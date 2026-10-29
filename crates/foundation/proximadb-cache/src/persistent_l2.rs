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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

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
    /// TD-TIER-1 SC3: residency pin, orthogonal to [`L2Class`]. `L2Class` is
    /// the *eviction order* under ordinary pressure; `protected` removes the
    /// entry from the ordinary-pressure victim set entirely and charges its
    /// bytes against [`StoreInner::reserved_budget`]. Explicit invalidation
    /// (`remove`/`remove_where`) still removes protected entries — pins
    /// protect immutable bytes, never visibility (SC-TE6). Runtime state only:
    /// intentionally NOT persisted in the entry file; the index rebuild drops
    /// protection and SC4's coverage reconciliation re-derives it from the
    /// grant ledger.
    protected: AtomicBool,
    /// TD-TIER-1 SC4-2: the grant group whose budget this pin is charged to
    /// (`None` = a manual pin, or unpinned). Set BEFORE `protected` flips to
    /// true and taken by whoever wins the `protected` swap back to false, so
    /// the group refund happens exactly once (see `release_pin`).
    pin_group: std::sync::Mutex<Option<Arc<GroupCounter>>>,
    touch: AtomicU64,
}

/// Test-only callback run on each selected eviction victim.
#[cfg(test)]
type EvictHook = Box<dyn Fn(&PersistentByteStore, &str) + Send + Sync>;

struct StoreInner {
    base: PathBuf,
    max_bytes: u64,
    resident_bytes: AtomicU64,
    class_resident_bytes: [AtomicU64; 2],
    /// TD-TIER-1 SC3: byte ceiling for protected entries. `0` (the default)
    /// makes every `protect` fail closed — the protection seam is inert until
    /// the reservation authority arms it (SC3-2 wiring).
    reserved_budget: AtomicU64,
    protected_bytes: AtomicU64,
    /// TD-TIER-1 SC4-2: the residency protection policy — one group per
    /// reservation grant, each with its own segment-root prefixes and its
    /// own byte budget (review B2: a single shared budget let one
    /// collection's pins consume every grant's share). The std mutex ALSO
    /// serializes every pin and unpin (review B5: concurrent same-key
    /// protects double-charged `protected_bytes`), so protection is
    /// synchronous and callable from inside `put` (which holds `mutation`)
    /// and from the sync residency bridge. Lock order: `mutation` →
    /// `protection`, never the reverse; no guard is held across an await.
    protection: std::sync::Mutex<ProtectionPolicy>,
    /// Fast path: false while the policy has no groups, so an inert store
    /// does not take the protection mutex on every `put` (review S4).
    policy_active: AtomicBool,
    /// TD-TIER-1 SC4-3: set when a policy pin was skipped for budget, so a
    /// later cache hit can retry it (coverage self-heal). Cleared by a full
    /// reconcile that skipped nothing.
    heal_pending: AtomicBool,
    /// Bounded heal attempts left: each released group pin re-arms it to
    /// [`HEAL_ATTEMPTS_PER_FREE`]; each hit-heal attempt spends one. Bounds
    /// the read-path cost of a steadily over-budget grant (review S1); the
    /// per-group `freed` flag decides whether an attempt can succeed.
    heal_attempts: AtomicU32,
    /// Test-only hook run between victim selection and its re-check, to
    /// exercise the selection→pin race deterministically.
    #[cfg(test)]
    evict_hook: std::sync::Mutex<Option<EvictHook>>,
    clock: AtomicU64,
    entries: DashMap<String, Arc<EntryMeta>>,
    mutation: tokio::sync::Mutex<()>,
}

impl std::fmt::Debug for StoreInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreInner")
            .field("base", &self.base)
            .field("max_bytes", &self.max_bytes)
            .field("entries", &self.entries.len())
            .finish_non_exhaustive()
    }
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
            reserved_budget: AtomicU64::new(0),
            protected_bytes: AtomicU64::new(0),
            protection: std::sync::Mutex::new(ProtectionPolicy::default()),
            policy_active: AtomicBool::new(false),
            heal_pending: AtomicBool::new(false),
            heal_attempts: AtomicU32::new(0),
            #[cfg(test)]
            evict_hook: std::sync::Mutex::new(None),
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

    /// Arm the protection budget (TD-TIER-1 SC3). The reservation authority
    /// sets this to its admitted grant bytes; `0` — the default — keeps the
    /// protection seam inert (every [`Self::protect`] fails closed). Shrinking
    /// below the current protected total does not evict: existing pins stand,
    /// and new protects fail until headroom exists.
    pub fn set_reserved_budget(&self, bytes: u64) {
        self.inner.reserved_budget.store(bytes, Ordering::Release);
    }

    /// The armed protection budget.
    pub fn reserved_budget(&self) -> u64 {
        self.inner.reserved_budget.load(Ordering::Acquire)
    }

    /// Bytes currently held by protected entries.
    pub fn protected_bytes(&self) -> u64 {
        self.inner.protected_bytes.load(Ordering::Acquire)
    }

    /// Pin one entry against ordinary eviction pressure (SC-TE4) — a MANUAL
    /// pin, charged to the store-wide budget and owned by no grant group.
    ///
    /// Fails closed when pinning would exceed the armed budget: an oversize
    /// pin is rejected at admission, never silently dropped later. Returns
    /// `Ok(false)` when the key is not indexed (nothing to pin), `Ok(true)`
    /// when pinned (including an already-protected idempotent re-pin). The
    /// pin is advisory against *pressure* only — explicit invalidation via
    /// [`Self::remove`] / [`Self::remove_where`] still applies (SC-TE6).
    ///
    /// Serialized on the protection mutex (review B5: lock-free
    /// check-then-act double-charged `protected_bytes` under concurrent
    /// same-key protects). Manual pins share the store-wide budget with the
    /// grant groups, so they can crowd groups out; production does not call
    /// this today (grants pin through [`Self::set_protection_groups`]).
    pub async fn protect(&self, key: &str) -> std::io::Result<bool> {
        let _guard = self.protection_guard();
        let Some(meta) = self.inner.entries.get(key).map(|entry| entry.clone()) else {
            return Ok(false);
        };
        self.pin_meta(key, &meta, None)
    }

    /// Drop one entry's pin (manual or policy). Returns whether it was
    /// pinned. Serialized on the protection mutex.
    pub async fn unprotect(&self, key: &str) -> bool {
        let _guard = self.protection_guard();
        let Some(meta) = self.inner.entries.get(key).map(|entry| entry.clone()) else {
            return false;
        };
        self.release_pin(&meta)
    }

    /// True when the entry is currently pinned (status projection helper).
    pub fn is_protected(&self, key: &str) -> bool {
        self.inner
            .entries
            .get(key)
            .is_some_and(|meta| meta.protected.load(Ordering::Acquire))
    }

    fn protection_guard(&self) -> std::sync::MutexGuard<'_, ProtectionPolicy> {
        self.inner
            .protection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Claim `len` against `counter` without exceeding `limit` (CAS loop).
    fn claim(counter: &AtomicU64, len: u64, limit: u64) -> Result<(), u64> {
        let mut current = counter.load(Ordering::Acquire);
        loop {
            let next = current.checked_add(len).ok_or(current)?;
            if next > limit {
                return Err(current);
            }
            match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Ok(()),
                Err(observed) => current = observed,
            }
        }
    }

    /// Pin `meta` (the caller holds the protection mutex). A group pin is
    /// charged to the group's own budget AND the store-wide budget; a
    /// manual pin (`group == None`) only to the store-wide budget.
    fn pin_meta(
        &self,
        key: &str,
        meta: &Arc<EntryMeta>,
        group: Option<(&Arc<GroupCounter>, u64)>,
    ) -> std::io::Result<bool> {
        if meta.protected.load(Ordering::Acquire) {
            return Ok(true);
        }
        if let Some((counter, budget)) = group
            && let Err(current) = Self::claim(&counter.pinned, meta.file_len, budget)
        {
            return Err(budget_error(&format!(
                "protecting key of {} bytes exceeds its grant's budget {budget} \
                 (grant already protects {current})",
                meta.file_len
            )));
        }
        let budget = self.inner.reserved_budget.load(Ordering::Acquire);
        if let Err(current) = Self::claim(&self.inner.protected_bytes, meta.file_len, budget) {
            if let Some((counter, _)) = group {
                counter.pinned.fetch_sub(meta.file_len, Ordering::AcqRel);
            }
            return Err(budget_error(&format!(
                "protecting key of {} bytes exceeds the reserved budget {budget} \
                 (already protected: {current}); raise the reservation or unprotect first",
                meta.file_len
            )));
        }
        // The `protected` flag and its owning group change together under
        // the entry's `pin_group` lock (verification round: a release that
        // swapped `protected` and THEN took the group could lose the old
        // group's refund to a concurrent re-pin that overwrote it).
        {
            let mut owner = meta
                .pin_group
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if meta.protected.load(Ordering::Acquire) {
                drop(owner);
                self.inner
                    .protected_bytes
                    .fetch_sub(meta.file_len, Ordering::AcqRel);
                if let Some((counter, _)) = group {
                    counter.pinned.fetch_sub(meta.file_len, Ordering::AcqRel);
                }
                return Ok(true);
            }
            *owner = group.map(|(counter, _)| counter.clone());
            meta.protected.store(true, Ordering::Release);
        }
        // `remove` is lock-free. Re-check that THIS entry (not merely the
        // key) is still indexed: a remove + re-put between our map read and
        // here would otherwise leave the old entry's bytes charged forever
        // (review S1). `release_pin`'s swap arbitrates the single refund.
        let still_indexed = self
            .inner
            .entries
            .get(key)
            .is_some_and(|current| Arc::ptr_eq(&current, meta));
        if !still_indexed {
            self.release_pin(meta);
            return Ok(false);
        }
        Ok(true)
    }

    /// Release a pin exactly once: whoever wins the `protected` swap refunds
    /// the store-wide counter and the owning group's counter.
    fn release_pin(&self, meta: &EntryMeta) -> bool {
        // Swap and take together under the entry's `pin_group` lock, so the
        // owner refunded is exactly the owner of the pin being released.
        let group = {
            let mut owner = meta
                .pin_group
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !meta.protected.swap(false, Ordering::AcqRel) {
                return false;
            }
            owner.take()
        };
        self.inner
            .protected_bytes
            .fetch_sub(meta.file_len, Ordering::AcqRel);
        if let Some(group) = group {
            group.pinned.fetch_sub(meta.file_len, Ordering::AcqRel);
            group.freed.store(true, Ordering::Release);
            self.inner
                .heal_attempts
                .store(HEAL_ATTEMPTS_PER_FREE, Ordering::Release);
        }
        true
    }

    /// The group whose budget a pin on `meta` is charged to, if any.
    fn pin_owner(meta: &EntryMeta) -> Option<Arc<GroupCounter>> {
        meta.pin_group
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// TD-TIER-1 SC4-2: replace the protection policy and reconcile resident
    /// entries against it. Each group is one reservation grant: its
    /// collection's segment-root prefixes and its own byte budget. A key
    /// belongs to the first group (by id) one of whose prefixes it contains,
    /// both compared in canonical form (see [`canonical_location`]); cache
    /// keys embed the published segment path mid-string, and each prefix
    /// ends in `/`, so containment cannot match a sibling collection.
    ///
    /// Reconcile: pins owned by a removed group, or whose key no longer
    /// matches its group, are released; a group whose budget shrank below
    /// its pinned bytes is clamped by releasing its pins; then unpinned
    /// matching entries are pinned against their group's budget —
    /// best-effort, skipped and counted when the budget is exhausted.
    /// Manual pins are never touched. O(entries × prefixes): admission
    /// cadence only.
    pub fn set_protection_groups(&self, groups: Vec<ProtectionGroup>) -> ProtectionSweep {
        let mut policy = self.protection_guard();
        let mut previous = std::mem::take(&mut policy.groups);
        for group in groups {
            let counter = previous
                .remove(&group.id)
                .map(|old| old.counter)
                .unwrap_or_else(|| Arc::new(GroupCounter::default()));
            policy.groups.insert(
                group.id,
                PolicyGroup {
                    prefixes: group
                        .prefixes
                        .iter()
                        .map(|prefix| canonical_location(prefix))
                        .collect(),
                    budget: group.budget_bytes,
                    counter,
                },
            );
        }
        self.inner
            .policy_active
            .store(!policy.groups.is_empty(), Ordering::Release);

        let mut sweep = ProtectionSweep::default();
        let entries: Vec<(String, Arc<EntryMeta>)> = self
            .inner
            .entries
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        // 1. Release policy pins whose group left or whose key left its group.
        for (key, meta) in &entries {
            let Some(owner) = Self::pin_owner(meta) else {
                continue;
            };
            let canonical = canonical_location(key);
            let keeps = policy.groups.values().any(|group| {
                Arc::ptr_eq(&group.counter, &owner)
                    && group
                        .prefixes
                        .iter()
                        .any(|p| canonical.contains(p.as_str()))
            });
            if !keeps && self.release_pin(meta) {
                sweep.unprotected += 1;
            }
        }
        // 2. Clamp groups whose budget shrank below their pinned bytes.
        for group in policy.groups.values() {
            for (_, meta) in &entries {
                if group.counter.pinned.load(Ordering::Acquire) <= group.budget {
                    break;
                }
                if Self::pin_owner(meta).is_some_and(|owner| Arc::ptr_eq(&owner, &group.counter))
                    && self.release_pin(meta)
                {
                    sweep.unprotected += 1;
                }
            }
        }
        // 3. Pin unpinned matching entries against their group's budget.
        for (key, meta) in &entries {
            if meta.protected.load(Ordering::Acquire) {
                continue;
            }
            if let Some(group) = policy.matching_group(key) {
                match self.pin_meta(key, meta, Some((&group.counter, group.budget))) {
                    Ok(true) => sweep.protected += 1,
                    Ok(false) => {}
                    Err(_) => sweep.skipped_over_budget += 1,
                }
            }
        }
        self.inner
            .heal_pending
            .store(sweep.skipped_over_budget > 0, Ordering::Release);
        sweep
    }

    /// The ids of the groups in the current policy (diagnostics/tests).
    pub fn protection_group_ids(&self) -> Vec<String> {
        self.protection_guard().groups.keys().cloned().collect()
    }

    /// Bytes currently pinned under one grant group — the coverage that
    /// grant has actually achieved (status/coverage input).
    pub fn covered_bytes(&self, group_id: &str) -> u64 {
        self.protection_guard()
            .groups
            .get(group_id)
            .map_or(0, |group| group.counter.pinned.load(Ordering::Acquire))
    }

    /// Bytes resident in the store under one grant group's prefixes, pinned
    /// or not (status input; `covered_bytes` ≤ this). O(entries): status
    /// reads only (the collection GET), never the query path.
    pub fn group_resident_bytes(&self, group_id: &str) -> u64 {
        // Copy the prefixes, then scan WITHOUT the protection mutex, so a
        // status read never blocks puts, eviction or pins.
        let prefixes = match self.protection_guard().groups.get(group_id) {
            Some(group) => group.prefixes.clone(),
            None => return 0,
        };
        self.inner
            .entries
            .iter()
            .filter(|entry| {
                let canonical = canonical_location(entry.key());
                prefixes
                    .iter()
                    .any(|prefix| canonical.contains(prefix.as_str()))
            })
            .map(|entry| entry.value().file_len)
            .sum()
    }

    /// Retry a skipped policy pin on a cache hit (SC4-3 self-heal). Only
    /// after a pin was skipped AND a group pin has since been released, a
    /// bounded number of hits ([`HEAL_ATTEMPTS_PER_FREE`] per release) try
    /// it; each attempt pins only when the hit key's OWN group has freed
    /// room, so an unrelated hit cannot consume another group's heal
    /// (review S1). Never waits on the protection mutex; reads otherwise
    /// take no lock and write no shared state.
    fn heal_on_hit(&self, key: &str, meta: &EntryMeta) {
        if !self.inner.heal_pending.load(Ordering::Acquire)
            || meta.protected.load(Ordering::Acquire)
            || self.inner.heal_attempts.load(Ordering::Acquire) == 0
        {
            return;
        }
        if self
            .inner
            .heal_attempts
            .try_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                left.checked_sub(1)
            })
            .is_err()
        {
            return;
        }
        let policy = match self.inner.protection.try_lock() {
            Ok(policy) => policy,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return,
        };
        let Some(group) = policy.matching_group(key) else {
            return;
        };
        if !group.counter.freed.load(Ordering::Acquire) {
            return;
        }
        let Some(current) = self.inner.entries.get(key).map(|entry| entry.clone()) else {
            return;
        };
        if self
            .pin_meta(key, &current, Some((&group.counter, group.budget)))
            .is_err()
        {
            // The freed room does not fit this entry: stop paying for
            // attempts in this group until another of its pins is released.
            group.counter.freed.store(false, Ordering::Release);
        }
    }

    /// Pin a just-admitted (or re-put) key when it falls under the policy.
    /// Called from `put`/`put_file_range` BEFORE eviction runs, so a
    /// protected key cannot be evicted between its write and its pin.
    /// Best-effort: an over-budget pin is skipped (coverage degrades; the
    /// write succeeds). Inert stores skip the mutex entirely.
    fn protect_on_admit(&self, key: &str) {
        if !self.inner.policy_active.load(Ordering::Acquire) {
            return;
        }
        let policy = self.protection_guard();
        self.pin_under_policy(&policy, key);
    }

    /// Pin `key` under its matching group; the caller holds the protection
    /// mutex. A budget refusal marks a heal as pending.
    fn pin_under_policy(&self, policy: &ProtectionPolicy, key: &str) {
        let Some(group) = policy.matching_group(key) else {
            return;
        };
        let Some(meta) = self.inner.entries.get(key).map(|entry| entry.clone()) else {
            return;
        };
        if self
            .pin_meta(key, &meta, Some((&group.counter, group.budget)))
            .is_err()
        {
            self.inner.heal_pending.store(true, Ordering::Release);
        }
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
        let admit_key = key.clone();
        let _guard = self.inner.mutation.lock().await;
        if let Some(existing) = self.inner.entries.get(&key) {
            existing.touch.store(self.next_tick(), Ordering::Relaxed);
            drop(existing);
            // A re-put of a resident key retries a pin that was skipped
            // earlier (budget exhausted then, maybe not now).
            self.protect_on_admit(&key);
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
        self.protect_on_admit(&admit_key);
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
        let admit_key = key.clone();
        let _guard = self.inner.mutation.lock().await;
        if let Some(existing) = self.inner.entries.get(&key) {
            existing.touch.store(self.next_tick(), Ordering::Relaxed);
            drop(existing);
            // A re-put of a resident key retries a pin that was skipped
            // earlier (budget exhausted then, maybe not now).
            self.protect_on_admit(&key);
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
        self.protect_on_admit(&admit_key);
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
                self.heal_on_hit(key, &meta);
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
                self.heal_on_hit(key, &meta);
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
            // Ordinary pressure never touches protected entries (SC-TE4). If
            // nothing evictable remains, stop over-budget rather than deleting
            // a pin — over-budget is an observable degradation the authority
            // reports, not something the cache silently resolves.
            let victim = self
                .inner
                .entries
                .iter()
                .filter(|entry| !entry.value().protected.load(Ordering::Acquire))
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
            // Re-check under the protection mutex: every pin is taken under
            // it, so a pin cannot land between this check and the removal
            // (final SC4-2 review: a concurrent registration sweep could pin
            // the chosen victim, and pressure then deleted a protected entry).
            // A victim pinned or removed since selection is simply skipped:
            // the next pass re-selects (the pinned one is filtered out), so
            // the loop still terminates when nothing evictable remains.
            #[cfg(test)]
            if let Some(hook) = self
                .inner
                .evict_hook
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_ref()
            {
                hook(self, &victim);
            }
            // The unlink runs after the guard drops (review S4): no blocking
            // file I/O under the protection mutex.
            let unlink = {
                let _guard = self.protection_guard();
                let path = self
                    .inner
                    .entries
                    .get(&victim)
                    .filter(|meta| !meta.protected.load(Ordering::Acquire))
                    .map(|meta| meta.path.clone());
                path.filter(|_| self.remove_indexed(&victim, false))
            };
            if let Some(path) = unlink {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    fn remove_indexed(&self, key: &str, delete_file: bool) -> bool {
        let Some((_, meta)) = self.inner.entries.remove(key) else {
            return false;
        };
        self.release_pin(&meta);
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
            protected: AtomicBool::new(false),
            pin_group: std::sync::Mutex::new(None),
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
            protected: AtomicBool::new(false),
            pin_group: std::sync::Mutex::new(None),
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
            protected: AtomicBool::new(false),
            pin_group: std::sync::Mutex::new(None),
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

/// TD-TIER-1 SC4-2: one reservation grant's protection share.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectionGroup {
    /// Stable group identity (the grant id).
    pub id: String,
    /// The collection's segment-root prefixes (each ending in `/`).
    pub prefixes: Vec<String>,
    /// Bytes this group may pin (the grant's reserved bytes).
    pub budget_bytes: u64,
}

/// Bytes pinned under one group (shared by the policy and the entries).
#[derive(Debug, Default)]
struct GroupCounter {
    pinned: AtomicU64,
    /// Set when one of this group's pins is released (its budget has room
    /// again); cleared when a heal attempt finds the room insufficient.
    freed: AtomicBool,
}

/// Hit-heal attempts granted per released group pin (SC4-3 review S1).
const HEAL_ATTEMPTS_PER_FREE: u32 = 256;

#[derive(Debug)]
struct PolicyGroup {
    /// Canonical-form prefixes.
    prefixes: Vec<String>,
    budget: u64,
    counter: Arc<GroupCounter>,
}

#[derive(Debug, Default)]
struct ProtectionPolicy {
    groups: std::collections::BTreeMap<String, PolicyGroup>,
}

impl ProtectionPolicy {
    /// The first group (by id) whose prefix the key contains, in canonical
    /// form.
    fn matching_group(&self, key: &str) -> Option<&PolicyGroup> {
        if self.groups.is_empty() {
            return None;
        }
        let canonical = canonical_location(key);
        self.groups.values().find(|group| {
            group
                .prefixes
                .iter()
                .any(|prefix| canonical.contains(prefix.as_str()))
        })
    }
}

/// TD-TIER-1 SC4-2: canonical form for comparing locations embedded in cache
/// keys (review S2). Azure scheme aliases (`abfs://`, `abfss://`, `adls://`,
/// `azure://`) become the `az://` form the Azure listing emits, `gcs://`
/// becomes `gs://`, and runs of
/// `/` collapse to one except the `//` that follows a scheme's `:`, so a
/// base written as `s3://b/x/` and keys listed as `s3://b/x/...` compare
/// equal. Applied to both prefixes and keys. Known trade-off: two bases that
/// differ only in a run of slashes (`s3://b/x//y/` vs `s3://b/x/y/`), which
/// object stores treat as distinct keys, compare equal here.
pub fn canonical_location(raw: &str) -> String {
    let mut text = raw.to_string();
    for alias in ["abfss://", "abfs://", "adls://", "azure://"] {
        text = text.replace(alias, "az://");
    }
    text = text.replace("gcs://", "gs://");
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch == '/' && out.ends_with('/') {
            let before_last = out[..out.len() - 1].chars().last();
            if before_last != Some(':') {
                continue;
            }
        }
        out.push(ch);
    }
    out
}

/// Outcome of a [`PersistentByteStore::set_protection_groups`] reconcile.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProtectionSweep {
    /// Entries newly pinned under the policy.
    pub protected: u64,
    /// Entries unpinned because their prefix left the policy.
    pub unprotected: u64,
    /// Matching entries left unpinned because the armed budget was exhausted
    /// (coverage degraded, never forced).
    pub skipped_over_budget: u64,
}

fn budget_error(message: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::ResourceBusy, /* semantically: admission refused */
        format!("TD-TIER-1 SC3 protection budget exceeded: {message}"),
    )
}

fn join_error(error: tokio::task::JoinError) -> std::io::Error {
    std::io::Error::other(format!("persistent cache worker failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{
        CHECKSUM_CHUNK_BYTES, L2Class, PersistentByteStore, ProtectionGroup, canonical_location,
    };
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

    // ---- TD-TIER-1 SC3: protection (orthogonal to L2Class) ----

    /// A 1 MiB store with a 1 MiB armed protection budget (the common
    /// protection-test fixture). Keep the TempDir alive for the test.
    fn armed_store() -> (tempfile::TempDir, PersistentByteStore) {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = PersistentByteStore::open(dir.path(), 1 << 20).expect("open");
        store.set_reserved_budget(1 << 20);
        (dir, store)
    }

    fn payload(len: usize) -> Arc<[u8]> {
        Arc::from(vec![7u8; len].into_boxed_slice())
    }

    // SC-TE4: an armed pin survives ordinary eviction pressure; unprotected
    // neighbors are the ones evicted.
    #[tokio::test]
    async fn protected_entry_survives_pressure() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = PersistentByteStore::open(dir.path(), 8 * 1024).expect("open");
        store.set_reserved_budget(8 * 1024);

        store
            .put("pinned", L2Class::Survivor, payload(4 * 1024))
            .await
            .expect("put pinned");
        assert!(store.protect("pinned").await.expect("protect pinned"));
        assert!(store.is_protected("pinned"));

        // Churn past the store budget: each put triggers eviction; the pinned
        // entry must never be the victim while unpinned peers are evictable.
        for i in 0..8 {
            store
                .put(&format!("churn-{i}"), L2Class::Survivor, payload(4 * 1024))
                .await
                .expect("put churn");
        }
        assert!(
            store.get("pinned").await.expect("get pinned").is_some(),
            "the pin survives the pressure"
        );
        assert!(store.protected_bytes() > 0);
    }

    // SC-TE4 admission half: an oversize pin is rejected at admission
    // (fail-closed), never silently granted and dropped later.
    #[tokio::test]
    async fn protect_beyond_reservation_fails_closed() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = PersistentByteStore::open(dir.path(), 1 << 20).expect("open");
        store
            .put("entry", L2Class::Survivor, payload(4 * 1024))
            .await
            .expect("put");

        store.set_reserved_budget(1024);
        let error = store
            .protect("entry")
            .await
            .expect_err("oversize pin refused");
        assert!(error.to_string().contains("reserved budget"), "{error}");
        assert!(!store.is_protected("entry"));
        assert_eq!(store.protected_bytes(), 0);

        // Unpinning accounting: arm full budget, pin, unpin, pin again.
        store.set_reserved_budget(1 << 20);
        assert!(store.protect("entry").await.expect("protect"));
        assert!(store.unprotect("entry").await);
        assert_eq!(store.protected_bytes(), 0);
        assert!(store.protect("entry").await.expect("re-protect"));
    }

    // SC-TE4: with every survivor pinned, ordinary pressure has no pin to
    // delete — an over-budget newcomer evicts ITSELF and the loop terminates
    // (no spin, no pin deletion). Over-budget is reported state, not something
    // the cache silently resolves at the pins' expense.
    #[tokio::test]
    async fn all_protected_over_budget_terminates() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = PersistentByteStore::open(dir.path(), 16 * 1024).expect("open");
        store.set_reserved_budget(64 * 1024);
        for i in 0..4 {
            store
                .put(&format!("pin-{i}"), L2Class::Survivor, payload(2 * 1024))
                .await
                .expect("put");
            assert!(store.protect(&format!("pin-{i}")).await.expect("protect"));
        }

        // An over-budget newcomer: only it is evictable, so it evicts itself
        // and eviction terminates rather than touching any pin.
        store
            .put("newcomer", L2Class::Survivor, payload(32 * 1024))
            .await
            .expect("put completes (self-evicting)");

        for i in 0..4 {
            assert!(store.is_protected(&format!("pin-{i}")));
            assert!(
                store.get(&format!("pin-{i}")).await.expect("get").is_some(),
                "no pin deleted to 'fix' the over-budget state"
            );
        }
        assert!(
            store.get("newcomer").await.expect("get").is_none(),
            "the unpinned newcomer self-evicted"
        );
        assert!(store.resident_bytes() <= 16 * 1024);
    }

    // SC-TE6: explicit invalidation removes a protected entry — pins protect
    // immutable bytes, never visibility — and the accounting follows.
    #[tokio::test]
    async fn explicit_remove_still_removes_protected() {
        let (_dir, store) = armed_store();
        store
            .put("entry", L2Class::Survivor, payload(4 * 1024))
            .await
            .expect("put");
        assert!(store.protect("entry").await.expect("protect"));
        let pinned_bytes = store.protected_bytes();
        assert!(pinned_bytes > 0);

        assert!(store.remove("entry"));
        assert_eq!(store.protected_bytes(), 0, "pin accounting released");
        assert!(store.get("entry").await.expect("get").is_none());
    }

    // The protection seam is inert until armed: default budget 0 refuses
    // every pin.
    #[tokio::test]
    async fn zero_budget_keeps_protection_inert() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = PersistentByteStore::open(dir.path(), 1 << 20).expect("open");
        store
            .put("entry", L2Class::Survivor, payload(64))
            .await
            .expect("put");
        assert_eq!(store.reserved_budget(), 0);
        let error = store
            .protect("entry")
            .await
            .expect_err("unarmed budget must refuse");
        assert!(error.to_string().contains("reserved budget"), "{error}");
        // A missing key is a no-op, not an error.
        assert!(!store.protect("absent").await.expect("absent key"));
    }

    // Review B5 regression: concurrent same-key protects must charge the
    // entry's bytes exactly ONCE — the pre-fix check-then-act race charged
    // twice (and the phantom was unclearable by any API).
    #[tokio::test]
    async fn concurrent_same_key_protects_charge_once() {
        let (_dir, store) = armed_store();
        store
            .put("entry", L2Class::Survivor, payload(4 * 1024))
            .await
            .expect("put");
        let entry_bytes = store.protected_bytes();
        let _ = entry_bytes;

        let mut handles = Vec::new();
        for _ in 0..16 {
            let store = store.clone();
            handles.push(tokio::spawn(async move { store.protect("entry").await }));
        }
        for handle in handles {
            assert!(handle.await.expect("join").expect("protect"));
        }
        let after = store.protected_bytes();
        let file_len = store.resident_bytes(); // single entry store
        assert_eq!(
            after, file_len,
            "16 concurrent protects charge exactly the entry's bytes once"
        );
        assert!(store.unprotect("entry").await);
        assert_eq!(
            store.protected_bytes(),
            0,
            "a single unprotect clears the full charge — no phantom"
        );
    }

    // Verification round 3 (B5 residual): the protects-only stress above
    // cannot fail on a membership-only refund; the actual arbitration is
    // protect vs the LOCK-FREE remove. Churn protects against concurrent
    // removes and assert the invariant that arbitration defends:
    // protected_bytes never exceeds the armed budget — a double-decrement
    // wraps to ~2^64 and blows straight past it.
    #[tokio::test]
    async fn protect_vs_remove_stress_never_breaks_accounting() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = PersistentByteStore::open(dir.path(), 1 << 20).expect("open");
        let budget = 64 * 1024u64;
        store.set_reserved_budget(budget);
        let payload = payload(4 * 1024);

        for round in 0..64 {
            store
                .put(&format!("k{round}"), L2Class::Survivor, payload.clone())
                .await
                .expect("put");
            let key = format!("k{round}");
            let mut handles = Vec::new();
            for i in 0..8 {
                let store = store.clone();
                let key = key.clone();
                handles.push(tokio::spawn(async move {
                    if i % 2 == 0 {
                        let _ = store.protect(&key).await;
                    } else {
                        let _ = store.unprotect(&key).await;
                    }
                }));
            }
            // The lock-free invalidator races the pins.
            let remover = store.clone();
            let remove_key = key.clone();
            let ((), _) = tokio::join!(
                async {
                    for handle in handles {
                        let _ = handle.await;
                    }
                },
                async move {
                    let _ = remover.remove(&remove_key);
                }
            );
            assert!(
                store.protected_bytes() <= budget,
                "round {round}: protected_bytes {} exceeds the armed budget {} \
                 (arithmetic wrapped or double-decremented)",
                store.protected_bytes(),
                budget
            );
        }
    }

    // ---- TD-TIER-1 SC4-2: per-grant protection groups ----

    const C1: &str = "s3://bucket/tenants/acme/12/data/";
    const C2: &str = "s3://bucket/tenants/beta/34/data/";

    fn key_in(prefix: &str, file: &str) -> String {
        format!("invariants/region-a/{prefix}{file}")
    }

    fn group(id: &str, prefix: &str, budget: u64) -> ProtectionGroup {
        ProtectionGroup {
            id: id.to_string(),
            prefixes: vec![prefix.to_string()],
            budget_bytes: budget,
        }
    }

    /// Keys under a group's prefix are pinned on admission; a sibling
    /// collection whose decimal id shares the prefix digits (`123` vs `12`)
    /// is NOT; a survivor-shaped key with the path mid-string is.
    #[tokio::test]
    async fn policy_pins_on_admission_with_exact_prefix_boundaries() {
        let (_dir, store) = armed_store();
        store.set_protection_groups(vec![group("g1", C1, 1 << 20)]);

        store
            .put(key_in(C1, "L0_a.pax"), L2Class::Invariants, payload(1024))
            .await
            .expect("put c1");
        let sibling = "invariants/region-a/s3://bucket/tenants/acme/123/data/L0_a.pax";
        store
            .put(sibling, L2Class::Invariants, payload(1024))
            .await
            .expect("put sibling");
        let survivor = format!("survivor/s/2:{C1}L0_a.pax:0:64");
        store
            .put(survivor.clone(), L2Class::Survivor, payload(64))
            .await
            .expect("put survivor");

        assert!(
            store.is_protected(&key_in(C1, "L0_a.pax")),
            "pinned on admit"
        );
        assert!(store.is_protected(&survivor), "mid-string path is matched");
        assert!(
            !store.is_protected(sibling),
            "id 123 must not match prefix 12/"
        );
        assert!(store.covered_bytes("g1") > 0);
    }

    /// Review B2: each grant pins only against its OWN budget — a hot
    /// collection cannot consume another grant's share, and releasing one
    /// grant leaves the store within the remaining budget.
    #[tokio::test]
    async fn grants_pin_within_their_own_budget() {
        let (_dir, store) = armed_store();
        // Measure one entry's on-disk size (it includes the key, so the probe
        // key has the same length as the keys below), then budget each grant
        // for exactly one entry.
        store
            .put(key_in(C1, "p"), L2Class::Invariants, payload(2048))
            .await
            .expect("probe");
        let one = store.resident_bytes();
        assert!(store.remove(&key_in(C1, "p")));
        store.set_reserved_budget(2 * one);
        store.set_protection_groups(vec![group("a", C1, one), group("b", C2, one)]);

        // A's hot data arrives first: only ONE of A's entries may pin.
        for file in ["x", "y"] {
            store
                .put(key_in(C1, file), L2Class::Invariants, payload(2048))
                .await
                .expect("put a");
        }
        store
            .put(key_in(C2, "z"), L2Class::Invariants, payload(2048))
            .await
            .expect("put b");
        assert_eq!(store.covered_bytes("a"), one, "A capped at its own budget");
        assert_eq!(store.covered_bytes("b"), one, "B keeps its share");

        // Release B, shrink the budget to A's share: the store stays within it.
        store.set_protection_groups(vec![group("a", C1, one)]);
        store.set_reserved_budget(one);
        assert!(store.protected_bytes() <= store.reserved_budget());
        assert!(!store.is_protected(&key_in(C2, "z")), "B's pin released");
    }

    /// Changing the groups pins already-resident matches, releases the pins
    /// of a removed group, and leaves MANUAL pins alone even when their key
    /// matches the removed prefix (review NIT).
    #[tokio::test]
    async fn sweep_pins_resident_and_releases_only_policy_pins() {
        let (_dir, store) = armed_store();
        store
            .put(key_in(C1, "L1_b.pax"), L2Class::Invariants, payload(512))
            .await
            .expect("put");
        store
            .put(key_in(C1, "manual"), L2Class::Survivor, payload(256))
            .await
            .expect("put manual");
        assert!(
            store
                .protect(&key_in(C1, "manual"))
                .await
                .expect("manual pin")
        );

        let sweep = store.set_protection_groups(vec![group("g1", C1, 1 << 20)]);
        assert_eq!(sweep.protected, 1, "resident match pinned retroactively");
        assert!(store.is_protected(&key_in(C1, "L1_b.pax")));

        let sweep = store.set_protection_groups(Vec::new());
        assert_eq!(sweep.unprotected, 1);
        assert!(
            !store.is_protected(&key_in(C1, "L1_b.pax")),
            "policy pin released"
        );
        assert!(
            store.is_protected(&key_in(C1, "manual")),
            "a manual pin survives removal of a prefix it matches"
        );
    }

    /// Coverage degrades instead of forcing: matching entries beyond the
    /// group budget stay unpinned and are counted.
    #[tokio::test]
    async fn sweep_skips_pins_beyond_the_group_budget() {
        let (_dir, store) = armed_store();
        for file in ["a", "b", "c"] {
            store
                .put(key_in(C1, file), L2Class::Invariants, payload(4 * 1024))
                .await
                .expect("put");
        }
        let one = store.resident_bytes() / 3;
        let sweep = store.set_protection_groups(vec![group("g1", C1, one)]);
        assert_eq!(sweep.protected, 1, "the group budget covers one entry");
        assert_eq!(sweep.skipped_over_budget, 2);
        assert_eq!(store.covered_bytes("g1"), one);
    }

    /// A group budget that shrinks below its pinned bytes is clamped.
    #[tokio::test]
    async fn shrinking_a_group_budget_clamps_its_pins() {
        let (_dir, store) = armed_store();
        for file in ["a", "b"] {
            store
                .put(key_in(C1, file), L2Class::Invariants, payload(2048))
                .await
                .expect("put");
        }
        let one = store.resident_bytes() / 2;
        store.set_protection_groups(vec![group("g1", C1, 2 * one)]);
        assert_eq!(store.covered_bytes("g1"), 2 * one);
        store.set_protection_groups(vec![group("g1", C1, one)]);
        assert!(
            store.covered_bytes("g1") <= one,
            "clamped to the new budget"
        );
    }

    /// SC4-3 self-heal: a pin skipped for budget is retried on the next cache
    /// hit once room frees (here: the pinned entry is invalidated), with no
    /// group change and no re-put.
    #[tokio::test]
    async fn skipped_pin_heals_on_hit_once_room_frees() {
        let (_dir, store) = armed_store();
        for file in ["a", "b"] {
            store
                .put(key_in(C1, file), L2Class::Invariants, payload(2048))
                .await
                .expect("put");
        }
        let one = store.resident_bytes() / 2;
        let sweep = store.set_protection_groups(vec![group("g1", C1, one)]);
        assert_eq!((sweep.protected, sweep.skipped_over_budget), (1, 1));
        let (pinned, skipped) = if store.is_protected(&key_in(C1, "a")) {
            (key_in(C1, "a"), key_in(C1, "b"))
        } else {
            (key_in(C1, "b"), key_in(C1, "a"))
        };
        store
            .put(key_in(C2, "other"), L2Class::Invariants, payload(64))
            .await
            .expect("put unrelated");
        assert!(store.remove(&pinned), "room frees");
        assert!(!store.is_protected(&skipped), "not healed until touched");
        // Review S1: a hit on a key outside every group must not consume
        // the heal that the freed room enables for g1.
        store
            .get(&key_in(C2, "other"))
            .await
            .expect("get unrelated")
            .expect("hit unrelated");
        store.get(&skipped).await.expect("get").expect("hit");
        assert!(
            store.is_protected(&skipped),
            "a hit retries the skipped pin"
        );
        assert_eq!(store.covered_bytes("g1"), one);
    }

    /// SC4-3 status input: resident bytes under a group count pinned and
    /// unpinned matches, never a sibling collection's.
    #[tokio::test]
    async fn group_resident_bytes_counts_only_its_prefixes() {
        let (_dir, store) = armed_store();
        for file in ["a", "b"] {
            store
                .put(key_in(C1, file), L2Class::Invariants, payload(1024))
                .await
                .expect("put");
        }
        store
            .put(key_in(C2, "a"), L2Class::Invariants, payload(1024))
            .await
            .expect("put other");
        let each = store.resident_bytes() / 3;
        store.set_protection_groups(vec![group("g1", C1, each)]);
        assert_eq!(store.group_resident_bytes("g1"), 2 * each);
        assert_eq!(store.covered_bytes("g1"), each, "covered <= resident");
        assert_eq!(store.group_resident_bytes("absent"), 0);
    }

    /// SC4-3: an entry pinned AFTER eviction selected it (a registration
    /// sweep racing the pressure loop) is not deleted. The hook pins the
    /// selected victim between selection and the re-check — the exact window
    /// the re-check closes; without it, pressure deletes the pin.
    #[tokio::test]
    async fn eviction_rechecks_a_victim_pinned_after_selection() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = PersistentByteStore::open(dir.path(), 16 * 1024).expect("small store");
        store.set_reserved_budget(1 << 30);
        // The victim is the only Survivor-class entry: eviction orders by
        // (class, touch), so it is selected first regardless of touch ties
        // (new entries start at touch 1; a tie would break by hash order).
        let victim = key_in(C1, "lru");
        store
            .put(victim.clone(), L2Class::Survivor, payload(6 * 1024))
            .await
            .expect("put lru");
        *store
            .inner
            .evict_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(Box::new(|store: &PersistentByteStore, selected: &str| {
                if selected.contains("/12/data/") && !store.is_protected(selected) {
                    store.set_protection_groups(vec![group("g1", C1, 1 << 30)]);
                }
            }));
        for i in 0..4 {
            store
                .put(
                    key_in(C2, &format!("f{i}")),
                    L2Class::Invariants,
                    payload(6 * 1024),
                )
                .await
                .expect("pressure put");
        }
        assert!(store.is_protected(&victim), "pinned by the racing sweep");
        assert!(
            store.get(&victim).await.expect("get").is_some(),
            "a victim pinned after selection must survive the pressure"
        );
    }

    /// Review S2: Azure scheme aliases and doubled slashes compare equal.
    #[test]
    fn canonical_location_unifies_aliases_and_slashes() {
        assert_eq!(
            canonical_location("adls://c/t/12/data/"),
            "az://c/t/12/data/"
        );
        assert_eq!(canonical_location("abfs://c/x"), "az://c/x");
        assert_eq!(
            canonical_location("s3://b/x//tenants/t/"),
            "s3://b/x/tenants/t/"
        );
        assert_eq!(
            canonical_location("invariants/region-a/s3://b/k"),
            "invariants/region-a/s3://b/k",
            "the scheme's own // is preserved"
        );
    }

    /// Review S2 end to end: a group registered with an `adls://` prefix pins
    /// keys the Azure listing emits as `az://`.
    #[tokio::test]
    async fn alias_prefix_matches_listed_keys() {
        let (_dir, store) = armed_store();
        store.set_protection_groups(vec![group("g1", "adls://cont/tenants/t/12/data/", 1 << 20)]);
        let listed = "invariants/control/az://cont/tenants/t/12/data/L0.pax";
        store
            .put(listed, L2Class::Invariants, payload(128))
            .await
            .expect("put");
        assert!(store.is_protected(listed));
    }

    // Protection is runtime state by design: the index rebuild drops it and
    // SC4's coverage reconciliation re-derives pins from the grant ledger.
    #[tokio::test]
    async fn rebuild_drops_protection() {
        let (dir, store) = armed_store();
        store
            .put("entry", L2Class::Survivor, payload(64))
            .await
            .expect("put");
        assert!(store.protect("entry").await.expect("protect"));
        drop(store);

        let reopened = PersistentByteStore::open(dir.path(), 1 << 20).expect("reopen");
        assert!(
            reopened.get("entry").await.expect("get").is_some(),
            "value survives the rebuild"
        );
        assert!(
            !reopened.is_protected("entry"),
            "protection is re-derived by SC4, not persisted"
        );
    }
}
