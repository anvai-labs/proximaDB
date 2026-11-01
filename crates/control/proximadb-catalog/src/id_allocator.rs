// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Monotonic per-type id allocator for stable catalog object ids (ADR-031).
//!
//! Design tenet (ADR-031): internal stable ids are compact `u64`, allocated **per
//! object type** (tenant, namespace, table/collection, …) by a **global monotonic
//! allocator**, **unique across all tenants within their type**, and **never
//! reused**. One [`IdAllocator`] instance owns one type's id space.
//!
//! Mirrors the WAL `GlobalLsnAllocator` (lock-free `AtomicU64`). Durability is by
//! **recovery, not a separate persisted counter**: at startup the owner calls
//! [`IdAllocator::raise_floor`] with `max(existing object_id) + 1`, so a restart can
//! never hand out an id already on disk. `0` is reserved as "unset" (so
//! `Option<u64>::None` / `0` both mean "no id yet"); the first real id is `1`.

use dashmap::DashMap;
use std::sync::atomic::{AtomicI32, AtomicU16, AtomicU32, AtomicU64, Ordering};

const RELAXED: Ordering = Ordering::Relaxed;

/// Lock-free monotonic allocator for one object-type's stable `u64` id space.
#[derive(Debug)]
pub struct IdAllocator {
    next: AtomicU64,
}

impl IdAllocator {
    /// New allocator whose first allocation is `start` (clamped to ≥ 1, since `0`
    /// is the reserved "unset" sentinel).
    pub fn new(start: u64) -> Self {
        Self {
            next: AtomicU64::new(start.max(1)),
        }
    }

    /// Allocate the next id (lock-free; strictly increasing; never reused).
    pub fn allocate(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    /// The id the next [`allocate`](Self::allocate) would return, without consuming it.
    pub fn peek(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }

    /// Recovery hook: ensure the next allocation is **at least** `floor` (monotone —
    /// never lowers the counter). Call at startup with `max(existing id) + 1` so a
    /// restart never reuses an id already persisted.
    pub fn raise_floor(&self, floor: u64) {
        self.next.fetch_max(floor, Ordering::Relaxed);
    }
}

impl Default for IdAllocator {
    fn default() -> Self {
        Self::new(1)
    }
}

/// The trait for stable-ID allocation (ADR-031 Phase 3).
///
/// **Contract**: `allocate()` is monotonic, never reuses IDs, lock-free.
/// `raise_floor()` is the recovery hook (call at startup with
/// `max(existing) + 1` so a restart never reuses a persisted ID).
///
/// **Single-pod**: [`IdAllocator`] (in-memory, persistent via scan + raise).
/// **Distributed** (follow-up): a gRPC service. Same contract — callers don't know.
pub trait StableIdAllocator: Send + Sync {
    fn allocate(&self) -> u64;
    fn peek(&self) -> u64;
    fn raise_floor(&self, floor: u64);
}

impl StableIdAllocator for IdAllocator {
    fn allocate(&self) -> u64 {
        IdAllocator::allocate(self)
    }
    fn peek(&self) -> u64 {
        IdAllocator::peek(self)
    }
    fn raise_floor(&self, floor: u64) {
        IdAllocator::raise_floor(self, floor)
    }
}

// ---------------------------------------------------------------------------
// CatalogIdService: per-scope typed-atomic ID allocation (ADR-031 Phase 4a).
// ---------------------------------------------------------------------------

/// Per-scope stable-ID allocation using **typed atomics** (no `u64` casts).
///
/// Each scoped type has a per-parent counter at the **exact atomic width** of
/// the target type (`AtomicU16` for namespace, `AtomicU32` for
/// collection/index/segment, `AtomicI32` for column). `fetch_add` returns the
/// correct type directly.
///
/// **Global uniqueness** is via the composite `(account, namespace, collection)`.
/// Bare IDs reuse across scopes (namespace 1 in account A ≠ namespace 1 in
/// account B) — the composite resolves them. This is the minting source for the
/// typed identity stored on `CatalogTableSchema` (`stable_account_id` /
/// `stable_namespace_id` / `stable_collection_id`); the root crate composes those
/// primitives into a `CollectionIdentity` at the path boundary.
///
/// Uses plain `u32`/`u16`/`i32` (the catalog crate cannot import the root's type
/// aliases — layering); the values are identical to `AccountId`/`NamespaceId`/…
/// defined in `src/core/stable_id.rs`.
pub struct CatalogIdService {
    /// Per-account namespace counters (AtomicU16 → namespace u16).
    namespace_allocators: DashMap<u32, AtomicU16>,
    /// Per-(account, namespace) collection counters (AtomicU32 → collection u32).
    collection_allocators: DashMap<(u32, u16), AtomicU32>,
    /// Per-collection column counters (AtomicI32 → column i32).
    column_allocators: DashMap<u32, AtomicI32>,
    /// Per-collection index counters (AtomicU32 → index u32).
    index_allocators: DashMap<u32, AtomicU32>,
    /// Per-collection SST segment counters (AtomicU32 → segment u32).
    segment_allocators: DashMap<u32, AtomicU32>,
}

impl Default for CatalogIdService {
    fn default() -> Self {
        Self::new()
    }
}

impl CatalogIdService {
    pub fn new() -> Self {
        Self {
            namespace_allocators: DashMap::new(),
            collection_allocators: DashMap::new(),
            column_allocators: DashMap::new(),
            index_allocators: DashMap::new(),
            segment_allocators: DashMap::new(),
        }
    }

    // ── Mint (typed atomics — no casts) ──────────────────────────────────

    /// Mint a namespace id (u16) scoped to `account_id` (1, 2, 3 within account).
    pub fn mint_namespace_id(&self, account_id: u32) -> u16 {
        self.namespace_allocators
            .entry(account_id)
            .or_insert(AtomicU16::new(1))
            .fetch_add(1, RELAXED)
    }

    /// Mint a collection id (u32) scoped to `(account_id, namespace_id)`.
    pub fn mint_collection_id(&self, account_id: u32, namespace_id: u16) -> u32 {
        self.collection_allocators
            .entry((account_id, namespace_id))
            .or_insert(AtomicU32::new(1))
            .fetch_add(1, RELAXED)
    }

    /// Mint a column id (i32) scoped to `collection_id`.
    pub fn mint_column_id(&self, collection_id: u32) -> i32 {
        self.column_allocators
            .entry(collection_id)
            .or_insert(AtomicI32::new(1))
            .fetch_add(1, RELAXED)
    }

    /// Mint an index id (u32) scoped to `collection_id`.
    pub fn mint_index_id(&self, collection_id: u32) -> u32 {
        self.index_allocators
            .entry(collection_id)
            .or_insert(AtomicU32::new(1))
            .fetch_add(1, RELAXED)
    }

    /// Mint an SST segment id (u32) scoped to `collection_id` (1, 2, 3 …).
    pub fn mint_segment_id(&self, collection_id: u32) -> u32 {
        self.segment_allocators
            .entry(collection_id)
            .or_insert(AtomicU32::new(1))
            .fetch_add(1, RELAXED)
    }

    // ── Per-scope recovery (typed — no casts) ────────────────────────────

    /// Recover the namespace allocator floor for `account_id`. Call at startup
    /// with `max(existing namespace u16 in this account)` so a restart never
    /// reuses a persisted id.
    pub fn recover_namespace_floor(&self, account_id: u32, max_existing: u16) {
        self.namespace_allocators
            .entry(account_id)
            .or_insert(AtomicU16::new(1))
            .fetch_max(max_existing + 1, RELAXED);
    }

    /// Recover the collection allocator floor for `(account_id, namespace_id)`.
    pub fn recover_collection_floor(&self, account_id: u32, namespace_id: u16, max_existing: u32) {
        self.collection_allocators
            .entry((account_id, namespace_id))
            .or_insert(AtomicU32::new(1))
            .fetch_max(max_existing + 1, RELAXED);
    }

    /// TD-PATH-2: raise a (possibly not-yet-created) account's namespace
    /// allocator above the durable identity high-water BEFORE minting — a
    /// fresh process has per-account allocators at 1, so without this the
    /// first mint after a restart could re-issue a DROPPED table's id.
    pub fn raise_namespace_floor(&self, account_id: u32, floor: u32) {
        // Clamp, never truncate: a foreign/corrupt sidecar value above the u16
        // space must not silently defeat the floor.
        let floor = u16::try_from(floor).unwrap_or(u16::MAX);
        self.namespace_allocators
            .entry(account_id)
            .or_insert(AtomicU16::new(1))
            .fetch_max(floor, RELAXED);
    }

    /// TD-PATH-2: raise a (possibly not-yet-created) `(account, namespace)`
    /// collection allocator above the durable identity high-water.
    pub fn raise_collection_floor(&self, account_id: u32, namespace_id: u16, floor: u32) {
        self.collection_allocators
            .entry((account_id, namespace_id))
            .or_insert(AtomicU32::new(1))
            .fetch_max(floor, RELAXED);
    }

    /// Recover the segment allocator floor for `collection_id`.
    pub fn recover_segment_floor(&self, collection_id: u32, max_existing: u32) {
        self.segment_allocators
            .entry(collection_id)
            .or_insert(AtomicU32::new(1))
            .fetch_max(max_existing + 1, RELAXED);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── IdAllocator tests ───────────────────────────────────────────────

    #[test]
    fn allocates_monotonic_distinct_ids_from_one() {
        let a = IdAllocator::default();
        assert_eq!(a.peek(), 1);
        assert_eq!(a.allocate(), 1);
        assert_eq!(a.allocate(), 2);
        assert_eq!(a.allocate(), 3);
        assert_eq!(a.peek(), 4, "peek does not consume");
    }

    #[test]
    fn zero_start_is_clamped_to_one() {
        // 0 is the reserved "unset" sentinel; allocation never returns it.
        let a = IdAllocator::new(0);
        assert_eq!(a.allocate(), 1);
    }

    #[test]
    fn raise_floor_recovers_high_water_and_is_monotone() {
        let a = IdAllocator::new(1);
        assert_eq!(a.allocate(), 1);
        assert_eq!(a.allocate(), 2);
        // Recovery: existing max id on disk is 7 → next must be 8.
        a.raise_floor(8);
        assert_eq!(
            a.allocate(),
            8,
            "never reuses ids below the recovered floor"
        );
        // A lower floor must never roll the counter backwards.
        a.raise_floor(3);
        assert_eq!(a.allocate(), 9);
    }

    #[test]
    fn per_type_spaces_are_independent() {
        // Two types (e.g. table vs namespace) each own their own space; the same
        // number can appear in both — they are never compared across types.
        let tables = IdAllocator::default();
        let namespaces = IdAllocator::default();
        assert_eq!(tables.allocate(), 1);
        assert_eq!(tables.allocate(), 2);
        assert_eq!(namespaces.allocate(), 1, "independent type space");
    }

    // ── CatalogIdService per-scope tests ────────────────────────────────

    #[test]
    fn namespace_ids_are_compact_per_account() {
        let svc = CatalogIdService::new();
        assert_eq!(svc.mint_namespace_id(1), 1);
        assert_eq!(svc.mint_namespace_id(1), 2);
        assert_eq!(svc.mint_namespace_id(1), 3);
    }

    #[test]
    fn different_accounts_restart_namespace_at_one() {
        let svc = CatalogIdService::new();
        assert_eq!(svc.mint_namespace_id(1), 1);
        assert_eq!(svc.mint_namespace_id(2), 1, "account 2 restarts at 1");
        assert_eq!(svc.mint_namespace_id(1), 2);
        assert_eq!(svc.mint_namespace_id(2), 2);
    }

    #[test]
    fn collection_ids_are_compact_per_namespace() {
        let svc = CatalogIdService::new();
        assert_eq!(svc.mint_collection_id(1, 1), 1);
        assert_eq!(svc.mint_collection_id(1, 1), 2);
        assert_eq!(
            svc.mint_collection_id(1, 2),
            1,
            "different namespace restarts at 1"
        );
    }

    #[test]
    fn segment_ids_are_per_collection() {
        let svc = CatalogIdService::new();
        assert_eq!(svc.mint_segment_id(1), 1);
        assert_eq!(svc.mint_segment_id(1), 2);
        assert_eq!(
            svc.mint_segment_id(2),
            1,
            "different collection restarts at 1"
        );
    }

    #[test]
    fn per_scope_recovery_prevents_reuse() {
        let svc = CatalogIdService::new();
        svc.mint_namespace_id(1);
        svc.mint_namespace_id(1);
        svc.recover_namespace_floor(1, 100);
        let next = svc.mint_namespace_id(1);
        assert!(next > 100, "after recovery, next must be >100, got {next}");
    }

    #[test]
    fn column_ids_are_compact_per_table() {
        let svc = CatalogIdService::new();
        assert_eq!(svc.mint_column_id(1), 1);
        assert_eq!(svc.mint_column_id(1), 2);
        assert_eq!(svc.mint_column_id(2), 1, "different table restarts at 1");
    }
}
