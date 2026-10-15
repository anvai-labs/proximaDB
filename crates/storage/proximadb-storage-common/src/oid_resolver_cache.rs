// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-DELVEC-1 WI-3c: in-memory OID→position resolver cache for cold-tier
//! deletion vectors.
//!
//! A sharded, byte-budgeted, recency-LRU cache of parsed `OidPositionResolver`s,
//! keyed by segment path (the final object-store URL). Mirrors the proven
//! `SegmentInvariantsCache` design (`segment_format.rs`) but simpler — single
//! tier (resolvers only), plain recency-LRU eviction — with an **independent**
//! byte budget so a delete storm can't starve the query-path invariants cache.
//!
//! Lazy-filled: empty on boot; the resolve path (`read_resolver`) loads a
//! resolver on first touch (cache miss) and `put`s it here. Compaction
//! invalidates on retire. This is the mechanism that makes the TD's O(segments)
//! delete-time scan tolerable across repeat deletes — in-memory `position_of`
//! probes instead of per-segment GETs.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use crate::oid_position_resolver::OidPositionResolver;

const OID_RESOLVER_CACHE_SHARDS: usize = 16;

/// Sharded, byte-budgeted, recency-LRU cache of per-segment OID→position
/// resolvers, keyed by segment path.
pub struct OidResolverCache {
    shards: Box<[RwLock<HashMap<String, ResolverEntry>>]>,
    bytes_used: AtomicUsize,
    /// Monotonic touch counter for recency (relaxed; eviction quality only).
    tick: AtomicU64,
    byte_budget: usize,
}

struct ResolverEntry {
    resolver: Arc<OidPositionResolver>,
    /// Tick of the last `get`/`put` touch (recency for eviction). Atomic so a
    /// hit updates it under the shard READ lock.
    last_hit: AtomicU64,
}

/// Rough in-memory footprint of a resident resolver (the `oids: Vec<String>` +
/// `by_oid: HashMap`). Exact sizing needs the resolver's private fields; this
/// bounds the budget (the `SegmentInvariantsCache` template uses the same
/// approximate approach with saturating arithmetic).
fn resolver_bytes(r: &OidPositionResolver) -> usize {
    // ~88 B/row: ~24 B String overhead + ~64 B HashMap bucket estimate.
    r.len() as usize * 88
}

impl OidResolverCache {
    /// `byte_budget` caps the total cached resolver bytes.
    pub fn new(byte_budget: usize) -> Self {
        let shards = (0..OID_RESOLVER_CACHE_SHARDS)
            .map(|_| RwLock::new(HashMap::new()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            shards,
            bytes_used: AtomicUsize::new(0),
            tick: AtomicU64::new(0),
            byte_budget,
        }
    }

    fn shard_for(&self, path: &str) -> &RwLock<HashMap<String, ResolverEntry>> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        path.hash(&mut h);
        &self.shards[(h.finish() as usize) % self.shards.len()]
    }

    fn next_tick(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn sub_bytes(&self, n: usize) {
        let _ = self
            .bytes_used
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(n))
            });
    }

    fn add_bytes(&self, n: usize) {
        self.bytes_used.fetch_add(n, Ordering::Relaxed);
    }

    /// On hit, return the cached resolver + stamp recency (shard READ lock only).
    pub fn get(&self, path: &str) -> Option<Arc<OidPositionResolver>> {
        let shard = self.shard_for(path).read().ok()?;
        let entry = shard.get(path)?;
        entry.last_hit.store(self.next_tick(), Ordering::Relaxed);
        Some(entry.resolver.clone())
    }

    /// Resident bytes currently held.
    pub fn bytes_used(&self) -> usize {
        self.bytes_used.load(Ordering::Relaxed)
    }

    /// Number of cached resolvers (diagnostic).
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.read().map(|shard| shard.len()).unwrap_or(0))
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert (replacing any existing entry for `path`); while over budget, evict
    /// the least-recently-touched entry. An entry larger than the whole budget is
    /// admitted anyway (mirrors the template's admit-anyway fallback) — eviction
    /// only helps when the entry itself fits.
    ///
    /// Victim scan: per-shard READ locks to find the global recency minimum, then
    /// confirm-and-remove under that shard's WRITE lock (no lock held across
    /// shards; a racing removal retries via the `while` loop).
    pub fn put(&self, path: String, resolver: Arc<OidPositionResolver>) {
        let entry_bytes = resolver_bytes(&resolver);
        {
            let Some(mut shard) = self.shard_for(&path).write().ok() else {
                return; // poisoned — best-effort cache, skip
            };
            if let Some(old) = shard.insert(
                path,
                ResolverEntry {
                    resolver,
                    last_hit: AtomicU64::new(self.next_tick()),
                },
            ) {
                self.sub_bytes(resolver_bytes(&old.resolver));
            }
        }
        self.add_bytes(entry_bytes);
        // Evict only if the entry itself fits the budget (oversize ⇒ admit anyway).
        if entry_bytes < self.byte_budget {
            while self.bytes_used.load(Ordering::Relaxed) > self.byte_budget {
                if !self.evict_one() {
                    break;
                }
            }
        }
    }

    /// Find + remove the least-recent entry across all shards. Returns false if
    /// the cache is empty.
    fn evict_one(&self) -> bool {
        let mut victim_key: Option<String> = None;
        let mut victim_tick = u64::MAX;
        let mut victim_shard_idx: Option<usize> = None;
        for (i, shard_lock) in self.shards.iter().enumerate() {
            if let Ok(shard) = shard_lock.read() {
                for (k, e) in shard.iter() {
                    let t = e.last_hit.load(Ordering::Relaxed);
                    if t < victim_tick {
                        victim_tick = t;
                        victim_key = Some(k.clone());
                        victim_shard_idx = Some(i);
                    }
                }
            }
        }
        let (idx, key) = match (victim_shard_idx, victim_key) {
            (Some(i), Some(k)) => (i, k),
            _ => return false,
        };
        let Some(mut shard) = self.shards[idx].write().ok() else {
            return false; // poisoned — caller retries
        };
        if let Some(entry) = shard.remove(&key) {
            self.sub_bytes(resolver_bytes(&entry.resolver));
            true
        } else {
            false // raced — the while loop retries
        }
    }

    /// Remove a single entry (compaction invalidation on segment retire).
    pub fn invalidate(&self, path: &str) {
        let Some(mut shard) = self.shard_for(path).write().ok() else {
            return; // poisoned — best-effort, skip
        };
        if let Some(entry) = shard.remove(path) {
            self.sub_bytes(resolver_bytes(&entry.resolver));
        }
    }

    /// Remove all entries whose key starts with `prefix` (collection-drop / bulk
    /// retire — TD-DELVEC-1 C3). Mirrors `invalidate` but scans every shard +
    /// matches by prefix. Miss-safe (no-op if nothing matches).
    pub fn invalidate_prefix(&self, prefix: &str) {
        for shard_lock in self.shards.iter() {
            let Some(mut shard) = shard_lock.write().ok() else {
                continue;
            };
            let to_remove: Vec<String> = shard
                .keys()
                .filter(|k| k.starts_with(prefix))
                .cloned()
                .collect();
            for key in to_remove {
                if let Some(entry) = shard.remove(&key) {
                    self.sub_bytes(resolver_bytes(&entry.resolver));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolver(n: usize) -> Arc<OidPositionResolver> {
        Arc::new(OidPositionResolver::from_stream_order(
            (0..n).map(|i| format!("oid-{i}")).collect(),
        ))
    }

    #[test]
    fn put_get_round_trips_and_tracks_bytes() {
        let cache = OidResolverCache::new(1024 * 1024);
        assert!(cache.is_empty());
        cache.put("seg-a".into(), resolver(10));
        assert_eq!(cache.len(), 1);
        let r = cache.get("seg-a").expect("hit");
        assert_eq!(r.len(), 10);
        assert_eq!(r.position_of("oid-3"), Some(3));
        assert!(cache.bytes_used() > 0);
        assert!(cache.get("seg-missing").is_none());
    }

    #[test]
    fn evicts_least_recent_under_budget() {
        // 10 rows × 88 B = 880 B; budget 1000 B fits one resolver.
        let cache = OidResolverCache::new(1000);
        cache.put("seg-a".into(), resolver(10));
        assert!(cache.get("seg-a").is_some()); // touch a → most recent
        cache.put("seg-b".into(), resolver(10)); // over budget → evict least-recent (a)
        assert!(cache.get("seg-a").is_none(), "least-recent seg-a evicted");
        assert!(cache.get("seg-b").is_some(), "seg-b resident");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn invalidate_removes_entry() {
        let cache = OidResolverCache::new(1024 * 1024);
        cache.put("seg-a".into(), resolver(5));
        assert!(cache.get("seg-a").is_some());
        cache.invalidate("seg-a");
        assert!(cache.get("seg-a").is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn invalidate_prefix_removes_matching_keeps_others() {
        // TD-DELVEC-1 C3: collection-drop invalidates all entries under a prefix.
        let cache = OidResolverCache::new(1024 * 1024);
        cache.put("file:///data/col-a/seg-1.pax".into(), resolver(3));
        cache.put("file:///data/col-a/seg-2.pax".into(), resolver(3));
        cache.put("file:///data/col-b/seg-1.pax".into(), resolver(3));
        assert_eq!(cache.len(), 3);

        // Drop collection col-a → its two resolvers invalidated; col-b survives.
        cache.invalidate_prefix("file:///data/col-a/");
        assert!(cache.get("file:///data/col-a/seg-1.pax").is_none());
        assert!(cache.get("file:///data/col-a/seg-2.pax").is_none());
        assert!(
            cache.get("file:///data/col-b/seg-1.pax").is_some(),
            "col-b must survive"
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn replace_credits_old_bytes() {
        let cache = OidResolverCache::new(1024 * 1024);
        cache.put("seg-a".into(), resolver(10));
        let before = cache.bytes_used();
        cache.put("seg-a".into(), resolver(20)); // replace with larger
        let after = cache.bytes_used();
        assert!(
            after > before,
            "replace with larger resolver grows the budget"
        );
        assert_eq!(cache.get("seg-a").unwrap().len(), 20);
    }

    #[test]
    fn oversize_entry_admitted_anyway() {
        // Budget 100 B; a 10-row resolver = 880 B (>> budget) is admitted.
        let cache = OidResolverCache::new(100);
        cache.put("seg-big".into(), resolver(10));
        assert!(
            cache.get("seg-big").is_some(),
            "oversize entry admitted even though it exceeds the budget"
        );
    }
}
