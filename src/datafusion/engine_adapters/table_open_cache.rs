// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Table-OPEN cache (TD-OLAP-4): memoize the per-query parquet *discovery*
//! (LIST + HEAD-per-file + footer read) so repeat queries against the same
//! external/materialized base skip that fixed floor.
//!
//! ## Why this exists
//! Measurement (1M ClickBench head-to-head vs DuckDB) showed every query paid a
//! ~20–40 ms table-OPEN floor: [`ObjectStoreParquetTable::open`] re-runs LIST +
//! HEAD + footer-read on *every* SELECT, even though the discovery result is a
//! pure function of the immutable parquet snapshot. That floor — not I/O bytes,
//! not the engine — is the median gap. Caching the discovery removes it on warm.
//!
//! ## What is cached — and what is NOT
//! Only the **footer-derived metadata** ([`TableOpenDiscovery`]: Arrow schema +
//! row-group [`FileSplit`]s + per-file byte sizes). The object store is
//! deliberately **NOT** cached: [`crate::observability::object_store_trace::TracingObjectStore`]
//! captures the per-query io-trace handle at construction, so a cached store
//! would misattribute every later query's `bytes_read` (billing) to the first.
//! The caller rebuilds + re-wraps the store fresh per query (cheap, no I/O) and
//! assembles the table from the cached metadata.
//!
//! ## Correctness / freshness
//! External parquet is immutable, so a cache entry is valid until the snapshot
//! changes. `ALTER TABLE … MATERIALIZE` republishes a table's base and MUST
//! [`invalidate_location`] it (wired in the DDL handler). Keyed by
//! `(tenant, location)` — isolation is structural (a cross-tenant read can never
//! hit another tenant's entry). Gated **default-OFF** via
//! `PROXIMADB_OLAP_TABLE_CACHE` per the default-OFF-until-baked mandate.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use arrow_schema::SchemaRef;
use parking_lot::RwLock;
use proximadb_iceberg_engine::deletion_vector::PositionDeletes;
use proximadb_storage_common::format_splits::FileSplit;

/// Immutable footer-derived discovery result for one parquet base/object: the
/// output of `open_files` minus the (per-query) object store. Cheap to clone
/// (Arc'd schema; the split vec is small — one entry per row group).
#[derive(Clone)]
pub struct TableOpenDiscovery {
    /// Arrow schema inferred from the parquet footer (first file wins).
    pub schema: SchemaRef,
    /// Row-group splits with their footer statistics (one entry per row group).
    pub splits: Vec<FileSplit>,
    /// Per-file byte sizes, keyed by the full object path.
    pub file_sizes: HashMap<String, u64>,
    /// The snapshot's decoded deletion vectors, keyed by the full data-file
    /// path (empty for snapshots without vectors). Part of the same immutable
    /// discovery as the splits — a cache hit without them would serve deleted
    /// rows.
    pub deletions: HashMap<String, Arc<PositionDeletes>>,
}

/// Cache key: tenant identity (fail-closed to `""` when single-tenant) + the
/// canonical object-store location. Keeping the tenant in the key makes
/// isolation structural rather than a per-lookup predicate.
type Key = (String, String);

/// Bound on distinct cached bases. External/materialized tables per tenant are
/// few; when the map exceeds this it is cleared wholesale (crude but bounded —
/// a metadata cache, not a hot-path result cache).
const MAX_ENTRIES: usize = 1024;

fn cache() -> &'static RwLock<HashMap<Key, Arc<TableOpenDiscovery>>> {
    static CACHE: OnceLock<RwLock<HashMap<Key, Arc<TableOpenDiscovery>>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Whether the table-OPEN cache is enabled (`PROXIMADB_OLAP_TABLE_CACHE=1`).
/// Default-OFF; evaluated once.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PROXIMADB_OLAP_TABLE_CACHE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn key(tenant: Option<&str>, location: &str) -> Key {
    (tenant.unwrap_or("").to_string(), location.to_string())
}

/// Look up the cached discovery for `(tenant, location)`, if present.
pub fn get(tenant: Option<&str>, location: &str) -> Option<Arc<TableOpenDiscovery>> {
    if !enabled() {
        return None;
    }
    cache().read().get(&key(tenant, location)).cloned()
}

/// Insert (or replace) the discovery for `(tenant, location)`.
pub fn insert(tenant: Option<&str>, location: &str, discovery: TableOpenDiscovery) {
    if !enabled() {
        return;
    }
    let mut map = cache().write();
    if map.len() >= MAX_ENTRIES && !map.contains_key(&key(tenant, location)) {
        map.clear();
    }
    map.insert(key(tenant, location), Arc::new(discovery));
}

/// Invalidate every cached entry for `location` across all tenants — called
/// when `ALTER TABLE … MATERIALIZE` republishes a base at that location, so a
/// stale snapshot is never reused. A no-op when the cache is disabled/empty.
pub fn invalidate_location(location: &str) {
    if cache().read().is_empty() {
        return;
    }
    cache().write().retain(|(_, loc), _| loc != location);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_enabled(on: bool) {
        // The `enabled()` gate memoizes on first read; these tests exercise the
        // map directly (bypassing the gate) so they are order-independent.
        let _ = on;
    }

    fn empty_discovery() -> TableOpenDiscovery {
        TableOpenDiscovery {
            schema: Arc::new(arrow_schema::Schema::empty()),
            splits: Vec::new(),
            file_sizes: HashMap::new(),
            deletions: HashMap::new(),
        }
    }

    #[test]
    fn invalidate_location_drops_only_that_location_all_tenants() {
        set_enabled(true);
        // Seed two tenants at one location + a third location; direct map access
        // so the test does not depend on the memoized env gate.
        let mut map = cache().write();
        map.insert(("t1".into(), "loc/a".into()), Arc::new(empty_discovery()));
        map.insert(("t2".into(), "loc/a".into()), Arc::new(empty_discovery()));
        map.insert(("t1".into(), "loc/b".into()), Arc::new(empty_discovery()));
        drop(map);

        invalidate_location("loc/a");

        let map = cache().read();
        assert!(!map.contains_key(&("t1".to_string(), "loc/a".to_string())));
        assert!(!map.contains_key(&("t2".to_string(), "loc/a".to_string())));
        assert!(
            map.contains_key(&("t1".to_string(), "loc/b".to_string())),
            "a different location is untouched"
        );
    }
}
