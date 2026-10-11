/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

//! REST operator endpoints for cache-affinity (Phase 7.2.4).
//!
//! The cache-affinity registry tracks "which node most recently
//! served queries for this collection" so reads can be biased to
//! the warm-cache holder (see `src/cluster/cache_affinity.rs`).
//! These endpoints give operators a way to inspect the registry and
//! invalidate entries — useful when a known cache eviction has
//! happened and the affinity hint is stale, or when an operator
//! wants to force a routing re-evaluation.
//!
//! Routes:
//!
//! * `GET /api/v2/collections/:collection_id/affinity` — read one
//!   entry. Returns `{"status":"affinitized", ...}` when the entry
//!   is present, regardless of TTL freshness (operator dashboards
//!   want to see stale entries too — `stale` field flags them).
//!   Returns `{"status":"not_affinitized", ...}` when the collection
//!   has no entry at all.
//! * `DELETE /api/v2/collections/:collection_id/affinity` — drop the
//!   entry for a collection. 200 with `dropped: true|false`.
//! * `GET /api/v2/collections/affinity` — list every entry on this
//!   node, sorted by collection_id. Includes stale entries so
//!   operators can see which collections went cold.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use serde::Serialize;

use crate::cluster::cache_affinity::AffinityEntry;
use crate::network::rest::canonical::handlers::AppState;

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AffinityResponse {
    Affinitized {
        collection_id: String,
        node_id: String,
        /// Monotonic count of queries served by this node for the
        /// collection while it remained the affinity holder.
        query_count: u64,
        /// Seconds since the most recent recorded query. Operators
        /// can compare this against the TTL to gauge freshness
        /// without parsing absolute timestamps.
        age_seconds: u64,
        /// True when the entry is older than the registry TTL.
        /// Routing has already stopped using stale entries, but the
        /// inspect view still surfaces them so an operator can see
        /// "collection X went cold."
        stale: bool,
    },
    NotAffinitized {
        collection_id: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AffinityDeleteResponse {
    Dropped {
        collection_id: String,
        dropped: bool,
    },
}

#[derive(Debug, Serialize)]
pub struct AffinityListItem {
    pub collection_id: String,
    pub node_id: String,
    pub query_count: u64,
    pub age_seconds: u64,
    pub stale: bool,
}

#[derive(Debug, Serialize)]
pub struct AffinityListResponse {
    pub count: usize,
    pub items: Vec<AffinityListItem>,
}

/// Render an `AffinityEntry` into the API shape. `ttl_seconds` is
/// the registry's configured TTL; entries older than that are
/// flagged `stale`.
fn entry_to_item(
    collection_id: String,
    entry: &AffinityEntry,
    ttl_seconds: u64,
) -> AffinityListItem {
    let age = entry.last_seen.elapsed();
    let age_seconds = age.as_secs();
    AffinityListItem {
        collection_id,
        node_id: entry.node_id.clone(),
        query_count: entry.query_count,
        age_seconds,
        stale: age_seconds > ttl_seconds,
    }
}

/// `GET /api/v2/collections/:collection_id/affinity`
pub async fn get_affinity(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
) -> Result<Json<AffinityResponse>, (StatusCode, String)> {
    let registry = &state.affinity_registry;

    match registry.entry(&collection_id) {
        Some(entry) => {
            let age_seconds = entry.last_seen.elapsed().as_secs();
            // The TTL isn't directly exposed by the registry; we
            // treat anything that `preferred_node` doesn't surface
            // as stale. Both checks happen against the same
            // current-time read.
            let stale = registry.preferred_node(&collection_id).is_none();
            Ok(Json(AffinityResponse::Affinitized {
                collection_id,
                node_id: entry.node_id,
                query_count: entry.query_count,
                age_seconds,
                stale,
            }))
        }
        None => Ok(Json(AffinityResponse::NotAffinitized { collection_id })),
    }
}

/// `DELETE /api/v2/collections/:collection_id/affinity`
pub async fn delete_affinity(
    State(state): State<AppState>,
    Path(collection_id): Path<String>,
) -> Result<Json<AffinityDeleteResponse>, (StatusCode, String)> {
    let registry = &state.affinity_registry;
    let dropped = registry.invalidate(&collection_id);
    Ok(Json(AffinityDeleteResponse::Dropped {
        collection_id,
        dropped,
    }))
}

/// `GET /api/v2/collections/affinity` — operator dashboard list.
pub async fn list_affinity(
    State(state): State<AppState>,
) -> Result<Json<AffinityListResponse>, (StatusCode, String)> {
    let registry = &state.affinity_registry;

    let listed = registry.list();
    // Determine staleness per-entry using the registry's own TTL
    // semantics (preferred_node returns None for stale entries).
    let items: Vec<AffinityListItem> = listed
        .into_iter()
        .map(|(collection_id, entry)| {
            // `has_fresh_affinity` checks TTL without allocating.
            let stale = !registry.has_fresh_affinity(&collection_id);
            let age_seconds = entry.last_seen.elapsed().as_secs();
            AffinityListItem {
                collection_id,
                node_id: entry.node_id,
                query_count: entry.query_count,
                age_seconds,
                stale,
            }
        })
        .collect();
    let count = items.len();
    Ok(Json(AffinityListResponse { count, items }))
}

/// Convenience for tests / future use cases that want a single
/// helper rather than per-field marshalling.
#[allow(dead_code)]
fn _entry_to_item_keep_in_scope() {
    let entry = AffinityEntry {
        node_id: "x".into(),
        last_seen: std::time::Instant::now(),
        query_count: 0,
    };
    let _ = entry_to_item("c".into(), &entry, 60);
}

#[cfg(test)]
mod tests {
    //! End-to-end inspection of the operator API is exercised via
    //! the routing/registry integration tests. These tests focus on
    //! the small pure-data conversion that's not covered there:
    //! `entry_to_item` correctly flags staleness from elapsed age.

    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn entry_to_item_marks_fresh_below_ttl() {
        let entry = AffinityEntry {
            node_id: "node-a".into(),
            last_seen: Instant::now(),
            query_count: 7,
        };
        let item = entry_to_item("coll".into(), &entry, 60);
        assert_eq!(item.node_id, "node-a");
        assert_eq!(item.query_count, 7);
        assert!(!item.stale);
    }

    #[test]
    fn entry_to_item_marks_stale_above_ttl() {
        // Manually back-date by subtracting the TTL+1s.
        let entry = AffinityEntry {
            node_id: "node-a".into(),
            last_seen: Instant::now() - Duration::from_secs(61),
            query_count: 1,
        };
        let item = entry_to_item("coll".into(), &entry, 60);
        assert!(item.stale, "61s entry must be flagged stale at TTL=60s");
        assert!(
            item.age_seconds >= 61,
            "age_seconds must reflect elapsed time"
        );
    }
}
