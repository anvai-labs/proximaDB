// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-DELVEC-1 WI-3c-1c: the oid→segment→position resolve API + lazy-load.
//!
//! Feature-gated `cold-deletion-vectors`. `resolve_oid_positions` is the
//! O(segments) delete-time scan: list the collection's `.pax` files, probe each
//! segment's cached resolver (`OidResolverCache`, WI-3c-1a/b), lazy-load on miss
//! (read the footer region via `fs.read_range`, mirroring Region-A/B). Returns
//! ALL hits (MVCC: an oid may be in >1 segment).

use crate::storage::engines::sst::segment_io::ResidentRangeRead as _;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use proximadb_storage_common::oid_position_resolver::OidPositionResolver;
use proximadb_storage_common::pax_block::SEGMENT_MAGIC;
use proximadb_storage_common::segment_layout::{SEG_TAIL_LEN, SegmentFooterIndex};

use crate::storage::engines::sst::oid_resolver_cache::OidResolverCache;
use crate::storage::persistence::filesystem::FilesystemFactory;

impl super::core::SstEngine {
    /// TD-DELVEC-1 WI-3c: resolve an oid to all `(segment_path, position)` hits
    /// across the collection's cold `.pax` segments. O(segments) in-memory probes
    /// (cache hits) + lazy footer-region reads on miss. Returns ALL hits (MVCC).
    pub async fn resolve_oid_positions(
        &self,
        collection_id: &str,
        oid: &str,
    ) -> Result<Vec<(String, u32)>> {
        let Some(ref cache) = self.oid_resolver_cache else {
            return Ok(Vec::new()); // cache disabled → tombstone fallback
        };
        let files = self.list_collection_files(collection_id).await?;
        let mut hits = Vec::new();
        for path in &files {
            if !path.ends_with(".pax") {
                continue;
            }
            if let Some(resolver) = self.get_or_load_resolver(path, cache).await?
                && let Some(pos) = resolver.position_of(oid)
            {
                hits.push((path.clone(), pos));
            }
        }
        Ok(hits)
    }

    /// Cache-then-lazy-load a segment's resolver. On cache miss, reads the
    /// footer region via `fs.read_range` (mirrors Region-A/B) + deserializes.
    async fn get_or_load_resolver(
        &self,
        path: &str,
        cache: &OidResolverCache,
    ) -> Result<Option<Arc<OidPositionResolver>>> {
        if let Some(r) = cache.get(path) {
            return Ok(Some(r)); // HIT
        }
        let loaded = self.read_resolver(path).await?;
        if let Some(ref r) = loaded {
            cache.put(path.to_string(), r.clone());
        }
        Ok(loaded)
    }

    /// Lazy-load a resolver from the segment's footer region. Reads the tail
    /// `[footer_len u64 | SEGMENT_MAGIC 8B]` to locate the footer, then the
    /// footer body (→ `opr_off`/`opr_len`), then the region bytes → deserialize
    /// (CRC-validated). Returns `None` if the segment has no resolver region
    /// (`opr_len == 0` → tombstone fallback) or isn't a coalesced PAX segment.
    async fn read_resolver(&self, path: &str) -> Result<Option<Arc<OidPositionResolver>>> {
        read_segment_resolver(self.filesystem(), path).await
    }
}

/// Lazy-load a resolver from the segment's footer region — a free fn usable by
/// callers that don't hold an `SstEngine` (e.g. compaction, TD-DELVEC-1 WI-6).
/// Reads the tail `[footer_len u64 | SEGMENT_MAGIC 8B]` to locate the footer,
/// then the footer body (→ `opr_off`/`opr_len`), then the region bytes →
/// deserialize (CRC-validated). Returns `None` if the segment has no resolver
/// region (`opr_len == 0` → tombstone fallback) or isn't a coalesced PAX segment.
pub async fn read_segment_resolver(
    filesystem: &FilesystemFactory,
    path: &str,
) -> Result<Option<Arc<OidPositionResolver>>> {
    let storage_url = path.to_string();
    let fs = filesystem
        .get_filesystem(&storage_url)
        .map_err(|e| anyhow!("resolver load: filesystem for {storage_url}: {e:?}"))?;

    // File size (to read the tail). A resident segment answers from its
    // copy (TD-TIER-1 SC4-4: no HEAD on a READY collection).
    let file_size = fs
        .object_size_resident(&storage_url)
        .await
        .map_err(|e| anyhow!("resolver load: metadata for {storage_url}: {e:?}"))?;
    if file_size < SEG_TAIL_LEN as u64 {
        return Ok(None); // too small
    }

    // Read the tail [footer_len u64 | SEGMENT_MAGIC 8B] to locate the footer.
    let tail_off = file_size - SEG_TAIL_LEN as u64;
    let tail = fs
        .read_range_resident(&storage_url, tail_off, SEG_TAIL_LEN as u64)
        .await
        .map_err(|e| anyhow!("resolver load: tail read for {storage_url}: {e:?}"))?;
    if &tail[8..16] != SEGMENT_MAGIC {
        return Ok(None); // not a coalesced PAX segment
    }
    let footer_len_bytes: [u8; 8] = tail[..8]
        .try_into()
        .map_err(|_| anyhow!("resolver load: bad tail for {storage_url}"))?;
    let footer_len = u64::from_le_bytes(footer_len_bytes) as usize;
    if footer_len == 0 || (tail_off as usize) < footer_len {
        return Ok(None);
    }

    // Read + parse the footer.
    let footer_off = tail_off - footer_len as u64;
    let footer_bytes = fs
        .read_range_resident(&storage_url, footer_off, footer_len as u64)
        .await
        .map_err(|e| anyhow!("resolver load: footer read for {storage_url}: {e:?}"))?;
    let footer = SegmentFooterIndex::parse(&footer_bytes)
        .map_err(|e| anyhow!("resolver load: footer parse for {storage_url}: {e}"))?;

    // No resolver region → None (tombstone fallback).
    if footer.opr_len == 0 {
        return Ok(None);
    }

    // Read + deserialize the resolver region (CRC-validated by deserialize).
    let region = fs
        .read_range_resident(&storage_url, footer.opr_off, footer.opr_len)
        .await
        .map_err(|e| anyhow!("resolver load: region read for {storage_url}: {e:?}"))?;
    let resolver = OidPositionResolver::deserialize(&region)
        .map_err(|e| anyhow!("resolver load: deserialize for {storage_url}: {e}"))?;
    Ok(Some(Arc::new(resolver)))
}
