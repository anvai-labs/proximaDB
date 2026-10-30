//! MVCC (Multi-Version Concurrency Control) Resolution for ProximaDB
//!
//! Centralized logic for resolving record versions according to MVCC rules:
//! 1. Records with valid_to_ns < current_time_ns are considered deleted (expired)
//! 2. Any record with non-empty ID and an expired version marks all versions deleted
//! 3. Records with empty OID are append-only (no versioning)
//! 4. For records with same ID, highest version without gaps wins
//! 5. record_version == 0 is treated as version 1 (legacy compatibility)
//! 6. For same version, earliest timestamp wins

use proximadb_records::ProximaRecord;
use std::collections::HashMap;
use tracing::debug;

/// MVCC resolver for ProximaRecord instances.
pub struct MvccResolver {
    /// Current timestamp in nanoseconds for expiry checks.
    current_timestamp_ns: i64,
}

impl MvccResolver {
    /// Create a new resolver using the system clock.
    pub fn new() -> Self {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64;
        Self {
            current_timestamp_ns: ns,
        }
    }

    /// Create with a specific nanosecond timestamp (for deterministic tests).
    pub fn with_timestamp_ns(ns: i64) -> Self {
        Self {
            current_timestamp_ns: ns,
        }
    }

    /// Backward-compat constructor — accepts seconds, converts to nanoseconds.
    pub fn with_timestamp(seconds: u32) -> Self {
        Self::with_timestamp_ns(seconds as i64 * 1_000_000_000)
    }

    // -----------------------------------------------------------------------
    // ProximaRecord API (canonical)
    // -----------------------------------------------------------------------

    /// Resolve a batch of `ProximaRecord`s according to MVCC rules.
    pub fn resolve_batch(&self, records: Vec<ProximaRecord>) -> Vec<ProximaRecord> {
        let mut id_groups: HashMap<String, Vec<ProximaRecord>> = HashMap::new();
        let mut append_only: Vec<ProximaRecord> = Vec::new();

        for record in records {
            let oid = &record.oid;
            if is_append_only_oid(oid) {
                append_only.push(record);
            } else {
                id_groups.entry(oid.clone()).or_default().push(record);
            }
        }

        let mut resolved = Vec::new();

        for (id, mut versions) in id_groups {
            // Any expired version marks the entire ID as deleted
            if versions.iter().any(|r| self.is_expired(r)) {
                debug!("MVCC: all versions of '{}' deleted due to expiry", id);
                continue;
            }

            // Sort ascending by (effective_version, created_at_ns)
            versions.sort_by(|a, b| {
                effective_version(a)
                    .cmp(&effective_version(b))
                    .then_with(|| a.created_at_ns.cmp(&b.created_at_ns))
            });

            // Require the sequence to start at 0 or 1. `versions.first()` is
            // safe to unwrap because the Vec was built via
            // `id_groups.entry(oid).or_default().push(record)`, so every
            // entry in `id_groups` has at least one record by construction.
            #[allow(clippy::unwrap_used)]
            let start = effective_version(versions.first().unwrap());
            if start > 1 {
                debug!(
                    "MVCC: version gap from start for '{}': starts at {}",
                    id, start
                );
                continue;
            }

            let mut expected = start;
            let mut last_valid: Option<ProximaRecord> = None;

            for record in versions {
                let ver = effective_version(&record);
                if ver == expected {
                    last_valid = Some(record);
                    expected += 1;
                } else if ver > expected {
                    debug!(
                        "MVCC: version gap for '{}': expected {}, found {}",
                        id, expected, ver
                    );
                    break;
                }
                // ver < expected → older duplicate; skip
            }

            if let Some(record) = last_valid {
                debug!(
                    "MVCC: selected version {} for '{}'",
                    effective_version(&record),
                    id
                );
                resolved.push(record);
            }
        }

        for record in append_only {
            if !self.is_expired(&record) {
                resolved.push(record);
            }
        }

        resolved
    }

    /// Resolve an owned compaction batch with bounded auxiliary memory.
    ///
    /// The general query-path resolver above intentionally accepts arbitrary
    /// input order, but its `HashMap<String, Vec<...>>` shape is pathological
    /// for compaction: a 10M-row segment with unique OIDs allocates 10M map
    /// entries and 10M one-element vectors while the full input remains live.
    /// Compaction already owns the entire batch, so it can sort by OID and scan
    /// contiguous version groups instead.
    ///
    /// This implementation:
    ///
    /// * sorts records in place with the allocation-free unstable sorter;
    /// * marks at most one winner per versioned OID in a one-byte-per-row mask;
    /// * compacts winners in the original allocation with `Vec::retain`;
    /// * never clones an embedding, OID, property tree, or record.
    ///
    /// Output is deterministic by OID. Within an OID, effective version then
    /// creation/update time provide the tie-break order; append-only rows use
    /// the same deterministic temporal order. The MVCC rules match
    /// [`Self::resolve_batch`].
    pub fn resolve_sorted_batch(&self, mut records: Vec<ProximaRecord>) -> Vec<ProximaRecord> {
        records.sort_unstable_by(|left, right| {
            left.oid
                .cmp(&right.oid)
                .then_with(|| effective_version(left).cmp(&effective_version(right)))
                .then_with(|| left.created_at_ns.cmp(&right.created_at_ns))
                .then_with(|| left.updated_at_ns.cmp(&right.updated_at_ns))
        });

        let mut keep = vec![false; records.len()];
        let mut group_start = 0usize;
        while group_start < records.len() {
            let group_oid = records[group_start].oid.as_str();
            let mut group_end = group_start + 1;
            while group_end < records.len() && records[group_end].oid == group_oid {
                group_end += 1;
            }

            if is_append_only_oid(group_oid) {
                for index in group_start..group_end {
                    keep[index] = !self.is_expired(&records[index]);
                }
                group_start = group_end;
                continue;
            }

            // Any expired version is a deletion marker for the entire OID.
            if records[group_start..group_end]
                .iter()
                .any(|record| self.is_expired(record))
            {
                group_start = group_end;
                continue;
            }

            let start_version = effective_version(&records[group_start]);
            if start_version > 1 {
                group_start = group_end;
                continue;
            }

            let mut expected = start_version;
            let mut winner = None;
            for (offset, record) in records[group_start..group_end].iter().enumerate() {
                let version = effective_version(record);
                if version == expected {
                    winner = Some(group_start + offset);
                    expected += 1;
                } else if version > expected {
                    break;
                }
                // version < expected is an older duplicate of a version that
                // already won through the deterministic timestamp ordering.
            }
            if let Some(index) = winner {
                keep[index] = true;
            }
            group_start = group_end;
        }

        let mut index = 0usize;
        records.retain(|_| {
            let retain = keep[index];
            index += 1;
            retain
        });
        records
    }

    /// Return `true` if the record's TTL has elapsed.
    pub fn is_expired(&self, record: &ProximaRecord) -> bool {
        record
            .valid_to_ns
            .is_some_and(|vt| vt < self.current_timestamp_ns)
    }

    /// Return `true` if `r1` should win over `r2` (for same-ID tie-breaking).
    ///
    /// Priority: non-expired > higher version > earlier timestamp.
    pub fn compare_records(&self, r1: &ProximaRecord, r2: &ProximaRecord) -> bool {
        let e1 = self.is_expired(r1);
        let e2 = self.is_expired(r2);

        match (e1, e2) {
            (true, false) => return false,
            (false, true) => return true,
            _ => {}
        }

        let v1 = effective_version(r1);
        let v2 = effective_version(r2);

        if v1 != v2 {
            return v1 > v2;
        }

        // Same version — earliest creation timestamp wins
        r1.created_at_ns <= r2.created_at_ns
    }
}

/// Treat `record_version == 0` as version 1 for historical stored records.
#[inline]
pub fn effective_version(r: &ProximaRecord) -> u64 {
    if r.record_version == 0 {
        1
    } else {
        r.record_version
    }
}

#[inline]
pub fn is_append_only_oid(oid: &str) -> bool {
    oid.is_empty() || oid == "null" || oid == "none" || oid.trim().is_empty()
}

impl Default for MvccResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::{EmbeddingCell, LabelSet, ProximaTree};

    fn create_record(
        id: Option<&str>,
        version: u64,
        created_at_s: i64,
        valid_to_s: Option<i64>,
    ) -> ProximaRecord {
        ProximaRecord {
            oid: id.unwrap_or("").to_string(),
            local_id: None,
            tid: None,
            variation_id: None,
            record_version: version,
            spec_version: 1,
            tenant_id: String::new(),
            permitted_principals: Vec::new(),
            rls_policy_id: None,
            created_at_ns: created_at_s * 1_000_000_000,
            updated_at_ns: created_at_s * 1_000_000_000,
            valid_from_ns: None,
            valid_to_ns: valid_to_s.map(|s| s * 1_000_000_000),
            origin: None,
            actor: None,
            method: None,
            memory_type: None,
            props: ProximaTree::new(),
            refs: Vec::new(),
            edge: None,
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "dense_vector",
                3,
                vec![1.0, 2.0, 3.0],
            )],
            sequence: None,
            labels: LabelSet::new(),
            ..Default::default()
        }
    }

    #[test]
    fn test_version_continuity() {
        // current time = 1000 s
        let resolver = MvccResolver::with_timestamp(1000);

        let records = vec![
            create_record(Some("v1"), 1, 100, None),
            create_record(Some("v1"), 3, 300, None), // Gap!
            create_record(Some("v1"), 2, 200, None),
        ];

        let resolved = resolver.resolve_batch(records);

        // Continuous sequence 1→2→3, highest wins
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].record_version, 3);
    }

    #[test]
    fn test_expiry_handling() {
        let resolver = MvccResolver::with_timestamp(1000);

        let records = vec![
            create_record(Some("v1"), 1, 100, None),
            create_record(Some("v1"), 2, 200, Some(500)), // expires at 500 s
            create_record(Some("v1"), 3, 300, None),
        ];

        let resolved = resolver.resolve_batch(records);
        assert_eq!(resolved.len(), 0); // All excluded due to one expired version
    }

    #[test]
    fn test_append_only_records() {
        let resolver = MvccResolver::with_timestamp(1000);

        let records = vec![
            create_record(None, 0, 100, None),
            create_record(Some(""), 0, 200, None),
            create_record(Some("null"), 0, 300, None),
            create_record(Some("  "), 0, 400, None),
        ];

        let resolved = resolver.resolve_batch(records);
        assert_eq!(resolved.len(), 4);
    }

    #[test]
    fn test_same_version_timestamp_resolution() {
        let resolver = MvccResolver::with_timestamp(1000);

        let records = vec![
            create_record(Some("v1"), 1, 200, None),
            create_record(Some("v1"), 1, 100, None), // Earlier wins
            create_record(Some("v1"), 1, 300, None),
        ];

        let resolved = resolver.resolve_batch(records);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].created_at_ns, 100 * 1_000_000_000);
    }

    #[test]
    fn test_multiple_ids_different_versions() {
        let resolver = MvccResolver::with_timestamp(1000);

        let records = vec![
            create_record(Some("id1"), 1, 100, None),
            create_record(Some("id1"), 2, 200, None),
            create_record(Some("id2"), 1, 150, None),
            create_record(Some("id2"), 3, 350, None), // Gap at version 2
            create_record(Some("id3"), 5, 500, None), // Starts at 5 — gap from beginning
        ];

        let resolved = resolver.resolve_batch(records);

        // id1 v2, id2 v1 (gap stops it), id3 excluded
        assert_eq!(resolved.len(), 2);

        let by_id: HashMap<String, &ProximaRecord> =
            resolved.iter().map(|r| (r.oid.clone(), r)).collect();
        assert_eq!(by_id["id1"].record_version, 2);
        assert_eq!(by_id["id2"].record_version, 1);
        assert!(!by_id.contains_key("id3"));
    }

    #[test]
    fn test_version_zero_treated_as_one() {
        let resolver = MvccResolver::with_timestamp(1000);

        let records = vec![
            create_record(Some("id1"), 0, 100, None), // treated as version 1
            create_record(Some("id1"), 2, 200, None),
            create_record(Some("id2"), 0, 150, None),
            create_record(Some("id2"), 0, 120, None), // same effective v1, earlier wins
        ];

        let resolved = resolver.resolve_batch(records);
        assert_eq!(resolved.len(), 2);

        let by_id: HashMap<String, &ProximaRecord> =
            resolved.iter().map(|r| (r.oid.clone(), r)).collect();
        assert_eq!(by_id["id1"].record_version, 2);
        assert_eq!(by_id["id2"].created_at_ns, 120 * 1_000_000_000);
    }

    #[test]
    fn test_expired_records_mark_entire_id_deleted() {
        let resolver = MvccResolver::with_timestamp(1000);

        let records = vec![
            create_record(Some("id1"), 1, 100, None),
            create_record(Some("id1"), 2, 200, Some(500)), // expired → all id1 deleted
            create_record(Some("id1"), 3, 300, None),
            create_record(Some("id2"), 1, 110, None),
            create_record(Some("id2"), 2, 210, None),
            create_record(None, 0, 130, Some(500)), // expired append-only, excluded
        ];

        let resolved = resolver.resolve_batch(records);
        assert_eq!(resolved.len(), 1);

        let by_id: HashMap<String, &ProximaRecord> =
            resolved.iter().map(|r| (r.oid.clone(), r)).collect();
        assert!(by_id.contains_key("id2"));
        assert_eq!(by_id["id2"].record_version, 2);
    }

    #[test]
    fn test_compare_records_function() {
        let resolver = MvccResolver::with_timestamp(1000);

        let v1 = create_record(Some("id"), 1, 100, None);
        let v2 = create_record(Some("id"), 2, 200, None);

        assert!(resolver.compare_records(&v2, &v1)); // v2 wins
        assert!(!resolver.compare_records(&v1, &v2)); // v1 loses

        let early = create_record(Some("id"), 1, 100, None);
        let late = create_record(Some("id"), 1, 200, None);

        assert!(resolver.compare_records(&early, &late)); // earlier ts wins
        assert!(!resolver.compare_records(&late, &early));

        let expired = create_record(Some("id"), 2, 100, Some(500));
        let valid = create_record(Some("id"), 1, 200, None);

        assert!(resolver.compare_records(&valid, &expired));
        assert!(!resolver.compare_records(&expired, &valid));
    }

    /// TD-COMPACT-2: the compaction resolver must not allocate one HashMap
    /// entry plus one Vec allocation per OID. The sort-and-scan path is
    /// behavior-equivalent to the general query-path resolver for mixed MVCC
    /// inputs, including append-only rows and expiry.
    #[test]
    fn sorted_batch_matches_general_resolver() {
        let resolver = MvccResolver::with_timestamp(1000);
        let records = vec![
            create_record(Some("id2"), 3, 350, None),
            create_record(None, 0, 130, None),
            create_record(Some("id1"), 2, 200, None),
            create_record(Some("id3"), 1, 100, Some(500)),
            create_record(Some("id2"), 1, 150, None),
            create_record(Some("id1"), 1, 100, None),
            create_record(Some("id2"), 1, 140, None),
            create_record(Some("id2"), 2, 250, None),
            create_record(Some("null"), 0, 300, None),
        ];

        let mut general: Vec<(String, u64, i64)> = resolver
            .resolve_batch(records.clone())
            .into_iter()
            .map(|record| (record.oid, record.record_version, record.created_at_ns))
            .collect();
        general.sort();

        let mut sorted: Vec<(String, u64, i64)> = resolver
            .resolve_sorted_batch(records)
            .into_iter()
            .map(|record| (record.oid, record.record_version, record.created_at_ns))
            .collect();
        sorted.sort();

        assert_eq!(sorted, general);
    }

    /// Moving through low-memory MVCC must preserve the embedding allocation.
    /// A pointer change here exposes a hidden full-vector clone even when the
    /// logical record is unchanged.
    #[test]
    fn sorted_batch_moves_embedding_without_cloning() {
        let resolver = MvccResolver::with_timestamp(1000);
        let record = create_record(Some("owned"), 1, 100, None);
        let original_ptr = record.embeddings[0]
            .values
            .as_fp32_slice()
            .map(<[f32]>::as_ptr);

        let resolved = resolver.resolve_sorted_batch(vec![record]);
        let resolved_ptr = resolved[0].embeddings[0]
            .values
            .as_fp32_slice()
            .map(<[f32]>::as_ptr);

        assert_eq!(resolved_ptr, original_ptr);
    }

    #[test]
    fn sorted_batch_output_is_deterministic_by_oid() {
        let resolver = MvccResolver::with_timestamp(1000);
        let records = vec![
            create_record(Some("z"), 1, 100, None),
            create_record(Some("a"), 1, 100, None),
            create_record(Some("m"), 1, 100, None),
        ];

        let resolved = resolver.resolve_sorted_batch(records);
        let ids: Vec<&str> = resolved.iter().map(|record| record.oid.as_str()).collect();
        assert_eq!(ids, vec!["a", "m", "z"]);
    }
}
