//! Deletion-vector planning for the incremental MATERIALIZE (TD-USUB-10,
//! TD-LIVEDEL-1).
//!
//! An incremental republication keeps the parent's data files, appends the
//! after-images of changed rows as new parts, and marks the OLD positions of
//! updated or deleted rows with deletion vectors. Iceberg allows at most one
//! vector per data file, so each file the delta touches gets ONE rewritten
//! vector: the parent's positions for that file, unioned with the new ones.
//! Every other parent vector is carried BY REFERENCE (its Puffin is reused,
//! nothing is decoded). This module is that decision, pure and unit-testable;
//! the caller does the I/O (reading touched parent vectors, writing the new
//! Puffin, committing).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{Result, anyhow};
use object_store::path::Path;
use proximadb_storage_common::object_store_bridge::IcebergSnapshotDeletion;

/// Above this fraction of deleted rows (deleted positions over all rows of
/// the carried files), the successor is a full rewrite instead: readers pay
/// for every hidden row, and a rewrite compacts them away.
pub(super) const MAX_DELETED_FRACTION_PERCENT: u64 = 20;

/// Above this many deleted positions in total, the successor is a full
/// rewrite: half the DuckDB view's 1M deleted-position cap, so external and
/// DuckDB readers keep headroom.
pub(super) const MAX_DELETED_POSITIONS: u64 = 500_000;

/// The deletion vectors a successor snapshot carries.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct DvPlan {
    /// Touched files: their complete, sorted, de-duplicated positions (the
    /// parent vector's union the new ones), to write into one new Puffin.
    pub rewrite: Vec<(Path, Vec<u64>)>,
    /// Untouched parent vectors, carried by reference.
    pub carry: Vec<IcebergSnapshotDeletion>,
}

impl DvPlan {
    /// Total deleted positions the successor will carry.
    pub fn deleted_positions(&self) -> u64 {
        let rewritten: u64 = self.rewrite.iter().map(|(_, p)| p.len() as u64).sum();
        let carried: u64 = self.carry.iter().map(|v| v.cardinality).sum();
        rewritten + carried
    }
}

/// The parent vectors whose files the delta touches: the caller must decode
/// exactly these (and only these) before [`plan_deletion_vectors`].
pub(super) fn touched_parent_vectors<'a>(
    parent: &'a [IcebergSnapshotDeletion],
    fresh: &BTreeMap<Path, Vec<u64>>,
) -> Vec<&'a IcebergSnapshotDeletion> {
    parent
        .iter()
        .filter(|v| fresh.contains_key(&v.data_file))
        .collect()
}

/// Plan the successor's deletion vectors.
///
/// * `carried` — the parent's data files, all kept by the successor;
/// * `parent` — the parent snapshot's vectors (at most one per file);
/// * `fresh` — new positions to delete, per carried file;
/// * `decoded` — the decoded positions of exactly the touched parent vectors
///   (see [`touched_parent_vectors`]);
/// * `file_rows` — each carried file's row count, to reject out-of-range
///   positions.
///
/// Fails closed on any inconsistency rather than publishing a vector that
/// names a file the snapshot does not carry, hides a position beyond a
/// file's end, or silently drops a parent vector.
pub(super) fn plan_deletion_vectors(
    carried: &[Path],
    parent: &[IcebergSnapshotDeletion],
    fresh: &BTreeMap<Path, Vec<u64>>,
    decoded: &HashMap<Path, Vec<u64>>,
    file_rows: &HashMap<Path, u64>,
) -> Result<DvPlan> {
    let carried_set: HashSet<&Path> = carried.iter().collect();
    let mut parent_by_file: HashMap<&Path, &IcebergSnapshotDeletion> = HashMap::new();
    for vector in parent {
        if !carried_set.contains(&vector.data_file) {
            return Err(anyhow!(
                "a parent deletion vector names {}, which the successor does not carry",
                vector.data_file
            ));
        }
        if parent_by_file.insert(&vector.data_file, vector).is_some() {
            return Err(anyhow!(
                "the parent snapshot has two deletion vectors for {}",
                vector.data_file
            ));
        }
    }

    let mut plan = DvPlan::default();
    for (file, positions) in fresh {
        if !carried_set.contains(file) {
            return Err(anyhow!(
                "new deletions name {file}, which the successor does not carry"
            ));
        }
        let mut merged: BTreeSet<u64> = positions.iter().copied().collect();
        if parent_by_file.contains_key(file) {
            let previous = decoded.get(file).ok_or_else(|| {
                anyhow!("the parent deletion vector of {file} was not decoded before planning")
            })?;
            merged.extend(previous.iter().copied());
        }
        if let Some(rows) = file_rows.get(file)
            && let Some(beyond) = merged.iter().find(|p| **p >= *rows)
        {
            return Err(anyhow!(
                "deletion position {beyond} is beyond the {rows} rows of {file}"
            ));
        }
        plan.rewrite
            .push((file.clone(), merged.into_iter().collect()));
    }
    plan.carry = parent
        .iter()
        .filter(|v| !fresh.contains_key(&v.data_file))
        .cloned()
        .collect();
    Ok(plan)
}

/// Whether the planned successor stays within the compaction bounds, or the
/// republication should rewrite (compact) the table instead.
pub(super) fn within_compaction_bounds(plan: &DvPlan, total_rows: u64) -> bool {
    let deleted = plan.deleted_positions();
    deleted <= MAX_DELETED_POSITIONS
        && deleted.saturating_mul(100) <= total_rows.saturating_mul(MAX_DELETED_FRACTION_PERCENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(file: &str, cardinality: u64) -> IcebergSnapshotDeletion {
        IcebergSnapshotDeletion {
            data_file: Path::from(file),
            puffin_file: Path::from("t/data/dv-live-1.puffin"),
            content_offset: 4,
            content_size_in_bytes: 10,
            puffin_file_size_in_bytes: 100,
            cardinality,
        }
    }

    fn paths(names: &[&str]) -> Vec<Path> {
        names.iter().map(|n| Path::from(*n)).collect()
    }

    #[test]
    fn untouched_parent_vectors_are_carried_by_reference() {
        let carried = paths(&["a", "b", "c"]);
        let parent = vec![vector("a", 2), vector("b", 1)];
        let fresh = BTreeMap::from([(Path::from("c"), vec![5, 1])]);
        let plan =
            plan_deletion_vectors(&carried, &parent, &fresh, &HashMap::new(), &HashMap::new())
                .expect("plan");
        assert_eq!(plan.rewrite, vec![(Path::from("c"), vec![1, 5])]);
        assert_eq!(plan.carry, parent, "a and b ride by reference, undecoded");
        assert_eq!(plan.deleted_positions(), 2 + 1 + 2);
    }

    #[test]
    fn a_touched_parent_vector_is_unioned_never_duplicated() {
        let carried = paths(&["a", "b"]);
        let parent = vec![vector("a", 2), vector("b", 1)];
        let fresh = BTreeMap::from([(Path::from("a"), vec![3, 7])]);
        let touched = touched_parent_vectors(&parent, &fresh);
        assert_eq!(touched, vec![&parent[0]], "only a's vector must be decoded");
        let decoded = HashMap::from([(Path::from("a"), vec![1, 3])]);
        let plan = plan_deletion_vectors(&carried, &parent, &fresh, &decoded, &HashMap::new())
            .expect("plan");
        assert_eq!(
            plan.rewrite,
            vec![(Path::from("a"), vec![1, 3, 7])],
            "one vector for a: the parent's {{1,3}} union the new {{3,7}}"
        );
        assert_eq!(
            plan.carry,
            vec![parent[1].clone()],
            "a's parent vector is NOT also carried"
        );
    }

    #[test]
    fn a_touched_parent_vector_must_have_been_decoded() {
        let parent = vec![vector("a", 1)];
        let fresh = BTreeMap::from([(Path::from("a"), vec![2])]);
        let err = plan_deletion_vectors(
            &paths(&["a"]),
            &parent,
            &fresh,
            &HashMap::new(),
            &HashMap::new(),
        )
        .expect_err("planning without the parent's positions would drop them");
        assert!(err.to_string().contains("not decoded"), "{err}");
    }

    #[test]
    fn deletions_on_a_file_the_successor_does_not_carry_fail_closed() {
        let fresh = BTreeMap::from([(Path::from("gone"), vec![0])]);
        let err = plan_deletion_vectors(
            &paths(&["a"]),
            &[],
            &fresh,
            &HashMap::new(),
            &HashMap::new(),
        )
        .expect_err("a vector must name a carried file");
        assert!(err.to_string().contains("does not carry"), "{err}");
        let err = plan_deletion_vectors(
            &paths(&["a"]),
            &[vector("gone", 1)],
            &BTreeMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        )
        .expect_err("a parent vector on a dropped file must not vanish silently");
        assert!(err.to_string().contains("does not carry"), "{err}");
    }

    #[test]
    fn duplicate_parent_vectors_and_out_of_range_positions_fail_closed() {
        let err = plan_deletion_vectors(
            &paths(&["a"]),
            &[vector("a", 1), vector("a", 2)],
            &BTreeMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        )
        .expect_err("two parent vectors for one file");
        assert!(err.to_string().contains("two deletion vectors"), "{err}");

        let fresh = BTreeMap::from([(Path::from("a"), vec![9])]);
        let rows = HashMap::from([(Path::from("a"), 9)]);
        let err = plan_deletion_vectors(&paths(&["a"]), &[], &fresh, &HashMap::new(), &rows)
            .expect_err("position 9 of a 9-row file");
        assert!(err.to_string().contains("beyond"), "{err}");
    }

    #[test]
    fn compaction_bounds_trip_on_fraction_and_on_total() {
        let plan = |n: u64| DvPlan {
            rewrite: vec![(Path::from("a"), (0..n).collect())],
            carry: vec![],
        };
        assert!(
            within_compaction_bounds(&plan(20), 100),
            "exactly 20% stays"
        );
        assert!(
            !within_compaction_bounds(&plan(21), 100),
            "over 20% compacts"
        );
        let big = DvPlan {
            rewrite: vec![],
            carry: vec![vector("a", MAX_DELETED_POSITIONS + 1)],
        };
        assert!(
            !within_compaction_bounds(&big, u64::MAX / 100),
            "over the absolute cap compacts regardless of fraction"
        );
    }
}
