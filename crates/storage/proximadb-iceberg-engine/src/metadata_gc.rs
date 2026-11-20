//! # metadata_gc — collect Iceberg metadata outside the generation window
//!
//! ADR-098 D4.2 / TD-SPINE-3. After a MATERIALIZE commit lands at version V,
//! the retained metadata versions are `window_floor(V, N)..` (N =
//! `PROXIMADB_MATERIALIZE_GC_RETAINED_GENERATIONS`, the TD-GCRETAIN-1 window
//! the data-file GC already uses). This pass deletes, under the table's
//! metadata prefix only:
//!
//! * `v{N}.metadata.json` documents older than the window — never the tip;
//! * manifest lists (`snap-{id}-v{N}-list.avro`) and manifests
//!   (`v{N}-{id}-m{k}.avro`) whose name places them older than the window AND
//!   that no retained metadata document reaches.
//!
//! Safety, all without a clock (TD-GCRETAIN-1 declined time-based grace):
//!
//! * **Reachability is positive evidence.** Every retained document (every
//!   listed version at or above the floor, including versions newer than V
//!   that a concurrent publisher committed) is read and its manifest lists and
//!   manifests resolved. Any read or parse failure skips the whole pass — an
//!   unreadable retained version cannot prove anything unreferenced.
//! * **The version in the name orders against in-flight publishers.** A
//!   publisher writes its manifests before its metadata CAS, so they are
//!   unreachable until it commits; but it commits the successor of a version
//!   it read as the latest, which is at least V, so its manifests are named
//!   with a version above the floor and are never candidates. Names that do
//!   not parse (legacy layouts, foreign writers) are kept.
//! * **Best effort.** Deletes that fail are counted, never surfaced as a
//!   commit error; the next pass retries them (they stay below the floor and
//!   unreachable).

use std::collections::HashSet;

use object_store::ObjectStoreExt;
use object_store::path::Path;
use proximadb_kernel::error::StorageError;
use proximadb_storage_common::object_store_bridge::IcebergMetadataGcReport;

use crate::{IcebergObjectStoreBridge, iceberg, read_err, retention};

/// The version of a `v{N}.metadata.json` file name.
pub(crate) fn metadata_version_of(name: &str) -> Option<u64> {
    let digits = name.strip_prefix('v')?.strip_suffix(".metadata.json")?;
    all_digits(digits).then(|| digits.parse().ok()).flatten()
}

/// The metadata version a generation-named Avro object was written for:
/// `v{N:020}-{snapshot}-m{k}.avro` (manifest) or
/// `snap-{snapshot}-v{N:020}-list.avro` (manifest list). `None` for any
/// other name, which the GC never deletes.
pub(crate) fn avro_version_of(name: &str) -> Option<u64> {
    if let Some(rest) = name.strip_prefix("snap-") {
        let body = rest.strip_suffix("-list.avro")?;
        let (snapshot, version) = body.rsplit_once("-v")?;
        return (is_snapshot_id(snapshot) && version.len() == 20 && all_digits(version))
            .then(|| version.parse().ok())
            .flatten();
    }
    let body = name.strip_prefix('v')?.strip_suffix(".avro")?;
    let (version, rest) = body.split_once('-')?;
    let (snapshot, manifest) = rest.rsplit_once("-m")?;
    (version.len() == 20 && all_digits(version) && is_snapshot_id(snapshot) && all_digits(manifest))
        .then(|| version.parse().ok())
        .flatten()
}

fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

fn is_snapshot_id(s: &str) -> bool {
    all_digits(s.strip_prefix('-').unwrap_or(s))
}

/// What one GC pass would delete, decided from names alone (pure; the bridge
/// supplies the listing and the reachable set).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetadataGcPlan {
    /// Oldest retained metadata version.
    pub floor: u64,
    /// Highest listed metadata version (never deleted).
    pub tip: Option<u64>,
    /// Objects to delete, as listed.
    pub delete: Vec<Path>,
}

/// Plan a pass over the direct children of a metadata prefix. `reachable`
/// holds the FILE NAMES of every manifest list and manifest a retained
/// document reaches (names, not paths: matching by name can only over-keep).
pub fn plan_metadata_gc(
    listing: &[Path],
    committed: u64,
    retained_generations: u64,
    reachable: &HashSet<String>,
) -> MetadataGcPlan {
    let floor = retention::window_floor(committed, retained_generations);
    let tip = listing
        .iter()
        .filter_map(|p| p.filename().and_then(metadata_version_of))
        .max();
    let delete = listing
        .iter()
        .filter(|path| {
            let Some(name) = path.filename() else {
                return false;
            };
            if let Some(version) = metadata_version_of(name) {
                return version < floor && Some(version) != tip;
            }
            avro_version_of(name).is_some_and(|v| v < floor) && !reachable.contains(name)
        })
        .cloned()
        .collect();
    MetadataGcPlan { floor, tip, delete }
}

impl IcebergObjectStoreBridge {
    /// Delete the metadata of `metadata_prefix` outside the generation window
    /// ending at `committed` (see the module docs). Never an error for a
    /// failed delete; an `Err` only for a failed LIST.
    pub async fn gc_metadata(
        &self,
        metadata_prefix: &str,
        committed: u64,
        retained_generations: u64,
    ) -> Result<IcebergMetadataGcReport, StorageError> {
        let prefix = metadata_prefix.trim_end_matches('/');
        let full_prefix = self.store.full_path(&Path::from(prefix));
        // Direct children only: anything nested deeper is not ours to judge.
        let listing: Vec<Path> = self
            .store
            .list(Some(&Path::from(prefix)))
            .await?
            .into_iter()
            .map(|m| m.location)
            .filter(|loc| {
                loc.filename()
                    .is_some_and(|n| loc.as_ref() == format!("{full_prefix}/{n}"))
            })
            .collect();
        let floor = retention::window_floor(committed, retained_generations);
        let retained_docs: Vec<&Path> = listing
            .iter()
            .filter(|p| {
                p.filename()
                    .and_then(metadata_version_of)
                    .is_some_and(|v| v >= floor)
            })
            .collect();
        let mut reachable = HashSet::new();
        for doc in retained_docs {
            if let Err(e) = self.collect_reachable(doc, prefix, &mut reachable).await {
                return Ok(IcebergMetadataGcReport {
                    skipped: Some(format!(
                        "retained metadata {doc} unreadable, nothing collected: {e}"
                    )),
                    ..Default::default()
                });
            }
        }
        let plan = plan_metadata_gc(&listing, committed, retained_generations, &reachable);
        let store = self.store.store();
        let mut report = IcebergMetadataGcReport::default();
        for path in &plan.delete {
            match store.delete(path).await {
                Ok(()) => report.deleted += 1,
                Err(e) => {
                    report.failed += 1;
                    tracing::debug!(%path, "iceberg metadata gc: delete failed: {e}");
                }
            }
        }
        Ok(report)
    }

    /// Add the file names of every manifest list and manifest the document at
    /// `doc` (a full store path) reaches, over ALL its snapshots.
    async fn collect_reachable(
        &self,
        doc: &Path,
        metadata_prefix: &str,
        reachable: &mut HashSet<String>,
    ) -> Result<(), StorageError> {
        let raw = self.store.store();
        let bytes = raw
            .get(doc)
            .await
            .map_err(|e| read_err("metadata gc: get metadata", e))?
            .bytes()
            .await
            .map_err(|e| read_err("metadata gc: read metadata", e))?;
        let json: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| read_err("metadata gc: parse metadata", e))?;
        let table_location = json.get("location").and_then(|l| l.as_str());
        let table_prefix = metadata_prefix
            .strip_suffix("metadata")
            .map(|p| p.trim_end_matches('/'))
            .unwrap_or("");
        let snapshots = json
            .get("snapshots")
            .and_then(|s| s.as_array())
            .cloned()
            .unwrap_or_default();
        for snapshot in snapshots {
            let Some(list) = snapshot.get("manifest-list").and_then(|l| l.as_str()) else {
                continue;
            };
            let list_path = self.resolve_recorded_path(list, table_location, table_prefix);
            if let Some(name) = list_path.filename() {
                reachable.insert(name.to_string());
            }
            let list_bytes = self.store.get(&list_path).await?;
            for manifest in iceberg::read_manifest_list(&list_bytes)? {
                let path = self.resolve_recorded_path(
                    &manifest.manifest_path,
                    table_location,
                    table_prefix,
                );
                if let Some(name) = path.filename() {
                    reachable.insert(name.to_string());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "metadata_gc_tests.rs"]
mod bridge_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn names(paths: &[Path]) -> Vec<String> {
        paths
            .iter()
            .filter_map(|p| p.filename().map(String::from))
            .collect()
    }

    fn meta(v: u64) -> Path {
        Path::from(format!("t/metadata/v{v:020}.metadata.json"))
    }
    fn manifest(v: u64) -> Path {
        Path::from(format!("t/metadata/v{v:020}-{}-m0.avro", 1000 + v))
    }
    fn list(v: u64) -> Path {
        Path::from(format!("t/metadata/snap-{}-v{v:020}-list.avro", 1000 + v))
    }

    #[test]
    fn name_parsing_is_strict() {
        assert_eq!(
            metadata_version_of("v00000000000000000007.metadata.json"),
            Some(7)
        );
        assert_eq!(metadata_version_of("v7.metadata.json"), Some(7));
        assert_eq!(metadata_version_of("vx.metadata.json"), None);
        assert_eq!(
            avro_version_of(manifest(4).filename().unwrap_or_default()),
            Some(4)
        );
        assert_eq!(
            avro_version_of(list(4).filename().unwrap_or_default()),
            Some(4)
        );
        // Legacy / foreign names never parse, so they are never collected.
        assert_eq!(avro_version_of("snap-1-list.avro"), None);
        assert_eq!(avro_version_of("1-m0.avro"), None);
        assert_eq!(avro_version_of("v00000000000000000004-1-m0.avro.tmp"), None);
        assert_eq!(
            avro_version_of("v4-1-m0.avro"),
            None,
            "unpadded is not ours"
        );
    }

    #[test]
    fn plan_keeps_the_window_and_the_tip() {
        let mut listing = Vec::new();
        for v in 0..6 {
            listing.extend([meta(v), manifest(v), list(v)]);
        }
        // committed v5, window 3 → retain v3, v4, v5.
        let reachable: HashSet<String> = (3..6)
            .flat_map(|v| names(&[manifest(v), list(v)]))
            .collect();
        let plan = plan_metadata_gc(&listing, 5, 3, &reachable);
        assert_eq!(plan.floor, 3);
        assert_eq!(plan.tip, Some(5));
        let mut deleted = names(&plan.delete);
        deleted.sort();
        let mut expected = names(
            &(0..3)
                .flat_map(|v| [meta(v), manifest(v), list(v)])
                .collect::<Vec<_>>(),
        );
        expected.sort();
        assert_eq!(deleted, expected);
    }

    #[test]
    fn plan_never_deletes_a_reachable_or_newer_object() {
        let listing = vec![
            meta(0),
            manifest(0),
            list(0),
            meta(9),
            manifest(9),
            manifest(12),
        ];
        // An old manifest a retained document still names survives.
        let reachable: HashSet<String> = names(&[manifest(0), list(0)]).into_iter().collect();
        let plan = plan_metadata_gc(&listing, 9, 2, &reachable);
        let deleted = names(&plan.delete);
        assert_eq!(deleted, names(&[meta(0)]));
        // An in-flight publisher's manifest (named above the committed
        // version, reachable from nothing yet) is never a candidate.
        assert!(!deleted.contains(&names(&[manifest(12)])[0]));
    }

    #[test]
    fn plan_never_deletes_the_tip_even_below_the_floor() {
        // A LIST that lags the commit (tip v1 < committed v5) keeps v1.
        let listing = vec![meta(0), meta(1)];
        let plan = plan_metadata_gc(&listing, 5, 2, &HashSet::new());
        assert_eq!(names(&plan.delete), names(&[meta(0)]));
    }

    #[test]
    fn plan_is_clock_free_and_deterministic() {
        let listing: Vec<Path> = (0..10)
            .flat_map(|v| [meta(v), manifest(v), list(v)])
            .collect();
        let reachable: HashSet<String> = (6..10)
            .flat_map(|v| names(&[manifest(v), list(v)]))
            .collect();
        let a = plan_metadata_gc(&listing, 9, 4, &reachable);
        let b = plan_metadata_gc(&listing, 9, 4, &reachable);
        assert_eq!(a, b);
        assert_eq!(a.delete.len(), 18);
    }
}
