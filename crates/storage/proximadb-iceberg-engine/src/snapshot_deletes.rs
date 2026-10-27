//! Snapshots with deletion vectors (P2.4 slice 3): reading a snapshot's data files
//! together with the vectors that delete rows of them, writing a Puffin file of
//! vectors, and publishing a snapshot that carries them.
//!
//! The invariants a published snapshot keeps, and every read checks:
//!
//! * only format v3 carries deletion vectors (v2 has position-delete files instead);
//! * each vector references a data file of the same snapshot;
//! * a data file has at most one vector, which holds every deleted position (a new
//!   vector for a file replaces the previous one and must include its positions).

use std::collections::HashSet;

use bytes::Bytes;
use object_store::path::Path;
use proximadb_kernel::error::StorageError;

use crate::deletion_vector::PositionDeletes;
use crate::{IcebergObjectStoreBridge, iceberg, metadata, puffin, read_err};

/// One deletion vector of a snapshot, resolved to bridge-relative keys.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotDeletionVector {
    /// The data file whose rows it deletes (one of the snapshot's `data_files`).
    pub data_file: Path,
    /// The Puffin file holding it, and its blob's byte range there.
    pub puffin_file: Path,
    pub content_offset: u64,
    pub content_size_in_bytes: u64,
    /// The Puffin file's full size (the delete-manifest entry's `file_size_in_bytes`),
    /// so a re-publisher can carry the vector without re-reading the object.
    pub puffin_file_size_in_bytes: u64,
    /// The number of deleted positions.
    pub cardinality: u64,
}

/// A snapshot's data files and the deletion vectors that apply to them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IcebergSnapshotFiles {
    pub data_files: Vec<Path>,
    pub deletion_vectors: Vec<SnapshotDeletionVector>,
}

fn invalid(context: impl std::fmt::Display) -> StorageError {
    StorageError::Serialization(format!("iceberg-engine: {context}"))
}

fn to_u64(v: i64, what: &str) -> Result<u64, StorageError> {
    u64::try_from(v).map_err(|_| invalid(format!("negative {what} {v}")))
}

impl IcebergObjectStoreBridge {
    /// The current snapshot of metadata `version` under `metadata_prefix`: its data
    /// files and its deletion vectors, both as bridge-relative keys.
    pub async fn read_iceberg_snapshot_with_deletes(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<IcebergSnapshotFiles, StorageError> {
        let committer = metadata::MetadataCommitter::new(self.store.clone(), metadata_prefix);
        let meta_bytes = committer.read_metadata(version).await?;
        let meta_json =
            std::str::from_utf8(&meta_bytes).map_err(|e| read_err("metadata utf8", e))?;
        let Some(list_loc) = iceberg::current_manifest_list(meta_json)? else {
            return Ok(IcebergSnapshotFiles::default());
        };
        // Recorded paths are absolute URIs (or legacy bridge-relative keys). The
        // table prefix re-roots URIs under the table `location` when this bridge is
        // opened somewhere other than the writer's root.
        let table_location = serde_json::from_str::<serde_json::Value>(meta_json)
            .ok()
            .and_then(|v| v.get("location").and_then(|l| l.as_str()).map(String::from));
        let table_prefix = metadata_prefix
            .trim_end_matches('/')
            .strip_suffix("metadata")
            .map(|p| p.trim_end_matches('/'))
            .unwrap_or("");
        let resolve =
            |p: &str| self.resolve_recorded_path(p, table_location.as_deref(), table_prefix);
        let list_bytes = self.store.get(&resolve(&list_loc)).await?;
        let mut out = IcebergSnapshotFiles::default();
        let mut vectors = Vec::new();
        for m in iceberg::read_manifest_list(&list_bytes)? {
            let bytes = self.store.get(&resolve(&m.manifest_path)).await?;
            match m.content {
                iceberg::ManifestContent::Data => {
                    for df in iceberg::read_manifest(&bytes)? {
                        out.data_files.push(resolve(&df.file_path));
                    }
                }
                iceberg::ManifestContent::Deletes => {
                    vectors.extend(iceberg::read_delete_manifest(&bytes)?);
                }
            }
        }
        let data: HashSet<&Path> = out.data_files.iter().collect();
        let mut seen: HashSet<Path> = HashSet::new();
        for v in vectors {
            let data_file = resolve(&v.referenced_data_file);
            if !data.contains(&data_file) {
                return Err(invalid(format!(
                    "a deletion vector references {data_file}, which is not a data file of \
                     snapshot {version}"
                )));
            }
            if !seen.insert(data_file.clone()) {
                return Err(invalid(format!(
                    "{data_file} has more than one deletion vector in snapshot {version}"
                )));
            }
            out.deletion_vectors.push(SnapshotDeletionVector {
                data_file,
                puffin_file: resolve(&v.file_path),
                content_offset: to_u64(v.content_offset, "content_offset")?,
                content_size_in_bytes: to_u64(v.content_size_in_bytes, "content_size_in_bytes")?,
                puffin_file_size_in_bytes: to_u64(v.file_size_in_bytes, "file_size_in_bytes")?,
                cardinality: to_u64(v.record_count, "record_count")?,
            });
        }
        Ok(out)
    }

    /// Fetch and decode one deletion vector (a ranged read of its blob only).
    pub async fn read_deletion_vector(
        &self,
        vector: &SnapshotDeletionVector,
    ) -> Result<PositionDeletes, StorageError> {
        let end = vector
            .content_offset
            .checked_add(vector.content_size_in_bytes)
            .ok_or_else(|| invalid("deletion vector range overflows"))?;
        let blob = self
            .store
            .get_range(&vector.puffin_file, vector.content_offset..end)
            .await?;
        let deletes = puffin::read_deletion_vector(&blob)?;
        if deletes.cardinality() != vector.cardinality {
            return Err(invalid(format!(
                "deletion vector for {} holds {} positions, its manifest entry says {}",
                vector.data_file,
                deletes.cardinality(),
                vector.cardinality
            )));
        }
        Ok(deletes)
    }

    /// Write `vectors` (data-file key → its complete deletion set) as one Puffin file
    /// at `puffin_key`, create-only. Returns the delete-manifest entries to publish
    /// with [`publish_iceberg_files_with_deletes`](Self::publish_iceberg_files_with_deletes).
    pub async fn write_deletion_vectors(
        &self,
        puffin_key: &Path,
        vectors: &[(Path, &PositionDeletes)],
    ) -> Result<Vec<iceberg::DeleteFileMeta>, StorageError> {
        let named: Vec<(String, &PositionDeletes)> = vectors
            .iter()
            .map(|(key, deletes)| (self.object_uri(key.as_ref()), *deletes))
            .collect();
        let (bytes, written) = puffin::write_deletion_vectors(&named)?;
        let size = bytes.len() as i64;
        self.store
            .put_if_absent(puffin_key, Bytes::from(bytes))
            .await?;
        let file_path = self.object_uri(puffin_key.as_ref());
        Ok(written
            .into_iter()
            .map(|w| iceberg::DeleteFileMeta {
                file_path: file_path.clone(),
                file_size_in_bytes: size,
                referenced_data_file: w.referenced_data_file,
                content_offset: w.content_offset,
                content_size_in_bytes: w.content_size_in_bytes,
                record_count: w.cardinality,
            })
            .collect())
    }

    /// [`publish_iceberg_files`](Self::publish_iceberg_files) with deletion vectors:
    /// the snapshot is `files` with `deletes` applied. Format v3 only; each vector must
    /// reference one of `files`, and no file may have two.
    #[allow(clippy::too_many_arguments)]
    pub async fn publish_iceberg_files_with_deletes(
        &self,
        files: &[Path],
        deletes: &[iceberg::DeleteFileMeta],
        parent: Option<u64>,
        metadata_prefix: &str,
        table_uuid: &str,
        table_location: &str,
        fields: &[iceberg::IcebergField],
        snapshot_id: i64,
        timestamp_ms: i64,
    ) -> Result<u64, StorageError> {
        let uris: HashSet<String> = files.iter().map(|f| self.object_uri(f.as_ref())).collect();
        let mut referenced = HashSet::new();
        for d in deletes {
            if !uris.contains(&d.referenced_data_file) {
                return Err(invalid(format!(
                    "a deletion vector references {}, which is not published here",
                    d.referenced_data_file
                )));
            }
            if !referenced.insert(d.referenced_data_file.as_str()) {
                return Err(invalid(format!(
                    "{} has more than one deletion vector",
                    d.referenced_data_file
                )));
            }
        }
        self.publish_files_inner(
            files,
            deletes,
            parent,
            metadata_prefix,
            table_uuid,
            table_location,
            fields,
            snapshot_id,
            timestamp_ms,
            iceberg::FormatVersion::V3,
        )
        .await
    }
}

/// TD-USUB-10 3c (live writer, foundation): where each deleted OID sits in a
/// published data file. The delete path keys on the canonical OID while a
/// Puffin deletion vector keys on the row's position within its file; the
/// hidden `__proxima_oid` column (written with every published file, P2.4
/// slice 2a) is the bridge. Positions are the file's write order — exactly
/// what `write_deletion_vectors` then encodes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PublishedOidPositions {
    /// One entry per file that holds at least one asked oid.
    pub per_file: Vec<PublishedPositions>,
    /// The asked oids found in NO file — top-level, never silently skipped:
    /// a delete keyed on a never-published oid may still be live in the WAL.
    pub oids_missing: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedPositions {
    /// The data file the positions belong to (bridge-relative key).
    pub file: Path,
    /// Row positions of the found oids, ascending (the DV is a set).
    pub positions: Vec<u32>,
    /// The oids found in this file, sorted (NOT index-aligned with
    /// `positions` — the pairing is not recoverable from this shape).
    pub oids_found: Vec<String>,
}

impl IcebergObjectStoreBridge {
    /// Map `oids` to their per-file row positions across `files`. The union of
    /// every file's `oids_missing` (deduplicated) rides the LAST entry so the
    /// caller sees never-published oids once.
    pub async fn published_positions_of(
        &self,
        files: &[Path],
        oids: &[&str],
    ) -> Result<PublishedOidPositions, StorageError> {
        // COST (co-design mandate, stated honestly): this reads the ENTIRE
        // `__proxima_oid` column of EVERY file — O(published oid bytes) of
        // network I/O per call, plus every oid materialized as a String
        // in memory. No row-group pruning, no cache. A footer/sidecar index
        // (TD-FPRUNE-1 pattern; the OidPositionResolver persistence slices)
        // is the known follow-up. Reachability note: this is an INHERENT
        // method; the DML delete path holds `&dyn ObjectStoreBridge` and
        // reaches the concrete type the way the DataFusion reader does
        // (`IcebergObjectStoreBridge::from_url`) or via a later trait method.
        use arrow_array::Array;
        use proximadb_storage_common::object_store_bridge::ObjectStoreBridge as _;
        use proximadb_storage_common::proxima_arrow::SYS_COL_OID;

        let asked: std::collections::HashSet<&str> = oids.iter().copied().collect();
        let mut found_all: std::collections::HashSet<&str> = Default::default();
        let mut per_file = Vec::new();
        for file in files {
            // Project just the oid column; only that column is decoded.
            let oid_field =
                arrow_schema::Field::new(SYS_COL_OID, arrow_schema::DataType::Utf8, false);
            let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![oid_field]));
            let mut stream = self
                .read_parquet_batches(file, schema, 8192, None)
                .await
                .map_err(|e| invalid(format!("opening {file:?} oid column: {e}")))?;
            use futures::TryStreamExt;
            let mut rows: Vec<String> = Vec::new();
            while let Some(batch) = stream
                .try_next()
                .await
                .map_err(|e| invalid(format!("reading {file:?} oid column: {e}")))?
            {
                // Round-1 review: a file written before the hidden oid column
                // (pre-P2.4-2a parts carried forward by appends) yields an
                // EMPTY projection — 0-column batches with row counts — and
                // a column access would PANIC. Fail closed with the file
                // named instead.
                if batch.num_columns() == 0 {
                    return Err(invalid(format!(
                        "{file:?}: no {SYS_COL_OID} column (written before the hidden \
                         oid column existed); the delete writer cannot map positions \
                         in this file"
                    )));
                }
                let column = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| {
                        invalid(format!("{file:?}: the {SYS_COL_OID} column is not utf8"))
                    })?;
                for i in 0..column.len() {
                    // A null oid cell (a foreign writer's shape; ours are
                    // non-null) can never match an asked oid — skip it.
                    if !column.is_null(i) {
                        rows.push(column.value(i).to_string());
                    }
                }
            }
            let resolver =
                proximadb_storage_common::oid_position_resolver::OidPositionResolver::from_stream_order(rows);
            let mut positions = Vec::new();
            let mut oids_found = Vec::new();
            for oid in &asked {
                if let Some(position) = resolver.position_of(oid) {
                    positions.push(position);
                    oids_found.push((*oid).to_string());
                    found_all.insert(oid);
                }
            }
            if !oids_found.is_empty() {
                positions.sort_unstable();
                positions.dedup();
                oids_found.sort();
                per_file.push(PublishedPositions {
                    file: file.clone(),
                    positions,
                    oids_found,
                });
            }
        }
        let mut oids_missing: Vec<String> = asked
            .iter()
            .filter(|o| !found_all.contains(*o))
            .map(|o| o.to_string())
            .collect();
        oids_missing.sort();
        Ok(PublishedOidPositions {
            per_file,
            oids_missing,
        })
    }
}

#[cfg(test)]
#[path = "snapshot_deletes_tests.rs"]
mod tests;
