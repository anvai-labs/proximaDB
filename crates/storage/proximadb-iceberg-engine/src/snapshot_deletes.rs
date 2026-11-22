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

/// A snapshot as a planner reads it (TD-SPINE-4): the data files and deletion
/// vectors of [`IcebergSnapshotFiles`], plus each data file's manifest column
/// metrics and the schema those metrics' field ids refer to. Every part is a
/// pure function of the (immutable) metadata version it was read from.
#[derive(Debug, Clone, Default)]
pub struct IcebergSnapshotPlan {
    pub files: IcebergSnapshotFiles,
    /// `(data file, its manifest column metrics)`, one per data file, in
    /// manifest order. Metrics are keyed by Iceberg field id.
    pub file_stats: Vec<(Path, Vec<iceberg::ColumnFileStats>)>,
    /// The metadata document's current schema (primitive fields only).
    pub fields: Vec<iceberg::IcebergField>,
}

/// The identity and manifest plan of a producer-published Iceberg snapshot.
#[derive(Debug, Clone)]
pub struct PinnedIcebergSnapshot {
    pub table_uuid: uuid::Uuid,
    pub snapshot_id: i64,
    pub format_version: u8,
    /// Source version reported by UniForm, when the property is present.
    pub converted_delta_version: Option<u64>,
    pub plan: IcebergSnapshotPlan,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct ProducerMetadata {
    table_uuid: String,
    location: String,
    format_version: u8,
    current_snapshot_id: i64,
    snapshots: Vec<ProducerSnapshot>,
    #[serde(default)]
    properties: std::collections::HashMap<String, String>,
    current_schema_id: Option<i32>,
    schemas: Option<Vec<serde_json::Value>>,
    partition_specs: Option<Vec<serde_json::Value>>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct ProducerSnapshot {
    snapshot_id: i64,
    manifest_list: Option<String>,
    key_id: Option<String>,
}

impl IcebergObjectStoreBridge {
    /// Plan the current snapshot of one immutable producer metadata object.
    ///
    /// The caller must authorize the table prefix independently of the document.
    /// Use a trusted store with filesystem confinement when applicable. This
    /// validates object keys lexically and does not enforce filesystem symlink
    /// isolation. Every reference must be an absolute URI under the authorized
    /// table root, using the same spelling as the bridge and unescaped keys.
    /// When supplied, the Delta version must match before any manifest I/O.
    /// Use [`open_pinned_iceberg_reader`](Self::open_pinned_iceberg_reader)
    /// for a reader that projects by field ID and applies deletion vectors.
    pub async fn read_pinned_iceberg_snapshot(
        &self,
        authorized_table_prefix: &Path,
        metadata_uri: &str,
        expected_delta_version: Option<u64>,
    ) -> Result<PinnedIcebergSnapshot, StorageError> {
        let (snapshot, _) = self
            .load_pinned_iceberg_snapshot(
                authorized_table_prefix,
                metadata_uri,
                expected_delta_version,
            )
            .await?;
        Ok(snapshot)
    }

    /// Open a flat, field-ID-bearing Parquet snapshot with the same storage
    /// authorization requirements as [`read_pinned_iceberg_snapshot`](Self::read_pinned_iceberg_snapshot).
    pub async fn open_pinned_iceberg_reader(
        &self,
        authorized_table_prefix: &Path,
        metadata_uri: &str,
        expected_delta_version: Option<u64>,
    ) -> Result<crate::PinnedIcebergReader, StorageError> {
        let (snapshot, metadata) = self
            .load_pinned_iceberg_snapshot(
                authorized_table_prefix,
                metadata_uri,
                expected_delta_version,
            )
            .await?;
        let id = metadata.current_schema_id.map(i64::from);
        let schema = metadata
            .schemas
            .as_deref()
            .unwrap_or_default()
            .iter()
            .find(|schema| id.is_some() && schema["schema-id"].as_i64() == id)
            .ok_or_else(|| invalid("pinned Iceberg current schema is absent"))?;
        crate::PinnedIcebergReader::try_new(
            self.clone(),
            snapshot,
            schema,
            metadata.partition_specs.as_deref(),
        )
    }

    async fn load_pinned_iceberg_snapshot(
        &self,
        authorized_table_prefix: &Path,
        metadata_uri: &str,
        expected_delta_version: Option<u64>,
    ) -> Result<(PinnedIcebergSnapshot, ProducerMetadata), StorageError> {
        if !self
            .root_url
            .as_deref()
            .is_some_and(|root| root.contains("://"))
        {
            return Err(invalid(
                "pinned Iceberg reads require an absolute bridge root",
            ));
        }
        let table_root = self.object_uri(authorized_table_prefix.as_ref());
        let table_root = table_root.trim_end_matches('/');
        let resolve = |uri: &str| {
            let key = uri
                .strip_prefix(table_root)
                .and_then(|suffix| suffix.strip_prefix('/'))
                .filter(|key| {
                    !key.is_empty()
                        && key
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"/-_.~=".contains(&b))
                })
                .ok_or_else(|| {
                    invalid(format!("unsupported Iceberg URI under {table_root}: {uri}"))
                })?;
            let path = Path::parse(key).map_err(|e| read_err("pinned Iceberg key", e))?;
            if path.as_ref() != key {
                return Err(invalid(format!("non-canonical Iceberg URI: {uri}")));
            }
            Ok(if authorized_table_prefix.as_ref().is_empty() {
                path
            } else {
                Path::parse(format!("{authorized_table_prefix}/{path}"))
                    .map_err(|e| read_err("pinned Iceberg key", e))?
            })
        };
        let bytes = self.store.get(&resolve(metadata_uri)?).await?;
        let json = std::str::from_utf8(&bytes).map_err(|e| read_err("metadata utf8", e))?;
        let meta: ProducerMetadata =
            serde_json::from_str(json).map_err(|e| read_err("producer metadata", e))?;
        if meta.location.trim_end_matches('/') != table_root {
            return Err(invalid(
                "producer metadata location differs from the authorized table root",
            ));
        }
        let table_uuid = uuid::Uuid::parse_str(&meta.table_uuid)
            .map_err(|e| read_err("producer table UUID", e))?;
        if !matches!(meta.format_version, 2 | 3) {
            return Err(invalid(format!(
                "unsupported Iceberg format v{}",
                meta.format_version
            )));
        }
        let converted_delta_version = meta
            .properties
            .get("converted_delta_version")
            .map(|version| version.parse::<u64>())
            .transpose()
            .map_err(|e| read_err("UniForm converted_delta_version", e))?;
        if let Some(expected) = expected_delta_version
            && converted_delta_version != Some(expected)
        {
            return Err(invalid(format!(
                "expected UniForm Delta version {expected}, metadata records {converted_delta_version:?}"
            )));
        }
        let snapshot_id = meta.current_snapshot_id;
        let snapshot = meta
            .snapshots
            .iter()
            .find(|s| s.snapshot_id == snapshot_id)
            .ok_or_else(|| invalid(format!("current snapshot {snapshot_id} is absent")))?;
        if snapshot.key_id.is_some() {
            return Err(invalid("encrypted Iceberg manifests are unsupported"));
        }
        let list = snapshot
            .manifest_list
            .as_deref()
            .ok_or_else(|| invalid(format!("snapshot {snapshot_id} has no manifest list")))?;
        let plan = self
            .read_manifest_plan(
                list,
                iceberg::current_schema_fields(json),
                snapshot_id,
                meta.format_version == 3,
                resolve,
            )
            .await?;
        let pinned = PinnedIcebergSnapshot {
            table_uuid,
            snapshot_id,
            format_version: meta.format_version,
            converted_delta_version,
            plan,
        };
        Ok((pinned, meta))
    }

    /// The current snapshot of metadata `version` under `metadata_prefix`: its data
    /// files and its deletion vectors, both as bridge-relative keys.
    pub async fn read_iceberg_snapshot_with_deletes(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<IcebergSnapshotFiles, StorageError> {
        Ok(self
            .read_iceberg_snapshot_plan(metadata_prefix, version)
            .await?
            .files)
    }

    /// [`read_iceberg_snapshot_with_deletes`](Self::read_iceberg_snapshot_with_deletes)
    /// plus the per-file manifest metrics and the schema they refer to — the
    /// same I/O (metadata document, manifest list, manifests), nothing more.
    pub async fn read_iceberg_snapshot_plan(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<IcebergSnapshotPlan, StorageError> {
        let committer = metadata::MetadataCommitter::new(self.store.clone(), metadata_prefix);
        let meta_bytes = committer.read_metadata(version).await?;
        let meta_json =
            std::str::from_utf8(&meta_bytes).map_err(|e| read_err("metadata utf8", e))?;
        let fields = iceberg::current_schema_fields(meta_json);
        let Some(list_loc) = iceberg::current_manifest_list(meta_json)? else {
            return Ok(IcebergSnapshotPlan {
                fields,
                ..Default::default()
            });
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
        self.read_manifest_plan(&list_loc, fields, version, true, |path| {
            Ok(self.resolve_recorded_path(path, table_location.as_deref(), table_prefix))
        })
        .await
    }

    async fn read_manifest_plan(
        &self,
        list_loc: &str,
        fields: Vec<iceberg::IcebergField>,
        snapshot: impl std::fmt::Display,
        allow_deletion_vectors: bool,
        resolve: impl Fn(&str) -> Result<Path, StorageError>,
    ) -> Result<IcebergSnapshotPlan, StorageError> {
        let list_bytes = self.store.get(&resolve(list_loc)?).await?;
        let mut out = IcebergSnapshotFiles::default();
        let mut file_stats = Vec::new();
        let mut vectors = Vec::new();
        for m in iceberg::read_manifest_list(&list_bytes)? {
            if m.content == iceberg::ManifestContent::Deletes && !allow_deletion_vectors {
                return Err(invalid("delete manifests in Iceberg v2 are unsupported"));
            }
            let bytes = self.store.get(&resolve(&m.manifest_path)?).await?;
            match m.content {
                iceberg::ManifestContent::Data => {
                    for df in iceberg::read_manifest(&bytes)? {
                        let path = resolve(&df.file_path)?;
                        file_stats.push((path.clone(), df.column_stats));
                        out.data_files.push(path);
                    }
                }
                iceberg::ManifestContent::Deletes => {
                    vectors.extend(iceberg::read_delete_manifest(&bytes)?);
                }
            }
        }
        let data: HashSet<&Path> = out.data_files.iter().collect();
        if data.len() != out.data_files.len() {
            return Err(invalid(format!(
                "duplicate data files in snapshot {snapshot}"
            )));
        }
        let mut seen: HashSet<Path> = HashSet::new();
        for v in vectors {
            let data_file = resolve(&v.referenced_data_file)?;
            if !data.contains(&data_file) {
                return Err(invalid(format!(
                    "a deletion vector references {data_file}, which is not a data file of \
                     snapshot {snapshot}"
                )));
            }
            if !seen.insert(data_file.clone()) {
                return Err(invalid(format!(
                    "{data_file} has more than one deletion vector in snapshot {snapshot}"
                )));
            }
            out.deletion_vectors.push(SnapshotDeletionVector {
                data_file,
                puffin_file: resolve(&v.file_path)?,
                content_offset: to_u64(v.content_offset, "content_offset")?,
                content_size_in_bytes: to_u64(v.content_size_in_bytes, "content_size_in_bytes")?,
                puffin_file_size_in_bytes: to_u64(v.file_size_in_bytes, "file_size_in_bytes")?,
                cardinality: to_u64(v.record_count, "record_count")?,
            });
        }
        Ok(IcebergSnapshotPlan {
            files: out,
            file_stats,
            fields,
        })
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
    /// Stream one data file's hidden `__proxima_oid` column, calling `visit`
    /// with every row's 0-based position and its oid (`None` for a null
    /// cell). Positions count EVERY row, so they stay aligned with the file.
    /// Only the oid column is decoded. A file written before the hidden
    /// column existed (pre-P2.4-2a parts carried forward by appends) yields
    /// an empty projection, and fails closed with the file named rather than
    /// panicking on a column access.
    async fn scan_oid_column(
        &self,
        file: &Path,
        mut visit: impl FnMut(u64, Option<&str>),
    ) -> Result<(), StorageError> {
        use arrow_array::Array;
        use futures::TryStreamExt;
        use proximadb_storage_common::object_store_bridge::ObjectStoreBridge as _;
        use proximadb_storage_common::proxima_arrow::SYS_COL_OID;

        let oid_field = arrow_schema::Field::new(SYS_COL_OID, arrow_schema::DataType::Utf8, false);
        let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![oid_field]));
        let mut stream = self
            .read_parquet_batches(file, schema, 8192, None)
            .await
            .map_err(|e| invalid(format!("opening {file:?} oid column: {e}")))?;
        let mut position: u64 = 0;
        while let Some(batch) = stream
            .try_next()
            .await
            .map_err(|e| invalid(format!("reading {file:?} oid column: {e}")))?
        {
            if batch.num_columns() == 0 {
                return Err(invalid(format!(
                    "{file:?}: no {SYS_COL_OID} column (written before the hidden oid \
                         column existed); positions cannot be mapped in this file"
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
                let cell = (!column.is_null(i)).then(|| column.value(i));
                visit(position, cell);
                position += 1;
            }
        }
        Ok(())
    }

    /// EXACT positions of `oids` in each of `files`: every occurrence of every
    /// asked oid, over every row (null cells counted, never matched). Files
    /// holding none are omitted; positions come back sorted. The incremental
    /// MATERIALIZE marks these old positions with deletion vectors, so unlike
    /// `published_positions_of` (first occurrence only, kept for the live
    /// publisher) nothing may be missed. COST: the whole oid column of each
    /// given file, matched in place (no per-row allocation); callers narrow
    /// `files` with the min/max + Bloom index first.
    pub async fn oid_positions_exact(
        &self,
        files: &[Path],
        oids: &[&str],
    ) -> Result<Vec<(Path, Vec<u64>)>, StorageError> {
        let asked: std::collections::HashSet<&str> = oids.iter().copied().collect();
        let mut out = Vec::new();
        for file in files {
            let mut positions = Vec::new();
            self.scan_oid_column(file, |position, cell| {
                if cell.is_some_and(|oid| asked.contains(oid)) {
                    positions.push(position);
                }
            })
            .await?;
            if !positions.is_empty() {
                out.push((file.clone(), positions));
            }
        }
        Ok(out)
    }

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
        let asked: std::collections::HashSet<&str> = oids.iter().copied().collect();
        let mut found_all: std::collections::HashSet<&str> = Default::default();
        let mut per_file = Vec::new();
        for file in files {
            let mut rows: Vec<String> = Vec::new();
            self.scan_oid_column(file, |_, cell| {
                // A null oid cell (a foreign writer's shape; ours are
                // non-null) can never match an asked oid — skip it.
                if let Some(oid) = cell {
                    rows.push(oid.to_string());
                }
            })
            .await?;
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
