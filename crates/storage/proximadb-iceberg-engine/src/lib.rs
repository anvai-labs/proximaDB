//! # proximadb-iceberg-engine — object-store `ObjectStoreBridge` implementation (F2)
//!
//! The durable **base tier** of the decoupled-storage warehouse: a concrete
//! [`ObjectStoreBridge`] over object storage that writes/reads Parquet for the
//! relational/analytics path and raw segment bytes for the vector/PAX path. It is the
//! capstone of the warehouse serialization stack — it owns no new serialization logic, it
//! *composes* the lower foundation:
//!
//! - **F1** [`ProximaObjectStore`] — the `object_store` plumbing (`put`/`get`/`get_range`).
//! - **F1.5** `proxima_arrow::proxima_records_to_record_batch` — schema-driven record→Arrow.
//! - **F1.6** `proxima_parquet` — in-memory `RecordBatch` ⇄ Parquet-`bytes`.
//! - schema inference (`proxima_arrow::infer_proxima_schema`) for the schema-less write API.
//!
//! ```text
//! write_records_to_parquet(path, records)
//!     = infer_proxima_schema(records)                       // F1.5 (inference)
//!     → proxima_records_to_parquet_bytes(records, &schema)  // F1.5 + F1.6
//!     → ProximaObjectStore::put(path, bytes)                // F1
//!
//! read_parquet_batches(path, schema, batch_size)
//!     = ParquetObjectReader::new(inner_store, full_key)      // F1 (footer + range reads)
//!     → ParquetRecordBatchStreamBuilder (projection by name) // parquet async reader
//!     → lazy RecordBatch stream (row groups read on demand)
//! ```
//!
//! ## v1 scope / follow-ups
//! - **Schema-less write.** The trait's `write_records_to_parquet` takes only `&[ProximaRecord]`,
//!   so the file schema is inferred from the records (see `infer_proxima_schema`). When a
//!   catalog-authoritative schema becomes available at the call site (F3 wiring), prefer
//!   passing it instead of inferring.
//! - **Read is footer-driven + projected.** `read_parquet_batches` opens a
//!   [`ParquetObjectReader`], so it fetches only the footer (suffix/range read) and then
//!   range-reads row groups lazily as the stream is polled — no whole-object download. A
//!   non-empty `schema` arg is applied as a column projection (by name); reorder/coercion
//!   to the requested schema and value-predicate row-group pruning (needs a predicate this
//!   signature lacks) remain follow-ups layered on the same metadata.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::ObjectStore;
use object_store::path::Path;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ProjectionMask;
// arrow-rs 59.2 deprecated ParquetObjectReader (wants direct AsyncFileReader impls,
// apache/arrow-rs#10308). Contained here; the migration is a tracked follow-up.
#[allow(deprecated)]
use parquet::arrow::async_reader::ParquetObjectReader;
use proximadb_kernel::error::StorageError;
use proximadb_object_store::ProximaObjectStore;
use proximadb_records::ProximaRecord;
use proximadb_storage_common::object_store_bridge::IcebergSnapshotDeletion as DeletionView;
use proximadb_storage_common::object_store_bridge::IcebergSnapshotField as FieldView;
use proximadb_storage_common::object_store_bridge::IcebergSnapshotFiles as FilesView;
use proximadb_storage_common::object_store_bridge::{CommitOutcome, ObjectStoreBridge};
use proximadb_storage_common::parquet_write_policy::{ParquetLayoutHints, ParquetWritePolicy};
use proximadb_storage_common::proxima_arrow::infer_proxima_schema_strict;
use proximadb_storage_common::proxima_arrow::proxima_records_to_record_batch;
use proximadb_storage_common::proxima_parquet::record_batches_to_parquet_bytes;
use proximadb_storage_common::proxima_schema::ProximaSchema;

/// Parquet writer policy for the warehouse publication tier (MATERIALIZE
/// snapshots + Iceberg interop files): row groups, page index, declared sort
/// order and spec SBBF Bloom filters — see
/// [`proximadb_storage_common::parquet_write_policy`]. Environment-overridable
/// (`PROXIMADB_MATERIALIZE_ROW_GROUP_ROWS`, `PROXIMADB_PARQUET_*`); every setting
/// yields a plain, mixed-read-safe Parquet file.
fn warehouse_write_policy() -> ParquetWritePolicy {
    ParquetWritePolicy::from_env()
}

/// Stamp Iceberg field ids (`PARQUET:field_id`) on the top-level columns: field id
/// `i + 1` for column `i`, the same assignment the publisher's Iceberg schema uses
/// (`materialize_table_to_parquet` numbers `schema.to_arrow_schema()` fields from 1).
/// Iceberg readers resolve Parquet columns by field id; the table's
/// `schema.name-mapping.default` covers files written before this.
fn with_iceberg_field_ids(batch: RecordBatch) -> Result<RecordBatch, StorageError> {
    let schema =
        proximadb_storage_common::parquet_multipart::iceberg_field_id_schema(&batch.schema());
    batch
        .with_schema(schema)
        .map_err(|e| StorageError::Serialization(format!("iceberg field ids: {e}")))
}

/// Iceberg v3 deletion vectors (`deletion-vector-v1` blobs, portable Roaring).
pub mod deletion_vector;
/// Spec-shaped Iceberg v2 manifest / manifest-list (Avro) + TableMetadata (JSON) content.
pub mod iceberg;
/// Iceberg-style atomic manifest commits (optimistic concurrency via `put_if_absent`).
pub mod manifest;
pub mod metadata;
mod snapshot_deletes;
pub use snapshot_deletes::{IcebergSnapshotFiles, SnapshotDeletionVector};
/// Puffin files: the container of a v3 table's deletion vectors.
pub mod puffin;
/// Parquet footer → Iceberg manifest column metrics.
pub mod stats;

/// Map a Parquet read/decode failure into a [`StorageError`] with operation context.
fn read_err(context: &str, e: impl std::fmt::Display) -> StorageError {
    StorageError::Serialization(format!("iceberg-engine: {context}: {e}"))
}

/// How a snapshot commit treats a data file whose Parquet footer cannot be read.
#[derive(Clone, Copy)]
enum FooterPolicy {
    /// A listed object of unknown provenance: record count 0, no column metrics.
    DegradeToEmpty,
    /// A file the publisher wrote: the commit fails.
    Required,
}

/// The table-level inputs of one Iceberg snapshot commit.
#[derive(Clone, Copy)]
struct SnapshotTarget<'a> {
    metadata_prefix: &'a str,
    table_uuid: &'a str,
    table_location: &'a str,
    fields: &'a [iceberg::IcebergField],
    snapshot_id: i64,
    timestamp_ms: i64,
    format_version: iceberg::FormatVersion,
}

/// Map storage-common Iceberg field descriptors (and the `v3` flag) to the engine's own.
/// Public so callers of the inherent publish methods (the live delete writer)
/// can convert the bridge view fields.
pub fn iceberg_fields_of(
    fields: &[proximadb_storage_common::object_store_bridge::IcebergSnapshotField],
    v3: bool,
) -> (Vec<iceberg::IcebergField>, iceberg::FormatVersion) {
    let mapped = fields
        .iter()
        .map(|f| iceberg::IcebergField {
            id: f.id,
            name: f.name.clone(),
            type_name: f.type_name.clone(),
            required: f.required,
        })
        .collect();
    let format_version = if v3 {
        iceberg::FormatVersion::V3
    } else {
        iceberg::FormatVersion::V2
    };
    (mapped, format_version)
}

/// The `current-snapshot-id` of an already-parsed Iceberg TableMetadata
/// json, the id a successor snapshot must exceed. Shared by every in-repo
/// publisher (MATERIALIZE, the bulk-append lane, the live delete publisher)
/// so their id-minting cannot drift (round-5 triage, finding 3). `None`
/// when the metadata carries no current snapshot (callers treat 0).
pub fn current_snapshot_id_of(meta: &serde_json::Value) -> Option<i64> {
    meta.get("current-snapshot-id").and_then(|v| v.as_i64())
}

/// An [`ObjectStoreBridge`] backed by a [`ProximaObjectStore`] (any `object_store` backend:
/// local `file://`, `memory://`, or cloud `s3://`/`gs://`/`az://` when the F1 crate's cloud
/// features are enabled). One bridge serves an entire store/bucket; paths select the object.
#[derive(Clone)]
pub struct IcebergObjectStoreBridge {
    store: ProximaObjectStore,
    /// The URL this bridge's base prefix was opened from, when known. Iceberg
    /// metadata must reference files by absolute URI for external readers, so the
    /// publisher joins bridge-relative keys onto this; without it, it falls back
    /// to bridge-relative keys (readable only by ProximaDB).
    root_url: Option<String>,
}

/// Join a bridge-relative key onto a store root URL (`memory://` + `k` stays
/// `memory://k`; `file:///wh` + `k` is `file:///wh/k`).
fn join_uri(root: &str, key: &str) -> String {
    if root.ends_with("://") {
        format!("{root}{key}")
    } else {
        format!("{}/{key}", root.trim_end_matches('/'))
    }
}

/// The absolute URI of a path v0.4.1 (and earlier) recorded relative to the
/// WAREHOUSE ROOT. Those releases wrote `location` as the absolute table data
/// directory, `<warehouse>/<table-rel>/data`, and every recorded path as
/// `<table-rel>/data/…` (data files) or `<table-rel>/metadata/…` (manifest
/// list, manifests). So the warehouse root is `location` minus
/// `/<table-rel>/data`, where `<table-rel>` is a leading run of the path's
/// components that both makes that suffix match AND is followed by `data` or
/// `metadata`. The second condition settles names that make the split
/// ambiguous (a table named `data` under a warehouse ending in `data`: both
/// `w` + `data/data` and `w/data` + `data` fit the location); the longest
/// such run wins. `None` when nothing qualifies: such a path is not guessed at.
fn legacy_warehouse_uri(location: &str, recorded: &str) -> Option<String> {
    let location = location.trim_end_matches('/');
    let parts: Vec<&str> = recorded.split('/').filter(|c| !c.is_empty()).collect();
    (1..parts.len().saturating_sub(1)).rev().find_map(|k| {
        if !matches!(parts[k], "data" | "metadata") {
            return None;
        }
        let suffix = format!("/{}/data", parts[..k].join("/"));
        location
            .strip_suffix(suffix.as_str())
            .map(|warehouse| join_uri(warehouse, &parts.join("/")))
    })
}

impl IcebergObjectStoreBridge {
    /// Wrap an existing object-store handle. Published Iceberg metadata then uses
    /// bridge-relative paths; call [`with_root_url`](Self::with_root_url) to make
    /// them absolute.
    pub fn new(store: ProximaObjectStore) -> Self {
        Self {
            store,
            root_url: None,
        }
    }

    /// Open the bridge directly from a URL (see [`ProximaObjectStore::from_url`]).
    pub fn from_url(url: &str) -> Result<Self, StorageError> {
        Ok(Self {
            store: ProximaObjectStore::from_url(url)?,
            root_url: Some(url.to_string()),
        })
    }

    /// Record the URL the wrapped store's base prefix corresponds to.
    pub fn with_root_url(mut self, url: impl Into<String>) -> Self {
        self.root_url = Some(url.into());
        self
    }

    /// The absolute URI of a bridge-relative key, or the key itself when the
    /// bridge has no root URL.
    pub fn object_uri(&self, key: &str) -> String {
        match &self.root_url {
            Some(root) => join_uri(root, key),
            None => key.to_string(),
        }
    }

    /// Map a path recorded in Iceberg metadata back to a bridge-relative key.
    ///
    /// Absolute URIs under this bridge's root are stripped to the key. A URI
    /// under the table `location` (but not under this bridge's root — e.g. the
    /// metadata was written through a warehouse-root bridge and is now read
    /// through a table-root bridge) is re-rooted at `table_prefix`, the
    /// bridge-relative prefix of that table. A relative path that v0.4.1-era
    /// metadata recorded relative to the warehouse root (see
    /// `legacy_warehouse_uri`) is rebuilt from `location` and accepted only
    /// when it lies under this bridge's root; every other relative path
    /// passes through.
    fn resolve_recorded_path(
        &self,
        recorded: &str,
        table_location: Option<&str>,
        table_prefix: &str,
    ) -> Path {
        if let Some(key) = self.strip_known_roots(recorded, table_location, table_prefix) {
            return key;
        }
        // v0.4.1 and earlier recorded paths RELATIVE TO THE WAREHOUSE ROOT;
        // rebuild the absolute URI from the document's own `location` and
        // accept it only when it lies under THIS bridge's root. Never through
        // the location re-rooting: a v0.4.1 `location` is the data directory,
        // not the table base it assumes, so it would drop `/data` from a data
        // file. A bridge rooted elsewhere keeps the recorded key as before.
        if !recorded.contains("://")
            && let Some(key) = table_location
                .and_then(|loc| legacy_warehouse_uri(loc, recorded))
                .and_then(|uri| self.strip_root(&uri))
        {
            return key;
        }
        Path::from(recorded)
    }

    /// `uri` as a key under this bridge when it is an absolute URI under the
    /// bridge's root URL; `None` otherwise.
    fn strip_root(&self, uri: &str) -> Option<Path> {
        let root_prefix = join_uri(self.root_url.as_deref()?, "");
        uri.strip_prefix(&root_prefix).map(Path::from)
    }

    /// `recorded` as a key under this bridge when it is an absolute URI under
    /// the bridge's root, or under the table `location` (re-rooted at
    /// `table_prefix`); `None` otherwise.
    fn strip_known_roots(
        &self,
        recorded: &str,
        table_location: Option<&str>,
        table_prefix: &str,
    ) -> Option<Path> {
        if let Some(key) = self.strip_root(recorded) {
            return Some(key);
        }
        let loc_prefix = join_uri(table_location?, "");
        let rest = recorded.strip_prefix(&loc_prefix)?;
        Some(if table_prefix.is_empty() {
            Path::from(rest)
        } else {
            Path::from(format!("{table_prefix}/{rest}"))
        })
    }

    /// Borrow the underlying [`ProximaObjectStore`].
    pub fn object_store(&self) -> &ProximaObjectStore {
        &self.store
    }

    /// Resolve a bridge-relative path to the concrete key used by the wrapped
    /// object store. Range-based readers use this to avoid a full object GET.
    pub fn full_object_path(&self, path: &Path) -> Path {
        self.store.full_path(path)
    }

    /// Read the data-file keys recorded by snapshot `version` of the manifest log at
    /// `manifest_prefix` (parses the v1 newline-delimited body). Each returned path is
    /// directly consumable by
    /// [`read_parquet_batches`](ObjectStoreBridge::read_parquet_batches).
    pub async fn read_snapshot(
        &self,
        manifest_prefix: &str,
        version: u64,
    ) -> Result<Vec<Path>, StorageError> {
        let committer = manifest::ManifestCommitter::new(self.store.clone(), manifest_prefix);
        let bytes = committer.read_manifest(version).await?;
        let text =
            std::str::from_utf8(&bytes).map_err(|e| read_err("decode manifest body (utf8)", e))?;
        Ok(text
            .lines()
            .filter(|l| !l.is_empty())
            .map(Path::from)
            .collect())
    }

    /// Footer metadata of a Parquet object (no row data).
    async fn parquet_footer(
        &self,
        path: &Path,
    ) -> Result<std::sync::Arc<parquet::file::metadata::ParquetMetaData>, StorageError> {
        #[allow(deprecated)]
        let reader = ParquetObjectReader::new(self.store.store(), self.full_object_path(path));
        let builder = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| read_err("open parquet footer", e))?;
        Ok(builder.metadata().clone())
    }

    /// Publish a snapshot as **spec-shaped Iceberg** (v3 by default) of every `*.parquet`
    /// object under `data_prefix`, as the successor of the latest committed metadata
    /// version. Returns the committed metadata version (or a `Conflict` to rebase onto).
    ///
    /// Listing selects whatever parquet is present, so this is only correct for a prefix
    /// that holds exactly one generation. Publishers that write generation-named parts
    /// use [`publish_iceberg_files`](Self::publish_iceberg_files) with the explicit list.
    #[allow(clippy::too_many_arguments)]
    pub async fn publish_iceberg_snapshot(
        &self,
        data_prefix: &Path,
        metadata_prefix: &str,
        table_uuid: &str,
        table_location: &str,
        fields: &[iceberg::IcebergField],
        snapshot_id: i64,
        timestamp_ms: i64,
        format_version: iceberg::FormatVersion,
    ) -> Result<CommitOutcome, StorageError> {
        let mut keys: Vec<Path> = self
            .list_objects(data_prefix)
            .await?
            .into_iter()
            .filter(|k| k.as_ref().ends_with(".parquet"))
            .collect();
        keys.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
        let committer = metadata::MetadataCommitter::new(self.store.clone(), metadata_prefix);
        let parent = committer.latest_version().await?;
        self.commit_iceberg_snapshot(
            &keys,
            &[],
            parent,
            FooterPolicy::DegradeToEmpty,
            &SnapshotTarget {
                metadata_prefix,
                table_uuid,
                table_location,
                fields,
                snapshot_id,
                timestamp_ms,
                format_version,
            },
        )
        .await
    }

    /// Publish exactly `files` (bridge-relative keys of parquet objects the caller
    /// wrote) as a spec-shaped Iceberg snapshot committed as the successor of metadata
    /// version `parent` (`None` ⇒ the first version). Returns the committed version.
    ///
    /// Unlike [`publish_iceberg_snapshot`](Self::publish_iceberg_snapshot) nothing is
    /// listed: a multi-part, multi-generation data prefix publishes only the generation
    /// named here. The Avro manifest and manifest list are generation-unique and written
    /// create-only, and every footer must be readable (the caller just wrote the files).
    /// Losing the race for the successor slot — another publisher committed after
    /// `parent`, or claimed its Avro paths — is an `Err(AlreadyExists)`, never success:
    /// the caller decides whether to rebase.
    #[allow(clippy::too_many_arguments)]
    pub async fn publish_iceberg_files(
        &self,
        files: &[Path],
        parent: Option<u64>,
        metadata_prefix: &str,
        table_uuid: &str,
        table_location: &str,
        fields: &[iceberg::IcebergField],
        snapshot_id: i64,
        timestamp_ms: i64,
        format_version: iceberg::FormatVersion,
    ) -> Result<u64, StorageError> {
        self.publish_files_inner(
            files,
            &[],
            parent,
            metadata_prefix,
            table_uuid,
            table_location,
            fields,
            snapshot_id,
            timestamp_ms,
            format_version,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn publish_files_inner(
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
        format_version: iceberg::FormatVersion,
    ) -> Result<u64, StorageError> {
        if files.is_empty() {
            return Err(StorageError::Serialization(
                "iceberg-engine: publish_iceberg_files needs at least one data file".into(),
            ));
        }
        if !deletes.is_empty() && format_version != iceberg::FormatVersion::V3 {
            return Err(StorageError::Serialization(
                "iceberg-engine: deletion vectors need format version 3".into(),
            ));
        }
        let target = SnapshotTarget {
            metadata_prefix,
            table_uuid,
            table_location,
            fields,
            snapshot_id,
            timestamp_ms,
            format_version,
        };
        match self
            .commit_iceberg_snapshot(files, deletes, parent, FooterPolicy::Required, &target)
            .await?
        {
            CommitOutcome::Committed(version) => Ok(version),
            CommitOutcome::Conflict { latest } => Err(StorageError::AlreadyExists(format!(
                "iceberg-engine: metadata under {metadata_prefix} moved past parent \
                 {parent:?} (latest {latest:?}); snapshot {snapshot_id} not published"
            ))),
        }
    }

    /// The shared commit: data-file metadata from the footers, lineage from `parent`,
    /// generation-unique Avro manifest + list written create-only, then the metadata CAS.
    async fn commit_iceberg_snapshot(
        &self,
        files: &[Path],
        deletes: &[iceberg::DeleteFileMeta],
        parent: Option<u64>,
        footers: FooterPolicy,
        target: &SnapshotTarget<'_>,
    ) -> Result<CommitOutcome, StorageError> {
        let SnapshotTarget {
            metadata_prefix,
            table_uuid,
            table_location,
            fields,
            snapshot_id,
            timestamp_ms,
            format_version,
        } = *target;
        // 1. Data files with size (HEAD) + record count and column metrics (footer).
        let mut data_files = Vec::with_capacity(files.len());
        for key in files {
            let size = self.store.object_size(key).await?;
            let (record_count, column_stats) = match (self.parquet_footer(key).await, footers) {
                (Ok(meta), _) => (
                    meta.file_metadata().num_rows(),
                    stats::iceberg_column_stats(&meta, fields),
                ),
                // A listed object of unknown provenance degrades to count 0 / no
                // metrics (never wrong); a file the caller wrote must be readable.
                (Err(_), FooterPolicy::DegradeToEmpty) => (0, Vec::new()),
                (Err(e), FooterPolicy::Required) => return Err(e),
            };
            data_files.push(iceberg::DataFileMeta {
                file_path: self.object_uri(key.as_ref()),
                record_count,
                file_size_in_bytes: size as i64,
                column_stats,
            });
        }
        let added_records: i64 = data_files.iter().map(|f| f.record_count).sum();
        let added_files = data_files.len() as i64;

        // 2. Lineage from the parent metadata document, if any.
        let committer = metadata::MetadataCommitter::new(self.store.clone(), metadata_prefix);
        let version = metadata::MetadataCommitter::successor(parent)?;
        let parent_json = match parent {
            Some(v) => Some(
                String::from_utf8(committer.read_metadata(v).await?.to_vec())
                    .map_err(|e| read_err("parent metadata utf8", e))?,
            ),
            None => None,
        };
        let parent_location = parent.map(|v| self.object_uri(&committer.metadata_location(v)));
        let lineage =
            iceberg::Lineage::from_parent(parent_json.as_deref(), parent_location.as_deref())?;
        // Snapshot ids must advance past the parent's current snapshot: an equal id
        // would make the snapshot its own ancestor and a lower one rewinds history.
        // Same-millisecond publishers (ids derived from a clock) hit this; a real
        // collision surfaces here instead of corrupting the lineage.
        if let Some(parent_snapshot_id) = lineage.parent_snapshot_id
            && snapshot_id <= parent_snapshot_id
        {
            return Err(StorageError::Serialization(format!(
                "iceberg-engine: snapshot id {snapshot_id} must exceed the parent snapshot \
                 {parent_snapshot_id} under {metadata_prefix}"
            )));
        }
        // ADR-097 D3 format policy: a table's format version never goes down. The
        // caller's `format_version` is a floor for a new table; once a snapshot
        // commits v3, every successor commits v3 even when the requested schema
        // would pick v2 — a republication must not implicitly downgrade an
        // upgraded table. (Upgrades stay caller-driven: a v2 parent with a v3
        // request publishes v3.)
        let format_version = match lineage.parent_format_version {
            Some(parent_v) if parent_v > format_version.as_i32() => {
                iceberg::FormatVersion::from_i32(parent_v).unwrap_or(format_version)
            }
            _ => format_version,
        };

        // 3. The Avro manifest + manifest list, named by the metadata version they are
        //    committed under, so two generations (or two publishers racing for one slot)
        //    never overwrite each other's files. Create-only: the loser of a race sees
        //    `AlreadyExists` here and reports a conflict instead of clobbering the
        //    winner's manifest out from under its committed metadata.
        let schema_json = iceberg::schema_json(fields).to_string();
        let manifest_bytes = iceberg::write_manifest(
            &data_files,
            snapshot_id,
            Some(&iceberg::ManifestHeader {
                schema_json: &schema_json,
                format_version,
            }),
        )?;
        let manifest_path = Path::from(format!(
            "{metadata_prefix}/v{version:020}-{snapshot_id}-m0.avro"
        ));
        let manifest_length = manifest_bytes.len() as i64;
        if let Some(conflict) = self
            .put_generation_object(&committer, &manifest_path, manifest_bytes)
            .await?
        {
            return Ok(conflict);
        }
        let mut manifest_list = vec![iceberg::ManifestFileMeta {
            manifest_path: self.object_uri(manifest_path.as_ref()),
            manifest_length,
            added_files_count: added_files as i32,
            added_rows_count: added_records,
            added_snapshot_id: snapshot_id,
            sequence_number: lineage.sequence_number,
            first_row_id: (format_version == iceberg::FormatVersion::V3)
                .then_some(lineage.first_row_id),
            content: iceberg::ManifestContent::Data,
        }];
        // The snapshot's deletion vectors, in a second (delete) manifest whose
        // entries inherit this snapshot's sequence number, so they apply to its
        // data files. Row ids are assigned to data manifests only.
        if !deletes.is_empty() {
            let delete_bytes = iceberg::write_delete_manifest(
                deletes,
                snapshot_id,
                &iceberg::ManifestHeader {
                    schema_json: &schema_json,
                    format_version,
                },
            )?;
            let delete_path = Path::from(format!(
                "{metadata_prefix}/v{version:020}-{snapshot_id}-m1.avro"
            ));
            let delete_length = delete_bytes.len() as i64;
            if let Some(conflict) = self
                .put_generation_object(&committer, &delete_path, delete_bytes)
                .await?
            {
                return Ok(conflict);
            }
            manifest_list.push(iceberg::ManifestFileMeta {
                manifest_path: self.object_uri(delete_path.as_ref()),
                manifest_length: delete_length,
                added_files_count: deletes.len() as i32,
                added_rows_count: deletes.iter().map(|d| d.record_count).sum(),
                added_snapshot_id: snapshot_id,
                sequence_number: lineage.sequence_number,
                first_row_id: None,
                content: iceberg::ManifestContent::Deletes,
            });
        }
        let list_bytes = iceberg::write_manifest_list(&manifest_list)?;
        let list_path = Path::from(format!(
            "{metadata_prefix}/snap-{snapshot_id}-v{version:020}-list.avro"
        ));
        if let Some(conflict) = self
            .put_generation_object(&committer, &list_path, list_bytes)
            .await?
        {
            return Ok(conflict);
        }

        // 4. Build + atomically commit the TableMetadata.
        let list_uri = self.object_uri(list_path.as_ref());
        let json = iceberg::build_table_metadata(
            &iceberg::SnapshotCommit {
                format_version,
                table_uuid,
                location: table_location,
                fields,
                snapshot_id,
                manifest_list_location: &list_uri,
                timestamp_ms,
                added_files,
                added_records,
            },
            &lineage,
        )?;
        committer.commit(parent, Bytes::from(json)).await
    }

    /// Create-only write of one generation-named metadata object. `Some(Conflict)` when
    /// another publisher already claimed the name (it is racing for the same version).
    async fn put_generation_object(
        &self,
        committer: &metadata::MetadataCommitter,
        path: &Path,
        bytes: Vec<u8>,
    ) -> Result<Option<CommitOutcome>, StorageError> {
        match self.store.put_if_absent(path, Bytes::from(bytes)).await {
            Ok(()) => Ok(None),
            Err(StorageError::AlreadyExists(_)) => Ok(Some(CommitOutcome::Conflict {
                latest: committer.latest_version().await?,
            })),
            Err(e) => Err(e),
        }
    }

    /// Absolute location of metadata `version` under `metadata_prefix` — what an
    /// Iceberg catalog serves as `metadata-location`.
    pub fn metadata_file_uri(&self, metadata_prefix: &str, version: u64) -> String {
        let committer = metadata::MetadataCommitter::new(self.store.clone(), metadata_prefix);
        self.object_uri(&committer.metadata_location(version))
    }

    /// Resolve the data-file keys of the current snapshot in the Iceberg metadata log at
    /// `metadata_prefix` (decodes `TableMetadata` → manifest list → manifests). Each returned
    /// path is consumable by [`read_parquet_batches`](ObjectStoreBridge::read_parquet_batches).
    ///
    /// A snapshot with deletion vectors is an error here: its data files alone would
    /// return deleted rows. Readers that apply them use
    /// [`read_iceberg_snapshot_with_deletes`](Self::read_iceberg_snapshot_with_deletes).
    pub async fn read_iceberg_snapshot(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<Vec<Path>, StorageError> {
        let snapshot = self
            .read_iceberg_snapshot_with_deletes(metadata_prefix, version)
            .await?;
        if !snapshot.deletion_vectors.is_empty() {
            return Err(StorageError::Serialization(format!(
                "iceberg-engine: snapshot {version} under {metadata_prefix} has {} deletion \
                 vectors, which this reader does not apply",
                snapshot.deletion_vectors.len()
            )));
        }
        Ok(snapshot.data_files)
    }
}

#[async_trait]
impl ObjectStoreBridge for IcebergObjectStoreBridge {
    fn inner_store(&self) -> Arc<dyn ObjectStore> {
        self.store.store()
    }

    fn object_store_path(&self, key: &Path) -> Path {
        self.full_object_path(key)
    }

    /// Footer-driven, range-read, lazily-streamed Parquet read.
    ///
    /// Instead of a full-object `get` followed by an in-memory decode, this opens
    /// a [`ParquetObjectReader`] over the wrapped object store, which fetches only
    /// the Parquet footer (via a suffix/range read), parses the file metadata, and
    /// then range-reads row groups **on demand** as the returned stream is polled.
    /// On a cloud store that turns a whole-file download into a footer read plus
    /// per-row-group ranges; on small in-memory objects it is equivalent.
    ///
    /// `schema` is honored as a **column projection**: when it carries fields, only
    /// the file columns whose (root) names appear in it are read (matched by name
    /// via [`ProjectionMask::columns`]; requested names absent from the file are
    /// skipped, and file order is preserved — reordering/coercion to the requested
    /// schema is a follow-up). An empty schema preserves the legacy "all columns"
    /// contract. Value-predicate row-group pruning needs a predicate this signature
    /// does not carry; it remains a follow-up layered on the same metadata.
    async fn read_parquet_batches(
        &self,
        path: &Path,
        schema: Arc<ArrowSchema>,
        batch_size: usize,
        _tenant_id: Option<&str>,
    ) -> Result<BoxStream<'static, Result<RecordBatch, StorageError>>, StorageError> {
        // Use the raw inner store + base-resolved key: ParquetObjectReader issues
        // its own range reads, so it must address the concrete object key the way
        // ProximaObjectStore's own methods do (which apply the base prefix).
        // arrow-rs 59.2 deprecated ParquetObjectReader (wants direct AsyncFileReader impls,
        // apache/arrow-rs#10308). Contained here; the migration is a tracked follow-up.
        #[allow(deprecated)]
        let reader = ParquetObjectReader::new(self.store.store(), self.full_object_path(path));
        let mut builder = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| read_err("open parquet stream (footer)", e))?
            .with_batch_size(batch_size);

        if !schema.fields().is_empty() {
            let mask = ProjectionMask::columns(
                builder.parquet_schema(),
                schema.fields().iter().map(|f| f.name().as_str()),
            );
            builder = builder.with_projection(mask);
        }

        let stream = builder
            .build()
            .map_err(|e| read_err("build parquet stream", e))?;
        Ok(stream
            .map(|batch| batch.map_err(|e| read_err("read parquet batch", e)))
            .boxed())
    }

    async fn write_records_to_parquet(
        &self,
        path: &Path,
        records: &[ProximaRecord],
        tenant_id: Option<&str>,
    ) -> Result<(), StorageError> {
        // Schema-less entry point: infer the schema from the records, then
        // delegate to the schema-explicit writer so both paths share one
        // implementation. The tenant flows through to the single I/O boundary.
        //
        // TD-USCHEMA-1: inference is a *union schema* (U-Schema, arXiv:2105.06494
        // §2.3) and collapsing structural variations is defined-lossy — a key
        // seen with two types keeps the first and NULLs the rest. On THIS path
        // there is no msgpack PROPS tail, so the typed column is the only copy
        // and such a value would be silently dropped into a well-formed Parquet
        // file. Fail closed instead; the error names the key, both types, and a
        // diverging record.
        let schema = infer_proxima_schema_strict(records)?;
        self.write_records_to_parquet_with_schema(path, records, &schema, tenant_id)
            .await
    }

    /// Schema-explicit write (overrides the trait default): serialize `records`
    /// through the CALLER-PROVIDED catalog-authoritative [`ProximaSchema`] so the
    /// file's column set/order/types/nullability match the catalog exactly —
    /// including columns absent from every record (written all-null), which
    /// inference would drop. The single I/O boundary is one atomic object `put`.
    ///
    /// `tenant_id` carries the write's owner to the I/O boundary for the billing
    /// follow-up (`proximadb_object_store_ops_total` / `_storage_bytes_seconds`);
    /// no telemetry sink is wired here yet.
    async fn write_records_to_parquet_with_schema(
        &self,
        path: &Path,
        records: &[ProximaRecord],
        schema: &ProximaSchema,
        tenant_id: Option<&str>,
    ) -> Result<(), StorageError> {
        self.write_records_to_parquet_with_layout(path, records, schema, &[], tenant_id)
            .await
    }

    /// Layout-aware write: the warehouse policy plus the caller's sort order
    /// (declared as Parquet `sorting_columns`). The single I/O boundary is one
    /// atomic object `put`.
    async fn write_records_to_parquet_with_layout(
        &self,
        path: &Path,
        records: &[ProximaRecord],
        schema: &ProximaSchema,
        sort_columns: &[String],
        _tenant_id: Option<&str>,
    ) -> Result<(), StorageError> {
        let hints = ParquetLayoutHints {
            sort_columns: sort_columns.to_vec(),
        };
        let batch = with_iceberg_field_ids(proxima_records_to_record_batch(records, schema)?)?;
        let props = warehouse_write_policy().writer_properties(&batch, &hints);
        let arrow_schema = batch.schema();
        let bytes = record_batches_to_parquet_bytes(&[batch], arrow_schema, Some(props))?;
        self.store.put(path, Bytes::from(bytes)).await
    }

    async fn fetch_vector_segment(
        &self,
        path: &Path,
        _tenant_id: Option<&str>,
    ) -> Result<Vec<u8>, StorageError> {
        Ok(self.store.get(path).await?.to_vec())
    }

    /// Metadata-only size (HEAD) — the prerequisite for footer-first ranged
    /// reads; no object body is transferred.
    async fn vector_segment_size(
        &self,
        path: &Path,
        _tenant_id: Option<&str>,
    ) -> Result<u64, StorageError> {
        self.store.object_size(path).await
    }

    /// True ranged GET over object storage (`Range: bytes=offset-…`): only the
    /// requested slice leaves the wire, so column/footer reads don't pull the
    /// whole segment.
    async fn fetch_vector_segment_range(
        &self,
        path: &Path,
        offset: u64,
        length: u64,
        _tenant_id: Option<&str>,
    ) -> Result<Vec<u8>, StorageError> {
        Ok(self
            .store
            .get_range(path, offset..offset + length)
            .await?
            .to_vec())
    }

    /// TD-167 / ADR-034 P1: batched multi-range GET via `object_store::get_ranges`,
    /// which coalesces adjacent ranges and issues the reads concurrently — so K
    /// surviving block bodies cost ~one round-trip instead of K serial GETs.
    async fn fetch_vector_segment_ranges(
        &self,
        path: &Path,
        ranges: &[std::ops::Range<u64>],
        _tenant_id: Option<&str>,
    ) -> Result<Vec<Vec<u8>>, StorageError> {
        Ok(self
            .store
            .get_ranges(path, ranges)
            .await?
            .into_iter()
            .map(|b| b.to_vec())
            .collect())
    }

    async fn persist_vector_segment(
        &self,
        path: &Path,
        data: &[u8],
        _tenant_id: Option<&str>,
    ) -> Result<(), StorageError> {
        self.store.put(path, Bytes::copy_from_slice(data)).await
    }

    /// Base-aware override of the trait default: list through [`ProximaObjectStore`]
    /// (which applies the store's base prefix) and strip that base from each
    /// returned key, so the paths are base-relative and round-trip through
    /// `read_parquet_batches` / `fetch_vector_segment` (which re-apply the base).
    async fn list_objects(&self, prefix: &Path) -> Result<Vec<Path>, StorageError> {
        let base = self.store.base().as_ref().to_string();
        let metas = self.store.list(Some(prefix)).await?;
        Ok(metas
            .into_iter()
            .map(|meta| {
                if base.is_empty() {
                    meta.location
                } else {
                    let key = meta.location.as_ref();
                    let relative = key
                        .strip_prefix(&base)
                        .map(|rest| rest.trim_start_matches('/'))
                        .unwrap_or(key);
                    Path::from(relative)
                }
            })
            .collect())
    }

    async fn latest_manifest_version(
        &self,
        manifest_prefix: &str,
    ) -> Result<Option<u64>, StorageError> {
        let committer = manifest::ManifestCommitter::new(self.store.clone(), manifest_prefix);
        committer.latest_version().await
    }

    async fn publish_snapshot(
        &self,
        data_prefix: &Path,
        manifest_prefix: &str,
        parent: Option<u64>,
    ) -> Result<CommitOutcome, StorageError> {
        let mut keys: Vec<String> = self
            .list_objects(data_prefix)
            .await?
            .iter()
            .map(|p| p.as_ref().to_string())
            .collect();
        keys.sort(); // deterministic manifest body regardless of list order
        let body = keys.join("\n");
        let committer = manifest::ManifestCommitter::new(self.store.clone(), manifest_prefix);
        committer.commit(parent, Bytes::from(body)).await
    }

    async fn latest_metadata_version(
        &self,
        metadata_prefix: &str,
    ) -> Result<Option<u64>, StorageError> {
        let committer = metadata::MetadataCommitter::new(self.store.clone(), metadata_prefix);
        committer.latest_version().await
    }

    async fn read_table_metadata(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<Vec<u8>, StorageError> {
        let committer = metadata::MetadataCommitter::new(self.store.clone(), metadata_prefix);
        Ok(committer.read_metadata(version).await?.to_vec())
    }

    async fn commit_table_metadata(
        &self,
        metadata_prefix: &str,
        parent: Option<u64>,
        metadata: Vec<u8>,
    ) -> Result<CommitOutcome, StorageError> {
        let committer = metadata::MetadataCommitter::new(self.store.clone(), metadata_prefix);
        committer.commit(parent, Bytes::from(metadata)).await
    }

    /// Map the storage-common field descriptors to Iceberg fields and delegate to the
    /// inherent [`publish_iceberg_snapshot`](IcebergObjectStoreBridge::publish_iceberg_snapshot).
    async fn publish_iceberg_table(
        &self,
        data_prefix: &Path,
        metadata_prefix: &str,
        table_uuid: &str,
        table_location: &str,
        fields: &[proximadb_storage_common::object_store_bridge::IcebergSnapshotField],
        snapshot_id: i64,
        timestamp_ms: i64,
        v3: bool,
    ) -> Result<CommitOutcome, StorageError> {
        let (mapped, format_version) = iceberg_fields_of(fields, v3);
        self.publish_iceberg_snapshot(
            data_prefix,
            metadata_prefix,
            table_uuid,
            table_location,
            &mapped,
            snapshot_id,
            timestamp_ms,
            format_version,
        )
        .await
    }

    async fn publish_iceberg_files(
        &self,
        files: &[Path],
        parent: Option<u64>,
        metadata_prefix: &str,
        table_uuid: &str,
        table_location: &str,
        fields: &[proximadb_storage_common::object_store_bridge::IcebergSnapshotField],
        snapshot_id: i64,
        timestamp_ms: i64,
        v3: bool,
    ) -> Result<u64, StorageError> {
        let (mapped, format_version) = iceberg_fields_of(fields, v3);
        IcebergObjectStoreBridge::publish_iceberg_files(
            self,
            files,
            parent,
            metadata_prefix,
            table_uuid,
            table_location,
            &mapped,
            snapshot_id,
            timestamp_ms,
            format_version,
        )
        .await
    }

    async fn read_iceberg_table(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<Vec<Path>, StorageError> {
        self.read_iceberg_snapshot(metadata_prefix, version).await
    }

    async fn read_iceberg_snapshot_files(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<FilesView, StorageError> {
        let snapshot = self
            .read_iceberg_snapshot_with_deletes(metadata_prefix, version)
            .await?;
        Ok(FilesView {
            data_files: snapshot.data_files,
            deletion_vectors: snapshot
                .deletion_vectors
                .into_iter()
                .map(|v| DeletionView {
                    data_file: v.data_file,
                    puffin_file: v.puffin_file,
                    content_offset: v.content_offset,
                    content_size_in_bytes: v.content_size_in_bytes,
                    puffin_file_size_in_bytes: v.puffin_file_size_in_bytes,
                    cardinality: v.cardinality,
                })
                .collect(),
        })
    }

    async fn publish_iceberg_files_deleting(
        &self,
        files: &[Path],
        deletes: &[DeletionView],
        parent: Option<u64>,
        metadata_prefix: &str,
        table_uuid: &str,
        table_location: &str,
        fields: &[FieldView],
        snapshot_id: i64,
        timestamp_ms: i64,
    ) -> Result<u64, StorageError> {
        // Map the carried view entries back to delete-manifest metas. A carried
        // vector names its files by the URI recorded when it was published;
        // `object_uri` of the bridge-relative key reproduces exactly that form.
        let overflow = |what: &str| {
            StorageError::Serialization(format!("iceberg-engine: carried {what} overflows i64"))
        };
        let metas: Vec<iceberg::DeleteFileMeta> = deletes
            .iter()
            .map(|d| {
                Ok(iceberg::DeleteFileMeta {
                    file_path: self.object_uri(d.puffin_file.as_ref()),
                    file_size_in_bytes: i64::try_from(d.puffin_file_size_in_bytes)
                        .map_err(|_| overflow("puffin size"))?,
                    referenced_data_file: self.object_uri(d.data_file.as_ref()),
                    content_offset: i64::try_from(d.content_offset)
                        .map_err(|_| overflow("content_offset"))?,
                    content_size_in_bytes: i64::try_from(d.content_size_in_bytes)
                        .map_err(|_| overflow("content_size_in_bytes"))?,
                    record_count: i64::try_from(d.cardinality)
                        .map_err(|_| overflow("cardinality"))?,
                })
            })
            .collect::<Result<Vec<_>, StorageError>>()?;
        self.publish_iceberg_files_with_deletes(
            files,
            &metas,
            parent,
            metadata_prefix,
            table_uuid,
            table_location,
            &iceberg_fields_of(fields, false).0,
            snapshot_id,
            timestamp_ms,
        )
        .await
    }

    async fn published_oid_positions(
        &self,
        files: &[Path],
        oids: &[&str],
    ) -> Result<Vec<proximadb_storage_common::object_store_bridge::FileOidPositions>, StorageError>
    {
        Ok(self
            .oid_positions_exact(files, oids)
            .await?
            .into_iter()
            .map(|(file, positions)| {
                proximadb_storage_common::object_store_bridge::FileOidPositions { file, positions }
            })
            .collect())
    }

    async fn read_deletion_vector_positions(
        &self,
        vector: &DeletionView,
    ) -> Result<Vec<u64>, StorageError> {
        let engine_form = snapshot_deletes::SnapshotDeletionVector {
            data_file: vector.data_file.clone(),
            puffin_file: vector.puffin_file.clone(),
            content_offset: vector.content_offset,
            content_size_in_bytes: vector.content_size_in_bytes,
            puffin_file_size_in_bytes: vector.puffin_file_size_in_bytes,
            cardinality: vector.cardinality,
        };
        Ok(self.read_deletion_vector(&engine_form).await?.positions())
    }

    async fn write_deletion_vector_positions(
        &self,
        puffin_key: &Path,
        vectors: &[(Path, Vec<u64>)],
    ) -> Result<Vec<DeletionView>, StorageError> {
        let encoded: Vec<(Path, deletion_vector::PositionDeletes)> = vectors
            .iter()
            .map(|(file, positions)| {
                let mut deletes = deletion_vector::PositionDeletes::new();
                for position in positions {
                    deletes.insert(*position);
                }
                (file.clone(), deletes)
            })
            .collect();
        let borrowed: Vec<(Path, &deletion_vector::PositionDeletes)> =
            encoded.iter().map(|(file, d)| (file.clone(), d)).collect();
        let metas = self.write_deletion_vectors(puffin_key, &borrowed).await?;
        // Map each written vector back to the caller's bridge-relative data
        // file by URI (never by output order).
        let by_uri: std::collections::HashMap<String, &Path> = vectors
            .iter()
            .map(|(file, _)| (self.object_uri(file.as_ref()), file))
            .collect();
        let negative = |what: &str| {
            StorageError::Serialization(format!("iceberg-engine: written {what} is negative"))
        };
        metas
            .into_iter()
            .map(|meta| {
                let data_file = by_uri.get(&meta.referenced_data_file).ok_or_else(|| {
                    StorageError::Serialization(format!(
                        "iceberg-engine: a written vector names {} which was not asked for",
                        meta.referenced_data_file
                    ))
                })?;
                Ok(DeletionView {
                    data_file: (*data_file).clone(),
                    puffin_file: puffin_key.clone(),
                    content_offset: u64::try_from(meta.content_offset)
                        .map_err(|_| negative("content_offset"))?,
                    content_size_in_bytes: u64::try_from(meta.content_size_in_bytes)
                        .map_err(|_| negative("content_size_in_bytes"))?,
                    puffin_file_size_in_bytes: u64::try_from(meta.file_size_in_bytes)
                        .map_err(|_| negative("puffin size"))?,
                    cardinality: u64::try_from(meta.record_count)
                        .map_err(|_| negative("cardinality"))?,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field};
    use object_store::memory::InMemory;
    use proximadb_records::{EmbeddingCell, ProximaTreeNode, ProximaValue};

    fn bridge() -> IcebergObjectStoreBridge {
        IcebergObjectStoreBridge::new(ProximaObjectStore::new(Arc::new(InMemory::new())))
    }

    fn record(oid: &str, props: Vec<(&str, ProximaValue)>) -> ProximaRecord {
        let mut r = ProximaRecord {
            oid: oid.to_string(),
            ..Default::default()
        };
        for (k, v) in props {
            r.props.insert(k.to_string(), ProximaTreeNode::Value(v));
        }
        r
    }

    /// records → write_records_to_parquet → read_parquet_batches must preserve rows + values.
    #[tokio::test]
    async fn records_round_trip_through_object_store() {
        let b = bridge();
        let path = Path::from("warehouse/t/data.parquet");

        let mut r0 = record(
            "r0",
            vec![
                ("name", ProximaValue::String("alice".into())),
                ("age", ProximaValue::Int64(30)),
            ],
        );
        r0.embeddings
            .push(EmbeddingCell::new_fp32("", "", 0, vec![1.0, 2.0, 3.0]));
        let r1 = record(
            "r1",
            vec![
                ("name", ProximaValue::String("bob".into())),
                ("age", ProximaValue::Int64(41)),
            ],
        );

        b.write_records_to_parquet(&path, &[r0, r1], None)
            .await
            .unwrap();

        let mut stream = b
            .read_parquet_batches(&path, Arc::new(ArrowSchema::empty()), 1024, None)
            .await
            .unwrap();

        let mut batches = Vec::new();
        while let Some(batch) = stream.next().await {
            batches.push(batch.unwrap());
        }
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);

        let batch = &batches[0];
        let names = batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "alice");
        assert_eq!(names.value(1), "bob");
        let ages = batch
            .column_by_name("age")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ages.value(0), 30);
        assert_eq!(ages.value(1), 41);
        // The inferred dense-vector column round-tripped as a column too.
        assert!(batch.column_by_name("vector").is_some());
    }

    /// Published Parquet carries Iceberg field ids (`i + 1`) on its columns, so
    /// Iceberg readers resolve columns by id rather than by name.
    #[test]
    fn published_parquet_columns_carry_iceberg_field_ids() {
        use parquet::file::reader::FileReader;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(StringArray::from(vec![Some("x")])),
            ],
        )
        .unwrap();
        let stamped = with_iceberg_field_ids(batch).unwrap();
        let bytes =
            record_batches_to_parquet_bytes(std::slice::from_ref(&stamped), stamped.schema(), None)
                .unwrap();
        let reader =
            parquet::file::serialized_reader::SerializedFileReader::new(Bytes::from(bytes))
                .unwrap();
        let descr = reader.metadata().file_metadata().schema_descr_ptr();
        let ids: Vec<i32> = descr
            .root_schema()
            .get_fields()
            .iter()
            .map(|f| f.get_basic_info().id())
            .collect();
        assert_eq!(ids, vec![1, 2], "field ids land in the Parquet schema");
    }

    /// ADR-097 D3: a URL-rooted bridge records absolute URIs (what external readers
    /// need), and a bridge opened at the TABLE root — the TD-USUB-10 manifest-driven
    /// reader — resolves them back to keys. Before absolute URIs, that reader got
    /// warehouse-relative keys it could not resolve and silently fell back to
    /// listing. A second publish chains onto the first.
    #[tokio::test]
    async fn absolute_uris_resolve_from_warehouse_and_table_roots() {
        let root = "memory:///warehouse";
        let wh = IcebergObjectStoreBridge::from_url(root).unwrap();
        let data_key = "t/ns/tbl/data/part-0.parquet";
        let r0 = record("r0", vec![("age", ProximaValue::Int64(7))]);
        wh.write_records_to_parquet(&Path::from(data_key), &[r0], None)
            .await
            .unwrap();
        let fields = vec![iceberg::IcebergField {
            id: 1,
            name: "age".into(),
            type_name: "long".into(),
            required: false,
        }];
        let table_location = format!("{root}/t/ns/tbl");
        for (snapshot_id, expected) in [
            (11, CommitOutcome::Committed(0)),
            (12, CommitOutcome::Committed(1)),
        ] {
            let out = wh
                .publish_iceberg_snapshot(
                    &Path::from("t/ns/tbl/data"),
                    "t/ns/tbl/metadata",
                    "uuid-abs",
                    &table_location,
                    &fields,
                    snapshot_id,
                    1_700_000_000_000 + snapshot_id,
                    iceberg::FormatVersion::V2,
                )
                .await
                .unwrap();
            assert_eq!(out, expected);
        }

        let meta = wh
            .read_table_metadata("t/ns/tbl/metadata", 1)
            .await
            .unwrap();
        let mj: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(mj["location"], table_location);
        assert_eq!(mj["snapshots"][0]["parent-snapshot-id"], 11);
        assert_eq!(mj["last-sequence-number"], 2);
        let list = mj["snapshots"][0]["manifest-list"].as_str().unwrap();
        assert!(
            list.starts_with("memory:///warehouse/t/ns/tbl/metadata/"),
            "{list}"
        );
        assert_eq!(
            mj["metadata-log"][0]["metadata-file"],
            wh.metadata_file_uri("t/ns/tbl/metadata", 0)
        );

        // Warehouse-root reader.
        let files = wh
            .read_iceberg_snapshot("t/ns/tbl/metadata", 1)
            .await
            .unwrap();
        assert_eq!(files, vec![Path::from(data_key)]);

        // Table-root reader (the TD-USUB-10 manifest-driven selection path). Both
        // bridges share the in-memory store only through the same process-global
        // `memory://` registry, so re-open through the warehouse store with a
        // table-root URL to model the second bridge.
        let table_bridge =
            IcebergObjectStoreBridge::new(ProximaObjectStore::new(wh.object_store().store()))
                .with_root_url("memory:///elsewhere");
        let rebased = table_bridge.resolve_recorded_path(
            &format!("{table_location}/data/part-0.parquet"),
            Some(&table_location),
            "",
        );
        assert_eq!(rebased.as_ref(), "data/part-0.parquet");
        let rebased_under_prefix = table_bridge.resolve_recorded_path(
            &format!("{table_location}/data/part-0.parquet"),
            Some(&table_location),
            "some/prefix",
        );
        assert_eq!(
            rebased_under_prefix.as_ref(),
            "some/prefix/data/part-0.parquet"
        );
        // Legacy relative paths pass through unchanged.
        assert_eq!(
            table_bridge
                .resolve_recorded_path("t/ns/tbl/data/x.parquet", None, "")
                .as_ref(),
            "t/ns/tbl/data/x.parquet"
        );
    }

    /// The v0.4.1 golden fixture (`testdata/v041`): the warehouse output of a
    /// REAL v0.4.1 server after `CREATE TABLE inv (id TEXT NOT NULL, qty INT,
    /// PRIMARY KEY (id))`, 20 INSERTs and `ALTER TABLE inv MATERIALIZE`. Bytes
    /// are verbatim except `location`, whose absolute warehouse root is the
    /// placeholder `file:///WAREHOUSE` (rewritten to a temp dir here).
    const V041_TABLE: &str = "data/proximadb/ns_584d9462-e9cf-4c0a-a353-1c5db38a7fe9/inv";

    fn v041_warehouse() -> (tempfile::TempDir, String) {
        fn copy(src: &std::path::Path, dst: &std::path::Path) {
            std::fs::create_dir_all(dst).unwrap();
            for entry in std::fs::read_dir(src).unwrap() {
                let path = entry.unwrap().path();
                let target = dst.join(path.file_name().unwrap());
                if path.is_dir() {
                    copy(&path, &target);
                } else {
                    std::fs::copy(&path, &target).unwrap();
                }
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/v041");
        copy(&fixture, dir.path());
        let root = format!("file://{}", dir.path().display());
        let metadata = dir
            .path()
            .join(V041_TABLE)
            .join("metadata/v00000000000000000000.metadata.json");
        let json = std::fs::read_to_string(&metadata).unwrap();
        assert_eq!(
            json.matches("file:///WAREHOUSE").count(),
            1,
            "one placeholder"
        );
        std::fs::write(&metadata, json.replace("file:///WAREHOUSE", &root)).unwrap();
        (dir, root)
    }

    /// v0.5.0 upgrade (both release blockers, on real v0.4.1 bytes): the
    /// table-base reader (OLAP reads) resolves v0.4.1's warehouse-relative
    /// paths instead of doubling them, the warehouse-root reader (MATERIALIZE)
    /// still resolves them, its lineage-less v3 metadata is adopted, and a
    /// successor commits onto it with lineage starting at 0.
    #[tokio::test]
    async fn a_real_v041_publication_reads_and_republishes() {
        let (_dir, root) = v041_warehouse();
        let table_meta = format!("{V041_TABLE}/metadata");

        // OLAP reader: a bridge at the table base, metadata under `metadata`.
        let table = IcebergObjectStoreBridge::from_url(&format!("{root}/{V041_TABLE}")).unwrap();
        let selected = table
            .read_iceberg_snapshot_with_deletes("metadata", 0)
            .await
            .expect("a v0.4.1 manifest decodes and resolves from the table base");
        assert_eq!(selected.data_files, vec![Path::from("data/part-0.parquet")]);
        assert!(selected.deletion_vectors.is_empty());
        table
            .object_store()
            .get(&selected.data_files[0])
            .await
            .expect("the resolved data file exists");

        // MATERIALIZE's bridge: the warehouse root.
        let warehouse = IcebergObjectStoreBridge::from_url(&root).unwrap();
        let files = warehouse
            .read_iceberg_snapshot_files(&table_meta, 0)
            .await
            .expect("resolves from the warehouse root");
        let part = Path::from(format!("{V041_TABLE}/data/part-0.parquet"));
        assert_eq!(files.data_files, vec![part.clone()]);

        // The real metadata is v3 without lineage, and is adopted at 0.
        let meta = warehouse.read_table_metadata(&table_meta, 0).await.unwrap();
        let parent: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parent["format-version"], 3);
        assert!(
            parent.get("next-row-id").is_none(),
            "v0.4.1 wrote no lineage"
        );
        let fields: Vec<iceberg::IcebergField> = parent["schemas"][0]["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| iceberg::IcebergField {
                id: f["id"].as_i64().unwrap() as i32,
                name: f["name"].as_str().unwrap().to_string(),
                type_name: f["type"].as_str().unwrap().to_string(),
                required: f["required"].as_bool().unwrap(),
            })
            .collect();

        // A successor commits onto the v0.4.1 parent (carrying its file).
        let v1 = warehouse
            .publish_iceberg_files(
                std::slice::from_ref(&part),
                Some(0),
                &table_meta,
                parent["table-uuid"].as_str().unwrap(),
                &format!("{root}/{V041_TABLE}"),
                &fields,
                parent["current-snapshot-id"].as_i64().unwrap() + 1,
                parent["last-updated-ms"].as_i64().unwrap() + 1,
                iceberg::FormatVersion::V3,
            )
            .await
            .expect("a successor commits onto a real v0.4.1 parent");
        assert_eq!(v1, 1);
        let successor: serde_json::Value =
            serde_json::from_slice(&warehouse.read_table_metadata(&table_meta, 1).await.unwrap())
                .unwrap();
        assert_eq!(successor["format-version"], 3);
        assert_eq!(
            successor["snapshots"][0]["first-row-id"], 0,
            "lineage starts at 0"
        );
        assert_eq!(successor["snapshots"][0]["added-rows"], 20);
        assert_eq!(successor["next-row-id"], 20);
        // And the table-base reader follows the successor.
        let next = table
            .read_iceberg_snapshot_with_deletes("metadata", 1)
            .await
            .unwrap();
        assert_eq!(next.data_files, vec![Path::from("data/part-0.parquet")]);
    }

    /// v0.5.0 upgrade: v0.4.1 recorded its manifest list, manifests and data
    /// files RELATIVE TO THE WAREHOUSE ROOT (`t/ns/tbl/metadata/…`) while its
    /// `location` is the absolute data directory (`<warehouse>/t/ns/tbl/data`).
    /// A reader opened at the table base must not append that relative path to
    /// its own root (doubling `t/ns/tbl`): the warehouse root is recovered from
    /// `location` minus `/<table-rel>/data`, and every bridge resolves the path.
    #[test]
    fn legacy_warehouse_relative_paths_resolve_from_any_bridge_root() {
        let location = "memory:///wh/t/ns/tbl/data";
        let at = |root: &str| {
            IcebergObjectStoreBridge::new(ProximaObjectStore::new(Arc::new(
                object_store::memory::InMemory::new(),
            )))
            .with_root_url(root)
        };
        let table_base = at("memory:///wh/t/ns/tbl");
        let warehouse = at("memory:///wh");
        for (recorded, from_base, from_warehouse) in [
            (
                "t/ns/tbl/metadata/snap-1-list.avro",
                "metadata/snap-1-list.avro",
                "t/ns/tbl/metadata/snap-1-list.avro",
            ),
            (
                "t/ns/tbl/metadata/1-m0.avro",
                "metadata/1-m0.avro",
                "t/ns/tbl/metadata/1-m0.avro",
            ),
            (
                "t/ns/tbl/data/part-0.parquet",
                "data/part-0.parquet",
                "t/ns/tbl/data/part-0.parquet",
            ),
        ] {
            assert_eq!(
                table_base
                    .resolve_recorded_path(recorded, Some(location), "")
                    .as_ref(),
                from_base,
                "{recorded} from the table base"
            );
            assert_eq!(
                warehouse
                    .resolve_recorded_path(recorded, Some(location), "")
                    .as_ref(),
                from_warehouse,
                "{recorded} from the warehouse root"
            );
        }
        // A relative path that does not match the location's table prefix is
        // not guessed at: it passes through as before.
        assert_eq!(
            table_base
                .resolve_recorded_path("other/x.parquet", Some(location), "")
                .as_ref(),
            "other/x.parquet"
        );
        // A bridge rooted ELSEWHERE (the warehouse moved mounts across the
        // upgrade) keeps the recorded warehouse-relative key: the rebuilt URI
        // must never go through the location re-rooting, which assumes 0.5.0's
        // table-base `location` and would drop `/data` from a v0.4.1 data file.
        let elsewhere = at("memory:///moved");
        for recorded in [
            "t/ns/tbl/metadata/snap-1-list.avro",
            "t/ns/tbl/metadata/1-m0.avro",
            "t/ns/tbl/data/part-0.parquet",
        ] {
            assert_eq!(
                elsewhere
                    .resolve_recorded_path(recorded, Some(location), "t/ns/tbl")
                    .as_ref(),
                recorded,
                "{recorded} from a bridge rooted elsewhere"
            );
        }
        // Names containing `data`: a table named `data` under a warehouse
        // whose root itself ends in `data` (one k matches).
        let data_location = "memory:///wh/data/t/ns/data/data";
        let data_base = at("memory:///wh/data/t/ns/data");
        assert_eq!(
            data_base
                .resolve_recorded_path("t/ns/data/data/part-0.parquet", Some(data_location), "")
                .as_ref(),
            "data/part-0.parquet"
        );
        assert_eq!(
            data_base
                .resolve_recorded_path("t/ns/data/metadata/m.avro", Some(data_location), "")
                .as_ref(),
            "metadata/m.avro"
        );
        // An ambiguous split: warehouse `…/w/data` + table-rel `data` gives the
        // location `…/w/data/data/data`, which `…/w` + table-rel `data/data`
        // also fits. v0.4.1's layout settles it: a recorded path continues
        // `<table-rel>/data/…` or `<table-rel>/metadata/…`, and only the
        // `w/data` split leaves `data/part-0.parquet` after the table-rel.
        let nested = at("memory:///w/data/data");
        assert_eq!(
            nested
                .resolve_recorded_path(
                    "data/data/part-0.parquet",
                    Some("memory:///w/data/data/data"),
                    ""
                )
                .as_ref(),
            "data/part-0.parquet",
            "the split whose remainder starts with data/ or metadata/ wins"
        );
    }

    fn age_field() -> Vec<iceberg::IcebergField> {
        vec![iceberg::IcebergField {
            id: 1,
            name: "age".into(),
            type_name: "long".into(),
            required: false,
        }]
    }

    async fn write_age_part(b: &IcebergObjectStoreBridge, key: &str, ages: &[i64]) -> Path {
        let records: Vec<ProximaRecord> = ages
            .iter()
            .map(|a| record(&format!("r{a}"), vec![("age", ProximaValue::Int64(*a))]))
            .collect();
        let path = Path::from(key);
        b.write_records_to_parquet(&path, &records, None)
            .await
            .unwrap();
        path
    }

    async fn publish_files(
        b: &IcebergObjectStoreBridge,
        files: &[Path],
        parent: Option<u64>,
        snapshot_id: i64,
    ) -> Result<u64, StorageError> {
        b.publish_iceberg_files(
            files,
            parent,
            "t/metadata",
            "uuid-files",
            "memory:///t",
            &age_field(),
            snapshot_id,
            1_700_000_000_000 + snapshot_id,
            iceberg::FormatVersion::V2,
        )
        .await
    }

    /// P2.4: an explicit file list publishes exactly that generation, whatever else
    /// sits in the data prefix — listing would union the other generation's parts.
    #[tokio::test]
    async fn publish_iceberg_files_publishes_only_the_named_generation() {
        let b = bridge();
        let g1 = write_age_part(&b, "t/data/g1-part-0.parquet", &[1, 2]).await;
        let v0 = publish_files(&b, std::slice::from_ref(&g1), None, 1)
            .await
            .unwrap();
        assert_eq!(v0, 0);

        let g2a = write_age_part(&b, "t/data/g2-part-0.parquet", &[3]).await;
        let g2b = write_age_part(&b, "t/data/g2-part-1.parquet", &[4, 5, 6]).await;
        let v1 = publish_files(&b, &[g2a.clone(), g2b.clone()], Some(v0), 2)
            .await
            .unwrap();
        assert_eq!(v1, 1);

        assert_eq!(
            b.read_iceberg_snapshot("t/metadata", 1).await.unwrap(),
            vec![g2a, g2b],
            "generation 2 only, though g1's part is still in t/data"
        );
        assert_eq!(
            b.read_iceberg_snapshot("t/metadata", 0).await.unwrap(),
            vec![g1],
            "the previous version's manifest is untouched"
        );
        let meta = b.read_table_metadata("t/metadata", 1).await.unwrap();
        let mj: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        let current = mj["snapshots"]
            .as_array()
            .unwrap()
            .iter()
            .find(|snap| snap["snapshot-id"] == mj["current-snapshot-id"])
            .unwrap();
        assert_eq!(current["summary"]["added-records"], "4");
    }

    /// P2.4: a publish whose parent is stale loses to the publisher that took the
    /// slot. It is an error (the old `publish_iceberg_table` caller treated the
    /// `Conflict` as success), and it must not overwrite the winner's Avro files.
    #[tokio::test]
    async fn publish_iceberg_files_on_a_stale_parent_is_an_error() {
        let b = bridge();
        let first = write_age_part(&b, "t/data/a.parquet", &[1]).await;
        publish_files(&b, std::slice::from_ref(&first), None, 10)
            .await
            .unwrap();
        let winner = write_age_part(&b, "t/data/b.parquet", &[2]).await;
        assert_eq!(
            publish_files(&b, std::slice::from_ref(&winner), Some(0), 11)
                .await
                .unwrap(),
            1
        );

        // Same parent, different snapshot: the metadata slot v1 is taken.
        let loser = write_age_part(&b, "t/data/c.parquet", &[3]).await;
        let err = publish_files(&b, std::slice::from_ref(&loser), Some(0), 12)
            .await
            .expect_err("a stale parent must not report success");
        assert!(matches!(err, StorageError::AlreadyExists(_)), "{err:?}");

        // Same parent AND same snapshot id: the generation's Avro names collide and
        // the create-only write refuses before anything of the winner is replaced.
        let err = publish_files(&b, std::slice::from_ref(&loser), Some(0), 11)
            .await
            .expect_err("colliding Avro names must not overwrite the winner");
        assert!(matches!(err, StorageError::AlreadyExists(_)), "{err:?}");
        assert_eq!(
            b.read_iceberg_snapshot("t/metadata", 1).await.unwrap(),
            vec![winner]
        );
    }

    /// ADR-097 D3 format policy: the requested version is a floor — a v2 request
    /// onto a v3 table commits v3, so a republication never downgrades an upgrade.
    #[tokio::test]
    async fn publish_iceberg_files_never_downgrades_a_v3_table() {
        let b = bridge();
        let first = write_age_part(&b, "t/data/a.parquet", &[1]).await;
        b.publish_iceberg_files(
            std::slice::from_ref(&first),
            None,
            "t/metadata",
            "uuid-fmt",
            "memory:///t",
            &age_field(),
            1,
            1,
            iceberg::FormatVersion::V3,
        )
        .await
        .unwrap();

        let second = write_age_part(&b, "t/data/b.parquet", &[2]).await;
        let v1 = b
            .publish_iceberg_files(
                std::slice::from_ref(&second),
                Some(0),
                "t/metadata",
                "uuid-fmt",
                "memory:///t",
                &age_field(),
                2,
                2,
                iceberg::FormatVersion::V2,
            )
            .await
            .unwrap();
        let meta = b.read_table_metadata("t/metadata", v1).await.unwrap();
        let mj: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(
            mj["format-version"], 3,
            "a v2 request onto a v3 table commits v3"
        );
        // The parent itself committed v3 (this is a floor, not a rewrite of v0),
        // and the upgrade direction stays caller-driven: a NEW table requested as
        // v2 starts v2, then adopts v3 when a successor requests it.
        let v0_meta = b.read_table_metadata("t/metadata", 0).await.unwrap();
        let v0_json: serde_json::Value = serde_json::from_slice(&v0_meta).unwrap();
        assert_eq!(v0_json["format-version"], 3);
        let c1 = write_age_part(&b, "t2/data/c1.parquet", &[3]).await;
        b.publish_iceberg_files(
            std::slice::from_ref(&c1),
            None,
            "t2/metadata",
            "uuid-fmt2",
            "memory:///t2",
            &age_field(),
            1,
            1,
            iceberg::FormatVersion::V2,
        )
        .await
        .unwrap();
        let fresh_meta = b.read_table_metadata("t2/metadata", 0).await.unwrap();
        let fresh: serde_json::Value = serde_json::from_slice(&fresh_meta).unwrap();
        assert_eq!(
            fresh["format-version"], 2,
            "a new v2-requested table starts v2"
        );
        let c2 = write_age_part(&b, "t2/data/c2.parquet", &[4]).await;
        b.publish_iceberg_files(
            std::slice::from_ref(&c2),
            Some(0),
            "t2/metadata",
            "uuid-fmt2",
            "memory:///t2",
            &age_field(),
            2,
            2,
            iceberg::FormatVersion::V3,
        )
        .await
        .unwrap();
        let upgraded_meta = b.read_table_metadata("t2/metadata", 1).await.unwrap();
        let upgraded: serde_json::Value = serde_json::from_slice(&upgraded_meta).unwrap();
        assert_eq!(
            upgraded["format-version"], 3,
            "a v3 request upgrades a v2 table"
        );
    }

    /// Snapshot ids must advance past the parent's current snapshot: an equal id
    /// is its own ancestor, a lower one rewinds history (same-millisecond
    /// publishers hit this; the collision surfaces instead of corrupting lineage).
    #[tokio::test]
    async fn publish_iceberg_files_rejects_a_non_advancing_snapshot_id() {
        let b = bridge();
        let first = write_age_part(&b, "t/data/a.parquet", &[1]).await;
        publish_files(&b, std::slice::from_ref(&first), None, 10)
            .await
            .unwrap();
        let second = write_age_part(&b, "t/data/b.parquet", &[2]).await;
        for bad_id in [10, 9] {
            let err = publish_files(&b, std::slice::from_ref(&second), Some(0), bad_id)
                .await
                .expect_err("a non-advancing snapshot id must be rejected");
            assert!(
                err.to_string().contains("must exceed the parent snapshot"),
                "{err:?}"
            );
        }
        // An id past the parent still commits.
        assert_eq!(
            publish_files(&b, std::slice::from_ref(&second), Some(0), 11)
                .await
                .unwrap(),
            1
        );
    }

    /// A file the publisher names must be a readable parquet: no silent count-0 entry.
    #[tokio::test]
    async fn publish_iceberg_files_requires_readable_footers() {
        let b = bridge();
        let bogus = Path::from("t/data/not-parquet.parquet");
        b.object_store()
            .put(&bogus, Bytes::from_static(b"not parquet"))
            .await
            .unwrap();
        assert!(
            publish_files(&b, &[bogus], None, 1).await.is_err(),
            "an unreadable footer fails the publish"
        );
        assert_eq!(b.latest_metadata_version("t/metadata").await.unwrap(), None);
        assert!(publish_files(&b, &[], None, 2).await.is_err(), "empty list");
    }

    /// End-to-end Iceberg publish: write a Parquet data file, publish a v3 snapshot
    /// (manifest + manifest list + TableMetadata), then resolve the snapshot back to its
    /// data files and read their rows. Proves ProximaDB emits a readable Iceberg table.
    #[tokio::test]
    async fn iceberg_snapshot_publishes_and_reads_back_v3() {
        let b = bridge();
        let data_prefix = Path::from("wh/t/ns/tbl/data");
        let metadata_prefix = "wh/t/ns/tbl/metadata";

        let r0 = record(
            "r0",
            vec![
                ("name", ProximaValue::String("alice".into())),
                ("age", ProximaValue::Int64(30)),
            ],
        );
        let r1 = record(
            "r1",
            vec![
                ("name", ProximaValue::String("bob".into())),
                ("age", ProximaValue::Int64(41)),
            ],
        );
        b.write_records_to_parquet(
            &Path::from("wh/t/ns/tbl/data/part-0.parquet"),
            &[r0, r1],
            None,
        )
        .await
        .unwrap();

        let fields = vec![
            iceberg::IcebergField {
                id: 1,
                name: "name".into(),
                type_name: "string".into(),
                required: false,
            },
            iceberg::IcebergField {
                id: 2,
                name: "age".into(),
                type_name: "long".into(),
                required: false,
            },
        ];
        let out = b
            .publish_iceberg_snapshot(
                &data_prefix,
                metadata_prefix,
                "uuid-1",
                "wh/t/ns/tbl",
                &fields,
                1001,
                1_700_000_000_000,
                iceberg::FormatVersion::V3,
            )
            .await
            .unwrap();
        assert_eq!(out, CommitOutcome::Committed(0));

        // The committed metadata is Iceberg format-version 3.
        let meta = b.read_table_metadata(metadata_prefix, 0).await.unwrap();
        let mj: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(mj["format-version"], 3);

        // Resolve the snapshot back to its data file(s) and read the rows.
        let files = b.read_iceberg_snapshot(metadata_prefix, 0).await.unwrap();
        assert_eq!(files.len(), 1, "one data file in the snapshot");
        let mut stream = b
            .read_parquet_batches(&files[0], Arc::new(ArrowSchema::empty()), 1024, None)
            .await
            .unwrap();
        let mut total = 0;
        while let Some(batch) = stream.next().await {
            total += batch.unwrap().num_rows();
        }
        assert_eq!(total, 2, "snapshot data files carry all rows");
    }

    /// A non-empty `schema` arg projects the read to only the named columns (by name),
    /// dropping the rest — and an absent requested name is silently skipped, not an error.
    #[tokio::test]
    async fn read_projects_to_requested_columns_by_name() {
        let b = bridge();
        let path = Path::from("warehouse/t/proj.parquet");

        let r0 = record(
            "r0",
            vec![
                ("name", ProximaValue::String("alice".into())),
                ("age", ProximaValue::Int64(30)),
                ("city", ProximaValue::String("paris".into())),
            ],
        );
        let r1 = record(
            "r1",
            vec![
                ("name", ProximaValue::String("bob".into())),
                ("age", ProximaValue::Int64(41)),
                ("city", ProximaValue::String("rome".into())),
            ],
        );
        b.write_records_to_parquet(&path, &[r0, r1], None)
            .await
            .unwrap();

        // Project to {name, age} plus a name that does not exist in the file.
        let projection = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
            Field::new("does_not_exist", DataType::Utf8, true),
        ]));

        let mut stream = b
            .read_parquet_batches(&path, projection, 1024, None)
            .await
            .unwrap();

        let mut batches = Vec::new();
        while let Some(batch) = stream.next().await {
            batches.push(batch.unwrap());
        }
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);

        let batch = &batches[0];
        // Only the two existing projected columns are present — `city` is pruned and the
        // bogus requested name does not materialize a column.
        assert_eq!(
            batch.num_columns(),
            2,
            "projection must drop unrequested columns"
        );
        assert!(batch.column_by_name("name").is_some());
        assert!(batch.column_by_name("age").is_some());
        assert!(
            batch.column_by_name("city").is_none(),
            "city was not requested"
        );
        assert!(batch.column_by_name("does_not_exist").is_none());

        let names = batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "alice");
        assert_eq!(names.value(1), "bob");
    }

    /// End-to-end warehouse base-tier publish: write data files, atomically publish a
    /// snapshot listing them, read the snapshot back, and re-read the data through it.
    #[tokio::test]
    async fn publish_and_read_snapshot_round_trips_data_files() {
        let b = bridge();
        // Two data files under t/data; the manifest log lives OUTSIDE that prefix.
        let p0 = Path::from("t/data/part-0.parquet");
        let p1 = Path::from("t/data/part-1.parquet");
        b.write_records_to_parquet(
            &p0,
            &[record(
                "r0",
                vec![("name", ProximaValue::String("alice".into()))],
            )],
            None,
        )
        .await
        .unwrap();
        b.write_records_to_parquet(
            &p1,
            &[record(
                "r1",
                vec![("name", ProximaValue::String("bob".into()))],
            )],
            None,
        )
        .await
        .unwrap();

        let out = b
            .publish_snapshot(&Path::from("t/data"), "t/_manifests", None)
            .await
            .unwrap();
        assert_eq!(out, manifest::CommitOutcome::Committed(0));

        let files = b.read_snapshot("t/_manifests", 0).await.unwrap();
        assert_eq!(files.len(), 2, "snapshot records both data files");
        assert!(files.contains(&p0) && files.contains(&p1));

        // The recorded keys re-read as data through the bridge — end to end.
        let mut total = 0usize;
        for f in &files {
            let mut s = b
                .read_parquet_batches(f, Arc::new(ArrowSchema::empty()), 1024, None)
                .await
                .unwrap();
            while let Some(batch) = s.next().await {
                total += batch.unwrap().num_rows();
            }
        }
        assert_eq!(total, 2, "both rows readable via the published snapshot");
    }

    /// Two publishes from the same parent conflict (atomic CAS) — the base tier cannot
    /// double-commit a version; the loser is told the latest to rebase onto.
    #[tokio::test]
    async fn publish_snapshot_is_atomic_cas() {
        let b = bridge();
        b.write_records_to_parquet(
            &Path::from("t/data/part-0.parquet"),
            &[record("r0", vec![("k", ProximaValue::Int64(1))])],
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            b.publish_snapshot(&Path::from("t/data"), "t/_manifests", None)
                .await
                .unwrap(),
            manifest::CommitOutcome::Committed(0)
        );
        // A second publish from the same parent (None ⇒ target v0) loses the race.
        assert_eq!(
            b.publish_snapshot(&Path::from("t/data"), "t/_manifests", None)
                .await
                .unwrap(),
            manifest::CommitOutcome::Conflict { latest: Some(0) }
        );
    }

    /// Reading a path that was never written must surface a NotFound, not panic.
    #[tokio::test]
    async fn read_missing_object_errors() {
        let b = bridge();
        let res = b
            .read_parquet_batches(
                &Path::from("nope.parquet"),
                Arc::new(ArrowSchema::empty()),
                64,
                None,
            )
            .await;
        assert!(res.is_err());
    }

    /// Raw vector-segment bytes round-trip through persist/fetch.
    #[tokio::test]
    async fn vector_segment_round_trip() {
        let b = bridge();
        let path = Path::from("warehouse/vec/seg_000.pax");
        let payload: Vec<u8> = (0u8..200).collect();

        b.persist_vector_segment(&path, &payload, None)
            .await
            .unwrap();
        let got = b.fetch_vector_segment(&path, None).await.unwrap();
        assert_eq!(got, payload);
    }

    /// With a NON-EMPTY store base, `list_objects` must return base-relative keys
    /// that round-trip through the read methods. (A raw `inner_store().list` would
    /// return base-prefixed keys that then double-prepend the base on read.)
    #[tokio::test]
    async fn list_objects_is_base_relative_and_round_trips_with_non_empty_base() {
        let b = IcebergObjectStoreBridge::from_url("memory:///warehouse").unwrap();
        assert!(
            !b.object_store().base().as_ref().is_empty(),
            "base must be non-empty"
        );

        let path = Path::from("t/data/part-0.parquet");
        let r0 = record("r0", vec![("name", ProximaValue::String("alice".into()))]);
        b.write_records_to_parquet(&path, &[r0], None)
            .await
            .unwrap();

        let listed = b.list_objects(&Path::from("t/data")).await.unwrap();
        assert_eq!(listed.len(), 1, "the written object must be discoverable");
        assert_eq!(
            listed[0].as_ref(),
            "t/data/part-0.parquet",
            "list_objects must strip the store base so the key round-trips through read"
        );

        // The listed key must be directly readable (the bridge re-applies the base).
        let mut stream = b
            .read_parquet_batches(&listed[0], Arc::new(ArrowSchema::empty()), 1024, None)
            .await
            .unwrap();
        let mut rows = 0usize;
        while let Some(batch) = stream.next().await {
            rows += batch.unwrap().num_rows();
        }
        assert_eq!(rows, 1, "the listed key must read back the written record");
    }

    async fn read_first_batch(b: &IcebergObjectStoreBridge, path: &Path) -> RecordBatch {
        let mut stream = b
            .read_parquet_batches(path, Arc::new(ArrowSchema::empty()), 1024, None)
            .await
            .unwrap();
        stream.next().await.expect("at least one batch").unwrap()
    }

    /// A schema-explicit write honors the CALLER's (catalog-authoritative)
    /// schema, not inference: a column declared in the schema but absent from
    /// every record is written as an all-null column, whereas the schema-less
    /// (inferred) write omits it entirely. This is the F3 contract — the
    /// on-disk file shape follows the catalog, not whichever optional fields a
    /// given batch happens to populate.
    #[tokio::test]
    async fn write_with_explicit_schema_null_fills_columns_absent_from_records() {
        use arrow_array::Float64Array;
        use proximadb_data_model::ProximaType;
        use proximadb_storage_common::proxima_schema::ProximaColumn;

        fn col(id: i32, name: &str, dt: ProximaType, nullable: bool) -> ProximaColumn {
            ProximaColumn {
                id,
                object_id: None,
                name: name.to_string(),
                data_type: dt,
                nullable,
                default_value: None,
                comment: None,
                properties: std::collections::HashMap::new(),
                is_deleted: false,
                original_id: None,
            }
        }

        let b = bridge();
        // Records carry only `name`; NONE carry `score`.
        let r0 = record("r0", vec![("name", ProximaValue::String("alice".into()))]);
        let r1 = record("r1", vec![("name", ProximaValue::String("bob".into()))]);

        // Catalog-authoritative schema declares `score` anyway.
        let schema = ProximaSchema::from_columns(
            "wh".to_string(),
            vec![
                col(1, "name", ProximaType::String, true),
                col(2, "score", ProximaType::Float64, true),
            ],
            vec![1],
        );

        // Schema-explicit write: `score` must survive as an all-null column.
        let explicit = Path::from("wh/explicit.parquet");
        b.write_records_to_parquet_with_schema(&explicit, &[r0.clone(), r1.clone()], &schema, None)
            .await
            .unwrap();
        let batch = read_first_batch(&b, &explicit).await;
        let score = batch
            .column_by_name("score")
            .expect("explicit schema must keep the declared `score` column")
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(score.len(), 2);
        assert!(
            score.is_null(0) && score.is_null(1),
            "a column absent from every record must be an all-null column"
        );

        // Schema-less (inferred) write: `score` is omitted (no record carries it).
        let inferred = Path::from("wh/inferred.parquet");
        b.write_records_to_parquet(&inferred, &[r0, r1], None)
            .await
            .unwrap();
        let batch = read_first_batch(&b, &inferred).await;
        assert!(
            batch.column_by_name("score").is_none(),
            "inference must omit a column that no record carries"
        );
    }
}
