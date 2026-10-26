use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::ObjectStore;
use object_store::path::Path;
use proximadb_kernel::error::StorageError;
use proximadb_records::ProximaRecord;
use std::sync::Arc;

use crate::CatalogTableSchema as ProximaSchema;

pub use object_store::ObjectStore as BridgeObjectStore;
pub use object_store::memory::InMemory as BridgeInMemoryObjectStore;
pub use object_store::path::Path as BridgeObjectPath;

/// Outcome of an attempted manifest commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitOutcome {
    /// The manifest was atomically published as this new version.
    Committed(u64),
    /// Another committer already claimed the target slot. `latest` is the highest
    /// version currently present — the snapshot to rebase onto and retry from.
    Conflict { latest: Option<u64> },
}

/// The canonical bridge interface between ProximaDB's compute/storage engines
/// and decoupled object storage (S3, GCS, Azure, Local).
///
/// This serves two primary architectural paths:
/// 1. Data Warehouse/Relational Workloads: Writing and reading Parquet files managed by Iceberg.
/// 2. Vector/ANN Workloads: Persisting and fetching specialized PAX blocks for high-performance retrieval.
#[async_trait]
pub trait ObjectStoreBridge: Send + Sync {
    /// Returns a reference to the underlying `ObjectStore` implementation.
    fn inner_store(&self) -> Arc<dyn ObjectStore>;

    /// The key in [`inner_store`](Self::inner_store) of a bridge-relative `key`
    /// (a bridge rooted under a base prefix prepends it). For writers that go to
    /// the store directly, e.g. multipart uploads. Default: the key itself.
    fn object_store_path(&self, key: &Path) -> Path {
        key.clone()
    }

    /// Reads a Parquet file from object storage and yields a stream of Arrow `RecordBatch`es.
    /// This is the primary ingestion path for DataFusion logical plans.
    async fn read_parquet_batches(
        &self,
        path: &Path,
        schema: Arc<ArrowSchema>,
        batch_size: usize,
        tenant_id: Option<&str>,
    ) -> Result<BoxStream<'static, Result<RecordBatch, StorageError>>, StorageError>;

    /// Writes a stream of `ProximaRecord`s into a Parquet file on object storage.
    /// This is the primary output path for DML/Analytics operations.
    /// Implementers MUST emit `proximadb_object_store_ops_total` and `proximadb_storage_bytes_seconds` for billing.
    async fn write_records_to_parquet(
        &self,
        path: &Path,
        records: &[ProximaRecord],
        tenant_id: Option<&str>,
    ) -> Result<(), StorageError>;

    /// Schema-explicit Parquet write: writes `records` using the CALLER-PROVIDED
    /// (catalog-authoritative) [`ProximaSchema`] instead of inferring one from the
    /// records, so the file's column set, order, types, and nullability match the
    /// catalog exactly — including columns declared in the schema but absent from
    /// every record (written as all-null), which inference would silently drop.
    /// This is the correct write path for warehouse materialization, where the
    /// published Parquet must round-trip through `SELECT *` against the catalog shape.
    ///
    /// Default: falls back to the schema-less (inferred) [`write_records_to_parquet`]
    /// for bridges without a schema-aware path.
    ///
    /// [`write_records_to_parquet`]: ObjectStoreBridge::write_records_to_parquet
    async fn write_records_to_parquet_with_schema(
        &self,
        path: &Path,
        records: &[ProximaRecord],
        schema: &ProximaSchema,
        tenant_id: Option<&str>,
    ) -> Result<(), StorageError> {
        let _ = schema;
        self.write_records_to_parquet(path, records, tenant_id)
            .await
    }

    /// Schema-explicit Parquet write that also declares the physical layout the
    /// caller produced: `sort_columns` names the columns `records` are already
    /// sorted by (ascending, nulls last). Warehouse bridges record it as Parquet
    /// `sorting_columns` and use it to shape page index / Bloom placement.
    ///
    /// Default: ignores the hint and falls back to
    /// [`write_records_to_parquet_with_schema`].
    ///
    /// [`write_records_to_parquet_with_schema`]: ObjectStoreBridge::write_records_to_parquet_with_schema
    async fn write_records_to_parquet_with_layout(
        &self,
        path: &Path,
        records: &[ProximaRecord],
        schema: &ProximaSchema,
        sort_columns: &[String],
        tenant_id: Option<&str>,
    ) -> Result<(), StorageError> {
        let _ = sort_columns;
        self.write_records_to_parquet_with_schema(path, records, schema, tenant_id)
            .await
    }

    /// Fetches a specialized PAX block or Segment (SST, HELIX, etc.) from object storage
    /// into memory for high-performance Vector SIMD scans.
    async fn fetch_vector_segment(
        &self,
        path: &Path,
        tenant_id: Option<&str>,
    ) -> Result<Vec<u8>, StorageError>;

    /// Byte length of a PAX segment object without reading its body — the
    /// prerequisite for footer-first ranged reads. The default falls back to a
    /// whole-object fetch (no I/O saving); object-store-backed implementations
    /// MUST override with a metadata-only `object_size`/HEAD.
    async fn vector_segment_size(
        &self,
        path: &Path,
        tenant_id: Option<&str>,
    ) -> Result<u64, StorageError> {
        Ok(self.fetch_vector_segment(path, tenant_id).await?.len() as u64)
    }

    /// Read a byte range `[offset, offset+length)` of a PAX segment — a true
    /// ranged GET on object storage. The default falls back to a whole-object
    /// fetch + slice (no I/O saving); object-store-backed implementations MUST
    /// override with `get_range` so only the requested bytes leave the wire.
    async fn fetch_vector_segment_range(
        &self,
        path: &Path,
        offset: u64,
        length: u64,
        tenant_id: Option<&str>,
    ) -> Result<Vec<u8>, StorageError> {
        let whole = self.fetch_vector_segment(path, tenant_id).await?;
        let start = (offset as usize).min(whole.len());
        let end = ((offset + length) as usize).min(whole.len());
        Ok(whole[start..end].to_vec())
    }

    /// Read **multiple** byte ranges of a PAX segment in one batched operation —
    /// the depth-collapsing primitive for object storage (TD-167 / ADR-034 P1). A
    /// pruned read fetches all surviving block bodies through this call instead of
    /// issuing one *dependent* ranged GET per block, turning `O(K)` serial
    /// round-trips into a single batched op (latency = depth × RTT, so batching K
    /// independent reads is one RTT, not K). Returns one buffer per input range, in
    /// order. The default loops [`fetch_vector_segment_range`] (correct, no I/O
    /// saving); object-store-backed implementations SHOULD override with
    /// `FileSystem::read_ranges` so the ranges coalesce/parallelize on the wire.
    async fn fetch_vector_segment_ranges(
        &self,
        path: &Path,
        ranges: &[std::ops::Range<u64>],
        tenant_id: Option<&str>,
    ) -> Result<Vec<Vec<u8>>, StorageError> {
        let mut out = Vec::with_capacity(ranges.len());
        for r in ranges {
            out.push(
                self.fetch_vector_segment_range(path, r.start, r.end - r.start, tenant_id)
                    .await?,
            );
        }
        Ok(out)
    }

    /// Persists a specialized PAX block or Segment to decoupled object storage.
    /// Implementers MUST emit `proximadb_object_store_ops_total` and `proximadb_storage_bytes_seconds` for billing.
    async fn persist_vector_segment(
        &self,
        path: &Path,
        data: &[u8],
        tenant_id: Option<&str>,
    ) -> Result<(), StorageError>;

    /// Enumerate the object keys under `prefix`, returned as paths that are
    /// directly consumable by this bridge's read methods (`read_parquet_batches`
    /// / `fetch_vector_segment`).
    ///
    /// This is the read-side discovery seam: a record store does not track which
    /// files the write path produced, so it lists them to read them back.
    ///
    /// The default implementation enumerates the raw [`inner_store`], returning
    /// each object's full key. This is correct for stores whose base prefix is
    /// empty (`memory://`, bucket-root). Implementations that wrap the store with
    /// a non-empty base prefix (e.g. `IcebergObjectStoreBridge` over
    /// `ProximaObjectStore`) MUST override this to return base-relative keys, so
    /// the returned paths round-trip through the bridge's read methods (which
    /// re-apply the base).
    ///
    /// [`inner_store`]: ObjectStoreBridge::inner_store
    async fn list_objects(&self, prefix: &Path) -> Result<Vec<Path>, StorageError> {
        let store = self.inner_store();
        let mut listing = store.list(Some(prefix));
        let mut paths = Vec::new();
        while let Some(meta) = listing.next().await {
            let meta = meta.map_err(|err| {
                StorageError::DiskIO(std::io::Error::other(format!(
                    "object-store list under `{prefix}` failed: {err}"
                )))
            })?;
            paths.push(meta.location);
        }
        Ok(paths)
    }

    /// Retrieve the latest committed manifest version for a given table prefix.
    async fn latest_manifest_version(
        &self,
        manifest_prefix: &str,
    ) -> Result<Option<u64>, StorageError> {
        let _ = manifest_prefix;
        Err(StorageError::Serialization(
            "latest_manifest_version not supported".into(),
        ))
    }

    /// Atomically publish the data objects currently under `data_prefix` as the next
    /// snapshot in the manifest log at `manifest_prefix`.
    async fn publish_snapshot(
        &self,
        data_prefix: &Path,
        manifest_prefix: &str,
        parent: Option<u64>,
    ) -> Result<CommitOutcome, StorageError> {
        let _ = (data_prefix, manifest_prefix, parent);
        Err(StorageError::Serialization(
            "publish_snapshot not supported".into(),
        ))
    }

    /// Latest committed Iceberg `metadata.json` version in the table metadata log at
    /// `metadata_prefix` (companion to [`latest_manifest_version`]; TD-119).
    ///
    /// [`latest_manifest_version`]: ObjectStoreBridge::latest_manifest_version
    async fn latest_metadata_version(
        &self,
        metadata_prefix: &str,
    ) -> Result<Option<u64>, StorageError> {
        let _ = metadata_prefix;
        Err(StorageError::Serialization(
            "latest_metadata_version not supported".into(),
        ))
    }

    /// Read the raw bytes of the Iceberg `metadata.json` at `version` under `metadata_prefix`.
    async fn read_table_metadata(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<Vec<u8>, StorageError> {
        let _ = (metadata_prefix, version);
        Err(StorageError::Serialization(
            "read_table_metadata not supported".into(),
        ))
    }

    /// Atomically publish `metadata` as the successor of `parent` in the table metadata log
    /// at `metadata_prefix` (optimistic-concurrency CAS, like [`publish_snapshot`]).
    ///
    /// [`publish_snapshot`]: ObjectStoreBridge::publish_snapshot
    async fn commit_table_metadata(
        &self,
        metadata_prefix: &str,
        parent: Option<u64>,
        metadata: Vec<u8>,
    ) -> Result<CommitOutcome, StorageError> {
        let _ = (metadata_prefix, parent, metadata);
        Err(StorageError::Serialization(
            "commit_table_metadata not supported".into(),
        ))
    }

    /// Publish the `*.parquet` data files under `data_prefix` as a **spec-shaped Iceberg
    /// snapshot** (Avro manifest + manifest list + `TableMetadata`) committed atomically
    /// through the metadata log at `metadata_prefix`, so external Iceberg engines can read
    /// the table. `v3` selects Iceberg format-version 3 (ProximaDB's default) vs v2. The
    /// field descriptors are storage-common-local (this trait can't depend on the Iceberg
    /// engine crate); the Iceberg bridge maps them to its own types. Default: unsupported.
    #[allow(clippy::too_many_arguments)]
    async fn publish_iceberg_table(
        &self,
        data_prefix: &Path,
        metadata_prefix: &str,
        table_uuid: &str,
        table_location: &str,
        fields: &[IcebergSnapshotField],
        snapshot_id: i64,
        timestamp_ms: i64,
        v3: bool,
    ) -> Result<CommitOutcome, StorageError> {
        let _ = (
            data_prefix,
            metadata_prefix,
            table_uuid,
            table_location,
            fields,
            snapshot_id,
            timestamp_ms,
            v3,
        );
        Err(StorageError::Serialization(
            "publish_iceberg_table not supported".into(),
        ))
    }

    /// Publish exactly `files` (keys of parquet objects the caller wrote) as a spec-shaped
    /// Iceberg snapshot committed as the successor of metadata version `parent` (`None` ⇒
    /// the first). Returns the committed version. Nothing is listed, so a data prefix
    /// holding several generations publishes only this one. Losing the race for the
    /// successor slot is an `Err(StorageError::AlreadyExists)`, never success.
    /// Default: unsupported.
    #[allow(clippy::too_many_arguments)]
    async fn publish_iceberg_files(
        &self,
        files: &[Path],
        parent: Option<u64>,
        metadata_prefix: &str,
        table_uuid: &str,
        table_location: &str,
        fields: &[IcebergSnapshotField],
        snapshot_id: i64,
        timestamp_ms: i64,
        v3: bool,
    ) -> Result<u64, StorageError> {
        let _ = (
            files,
            parent,
            metadata_prefix,
            table_uuid,
            table_location,
            fields,
            snapshot_id,
            timestamp_ms,
            v3,
        );
        Err(StorageError::Serialization(
            "publish_iceberg_files not supported".into(),
        ))
    }

    /// Resolve the current snapshot's data-file paths from the Iceberg metadata log at
    /// `metadata_prefix` (decodes `TableMetadata` → manifest list → manifests). Default: empty.
    async fn read_iceberg_table(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<Vec<Path>, StorageError> {
        let _ = (metadata_prefix, version);
        Ok(Vec::new())
    }

    /// The current snapshot's data files **and the deletion vectors that apply to
    /// them**, as one view (the snapshot's rows are its data files minus each
    /// file's deleted positions).
    ///
    /// The default keeps the pre-delete behavior of bridges that never publish
    /// vectors: the data files of [`read_iceberg_table`](Self::read_iceberg_table)
    /// and no deletions. A bridge that CAN publish vectors must override this and
    /// refuse a snapshot carrying them if it cannot represent them — never return
    /// the deleted rows as live.
    async fn read_iceberg_snapshot_files(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<IcebergSnapshotFiles, StorageError> {
        Ok(IcebergSnapshotFiles {
            data_files: self.read_iceberg_table(metadata_prefix, version).await?,
            deletion_vectors: Vec::new(),
        })
    }

    /// [`publish_iceberg_files`](Self::publish_iceberg_files) carrying `deletes`:
    /// the committed snapshot is `files` with each vector's positions deleted.
    /// Format v3 only; the engine validates that every vector references one of
    /// `files` and no file has two. Default: unsupported.
    #[allow(clippy::too_many_arguments)]
    async fn publish_iceberg_files_deleting(
        &self,
        files: &[Path],
        deletes: &[IcebergSnapshotDeletion],
        parent: Option<u64>,
        metadata_prefix: &str,
        table_uuid: &str,
        table_location: &str,
        fields: &[IcebergSnapshotField],
        snapshot_id: i64,
        timestamp_ms: i64,
    ) -> Result<u64, StorageError> {
        let _ = (
            files,
            deletes,
            parent,
            metadata_prefix,
            table_uuid,
            table_location,
            fields,
            snapshot_id,
            timestamp_ms,
        );
        Err(StorageError::Serialization(
            "publish_iceberg_files_deleting not supported".into(),
        ))
    }
}

/// Storage-common-local descriptor of one Iceberg schema field (the trait can't reference
/// the Iceberg engine's own type). The Iceberg bridge maps `type_name` to an Iceberg
/// primitive type ("long", "int", "string", "double", "boolean", "date", "timestamp", …).
#[derive(Debug, Clone)]
pub struct IcebergSnapshotField {
    pub id: i32,
    pub name: String,
    pub type_name: String,
    pub required: bool,
}

/// One deletion vector of a published snapshot, resolved to bridge-relative keys
/// (foundation-local mirror of the engine's descriptor — same pattern as
/// [`IcebergSnapshotField`]: the trait cannot name the engine's types). A reader
/// applies it by hiding `data_file`'s positions named by the vector's blob in
/// `puffin_file`; a re-publisher passes it back to
/// [`ObjectStoreBridge::publish_iceberg_files_deleting`] to carry the deletes forward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergSnapshotDeletion {
    /// The data file whose rows it deletes (one of the snapshot's data files).
    pub data_file: Path,
    /// The Puffin file holding the vector, and its blob's byte range + size there.
    pub puffin_file: Path,
    pub content_offset: u64,
    pub content_size_in_bytes: u64,
    /// The Puffin file's full size (the delete-manifest entry's `file_size_in_bytes`).
    pub puffin_file_size_in_bytes: u64,
    /// The number of deleted positions.
    pub cardinality: u64,
}

/// A published snapshot's data files and the deletion vectors that apply to them
/// (its rows are the data files minus each vector's positions).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IcebergSnapshotFiles {
    pub data_files: Vec<Path>,
    pub deletion_vectors: Vec<IcebergSnapshotDeletion>,
}

impl IcebergSnapshotFiles {
    /// True when the snapshot carries deletion vectors — a reader that cannot
    /// apply them must refuse rather than serve the deleted rows.
    pub fn has_deletions(&self) -> bool {
        !self.deletion_vectors.is_empty()
    }
}

/// Map an Arrow schema to the Iceberg snapshot fields, ids 1..n in arrow order
/// (the single mapping every publisher uses — MATERIALIZE and the DataFusion
/// bulk-append executor must not drift apart on field ids or type names).
pub fn iceberg_snapshot_fields(schema: &arrow_schema::Schema) -> Vec<IcebergSnapshotField> {
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| IcebergSnapshotField {
            id: (i + 1) as i32,
            name: f.name().clone(),
            type_name: iceberg_type_for(f.data_type()),
            required: !f.is_nullable(),
        })
        .collect()
}

/// The Arrow → Iceberg type-name mapping, chosen to keep external Iceberg readers'
/// schema aligned with the Parquet physical types. Timestamps map to the v3
/// nanosecond types (ProximaDB is nanosecond-native).
pub fn iceberg_type_for(arrow: &arrow_schema::DataType) -> String {
    use arrow_schema::{DataType as D, TimeUnit};
    match arrow {
        D::Boolean => "boolean".into(),
        D::Int8 | D::Int16 | D::Int32 => "int".into(),
        D::Int64 | D::UInt8 | D::UInt16 | D::UInt32 | D::UInt64 => "long".into(),
        D::Float16 | D::Float32 => "float".into(),
        D::Float64 => "double".into(),
        D::Date32 | D::Date64 => "date".into(),
        D::Decimal128(p, s) | D::Decimal256(p, s) => format!("decimal({p},{s})"),
        // Iceberg's v2 `timestamp` is microseconds; nanoseconds need v3's `_ns` types
        // (see `iceberg_needs_v3`).
        D::Timestamp(TimeUnit::Nanosecond, Some(_)) => "timestamptz_ns".into(),
        D::Timestamp(TimeUnit::Nanosecond, None) => "timestamp_ns".into(),
        D::Timestamp(_, Some(_)) => "timestamptz".into(),
        D::Timestamp(_, None) => "timestamp".into(),
        D::Utf8 | D::LargeUtf8 => "string".into(),
        D::FixedSizeBinary(n) => format!("fixed[{n}]"),
        D::Binary | D::LargeBinary => "binary".into(),
        // JSON columns materialize as Utf8 today; v3 `variant` is a follow-up.
        _ => "string".into(),
    }
}

/// ADR-097 D3: publish Iceberg **v2** — the version external readers (PyIceberg,
/// DuckDB, Trino) verifiably read — unless a column needs a v3-only type
/// (nanosecond timestamps). v3 deletion vectors / row lineage / `variant` follow
/// once an external v3 reader is in the read-back gate. This is the *requested*
/// version; the engine additionally refuses to downgrade an already-v3 table.
pub fn iceberg_needs_v3(fields: &[IcebergSnapshotField]) -> bool {
    fields.iter().any(|f| f.type_name.ends_with("_ns"))
}

/// A stable, well-formed UUID string derived deterministically from the table's object
/// prefix, so an Iceberg table keeps the same `table-uuid` across re-materializations
/// (no `uuid` v5 dependency; two seeded hashes fill the 16 bytes).
pub fn deterministic_table_uuid(prefix: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h1 = std::collections::hash_map::DefaultHasher::new();
    prefix.hash(&mut h1);
    let a = h1.finish();
    let mut h2 = std::collections::hash_map::DefaultHasher::new();
    (prefix, 0x50524f58_4944425fu64).hash(&mut h2); // "PROX_IDB_" salt
    let b = h2.finish();
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        (a >> 32) as u32,
        (a >> 16) as u16,
        a as u16,
        (b >> 48) as u16,
        b & 0x0000_ffff_ffff_ffff
    )
}
