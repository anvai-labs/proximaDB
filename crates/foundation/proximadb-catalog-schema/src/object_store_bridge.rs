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
