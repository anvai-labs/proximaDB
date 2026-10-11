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
use parquet::file::properties::WriterProperties;
use proximadb_kernel::error::StorageError;
use proximadb_object_store::ProximaObjectStore;
use proximadb_records::ProximaRecord;
use proximadb_storage_common::object_store_bridge::{CommitOutcome, ObjectStoreBridge};
use proximadb_storage_common::proxima_arrow::infer_proxima_schema_strict;
use proximadb_storage_common::proxima_parquet::proxima_records_to_parquet_bytes;
use proximadb_storage_common::proxima_schema::ProximaSchema;

/// Parquet writer configuration for the warehouse publication tier
/// (MATERIALIZE snapshots + Iceberg interop files).
///
/// The load-bearing setting is the row-group cap: the Arrow default is
/// 1,048,576 rows, which packs every table below ~1M rows into a SINGLE row
/// group — one split, whole-domain min/max, and zone-map/runtime-filter
/// pruning provably cannot skip anything (measured 0% in
/// `docs/_internal/status/TPC_PERF_GATE_EVIDENCE_2026_07_04.adoc`). The
/// default of 65,536 rows approximates the co-design blueprint's 8–16 MiB
/// scan/aggregate geometry for typical fact rows; override per deployment
/// with `PROXIMADB_MATERIALIZE_ROW_GROUP_ROWS`. Row-group count is reader
/// geometry, not a format change — mixed-read-safe by construction.
fn warehouse_writer_properties() -> WriterProperties {
    let rows = std::env::var("PROXIMADB_MATERIALIZE_ROW_GROUP_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(65_536);
    WriterProperties::builder()
        .set_max_row_group_row_count(Some(rows))
        .build()
}

/// Spec-shaped Iceberg v2 manifest / manifest-list (Avro) + TableMetadata (JSON) content.
pub mod iceberg;
/// Iceberg-style atomic manifest commits (optimistic concurrency via `put_if_absent`).
pub mod manifest;
pub mod metadata;

/// Map a Parquet read/decode failure into a [`StorageError`] with operation context.
fn read_err(context: &str, e: impl std::fmt::Display) -> StorageError {
    StorageError::Serialization(format!("iceberg-engine: {context}: {e}"))
}

/// An [`ObjectStoreBridge`] backed by a [`ProximaObjectStore`] (any `object_store` backend:
/// local `file://`, `memory://`, or cloud `s3://`/`gs://`/`az://` when the F1 crate's cloud
/// features are enabled). One bridge serves an entire store/bucket; paths select the object.
#[derive(Clone)]
pub struct IcebergObjectStoreBridge {
    store: ProximaObjectStore,
}

impl IcebergObjectStoreBridge {
    /// Wrap an existing object-store handle.
    pub fn new(store: ProximaObjectStore) -> Self {
        Self { store }
    }

    /// Open the bridge directly from a URL (see [`ProximaObjectStore::from_url`]).
    pub fn from_url(url: &str) -> Result<Self, StorageError> {
        Ok(Self {
            store: ProximaObjectStore::from_url(url)?,
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

    /// Row count of a Parquet object, read from the footer metadata only (no row data).
    async fn parquet_num_rows(&self, path: &Path) -> Result<i64, StorageError> {
        // arrow-rs 59.2 deprecated ParquetObjectReader (wants direct AsyncFileReader impls,
        // apache/arrow-rs#10308). Contained here; the migration is a tracked follow-up.
        #[allow(deprecated)]
        let reader = ParquetObjectReader::new(self.store.store(), self.full_object_path(path));
        let builder = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| read_err("open parquet footer", e))?;
        Ok(builder.metadata().file_metadata().num_rows())
    }

    /// Publish a snapshot as **spec-shaped Iceberg** (v3 by default): write an Avro manifest
    /// for the `*.parquet` data files under `data_prefix`, an Avro manifest list, and a
    /// `TableMetadata` JSON referencing it, committed atomically through the metadata log at
    /// `metadata_prefix`. External Iceberg engines can then read the table. `snapshot_id` and
    /// `timestamp_ms` are caller-supplied (no wall-clock here, for determinism). Returns the
    /// committed metadata version (or a `Conflict` to rebase onto).
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
        // 1. Enumerate data files with size (HEAD) + record count (footer).
        let mut keys: Vec<Path> = self.list_objects(data_prefix).await?;
        keys.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
        let mut data_files = Vec::new();
        for key in &keys {
            if !key.as_ref().ends_with(".parquet") {
                continue;
            }
            let size = self.store.object_size(key).await?;
            let record_count = self.parquet_num_rows(key).await.unwrap_or(0);
            data_files.push(iceberg::DataFileMeta {
                file_path: key.as_ref().to_string(),
                record_count,
                file_size_in_bytes: size as i64,
            });
        }
        let added_records: i64 = data_files.iter().map(|f| f.record_count).sum();
        let added_files = data_files.len() as i64;

        // 2. Write the Avro manifest + manifest list.
        let manifest_bytes = iceberg::write_manifest(&data_files, snapshot_id)?;
        let manifest_path = Path::from(format!("{metadata_prefix}/{snapshot_id}-m0.avro"));
        self.store
            .put(&manifest_path, Bytes::from(manifest_bytes.clone()))
            .await?;
        let manifest_list = vec![iceberg::ManifestFileMeta {
            manifest_path: manifest_path.as_ref().to_string(),
            manifest_length: manifest_bytes.len() as i64,
            added_files_count: added_files as i32,
            added_rows_count: added_records,
            added_snapshot_id: snapshot_id,
        }];
        let list_bytes = iceberg::write_manifest_list(&manifest_list)?;
        let list_path = Path::from(format!("{metadata_prefix}/snap-{snapshot_id}-list.avro"));
        self.store.put(&list_path, Bytes::from(list_bytes)).await?;

        // 3. Build + atomically commit the TableMetadata.
        let committer = metadata::MetadataCommitter::new(self.store.clone(), metadata_prefix);
        let parent = committer.latest_version().await?;
        let sequence_number = parent.map(|p| p as i64 + 1).unwrap_or(1);
        let json = iceberg::build_table_metadata(
            format_version,
            table_uuid,
            table_location,
            fields,
            snapshot_id,
            sequence_number,
            list_path.as_ref(),
            timestamp_ms,
            added_files,
            added_records,
        )?;
        committer.commit(parent, Bytes::from(json)).await
    }

    /// Resolve the data-file keys of the current snapshot in the Iceberg metadata log at
    /// `metadata_prefix` (decodes `TableMetadata` → manifest list → manifests). Each returned
    /// path is consumable by [`read_parquet_batches`](ObjectStoreBridge::read_parquet_batches).
    pub async fn read_iceberg_snapshot(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<Vec<Path>, StorageError> {
        let committer = metadata::MetadataCommitter::new(self.store.clone(), metadata_prefix);
        let meta_bytes = committer.read_metadata(version).await?;
        let meta_json =
            std::str::from_utf8(&meta_bytes).map_err(|e| read_err("metadata utf8", e))?;
        let Some(list_loc) = iceberg::current_manifest_list(meta_json)? else {
            return Ok(Vec::new());
        };
        let list_bytes = self.store.get(&Path::from(list_loc)).await?;
        let manifests = iceberg::read_manifest_list(&list_bytes)?;
        let mut out = Vec::new();
        for m in manifests {
            let mbytes = self.store.get(&Path::from(m.manifest_path)).await?;
            for df in iceberg::read_manifest(&mbytes)? {
                out.push(Path::from(df.file_path));
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl ObjectStoreBridge for IcebergObjectStoreBridge {
    fn inner_store(&self) -> Arc<dyn ObjectStore> {
        self.store.store()
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
        _tenant_id: Option<&str>,
    ) -> Result<(), StorageError> {
        let bytes =
            proxima_records_to_parquet_bytes(records, schema, Some(warehouse_writer_properties()))?;
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
        let mapped: Vec<iceberg::IcebergField> = fields
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

    async fn read_iceberg_table(
        &self,
        metadata_prefix: &str,
        version: u64,
    ) -> Result<Vec<Path>, StorageError> {
        self.read_iceberg_snapshot(metadata_prefix, version).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field};
    use object_store::memory::InMemory;
    use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaTreeNode, ProximaValue};

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
        r0.embeddings.push(EmbeddingCell {
            values: EmbeddingValues::Fp32(vec![1.0, 2.0, 3.0]),
            ..Default::default()
        });
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
