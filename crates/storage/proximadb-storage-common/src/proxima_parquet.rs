//! # proxima_parquet — in-memory `RecordBatch` ↔ Parquet-bytes codec (F1.6)
//!
//! The missing serialization layer between Arrow and the warehouse base tier. Every existing
//! Parquet helper in the tree (`columnar_io::write_parquet`, `file_export::read_parquet_file`,
//! the open-format `iceberg.rs` reader) is **FileSystem-path-based** — it writes to / reads
//! from a `&str` path. The decoupled-storage warehouse base tier does not have a path: the
//! [`ObjectStoreBridge`](crate::object_store_bridge) and the `proximadb-object-store`
//! `ProximaObjectStore` speak **`Bytes`** (`put`/`get`/`get_range`). This module supplies the
//! pure in-memory `RecordBatch` ⇄ Parquet-`bytes` conversion those need, with no FileSystem
//! and no `object_store` dependency (it operates on `Vec<u8>`/`Bytes`, so it is independent of
//! the `object_store` crate version).
//!
//! Composed with the canonical [`proxima_records_to_record_batch`] (F1.5), this is exactly the
//! body of `ObjectStoreBridge::write_records_to_parquet` minus the final `put`:
//!
//! ```text
//! write_records_to_parquet(path, records) =
//!     proxima_records_to_parquet_bytes(records, schema)   // this module + F1.5
//!     → ProximaObjectStore::put(path, bytes)              // F1
//!
//! read_parquet_batches(path) =
//!     ProximaObjectStore::get(path) → bytes               // F1
//!     → parquet_bytes_to_record_batches(bytes)            // this module
//! ```
//!
//! Canonical home: `proximadb-storage-common`, beside the bridge trait, `ProximaSchema`, and
//! `proxima_arrow`. (`proximadb-codec` is vector/quantization columnar encoding only; its
//! "wraps the parquet crate" framing is aspirational, so Parquet (de)serialization does NOT
//! belong there.)

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use proximadb_kernel::error::StorageError;
use proximadb_records::ProximaRecord;

use crate::proxima_arrow::proxima_records_to_record_batch;
use crate::proxima_schema::ProximaSchema;

fn ser_err(context: &str, e: impl std::fmt::Display) -> StorageError {
    StorageError::Serialization(format!("proxima_parquet: {context}: {e}"))
}

/// Serialize Arrow [`RecordBatch`]es into the bytes of a single in-memory Parquet file.
///
/// `schema` is the file schema (every batch must match it); passing zero batches writes a
/// valid Parquet file with the schema and no rows. `props` is the Parquet writer
/// configuration — pass `None` for the Arrow default (uncompressed). Callers wanting a
/// warehouse-grade default can pass e.g. `WriterProperties::builder().set_compression(
/// Compression::SNAPPY).build()`.
pub fn record_batches_to_parquet_bytes(
    batches: &[RecordBatch],
    schema: SchemaRef,
    props: Option<WriterProperties>,
) -> Result<Vec<u8>, StorageError> {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buf, schema, props)
            .map_err(|e| ser_err("ArrowWriter::try_new", e))?;
        for batch in batches {
            writer.write(batch).map_err(|e| ser_err("write", e))?;
        }
        writer.close().map_err(|e| ser_err("close", e))?;
    }
    Ok(buf)
}

/// Read every [`RecordBatch`] from the bytes of an in-memory Parquet file.
///
/// `Bytes` is cheap to clone and implements the parquet `ChunkReader` directly, so no temp
/// file or extra copy is needed. The batch boundaries are chosen by the reader (typically one
/// per row group); callers that need a single batch should concatenate.
pub fn parquet_bytes_to_record_batches(bytes: Bytes) -> Result<Vec<RecordBatch>, StorageError> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .map_err(|e| ser_err("ParquetRecordBatchReaderBuilder::try_new", e))?;
    let reader = builder.build().map_err(|e| ser_err("reader build", e))?;
    let mut out = Vec::new();
    for batch in reader {
        out.push(batch.map_err(|e| ser_err("read batch", e))?);
    }
    Ok(out)
}

/// Like [`parquet_bytes_to_record_batches`] but caps each yielded batch at `batch_size` rows.
///
/// This is the read path `ObjectStoreBridge::read_parquet_batches` uses, where the caller
/// chooses the batch size that flows into the downstream (DataFusion) operator pipeline.
pub fn parquet_bytes_to_record_batches_with_batch_size(
    bytes: Bytes,
    batch_size: usize,
) -> Result<Vec<RecordBatch>, StorageError> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .map_err(|e| ser_err("ParquetRecordBatchReaderBuilder::try_new", e))?
        .with_batch_size(batch_size)
        .build()
        .map_err(|e| ser_err("reader build", e))?;
    let mut out = Vec::new();
    for batch in reader {
        out.push(batch.map_err(|e| ser_err("read batch", e))?);
    }
    Ok(out)
}

/// Encode canonical [`ProximaRecord`]s straight to Parquet bytes via the canonical
/// `ProximaSchema`-driven Arrow mapping (F1.5) + a Parquet writer. This is the function the
/// F2 `ObjectStoreBridge::write_records_to_parquet` impl calls before the object-store `put`.
pub fn proxima_records_to_parquet_bytes(
    records: &[ProximaRecord],
    schema: &ProximaSchema,
    props: Option<WriterProperties>,
) -> Result<Vec<u8>, StorageError> {
    let batch = proxima_records_to_record_batch(records, schema)?;
    let arrow_schema = batch.schema();
    record_batches_to_parquet_bytes(&[batch], arrow_schema, props)
}

/// Like [`proxima_records_to_parquet_bytes`], but the writer properties are derived
/// from `policy` against the encoded batch itself (Bloom NDV sizing needs the data),
/// so the file carries the warehouse layout: row groups, page index, declared sort
/// order, and spec SBBF Bloom filters.
pub fn proxima_records_to_parquet_bytes_with_policy(
    records: &[ProximaRecord],
    schema: &ProximaSchema,
    policy: &crate::parquet_write_policy::ParquetWritePolicy,
    hints: &crate::parquet_write_policy::ParquetLayoutHints,
) -> Result<Vec<u8>, StorageError> {
    let batch = proxima_records_to_record_batch(records, schema)?;
    let props = policy.writer_properties(&batch, hints);
    let arrow_schema = batch.schema();
    record_batches_to_parquet_bytes(&[batch], arrow_schema, Some(props))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Array, FixedSizeBinaryArray, Float64Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaTreeNode, ProximaValue};

    use proximadb_data_model::{ProximaType, VectorElement as DmVectorElement};

    use crate::proxima_schema::ProximaColumn;

    fn col(id: i32, name: &str, dt: ProximaType, nullable: bool) -> ProximaColumn {
        ProximaColumn {
            id,
            object_id: None,
            name: name.to_string(),
            data_type: dt,
            nullable,
            default_value: None,
            comment: None,
            properties: HashMap::new(),
            is_deleted: false,
            original_id: None,
        }
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

    /// Records → Parquet bytes → RecordBatches must preserve every typed column, the null
    /// mask, and the fp32 vector column (the warehouse write→read round-trip F2 relies on).
    #[test]
    fn records_round_trip_through_parquet_bytes() {
        let dim = 3u32;
        let schema = ProximaSchema::from_columns(
            "wh".to_string(),
            vec![
                col(1, "name", ProximaType::String, true),
                col(2, "age", ProximaType::Int64, true),
                col(3, "score", ProximaType::Float64, true),
                col(
                    4,
                    "embedding",
                    ProximaType::DenseVector {
                        element: DmVectorElement::Float32,
                        dim: dim as usize,
                    },
                    true,
                ),
            ],
            vec![1],
        );

        let mut r0 = record(
            "r0",
            vec![
                ("name", ProximaValue::String("alice".into())),
                ("age", ProximaValue::Int64(30)),
                ("score", ProximaValue::Float64(9.5)),
            ],
        );
        r0.embeddings.push(EmbeddingCell {
            values: EmbeddingValues::Fp32(vec![1.0, 2.0, 3.0]),
            ..Default::default()
        });
        // r1: `age` absent (→ null), no embedding (→ null vector).
        let r1 = record(
            "r1",
            vec![
                ("name", ProximaValue::String("bob".into())),
                ("score", ProximaValue::Float64(1.25)),
            ],
        );

        let bytes = proxima_records_to_parquet_bytes(&[r0, r1], &schema, None).unwrap();
        assert!(!bytes.is_empty());

        let batches = parquet_bytes_to_record_batches(Bytes::from(bytes)).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
        let batch = &batches[0];

        // Field names + types survive (compare structurally, ignoring custom field metadata).
        let want = schema.to_arrow_schema();
        assert_eq!(batch.num_columns(), want.fields().len());
        for (got, exp) in batch.schema().fields().iter().zip(want.fields().iter()) {
            assert_eq!(got.name(), exp.name());
            assert_eq!(got.data_type(), exp.data_type());
        }

        let names = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "alice");
        assert_eq!(names.value(1), "bob");

        let ages = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ages.value(0), 30);
        assert!(ages.is_null(1), "absent prop must survive as a null");

        let scores = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(scores.value(0), 9.5);
        assert_eq!(scores.value(1), 1.25);

        let vecs = batch
            .column(3)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        assert_eq!(vecs.value_length(), (dim * 4) as i32);
        let decoded: Vec<f32> = vecs
            .value(0)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(decoded, vec![1.0, 2.0, 3.0]);
        assert!(vecs.is_null(1), "record without embedding stays null");
    }

    /// Zero records still produces a readable Parquet file carrying the schema and no rows.
    #[test]
    fn empty_records_write_schema_only() {
        let schema = ProximaSchema::from_columns(
            "empty".to_string(),
            vec![
                col(1, "id", ProximaType::String, false),
                col(2, "n", ProximaType::Int64, true),
            ],
            vec![1],
        );
        let bytes = proxima_records_to_parquet_bytes(&[], &schema, None).unwrap();
        let batches = parquet_bytes_to_record_batches(Bytes::from(bytes)).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }

    /// The lower-level batch codec round-trips an arbitrary RecordBatch.
    #[test]
    fn record_batch_bytes_round_trip() {
        let schema: SchemaRef = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            ],
        )
        .unwrap();

        let bytes = record_batches_to_parquet_bytes(&[batch], schema, None).unwrap();
        let back = parquet_bytes_to_record_batches(Bytes::from(bytes)).unwrap();
        let total: usize = back.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3);
        let v = back[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(v.value(0), "a");
        assert!(v.is_null(1));
        assert_eq!(v.value(2), "c");
    }
}
