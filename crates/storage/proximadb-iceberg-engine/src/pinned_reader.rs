//! Field-ID projection and position deletes for producer-pinned Parquet snapshots.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use arrow_array::{BooleanArray, RecordBatch, RecordBatchOptions, new_null_array};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt, stream};
use object_store::path::Path;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::{PARQUET_FIELD_ID_META_KEY, ProjectionMask};
use proximadb_kernel::error::StorageError;
use serde::Deserialize;
use serde_json::Value;

use crate::{IcebergObjectStoreBridge, PinnedIcebergSnapshot, SnapshotDeletionVector, read_err};

type BatchStream = BoxStream<'static, Result<RecordBatch, StorageError>>;

/// A scan of one producer-pinned snapshot. The schema comes from Iceberg
/// metadata, with the current names and logical types. Data files are opened
/// lazily, projected by their physical Parquet field IDs, and filtered by the
/// snapshot's deletion vectors before emitting Arrow batches.
pub struct PinnedIcebergReader {
    bridge: IcebergObjectStoreBridge,
    snapshot: PinnedIcebergSnapshot,
    schema: SchemaRef,
    fields: Vec<ProjectedField>,
    deletes: HashMap<Path, SnapshotDeletionVector>,
}

#[derive(Clone, Copy)]
struct ProjectedField {
    id: i32,
    can_fill_null: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct IcebergColumn {
    id: i32,
    name: String,
    required: bool,
    #[serde(rename = "type")]
    type_name: String,
    initial_default: Option<Value>,
}

#[derive(Deserialize)]
struct PartitionSpec {
    fields: Vec<PartitionField>,
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct PartitionField {
    source_id: i32,
    transform: String,
}

impl PinnedIcebergReader {
    pub(crate) fn try_new(
        bridge: IcebergObjectStoreBridge,
        snapshot: PinnedIcebergSnapshot,
        schema: &Value,
        partition_specs: Option<&[Value]>,
    ) -> Result<Self, StorageError> {
        if schema["type"].as_str() != Some("struct") {
            return Err(read_err("pinned Iceberg schema", "expected a struct"));
        }
        let columns: Vec<IcebergColumn> = Vec::deserialize(&schema["fields"])
            .map_err(|e| read_err("pinned Iceberg flat columns", e))?;
        let mut identity_sources = HashSet::new();
        if let Some(specs) = partition_specs {
            for spec in specs {
                let spec = PartitionSpec::deserialize(spec)
                    .map_err(|e| read_err("pinned Iceberg partition spec", e))?;
                identity_sources.extend(
                    spec.fields
                        .into_iter()
                        .filter(|field| field.transform == "identity")
                        .map(|field| field.source_id),
                );
            }
        }
        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        let mut fields = Vec::with_capacity(columns.len());
        let mut arrow_fields = Vec::with_capacity(columns.len());
        for column in columns {
            if column.id <= 0 || !ids.insert(column.id) || !names.insert(column.name.clone()) {
                return Err(read_err(
                    "pinned Iceberg schema",
                    format!(
                        "invalid or duplicate column {} with field ID {}",
                        column.name, column.id,
                    ),
                ));
            }
            fields.push(ProjectedField {
                id: column.id,
                // Identity-partition migrations may omit actual values from the file.
                can_fill_null: !column.required
                    && column.initial_default.is_none()
                    && partition_specs.is_some_and(|specs| !specs.is_empty())
                    && !identity_sources.contains(&column.id),
            });
            arrow_fields.push(
                Field::new(column.name, ice_type(&column.type_name)?, !column.required)
                    .with_metadata(HashMap::from([(
                        PARQUET_FIELD_ID_META_KEY.to_string(),
                        column.id.to_string(),
                    )])),
            );
        }
        let deletes = snapshot
            .plan
            .files
            .deletion_vectors
            .iter()
            .map(|vector| (vector.data_file.clone(), vector.clone()))
            .collect();
        Ok(Self {
            bridge,
            snapshot,
            schema: Arc::new(Schema::new(arrow_fields)),
            fields,
            deletes,
        })
    }

    pub fn snapshot(&self) -> &PinnedIcebergSnapshot {
        &self.snapshot
    }

    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// Stream projected columns in requested order, with zero columns allowed
    /// for consumers that only need row counts. Missing optional fields without
    /// defaults or identity partition values are filled with nulls.
    pub fn scan(
        &self,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> Result<BatchStream, StorageError> {
        self.scan_files(
            0..self.snapshot.plan.files.data_files.len(),
            projection,
            batch_size,
        )
    }

    /// Scan a range of file indices in the pinned manifest plan.
    pub fn scan_files(
        &self,
        range: Range<usize>,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> Result<BatchStream, StorageError> {
        if batch_size == 0 {
            return Err(read_err(
                "pinned Iceberg scan",
                "batch size must be positive",
            ));
        }
        let (schema, fields) = if let Some(indices) = projection {
            let schema = self
                .schema
                .project(indices)
                .map_err(|e| read_err("pinned Iceberg projection", e))?;
            let fields = indices.iter().map(|&index| self.fields[index]).collect();
            (Arc::new(schema), fields)
        } else {
            (self.schema(), self.fields.clone())
        };
        let fields = Arc::new(fields);
        let files: Vec<_> = self
            .snapshot
            .plan
            .files
            .data_files
            .get(range)
            .ok_or_else(|| read_err("pinned Iceberg scan", "file range outside the pinned plan"))?
            .iter()
            .map(|path| (path.clone(), self.deletes.get(path).cloned()))
            .collect();

        let bridge = self.bridge.clone();
        Ok(stream::iter(files)
            .then(move |(path, vector)| {
                read_file(
                    bridge.clone(),
                    path,
                    vector,
                    schema.clone(),
                    fields.clone(),
                    batch_size,
                )
            })
            .try_flatten()
            .boxed())
    }
}

async fn read_file(
    bridge: IcebergObjectStoreBridge,
    path: Path,
    vector: Option<SnapshotDeletionVector>,
    schema: SchemaRef,
    fields: Arc<Vec<ProjectedField>>,
    batch_size: usize,
) -> Result<BatchStream, StorageError> {
    #[allow(deprecated)]
    let reader = parquet::arrow::async_reader::ParquetObjectReader::new(
        bridge.object_store().store(),
        bridge.full_object_path(&path),
    );
    let options = ArrowReaderOptions::new().with_skip_arrow_metadata(true);
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(reader, options)
        .await
        .map_err(|e| read_err("open pinned Iceberg Parquet footer", e))?;
    let mut ids = HashMap::new();
    for (index, root) in builder
        .parquet_schema()
        .root_schema()
        .get_fields()
        .iter()
        .enumerate()
    {
        let info = root.get_basic_info();
        if !info.has_id() {
            return Err(read_err(
                "pinned Iceberg projection",
                format!("{path}: Parquet column {} has no field ID", info.name(),),
            ));
        }
        if ids.insert(info.id(), index).is_some() {
            return Err(read_err(
                "pinned Iceberg projection",
                format!("{path}: duplicate Parquet field ID {}", info.id(),),
            ));
        }
    }

    let file_columns: Vec<_> = fields
        .iter()
        .zip(schema.fields())
        .map(|(field, output)| match ids.get(&field.id).copied() {
            Some(index) => {
                let actual = builder.schema().field(index).data_type();
                if !can_read_as(actual, output.data_type()) {
                    return Err(read_err(
                        "pinned Iceberg projection",
                        format!(
                            "{path}: field ID {} has type {actual}, expected {}",
                            field.id,
                            output.data_type(),
                        ),
                    ));
                }
                Ok(Some(index))
            }
            None if field.can_fill_null => Ok(None),
            None => Err(read_err(
                "pinned Iceberg projection",
                format!("{path}: missing field ID {} requires a value", field.id,),
            )),
        })
        .collect::<Result<_, _>>()?;
    let mut file_order: Vec<_> = file_columns.iter().flatten().copied().collect();
    file_order.sort_unstable();
    file_order.dedup();
    // Parquet returns selected roots in physical order. Translate the original
    // absolute root index into its index in the projected batch.
    let batch_columns: Vec<_> = file_columns
        .into_iter()
        .map(|root| root.map(|index| file_order.partition_point(|&selected| selected < index)))
        .collect();
    let deletes = if let Some(vector) = vector {
        let rows = u64::try_from(builder.metadata().file_metadata().num_rows())
            .map_err(|e| read_err("pinned Iceberg row count", e))?;
        let deletes = bridge.read_deletion_vector(&vector).await?;
        if deletes.count_range(0, rows) != deletes.cardinality() {
            return Err(read_err(
                "pinned Iceberg deletion vector",
                format!("{path}: delete position outside {rows} physical rows",),
            ));
        }
        Some(deletes)
    } else {
        None
    };
    let mask = ProjectionMask::roots(builder.parquet_schema(), file_order);
    let batches = builder
        .with_projection(mask)
        .with_batch_size(batch_size)
        .build()
        .map_err(|e| read_err("build pinned Iceberg Parquet stream", e))?;
    let mut next_position = 0u64;
    Ok(batches
        .map(move |batch| {
            let batch = batch.map_err(|e| read_err("read pinned Iceberg Parquet batch", e))?;
            let mut rows = batch.num_rows();
            let keep = if let Some(deletes) = &deletes {
                let start = next_position;
                next_position = start.checked_add(rows as u64).ok_or_else(|| {
                    read_err("pinned Iceberg deletion vector", "row position overflow")
                })?;
                if deletes.count_range(start, next_position) > 0 {
                    let keep: BooleanArray = (start..next_position)
                        .map(|row| !deletes.contains(row))
                        .collect();
                    rows = keep.true_count();
                    Some(keep)
                } else {
                    None
                }
            } else {
                None
            };
            let columns = batch_columns
                .iter()
                .zip(schema.fields())
                .map(|(index, field)| {
                    let Some(index) = index else {
                        return Ok(new_null_array(field.data_type(), rows));
                    };
                    let column = batch.column(*index);
                    let column = if let Some(keep) = &keep {
                        arrow_select::filter::filter(column.as_ref(), keep)
                            .map_err(|e| read_err("apply pinned Iceberg deletion vector", e))?
                    } else {
                        column.clone()
                    };
                    if column.data_type() == field.data_type() {
                        Ok(column)
                    } else {
                        arrow_cast::cast(column.as_ref(), field.data_type())
                            .map_err(|e| read_err("promote pinned Iceberg field", e))
                    }
                })
                .collect::<Result<_, _>>()?;
            let options = RecordBatchOptions::new().with_row_count(Some(rows));
            RecordBatch::try_new_with_options(schema.clone(), columns, &options)
                .map_err(|e| read_err("assemble pinned Iceberg batch", e))
        })
        .boxed())
}

fn ice_type(name: &str) -> Result<DataType, StorageError> {
    use DataType::*;
    let unsupported = || read_err("pinned Iceberg type", format!("unsupported type {name}"));
    Ok(match name {
        "boolean" => Boolean,
        "int" => Int32,
        "long" => Int64,
        "float" => Float32,
        "double" => Float64,
        "string" => Utf8,
        "binary" => Binary,
        "date" => Date32,
        "time" => Time64(TimeUnit::Microsecond),
        "timestamp" => Timestamp(TimeUnit::Microsecond, None),
        "timestamptz" => Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        "timestamp_ns" => Timestamp(TimeUnit::Nanosecond, None),
        "timestamptz_ns" => Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
        "uuid" => FixedSizeBinary(16),
        _ => {
            if let Some(length) = name
                .strip_prefix("fixed[")
                .and_then(|v| v.strip_suffix(']'))
            {
                let length = length
                    .parse::<i32>()
                    .ok()
                    .filter(|&v| v > 0)
                    .ok_or_else(unsupported)?;
                FixedSizeBinary(length)
            } else if let Some((precision, scale)) = name
                .strip_prefix("decimal(")
                .and_then(|v| v.strip_suffix(')'))
                .and_then(|v| v.split_once(','))
            {
                let precision = precision
                    .trim()
                    .parse::<u8>()
                    .ok()
                    .filter(|v| (1..=38).contains(v))
                    .ok_or_else(unsupported)?;
                let scale = scale
                    .trim()
                    .parse::<i8>()
                    .ok()
                    .filter(|&v| v >= 0 && v as u8 <= precision)
                    .ok_or_else(unsupported)?;
                Decimal128(precision, scale)
            } else {
                return Err(unsupported());
            }
        }
    })
}

fn can_read_as(actual: &DataType, desired: &DataType) -> bool {
    use DataType::*;
    actual == desired
        || match (actual, desired) {
            (Int32, Int64)
            | (Float32, Float64)
            | (Utf8View | LargeUtf8, Utf8)
            | (BinaryView | LargeBinary, Binary) => true,
            (Decimal128(source, scale), Decimal128(target, target_scale)) => {
                source <= target && scale == target_scale
            }
            (Timestamp(source, Some(tz)), Timestamp(target, Some(target_tz))) => {
                source == target && tz.as_ref() == "+00:00" && target_tz.as_ref() == "UTC"
            }
            _ => false,
        }
}
