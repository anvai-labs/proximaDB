//! Verified local NDJSON ranges decoded into typed Arrow batches.

use std::ops::Range;
use std::sync::Arc;

use anyhow::{Context, ensure};
use arrow::json::ReaderBuilder;
use arrow::json::reader::{ArrayDecoder, DecoderContext, DecoderFactory, Tape, TapeElement};
use arrow_array::builder::Decimal128Builder;
use arrow_array::{ArrayRef, types::Decimal128Type};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema, SchemaRef};
use futures::{StreamExt, TryStreamExt, stream};
use object_store::{ObjectStoreExt, path::Path};
use proximadb_catalog::CatalogColumn;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::confined_file_store::ConfinedFileStore;
use crate::query::execution::pinned_source::{PinnedBatchSource, PinnedBatchStream};
use crate::services::ddl::pinned_ndjson::PinnedNdjsonLocation;

const MAX_RANGE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    size: u64,
    ranges: Vec<ManifestRange>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestRange {
    end: u64,
    sha256: String,
}

#[derive(Clone)]
struct VerifiedRange {
    bytes: Range<u64>,
    digest: [u8; 32],
}

pub(crate) struct PinnedNdjsonReader {
    store: Arc<ConfinedFileStore>,
    file: Path,
    schema: SchemaRef,
    ranges: Vec<VerifiedRange>,
}

impl PinnedNdjsonReader {
    pub(crate) async fn open(
        location: PinnedNdjsonLocation,
        columns: &[CatalogColumn],
    ) -> anyhow::Result<Self> {
        let schema = ndjson_schema(columns)?;
        let store = Arc::new(ConfinedFileStore::open(location.root, Path::ROOT).await?);
        let file = location.file;
        let manifest_path = Path::parse(format!("{file}.manifest.json"))?;
        let manifest_size = store.head(&manifest_path).await?.size;
        ensure!(
            (1..=MAX_MANIFEST_BYTES).contains(&manifest_size),
            "external NDJSON manifest exceeds its size limit or is empty"
        );
        let bytes = store.get_range(&manifest_path, 0..manifest_size).await?;
        ensure!(
            Sha256::digest(&bytes).as_slice() == location.manifest_digest,
            "external NDJSON manifest SHA-256 mismatch"
        );
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        ensure!(
            manifest.version == 1,
            "unsupported external NDJSON manifest version"
        );
        let mut start = 0;
        let mut ranges = Vec::with_capacity(manifest.ranges.len());
        for range in manifest.ranges {
            ensure!(
                range.end > start && range.end - start <= MAX_RANGE_BYTES,
                "external NDJSON range is empty, unordered, or exceeds 8 MiB"
            );
            let mut digest = [0; 32];
            hex::decode_to_slice(range.sha256, &mut digest)
                .context("invalid NDJSON range SHA-256")?;
            ranges.push(VerifiedRange {
                bytes: start..range.end,
                digest,
            });
            start = range.end;
        }
        ensure!(
            start == manifest.size,
            "external NDJSON ranges do not cover the pinned file size"
        );
        ensure!(
            store.head(&file).await?.size == manifest.size,
            "external NDJSON file size differs from the pinned manifest"
        );
        Ok(Self {
            store,
            file,
            schema,
            ranges,
        })
    }
}

pub(crate) fn ndjson_schema(columns: &[CatalogColumn]) -> anyhow::Result<SchemaRef> {
    let fields = columns
        .iter()
        .map(|column| {
            let ty = crate::datafusion::schema_inference::proxima_to_arrow_type(&column.data_type);
            ensure!(
                matches!(
                    ty,
                    DataType::Boolean
                        | DataType::Int32
                        | DataType::Int64
                        | DataType::Float32
                        | DataType::Float64
                        | DataType::Date32
                        | DataType::Decimal128(_, _)
                ) || (ty == DataType::Utf8
                    && matches!(column.data_type, proximadb_data_model::ProximaType::String)),
                "unsupported external NDJSON type for column {}: {:?}",
                column.name,
                column.data_type
            );
            Ok(Field::new(&column.name, ty, column.nullable))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

impl PinnedBatchSource for PinnedNdjsonReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn split_count(&self) -> usize {
        self.ranges.len()
    }

    fn scan_splits(
        &self,
        splits: Range<usize>,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> anyhow::Result<PinnedBatchStream> {
        let ranges = self
            .ranges
            .get(splits)
            .context("NDJSON split outside pinned inventory")?
            .to_vec();
        let store = self.store.clone();
        let file = self.file.clone();
        let schema = self.schema();
        let projection = projection.map(<[usize]>::to_vec);
        Ok(stream::iter(ranges)
            .then(move |range| {
                let store = store.clone();
                let file = file.clone();
                let schema = schema.clone();
                let projection = projection.clone();
                async move {
                    let bytes = store.get_range(&file, range.bytes.clone()).await?;
                    ensure!(
                        bytes.len() as u64 == range.bytes.end - range.bytes.start
                            && Sha256::digest(&bytes).as_slice() == range.digest,
                        "external NDJSON range SHA-256 mismatch for {file} at byte {}",
                        range.bytes.start
                    );
                    ensure!(
                        bytes.ends_with(b"\n"),
                        "external NDJSON range must end at a newline"
                    );
                    let reader = ReaderBuilder::new(schema)
                        .with_batch_size(batch_size)
                        .with_strict_mode(true)
                        .with_decoder_factory(Arc::new(ExactDecimalFactory))
                        .build(std::io::Cursor::new(bytes))?;
                    Ok(stream::iter(reader.map(move |batch| {
                        let batch = batch?;
                        Ok(match &projection {
                            Some(indices) => batch.project(indices)?,
                            None => batch,
                        })
                    })))
                }
            })
            .try_flatten()
            .boxed())
    }
}

#[derive(Debug)]
struct ExactDecimalFactory;

impl DecoderFactory for ExactDecimalFactory {
    fn make_default_decoder(
        &self,
        _ctx: &DecoderContext,
        field: &FieldRef,
        _is_nullable: bool,
    ) -> Result<Option<Box<dyn ArrayDecoder>>, ArrowError> {
        Ok(match *field.data_type() {
            DataType::Decimal128(precision, scale) => {
                Some(Box::new(ExactDecimal { precision, scale }))
            }
            _ => None,
        })
    }
}

struct ExactDecimal {
    precision: u8,
    scale: i8,
}

impl ArrayDecoder for ExactDecimal {
    fn decode(&mut self, tape: &Tape<'_>, positions: &[u32]) -> Result<ArrayRef, ArrowError> {
        let mut builder = Decimal128Builder::with_capacity(positions.len());
        for &position in positions {
            let value = match tape.get(position) {
                TapeElement::Null => {
                    builder.append_null();
                    continue;
                }
                TapeElement::String(value) | TapeElement::Number(value) => tape.get_string(value),
                _ => return Err(tape.error(position, "decimal")),
            };
            // Arrow's default decimal parser rounds excess scale. Exact input
            // accepts fixed-point form with any excess fractional digits zero.
            if value.contains(['e', 'E'])
                || value.split_once('.').is_some_and(|(_, fraction)| {
                    fraction
                        .bytes()
                        .skip(self.scale as usize)
                        .any(|digit| digit != b'0')
                })
            {
                return Err(ArrowError::ParseError(format!(
                    "NDJSON decimal cannot be represented exactly at scale {}",
                    self.scale
                )));
            }
            builder.append_value(arrow::compute::kernels::cast_utils::parse_decimal::<
                Decimal128Type,
            >(value, self.precision, self.scale)?);
        }
        Ok(Arc::new(
            builder
                .finish()
                .with_precision_and_scale(self.precision, self.scale)?,
        ))
    }
}
