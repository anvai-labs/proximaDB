//! Bind a Delta log version once and read its active Parquet files with Kernel.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, ensure};
use arrow_array::{
    Array, BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray, StructArray,
};
use arrow_schema::{Schema, SchemaRef};
use delta_kernel::actions::deletion_vector::{DeletionVectorDescriptor, DeletionVectorStorageType};
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::expressions::{Expression, ExpressionStructPatch};
use delta_kernel::scan::state::{ScanFile, transform_to_logical};
use delta_kernel::schema::SchemaRef as DeltaSchemaRef;
use delta_kernel::{
    CancellationToken, CancellationTokenRef, CancelledFuture, Engine, ExpressionRef, FileMeta,
    Snapshot,
};
use futures::stream::{self, StreamExt};
use proximadb_catalog::CatalogColumn;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken as StopToken;
use url::Url;

use super::confined_delta_engine::{ConfinedDeltaEngine, DELTA_ROOT, within_delta_root};
use super::confined_file_store::ConfinedFileStore;
use crate::query::execution::engine::ExecutionControls;
use crate::query::execution::pinned_source::{PinnedBatchSource, PinnedBatchStream};
use crate::services::ddl::pinned_delta::PinnedDeltaLocation;

const MAX_ACTIVE_FILES: usize = 16_384;
const MAX_DV_POSITIONS: u64 = 1_048_576;

pub(crate) struct PinnedDeltaReader {
    engine: Arc<ConfinedDeltaEngine>,
    schema: SchemaRef,
    physical: DeltaSchemaRef,
    logical: DeltaSchemaRef,
    files: Arc<[DeltaFile]>,
    controls: ExecutionControls,
}

struct DeltaFile {
    meta: FileMeta,
    transform: Option<ExpressionRef>,
    deletion_vector: Option<DeletionVectorDescriptor>,
}

struct ScanProjection {
    logical: Option<DeltaSchemaRef>,
    positional_physical: Option<DeltaSchemaRef>,
    indices: Vec<usize>,
    logical_width: usize,
}

struct ProjectedRead {
    physical: DeltaSchemaRef,
    logical: DeltaSchemaRef,
    transform: Option<ExpressionRef>,
}

impl ScanProjection {
    fn new(
        indices: &[usize],
        physical: &DeltaSchemaRef,
        logical: &DeltaSchemaRef,
    ) -> anyhow::Result<Self> {
        fn select(schema: &DeltaSchemaRef, indices: &[usize]) -> anyhow::Result<DeltaSchemaRef> {
            let names = indices
                .iter()
                .map(|&index| {
                    schema
                        .field_at_index(index)
                        .map(|field| field.name.as_str())
                        .context("projection index outside Delta schema")
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok(schema.project(&names)?)
        }
        // Kernel's default Arrow engine needs a field when constructing its
        // struct representation. Keep a single column for row-only scans.
        let decode_indices = if indices.is_empty() {
            &[0][..]
        } else {
            indices
        };
        let unique = indices.iter().copied().collect::<HashSet<_>>().len() == indices.len();
        Ok(Self {
            logical: unique
                .then(|| select(logical, decode_indices))
                .transpose()?,
            positional_physical: (unique && physical.fields().len() == logical.fields().len())
                .then(|| select(physical, decode_indices))
                .transpose()?,
            indices: indices.to_vec(),
            logical_width: logical.fields().len(),
        })
    }

    fn for_file(
        &self,
        file: &DeltaFile,
        physical: &DeltaSchemaRef,
    ) -> anyhow::Result<Option<ProjectedRead>> {
        let Some(logical) = &self.logical else {
            return Ok(None);
        };
        let selected = match file.transform.as_deref() {
            None => self
                .positional_physical
                .as_ref()
                .filter(|schema| *schema == logical)
                .cloned()
                .map(|physical| ProjectedRead {
                    physical,
                    logical: logical.clone(),
                    transform: None,
                }),
            // An empty, top-level patch maps names by output position alone.
            Some(Expression::StructPatch(patch))
                if patch.input_path().is_none() && patch.is_empty() =>
            {
                self.positional_physical
                    .clone()
                    .map(|physical| ProjectedRead {
                        physical,
                        logical: logical.clone(),
                        transform: file.transform.clone(),
                    })
            }
            Some(Expression::StructPatch(patch)) => {
                self.partition_read(patch, physical, logical)?
            }
            _ => None,
        };
        Ok(selected)
    }

    /// Kernel reconstructs standard partition values with literal insertions.
    /// Keep other patch forms on the full-schema path.
    fn partition_read(
        &self,
        patch: &ExpressionStructPatch,
        physical: &DeltaSchemaRef,
        logical: &DeltaSchemaRef,
    ) -> anyhow::Result<Option<ProjectedRead>> {
        if patch.input_path().is_some()
            || patch
                .field_patches
                .iter()
                .any(|(field, edit)| !edit.keep_input || physical.field(field).is_none())
        {
            return Ok(None);
        }

        // Entries pair a logical output expression with the physical field
        // it reads. Kernel evaluates the retained literals using their types.
        let mut outputs = Vec::new();
        fn insert_literals<'a>(
            outputs: &mut Vec<(Option<&'a str>, ExpressionRef)>,
            expressions: &[ExpressionRef],
        ) -> bool {
            if expressions
                .iter()
                .any(|expr| !matches!(expr.as_ref(), Expression::Literal(_)))
            {
                return false;
            }
            outputs.extend(expressions.iter().cloned().map(|expr| (None, expr)));
            true
        }

        if !insert_literals(&mut outputs, &patch.prepended_fields) {
            return Ok(None);
        }
        for field in physical.fields() {
            let name = field.name.as_str();
            outputs.push((Some(name), Arc::new(Expression::column([name]))));
            if let Some(edit) = patch.field_patches.get(name)
                && !insert_literals(&mut outputs, &edit.insertions)
            {
                return Ok(None);
            }
        }
        if !insert_literals(&mut outputs, &patch.appended_fields)
            || outputs.len() != self.logical_width
        {
            return Ok(None);
        }

        let indices = if self.indices.is_empty() {
            &[0][..]
        } else {
            &self.indices
        };
        let mut names = Vec::new();
        let mut expressions = Vec::with_capacity(indices.len());
        for &index in indices {
            let Some((name, expr)) = outputs.get(index) else {
                return Ok(None);
            };
            names.extend(*name);
            expressions.push(expr.clone());
        }
        if names.is_empty() {
            // A literal-only projection still needs a physical column to
            // drive row counts. Do not derive them from Delta statistics.
            let Some(anchor) = physical.fields().next() else {
                return Ok(None);
            };
            names.push(anchor.name.as_str());
        }
        Ok(Some(ProjectedRead {
            physical: physical.project(&names)?,
            logical: logical.clone(),
            transform: Some(Arc::new(Expression::Struct(expressions, None))),
        }))
    }
}

struct QueryCancellation {
    controls: ExecutionControls,
    stop: StopToken,
}

impl CancellationToken for QueryCancellation {
    fn is_cancelled(&self) -> bool {
        self.stop.is_cancelled() || self.controls.check_cancelled().is_err()
    }

    fn cancelled_future(&self) -> CancelledFuture<'_> {
        Box::pin(async move {
            if self.controls.cancellation_flag.is_none() {
                self.stop.cancelled().await;
                return;
            }
            while !self.is_cancelled() {
                // Pgwire carries an atomic flag. Poll it only while Kernel
                // has a cancellable I/O wait in flight.
                tokio::select! {
                    _ = self.stop.cancelled() => return,
                    _ = tokio::time::sleep(Duration::from_millis(20)) => {}
                }
            }
        })
    }
}

impl PinnedDeltaReader {
    pub(crate) async fn open(
        location: PinnedDeltaLocation,
        columns: &[CatalogColumn],
        controls: ExecutionControls,
    ) -> anyhow::Result<Self> {
        let store = ConfinedFileStore::open(location.root, location.relative)
            .await?
            .with_delta_log();
        let engine = Arc::new(ConfinedDeltaEngine::new(Arc::new(store)));
        let columns = columns.to_vec();
        let stop = StopToken::new();
        let _stop_on_drop = stop.clone().drop_guard();
        let token: CancellationTokenRef = Arc::new(QueryCancellation {
            controls: controls.clone(),
            stop,
        });
        tokio::task::spawn_blocking(move || {
            let snapshot = Snapshot::builder_for(DELTA_ROOT)
                .at_version(location.version)
                .with_cancellation_token(token.clone())
                .build(engine.as_ref())
                .with_context(|| format!("cannot open Delta version {}", location.version))?;
            let scan = snapshot
                .scan_builder()
                .with_cancellation_token(token.clone())
                .build()?;
            let schema: Schema = scan.logical_schema().as_ref().try_into_arrow()?;
            super::pinned_external::validate_external_schema(&schema, &columns)
                .context("external Delta catalog columns must match the pinned schema")?;
            let root = Url::parse(DELTA_ROOT)?;
            let mut files = Vec::new();
            let mut paths = HashSet::new();
            for metadata in scan.scan_metadata(engine.as_ref())? {
                let metadata = metadata?;
                let mut dvs = deletion_vectors(&metadata)?;
                let entries = metadata.visit_scan_files(Vec::new(), Vec::push)?;
                for entry in entries {
                    ensure!(
                        files.len() < MAX_ACTIVE_FILES,
                        "Delta snapshot exceeds 16384 active files"
                    );
                    let dv = dvs.remove(&entry.path);
                    ensure!(
                        entry.dv_info.has_vector() == dv.is_some(),
                        "Delta deletion vector does not match scan metadata"
                    );
                    let file = bind_file(&root, entry, dv)?;
                    ensure!(
                        paths.insert(within_delta_root(&file.meta.location)?),
                        "duplicate active Delta data path"
                    );
                    files.push(file);
                }
                ensure!(dvs.is_empty(), "unmatched Delta deletion vector");
            }
            controls.check_cancelled()?;
            Ok(Self {
                engine,
                schema: Arc::new(schema),
                physical: scan.physical_schema().clone(),
                logical: scan.logical_schema().clone(),
                files: files.into(),
                controls,
            })
        })
        .await?
    }

    fn scan_file(
        &self,
        file: &DeltaFile,
        projection: Option<&ScanProjection>,
        token: &CancellationTokenRef,
        output: &mpsc::Sender<anyhow::Result<RecordBatch>>,
    ) -> anyhow::Result<()> {
        ensure!(!token.is_cancelled(), "Delta scan cancelled");
        // Decode with Kernel and keep sparse positions. A high row number
        // must not allocate a dense mask for the entire Parquet file.
        let deletes = if let Some(descriptor) = &file.deletion_vector {
            let root = Url::parse(DELTA_ROOT)?;
            let vector = descriptor.read(self.engine.storage_handler(), &root)?;
            ensure!(
                vector.len() == u64::try_from(descriptor.cardinality)?,
                "invalid Delta deletion vector cardinality"
            );
            Some(vector)
        } else {
            None
        };
        let projected = projection
            .map(|projection| projection.for_file(file, &self.physical))
            .transpose()?
            .flatten();
        let (physical, logical, transform) = match &projected {
            Some(read) => (&read.physical, &read.logical, &read.transform),
            None => (&self.physical, &self.logical, &file.transform),
        };
        let residual_projection = projection
            .map(|projection| projection.indices.as_slice())
            .filter(|indices| projected.is_none() || indices.is_empty());
        let source = self
            .engine
            .parquet_handler()
            .read_parquet_files_with_cancellation(
                std::slice::from_ref(&file.meta),
                physical.clone(),
                None,
                Some(token.clone()),
            )?;
        let mut position = 0u64;
        for batch in source {
            ensure!(!token.is_cancelled(), "Delta scan cancelled");
            let batch = transform_to_logical(
                self.engine.as_ref(),
                batch?,
                physical,
                logical,
                transform.clone(),
            )?
            .try_into_record_batch()?;
            let end = position
                .checked_add(batch.num_rows() as u64)
                .context("Delta row position overflow")?;
            let batch = match &deletes {
                Some(vector) if vector.range_cardinality(position..end) != 0 => {
                    let keep: BooleanArray =
                        (position..end).map(|row| !vector.contains(row)).collect();
                    arrow_select::filter::filter_record_batch(&batch, &keep)?
                }
                _ => batch,
            };
            position = end;
            let batch = match residual_projection {
                Some(projection) => batch.project(projection)?,
                None => batch,
            };
            if output.blocking_send(Ok(batch)).is_err() {
                return Ok(());
            }
        }
        ensure!(
            deletes
                .as_ref()
                .and_then(|vector| vector.max())
                .is_none_or(|row| row < position),
            "Delta deletion vector references a position outside its Parquet file"
        );
        Ok(())
    }
}

impl PinnedBatchSource for PinnedDeltaReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn split_count(&self) -> usize {
        self.files.len()
    }

    fn scan_splits(
        &self,
        splits: Range<usize>,
        projection: Option<&[usize]>,
        _batch_size: usize,
    ) -> anyhow::Result<PinnedBatchStream> {
        ensure!(
            self.files.get(splits.clone()).is_some(),
            "Delta split outside bound snapshot"
        );
        let reader = Self {
            engine: self.engine.clone(),
            schema: self.schema.clone(),
            physical: self.physical.clone(),
            logical: self.logical.clone(),
            files: self.files.clone(),
            controls: self.controls.clone(),
        };
        let projection = projection
            .map(|indices| ScanProjection::new(indices, &self.physical, &self.logical))
            .transpose()?;
        let stop = StopToken::new();
        let drop_guard = stop.clone().drop_guard();
        let token: CancellationTokenRef = Arc::new(QueryCancellation {
            controls: self.controls.clone(),
            stop,
        });
        let (output, input) = mpsc::channel(1);
        // The Kernel iterator is synchronous. Keep it off the async workers
        // and bound queued Arrow batches until the downstream consumer pulls.
        let worker = tokio::task::spawn_blocking(move || {
            for file in &reader.files[splits] {
                if let Err(error) = reader.scan_file(file, projection.as_ref(), &token, &output) {
                    let _ = output.blocking_send(Err(error));
                    return;
                }
                if output.is_closed() {
                    return;
                }
            }
        });
        Ok(stream::unfold(
            (input, Some(worker), drop_guard),
            |(mut input, mut worker, guard)| async move {
                if let Some(batch) = input.recv().await {
                    return Some((batch, (input, worker, guard)));
                }
                let ended = worker.take()?.await;
                ended
                    .err()
                    .map(|error| (Err(error.into()), (input, worker, guard)))
            },
        )
        .boxed())
    }
}

fn bind_file(
    root: &Url,
    entry: ScanFile,
    deletion_vector: Option<DeletionVectorDescriptor>,
) -> anyhow::Result<DeltaFile> {
    ensure!(
        !entry.path.is_empty()
            && entry.path.len() <= 4096
            && entry
                .path
                .split('/')
                .all(|part| !part.is_empty() && part != ".." && part != ".")
            && !entry.path.contains(['\\', ':', '?', '#']),
        "invalid relative Delta data path"
    );
    let path = root.join(&entry.path)?;
    let key = within_delta_root(&path)?;
    ensure!(
        key == object_store::path::Path::from_url_path(&entry.path)?
            && key.as_ref().ends_with(".parquet")
            && !key.as_ref().starts_with("_delta_log/"),
        "Delta data reference must name Parquet beneath the table"
    );
    if let Some(dv) = &deletion_vector {
        ensure!(
            matches!(
                dv.storage_type,
                DeletionVectorStorageType::Inline | DeletionVectorStorageType::PersistedRelative
            ) && (1..=16 * 1024 * 1024).contains(&dv.size_in_bytes)
                && u64::try_from(dv.cardinality)? <= MAX_DV_POSITIONS,
            "unsupported Delta deletion vector storage or size"
        );
        if let Some(location) = dv.absolute_path(root)? {
            let key = within_delta_root(&location)?;
            ensure!(
                key.as_ref().ends_with(".bin") && !key.as_ref().starts_with("_delta_log/"),
                "invalid Delta deletion vector path"
            );
        }
    }
    Ok(DeltaFile {
        meta: FileMeta::new(path, entry.modification_time, u64::try_from(entry.size)?),
        transform: entry.transform,
        deletion_vector,
    })
}

// Kernel's public scan file exposes presence and cardinality. Read the typed
// descriptor from the same filtered metadata batch to get bounded sparse decoding.
fn deletion_vectors(
    metadata: &delta_kernel::scan::ScanMetadata,
) -> anyhow::Result<HashMap<String, DeletionVectorDescriptor>> {
    fn column<'a, A: Array + 'static>(array: &'a dyn Array, name: &str) -> anyhow::Result<&'a A> {
        array
            .as_any()
            .downcast_ref()
            .with_context(|| format!("invalid Delta scan column {name}"))
    }

    let batch = metadata
        .scan_files
        .data()
        .any_ref()
        .downcast_ref::<ArrowEngineData>()
        .context("expected Delta Kernel Arrow metadata")?
        .record_batch();
    let dvs = column::<StructArray>(
        batch
            .column_by_name("deletionVector")
            .context("missing Delta deletionVector metadata")?
            .as_ref(),
        "deletionVector",
    )?;
    if dvs.null_count() == dvs.len() {
        return Ok(HashMap::new());
    }
    let paths = column::<StringArray>(
        batch
            .column_by_name("path")
            .context("missing Delta path metadata")?
            .as_ref(),
        "path",
    )?;
    let field = |name| {
        dvs.column_by_name(name)
            .map(AsRef::as_ref)
            .with_context(|| format!("missing Delta deletionVector field {name}"))
    };
    let storage = column::<StringArray>(field("storageType")?, "storageType")?;
    let value = column::<StringArray>(field("pathOrInlineDv")?, "pathOrInlineDv")?;
    let offset = column::<Int32Array>(field("offset")?, "offset")?;
    let size = column::<Int32Array>(field("sizeInBytes")?, "sizeInBytes")?;
    let cardinality = column::<Int64Array>(field("cardinality")?, "cardinality")?;
    let mut by_path = HashMap::new();
    for i in 0..batch.num_rows() {
        if dvs.is_null(i)
            || !metadata
                .scan_files
                .selection_vector()
                .get(i)
                .copied()
                .unwrap_or(true)
        {
            continue;
        }
        ensure!(
            [paths as &dyn Array, storage, value, size, cardinality]
                .iter()
                .all(|array| array.is_valid(i)),
            "incomplete Delta deletion vector descriptor"
        );
        let descriptor = DeletionVectorDescriptor {
            storage_type: storage.value(i).parse()?,
            path_or_inline_dv: value.value(i).to_owned(),
            offset: offset.is_valid(i).then(|| offset.value(i)),
            size_in_bytes: size.value(i),
            cardinality: cardinality.value(i),
        };
        ensure!(
            by_path
                .insert(paths.value(i).to_owned(), descriptor)
                .is_none(),
            "duplicate Delta deletion vector path"
        );
    }
    Ok(by_path)
}
