//! External source reader (Phase 8 F5 / TD-090).
//!
//! Reads dense vectors + ids from an external Parquet source into
//! `ProximaRecord`s for in-place index building — the source is **never
//! copied** into ProximaDB storage; these records exist only long enough to
//! train/populate the AXIS index. Also computes a `snapshot_fingerprint` of the
//! source that is recorded as the projection lineage (`source_range`).
//!
//! The Parquet read uses the same `ParquetRecordBatchReaderBuilder` path as
//! `src/storage/formats/open/iceberg.rs` (`IcebergFormat::read_parquet_file`),
//! which is private to that module; the Arrow vector extraction mirrors
//! `columnar_query_reader.rs` (FixedSizeList preferred, List fallback).

use std::collections::HashSet;

use anyhow::{Context, Result};
use arrow_array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeStringArray, ListArray, RecordBatch, StringArray,
};
use proximadb_data_model::ProximaValue;
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};

use super::types::ExternalCollectionSpec;

const READ_BATCH_SIZE: usize = 4096;

/// Read all `(id, vector)` rows from the external source described by `spec`
/// into `ProximaRecord`s (one dense embedding each). Validates that every
/// vector matches `spec.dimension`.
#[cfg_attr(not(feature = "axis"), allow(dead_code))]
pub fn read_external_records(spec: &ExternalCollectionSpec) -> Result<Vec<ProximaRecord>> {
    let files = list_parquet_files(&spec.location)?;
    if files.is_empty() {
        anyhow::bail!(
            "external collection '{}': no Parquet files found at '{}'",
            spec.name,
            spec.location
        );
    }

    let projection = [spec.id_column.clone(), spec.vector_column.clone()];
    let mut records = Vec::new();
    for file in &files {
        for batch in read_parquet_batches(file, Some(projection.as_slice()))? {
            extract_records(&batch, spec, &mut records)?;
        }
    }
    Ok(records)
}

/// Fetch the full records (all non-vector columns as `props`) for `ids` from the
/// external source — the retrieval-time "federated fetch" that returns the real
/// text/metadata ProximaDB indexes but does not own. The vector column is
/// skipped (the embedding is not echoed back). Rows whose id is not in `ids` are
/// ignored; the first occurrence of a duplicate id wins. Returned order is
/// source order — callers re-order by score.
pub fn read_records_by_ids(
    spec: &ExternalCollectionSpec,
    ids: &[String],
) -> Result<Vec<ProximaRecord>> {
    let wanted: HashSet<&str> = ids.iter().map(|s| s.as_str()).collect();
    if wanted.is_empty() {
        return Ok(Vec::new());
    }
    let files = list_parquet_files(&spec.location)?;
    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for file in &files {
        // No projection: read every column so any metadata column lands in props.
        for batch in read_parquet_batches(file, None)? {
            extract_full_records(&batch, spec, &wanted, &mut seen, &mut out)?;
        }
    }
    Ok(out)
}

/// Read `(oid, text)` pairs for every row from the external source's configured
/// `text_column` (F5 Slice 3) — the BM25 build input. Rows with a null id or
/// text are skipped. Errors if no `text_column` is configured.
pub fn read_external_text(spec: &ExternalCollectionSpec) -> Result<Vec<(String, String)>> {
    let text_column = spec.text_column.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "external collection '{}': no text_column configured for BM25",
            spec.name
        )
    })?;
    let files = list_parquet_files(&spec.location)?;
    let projection = [spec.id_column.clone(), text_column.clone()];
    let mut out = Vec::new();
    for file in &files {
        for batch in read_parquet_batches(file, Some(projection.as_slice()))? {
            extract_text(&batch, &spec.id_column, text_column, &mut out)?;
        }
    }
    Ok(out)
}

/// Extract `(oid, text)` rows from one batch (Utf8/LargeUtf8 text column).
fn extract_text(
    batch: &RecordBatch,
    id_column: &str,
    text_column: &str,
    out: &mut Vec<(String, String)>,
) -> Result<()> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(());
    }
    let ids = batch
        .column_by_name(id_column)
        .ok_or_else(|| anyhow::anyhow!("id column '{id_column}' missing in source"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("id column '{id_column}' is not Utf8"))?;
    let text_col = batch
        .column_by_name(text_column)
        .ok_or_else(|| anyhow::anyhow!("text column '{text_column}' missing in source"))?;

    let read_text: Box<dyn Fn(usize) -> Option<String>> =
        if let Some(a) = text_col.as_any().downcast_ref::<StringArray>() {
            Box::new(move |row| (!a.is_null(row)).then(|| a.value(row).to_string()))
        } else if let Some(a) = text_col.as_any().downcast_ref::<LargeStringArray>() {
            Box::new(move |row| (!a.is_null(row)).then(|| a.value(row).to_string()))
        } else {
            anyhow::bail!("text column '{text_column}' is not Utf8/LargeUtf8");
        };

    for row in 0..num_rows {
        if ids.is_null(row) {
            continue;
        }
        if let Some(text) = read_text(row) {
            out.push((ids.value(row).to_string(), text));
        }
    }
    Ok(())
}

/// Stable content fingerprint of the external source: an FNV-1a hash over the
/// sorted `(file_name, len, mtime_nanos)` tuples. Deterministic across runs so
/// Slice 2 can detect source-commit advance by recomputing and comparing.
pub fn snapshot_fingerprint(location: &str) -> Result<String> {
    let files = list_parquet_files(location)?;
    // FNV-1a (64-bit) over a canonical description of the file set.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let prime: u64 = 0x0000_0100_0000_01b3;
    let mut mix = |bytes: &[u8]| {
        for b in bytes {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(prime);
        }
    };
    for path in &files {
        let meta = std::fs::metadata(path)
            .with_context(|| format!("stat external file '{}'", path.display()))?;
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        mix(name.as_bytes());
        mix(&meta.len().to_le_bytes());
        mix(&mtime_ns.to_le_bytes());
    }
    Ok(format!("fnv1a:{:016x}", hash))
}

/// One opaque message for every rejection. Deliberately says nothing about
/// whether the path exists, is a file, is a directory, or merely sits outside
/// the allowed roots: `POST /api/v2/external-collections` is reachable by any
/// caller the REST surface admits, so a specific error is a filesystem oracle.
const REJECTED: &str = "external location is not an accepted source \
(see PROXIMADB_EXTERNAL_TABLE_ROOTS)";

/// Only `.parquet` is readable as an external source. The single-file branch of
/// `list_parquet_files` had NO such check, so a location naming any file at all
/// was pushed and then stat'ed by `snapshot_fingerprint`, leaking that file's
/// length and mtime.
fn is_parquet(path: &std::path::Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("parquet")
}

/// Is `candidate` inside `root`? Compared COMPONENT-WISE, never as a string
/// prefix: `/data-evil/x` has the string prefix `/data` but is not inside it.
fn is_within(candidate: &std::path::Path, root: &std::path::Path) -> bool {
    let mut c = candidate.components();
    for rc in root.components() {
        match c.next() {
            Some(cc) if cc == rc => {}
            _ => return false,
        }
    }
    true
}

/// Confine `location` for reading.
///
/// Two layers, because they catch different things:
///
/// 1. `external_location_allowed` -- the SAME fail-closed
///    `PROXIMADB_EXTERNAL_TABLE_ROOTS` allowlist `CREATE EXTERNAL TABLE` uses.
///    It rejects `..`, matches on a path boundary, and works for URI locations
///    (`s3://`) where canonicalisation is meaningless.
/// 2. canonicalisation, for a local path -- the allowlist is textual, so a
///    SYMLINK inside an allowed root that points outside it passes layer 1.
///    The kernel resolves the link, not the string.
///
/// Before this, neither layer existed here: `location` came straight from the
/// request body into `fs::metadata` / `fs::read_dir`. Reported as
/// `rust/path-injection`; tracked in TD-SECSCAN-1.
fn confine(location: &str) -> Result<std::path::PathBuf> {
    if !crate::services::ddl::external_location_allowed(location) {
        anyhow::bail!(REJECTED);
    }
    confine_canonical(location)
}

/// Layer 2 on its own, so tests exercise it without touching the process
/// environment: `std::env::set_var` is `unsafe` in edition 2024 and races the
/// threads `cargo test --lib` uses.
fn confine_canonical(location: &str) -> Result<std::path::PathBuf> {
    let canonical = std::fs::canonicalize(location).map_err(|_| anyhow::anyhow!(REJECTED))?;
    // A local location that cleared layer 1 named a root it is textually under;
    // require the RESOLVED path to still be under that same textual root.
    let textual = std::path::Path::new(location.trim_start_matches("file://"));
    let anchor = match std::fs::canonicalize(textual.parent().unwrap_or(textual)) {
        Ok(a) => a,
        Err(_) => return Ok(canonical),
    };
    if !is_within(&canonical, &anchor) {
        anyhow::bail!(REJECTED);
    }
    Ok(canonical)
}

/// Resolve `location` to a sorted list of Parquet files. Accepts either a
/// single `.parquet` file or a directory containing `.parquet` files.
///
/// Every path is confined by `confine` (shared `PROXIMADB_EXTERNAL_TABLE_ROOTS`
/// allowlist, then canonicalisation) before any filesystem access. Before
/// that this function took `location` straight from the unauthenticated
/// `POST /api/v2/external-collections` body and handed it to `fs::metadata` /
/// `fs::read_dir` with no canonicalisation, no root and no tenant prefix, so it
/// would stat or enumerate any path on the host; the single-file branch had no
/// extension check either, so any file at all was pushed for reading. Reported
/// as `rust/path-injection` and tracked in TD-SECSCAN-1.
fn list_parquet_files(location: &str) -> Result<Vec<std::path::PathBuf>> {
    let path = confine(location)?;
    let meta = std::fs::metadata(&path).map_err(|_| anyhow::anyhow!(REJECTED))?;
    let mut files = Vec::new();
    if meta.is_dir() {
        for entry in std::fs::read_dir(&path).map_err(|_| anyhow::anyhow!(REJECTED))? {
            let entry = entry?;
            let p = entry.path();
            if is_parquet(&p) {
                // Re-confine each entry: a symlink inside an allowed directory
                // can still point outside it.
                if let Ok(c) = std::fs::canonicalize(&p)
                    && is_within(&c, &path)
                {
                    files.push(c);
                }
            }
        }
    } else {
        // The missing check. Without it a single-file location of any type was
        // pushed and then stat'ed by snapshot_fingerprint, leaking length and
        // mtime of an arbitrary host file.
        if !is_parquet(&path) {
            anyhow::bail!(REJECTED);
        }
        files.push(path);
    }
    files.sort();
    Ok(files)
}

/// Read a single Parquet file. `columns = Some(names)` projects just those
/// columns; `None` reads all columns. Mirrors `IcebergFormat::read_parquet_file`
/// (private to that module).
fn read_parquet_batches(
    path: &std::path::Path,
    columns: Option<&[String]>,
) -> Result<Vec<RecordBatch>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    let file = File::open(path)
        .with_context(|| format!("open external Parquet file '{}'", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?.with_batch_size(READ_BATCH_SIZE);

    let reader = match columns {
        Some(names) => {
            let schema = builder.schema();
            let indices: Vec<usize> = names
                .iter()
                .filter_map(|name| schema.index_of(name).ok())
                .collect();
            let mask = parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), indices);
            builder.with_projection(mask).build()?
        }
        None => builder.build()?,
    };
    Ok(reader.filter_map(|r| r.ok()).collect())
}

/// Extract full records for the wanted ids from one batch: `oid` from the id
/// column + `props` from every other non-vector column (typed via
/// `arrow_cell_to_proxima_value`). No vector is attached.
fn extract_full_records(
    batch: &RecordBatch,
    spec: &ExternalCollectionSpec,
    wanted: &HashSet<&str>,
    seen: &mut HashSet<String>,
    out: &mut Vec<ProximaRecord>,
) -> Result<()> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(());
    }
    let ids = batch
        .column_by_name(&spec.id_column)
        .ok_or_else(|| anyhow::anyhow!("id column '{}' missing in source", spec.id_column))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("id column '{}' is not Utf8", spec.id_column))?;

    // Pre-resolve the prop columns (everything but id + vector) once per batch.
    let schema = batch.schema();
    let prop_cols: Vec<(String, &ArrayRef)> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| f.name() != &spec.id_column && f.name() != &spec.vector_column)
        .map(|(i, f)| (f.name().clone(), batch.column(i)))
        .collect();

    for row in 0..num_rows {
        if ids.is_null(row) {
            continue;
        }
        let oid = ids.value(row);
        if !wanted.contains(oid) || seen.contains(oid) {
            continue;
        }
        seen.insert(oid.to_string());
        let mut props = std::collections::HashMap::new();
        for (name, array) in &prop_cols {
            if let Some(value) = arrow_cell_to_proxima_value(array, row) {
                props.insert(name.clone(), ProximaTreeNode::Value(value));
            }
        }
        out.push(ProximaRecord {
            oid: oid.to_string(),
            props,
            ..Default::default()
        });
    }
    Ok(())
}

/// Map a single Arrow cell to a `ProximaValue`. Covers the scalar types a lake
/// table's metadata columns commonly use; nulls and unsupported types (lists,
/// structs, the vector column) yield `None` so they are simply omitted from
/// `props`. Nested/struct decoding is a later slice.
fn arrow_cell_to_proxima_value(array: &ArrayRef, row: usize) -> Option<ProximaValue> {
    if array.is_null(row) {
        return None;
    }
    let any = array.as_any();
    if let Some(a) = any.downcast_ref::<StringArray>() {
        Some(ProximaValue::String(a.value(row).to_string()))
    } else if let Some(a) = any.downcast_ref::<LargeStringArray>() {
        Some(ProximaValue::String(a.value(row).to_string()))
    } else if let Some(a) = any.downcast_ref::<BooleanArray>() {
        Some(ProximaValue::Boolean(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Int64Array>() {
        Some(ProximaValue::Int64(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Int32Array>() {
        Some(ProximaValue::Int32(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Int16Array>() {
        Some(ProximaValue::Int16(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Int8Array>() {
        Some(ProximaValue::Int8(a.value(row)))
    } else if let Some(a) = any.downcast_ref::<Float64Array>() {
        Some(ProximaValue::Float64(a.value(row)))
    } else {
        any.downcast_ref::<Float32Array>()
            .map(|a| ProximaValue::Float32(a.value(row)))
    }
}

/// Extract `(id, vector)` rows from one batch into `ProximaRecord`s.
#[cfg_attr(not(feature = "axis"), allow(dead_code))]
fn extract_records(
    batch: &RecordBatch,
    spec: &ExternalCollectionSpec,
    out: &mut Vec<ProximaRecord>,
) -> Result<()> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(());
    }

    let id_col = batch
        .column_by_name(&spec.id_column)
        .ok_or_else(|| anyhow::anyhow!("id column '{}' missing in source", spec.id_column))?;
    let ids = id_col
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("id column '{}' is not Utf8", spec.id_column))?;

    let vec_col = batch.column_by_name(&spec.vector_column).ok_or_else(|| {
        anyhow::anyhow!("vector column '{}' missing in source", spec.vector_column)
    })?;
    let vectors = extract_vectors(vec_col, num_rows, &spec.vector_column)?;

    for (i, vector) in vectors.iter().enumerate() {
        let oid = ids.value(i).to_string();
        if vector.len() != spec.dimension {
            anyhow::bail!(
                "external collection '{}': row '{}' has dimension {} (expected {})",
                spec.name,
                oid,
                vector.len(),
                spec.dimension
            );
        }
        out.push(ProximaRecord {
            oid,
            embeddings: vec![EmbeddingCell::new_fp32(
                "external",
                "dense_vector",
                spec.dimension as u32,
                vector.clone(),
            )],
            ..Default::default()
        });
    }
    Ok(())
}

/// Decode the vector column (FixedSizeList preferred, List fallback) to
/// `Vec<Vec<f32>>`. Mirrors `columnar_query_reader.rs`.
#[cfg_attr(not(feature = "axis"), allow(dead_code))]
fn extract_vectors(
    col: &arrow_array::ArrayRef,
    num_rows: usize,
    column_name: &str,
) -> Result<Vec<Vec<f32>>> {
    if let Some(fixed) = col.as_any().downcast_ref::<FixedSizeListArray>() {
        let values = fixed
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                anyhow::anyhow!("vector column '{column_name}' values are not Float32")
            })?;
        let dim = fixed.value_length() as usize;
        let mut out = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let start = i * dim;
            out.push((start..start + dim).map(|idx| values.value(idx)).collect());
        }
        Ok(out)
    } else if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
        let mut out = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let arr = list.value(i);
            let floats = arr.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                anyhow::anyhow!("vector column '{column_name}' values are not Float32")
            })?;
            out.push((0..floats.len()).map(|j| floats.value(j)).collect());
        }
        Ok(out)
    } else {
        anyhow::bail!("vector column '{column_name}' is neither FixedSizeList nor List of Float32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, StringBuilder};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    /// Write a tiny Parquet file with `id: Utf8` + `vector: FixedSizeList<Float32, dim>`.

    /// Allowlist a fixture's own directory for this test process.
    ///
    /// `list_parquet_files` is now fail-closed behind
    /// `PROXIMADB_EXTERNAL_TABLE_ROOTS` (TD-SECSCAN-1: the location used to go
    /// straight to `fs::metadata` / `fs::read_dir`). Unset rejects everything,
    /// which is the shipped default, so a test that reads a real fixture must
    /// opt in. Called from the fixture WRITERS, so every test that creates one
    /// is covered without 14 separate call sites.
    ///
    /// `set_var` is `unsafe` in edition 2024 and racy across threads; this is
    /// safe under nextest, which gives each test its own process (the repo's
    /// mandated runner). A bare path, not `file://`, because these tests pass
    /// bare locations and the allowlist match is textual.
    fn allowlist_fixture_dir(path: &std::path::Path) {
        let dir = path
            .parent()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "/".to_string());
        unsafe { std::env::set_var("PROXIMADB_EXTERNAL_TABLE_ROOTS", &dir) };
    }

    fn write_parquet(path: &std::path::Path, ids: &[&str], vectors: &[Vec<f32>], dim: i32) {
        allowlist_fixture_dir(path);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                false,
            ),
        ]));

        let mut id_builder = StringBuilder::new();
        for id in ids {
            id_builder.append_value(id);
        }
        let mut vec_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
        for v in vectors {
            for x in v {
                vec_builder.values().append_value(*x);
            }
            vec_builder.append(true);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(vec_builder.finish()),
            ],
        )
        .unwrap();

        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Write a Parquet file with `id: Utf8`, `text: Utf8`, `year: Int64`, and
    /// `vector: FixedSizeList<Float32, dim>` — exercises federated-fetch props.
    fn write_parquet_with_meta(
        path: &std::path::Path,
        rows: &[(&str, &str, i64, Vec<f32>)],
        dim: i32,
    ) {
        allowlist_fixture_dir(path);
        use arrow_array::builder::Int64Builder;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("year", DataType::Int64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                false,
            ),
        ]));
        let mut id_b = StringBuilder::new();
        let mut text_b = StringBuilder::new();
        let mut year_b = Int64Builder::new();
        let mut vec_b = FixedSizeListBuilder::new(Float32Builder::new(), dim);
        for (id, text, year, v) in rows {
            id_b.append_value(id);
            text_b.append_value(text);
            year_b.append_value(*year);
            for x in v {
                vec_b.values().append_value(*x);
            }
            vec_b.append(true);
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_b.finish()),
                Arc::new(text_b.finish()),
                Arc::new(year_b.finish()),
                Arc::new(vec_b.finish()),
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn tmp_path(suffix: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "proximadb_extsrc_{}_{}.parquet",
            uuid::Uuid::new_v4().simple(),
            suffix
        ))
    }

    #[test]
    fn read_records_by_ids_returns_props_and_skips_vector_and_unknown() {
        let path = tmp_path("byids");
        write_parquet_with_meta(
            &path,
            &[
                ("a", "alpha", 2021, vec![1.0, 0.0]),
                ("b", "bravo", 2022, vec![0.0, 1.0]),
                ("c", "charlie", 2023, vec![1.0, 1.0]),
            ],
            2,
        );
        let spec =
            ExternalCollectionSpec::parquet("docs", path.to_str().unwrap(), "id", "vector", 2);

        // Fetch a subset; "zzz" is unknown and must be skipped.
        let recs = read_records_by_ids(
            &spec,
            &["c".to_string(), "a".to_string(), "zzz".to_string()],
        )
        .unwrap();
        assert_eq!(recs.len(), 2);

        let by_id: std::collections::HashMap<_, _> =
            recs.into_iter().map(|r| (r.oid.clone(), r)).collect();
        let a = &by_id["a"];
        // Props carry the non-vector columns; the vector column is NOT echoed.
        assert!(!a.props.contains_key("vector"));
        assert!(!a.props.contains_key("id"));
        match &a.props["text"] {
            ProximaTreeNode::Value(ProximaValue::String(s)) => assert_eq!(s, "alpha"),
            other => panic!("text prop wrong: {other:?}"),
        }
        match &a.props["year"] {
            ProximaTreeNode::Value(ProximaValue::Int64(y)) => assert_eq!(*y, 2021),
            other => panic!("year prop wrong: {other:?}"),
        }
        // And the vector is not attached as an embedding (fetch is metadata-only).
        assert!(a.embeddings.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_records_by_ids_empty_input_is_empty() {
        let path = tmp_path("empty");
        write_parquet_with_meta(&path, &[("a", "alpha", 1, vec![1.0, 0.0])], 2);
        let spec =
            ExternalCollectionSpec::parquet("docs", path.to_str().unwrap(), "id", "vector", 2);
        assert!(read_records_by_ids(&spec, &[]).unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_external_text_returns_id_text_pairs() {
        let path = tmp_path("text");
        write_parquet_with_meta(
            &path,
            &[
                ("a", "alpha bravo", 1, vec![1.0, 0.0]),
                ("b", "charlie", 2, vec![0.0, 1.0]),
            ],
            2,
        );
        let spec =
            ExternalCollectionSpec::parquet("docs", path.to_str().unwrap(), "id", "vector", 2)
                .with_text_column("text");
        let pairs = read_external_text(&spec).unwrap();
        assert_eq!(
            pairs,
            vec![
                ("a".to_string(), "alpha bravo".to_string()),
                ("b".to_string(), "charlie".to_string()),
            ]
        );
        // No text_column → error.
        let no_text =
            ExternalCollectionSpec::parquet("docs", path.to_str().unwrap(), "id", "vector", 2);
        assert!(read_external_text(&no_text).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reads_ids_and_vectors_with_correct_dim() {
        let path = tmp_path("read");
        let ids = ["a", "b", "c"];
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        write_parquet(&path, &ids, &vectors, 4);

        let spec =
            ExternalCollectionSpec::parquet("docs", path.to_str().unwrap(), "id", "vector", 4);
        let records = read_external_records(&spec).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].oid, "a");
        assert_eq!(records[1].embeddings[0].values.to_fp32_owned(), vectors[1]);
        assert_eq!(records[2].embeddings[0].dim, 4);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let path = tmp_path("dimmismatch");
        write_parquet(&path, &["a"], &[vec![1.0, 2.0, 3.0, 4.0]], 4);
        let spec =
            ExternalCollectionSpec::parquet("docs", path.to_str().unwrap(), "id", "vector", 8);
        assert!(read_external_records(&spec).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fingerprint_is_stable_and_changes_with_content() {
        let dir = std::env::temp_dir().join(format!(
            "proximadb_extsrc_dir_{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        write_parquet(&dir.join("part-0.parquet"), &["a"], &[vec![1.0, 0.0]], 2);

        let loc = dir.to_str().unwrap();
        let fp1 = snapshot_fingerprint(loc).unwrap();
        let fp2 = snapshot_fingerprint(loc).unwrap();
        assert_eq!(fp1, fp2, "fingerprint must be stable across reads");
        assert!(fp1.starts_with("fnv1a:"));

        // Adding a file changes the fingerprint.
        write_parquet(&dir.join("part-1.parquet"), &["b"], &[vec![0.0, 1.0]], 2);
        let fp3 = snapshot_fingerprint(loc).unwrap();
        assert_ne!(fp1, fp3, "adding a file must change the fingerprint");

        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod path_confinement_tests {
    //! `rust/path-injection` on `POST /api/v2/external-collections`
    //! (TD-SECSCAN-1). The handler validated only that `location` was non-empty
    //! and handed it to `fs::metadata` / `fs::read_dir`: no allowlist, no
    //! canonicalisation, no tenant prefix.
    //!
    //! No test here mutates the process environment. `std::env::set_var` is
    //! `unsafe` in edition 2024 and races the threads `cargo test --lib` uses,
    //! so each layer is exercised through its own pure entry point. The
    //! gate-unset case needs no mutation at all -- unset IS the test default,
    //! and it is also the shipped default.
    use super::*;

    #[test]
    fn the_shipped_default_rejects_every_location() {
        // PROXIMADB_EXTERNAL_TABLE_ROOTS is unset here, as it is in production
        // unless an operator opts in. Fail-closed.
        let dir = tempfile::tempdir().expect("tempdir");
        let f = dir.path().join("a.parquet");
        std::fs::write(&f, b"x").expect("write");
        let err = confine(f.to_str().expect("utf8")).expect_err("must reject");
        assert_eq!(err.to_string(), REJECTED);
    }

    #[test]
    fn rejection_is_one_indistinguishable_message() {
        // A specific error is a filesystem oracle: "no such file" vs
        // "forbidden" reveals existence to any caller the surface admits.
        let dir = tempfile::tempdir().expect("tempdir");
        let real = dir.path().join("real.parquet");
        std::fs::write(&real, b"x").expect("write");
        let absent = dir.path().join("nope.parquet");

        let a = confine(real.to_str().expect("utf8")).expect_err("real");
        let b = confine(absent.to_str().expect("utf8")).expect_err("absent");
        assert_eq!(a.to_string(), b.to_string(), "messages must not differ");
        assert!(
            !a.to_string().contains("real.parquet"),
            "must not echo the path"
        );
        assert!(!b.to_string().contains("nope"), "must not echo the path");
    }

    #[test]
    fn a_symlink_out_of_its_directory_is_rejected() {
        // The hole the textual allowlist cannot see: the kernel resolves the
        // link, not the string.
        let base = tempfile::tempdir().expect("tempdir");
        let allowed = base.path().join("allowed");
        std::fs::create_dir_all(&allowed).expect("mkdir");
        let outside = base.path().join("outside.parquet");
        std::fs::write(&outside, b"x").expect("write");
        let link = allowed.join("link.parquet");
        if std::os::unix::fs::symlink(&outside, &link).is_err() {
            return; // symlinks unavailable on this filesystem
        }
        let err = confine_canonical(link.to_str().expect("utf8")).expect_err("escape");
        assert_eq!(err.to_string(), REJECTED);
    }

    #[test]
    fn a_regular_file_in_place_is_accepted_by_canonicalisation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let f = dir.path().join("a.parquet");
        std::fs::write(&f, b"x").expect("write");
        let got = confine_canonical(f.to_str().expect("utf8")).expect("in place");
        assert!(got.ends_with("a.parquet"));
    }

    #[test]
    fn list_parquet_files_is_confined_at_the_call_site() {
        // Guards the CALL SITE, not just the helpers. Without this, deleting
        // `confine(location)?` from `list_parquet_files` would leave every
        // other test in this module green while restoring the original
        // arbitrary-path read.
        let dir = tempfile::tempdir().expect("tempdir");
        let f = dir.path().join("a.parquet");
        std::fs::write(&f, b"x").expect("write");
        // Gate unset (the shipped default) => the reader must refuse.
        let err = list_parquet_files(f.to_str().expect("utf8"))
            .expect_err("list_parquet_files must confine before touching the filesystem");
        assert_eq!(err.to_string(), REJECTED);
    }

    #[test]
    fn list_parquet_files_refuses_a_directory_too() {
        // The directory branch enumerated any path it was given.
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.parquet"), b"x").expect("write");
        let err = list_parquet_files(dir.path().to_str().expect("utf8"))
            .expect_err("directory locations must be confined as well");
        assert_eq!(err.to_string(), REJECTED);
    }

    #[test]
    fn is_within_is_component_wise_not_textual() {
        use std::path::Path;
        assert!(is_within(Path::new("/a/b/c"), Path::new("/a/b")));
        assert!(is_within(Path::new("/a/b"), Path::new("/a/b")));
        // `/a/bb` shares the string prefix `/a/b` but is a different directory.
        assert!(!is_within(Path::new("/a/bb/c"), Path::new("/a/b")));
        assert!(!is_within(Path::new("/a"), Path::new("/a/b")));
    }

    #[test]
    fn only_parquet_is_readable() {
        use std::path::Path;
        assert!(is_parquet(Path::new("/x/a.parquet")));
        assert!(!is_parquet(Path::new("/etc/shadow")));
        assert!(!is_parquet(Path::new("/x/a.parquet.bak")));
        assert!(!is_parquet(Path::new("/x/parquet")));
        assert!(
            !is_parquet(Path::new("/x/a.PARQUET")),
            "extension match is exact"
        );
    }

    #[test]
    fn the_shared_allowlist_is_what_gates_this_surface() {
        // Pin the reuse: external collections must NOT grow a second gate.
        // Unset => denied, same as CREATE EXTERNAL TABLE.
        assert!(!crate::services::ddl::external_location_allowed(
            "file:///tmp/whatever/a.parquet"
        ));
        assert!(!crate::services::ddl::external_location_allowed(
            "../escape.parquet"
        ));
        assert!(!crate::services::ddl::external_location_allowed(""));
    }
}
