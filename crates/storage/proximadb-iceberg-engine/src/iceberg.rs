//! # iceberg — spec-shaped Iceberg v2 manifests, manifest lists, and table metadata
//!
//! [`crate::manifest::ManifestCommitter`] / [`crate::metadata::MetadataCommitter`] own the
//! *atomicity* of publishing a snapshot (versioned create-only CAS + generation fence) but
//! treat the published bytes as opaque. Historically the bytes were a newline-delimited list
//! of data-file keys — readable only by ProximaDB.
//!
//! This module supplies the **content**: the Apache Iceberg v2 on-disk artifacts so that
//! external engines (Spark, Trino, DuckDB, PyIceberg) can read ProximaDB-published tables:
//!
//! - **manifest** (Avro): one `manifest_entry` per data file (status + `data_file` struct).
//! - **manifest list** (Avro): one `manifest_file` per manifest in a snapshot.
//! - **table metadata** (JSON): format-version 2 (or 3) `TableMetadata` — schema, partition
//!   spec (unpartitioned here), the current snapshot chained to its parent, the
//!   `schema.name-mapping.default` property, `current-snapshot-id`, and the `main` ref.
//!
//! ADR-097 D3: this module is the single Iceberg writer. Paths it records are absolute
//! URIs when the bridge knows its store URL (`IcebergObjectStoreBridge::from_url`), which
//! is what external readers require; `tests/iceberg_interop_e2e.rs` holds the
//! PyIceberg / DuckDB read-back gate.
//!
//! The Avro schemas carry Iceberg `field-id`s on every field (the spec's identity contract).
//! Tables are written **unpartitioned** for the first cut (partition spec id 0, empty struct);
//! partitioned layout is a follow-up. The committers still own atomicity — this module only
//! shapes bytes and is fully round-trip tested.

use apache_avro::{Reader, Schema, Writer};
use proximadb_kernel::error::StorageError;
use serde::{Deserialize, Serialize};

fn ser_err(context: &str, e: impl std::fmt::Display) -> StorageError {
    StorageError::Serialization(format!("iceberg: {context}: {e}"))
}

// --- Iceberg v2 manifest Avro schema (unpartitioned) -----------------------
// field-ids per the Iceberg spec (data_file: 134/100/101/102/103/104; entry: 0/1/3/4/2).
const MANIFEST_SCHEMA_JSON: &str = r#"
{
  "type": "record",
  "name": "manifest_entry",
  "fields": [
    {"name": "status", "type": "int", "field-id": 0},
    {"name": "snapshot_id", "type": ["null", "long"], "default": null, "field-id": 1},
    {"name": "sequence_number", "type": ["null", "long"], "default": null, "field-id": 3},
    {"name": "file_sequence_number", "type": ["null", "long"], "default": null, "field-id": 4},
    {"name": "data_file", "field-id": 2, "type": {
      "type": "record",
      "name": "r2",
      "fields": [
        {"name": "content", "type": "int", "field-id": 134},
        {"name": "file_path", "type": "string", "field-id": 100},
        {"name": "file_format", "type": "string", "field-id": 101},
        {"name": "partition", "field-id": 102, "type": {
          "type": "record", "name": "r102", "fields": []
        }},
        {"name": "record_count", "type": "long", "field-id": 103},
        {"name": "file_size_in_bytes", "type": "long", "field-id": 104},
        {"name": "column_sizes", "default": null, "field-id": 108, "type": ["null", {
          "type": "array", "logicalType": "map", "items": {"type": "record", "name": "k117_v118",
            "fields": [{"name": "key", "type": "int", "field-id": 117},
                       {"name": "value", "type": "long", "field-id": 118}]}}]},
        {"name": "value_counts", "default": null, "field-id": 109, "type": ["null", {
          "type": "array", "logicalType": "map", "items": {"type": "record", "name": "k119_v120",
            "fields": [{"name": "key", "type": "int", "field-id": 119},
                       {"name": "value", "type": "long", "field-id": 120}]}}]},
        {"name": "null_value_counts", "default": null, "field-id": 110, "type": ["null", {
          "type": "array", "logicalType": "map", "items": {"type": "record", "name": "k121_v122",
            "fields": [{"name": "key", "type": "int", "field-id": 121},
                       {"name": "value", "type": "long", "field-id": 122}]}}]},
        {"name": "lower_bounds", "default": null, "field-id": 125, "type": ["null", {
          "type": "array", "logicalType": "map", "items": {"type": "record", "name": "k126_v127",
            "fields": [{"name": "key", "type": "int", "field-id": 126},
                       {"name": "value", "type": "bytes", "field-id": 127}]}}]},
        {"name": "upper_bounds", "default": null, "field-id": 128, "type": ["null", {
          "type": "array", "logicalType": "map", "items": {"type": "record", "name": "k129_v130",
            "fields": [{"name": "key", "type": "int", "field-id": 129},
                       {"name": "value", "type": "bytes", "field-id": 130}]}}]}
      ]
    }}
  ]
}
"#;

// --- Iceberg v2 manifest-list Avro schema ----------------------------------
const MANIFEST_LIST_SCHEMA_JSON: &str = r#"
{
  "type": "record",
  "name": "manifest_file",
  "fields": [
    {"name": "manifest_path", "type": "string", "field-id": 500},
    {"name": "manifest_length", "type": "long", "field-id": 501},
    {"name": "partition_spec_id", "type": "int", "field-id": 502},
    {"name": "content", "type": "int", "field-id": 517},
    {"name": "sequence_number", "type": "long", "field-id": 515},
    {"name": "min_sequence_number", "type": "long", "field-id": 516},
    {"name": "added_snapshot_id", "type": "long", "field-id": 503},
    {"name": "added_files_count", "type": "int", "field-id": 504},
    {"name": "existing_files_count", "type": "int", "field-id": 505},
    {"name": "deleted_files_count", "type": "int", "field-id": 506},
    {"name": "added_rows_count", "type": "long", "field-id": 512},
    {"name": "existing_rows_count", "type": "long", "field-id": 513},
    {"name": "deleted_rows_count", "type": "long", "field-id": 514},
    {"name": "first_row_id", "type": ["null", "long"], "default": null, "field-id": 520}
  ]
}
"#;

// --- Avro container header: keep Iceberg field ids --------------------------
//
// apache-avro 0.16 re-serializes the schema into the container header's
// `avro.schema` and drops custom field attributes — Iceberg's `field-id` — on the
// way. Iceberg readers (PyIceberg, Java) resolve manifest fields by those ids and
// reject a manifest without them ("missing field-id"). So after writing, the
// header is re-encoded with the schema JSON verbatim. Data blocks follow the
// header's sync marker and are copied byte-for-byte.

const AVRO_MAGIC: &[u8; 4] = b"Obj\x01";

fn read_avro_long(buf: &[u8], pos: &mut usize) -> Result<i64, StorageError> {
    let mut n: u64 = 0;
    let mut shift = 0;
    loop {
        let byte = *buf
            .get(*pos)
            .ok_or_else(|| ser_err("avro header", "truncated varint"))?;
        *pos += 1;
        n |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 63 {
            return Err(ser_err("avro header", "varint overflow"));
        }
    }
    Ok((n >> 1) as i64 ^ -((n & 1) as i64))
}

fn write_avro_long(out: &mut Vec<u8>, v: i64) {
    let mut n = ((v << 1) ^ (v >> 63)) as u64;
    while n >= 0x80 {
        out.push((n as u8) | 0x80);
        n >>= 7;
    }
    out.push(n as u8);
}

fn read_avro_bytes<'a>(buf: &'a [u8], pos: &mut usize) -> Result<&'a [u8], StorageError> {
    let len = usize::try_from(read_avro_long(buf, pos)?)
        .map_err(|_| ser_err("avro header", "negative length"))?;
    let end = pos
        .checked_add(len)
        .filter(|end| *end <= buf.len())
        .ok_or_else(|| ser_err("avro header", "truncated entry"))?;
    let out = &buf[*pos..end];
    *pos = end;
    Ok(out)
}

/// Avro container header metadata: `(key, value)` byte pairs, in file order.
type AvroHeaderEntries = Vec<(Vec<u8>, Vec<u8>)>;

/// The header metadata entries of an Avro object container file, in order, and
/// the offset where the sync marker begins.
fn avro_header_entries(buf: &[u8]) -> Result<(AvroHeaderEntries, usize), StorageError> {
    if buf.len() < 4 || &buf[..4] != AVRO_MAGIC {
        return Err(ser_err("avro header", "not an Avro object container"));
    }
    let mut pos = 4;
    let mut entries = Vec::new();
    loop {
        let mut count = read_avro_long(buf, &mut pos)?;
        if count == 0 {
            break;
        }
        if count < 0 {
            count = -count;
            read_avro_long(buf, &mut pos)?; // block size in bytes — unused
        }
        for _ in 0..count {
            let key = read_avro_bytes(buf, &mut pos)?.to_vec();
            let value = read_avro_bytes(buf, &mut pos)?.to_vec();
            entries.push((key, value));
        }
    }
    Ok((entries, pos))
}

/// Re-encode `container`'s header with `avro.schema` set to `schema_json` verbatim.
///
/// apache-avro <= 0.16 wrote no header at all when nothing was appended (an
/// empty manifest); since 0.21 `into_inner()` writes the header first, so this
/// branch is defensive. An empty `container` gets a fresh header — `avro.schema`,
/// `avro.codec` = `null`, `user_metadata` and a new sync marker — with no data
/// blocks, which is a valid, empty Avro object container.
fn with_verbatim_schema(
    container: Vec<u8>,
    schema_json: &str,
    user_metadata: &[(String, String)],
) -> Result<Vec<u8>, StorageError> {
    let compact = serde_json::from_str::<serde_json::Value>(schema_json)
        .map_err(|e| ser_err("avro header schema", e))?
        .to_string();
    let empty_body;
    let (entries, body): (AvroHeaderEntries, &[u8]) = if container.is_empty() {
        let mut entries = vec![
            (b"avro.schema".to_vec(), Vec::new()),
            (b"avro.codec".to_vec(), b"null".to_vec()),
        ];
        entries.extend(
            user_metadata
                .iter()
                .map(|(k, v)| (k.as_bytes().to_vec(), v.as_bytes().to_vec())),
        );
        empty_body = uuid::Uuid::new_v4().into_bytes();
        (entries, &empty_body[..])
    } else {
        let (entries, body_start) = avro_header_entries(&container)?;
        (entries, &container[body_start..])
    };
    let mut out = Vec::with_capacity(container.len() + compact.len());
    out.extend_from_slice(AVRO_MAGIC);
    write_avro_long(&mut out, entries.len() as i64);
    for (key, value) in &entries {
        let value: &[u8] = if key == b"avro.schema" {
            compact.as_bytes()
        } else {
            value
        };
        write_avro_long(&mut out, key.len() as i64);
        out.extend_from_slice(key);
        write_avro_long(&mut out, value.len() as i64);
        out.extend_from_slice(value);
    }
    write_avro_long(&mut out, 0);
    out.extend_from_slice(body);
    Ok(out)
}

/// Public description of one data file in a snapshot.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DataFileMeta {
    pub file_path: String,
    pub record_count: i64,
    pub file_size_in_bytes: i64,
    /// Per-column file statistics (Iceberg `column_sizes` / `value_counts` /
    /// `null_value_counts` / `lower_bounds` / `upper_bounds`), keyed by field id.
    /// External engines prune whole data files with these at planning time,
    /// before opening any Parquet footer.
    pub column_stats: Vec<ColumnFileStats>,
}

/// Iceberg per-column metrics for one data file. Bounds use the spec's
/// single-value binary serialization (see [`crate::iceberg::bound_bytes`]
/// callers in the bridge); `None` = unknown (never prunes).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ColumnFileStats {
    pub field_id: i32,
    pub column_size: Option<i64>,
    pub value_count: Option<i64>,
    pub null_value_count: Option<i64>,
    pub lower_bound: Option<Vec<u8>>,
    pub upper_bound: Option<Vec<u8>>,
}

/// Public description of one manifest in a manifest list.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ManifestFileMeta {
    pub manifest_path: String,
    pub manifest_length: i64,
    pub added_files_count: i32,
    pub added_rows_count: i64,
    pub added_snapshot_id: i64,
    /// The data sequence number of the snapshot that added this manifest. Added
    /// entries inherit it (their own `sequence_number` is null), so it must be the
    /// committing snapshot's sequence number — `0` would place every file before
    /// all deletes and break v2 sequence-number semantics.
    pub sequence_number: i64,
    /// v3 row lineage: first row id assigned to this manifest's added rows.
    pub first_row_id: Option<i64>,
}

// --- Avro serde mirrors (field order MUST match the schemas above) ----------
#[derive(Serialize, Deserialize)]
struct AvroPartition {}

/// `bytes` (de)serialization for Avro: serde's default `Vec<u8>` is a sequence,
/// which Avro would encode as `array<int>`, not `bytes`.
mod avro_bytes {
    use serde::{Deserializer, Serializer, de};

    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(v)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        struct V;
        impl<'de> de::Visitor<'de> for V {
            type Value = Vec<u8>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("bytes")
            }
            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Vec<u8>, E> {
                Ok(v.to_vec())
            }
            fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Vec<u8>, E> {
                Ok(v)
            }
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<u8>, A::Error> {
                let mut out = Vec::new();
                while let Some(b) = seq.next_element::<u8>()? {
                    out.push(b);
                }
                Ok(out)
            }
        }
        d.deserialize_bytes(V)
    }
}

#[derive(Serialize, Deserialize)]
struct AvroIntLong {
    key: i32,
    value: i64,
}

#[derive(Serialize, Deserialize)]
struct AvroIntBytes {
    key: i32,
    #[serde(with = "avro_bytes")]
    value: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct AvroDataFile {
    content: i32,
    file_path: String,
    file_format: String,
    partition: AvroPartition,
    record_count: i64,
    file_size_in_bytes: i64,
    #[serde(default)]
    column_sizes: Option<Vec<AvroIntLong>>,
    #[serde(default)]
    value_counts: Option<Vec<AvroIntLong>>,
    #[serde(default)]
    null_value_counts: Option<Vec<AvroIntLong>>,
    #[serde(default)]
    lower_bounds: Option<Vec<AvroIntBytes>>,
    #[serde(default)]
    upper_bounds: Option<Vec<AvroIntBytes>>,
}

fn long_map(
    stats: &[ColumnFileStats],
    pick: impl Fn(&ColumnFileStats) -> Option<i64>,
) -> Option<Vec<AvroIntLong>> {
    let v: Vec<AvroIntLong> = stats
        .iter()
        .filter_map(|c| {
            pick(c).map(|value| AvroIntLong {
                key: c.field_id,
                value,
            })
        })
        .collect();
    (!v.is_empty()).then_some(v)
}

fn bytes_map(
    stats: &[ColumnFileStats],
    pick: impl Fn(&ColumnFileStats) -> Option<&Vec<u8>>,
) -> Option<Vec<AvroIntBytes>> {
    let v: Vec<AvroIntBytes> = stats
        .iter()
        .filter_map(|c| {
            pick(c).map(|value| AvroIntBytes {
                key: c.field_id,
                value: value.clone(),
            })
        })
        .collect();
    (!v.is_empty()).then_some(v)
}

fn stats_from_avro(f: &AvroDataFile) -> Vec<ColumnFileStats> {
    use std::collections::BTreeMap;
    fn get(out: &mut BTreeMap<i32, ColumnFileStats>, id: i32) -> &mut ColumnFileStats {
        out.entry(id).or_insert_with(|| ColumnFileStats {
            field_id: id,
            ..Default::default()
        })
    }
    let mut out: BTreeMap<i32, ColumnFileStats> = BTreeMap::new();
    for kv in f.column_sizes.iter().flatten() {
        get(&mut out, kv.key).column_size = Some(kv.value);
    }
    for kv in f.value_counts.iter().flatten() {
        get(&mut out, kv.key).value_count = Some(kv.value);
    }
    for kv in f.null_value_counts.iter().flatten() {
        get(&mut out, kv.key).null_value_count = Some(kv.value);
    }
    for kv in f.lower_bounds.iter().flatten() {
        get(&mut out, kv.key).lower_bound = Some(kv.value.clone());
    }
    for kv in f.upper_bounds.iter().flatten() {
        get(&mut out, kv.key).upper_bound = Some(kv.value.clone());
    }
    out.into_values().collect()
}

#[derive(Serialize, Deserialize)]
struct AvroManifestEntry {
    status: i32,
    snapshot_id: Option<i64>,
    sequence_number: Option<i64>,
    file_sequence_number: Option<i64>,
    data_file: AvroDataFile,
}

#[derive(Serialize, Deserialize)]
struct AvroManifestFile {
    manifest_path: String,
    manifest_length: i64,
    partition_spec_id: i32,
    content: i32,
    sequence_number: i64,
    min_sequence_number: i64,
    added_snapshot_id: i64,
    added_files_count: i32,
    existing_files_count: i32,
    deleted_files_count: i32,
    added_rows_count: i64,
    existing_rows_count: i64,
    deleted_rows_count: i64,
    #[serde(default)]
    first_row_id: Option<i64>,
}

/// Avro file-level metadata Iceberg writers put in every manifest header. Java and
/// PyIceberg readers take the manifest's schema, partition spec and format version
/// from these keys; without them a reader must guess from the table metadata.
pub struct ManifestHeader<'a> {
    /// The table schema as Iceberg schema JSON (`{"type":"struct",...}`).
    pub schema_json: &'a str,
    pub format_version: FormatVersion,
}

/// Serialize a manifest (one `manifest_entry` per data file) as Iceberg v2 Avro.
/// `status` is ADDED (1), `content` is DATA (0), format is PARQUET, unpartitioned.
pub fn write_manifest(
    files: &[DataFileMeta],
    snapshot_id: i64,
    header: Option<&ManifestHeader<'_>>,
) -> Result<Vec<u8>, StorageError> {
    let schema =
        Schema::parse_str(MANIFEST_SCHEMA_JSON).map_err(|e| ser_err("parse manifest schema", e))?;
    let mut writer = Writer::new(&schema, Vec::new());
    let user_metadata: Vec<(String, String)> = header
        .map(|h| {
            vec![
                ("schema".to_string(), h.schema_json.to_string()),
                ("schema-id".to_string(), "0".to_string()),
                ("partition-spec".to_string(), "[]".to_string()),
                ("partition-spec-id".to_string(), "0".to_string()),
                (
                    "format-version".to_string(),
                    h.format_version.as_i32().to_string(),
                ),
                ("content".to_string(), "data".to_string()),
            ]
        })
        .unwrap_or_default();
    for (k, v) in &user_metadata {
        writer
            .add_user_metadata(k.clone(), v)
            .map_err(|e| ser_err("manifest header metadata", e))?;
    }
    for f in files {
        let entry = AvroManifestEntry {
            status: 1, // ADDED
            snapshot_id: Some(snapshot_id),
            sequence_number: None,
            file_sequence_number: None,
            data_file: AvroDataFile {
                content: 0, // DATA
                file_path: f.file_path.clone(),
                file_format: "PARQUET".to_string(),
                partition: AvroPartition {},
                record_count: f.record_count,
                file_size_in_bytes: f.file_size_in_bytes,
                column_sizes: long_map(&f.column_stats, |c| c.column_size),
                value_counts: long_map(&f.column_stats, |c| c.value_count),
                null_value_counts: long_map(&f.column_stats, |c| c.null_value_count),
                lower_bounds: bytes_map(&f.column_stats, |c| c.lower_bound.as_ref()),
                upper_bounds: bytes_map(&f.column_stats, |c| c.upper_bound.as_ref()),
            },
        };
        writer
            .append_ser(&entry)
            .map_err(|e| ser_err("append manifest entry", e))?;
    }
    let container = writer
        .into_inner()
        .map_err(|e| ser_err("finish manifest", e))?;
    with_verbatim_schema(container, MANIFEST_SCHEMA_JSON, &user_metadata)
}

/// Read an Iceberg v2 manifest Avro back into the data-file list.
pub fn read_manifest(bytes: &[u8]) -> Result<Vec<DataFileMeta>, StorageError> {
    let schema =
        Schema::parse_str(MANIFEST_SCHEMA_JSON).map_err(|e| ser_err("parse manifest schema", e))?;
    let reader =
        Reader::with_schema(&schema, bytes).map_err(|e| ser_err("open manifest reader", e))?;
    let mut out = Vec::new();
    for value in reader {
        let value = value.map_err(|e| ser_err("read manifest value", e))?;
        let entry: AvroManifestEntry =
            apache_avro::from_value(&value).map_err(|e| ser_err("decode manifest entry", e))?;
        // status 2 == DELETED; skip deleted entries when surfacing live data files.
        if entry.status == 2 {
            continue;
        }
        let column_stats = stats_from_avro(&entry.data_file);
        out.push(DataFileMeta {
            file_path: entry.data_file.file_path,
            record_count: entry.data_file.record_count,
            file_size_in_bytes: entry.data_file.file_size_in_bytes,
            column_stats,
        });
    }
    Ok(out)
}

/// Serialize a manifest list (one `manifest_file` per manifest) as Iceberg v2 Avro.
pub fn write_manifest_list(manifests: &[ManifestFileMeta]) -> Result<Vec<u8>, StorageError> {
    let schema = Schema::parse_str(MANIFEST_LIST_SCHEMA_JSON)
        .map_err(|e| ser_err("parse manifest-list schema", e))?;
    let mut writer = Writer::new(&schema, Vec::new());
    for m in manifests {
        let mf = AvroManifestFile {
            manifest_path: m.manifest_path.clone(),
            manifest_length: m.manifest_length,
            partition_spec_id: 0,
            content: 0, // DATA
            sequence_number: m.sequence_number,
            min_sequence_number: m.sequence_number,
            added_snapshot_id: m.added_snapshot_id,
            added_files_count: m.added_files_count,
            existing_files_count: 0,
            deleted_files_count: 0,
            added_rows_count: m.added_rows_count,
            existing_rows_count: 0,
            deleted_rows_count: 0,
            first_row_id: m.first_row_id,
        };
        writer
            .append_ser(&mf)
            .map_err(|e| ser_err("append manifest_file", e))?;
    }
    let container = writer
        .into_inner()
        .map_err(|e| ser_err("finish manifest list", e))?;
    with_verbatim_schema(container, MANIFEST_LIST_SCHEMA_JSON, &[])
}

/// Read an Iceberg v2 manifest list back into its entries.
pub fn read_manifest_list(bytes: &[u8]) -> Result<Vec<ManifestFileMeta>, StorageError> {
    let schema = Schema::parse_str(MANIFEST_LIST_SCHEMA_JSON)
        .map_err(|e| ser_err("parse manifest-list schema", e))?;
    let reader =
        Reader::with_schema(&schema, bytes).map_err(|e| ser_err("open manifest-list reader", e))?;
    let mut out = Vec::new();
    for value in reader {
        let value = value.map_err(|e| ser_err("read manifest-list value", e))?;
        let mf: AvroManifestFile =
            apache_avro::from_value(&value).map_err(|e| ser_err("decode manifest_file", e))?;
        out.push(ManifestFileMeta {
            manifest_path: mf.manifest_path,
            manifest_length: mf.manifest_length,
            added_files_count: mf.added_files_count,
            added_rows_count: mf.added_rows_count,
            added_snapshot_id: mf.added_snapshot_id,
            sequence_number: mf.sequence_number,
            first_row_id: mf.first_row_id,
        });
    }
    Ok(out)
}

/// One Iceberg schema field (subset of the spec needed to publish flat relational tables).
#[derive(Debug, Clone)]
pub struct IcebergField {
    pub id: i32,
    pub name: String,
    /// Iceberg primitive type name, e.g. "long", "string", "double", "boolean", "date".
    pub type_name: String,
    pub required: bool,
}

/// Iceberg table format version.
///
/// **v3 is ProximaDB's target and default** — its headline features map onto ProximaDB's
/// differentiators (row lineage ↔ CDC, deletion vectors ↔ dedup, `variant` ↔ documents,
/// `timestamp_ns` ↔ our nanosecond-native format), so building toward v3 directly avoids
/// double-spending on v2-only mechanisms (positional/equality deletes, microsecond-only
/// timestamps) that v3 supersedes. v2 stays reachable for the subset of external readers
/// that are still v2-only (capability negotiation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FormatVersion {
    /// format-version 2 — reachable for legacy external readers.
    V2,
    /// format-version 3 — ProximaDB's target (lineage / DV / variant / ns-timestamps).
    #[default]
    V3,
}

impl FormatVersion {
    pub fn as_i32(self) -> i32 {
        match self {
            FormatVersion::V2 => 2,
            FormatVersion::V3 => 3,
        }
    }
}

/// Iceberg schema JSON (`schema-id` 0) for `fields`. Shared by the table metadata
/// and the manifest header so both describe the same schema.
pub fn schema_json(fields: &[IcebergField]) -> serde_json::Value {
    let schema_fields: Vec<serde_json::Value> = fields
        .iter()
        .map(|f| {
            serde_json::json!({
                "id": f.id,
                "name": f.name,
                "required": f.required,
                "type": f.type_name,
            })
        })
        .collect();
    serde_json::json!({
        "type": "struct",
        "schema-id": 0,
        "fields": schema_fields,
    })
}

/// `schema.name-mapping.default`: maps Iceberg field ids to Parquet column names.
/// ProximaDB's Parquet files carry no embedded field ids, so without this mapping
/// PyIceberg refuses to read them and other engines fall back to guessing.
fn name_mapping(fields: &[IcebergField]) -> String {
    let entries: Vec<serde_json::Value> = fields
        .iter()
        .map(|f| serde_json::json!({"field-id": f.id, "names": [f.name]}))
        .collect();
    serde_json::Value::Array(entries).to_string()
}

/// What a new snapshot inherits from the table's previous `metadata.json`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Lineage {
    /// Data sequence number for the new snapshot (previous `last-sequence-number` + 1).
    pub sequence_number: i64,
    /// The previous current snapshot, if any.
    pub parent_snapshot_id: Option<i64>,
    /// v3 row lineage: first row id for the new snapshot's added rows.
    pub first_row_id: i64,
    /// The previous metadata log, extended with the previous metadata file.
    metadata_log: Vec<serde_json::Value>,
}

impl Lineage {
    /// Derive lineage from the previous metadata document (`None` for the first commit).
    /// `parent_location` is the previous document's absolute location, recorded in the
    /// new `metadata-log`.
    pub fn from_parent(
        parent_json: Option<&str>,
        parent_location: Option<&str>,
    ) -> Result<Self, StorageError> {
        let Some(json) = parent_json else {
            return Ok(Self {
                sequence_number: 1,
                ..Default::default()
            });
        };
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| ser_err("parse parent metadata", e))?;
        let last_seq = v
            .get("last-sequence-number")
            .and_then(|x| x.as_i64())
            .unwrap_or(0);
        let parent_snapshot_id = v
            .get("current-snapshot-id")
            .and_then(|x| x.as_i64())
            .filter(|id| *id >= 0);
        let first_row_id = v.get("next-row-id").and_then(|x| x.as_i64()).unwrap_or(0);
        let mut metadata_log = v
            .get("metadata-log")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        if let Some(loc) = parent_location {
            metadata_log.push(serde_json::json!({
                "timestamp-ms": v.get("last-updated-ms").and_then(|x| x.as_i64()).unwrap_or(0),
                "metadata-file": loc,
            }));
        }
        Ok(Self {
            sequence_number: last_seq + 1,
            parent_snapshot_id,
            first_row_id,
            metadata_log,
        })
    }
}

/// One snapshot commit: the inputs to [`build_table_metadata`].
#[derive(Debug, Clone)]
pub struct SnapshotCommit<'a> {
    pub format_version: FormatVersion,
    pub table_uuid: &'a str,
    /// Table root location (absolute URI when the writer knows its store URL).
    pub location: &'a str,
    pub fields: &'a [IcebergField],
    pub snapshot_id: i64,
    pub manifest_list_location: &'a str,
    /// Commit time (passed in — this crate avoids wall-clock for determinism).
    pub timestamp_ms: i64,
    pub added_files: i64,
    pub added_records: i64,
}

/// Build the `TableMetadata` JSON for `commit`, chained onto `lineage`.
///
/// The snapshot is the table's complete current file set, so a commit with a parent
/// is an `overwrite` and the parent snapshot is not carried forward: a rewrite replaces
/// the parent's files, which are then collected, and an append re-lists the files it
/// carries (as added, so `added-files`/`added-records` count them). The parent stays referenced through `parent-snapshot-id` and the
/// previous metadata file through `metadata-log`, exactly as after snapshot expiry.
/// The table is unpartitioned (spec id 0). v3 adds the row-lineage fields
/// (`next-row-id`, `first-row-id`, `added-rows`).
pub fn build_table_metadata(
    commit: &SnapshotCommit<'_>,
    lineage: &Lineage,
) -> Result<String, StorageError> {
    let fields = commit.fields;
    let last_column_id = fields.iter().map(|f| f.id).max().unwrap_or(0);
    let operation = if lineage.parent_snapshot_id.is_some() {
        "overwrite"
    } else {
        "append"
    };
    let mut snapshot = serde_json::json!({
        "snapshot-id": commit.snapshot_id,
        "sequence-number": lineage.sequence_number,
        "timestamp-ms": commit.timestamp_ms,
        "summary": {
            "operation": operation,
            "added-data-files": commit.added_files.to_string(),
            "added-records": commit.added_records.to_string(),
            "total-data-files": commit.added_files.to_string(),
            "total-records": commit.added_records.to_string(),
        },
        "manifest-list": commit.manifest_list_location,
        "schema-id": 0,
    });
    if let Some(parent) = lineage.parent_snapshot_id {
        snapshot["parent-snapshot-id"] = serde_json::json!(parent);
    }
    let mut metadata = serde_json::json!({
        "format-version": commit.format_version.as_i32(),
        "table-uuid": commit.table_uuid,
        "location": commit.location,
        "last-sequence-number": lineage.sequence_number,
        "last-updated-ms": commit.timestamp_ms,
        "last-column-id": last_column_id,
        "current-schema-id": 0,
        "schemas": [schema_json(fields)],
        "default-spec-id": 0,
        "partition-specs": [{"spec-id": 0, "fields": []}],
        "last-partition-id": 999,
        "default-sort-order-id": 0,
        "sort-orders": [{"order-id": 0, "fields": []}],
        "properties": {
            "schema.name-mapping.default": name_mapping(fields),
            "write.format.default": "parquet",
        },
        "current-snapshot-id": commit.snapshot_id,
        "snapshots": [],
        "snapshot-log": [{"snapshot-id": commit.snapshot_id, "timestamp-ms": commit.timestamp_ms}],
        "metadata-log": lineage.metadata_log,
        "refs": {"main": {"snapshot-id": commit.snapshot_id, "type": "branch"}},
    });
    if commit.format_version == FormatVersion::V3 {
        snapshot["first-row-id"] = serde_json::json!(lineage.first_row_id);
        snapshot["added-rows"] = serde_json::json!(commit.added_records);
        metadata["next-row-id"] = serde_json::json!(lineage.first_row_id + commit.added_records);
    }
    metadata["snapshots"] = serde_json::json!([snapshot]);
    serde_json::to_string_pretty(&metadata).map_err(|e| ser_err("serialize table metadata", e))
}

/// Extract the current snapshot's manifest-list location from a `TableMetadata` JSON.
pub fn current_manifest_list(metadata_json: &str) -> Result<Option<String>, StorageError> {
    let v: serde_json::Value =
        serde_json::from_str(metadata_json).map_err(|e| ser_err("parse table metadata", e))?;
    let current = v.get("current-snapshot-id").and_then(|x| x.as_i64());
    let Some(current) = current else {
        return Ok(None);
    };
    let snapshots = v.get("snapshots").and_then(|x| x.as_array());
    let Some(snapshots) = snapshots else {
        return Ok(None);
    };
    Ok(snapshots
        .iter()
        .find(|s| s.get("snapshot-id").and_then(|x| x.as_i64()) == Some(current))
        .and_then(|s| s.get("manifest-list").and_then(|x| x.as_str()))
        .map(String::from))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files() -> Vec<DataFileMeta> {
        vec![
            DataFileMeta {
                file_path: "data/t/ns/tbl/data/part-0.parquet".into(),
                record_count: 100,
                file_size_in_bytes: 4096,
                ..Default::default()
            },
            DataFileMeta {
                file_path: "data/t/ns/tbl/data/part-1.parquet".into(),
                record_count: 50,
                file_size_in_bytes: 2048,
                ..Default::default()
            },
        ]
    }

    #[test]
    fn manifest_round_trips_column_metrics() {
        let mut f = files();
        f[0].column_stats = vec![
            ColumnFileStats {
                field_id: 1,
                column_size: Some(512),
                value_count: Some(100),
                null_value_count: Some(3),
                lower_bound: Some(7i64.to_le_bytes().to_vec()),
                upper_bound: Some(900i64.to_le_bytes().to_vec()),
            },
            ColumnFileStats {
                field_id: 2,
                column_size: Some(64),
                value_count: Some(100),
                null_value_count: Some(0),
                lower_bound: Some(b"AL".to_vec()),
                upper_bound: None,
            },
        ];
        let bytes = write_manifest(&f, 42, None).unwrap();
        let back = read_manifest(&bytes).unwrap();
        assert_eq!(
            back, f,
            "metrics survive the Avro round trip; absent maps stay absent"
        );
    }

    #[test]
    fn manifest_round_trips() {
        let bytes = write_manifest(&files(), 12345, None).unwrap();
        let back = read_manifest(&bytes).unwrap();
        assert_eq!(back, files());
    }

    #[test]
    fn manifest_list_round_trips() {
        let manifests = vec![ManifestFileMeta {
            manifest_path: "data/t/ns/tbl/metadata/abc-m0.avro".into(),
            manifest_length: 1234,
            added_files_count: 2,
            added_rows_count: 150,
            added_snapshot_id: 12345,
            sequence_number: 3,
            first_row_id: Some(10),
        }];
        let bytes = write_manifest_list(&manifests).unwrap();
        let back = read_manifest_list(&bytes).unwrap();
        assert_eq!(back, manifests);
    }

    fn two_fields() -> Vec<IcebergField> {
        vec![
            IcebergField {
                id: 1,
                name: "id".into(),
                type_name: "long".into(),
                required: true,
            },
            IcebergField {
                id: 2,
                name: "name".into(),
                type_name: "string".into(),
                required: false,
            },
        ]
    }

    fn commit<'a>(
        fields: &'a [IcebergField],
        version: FormatVersion,
        snapshot_id: i64,
        list: &'a str,
    ) -> SnapshotCommit<'a> {
        SnapshotCommit {
            format_version: version,
            table_uuid: "uuid-1",
            location: "file:///wh/t/ns/tbl",
            fields,
            snapshot_id,
            manifest_list_location: list,
            timestamp_ms: 1_700_000_000_000 + snapshot_id,
            added_files: 2,
            added_records: 150,
        }
    }

    #[test]
    fn table_metadata_points_at_manifest_list() {
        let fields = two_fields();
        let list = "file:///wh/t/ns/tbl/metadata/snap-777-list.avro";
        let json = build_table_metadata(
            &commit(&fields, FormatVersion::V2, 777, list),
            &Lineage::from_parent(None, None).unwrap(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["format-version"], 2);
        assert_eq!(v["last-column-id"], 2);
        assert_eq!(v["last-sequence-number"], 1);
        assert_eq!(v["snapshots"][0]["summary"]["operation"], "append");
        assert!(v["snapshots"][0].get("parent-snapshot-id").is_none());
        assert!(v.get("next-row-id").is_none(), "v2 has no row lineage");
        assert_eq!(current_manifest_list(&json).unwrap().as_deref(), Some(list));

        // External readers resolve Parquet columns through the name mapping.
        let mapping: serde_json::Value = serde_json::from_str(
            v["properties"]["schema.name-mapping.default"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            mapping,
            serde_json::json!([
                {"field-id": 1, "names": ["id"]},
                {"field-id": 2, "names": ["name"]}
            ])
        );
    }

    #[test]
    fn successor_metadata_chains_onto_parent() {
        let fields = two_fields();
        let first = build_table_metadata(
            &commit(&fields, FormatVersion::V3, 777, "l0"),
            &Lineage::from_parent(None, None).unwrap(),
        )
        .unwrap();
        let lineage =
            Lineage::from_parent(Some(&first), Some("file:///wh/m/v0.metadata.json")).unwrap();
        assert_eq!(lineage.sequence_number, 2);
        assert_eq!(lineage.parent_snapshot_id, Some(777));
        assert_eq!(lineage.first_row_id, 150, "next-row-id of the parent");

        let second =
            build_table_metadata(&commit(&fields, FormatVersion::V3, 778, "l1"), &lineage).unwrap();
        let v: serde_json::Value = serde_json::from_str(&second).unwrap();
        assert_eq!(v["format-version"], 3);
        assert_eq!(v["last-sequence-number"], 2);
        let snap = &v["snapshots"][0];
        assert_eq!(snap["parent-snapshot-id"], 777);
        assert_eq!(snap["sequence-number"], 2);
        assert_eq!(snap["summary"]["operation"], "overwrite");
        assert_eq!(snap["first-row-id"], 150);
        assert_eq!(v["next-row-id"], 300);
        assert_eq!(
            v["snapshots"].as_array().unwrap().len(),
            1,
            "the in-place-rewritten parent snapshot is not carried forward"
        );
        assert_eq!(
            v["metadata-log"][0]["metadata-file"],
            "file:///wh/m/v0.metadata.json"
        );
    }

    /// Iceberg readers resolve manifest fields by `field-id`; the container header
    /// must carry them (apache-avro alone drops them), and the rewritten header
    /// must still decode.
    #[test]
    fn avro_header_schema_keeps_iceberg_field_ids() {
        for bytes in [
            write_manifest(&files(), 9, None).unwrap(),
            write_manifest_list(&[ManifestFileMeta::default()]).unwrap(),
        ] {
            let (entries, _) = avro_header_entries(&bytes).unwrap();
            let schema = entries
                .iter()
                .find(|(k, _)| k == b"avro.schema")
                .map(|(_, v)| String::from_utf8(v.clone()).unwrap())
                .expect("avro.schema header");
            let v: serde_json::Value = serde_json::from_str(&schema).unwrap();
            let fields = v["fields"].as_array().unwrap();
            assert!(
                fields.iter().all(|f| f.get("field-id").is_some()),
                "every top-level field carries its Iceberg field-id: {schema}"
            );
            assert!(
                Reader::new(&bytes[..]).is_ok(),
                "rewritten container decodes"
            );
        }
        assert_eq!(
            read_manifest(&write_manifest(&files(), 9, None).unwrap()).unwrap(),
            files()
        );
    }

    /// A snapshot with no data files still writes valid, readable (empty) Avro
    /// containers — apache-avro alone emits zero bytes here.
    #[test]
    fn empty_manifest_and_list_are_valid_avro() {
        let schema = schema_json(&two_fields()).to_string();
        let manifest = write_manifest(
            &[],
            1,
            Some(&ManifestHeader {
                schema_json: &schema,
                format_version: FormatVersion::V2,
            }),
        )
        .unwrap();
        assert!(read_manifest(&manifest).unwrap().is_empty());
        let reader = Reader::new(&manifest[..]).unwrap();
        assert_eq!(
            reader.user_metadata().get("format-version").map(|v| &v[..]),
            Some(&b"2"[..])
        );
        let list = write_manifest_list(&[]).unwrap();
        assert!(read_manifest_list(&list).unwrap().is_empty());
    }

    #[test]
    fn avro_long_round_trips() {
        for v in [0i64, 1, -1, 63, -64, 64, 300, -300, i64::MAX, i64::MIN] {
            let mut buf = Vec::new();
            write_avro_long(&mut buf, v);
            let mut pos = 0;
            assert_eq!(read_avro_long(&buf, &mut pos).unwrap(), v);
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn manifest_header_carries_schema_spec_and_format_version() {
        let fields = two_fields();
        let schema = schema_json(&fields).to_string();
        let bytes = write_manifest(
            &files(),
            9,
            Some(&ManifestHeader {
                schema_json: &schema,
                format_version: FormatVersion::V2,
            }),
        )
        .unwrap();
        let reader = Reader::new(&bytes[..]).unwrap();
        let meta = reader.user_metadata();
        assert_eq!(meta.get("format-version").map(|v| &v[..]), Some(&b"2"[..]));
        assert_eq!(meta.get("content").map(|v| &v[..]), Some(&b"data"[..]));
        assert_eq!(meta.get("partition-spec").map(|v| &v[..]), Some(&b"[]"[..]));
        assert_eq!(
            meta.get("schema").map(|v| v.as_slice()),
            Some(schema.as_bytes())
        );
        // The header does not change the entries.
        assert_eq!(read_manifest(&bytes).unwrap(), files());
    }

    /// End-to-end content round-trip: metadata → manifest-list → manifest → data files.
    #[test]
    fn snapshot_content_round_trips_end_to_end() {
        let data = files();
        let manifest_bytes = write_manifest(&data, 777, None).unwrap();
        let manifests = vec![ManifestFileMeta {
            manifest_path: "m0.avro".into(),
            manifest_length: manifest_bytes.len() as i64,
            added_files_count: data.len() as i32,
            added_rows_count: data.iter().map(|f| f.record_count).sum(),
            added_snapshot_id: 777,
            ..Default::default()
        }];
        let list_bytes = write_manifest_list(&manifests).unwrap();
        let fields = [IcebergField {
            id: 1,
            name: "id".into(),
            type_name: "long".into(),
            required: true,
        }];
        let json = build_table_metadata(
            &SnapshotCommit {
                format_version: FormatVersion::V2,
                table_uuid: "uuid-1",
                location: "loc",
                fields: &fields,
                snapshot_id: 777,
                manifest_list_location: "snap-777-list.avro",
                timestamp_ms: 1,
                added_files: data.len() as i64,
                added_records: data.iter().map(|f| f.record_count).sum(),
            },
            &Lineage::from_parent(None, None).unwrap(),
        )
        .unwrap();

        // Resolve back down the chain.
        assert!(current_manifest_list(&json).unwrap().is_some());
        let list = read_manifest_list(&list_bytes).unwrap();
        assert_eq!(list.len(), 1);
        let recovered = read_manifest(&manifest_bytes).unwrap();
        assert_eq!(recovered, data);
    }

    /// Mixed-read gate for the apache-avro 0.16 -> 0.21 upgrade (mandate #8):
    /// manifests already in object storage were written by 0.16, so the
    /// current reader must decode those exact bytes. The fixtures were produced
    /// by `write_manifest` / `write_manifest_list` built against apache-avro
    /// 0.16.0 and are pinned here unchanged.
    #[test]
    fn reads_manifests_written_by_apache_avro_0_16() {
        let mut expected = files();
        expected[0].column_stats = vec![ColumnFileStats {
            field_id: 1,
            column_size: Some(512),
            value_count: Some(100),
            null_value_count: Some(3),
            lower_bound: Some(7i64.to_le_bytes().to_vec()),
            upper_bound: Some(900i64.to_le_bytes().to_vec()),
        }];
        let manifest = include_bytes!("../testdata/manifest_avro016.avro");
        assert_eq!(read_manifest(manifest).unwrap(), expected);

        let list = include_bytes!("../testdata/manifest_list_avro016.avro");
        assert_eq!(
            read_manifest_list(list).unwrap(),
            vec![ManifestFileMeta {
                manifest_path: "data/t/ns/tbl/metadata/m-0.avro".into(),
                manifest_length: 2254,
                added_files_count: 2,
                added_rows_count: 150,
                added_snapshot_id: 42,
                sequence_number: 9,
                first_row_id: Some(1000),
            }]
        );
    }
}
