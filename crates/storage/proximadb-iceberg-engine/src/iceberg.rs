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
//! - **table metadata** (JSON): format-version 2 `TableMetadata` — schema, partition spec
//!   (unpartitioned here), snapshots (each pointing at its manifest list), `current-snapshot-id`,
//!   and the `main` ref.
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
    {"name": "deleted_rows_count", "type": "long", "field-id": 514}
  ]
}
"#;

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
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestFileMeta {
    pub manifest_path: String,
    pub manifest_length: i64,
    pub added_files_count: i32,
    pub added_rows_count: i64,
    pub added_snapshot_id: i64,
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
}

/// Serialize a manifest (one `manifest_entry` per data file) as Iceberg v2 Avro.
/// `status` is ADDED (1), `content` is DATA (0), format is PARQUET, unpartitioned.
pub fn write_manifest(files: &[DataFileMeta], snapshot_id: i64) -> Result<Vec<u8>, StorageError> {
    let schema =
        Schema::parse_str(MANIFEST_SCHEMA_JSON).map_err(|e| ser_err("parse manifest schema", e))?;
    let mut writer = Writer::new(&schema, Vec::new());
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
    writer
        .into_inner()
        .map_err(|e| ser_err("finish manifest", e))
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
            sequence_number: 0,
            min_sequence_number: 0,
            added_snapshot_id: m.added_snapshot_id,
            added_files_count: m.added_files_count,
            existing_files_count: 0,
            deleted_files_count: 0,
            added_rows_count: m.added_rows_count,
            existing_rows_count: 0,
            deleted_rows_count: 0,
        };
        writer
            .append_ser(&mf)
            .map_err(|e| ser_err("append manifest_file", e))?;
    }
    writer
        .into_inner()
        .map_err(|e| ser_err("finish manifest list", e))
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
    fn as_i32(self) -> i32 {
        match self {
            FormatVersion::V2 => 2,
            FormatVersion::V3 => 3,
        }
    }
}

/// Build a `TableMetadata` JSON document for a single snapshot at the given format version.
///
/// `snapshot_id` identifies the snapshot; `manifest_list_location` is the object key of the
/// snapshot's manifest list; `timestamp_ms` is the commit time (passed in — this crate avoids
/// wall-clock for determinism/testability). The table is unpartitioned (spec id 0). Pass
/// [`FormatVersion::V2`] for maximum external-reader compatibility, [`FormatVersion::V3`] to
/// opt into the lineage/DV/variant-aligned format.
#[allow(clippy::too_many_arguments)]
pub fn build_table_metadata(
    format_version: FormatVersion,
    table_uuid: &str,
    location: &str,
    fields: &[IcebergField],
    snapshot_id: i64,
    sequence_number: i64,
    manifest_list_location: &str,
    timestamp_ms: i64,
    added_files: i64,
    added_records: i64,
) -> Result<String, StorageError> {
    let last_column_id = fields.iter().map(|f| f.id).max().unwrap_or(0);
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
    let metadata = serde_json::json!({
        "format-version": format_version.as_i32(),
        "table-uuid": table_uuid,
        "location": location,
        "last-sequence-number": sequence_number,
        "last-updated-ms": timestamp_ms,
        "last-column-id": last_column_id,
        "current-schema-id": 0,
        "schemas": [{
            "type": "struct",
            "schema-id": 0,
            "fields": schema_fields,
        }],
        "default-spec-id": 0,
        "partition-specs": [{"spec-id": 0, "fields": []}],
        "last-partition-id": 999,
        "default-sort-order-id": 0,
        "sort-orders": [{"order-id": 0, "fields": []}],
        "properties": {},
        "current-snapshot-id": snapshot_id,
        "snapshots": [{
            "snapshot-id": snapshot_id,
            "sequence-number": sequence_number,
            "timestamp-ms": timestamp_ms,
            "summary": {
                "operation": "append",
                "added-data-files": added_files.to_string(),
                "added-records": added_records.to_string(),
            },
            "manifest-list": manifest_list_location,
            "schema-id": 0,
        }],
        "snapshot-log": [{"snapshot-id": snapshot_id, "timestamp-ms": timestamp_ms}],
        "metadata-log": [],
        "refs": {"main": {"snapshot-id": snapshot_id, "type": "branch"}},
    });
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
        let bytes = write_manifest(&f, 42).unwrap();
        let back = read_manifest(&bytes).unwrap();
        assert_eq!(
            back, f,
            "metrics survive the Avro round trip; absent maps stay absent"
        );
    }

    #[test]
    fn manifest_round_trips() {
        let bytes = write_manifest(&files(), 12345).unwrap();
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
        }];
        let bytes = write_manifest_list(&manifests).unwrap();
        let back = read_manifest_list(&bytes).unwrap();
        assert_eq!(back, manifests);
    }

    #[test]
    fn table_metadata_points_at_manifest_list() {
        let fields = vec![
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
        ];
        let json = build_table_metadata(
            FormatVersion::V2,
            "uuid-1",
            "data/t/ns/tbl",
            &fields,
            777,
            1,
            "data/t/ns/tbl/metadata/snap-777-list.avro",
            1_700_000_000_000,
            2,
            150,
        )
        .unwrap();
        // Valid JSON, format v2, and resolvable current manifest list.
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["format-version"], 2);
        assert_eq!(v["last-column-id"], 2);
        assert_eq!(
            current_manifest_list(&json).unwrap().as_deref(),
            Some("data/t/ns/tbl/metadata/snap-777-list.avro")
        );

        // v3 is reachable (opt-in) — same structure, format-version flips to 3.
        let v3 = build_table_metadata(
            FormatVersion::V3,
            "uuid-1",
            "data/t/ns/tbl",
            &fields,
            778,
            2,
            "data/t/ns/tbl/metadata/snap-778-list.avro",
            1_700_000_000_001,
            1,
            10,
        )
        .unwrap();
        let v3v: serde_json::Value = serde_json::from_str(&v3).unwrap();
        assert_eq!(v3v["format-version"], 3);
        assert!(current_manifest_list(&v3).unwrap().is_some());
    }

    /// End-to-end content round-trip: metadata → manifest-list → manifest → data files.
    #[test]
    fn snapshot_content_round_trips_end_to_end() {
        let data = files();
        let manifest_bytes = write_manifest(&data, 777).unwrap();
        let manifests = vec![ManifestFileMeta {
            manifest_path: "m0.avro".into(),
            manifest_length: manifest_bytes.len() as i64,
            added_files_count: data.len() as i32,
            added_rows_count: data.iter().map(|f| f.record_count).sum(),
            added_snapshot_id: 777,
        }];
        let list_bytes = write_manifest_list(&manifests).unwrap();
        let json = build_table_metadata(
            FormatVersion::V2,
            "uuid-1",
            "loc",
            &[IcebergField {
                id: 1,
                name: "id".into(),
                type_name: "long".into(),
                required: true,
            }],
            777,
            1,
            "snap-777-list.avro",
            1,
            data.len() as i64,
            data.iter().map(|f| f.record_count).sum(),
        )
        .unwrap();

        // Resolve back down the chain.
        assert!(current_manifest_list(&json).unwrap().is_some());
        let list = read_manifest_list(&list_bytes).unwrap();
        assert_eq!(list.len(), 1);
        let recovered = read_manifest(&manifest_bytes).unwrap();
        assert_eq!(recovered, data);
    }
}
