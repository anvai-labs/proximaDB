//! Delete manifests: the manifest of a v3 snapshot's deletion vectors (P2.4 slice 3).
//!
//! Each entry is one deletion vector: `content` POSITION_DELETES (1), format PUFFIN,
//! `record_count` its cardinality, and the v3 fields that pin it to one blob of one
//! Puffin file and to the data file it deletes from — `referenced_data_file` (143),
//! `content_offset` (144) and `content_size_in_bytes` (145). Entries inherit the
//! manifest's sequence number, so a vector applies to every data file of its own
//! snapshot (position deletes apply to data sequence numbers `<=` their own).
//!
//! Only deletion vectors are read back. A v2 position-delete Parquet file or an
//! equality delete (another writer's) is an error, so a reader fails closed instead
//! of returning rows it was told were deleted.

use apache_avro::{Reader, Schema, Writer};
use proximadb_kernel::error::StorageError;
use serde::{Deserialize, Serialize};

use super::{AvroPartition, ManifestHeader, ser_err, with_verbatim_schema};

const DELETE_MANIFEST_SCHEMA_JSON: &str = r#"
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
        {"name": "referenced_data_file", "type": ["null", "string"], "default": null,
         "field-id": 143},
        {"name": "content_offset", "type": ["null", "long"], "default": null, "field-id": 144},
        {"name": "content_size_in_bytes", "type": ["null", "long"], "default": null,
         "field-id": 145}
      ]
    }}
  ]
}
"#;

/// `data_file.content` of a position delete (a deletion vector is one).
const POSITION_DELETES: i32 = 1;
const PUFFIN: &str = "PUFFIN";

/// One deletion vector in a snapshot: a blob of a Puffin file, deleting positions
/// of one data file.
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteFileMeta {
    /// The Puffin file.
    pub file_path: String,
    pub file_size_in_bytes: i64,
    /// The data file whose rows it deletes, exactly as that file's entry records it.
    pub referenced_data_file: String,
    pub content_offset: i64,
    pub content_size_in_bytes: i64,
    /// The number of deleted positions.
    pub record_count: i64,
}

#[derive(Serialize, Deserialize)]
struct AvroDeleteFile {
    content: i32,
    file_path: String,
    file_format: String,
    partition: AvroPartition,
    record_count: i64,
    file_size_in_bytes: i64,
    referenced_data_file: Option<String>,
    content_offset: Option<i64>,
    content_size_in_bytes: Option<i64>,
}

#[derive(Serialize, Deserialize)]
struct AvroDeleteEntry {
    status: i32,
    snapshot_id: Option<i64>,
    sequence_number: Option<i64>,
    file_sequence_number: Option<i64>,
    data_file: AvroDeleteFile,
}

/// Serialize a delete manifest: one ADDED entry per deletion vector.
pub fn write_delete_manifest(
    vectors: &[DeleteFileMeta],
    snapshot_id: i64,
    header: &ManifestHeader<'_>,
) -> Result<Vec<u8>, StorageError> {
    let schema = Schema::parse_str(DELETE_MANIFEST_SCHEMA_JSON)
        .map_err(|e| ser_err("parse delete manifest schema", e))?;
    let mut writer = Writer::new(&schema, Vec::new());
    let user_metadata = vec![
        ("schema".to_string(), header.schema_json.to_string()),
        ("schema-id".to_string(), "0".to_string()),
        ("partition-spec".to_string(), "[]".to_string()),
        ("partition-spec-id".to_string(), "0".to_string()),
        (
            "format-version".to_string(),
            header.format_version.as_i32().to_string(),
        ),
        ("content".to_string(), "deletes".to_string()),
    ];
    for (k, v) in &user_metadata {
        writer
            .add_user_metadata(k.clone(), v)
            .map_err(|e| ser_err("delete manifest header metadata", e))?;
    }
    for v in vectors {
        let entry = AvroDeleteEntry {
            status: 1, // ADDED
            snapshot_id: Some(snapshot_id),
            sequence_number: None,
            file_sequence_number: None,
            data_file: AvroDeleteFile {
                content: POSITION_DELETES,
                file_path: v.file_path.clone(),
                file_format: PUFFIN.to_string(),
                partition: AvroPartition {},
                record_count: v.record_count,
                file_size_in_bytes: v.file_size_in_bytes,
                referenced_data_file: Some(v.referenced_data_file.clone()),
                content_offset: Some(v.content_offset),
                content_size_in_bytes: Some(v.content_size_in_bytes),
            },
        };
        writer
            .append_ser(&entry)
            .map_err(|e| ser_err("append delete manifest entry", e))?;
    }
    let container = writer
        .into_inner()
        .map_err(|e| ser_err("finish delete manifest", e))?;
    with_verbatim_schema(container, DELETE_MANIFEST_SCHEMA_JSON, &user_metadata)
}

/// Read a delete manifest's live deletion vectors. Any other kind of delete file is
/// an error (see the module docs).
pub fn read_delete_manifest(bytes: &[u8]) -> Result<Vec<DeleteFileMeta>, StorageError> {
    let schema = Schema::parse_str(DELETE_MANIFEST_SCHEMA_JSON)
        .map_err(|e| ser_err("parse delete manifest schema", e))?;
    let reader = Reader::with_schema(&schema, bytes)
        .map_err(|e| ser_err("open delete manifest reader", e))?;
    let mut out = Vec::new();
    for value in reader {
        let value = value.map_err(|e| ser_err("read delete manifest value", e))?;
        let entry: AvroDeleteEntry = apache_avro::from_value(&value)
            .map_err(|e| ser_err("decode delete manifest entry", e))?;
        if entry.status == 2 {
            continue; // DELETED
        }
        let f = entry.data_file;
        let unsupported = |what: &str| {
            ser_err(
                "delete manifest",
                format!("{what} in {} is not supported", f.file_path),
            )
        };
        match f.content {
            POSITION_DELETES => {}
            0 => return Err(unsupported("a data file")),
            _ => return Err(unsupported("an equality delete")),
        }
        if !f.file_format.eq_ignore_ascii_case(PUFFIN) {
            return Err(unsupported("a position-delete data file"));
        }
        let (Some(referenced_data_file), Some(content_offset), Some(content_size_in_bytes)) = (
            f.referenced_data_file.clone(),
            f.content_offset,
            f.content_size_in_bytes,
        ) else {
            return Err(unsupported("a deletion vector without its blob location"));
        };
        out.push(DeleteFileMeta {
            file_path: f.file_path,
            file_size_in_bytes: f.file_size_in_bytes,
            referenced_data_file,
            content_offset,
            content_size_in_bytes,
            record_count: f.record_count,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::{FormatVersion, ManifestHeader, write_manifest};
    use super::*;

    fn header() -> ManifestHeader<'static> {
        ManifestHeader {
            schema_json: r#"{"type":"struct","schema-id":0,"fields":[]}"#,
            format_version: FormatVersion::V3,
        }
    }

    fn vectors() -> Vec<DeleteFileMeta> {
        vec![
            DeleteFileMeta {
                file_path: "s3://w/t/data/dv-1.puffin".into(),
                file_size_in_bytes: 300,
                referenced_data_file: "s3://w/t/data/a.parquet".into(),
                content_offset: 4,
                content_size_in_bytes: 60,
                record_count: 3,
            },
            DeleteFileMeta {
                file_path: "s3://w/t/data/dv-1.puffin".into(),
                file_size_in_bytes: 300,
                referenced_data_file: "s3://w/t/data/b.parquet".into(),
                content_offset: 64,
                content_size_in_bytes: 40,
                record_count: 1,
            },
        ]
    }

    #[test]
    fn delete_manifest_round_trips() {
        let bytes = write_delete_manifest(&vectors(), 9, &header()).unwrap();
        assert_eq!(read_delete_manifest(&bytes).unwrap(), vectors());
    }

    #[test]
    fn the_header_says_deletes_and_every_field_carries_its_id() {
        let bytes = write_delete_manifest(&vectors(), 9, &header()).unwrap();
        let reader = Reader::new(&bytes[..]).unwrap();
        let meta = reader.user_metadata();
        assert_eq!(meta.get("content").map(|v| &v[..]), Some(&b"deletes"[..]));
        let schema_text = String::from_utf8_lossy(&bytes[..bytes.len().min(4096)]);
        for id in ["\"field-id\":143", "\"field-id\":144", "\"field-id\":145"] {
            assert!(
                schema_text.replace(' ', "").contains(id),
                "{id} in the header schema"
            );
        }
    }

    #[test]
    fn a_data_manifest_is_not_a_delete_manifest() {
        let data = write_manifest(
            &[super::super::DataFileMeta {
                file_path: "a.parquet".into(),
                record_count: 1,
                file_size_in_bytes: 10,
                column_stats: Vec::new(),
            }],
            9,
            None,
        )
        .unwrap();
        assert!(read_delete_manifest(&data).is_err());
    }
}
