//! # puffin — the Puffin file that holds a v3 table's deletion vectors (P2.4 slice 3)
//!
//! ```text
//! file   = magic blob* footer
//! footer = magic payload u32-LE(payload length) flags[4] magic
//! magic  = "PFA1"
//! ```
//!
//! The footer payload is JSON: `{"blobs": [BlobMetadata], "properties": {...}}`, each
//! blob naming its `type`, `offset` and `length` in the file. A deletion vector is a
//! `deletion-vector-v1` blob over the reserved `_pos` field, with snapshot id and
//! sequence number `-1` (they are inherited from the manifest entry) and the
//! `referenced-data-file` and `cardinality` properties. The manifest entry points
//! at the blob directly (`content_offset` / `content_size_in_bytes`), so a reader
//! needs the footer only to validate the file.
//!
//! Blobs are written uncompressed, as the spec requires for deletion vectors; a
//! compressed footer is rejected on read.

use std::collections::BTreeMap;

use proximadb_kernel::error::StorageError;
use serde::{Deserialize, Serialize};

use crate::deletion_vector::PositionDeletes;

const MAGIC: &[u8; 4] = b"PFA1";
/// The blob type of a deletion vector.
pub const DELETION_VECTOR_V1: &str = "deletion-vector-v1";
/// Iceberg's reserved field id of the row position (`_pos`).
pub const ROW_POSITION_FIELD_ID: i32 = 2_147_483_645;
/// Footer flag (byte 0, bit 0): the payload is compressed.
const FOOTER_PAYLOAD_COMPRESSED: u8 = 1;

fn puffin_err(context: impl std::fmt::Display) -> StorageError {
    StorageError::Serialization(format!("puffin: {context}"))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlobMetadata {
    #[serde(rename = "type")]
    pub blob_type: String,
    pub fields: Vec<i32>,
    #[serde(rename = "snapshot-id")]
    pub snapshot_id: i64,
    #[serde(rename = "sequence-number")]
    pub sequence_number: i64,
    pub offset: i64,
    pub length: i64,
    #[serde(
        rename = "compression-codec",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub compression_codec: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub properties: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileMetadata {
    pub blobs: Vec<BlobMetadata>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub properties: BTreeMap<String, String>,
}

/// One written deletion vector: where its blob sits in the Puffin file, which is
/// what the delete manifest entry records.
#[derive(Debug, Clone, PartialEq)]
pub struct WrittenVector {
    pub referenced_data_file: String,
    pub content_offset: i64,
    pub content_size_in_bytes: i64,
    pub cardinality: i64,
}

/// A Puffin file holding one deletion vector per `(data file, deletes)` pair.
/// Returns the file bytes and, in input order, each vector's location in them.
pub fn write_deletion_vectors(
    vectors: &[(String, &PositionDeletes)],
) -> Result<(Vec<u8>, Vec<WrittenVector>), StorageError> {
    let mut out = MAGIC.to_vec();
    let mut blobs = Vec::with_capacity(vectors.len());
    let mut written = Vec::with_capacity(vectors.len());
    for (data_file, deletes) in vectors {
        if deletes.is_empty() {
            return Err(puffin_err(format!("empty deletion vector for {data_file}")));
        }
        let blob = deletes.encode_blob();
        let offset = out.len() as i64;
        let length = blob.len() as i64;
        let cardinality = deletes.cardinality() as i64;
        out.extend_from_slice(&blob);
        blobs.push(BlobMetadata {
            blob_type: DELETION_VECTOR_V1.to_string(),
            fields: vec![ROW_POSITION_FIELD_ID],
            snapshot_id: -1,
            sequence_number: -1,
            offset,
            length,
            compression_codec: None,
            properties: BTreeMap::from([
                ("referenced-data-file".to_string(), data_file.clone()),
                ("cardinality".to_string(), cardinality.to_string()),
            ]),
        });
        written.push(WrittenVector {
            referenced_data_file: data_file.clone(),
            content_offset: offset,
            content_size_in_bytes: length,
            cardinality,
        });
    }
    let payload = serde_json::to_vec(&FileMetadata {
        blobs,
        properties: BTreeMap::from([("created-by".to_string(), "ProximaDB".to_string())]),
    })
    .map_err(puffin_err)?;
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&payload);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(MAGIC);
    Ok((out, written))
}

/// The footer of a whole Puffin file.
pub fn read_footer(file: &[u8]) -> Result<FileMetadata, StorageError> {
    // magic + footer magic + length + flags + magic, around an empty payload.
    if file.len() < 4 + 4 + 4 + 4 + 4 || &file[..4] != MAGIC || &file[file.len() - 4..] != MAGIC {
        return Err(puffin_err("missing magic"));
    }
    let flags = &file[file.len() - 8..file.len() - 4];
    if flags[0] & FOOTER_PAYLOAD_COMPRESSED != 0 {
        return Err(puffin_err("compressed footers are not supported"));
    }
    let len_at = file.len() - 12;
    let payload_len = u32::from_le_bytes(file[len_at..len_at + 4].try_into().expect("4 bytes"));
    let payload_start = len_at
        .checked_sub(payload_len as usize)
        .filter(|start| *start >= 8)
        .ok_or_else(|| puffin_err("footer longer than the file"))?;
    if &file[payload_start - 4..payload_start] != MAGIC {
        return Err(puffin_err("missing footer magic"));
    }
    serde_json::from_slice(&file[payload_start..len_at]).map_err(puffin_err)
}

/// Decode the deletion vector a manifest entry points at (`content_offset` /
/// `content_size_in_bytes`), given only those bytes of the file.
pub fn read_deletion_vector(blob: &[u8]) -> Result<PositionDeletes, StorageError> {
    PositionDeletes::decode_blob(blob)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deletes(positions: &[u64]) -> PositionDeletes {
        let mut d = PositionDeletes::new();
        for p in positions {
            d.insert(*p);
        }
        d
    }

    #[test]
    fn a_written_file_reads_back_through_its_footer_and_offsets() {
        let a = deletes(&[0, 3, 9]);
        let b = deletes(&[1 << 20]);
        let (file, written) = write_deletion_vectors(&[
            ("s3://w/t/data/a.parquet".to_string(), &a),
            ("s3://w/t/data/b.parquet".to_string(), &b),
        ])
        .expect("write");
        let footer = read_footer(&file).expect("footer");
        assert_eq!(footer.blobs.len(), 2);
        assert_eq!(footer.properties["created-by"], "ProximaDB");
        for (meta, (w, expect)) in footer.blobs.iter().zip(written.iter().zip([&a, &b])) {
            assert_eq!(meta.blob_type, DELETION_VECTOR_V1);
            assert_eq!(meta.fields, vec![ROW_POSITION_FIELD_ID]);
            assert_eq!((meta.snapshot_id, meta.sequence_number), (-1, -1));
            assert_eq!(
                (meta.offset, meta.length),
                (w.content_offset, w.content_size_in_bytes)
            );
            assert_eq!(
                meta.properties["referenced-data-file"],
                w.referenced_data_file
            );
            assert_eq!(meta.properties["cardinality"], w.cardinality.to_string());
            let start = w.content_offset as usize;
            let blob = &file[start..start + w.content_size_in_bytes as usize];
            assert_eq!(
                read_deletion_vector(blob).expect("dv").positions(),
                expect.positions()
            );
        }
        assert_eq!(
            written[0].content_offset, 4,
            "the first blob follows the magic"
        );
    }

    #[test]
    fn malformed_files_are_rejected() {
        let (file, _) =
            write_deletion_vectors(&[("f".to_string(), &deletes(&[1]))]).expect("write");
        assert!(read_footer(&file[..file.len() - 1]).is_err());
        assert!(read_footer(&file[1..]).is_err());
        let mut compressed = file.clone();
        let flags = compressed.len() - 8;
        compressed[flags] = FOOTER_PAYLOAD_COMPRESSED;
        assert!(read_footer(&compressed).is_err());
        assert!(write_deletion_vectors(&[("f".to_string(), &PositionDeletes::new())]).is_err());
    }
}
