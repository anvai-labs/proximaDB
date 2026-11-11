//! Decoding of the per-row version facts (`version`, `updated_at`, `expires_at`)
//! from a columnar batch into `ProximaRecord` — ONE place, shared by every
//! record builder that reads flushed Parquet segments.
//!
//! Why this exists (TD-ENGINE-MVCC-1): the builders each re-derived these
//! columns and got them wrong in different ways. One downcast `version` to
//! `Int64Array` while the writer stores `UInt32` (so every row read back as
//! version 0); none read `updated_at` (so every row's last-write time was 0);
//! and until #2518 none read `expires_at` (so every row looked immortal).
//! Without these facts no reader can tell which copy of an id is the newest.
//!
//! Writer contract (`parquet_write_engine::schema_builder` +
//! `streaming_writer`): `updated_at` Int64 milliseconds (nullable), `version`
//! UInt32 (nullable), `expires_at` Int64 milliseconds = `valid_to_ns / 1e6`
//! (nullable; 0 = tombstone). Mixed-read safe: a missing column or a null cell
//! leaves the record's default, and `version` also accepts `Int64` so a file
//! written with the other integer width still decodes.

use arrow::record_batch::RecordBatch;
use arrow_array::{Array, Int64Array, UInt32Array};
use proximadb_records::ProximaRecord;
use proximadb_storage_common::{FIELD_EXPIRES_AT, FIELD_UPDATED_AT, FIELD_VERSION};

enum VersionColumn<'a> {
    U32(&'a UInt32Array),
    I64(&'a Int64Array),
    Absent,
}

/// Borrowed views of one batch's version-fact columns.
pub(crate) struct VersionColumns<'a> {
    version: VersionColumn<'a>,
    updated_at_ms: Option<&'a Int64Array>,
    expires_at_ms: Option<&'a Int64Array>,
}

fn int64<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a Int64Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
}

impl<'a> VersionColumns<'a> {
    pub(crate) fn from_batch(batch: &'a RecordBatch) -> Self {
        let version = match batch.column_by_name(FIELD_VERSION) {
            Some(col) => {
                if let Some(a) = col.as_any().downcast_ref::<UInt32Array>() {
                    VersionColumn::U32(a)
                } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                    VersionColumn::I64(a)
                } else {
                    VersionColumn::Absent
                }
            }
            None => VersionColumn::Absent,
        };
        Self {
            version,
            updated_at_ms: int64(batch, FIELD_UPDATED_AT),
            expires_at_ms: int64(batch, FIELD_EXPIRES_AT),
        }
    }

    /// The row's `version`, if the column is present and the cell non-null.
    pub(crate) fn version(&self, row: usize) -> Option<u64> {
        match self.version {
            VersionColumn::U32(a) if a.is_valid(row) => Some(u64::from(a.value(row))),
            VersionColumn::I64(a) if a.is_valid(row) => u64::try_from(a.value(row)).ok(),
            _ => None,
        }
    }

    /// Stamp the row's version facts onto `record`. Absent/null facts leave
    /// the record's existing value (mixed-read safe).
    pub(crate) fn stamp(&self, record: &mut ProximaRecord, row: usize) {
        if let Some(v) = self.version(row) {
            record.record_version = v;
        }
        if let Some(a) = self.updated_at_ms
            && a.is_valid(row)
        {
            record.updated_at_ns = a.value(row).saturating_mul(1_000_000);
        }
        if let Some(a) = self.expires_at_ms
            && a.is_valid(row)
        {
            // ms -> ns. A tombstone (0) stays 0.
            record.valid_to_ns = Some(a.value(row).saturating_mul(1_000_000));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch(version: Arc<dyn Array>, version_type: DataType) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(FIELD_VERSION, version_type, true),
            Field::new(FIELD_UPDATED_AT, DataType::Int64, true),
            Field::new(FIELD_EXPIRES_AT, DataType::Int64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                version,
                Arc::new(Int64Array::from(vec![Some(1_700_000_000_123), None, None])),
                Arc::new(Int64Array::from(vec![
                    Some(1_800_000_000_000),
                    Some(0),
                    None,
                ])),
            ],
        )
        .unwrap_or_else(|e| panic!("batch: {e}"))
    }

    #[test]
    fn decodes_the_writer_contract_uint32_version_and_ms_times() {
        let b = batch(
            Arc::new(UInt32Array::from(vec![Some(7), Some(2), None])),
            DataType::UInt32,
        );
        let cols = VersionColumns::from_batch(&b);

        let mut r0 = ProximaRecord::default();
        cols.stamp(&mut r0, 0);
        assert_eq!(
            r0.record_version, 7,
            "UInt32 version must decode (was always 0)"
        );
        assert_eq!(r0.updated_at_ns, 1_700_000_000_123_000_000);
        assert_eq!(r0.valid_to_ns, Some(1_800_000_000_000_000_000));

        let mut r1 = ProximaRecord::default();
        // `Default` stamps `updated_at_ns` with the current time, so compare
        // against the pre-stamp value, not 0.
        let r1_updated_before = r1.updated_at_ns;
        cols.stamp(&mut r1, 1);
        assert_eq!(r1.record_version, 2);
        assert_eq!(r1.valid_to_ns, Some(0), "tombstone stays 0");
        assert_eq!(
            r1.updated_at_ns, r1_updated_before,
            "null updated_at leaves the existing value"
        );

        let mut r2 = ProximaRecord {
            record_version: 9,
            ..Default::default()
        };
        cols.stamp(&mut r2, 2);
        assert_eq!(
            r2.record_version, 9,
            "null version leaves the existing value"
        );
        assert_eq!(r2.valid_to_ns, None, "null expires_at = no TTL");
    }

    #[test]
    fn accepts_int64_version_and_missing_columns() {
        let b = batch(
            Arc::new(Int64Array::from(vec![Some(5), Some(-1), None])),
            DataType::Int64,
        );
        let cols = VersionColumns::from_batch(&b);
        assert_eq!(cols.version(0), Some(5));
        assert_eq!(
            cols.version(1),
            None,
            "negative versions are rejected, not wrapped"
        );

        let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
        let none = VersionColumns::from_batch(&empty);
        let mut r = ProximaRecord {
            record_version: 3,
            ..Default::default()
        };
        none.stamp(&mut r, 0);
        assert_eq!(r.record_version, 3);
        assert_eq!(r.valid_to_ns, None);
    }
}
