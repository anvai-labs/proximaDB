//! Insert-only incremental MATERIALIZE (P2.4 slice 2, TD-USUB-10).
//!
//! A republication whose changes since the published `snapshot_lsn` are all new
//! rows appends them as extra parts instead of rewriting the table. Whether a
//! changed oid is new is decided by probing the published files' hidden
//! `__proxima_oid` column: per row group, its min/max statistics and its Bloom
//! filter. A changed oid that may be in a published file is possibly an update
//! or a delete, which an append cannot express, so the publication falls back to
//! a full rewrite. The probe never misses a published oid (min/max and Bloom
//! filters have no false negatives); a false positive only costs a full rewrite.

use anyhow::{Result, anyhow};
use object_store::path::Path;
#[allow(deprecated)]
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use parquet::bloom_filter::Sbbf;
use parquet::file::statistics::Statistics;
use proximadb_storage_common::object_store_bridge::ObjectStoreBridge;
use proximadb_storage_common::proxima_arrow::SYS_COL_OID;

/// Env gate: the most rows one incremental append may carry; a larger delta is
/// published by a full rewrite. `0`/`off` disables incremental append.
pub const APPEND_MAX_ROWS_ENV: &str = "PROXIMADB_MATERIALIZE_APPEND_MAX_ROWS";

/// Default for [`APPEND_MAX_ROWS_ENV`]: the delta's rows are held in memory
/// while they are written, so this bounds that memory.
pub const DEFAULT_APPEND_MAX_ROWS: u64 = 1_000_000;

/// The most files a snapshot may hold for a republication to append to it. Each
/// append adds a part, and every republication probes every part's footer, so
/// past this a full rewrite compacts them.
pub const APPEND_MAX_FILES: usize = 64;

/// The configured append bound, or `None` when incremental append is off.
pub(crate) fn configured_append_max_rows() -> Option<usize> {
    crate::services::storage_format::default_on_gate(APPEND_MAX_ROWS_ENV, DEFAULT_APPEND_MAX_ROWS)
        .map(|n| usize::try_from(n).unwrap_or(usize::MAX))
}

/// One published row group's oid bounds and (when it has one) Bloom filter.
struct OidGroup {
    min: Vec<u8>,
    max: Vec<u8>,
    bloom: Option<Sbbf>,
}

/// The published oids of one snapshot, as probe-able row-group summaries.
pub(crate) struct PublishedOids {
    groups: Vec<OidGroup>,
}

impl PublishedOids {
    /// `false` only when no published file can hold `oid`.
    pub(crate) fn may_contain(&self, oid: &str) -> bool {
        let key = oid.as_bytes();
        self.groups.iter().any(|g| {
            g.min.as_slice() <= key
                && key <= g.max.as_slice()
                && g.bloom.as_ref().is_none_or(|b| b.check(oid))
        })
    }

    /// Read the summaries of `files` (bridge-relative keys). `Ok(None)` when a
    /// file cannot be appended to: its columns differ from `expected` (the
    /// schema this publication writes, so a schema change since the last
    /// publication rewrites), it was written before the hidden oid column, or a
    /// row group has no oid statistics.
    pub(crate) async fn read(
        bridge: &dyn ObjectStoreBridge,
        files: &[Path],
        expected: &arrow_schema::Schema,
    ) -> Result<Option<Self>> {
        let store = bridge.inner_store();
        let mut groups = Vec::new();
        for file in files {
            #[allow(deprecated)]
            let reader = ParquetObjectReader::new(store.clone(), bridge.object_store_path(file));
            let mut builder = ParquetRecordBatchStreamBuilder::new(reader)
                .await
                .map_err(|e| anyhow!("MATERIALIZE append: open '{file}': {e}"))?;
            if !same_columns(builder.schema(), expected) {
                return Ok(None);
            }
            let metadata = builder.metadata().clone();
            // The Parquet leaf index, which differs from the Arrow field index
            // once a nested column precedes the oid column.
            let Some(column) = metadata
                .file_metadata()
                .schema_descr()
                .columns()
                .iter()
                .position(|c| c.path().string() == SYS_COL_OID)
            else {
                return Ok(None);
            };
            for (index, row_group) in metadata.row_groups().iter().enumerate() {
                if row_group.num_rows() == 0 {
                    continue;
                }
                let Some(Statistics::ByteArray(stats)) = row_group.column(column).statistics()
                else {
                    return Ok(None);
                };
                let (Some(min), Some(max)) = (stats.min_opt(), stats.max_opt()) else {
                    return Ok(None);
                };
                let bloom = if row_group.column(column).bloom_filter_offset().is_some() {
                    builder
                        .get_row_group_column_bloom_filter(index, column)
                        .await
                        .map_err(|e| anyhow!("MATERIALIZE append: bloom of '{file}': {e}"))?
                } else {
                    None
                };
                groups.push(OidGroup {
                    min: min.data().to_vec(),
                    max: max.data().to_vec(),
                    bloom,
                });
            }
        }
        Ok(Some(Self { groups }))
    }
}

/// Same column names, types and nullability, in order (field metadata such as
/// Iceberg field ids aside).
fn same_columns(file: &arrow_schema::Schema, expected: &arrow_schema::Schema) -> bool {
    file.fields().len() == expected.fields().len()
        && file.fields().iter().zip(expected.fields()).all(|(a, b)| {
            a.name() == b.name()
                && a.data_type() == b.data_type()
                && a.is_nullable() == b.is_nullable()
        })
}

/// How a republication proceeds.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AppendPlan {
    /// Rewrite the table (the reason, for the log).
    Rewrite(&'static str),
    /// Nothing changed since the published snapshot.
    Unchanged,
    /// Append these new oids to the published files.
    Append(Vec<String>),
}

/// Classify `changed` (the oids written since the published snapshot) against
/// `published`. `max_rows` bounds an append.
pub(crate) fn classify(
    changed: Vec<String>,
    published: &PublishedOids,
    max_rows: usize,
) -> AppendPlan {
    if changed.is_empty() {
        return AppendPlan::Unchanged;
    }
    if changed.len() > max_rows {
        return AppendPlan::Rewrite("the delta exceeds the append bound");
    }
    if changed.iter().any(|oid| published.may_contain(oid)) {
        return AppendPlan::Rewrite("a changed row may already be published (update or delete)");
    }
    AppendPlan::Append(changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(min: &str, max: &str) -> OidGroup {
        OidGroup {
            min: min.as_bytes().to_vec(),
            max: max.as_bytes().to_vec(),
            bloom: None,
        }
    }

    #[test]
    fn classification_appends_only_rows_no_published_file_can_hold() {
        let published = PublishedOids {
            groups: vec![group("t:row:100", "t:row:199")],
        };
        let new = |oids: &[&str]| oids.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(classify(vec![], &published, 10), AppendPlan::Unchanged);
        assert_eq!(
            classify(new(&["t:row:200", "t:row:201"]), &published, 10),
            AppendPlan::Append(new(&["t:row:200", "t:row:201"]))
        );
        assert!(matches!(
            classify(new(&["t:row:200", "t:row:150"]), &published, 10),
            AppendPlan::Rewrite(_)
        ));
        assert!(matches!(
            classify(new(&["t:row:200", "t:row:201"]), &published, 1),
            AppendPlan::Rewrite(_)
        ));
    }
}
