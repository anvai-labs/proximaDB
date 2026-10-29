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
    /// Index into [`PublishedOids::files`] of the file this row group is in.
    file: usize,
    min: Vec<u8>,
    max: Vec<u8>,
    bloom: Option<Sbbf>,
}

/// The published oids of one snapshot, as probe-able row-group summaries.
pub(crate) struct PublishedOids {
    groups: Vec<OidGroup>,
    /// The probed files and each one's row count, in `read` order.
    files: Vec<(Path, u64)>,
}

impl OidGroup {
    fn may_contain(&self, oid: &str) -> bool {
        let key = oid.as_bytes();
        self.min.as_slice() <= key
            && key <= self.max.as_slice()
            && self.bloom.as_ref().is_none_or(|b| b.check(oid))
    }
}

impl PublishedOids {
    /// `false` only when no published file can hold `oid`.
    pub(crate) fn may_contain(&self, oid: &str) -> bool {
        self.groups.iter().any(|g| g.may_contain(oid))
    }

    /// The files whose row groups may hold any of `oids` (min/max and Bloom
    /// have no false negatives): the exact position scan reads only these.
    pub(crate) fn suspect_files(&self, oids: &[String]) -> Vec<Path> {
        let mut suspect = vec![false; self.files.len()];
        for group in &self.groups {
            if !suspect[group.file] && oids.iter().any(|oid| group.may_contain(oid)) {
                suspect[group.file] = true;
            }
        }
        self.files
            .iter()
            .zip(suspect)
            .filter(|(_, is)| *is)
            .map(|((file, _), _)| file.clone())
            .collect()
    }

    /// Each probed file's row count.
    pub(crate) fn file_rows(&self) -> std::collections::HashMap<Path, u64> {
        self.files.iter().cloned().collect()
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
        let mut file_list = Vec::with_capacity(files.len());
        for (file_index, file) in files.iter().enumerate() {
            #[allow(deprecated)]
            let reader = ParquetObjectReader::new(store.clone(), bridge.object_store_path(file));
            let mut builder = ParquetRecordBatchStreamBuilder::new(reader)
                .await
                .map_err(|e| anyhow!("MATERIALIZE append: open '{file}': {e}"))?;
            if !same_columns(builder.schema(), expected) {
                return Ok(None);
            }
            let metadata = builder.metadata().clone();
            let rows = u64::try_from(metadata.file_metadata().num_rows()).unwrap_or(0);
            file_list.push((file.clone(), rows));
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
                    file: file_index,
                    min: min.data().to_vec(),
                    max: max.data().to_vec(),
                    bloom,
                });
            }
        }
        Ok(Some(Self {
            groups,
            files: file_list,
        }))
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
    /// Incremental update/delete (`PROXIMADB_MATERIALIZE_INCREMENTAL_DV`):
    /// append the changed oids' after-images and mark their OLD positions in
    /// `suspect_files` with deletion vectors.
    Delta {
        oids: Vec<String>,
        suspect_files: Vec<Path>,
        file_rows: std::collections::HashMap<Path, u64>,
    },
}

/// `PROXIMADB_MATERIALIZE_INCREMENTAL_DV` (default OFF): an update or delete
/// since the last publication republishes incrementally — after-images
/// appended, old positions marked with deletion vectors — instead of
/// rewriting the table. It changes the publication format (DVs force Iceberg
/// v3), so it ships off until baked (the storage-format mandate).
pub(crate) fn incremental_dv_enabled() -> bool {
    std::env::var("PROXIMADB_MATERIALIZE_INCREMENTAL_DV").is_ok_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Classify `changed` (the oids written since the published snapshot) against
/// `published`. `max_rows` bounds an append.
pub(crate) fn classify(
    changed: Vec<String>,
    published: &PublishedOids,
    max_rows: usize,
    incremental: bool,
) -> AppendPlan {
    if changed.is_empty() {
        return AppendPlan::Unchanged;
    }
    if changed.len() > max_rows {
        return AppendPlan::Rewrite("the delta exceeds the append bound");
    }
    if changed.iter().any(|oid| published.may_contain(oid)) {
        if !incremental {
            return AppendPlan::Rewrite(
                "a changed row may already be published (update or delete)",
            );
        }
        let suspect_files = published.suspect_files(&changed);
        return AppendPlan::Delta {
            oids: changed,
            suspect_files,
            file_rows: published.file_rows(),
        };
    }
    AppendPlan::Append(changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(min: &str, max: &str) -> OidGroup {
        OidGroup {
            file: 0,
            min: min.as_bytes().to_vec(),
            max: max.as_bytes().to_vec(),
            bloom: None,
        }
    }

    #[test]
    fn classification_appends_only_rows_no_published_file_can_hold() {
        let published = PublishedOids {
            groups: vec![group("t:row:100", "t:row:199")],
            files: vec![(Path::from("p0"), 100)],
        };
        let new = |oids: &[&str]| oids.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            classify(vec![], &published, 10, false),
            AppendPlan::Unchanged
        );
        assert_eq!(
            classify(new(&["t:row:200", "t:row:201"]), &published, 10, false),
            AppendPlan::Append(new(&["t:row:200", "t:row:201"]))
        );
        assert!(matches!(
            classify(new(&["t:row:200", "t:row:150"]), &published, 10, false),
            AppendPlan::Rewrite(_)
        ));
        assert!(matches!(
            classify(new(&["t:row:200", "t:row:201"]), &published, 1, false),
            AppendPlan::Rewrite(_)
        ));
    }

    /// With the incremental gate, a changed oid that may be published is a
    /// Delta — and the exact scan is narrowed to the files whose row groups
    /// may hold a changed oid (the other file is never read).
    #[test]
    fn incremental_classification_narrows_the_scan_to_suspect_files() {
        let mut far = group("t:row:500", "t:row:599");
        far.file = 1;
        let published = PublishedOids {
            groups: vec![group("t:row:100", "t:row:199"), far],
            files: vec![(Path::from("p0"), 100), (Path::from("p1"), 100)],
        };
        let changed = vec!["t:row:150".to_string(), "t:row:800".to_string()];
        assert_eq!(
            classify(changed.clone(), &published, 10, true),
            AppendPlan::Delta {
                oids: changed,
                suspect_files: vec![Path::from("p0")],
                file_rows: std::collections::HashMap::from([
                    (Path::from("p0"), 100),
                    (Path::from("p1"), 100),
                ]),
            },
            "only p0 may hold t:row:150; t:row:800 is in no file's range"
        );
        // Pure inserts still append, gate or not.
        assert!(matches!(
            classify(vec!["t:row:900".to_string()], &published, 10, true),
            AppendPlan::Append(_)
        ));
    }
}
