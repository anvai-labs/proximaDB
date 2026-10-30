// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Oid-range partitions for the Layer A scan (ADR-097 P3.1, PR 7b).
//!
//! The scan splits into DataFusion partitions by **oid range**: every copy of
//! an oid — in every segment, resident or tombstoned — falls in exactly one
//! partition, so newest-wins stays a per-partition merge and the partitions
//! never disagree about which copy is served.
//!
//! The boundaries are chosen once per scan, when it is planned, at quantiles
//! of the segments' row groups by their minimum oid (Parquet statistics of
//! `__proxima_oid`); the scan declares exactly as many partitions as there are
//! ranges, and every partition shares them and the footers loaded with them.
//! A segment that is not oid-ordered Parquet (read whole, see the provider)
//! keeps the scan to one range, so it is never decoded more than once.
//!
//! A partition reads only the row groups whose oid statistics overlap its
//! range — statistics may be truncated, but a truncated minimum is never above
//! the true one nor a truncated maximum below, so the overlap test only errs
//! towards reading — and masks the rows of a row group straddling a boundary
//! to its range. Absent statistics read as "overlaps every range".
//!
//! Scan metrics are sums over the partitions: a row group straddling a
//! boundary is counted by each partition that reads it.

use std::sync::Arc;

use anyhow::Result;
use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch, Scalar, StringArray};
use arrow::compute::kernels::boolean::and;
use arrow::compute::kernels::cmp::{gt_eq, lt};
use arrow::compute::{cast, filter_record_batch};
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::file::metadata::ParquetMetaData;
use proximadb_storage_common::proxima_arrow::SYS_COL_OID;

use crate::storage::persistence::filesystem::FileSystem;
use crate::storage::persistence::filesystem::parquet_async_reader::{
    FileSystemParquetReader, is_io_error,
};

/// One partition's half-open oid range `[lo, hi)`; `None` is unbounded. An
/// `empty` range holds no oid.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct OidRange {
    lo: Option<String>,
    hi: Option<String>,
    empty: bool,
}

impl OidRange {
    /// Every oid.
    pub fn all() -> Self {
        Self::default()
    }

    /// No oid.
    pub fn empty() -> Self {
        Self {
            empty: true,
            ..Self::default()
        }
    }

    pub fn is_all(&self) -> bool {
        !self.empty && self.lo.is_none() && self.hi.is_none()
    }

    pub fn contains(&self, oid: &str) -> bool {
        !self.empty
            && self.lo.as_deref().is_none_or(|lo| oid >= lo)
            && self.hi.as_deref().is_none_or(|hi| oid < hi)
    }

    /// May a row group whose oids lie in `[min, max]` (each `None` when
    /// unknown) hold an oid of this range?
    pub fn overlaps(&self, min: Option<&str>, max: Option<&str>) -> bool {
        if self.empty {
            return false;
        }
        let below_hi = match (self.hi.as_deref(), min) {
            (Some(hi), Some(min)) => min < hi,
            _ => true,
        };
        let above_lo = match (self.lo.as_deref(), max) {
            (Some(lo), Some(max)) => max >= lo,
            _ => true,
        };
        below_hi && above_lo
    }

    /// `batch` with only the rows whose `__proxima_oid` lies in this range.
    pub fn mask(&self, batch: RecordBatch) -> Result<RecordBatch> {
        if self.is_all() {
            return Ok(batch);
        }
        let oids = batch.column(batch.schema().index_of(SYS_COL_OID)?);
        // Each bound in the column's own string type (Utf8, LargeUtf8, view).
        let bound = |value: &str| -> Result<Scalar<ArrayRef>> {
            let one: ArrayRef = Arc::new(StringArray::from(vec![value]));
            Ok(Scalar::new(cast(&one, oids.data_type())?))
        };
        let mut keep = BooleanArray::from(vec![!self.empty; batch.num_rows()]);
        if let Some(lo) = &self.lo {
            keep = and(&keep, &gt_eq(oids, &bound(lo)?)?)?;
        }
        if let Some(hi) = &self.hi {
            keep = and(&keep, &lt(oids, &bound(hi)?)?)?;
        }
        Ok(filter_record_batch(&batch, &keep)?)
    }

    /// Per row group of an oid-ordered segment: may it hold an oid of this
    /// range?
    pub fn row_groups(&self, metadata: &ParquetMetaData, arrow: &ArrowReaderMetadata) -> Vec<bool> {
        let n = metadata.num_row_groups();
        if self.is_all() {
            return vec![true; n];
        }
        let Some((mins, maxes)) = oid_bounds(metadata, arrow) else {
            return vec![!self.empty; n];
        };
        (0..n)
            .map(|i| self.overlaps(bound(&mins, i), bound(&maxes, i)))
            .collect()
    }
}

/// Row group `i`'s statistic, if present.
fn bound(values: &StringArray, i: usize) -> Option<&str> {
    (i < values.len() && values.is_valid(i)).then(|| values.value(i))
}

/// Per row group: the minimum and maximum oid statistics.
fn oid_bounds(
    metadata: &ParquetMetaData,
    arrow: &ArrowReaderMetadata,
) -> Option<(StringArray, StringArray)> {
    let converter = StatisticsConverter::try_new(
        SYS_COL_OID,
        arrow.schema(),
        metadata.file_metadata().schema_descr(),
    )
    .ok()?;
    let as_strings = |a: ArrayRef| {
        cast(&a, &arrow::datatypes::DataType::Utf8)
            .ok()?
            .as_any()
            .downcast_ref::<StringArray>()
            .cloned()
    };
    let mins = as_strings(
        converter
            .row_group_mins(metadata.row_groups().iter())
            .ok()?,
    )?;
    let maxes = as_strings(
        converter
            .row_group_maxes(metadata.row_groups().iter())
            .ok()?,
    )?;
    Some((mins, maxes))
}

/// At most `partitions` ranges covering every oid, split at quantiles of
/// `mins` (the row groups' minimum oids); fewer when there are fewer distinct
/// split points.
pub(crate) fn ranges(mut mins: Vec<String>, partitions: usize) -> Vec<OidRange> {
    let partitions = partitions.max(1);
    mins.sort();
    let mut bounds: Vec<String> = Vec::new();
    for k in 1..partitions {
        let Some(candidate) = mins.get(k * mins.len() / partitions) else {
            continue;
        };
        // Strictly increasing, and never the smallest minimum: a boundary
        // there would leave the first range empty.
        if bounds.last().is_none_or(|last| candidate > last) && Some(candidate) != mins.first() {
            bounds.push(candidate.clone());
        }
    }
    let mut out = Vec::with_capacity(bounds.len() + 1);
    let mut lo: Option<String> = None;
    for hi in bounds {
        out.push(OidRange {
            lo: lo.take(),
            hi: Some(hi.clone()),
            empty: false,
        });
        lo = Some(hi);
    }
    out.push(OidRange {
        lo,
        hi: None,
        empty: false,
    });
    out
}

/// The scan's partition layout: each partition's oid range, and every
/// segment's footer (`None` for an object that is not Parquet), loaded once
/// and shared by the partitions.
#[derive(Debug)]
pub(crate) struct Layout {
    pub ranges: Vec<OidRange>,
    pub footers: Vec<Option<Arc<ParquetMetaData>>>,
}

impl Layout {
    /// `oid_ordered` tells a segment the scan streams in oid order from one it
    /// reads whole; any of the latter keeps the scan to one range.
    pub async fn load(
        filesystem: Arc<dyn FileSystem>,
        segments: &[String],
        partitions: usize,
        oid_ordered: fn(&ParquetMetaData) -> bool,
    ) -> Result<Self> {
        let footers = futures::future::try_join_all(segments.iter().map(|path| {
            let filesystem = filesystem.clone();
            async move {
                let mut reader = FileSystemParquetReader::new(filesystem, path.clone());
                match reader.load_metadata(None).await {
                    Ok(metadata) => Ok(Some(metadata)),
                    Err(e) if is_io_error(&e) => {
                        Err(anyhow::anyhow!("layer A: read segment '{path}': {e}"))
                    }
                    Err(_) => Ok(None),
                }
            }
        }))
        .await?;
        if !footers
            .iter()
            .all(|f| f.as_deref().is_some_and(oid_ordered))
        {
            return Ok(Self {
                ranges: vec![OidRange::all()],
                footers,
            });
        }
        let mut mins = Vec::new();
        for metadata in footers.iter().flatten() {
            // A footer whose schema does not convert gives no split points; the
            // segment itself is still read (and fails or not) as before.
            let Ok(arrow) =
                ArrowReaderMetadata::try_new(metadata.clone(), ArrowReaderOptions::new())
            else {
                continue;
            };
            if let Some((segment_mins, _)) = oid_bounds(metadata, &arrow) {
                mins.extend(segment_mins.iter().flatten().map(str::to_string));
            }
        }
        Ok(Self {
            ranges: ranges(mins, partitions),
            footers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};

    fn s(v: &str) -> String {
        v.to_string()
    }

    #[test]
    fn ranges_cover_every_oid_once_at_quantiles() {
        let mins: Vec<String> = (0..8).map(|i| format!("{i:02}")).collect();
        let r = ranges(mins, 4);
        assert_eq!(r.len(), 4);
        for oid in ["", "00", "01", "02", "03", "05", "07", "99", "zz"] {
            assert_eq!(
                r.iter().filter(|range| range.contains(oid)).count(),
                1,
                "{oid}"
            );
        }
        assert!(r[0].contains("01") && r[1].contains("02") && r[3].contains("07"));
    }

    #[test]
    fn too_few_split_points_give_fewer_ranges() {
        assert_eq!(
            ranges(vec![s("a"), s("a"), s("a")], 4),
            vec![OidRange::all()]
        );
        assert_eq!(ranges(Vec::new(), 3), vec![OidRange::all()]);
        assert_eq!(ranges(vec![s("x")], 1), vec![OidRange::all()]);
        let two = ranges(vec![s("a"), s("a"), s("b"), s("b")], 64);
        assert_eq!(two.len(), 2, "{two:?}");
        assert!(two[0].contains("a") && two[1].contains("b") && !two[0].contains("b"));
        assert!(!OidRange::empty().contains("a") && !OidRange::empty().overlaps(None, None));
    }

    #[test]
    fn overlap_is_conservative_at_the_edges() {
        let r = OidRange {
            lo: Some(s("b")),
            hi: Some(s("d")),
            empty: false,
        };
        assert!(r.overlaps(Some("a"), Some("b")), "max == lo overlaps");
        assert!(!r.overlaps(Some("a"), Some("az")));
        assert!(r.overlaps(Some("cz"), Some("z")));
        assert!(!r.overlaps(Some("d"), Some("z")), "min == hi is outside");
        assert!(!r.overlaps(None, Some("a0")), "unknown min, max below lo");
        assert!(r.overlaps(Some("c"), None), "unknown max");
        assert!(r.overlaps(None, None), "unknown statistics overlap");
        assert!(r.contains("b") && r.contains("c") && !r.contains("d") && !r.contains("a"));
    }

    #[test]
    fn mask_keeps_only_the_range() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            SYS_COL_OID,
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["a", "b", "c", "d"]))],
        )
        .unwrap();
        let r = OidRange {
            lo: Some(s("b")),
            hi: Some(s("d")),
            empty: false,
        };
        let masked = r.mask(batch.clone()).unwrap();
        let oids = masked
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(oids.iter().flatten().collect::<Vec<_>>(), vec!["b", "c"]);
        assert_eq!(OidRange::empty().mask(batch.clone()).unwrap().num_rows(), 0);
        // The bounds take the column's own string type.
        let large = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                SYS_COL_OID,
                DataType::LargeUtf8,
                false,
            )])),
            vec![cast(batch.column(0), &DataType::LargeUtf8).unwrap()],
        )
        .unwrap();
        assert_eq!(r.mask(large).unwrap().num_rows(), 2);
    }
}
