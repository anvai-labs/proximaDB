//! # `RecordReader` — the one decode seam for persisted segments (TD-USUB-5)
//!
//! Before this module, decoding a persisted segment back to canonical
//! [`ProximaRecord`]s was an **inline two-arm `match`** in the root crate
//! (`storage::engines::sst::segment_format::read_segment_records`). That shape
//! has two costs ADR-094 §4 calls out:
//!
//! 1. **A new format cannot be added without editing the root monolith**, so the
//!    relational/warehouse arm (Parquet) had no way in — and a Parquet object
//!    reaching the router fell through to the legacy default and was handed to a
//!    decoder that cannot read it.
//! 2. The decode rule for each format is not addressable on its own, so nothing
//!    below the root crate can decode a segment.
//!
//! This module makes the seam explicit: one trait, one implementation per
//! [`SegmentFormat`], selected by magic. The two format-layer implementations
//! ([`PaxRecordReader`], [`ParquetRecordReader`]) live here beside the formats
//! they read; the legacy `ProximaDataBlock` implementation and the dispatch table
//! live in the root crate, where that type is defined.
//!
//! ## What this is NOT
//!
//! [`crate::format_traits`] (`StorageFormat`/`InternalFormat`/`OpenTableFormat`)
//! is a *different* seam: async, path-addressed, and batch-stream shaped
//! (`read_batches(&ReadContext) -> RecordBatchStream`). `RecordReader` is the
//! synchronous bytes→records inverse the mixed-format router needs, on bytes that
//! have already been fetched. They compose; neither subsumes the other.
//!
//! ## Reuse, not re-implementation
//!
//! Per the storage-format-migration mandate, every implementation delegates to the
//! canonical inverse for its format — [`crate::pax_block::read_pax_segment_records`]
//! and [`crate::proxima_parquet::parquet_bytes_to_record_batches`] +
//! [`crate::proxima_arrow::record_batch_to_proxima_records`]. No decoder is
//! hand-rolled here.

#![forbid(unsafe_code)]

use anyhow::{Context, Result};
use proximadb_records::ProximaRecord;

use crate::segment_layout::SegmentFormat;

/// The catalog-derived context a decoder needs to reconstruct canonical records.
///
/// PAX stores user columns and embeddings positionally, so the schema keys are
/// required to name them; Parquet and the legacy block format are self-describing
/// and ignore those fields. `tenant_ctx` applies to every format: it is the
/// segment's owning tenant, stamped onto rows whose tenant column was dropped by
/// catalog-resolution, and ignored when a stored value is present.
///
/// Bundling these into one borrowed struct (rather than three positional
/// arguments) is what lets the trait stay object-safe with a stable signature as
/// formats are added.
#[derive(Debug, Clone, Copy, Default)]
pub struct SegmentReadContext<'a> {
    /// Embedding model ids, positionally aligned with the segment's embedding
    /// stripes. Empty = best-effort defaults (`model_{i}`).
    pub embedding_model_ids: &'a [String],
    /// User-defined column keys, positionally aligned with the segment's user
    /// stripes. Empty = best-effort defaults.
    pub user_column_keys: &'a [String],
    /// The segment's owning tenant, from the catalog/path. `None` keeps stored
    /// values verbatim.
    pub tenant_ctx: Option<&'a str>,
}

impl<'a> SegmentReadContext<'a> {
    /// Construct a context from the three catalog-derived inputs.
    pub fn new(
        embedding_model_ids: &'a [String],
        user_column_keys: &'a [String],
        tenant_ctx: Option<&'a str>,
    ) -> Self {
        Self {
            embedding_model_ids,
            user_column_keys,
            tenant_ctx,
        }
    }
}

/// Decode the bytes of one persisted segment back to canonical records.
///
/// One implementation per [`SegmentFormat`]. Implementations MUST fail loudly on
/// input they cannot decode — returning fewer records, or records of a format
/// they did not actually read, is the silently-wrong-answer failure mandate #1
/// forbids.
pub trait RecordReader: Send + Sync {
    /// The on-disk format this reader decodes. Used by the dispatch table to
    /// assert reader/format agreement.
    fn format(&self) -> SegmentFormat;

    /// Decode `bytes` into canonical records.
    fn read_records(
        &self,
        bytes: &[u8],
        ctx: &SegmentReadContext<'_>,
    ) -> Result<Vec<ProximaRecord>>;
}

/// Reads columnar PAX segments (`PBLK` / `PXH1` head, `PAXSEG01` tail) via the
/// canonical inverse [`crate::pax_block::read_pax_segment_records`].
#[derive(Debug, Clone, Copy, Default)]
pub struct PaxRecordReader;

impl RecordReader for PaxRecordReader {
    fn format(&self) -> SegmentFormat {
        SegmentFormat::Pax
    }

    fn read_records(
        &self,
        bytes: &[u8],
        ctx: &SegmentReadContext<'_>,
    ) -> Result<Vec<ProximaRecord>> {
        crate::pax_block::read_pax_segment_records(
            bytes,
            ctx.embedding_model_ids,
            ctx.user_column_keys,
            ctx.tenant_ctx,
        )
    }
}

/// Reads Apache Parquet files (`PAR1` at both ends) — the relational/warehouse
/// landing format under ADR-094 "format follows modality", and the Iceberg
/// interop seam.
///
/// Parquet is self-describing, so `embedding_model_ids` / `user_column_keys` are
/// unused; columns become `props` and a `FixedSizeBinary` column is decoded as a
/// dense fp32 embedding, exactly as
/// [`crate::proxima_arrow::record_batch_to_proxima_records`] defines. Identity
/// (`oid`) is not inferred — a self-describing Arrow batch does not carry it, and
/// the catalog-aware caller stamps it.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParquetRecordReader;

impl RecordReader for ParquetRecordReader {
    fn format(&self) -> SegmentFormat {
        SegmentFormat::Parquet
    }

    fn read_records(
        &self,
        bytes: &[u8],
        ctx: &SegmentReadContext<'_>,
    ) -> Result<Vec<ProximaRecord>> {
        let batches = crate::proxima_parquet::parquet_bytes_to_record_batches(
            bytes::Bytes::copy_from_slice(bytes),
        )
        .context("RecordReader(Parquet): decode parquet bytes")?;

        let mut out = Vec::new();
        for batch in &batches {
            out.extend(crate::proxima_arrow::record_batch_to_proxima_records(batch));
        }

        // Same catalog-resolution rule the PAX path applies in
        // `proximadb_block_format::record`: a stored tenant wins; an empty one is
        // stamped from the segment's owning tenant context.
        if let Some(tenant) = ctx.tenant_ctx {
            for record in out.iter_mut().filter(|r| r.tenant_id.is_empty()) {
                record.tenant_id = tenant.to_string();
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxima_parquet::record_batches_to_parquet_bytes;
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn parquet_bytes_with(col: &str, values: &[&str]) -> Vec<u8> {
        let schema = Arc::new(Schema::new(vec![Field::new(col, DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(values.to_vec()))],
        )
        .expect("batch");
        record_batches_to_parquet_bytes(&[batch], schema, None).expect("parquet bytes")
    }

    /// The Parquet arm decodes real Parquet bytes into records — and, crucially,
    /// those bytes are recognised as `SegmentFormat::Parquet`, so the router
    /// reaches this reader rather than the legacy default.
    #[test]
    fn parquet_reader_decodes_and_is_detected() {
        let bytes = parquet_bytes_with("name", &["a", "b", "c"]);
        assert_eq!(SegmentFormat::detect(&bytes), SegmentFormat::Parquet);

        let reader = ParquetRecordReader;
        assert_eq!(reader.format(), SegmentFormat::Parquet);

        let ctx = SegmentReadContext::default();
        let records = reader.read_records(&bytes, &ctx).expect("decode");
        assert_eq!(records.len(), 3);
        assert!(records.iter().all(|r| r.props.contains_key("name")));
    }

    /// `tenant_ctx` is stamped onto rows with no stored tenant — the same
    /// catalog-resolution rule the PAX path applies, so the two formats agree on
    /// tenant provenance and isolation is not format-dependent.
    #[test]
    fn parquet_reader_stamps_tenant_context() {
        let bytes = parquet_bytes_with("name", &["x", "y"]);

        let unstamped = ParquetRecordReader
            .read_records(&bytes, &SegmentReadContext::default())
            .expect("decode");
        assert!(unstamped.iter().all(|r| r.tenant_id.is_empty()));

        let stamped = ParquetRecordReader
            .read_records(&bytes, &SegmentReadContext::new(&[], &[], Some("tenant-7")))
            .expect("decode");
        assert!(stamped.iter().all(|r| r.tenant_id == "tenant-7"));
    }

    /// Truncated/corrupt Parquet must ERROR, never yield a partial or empty
    /// success — mandate #1, fail closed.
    #[test]
    fn parquet_reader_errors_on_corrupt_input() {
        let mut bytes = parquet_bytes_with("name", &["a"]);
        // Keep both magics intact (so detection still routes here) but destroy
        // the footer in between.
        let len = bytes.len();
        for b in bytes[4..len - 8].iter_mut() {
            *b = 0xFF;
        }
        assert_eq!(SegmentFormat::detect(&bytes), SegmentFormat::Parquet);
        assert!(
            ParquetRecordReader
                .read_records(&bytes, &SegmentReadContext::default())
                .is_err(),
            "corrupt parquet must fail closed, not decode to an empty result"
        );
    }

    /// The trait is object-safe — the property the root dispatch table depends
    /// on, and the reason a new format needs no change to the router.
    #[test]
    fn record_reader_is_object_safe() {
        let readers: Vec<Box<dyn RecordReader>> =
            vec![Box::new(PaxRecordReader), Box::new(ParquetRecordReader)];
        let formats: Vec<SegmentFormat> = readers.iter().map(|r| r.format()).collect();
        assert_eq!(formats, vec![SegmentFormat::Pax, SegmentFormat::Parquet]);
    }
}
