//! Bulk record-operations port (TD-104 S3).
//!
//! The Arrow Flight ingest path (`do_put`) consumes record-batch insert/upsert/
//! delete. This port lets the Flight service depend on the contract instead of a
//! concrete root-crate service. Implemented by the root crate's
//! `RecordOpsService` (`src/api_handlers/record_ops_service.rs`), which owns the
//! `handle_record_*_for_tenant` write path (the legacy `UnifiedHandlers` wrapper
//! was deleted in TD-104 S3-f).
//!
//! Inputs are canonical (`ProximaRecord` from `proximadb-records`); the result is the
//! relocated [`crate::batch_result::BatchOperationResult`]. No durable authority lives
//! here — this is a façade over the same vector/record services.

use anyhow::Result;
use async_trait::async_trait;
use proximadb_filter_expression::FilterExpression;
use proximadb_records::ProximaRecord;
use crate::rich_record::RichRecordBatchRequest;

use crate::batch_result::BatchOperationResult;

#[async_trait]
pub trait RecordOpsPort: Send + Sync {
    /// Insert a batch of canonical records into `collection_id`.
    async fn insert_record_batch(
        &self,
        collection_id: &str,
        records: Vec<ProximaRecord>,
        tenant_id: Option<&str>,
    ) -> Result<BatchOperationResult>;

    /// Upsert a batch of canonical records into `collection_id`.
    async fn upsert_record_batch(
        &self,
        collection_id: &str,
        records: Vec<ProximaRecord>,
        tenant_id: Option<&str>,
    ) -> Result<BatchOperationResult>;

    /// Delete records by id from `collection_id`.
    async fn delete_record_batch(
        &self,
        collection_id: &str,
        record_ids: Vec<String>,
        tenant_id: Option<&str>,
    ) -> Result<BatchOperationResult>;

    // ── ADR-094: the REST v2 handler surface (records.rs consumers) ──

    /// Full record-batch write orchestration (insert semantics; validation,
    /// routing and metrics included).
    async fn handle_record_batch_for_tenant(
        &self,
        request: RichRecordBatchRequest,
        tenant_id: Option<&str>,
    ) -> Result<BatchOperationResult> {
        let _ = request;
        Err(anyhow::anyhow!(
            "record batch handling is not implemented by this runtime port"
        ))
    }

    /// Full record-batch delete orchestration.
    async fn handle_record_delete_batch_for_tenant(
        &self,
        request: crate::rich_record::RichRecordDeleteBatchRequest,
        tenant_id: Option<&str>,
    ) -> Result<BatchOperationResult> {
        let _ = request;
        Err(anyhow::anyhow!(
            "record delete handling is not implemented by this runtime port"
        ))
    }

    /// Get one record (canonical v2 semantics). Returns the matched search
    /// result, if any (the root alias `RichRecordGetResponse = Option<
    /// RichSearchResult>`).
    async fn handle_record_get_for_tenant(
        &self,
        request: crate::rich_record::RichRecordGetRequest,
        identity: crate::service_ports::PortIdentity<'_>,
    ) -> Result<Option<crate::rich_search::RichSearchResult>> {
        let _ = request;
        Err(anyhow::anyhow!(
            "record get is not implemented by this runtime port"
        ))
    }

    /// Paginated scan (cursor-based) over visible records.
    async fn handle_record_scan_paginated_for_tenant(
        &self,
        collection_id: &str,
        cursor: Option<&proximadb_scan_cursor::scan_cursor::ScanCursor>,
        limit: usize,
        include_vector: bool,
        include_props: bool,
        tenant_id: Option<&str>,
        filter: Option<&FilterExpression>,
        now_ns: i64,
    ) -> Result<(
        Vec<ProximaRecord>,
        Option<proximadb_scan_cursor::scan_cursor::ScanCursor>,
    )> {
        let _ = (
            collection_id,
            cursor,
            limit,
            include_vector,
            include_props,
            tenant_id,
            filter,
            now_ns,
        );
        Err(anyhow::anyhow!(
            "record scan is not implemented by this runtime port"
        ))
    }

    /// Canonical v2 search (ABAC + tenant-scoped).
    async fn handle_record_search_for_tenant(
        &self,
        request: crate::rich_search::RichSearchRequest,
        identity: crate::service_ports::PortIdentity<'_>,
    ) -> Result<crate::rich_search::RichSearchResponse> {
        let _ = request;
        Err(anyhow::anyhow!(
            "record search is not implemented by this runtime port"
        ))
    }
}
