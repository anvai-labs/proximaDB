// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Low-cardinality delivery-state metrics for the ADR-066 trace sink.

use lazy_static::lazy_static;
use prometheus::{IntCounter, IntGauge};

use super::operational_metrics::{counter, gauge};

lazy_static! {
    pub static ref RECORDS_DROPPED_TOTAL: IntCounter = counter(
        "proximadb_io_trace_sink_records_dropped_total",
        "Trace records dropped from best-effort ingress or after an unrecoverable seal failure",
    );
    pub static ref BYTES_DROPPED_TOTAL: IntCounter = counter(
        "proximadb_io_trace_sink_bytes_dropped_total",
        "Uncompressed trace bytes dropped by the best-effort sink",
    );
    pub static ref SEAL_FAILURES_TOTAL: IntCounter = counter(
        "proximadb_io_trace_sink_seal_failures_total",
        "Trace segment compression or durable local-seal failures",
    );
    pub static ref UPLOAD_FAILURES_TOTAL: IntCounter = counter(
        "proximadb_io_trace_sink_upload_failures_total",
        "Trace object conditional-create or verification failures",
    );
    pub static ref UPLOAD_RETRIES_TOTAL: IntCounter = counter(
        "proximadb_io_trace_sink_upload_retries_total",
        "Durable pending trace files retried after their initial upload attempt",
    );
    pub static ref PENDING_FILES: IntGauge = gauge(
        "proximadb_io_trace_sink_pending_files",
        "Immutable trace files awaiting verified object-store delivery",
    );
    pub static ref PENDING_BYTES: IntGauge = gauge(
        "proximadb_io_trace_sink_pending_bytes",
        "Compressed bytes awaiting verified object-store delivery",
    );
    pub static ref LAST_SUCCESS_UNIX_SECONDS: IntGauge = gauge(
        "proximadb_io_trace_sink_last_success_unix_seconds",
        "Unix timestamp of the latest verified trace object delivery",
    );
    pub static ref DUPLICATE_INSTALLS_TOTAL: IntCounter = counter(
        "proximadb_io_trace_sink_duplicate_installs_total",
        "Trace sink installation attempts rejected because a worker was already live",
    );
    pub static ref SHUTDOWN_TIMEOUTS_TOTAL: IntCounter = counter(
        "proximadb_io_trace_sink_shutdown_timeouts_total",
        "Trace sink graceful shutdown attempts that exceeded the deadline",
    );
}

pub fn record_drop(records: u64, bytes: u64) {
    RECORDS_DROPPED_TOTAL.inc_by(records);
    BYTES_DROPPED_TOTAL.inc_by(bytes);
}

pub fn set_pending(files: u64, bytes: u64) {
    PENDING_FILES.set(files.min(i64::MAX as u64) as i64);
    PENDING_BYTES.set(bytes.min(i64::MAX as u64) as i64);
}

pub fn record_delivery_success() {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    LAST_SUCCESS_UNIX_SECONDS.set(now.min(i64::MAX as u64) as i64);
}
