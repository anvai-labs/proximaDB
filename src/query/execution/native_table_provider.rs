// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-USUB-2: register native (non-Parquet) relational tables with the DataFusion
//! destination, so SQL capability stops depending on physical storage format.
//!
//! # Why this exists
//!
//! `allow_engine_sql_fallback = true` appears at exactly one call site — the
//! DataFusion floor in `relational_pipeline.rs`. Because `ctx.sql` is the escape
//! hatch for every shape the shared relational frontend declines, CTEs, window
//! functions, joins, subqueries, `HAVING` and `DISTINCT` have worked **only** on
//! Parquet-backed tables. A user hits `0A000` telling them to run
//! `ALTER TABLE … MATERIALIZE` before their CTE will run.
//!
//! Parquet is incidental to that: it is simply the only format with a **registered
//! table source** on the relational path. Registering a source for native tables
//! closes the gap with **no format change, no publication and no write
//! amplification** (ADR-094 defect **b**).
//!
//! # How
//!
//! Rows come from [`crate::services::dml::DmlService::scan_table_relational`] via the
//! [`NativeTableSource`] port, are encoded with the same
//! `value_row_to_relational_record` → `proxima_records_to_record_batch` builder the
//! ADR-025 OLAP read-merge already uses, and are registered as a DataFusion
//! `MemTable`. Nothing new is invented; this is wiring.
//!
//! # Two properties worth stating plainly
//!
//! **ABAC is enforced by construction.** `scan_table_relational` resolves the ABAC
//! row filter itself — a `Denied` subject yields zero rows, `Restricted(p)` applies
//! the predicate, and an identity/storage tenant mismatch is an error. So rows handed
//! to DataFusion are already governed. This is the structural difference from the
//! Parquet path, which reads files directly and therefore cannot enforce a row filter
//! — the reason `parquet_backed` is forced false under ABAC today. This module does
//! **not** yet relax that gate (see `TD-USUB-3`); it only avoids adding a new bypass.
//!
//! **It is memory-bounded, and declines rather than degrading.** The whole table is
//! materialized in memory, exactly as the ADR-025 read-merge base already is. A table
//! larger than `row_cap` returns `None`, and the caller keeps its existing behaviour
//! (Volcano + the honest decline) rather than risking an OOM. Streaming is future work
//! and is tracked with the same note the read-merge carries.

use std::collections::HashMap;
use std::sync::Arc;

use proximadb_catalog_schema::CatalogTableSchema;
use proximadb_records::ProximaRecord;
use proximadb_runtime::PortIdentity;

/// Default cap on rows materialized per natively-registered table.
///
/// Chosen to be comfortably larger than the conformance/regression corpora while
/// remaining a bounded allocation. Override with `PROXIMADB_NATIVE_TABLE_ROW_CAP`.
pub const DEFAULT_NATIVE_TABLE_ROW_CAP: usize = 1_000_000;

/// Env gate for the whole feature. Default **OFF** (mandate #8): with the gate unset
/// this module is inert and the pipeline behaves byte-identically to before.
const NATIVE_TABLE_PROVIDER_ENV: &str = "PROXIMADB_NATIVE_TABLE_PROVIDER";
/// Env override for [`DEFAULT_NATIVE_TABLE_ROW_CAP`].
const NATIVE_TABLE_ROW_CAP_ENV: &str = "PROXIMADB_NATIVE_TABLE_ROW_CAP";

/// Is native relational table registration enabled? Default `false`.
pub fn native_table_provider_enabled() -> bool {
    matches!(
        std::env::var(NATIVE_TABLE_PROVIDER_ENV)
            .unwrap_or_default()
            .trim(),
        "1" | "true" | "TRUE" | "on" | "ON"
    )
}

/// Row cap for a natively-registered table (see [`DEFAULT_NATIVE_TABLE_ROW_CAP`]).
pub fn native_table_row_cap() -> usize {
    std::env::var(NATIVE_TABLE_ROW_CAP_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_NATIVE_TABLE_ROW_CAP)
}

/// Authoritative source of native relational rows.
///
/// Deliberately engine-neutral — no `datafusion::` types appear here, so the port
/// stays inside ADR-039's leakage boundary and the native lane can use it too.
#[async_trait::async_trait]
pub trait NativeTableSource: Send + Sync {
    /// The table's catalog schema plus **all** current live rows, each built with every
    /// column materialized in `props` (the shape `proxima_records_to_record_batch`
    /// expects).
    ///
    /// Returns `Ok(None)` when the table exceeds `row_cap`, so the caller declines
    /// cleanly instead of materializing an unbounded result.
    ///
    /// Implementations MUST apply the caller's ABAC row filter — see the module note.
    async fn full_table_records(
        &self,
        table: &str,
        tenant: Option<&str>,
        identity: PortIdentity<'_>,
        row_cap: usize,
    ) -> anyhow::Result<Option<(CatalogTableSchema, Vec<ProximaRecord>)>>;
}

/// Native tables eligible for registration on this query.
#[derive(Clone)]
pub struct NativeTableConfig {
    /// Authoritative row source (the `DmlService`).
    pub source: Arc<dyn NativeTableSource>,
    /// Table name as the SQL references it, keyed by normalized table key.
    pub tables: HashMap<String, String>,
    /// Per-table cap on materialized rows.
    pub row_cap: usize,
}

impl std::fmt::Debug for NativeTableConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeTableConfig")
            .field("tables", &self.tables)
            .field("row_cap", &self.row_cap)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate is OFF unless explicitly enabled — mandate #8 (ship default-off).
    /// Asserted against the real reader so a future refactor that flips the default
    /// trips here.
    #[test]
    fn gate_defaults_off_and_accepts_only_truthy_values() {
        // SAFETY: single-threaded unit test mutating process env, restored below.
        unsafe { std::env::remove_var(NATIVE_TABLE_PROVIDER_ENV) };
        assert!(
            !native_table_provider_enabled(),
            "native table provider must default OFF"
        );
        for truthy in ["1", "true", "on"] {
            unsafe { std::env::set_var(NATIVE_TABLE_PROVIDER_ENV, truthy) };
            assert!(native_table_provider_enabled(), "{truthy} should enable");
        }
        for falsy in ["0", "false", "off", ""] {
            unsafe { std::env::set_var(NATIVE_TABLE_PROVIDER_ENV, falsy) };
            assert!(!native_table_provider_enabled(), "{falsy} must not enable");
        }
        unsafe { std::env::remove_var(NATIVE_TABLE_PROVIDER_ENV) };
    }

    #[test]
    fn row_cap_falls_back_to_default_on_absent_or_invalid() {
        // SAFETY: single-threaded unit test mutating process env, restored below.
        unsafe { std::env::remove_var(NATIVE_TABLE_ROW_CAP_ENV) };
        assert_eq!(native_table_row_cap(), DEFAULT_NATIVE_TABLE_ROW_CAP);
        for bad in ["0", "-5", "abc", ""] {
            unsafe { std::env::set_var(NATIVE_TABLE_ROW_CAP_ENV, bad) };
            assert_eq!(
                native_table_row_cap(),
                DEFAULT_NATIVE_TABLE_ROW_CAP,
                "invalid cap {bad:?} must fall back to the default, never 0"
            );
        }
        unsafe { std::env::set_var(NATIVE_TABLE_ROW_CAP_ENV, "128") };
        assert_eq!(native_table_row_cap(), 128);
        unsafe { std::env::remove_var(NATIVE_TABLE_ROW_CAP_ENV) };
    }
}
