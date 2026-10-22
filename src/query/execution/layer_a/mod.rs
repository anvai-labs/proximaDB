// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! ADR-097 P3.1: the Layer A table provider — format-independent native SQL.
//!
//! With `PROXIMADB_NATIVE_TABLE_PROVIDER` on, a native relational table used to
//! reach DataFusion as one `MemTable` built from a fully materialized scan, and a
//! table past `PROXIMADB_NATIVE_TABLE_ROW_CAP` (1M rows) was not registered at all.
//! This module serves the same rows by **streaming** the record tier instead:
//!
//! 1. a flush-consistent snapshot of the table's partition — resident rows,
//!    memory tombstones, segment paths — taken under one lock, with no segment
//!    read ([`crate::services::layer_a_snapshot`]);
//! 2. each spill segment read row group by row group through the `FileSystem`
//!    port ([`FileSystemParquetReader`]);
//! 3. a newest-wins k-way merge over the oid-sorted segments, under the resident
//!    rows and tombstones ([`merge`]);
//! 4. every row conformed to the catalog schema exactly as the materialized path
//!    builds it, with ABAC applied per row before any limit ([`conform`]).
//!
//! # Kill switch
//!
//! `PROXIMADB_NATIVE_TABLE_LAYER_A=0` puts registration back on the `MemTable`
//! path. Default on — but only inside the native-provider gate, which is itself
//! default OFF, so nothing changes for a default deployment.
//!
//! [`FileSystemParquetReader`]: crate::storage::persistence::filesystem::parquet_async_reader::FileSystemParquetReader

pub(crate) mod conform;
pub(crate) mod merge;
pub mod provider;

pub use provider::{LayerAScanExec, LayerATableProvider};

/// Kill switch for the streaming provider (see the module note).
const LAYER_A_ENV: &str = "PROXIMADB_NATIVE_TABLE_LAYER_A";

/// Serve native tables through [`LayerATableProvider`]? Default `true`; only an
/// explicit falsy value turns it off.
pub fn layer_a_provider_enabled() -> bool {
    !matches!(
        std::env::var(LAYER_A_ENV).unwrap_or_default().trim(),
        "0" | "false" | "FALSE" | "off" | "OFF"
    )
}

#[cfg(test)]
mod tests;
