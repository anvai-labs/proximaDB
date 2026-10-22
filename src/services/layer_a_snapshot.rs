// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! The snapshot port of the Layer A table provider (ADR-097 P3.1, PR 2).
//!
//! Today a native SQL scan asks the record store for a merged, projected result
//! (`scan_table_relational`), which reads every spill segment into memory before
//! the first row reaches the engine. The streaming provider instead takes this
//! snapshot — the partition's [`RecordScanView`] plus the handle its segments
//! are read through — and merges the layers itself, row group by row group.
//!
//! The snapshot carries **no rows from a segment**: taking it costs one lock and
//! a copy of the resident rows, never object I/O.
//!
//! [`TableRecordStore::layer_a_snapshot`](super::record_store::TableRecordStore::layer_a_snapshot)
//! returns `Ok(None)` for every store that cannot offer one; the caller then
//! keeps the materialized path, so an unsupported store degrades to today's
//! behaviour rather than to a wrong answer.

use std::sync::Arc;

use proximadb_catalog_schema::CatalogTableSchema;
use proximadb_records::{ProximaRecord, RecordScanView};

use crate::storage::persistence::filesystem::FileSystem;

/// One table partition's flush-consistent state, ready to be merged.
#[derive(Clone)]
pub struct LayerASnapshot {
    /// The partition's resident rows, memory tombstones and segment paths.
    pub view: RecordScanView,
    /// Where `view.segments` are read from. Always `Some` when there are
    /// segments; a store that cannot name it returns no snapshot at all.
    pub filesystem: Option<Arc<dyn FileSystem>>,
    /// A row belongs to the table iff its `variation_id` is unset or equals
    /// this — the same membership rule every catalog-table scan applies.
    pub table_name: String,
}

impl std::fmt::Debug for LayerASnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerASnapshot")
            .field("resident", &self.view.resident.len())
            .field("tombstones", &self.view.tombstones.len())
            .field("segments", &self.view.segments)
            .field("table_name", &self.table_name)
            .finish_non_exhaustive()
    }
}

impl LayerASnapshot {
    /// Does `record` belong to this snapshot's table?
    pub fn is_member(&self, record: &ProximaRecord) -> bool {
        record
            .variation_id
            .as_deref()
            .is_none_or(|variation| variation == self.table_name)
    }
}

/// How the caller's identity may see a table's rows (FA-c Phase 2), resolved
/// once by the same prelude `scan_table_relational` uses.
pub enum LayerARowAccess {
    /// No ABAC row filter applies.
    Unrestricted,
    /// Only rows the predicate keeps are visible. The provider evaluates it on
    /// canonical records, BEFORE any limit.
    Restricted(Box<dyn Fn(&ProximaRecord) -> bool + Send + Sync>),
}

impl std::fmt::Debug for LayerARowAccess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Unrestricted => "Unrestricted",
            Self::Restricted(_) => "Restricted",
        })
    }
}

/// A governed snapshot of one catalog table, as the native SQL path needs it.
#[derive(Debug)]
pub struct NativeLayerASnapshot {
    /// The table's catalog schema — the shape every row is conformed to.
    pub schema: CatalogTableSchema,
    /// `None` when the caller's identity is denied every row: the table is then
    /// empty for this query, and no partition state was even read.
    pub snapshot: Option<LayerASnapshot>,
    pub access: LayerARowAccess,
}
