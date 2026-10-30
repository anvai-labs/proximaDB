//! Writable in-memory relational engine (ADR-019 L5 / S5b).
//!
//! First reference implementation of ProximaDB's full
//! read+write contract surface:
//!
//! - [`RelationalReader`] from `proximadb-relational-reader` — the
//!   scan/PK-lookup path the executor drives.
//! - [`RelationalWriter`] defined here — `create_table`,
//!   `insert_rows`, `update_by_pk`, `delete_by_pk`.
//! - [`ReaderFactory`] from `proximadb-relational-executor` so the
//!   executor can resolve `Scan { table }` directly to a reader
//!   over this engine.
//!
//! Storage shape (MVP):
//!
//! ```text
//! InMemoryRelationalEngine
//!   tables: parking_lot::RwLock<HashMap<String, TableState>>
//!     TableState { schema, pk_columns, rows }
//! ```
//!
//! On `open_reader`, we **snapshot** the table's `rows` Vec into
//! the returned cursor. Long scans therefore don't block writes
//! and writes don't block scans, at the cost of point-in-time
//! semantics (no MVCC — Phase 3).
//!
//! PK lookups go straight against the live table (no snapshot)
//! since they're O(N) probes today; Phase 3 indexes them.
//!
//! Concurrency: all mutating ops take a write lock on the table
//! map; per-table state lives inside an Arc so individual tables
//! can be unlocked once we move to per-table sharded locks.

use async_trait::async_trait;
use parking_lot::RwLock;
use proximadb_data_model::ProximaValue;
use proximadb_relational_algebra::TableId;
use proximadb_relational_executor::{ExecError, ReaderFactory};
use proximadb_relational_reader::{ReaderCapabilities, ReaderError, RelationalReader, ScanContext};
use proximadb_relational_types::{Expr, NoFunctions, RelationalRow, RelationalSchema};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

pub mod transaction;
pub use transaction::{PkKey, TableWrites, Transaction, TransactionBuffer, TxReaderFactory};

// =========================================================================
// Errors
// =========================================================================

#[derive(Debug, Error, Clone, PartialEq)]
pub enum EngineError {
    #[error("table not found: {0}")]
    TableNotFound(String),

    #[error("table already exists: {0}")]
    TableExists(String),

    #[error("row arity mismatch: expected {expected}, got {actual}")]
    RowArity { expected: usize, actual: usize },

    #[error("primary-key arity mismatch: expected {expected}, got {actual}")]
    PkArity { expected: usize, actual: usize },

    #[error("primary key required for update/delete but table {0} has none")]
    NoPrimaryKey(String),

    #[error("reader error: {0}")]
    Reader(#[from] ReaderError),

    #[error("internal engine error: {0}")]
    Internal(String),
}

impl From<EngineError> for ExecError {
    fn from(e: EngineError) -> ExecError {
        match e {
            EngineError::Reader(r) => ExecError::Reader(r),
            other => ExecError::Internal(other.to_string()),
        }
    }
}

// =========================================================================
// Writer trait
// =========================================================================

/// Write-side contract every relational engine implements.
/// `RelationalReader` covers reads; this covers DDL + DML.
///
/// Implementations MUST be `Send + Sync` so they can sit behind
/// an `Arc` shared across async tasks.
pub trait RelationalWriter: Send + Sync {
    /// Create a new table. Errors if a table with this name
    /// already exists. `pk_columns` lists ordinals (within
    /// `schema`) that form the primary key; pass an empty Vec
    /// for tables without a PK.
    fn create_table(
        &self,
        name: &str,
        schema: RelationalSchema,
        pk_columns: Vec<usize>,
    ) -> Result<(), EngineError>;

    /// Drop a table. Idempotent — succeeds even if the table
    /// doesn't exist.
    fn drop_table(&self, name: &str) -> Result<(), EngineError>;

    /// Insert one or more rows. Each row must match the table's
    /// schema arity (mismatched rows return `RowArity` without
    /// inserting any). Returns the count of inserted rows.
    fn insert_rows(&self, table: &str, rows: Vec<RelationalRow>) -> Result<usize, EngineError>;

    /// Replace the row matching `key` (in the table's PK
    /// ordinals) with `new_row`. Returns `true` if a matching
    /// row was found and updated.
    fn update_by_pk(
        &self,
        table: &str,
        key: &[ProximaValue],
        new_row: RelationalRow,
    ) -> Result<bool, EngineError>;

    /// Delete the row matching `key`. Returns `true` if a
    /// matching row was found and removed.
    fn delete_by_pk(&self, table: &str, key: &[ProximaValue]) -> Result<bool, EngineError>;
}

// =========================================================================
// InMemoryRelationalEngine
// =========================================================================

/// Per-table runtime state inside the engine.
#[derive(Debug, Clone)]
struct TableState {
    schema: RelationalSchema,
    pk_columns: Vec<usize>,
    rows: Vec<RelationalRow>,
}

/// The engine itself. Always wrapped in `Arc` for sharing.
#[derive(Debug, Default)]
pub struct InMemoryRelationalEngine {
    tables: RwLock<HashMap<String, TableState>>,
}

impl InMemoryRelationalEngine {
    /// PK-shape guard shared by the PK-keyed writes: lock the table map,
    /// resolve `table`, and check that it HAS a PK matching `key`'s arity.
    fn pk_checked_state<'a>(
        tables: &'a mut HashMap<String, TableState>,
        table: &str,
        key: &[ProximaValue],
    ) -> Result<&'a mut TableState, EngineError> {
        let state = tables
            .get_mut(table)
            .ok_or_else(|| EngineError::TableNotFound(table.to_string()))?;
        if state.pk_columns.is_empty() {
            return Err(EngineError::NoPrimaryKey(table.to_string()));
        }
        if key.len() != state.pk_columns.len() {
            return Err(EngineError::PkArity {
                expected: state.pk_columns.len(),
                actual: key.len(),
            });
        }
        Ok(state)
    }
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Read-only access to a table's schema. Returns `None` if
    /// the table doesn't exist.
    pub fn schema_of(&self, name: &str) -> Option<RelationalSchema> {
        self.tables.read().get(name).map(|t| t.schema.clone())
    }

    /// Read-only access to a table's primary-key column ordinals.
    pub fn pk_columns(&self, name: &str) -> Option<Vec<usize>> {
        self.tables.read().get(name).map(|t| t.pk_columns.clone())
    }

    /// Snapshot a table's current rows. Used internally by the
    /// reader at `open` time; exposed for tests and tools.
    pub fn snapshot_rows(&self, name: &str) -> Option<Vec<RelationalRow>> {
        self.tables.read().get(name).map(|t| t.rows.clone())
    }

    /// Total row count across all tables (testing/diagnostics).
    pub fn total_rows(&self) -> usize {
        self.tables.read().values().map(|t| t.rows.len()).sum()
    }

    fn find_pk_index(state: &TableState, key: &[ProximaValue]) -> Option<usize> {
        state.rows.iter().position(|row| {
            state
                .pk_columns
                .iter()
                .zip(key.iter())
                .all(|(&idx, expected)| &row[idx] == expected)
        })
    }
}

impl RelationalWriter for InMemoryRelationalEngine {
    fn create_table(
        &self,
        name: &str,
        schema: RelationalSchema,
        pk_columns: Vec<usize>,
    ) -> Result<(), EngineError> {
        // Validate that pk_columns are in bounds.
        for &idx in &pk_columns {
            if idx >= schema.len() {
                return Err(EngineError::Internal(format!(
                    "pk column ordinal {idx} out of bounds for schema with {} columns",
                    schema.len()
                )));
            }
        }
        let mut tables = self.tables.write();
        if tables.contains_key(name) {
            return Err(EngineError::TableExists(name.to_string()));
        }
        tables.insert(
            name.to_string(),
            TableState {
                schema,
                pk_columns,
                rows: Vec::new(),
            },
        );
        Ok(())
    }

    fn drop_table(&self, name: &str) -> Result<(), EngineError> {
        self.tables.write().remove(name);
        Ok(())
    }

    fn insert_rows(&self, table: &str, rows: Vec<RelationalRow>) -> Result<usize, EngineError> {
        let mut tables = self.tables.write();
        let state = tables
            .get_mut(table)
            .ok_or_else(|| EngineError::TableNotFound(table.to_string()))?;
        let expected = state.schema.len();
        for r in &rows {
            if r.len() != expected {
                return Err(EngineError::RowArity {
                    expected,
                    actual: r.len(),
                });
            }
        }
        let n = rows.len();
        state.rows.extend(rows);
        Ok(n)
    }

    fn update_by_pk(
        &self,
        table: &str,
        key: &[ProximaValue],
        new_row: RelationalRow,
    ) -> Result<bool, EngineError> {
        let mut tables = self.tables.write();
        let state = Self::pk_checked_state(&mut tables, table, key)?;
        if new_row.len() != state.schema.len() {
            return Err(EngineError::RowArity {
                expected: state.schema.len(),
                actual: new_row.len(),
            });
        }
        Ok(match Self::find_pk_index(state, key) {
            Some(idx) => {
                state.rows[idx] = new_row;
                true
            }
            None => false,
        })
    }

    fn delete_by_pk(&self, table: &str, key: &[ProximaValue]) -> Result<bool, EngineError> {
        let mut tables = self.tables.write();
        let state = Self::pk_checked_state(&mut tables, table, key)?;
        Ok(match Self::find_pk_index(state, key) {
            Some(idx) => {
                state.rows.remove(idx);
                true
            }
            None => false,
        })
    }
}

// =========================================================================
// EngineReaderFactory — bridges the engine to the executor
// =========================================================================

/// Adapter exposing the engine through the executor's
/// [`ReaderFactory`] trait. The factory holds a clone of the
/// `Arc`; constructing a reader copies the table's rows into the
/// reader at `open_reader` time.
#[derive(Debug, Clone)]
pub struct EngineReaderFactory {
    engine: Arc<InMemoryRelationalEngine>,
}

impl EngineReaderFactory {
    pub fn new(engine: Arc<InMemoryRelationalEngine>) -> Self {
        Self { engine }
    }
}

impl ReaderFactory for EngineReaderFactory {
    fn open_reader(&self, table: &TableId) -> Result<Box<dyn RelationalReader>, ExecError> {
        let name = &table.name;
        let snapshot = self
            .engine
            .tables
            .read()
            .get(name)
            .ok_or_else(|| ExecError::from(EngineError::TableNotFound(name.clone())))?
            .clone();
        Ok(Box::new(InMemoryReader::new(
            name.clone(),
            snapshot.schema,
            snapshot.pk_columns,
            snapshot.rows,
        )))
    }
}

// =========================================================================
// InMemoryReader — the cursor over a snapshot
// =========================================================================

/// Scan-state of one open [`VecRowScan`].
pub(crate) struct OpenState {
    output_schema: RelationalSchema,
    projection_indices: Vec<usize>,
    predicate: Option<Expr>,
    limit: Option<u64>,
    cursor: usize,
    rows_emitted: u64,
}

/// Scan core over a materialized `Vec<RelationalRow>`: projection
/// resolution, predicate type-check + row filter, LIMIT, and O(N) PK
/// point-lookup. Shared by [`InMemoryReader`] (engine snapshot) and
/// [`transaction::TxReader`] (engine ∪ transaction buffer) — they differ
/// only in how `rows` is populated.
pub(crate) struct VecRowScan {
    full_schema: RelationalSchema,
    pk_columns: Vec<usize>,
    rows: Vec<RelationalRow>,
    open_state: Option<OpenState>,
}

impl VecRowScan {
    pub(crate) fn new(
        full_schema: RelationalSchema,
        pk_columns: Vec<usize>,
        rows: Vec<RelationalRow>,
    ) -> Self {
        Self {
            full_schema,
            pk_columns,
            rows,
            open_state: None,
        }
    }

    fn resolve_projection(
        &self,
        projection: &Option<Vec<String>>,
    ) -> Result<(RelationalSchema, Vec<usize>), ReaderError> {
        let Some(names) = projection else {
            return Ok((
                (self.full_schema.clone()),
                (0..self.full_schema.len()).collect(),
            ));
        };
        let mut cols = Vec::with_capacity(names.len());
        let mut indices = Vec::with_capacity(names.len());
        for n in names {
            let (idx, info) = self
                .full_schema
                .column_by_name(n)
                .ok_or_else(|| ReaderError::InvalidProjection(n.clone()))?;
            cols.push(info.clone());
            indices.push(idx);
        }
        Ok((RelationalSchema::new(cols), indices))
    }

    pub(crate) fn capabilities(&self) -> ReaderCapabilities {
        // Same shape as the test VecReader: every flag honoured.
        ReaderCapabilities::full().with_pk_lookup(!self.pk_columns.is_empty())
    }

    pub(crate) fn schema(&self) -> &RelationalSchema {
        match &self.open_state {
            Some(s) => &s.output_schema,
            None => &self.full_schema,
        }
    }

    pub(crate) fn open(&mut self, ctx: &ScanContext) -> Result<(), ReaderError> {
        let (output_schema, projection_indices) = self.resolve_projection(&ctx.projection)?;
        if let Some(pred) = &ctx.predicate {
            pred.type_check(&self.full_schema)
                .map_err(ReaderError::PredicateEval)?;
        }
        self.open_state = Some(OpenState {
            output_schema,
            projection_indices,
            predicate: ctx.predicate.clone(),
            limit: ctx.limit,
            cursor: 0,
            rows_emitted: 0,
        });
        Ok(())
    }

    pub(crate) fn next_row(&mut self) -> Result<Option<RelationalRow>, ReaderError> {
        let state = self.open_state.as_mut().ok_or(ReaderError::NotOpen)?;
        loop {
            if state.cursor >= self.rows.len() {
                return Ok(None);
            }
            if let Some(limit) = state.limit
                && state.rows_emitted >= limit
            {
                return Ok(None);
            }
            let row = &self.rows[state.cursor];
            state.cursor += 1;
            if let Some(pred) = &state.predicate {
                let v = pred
                    .eval(row, &NoFunctions)
                    .map_err(ReaderError::PredicateEval)?;
                if !matches!(v, ProximaValue::Boolean(true)) {
                    continue;
                }
            }
            let projected: RelationalRow = state
                .projection_indices
                .iter()
                .map(|&i| row[i].clone())
                .collect();
            state.rows_emitted += 1;
            return Ok(Some(projected));
        }
    }

    pub(crate) fn lookup_pk(
        &self,
        key: &[ProximaValue],
    ) -> Result<Option<RelationalRow>, ReaderError> {
        if self.pk_columns.is_empty() {
            return Err(ReaderError::PkLookupUnsupported);
        }
        if key.len() != self.pk_columns.len() {
            return Err(ReaderError::PkArityMismatch {
                expected: self.pk_columns.len(),
                actual: key.len(),
            });
        }
        for row in &self.rows {
            let matches = self
                .pk_columns
                .iter()
                .zip(key.iter())
                .all(|(&idx, expected)| &row[idx] == expected);
            if matches {
                return Ok(Some(row.clone()));
            }
        }
        Ok(None)
    }

    pub(crate) fn close(&mut self) {
        self.open_state = None;
    }
}

/// Read cursor. Identical contract to the test `VecReader` in
/// the reader crate, but lives in the engine crate so we can
/// evolve it independently (e.g. add bloom/zone-map metadata
/// in Phase 3). Scan mechanics live in the shared [`VecRowScan`].
pub struct InMemoryReader {
    /// Table name for diagnostics. Read by future EXPLAIN integration;
    /// currently constructed for observability and kept against
    /// dead-code lint by `#[allow]`.
    #[allow(dead_code)]
    table_name: String,
    core: VecRowScan,
}

impl InMemoryReader {
    fn new(
        table_name: String,
        schema: RelationalSchema,
        pk_columns: Vec<usize>,
        rows: Vec<RelationalRow>,
    ) -> Self {
        Self {
            table_name,
            core: VecRowScan::new(schema, pk_columns, rows),
        }
    }
}

#[async_trait]
impl RelationalReader for InMemoryReader {
    fn name(&self) -> &'static str {
        // Cannot return the table-specific name — the trait
        // wants a 'static str. Use the engine-level identifier;
        // callers consult `.schema()` for table-specific info.
        "in_memory"
    }

    fn capabilities(&self) -> ReaderCapabilities {
        self.core.capabilities()
    }

    fn schema(&self) -> &RelationalSchema {
        self.core.schema()
    }

    async fn open(&mut self, ctx: &ScanContext) -> Result<(), ReaderError> {
        self.core.open(ctx)
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ReaderError> {
        self.core.next_row()
    }

    async fn lookup_pk(&self, key: &[ProximaValue]) -> Result<Option<RelationalRow>, ReaderError> {
        self.core.lookup_pk(key)
    }

    async fn close(&mut self) -> Result<(), ReaderError> {
        self.core.close();
        Ok(())
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_data_model::ProximaType;
    use proximadb_relational_algebra::TableId;
    use proximadb_relational_executor::{ExecutionContext, build_executor, collect};
    use proximadb_relational_planner::{Planner, StaticCapabilities};
    use proximadb_relational_reader::ReaderCapabilities as RC;
    use proximadb_relational_types::ColumnInfo;

    /// Stamped prologue: a `users` table (PK on `id`) populated with `rows`.
    fn users_engine(rows: Vec<RelationalRow>) -> Arc<InMemoryRelationalEngine> {
        let engine = InMemoryRelationalEngine::new();
        engine
            .create_table("users", users_schema(), vec![0])
            .unwrap();
        engine.insert_rows("users", rows).unwrap();
        engine
    }

    /// Frontend catalog over the engine (SQL e2e tests).
    struct EngineCatalog(Arc<InMemoryRelationalEngine>);
    impl proximadb_relational_frontend::CatalogLookup for EngineCatalog {
        fn lookup_table(&self, name: &str) -> Option<std::sync::Arc<RelationalSchema>> {
            self.0.schema_of(name).map(std::sync::Arc::new)
        }
    }

    /// Lower `sql` → plan → execute against `engine`, returning the rows.
    /// Plans with a capability resolver that knows the users PK (`id`).
    async fn run_sql(engine: &Arc<InMemoryRelationalEngine>, sql: &str) -> Vec<RelationalRow> {
        let catalog = EngineCatalog(engine.clone());
        let logical = proximadb_relational_frontend::lower_sql(sql, &catalog).unwrap();
        let planner = Planner::new(StaticCapabilities {
            caps: RC::full(),
            pk_columns: vec![0],
            secondary_columns: Vec::new(),
        });
        let physical = planner.plan(logical).unwrap();
        let factory = EngineReaderFactory::new(engine.clone());
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(physical, &factory, &ctx).unwrap();
        exec.open().await.unwrap();
        collect(&mut *exec).await.unwrap()
    }

    fn users_schema() -> RelationalSchema {
        RelationalSchema::new(vec![
            ColumnInfo::new("id", ProximaType::Int64, false),
            ColumnInfo::new("name", ProximaType::String, true),
            ColumnInfo::new("age", ProximaType::Int32, true),
        ])
    }

    fn user_row(id: i64, name: &str, age: i32) -> RelationalRow {
        vec![
            ProximaValue::Int64(id),
            ProximaValue::String(name.into()),
            ProximaValue::Int32(age),
        ]
    }

    // ----- create_table / DDL ----------------------------------------

    #[test]
    fn create_table_then_schema_lookup() {
        let engine = InMemoryRelationalEngine::new();
        engine
            .create_table("users", users_schema(), vec![0])
            .unwrap();
        assert_eq!(engine.schema_of("users").unwrap().len(), 3);
        assert_eq!(engine.pk_columns("users"), Some(vec![0]));
    }

    #[test]
    fn create_table_twice_errors() {
        let engine = InMemoryRelationalEngine::new();
        engine
            .create_table("users", users_schema(), vec![0])
            .unwrap();
        let err = engine
            .create_table("users", users_schema(), vec![0])
            .unwrap_err();
        assert!(matches!(err, EngineError::TableExists(_)));
    }

    #[test]
    fn drop_table_is_idempotent() {
        let engine = InMemoryRelationalEngine::new();
        engine.drop_table("nonexistent").unwrap();
        engine
            .create_table("users", users_schema(), vec![0])
            .unwrap();
        engine.drop_table("users").unwrap();
        assert!(engine.schema_of("users").is_none());
    }

    #[test]
    fn pk_column_ordinal_out_of_bounds_errors() {
        let engine = InMemoryRelationalEngine::new();
        let err = engine
            .create_table("users", users_schema(), vec![99])
            .unwrap_err();
        assert!(matches!(err, EngineError::Internal(_)));
    }

    // ----- insert / update / delete ----------------------------------

    #[test]
    fn insert_rows_appends() {
        let engine = InMemoryRelationalEngine::new();
        engine
            .create_table("users", users_schema(), vec![0])
            .unwrap();
        let n = engine
            .insert_rows(
                "users",
                vec![user_row(1, "alice", 30), user_row(2, "bob", 25)],
            )
            .unwrap();
        assert_eq!(n, 2);
        assert_eq!(engine.total_rows(), 2);
    }

    #[test]
    fn insert_arity_mismatch_rejects_all() {
        let engine = InMemoryRelationalEngine::new();
        engine
            .create_table("users", users_schema(), vec![0])
            .unwrap();
        let err = engine
            .insert_rows(
                "users",
                vec![
                    user_row(1, "alice", 30),
                    vec![ProximaValue::Int64(2)], // wrong arity
                ],
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::RowArity { .. }));
        // No partial write.
        assert_eq!(engine.total_rows(), 0);
    }

    #[test]
    fn update_by_pk_replaces_row() {
        let engine = users_engine(vec![user_row(1, "alice", 30)]);
        let updated = engine
            .update_by_pk(
                "users",
                &[ProximaValue::Int64(1)],
                user_row(1, "alice_v2", 31),
            )
            .unwrap();
        assert!(updated);
        let rows = engine.snapshot_rows("users").unwrap();
        assert_eq!(rows[0][1], ProximaValue::String("alice_v2".into()));
    }

    #[test]
    fn update_missing_pk_returns_false() {
        let engine = users_engine(Vec::new());
        let updated = engine
            .update_by_pk(
                "users",
                &[ProximaValue::Int64(999)],
                user_row(999, "ghost", 0),
            )
            .unwrap();
        assert!(!updated);
    }

    #[test]
    fn delete_by_pk_removes_row() {
        let engine = users_engine(vec![user_row(1, "alice", 30), user_row(2, "bob", 25)]);
        let deleted = engine
            .delete_by_pk("users", &[ProximaValue::Int64(1)])
            .unwrap();
        assert!(deleted);
        let rows = engine.snapshot_rows("users").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], ProximaValue::Int64(2));
    }

    #[test]
    fn delete_without_pk_errors() {
        let engine = InMemoryRelationalEngine::new();
        // pk_columns empty.
        engine
            .create_table("users", users_schema(), vec![])
            .unwrap();
        let err = engine
            .delete_by_pk("users", &[ProximaValue::Int64(1)])
            .unwrap_err();
        assert!(matches!(err, EngineError::NoPrimaryKey(_)));
    }

    #[test]
    fn pk_arity_mismatch_errors() {
        let engine = InMemoryRelationalEngine::new();
        engine
            .create_table("users", users_schema(), vec![0])
            .unwrap();
        let err = engine
            .delete_by_pk("users", &[ProximaValue::Int64(1), ProximaValue::Int64(2)])
            .unwrap_err();
        assert!(matches!(err, EngineError::PkArity { .. }));
    }

    // ----- ReaderFactory + executor integration ----------------------

    #[tokio::test]
    async fn scan_via_factory_emits_all_rows() {
        let engine = users_engine(vec![
            user_row(1, "alice", 30),
            user_row(2, "bob", 25),
            user_row(3, "carol", 40),
        ]);
        let factory = EngineReaderFactory::new(engine.clone());
        let plan = proximadb_relational_planner::PhysicalPlan::Scan {
            table: TableId::new("users"),
            output_schema: users_schema(),
            projection: None,
            predicate: None,
            limit: None,
            access: proximadb_relational_planner::ScanAccess::FullScan,
        };
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &factory, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[tokio::test]
    async fn end_to_end_select_via_frontend() {
        // Round-trip: create+insert via writer, lower SQL via
        // frontend, plan, execute, verify result.
        let engine = users_engine(vec![
            user_row(1, "alice", 30),
            user_row(2, "bob", 25),
            user_row(3, "carol", 40),
        ]);
        // SELECT name FROM users WHERE age > 25
        let rows = run_sql(&engine, "SELECT name FROM users WHERE age > 25").await;
        // alice (30) + carol (40); bob (25) excluded.
        assert_eq!(rows.len(), 2);
        let names: Vec<String> = rows
            .iter()
            .map(|r| match &r[0] {
                ProximaValue::String(s) => s.clone(),
                _ => panic!(),
            })
            .collect();
        assert!(names.contains(&"alice".to_string()));
        assert!(names.contains(&"carol".to_string()));
    }

    #[tokio::test]
    async fn end_to_end_pk_lookup_with_projection() {
        // SELECT name FROM users WHERE id = 2
        // Exercises the full pipeline: frontend lowers SQL, planner
        // rewrites `id = lit` into ScanAccess::PkLookup and pushes
        // projection through the PkLookup, executor narrows the row
        // returned by `lookup_pk`. Final result must be a single-row,
        // single-column ("bob") output.
        let engine = users_engine(vec![
            user_row(1, "alice", 30),
            user_row(2, "bob", 25),
            user_row(3, "carol", 40),
        ]);
        let rows = run_sql(&engine, "SELECT name FROM users WHERE id = 2").await;
        assert_eq!(rows.len(), 1, "PK lookup must return exactly one row");
        assert_eq!(rows[0].len(), 1, "row must be narrowed to one column");
        assert_eq!(rows[0][0], ProximaValue::String("bob".into()));
    }

    #[tokio::test]
    async fn end_to_end_count_star() {
        let engine = users_engine(vec![
            user_row(1, "alice", 30),
            user_row(2, "bob", 25),
            user_row(3, "carol", 40),
        ]);
        let rows = run_sql(&engine, "SELECT COUNT(*) FROM users").await;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], ProximaValue::Int64(3));
    }
}
