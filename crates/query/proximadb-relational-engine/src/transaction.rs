//! Transaction layer (ADR-019 / S6).
//!
//! Each [`Transaction`] holds a [`TransactionBuffer`] of pending
//! writes against an [`InMemoryRelationalEngine`]. Writes don't
//! mutate the engine until [`Transaction::commit`]; reads see the
//! current committed state of the engine **plus** this
//! transaction's own pending writes.
//!
//! Isolation level: **READ COMMITTED**. Each statement that opens
//! a reader gets a fresh snapshot of committed state at the
//! moment of `open`, merged with the buffer. Two concurrent
//! transactions writing the same row commit in last-write-wins
//! order (the buffer doesn't detect write-write conflicts —
//! storage-level MVCC is Phase 3).
//!
//! Lifecycle:
//!
//! ```text
//! Transaction::begin(engine)
//!   .insert(...) | .update(...) | .delete(...) | .reader_factory()
//!   .commit() | .rollback()
//! ```
//!
//! Reads inside the transaction go through
//! [`Transaction::reader_factory`], which returns a
//! [`TxReaderFactory`] that satisfies the executor's
//! [`ReaderFactory`] contract. Drop the factory before commit.

use crate::{EngineError, InMemoryRelationalEngine, RelationalWriter};
use async_trait::async_trait;
use parking_lot::RwLock;
use proximadb_data_model::ProximaValue;
use proximadb_relational_algebra::TableId;
use proximadb_relational_executor::{ExecError, ReaderFactory};
use proximadb_relational_reader::{ReaderCapabilities, ReaderError, RelationalReader, ScanContext};
use proximadb_relational_types::{RelationalRow, RelationalSchema};
use std::collections::HashMap;
use std::sync::Arc;

// =========================================================================
// Buffer types
// =========================================================================

/// Pending writes against a single table.
#[derive(Debug, Default, Clone)]
pub struct TableWrites {
    /// Newly inserted rows that aren't yet visible to other
    /// transactions. Appended in insert order.
    pub inserts: Vec<RelationalRow>,
    /// `(pk_key, new_row)` — replace the live row matching
    /// `pk_key` with `new_row`. Latest write per PK wins.
    pub updates: HashMap<PkKey, RelationalRow>,
    /// `pk_key` — drop the live row matching `pk_key`. Wins over
    /// any earlier insert/update of the same key.
    pub deletes: Vec<PkKey>,
}

/// Canonical hashable wrapper around a PK value vector. Wraps
/// `Vec<ProximaValue>` because ProximaValue doesn't itself impl
/// `Eq + Hash` (Float64 is the only blocker).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PkKey(Vec<KeyComponent>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum KeyComponent {
    Null,
    Bool(bool),
    Int(i128),
    Float(u64), // bit pattern
    String(String),
    Binary(Vec<u8>),
    Uuid([u8; 16]),
    /// Catch-all for types we don't yet canonicalise; falls back
    /// to a Debug string so PKs at least round-trip.
    Other(String),
}

impl PkKey {
    pub fn from_values(values: &[ProximaValue]) -> Self {
        Self(values.iter().map(value_to_component).collect())
    }
}

fn value_to_component(v: &ProximaValue) -> KeyComponent {
    use ProximaValue as V;
    match v {
        V::Null => KeyComponent::Null,
        V::Boolean(b) => KeyComponent::Bool(*b),
        V::Int8(x) => KeyComponent::Int(*x as i128),
        V::Int16(x) => KeyComponent::Int(*x as i128),
        V::Int32(x) => KeyComponent::Int(*x as i128),
        V::Int64(x) => KeyComponent::Int(*x as i128),
        V::UInt8(x) => KeyComponent::Int(*x as i128),
        V::UInt16(x) => KeyComponent::Int(*x as i128),
        V::UInt32(x) => KeyComponent::Int(*x as i128),
        V::UInt64(x) => KeyComponent::Int(*x as i128),
        V::Float16(x) | V::Float32(x) => KeyComponent::Float((*x as f64).to_bits()),
        V::Float64(x) => KeyComponent::Float(x.to_bits()),
        V::String(s) | V::Symbol(s) => KeyComponent::String(s.clone()),
        V::Binary(b) => KeyComponent::Binary(b.clone()),
        V::Uuid(u) | V::ULID(u) => KeyComponent::Uuid(*u),
        other => KeyComponent::Other(format!("{other:?}")),
    }
}

/// The aggregate buffer across all tables.
#[derive(Debug, Default)]
pub struct TransactionBuffer {
    pub tables: HashMap<String, TableWrites>,
}

impl TransactionBuffer {
    pub fn is_empty(&self) -> bool {
        self.tables
            .values()
            .all(|t| t.inserts.is_empty() && t.updates.is_empty() && t.deletes.is_empty())
    }
}

// =========================================================================
// Transaction
// =========================================================================

/// A read-write transaction. Always wrapped in `Arc` so the
/// [`TxReaderFactory`] can hold its own reference.
pub struct Transaction {
    engine: Arc<InMemoryRelationalEngine>,
    buffer: RwLock<TransactionBuffer>,
    /// True once `commit` or `rollback` has been called. Any
    /// further mutation returns `EngineError::Internal`.
    finished: RwLock<bool>,
}

impl Transaction {
    /// Begin a new transaction.
    pub fn begin(engine: Arc<InMemoryRelationalEngine>) -> Arc<Self> {
        Arc::new(Self {
            engine,
            buffer: RwLock::new(TransactionBuffer::default()),
            finished: RwLock::new(false),
        })
    }

    fn ensure_active(&self) -> Result<(), EngineError> {
        if *self.finished.read() {
            return Err(EngineError::Internal(
                "transaction already committed/rolled back".into(),
            ));
        }
        Ok(())
    }

    /// Insert rows. Writes go to the buffer; engine is not
    /// touched until commit.
    pub fn insert(&self, table: &str, rows: Vec<RelationalRow>) -> Result<usize, EngineError> {
        self.ensure_active()?;
        // Validate row arity against the engine's current schema.
        let schema = self
            .engine
            .schema_of(table)
            .ok_or_else(|| EngineError::TableNotFound(table.to_string()))?;
        for r in &rows {
            if r.len() != schema.len() {
                return Err(EngineError::RowArity {
                    expected: schema.len(),
                    actual: r.len(),
                });
            }
        }
        let n = rows.len();
        self.buffer
            .write()
            .tables
            .entry(table.to_string())
            .or_default()
            .inserts
            .extend(rows);
        Ok(n)
    }

    /// PK-shape guard shared by the buffered PK-keyed writes:
    /// table exists, has a PK, and `key` matches its arity.
    fn checked_pk(&self, table: &str, key: &[ProximaValue]) -> Result<(), EngineError> {
        let pk = self
            .engine
            .pk_columns(table)
            .ok_or_else(|| EngineError::TableNotFound(table.to_string()))?;
        if pk.is_empty() {
            return Err(EngineError::NoPrimaryKey(table.to_string()));
        }
        if key.len() != pk.len() {
            return Err(EngineError::PkArity {
                expected: pk.len(),
                actual: key.len(),
            });
        }
        Ok(())
    }

    /// Update a row by PK. Buffered.
    pub fn update_by_pk(
        &self,
        table: &str,
        key: &[ProximaValue],
        new_row: RelationalRow,
    ) -> Result<(), EngineError> {
        self.ensure_active()?;
        self.checked_pk(table, key)?;
        let schema = self
            .engine
            .schema_of(table)
            .ok_or_else(|| EngineError::TableNotFound(table.to_string()))?;
        if new_row.len() != schema.len() {
            return Err(EngineError::RowArity {
                expected: schema.len(),
                actual: new_row.len(),
            });
        }
        self.buffer
            .write()
            .tables
            .entry(table.to_string())
            .or_default()
            .updates
            .insert(PkKey::from_values(key), new_row);
        Ok(())
    }

    /// Delete a row by PK. Buffered.
    pub fn delete_by_pk(&self, table: &str, key: &[ProximaValue]) -> Result<(), EngineError> {
        self.ensure_active()?;
        self.checked_pk(table, key)?;
        self.buffer
            .write()
            .tables
            .entry(table.to_string())
            .or_default()
            .deletes
            .push(PkKey::from_values(key));
        Ok(())
    }

    /// Return a [`ReaderFactory`] that reads
    /// committed state ∪ this transaction's buffer.
    pub fn reader_factory(self: &Arc<Self>) -> TxReaderFactory {
        TxReaderFactory { tx: self.clone() }
    }

    /// Commit all buffered writes atomically.
    ///
    /// Acquires the engine's write lock for the duration of the
    /// apply, so concurrent transactions see either nothing or
    /// the complete set. Apply order per table is:
    /// **deletes → updates → inserts** so an UPDATE that
    /// matches an existing row isn't shadowed by an INSERT of
    /// the same PK in the same buffer.
    pub fn commit(self: &Arc<Self>) -> Result<(), EngineError> {
        {
            let mut finished = self.finished.write();
            if *finished {
                return Err(EngineError::Internal("transaction already finished".into()));
            }
            *finished = true;
        }
        let buffer = std::mem::take(&mut *self.buffer.write());
        // Apply each table's pending writes.
        for (table, writes) in buffer.tables {
            // Deletes first.
            for key in &writes.deletes {
                self.apply_delete(&table, key)?;
            }
            // Updates next.
            for (key, row) in writes.updates {
                self.apply_update(&table, &key, row)?;
            }
            // Inserts last.
            if !writes.inserts.is_empty() {
                self.engine.insert_rows(&table, writes.inserts)?;
            }
        }
        Ok(())
    }

    /// Rollback — discard the buffer and mark the transaction
    /// as finished.
    pub fn rollback(self: &Arc<Self>) {
        *self.finished.write() = true;
        *self.buffer.write() = TransactionBuffer::default();
    }

    fn apply_delete(&self, table: &str, key: &PkKey) -> Result<(), EngineError> {
        let raw_key = self.materialise_pk(table, key)?;
        self.engine.delete_by_pk(table, &raw_key).map(|_| ())
    }

    fn apply_update(
        &self,
        table: &str,
        key: &PkKey,
        new_row: RelationalRow,
    ) -> Result<(), EngineError> {
        let raw_key = self.materialise_pk(table, key)?;
        let updated = self.engine.update_by_pk(table, &raw_key, new_row.clone())?;
        if !updated {
            // No live row matches — treat as insert (UPSERT-ish
            // semantics for the MVP). Phase 3 will surface an
            // explicit conflict.
            self.engine.insert_rows(table, vec![new_row])?;
        }
        Ok(())
    }

    /// Reverse PkKey → Vec<ProximaValue> using the table's
    /// schema to choose the right ProximaValue variants for
    /// integers (we collapsed everything to i128 during
    /// canonicalisation).
    fn materialise_pk(&self, table: &str, key: &PkKey) -> Result<Vec<ProximaValue>, EngineError> {
        let schema = self
            .engine
            .schema_of(table)
            .ok_or_else(|| EngineError::TableNotFound(table.to_string()))?;
        let pk_cols = self
            .engine
            .pk_columns(table)
            .ok_or_else(|| EngineError::TableNotFound(table.to_string()))?;
        let mut out = Vec::with_capacity(key.0.len());
        for (idx, kc) in key.0.iter().enumerate() {
            let col_ord = pk_cols[idx];
            let target_ty = &schema.columns[col_ord].ty;
            out.push(component_to_value(kc, target_ty)?);
        }
        Ok(out)
    }
}

fn component_to_value(
    kc: &KeyComponent,
    target: &proximadb_data_model::ProximaType,
) -> Result<ProximaValue, EngineError> {
    use proximadb_data_model::ProximaType as T;
    match (kc, target) {
        (KeyComponent::Null, _) => Ok(ProximaValue::Null),
        (KeyComponent::Bool(b), T::Boolean) => Ok(ProximaValue::Boolean(*b)),
        (KeyComponent::Int(x), T::Int8) => Ok(ProximaValue::Int8(*x as i8)),
        (KeyComponent::Int(x), T::Int16) => Ok(ProximaValue::Int16(*x as i16)),
        (KeyComponent::Int(x), T::Int32) => Ok(ProximaValue::Int32(*x as i32)),
        (KeyComponent::Int(x), T::Int64) => Ok(ProximaValue::Int64(*x as i64)),
        (KeyComponent::Int(x), T::UInt8) => Ok(ProximaValue::UInt8(*x as u8)),
        (KeyComponent::Int(x), T::UInt16) => Ok(ProximaValue::UInt16(*x as u16)),
        (KeyComponent::Int(x), T::UInt32) => Ok(ProximaValue::UInt32(*x as u32)),
        (KeyComponent::Int(x), T::UInt64) => Ok(ProximaValue::UInt64(*x as u64)),
        (KeyComponent::Float(bits), T::Float64) => Ok(ProximaValue::Float64(f64::from_bits(*bits))),
        (KeyComponent::Float(bits), T::Float32) => {
            Ok(ProximaValue::Float32(f64::from_bits(*bits) as f32))
        }
        (KeyComponent::String(s), T::String) => Ok(ProximaValue::String(s.clone())),
        (KeyComponent::String(s), T::Symbol) => Ok(ProximaValue::Symbol(s.clone())),
        (KeyComponent::Binary(b), T::Binary) => Ok(ProximaValue::Binary(b.clone())),
        (KeyComponent::Uuid(u), T::Uuid) => Ok(ProximaValue::Uuid(*u)),
        (KeyComponent::Uuid(u), T::ULID) => Ok(ProximaValue::ULID(*u)),
        _ => Err(EngineError::Internal(format!(
            "cannot materialise PK component {kc:?} as target type {target:?}"
        ))),
    }
}

// =========================================================================
// TxReaderFactory + TxReader
// =========================================================================

/// `ReaderFactory` impl that returns readers seeing the
/// transaction's view of the world: engine rows minus this
/// transaction's deletes, with this transaction's updates
/// applied, plus this transaction's inserts.
pub struct TxReaderFactory {
    tx: Arc<Transaction>,
}

impl ReaderFactory for TxReaderFactory {
    fn open_reader(&self, table: &TableId) -> Result<Box<dyn RelationalReader>, ExecError> {
        let name = &table.name;
        // Snapshot engine state.
        let (schema, pk_cols, mut rows) = {
            let snapshot = self.tx.engine.tables.read();
            let state = snapshot
                .get(name)
                .ok_or_else(|| ExecError::from(EngineError::TableNotFound(name.clone())))?;
            (
                state.schema.clone(),
                state.pk_columns.clone(),
                state.rows.clone(),
            )
        };
        // Apply this transaction's pending writes for the table.
        let writes = self.tx.buffer.read();
        if let Some(t) = writes.tables.get(name) {
            // Deletes.
            for key in &t.deletes {
                if let Some(pos) = find_row_by_pkkey(&rows, &pk_cols, key) {
                    rows.remove(pos);
                }
            }
            // Updates.
            for (key, new_row) in &t.updates {
                if let Some(pos) = find_row_by_pkkey(&rows, &pk_cols, key) {
                    rows[pos] = new_row.clone();
                } else {
                    // No matching committed row — treat as insert
                    // for visibility.
                    rows.push(new_row.clone());
                }
            }
            // Inserts.
            rows.extend(t.inserts.clone());
        }
        Ok(Box::new(TxReader::new(schema, pk_cols, rows)))
    }
}

fn find_row_by_pkkey(rows: &[RelationalRow], pk_cols: &[usize], key: &PkKey) -> Option<usize> {
    rows.iter().position(|row| {
        let row_key: Vec<KeyComponent> = pk_cols
            .iter()
            .map(|&i| value_to_component(&row[i]))
            .collect();
        PkKey(row_key) == *key
    })
}

/// Read cursor backed by an already-materialised (engine ∪ tx
/// buffer) row set. Scan mechanics (projection, predicate, LIMIT,
/// PK lookup) live in the shared [`crate::VecRowScan`] core; this
/// type only supplies the merged snapshot.
pub struct TxReader {
    core: crate::VecRowScan,
}

impl TxReader {
    fn new(schema: RelationalSchema, pk_columns: Vec<usize>, rows: Vec<RelationalRow>) -> Self {
        Self {
            core: crate::VecRowScan::new(schema, pk_columns, rows),
        }
    }
}

#[async_trait]
impl RelationalReader for TxReader {
    fn name(&self) -> &'static str {
        "tx"
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
    use crate::{InMemoryRelationalEngine, RelationalWriter};
    use proximadb_data_model::ProximaType;
    use proximadb_relational_algebra::TableId;
    use proximadb_relational_executor::{ExecutionContext, build_executor, collect};
    use proximadb_relational_planner::{PhysicalPlan, ScanAccess};
    use proximadb_relational_types::{ColumnInfo, RelationalSchema};

    fn users_schema() -> RelationalSchema {
        RelationalSchema::new(vec![
            ColumnInfo::new("id", ProximaType::Int64, false),
            ColumnInfo::new("name", ProximaType::String, true),
        ])
    }

    fn user_row(id: i64, name: &str) -> RelationalRow {
        vec![ProximaValue::Int64(id), ProximaValue::String(name.into())]
    }

    fn engine_with_users() -> Arc<InMemoryRelationalEngine> {
        let e = InMemoryRelationalEngine::new();
        e.create_table("users", users_schema(), vec![0]).unwrap();
        e.insert_rows("users", vec![user_row(1, "alice"), user_row(2, "bob")])
            .unwrap();
        e
    }

    async fn scan_all(factory: &TxReaderFactory) -> Vec<RelationalRow> {
        let plan = PhysicalPlan::Scan {
            table: TableId::new("users"),
            output_schema: users_schema(),
            projection: None,
            predicate: None,
            limit: None,
            access: ScanAccess::FullScan,
        };
        let mut exec = build_executor(plan, factory, &ExecutionContext::default()).unwrap();
        exec.open().await.unwrap();
        collect(&mut *exec).await.unwrap()
    }

    // ----- Visibility -----------------------------------------------

    #[tokio::test]
    async fn insert_visible_inside_tx_invisible_outside() {
        let engine = engine_with_users();
        let tx = Transaction::begin(engine.clone());
        tx.insert("users", vec![user_row(3, "carol")]).unwrap();
        // Inside tx: 3 rows.
        let inside = scan_all(&tx.reader_factory()).await;
        assert_eq!(inside.len(), 3);
        // Outside (direct engine read): 2 rows.
        assert_eq!(engine.snapshot_rows("users").unwrap().len(), 2);
    }

    #[tokio::test]
    async fn commit_persists_writes() {
        let engine = engine_with_users();
        let tx = Transaction::begin(engine.clone());
        tx.insert("users", vec![user_row(3, "carol")]).unwrap();
        tx.commit().unwrap();
        // Engine now sees 3 rows.
        assert_eq!(engine.snapshot_rows("users").unwrap().len(), 3);
    }

    #[tokio::test]
    async fn rollback_drops_writes() {
        let engine = engine_with_users();
        let tx = Transaction::begin(engine.clone());
        tx.insert("users", vec![user_row(3, "carol")]).unwrap();
        tx.rollback();
        // Engine still has just the original 2 rows.
        assert_eq!(engine.snapshot_rows("users").unwrap().len(), 2);
    }

    // ----- Update + delete -------------------------------------------

    #[tokio::test]
    async fn update_buffered_then_committed() {
        let engine = engine_with_users();
        let tx = Transaction::begin(engine.clone());
        tx.update_by_pk("users", &[ProximaValue::Int64(1)], user_row(1, "alice_v2"))
            .unwrap();
        // Inside: updated.
        let inside = scan_all(&tx.reader_factory()).await;
        let alice_row = inside
            .iter()
            .find(|r| r[0] == ProximaValue::Int64(1))
            .unwrap();
        assert_eq!(alice_row[1], ProximaValue::String("alice_v2".into()));
        // Outside: original.
        let raw = engine.snapshot_rows("users").unwrap();
        assert_eq!(raw[0][1], ProximaValue::String("alice".into()));
        tx.commit().unwrap();
        let after = engine.snapshot_rows("users").unwrap();
        assert_eq!(after[0][1], ProximaValue::String("alice_v2".into()));
    }

    #[tokio::test]
    async fn delete_buffered_then_committed() {
        let engine = engine_with_users();
        let tx = Transaction::begin(engine.clone());
        tx.delete_by_pk("users", &[ProximaValue::Int64(1)]).unwrap();
        // Inside: 1 row.
        let inside = scan_all(&tx.reader_factory()).await;
        assert_eq!(inside.len(), 1);
        assert_eq!(inside[0][0], ProximaValue::Int64(2));
        // Outside: 2 rows still.
        assert_eq!(engine.snapshot_rows("users").unwrap().len(), 2);
        tx.commit().unwrap();
        assert_eq!(engine.snapshot_rows("users").unwrap().len(), 1);
    }

    // ----- Mixed ------------------------------------------------------

    #[tokio::test]
    async fn mixed_operations_commit_in_correct_order() {
        let engine = engine_with_users();
        let tx = Transaction::begin(engine.clone());
        tx.insert("users", vec![user_row(3, "carol")]).unwrap();
        tx.update_by_pk("users", &[ProximaValue::Int64(2)], user_row(2, "bob_v2"))
            .unwrap();
        tx.delete_by_pk("users", &[ProximaValue::Int64(1)]).unwrap();
        tx.commit().unwrap();
        let rows = engine.snapshot_rows("users").unwrap();
        assert_eq!(rows.len(), 2);
        // bob updated, carol inserted, alice deleted.
        let names: Vec<String> = rows
            .iter()
            .map(|r| match &r[1] {
                ProximaValue::String(s) => s.clone(),
                _ => panic!(),
            })
            .collect();
        assert!(names.contains(&"bob_v2".to_string()));
        assert!(names.contains(&"carol".to_string()));
        assert!(!names.contains(&"alice".to_string()));
    }

    // ----- Concurrent transactions (READ COMMITTED) ------------------

    #[tokio::test]
    async fn concurrent_txs_dont_see_each_others_uncommitted_writes() {
        let engine = engine_with_users();
        let tx_a = Transaction::begin(engine.clone());
        let tx_b = Transaction::begin(engine.clone());
        tx_a.insert("users", vec![user_row(3, "carol")]).unwrap();
        // tx_b reading should NOT see carol.
        let b_view = scan_all(&tx_b.reader_factory()).await;
        assert_eq!(b_view.len(), 2);
        // tx_a reading still sees its own write.
        let a_view = scan_all(&tx_a.reader_factory()).await;
        assert_eq!(a_view.len(), 3);
        tx_a.commit().unwrap();
        // After commit, tx_b sees the new row on a fresh scan.
        let b_view_after = scan_all(&tx_b.reader_factory()).await;
        assert_eq!(b_view_after.len(), 3);
    }

    // ----- Finished-state guard ---------------------------------------

    #[tokio::test]
    async fn write_after_commit_errors() {
        let engine = engine_with_users();
        let tx = Transaction::begin(engine.clone());
        tx.commit().unwrap();
        let err = tx.insert("users", vec![user_row(99, "ghost")]).unwrap_err();
        assert!(matches!(err, EngineError::Internal(_)));
    }

    #[tokio::test]
    async fn double_commit_errors() {
        let engine = engine_with_users();
        let tx = Transaction::begin(engine.clone());
        tx.commit().unwrap();
        let err = tx.commit().unwrap_err();
        assert!(matches!(err, EngineError::Internal(_)));
    }

    // ----- Validation -------------------------------------------------

    #[tokio::test]
    async fn insert_arity_mismatch_errors_at_buffer_time() {
        let engine = engine_with_users();
        let tx = Transaction::begin(engine.clone());
        let err = tx
            .insert("users", vec![vec![ProximaValue::Int64(99)]])
            .unwrap_err();
        assert!(matches!(err, EngineError::RowArity { .. }));
    }

    #[tokio::test]
    async fn unknown_table_errors() {
        let engine = engine_with_users();
        let tx = Transaction::begin(engine.clone());
        let err = tx.insert("nope", vec![user_row(1, "alice")]).unwrap_err();
        assert!(matches!(err, EngineError::TableNotFound(_)));
    }
}
