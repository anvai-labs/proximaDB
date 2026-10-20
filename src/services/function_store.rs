//! Canonical-WAL-backed durable storage for SQL user functions (F5).
//!
//! `CREATE FUNCTION` definitions are catalog-level metadata that must survive restarts. Mirroring
//! [`crate::services::rank_profile_store`], each put/remove is encoded as a
//! `CanonicalOperation::RecordUpsert` / `RecordDelete` on a synthetic `__proxima_functions__`
//! collection through the shared [`TableWalAppender`]; the WAL is replayed at startup to rebuild
//! the visible function set, which is then re-registered into the engine-neutral
//! `proximadb_functions::ProximaFunctionRegistry` so the user functions are live on both engines
//! again after a restart.
//!
//! The whole [`StoredFunction`] is serialized as one JSON `definition` prop (its fields —
//! including `ProximaType` — are `serde`), keeping the record mapping trivial.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use parking_lot::RwLock;
use proximadb_data_model::{ProximaType, ProximaValue};
use proximadb_records::{ProximaRecord, ProximaTreeNode};
use proximadb_storage_common::{CanonicalOperation, CanonicalWalEntry};
use serde::{Deserialize, Serialize};

use crate::services::record_store::TableWalAppender;

/// Synthetic collection id namespacing function entries inside the shared canonical WAL. Shaped
/// to never collide with a user collection name.
pub const FUNCTIONS_COLLECTION_ID: &str = "__proxima_functions__";

/// A durable SQL user-function definition — the shape registered as a
/// `proximadb_functions::sql_bodied_scalar`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredFunction {
    /// Function name (catalog key).
    pub name: String,
    /// Parameters in order: `(name, type)`.
    pub params: Vec<(String, ProximaType)>,
    /// Declared return type.
    pub return_ty: ProximaType,
    /// SQL body — a scalar expression over the parameters.
    pub body: String,
    /// Wall-clock install time (ms since Unix epoch).
    pub created_at_ms: i64,
}

/// Durable function-catalog API. Implementations persist puts/removes through the canonical WAL
/// so registered user functions survive restarts; reads are served from an in-memory snapshot
/// rebuilt at construction.
#[async_trait]
pub trait FunctionStore: Send + Sync {
    /// Persist (insert or replace) a function definition.
    async fn put(&self, function: StoredFunction) -> Result<()>;
    /// List every persisted function (used by startup recovery to re-register them).
    async fn list_all(&self) -> Result<Vec<StoredFunction>>;
    /// Remove a function by name. Returns `true` if one was present.
    async fn remove(&self, name: &str) -> Result<bool>;
    /// Re-append every live definition to the head of the canonical WAL and
    /// return the sequence of the first re-appended entry: this store no
    /// longer needs any WAL entry below it (TD-SEAM-12). `None` when no
    /// function is defined.
    async fn checkpoint(&self) -> Result<Option<u64>>;
}

/// Canonical-WAL-backed [`FunctionStore`].
pub struct CanonicalWalFunctionStore {
    appender: Arc<dyn TableWalAppender>,
    state: RwLock<HashMap<String, StoredFunction>>,
    /// Serializes puts, removes and [`FunctionStore::checkpoint`], so a
    /// checkpoint never re-appends a definition older than one already
    /// written after it (replay is last-write-wins by name).
    write_gate: tokio::sync::Mutex<()>,
}

impl CanonicalWalFunctionStore {
    /// Build an empty store over a fresh appender (tests / known-empty WAL).
    pub fn new(appender: Arc<dyn TableWalAppender>) -> Self {
        Self {
            appender,
            state: RwLock::new(HashMap::new()),
            write_gate: tokio::sync::Mutex::new(()),
        }
    }

    /// Build a store and replay an existing canonical WAL slice into it (startup recovery).
    pub fn from_wal_entries(
        appender: Arc<dyn TableWalAppender>,
        entries: &[CanonicalWalEntry],
    ) -> Self {
        let store = Self::new(appender);
        store.replay_entries(entries);
        store
    }

    fn replay_entries(&self, entries: &[CanonicalWalEntry]) {
        let mut state = self.state.write();
        for entry in entries {
            match &entry.operation {
                CanonicalOperation::RecordUpsert {
                    collection_id,
                    record,
                    ..
                } if collection_id == FUNCTIONS_COLLECTION_ID => {
                    if let Some(f) = record_to_function(record.as_ref()) {
                        state.insert(f.name.clone(), f);
                    }
                }
                CanonicalOperation::RecordDelete {
                    collection_id, oid, ..
                } if collection_id == FUNCTIONS_COLLECTION_ID => {
                    state.remove(oid);
                }
                _ => {}
            }
        }
    }
}

/// Lower a [`StoredFunction`]'s body and register it as a `sql_bodied_scalar` in the shared
/// `proximadb_functions::builtins()` registry — making the user function callable on both
/// engines. Shared by the `CREATE FUNCTION` DDL handler and startup recovery (which replays the
/// persisted catalog through here).
pub fn register_stored_function(f: &StoredFunction) -> Result<()> {
    let body_expr = proximadb_relational_frontend::lower_function_body(&f.body, &f.params)
        .map_err(|e| anyhow::anyhow!("function '{}': {e}", f.name))?;
    let arg_types: Vec<ProximaType> = f.params.iter().map(|(_, t)| t.clone()).collect();
    proximadb_functions::builtins().register_scalar(proximadb_functions::sql_bodied_scalar(
        &f.name,
        arg_types,
        f.return_ty.clone(),
        body_expr,
    ));
    Ok(())
}

/// Current wall-clock time in nanoseconds since the Unix epoch.
fn now_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

fn function_to_record(f: &StoredFunction) -> ProximaRecord {
    let json = serde_json::to_string(f).unwrap_or_default();
    let mut props: HashMap<String, ProximaTreeNode> = HashMap::new();
    props.insert(
        "definition".to_string(),
        ProximaTreeNode::Value(ProximaValue::String(json)),
    );
    ProximaRecord {
        oid: f.name.clone(),
        variation_id: Some(FUNCTIONS_COLLECTION_ID.to_string()),
        created_at_ns: now_ns(),
        props,
        ..Default::default()
    }
}

fn record_to_function(record: &ProximaRecord) -> Option<StoredFunction> {
    match record.props.get("definition")? {
        ProximaTreeNode::Value(ProximaValue::String(s)) => serde_json::from_str(s).ok(),
        _ => None,
    }
}

#[async_trait]
impl FunctionStore for CanonicalWalFunctionStore {
    async fn put(&self, function: StoredFunction) -> Result<()> {
        let _gate = self.write_gate.lock().await;
        let record = function_to_record(&function);
        self.appender
            .append_operations(
                vec![CanonicalOperation::RecordUpsert {
                    collection_id: FUNCTIONS_COLLECTION_ID.to_string(),
                    record: Box::new(record),
                    projections: Vec::new(),
                }],
                None,
            )
            .await
            .context("appending CREATE FUNCTION to canonical WAL")?;
        self.state.write().insert(function.name.clone(), function);
        Ok(())
    }

    async fn list_all(&self) -> Result<Vec<StoredFunction>> {
        Ok(self.state.read().values().cloned().collect())
    }

    async fn remove(&self, name: &str) -> Result<bool> {
        let _gate = self.write_gate.lock().await;
        if self.state.read().get(name).is_none() {
            return Ok(false);
        }
        self.appender
            .append_operations(
                vec![CanonicalOperation::RecordDelete {
                    collection_id: FUNCTIONS_COLLECTION_ID.to_string(),
                    oid: name.to_string(),
                    projections: Vec::new(),
                }],
                None,
            )
            .await
            .context("appending DROP FUNCTION to canonical WAL")?;
        self.state.write().remove(name);
        Ok(true)
    }

    async fn checkpoint(&self) -> Result<Option<u64>> {
        let _gate = self.write_gate.lock().await;
        let mut live: Vec<StoredFunction> = self.state.read().values().cloned().collect();
        live.sort_by(|a, b| a.name.cmp(&b.name));
        if live.is_empty() {
            return Ok(None);
        }
        let operations = live
            .iter()
            .map(|f| CanonicalOperation::RecordUpsert {
                collection_id: FUNCTIONS_COLLECTION_ID.to_string(),
                record: Box::new(function_to_record(f)),
                projections: Vec::new(),
            })
            .collect();
        let entries = self
            .appender
            .append_operations(operations, None)
            .await
            .context("re-appending function definitions for a WAL checkpoint")?;
        Ok(entries.first().map(|e| e.sequence_number))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::canonical_wal::FramedTableWalAppender;
    use tempfile::tempdir;

    fn sample(name: &str) -> StoredFunction {
        StoredFunction {
            name: name.to_string(),
            params: vec![("x".to_string(), ProximaType::Int64)],
            return_ty: ProximaType::Int64,
            body: "x * 2".to_string(),
            created_at_ms: 0,
        }
    }

    #[tokio::test]
    async fn put_list_and_remove_roundtrip() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("functions.wal");
        let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
        let store = CanonicalWalFunctionStore::new(appender);

        store.put(sample("double")).await.unwrap();
        assert_eq!(store.list_all().await.unwrap().len(), 1);
        assert!(store.remove("double").await.unwrap());
        assert!(store.list_all().await.unwrap().is_empty());
        assert!(!store.remove("double").await.unwrap()); // already gone
    }

    #[tokio::test]
    async fn survives_restart_via_wal_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("functions.wal");
        {
            let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
            let store = CanonicalWalFunctionStore::new(appender);
            store.put(sample("double")).await.unwrap();
        }
        // "Restart": reopen the WAL, read its entries, rebuild the store from them.
        let entries = FramedTableWalAppender::read_entries_from_path(&wal_path)
            .await
            .unwrap();
        let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
        let store = CanonicalWalFunctionStore::from_wal_entries(appender, &entries);

        let all = store.list_all().await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name, "double");
        assert_eq!(all[0].body, "x * 2");
        assert_eq!(all[0].return_ty, ProximaType::Int64);
        assert_eq!(all[0].params, vec![("x".to_string(), ProximaType::Int64)]);
    }
}
