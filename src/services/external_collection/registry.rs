//! `ExternalCollectionRegistry` — durable in-memory registry of external
//! collections (Phase 8 F5).
//!
//! Mirrors `crate::services::discovery::DiscoveryRegistry`: a `DashMap` is
//! authoritative in memory; when constructed with a path, every mutation
//! atomically rewrites a JSON sidecar (temp file + rename). Write failures are
//! logged, never propagated — the in-memory state stays authoritative.

use std::path::{Path, PathBuf};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use super::types::ExternalCollection;
use proximadb_kernel::time::now_unix_ms;

const REGISTRY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct PersistedRegistry {
    schema_version: u32,
    collections: Vec<ExternalCollection>,
}

/// Durable registry of external collections keyed by `id`.
#[derive(Default)]
pub struct ExternalCollectionRegistry {
    collections: DashMap<String, ExternalCollection>,
    persistence_path: Option<PathBuf>,
}

impl ExternalCollectionRegistry {
    /// In-memory registry with no persistence (tests / pre-bootstrap).
    pub fn new() -> Self {
        Self::default()
    }

    /// Registry that auto-persists to `path` on every mutation, recovering
    /// prior records if the file exists and is valid. Corrupt/missing files
    /// start empty (the next mutation overwrites them).
    pub fn load_or_create_at(path: PathBuf) -> Self {
        let registry = Self {
            collections: DashMap::new(),
            persistence_path: Some(path.clone()),
        };

        match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<PersistedRegistry>(&bytes) {
                Ok(persisted) => {
                    tracing::info!(
                        "ExternalCollectionRegistry: loaded {} collections from {}",
                        persisted.collections.len(),
                        path.display()
                    );
                    for ec in persisted.collections {
                        registry.collections.insert(ec.id.clone(), ec);
                    }
                }
                Err(err) => tracing::warn!(
                    "ExternalCollectionRegistry: file at {} is corrupt ({}); starting empty",
                    path.display(),
                    err
                ),
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => tracing::debug!(
                "ExternalCollectionRegistry: no existing file at {}; starting empty",
                path.display()
            ),
            Err(err) => tracing::warn!(
                "ExternalCollectionRegistry: cannot read {} ({}); starting empty",
                path.display(),
                err
            ),
        }

        registry
    }

    /// Insert or replace a record, bumping its `updated_at_ms`.
    pub fn upsert(&self, mut ec: ExternalCollection) {
        ec.updated_at_ms = now_unix_ms();
        self.collections.insert(ec.id.clone(), ec);
        self.persist_if_configured();
    }

    /// Look up a record by id.
    pub fn get(&self, id: &str) -> Option<ExternalCollection> {
        self.collections.get(id).map(|e| e.clone())
    }

    /// Look up a record by its logical collection name.
    pub fn get_by_name(&self, name: &str) -> Option<ExternalCollection> {
        self.collections
            .iter()
            .find(|e| e.value().spec.name == name)
            .map(|e| e.value().clone())
    }

    /// Every registered external collection (newest first).
    pub fn list_all(&self) -> Vec<ExternalCollection> {
        let mut all: Vec<ExternalCollection> =
            self.collections.iter().map(|e| e.value().clone()).collect();
        all.sort_by_key(|e| std::cmp::Reverse(e.created_at_ms));
        all
    }

    fn persist_if_configured(&self) {
        let Some(path) = self.persistence_path.as_ref() else {
            return;
        };
        let persisted = PersistedRegistry {
            schema_version: REGISTRY_SCHEMA_VERSION,
            collections: self.collections.iter().map(|e| e.value().clone()).collect(),
        };
        if let Err(err) = atomic_write_json(path, &persisted) {
            tracing::warn!(
                "ExternalCollectionRegistry: failed to persist to {} ({}); in-memory state remains authoritative",
                path.display(),
                err
            );
        }
    }
}

fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let serialized = serde_json::to_vec_pretty(value)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serialized)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::external_collection::types::{
        ExternalCollectionSpec, ExternalCollectionStatus,
    };

    fn sample(name: &str) -> ExternalCollection {
        let spec = ExternalCollectionSpec::parquet(name, "/tmp/x.parquet", "id", "vector", 8);
        ExternalCollection::new(spec, "snap-1")
    }

    #[test]
    fn upsert_then_get_by_id_and_name() {
        let reg = ExternalCollectionRegistry::new();
        let ec = sample("docs");
        reg.upsert(ec.clone());
        assert_eq!(reg.get(&ec.id).unwrap().spec.name, "docs");
        assert_eq!(reg.get_by_name("docs").unwrap().id, ec.id);
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn persistence_round_trip_restores_records() {
        let tmp = std::env::temp_dir().join(format!(
            "proximadb_extcoll_reg_{}.json",
            uuid::Uuid::new_v4().simple()
        ));
        let id = {
            let reg = ExternalCollectionRegistry::load_or_create_at(tmp.clone());
            let mut ec = sample("docs");
            ec.status = ExternalCollectionStatus::Ready;
            ec.indexed_record_count = 42;
            reg.upsert(ec.clone());
            ec.id
        };
        let restored = ExternalCollectionRegistry::load_or_create_at(tmp.clone());
        let got = restored.get(&id).unwrap();
        assert_eq!(got.status, ExternalCollectionStatus::Ready);
        assert_eq!(got.indexed_record_count, 42);
        let _ = std::fs::remove_file(&tmp);
    }
}
