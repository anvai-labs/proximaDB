//! `DiscoveryRegistry` — durable in-memory registry of discovery jobs.
//!
//! A `DashMap` is authoritative in memory. Local persistence atomically rewrites
//! a JSON sidecar (temp file + rename); object-store persistence sends ordered
//! full-snapshot PUTs through the unified filesystem. Write failures are logged,
//! never propagated — the in-memory state stays authoritative.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use super::job::{DiscoveryJob, DiscoveryJobStatus};
use proximadb_kernel::time::now_unix_ms;

const REGISTRY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct PersistedRegistry {
    schema_version: u32,
    jobs: Vec<DiscoveryJob>,
}

/// Durable registry of discovery jobs keyed by `job_id`.
pub struct DiscoveryRegistry {
    jobs: DashMap<String, DiscoveryJob>,
    persistence: Option<RegistryPersistence>,
}

enum RegistryPersistence {
    Local(PathBuf),
    ObjectStore {
        url: String,
        writer: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
    },
}

impl Default for DiscoveryRegistry {
    fn default() -> Self {
        Self {
            jobs: DashMap::new(),
            persistence: None,
        }
    }
}

impl DiscoveryRegistry {
    /// In-memory registry with no persistence (tests / pre-bootstrap).
    pub fn new() -> Self {
        Self::default()
    }

    /// Registry that auto-persists to `path` on every mutation, recovering
    /// prior jobs if the file exists and is valid. Corrupt/missing files start
    /// empty (the next mutation overwrites them).
    pub fn load_or_create_at(path: PathBuf) -> Self {
        let registry = Self {
            jobs: DashMap::new(),
            persistence: Some(RegistryPersistence::Local(path.clone())),
        };

        match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<PersistedRegistry>(&bytes) {
                Ok(persisted) => {
                    tracing::info!(
                        "DiscoveryRegistry: loaded {} jobs from {}",
                        persisted.jobs.len(),
                        path.display()
                    );
                    for job in persisted.jobs {
                        registry.jobs.insert(job.job_id.clone(), job);
                    }
                }
                Err(err) => tracing::warn!(
                    "DiscoveryRegistry: file at {} is corrupt ({}); starting empty",
                    path.display(),
                    err
                ),
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => tracing::debug!(
                "DiscoveryRegistry: no existing file at {}; starting empty",
                path.display()
            ),
            Err(err) => tracing::warn!(
                "DiscoveryRegistry: cannot read {} ({}); starting empty",
                path.display(),
                err
            ),
        }

        registry
    }

    /// Registry persisted through the unified filesystem at a scheme-qualified
    /// URL. This is the production constructor for object-store metadata; URL-
    /// shaped paths never reach `std::fs`.
    pub async fn load_or_create_at_url(
        url: String,
        filesystem_factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    ) -> anyhow::Result<Self> {
        let filesystem = filesystem_factory.get_filesystem(&url)?;
        let (writer, mut writes) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
        let writer_url = url.clone();
        let writer_filesystem = filesystem.clone();
        // One worker serializes full-snapshot PUTs in mutation order. Spawning
        // one task per mutation could let an older snapshot land after a newer
        // one and roll the registry backward.
        tokio::spawn(async move {
            while let Some(serialized) = writes.recv().await {
                if let Err(err) = writer_filesystem
                    .write_atomic(&writer_url, &serialized, None)
                    .await
                {
                    tracing::warn!(
                        "DiscoveryRegistry: failed to persist to {} ({}); in-memory state remains authoritative",
                        writer_url,
                        err
                    );
                }
            }
        });
        let registry = Self {
            jobs: DashMap::new(),
            persistence: Some(RegistryPersistence::ObjectStore {
                url: url.clone(),
                writer,
            }),
        };
        if filesystem.exists(&url).await? {
            match filesystem.read(&url).await {
                Ok(bytes) => match serde_json::from_slice::<PersistedRegistry>(&bytes) {
                    Ok(persisted) => {
                        tracing::info!(
                            "DiscoveryRegistry: loaded {} jobs from {}",
                            persisted.jobs.len(),
                            url
                        );
                        for job in persisted.jobs {
                            registry.jobs.insert(job.job_id.clone(), job);
                        }
                    }
                    Err(err) => tracing::warn!(
                        "DiscoveryRegistry: object at {} is corrupt ({}); starting empty",
                        url,
                        err
                    ),
                },
                Err(err) => tracing::warn!(
                    "DiscoveryRegistry: cannot read {} ({}); starting empty",
                    url,
                    err
                ),
            }
        }
        Ok(registry)
    }

    /// Record a newly created (`Scheduled`) job.
    pub fn schedule(&self, job: DiscoveryJob) -> DiscoveryJob {
        self.jobs.insert(job.job_id.clone(), job.clone());
        self.persist_if_configured();
        job
    }

    /// Replace a job record (status/result updates).
    pub fn upsert(&self, job: DiscoveryJob) {
        self.jobs.insert(job.job_id.clone(), job);
        self.persist_if_configured();
    }

    /// Look up a job by id.
    pub fn get(&self, job_id: &str) -> Option<DiscoveryJob> {
        self.jobs.get(job_id).map(|j| j.clone())
    }

    /// All jobs for a collection (newest first).
    pub fn list_for_collection(&self, collection_id: &str) -> Vec<DiscoveryJob> {
        let mut jobs: Vec<DiscoveryJob> = self
            .jobs
            .iter()
            .filter(|e| e.value().collection_id == collection_id)
            .map(|e| e.value().clone())
            .collect();
        jobs.sort_by_key(|j| std::cmp::Reverse(j.created_at_ms));
        jobs
    }

    /// Every job (for operator dashboards).
    pub fn list_all(&self) -> Vec<DiscoveryJob> {
        self.jobs.iter().map(|e| e.value().clone()).collect()
    }

    /// Distinct collection ids that have any discovery history. Used by the
    /// drift watcher to decide which collections to sweep.
    pub fn collections(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        self.jobs
            .iter()
            .filter_map(|e| {
                let id = e.value().collection_id.clone();
                seen.insert(id.clone()).then_some(id)
            })
            .collect()
    }

    /// Atomically claim the oldest `Scheduled` job, transitioning it to
    /// `Running`. Returns the claimed job, or `None` if none are pending.
    pub fn claim_next_scheduled(&self) -> Option<DiscoveryJob> {
        let mut candidate: Option<DiscoveryJob> = None;
        for entry in self.jobs.iter() {
            if entry.value().status != DiscoveryJobStatus::Scheduled {
                continue;
            }
            let job = entry.value().clone();
            candidate = match candidate {
                Some(current) if current.created_at_ms <= job.created_at_ms => Some(current),
                _ => Some(job),
            };
        }

        let mut job = candidate?;
        job.status = DiscoveryJobStatus::Running;
        job.started_at_ms = Some(now_unix_ms());
        self.jobs.insert(job.job_id.clone(), job.clone());
        self.persist_if_configured();
        Some(job)
    }

    fn persist_if_configured(&self) {
        let Some(persistence) = self.persistence.as_ref() else {
            return;
        };
        let persisted = PersistedRegistry {
            schema_version: REGISTRY_SCHEMA_VERSION,
            jobs: self.jobs.iter().map(|e| e.value().clone()).collect(),
        };
        match persistence {
            RegistryPersistence::Local(path) => {
                if let Err(err) = atomic_write_json(path, &persisted) {
                    tracing::warn!(
                        "DiscoveryRegistry: failed to persist to {} ({}); in-memory state remains authoritative",
                        path.display(),
                        err
                    );
                }
            }
            RegistryPersistence::ObjectStore { url, writer } => {
                let serialized = match serde_json::to_vec_pretty(&persisted) {
                    Ok(serialized) => serialized,
                    Err(err) => {
                        tracing::warn!("DiscoveryRegistry: serialization failed: {err}");
                        return;
                    }
                };
                // Mutations are synchronous control-plane calls today. Enqueue
                // the full snapshot to the single ordered remote writer rather
                // than blocking the caller's worker thread on object-store I/O.
                if writer.send(serialized).is_err() {
                    tracing::warn!(
                        "DiscoveryRegistry: persistence worker for {} stopped; in-memory state remains authoritative",
                        url
                    );
                }
            }
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
    use crate::services::discovery::job::DiscoveryJobKind;

    #[test]
    fn schedule_then_claim_transitions_to_running() {
        let reg = DiscoveryRegistry::new();
        let job = reg.schedule(DiscoveryJob::new("c1", DiscoveryJobKind::Dedup));
        assert_eq!(
            reg.get(&job.job_id).unwrap().status,
            DiscoveryJobStatus::Scheduled
        );

        let claimed = reg.claim_next_scheduled().unwrap();
        assert_eq!(claimed.job_id, job.job_id);
        assert_eq!(claimed.status, DiscoveryJobStatus::Running);
        assert!(claimed.started_at_ms.is_some());
        // No more scheduled jobs.
        assert!(reg.claim_next_scheduled().is_none());
    }

    #[test]
    fn persistence_round_trip_restores_jobs() {
        let tmp = std::env::temp_dir().join(format!(
            "proximadb_discovery_reg_{}.json",
            uuid::Uuid::new_v4().simple()
        ));
        {
            let reg = DiscoveryRegistry::load_or_create_at(tmp.clone());
            reg.schedule(DiscoveryJob::new("c1", DiscoveryJobKind::Dedup));
            reg.schedule(DiscoveryJob::new("c2", DiscoveryJobKind::Recluster));
        }
        let restored = DiscoveryRegistry::load_or_create_at(tmp.clone());
        assert_eq!(restored.list_all().len(), 2);
        let _ = std::fs::remove_file(&tmp);
    }
}
