//! Repository interface for profile persistence + an in-memory impl.
//!
//! The xCatalog-backed implementation will live in the catalog crate
//! (out of scope for R-4). Until then, the in-memory variant is what
//! tests and embedded mode use.

use crate::spec::RankProfileSpec;
use async_trait::async_trait;
use dashmap::DashMap;
use proximadb_rank_core::{RankError, RankResult};
use tokio::sync::broadcast;

/// Notifications emitted by the repository so registries (and CDC
/// subscribers in R-7) can react to profile changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileEvent {
    Created { name: String, version: u32 },
    Updated { name: String, version: u32 },
    Deleted { name: String },
}

#[async_trait]
pub trait RankProfileRepository: Send + Sync {
    async fn create(&self, spec: RankProfileSpec) -> RankResult<u32>;
    async fn update(&self, spec: RankProfileSpec) -> RankResult<u32>;
    async fn delete(&self, name: &str) -> RankResult<()>;
    async fn get(&self, name: &str) -> RankResult<Option<RankProfileSpec>>;
    async fn list(&self) -> RankResult<Vec<RankProfileSpec>>;
    fn watch(&self) -> broadcast::Receiver<ProfileEvent>;
}

/// Thread-safe, lock-free in-memory repository. Drops on process exit;
/// not durable. Use for tests + embedded mode.
pub struct InMemoryRankProfileRepository {
    inner: DashMap<String, RankProfileSpec>,
    events: broadcast::Sender<ProfileEvent>,
}

impl InMemoryRankProfileRepository {
    pub fn new() -> Self {
        let (tx, _rx) = broadcast::channel(128);
        Self {
            inner: DashMap::new(),
            events: tx,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for InMemoryRankProfileRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RankProfileRepository for InMemoryRankProfileRepository {
    async fn create(&self, mut spec: RankProfileSpec) -> RankResult<u32> {
        if self.inner.contains_key(&spec.name) {
            return Err(RankError::InvalidProfile(format!(
                "profile '{}' already exists; use update() instead",
                spec.name
            )));
        }
        spec.version = 1;
        spec.created_at_ms = proximadb_kernel::time::now_unix_ms();
        let v = spec.version;
        let name = spec.name.clone();
        self.inner.insert(name.clone(), spec);
        let _ = self.events.send(ProfileEvent::Created { name, version: v });
        Ok(v)
    }

    async fn update(&self, mut spec: RankProfileSpec) -> RankResult<u32> {
        let prior = self
            .inner
            .get(&spec.name)
            .map(|r| r.version)
            .ok_or_else(|| {
                RankError::ProfileNotFound(format!(
                    "cannot update unknown profile '{}'; create() first",
                    spec.name
                ))
            })?;
        spec.version = prior + 1;
        spec.created_at_ms = proximadb_kernel::time::now_unix_ms();
        let v = spec.version;
        let name = spec.name.clone();
        self.inner.insert(name.clone(), spec);
        let _ = self.events.send(ProfileEvent::Updated { name, version: v });
        Ok(v)
    }

    async fn delete(&self, name: &str) -> RankResult<()> {
        if self.inner.remove(name).is_none() {
            return Err(RankError::ProfileNotFound(name.to_string()));
        }
        let _ = self.events.send(ProfileEvent::Deleted {
            name: name.to_string(),
        });
        Ok(())
    }

    async fn get(&self, name: &str) -> RankResult<Option<RankProfileSpec>> {
        Ok(self.inner.get(name).map(|r| r.value().clone()))
    }

    async fn list(&self) -> RankResult<Vec<RankProfileSpec>> {
        Ok(self.inner.iter().map(|r| r.value().clone()).collect())
    }

    fn watch(&self) -> broadcast::Receiver<ProfileEvent> {
        self.events.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{PhaseSpec, RankProfileSpec};
    use std::sync::Arc;

    fn minimal(name: &str) -> RankProfileSpec {
        let mut s = RankProfileSpec::new(name);
        s.first_phase = Some(PhaseSpec {
            expression: "bm25(\"t\")".into(),
            heap_size: Some(10),
            rerank_count: None,
            batch_size: None,
        });
        s
    }

    #[tokio::test]
    async fn create_then_get() {
        let repo = InMemoryRankProfileRepository::new();
        let v = repo.create(minimal("alpha")).await.unwrap();
        assert_eq!(v, 1);
        let fetched = repo.get("alpha").await.unwrap().unwrap();
        assert_eq!(fetched.name, "alpha");
        assert_eq!(fetched.version, 1);
        assert!(fetched.created_at_ms > 0);
    }

    #[tokio::test]
    async fn create_rejects_duplicate() {
        let repo = InMemoryRankProfileRepository::new();
        repo.create(minimal("a")).await.unwrap();
        match repo.create(minimal("a")).await {
            Err(RankError::InvalidProfile(msg)) => assert!(msg.contains("already exists")),
            other => panic!("expected InvalidProfile, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn update_increments_version() {
        let repo = InMemoryRankProfileRepository::new();
        repo.create(minimal("a")).await.unwrap();
        let v2 = repo.update(minimal("a")).await.unwrap();
        assert_eq!(v2, 2);
        let v3 = repo.update(minimal("a")).await.unwrap();
        assert_eq!(v3, 3);
    }

    #[tokio::test]
    async fn update_unknown_errors() {
        let repo = InMemoryRankProfileRepository::new();
        match repo.update(minimal("ghost")).await {
            Err(RankError::ProfileNotFound(msg)) => assert!(msg.contains("ghost")),
            Err(_) => panic!("expected ProfileNotFound, got a different RankError"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[tokio::test]
    async fn delete_removes_and_emits_event() {
        let repo = Arc::new(InMemoryRankProfileRepository::new());
        let mut rx = repo.watch();
        repo.create(minimal("a")).await.unwrap();
        let _ = rx.recv().await.unwrap();
        repo.delete("a").await.unwrap();
        let evt = rx.recv().await.unwrap();
        assert_eq!(
            evt,
            ProfileEvent::Deleted {
                name: "a".to_string()
            }
        );
        assert!(repo.get("a").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_unknown_errors() {
        let repo = InMemoryRankProfileRepository::new();
        match repo.delete("ghost").await {
            Err(RankError::ProfileNotFound(_)) => {}
            other => panic!("expected ProfileNotFound: {other:?}"),
        }
    }

    #[tokio::test]
    async fn list_returns_all() {
        let repo = InMemoryRankProfileRepository::new();
        repo.create(minimal("a")).await.unwrap();
        repo.create(minimal("b")).await.unwrap();
        repo.create(minimal("c")).await.unwrap();
        let all = repo.list().await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn watch_observes_create_and_update() {
        let repo = InMemoryRankProfileRepository::new();
        let mut rx = repo.watch();
        repo.create(minimal("a")).await.unwrap();
        repo.update(minimal("a")).await.unwrap();
        let e1 = rx.recv().await.unwrap();
        let e2 = rx.recv().await.unwrap();
        assert!(matches!(e1, ProfileEvent::Created { version: 1, .. }));
        assert!(matches!(e2, ProfileEvent::Updated { version: 2, .. }));
    }
}
