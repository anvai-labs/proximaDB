//! `ModelRegistry` — durable directory of `ModelDescriptor`s.
//!
//! The xCatalog-backed implementation will live in the catalog crate
//! (out of scope for R-5). The in-memory variant here covers tests +
//! embedded mode. R-5b will add `acquire_with_loader(cache, registry, key)`
//! to drive the lazy download → cache install path.

use crate::descriptor::{ModelDescriptor, ModelKey};
use async_trait::async_trait;
use dashmap::DashMap;
use proximadb_rank_core::{RankError, RankResult};
use std::sync::atomic::{AtomicU64, Ordering};

#[async_trait]
pub trait ModelRegistry: Send + Sync {
    /// Register a new descriptor. Returns the assigned monotonic `seq`.
    /// Errors if a descriptor with the same `ModelKey` already exists.
    async fn register(&self, desc: ModelDescriptor) -> RankResult<u64>;
    async fn get(&self, key: &ModelKey) -> RankResult<Option<ModelDescriptor>>;
    /// List all descriptors, optionally filtered to a tenant scope.
    async fn list(&self, tenant: Option<&str>) -> RankResult<Vec<ModelDescriptor>>;
    async fn delete(&self, key: &ModelKey) -> RankResult<()>;
}

pub struct InMemoryModelRegistry {
    inner: DashMap<ModelKey, ModelDescriptor>,
    seq: AtomicU64,
}

impl InMemoryModelRegistry {
    pub fn new() -> Self {
        Self {
            inner: DashMap::new(),
            seq: AtomicU64::new(0),
        }
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for InMemoryModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelRegistry for InMemoryModelRegistry {
    async fn register(&self, mut desc: ModelDescriptor) -> RankResult<u64> {
        if self.inner.contains_key(&desc.key) {
            return Err(RankError::InvalidProfile(format!(
                "model '{}' already registered; delete first to replace",
                desc.key
            )));
        }
        let seq = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        desc.seq = seq;
        desc.created_at_ms = proximadb_kernel::time::now_unix_ms();
        self.inner.insert(desc.key.clone(), desc);
        Ok(seq)
    }

    async fn get(&self, key: &ModelKey) -> RankResult<Option<ModelDescriptor>> {
        Ok(self.inner.get(key).map(|r| r.value().clone()))
    }

    async fn list(&self, tenant: Option<&str>) -> RankResult<Vec<ModelDescriptor>> {
        Ok(self
            .inner
            .iter()
            .filter(|r| match tenant {
                Some(t) => r.value().tenant.as_deref() == Some(t),
                None => true,
            })
            .map(|r| r.value().clone())
            .collect())
    }

    async fn delete(&self, key: &ModelKey) -> RankResult<()> {
        if self.inner.remove(key).is_none() {
            return Err(RankError::ProfileNotFound(format!(
                "no such model in registry: {key}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{DType, ModelFramework};

    fn desc(model: &str, version: &str, tenant: Option<&str>) -> ModelDescriptor {
        ModelDescriptor {
            key: ModelKey::new(model, version),
            tenant: tenant.map(|s| s.to_string()),
            uri: format!("file:///models/{model}/{version}.onnx"),
            sha256: [0; 32],
            size_bytes: 1024,
            framework: ModelFramework::Onnx,
            dtype: DType::Fp32,
            input_spec: vec![],
            output_spec: vec![],
            max_batch_size: 32,
            seq: 0,
            created_at_ms: 0,
        }
    }

    #[tokio::test]
    async fn register_assigns_seq_and_timestamp() {
        let reg = InMemoryModelRegistry::new();
        let s = reg.register(desc("rerank", "v1", None)).await.unwrap();
        assert_eq!(s, 1);
        let fetched = reg
            .get(&ModelKey::new("rerank", "v1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.seq, 1);
        assert!(fetched.created_at_ms > 0);
    }

    #[tokio::test]
    async fn register_rejects_duplicate_key() {
        let reg = InMemoryModelRegistry::new();
        reg.register(desc("rerank", "v1", None)).await.unwrap();
        match reg.register(desc("rerank", "v1", None)).await {
            Err(RankError::InvalidProfile(msg)) => assert!(msg.contains("already registered")),
            other => panic!("expected InvalidProfile: {other:?}"),
        }
    }

    #[tokio::test]
    async fn seq_increments_across_distinct_keys() {
        let reg = InMemoryModelRegistry::new();
        let s1 = reg.register(desc("a", "v1", None)).await.unwrap();
        let s2 = reg.register(desc("b", "v1", None)).await.unwrap();
        let s3 = reg.register(desc("a", "v2", None)).await.unwrap();
        assert!(
            s1 < s2 && s2 < s3,
            "seq monotonically increases: {s1}, {s2}, {s3}"
        );
    }

    #[tokio::test]
    async fn get_unknown_returns_none() {
        let reg = InMemoryModelRegistry::new();
        assert!(
            reg.get(&ModelKey::new("ghost", "v1"))
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn list_filters_by_tenant() {
        let reg = InMemoryModelRegistry::new();
        reg.register(desc("m1", "v1", Some("tenant-a")))
            .await
            .unwrap();
        reg.register(desc("m2", "v1", Some("tenant-a")))
            .await
            .unwrap();
        reg.register(desc("m3", "v1", Some("tenant-b")))
            .await
            .unwrap();
        reg.register(desc("m4", "v1", None)).await.unwrap();
        let a = reg.list(Some("tenant-a")).await.unwrap();
        assert_eq!(a.len(), 2);
        let b = reg.list(Some("tenant-b")).await.unwrap();
        assert_eq!(b.len(), 1);
        let all = reg.list(None).await.unwrap();
        assert_eq!(all.len(), 4);
    }

    #[tokio::test]
    async fn delete_removes_descriptor() {
        let reg = InMemoryModelRegistry::new();
        reg.register(desc("rerank", "v1", None)).await.unwrap();
        reg.delete(&ModelKey::new("rerank", "v1")).await.unwrap();
        assert!(
            reg.get(&ModelKey::new("rerank", "v1"))
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn delete_unknown_errors() {
        let reg = InMemoryModelRegistry::new();
        match reg.delete(&ModelKey::new("ghost", "v1")).await {
            Err(RankError::ProfileNotFound(_)) => {}
            other => panic!("expected ProfileNotFound: {other:?}"),
        }
    }
}
