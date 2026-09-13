//! Tracked-S3 [`ArtifactBackend`] (TD-MLOPS-4 S1-remainder).
//!
//! The SECOND implementation behind the seam — added, not rewritten: the
//! hardened local store (#1891) is untouched and stays the default. This
//! backend rides the platform's object-store plumbing
//! (`ProximaObjectStore`: standard `s3://bucket[/prefix]` URL + AWS_* env
//! credentials, async-native — no blocking to offload, so port rule 2
//! holds by construction). Tenant binding is rule 1: the factory derives
//! an injective per-tenant prefix (the same `<len>/<hex>` encoding as the
//! local layout) and paths never carry tenant identity.
//!
//! Object stores have no real directories: a GET miss falls back to a
//! prefix listing so the port's GET-on-directory-LISTS contract holds,
//! and DELETE is naturally idempotent (S3 semantics).

#[cfg(feature = "aws")]
mod imp {
    use bytes::Bytes;
    use proximadb_catalog::run_store::{
        ArtifactBackend, ArtifactBackendCapabilities, ArtifactBackendContents,
        ArtifactBackendEntry, ArtifactBackendError, ArtifactBackendFactory, ArtifactPath,
    };
    use proximadb_kernel::error::StorageError;
    use proximadb_object_store::ProximaObjectStore;

    const MAX_ARTIFACT_TENANT_ID_BYTES: usize = 64;

    fn err(error: StorageError) -> ArtifactBackendError {
        ArtifactBackendError::Internal(error.to_string())
    }

    /// Factory from a tracked-store URL (`s3://bucket[/prefix]`).
    pub struct S3ArtifactBackendFactory {
        url: String,
    }

    impl S3ArtifactBackendFactory {
        pub fn new(url: impl Into<String>) -> Self {
            Self { url: url.into() }
        }

        /// The injective per-tenant prefix (same encoding as the local
        /// layout — byte length separates roots so no tenant's prefix is
        /// an ancestor of another's).
        fn tenant_prefix(tenant_id: &str) -> Result<String, ArtifactBackendError> {
            if tenant_id.is_empty() || tenant_id.len() > MAX_ARTIFACT_TENANT_ID_BYTES {
                return Err(ArtifactBackendError::Invalid(format!(
                    "artifact tenant id must be 1..={MAX_ARTIFACT_TENANT_ID_BYTES} UTF-8 bytes"
                )));
            }
            Ok(format!(
                "mlflow_artifacts_v2/{}/{}",
                tenant_id.len(),
                hex::encode(tenant_id.as_bytes())
            ))
        }
    }

    impl ArtifactBackendFactory for S3ArtifactBackendFactory {
        fn backend_for(
            &self,
            tenant_id: &str,
        ) -> Result<std::sync::Arc<dyn ArtifactBackend>, ArtifactBackendError> {
            let prefix = Self::tenant_prefix(tenant_id)?;
            let store = ProximaObjectStore::from_url(&self.url).map_err(err)?;
            let base = object_store::path::Path::parse(prefix)
                .map_err(|e| ArtifactBackendError::Internal(format!("tenant prefix: {e}")))?;
            Ok(std::sync::Arc::new(S3ArtifactBackend { store, base }))
        }
    }

    pub struct S3ArtifactBackend {
        store: ProximaObjectStore,
        /// Tenant-scoped prefix inside the bucket (rule 1).
        base: object_store::path::Path,
    }

    impl S3ArtifactBackend {
        fn full(&self, path: &ArtifactPath) -> object_store::path::Path {
            let mut segments: Vec<String> =
                self.base.parts().map(|p| p.as_ref().to_string()).collect();
            segments.extend(path.segments().iter().cloned());
            object_store::path::Path::from_iter(segments)
        }

        /// Direct children under a prefix: object keys are flat, so a
        /// child is the next segment; anything deeper makes it a
        /// directory node.
        async fn children(
            &self,
            prefix: &object_store::path::Path,
        ) -> Result<Vec<ArtifactBackendEntry>, ArtifactBackendError> {
            let metas = self.store.list(Some(prefix)).await.map_err(err)?;
            // `ObjectMeta.location` is the ABSOLUTE key (the URL's base
            // prefix included); ops take caller-relative paths and join
            // internally — slice by the RESOLVED prefix, not the
            // caller-relative one (the misaligned slice returned garbage
            // children like the tenant-length segment).
            let prefix_len = self.store.full_path(prefix).as_ref().len();
            let mut out: std::collections::BTreeMap<String, bool> =
                std::collections::BTreeMap::new();
            for meta in metas {
                let rest = &meta.location.as_ref()[prefix_len..];
                let rest = rest.trim_start_matches('/');
                if rest.is_empty() {
                    continue;
                }
                match rest.split_once('/') {
                    Some((child, _deeper)) => {
                        out.insert(child.to_string(), true);
                    }
                    None => {
                        // A file at this level; is_dir only if also seen
                        // as a parent elsewhere.
                        out.entry(rest.to_string()).or_insert(false);
                    }
                }
            }
            Ok(out
                .into_iter()
                .map(|(name, is_dir)| ArtifactBackendEntry {
                    name,
                    is_dir,
                    size_bytes: 0,
                })
                .collect())
        }
    }

    #[async_trait::async_trait]
    impl ArtifactBackend for S3ArtifactBackend {
        fn capabilities(&self) -> ArtifactBackendCapabilities {
            // No symlink/parent-swap semantics — the fs-shaped battery
            // cases do not apply (capability-shaped, per the TD).
            ArtifactBackendCapabilities {
                filesystem_semantics: false,
            }
        }

        async fn put(&self, path: &ArtifactPath, bytes: &[u8]) -> Result<(), ArtifactBackendError> {
            self.store
                .put(&self.full(path), Bytes::copy_from_slice(bytes))
                .await
                .map_err(err)
        }

        async fn get(
            &self,
            path: &ArtifactPath,
        ) -> Result<Option<ArtifactBackendContents>, ArtifactBackendError> {
            let full = self.full(path);
            match self.store.get(&full).await {
                Ok(bytes) => Ok(Some(ArtifactBackendContents::File(bytes.to_vec()))),
                Err(StorageError::NotFound { .. }) => {
                    // Object stores have no directories: a GET miss may be
                    // a prefix — the GET-on-directory-LISTS contract.
                    let children = self.children(&full).await?;
                    if children.is_empty() {
                        Ok(None)
                    } else {
                        Ok(Some(ArtifactBackendContents::Directory(children)))
                    }
                }
                Err(e) => Err(err(e)),
            }
        }

        async fn list(
            &self,
            path: &ArtifactPath,
        ) -> Result<Vec<ArtifactBackendEntry>, ArtifactBackendError> {
            self.children(&self.full(path)).await
        }

        async fn delete(&self, path: &ArtifactPath) -> Result<(), ArtifactBackendError> {
            // S3 DELETE of an absent key is a success — idempotence for
            // free (the port contract).
            self.store.delete(&self.full(path)).await.map_err(err)
        }
    }
}

#[cfg(feature = "aws")]
pub use imp::{S3ArtifactBackend, S3ArtifactBackendFactory};

#[cfg(not(feature = "aws"))]
mod imp {
    use proximadb_catalog::run_store::ArtifactBackendError;

    /// Feature-off stub: constructing S3 backends fails closed with a
    /// clear, actionable message (the local hardened backend remains the
    /// default — nothing degrades).
    pub struct S3ArtifactBackendFactory;

    impl S3ArtifactBackendFactory {
        pub fn new(_url: impl Into<String>) -> Self {
            Self
        }
    }

    impl proximadb_catalog::run_store::ArtifactBackendFactory for S3ArtifactBackendFactory {
        fn backend_for(
            &self,
            _tenant_id: &str,
        ) -> Result<
            std::sync::Arc<dyn proximadb_catalog::run_store::ArtifactBackend>,
            ArtifactBackendError,
        > {
            Err(ArtifactBackendError::UnsupportedPlatform(
                "the MLflow S3 artifact backend requires building with --features aws \
                 (unset PROXIMADB_MLFLOW_ARTIFACTS_URL to use the hardened local backend)"
                    .to_string(),
            ))
        }
    }
}

#[cfg(not(feature = "aws"))]
pub use imp::S3ArtifactBackendFactory;

#[cfg(all(test, feature = "aws"))]
mod tests {
    use super::*;
    use proximadb_catalog::run_store::conformance_tests::artifact_backend_conformance;

    /// The seam battery against the REAL tracked backend. Gated on
    /// PROXIMADB_MLFLOW_ARTIFACTS_TEST_URL pointing at an S3-compatible
    /// endpoint (CI: the emulator lane's MinIO; local: the dataserver2
    /// MinIO). Skips when unset — never fabricates success.
    #[tokio::test]
    async fn s3_backend_passes_seam_conformance() {
        let Ok(url) = std::env::var("PROXIMADB_MLFLOW_ARTIFACTS_TEST_URL") else {
            eprintln!("skipping: PROXIMADB_MLFLOW_ARTIFACTS_TEST_URL not set");
            return;
        };
        let factory = S3ArtifactBackendFactory::new(url);
        artifact_backend_conformance(&factory).await;
    }
}
