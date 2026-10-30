/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Azure Blob Storage [`FileSystem`] backend on the canonical `object_store`
//! crate. Feature-gated behind `azure`.
//!
//! ## Orientation (ADR-036): this is the **Blob endpoint**, not DFS/HNS
//!
//! `MicrosoftAzureBuilder` talks to the **Blob endpoint** (`*.blob.core.windows.net`)
//! with **flat object keys**. The factory registers four schemes for this one
//! backend — `az`/`azure` (canonical) and `adls`/`abfs` (ergonomic aliases) — but
//! all four take the *same* Blob path here (`scheme://container/blob`). The aliases
//! do **not** engage the ADLS Gen2 DFS endpoint (`*.dfs.core.windows.net`), the
//! Hadoop ABFS driver, or Hierarchical Namespace. We deliberately run **flat Blob
//! (HNS-off)**: our workload is flat-key, immutable, ranged-read, so HNS adds
//! per-operation cost with no benefit. The object-storage **cost lever is the
//! access tier** (`x-ms-access-tier`: Hot/Cool/Cold/Archive), set per-PUT from
//! `FileOptions.storage_class` (see [`AzureBlobFileSystem::write`]) — not the
//! scheme and not the namespace mode.
//!
//! `object_store` is the battle-tested Rust object-store abstraction (Arrow /
//! DataFusion / Delta ecosystem). Unlike the pre-GA `azure_storage_blobs` 0.19
//! (whose ranged streaming GET hangs against Azurite), `object_store::get_range`
//! is a **true byte-range read** — the bandwidth-optimal ADR-023 cold-load
//! primitive — and `with_use_emulator(true)` targets Azurite as well as real
//! Azure / shared-key accounts.

use async_trait::async_trait;
use dashmap::DashMap;
use futures::StreamExt;
use object_store::ObjectStore;
// Extension trait providing get/get_range/put/delete/head on the object_store
// handle. Required in scope here or the bridge calls below fail to resolve
// (E0599) — this file is feature-gated (`azure`) and not built by default CI.
use object_store::ObjectStoreExt;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::path::Path as ObjPath;
use object_store::{
    Attribute, Attributes, PutMode, PutMultipartOptions, PutOptions, PutPayload, WriteMultipart,
};
use std::sync::Arc;
use tokio::io::AsyncReadExt;

use super::{
    DirEntry, FileOptions, FileSystem, FilesystemError, FilesystemFile, FsFileMetadata, FsResult,
};

/// Configuration for [`AzureBlobFileSystem`].
///
/// Auth precedence (delegated to `object_store`'s `MicrosoftAzureBuilder`):
/// 1. `use_emulator` → Azurite (well-known `devstoreaccount1` creds).
/// 2. `access_key` → shared-key (a long-lived secret — least preferred; avoid
///    in cloud where a workload/managed identity is available).
/// 3. `client_id` + `tenant_id` + `federated_token_file` → **AKS Workload
///    Identity** (OIDC federation; no secret material — the recommended posture
///    for the Azure MVP on AKS).
/// 4. `client_id` alone → **user-assigned Managed Identity** (IMDS).
/// 5. none of the above → **system-assigned Managed Identity** (IMDS) — the
///    secret-less default when the pod has an attached identity.
///
/// The identity fields are normally populated from the standard env vars the
/// AKS workload-identity webhook injects (`AZURE_CLIENT_ID` / `AZURE_TENANT_ID`
/// / `AZURE_FEDERATED_TOKEN_FILE`) — see `FilesystemFactory::azure_config_from_env`.
#[derive(Debug, Clone)]
pub struct AzureBlobConfig {
    /// Storage account name (ignored in emulator mode).
    pub account: String,
    /// Shared-key access key (not needed when `use_emulator` or when an
    /// identity is configured).
    pub access_key: Option<String>,
    /// Use the Azurite emulator (object_store fills in the well-known creds + host).
    pub use_emulator: bool,
    /// Optional custom blob endpoint (non-emulator).
    pub endpoint: Option<String>,
    /// AAD client (application/managed-identity) id. With `tenant_id` +
    /// `federated_token_file` selects Workload Identity; alone selects a
    /// user-assigned Managed Identity.
    pub client_id: Option<String>,
    /// AAD tenant id (Workload Identity).
    pub tenant_id: Option<String>,
    /// Path to the projected federated token file (Workload Identity, AKS).
    pub federated_token_file: Option<String>,
}

impl Default for AzureBlobConfig {
    fn default() -> Self {
        Self {
            account: "devstoreaccount1".to_string(),
            access_key: None,
            use_emulator: true,
            endpoint: None,
            client_id: None,
            tenant_id: None,
            federated_token_file: None,
        }
    }
}

/// Azure Blob `FileSystem` backend over `object_store`. One `ObjectStore` is
/// built (and cached) per container.
pub struct AzureBlobFileSystem {
    config: AzureBlobConfig,
    stores: DashMap<String, Arc<dyn ObjectStore>>,
}

impl std::fmt::Debug for AzureBlobFileSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureBlobFileSystem").finish()
    }
}

impl AzureBlobFileSystem {
    pub async fn new(cfg: AzureBlobConfig) -> FsResult<Self> {
        Ok(Self {
            config: cfg,
            stores: DashMap::new(),
        })
    }

    /// Parse `abfs://container/blob` (or `adls://`/`az://`/`azure://`).
    fn parse(path: &str) -> FsResult<(String, String)> {
        super::split_object_url(
            path,
            &["abfs://", "adls://", "az://", "azure://"],
            ("an", "azure"),
            "blob",
        )
    }

    /// Build (and cache) an `ObjectStore` for a container.
    fn store_for(&self, container: &str) -> FsResult<Arc<dyn ObjectStore>> {
        if let Some(s) = self.stores.get(container) {
            return Ok(s.clone());
        }
        let mut builder = MicrosoftAzureBuilder::new().with_container_name(container);
        if self.config.use_emulator {
            // Azurite speaks plaintext HTTP; allow it ONLY for the emulator.
            // Real ADLS keeps the object_store default (HTTPS enforced) so MVP
            // traffic is never downgraded to cleartext.
            builder = builder.with_allow_http(true).with_use_emulator(true);
        } else {
            builder = builder.with_account(self.config.account.clone());
            // Shared key (a secret) — only when explicitly supplied.
            if let Some(key) = &self.config.access_key {
                builder = builder.with_access_key(key.clone());
            }
            // Secret-less identity. object_store's build() resolves credentials
            // in this order: access_key → (client_id + tenant_id +
            // federated_token_file) Workload Identity → client-secret → Azure
            // CLI → IMDS Managed Identity. Setting just `client_id` selects a
            // user-assigned MI; setting nothing here leaves the system-assigned
            // MI (IMDS) as the default — so an AKS pod with an attached identity
            // needs no secret in config.
            if let Some(client_id) = &self.config.client_id {
                builder = builder.with_client_id(client_id.clone());
            }
            if let Some(tenant_id) = &self.config.tenant_id {
                builder = builder.with_tenant_id(tenant_id.clone());
            }
            if let Some(token_file) = &self.config.federated_token_file {
                builder = builder.with_federated_token_file(token_file.clone());
            }
            if let Some(ep) = &self.config.endpoint {
                builder = builder.with_endpoint(ep.clone());
            }
        }
        let store: Arc<dyn ObjectStore> = Arc::new(
            builder
                .build()
                .map_err(|e| FilesystemError::Config(format!("Azure store build: {e}")))?,
        );
        self.stores.insert(container.to_string(), store.clone());
        Ok(store)
    }

    fn net(ctx: &str, e: impl std::fmt::Display) -> FilesystemError {
        FilesystemError::Network(format!("{ctx}: {e}"))
    }
}

#[async_trait]
impl FileSystem for AzureBlobFileSystem {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
        let started = std::time::Instant::now();
        let (container, blob) = Self::parse(path)?;
        let store = self.store_for(&container)?;
        let result = store
            .get(&ObjPath::from(blob))
            .await
            .map_err(|e| super::object_store_errors::object_error("Azure get", path, e))?;
        let bytes = result
            .bytes()
            .await
            .map_err(|e| Self::net("Azure body", e))?;
        let bytes = bytes.to_vec();
        super::record_physical_full_read(bytes.len() as u64, started);
        Ok(bytes)
    }

    /// ADR-023 cold path: a TRUE ranged blob GET — fetches only `[offset, +length)`.
    async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        let started = std::time::Instant::now();
        if length == 0 {
            return Ok(Vec::new());
        }
        let (container, blob) = Self::parse(path)?;
        let store = self.store_for(&container)?;
        let bytes = store
            .get_range(&ObjPath::from(blob), offset..(offset + length))
            .await
            .map_err(|e| super::object_store_errors::object_error("Azure get_range", path, e))?;
        // ADR-030 / TD-158: physical GET boundary — feed the per-query I/O accumulator
        // (task-local; no-op outside a query scope). Always-on core counters.
        super::record_physical_range_read(bytes.len() as u64, started);
        Ok(bytes.to_vec())
    }

    async fn write(&self, path: &str, data: &[u8], options: Option<FileOptions>) -> FsResult<()> {
        let (container, blob) = Self::parse(path)?;
        let store = self.store_for(&container)?;
        let dst = ObjPath::from(blob);
        let payload = PutPayload::from(data.to_vec());
        // Access tier is the object-storage cost lever (ADR-036): map the canonical
        // `FileOptions.storage_class` to `x-ms-access-tier` on the PUT. Unset/typo ⇒
        // `None` ⇒ the account default tier (today's behavior, unchanged). The tier
        // is set per-object on the flat Blob endpoint — no ADLS HNS required.
        match options.as_ref().and_then(FileOptions::access_tier) {
            Some(tier) => {
                let mut attributes = Attributes::new();
                attributes.insert(Attribute::StorageClass, tier.as_azure_access_tier().into());
                let opts = PutOptions {
                    attributes,
                    ..Default::default()
                };
                store
                    .put_opts(&dst, payload, opts)
                    .await
                    .map_err(|e| super::object_store_errors::denied("Azure put_opts", e))?;
            }
            None => {
                store
                    .put(&dst, payload)
                    .await
                    .map_err(|e| super::object_store_errors::denied("Azure put", e))?;
            }
        }
        Ok(())
    }

    fn supports_bounded_local_file_write(&self) -> bool {
        true
    }

    async fn write_local_file(
        &self,
        path: &str,
        local_path: &std::path::Path,
        options: Option<FileOptions>,
    ) -> FsResult<u64> {
        const CHUNK_BYTES: usize = 8 * 1024 * 1024;
        const MAX_IN_FLIGHT: usize = 4;

        let (container, blob) = Self::parse(path)?;
        let store = self.store_for(&container)?;
        let mut attributes = Attributes::new();
        if let Some(tier) = options.as_ref().and_then(FileOptions::access_tier) {
            attributes.insert(Attribute::StorageClass, tier.as_azure_access_tier().into());
        }
        let mut file = tokio::fs::File::open(local_path).await?;
        let bytes = file.metadata().await?.len();
        let upload = store
            .put_multipart_opts(
                &ObjPath::from(blob),
                PutMultipartOptions {
                    attributes,
                    ..Default::default()
                },
            )
            .await
            .map_err(|error| Self::net("Azure multipart begin", error))?;
        let mut multipart = WriteMultipart::new_with_chunk_size(upload, CHUNK_BYTES);
        let mut buffer = vec![0u8; CHUNK_BYTES];
        loop {
            if let Err(error) = multipart.wait_for_capacity(MAX_IN_FLIGHT).await {
                let _ = multipart.abort().await;
                return Err(Self::net("Azure multipart part", error));
            }
            let read = match file.read(&mut buffer).await {
                Ok(read) => read,
                Err(error) => {
                    let _ = multipart.abort().await;
                    return Err(error.into());
                }
            };
            if read == 0 {
                break;
            }
            multipart.write(&buffer[..read]);
        }
        multipart
            .finish()
            .await
            .map_err(|error| Self::net("Azure multipart complete", error))?;
        Ok(bytes)
    }

    async fn write_if_absent(
        &self,
        path: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        let (container, blob) = Self::parse(path)?;
        let store = self.store_for(&container)?;
        let dst = ObjPath::from(blob);
        let mut attributes = Attributes::new();
        if let Some(tier) = options.as_ref().and_then(FileOptions::access_tier) {
            attributes.insert(Attribute::StorageClass, tier.as_azure_access_tier().into());
        }
        let result = store
            .put_opts(
                &dst,
                PutPayload::from(data.to_vec()),
                PutOptions {
                    mode: PutMode::Create,
                    attributes,
                    ..Default::default()
                },
            )
            .await;
        match result {
            Ok(_) => Ok(()),
            Err(object_store::Error::AlreadyExists { .. }) => {
                Err(FilesystemError::AlreadyExists(path.to_string()))
            }
            Err(error) => Err(super::object_store_errors::denied(
                "Azure conditional put",
                error,
            )),
        }
    }

    async fn append(&self, _path: &str, _data: &[u8]) -> FsResult<()> {
        Err(FilesystemError::InvalidOperation(
            "block blobs do not support append".to_string(),
        ))
    }

    async fn delete(&self, path: &str) -> FsResult<()> {
        let (container, blob) = Self::parse(path)?;
        let store = self.store_for(&container)?;
        store
            .delete(&ObjPath::from(blob))
            .await
            .map_err(|e| super::object_store_errors::object_error("Azure delete", path, e))?;
        Ok(())
    }

    async fn exists(&self, path: &str) -> FsResult<bool> {
        super::traced_request(super::TracedOp::Head, async move {
            let (container, blob) = Self::parse(path)?;
            let store = self.store_for(&container)?;
            super::object_store_errors::exists_from_head(
                "Azure head",
                store.head(&ObjPath::from(blob)).await,
            )
        })
        .await
    }

    async fn metadata(&self, path: &str) -> FsResult<FsFileMetadata> {
        super::traced_request(super::TracedOp::Head, async move {
            let (container, blob) = Self::parse(path)?;
            let store = self.store_for(&container)?;
            let meta = store
                .head(&ObjPath::from(blob))
                .await
                .map_err(|e| super::object_store_errors::object_error("Azure head", path, e))?;
            Ok(FsFileMetadata {
                path: path.to_string(),
                size: meta.size,
                is_directory: false,
                etag: meta.e_tag,
                ..Default::default()
            })
        })
        .await
    }

    async fn list(&self, path: &str) -> FsResult<Vec<DirEntry>> {
        super::traced_request(super::TracedOp::List, async move {
            let (container, prefix) = Self::parse(path)?;
            let store = self.store_for(&container)?;
            let mut stream = store.list(Some(&ObjPath::from(prefix)));
            let mut entries = Vec::new();
            while let Some(meta) = stream.next().await {
                let meta = meta.map_err(|e| super::object_store_errors::denied("Azure list", e))?;
                let key = meta.location.to_string();
                let name = key.rsplit('/').next().unwrap_or(&key).to_string();
                entries.push(DirEntry {
                    name,
                    url: format!("az://{container}/{key}"),
                    metadata: FsFileMetadata {
                        path: format!("az://{container}/{key}"),
                        size: meta.size,
                        is_directory: false,
                        etag: meta.e_tag,
                        ..Default::default()
                    },
                });
            }
            Ok(entries)
        })
        .await
    }

    async fn create_dir(&self, _path: &str) -> FsResult<()> {
        Ok(())
    }

    async fn create_dir_all(&self, _path: &str) -> FsResult<()> {
        Ok(())
    }

    fn filesystem_type(&self) -> &'static str {
        "adls"
    }

    async fn sync(&self) -> FsResult<()> {
        Ok(())
    }

    async fn open_file(&self, _path: &str, _create: bool) -> FsResult<Box<dyn FilesystemFile>> {
        Err(FilesystemError::InvalidOperation(
            "streaming open_file is not supported on the Azure backend".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_all_azure_schemes() {
        for path in [
            "abfs://container/blob.bin",
            "adls://container/blob.bin",
            "az://container/blob.bin",
            "azure://container/blob.bin",
        ] {
            assert_eq!(
                AzureBlobFileSystem::parse(path).unwrap(),
                ("container".to_string(), "blob.bin".to_string()),
                "scheme failed: {path}"
            );
        }
    }

    #[test]
    fn parse_keeps_nested_blob_path() {
        assert_eq!(
            AzureBlobFileSystem::parse("abfs://c/a/b/c.bin").unwrap(),
            ("c".to_string(), "a/b/c.bin".to_string())
        );
    }

    #[test]
    fn parse_rejects_non_azure_and_incomplete() {
        assert!(AzureBlobFileSystem::parse("s3://b/k").is_err());
        assert!(AzureBlobFileSystem::parse("abfs://container-only").is_err());
        assert!(AzureBlobFileSystem::parse("abfs://c/").is_err());
        assert!(AzureBlobFileSystem::parse("abfs:///b").is_err());
    }

    #[test]
    fn config_default_targets_emulator() {
        let cfg = AzureBlobConfig::default();
        assert_eq!(cfg.account, "devstoreaccount1");
        assert!(cfg.use_emulator);
    }

    #[tokio::test]
    async fn store_for_caches_per_container() {
        let fs = AzureBlobFileSystem::new(AzureBlobConfig::default())
            .await
            .unwrap();
        let a = fs.store_for("c1").unwrap();
        let b = fs.store_for("c1").unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        let _ = fs.store_for("c2").unwrap();
        assert_eq!(fs.stores.len(), 2);
    }

    /// Azure-MVP hardening: an AKS Workload Identity config (no shared key,
    /// just client/tenant/federated-token) must build a usable store — proving
    /// ADLS can be reached secret-lessly.
    #[tokio::test]
    async fn store_for_builds_with_workload_identity_no_secret() {
        let cfg = AzureBlobConfig {
            account: "teststorageacct".to_string(),
            access_key: None,
            use_emulator: false,
            endpoint: None,
            client_id: Some("00000000-0000-0000-0000-000000000001".to_string()),
            tenant_id: Some("00000000-0000-0000-0000-000000000002".to_string()),
            federated_token_file: Some(
                "/var/run/secrets/azure/tokens/azure-identity-token".to_string(),
            ),
        };
        let fs = AzureBlobFileSystem::new(cfg).await.unwrap();
        // Builds the credential provider (token file is read lazily per-request,
        // so no file needs to exist here) — i.e. workload identity is wired.
        assert!(fs.store_for("warehouse").is_ok());
    }

    /// A user-assigned Managed Identity config (client_id only, no key) must
    /// also build — the IMDS fallback path.
    #[tokio::test]
    async fn store_for_builds_with_user_assigned_managed_identity() {
        let cfg = AzureBlobConfig {
            account: "teststorageacct".to_string(),
            access_key: None,
            use_emulator: false,
            client_id: Some("00000000-0000-0000-0000-000000000003".to_string()),
            ..Default::default()
        };
        let fs = AzureBlobFileSystem::new(cfg).await.unwrap();
        assert!(fs.store_for("warehouse").is_ok());
    }
}
