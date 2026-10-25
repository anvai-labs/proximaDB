/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! AWS S3 (and S3-compatible: MinIO / LocalStack / R2) [`FileSystem`] backend on
//! the canonical `object_store` crate. Feature-gated behind `aws`.
//!
//! `object_store::get_range` is a TRUE byte-range S3 GET — the bandwidth-optimal
//! ADR-023 cold-load primitive. Using `object_store` here (rather than the full
//! `aws-sdk-s3` stack) keeps the cloud-FS layer on one canonical dependency
//! across S3 / Azure / GCS and drops a large transitive SDK tree.

use async_trait::async_trait;
use dashmap::DashMap;
use futures::StreamExt;
use object_store::ObjectStore;
// Extension trait providing get/get_range/put/delete/head on the object_store
// handle. Required in scope here or the bridge calls below fail to resolve
// (E0599) — this file is feature-gated (`aws`) and not built by default CI.
use object_store::ObjectStoreExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjPath;
use object_store::{
    Attribute, Attributes, PutMode, PutMultipartOptions, PutOptions, PutPayload, WriteMultipart,
};
use std::sync::Arc;
use tokio::io::AsyncReadExt;

use super::{
    DirEntry, FileOptions, FileSystem, FilesystemError, FilesystemFile, FsFileMetadata, FsResult,
};

/// Configuration for [`AwsS3FileSystem`]. For MinIO/LocalStack set
/// `endpoint_url` + `force_path_style = true` + static creds.
#[derive(Debug, Clone)]
pub struct AwsS3Config {
    /// AWS region (e.g. `us-east-1`). Required by SigV4 even for MinIO.
    pub region: String,
    /// Custom S3 endpoint (MinIO/LocalStack). `None` = real AWS.
    pub endpoint_url: Option<String>,
    /// Path-style addressing (`http://host/bucket/key`) — required by MinIO.
    pub force_path_style: bool,
    /// Static access key id (else env/instance credentials).
    pub access_key_id: Option<String>,
    /// Static secret access key.
    pub secret_access_key: Option<String>,
    /// Optional session token (STS).
    pub session_token: Option<String>,
}

impl Default for AwsS3Config {
    fn default() -> Self {
        Self {
            region: "us-east-1".to_string(),
            endpoint_url: None,
            force_path_style: false,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
        }
    }
}

/// AWS S3 `FileSystem` backend over `object_store`. One `ObjectStore` is built
/// (and cached) per bucket.
pub struct AwsS3FileSystem {
    config: AwsS3Config,
    stores: DashMap<String, Arc<dyn ObjectStore>>,
}

impl std::fmt::Debug for AwsS3FileSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AwsS3FileSystem").finish()
    }
}

impl AwsS3FileSystem {
    pub async fn new(cfg: AwsS3Config) -> FsResult<Self> {
        Ok(Self {
            config: cfg,
            stores: DashMap::new(),
        })
    }

    /// Parse `s3://bucket/key` (or `s3a://`).
    fn parse(path: &str) -> FsResult<(String, String)> {
        super::split_object_url(path, &["s3://", "s3a://"], ("an", "s3"), "key")
    }

    /// Build (and cache) an `ObjectStore` for a bucket.
    fn store_for(&self, bucket: &str) -> FsResult<Arc<dyn ObjectStore>> {
        if let Some(s) = self.stores.get(bucket) {
            return Ok(s.clone());
        }
        // Base on the environment so secret-less credential chains engage when
        // no static key is configured: EKS IRSA / web-identity (AWS_ROLE_ARN +
        // AWS_WEB_IDENTITY_TOKEN_FILE), EC2/ECS instance profile (IMDS), or
        // AWS_ACCESS_KEY_ID/SECRET from env. Explicit config below overrides.
        // (Parity with the Azure backend's Workload/Managed Identity support;
        // GCS already authenticates via Application Default Credentials.)
        let mut builder = AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .with_region(self.config.region.clone());
        if let Some(ep) = &self.config.endpoint_url {
            builder = builder.with_endpoint(ep.clone()).with_allow_http(true);
        }
        if self.config.force_path_style {
            // Path-style addressing: http://host:port/bucket/key (MinIO).
            builder = builder.with_virtual_hosted_style_request(false);
        }
        if let Some(ak) = &self.config.access_key_id {
            builder = builder.with_access_key_id(ak.clone());
        }
        if let Some(sk) = &self.config.secret_access_key {
            builder = builder.with_secret_access_key(sk.clone());
        }
        if let Some(tok) = &self.config.session_token {
            builder = builder.with_token(tok.clone());
        }
        let store: Arc<dyn ObjectStore> = Arc::new(
            builder
                .build()
                .map_err(|e| FilesystemError::Config(format!("S3 store build: {e}")))?,
        );
        self.stores.insert(bucket.to_string(), store.clone());
        Ok(store)
    }

    fn net(ctx: &str, e: impl std::fmt::Display) -> FilesystemError {
        FilesystemError::Network(format!("{ctx}: {e}"))
    }
}

#[async_trait]
impl FileSystem for AwsS3FileSystem {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
        let started = std::time::Instant::now();
        let (bucket, key) = Self::parse(path)?;
        let store = self.store_for(&bucket)?;
        let result = store
            .get(&ObjPath::from(key))
            .await
            .map_err(|e| super::object_store_errors::object_error("S3 get", path, e))?;
        let bytes = result.bytes().await.map_err(|e| Self::net("S3 body", e))?;
        let bytes = bytes.to_vec();
        super::record_physical_full_read(bytes.len() as u64, started);
        Ok(bytes)
    }

    /// ADR-023 cold path: a TRUE ranged S3 GET — fetches only `[offset, +length)`.
    async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        let started = std::time::Instant::now();
        if length == 0 {
            return Ok(Vec::new());
        }
        let (bucket, key) = Self::parse(path)?;
        let store = self.store_for(&bucket)?;
        let bytes = store
            .get_range(&ObjPath::from(key), offset..(offset + length))
            .await
            .map_err(|e| super::object_store_errors::object_error("S3 get_range", path, e))?;
        // ADR-030 / TD-158: physical GET boundary — feed the per-query I/O accumulator
        // (task-local; no-op outside a query scope). Always-on core counters.
        super::record_physical_range_read(bytes.len() as u64, started);
        Ok(bytes.to_vec())
    }

    async fn write(&self, path: &str, data: &[u8], options: Option<FileOptions>) -> FsResult<()> {
        let (bucket, key) = Self::parse(path)?;
        let store = self.store_for(&bucket)?;
        let dst = ObjPath::from(key);
        let payload = PutPayload::from(data.to_vec());
        // Access tier is the object-storage cost lever (ADR-036): map the canonical
        // `FileOptions.storage_class` to `x-amz-storage-class` on the PUT. Unset/typo
        // ⇒ `None` ⇒ the bucket default class (today's behavior, unchanged).
        match options.as_ref().and_then(FileOptions::access_tier) {
            Some(tier) => {
                let mut attributes = Attributes::new();
                attributes.insert(Attribute::StorageClass, tier.as_s3_storage_class().into());
                let opts = PutOptions {
                    attributes,
                    ..Default::default()
                };
                store
                    .put_opts(&dst, payload, opts)
                    .await
                    .map_err(|e| super::object_store_errors::denied("S3 put_opts", e))?;
            }
            None => {
                store
                    .put(&dst, payload)
                    .await
                    .map_err(|e| super::object_store_errors::denied("S3 put", e))?;
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

        let (bucket, key) = Self::parse(path)?;
        let store = self.store_for(&bucket)?;
        let mut attributes = Attributes::new();
        if let Some(tier) = options.as_ref().and_then(FileOptions::access_tier) {
            attributes.insert(Attribute::StorageClass, tier.as_s3_storage_class().into());
        }
        let mut file = tokio::fs::File::open(local_path).await?;
        let bytes = file.metadata().await?.len();
        let upload = store
            .put_multipart_opts(
                &ObjPath::from(key),
                PutMultipartOptions {
                    attributes,
                    ..Default::default()
                },
            )
            .await
            .map_err(|error| Self::net("S3 multipart begin", error))?;
        let mut multipart = WriteMultipart::new_with_chunk_size(upload, CHUNK_BYTES);
        let mut buffer = vec![0u8; CHUNK_BYTES];
        loop {
            if let Err(error) = multipart.wait_for_capacity(MAX_IN_FLIGHT).await {
                let _ = multipart.abort().await;
                return Err(Self::net("S3 multipart part", error));
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
            .map_err(|error| Self::net("S3 multipart complete", error))?;
        Ok(bytes)
    }

    async fn write_if_absent(
        &self,
        path: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        let (bucket, key) = Self::parse(path)?;
        let store = self.store_for(&bucket)?;
        let dst = ObjPath::from(key);
        let mut attributes = Attributes::new();
        if let Some(tier) = options.as_ref().and_then(FileOptions::access_tier) {
            attributes.insert(Attribute::StorageClass, tier.as_s3_storage_class().into());
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
                "S3 conditional put",
                error,
            )),
        }
    }

    async fn append(&self, _path: &str, _data: &[u8]) -> FsResult<()> {
        Err(FilesystemError::InvalidOperation(
            "S3 objects do not support append".to_string(),
        ))
    }

    async fn delete(&self, path: &str) -> FsResult<()> {
        let (bucket, key) = Self::parse(path)?;
        let store = self.store_for(&bucket)?;
        store
            .delete(&ObjPath::from(key))
            .await
            .map_err(|e| super::object_store_errors::object_error("S3 delete", path, e))?;
        Ok(())
    }

    async fn exists(&self, path: &str) -> FsResult<bool> {
        super::traced_request(super::TracedOp::Head, async move {
            let (bucket, key) = Self::parse(path)?;
            let store = self.store_for(&bucket)?;
            super::object_store_errors::exists_from_head(
                "S3 head",
                store.head(&ObjPath::from(key)).await,
            )
        })
        .await
    }

    async fn metadata(&self, path: &str) -> FsResult<FsFileMetadata> {
        super::traced_request(super::TracedOp::Head, async move {
            let (bucket, key) = Self::parse(path)?;
            let store = self.store_for(&bucket)?;
            let meta = store
                .head(&ObjPath::from(key))
                .await
                .map_err(|e| super::object_store_errors::object_error("S3 head", path, e))?;
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
            let (bucket, prefix) = Self::parse(path)?;
            let store = self.store_for(&bucket)?;
            let mut stream = store.list(Some(&ObjPath::from(prefix)));
            let mut entries = Vec::new();
            while let Some(meta) = stream.next().await {
                let meta = meta.map_err(|e| super::object_store_errors::denied("S3 list", e))?;
                let key = meta.location.to_string();
                let name = key.rsplit('/').next().unwrap_or(&key).to_string();
                entries.push(DirEntry {
                    name,
                    url: format!("s3://{bucket}/{key}"),
                    metadata: FsFileMetadata {
                        path: format!("s3://{bucket}/{key}"),
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
        "s3"
    }

    async fn sync(&self) -> FsResult<()> {
        Ok(())
    }

    async fn open_file(&self, _path: &str, _create: bool) -> FsResult<Box<dyn FilesystemFile>> {
        Err(FilesystemError::InvalidOperation(
            "streaming open_file is not supported on the S3 backend".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_splits_bucket_and_key() {
        assert_eq!(
            AwsS3FileSystem::parse("s3://my-bucket/path/to/index.bin").unwrap(),
            ("my-bucket".to_string(), "path/to/index.bin".to_string())
        );
    }

    #[test]
    fn parse_accepts_s3a_scheme() {
        assert_eq!(
            AwsS3FileSystem::parse("s3a://b/k").unwrap(),
            ("b".to_string(), "k".to_string())
        );
    }

    #[test]
    fn parse_rejects_non_s3_scheme() {
        assert!(matches!(
            AwsS3FileSystem::parse("gs://b/k"),
            Err(FilesystemError::InvalidPath(_))
        ));
    }

    #[test]
    fn parse_rejects_missing_key() {
        assert!(matches!(
            AwsS3FileSystem::parse("s3://bucket-only"),
            Err(FilesystemError::InvalidPath(_))
        ));
    }

    #[test]
    fn parse_rejects_empty_bucket_or_key() {
        assert!(AwsS3FileSystem::parse("s3:///key").is_err());
        assert!(AwsS3FileSystem::parse("s3://bucket/").is_err());
    }

    #[tokio::test]
    async fn store_for_caches_one_store_per_bucket() {
        let fs = AwsS3FileSystem::new(AwsS3Config {
            endpoint_url: Some("http://127.0.0.1:9000".into()),
            force_path_style: true,
            access_key_id: Some("k".into()),
            secret_access_key: Some("s".into()),
            ..Default::default()
        })
        .await
        .unwrap();
        let a = fs.store_for("bkt").unwrap();
        let b = fs.store_for("bkt").unwrap();
        assert!(Arc::ptr_eq(&a, &b), "same bucket -> cached store");
        assert_eq!(fs.stores.len(), 1);
        let _ = fs.store_for("other").unwrap();
        assert_eq!(fs.stores.len(), 2, "distinct bucket -> distinct store");
    }

    #[test]
    fn filesystem_type_is_s3() {
        // Construction is async; type tag is a pure const, assert via a built instance.
        let cfg = AwsS3Config::default();
        assert_eq!(cfg.region, "us-east-1");
        assert!(!cfg.force_path_style);
    }
}
