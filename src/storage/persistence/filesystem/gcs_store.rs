/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Google Cloud Storage [`FileSystem`] backend on the official
//! `google-cloud-storage` crate (Google's 1.x SDK; the yoshidan 0.x line the
//! name was donated from continues as `gcloud-storage`). Feature-gated behind
//! `gcp`.
//!
//! The 1.x data client (`client::Storage`) speaks the JSON API over HTTP
//! (`/storage/v1/…`), the same wire surface the 0.x `http` module used, so a
//! custom `storage_endpoint` still works against `fake-gcs-server`
//! (anonymous + custom endpoint). Metadata operations (get / delete / list)
//! live on the separate `client::StorageControl`; both clients are pointed at
//! the same endpoint. `read_range` issues a ranged object download — the
//! ADR-023 cold-load primitive. Resumable uploads (chunked, checksum-verified)
//! are SDK-managed since 1.x, replacing the hand-rolled chunk loop the 0.15
//! `UploadStatus::ResumeIncomplete` Range-header gap forced on us.

use async_trait::async_trait;
use google_cloud_auth::credentials::anonymous;
use google_cloud_gax::error::rpc::Code as GaxCode;
use google_cloud_gax::paginator::Paginator;
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model_ext::ReadRange;

use super::{
    DirEntry, FileOptions, FileSystem, FilesystemError, FilesystemFile, FsFileMetadata, FsResult,
};

/// Configuration for [`GcsFileSystem`]. `endpoint_url` + `anonymous` target a
/// local `fake-gcs-server`; leave them unset for real GCS (default credentials).
#[derive(Debug, Clone, Default)]
pub struct GcsConfig {
    /// Custom storage endpoint (e.g. `http://localhost:4443` for fake-gcs-server).
    pub endpoint_url: Option<String>,
    /// Use anonymous (credential-less) access — required for fake-gcs-server.
    pub anonymous: bool,
    /// GCP project id (optional; not needed for fake-gcs). Reserved — the 1.x
    /// SDK keys requests by bucket, not project; kept for config compatibility.
    pub project_id: Option<String>,
}

/// GCS `FileSystem` backend over `google-cloud-storage` 1.x.
pub struct GcsFileSystem {
    /// JSON API data plane: object upload / download (incl. ranged reads).
    data: Storage,
    /// Metadata plane: get / delete / list (buckets addressed as
    /// `projects/_/buckets/{bucket}`).
    control: StorageControl,
}

impl std::fmt::Debug for GcsFileSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GcsFileSystem").finish()
    }
}

impl GcsFileSystem {
    pub async fn new(cfg: GcsConfig) -> FsResult<Self> {
        let (data_builder, control_builder) = match cfg.anonymous {
            true => {
                let creds = anonymous::Builder::new().build();
                (
                    Storage::builder().with_credentials(creds.clone()),
                    StorageControl::builder().with_credentials(creds),
                )
            }
            // Default: Application Default Credentials, resolved at build time.
            false => (Storage::builder(), StorageControl::builder()),
        };
        // Both planes must agree: fake-gcs-server serves the JSON API for
        // object data AND metadata on the one endpoint.
        let (data_builder, control_builder) = match cfg.endpoint_url {
            Some(endpoint) => (
                data_builder.with_endpoint(endpoint.clone()),
                control_builder.with_endpoint(endpoint),
            ),
            None => (data_builder, control_builder),
        };
        let data = data_builder
            .build()
            .await
            .map_err(|e| FilesystemError::Auth(format!("GCS data client: {e}")))?;
        let control = control_builder
            .build()
            .await
            .map_err(|e| FilesystemError::Auth(format!("GCS control client: {e}")))?;
        Ok(Self { data, control })
    }

    /// The 1.x API addresses buckets as `projects/_/buckets/{bucket}` on both
    /// the data and control planes.
    fn bucket_param(bucket: &str) -> String {
        format!("projects/_/buckets/{bucket}")
    }

    /// Parse `gs://bucket/object` (or `gcs://`) into `(bucket, object)`.
    fn parse(path: &str) -> FsResult<(String, String)> {
        let rest = path
            .strip_prefix("gs://")
            .or_else(|| path.strip_prefix("gcs://"))
            .ok_or_else(|| FilesystemError::InvalidPath(format!("not a gs:// path: {path}")))?;
        let (bucket, object) = rest.split_once('/').ok_or_else(|| {
            FilesystemError::InvalidPath(format!("missing object in gs path: {path}"))
        })?;
        if bucket.is_empty() || object.is_empty() {
            return Err(FilesystemError::InvalidPath(format!("bad gs path: {path}")));
        }
        Ok((bucket.to_string(), object.to_string()))
    }

    fn net(ctx: &str, e: impl std::fmt::Display) -> FilesystemError {
        FilesystemError::Network(format!("{ctx}: {e}"))
    }

    /// Conditional-write preconditions surface as the GCS precondition codes;
    /// map both spellings (409 / 412) to `AlreadyExists`.
    fn is_precondition_conflict(e: &google_cloud_storage::Error) -> bool {
        e.status().is_some_and(|status| {
            matches!(
                status.code,
                GaxCode::AlreadyExists | GaxCode::FailedPrecondition
            )
        })
    }

    async fn read_all(
        &self,
        ctx: &str,
        bucket: &str,
        object: &str,
        range: Option<ReadRange>,
    ) -> FsResult<Vec<u8>> {
        let mut builder = self.data.read_object(Self::bucket_param(bucket), object);
        if let Some(range) = range {
            builder = builder.set_read_range(range);
        }
        let mut response = builder.send().await.map_err(|e| Self::net(ctx, e))?;
        let mut out = Vec::new();
        while let Some(chunk) = response.next().await {
            let chunk = chunk.map_err(|e| Self::net(ctx, e))?;
            out.extend_from_slice(&chunk);
        }
        Ok(out)
    }

    async fn list_paginated(
        &self,
        path: &str,
        // NOTE: the 1.x ListObjectsRequest has no page-size control (no
        // maxResults in the v2 model) — the parameter is accepted for
        // `FileSystem::list` compatibility and ignored; the token loop below
        // still consumes every page the service chooses to send.
        _max_results: Option<i32>,
    ) -> FsResult<Vec<DirEntry>> {
        let (bucket, prefix) = Self::parse(path)?;
        let mut entries = Vec::new();
        let mut page_token = None;

        loop {
            let requested_page_token = page_token.clone();
            let mut builder = self
                .control
                .list_objects()
                .set_parent(Self::bucket_param(&bucket))
                .set_prefix(&prefix);
            if let Some(token) = &requested_page_token {
                builder = builder.set_page_token(token);
            }
            let mut pages = builder.by_page();
            let page = pages.next().await;
            let response = match page {
                Some(Ok(response)) => response,
                Some(Err(e)) => return Err(Self::net("GCS list_objects", e)),
                None => break,
            };

            for obj in response.objects {
                let name = obj.name.rsplit('/').next().unwrap_or(&obj.name).to_string();
                entries.push(DirEntry {
                    name,
                    url: format!("gs://{bucket}/{}", obj.name),
                    metadata: FsFileMetadata {
                        path: format!("gs://{bucket}/{}", obj.name),
                        size: obj.size.max(0) as u64,
                        is_directory: false,
                        etag: Some(obj.etag),
                        ..Default::default()
                    },
                });
            }

            // 1.x models the token as a plain String; empty means "no more pages".
            page_token = if response.next_page_token.is_empty() {
                None
            } else {
                Some(response.next_page_token)
            };
            match &page_token {
                None => break,
                Some(next) if Some(next) == requested_page_token.as_ref() => {
                    return Err(FilesystemError::Network(format!(
                        "GCS list_objects returned a repeated page token for {path}"
                    )));
                }
                Some(_) => {}
            }
        }

        Ok(entries)
    }
}

#[async_trait]
impl FileSystem for GcsFileSystem {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
        let (bucket, object) = Self::parse(path)?;
        let bytes = self
            .read_all("GCS read_object", &bucket, &object, None)
            .await?;
        super::record_physical_full_read(bytes.len() as u64);
        Ok(bytes)
    }

    /// ADR-023 cold path: a ranged object download — fetches only `[offset, +length)`.
    async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        if length == 0 {
            return Ok(Vec::new());
        }
        let (bucket, object) = Self::parse(path)?;
        let bytes = self
            .read_all(
                "GCS read_object range",
                &bucket,
                &object,
                Some(ReadRange::segment(offset, length)),
            )
            .await?;
        // ADR-030 / TD-158: physical GET boundary — feed the per-query I/O accumulator
        // (task-local; no-op outside a query scope). Always-on core counters.
        super::record_physical_range_read(bytes.len() as u64);
        Ok(bytes)
    }

    async fn write(&self, path: &str, data: &[u8], options: Option<FileOptions>) -> FsResult<()> {
        let (bucket, object) = Self::parse(path)?;
        let mut builder = self.data.write_object(
            Self::bucket_param(&bucket),
            &object,
            bytes::Bytes::copy_from_slice(data),
        );
        // ADR-036: the per-object access tier is the dominant at-rest cost
        // lever. TD-173 residual (a) closed: the 1.x write carries resource
        // metadata on the simple upload, so `ObjectAccessTier::as_gcs_storage_class`
        // wires through directly. Default-safe: unset tier ⇒ bucket default.
        if let Some(tier) = options.as_ref().and_then(FileOptions::access_tier) {
            builder = builder.set_storage_class(tier.as_gcs_storage_class());
        }
        builder
            .send_unbuffered()
            .await
            .map_err(|e| Self::net("GCS write_object", e))?;
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
        let (bucket, object) = Self::parse(path)?;
        let file = tokio::fs::File::open(local_path)
            .await
            .map_err(|e| Self::net("GCS open local file", e))?;
        let bytes = file
            .metadata()
            .await
            .map_err(|e| Self::net("GCS local file metadata", e))?
            .len();
        if bytes == 0 {
            self.data
                .write_object(Self::bucket_param(&bucket), &object, bytes::Bytes::new())
                .send_unbuffered()
                .await
                .map_err(|e| Self::net("GCS empty upload", e))?;
            return Ok(0);
        }
        // SDK-managed resumable upload: `send_buffered` streams the file in
        // bounded buffers with checksum verification above the resumable
        // threshold — no hand-rolled chunk/ack loop (the 0.15 workaround).
        // Tier rides the session's resource metadata, same as `write` (S3
        // parity: aws_s3.rs applies it here too).
        let mut builder = self
            .data
            .write_object(Self::bucket_param(&bucket), &object, file);
        if let Some(tier) = options.as_ref().and_then(FileOptions::access_tier) {
            builder = builder.set_storage_class(tier.as_gcs_storage_class());
        }
        builder
            .send_buffered()
            .await
            .map_err(|e| Self::net("GCS resumable upload", e))?;
        Ok(bytes)
    }

    async fn write_if_absent(
        &self,
        path: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        let (bucket, object) = Self::parse(path)?;
        let mut builder = self
            .data
            .write_object(
                Self::bucket_param(&bucket),
                &object,
                bytes::Bytes::copy_from_slice(data),
            )
            .set_if_generation_match(0i64);
        if let Some(tier) = options.as_ref().and_then(FileOptions::access_tier) {
            builder = builder.set_storage_class(tier.as_gcs_storage_class());
        }
        match builder.send_unbuffered().await {
            Ok(_) => Ok(()),
            Err(e) if Self::is_precondition_conflict(&e) => {
                Err(FilesystemError::AlreadyExists(path.to_string()))
            }
            Err(e) => Err(Self::net("GCS conditional upload", e)),
        }
    }

    async fn append(&self, _path: &str, _data: &[u8]) -> FsResult<()> {
        Err(FilesystemError::InvalidOperation(
            "GCS objects are immutable; append is unsupported".to_string(),
        ))
    }

    async fn delete(&self, path: &str) -> FsResult<()> {
        let (bucket, object) = Self::parse(path)?;
        self.control
            .delete_object()
            .set_bucket(Self::bucket_param(&bucket))
            .set_object(&object)
            .send()
            .await
            .map_err(|e| Self::net("GCS delete_object", e))?;
        Ok(())
    }

    async fn exists(&self, path: &str) -> FsResult<bool> {
        let (bucket, object) = Self::parse(path)?;
        Ok(self
            .control
            .get_object()
            .set_bucket(Self::bucket_param(&bucket))
            .set_object(&object)
            .send()
            .await
            .is_ok())
    }

    async fn metadata(&self, path: &str) -> FsResult<FsFileMetadata> {
        let (bucket, object) = Self::parse(path)?;
        let obj = self
            .control
            .get_object()
            .set_bucket(Self::bucket_param(&bucket))
            .set_object(&object)
            .send()
            .await
            .map_err(|e| Self::net("GCS get_object", e))?;
        Ok(FsFileMetadata {
            path: path.to_string(),
            size: obj.size.max(0) as u64,
            is_directory: false,
            etag: Some(obj.etag),
            ..Default::default()
        })
    }

    async fn list(&self, path: &str) -> FsResult<Vec<DirEntry>> {
        self.list_paginated(path, None).await
    }

    async fn create_dir(&self, _path: &str) -> FsResult<()> {
        Ok(())
    }

    async fn create_dir_all(&self, _path: &str) -> FsResult<()> {
        Ok(())
    }

    async fn copy(&self, from: &str, to: &str) -> FsResult<()> {
        // GCS has a native copy/rewrite; download+upload keeps this backend lean
        // and is sufficient for the cold-path use (small index objects).
        let data = self.read(from).await?;
        self.write(to, &data, None).await
    }

    async fn move_file(&self, from: &str, to: &str) -> FsResult<()> {
        self.copy(from, to).await?;
        self.delete(from).await
    }

    fn filesystem_type(&self) -> &'static str {
        "gcs"
    }

    async fn sync(&self) -> FsResult<()> {
        Ok(())
    }

    async fn open_file(&self, _path: &str, _create: bool) -> FsResult<Box<dyn FilesystemFile>> {
        Err(FilesystemError::InvalidOperation(
            "streaming open_file is not supported on the GCS backend".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_gs_and_gcs_schemes() {
        assert_eq!(
            GcsFileSystem::parse("gs://bucket/obj.bin").unwrap(),
            ("bucket".to_string(), "obj.bin".to_string())
        );
        assert_eq!(
            GcsFileSystem::parse("gcs://bucket/a/b.bin").unwrap(),
            ("bucket".to_string(), "a/b.bin".to_string())
        );
    }

    #[test]
    fn parse_rejects_non_gcs_and_incomplete() {
        assert!(GcsFileSystem::parse("s3://b/k").is_err());
        assert!(GcsFileSystem::parse("gs://bucket-only").is_err());
        assert!(GcsFileSystem::parse("gs://b/").is_err());
        assert!(GcsFileSystem::parse("gs:///obj").is_err());
    }

    #[tokio::test]
    async fn anonymous_endpoint_config_builds_a_client() {
        // fake-gcs shape: anonymous + custom endpoint must construct without auth I/O,
        // on both the data and control planes.
        let fs = GcsFileSystem::new(GcsConfig {
            endpoint_url: Some("http://127.0.0.1:4443".into()),
            anonymous: true,
            project_id: None,
        })
        .await;
        assert!(
            fs.is_ok(),
            "anonymous+endpoint clients should build offline"
        );
    }

    #[tokio::test]
    #[ignore = "needs fake-gcs-server — set PROXIMADB_GCS_TEST_ENDPOINT"]
    async fn list_consumes_all_pages_against_fake_gcs() {
        let Ok(endpoint) = std::env::var("PROXIMADB_GCS_TEST_ENDPOINT") else {
            eprintln!("skip: set PROXIMADB_GCS_TEST_ENDPOINT with fake-gcs-server running");
            return;
        };
        let bucket = std::env::var("PROXIMADB_GCS_TEST_BUCKET")
            .unwrap_or_else(|_| "proximadb-test".to_string());
        let fs = GcsFileSystem::new(GcsConfig {
            endpoint_url: Some(endpoint),
            anonymous: true,
            project_id: Some("proximadb".to_string()),
        })
        .await
        .expect("fake-gcs client");
        let prefix = format!("td-objstore-4/{}/", uuid::Uuid::new_v4());

        for index in 0..5 {
            let path = format!("gs://{bucket}/{prefix}{index}.bcwal");
            fs.write(&path, &[index], None).await.expect("seed object");
        }

        let prefix_path = format!("gs://{bucket}/{prefix}");
        assert!(
            !fs.exists(&prefix_path).await.expect("prefix HEAD"),
            "a flat-key prefix must not exist as an exact object"
        );
        // The 1.x SDK exposes no page-size control, so this no longer forces
        // multiple pages; it pins the prefix-list contract: every seeded object
        // is returned, none of its own prefix tombstones are.
        let entries = fs
            .list_paginated(&prefix_path, Some(2))
            .await
            .expect("prefix LIST");
        assert_eq!(entries.len(), 5, "LIST must return every seeded object");

        for entry in entries {
            fs.delete(&entry.url).await.expect("cleanup object");
        }
    }
}
