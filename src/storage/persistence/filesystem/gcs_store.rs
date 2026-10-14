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
//! Two planes, deliberately split:
//!
//! * **Data** — `client::Storage`, the SDK's JSON API client
//!   (`/storage/v1/…` over HTTP). Ranged reads (the ADR-023 cold-load
//!   primitive) and resumable uploads are SDK-managed; `write_if_absent`
//!   rides the `ifGenerationMatch=0` precondition. A custom
//!   `storage_endpoint` still works against `fake-gcs-server` (anonymous +
//!   custom endpoint).
//! * **Metadata** (get / delete / list) — a small hand-rolled JSON API client
//!   on the workspace `reqwest`, tokened by `google-cloud-auth`. The 1.x SDK's
//!   metadata client (`client::StorageControl`) speaks `google.storage.v2`
//!   over gRPC only, and `fake-gcs-server` — the emulator every GCS contract
//!   test and the qa-gate lane verify against (TD-OBJSTORE-5) — does not
//!   serve gRPC at all. The JSON API is the surface fake-gcs emulates and the
//!   same wire surface the 0.x client used for these verbs, so both the
//!   emulator verification path and production behavior are preserved.

use async_trait::async_trait;
use google_cloud_auth::credentials::{self, CacheableResource, Credentials, anonymous};
use google_cloud_storage::client::Storage;
use google_cloud_storage::model_ext::ReadRange;
use serde::Deserialize;

use super::{
    DirEntry, FileOptions, FileSystem, FilesystemError, FilesystemFile, FsFileMetadata, FsResult,
};

/// Configuration for [`GcsFileSystem`]. `endpoint_url` + `anonymous` target a
/// local `fake-gcs-server`; leave them unset for real GCS (default credentials).
#[derive(Debug, Clone, Default)]
pub struct GcsConfig {
    /// Custom storage endpoint (e.g. `http://localhost:4443` for fake-gcs-server).
    /// Serves as the base URL for BOTH planes (data SDK and metadata JSON API).
    pub endpoint_url: Option<String>,
    /// Use anonymous (credential-less) access — required for fake-gcs-server.
    pub anonymous: bool,
    /// GCP project id (optional; not needed for fake-gcs). Reserved — the 1.x
    /// SDK keys requests by bucket, not project; kept for config compatibility.
    pub project_id: Option<String>,
}

/// GCS `FileSystem` backend over `google-cloud-storage` 1.x (data plane) plus
/// a JSON API metadata client.
pub struct GcsFileSystem {
    /// JSON API data plane: object upload / download (incl. ranged reads).
    data: Storage,
    /// Metadata plane: get / delete / list over the JSON API.
    json: MetadataClient,
}

impl std::fmt::Debug for GcsFileSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GcsFileSystem").finish()
    }
}

/// Metadata plane over the GCS JSON API (`/storage/v1`).
///
/// See the module docs for why this is not `client::StorageControl`: that
/// client is gRPC-only, which the fake-gcs verification path cannot serve.
struct MetadataClient {
    http: reqwest::Client,
    /// JSON API base: custom endpoint (fake-gcs) or the production host.
    base: String,
    /// ADC credentials for real GCS; `None` = anonymous (fake-gcs).
    creds: Option<Credentials>,
}

/// Metadata calls are bounded single-object / single-page requests.
const METADATA_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

impl MetadataClient {
    fn new(endpoint_url: Option<String>, creds: Option<Credentials>) -> FsResult<Self> {
        let http = reqwest::Client::builder()
            .timeout(METADATA_TIMEOUT)
            .build()
            .map_err(|e| FilesystemError::Auth(format!("GCS metadata client: {e}")))?;
        Ok(Self {
            http,
            base: endpoint_url.unwrap_or_else(|| "https://storage.googleapis.com".to_string()),
            creds,
        })
    }

    fn net(ctx: &str, e: impl std::fmt::Display) -> FilesystemError {
        FilesystemError::Network(format!("{ctx}: {e}"))
    }

    /// Attach ADC auth headers when running against real GCS. Token fetch and
    /// refresh are handled by google-cloud-auth (cached, background-refreshed).
    async fn authorized(
        &self,
        mut rb: reqwest::RequestBuilder,
    ) -> FsResult<reqwest::RequestBuilder> {
        if let Some(creds) = &self.creds {
            let headers = creds
                .headers(http::Extensions::new())
                .await
                .map_err(|e| FilesystemError::Auth(format!("GCS auth headers: {e}")))?;
            if let CacheableResource::New { data, .. } = headers {
                for (name, value) in data.iter() {
                    if let Ok(v) = value.to_str() {
                        rb = rb.header(name.as_str(), v);
                    }
                }
            }
        }
        Ok(rb)
    }

    async fn get(&self, url: &str) -> FsResult<reqwest::Response> {
        self.authorized(self.http.get(url))
            .await?
            .send()
            .await
            .map_err(|e| Self::net("GCS metadata GET", e))
    }

    async fn delete(&self, url: &str) -> FsResult<reqwest::Response> {
        self.authorized(self.http.delete(url))
            .await?
            .send()
            .await
            .map_err(|e| Self::net("GCS metadata DELETE", e))
    }

    /// `{base}/storage/v1/b/{bucket}/o/{object}` — the JSON API object resource.
    fn object_url(&self, bucket: &str, object: &str) -> String {
        format!(
            "{}/storage/v1/b/{}/o/{}",
            self.base,
            percent_encode(bucket, true),
            percent_encode(object, true)
        )
    }

    /// GET the object resource: `Some(metadata)` on 200, `None` on 404.
    async fn get_object(&self, bucket: &str, object: &str) -> FsResult<Option<serde_json::Value>> {
        let url = self.object_url(bucket, object);
        let response = self.get(&url).await?;
        let status = response.status().as_u16();
        let body = response
            .bytes()
            .await
            .map_err(|e| Self::net("GCS metadata body", e))?;
        match status {
            200 => Ok(Some(
                serde_json::from_slice(&body).map_err(|e| Self::net("GCS metadata decode", e))?,
            )),
            404 => Ok(None),
            code => Err(Self::net(
                "GCS metadata GET",
                format!("HTTP {code}: {}", Self::body_snippet(&body)),
            )),
        }
    }

    /// One LIST page: `max_results` → `maxResults`, `page_token` → `pageToken`.
    async fn list_page(
        &self,
        bucket: &str,
        prefix: &str,
        page_token: Option<&str>,
        max_results: Option<i32>,
    ) -> FsResult<GcsListPage> {
        let mut url = format!(
            "{}/storage/v1/b/{}/o?prefix={}",
            self.base,
            percent_encode(bucket, true),
            percent_encode(prefix, true)
        );
        if let Some(token) = page_token {
            url.push_str("&pageToken=");
            url.push_str(&percent_encode(token, false));
        }
        if let Some(max) = max_results {
            url.push_str(&format!("&maxResults={max}"));
        }
        let response = self.get(&url).await?;
        let status = response.status().as_u16();
        let body = response
            .bytes()
            .await
            .map_err(|e| Self::net("GCS list body", e))?;
        if status != 200 {
            return Err(Self::net(
                "GCS list",
                format!("HTTP {status}: {}", Self::body_snippet(&body)),
            ));
        }
        serde_json::from_slice(&body).map_err(|e| Self::net("GCS list decode", e))
    }

    fn body_snippet(body: &[u8]) -> String {
        String::from_utf8_lossy(body).chars().take(200).collect()
    }
}

/// RFC 3986 unreserved set stays bare; everything else percent-escaped.
/// Object names and prefixes are slash-separated — keep `/` bare in path
/// positions (`keep_slash`), escape it in query positions (page tokens).
fn percent_encode(value: &str, keep_slash: bool) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        let bare = byte.is_ascii_alphanumeric()
            || matches!(byte, b'-' | b'.' | b'_' | b'~')
            || (keep_slash && byte == b'/');
        if bare {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GcsListPage {
    #[serde(default)]
    items: Vec<GcsListItem>,
    /// JSON API: absent on the last page.
    #[serde(default)]
    next_page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GcsListItem {
    name: String,
    /// JSON API sizes arrive as strings.
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    etag: Option<String>,
}

impl GcsFileSystem {
    pub async fn new(cfg: GcsConfig) -> FsResult<Self> {
        // One credential source for both planes: anonymous (fake-gcs) or ADC
        // (resolved once here, shared with the data client).
        let (creds, data_builder) = match cfg.anonymous {
            true => (
                None,
                Storage::builder().with_credentials(anonymous::Builder::new().build()),
            ),
            false => {
                let creds = credentials::Builder::default()
                    .build()
                    .map_err(|e| FilesystemError::Auth(format!("GCS default credentials: {e}")))?;
                (
                    Some(creds.clone()),
                    Storage::builder().with_credentials(creds),
                )
            }
        };
        let data_builder = match cfg.endpoint_url.clone() {
            Some(endpoint) => data_builder.with_endpoint(endpoint),
            None => data_builder,
        };
        let data = data_builder
            .build()
            .await
            .map_err(|e| FilesystemError::Auth(format!("GCS data client: {e}")))?;
        let json = MetadataClient::new(cfg.endpoint_url, creds)?;
        Ok(Self { data, json })
    }

    /// The 1.x API addresses buckets as `projects/_/buckets/{bucket}` on the
    /// data plane (the JSON API metadata plane takes the bare bucket name).
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

    /// Conditional-write preconditions surface as HTTP 409/412 on the JSON
    /// API. google-cloud-gax derives `status.code` ONLY from the response
    /// body's `status` string — GCS error bodies carry `error.code` +
    /// `error.errors[].reason` and no `status` string, so the parsed `Code`
    /// comes back `Unknown`; the HTTP status is the reliable signal (this is
    /// how the 0.x backend detected it too).
    fn is_precondition_conflict(e: &google_cloud_storage::Error) -> bool {
        e.http_status_code()
            .is_some_and(|code| matches!(code, 409 | 412))
    }

    /// One sequential pass over `file` computing its CRC32C, then rewind to
    /// byte 0 so the caller can hand the file to the SDK for upload. Used to
    /// supply the upfront checksum the write paths send instead of the 1.x
    /// SDK's trailing-`checksums` multipart part (see the `crc32c` dep comment
    /// in Cargo.toml).
    async fn hash_local_file(file: &mut tokio::fs::File) -> FsResult<u32> {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};
        let mut sum: u32 = 0;
        let mut buf = vec![0u8; 256 * 1024];
        loop {
            let n = file
                .read(&mut buf)
                .await
                .map_err(|e| Self::net("GCS local file checksum pass", e))?;
            if n == 0 {
                break;
            }
            sum = crc32c::crc32c_append(sum, &buf[..n]);
        }
        file.seek(std::io::SeekFrom::Start(0))
            .await
            .map_err(|e| Self::net("GCS local file rewind", e))?;
        Ok(sum)
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
        // Page-size pin (ops/test hook via PROXIMADB_GCS_LIST_PAGE_SIZE);
        // `None` = service default page.
        max_results: Option<i32>,
    ) -> FsResult<Vec<DirEntry>> {
        let (bucket, prefix) = Self::parse(path)?;
        let mut entries = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let requested_page_token = page_token.clone();
            let page = self
                .json
                .list_page(
                    &bucket,
                    &prefix,
                    requested_page_token.as_deref(),
                    max_results,
                )
                .await?;

            for item in &page.items {
                let name = item
                    .name
                    .rsplit('/')
                    .next()
                    .unwrap_or(&item.name)
                    .to_string();
                entries.push(DirEntry {
                    name,
                    url: format!("gs://{bucket}/{}", item.name),
                    metadata: FsFileMetadata {
                        path: format!("gs://{bucket}/{}", item.name),
                        size: item
                            .size
                            .as_deref()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0),
                        is_directory: false,
                        etag: item.etag.clone(),
                        ..Default::default()
                    },
                });
            }

            page_token = page.next_page_token;
            match &page_token {
                None => break,
                Some(next) if Some(next.as_str()) == requested_page_token.as_deref() => {
                    return Err(FilesystemError::Network(format!(
                        "GCS list returned a repeated page token for {path}"
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
        let mut builder = self
            .data
            .write_object(
                Self::bucket_param(&bucket),
                &object,
                bytes::Bytes::copy_from_slice(data),
            )
            // Supply the payload's CRC32C upfront (see the `crc32c` dep comment
            // in Cargo.toml): the 1.x SDK's default single-shot upload appends
            // a trailing `checksums` multipart part that fake-gcs-server
            // corrupts into the stored object. Upfront checksum keeps
            // end-to-end verification and adds real-GCS server-side upload
            // validation.
            .with_known_crc32c(crc32c::crc32c(data));
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
                .with_known_crc32c(0u32) // crc32c of the empty payload
                .send_unbuffered()
                .await
                .map_err(|e| Self::net("GCS empty upload", e))?;
            return Ok(0);
        }
        // Same upfront-CRC32C protocol as `write` (see comment there): hash the
        // file in one sequential pass (rewinding it for the upload). The pass
        // is page-cache-warm in the flush path and replaces the SDK's own
        // pre-upload hashing pass, so it is not extra I/O.
        let mut file = file;
        let sum = Self::hash_local_file(&mut file).await?;
        // SDK-managed resumable upload: `send_buffered` streams the file in
        // bounded buffers with checksum verification above the resumable
        // threshold — no hand-rolled chunk/ack loop (the 0.15 workaround).
        // Tier rides the session's resource metadata, same as `write` (S3
        // parity: aws_s3.rs applies it here too).
        let mut builder = self
            .data
            .write_object(Self::bucket_param(&bucket), &object, file)
            .with_known_crc32c(sum);
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
            .with_known_crc32c(crc32c::crc32c(data)) // upfront checksum, see `write`
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
        let url = self.json.object_url(&bucket, &object);
        let response = self.json.delete(&url).await?;
        let status = response.status().as_u16();
        if status == 200 || status == 204 {
            Ok(())
        } else {
            // Missing-object deletes error (Network) — same contract as the
            // 0.x backend's delete_object.
            Err(FilesystemError::Network(format!(
                "GCS delete: HTTP {status} for {path}"
            )))
        }
    }

    async fn exists(&self, path: &str) -> FsResult<bool> {
        let (bucket, object) = Self::parse(path)?;
        // A failed metadata GET (auth/quota/…) reports `false` — same
        // swallow-the-error contract the 0.x backend had (`is_ok()`).
        Ok(self.json.get_object(&bucket, &object).await?.is_some())
    }

    async fn metadata(&self, path: &str) -> FsResult<FsFileMetadata> {
        let (bucket, object) = Self::parse(path)?;
        let obj = self
            .json
            .get_object(&bucket, &object)
            .await?
            .ok_or_else(|| FilesystemError::Network(format!("GCS object not found: {path}")))?;
        Ok(FsFileMetadata {
            path: path.to_string(),
            size: obj
                .get("size")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            is_directory: false,
            etag: obj.get("etag").and_then(|v| v.as_str()).map(str::to_string),
            ..Default::default()
        })
    }

    async fn list(&self, path: &str) -> FsResult<Vec<DirEntry>> {
        // Ops/test hook: pin the page size to force multi-page traversal —
        // the pagination regression net (TD-OBJSTORE-4 S1). Unset ⇒ service
        // default page.
        let max_results = std::env::var("PROXIMADB_GCS_LIST_PAGE_SIZE")
            .ok()
            .and_then(|v| v.parse::<i32>().ok());
        self.list_paginated(path, max_results).await
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

    #[test]
    fn percent_encode_escapes_query_and_path_positions() {
        // Path position: `/` and unreserved stay bare.
        assert_eq!(
            percent_encode("a/b~c_1.bin", true),
            "a/b~c_1.bin".to_string()
        );
        // Query position: `/` is escaped (page tokens are base64-ish URLs).
        assert_eq!(
            percent_encode("a/b+c=d&e", false),
            "a%2Fb%2Bc%3Dd%26e".to_string()
        );
    }

    #[tokio::test]
    async fn anonymous_endpoint_config_builds_a_client() {
        // fake-gcs shape: anonymous + custom endpoint must construct without auth I/O,
        // on both the data client and the metadata layer.
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
        // maxResults=2 against 5 seeded objects forces three pages — exercising
        // the pageToken continuation loop and the repeated-token guard.
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
