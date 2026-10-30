//! Local-scratch staging for segment writers that target object-store URLs.
//!
//! The SST segment writers (`write_pax_segment_full`, `ArrowBlockWriter`) are
//! LOCAL-file writers. String-stripping a cloud staging URL into a "path"
//! creates a literal local `az:...` directory: the atomic promote then LISTs
//! the REMOTE staging prefix, finds nothing, moves nothing, and the operation
//! **false-succeeds** — on the flush path that let WAL deletion destroy the
//! only durable copy (TD-OBJSTORE-4 defect 6); on the compaction path it
//! would drop the merged output while the inputs get deleted. This primitive
//! makes the write coherent for BOTH bases: cloud targets upload a local file
//! through the backend's bounded multipart/resumable primitive; `file://` and
//! bare-local targets write a hidden sibling and atomically rename it. The
//! sibling placement is essential: configurable scratch may be another mount,
//! where rename is neither available nor atomic.

use std::sync::Arc;

use anyhow::{Context, Result};

use proximadb_storage_ports::FilesystemPort;

/// A segment write staged for a possibly-remote target URL.
pub struct StagedSegmentWrite {
    /// Remote URL to upload to on finalize (`None` ⇒ local atomic rename).
    remote_url: Option<String>,
    /// Path the local-file segment writer must write to.
    local_path: String,
    /// Final local path published by a same-directory atomic rename.
    local_final_path: Option<String>,
    /// TD-TIER-1 SC5: access tier for the uploaded segment and its sidecar
    /// (`None` ⇒ the account default — the upload is today's call).
    access_tier: Option<proximadb_storage_filesystem_types::ObjectAccessTier>,
}

impl StagedSegmentWrite {
    /// Whether `target_url` is an object-store key rather than a local path.
    /// A completed multipart upload may publish such a key directly: its
    /// uncommitted blocks are invisible until the final block-list commit.
    pub fn is_remote_target(target_url: &str) -> bool {
        target_url.contains("://") && !target_url.starts_with("file://")
    }

    /// Prepare a staged write for `target_url` (the full staging FILE url,
    /// e.g. `az://c/…/__flush/L0_x.pax` or `file:///…/__flush/L0_x.pax`).
    pub async fn begin(target_url: &str) -> Result<Self> {
        Self::begin_with_scratch(target_url, None).await
    }

    /// Prepare a staged write while placing remote-upload scratch below the
    /// caller's already-admitted local directory. Spill compaction must not
    /// escape its reserved filesystem by silently falling back to system tmp.
    pub async fn begin_in(target_url: &str, scratch_root: &std::path::Path) -> Result<Self> {
        Self::begin_with_scratch(target_url, Some(scratch_root)).await
    }

    async fn begin_with_scratch(
        target_url: &str,
        scratch_root: Option<&std::path::Path>,
    ) -> Result<Self> {
        let is_remote = Self::is_remote_target(target_url);
        if is_remote {
            let scratch = scratch_root
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(|| std::env::temp_dir().join("proximadb-flush-staging"));
            tokio::fs::create_dir_all(&scratch)
                .await
                .context("create local segment-staging scratch dir")?;
            let name = target_url.rsplit('/').next().unwrap_or("segment.bin");
            let local = scratch.join(format!("{}-{}", uuid::Uuid::new_v4().simple(), name));
            Ok(Self {
                remote_url: Some(target_url.to_string()),
                local_path: local.to_string_lossy().into_owned(),
                local_final_path: None,
                access_tier: None,
            })
        } else {
            let final_path = target_url
                .strip_prefix("file://")
                .unwrap_or(target_url)
                .to_string();
            let final_path_ref = std::path::Path::new(&final_path);
            let parent = final_path_ref
                .parent()
                .filter(|candidate| !candidate.as_os_str().is_empty())
                .unwrap_or_else(|| std::path::Path::new("."));
            tokio::fs::create_dir_all(parent)
                .await
                .context("create local staging parent dir")?;
            let name = final_path_ref
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("segment.bin");
            // The temp file MUST share the final file's directory. Only then
            // is rename guaranteed to stay on one filesystem and provide the
            // atomic visibility boundary required by embedded/file:// mode.
            let local = parent.join(format!(
                ".{name}.proximadb-{}.tmp",
                uuid::Uuid::new_v4().simple()
            ));
            Ok(Self {
                remote_url: None,
                local_path: local.to_string_lossy().into_owned(),
                local_final_path: Some(final_path),
                access_tier: None,
            })
        }
    }

    /// TD-TIER-1 SC5: publish the segment (and its Arrow sidecar, which is
    /// read with it and shares its lifetime) at `tier`. A local target has no
    /// tier and ignores it.
    #[must_use]
    pub fn with_access_tier(
        mut self,
        tier: Option<proximadb_storage_filesystem_types::ObjectAccessTier>,
    ) -> Self {
        self.access_tier = tier;
        self
    }

    fn upload_options(&self) -> Option<proximadb_storage_filesystem_types::FileOptions> {
        proximadb_storage_filesystem_types::FileOptions::for_access_tier(self.access_tier)
    }

    /// The local path the segment writer must write to.
    pub fn local_path(&self) -> &str {
        &self.local_path
    }

    /// Promote the staged bytes to the remote staging URL (no-op for local
    /// bases) and return the segment's byte count — callers must NOT probe the
    /// scratch path afterwards (it is removed here; the post-finalize size
    /// probe reading the deleted scratch file made cloud compaction fail 100%
    /// of the time in review round 1 of this fix). Sidecar-aware: an Arrow
    /// segment's `{path}.idx` (required by `ArrowBlockReader::open`) is
    /// uploaded alongside as `{remote}.idx` and its scratch removed too.
    pub async fn finalize(self, factory: &Arc<dyn FilesystemPort>) -> Result<u64> {
        self.finalize_with_policy(factory, false).await
    }

    /// Finalize only when a remote backend guarantees bounded-memory upload.
    /// The local direct-write path is already bounded and needs no backend
    /// capability check.
    pub async fn finalize_bounded(self, factory: &Arc<dyn FilesystemPort>) -> Result<u64> {
        self.finalize_with_policy(factory, true).await
    }

    /// Upload through a bounded backend while retaining remote scratch until
    /// this guard is dropped. Spill compaction uses the retained local PAX for
    /// post-publication cache promotion, avoiding an object-store reread and a
    /// segment-sized in-memory seed.
    pub async fn upload_bounded_retaining_local(
        &self,
        factory: &Arc<dyn FilesystemPort>,
    ) -> Result<u64> {
        if let Some(remote) = &self.remote_url {
            let fs = factory
                .get_filesystem(remote)
                .map_err(|e| anyhow::anyhow!("staging filesystem for {remote}: {e}"))?;
            if !fs.supports_bounded_local_file_write() {
                anyhow::bail!(
                    "{} backend does not guarantee bounded local-file publication for {remote}",
                    fs.filesystem_type()
                );
            }
            let bytes = fs
                .write_local_file(
                    remote,
                    std::path::Path::new(&self.local_path),
                    self.upload_options(),
                )
                .await
                .map_err(|e| anyhow::anyhow!("upload staged segment to {remote}: {e}"))?;
            let sidecar = format!("{}.idx", self.local_path);
            if tokio::fs::try_exists(&sidecar)
                .await
                .context("probe staged Arrow sidecar")?
            {
                fs.write_local_file(
                    &format!("{remote}.idx"),
                    std::path::Path::new(&sidecar),
                    self.upload_options(),
                )
                .await
                .map_err(|e| anyhow::anyhow!("upload Arrow sidecar to {remote}.idx: {e}"))?;
            }
            tracing::debug!(remote = %remote, bytes, "staged segment uploaded and retained locally");
            Ok(bytes)
        } else {
            self.publish_local_atomic().await
        }
    }

    async fn publish_local_atomic(&self) -> Result<u64> {
        let final_path = self
            .local_final_path
            .as_deref()
            .context("local staged segment is missing its final path")?;
        let bytes = tokio::fs::metadata(&self.local_path)
            .await
            .context("stat locally staged segment")?
            .len();

        // Publish an Arrow sidecar first when present. The PAX/segment rename
        // remains the visibility boundary, so a visible segment never points
        // at a sidecar that has not yet been published. A crash between the
        // two renames can leave only an unreferenced sidecar, which cleanup may
        // reclaim; it cannot expose a partial segment.
        let staged_sidecar = format!("{}.idx", self.local_path);
        if tokio::fs::try_exists(&staged_sidecar)
            .await
            .context("probe locally staged Arrow sidecar")?
        {
            tokio::fs::rename(&staged_sidecar, format!("{final_path}.idx"))
                .await
                .context("atomically publish local Arrow sidecar")?;
        }
        tokio::fs::rename(&self.local_path, final_path)
            .await
            .context("atomically publish local segment")?;
        Ok(bytes)
    }

    async fn finalize_with_policy(
        mut self,
        factory: &Arc<dyn FilesystemPort>,
        require_bounded_remote: bool,
    ) -> Result<u64> {
        if let Some(remote) = self.remote_url.clone() {
            let fs = factory
                .get_filesystem(&remote)
                .map_err(|e| anyhow::anyhow!("staging filesystem for {remote}: {e}"))?;
            if require_bounded_remote && !fs.supports_bounded_local_file_write() {
                anyhow::bail!(
                    "{} backend does not guarantee bounded local-file publication for {remote}",
                    fs.filesystem_type()
                );
            }
            let bytes = fs
                .write_local_file(
                    &remote,
                    std::path::Path::new(&self.local_path),
                    self.upload_options(),
                )
                .await
                .map_err(|e| anyhow::anyhow!("upload staged segment to {remote}: {e}"))?;
            let _ = tokio::fs::remove_file(&self.local_path).await;
            // Arrow sidecar pair (best-effort presence, mandatory upload if present).
            let sidecar = format!("{}.idx", self.local_path);
            if tokio::fs::try_exists(&sidecar).await.unwrap_or(false) {
                fs.write_local_file(
                    &format!("{remote}.idx"),
                    std::path::Path::new(&sidecar),
                    self.upload_options(),
                )
                .await
                .map_err(|e| anyhow::anyhow!("upload Arrow sidecar to {remote}.idx: {e}"))?;
                let _ = tokio::fs::remove_file(&sidecar).await;
            }
            self.remote_url.take();
            tracing::debug!(remote = %remote, bytes, "staged segment uploaded");
            Ok(bytes)
        } else {
            self.publish_local_atomic().await
        }
    }
}

impl Drop for StagedSegmentWrite {
    /// Leak guard: a writer failure between `begin` and `finalize` must not
    /// strand the scratch file (+ possible Arrow sidecar). After successful
    /// cloud upload or local rename the scratch paths no longer exist, so this
    /// cleanup is a no-op.
    fn drop(&mut self) {
        if self.remote_url.is_some() || self.local_final_path.is_some() {
            let _ = std::fs::remove_file(&self.local_path);
            let _ = std::fs::remove_file(format!("{}.idx", self.local_path));
        }
    }
}

/// Read a whole segment object through the filesystem port for every scheme.
/// Besides keeping cloud URL routing correct, this preserves the physical-read
/// instrumentation boundary for local segments as well.
pub async fn read_object_bytes(factory: &Arc<dyn FilesystemPort>, url: &str) -> Result<Vec<u8>> {
    factory
        .read(url)
        .await
        .map_err(|e| anyhow::anyhow!("read segment object {url}: {e}"))
}

/// A segment made readable at a LOCAL path for path-based readers
/// (`ArrowBlockReader::open`). Cloud objects download to a scratch file that is
/// removed on drop; local paths pass through untouched.
pub struct LocalizedSegment {
    path: String,
    scratch: bool,
}

impl LocalizedSegment {
    pub async fn fetch(factory: &Arc<dyn FilesystemPort>, url: &str) -> Result<Self> {
        if url.contains("://") && !url.starts_with("file://") {
            let bytes = read_object_bytes(factory, url).await?;
            let dir = std::env::temp_dir().join("proximadb-segment-scratch");
            tokio::fs::create_dir_all(&dir)
                .await
                .context("create segment scratch dir")?;
            let name = url.rsplit('/').next().unwrap_or("segment.bin");
            let path = dir.join(format!("{}-{}", uuid::Uuid::new_v4().simple(), name));
            tokio::fs::write(&path, &bytes)
                .await
                .context("stage segment for local reader")?;
            // Sidecar pair: Arrow readers hard-require `{path}.idx` — fetch it
            // when the remote has one (absent for PAX; best-effort probe).
            if let Ok(fs) = factory.get_filesystem(url) {
                let remote_idx = format!("{url}.idx");
                if let Ok(idx_bytes) = fs.read(&remote_idx).await {
                    let _ = tokio::fs::write(format!("{}.idx", path.display()), &idx_bytes).await;
                }
            }
            Ok(Self {
                path: path.to_string_lossy().into_owned(),
                scratch: true,
            })
        } else {
            Ok(Self {
                path: url.strip_prefix("file://").unwrap_or(url).to_string(),
                scratch: false,
            })
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }
}

impl Drop for LocalizedSegment {
    fn drop(&mut self) {
        if self.scratch {
            let _ = std::fs::remove_file(&self.path);
            let _ = std::fs::remove_file(format!("{}.idx", self.path));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_target_is_invisible_until_same_directory_atomic_rename() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let target = format!("file://{}/coll/data/__flush/L0_x.pax", tmp.path().display());
        let staged = StagedSegmentWrite::begin(&target).await.expect("begin");
        let final_path = tmp.path().join("coll/data/__flush/L0_x.pax");
        let staging_path = std::path::PathBuf::from(staged.local_path());
        assert_eq!(staging_path.parent(), final_path.parent());
        assert_ne!(staging_path, final_path);
        assert!(!final_path.exists(), "target must be absent before commit");
        assert!(!staged.local_path().contains("proximadb-flush-staging"));
        std::fs::write(staged.local_path(), b"segment").expect("write");
        std::fs::write(format!("{}.idx", staged.local_path()), b"index").expect("write sidecar");
        assert!(
            !final_path.exists(),
            "staged bytes must not be query-visible"
        );
        let factory = crate::test_local_port::local_port::local_port();
        staged.finalize(&factory).await.expect("finalize");
        assert_eq!(std::fs::read(&final_path).expect("read final"), b"segment");
        assert_eq!(
            std::fs::read(format!("{}.idx", final_path.display())).expect("read final sidecar"),
            b"index"
        );
        assert!(!staging_path.exists(), "atomic rename must consume staging");
    }

    #[tokio::test]
    async fn dropping_failed_local_write_reclaims_hidden_sibling() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let target = format!("file://{}/coll/data/L0_failed.pax", tmp.path().display());
        let staged = StagedSegmentWrite::begin(&target).await.expect("begin");
        let staging_path = std::path::PathBuf::from(staged.local_path());
        std::fs::write(&staging_path, b"partial").expect("write partial segment");
        std::fs::write(format!("{}.idx", staging_path.display()), b"partial-index")
            .expect("write partial sidecar");

        drop(staged);

        assert!(!staging_path.exists(), "failed staging file must be reaped");
        assert!(
            !std::path::Path::new(&format!("{}.idx", staging_path.display())).exists(),
            "failed staging sidecar must be reaped"
        );
        assert!(
            !tmp.path().join("coll/data/L0_failed.pax").exists(),
            "failed write must never publish its target"
        );
    }

    /// TD-TIER-1 SC5: a tier set on the staged write rides every upload
    /// (segment and sidecar); unset, the upload options stay `None` — the
    /// call is byte-for-byte today's.
    #[tokio::test]
    async fn access_tier_rides_the_upload_options() {
        use proximadb_storage_filesystem_types::ObjectAccessTier;
        let staged = StagedSegmentWrite::begin("s3://bucket/coll/data/L1_z.pax")
            .await
            .expect("begin");
        assert!(staged.upload_options().is_none());
        let staged = staged.with_access_tier(Some(ObjectAccessTier::Cool));
        let options = staged.upload_options().expect("tiered");
        assert_eq!(options.access_tier(), Some(ObjectAccessTier::Cool));
        assert!(options.overwrite, "same semantics as `None` options");
        // Every upload in this file goes through `upload_options` (none
        // passes a literal `None` any more).
        let source = include_str!("staged_write.rs");
        let production = source.split("#[cfg(test)]").next().unwrap_or_default();
        assert_eq!(production.matches("self.upload_options()").count(), 4);
        assert!(!production.contains("&self.local_path), None)"));
    }

    #[tokio::test]
    async fn remote_target_stages_to_scratch_never_a_literal_scheme_dir() {
        let staged = StagedSegmentWrite::begin("az://container/coll/data/__flush/L0_y.pax")
            .await
            .expect("begin");
        // The writer path must be a REAL local scratch file — never a literal
        // `az:...` path (the URL-as-local-path artifact class).
        assert!(staged.local_path().contains("proximadb-flush-staging"));
        assert!(!staged.local_path().contains("az:"));
    }

    #[test]
    fn remote_target_detection_keeps_local_paths_on_rename_path() {
        assert!(StagedSegmentWrite::is_remote_target(
            "az://container/coll/data/L3.pax"
        ));
        assert!(StagedSegmentWrite::is_remote_target(
            "s3://bucket/coll/data/L3.pax"
        ));
        assert!(StagedSegmentWrite::is_remote_target(
            "gs://bucket/coll/data/L3.pax"
        ));
        assert!(StagedSegmentWrite::is_remote_target(
            "gcs://bucket/coll/data/L3.pax"
        ));
        assert!(!StagedSegmentWrite::is_remote_target(
            "file:///var/lib/proximadb/L3.pax"
        ));
        assert!(!StagedSegmentWrite::is_remote_target(
            "/var/lib/proximadb/L3.pax"
        ));
    }
}
