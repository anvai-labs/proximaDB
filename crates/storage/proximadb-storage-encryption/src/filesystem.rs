// Encrypted filesystem wrapper
//
// Provides transparent encryption for whole-file filesystem operations.
// This keeps storage engines unchanged while moving encryption concerns into
// a single adapter layer.

use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use tracing::debug;

use crate::{FileEncryptionLayer, KeyVersionManager};
use proximadb_storage_filesystem_types::{
    DirEntry, FileMetadata, FileOptions, FileSystem, FilesystemError, FilesystemFile, FsResult,
};

/// Encrypted filesystem wrapper that transparently encrypts data at rest.
///
/// This adapter is intentionally whole-file oriented today. Standard read/write
/// calls work transparently, but streaming handles are rejected until a correct
/// truncate/rewrite story exists for encrypted files.
pub struct EncryptedFilesystem {
    /// Underlying filesystem (local, S3, Azure, GCS, etc.).
    underlying: Arc<dyn FileSystem>,
    /// File encryption layer.
    encryption: Arc<FileEncryptionLayer>,
    /// Optional extension used to distinguish encrypted files on disk.
    encrypted_extension: Option<String>,
}

impl std::fmt::Debug for EncryptedFilesystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptedFilesystem")
            .field("underlying", &"<dyn FileSystem>")
            .field("encryption_enabled", &self.encryption.is_enabled())
            .field("encrypted_extension", &self.encrypted_extension)
            .finish()
    }
}

impl EncryptedFilesystem {
    /// Create a new encrypted filesystem wrapper.
    pub fn new(
        underlying: Arc<dyn FileSystem>,
        key_manager: Arc<KeyVersionManager>,
        encryption_enabled: bool,
    ) -> Self {
        let encryption = Arc::new(FileEncryptionLayer::new(
            key_manager,
            encryption_enabled,
            4096,
        ));

        Self {
            underlying,
            encryption,
            encrypted_extension: Some(".enc".to_string()),
        }
    }

    /// Create with a custom encrypted file extension.
    pub fn with_extension(mut self, extension: String) -> Self {
        self.encrypted_extension = Some(extension);
        self
    }

    /// Use the same filename for encrypted files.
    pub fn without_extension(mut self) -> Self {
        self.encrypted_extension = None;
        self
    }

    fn actual_path(&self, path: &str) -> String {
        if let Some(ref ext) = self.encrypted_extension
            && !path.ends_with(ext)
        {
            return format!("{}{}", path, ext);
        }
        path.to_string()
    }

    fn crypto_error(action: &str, path: &str, err: impl std::fmt::Display) -> FilesystemError {
        FilesystemError::InvalidOperation(format!("{} failed for {}: {}", action, path, err))
    }

    fn slice_range(data: &[u8], range: Range<u64>) -> Vec<u8> {
        let start = range.start as usize;
        if start >= data.len() {
            return vec![];
        }

        let end = (range.end as usize).min(data.len());
        if end <= start {
            return vec![];
        }

        data[start..end].to_vec()
    }

    async fn encrypted_exists(&self, path: &str) -> FsResult<bool> {
        let actual_path = self.actual_path(path);
        if actual_path == path {
            return self.underlying.exists(path).await;
        }
        self.underlying.exists(&actual_path).await
    }

    async fn read_encrypted(&self, path: &str) -> FsResult<Option<Vec<u8>>> {
        let actual_path = self.actual_path(path);
        if !self.encrypted_exists(path).await? {
            return Ok(None);
        }

        if !self.encryption.is_enabled() {
            return self.underlying.read(&actual_path).await.map(Some);
        }

        let encrypted = self.underlying.read(&actual_path).await?;
        let decrypted = self
            .encryption
            .decrypt_file(path, &encrypted)
            .map_err(|e| Self::crypto_error("decryption", path, e))?;
        Ok(Some(decrypted))
    }

    async fn write_encrypted(
        &self,
        path: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        let actual_path = self.actual_path(path);
        let encrypted = self
            .encryption
            .encrypt_file(path, data)
            .map_err(|e| Self::crypto_error("encryption", path, e))?;
        self.underlying
            .write(&actual_path, &encrypted, options)
            .await
    }
}

#[async_trait]
impl FileSystem for EncryptedFilesystem {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
        if let Some(decrypted) = self.read_encrypted(path).await? {
            return Ok(decrypted);
        }

        debug!(
            "Reading plaintext file for backward compatibility: {}",
            path
        );
        self.underlying.read(path).await
    }

    async fn get_mmap(&self, _path: &str) -> FsResult<Option<memmap2::Mmap>> {
        // Encrypted data cannot be exposed as a stable mmap without a decrypted
        // materialization layer, so fall back to regular reads.
        Ok(None)
    }

    async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        if let Some(decrypted) = self.read_encrypted(path).await? {
            return Ok(Self::slice_range(
                &decrypted,
                offset..offset.saturating_add(length),
            ));
        }

        self.underlying.read_range(path, offset, length).await
    }

    fn range_coalesce_policy(
        &self,
    ) -> Option<proximadb_storage_filesystem_types::RangeCoalescePolicy> {
        // Delegate so the policy query reaches whichever layer holds it.
        //
        // Known divergence, pre-existing and NOT introduced by coalescing: for an
        // encrypted file the branch below issues ONE whole-object read and slices
        // it, while a `CountingFileSystem` above records one physical read per
        // planned range. The plan describes the unencrypted path; encryption
        // short-circuits it. Tracked separately — do not read the physical-GET
        // meter as truth for encrypted collections.
        self.underlying.range_coalesce_policy()
    }

    async fn read_ranges(
        &self,
        path: &str,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> FsResult<Vec<Vec<u8>>> {
        if let Some(decrypted) = self.read_encrypted(path).await? {
            return Ok(ranges
                .into_iter()
                .map(|range| Self::slice_range(&decrypted, range))
                .collect());
        }

        self.underlying.read_ranges(path, ranges).await
    }

    async fn write(&self, path: &str, data: &[u8], options: Option<FileOptions>) -> FsResult<()> {
        self.write_encrypted(path, data, options).await
    }

    async fn write_if_absent(
        &self,
        path: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        let actual_path = self.actual_path(path);
        let encrypted = self
            .encryption
            .encrypt_file(path, data)
            .map_err(|e| Self::crypto_error("encryption", path, e))?;
        self.underlying
            .write_if_absent(&actual_path, &encrypted, options)
            .await
    }

    async fn sync_file(&self, path: &str) -> FsResult<()> {
        if self.encrypted_exists(path).await? {
            let actual_path = self.actual_path(path);
            return self.underlying.sync_file(&actual_path).await;
        }

        self.underlying.sync_file(path).await
    }

    async fn append(&self, path: &str, data: &[u8]) -> FsResult<()> {
        let mut existing = if self.exists(path).await? {
            self.read(path).await?
        } else {
            Vec::new()
        };
        existing.extend_from_slice(data);
        self.write(path, &existing, None).await
    }

    fn supports_append(&self) -> bool {
        self.underlying.supports_append()
    }

    async fn delete(&self, path: &str) -> FsResult<()> {
        let actual_path = self.actual_path(path);
        let mut deleted = false;

        if self.encrypted_exists(path).await? {
            self.underlying.delete(&actual_path).await?;
            deleted = true;
        }

        if actual_path != path && self.underlying.exists(path).await? {
            self.underlying.delete(path).await?;
            deleted = true;
        }

        if deleted {
            Ok(())
        } else {
            Err(FilesystemError::NotFound(format!(
                "File not found: {}",
                path
            )))
        }
    }

    async fn exists(&self, path: &str) -> FsResult<bool> {
        if self.encrypted_exists(path).await? {
            return Ok(true);
        }
        self.underlying.exists(path).await
    }

    async fn metadata(&self, path: &str) -> FsResult<FileMetadata> {
        if self.encrypted_exists(path).await? {
            let actual_path = self.actual_path(path);
            let mut metadata = self.underlying.metadata(&actual_path).await?;
            metadata.path = path.to_string();

            if let Ok(header) = self.underlying.read_range(&actual_path, 0, 37).await
                && let Ok(encrypted_metadata) = self.encryption.get_metadata(&header)
            {
                metadata.size = encrypted_metadata.original_size;
            }

            return Ok(metadata);
        }

        self.underlying.metadata(path).await
    }

    /// List a prefix, reporting each entry under the name the CALLER uses.
    ///
    /// This wrapper is meant to be transparent: `actual_path` appends
    /// `encrypted_extension` to every `read`, `write`, `write_if_absent`,
    /// `delete` and `exists`, so the on-disk key for `foo.parquet` is
    /// `foo.parquet.enc`. A `list` that reported the on-disk name would hand
    /// callers a name they cannot read back through this same wrapper, and —
    /// worse — would break every caller that filters a listing BY EXTENSION.
    ///
    /// That is not hypothetical. `FilesystemFactory` wraps every filesystem it
    /// produces when encryption is configured, and the affected caller is the
    /// **WAL manifest service**, which filters `manifest_*.jsonl`: under
    /// encryption every manifest is `manifest_X_Y.jsonl.enc`, the filter rejects
    /// all of them, and recovery logs "No existing manifest segments found,
    /// starting fresh" — the flushed-segment registry is silently discarded, and
    /// `cleanup_old_manifest_segments` never reclaims either (TD-ENCFS-1).
    ///
    /// Two more filter by extension through this wrapper, both narrower:
    /// RAPTOR's `SCAN_DISK` (`.raptor`/`.data`), gated behind
    /// `experimental-engines`; and the SST engine's `contains_vector`
    /// (`.sst`), which is production-wired but whose bloom check is a stub
    /// returning `true` unconditionally — so there the fix makes the encrypted
    /// answer match the unencrypted one rather than changing a correct answer.
    ///
    /// `io_trace_warehouse` is NOT affected, despite filtering `.jsonl.zst` and
    /// `.parquet` after a listing: it lists through `object_store` directly
    /// (`ObjectPath`, `meta.location`), so this wrapper is never in its path.
    /// Recorded explicitly because the extension-filter PATTERN matches there
    /// while the mechanism does not.
    ///
    /// Stripping is confined to FILES. `create_dir`/`create_dir_all` pass the
    /// path through unmangled, so a directory name never carries the extension
    /// — a directory legitimately ending in it is not ours to rewrite.
    async fn list(&self, path: &str) -> FsResult<Vec<DirEntry>> {
        let mut entries = self.underlying.list(path).await?;
        let Some(ext) = self.encrypted_extension.as_deref() else {
            // `without_extension()`: on-disk and logical names already agree.
            return Ok(entries);
        };
        for entry in &mut entries {
            if entry.metadata.is_directory {
                continue;
            }
            // Each field independently: a backend may populate any subset, and
            // `name`/`url`/`metadata.path` are all names a caller may use.
            if let Some(base) = entry.name.strip_suffix(ext) {
                entry.name = base.to_string();
            }
            if let Some(base) = entry.url.strip_suffix(ext) {
                entry.url = base.to_string();
            }
            if let Some(base) = entry.metadata.path.strip_suffix(ext) {
                entry.metadata.path = base.to_string();
            }
        }
        Ok(entries)
    }

    async fn create_dir(&self, path: &str) -> FsResult<()> {
        self.underlying.create_dir(path).await
    }

    async fn create_dir_all(&self, path: &str) -> FsResult<()> {
        self.underlying.create_dir_all(path).await
    }

    async fn copy(&self, from: &str, to: &str) -> FsResult<()> {
        let data = self.read(from).await?;
        self.write(
            to,
            &data,
            Some(FileOptions {
                create_dirs: true,
                overwrite: true,
                ..Default::default()
            }),
        )
        .await
    }

    async fn move_file(&self, from: &str, to: &str) -> FsResult<()> {
        self.copy(from, to).await?;
        self.delete(from).await
    }

    fn filesystem_type(&self) -> &'static str {
        "encrypted"
    }

    async fn write_atomic(
        &self,
        path: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        let actual_path = self.actual_path(path);
        let encrypted = self
            .encryption
            .encrypt_file(path, data)
            .map_err(|e| Self::crypto_error("encryption", path, e))?;
        self.underlying
            .write_atomic(&actual_path, &encrypted, options)
            .await
    }

    async fn sync(&self) -> FsResult<()> {
        self.underlying.sync().await
    }

    async fn open_file(&self, path: &str, _create: bool) -> FsResult<Box<dyn FilesystemFile>> {
        Err(FilesystemError::InvalidOperation(format!(
            "Streaming open_file is not wired for encrypted filesystem: {}",
            path
        )))
    }
}

#[cfg(test)]
mod list_transparency_tests {
    use super::*;
    use crate::KeyManager;
    use std::sync::Mutex;

    /// A filesystem whose `list` returns a fixed set of entries and whose
    /// `exists` records the path it was probed with.
    ///
    /// The recorded probe is what makes transparency testable without the
    /// crypto path: a listing is transparent only if the name it reports can be
    /// handed straight back to this same wrapper, which must then reach the
    /// ON-DISK key.
    #[derive(Debug, Default)]
    struct ListingFs {
        entries: Vec<DirEntry>,
        probed: Mutex<Vec<String>>,
    }

    impl ListingFs {
        fn entry(url: &str, is_directory: bool) -> DirEntry {
            DirEntry {
                name: url.rsplit('/').next().unwrap_or(url).to_string(),
                url: url.to_string(),
                metadata: FileMetadata {
                    path: url.to_string(),
                    is_directory,
                    ..Default::default()
                },
            }
        }
        fn with(entries: Vec<DirEntry>) -> Arc<Self> {
            Arc::new(Self {
                entries,
                probed: Mutex::new(Vec::new()),
            })
        }
        fn probes(&self) -> Vec<String> {
            self.probed.lock().expect("probe log").clone()
        }
    }

    #[async_trait]
    impl FileSystem for ListingFs {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        async fn list(&self, _path: &str) -> FsResult<Vec<DirEntry>> {
            Ok(self.entries.clone())
        }
        async fn exists(&self, path: &str) -> FsResult<bool> {
            self.probed
                .lock()
                .expect("probe log")
                .push(path.to_string());
            Ok(self.entries.iter().any(|e| e.url == path))
        }
        async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
            Err(FilesystemError::NotFound(path.to_string()))
        }
        async fn write(&self, _p: &str, _d: &[u8], _o: Option<FileOptions>) -> FsResult<()> {
            Ok(())
        }
        async fn append(&self, _p: &str, _d: &[u8]) -> FsResult<()> {
            Ok(())
        }
        async fn delete(&self, _p: &str) -> FsResult<()> {
            Ok(())
        }
        async fn metadata(&self, path: &str) -> FsResult<FileMetadata> {
            Err(FilesystemError::NotFound(path.to_string()))
        }
        async fn create_dir(&self, _p: &str) -> FsResult<()> {
            Ok(())
        }
        async fn create_dir_all(&self, _p: &str) -> FsResult<()> {
            Ok(())
        }
        async fn copy(&self, _f: &str, _t: &str) -> FsResult<()> {
            Ok(())
        }
        async fn move_file(&self, _f: &str, _t: &str) -> FsResult<()> {
            Ok(())
        }
        fn filesystem_type(&self) -> &'static str {
            "listing-test"
        }
        async fn sync(&self) -> FsResult<()> {
            Ok(())
        }
        async fn open_file(&self, path: &str, _c: bool) -> FsResult<Box<dyn FilesystemFile>> {
            Err(FilesystemError::NotFound(path.to_string()))
        }
    }

    fn wrapper(underlying: Arc<dyn FileSystem>) -> EncryptedFilesystem {
        // Established pattern in this crate (see `file_encryption.rs` tests).
        unsafe {
            std::env::set_var(
                "TEST_ENCFS_LIST_MASTER_KEY",
                "test-master-key-32-bytes-long-here!!",
            );
        }
        let key_manager =
            Arc::new(KeyManager::from_env("TEST_ENCFS_LIST_MASTER_KEY").expect("test master key"));
        EncryptedFilesystem::new(
            underlying,
            Arc::new(KeyVersionManager::new(key_manager)),
            true,
        )
    }

    /// A listing must report the name the caller uses, and that name must reach
    /// the on-disk key when handed back to this wrapper.
    #[tokio::test]
    async fn list_reports_logical_names_that_round_trip() {
        let fs = ListingFs::with(vec![ListingFs::entry("s3://b/w/foo.parquet.enc", false)]);
        let wrapped = wrapper(fs.clone());

        let entries = wrapped.list("s3://b/w/").await.expect("list");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "foo.parquet", "name must be un-mangled");
        assert_eq!(entries[0].url, "s3://b/w/foo.parquet");
        assert_eq!(entries[0].metadata.path, "s3://b/w/foo.parquet");

        // The reported name must be usable through the same wrapper.
        assert!(
            wrapped.exists(&entries[0].url).await.expect("exists"),
            "the listed name must resolve back to the on-disk object"
        );
        assert_eq!(
            fs.probes(),
            vec!["s3://b/w/foo.parquet.enc".to_string()],
            "the wrapper must have probed the ON-DISK key"
        );
    }

    /// The defect this fixes: a caller that filters a listing by extension.
    ///
    /// The WAL manifest service filters `entry.name.ends_with(".jsonl")`. With
    /// the mangled name it matches nothing and manifest discovery returns an
    /// EMPTY set under encryption (TD-ENCFS-1).
    #[tokio::test]
    async fn an_extension_filter_matches_encrypted_files() {
        let fs = ListingFs::with(vec![
            ListingFs::entry("file:///wal/manifest_1_2.jsonl.enc", false),
            ListingFs::entry("file:///wal/manifest_3_4.jsonl.enc", false),
            ListingFs::entry("file:///wal/unrelated.bin.enc", false),
        ]);
        let entries = wrapper(fs).list("file:///wal/").await.expect("list");

        let manifests: Vec<&str> = entries
            .iter()
            .filter(|e| e.name.contains("manifest_") && e.name.ends_with(".jsonl"))
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(
            manifests,
            vec!["manifest_1_2.jsonl", "manifest_3_4.jsonl"],
            "an extension filter must find encrypted manifests"
        );
    }

    /// A DIRECTORY is never mangled on write, so its name is not ours to strip.
    #[tokio::test]
    async fn list_does_not_strip_a_directory_name() {
        let fs = ListingFs::with(vec![ListingFs::entry("file:///data/bits.enc", true)]);
        let entries = wrapper(fs).list("file:///data/").await.expect("list");
        assert_eq!(
            entries[0].name, "bits.enc",
            "`create_dir` passes the path through, so a directory keeps its name"
        );
    }

    /// An unencrypted file in a mixed directory passes through untouched.
    #[tokio::test]
    async fn list_leaves_a_plain_file_alone() {
        let fs = ListingFs::with(vec![ListingFs::entry("file:///data/plain.parquet", false)]);
        let entries = wrapper(fs).list("file:///data/").await.expect("list");
        assert_eq!(entries[0].name, "plain.parquet");
    }

    /// With `without_extension()` the on-disk and logical names already agree,
    /// so nothing may be stripped.
    #[tokio::test]
    async fn without_extension_leaves_the_listing_untouched() {
        let fs = ListingFs::with(vec![ListingFs::entry("file:///data/foo.enc", false)]);
        let entries = wrapper(fs)
            .without_extension()
            .list("file:///data/")
            .await
            .expect("list");
        assert_eq!(
            entries[0].name, "foo.enc",
            "no extension configured means no name rewriting"
        );
    }
}
