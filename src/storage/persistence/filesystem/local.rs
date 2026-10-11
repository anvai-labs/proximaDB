/*
 * Copyright 2025 Vijaykumar Singh
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

//! Local filesystem implementation supporting Windows, Linux, and other OS platforms
//!
//! IMPORTANT: This implementation uses sync_all() for writes when sync_enabled is true
//! to ensure data durability and prevent file truncation issues that can occur when
//! files are moved/renamed before data is fully flushed to disk.

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, trace, warn};

use super::{
    DirEntry, FileMetadata, FileOptions, FileSystem, FilesystemError, FilesystemFile, FsResult,
};

/// Local filesystem configuration
#[derive(Debug, Clone)]
pub struct LocalConfig {
    /// Root directory for relative paths
    pub root_dir: Option<PathBuf>,

    /// Enable symbolic link resolution
    pub follow_symlinks: bool,

    /// Default file permissions (Unix-style octal)
    pub default_permissions: Option<u32>,

    /// Enable filesystem-level sync operations
    pub sync_enabled: bool,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            root_dir: None,
            follow_symlinks: true,
            default_permissions: None,
            sync_enabled: true,
        }
    }
}

/// Local filesystem implementation with mmap caching for faster reads
pub struct LocalFileSystem {
    config: LocalConfig,
    /// LRU cache for memory-mapped files (thread-safe)
    /// Files above MIN_MMAP_SIZE are cached for faster subsequent reads
    mmap_cache: std::sync::Arc<
        parking_lot::RwLock<
            proximadb_runtime_common::cache::LruCache<PathBuf, std::sync::Arc<memmap2::Mmap>>,
        >,
    >,
    /// Optional file encryption layer for transparent encryption at rest
    encryption_layer: Option<std::sync::Arc<crate::storage::encryption::FileEncryptionLayer>>,
}

impl std::fmt::Debug for LocalFileSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalFileSystem")
            .field("config", &self.config)
            .field("mmap_cache_size", &self.mmap_cache.read().len())
            .finish()
    }
}

/// Minimum file size to use mmap caching (512KB)
/// Files smaller than this are read directly (mmap overhead not worth it)
const MIN_MMAP_SIZE: u64 = 512 * 1024;

/// Maximum number of files to keep in mmap cache
const MMAP_CACHE_SIZE: usize = 64;

impl LocalFileSystem {
    /// Extract path from URL for stateless operations (uses centralized utility)
    fn resolve_path(&self, url: &str) -> FsResult<String> {
        use crate::storage::persistence::filesystem::FilesystemFactory;

        let path = FilesystemFactory::resolve_path(url)?;
        // For bare relative paths (no scheme), anchor to root_dir so each
        // LocalFileSystem instance with its own root_dir stays isolated.
        // Paths that came in with a scheme (file://, etc.) keep their
        // extracted form so callers can reason about them directly.
        let path_buf = PathBuf::from(&path);
        if !path_buf.is_absolute()
            && !url.contains("://")
            && let Some(ref root) = self.config.root_dir
        {
            return Ok(root.join(&path_buf).to_string_lossy().into_owned());
        }
        Ok(path)
    }

    /// Create new local filesystem instance with encryption support
    pub async fn new_with_encryption(
        config: LocalConfig,
        encryption_layer: Option<std::sync::Arc<crate::storage::encryption::FileEncryptionLayer>>,
    ) -> FsResult<Self> {
        // Capture the process start directory information
        match std::env::current_dir() {
            Ok(_cwd) => {
                // Try to get parent to see if it's accessible
                if let Some(_parent) = _cwd.parent() {}
            }
            Err(_e) => {
                // Try alternative methods to understand the environment
                if let Ok(exe_path) = std::env::current_exe()
                    && let Some(_exe_dir) = exe_path.parent()
                {}

                // Check environment variables
                if let Ok(_pwd) = std::env::var("PWD") {}

                // Check if we're in a test environment
                if let Ok(_cargo_manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {}
            }
        }

        // Create and validate root directory if specified
        if let Some(ref root_dir) = config.root_dir {
            if !root_dir.exists() {
                // Create the root directory recursively if it doesn't exist
                fs::create_dir_all(root_dir).await.map_err(|e| {
                    FilesystemError::Config(format!(
                        "Failed to create root directory {}: {}",
                        root_dir.display(),
                        e
                    ))
                })?;
                tracing::info!("📁 Created root directory: {}", root_dir.display());
            }
            if !root_dir.is_dir() {
                return Err(FilesystemError::Config(format!(
                    "Root path is not a directory: {}",
                    root_dir.display()
                )));
            }
        }

        Ok(Self {
            config,
            mmap_cache: std::sync::Arc::new(parking_lot::RwLock::new(
                proximadb_runtime_common::cache::LruCache::new(MMAP_CACHE_SIZE),
            )),
            encryption_layer,
        })
    }

    /// Get the encryption layer if enabled
    pub fn encryption_layer(
        &self,
    ) -> Option<&std::sync::Arc<crate::storage::encryption::FileEncryptionLayer>> {
        self.encryption_layer.as_ref()
    }

    /// Resolve a path string to an absolute PathBuf, handling cases where current_dir() fails
    fn resolve_to_absolute_path(&self, path_str: &str) -> PathBuf {
        let mut resolved_path = PathBuf::from(path_str);

        // If path is already absolute, return it
        if resolved_path.is_absolute() {
            return resolved_path;
        }

        // Try to get current directory, otherwise use fallbacks
        match std::env::current_dir() {
            Ok(cwd) => {
                resolved_path = cwd.join(path_str);
            }
            Err(_e) => {
                // Use PWD or CARGO_MANIFEST_DIR as fallback
                let fallback_base = if let Ok(pwd) = std::env::var("PWD") {
                    pwd
                } else if let Ok(cargo_dir) = std::env::var("CARGO_MANIFEST_DIR") {
                    cargo_dir
                } else {
                    "/tmp".to_string()
                };

                // Properly resolve the relative path - order matters!
                if path_str.starts_with("../") {
                    // Handle parent directory references with path
                    let base = PathBuf::from(&fallback_base);
                    let mut current = base.clone();
                    let mut remaining: &str = path_str;

                    // Handle multiple ../ sequences
                    while remaining.starts_with("../") {
                        if let Some(parent) = current.parent() {
                            current = parent.to_path_buf();
                            remaining = match remaining.strip_prefix("../") {
                                Some(s) => s,
                                None => remaining,
                            };
                        } else {
                            // Can't go up further, just use what we have
                            break;
                        }
                    }
                    resolved_path = current.join(remaining);
                } else if path_str == ".." {
                    // Just parent directory
                    let base = PathBuf::from(&fallback_base);
                    if let Some(parent) = base.parent() {
                        resolved_path = parent.to_path_buf();
                    } else {
                        resolved_path = base;
                    }
                } else if path_str.starts_with("./") {
                    // Replace ./ with the base directory
                    let clean_path = match path_str.strip_prefix("./") {
                        Some(p) => p,
                        None => path_str,
                    };
                    resolved_path = PathBuf::from(fallback_base).join(clean_path);
                } else if path_str == "." {
                    // Just current directory
                    resolved_path = PathBuf::from(fallback_base);
                } else {
                    // Regular relative path
                    resolved_path = PathBuf::from(fallback_base).join(path_str);
                }
            }
        }

        resolved_path
    }

    /// Get or create mmap for a file (cached for efficiency)
    ///
    /// Returns None if:
    /// - File is smaller than MIN_MMAP_SIZE
    /// - File doesn't exist or can't be opened
    /// - Mmap creation fails
    fn get_or_create_mmap(&self, path: &PathBuf) -> Option<std::sync::Arc<memmap2::Mmap>> {
        use memmap2::MmapOptions;
        use std::fs::File;

        // Check cache first (read lock)
        {
            let cache = self.mmap_cache.read();
            if let Some(mmap) = cache.peek(path) {
                return Some(std::sync::Arc::clone(mmap));
            }
        }

        // Not in cache, try to create mmap (need to check file size first)
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return None,
        };

        let metadata = match file.metadata() {
            Ok(m) => m,
            Err(_) => return None,
        };

        // Only mmap files above the threshold
        if metadata.len() < MIN_MMAP_SIZE {
            return None;
        }

        // Create the mmap
        let mmap = match unsafe { MmapOptions::new().map(&file) } {
            Ok(m) => std::sync::Arc::new(m),
            Err(e) => {
                tracing::debug!("Failed to mmap {}: {}", path.display(), e);
                return None;
            }
        };

        // Cache it (write lock)
        {
            let mut cache = self.mmap_cache.write();
            cache.put(path.clone(), std::sync::Arc::clone(&mmap));
        }

        tracing::debug!(
            "Created mmap for {} ({} bytes)",
            path.display(),
            metadata.len()
        );

        Some(mmap)
    }

    /// Invalidate mmap cache entry (call after file modification)
    pub fn invalidate_mmap_cache(&self, path: &PathBuf) {
        let mut cache = self.mmap_cache.write();
        cache.pop(path);
    }

    /// Create new local filesystem instance
    pub async fn new(config: LocalConfig) -> FsResult<Self> {
        // Capture the process start directory information
        match std::env::current_dir() {
            Ok(_cwd) => {
                // Try to get parent to see if it's accessible
                if let Some(_parent) = _cwd.parent() {}
            }
            Err(_e) => {
                // Try alternative methods to understand the environment
                if let Ok(exe_path) = std::env::current_exe()
                    && let Some(_exe_dir) = exe_path.parent()
                {}

                // Check environment variables
                if let Ok(_pwd) = std::env::var("PWD") {}

                // Check if we're in a test environment
                if let Ok(_cargo_manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {}
            }
        }

        // Create and validate root directory if specified
        if let Some(ref root_dir) = config.root_dir {
            if !root_dir.exists() {
                // Create the root directory recursively if it doesn't exist
                fs::create_dir_all(root_dir).await.map_err(|e| {
                    FilesystemError::Config(format!(
                        "Failed to create root directory {}: {}",
                        root_dir.display(),
                        e
                    ))
                })?;
                tracing::info!("📁 Created root directory: {}", root_dir.display());
            }
            if !root_dir.is_dir() {
                return Err(FilesystemError::Config(format!(
                    "Root path is not a directory: {}",
                    root_dir.display()
                )));
            }
        }

        Ok(Self {
            config,
            mmap_cache: std::sync::Arc::new(parking_lot::RwLock::new(
                proximadb_runtime_common::cache::LruCache::new(MMAP_CACHE_SIZE),
            )),
            encryption_layer: None,
        })
    }

    /// Convert std::fs::Metadata to FileMetadata
    fn convert_metadata(&self, path: &Path, metadata: &std::fs::Metadata) -> FileMetadata {
        use std::time::UNIX_EPOCH;

        let to_datetime = |time: std::io::Result<std::time::SystemTime>| {
            time.ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .and_then(|d| {
                    chrono::DateTime::from_timestamp(d.as_secs() as i64, d.subsec_nanos())
                })
        };

        FileMetadata {
            path: path.display().to_string(),
            size: metadata.len(),
            created: to_datetime(metadata.created()),
            modified: to_datetime(metadata.modified()),
            is_directory: metadata.is_dir(),
            permissions: self.get_permissions_string(metadata),
            etag: None,          // Not applicable for local filesystem
            storage_class: None, // Not applicable for local filesystem
        }
    }

    /// Get permissions string from metadata
    fn get_permissions_string(&self, metadata: &std::fs::Metadata) -> Option<String> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = metadata.permissions().mode();
            Some(format!("{:o}", mode & 0o777))
        }

        #[cfg(not(unix))]
        {
            if metadata.permissions().readonly() {
                Some("readonly".to_string())
            } else {
                Some("readwrite".to_string())
            }
        }
    }
}

#[async_trait]
impl FileSystem for LocalFileSystem {
    async fn compare_exchange(
        &self,
        path: &str,
        expected: Option<&[u8]>,
        replacement: &[u8],
        guards: &[(&str, Option<&[u8]>)],
    ) -> FsResult<bool> {
        if self.encryption_layer.is_some() {
            return Err(FilesystemError::InvalidOperation(
                "conditional coordination-file publication is not qualified for encrypted storage"
                    .into(),
            ));
        }
        let resolved = PathBuf::from(self.resolve_path(path)?);
        let expected = expected.map(<[u8]>::to_vec);
        let replacement = replacement.to_vec();
        let guards = guards
            .iter()
            .map(|(path, expected)| {
                Ok((
                    PathBuf::from(self.resolve_path(path)?),
                    expected.map(<[u8]>::to_vec),
                ))
            })
            .collect::<FsResult<Vec<_>>>()?;
        let cache = self.mmap_cache.clone();
        tokio::task::spawn_blocking(move || {
            let guards: Vec<_> = guards
                .iter()
                .map(|(path, expected)| (path.as_path(), expected.as_deref()))
                .collect();
            let result =
                proximadb_runtime_common::file_lock::FileLockManager::compare_exchange_file(
                    &resolved,
                    expected.as_deref(),
                    &replacement,
                    &guards,
                );
            // Invalidate even on an indeterminate error. This remains in the same
            // blocking task if the caller is cancelled after publication.
            cache.write().pop(&resolved);
            result.map_err(FilesystemError::Io)
        })
        .await
        .map_err(|e| FilesystemError::Io(std::io::Error::other(e)))?
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
        // Extract path from URL
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(path_str);

        let result = match fs::read(&resolved_path).await {
            Ok(data) => {
                // Decrypt data if encryption is enabled
                if let Some(ref encryption) = self.encryption_layer {
                    match encryption.decrypt_file(path, &data) {
                        Ok(decrypted) => Ok(decrypted),
                        Err(e) => {
                            // If decryption fails, the file might not be encrypted
                            // Check for encrypted file magic number
                            if data.len() >= 4 && &data[0..4] == b"PEDE" {
                                Err(FilesystemError::Io(std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    format!("Decryption failed for {}: {}", path, e),
                                )))
                            } else {
                                // File is not encrypted, return as-is
                                Ok(data)
                            }
                        }
                    }
                } else {
                    Ok(data)
                }
            }
            Err(e) => match e.kind() {
                std::io::ErrorKind::NotFound => Err(FilesystemError::NotFound(format!(
                    "File not found: {}",
                    path
                ))),
                std::io::ErrorKind::PermissionDenied => Err(FilesystemError::PermissionDenied(
                    resolved_path.display().to_string(),
                )),
                _ => Err(FilesystemError::Io(e)),
            },
        };
        if let Ok(ref bytes) = result {
            super::record_physical_full_read(bytes.len() as u64);
        }
        result
    }

    async fn get_mmap(&self, path: &str) -> FsResult<Option<memmap2::Mmap>> {
        use memmap2::MmapOptions;
        use std::fs::File;

        // Extract path from URL
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(path_str);

        // Open file for memory mapping (synchronous operation)
        let file = File::open(&resolved_path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                FilesystemError::NotFound(format!("File not found for mmap: {}", path))
            }
            std::io::ErrorKind::PermissionDenied => {
                FilesystemError::PermissionDenied(resolved_path.display().to_string())
            }
            _ => FilesystemError::Io(e),
        })?;

        // Create memory map
        // Safety: We're creating a read-only mmap which is safe for concurrent access
        let mmap = unsafe { MmapOptions::new().map(&file).map_err(FilesystemError::Io)? };

        tracing::debug!(
            "Created memory map for file: {} (size: {} bytes)",
            path,
            mmap.len()
        );
        Ok(Some(mmap))
    }

    fn supports_mmap(&self) -> bool {
        true // Local filesystem supports memory mapping
    }

    async fn write(&self, path: &str, data: &[u8], options: Option<FileOptions>) -> FsResult<()> {
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(path_str);
        let options = options.clone();

        // Create parent directories if requested
        if options.as_ref().is_some_and(|o| o.create_dirs)
            && let Some(parent) = resolved_path.parent()
        {
            fs::create_dir_all(parent)
                .await
                .map_err(FilesystemError::Io)?;
        }

        // Check if file exists and handle overwrite option
        if !options.as_ref().is_none_or(|o| o.overwrite) && resolved_path.exists() {
            return Err(FilesystemError::AlreadyExists(
                resolved_path.display().to_string(),
            ));
        }

        // Encrypt data if encryption is enabled
        let data_to_write = if let Some(ref encryption) = self.encryption_layer {
            match encryption.encrypt_file(path, data) {
                Ok(encrypted) => encrypted,
                Err(e) => {
                    return Err(FilesystemError::Io(std::io::Error::other(format!(
                        "Encryption failed for {}: {}",
                        path, e
                    ))));
                }
            }
        } else {
            data.to_vec()
        };

        // Write file with proper sync to ensure data is flushed to disk
        // This is critical for atomic writes to prevent truncation
        if self.config.sync_enabled {
            // Open file for writing with sync
            let mut file = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&resolved_path)
                .await
                .map_err(FilesystemError::Io)?;

            // Write all data
            file.write_all(&data_to_write)
                .await
                .map_err(FilesystemError::Io)?;

            // Sync data to disk to prevent truncation issues
            file.sync_all().await.map_err(FilesystemError::Io)?;
        } else {
            // Fast write without sync (for development/testing only)
            fs::write(&resolved_path, data_to_write)
                .await
                .map_err(FilesystemError::Io)?;
        }

        // Set permissions if specified
        #[cfg(unix)]
        if let Some(permissions) = self.config.default_permissions {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(permissions);
            fs::set_permissions(&resolved_path, perms)
                .await
                .map_err(FilesystemError::Io)?;
        }

        Ok(())
    }

    async fn write_if_absent(
        &self,
        path: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        let resolved_path = PathBuf::from(self.resolve_path(path)?);
        if options.as_ref().is_some_and(|o| o.create_dirs)
            && let Some(parent) = resolved_path.parent()
        {
            fs::create_dir_all(parent)
                .await
                .map_err(FilesystemError::Io)?;
        }

        let data_to_write = if let Some(ref encryption) = self.encryption_layer {
            encryption.encrypt_file(path, data).map_err(|e| {
                FilesystemError::Io(std::io::Error::other(format!(
                    "Encryption failed for {path}: {e}"
                )))
            })?
        } else {
            data.to_vec()
        };

        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&resolved_path)
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    FilesystemError::AlreadyExists(resolved_path.display().to_string())
                } else {
                    FilesystemError::Io(e)
                }
            })?;
        file.write_all(&data_to_write)
            .await
            .map_err(FilesystemError::Io)?;
        if self.config.sync_enabled {
            file.sync_all().await.map_err(FilesystemError::Io)?;
        }
        Ok(())
    }

    async fn append(&self, path: &str, data: &[u8]) -> FsResult<()> {
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(path_str);

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&resolved_path)
            .await
            .map_err(FilesystemError::Io)?;

        file.write_all(data).await.map_err(FilesystemError::Io)?;

        if self.config.sync_enabled {
            file.sync_all().await.map_err(FilesystemError::Io)?;
        }

        Ok(())
    }

    fn supports_append(&self) -> bool {
        true
    }

    fn supports_conditional_replace(&self) -> bool {
        cfg!(unix) && self.encryption_layer.is_none()
    }

    async fn delete(&self, path: &str) -> FsResult<()> {
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(path_str);

        if !resolved_path.exists() {
            return Err(FilesystemError::NotFound(
                resolved_path.display().to_string(),
            ));
        }

        if resolved_path.is_dir() {
            fs::remove_dir_all(&resolved_path)
                .await
                .map_err(FilesystemError::Io)
        } else {
            fs::remove_file(&resolved_path)
                .await
                .map_err(FilesystemError::Io)
        }
    }

    async fn exists(&self, path: &str) -> FsResult<bool> {
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(path_str);
        Ok(resolved_path.exists())
    }

    async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        // ADR-030 / TD-158: this is the leaf physical-GET boundary. Feed the per-query
        // I/O accumulator once per successful ranged read — `UnifiedCachingFilesystem`
        // serves cache hits without delegating here, so `range_gets` reflects TRUE
        // physical GETs (the `avg_get_size` the route cost model prices == the bytes the
        // KRU read-meter charges, by construction). The counters are task-local: they
        // no-op outside an active `io_trace` scope, and the core counters are always-on
        // (not behind `io-trace`) so the billing observer reads real values at drain.
        let result: FsResult<Vec<u8>> = async {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};

        // Extract path from URL
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(&path_str);

        // DEBUG: Log path resolution for RAPTOR files
        if path.contains(".raptor") {
            tracing::debug!(
                "🔍 LocalFS read_range: input='{}', resolved='{}', offset={}, length={}",
                path,
                resolved_path.display(),
                offset,
                length
            );
        }

        // OPTIMIZATION: Try mmap-based read first for large files
        // Mmap is faster because:
        // 1. Data is already in page cache (no syscall for subsequent reads)
        // 2. No file handle overhead
        // 3. Better read-ahead behavior from OS
        if let Some(mmap) = self.get_or_create_mmap(&resolved_path) {
            let start = offset as usize;
            let end = (offset + length) as usize;

            // Bounds check
            if start >= mmap.len() {
                return Ok(vec![]);
            }
            let end = end.min(mmap.len());

            // Copy the slice (still faster than seek+read due to page cache)
            return Ok(mmap[start..end].to_vec());
        }

        // Fall back to traditional read for small files or when mmap fails
        let mut file = match tokio::fs::File::open(&resolved_path).await {
            Ok(f) => f,
            Err(e) => match e.kind() {
                std::io::ErrorKind::NotFound => {
                    return Err(FilesystemError::NotFound(format!(
                        "File not found: {}",
                        path
                    )));
                }
                std::io::ErrorKind::PermissionDenied => {
                    return Err(FilesystemError::PermissionDenied(
                        resolved_path.display().to_string(),
                    ));
                }
                _ => return Err(FilesystemError::Io(e)),
            },
        };

        // Get file size to validate the range
        let metadata = file.metadata().await.map_err(FilesystemError::Io)?;
        let file_size = metadata.len();

        // DEBUG: Log file size for RAPTOR files
        if path.contains(".raptor") {
            tracing::debug!(
                "🔍 LocalFS read_range: file_size={}, offset={}, offset>=file_size={}",
                file_size,
                offset,
                offset >= file_size
            );
        }

        // Check if offset is beyond file size
        if offset >= file_size {
            return Ok(vec![]);
        }

        // Seek to the offset
        file.seek(std::io::SeekFrom::Start(offset))
            .await
            .map_err(FilesystemError::Io)?;

        // Calculate actual bytes to read (handle case where offset + length > file_size)
        let bytes_to_read = std::cmp::min(length, file_size - offset) as usize;

        // Debug log for SSTable bloom filter issue
        if path.contains(".sstable") && offset < 1000 {
            tracing::debug!(
                "DEBUG LocalFS read_range: path={}, offset={}, length={}, file_size={}, bytes_to_read={}",
                path,
                offset,
                length,
                file_size,
                bytes_to_read
            );
        }

        // Extra detailed tracing for our SST debug

        // Read the exact amount of bytes
        let mut buffer = vec![0u8; bytes_to_read];
        match file.read_exact(&mut buffer).await {
            Ok(_) => {
                // Success
            }
            Err(e) => {
                // Get current position for debugging
                let current_pos = file.stream_position().await.map_err(FilesystemError::Io)?;
                tracing::error!(
                    "LocalFS read_exact failed: path={}, offset={}, bytes_to_read={}, current_pos={}, file_size={}, error={:?}",
                    path,
                    offset,
                    bytes_to_read,
                    current_pos,
                    file_size,
                    e
                );
                return Err(FilesystemError::Io(e));
            }
        }

        Ok(buffer)
        }
        .await;
        if let Ok(ref bytes) = result {
            super::record_physical_range_read(bytes.len() as u64);
            if std::env::var_os("PROXIMADB_TRACE_GETS").is_some() {
                eprintln!("[FS GET] {path} off={offset} len={length}");
            }
        }
        result
    }

    async fn metadata(&self, path: &str) -> FsResult<FileMetadata> {
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(path_str);

        let metadata = if self.config.follow_symlinks {
            fs::metadata(&resolved_path).await
        } else {
            fs::symlink_metadata(&resolved_path).await
        };

        match metadata {
            Ok(meta) => Ok(self.convert_metadata(&resolved_path, &meta)),
            Err(e) => match e.kind() {
                std::io::ErrorKind::NotFound => Err(FilesystemError::NotFound(
                    resolved_path.display().to_string(),
                )),
                _ => Err(FilesystemError::Io(e)),
            },
        }
    }

    async fn list(&self, path: &str) -> FsResult<Vec<DirEntry>> {
        // Extract path from URL
        let path_str = self.resolve_path(path)?;

        let resolved_path = PathBuf::from(path_str);

        let mut entries = Vec::new();
        let mut dir = fs::read_dir(&resolved_path)
            .await
            .map_err(FilesystemError::Io)?;

        while let Some(entry) = dir.next_entry().await.map_err(FilesystemError::Io)? {
            let entry_path = entry.path();
            let name = entry_path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| {
                    FilesystemError::InvalidPath(format!(
                        "Invalid filename encoding for path: {}",
                        entry_path.display()
                    ))
                })?
                .to_string();

            let metadata = entry.metadata().await.map_err(FilesystemError::Io)?;
            let file_metadata = self.convert_metadata(&entry_path, &metadata);

            // Create full URL for entry (stateless design)
            let entry_url = if path.starts_with("file://./") || path.starts_with("./") {
                // For relative URLs, preserve the relative nature
                if let Some(root_dir) = &self.config.root_dir {
                    let relative_from_root = entry_path.strip_prefix(root_dir).map_or_else(
                        |_| entry_path.display().to_string(),
                        |p| p.to_string_lossy().to_string(),
                    );
                    format!("file://./{}", relative_from_root)
                } else {
                    // No root_dir, but path is relative - preserve relative nature
                    format!("file://{}", entry_path.display())
                }
            } else {
                // Absolute path
                format!("file://{}", entry_path.display())
            };
            // Per-file logging removed - too verbose for production
            // Use trace only if debugging specific file issues
            trace!("LocalFileSystem entry: '{}' -> {:?}", entry_url, entry_path);

            entries.push(DirEntry {
                name,
                url: entry_url, // Use full URL instead of path
                metadata: file_metadata,
            });
        }

        // Sort entries by name for consistent ordering
        entries.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(entries)
    }

    async fn create_dir(&self, path: &str) -> FsResult<()> {
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(path_str);

        fs::create_dir(&resolved_path)
            .await
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::AlreadyExists => {
                    FilesystemError::AlreadyExists(resolved_path.display().to_string())
                }
                _ => FilesystemError::Io(e),
            })
    }

    async fn create_dir_all(&self, path: &str) -> FsResult<()> {
        // First extract path from URL
        let path_str = self.resolve_path(path)?;

        // Then resolve to absolute path
        let resolved_path = self.resolve_to_absolute_path(&path_str);

        // Now create the directory with the resolved absolute path
        match std::fs::create_dir_all(&resolved_path) {
            Ok(()) => Ok(()),
            Err(e) => match e.kind() {
                std::io::ErrorKind::AlreadyExists => Ok(()),
                _ => Err(FilesystemError::Io(e)),
            },
        }
    }

    async fn copy(&self, from: &str, to: &str) -> FsResult<()> {
        let from_path_str = self.resolve_path(from)?;
        let from_path = PathBuf::from(from_path_str);
        let to_path_str = self.resolve_path(to)?;
        let to_path = PathBuf::from(to_path_str);

        // Create parent directory for destination if needed
        if let Some(parent) = to_path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(FilesystemError::Io)?;
        }

        fs::copy(&from_path, &to_path)
            .await
            .map_err(FilesystemError::Io)?;

        Ok(())
    }

    async fn move_file(&self, from: &str, to: &str) -> FsResult<()> {
        trace!("move_file called: from {} to {}", from, to);

        // Extract paths from URLs
        let from_path_str = self.resolve_path(from)?;
        let from_path = PathBuf::from(&from_path_str);
        trace!("Resolved from_path: {:?}", from_path);

        let to_path_str = self.resolve_path(to)?;
        let to_path = PathBuf::from(&to_path_str);
        trace!("Resolved to_path: {:?}", to_path);

        // Check if source file exists
        if !from_path.exists() {
            warn!("Source file does not exist: {:?}", from_path);
            return Err(FilesystemError::NotFound(from_path.display().to_string()));
        }

        // Check source file size
        if let Ok(metadata) = std::fs::metadata(&from_path) {
            trace!("Source file size: {} bytes", metadata.len());
        }

        // Create parent directory for destination if needed
        if let Some(parent) = to_path.parent() {
            trace!("Creating parent directory: {:?}", parent);
            fs::create_dir_all(parent).await.map_err(|e| {
                warn!("Error creating parent dir: {}", e);
                FilesystemError::Io(e)
            })?;
        }

        trace!("Attempting rename from {:?} to {:?}", from_path, to_path);
        let result = fs::rename(&from_path, &to_path).await.map_err(|e| {
            debug!("Rename failed, will try copy: {}", e);
            FilesystemError::Io(e)
        });

        if result.is_ok() {
            trace!("File moved successfully");

            // Verify destination exists
            if to_path.exists() {
                if let Ok(metadata) = std::fs::metadata(&to_path) {
                    trace!(
                        "Verified destination file exists with size: {} bytes",
                        metadata.len()
                    );
                }
            } else {
                warn!("Destination file doesn't exist after move!");
            }
        }

        result
    }

    fn filesystem_type(&self) -> &'static str {
        "local"
    }

    async fn sync(&self) -> FsResult<()> {
        // For local filesystem, sync is handled at the file level
        // This is a no-op for the filesystem as a whole
        Ok(())
    }

    async fn sync_file(&self, path: &str) -> FsResult<()> {
        // Extract path from URL
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(path_str);

        // Only sync if enabled in config
        if !self.config.sync_enabled {
            return Ok(());
        }

        // Open the file for sync
        let file = tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&resolved_path)
            .await
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => {
                    FilesystemError::NotFound(resolved_path.display().to_string())
                }
                _ => FilesystemError::Io(e),
            })?;

        // Call sync_all() which performs fsync on the file
        file.sync_all().await.map_err(FilesystemError::Io)?;

        tracing::debug!("✅ File synced to disk: {}", resolved_path.display());
        Ok(())
    }

    async fn sync_data(&self, path: &str) -> FsResult<()> {
        // fdatasync: flush the file's data (and only the metadata required to
        // read it back) without the full inode metadata flush of `sync_all`.
        // This is the genuine `DurabilityLevel::SyncData` fast-path on local
        // disks; remote backends fall back to a full `sync_file` via the trait
        // default.
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(path_str);

        if !self.config.sync_enabled {
            return Ok(());
        }

        let file = tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&resolved_path)
            .await
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => {
                    FilesystemError::NotFound(resolved_path.display().to_string())
                }
                _ => FilesystemError::Io(e),
            })?;

        // sync_data() maps to fdatasync(2) — data durability without the extra
        // metadata barrier of fsync(2)/sync_all().
        file.sync_data().await.map_err(FilesystemError::Io)?;

        tracing::debug!("✅ File data synced to disk: {}", resolved_path.display());
        Ok(())
    }

    async fn open_file(&self, path: &str, create: bool) -> FsResult<Box<dyn FilesystemFile>> {
        let path_str = self.resolve_path(path)?;
        let resolved_path = PathBuf::from(path_str);

        let file = if create {
            tokio::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&resolved_path)
                .await
        } else {
            tokio::fs::File::open(&resolved_path).await
        }
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                FilesystemError::NotFound(format!("File not found: {}", path))
            }
            std::io::ErrorKind::PermissionDenied => {
                FilesystemError::PermissionDenied(resolved_path.display().to_string())
            }
            _ => FilesystemError::Io(e),
        })?;

        Ok(Box::new(LocalFile {
            file,
            path: resolved_path,
        }))
    }
}

/// Local file handle for streaming operations
#[derive(Debug)]
struct LocalFile {
    file: tokio::fs::File,
    path: PathBuf,
}

#[async_trait]
impl FilesystemFile for LocalFile {
    async fn read(&mut self, buf: &mut [u8]) -> FsResult<usize> {
        use tokio::io::AsyncReadExt;
        self.file.read(buf).await.map_err(FilesystemError::Io)
    }

    async fn write(&mut self, buf: &[u8]) -> FsResult<usize> {
        use tokio::io::AsyncWriteExt;
        self.file.write(buf).await.map_err(FilesystemError::Io)
    }

    async fn flush(&mut self) -> FsResult<()> {
        use tokio::io::AsyncWriteExt;
        self.file.flush().await.map_err(FilesystemError::Io)
    }

    async fn seek(&mut self, pos: u64) -> FsResult<u64> {
        use tokio::io::{AsyncSeekExt, SeekFrom};
        self.file
            .seek(SeekFrom::Start(pos))
            .await
            .map_err(FilesystemError::Io)
    }

    async fn position(&self) -> FsResult<u64> {
        // Note: Getting current position requires a mutable reference in tokio
        // This is a limitation of the tokio API
        Ok(0) // Would need to track position separately or use seek(SeekFrom::Current(0))
    }

    async fn file_size(&self) -> FsResult<u64> {
        let metadata = tokio::fs::metadata(&self.path)
            .await
            .map_err(FilesystemError::Io)?;
        Ok(metadata.len())
    }

    async fn sync_all(&mut self) -> FsResult<()> {
        self.file.sync_all().await.map_err(FilesystemError::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tracing::{debug, error, info};

    /// Async mutex to serialize tests that change the current working directory.
    /// `tokio::sync::Mutex` so the guard may be held across `.await` without
    /// blocking the runtime (clears clippy::await_holding_lock); each test is a
    /// single task, so holding it for the test body serializes the suite safely.
    static CWD_MUTEX: Lazy<tokio::sync::Mutex<()>> = Lazy::new(|| tokio::sync::Mutex::new(()));

    #[tokio::test]
    async fn test_relative_path_url_handling() {
        // Serialize tests that change the current working directory
        let _cwd_guard = CWD_MUTEX.lock().await;

        // Save current directory to restore later
        let original_dir = std::env::current_dir().unwrap();

        // Create a temporary directory and change to it
        let temp_dir = TempDir::new().unwrap();
        std::env::set_current_dir(temp_dir.path()).unwrap();

        // Clean up any existing test structure first
        if std::path::Path::new("test_metadata_info").exists() {
            std::fs::remove_dir_all("test_metadata_info").ok();
        }

        // Create test structure using relative paths in current directory
        std::fs::create_dir_all("test_metadata/current").unwrap();
        std::fs::write("test_metadata/current/test.txt", b"test content").unwrap();

        // Create filesystem without root_dir - it should handle relative paths as-is
        let config = LocalConfig {
            root_dir: None,
            follow_symlinks: true,
            default_permissions: None,
            sync_enabled: false,
        };

        let fs = LocalFileSystem::new(config).await.unwrap();

        // Debug: Check current directory and file existence
        let current_dir = std::env::current_dir().unwrap();
        debug!("🔍 Current directory: {}", current_dir.display());
        debug!(
            "🔍 Checking if test_metadata/current exists: {}",
            std::path::Path::new("test_metadata/current").exists()
        );
        debug!(
            "🔍 Checking if test_metadata/current/test.txt exists: {}",
            std::path::Path::new("test_metadata/current/test.txt").exists()
        );

        // Verify the directory exists before testing
        assert!(
            std::path::Path::new("test_metadata/current").exists(),
            "test_metadata/current directory should exist"
        );
        assert!(
            std::path::Path::new("test_metadata/current/test.txt").exists(),
            "test.txt file should exist"
        );

        // Test listing with relative path - this should work with the relative path as provided
        let relative_url = "file://test_metadata/current";
        debug!("🔍 About to list: {}", relative_url);
        let entries = fs.list(relative_url).await.unwrap();

        // Restore original directory AFTER filesystem operations to avoid test pollution
        std::env::set_current_dir(original_dir).unwrap();

        // Verify entries
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "test.txt");

        // The key test: verify the URL doesn't have duplicated paths
        let entry_url = &entries[0].url;
        debug!("Entry URL: {}", entry_url);

        // The URL should be properly formatted without duplications
        assert!(entry_url.contains("test_metadata/current/test.txt"));
        assert!(!entry_url.contains("test_metadata/test_metadata_info"));
    }

    /// ADR-030 / TD-158: the leaf `read_range` feeds the per-query I/O accumulator.
    /// A ranged read inside an `io_trace` scope records one physical GET plus the exact
    /// bytes returned, so `range_gets > 0` and `avg_get_bytes()` (the read-granularity
    /// signal the route cost model prices and the KRU meter charges) is populated.
    /// The core counters are always-on, so this holds in the default build (no
    /// `io-trace` feature). Outside any scope the same call is a silent no-op.
    #[tokio::test]
    async fn read_range_feeds_io_trace_accumulator_td158() {
        use crate::observability::io_trace;

        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("seg.bin");
        let payload = vec![7u8; 4096];
        std::fs::write(&file_path, &payload).unwrap();

        let fs = LocalFileSystem::new(LocalConfig::default()).await.unwrap();
        let url = format!("file://{}", file_path.display());

        // Inside an active scope: both physical reads are accounted to the per-query trace.
        let snap = io_trace::scope(async {
            let first = fs.read_range(&url, 0, payload.len() as u64).await.unwrap();
            assert_eq!(first.len(), payload.len());
            // A second, smaller ranged read proves GETs accumulate (this is the bare
            // leaf FS — no caching wrapper, so every read is a physical GET).
            let second = fs.read_range(&url, 0, 1024).await.unwrap();
            assert_eq!(second.len(), 1024);
            io_trace::snapshot().expect("snapshot must exist inside an active scope")
        })
        .await;

        assert!(
            snap.range_gets >= 2,
            "each physical ranged read counts one GET (got {})",
            snap.range_gets
        );
        assert_eq!(
            snap.bytes_read,
            (payload.len() + 1024) as u64,
            "bytes_read must equal the exact bytes returned across both GETs"
        );
        assert!(
            snap.avg_get_bytes().is_some(),
            "avg_get_bytes (read-granularity signal) must be populated once GETs are recorded"
        );

        // Outside any scope the same call is a silent no-op — task-local, no global state.
        let outside = fs.read_range(&url, 0, 16).await.unwrap();
        assert_eq!(outside.len(), 16);
        assert!(
            io_trace::snapshot().is_none(),
            "no active scope ⇒ no trace recorded"
        );
    }

    #[tokio::test]
    async fn test_absolute_path_url_handling() {
        // Create a temporary directory
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        // Create test structure
        let test_dir = temp_path.join("test_metadata/current");
        std::fs::create_dir_all(&test_dir).unwrap();
        std::fs::write(test_dir.join("test.txt"), b"test content").unwrap();

        // Create filesystem without root_dir (absolute paths)
        let config = LocalConfig::default();
        let fs = LocalFileSystem::new(config).await.unwrap();

        // Test listing with absolute URL
        let absolute_url = format!("file://{}", test_dir.display());
        let entries = fs.list(&absolute_url).await.unwrap();

        // Verify entries
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "test.txt");

        // Verify the URL is properly formatted
        let entry_url = &entries[0].url;
        debug!("Entry URL: {}", entry_url);
        assert!(entry_url.starts_with("file://"));
        assert!(entry_url.ends_with("/test.txt"));
    }

    #[tokio::test]
    async fn test_resolve_path() {
        let config = LocalConfig::default();
        let fs = LocalFileSystem::new(config).await.unwrap();

        // Test various URL formats
        assert_eq!(fs.resolve_path("file:///tmp/test").unwrap(), "/tmp/test");
        assert_eq!(
            fs.resolve_path("file://./relative/path").unwrap(),
            "./relative/path"
        );
        assert_eq!(fs.resolve_path("/absolute/path").unwrap(), "/absolute/path");
        assert_eq!(fs.resolve_path("relative/path").unwrap(), "relative/path");
    }

    #[tokio::test]
    async fn test_no_path_duplication_in_operations() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        // Setup filesystem with root directory
        let config = LocalConfig {
            root_dir: Some(temp_path.to_path_buf()),
            follow_symlinks: true,
            default_permissions: None,
            sync_enabled: false,
        };

        let fs = LocalFileSystem::new(config).await.unwrap();

        // Use absolute paths based on temp dir to avoid CWD-dependent relative paths
        let base = format!("file://{}", temp_path.display());
        let test_url = format!("{}/metadata/current", base);
        let file_url = format!("{}/metadata/current/test.oplog", base);

        // Create a directory structure
        fs.create_dir_all(&test_url).await.unwrap();

        // Write a file
        let _ = fs.delete(&file_url).await;
        fs.write(&file_url, b"test data", None).await.unwrap();

        // List the directory
        let entries = fs.list(&test_url).await.unwrap();

        // Filter out hidden files (like .DS_Store on macOS) and staging directories
        let visible_entries: Vec<_> = entries
            .iter()
            .filter(|e| !e.name.starts_with('.') && !e.name.starts_with("__"))
            .collect();

        // Verify no path duplication
        assert_eq!(visible_entries.len(), 1);
        let entry_url = &visible_entries[0].url;

        // Count occurrences of "metadata" in the URL path segments
        let metadata_count = entry_url.matches("metadata").count();
        assert_eq!(
            metadata_count, 1,
            "Path 'metadata' should appear only once in URL: {}",
            entry_url
        );

        // Verify we can read the file using both the listed URL and original URL
        // First, normalize the entry URL to handle any path duplication issues
        let normalized_entry_url = entry_url.replace("././", "./");

        // Try the normalized entry URL first, fall back to original file_url if needed
        let content = match fs.read(&normalized_entry_url).await {
            Ok(data) => data,
            Err(_) => {
                // If normalized entry URL doesn't work, try the entry URL as-is
                match fs.read(entry_url).await {
                    Ok(data) => data,
                    Err(_) => {
                        // Final fallback: try the original file_url
                        fs.read(&file_url).await.unwrap_or_else(|e| {
                            panic!("Failed to read file with entry URL ({}), normalized entry URL ({}), and original URL ({}): {}",
                                   entry_url, normalized_entry_url, file_url, e);
                        })
                    }
                }
            }
        };
        assert_eq!(content, b"test data");
    }

    #[tokio::test]
    async fn test_filesystem_without_root_dir() {
        // Serialize tests that change the current working directory
        let _cwd_guard = CWD_MUTEX.lock().await;

        // Test that filesystem without root_dir handles URLs correctly
        let temp_dir = TempDir::new().unwrap();
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(temp_dir.path()).unwrap();

        // Create filesystem WITHOUT root_dir
        let config = LocalConfig::default();
        let fs = LocalFileSystem::new(config).await.unwrap();

        // Clean up any existing test directories first
        if fs
            .exists("file://./test_metadata_info")
            .await
            .unwrap_or(false)
        {
            // List and delete all files recursively
            if let Ok(entries) = fs.list("file://./test_metadata/current").await {
                for entry in entries {
                    let _ = fs.delete(&entry.url).await;
                }
            }
            let _ = fs.delete("file://./test_metadata/current").await;
            let _ = fs.delete("file://./test_metadata_info").await;
        }

        // Test with relative URL
        let relative_url = "file://./test_metadata/current";
        fs.create_dir_all(relative_url).await.unwrap();

        // Verify the directory was created at the correct location
        assert!(std::path::Path::new("./test_metadata/current").exists());

        // Write a file
        fs.write("file://./test_metadata/current/test.txt", b"test", None)
            .await
            .unwrap();

        // List directory
        let entries = fs.list(relative_url).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].url.contains("test.txt"));

        // Verify no path duplication in the URL
        assert!(!entries[0].url.contains("test_metadata/test_metadata_info"));

        // Restore original directory
        std::env::set_current_dir(original_dir).unwrap();
    }

    #[tokio::test]
    async fn test_resolve_path_consistency() {
        // Test that paths are extracted consistently without root_dir resolution
        let temp_dir = TempDir::new().unwrap();

        // Test with root_dir set (but it shouldn't affect path extraction)
        let config_with_root = LocalConfig {
            root_dir: Some(temp_dir.path().to_path_buf()),
            ..Default::default()
        };
        let fs_with_root = LocalFileSystem::new(config_with_root).await.unwrap();

        // Path extraction should return the path as-is
        let path_str = fs_with_root
            .resolve_path("file://./subdir/file.txt")
            .unwrap();
        assert_eq!(path_str, "./subdir/file.txt");

        // Test without root_dir - should be the same
        let config_no_root = LocalConfig::default();
        let fs_no_root = LocalFileSystem::new(config_no_root).await.unwrap();

        let path_str = fs_no_root.resolve_path("file://./subdir/file.txt").unwrap();
        assert_eq!(path_str, "./subdir/file.txt");
    }

    #[tokio::test]
    async fn test_prevent_double_relative_path_resolution() {
        // Serialize tests that change the current working directory
        let _cwd_guard = CWD_MUTEX.lock().await;

        // This test verifies that paths are used as-is without root_dir resolution
        // Previously there was a bug where paths were being resolved against root_dir

        let temp_dir = TempDir::new().unwrap();
        let working_dir = temp_dir.path();

        // Change to temp directory so relative paths work
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(working_dir).unwrap();

        // Create filesystem WITHOUT root_dir - paths should be used as-is
        let config = LocalConfig {
            root_dir: None,      // No root_dir - paths are used as-is
            sync_enabled: false, // Disable sync for testing
            ..Default::default()
        };
        let fs = LocalFileSystem::new(config).await.unwrap();

        // Test with URL - use FileOptions to ensure proper creation
        let options = Some(crate::storage::persistence::filesystem::FileOptions {
            create_dirs: true,
            overwrite: true,
            buffer_size: None,
            encryption: None,
            storage_class: None,
            metadata: None,
            temp_path: None,
        });

        // Write using a file:// URL with proper relative path format
        let test_path = "metadata/current/test.txt";
        let test_url = format!("file://./{}", test_path);
        fs.write(&test_url, b"test data", options.clone())
            .await
            .unwrap();

        // Verify the file was created at the expected location
        assert!(
            std::path::Path::new(test_path).exists(),
            "File should exist at {}",
            test_path
        );

        // Also test with a relative path (no file:// prefix)
        let another_path = "metadata/current/another.txt";
        fs.write(another_path, b"more data", options).await.unwrap();
        assert!(
            std::path::Path::new(another_path).exists(),
            "File should exist at {}",
            another_path
        );

        // Test that previously problematic paths work correctly
        // This used to cause double path resolution
        let problematic_path = "./test_metadata/current/file.txt";
        let problematic_url = format!("file://{}", problematic_path);
        // Use options to create directories
        let create_dirs_options = Some(crate::storage::persistence::filesystem::FileOptions {
            create_dirs: true,
            overwrite: true,
            buffer_size: None,
            encryption: None,
            storage_class: None,
            metadata: None,
            temp_path: None,
        });
        fs.write(&problematic_url, b"problematic data", create_dirs_options)
            .await
            .unwrap();
        assert!(
            std::path::Path::new("test_metadata/current/file.txt").exists(),
            "File should exist at test_metadata/current/file.txt"
        );

        // Restore original directory
        std::env::set_current_dir(original_dir).unwrap();
    }

    #[tokio::test]
    async fn test_exists_method_comprehensive() {
        debug!("Starting exists method comprehensive test");
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();
        debug!("Created temp directory: {}", temp_path.display());

        // Test with root_dir configuration - note: root_dir is validated/created but
        // path resolution uses the provided paths as-is, so we use absolute paths
        let config = LocalConfig {
            root_dir: Some(temp_path.to_path_buf()),
            follow_symlinks: true,
            default_permissions: None,
            sync_enabled: false,
        };
        debug!("Created config: {:?}", config);

        debug!("Creating LocalFileSystem...");
        let fs = match LocalFileSystem::new(config).await {
            Ok(fs) => {
                info!("LocalFileSystem created successfully");
                fs
            }
            Err(e) => {
                error!("Failed to create LocalFileSystem: {:?}", e);
                panic!("Failed to create filesystem: {:?}", e);
            }
        };

        // Use absolute paths based on temp_path for isolation
        let base_url = format!("file://{}", temp_path.display());

        // Test 1: Non-existent file should return false
        let non_existent_url = format!("{}/non_existent_file.txt", base_url);
        debug!("Testing non-existent file: {}", non_existent_url);
        let exists_result = fs.exists(&non_existent_url).await.unwrap();
        assert!(!exists_result, "Non-existent file should return false");
        info!("Non-existent file test passed");

        // Test 2: Create a file and test it exists
        let test_file_url = format!("{}/test_exists_file.txt", base_url);
        debug!("Testing file creation: {}", test_file_url);

        // Clean up any existing file first
        if fs.exists(&test_file_url).await.unwrap_or(false) {
            debug!("Cleaning up existing file: {}", test_file_url);
            let _ = fs.delete(&test_file_url).await;
        }

        debug!("Writing test file: {}", test_file_url);
        match fs.write(&test_file_url, b"test content", None).await {
            Ok(_) => info!("File written successfully"),
            Err(e) => {
                error!("Failed to write file: {:?}", e);
                panic!("Failed to write test file: {:?}", e);
            }
        }
        let exists_result = fs.exists(&test_file_url).await.unwrap();
        assert!(exists_result, "Created file should exist");

        // Test 3: Non-existent directory should return false
        let non_existent_dir_url = format!("{}/non_existent_dir", base_url);
        let exists_result = fs.exists(&non_existent_dir_url).await.unwrap();
        assert!(!exists_result, "Non-existent directory should return false");

        // Test 4: Create a directory and test it exists
        let test_dir_url = format!("{}/test_exists_dir", base_url);

        // Clean up any existing directory first
        if fs.exists(&test_dir_url).await.unwrap_or(false) {
            debug!("Cleaning up existing directory: {}", test_dir_url);
            let _ = fs.delete(&test_dir_url).await;
        }

        fs.create_dir(&test_dir_url).await.unwrap();
        let exists_result = fs.exists(&test_dir_url).await.unwrap();
        assert!(exists_result, "Created directory should exist");

        // Test 5: Test with nested path
        let nested_file_url = format!("{}/test_exists_dir/nested_file.txt", base_url);
        fs.write(&nested_file_url, b"nested content", None)
            .await
            .unwrap();
        let exists_result = fs.exists(&nested_file_url).await.unwrap();
        assert!(exists_result, "Nested file should exist");

        // Test 6: Test after deletion
        fs.delete(&test_file_url).await.unwrap();
        let exists_result = fs.exists(&test_file_url).await.unwrap();
        assert!(!exists_result, "Deleted file should not exist");

        info!("All exists() method tests passed!");
    }

    #[tokio::test]
    async fn test_exists_method_without_root_dir() {
        // Serialize tests that change the current working directory
        let _cwd_guard = CWD_MUTEX.lock().await;

        // Save current directory to restore later
        let original_dir = std::env::current_dir().unwrap();

        // Create a temporary directory and change to it
        let temp_dir = TempDir::new().unwrap();
        std::env::set_current_dir(temp_dir.path()).unwrap();

        // Test without root_dir (relative paths)
        let config = LocalConfig {
            root_dir: None,
            follow_symlinks: true,
            default_permissions: None,
            sync_enabled: false,
        };

        let fs = LocalFileSystem::new(config).await.unwrap();

        // Clean up any existing test files
        if std::path::Path::new("test_exists_relative.txt").exists() {
            std::fs::remove_file("test_exists_relative.txt").ok();
        }

        // Test 1: Non-existent relative file
        let relative_file_url = "file://./test_exists_relative.txt";
        let exists_result = fs.exists(relative_file_url).await.unwrap();
        assert!(
            !exists_result,
            "Non-existent relative file should return false"
        );

        // Test 2: Create and test relative file
        fs.write(relative_file_url, b"relative test content", None)
            .await
            .unwrap();
        let exists_result = fs.exists(relative_file_url).await.unwrap();
        assert!(exists_result, "Created relative file should exist");

        // Test 3: Verify the file actually exists on disk
        assert!(
            std::path::Path::new("test_exists_relative.txt").exists(),
            "File should exist on disk at relative path"
        );

        // Clean up
        fs.delete(relative_file_url).await.unwrap();
        let exists_result = fs.exists(relative_file_url).await.unwrap();
        assert!(!exists_result, "Deleted relative file should not exist");

        // Restore original directory
        std::env::set_current_dir(original_dir).unwrap();

        info!("✅ All exists() relative path tests passed!");
    }

    // Additional sync tests from standalone test file

    #[tokio::test]
    async fn test_local_filesystem_sync_file() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        let config = LocalConfig {
            root_dir: Some(base_path.into()),
            sync_enabled: true,
            ..Default::default()
        };

        let fs = LocalFileSystem::new(config).await.unwrap();

        // Write test data
        let test_path = "test_sync.dat";
        let test_data = b"Critical data that must be synced";
        // Clean up if file exists from previous run
        if fs.exists(test_path).await.unwrap_or(false) {
            let _ = fs.delete(test_path).await;
        }
        fs.write(test_path, test_data, None).await.unwrap();

        // Yield to ensure async write completes before sync
        tokio::task::yield_now().await;

        // Call sync_file - should succeed
        fs.sync_file(test_path).await.unwrap();

        // Verify data can be read back
        let read_data = fs.read(test_path).await.unwrap();
        assert_eq!(read_data, test_data);
    }

    #[tokio::test]
    async fn test_local_filesystem_sync_disabled() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        let config = LocalConfig {
            root_dir: Some(base_path.into()),
            sync_enabled: false, // Sync disabled
            ..Default::default()
        };

        let fs = LocalFileSystem::new(config).await.unwrap();

        // Write test data
        let test_path = "test_no_sync.dat";
        let test_data = b"Data without sync";
        // Clean up if file exists from previous run
        if fs.exists(test_path).await.unwrap_or(false) {
            let _ = fs.delete(test_path).await;
        }
        fs.write(test_path, test_data, None).await.unwrap();

        // Call sync_file - should succeed but do nothing
        fs.sync_file(test_path).await.unwrap();

        // Data should still be readable
        let read_data = fs.read(test_path).await.unwrap();
        assert_eq!(read_data, test_data);
    }

    #[tokio::test]
    async fn test_sync_file_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        let config = LocalConfig {
            root_dir: Some(base_path.into()),
            sync_enabled: true,
            ..Default::default()
        };

        let fs = LocalFileSystem::new(config).await.unwrap();

        // Try to sync non-existent file
        let result = fs.sync_file("non_existent.dat").await;

        // Should fail with appropriate error
        assert!(result.is_err());
        match result.unwrap_err() {
            FilesystemError::NotFound(_) => {} // Expected
            e => panic!(
                "Expected NotFound error for non-existent file, got: {:?}",
                e
            ),
        }
    }

    #[tokio::test]
    async fn test_sync_after_append() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        let config = LocalConfig {
            root_dir: Some(base_path.into()),
            sync_enabled: true,
            ..Default::default()
        };

        let fs = LocalFileSystem::new(config).await.unwrap();

        let test_path = "append_sync.dat";

        // Clean up if file exists from previous run
        if fs.exists(test_path).await.unwrap_or(false) {
            let _ = fs.delete(test_path).await;
        }

        // Initial write
        fs.write(test_path, b"Initial data", None).await.unwrap();
        fs.sync_file(test_path).await.unwrap();

        // Append more data
        fs.append(test_path, b" - Appended data").await.unwrap();

        // Yield to ensure async operations complete
        tokio::task::yield_now().await;

        // Sync after append
        fs.sync_file(test_path).await.unwrap();

        // Verify complete data
        let read_data = fs.read(test_path).await.unwrap();
        assert_eq!(read_data, b"Initial data - Appended data");
    }

    #[tokio::test]
    async fn test_concurrent_sync_operations() {
        // Use a fixture-based approach with proper async synchronization
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        let config = LocalConfig {
            root_dir: Some(base_path.into()),
            sync_enabled: true,
            ..Default::default()
        };

        let fs = Arc::new(LocalFileSystem::new(config).await.unwrap());

        // Test data: file paths and their contents
        let test_files = vec![
            ("file1.dat", b"Data 1"),
            ("file2.dat", b"Data 2"),
            ("file3.dat", b"Data 3"),
        ];

        // Setup phase: Write all files synchronously with verification
        // This ensures all files exist and are readable before concurrent operations
        for (path, data) in &test_files {
            // Clean up any existing file
            if fs.exists(path).await.unwrap_or(false) {
                fs.delete(path).await.unwrap();
            }

            // Write and immediately verify
            fs.write(path, &data[..], None).await.unwrap();

            // Synchronously verify the write was successful
            let read_data = fs.read(path).await.unwrap();
            assert_eq!(
                &read_data, data,
                "Initial write verification failed for {}",
                path
            );
        }

        // Use a barrier to ensure all writes are flushed before concurrent syncs
        // This is more reliable than sleep-based approaches
        // Use tokio::sync::Barrier for async-compatible synchronization
        let barrier = Arc::new(tokio::sync::Barrier::new(test_files.len() + 1));
        let mut handles = vec![];

        // Spawn concurrent sync operations with barrier synchronization
        for (path, _) in &test_files {
            let fs_clone = Arc::clone(&fs);
            let barrier_clone = Arc::clone(&barrier);
            let path = path.to_string();

            handles.push(tokio::spawn(async move {
                // Wait for all tasks to be ready before starting
                barrier_clone.wait().await;

                // Perform sync operation with simple error handling
                // No retries or timeouts - let the test fail if there's a real issue
                match fs_clone.sync_file(&path).await {
                    Ok(()) => Ok(path),
                    Err(e) => Err((path, e)),
                }
            }));
        }

        // Wait for all tasks to be ready, then release the barrier
        barrier.wait().await;

        // Collect results with clear error messages
        let mut results = vec![];
        for handle in handles {
            match handle.await {
                Ok(Ok(path)) => results.push((path, true)),
                Ok(Err((path, e))) => {
                    eprintln!("Sync failed for {}: {}", path, e);
                    results.push((path, false));
                }
                Err(e) => {
                    eprintln!("Sync task panicked: {}", e);
                    results.push(("unknown".to_string(), false));
                }
            }
        }

        // Assert with clear, stable error messages
        let failed_syncs: Vec<_> = results
            .into_iter()
            .filter(|(_, success)| !success)
            .collect();
        assert!(
            failed_syncs.is_empty(),
            "Sync operations failed for files: {:?}",
            failed_syncs
                .iter()
                .map(|(path, _)| path)
                .collect::<Vec<_>>()
        );

        // Final verification: Ensure all data is intact after concurrent syncs
        for (path, expected_data) in &test_files {
            let read_data = fs.read(path).await.unwrap();
            assert_eq!(
                &read_data, expected_data,
                "Data verification failed after concurrent sync for {}",
                path
            );
        }
    }

    #[tokio::test]
    async fn write_if_absent_is_atomic_and_preserves_first_value() {
        let temp_dir = TempDir::new().unwrap();
        let fs = LocalFileSystem::new(LocalConfig {
            root_dir: Some(temp_dir.path().to_path_buf()),
            sync_enabled: true,
            ..Default::default()
        })
        .await
        .unwrap();
        fs.write_if_absent(
            "commit.pax",
            b"first",
            Some(FileOptions {
                create_dirs: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let error = fs
            .write_if_absent("commit.pax", b"second", None)
            .await
            .unwrap_err();
        assert!(matches!(error, FilesystemError::AlreadyExists(_)));
        assert_eq!(fs.read("commit.pax").await.unwrap(), b"first");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn conditional_publication_decorators_forward_and_encryption_rejects() {
        use crate::storage::encryption::{
            FileEncryptionLayer,
            key_manager::{KeyManager, KeyVersionManager},
        };
        use proximadb_storage_filesystem_types::counting::{CountingFileSystem, GetCounters};
        use proximadb_storage_filesystem_types::faults::FaultInjectingFileSystem;
        use std::sync::Arc;
        let dir = TempDir::new().unwrap();
        let config = LocalConfig {
            root_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        let local: Arc<dyn FileSystem> =
            Arc::new(LocalFileSystem::new(config.clone()).await.unwrap());
        let counted: Arc<dyn FileSystem> = Arc::new(CountingFileSystem::new(
            local,
            Arc::new(GetCounters::default()),
        ));
        let fs = FaultInjectingFileSystem::new(counted);
        assert!(fs.supports_conditional_replace());
        assert!(
            fs.compare_exchange("lease.meta", None, b"first", &[])
                .await
                .unwrap()
        );
        assert!(
            !fs.compare_exchange("lease.meta", None, b"bad", &[])
                .await
                .unwrap()
        );
        assert!(
            fs.compare_exchange("lease.meta", Some(b"first"), b"second", &[])
                .await
                .unwrap()
        );
        assert_eq!(fs.read("lease.meta").await.unwrap(), b"second");
        assert!(
            !fs.compare_exchange(
                "offset.meta",
                None,
                b"progress",
                &[("lease.meta", Some(b"first"))]
            )
            .await
            .unwrap()
        );
        assert!(
            fs.compare_exchange(
                "offset.meta",
                None,
                b"progress",
                &[("lease.meta", Some(b"second"))]
            )
            .await
            .unwrap()
        );
        assert_eq!(fs.read("offset.meta").await.unwrap(), b"progress");
        let mut key = [0; 32];
        key[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        key[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        let keys = Arc::new(KeyVersionManager::new(Arc::new(KeyManager::new(key))));
        let encrypted = LocalFileSystem::new_with_encryption(
            config,
            Some(Arc::new(FileEncryptionLayer::new(keys, true, 4096))),
        )
        .await
        .unwrap();
        assert!(!encrypted.supports_conditional_replace());
        assert!(matches!(
            encrypted
                .compare_exchange("lease.meta", Some(b"second"), b"bad", &[])
                .await,
            Err(FilesystemError::InvalidOperation(_))
        ));
        assert_eq!(
            std::fs::read(dir.path().join("lease.meta")).unwrap(),
            b"second"
        );
    }
}
