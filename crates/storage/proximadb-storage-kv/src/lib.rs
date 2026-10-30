//! Simple key-value storage abstraction
//!
//! This crate provides a minimal key-value storage interface with a filesystem-based
//! implementation. Used for storing configuration, metadata, and other small data items.
//!
//! Extracted from the root crate's `src/storage/kv` as the first slice of the root-crate
//! decomposition (see `docs/history/planning/ROOT_CRATE_DECOMPOSITION_PLAN_2026_06_21.adoc`).
//! The root crate re-exports it as `crate::storage::kv` for source compatibility.

use anyhow::Result;
use async_trait::async_trait;

/// Key-value storage trait for simple binary data operations
///
/// Provides basic CRUD operations for binary key-value storage.
/// Keys are strings, values are byte arrays.
#[async_trait]
pub trait StorageKV: Send + Sync {
    /// Store a value at the given key
    async fn put(&self, key: &str, value: &[u8]) -> Result<()>;

    /// Retrieve a value by key, returns None if not found
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Delete a value by key (no error if key doesn't exist)
    async fn delete(&self, key: &str) -> Result<()>;
}

/// Filesystem-based key-value storage implementation
///
/// Stores each key-value pair as a separate file in a directory hierarchy.
/// Keys with "/" are sanitized to "__" for filesystem compatibility.
///
/// # Example
///
/// ```rust,ignore
/// use proximadb::storage::kv::FsKV;
///
/// let kv = FsKV::new("/var/lib/proximadb/kv");
/// kv.put("config", b"{\"setting\": true}").await?;
/// let value = kv.get("config").await?;
/// ```
pub struct FsKV {
    /// Base directory for storing key-value files
    base_dir: std::path::PathBuf,
}

impl FsKV {
    /// Create a new filesystem KV store at the given base directory
    ///
    /// # Arguments
    ///
    /// * `base_dir`: Directory where KV files will be stored
    pub fn new<P: Into<std::path::PathBuf>>(base_dir: P) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// Convert a key to a filesystem path
    fn path_for(&self, key: &str) -> std::path::PathBuf {
        let mut p = self.base_dir.clone();
        p.push(format!("{}.bin", key.replace('/', "__")));
        p
    }
}

#[async_trait]
impl StorageKV for FsKV {
    /// Store a value at the given key
    async fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        let path = self.path_for(key);
        if let Some(parent) = path.parent() {
            // Create parent directories - errors will surface when creating file
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                tracing::debug!("Note: Failed to create parent dirs {:?}: {}", parent, e);
            }
        }
        let mut f = tokio::fs::File::create(&path).await?;
        use tokio::io::AsyncWriteExt;
        f.write_all(value).await?;
        Ok(())
    }

    /// Retrieve a value by key, returns None if not found
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let path = self.path_for(key);
        match tokio::fs::read(&path).await {
            Ok(b) => Ok(Some(b)),
            Err(_) => Ok(None),
        }
    }

    /// Delete a value by key (no error if key doesn't exist)
    async fn delete(&self, key: &str) -> Result<()> {
        let path = self.path_for(key);
        let _ = tokio::fs::remove_file(&path).await;
        Ok(())
    }
}
