//! Disposable local whole-file cache. Reads retain a file lease while mutations
//! serialize index publication and capacity accounting. Source keys stay opaque.
use dashmap::DashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Mutex;
use tracing::warn;

pub struct DiskCacheManager {
    cache_dir: PathBuf,
    max_size_bytes: usize,
    enabled: bool,
    mutation: Mutex<()>,
    current_size: AtomicUsize,
    entries: DashMap<String, CacheEntry>,
    stats: CacheStats,
}
#[derive(Clone)]
struct CacheEntry {
    // The final lease removes the file, after any in-flight byte read finishes.
    file: Arc<tempfile::TempPath>,
    size: usize,
    last_accessed: Instant,
}
#[derive(Default)]
struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    bytes_saved: AtomicU64,
}
impl DiskCacheManager {
    pub fn new(cache_dir: PathBuf, max_size_gb: usize) -> Self {
        let enabled = if cache_dir.to_string_lossy().contains("://") {
            tracing::error!(
                "disk cache requires a local directory: {}",
                cache_dir.display()
            );
            false
        } else if let Err(error) = std::fs::create_dir_all(&cache_dir) {
            tracing::error!("disk cache directory unavailable: {error}");
            false
        } else {
            true
        };
        Self {
            cache_dir,
            max_size_bytes: max_size_gb.saturating_mul(1024 * 1024 * 1024),
            enabled,
            mutation: Mutex::new(()),
            current_size: AtomicUsize::new(0),
            entries: DashMap::new(),
            stats: CacheStats::default(),
        }
    }
    /// Compatibility path lookup. Serving callers should use get_data/get_range
    /// to retain the local file through their read instead of routing this path.
    /// This non-serving lookup does not count hits or bytes saved.
    pub async fn get(&self, key: &str) -> Option<String> {
        let entry = self.lookup(key)?;
        if entry.file.exists() {
            Some(entry.file.to_string_lossy().into_owned())
        } else {
            self.discard_if_current(key, &entry).await;
            None
        }
    }
    fn lookup(&self, key: &str) -> Option<CacheEntry> {
        self.entries.get_mut(key).map(|mut entry| {
            entry.last_accessed = Instant::now();
            entry.clone()
        })
    }
    pub async fn get_data(&self, key: &str) -> Option<Vec<u8>> {
        let Some(entry) = self.lookup(key) else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        self.read_entry(key, entry, 0, None).await
    }
    /// Read just the requested range from a cached whole body.
    pub async fn get_range(&self, key: &str, offset: u64, length: u64) -> Option<Vec<u8>> {
        let Some(entry) = self.lookup(key) else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        self.read_entry(key, entry, offset, Some(length)).await
    }
    async fn read_entry(
        &self,
        key: &str,
        entry: CacheEntry,
        offset: u64,
        length: Option<u64>,
    ) -> Option<Vec<u8>> {
        let length = length.unwrap_or(entry.size as u64);
        if offset
            .checked_add(length)
            .is_none_or(|end| end > entry.size as u64)
        {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let Ok(size) = usize::try_from(length) else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let result: std::io::Result<Vec<u8>> = async {
            let mut file = tokio::fs::File::open(entry.file.as_ref()).await?;
            if file.metadata().await?.len() != entry.size as u64 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "cached size changed",
                ));
            }
            file.seek(std::io::SeekFrom::Start(offset)).await?;
            let mut data = vec![0; size];
            file.read_exact(&mut data).await?;
            Ok(data)
        }
        .await;
        match result {
            Ok(data) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.stats.bytes_saved.fetch_add(length, Ordering::Relaxed);
                Some(data)
            }
            Err(error) => {
                warn!("local cache read failed: {error}");
                self.discard_if_current(key, &entry).await;
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }
    async fn discard_if_current(&self, key: &str, entry: &CacheEntry) {
        let _guard = self.mutation.lock().await;
        let current = self
            .entries
            .get(key)
            .is_some_and(|current| Arc::ptr_eq(&current.file, &entry.file));
        if current {
            self.remove_locked(key);
        }
    }
    /// Publish fully written bytes. Failed/oversized admissions invalidate any
    /// stale same-key body but preserve unrelated entries. Source data is untouched.
    pub async fn put(&self, key: &str, data: &[u8]) {
        let _guard = self.mutation.lock().await;
        // A cancelled replacement must not retain stale bytes at this key.
        // Existing readers keep their lease to the old body until they finish.
        self.remove_locked(key);
        if !self.enabled || data.len() > self.max_size_bytes || self.max_size_bytes == 0 {
            return;
        }
        let result: std::io::Result<tempfile::TempPath> = async {
            let temporary = tempfile::Builder::new()
                .prefix("cache-")
                .tempfile_in(&self.cache_dir)?;
            let (file, path) = temporary.into_parts();
            let mut file = tokio::fs::File::from_std(file);
            file.write_all(data).await?;
            file.flush().await?;
            Ok(path)
        }
        .await;
        let file = match result {
            Ok(file) => file,
            Err(error) => {
                warn!("local cache fill failed: {error}");
                return;
            }
        };
        // No awaits between accounting, eviction and publication. File leases
        // keep evicted/replaced bodies alive only while a read uses them.
        let mut victims: Vec<_> = self
            .entries
            .iter()
            .map(|entry| (entry.key().clone(), entry.last_accessed))
            .collect();
        victims.sort_by_key(|(_, touched)| *touched);
        for (victim, _) in victims {
            if self.current_size.load(Ordering::Relaxed) <= self.max_size_bytes - data.len() {
                break;
            }
            if self.remove_locked(&victim) {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.entries.insert(
            key.to_owned(),
            CacheEntry {
                file: Arc::new(file),
                size: data.len(),
                last_accessed: Instant::now(),
            },
        );
        self.current_size.fetch_add(data.len(), Ordering::Relaxed);
    }
    fn remove_locked(&self, key: &str) -> bool {
        if let Some((_, entry)) = self.entries.remove(key) {
            self.current_size.fetch_sub(entry.size, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
    pub async fn invalidate(&self, key: &str) {
        let _guard = self.mutation.lock().await;
        self.remove_locked(key);
    }
    pub async fn invalidate_prefix(&self, prefix: &str) {
        let _guard = self.mutation.lock().await;
        let keys: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.key().starts_with(prefix))
            .map(|e| e.key().clone())
            .collect();
        for key in keys {
            self.remove_locked(&key);
        }
    }
    pub fn stats(&self) -> DiskCacheStatistics {
        DiskCacheStatistics {
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            bytes_saved: self.stats.bytes_saved.load(Ordering::Relaxed),
            resident_bytes: self.current_size.load(Ordering::Relaxed),
            entries: self.entries.len(),
        }
    }
}
#[derive(Debug, Clone)]
pub struct DiskCacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_saved: u64,
    /// Bytes in the resident index, excluding temporary fills and active read leases.
    pub resident_bytes: usize,
    pub entries: usize,
}
impl DiskCacheStatistics {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn cache(capacity: usize) -> (DiskCacheManager, tempfile::TempDir) {
        let directory = tempfile::tempdir().expect("cache directory");
        let mut cache = DiskCacheManager::new(directory.path().to_owned(), 1);
        cache.max_size_bytes = capacity;
        (cache, directory)
    }

    #[tokio::test]
    async fn capacity_eviction_finishes_and_replacement_accounts_exact_bytes() {
        let (cache, _directory) = cache(6);
        cache.put("a", b"1111").await;
        tokio::time::timeout(Duration::from_secs(2), cache.put("b", b"2222"))
            .await
            .expect("capacity eviction must not self-deadlock");
        assert!(cache.get_data("a").await.is_none());
        assert_eq!(
            cache.get_data("b").await.as_deref(),
            Some(b"2222".as_slice())
        );
        assert_eq!(cache.stats().resident_bytes, 4);
        assert_eq!(cache.stats().evictions, 1);
        cache.put("b", b"33").await;
        cache.put("c", b"4444").await;
        assert_eq!(cache.stats().resident_bytes, 6);
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.get_data("b").await.as_deref(), Some(b"33".as_slice()));
        cache.put("b", b"too large").await;
        assert!(
            cache.get_data("b").await.is_none(),
            "oversized replacement must not retain stale data"
        );
        assert_eq!(cache.stats().resident_bytes, 4);
        assert_eq!(
            cache.get_data("c").await.as_deref(),
            Some(b"4444".as_slice())
        );
    }

    #[tokio::test]
    async fn ranges_count_only_served_bytes_and_reject_overflow() {
        let (cache, _directory) = cache(20);
        cache.put("s3://bucket/key", b"abcdefghij").await;
        assert_eq!(
            cache.get_range("s3://bucket/key", 2, 3).await.as_deref(),
            Some(b"cde".as_slice())
        );
        assert!(
            cache
                .get_range("s3://bucket/key", u64::MAX, 2)
                .await
                .is_none()
        );
        assert!(cache.get_range("s3://bucket/key", 8, 3).await.is_none());
        assert_eq!(cache.stats().bytes_saved, 3);
        assert_eq!(cache.stats().resident_bytes, 10);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 2);
    }

    #[tokio::test]
    async fn missing_and_truncated_files_remove_resident_accounting() {
        let (cache, _directory) = cache(20);
        cache.put("missing", b"abc").await;
        let path = cache.get("missing").await.expect("cached path");
        tokio::fs::remove_file(path)
            .await
            .expect("remove cache file");
        assert!(cache.get_data("missing").await.is_none());
        assert_eq!(cache.stats().resident_bytes, 0);
        cache.put("truncated", b"abcdef").await;
        let path = cache.get("truncated").await.expect("cached path");
        tokio::fs::write(path, b"a")
            .await
            .expect("truncate cache file");
        assert!(cache.get_range("truncated", 0, 1).await.is_none());
        assert_eq!(cache.stats().resident_bytes, 0);
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().hits, 0);
    }

    #[tokio::test]
    async fn read_lease_survives_replacement_and_stale_cleanup_keeps_new_body() {
        let (cache, _directory) = cache(20);
        cache.put("key", b"old").await;
        let old = cache.lookup("key").expect("read lease");
        let old_path: PathBuf = old.file.to_path_buf();
        cache.put("key", b"newer").await;
        assert_eq!(
            tokio::fs::read(&old_path).await.expect("leased body"),
            b"old"
        );
        cache.discard_if_current("key", &old).await;
        assert_eq!(
            cache.get_data("key").await.as_deref(),
            Some(b"newer".as_slice())
        );
        assert_eq!(cache.stats().resident_bytes, 5);
        drop(old);
        assert!(!old_path.exists(), "last reader releases obsolete file");
        cache.invalidate("key").await;
        assert_eq!(cache.stats().resident_bytes, 0);
    }

    #[tokio::test]
    async fn failed_fill_invalidates_same_key_and_preserves_unrelated_entries() {
        let (mut cache, directory) = cache(20);
        cache.put("key", b"old").await;
        cache.put("other", b"keep").await;
        let blocked = directory.path().join("not-a-directory");
        tokio::fs::write(&blocked, b"x")
            .await
            .expect("block cache directory");
        cache.cache_dir = blocked;
        cache.put("key", b"new").await;
        assert!(cache.get_data("key").await.is_none());
        assert_eq!(
            cache.get_data("other").await.as_deref(),
            Some(b"keep".as_slice())
        );
        assert_eq!(cache.stats().resident_bytes, 4);
    }
    #[tokio::test]
    async fn concurrent_mutations_and_prefix_invalidation_remain_bounded() {
        let (cache, _directory) = cache(12);
        let cache = Arc::new(cache);
        let mut workers = tokio::task::JoinSet::new();
        for worker in 0..8 {
            let cache = cache.clone();
            workers.spawn(async move {
                let key = format!("collection/{worker}");
                for _ in 0..8 {
                    cache.put(&key, b"body").await;
                    let _ = cache.get_data(&key).await;
                    cache.invalidate(&key).await;
                }
            });
        }
        tokio::time::timeout(Duration::from_secs(3), async {
            while let Some(result) = workers.join_next().await {
                result.expect("cache worker");
            }
        })
        .await
        .expect("concurrent mutation must complete");
        assert_eq!(cache.stats().resident_bytes, 0);
        cache.put("collection/a", b"aaaa").await;
        cache.put("collection/b", b"bbbb").await;
        cache.put("other/c", b"cccc").await;
        cache.invalidate_prefix("collection/").await;
        assert_eq!(cache.stats().resident_bytes, 4);
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(
            cache.get_data("other/c").await.as_deref(),
            Some(b"cccc".as_slice())
        );
    }
}
