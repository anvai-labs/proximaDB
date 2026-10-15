//! Shared runtime utilities that are reusable across storage, query, modality,
//! platform, and application crates.
//!
//! This crate must stay horizontal: it may depend on foundation crates when
//! necessary, but it must not depend upward into domain, platform, integration,
//! root, application, or binding crates.

pub mod btree;
pub mod cache;
pub mod disk_cache;
pub mod file_lock;
pub mod pool;
pub mod query_cache;
pub mod skiplist;
pub mod vector_ops;

pub use btree::{BPlusTree, BTreeError, BTreeIterator, BTreeStats, DiskNodeInfo};
pub use cache::{CacheEntry, CacheError, CacheStats, LruCache, ThreadSafeLruCache};
pub use disk_cache::{DiskCacheManager, DiskCacheStatistics};
pub use file_lock::{
    AccessMode, FileLockError, FileLockManager, FileLockSet, LockOwner, is_coordination_lock_file,
};
pub use pool::{Pool, PoolConfig, PoolStats, PooledItem, VectorMemoryPool, VectorPoolStats};
pub use query_cache::{Cache, ShardedMapCache};
pub use skiplist::{SkipList, SkipListIterator};
pub use vector_ops::{
    cosine_similarity, dot_product, mean, normalize_l2, resize_vector, standard_deviation,
    validate_vector,
};
