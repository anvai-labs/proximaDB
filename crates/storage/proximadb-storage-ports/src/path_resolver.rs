//! Collection path resolver port (DIP-compliant interface).
//!
//! Abstracts the resolution of storage paths for collections, replacing global
//! singletons with dependency injection. Originally rooted at
//! `src/storage/trait_components/path_resolver.rs`; the **port trait + value
//! types** are hoisted here (root-crate decomposition, Slice D) so that
//! `src/storage` can depend *down* on this abstraction without dragging the
//! root-catalog-coupled concrete resolvers (`ConfigFallbackResolver`,
//! `CachedResolver`, `CompositeResolver`) along with it. Those impls stay in
//! the root crate and `impl` this trait — a downward edge (root→storage-ports
//! is layering-allowed; the forbidden direction was storage→root-internal).
//!
//! The trait + `StorageAssignment` are deliberately primitive-typed
//! (`String`/`bool`/`u32`/`Vec<String>`): NO root-internal type crosses the
//! seam, so the port is facade-free.

use anyhow::Result;
use async_trait::async_trait;
use proximadb_kernel::stable_id::CollectionIdentity;
use proximadb_proto::proximadb_v1::StorageAssignment as ProtoStorageAssignment;
use proximadb_storage_common::StoragePath;

/// Storage location assignment for a collection.
#[derive(Debug, Clone)]
pub struct StorageAssignment {
    /// Primary storage URL (e.g., "file:///data/proximadb/d1")
    pub primary_url: String,
    /// Weight for load balancing (1-100)
    pub weight: u32,
    /// Whether this location is available
    pub available: bool,
    /// Optional: Replica URLs for high availability
    pub replica_urls: Vec<String>,
}

impl Default for StorageAssignment {
    fn default() -> Self {
        Self {
            primary_url: "file:///tmp/proximadb/data".to_string(),
            weight: 1,
            available: true,
            replica_urls: Vec::new(),
        }
    }
}

/// Collection path resolver trait (DIP-compliant interface).
///
/// Abstracts the resolution of storage paths for collections,
/// replacing global singletons with dependency injection.
///
/// Available root-crate implementations (kept in
/// `src/storage/trait_components/path_resolver.rs` because they are
/// root-catalog-coupled):
/// - `ConfigFallbackResolver`: Uses WAL config paths (for testing)
/// - `CachedResolver`: Caches resolved paths (for performance)
/// - `CompositeResolver`: Fallback chain
#[async_trait]
pub trait CollectionPathResolver: Send + Sync {
    /// Resolver name for logging/debugging
    fn name(&self) -> &'static str;

    /// Resolve the base storage location for a collection.
    ///
    /// # Arguments
    /// * `collection_id` - The collection identifier
    ///
    /// # Returns
    /// The base URL for the collection's storage (e.g., "file:///data/proximadb/collections/my_collection")
    async fn resolve_base_location(&self, collection_id: &str) -> Result<String>;

    /// Resolve the storage assignment for a collection.
    ///
    /// # Arguments
    /// * `collection_id` - The collection identifier
    ///
    /// # Returns
    /// Storage assignment details including primary URL and replicas
    async fn resolve_storage_assignment(&self, collection_id: &str) -> Result<StorageAssignment>;

    /// Resolve the WAL directory for a collection.
    ///
    /// # Arguments
    /// * `collection_id` - The collection identifier
    ///
    /// # Returns
    /// The WAL directory URL (e.g., "file:///data/proximadb/collections/my_collection/wal")
    async fn resolve_wal_location(&self, collection_id: &str) -> Result<String> {
        let base = self.resolve_base_location(collection_id).await?;
        Ok(format!("{}/wal", base))
    }

    /// Resolve the SST directory for a collection.
    ///
    /// # Arguments
    /// * `collection_id` - The collection identifier
    ///
    /// # Returns
    /// The SST files directory URL
    async fn resolve_sst_location(&self, collection_id: &str) -> Result<String> {
        let base = self.resolve_base_location(collection_id).await?;
        Ok(format!("{}/sst", base))
    }

    /// Check if a collection exists.
    async fn collection_exists(&self, collection_id: &str) -> Result<bool>;
}

// ---------------------------------------------------------------------------
// ADR-031 Phase 4c/4d typed-path helpers (hoisted from the root crate).
// ---------------------------------------------------------------------------
//
// `CollectionIdentity` is a foundation type (`proximadb_kernel::stable_id`);
// `StoragePath` lives in `proximadb_storage_common`, which CAN now name
// `CollectionIdentity` (both foundation-tier). So the typed variants live HERE
// — a port-layer helper that wraps the legacy `StoragePath` calls for the
// `None` (legacy) branch and composes the account-rooted zero-padded base62
// path for the `Some(identity)` branch.
//
// Both branches share the SAME trailing subpath suffix as the legacy
// `StoragePath::collection_*_path` (`/data`, … — NO trailing slash), so the
// `None` branch is byte-identical to the pre-4c path and the `Some` branch
// differs only in the prefix (mixed-read-safe per-collection).

/// ADR-031 Phase 4d: recover a [`CollectionIdentity`] from a proto
/// [`ProtoStorageAssignment`]'s typed triple, for the **catalog-free engine
/// read paths**.
///
/// Engines resolve data/wal/index paths deep in the search/flush stack with no
/// catalog/schema access — the typed identity cannot be re-minted at read time,
/// so it is carried on the proto collection (set at create by the manager when
/// `PROXIMADB_TYPED_PATHS=1`) and reconstituted here. All three fields are `Some`
/// together (the manager sets them atomically) or all `None` (env OFF / legacy
/// collection created before 4d) → `None` → the typed path helpers fall back to
/// the byte-identical legacy path (mixed-read-safe per-collection).
///
/// `namespace_id` is a `u16` in the typed identity but stored as `uint32` in
/// proto (proto has no `uint16`); it is narrowed here. Values > `u16::MAX` are
/// impossible by construction (the catalog mints `NamespaceId = u16`), so the
/// narrowing is infallible in practice — `None` is returned defensively if a
/// future caller somehow stored an out-of-range value.
pub fn typed_identity_from_storage_assignment(
    storage_assignment: Option<&ProtoStorageAssignment>,
) -> Option<CollectionIdentity> {
    let sa = storage_assignment?;
    let account_id = sa.typed_account_id?;
    let namespace_id = sa.typed_namespace_id?;
    let collection_id = sa.typed_collection_id?;
    // Proto has no uint16; narrow back to the typed NamespaceId (u16).
    let namespace_id = if namespace_id <= u32::from(u16::MAX) {
        namespace_id as u16
    } else {
        // Defensive: out-of-range means the triple wasn't minted by the catalog
        // — treat as legacy rather than truncate silently.
        return None;
    };
    Some(CollectionIdentity {
        account_id,
        namespace_id,
        collection_id,
    })
}

/// Select the catalog identity for a physical typed-path lookup.
///
/// ADR-0083 makes the numeric identity authoritative and therefore persists it
/// regardless of path layout. ADR-031 keeps the physical base62 layout behind
/// `PROXIMADB_TYPED_PATHS` until its mixed-read migration is complete. Keeping
/// those decisions separate is essential: an identity being present must not
/// silently opt a collection into a different object-store prefix.
pub fn typed_path_identity(identity: Option<CollectionIdentity>) -> Option<CollectionIdentity> {
    typed_path_identity_when(identity, typed_paths_enabled())
}

/// Pure policy seam for deterministic tests and config-driven callers.
pub fn typed_path_identity_when(
    identity: Option<CollectionIdentity>,
    enabled: bool,
) -> Option<CollectionIdentity> {
    enabled.then_some(identity).flatten()
}

proximadb_config::env_gate! {
    /// Whether the base62 account/namespace/collection path layout is enabled.
    ///
    /// Read once per process so every engine path observes one layout decision and
    /// cannot split reads from writes if the process environment changes later.
    pub fn typed_paths_enabled() { gate: "PROXIMADB_TYPED_PATHS", default: false }
}

/// ADR-031 Phase 4c: typed collection **data** directory path.
///
/// * `Some(identity)` → `{base}/accounts/{acct}/{ns}/{coll}/data`
///   (zero-padded base62, no tenant slot — Phase 4 hierarchy collapse).
/// * `None`           → byte-identical legacy
///   [`StoragePath::collection_data_path`] (`{base}/{collection_id}/data`).
///
/// The trailing suffix (`/data`, no slash) matches the legacy contract exactly
/// so reads/writes against a legacy collection (`None`) resolve unchanged.
pub fn collection_data_path_typed(
    base: &str,
    collection_id: &str,
    identity: Option<CollectionIdentity>,
) -> String {
    match identity {
        Some(id) => {
            let (acct, ns, coll) = id.path_segments();
            format!("{base}/accounts/{acct}/{ns}/{coll}/data")
        }
        None => StoragePath::collection_data_path(base, collection_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_assignment_default_is_sane() {
        let a = StorageAssignment::default();
        assert!(a.available);
        assert_eq!(a.weight, 1);
        assert!(a.replica_urls.is_empty());
        assert!(!a.primary_url.is_empty());
    }
}
