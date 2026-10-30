// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! # Collection Service - Core Business Logic and Collection Management
//!
//! This service is the central orchestrator for all collection-related operations in ProximaDB.
//! It provides a unified interface that abstracts storage details from the API layer while
//! managing collection lifecycle, metadata, and coordination with storage engines.
//!
//! ## Role in ProximaDB Architecture
//!
//! The CollectionService sits at the heart of the service layer:
//! ```text
//! API Handlers → CollectionService → Storage/Index/WAL
//!                     ↓
//!              Metadata Backend
//! ```
//!
//! ## Key Responsibilities
//!
//! 1. **Collection Lifecycle Management**:
//!    - Create, update, delete collections
//!    - UUID generation and management
//!    - Schema validation and evolution
//!
//! 2. **Storage Coordination**:
//!    - Storage engine selection based on workload
//!    - Multi-disk path assignment
//!    - Collection-to-storage affinity
//!
//! 3. **Metadata Management**:
//!    - Persistent metadata storage
//!    - Configuration caching with DashMap
//!    - Index configuration management
//!
//! 4. **Business Logic**:
//!    - Validation of collection parameters
//!    - Default value resolution
//!    - Compression strategy selection
//!    - Quantization configuration
//!
//! ## Design Principles
//!
//! - **Proto-First**: Uses native protocol buffer types (Collection, CollectionConfig)
//! - **Zero-Copy**: Minimal allocations and translations
//! - **UUID-Based**: All storage paths use UUIDs for uniqueness
//! - **Atomic Operations**: All operations are atomic with proper rollback
//! - **Cache-Friendly**: DashMap for lock-free concurrent access to metadata
//!
//! ## Integration Points
//!
//! - **Upstream**: Called by `UnifiedHandlers` for all collection operations
//! - **Downstream**:
//!   - `CatalogManager` (xCatalog) for collection metadata persistence
//!   - `FilesystemFactory` for storage access
//!
//! ## Performance Optimizations
//!
//! - **Lock-Free Caching**: DashMap eliminates lock contention
//! - **Lazy Loading**: Metadata loaded on-demand
//! - **Batch Operations**: Support for bulk collection operations
//! - **Smart Defaults**: Automatic selection of optimal configurations

use anyhow::{Context, Result};

/// Serializes collection-config read-modify-write (`update_collection`, the
/// owner backfill) so a stamped owner can never be overwritten by a write that
/// read the record before the stamp.
static COLLECTION_CONFIG_RMW: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
use std::collections::HashSet;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

// Using String directly instead of String alias for proto-first architecture
use crate::catalog::CatalogManager;
use crate::core::config::StorageConfig;
use crate::core::stable_id::CollectionIdentity;
use crate::proto::proximadb_v1::{Collection, CollectionConfig, StorageEngine};
use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::trait_components::path_resolver::{
    collection_data_path_typed, collection_index_path_typed, collection_wal_path_typed,
};
use proximadb_storage_common::storage_path::StoragePath;

/// The `(wal, data, indexes)` directories of a collection under its
/// tenant-scoped base: the typed `{base}/accounts/{a}/{n}/{c}/…` layout for a
/// typed identity, else the legacy `{base}/{uuid}/…` — the same helpers flush
/// and compaction resolve their paths with.
fn collection_storage_dirs(
    tenant_base: &str,
    collection_uuid: &str,
    identity: Option<CollectionIdentity>,
) -> (String, String, String) {
    (
        collection_wal_path_typed(tenant_base, collection_uuid, identity),
        collection_data_path_typed(tenant_base, collection_uuid, identity),
        collection_index_path_typed(tenant_base, collection_uuid, identity),
    )
}

/// Every root directory a collection's storage may live under: the legacy
/// `{base}/{uuid}` always and, when a typed identity is recorded, the typed
/// `{base}/accounts/{a}/{n}/{c}` — both regardless of the current
/// `PROXIMADB_TYPED_PATHS`, because a restart can flip which layout flushes
/// write (the residency prefixes register both for the same reason).
fn collection_storage_roots(
    base: &str,
    collection_uuid: &str,
    identity: Option<CollectionIdentity>,
) -> Vec<String> {
    let base = base.trim_end_matches('/');
    let mut roots = vec![format!("{base}/{collection_uuid}")];
    if identity.is_some()
        && let Some(root) = collection_data_path_typed(base, collection_uuid, identity)
            .strip_suffix("/data")
            .map(str::to_string)
    {
        roots.push(root);
    }
    roots
}

// Proto-first architecture - use crate::proto::proximadb_v1::Collection directly

// Local types to replace assignment service
/// Storage component type used for multi-disk path assignment.
#[derive(Debug, Clone)]
enum StorageComponentType {
    /// Write-ahead log component
    Wal,
    /// Data storage component
    Storage,
    /// Index storage component
    Index,
}

/// Collection service for unified business logic with multi-disk coordination
pub struct CollectionService {
    /// Factory for creating filesystem instances per collection path
    filesystem_factory: Arc<FilesystemFactory>,
    /// Cache for IndexConfig to avoid repeated deserialization
    /// Using dashmap for lock-free concurrent access
    index_config_cache: Arc<dashmap::DashMap<String, crate::index::config::RuntimeIndexConfig>>,
    /// Global storage configuration for engine and WAL settings
    storage_config: StorageConfig,
    /// TD-TIER-1 SC3-2: reservation pools opened at startup (None ⇒ inert).
    reservations: Option<Arc<crate::services::collection::ReservationRegistry>>,
    /// xCatalog manager — the sole authoritative store for collection lifecycle
    /// metadata. When wired, collection reads/writes resolve through xCatalog.
    catalog_manager: Option<Arc<CatalogManager>>,
    /// ADR-035 D2 / TD-SC-1b: hot read cache fronting `collection()` so repeated
    /// metadata reads avoid the catalog's 1+N+M object-store round-trips. Built
    /// when the catalog manager is wired; entries are corpus-version-stamped so a
    /// write self-invalidates them. Active only for the single-tenant,
    /// name-keyed read path (see `collection()`).
    syscat_cache: Option<Arc<crate::catalog::syscat_cache::HotSysCatCache>>,
    /// ADR-035 D2 / TD-SC-2b: local directory for the on-disk warm tier. When set
    /// (explicitly, or from `PROXIMADB_SYSCAT_WARM_DIR`), the hot cache reads
    /// through a [`WarmDiskStore`](crate::catalog::syscat_warm::WarmDiskStore)
    /// before the canonical catalog, so a hot miss is served from the OS page
    /// cache instead of object-store round-trips. Must be set before
    /// `with_catalog_manager` (which builds the cache). `None` ⇒ hot → canonical.
    syscat_warm_dir: Option<std::path::PathBuf>,

    /// TD-CAT-2b (S3a): whether collection catalog assets are stored under a
    /// tenant-prefixed namespace (`[tenant, ...levels]`, the convention the
    /// DDL/DML path already uses via `resolve_table_scoped`). Read once from
    /// `PROXIMADB_CATALOG_TENANT_NAMESPACES` at construction (default-OFF);
    /// interior mutability so tests toggle it deterministically without env
    /// races. Off ⇒ today's bare-namespace behavior.
    tenant_namespaces: std::sync::atomic::AtomicBool,

    // NEW: Multi-tenant integration
    /// Optional tenant manager for multi-tenant isolation
    tenant_manager: Option<Arc<crate::storage::tenant::TenantManager>>,
    /// Optional RBAC enforcer for role-based access control
    rbac_enforcer: Option<Arc<crate::storage::tenant::EnhancedRBACManager>>,

    /// Per-collection TurboQuant store registry (Phase P — Quantization
    /// Trait Convergence Plan). When present, `create_collection` with
    /// `enable_turboquant=true` registers the per-collection store
    /// immediately (no first-search latency hit) via
    /// `registry.get_or_create(...)`. When absent (default test paths
    /// + non-TurboQuant deployments), the create-time block falls back
    /// to logging-only behavior so existing fixtures keep working.
    ///
    /// Same `Arc<dyn>` instance lives on `SharedServices.turboquant_registry`
    /// — Phase P's hoist in `SharedServices::new` ensures the create-time
    /// wire and the boot-time hydration share one map.
    #[cfg(feature = "experimental-turboquant")]
    turboquant_registry: Option<
        Arc<dyn crate::compute::quantization::turboquant_store_registry::TurboQuantStoreRegistry>,
    >,
}

impl CollectionService {
    /// Create new collection service with multi-disk coordination
    pub async fn new(storage_config: StorageConfig) -> Result<Self> {
        // Create filesystem factory with proper config from storage_config
        let fs_config = crate::storage::persistence::filesystem::FilesystemConfig {
            default_fs: Some(storage_config.metadata_url.clone()),
            local: None,
            global_options: Default::default(),
            auth_config: None,
            performance_config: Default::default(),
            scheme_mapping: Default::default(),
        };

        let filesystem_factory = Arc::new(
            FilesystemFactory::create(fs_config)
                .await
                .context("Failed to initialize filesystem factory")?,
        );

        // TD-TIER-1 SC3-2: open the reservation pools declared in
        // `[storage.reservation_pools]` (validated at config load). Empty
        // list ⇒ None ⇒ every reservation seam is inert. A pool another
        // process exclusively owns fails construction — an operator wants
        // that at startup, not at first PERFORMANCE admission.
        let reservations =
            crate::services::collection::reservation::ReservationRegistry::open_from_config(
                &storage_config.reservation_pools,
            )
            .await
            .context("TD-TIER-1 SC3: reservation pool startup failed")?;

        let service = Self {
            filesystem_factory,
            index_config_cache: Arc::new(dashmap::DashMap::new()),
            storage_config,
            reservations,
            catalog_manager: None,
            syscat_cache: None,    // Built in `with_catalog_manager`.
            syscat_warm_dir: None, // Set via `with_syscat_warm_dir` / env.
            tenant_namespaces: std::sync::atomic::AtomicBool::new(Self::tenant_namespaces_enabled()),
            tenant_manager: None, // Will be set via with_tenant_manager()
            rbac_enforcer: None,  // Will be set via with_rbac_enforcer()
            #[cfg(feature = "experimental-turboquant")]
            turboquant_registry: None, // Will be set via with_turboquant_registry()
        };
        // TD-TIER-1 SC4-4: the restart backfill runs when the SST engine
        // registers the process store (`sst/core.rs`), not here — grants are
        // restored before the store exists (review S4).
        Ok(service)
    }

    /// TD-TIER-1 SC4-3: report a PERFORMANCE collection's live grant on its
    /// status — reserved bytes from the ledger, resident and covered bytes
    /// from the registered process store. Only the collection GET operation
    /// (REST v2 and gRPC v2) calls this, through
    /// `CollectionPort::enrich_collection_status` — never `get_collection`,
    /// which every search resolves through (review B1). Non-PERFORMANCE
    /// collections and an inert registry return immediately.
    pub async fn overlay_live_residency(&self, collection: &mut Collection) {
        let Some(registry) = &self.reservations else {
            return;
        };
        let Some(config) = collection.config.as_ref() else {
            return;
        };
        if crate::services::collection::storage_class::storage_class_or_balanced(config)
            != crate::proto::proximadb_v1::StorageClass::Performance
        {
            return;
        }
        let Some(grant) = registry.live_grant_for_collection(&collection.id).await else {
            return;
        };
        let (resident, covered) = proximadb_cache::process_store()
            .map(|store| {
                (
                    store.group_resident_bytes(&grant.id),
                    store.covered_bytes(&grant.id),
                )
            })
            .unwrap_or((0, 0));
        // SC4-5: prove coverage of the committed segment set (one LIST per
        // segment root, on every status read — not the query path). A failed
        // proof reports DEGRADED with the cause and keeps the byte figures.
        let proof = match crate::storage::engines::sst::segment_io::prove_residency(
            &self.filesystem_factory,
            &grant.residency_prefixes,
        )
        .await
        {
            Ok(proof) => proof,
            Err(error) => {
                warn!(
                    "TD-TIER-1 SC4: residency proof for {} failed: {error}",
                    collection.id
                );
                let base = collection.storage_class_status.take().unwrap_or_else(|| {
                    crate::services::collection::storage_class::storage_class_status(config)
                });
                collection.storage_class_status = Some(
                    crate::services::collection::storage_class::with_failed_proof(
                        base,
                        grant.bytes,
                        resident,
                        covered,
                        &error.to_string(),
                    ),
                );
                return;
            }
        };
        // Review S4 / S-1: the proven profile is the SST PAX scan — AXIS-routed
        // collections (non-empty index_configs) and other engines (whose
        // readers and object kinds the proof does not count) are outside it.
        let in_profile = crate::services::collection::storage_class::in_ready_profile(config);
        let base = collection.storage_class_status.take().unwrap_or_else(|| {
            crate::services::collection::storage_class::storage_class_status(config)
        });
        collection.storage_class_status = Some(
            crate::services::collection::storage_class::with_residency_proof(
                base,
                grant.bytes,
                resident,
                covered,
                in_profile,
                &proof,
            ),
        );
    }

    /// TD-TIER-1 SC3-2: the shared reservation registry, when pools are
    /// declared. The metering cadence ticks it (release tails + accrual).
    pub fn reservations(
        &self,
    ) -> Option<&Arc<crate::services::collection::reservation::ReservationRegistry>> {
        self.reservations.as_ref()
    }

    /// Attach a TurboQuant store registry (Phase P — Quantization Trait
    /// Convergence Plan). When set, `create_collection` with
    /// `enable_turboquant=true` registers the per-collection store
    /// immediately via `registry.get_or_create(...)`. Mirrors the
    /// `with_catalog_manager` pattern below.
    ///
    /// Production wiring: `SharedServices::new` hoists the registry
    /// construction (Phase P Site 2) and threads the same `Arc<dyn>`
    /// instance through here. Sharing one `Arc` means create-time
    /// registrations land in the same map the boot-time hydration loop
    /// populates.
    #[cfg(feature = "experimental-turboquant")]
    pub fn with_turboquant_registry(
        mut self,
        registry: Arc<
            dyn crate::compute::quantization::turboquant_store_registry::TurboQuantStoreRegistry,
        >,
    ) -> Self {
        self.turboquant_registry = Some(registry);
        self
    }

    /// Attach the shared xCatalog manager.
    ///
    /// During migration, xCatalog is the lifecycle metadata authority when configured while the
    /// legacy metadata backend is kept in sync for storage-engine callers that still read it.
    pub fn with_catalog_manager(mut self, catalog_manager: Arc<CatalogManager>) -> Self {
        // Build the hot read cache over a catalog-manager-only inner source (no
        // `Arc` cycle back to this service). Usage is gated at read time on
        // single-tenant mode + a name (non-UUID) key — see `collection()`.
        let canonical: Arc<dyn crate::catalog::syscat_cache::CatalogMetadataSource> =
            Arc::new(CatalogAssetSource {
                catalog_manager: catalog_manager.clone(),
            });

        // TD-SC-2b: insert the on-disk warm tier between hot and canonical when a
        // cache dir is configured (explicit builder, else `PROXIMADB_SYSCAT_WARM_DIR`).
        // Unset ⇒ hot → canonical, unchanged. The warm tier is corpus-version
        // stamped, so it stays coherent with no write-path coupling.
        let warm_dir = self.syscat_warm_dir.clone().or_else(|| {
            std::env::var("PROXIMADB_SYSCAT_WARM_DIR")
                .ok()
                .filter(|dir| !dir.is_empty())
                .map(std::path::PathBuf::from)
        });
        let inner: Arc<dyn crate::catalog::syscat_cache::CatalogMetadataSource> = match warm_dir {
            Some(dir) => Arc::new(crate::catalog::syscat_warm::WarmDiskStore::new(
                dir, canonical,
            )),
            None => canonical,
        };

        self.syscat_cache = Some(Arc::new(
            crate::catalog::syscat_cache::HotSysCatCache::with_defaults(
                SYSCAT_CACHE_POOL_BYTES,
                inner,
            ),
        ));
        self.catalog_manager = Some(catalog_manager);
        self
    }

    /// ADR-035 D2 / TD-SC-2b: enable the on-disk warm tier rooted at `dir`. Call
    /// **before** [`with_catalog_manager`](Self::with_catalog_manager) (which
    /// builds the cache). Production wires this from `PROXIMADB_SYSCAT_WARM_DIR`;
    /// the explicit builder keeps it testable without mutating process env.
    pub fn with_syscat_warm_dir(mut self, dir: std::path::PathBuf) -> Self {
        self.syscat_warm_dir = Some(dir);
        self
    }

    /// Set tenant manager for multi-tenant support
    pub fn with_tenant_manager(
        mut self,
        tenant_manager: Arc<crate::storage::tenant::TenantManager>,
    ) -> Self {
        self.tenant_manager = Some(tenant_manager);
        self
    }

    /// Set RBAC enforcer for permission validation
    pub fn with_rbac_enforcer(
        mut self,
        rbac_enforcer: Arc<crate::storage::tenant::EnhancedRBACManager>,
    ) -> Self {
        self.rbac_enforcer = Some(rbac_enforcer);
        self
    }

    /// Get storage configuration
    ///
    /// Returns the storage configuration for accessing storage locations.
    /// Used by Arrow Flight service to find .arrow files.
    pub fn storage_config(&self) -> &StorageConfig {
        &self.storage_config
    }

    /// Returns `true` if multi-tenant mode is enabled (a tenant manager is configured).
    pub fn multi_tenant_enabled(&self) -> bool {
        self.tenant_manager.is_some()
    }

    /// Load tenant context for the given tenant ID.
    ///
    /// Returns `Ok(None)` when multi-tenant mode is disabled.
    /// Returns an error if the tenant ID is missing or the tenant is not found.
    pub fn load_tenant_context(
        &self,
        tenant_id: Option<&str>,
    ) -> Result<Option<crate::storage::tenant::TenantContext>> {
        match &self.tenant_manager {
            Some(tenant_manager) => {
                let tenant_id = tenant_id
                    .map(str::trim)
                    .filter(|tenant_id| !tenant_id.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!("Tenant context is required for this operation")
                    })?;
                let tenant_ctx = tenant_manager
                    .get_tenant(tenant_id)
                    .with_context(|| format!("Tenant '{}' not found", tenant_id))?;
                Ok(Some(tenant_ctx))
            }
            None => Ok(None),
        }
    }

    /// Resolve the owning tenant for a collection.
    ///
    /// Thin alias over the canonical [`proximadb_tenant::tenant_id_of`] resolver
    /// (foundation tier) so service-internal callers keep a local name while the
    /// tag/owner precedence stays identical to the storage paths and network
    /// gates that share that one primitive.
    pub(crate) fn collection_tenant_id(collection: &Collection) -> Option<String> {
        proximadb_tenant::tenant_id_of(collection)
    }

    /// TD-CAT-2b (S3a): presence-based, default-OFF gate for storing collection
    /// catalog assets under a tenant-prefixed namespace. Read once at
    /// construction into [`tenant_namespaces`](Self::tenant_namespaces); call
    /// sites consult [`tenant_namespaces_on`](Self::tenant_namespaces_on).
    fn tenant_namespaces_enabled() -> bool {
        std::env::var_os("PROXIMADB_CATALOG_TENANT_NAMESPACES").is_some()
    }

    /// Whether collection assets are written under a tenant-prefixed namespace.
    fn tenant_namespaces_on(&self) -> bool {
        self.tenant_namespaces
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Test-only deterministic toggle for the tenant-namespaces gate, avoiding
    /// the process-global env races a parallel test suite would otherwise hit.
    #[cfg(test)]
    fn set_tenant_namespaces_for_test(&self, on: bool) {
        self.tenant_namespaces
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// TD-CAT-2b (S3a): tenant-scope a collection's catalog identifier by
    /// prepending the tenant as namespace level 0 — the same convention the
    /// DDL/DML path uses via `CatalogManager::resolve_table_scoped`. This keeps
    /// two tenants' identically-named namespaces (`default`, …) structurally
    /// distinct in the single multi-tenant catalog instead of colliding on the
    /// bare `levels.join(".")` key. A no-op when the gate is off or the
    /// collection is not tenant-scoped, so legacy bare assets are unchanged.
    fn tenant_scoped_identifier(
        enabled: bool,
        tenant: Option<&str>,
        identifier: crate::catalog::TableIdentifier,
    ) -> crate::catalog::TableIdentifier {
        match tenant {
            Some(tenant) if enabled && !tenant.is_empty() => {
                let mut namespace = Vec::with_capacity(identifier.namespace.len() + 1);
                namespace.push(tenant.to_string());
                namespace.extend(identifier.namespace.iter().cloned());
                crate::catalog::TableIdentifier::new(namespace, identifier.name)
            }
            _ => identifier,
        }
    }

    /// TD-SC-4 (S4): provision a tenant's bare-minimum system-catalog skeleton at
    /// signup (idempotent). A no-op unless tenant-prefixed namespaces are enabled
    /// (a single-tenant deployment needs no per-tenant skeleton) and a catalog is
    /// configured — so it's safe to call unconditionally from any onboarding
    /// path. Delegates to the catalog-manager-only free function of the same name.
    pub async fn provision_tenant_system_catalog(&self, tenant: &str) -> Result<()> {
        if !self.tenant_namespaces_on() {
            return Ok(());
        }
        let Some(catalog_manager) = &self.catalog_manager else {
            return Ok(());
        };
        provision_tenant_system_catalog(catalog_manager, tenant).await
    }

    /// Default corpus-version bucket for writes without a threaded tenant context
    /// (single-tenant / anonymous deployments). Reads of the same
    /// corpus-version-keyed caches use the same bucket, so invalidation stays
    /// consistent regardless of whether a tenant context is present.
    pub(crate) const DEFAULT_VERSION_TENANT: &str = crate::catalog::DEFAULT_CORPUS_VERSION_TENANT;

    /// The tenant id to key corpus-version (cache-invalidation) bumps by. Falls
    /// back to [`DEFAULT_VERSION_TENANT`](Self::DEFAULT_VERSION_TENANT) so the
    /// bump — and therefore cache invalidation — fires **unconditionally** on
    /// writes. Previously the bump was gated on a tenant context being present,
    /// so in single-tenant/anonymous deployments (the common case today) writes
    /// never bumped the version, leaving corpus-version-keyed caches stale.
    fn version_tenant(tenant_context: Option<&crate::storage::tenant::TenantContext>) -> &str {
        tenant_context
            .map(|ctx| ctx.tenant_id.as_str())
            .filter(|id| !id.is_empty())
            .unwrap_or(Self::DEFAULT_VERSION_TENANT)
    }

    /// Bump the corpus version of `name` for `tenant` and, when that is not the
    /// default bucket, also for the default bucket: the name-keyed collection
    /// cache ([`Self::collection`]) reads the default bucket, so a named
    /// tenant's create or delete must invalidate it too. Returns `tenant`'s
    /// new version.
    async fn bump_corpus_versions(tenant: &str, name: &str) -> u64 {
        let registry = crate::catalog::CorpusVersionRegistry::global();
        let version = registry.bump(tenant, name).await;
        if tenant != Self::DEFAULT_VERSION_TENANT {
            let _ = registry.bump(Self::DEFAULT_VERSION_TENANT, name).await;
        }
        version
    }

    async fn count_tenant_collections(&self, tenant_id: &str) -> Result<usize> {
        Ok(self
            .list_collections()
            .await?
            .into_iter()
            .filter(|collection| {
                Self::collection_tenant_id(collection).as_deref() == Some(tenant_id)
            })
            .count())
    }

    async fn validate_tenant_collection_access(
        &self,
        collection_identifier: &str,
        tenant_ctx: &crate::storage::tenant::TenantContext,
    ) -> Result<Option<Collection>> {
        if let Some(ref tenant_manager) = self.tenant_manager
            && !tenant_manager.is_tenant_active(&tenant_ctx.tenant_id)
        {
            warn!(
                "🚨 Tenant '{}' is not active; denying access to collection '{}'",
                tenant_ctx.tenant_id, collection_identifier
            );
            return Ok(None);
        }

        let collection = self.collection(collection_identifier).await?;

        let Some(collection) = collection else {
            return Ok(None);
        };

        if !proximadb_tenant::collection_visible_to(&collection, Some(&tenant_ctx.tenant_id)) {
            debug!(
                tenant = %tenant_ctx.tenant_id,
                collection = %collection_identifier,
                "collection is not resolvable for this tenant"
            );
            return Ok(None);
        }

        if self.rbac_enforcer.is_some() {
            debug!(
                "RBAC enforcer configured for tenant '{}', but collection service access checks still need user context wiring",
                tenant_ctx.tenant_id
            );
        }

        Ok(Some(collection))
    }

    /// Create collection - single method for all handlers (REST, gRPC, etc)
    /// Takes native types directly, no proto/avro conversions needed
    /// NOW WITH MULTI-TENANT SUPPORT
    pub async fn create_collection(
        &self,
        config: &crate::proto::proximadb_v1::CollectionConfig,
    ) -> Result<CollectionServiceResponse> {
        self.create_collection_with_tenant_context(config, None)
            .await
    }

    /// Resolve a collection for a request acting as `request_tenant`.
    ///
    /// `Some(tenant)` (a request; blank = the default tenant) returns the
    /// collection only when that tenant may resolve it under the one ownership
    /// rule (`proximadb_tenant::collection_visible_to`); otherwise `None`,
    /// exactly as for a collection that does not exist. `None` (no tenant)
    /// denies; internal lookups use [`Self::collection_internal`].
    pub async fn collection_for_tenant(
        &self,
        identifier: &str,
        request_tenant: Option<&str>,
    ) -> Result<Option<Collection>> {
        // No request tenant ⇒ deny. Internal lookups use `collection_internal`.
        let Some(tenant) = request_tenant else {
            return Ok(None);
        };
        let ctx = self.request_tenant_context(Some(tenant))?;
        self.get_collection_with_tenant_context(identifier, ctx.as_ref())
            .await
    }

    /// The tenant context a request acting as `request_tenant` resolves
    /// collections in: the managed context when a TenantManager is wired,
    /// else a context for the tenant itself. `None` tenant ⇒ `None` (deny).
    pub fn request_tenant_context(
        &self,
        request_tenant: Option<&str>,
    ) -> Result<Option<crate::storage::tenant::TenantContext>> {
        let Some(tenant) = request_tenant else {
            return Ok(None);
        };
        Ok(Some(
            self.load_tenant_context(Some(tenant))?.unwrap_or_else(|| {
                crate::storage::tenant::TenantContext::for_tenant_id(
                    proximadb_tenant::resolve_request_tenant(Some(tenant)),
                )
            }),
        ))
    }

    /// Record `tenant` as the owner of a collection that has no recorded owner,
    /// through the same catalog write as `update_collection`. Never re-assigns
    /// a collection that already records an owner (or conflicting owners).
    /// Returns whether an owner was written.
    pub async fn assign_owner_if_unattributed(
        &self,
        identifier: &str,
        tenant: &str,
    ) -> Result<bool> {
        // Serialized with `update_collection` so the read and the write see
        // the same record (no lost owner, no lost update).
        let _rmw = COLLECTION_CONFIG_RMW.lock().await;
        // Under tenant-prefixed catalog namespaces (TD-CAT-2b gate) an owner
        // change would move the catalog asset to another namespace; leave such
        // collections to the default tenant (fail closed) rather than rewrite
        // the asset key here.
        if self.tenant_namespaces_on() {
            return Ok(false);
        }
        let Some(mut record) = self.collection(identifier).await? else {
            return Ok(false);
        };
        if proximadb_tenant::collection_owner(&record)
            != proximadb_tenant::CollectionOwner::Unattributed
        {
            return Ok(false);
        }
        let Some(config) = record.config.as_mut() else {
            return Ok(false);
        };
        proximadb_tenant::stamp_collection_owner(config, tenant);
        let name = config.name.clone();
        if !self
            .write_owner_metadata_in_place(&record)
            .await
            .context("Failed to record collection owner")?
        {
            return Ok(false);
        }
        let _ = crate::catalog::CorpusVersionRegistry::global()
            .bump(Self::DEFAULT_VERSION_TENANT, &name)
            .await;
        Ok(true)
    }

    /// Write `record`'s owner metadata (the `owner` and `tags` properties) onto
    /// its existing catalog entry and change nothing else: columns, layouts,
    /// identity and every other property stay as they are. Only an entry that
    /// records exactly this collection is touched. Returns whether it wrote.
    async fn write_owner_metadata_in_place(&self, record: &Collection) -> Result<bool> {
        let (Some(catalog_manager), Some(config)) = (&self.catalog_manager, record.config.as_ref())
        else {
            return Ok(false);
        };
        let catalog = catalog_manager.default_catalog().await?;
        let identifier = Self::tenant_scoped_identifier(
            self.tenant_namespaces_on(),
            Self::collection_tenant_id(record).as_deref(),
            crate::storage::metadata::collection_mapping::collection_table_identifier(config),
        );
        if !catalog.table_exists(&identifier).await? {
            return Ok(false);
        }
        let mut schema = catalog.get_table(&identifier).await?;
        if schema.properties.get("collection.id") != Some(&record.id) {
            return Ok(false);
        }
        match &config.owner {
            Some(owner) => {
                schema.properties.insert("owner".to_string(), owner.clone());
            }
            None => {
                schema.properties.remove("owner");
            }
        }
        if config.tags.is_empty() {
            schema.properties.remove("tags");
        } else {
            schema
                .properties
                .insert("tags".to_string(), config.tags.join(","));
        }
        // The full config is also recorded as JSON (the reader restores it from
        // there); carry the same two fields into it and nothing else.
        if let Some(json) = schema.properties.get("collection.config_json")
            && let Ok(mut full) =
                serde_json::from_str::<crate::proto::proximadb_v1::CollectionConfig>(json)
        {
            full.owner = config.owner.clone();
            full.tags = config.tags.clone();
            schema.properties.insert(
                "collection.config_json".to_string(),
                serde_json::to_string(&full).context("serializing collection config")?,
            );
        }
        // TD-CATREPLACE-1: replace in place — no drop window (this path used
        // the drop+create pattern this TD retires).
        catalog.replace_table(&identifier, schema).await?;
        Ok(true)
    }

    /// Record that the owner backfill could not attribute a collection (it stays
    /// owned by the default tenant): adds `marker` to its tags unless it
    /// meanwhile gained an owner. Same catalog write and lock as
    /// [`Self::assign_owner_if_unattributed`].
    pub async fn mark_owner_unattributable(&self, identifier: &str, marker: &str) -> Result<()> {
        let _rmw = COLLECTION_CONFIG_RMW.lock().await;
        let Some(mut record) = self.collection(identifier).await? else {
            return Ok(());
        };
        if proximadb_tenant::collection_owner(&record)
            != proximadb_tenant::CollectionOwner::Unattributed
        {
            return Ok(());
        }
        let Some(config) = record.config.as_mut() else {
            return Ok(());
        };
        if config.tags.iter().any(|tag| tag == marker) {
            return Ok(());
        }
        config.tags.push(marker.to_string());
        let name = config.name.clone();
        self.write_owner_metadata_in_place(&record)
            .await
            .context("Failed to record collection owner backfill outcome")?;
        let _ = crate::catalog::CorpusVersionRegistry::global()
            .bump(Self::DEFAULT_VERSION_TENANT, &name)
            .await;
        Ok(())
    }

    /// Get collection with tenant validation
    pub async fn get_collection_with_tenant_context(
        &self,
        collection_name: &str,
        tenant_context: Option<&crate::storage::tenant::TenantContext>,
    ) -> Result<Option<crate::proto::proximadb_v1::Collection>> {
        // No tenant context ⇒ deny. Internal (system) lookups use
        // `collection_internal`.
        let Some(tenant_ctx) = tenant_context else {
            return Ok(None);
        };
        if self.tenant_manager.is_none() {
            return Ok(self
                .collection(collection_name)
                .await?
                .filter(|collection| {
                    proximadb_tenant::collection_visible_to(collection, Some(&tenant_ctx.tenant_id))
                }));
        }
        let collection = self
            .validate_tenant_collection_access(collection_name, tenant_ctx)
            .await?;
        if collection.is_some() {
            debug!(
                "✅ Tenant ownership validation passed for collection access: tenant={}, collection={}",
                tenant_ctx.tenant_id, collection_name
            );
        }
        Ok(collection)
    }

    /// Internal (system) lookup by name or id, not scoped to any tenant: for
    /// background flush, compaction, recovery and the owner backfill. Request
    /// paths resolve through `collection_for_tenant`.
    pub async fn collection_internal(&self, identifier: &str) -> Result<Option<Collection>> {
        self.collection(identifier).await
    }

    /// List all collections, filtered to the given tenant context if multi-tenant mode is active.
    pub async fn list_collections_with_tenant_context(
        &self,
        tenant_context: Option<&crate::storage::tenant::TenantContext>,
    ) -> Result<Vec<Collection>> {
        let collections = self.list_collections().await?;

        // No tenant context ⇒ deny (an empty listing).
        let Some(tenant_ctx) = tenant_context else {
            return Ok(Vec::new());
        };
        if self.tenant_manager.is_none() {
            return Ok(collections
                .into_iter()
                .filter(|collection| {
                    proximadb_tenant::collection_visible_to(collection, Some(&tenant_ctx.tenant_id))
                })
                .collect());
        }
        {
            if let Some(ref tenant_manager) = self.tenant_manager
                && !tenant_manager.is_tenant_active(&tenant_ctx.tenant_id)
            {
                warn!(
                    "🚨 Tenant '{}' is not active; returning empty collection list",
                    tenant_ctx.tenant_id
                );
                return Ok(Vec::new());
            }

            Ok(collections
                .into_iter()
                .filter(|collection| {
                    proximadb_tenant::collection_visible_to(collection, Some(&tenant_ctx.tenant_id))
                })
                .collect())
        }
    }

    /// Delete collection with tenant validation
    pub async fn delete_collection_with_tenant_context(
        &self,
        collection_name: &str,
        tenant_context: Option<&crate::storage::tenant::TenantContext>,
    ) -> Result<CollectionServiceResponse> {
        debug!("🗑️ Deleting collection: {}", collection_name);

        // Resolve within the request tenant (no context ⇒ deny). A collection
        // this tenant may not resolve is answered exactly like a missing one,
        // and nothing is touched.
        let Some(target) = self
            .get_collection_with_tenant_context(collection_name, tenant_context)
            .await?
        else {
            return Ok(CollectionServiceResponse {
                success: false,
                collection: None,
                storage_path: None,
                error_code: Some("COLLECTION_NOT_FOUND".to_string()),
                processing_time_us: 0,
            });
        };

        let response = self.delete_collection(collection_name).await?;
        // TD-TIER-1 SC3-2: the inner delete is the authoritative release
        // (keyed by the record it removed). This belt-and-braces release keys
        // the REQUEST tenant against the outer-resolved target — idempotent
        // where the inner path already released, and it covers an owner field
        // that disagrees with the request tenant.
        if response.success
            && let Some(registry) = &self.reservations
        {
            registry
                .release_for_collection(
                    Self::version_tenant(tenant_context),
                    &target.id,
                    chrono::Utc::now().timestamp(),
                )
                .await;
        }
        // The name-keyed cache is also keyed by id and by the name the caller
        // used; invalidate every key the deleted collection can be read by.
        if response.success {
            let tenant = Self::version_tenant(tenant_context);
            let mut keys = vec![target.id.clone()];
            if let Some(config) = &target.config {
                keys.push(config.name.clone());
            }
            for key in keys.iter().filter(|key| key.as_str() != collection_name) {
                let _ = Self::bump_corpus_versions(tenant, key).await;
            }
        }

        // Bump the corpus_version for (tenant, collection) so corpus-version-keyed
        // caches (PlanCache, the per-tenant system-catalog cache) invalidate on
        // the next lookup. A delete definitionally invalidates any cached
        // metadata/plans. Bumped **unconditionally** via the effective tenant —
        // anonymous/single-tenant deletes (no threaded context) must invalidate
        // too, keyed by the default bucket reads use.
        if response.success {
            let tenant = Self::version_tenant(tenant_context);
            let version = Self::bump_corpus_versions(tenant, collection_name).await;
            debug!(
                "🔄 corpus_version bumped after delete: tenant={} collection={} version={}",
                tenant, collection_name, version
            );
        }

        Ok(response)
    }

    /// Idempotent get-or-create by name for the document canonical-vector route (ADR-055
    /// P-Provision): build a minimal `CollectionConfig` and create it, treating an already-existing
    /// collection as success. `dimension == 0` ⇒ a vectorless (pure-document) collection. The v1
    /// `CollectionConfig`/`StorageEngine` construction lives HERE (where those types are already in
    /// scope) so callers (e.g. `RecordOpsService::ensure_collection`) stay v1-proto-free (TD-123).
    pub async fn get_or_create_by_name(
        &self,
        name: &str,
        dimension: u32,
        tenant_context: Option<&crate::storage::tenant::TenantContext>,
        request_tenant: Option<&str>,
        promote_keys: &[String],
    ) -> Result<()> {
        // P-Shred follow-up (ADR-055): declared hot keys become filterable columns (typed, id >= 100
        // in the catalog schema), which `catalog_schema_from_collection` then registers as
        // props-auto-promotion `promoted_keys` for document collections — so those props shred into
        // typed user-columns at flush. Default type Text/String; the shred writer coerces per value.
        let filterable_columns = promote_keys
            .iter()
            .filter(|k| !k.is_empty())
            .map(|k| crate::proto::proximadb_v1::FilterableColumnSpec {
                name: k.clone(),
                indexed: true,
                supports_range: true,
                ..Default::default()
            })
            .collect();
        let mut config = CollectionConfig {
            name: name.to_string(),
            dimension,
            storage_engine: Some(StorageEngine::Sst as i32),
            enable_proxima_record: Some(true),
            filterable_columns,
            ..Default::default()
        };
        // Record the requesting tenant as owner.
        if let Some(tenant) = request_tenant {
            proximadb_tenant::stamp_collection_owner(&mut config, tenant);
        }
        let resp = self
            .create_collection_with_tenant_context(&config, tenant_context)
            .await?;
        if resp.success || resp.error_code.as_deref() == Some("COLLECTION_EXISTS") {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "get_or_create_by_name '{}' failed: {:?}",
                name,
                resp.error_code
            ))
        }
    }

    /// Create collection with tenant context validation
    pub async fn create_collection_with_tenant_context(
        &self,
        config: &crate::proto::proximadb_v1::CollectionConfig,
        tenant_context: Option<&crate::storage::tenant::TenantContext>,
    ) -> Result<CollectionServiceResponse> {
        debug!(
            "🆕 Creating collection: {} with distance_metric={:?}",
            config.name, config.distance_metric
        );
        let start_time = std::time::Instant::now();

        if let Some(ref tenant_manager) = self.tenant_manager {
            let Some(tenant_ctx) = tenant_context else {
                return Ok(CollectionServiceResponse::error(
                    "TENANT_CONTEXT_REQUIRED: tenant context is required when multi-tenant mode is enabled".to_string(),
                    start_time.elapsed().as_micros() as i64,
                ));
            };

            if !tenant_manager.is_tenant_active(&tenant_ctx.tenant_id) {
                return Ok(CollectionServiceResponse::error(
                    format!(
                        "TENANT_INACTIVE: tenant {} is not active",
                        tenant_ctx.tenant_id
                    ),
                    start_time.elapsed().as_micros() as i64,
                ));
            }

            let tenant_collection_count =
                self.count_tenant_collections(&tenant_ctx.tenant_id).await?;
            let tenant_limit = tenant_ctx.resource_limits.max_collections as usize;

            if tenant_collection_count >= tenant_limit {
                return Ok(CollectionServiceResponse::error(
                    format!(
                        "TENANT_COLLECTION_LIMIT_EXCEEDED: tenant {} has reached its collection limit ({})",
                        tenant_ctx.tenant_id, tenant_limit
                    ),
                    start_time.elapsed().as_micros() as i64,
                ));
            }

            if self.rbac_enforcer.is_some() {
                debug!(
                    "RBAC enforcer configured for tenant '{}', but collection creation checks still need user context wiring",
                    tenant_ctx.tenant_id
                );
            }

            debug!(
                "✅ Tenant validation passed for collection creation: tenant={}, existing_collections={}, limit={}",
                tenant_ctx.tenant_id, tenant_collection_count, tenant_limit
            );
        }

        let mut enriched_config = config.clone();

        // TD-TIER-1 SC1: validate the desired storage class at admission on
        // every facade path (REST strings are pre-parsed by the facade; gRPC /
        // Flight / SQL reach here with the raw enum number, which prost
        // preserves intact). Unknown ⇒ error; absent/UNSPECIFIED ⇒ balanced,
        // persisted explicitly below so reads echo one vocabulary.
        let resolved_storage_class =
            crate::services::collection::storage_class::validate_storage_class_raw(
                enriched_config.storage_class,
            )?;

        // Persist an explicit default metric so every downstream subsystem
        // sees the same collection semantics instead of applying its own fallback.
        let resolved_distance_metric = enriched_config
            .distance_metric
            .and_then(|metric| crate::proto::proximadb_v1::DistanceMetric::try_from(metric).ok())
            .filter(|metric| *metric != crate::proto::proximadb_v1::DistanceMetric::Unspecified)
            .unwrap_or(crate::proto::proximadb_v1::DistanceMetric::Cosine);
        enriched_config.distance_metric = Some(resolved_distance_metric as i32);

        // Heuristic engine routing: collections that don't pin a
        // storage_engine fall through to the rules in
        // crate::services::collection::engine_selector. Vector
        // collections with neither an index nor quantization land on
        // HELIX (Hilbert-sorted blocks → usable recall without an
        // external index); everything else stays on SST. Caller-pinned
        // engine choices are passed through untouched.
        let (selected_engine, selection_reason) =
            crate::services::collection::engine_selector::infer_storage_engine(&enriched_config);
        // TD-VIPER-1 S2 (ADR-093): reject NEW VIPER selections at creation.
        // Existing VIPER collections remain readable via the context-free
        // factory; only new creations are refused (staged retirement S1 → S2).
        if selected_engine == crate::proto::proximadb_v1::StorageEngine::Viper {
            return Err(anyhow::anyhow!(
                "VIPER is deprecated (ADR-093) and can no longer be selected for new \
                 collections: use SST (default), NOVA (columnar analytics), HELIX, or the \
                 DataFusion/Parquet path for warehouse-shaped analytics"
            ));
        }
        // Fail loud at creation, not silently at first read/flush: MMAP/Hybrid
        // (unimplemented) and any experimental engine selected without its
        // cargo feature would otherwise report `success: true` here and only
        // surface as a background flush warning or a first-read error.
        // `create_from_proto_async` is the same stateless, cheap constructor
        // the read path (`resolver.rs`) calls on every first access; discard
        // the instance, we only need its constructability as a precondition.
        crate::storage::engines::factory::StorageFormatFactory::create_from_proto_async(
            selected_engine,
        )
        .await
        .map_err(|error| {
            anyhow::anyhow!("storage engine {selected_engine:?} is unavailable: {error:#}")
        })?;
        let previous_engine_field = enriched_config.storage_engine;
        enriched_config.storage_engine = Some(selected_engine as i32);
        // Persist the RESOLVED class (absent/UNSPECIFIED ⇒ BALANCED) so every
        // read path and the catalog round-trip see one vocabulary; pre-SC1
        // records keep their absent field and resolve balanced on read.
        enriched_config.storage_class = Some(resolved_storage_class as i32);
        tracing::info!(
            target: "collection.engine_selector",
            collection = %enriched_config.name,
            chosen_engine = ?selected_engine,
            reason = selection_reason,
            previous_field = ?previous_engine_field,
            dimension = enriched_config.dimension,
            has_index = !enriched_config.index_configs.is_empty(),
            has_quantization = enriched_config
                .quantization
                .as_ref()
                .and_then(|q| q.enabled)
                .unwrap_or(false),
            "auto-selected storage engine"
        );

        // Recall-target advisor wiring: when the caller asked for a
        // specific recall (via a `recall_target:<f32>` tag), invoke
        // the algorithm-agnostic advisor. The selector picks HNSW
        // vs IVF based on declared budgets (max_memory_mb,
        // max_query_latency_ms) and sizes the chosen algorithm's
        // params; results are stamped into the matching IndexConfig.
        // See crate::services::collection::recall_target for the
        // parse + apply contract.
        if let Some(recall_target) =
            crate::services::collection::recall_target::parse_recall_target(&enriched_config)
        {
            // The advisor sizes AXIS HNSW/IVF params — skipped in a PAX-exact-scan
            // (`axis` off) build; the recall_target tag is still parsed above so the
            // rest of collection-create is unaffected.
            #[cfg(feature = "axis")]
            {
                let applied = crate::services::collection::recall_target::apply_advisor_to_indexes(
                    &mut enriched_config,
                    recall_target,
                );
                for advice in &applied {
                    tracing::info!(
                        target: "collection.recall_target",
                        collection = %enriched_config.name,
                        index = %advice.index_name,
                        recall_target = recall_target,
                        algorithm = %advice.output.kind.label(),
                        clamped_by_budget = advice.output.clamped_by_budget,
                        projected_recall = ?advice.output.projected_recall,
                        estimated_memory_mb = advice.output.estimated_memory_mb,
                        estimated_per_query_work = advice.output.estimated_per_query_work,
                        rationale = %advice.output.rationale,
                        "auto-sized index from recall_target"
                    );
                }
            }
            #[cfg(not(feature = "axis"))]
            {
                let _ = recall_target;
            }
        }

        // Resolve compression and storage configuration

        // Record the owning tenant on the collection (the one ownership rule,
        // `proximadb_tenant::ownership`). The port-level create stamps the
        // request tenant even without a tenant context; this covers callers
        // that pass a context.
        if let Some(tenant_ctx) = tenant_context {
            proximadb_tenant::stamp_collection_owner(&mut enriched_config, &tenant_ctx.tenant_id);
            enriched_config
                .tags
                .push(format!("created_at:{}", chrono::Utc::now().to_rfc3339()));
            debug!(
                "✅ Added tenant metadata to collection: tenant_id={}",
                tenant_ctx.tenant_id
            );
        }

        // Ensure storage_config exists and set compression within it
        if enriched_config.storage_config.is_none() {
            enriched_config.storage_config = Some(crate::proto::proximadb_v1::StorageConfig {
                ..Default::default()
            });
        }

        // Resolve compression within storage_config
        if let Some(ref mut storage_cfg) = enriched_config.storage_config {
            let resolved_compression = self.resolve_compression_config(
                None, // No existing compression config to resolve from
                enriched_config
                    .storage_engine
                    .unwrap_or(StorageEngine::Sst as i32),
            );
            if let Some(compression_config) = resolved_compression {
                storage_cfg.compression = Some(compression_config.algorithm);
            }
        }

        // Add default quantization configuration if not provided
        // Use smart defaults based on vector dimension for optimal performance
        if enriched_config.quantization.is_none() {
            use crate::compute::quantization::QuantizationSmartDefaults;

            match QuantizationSmartDefaults::generate_for_dimension(config.dimension as usize) {
                Ok(smart_config) => {
                    enriched_config.quantization = Some(smart_config);
                    info!(
                        "🧠 Generated smart quantization defaults for collection '{}' (dimension: {})",
                        config.name, config.dimension
                    );
                }
                Err(e) => {
                    warn!(
                        "⚠️ Failed to generate smart defaults, using fallback: {}",
                        e
                    );
                    // Fallback to simple default
                    enriched_config.quantization = Some(crate::proto::proximadb_v1::QuantizationConfig {
                        enabled: Some(true),
                        strategy: Some(crate::proto::proximadb_v1::quantization_config::Strategy::SmartDefaults as i32),
                        custom_levels: vec![],
                        enable_progressive_search: Some(true),
                        binary_filter_selectivity: Some(0.3),
                        int8_ranking_selectivity: Some(0.1),
                        pq_ranking_selectivity: Some(0.05),
                        training_sample_size: Some(10000),
                        quality_threshold: Some(0.95),
                        enable_adaptive_training: Some(true),
                        optimize_for_storage: Some(false),
                        optimize_for_memory: Some(false),
                        enable_simd_acceleration: Some(true),
                        // NEW: Direct quantization type enables
                        enable_binary: Some(true),
                        enable_int8: Some(true),
                        enable_pq: Some(true),
                        // Product Quantization specific settings
                        pq_segments: Some(8),
                        pq_bits: Some(8),
                        pq_codebooks: Some(0),
                        // Thresholds for progressive search
                        binary_threshold: Some(0.3),
                        int8_threshold: Some(0.1),
                        pq_threshold: Some(0.05),
                        enable_turboquant: Some(false),
                    });
                }
            }
        }

        // Phase N (opt-in plumbing) + Phase P (create-time register) —
        // Quantization Trait Convergence Plan. When the SDK / handler
        // sets `quantization.enable_turboquant = true`:
        //
        // 1. On `cfg(experimental-turboquant)` builds with a registry
        //    attached: call `registry.get_or_create(...)` so the
        //    per-collection TurboQuant store is registered NOW. The first
        //    search reaches the kernel instead of a silent full-precision
        //    fallback. Failures are logged but DO NOT abort collection
        //    creation (Phase O "log + continue" pattern — registry
        //    transient errors must not block the catalog write).
        // 2. On `cfg(experimental-turboquant)` builds without a registry
        //    (test fixtures + paths constructed via `CollectionService::new`
        //    without the builder): emit the Phase N structured event so
        //    operator dashboards still see the intent.
        // 3. On builds without the feature: emit a `warn!` so silent
        //    drops never go unnoticed.
        //
        // Defaults surfaced (per ADR-021 §"Authority mode"):
        //   - `bit_width = 4`
        //   - `calibration_mode = tq_plus`
        //   - `rotation_seed = derive_rotation_seed(&collection_name)` —
        //     same FNV-1a hash every other Phase-A→O surface uses, so
        //     the runtime store, the EXPLAIN payload, and any future
        //     catalog row all agree on the per-collection seed.
        let opt_in = enriched_config
            .quantization
            .as_ref()
            .and_then(|q| q.enable_turboquant)
            .unwrap_or(false);
        if opt_in {
            #[cfg(feature = "experimental-turboquant")]
            {
                use proximadb_quantization_types::CalibrationMode;
                let seed = proximadb_quantization_types::derive_rotation_seed(&config.name);
                let bit_width: u8 = 4;
                if let Some(registry) = &self.turboquant_registry {
                    match registry
                        .get_or_create(
                            &config.name,
                            config.dimension as usize,
                            bit_width,
                            CalibrationMode::TqPlus,
                            seed,
                        )
                        .await
                    {
                        Ok(_store) => {
                            tracing::info!(
                                target: "proximadb::turboquant::opt_in",
                                collection = %config.name,
                                bit_width,
                                calibration_mode = "tq_plus",
                                rotation_seed = format!("{:#x}", seed),
                                "Phase P opt-in: TurboQuant store registered for new collection",
                            );
                        }
                        Err(e) => {
                            // Log + continue. Collection-create must NOT
                            // fail just because the registry hit an
                            // error — boot-time hydration recovers on
                            // next restart, and the next search retries
                            // `get_or_create` lazily.
                            tracing::warn!(
                                target: "proximadb::turboquant::opt_in",
                                collection = %config.name,
                                error = %e,
                                "Phase P opt-in: get_or_create failed; collection will fall \
                                 back to full-precision scoring until next boot",
                            );
                        }
                    }
                } else {
                    // Registry not attached (test path). Keep the Phase
                    // N logging-only behavior so existing fixtures don't
                    // break — the operator-visible intent still surfaces.
                    tracing::info!(
                        target: "proximadb::turboquant::opt_in",
                        collection = %config.name,
                        bit_width,
                        calibration_mode = "tq_plus",
                        rotation_seed = format!("{:#x}", seed),
                        "Phase N opt-in (no registry attached): TurboQuant registered for collection",
                    );
                }
            }
            #[cfg(not(feature = "experimental-turboquant"))]
            {
                tracing::warn!(
                    collection = %config.name,
                    "Collection requested enable_turboquant=true but the server build \
                     does not have the `experimental-turboquant` feature enabled; \
                     opt-in is silently dropped",
                );
            }
        }

        if enriched_config.index_configs.is_empty() {
            info!(
                "📊 Collection '{}' created without an ANN index; exact/brute-force retrieval remains the default until indexes are explicitly configured",
                config.name
            );
        }

        // Validate compression algorithm is supported by the storage engine
        // SDK defines compression config in collection metadata and it drives datablock compression
        if let Some(ref storage_cfg) = enriched_config.storage_config {
            // storage_cfg.compression is i32 in proto v1, check if it's set
            if storage_cfg.compression.unwrap_or(0) != 0 {
                use crate::proto::proximadb_v1::CompressionAlgorithm;
                use crate::storage::engine_capabilities::EngineCapabilities;

                // Convert engine type to enum
                let engine = EngineCapabilities::engine_from_int(
                    config.storage_engine.unwrap_or(StorageEngine::Sst as i32),
                );

                // Try to convert compression algorithm from i32
                if let Ok(algorithm) =
                    CompressionAlgorithm::try_from(storage_cfg.compression.unwrap_or(0))
                {
                    if !EngineCapabilities::is_compression_supported(engine, algorithm) {
                        let engine_name = EngineCapabilities::get_engine_name(engine);
                        let unsupported =
                            EngineCapabilities::get_unsupported_compression_algorithms(engine);
                        return Ok(CollectionServiceResponse::error(
                            format!(
                                "UNSUPPORTED_COMPRESSION: Compression algorithm {:?} is not supported by {} engine. Unsupported algorithms: {:?}",
                                algorithm, engine_name, unsupported
                            ),
                            start_time.elapsed().as_micros() as i64,
                        ));
                    }
                } else {
                    return Ok(CollectionServiceResponse::error(
                        format!(
                            "INVALID_COMPRESSION: Invalid compression algorithm: {:?}",
                            storage_cfg.compression
                        ),
                        start_time.elapsed().as_micros() as i64,
                    ));
                }
            }
        }

        // Input validation
        if config.name.is_empty() {
            return Ok(CollectionServiceResponse::error(
                "INVALID_NAME: Collection name cannot be empty".to_string(),
                start_time.elapsed().as_micros() as i64,
            ));
        }

        // No artificial minimum name length. SQL/ANSI identifiers — and hence
        // relational tables created over pgwire (e.g. TPC-H `part`, `orders`,
        // `region`) — are routinely short. Name shape is validated elsewhere
        // (CollectionNameValidator: non-empty, valid pattern, not reserved); a
        // length floor here only blocked legitimate short table names.

        if config.dimension == 0 || config.dimension > 1_000_000 {
            return Ok(CollectionServiceResponse::error(
                "INVALID_DIMENSION: Invalid dimension: must be between 1 and 1,000,000".to_string(),
                start_time.elapsed().as_micros() as i64,
            ));
        }

        // Validate quantization configuration
        if let Some(quant_config) = &enriched_config.quantization
            && quant_config.enabled.unwrap_or(false)
        {
            info!(
                "⚠️ Collection '{}' has quantization enabled. All vectors MUST have unique IDs for tracking quantized representations",
                config.name
            );
            // Note: We don't fail here, but log a warning. The actual validation happens during insert
            // This allows collections to be created with quantization enabled, but enforces IDs at insert time
        }

        if self.collection(&config.name).await?.is_some() {
            return Ok(CollectionServiceResponse {
                success: false,
                collection: None,
                storage_path: None,
                error_code: Some("COLLECTION_EXISTS".to_string()),
                processing_time_us: start_time.elapsed().as_micros() as i64,
            });
        }

        // Create proto collection directly - no Avro conversion needed!
        // Collection IDs are UUIDs. Legacy base62/time IDs remain resolvable from older metadata,
        // but new catalog assets use opaque UUID identity and keep names as stable logical aliases.
        let uuid = self.generate_unique_collection_id().await?;
        let now = chrono::Utc::now().timestamp_micros();

        // Get storage location - use provided or pick randomly from config
        let base_location = if let Some(ref storage_config) = enriched_config.storage_config {
            if storage_config
                .storage_path
                .as_ref()
                .is_some_and(|p| !p.is_empty())
            {
                // User provided storage location
                storage_config.storage_path.clone().unwrap_or_default()
            } else {
                // Pick randomly from configured locations
                use rand::seq::SliceRandom;
                self.storage_config
                    .storage_locations
                    .choose(&mut rand::thread_rng())
                    .ok_or_else(|| anyhow::anyhow!("No storage locations configured"))?
                    .url
                    .clone()
            }
        } else {
            // Pick randomly from configured locations
            use rand::seq::SliceRandom;
            self.storage_config
                .storage_locations
                .choose(&mut rand::thread_rng())
                .ok_or_else(|| anyhow::anyhow!("No storage locations configured"))?
                .url
                .clone()
        };

        // TD-TIER-1 SC2: admission runs on the final RESOLVED placement — the
        // client's explicit storage_path, or the operator-CONFIGURED location
        // chosen above — plus the resolved engine and the requesting tenant,
        // and BEFORE any side effects (identity minting, directory creation).
        // A configured remote default is admissible; another tenant's explicit
        // prefix is not; anything not object-store-backed on the PAX path is
        // rejected with an actionable message, never silently degraded.
        let admission_tenant_id = tenant_context.map(|ctx| ctx.tenant_id.as_str());
        crate::services::collection::storage_class::check_performance_admission(
            &enriched_config,
            selected_engine,
            &base_location,
            admission_tenant_id,
        )?;
        // TD-TIER-1 SC5: an explicit access tier is validated and needs a
        // tiered object store on the same resolved placement.
        crate::services::collection::storage_class::check_access_tier_admission(
            &enriched_config,
            &base_location,
        )?;

        // ADR-031 Phase 4c: pre-mint the typed collection identity
        // `(account, namespace, collection)` BEFORE storage-dir creation, so the
        // DATA/WAL/index dirs are created at the typed account-rooted path
        // (`accounts/{base62}/…`) when `PROXIMADB_TYPED_PATHS=1` AND an account
        // is known. The same triple is threaded to `upsert_collection_catalog_asset`
        // so the persisted schema's `stable_namespace_id`/`stable_collection_id`
        // match the path (create_table→`mint_stable_identity` then preserves them
        // — idempotent, no double-mint). `None` when env OFF or no account → every
        // path is byte-identical legacy (mixed-read-safe per-collection).
        //
        // The account string is derived from the tenant context inside
        // `mint_typed_identity_for_collection` (Phase 4 collapses tenant into
        // account; see the helper for the rationale + the Phase 5 forward-note).
        // ADR-0083 rev2 D3: the composite is ALWAYS minted (not env-gated) —
        // it is the sole collection identity. The PATH layout stays env-gated
        // (typed_paths_enabled), but the composite on StorageAssignment is
        // always Some so admission/flush/compaction can key on it.
        let typed_identity = self
            .mint_typed_identity_for_collection(tenant_context, &enriched_config.name, &uuid)
            .await;

        // TD-TIER-1 SC3-2: a PERFORMANCE create acquires its reservation from
        // the default pool before any collection-KEYED durable side effect
        // (directories, catalog upsert): a refused admission leaves the
        // request rejected with the old policy unchanged — the pool ledger is
        // the durable record of what was granted, keyed (tenant, collection
        // uuid). The key tenant is the same default-folding `version_tenant`
        // uses (never the empty string — an empty tenant id is rejected by the
        // metering path resolver, so it must never reach a grant). Kill-switch
        // and pool misconfiguration surface as actionable errors here, never a
        // silent zero-byte grant. Non-PERFORMANCE classes and an empty pool
        // list skip this seam entirely. If a later create step fails, the
        // compensation below releases the grant: a leaked grant would charge
        // capacity and accrue billing for a collection that never existed.
        // The tenant-scoped base every storage consumer reads: the storage
        // assignment records it, flush/compaction publish under it, the grant
        // protects it — and the create-time directories are built from it.
        let tenant_id = tenant_context.map(|ctx| ctx.tenant_id.as_str());
        let tenant_base_location = match tenant_id {
            Some(tid) => StoragePath::tenant_root_path(&base_location, tid),
            None => base_location.clone(),
        };

        let mut reserved_grant =
            crate::services::collection::reservation::GrantCompensation::none();
        if crate::services::collection::storage_class::storage_class_or_balanced(&enriched_config)
            == crate::proto::proximadb_v1::StorageClass::Performance
            && let Some(registry) = &self.reservations
        {
            let reservation_tenant = Self::version_tenant(tenant_context).to_string();
            // SC4-2: the grant protects the collection's segment roots — built
            // from the same tenant-scoped base the storage assignment records
            // below, so the prefixes match what flush/compaction publish.
            let protect_prefixes = crate::services::collection::storage_class::residency_prefixes(
                &tenant_base_location,
                &uuid,
                typed_identity,
            );
            reserved_grant = crate::services::collection::reservation::GrantCompensation::reserve(
                registry,
                reservation_tenant,
                uuid.clone(),
                chrono::Utc::now().timestamp(),
                &protect_prefixes,
            );
            reserved_grant
                .committed()
                .await
                .context("TD-TIER-1 SC3: PERFORMANCE admission refused")?;
        }
        // Compensating release for every early exit after a successful
        // reserve (review B1: the grant must not outlive a failed create). A
        // DROPPED create future (client disconnect) releases too: the guard's
        // Drop spawns the release (SC4-3 review S3); it is disarmed once the
        // catalog record — the collection's existence — is durable.
        macro_rules! compensate_reservation {
            () => {
                reserved_grant.release_now().await;
            };
        }

        // Create storage directories under the tenant-scoped base (typed or
        // legacy layout) — exactly where flush and compaction will publish.
        let _storage_created = match self
            .create_storage_directories(
                &tenant_base_location,
                &enriched_config.name,
                &uuid,
                tenant_id,
                crate::storage::trait_components::path_resolver::typed_path_identity(
                    typed_identity,
                ),
            )
            .await
        {
            Ok(created) => created,
            Err(e) => {
                compensate_reservation!();
                return Err(e.context("Failed to create storage directories"));
            }
        };

        // ADR-031 Phase 4d: when a typed identity was minted (env ON + account
        // known), carry the triple on the proto `StorageAssignment` so the
        // catalog-free engines can resolve the account-rooted read path
        // (`accounts/{acct}/{ns}/{coll}/…`) at read time — they have no catalog
        // access and can't mint it themselves. All three are Some together, or
        // all None (env OFF / no account → legacy byte-identical path).
        let (typed_account_id, typed_namespace_id, typed_collection_id) = match typed_identity {
            Some(id) => (
                Some(id.account_id),
                Some(id.namespace_id as u32),
                Some(id.collection_id),
            ),
            None => (None, None, None),
        };

        // Create proto collection with stats and storage assignment
        let proto_collection = Collection {
            storage_class_status: Some(
                crate::services::collection::storage_class::storage_class_status(&enriched_config),
            ),
            id: uuid.clone(),
            config: Some(enriched_config.clone()), // Use enriched config with compression
            stats: Some(crate::proto::proximadb_v1::CollectionStats {
                vector_count: 0,
                index_size_bytes: 0,
                data_size_bytes: 0,
            }),
            created_at: now,
            updated_at: now,
            storage_assignment: Some(crate::proto::proximadb_v1::StorageAssignment {
                primary_path: tenant_base_location.clone(),
                backup_paths: vec![],
                engine: enriched_config
                    .storage_engine
                    .unwrap_or(StorageEngine::Sst as i32),
                engine_config: std::collections::HashMap::new(),
                base_location: tenant_base_location.clone(), // Tenant-prefixed path
                assigned_at: chrono::Utc::now().timestamp_micros(),
                typed_account_id,
                typed_namespace_id,
                typed_collection_id,
            }),
        };

        if let Err(e) = self
            .upsert_collection_catalog_asset(&proto_collection, typed_identity)
            .await
        {
            compensate_reservation!();
            return Ok(CollectionServiceResponse::error(
                format!("CATALOG_CREATE_FAILED: {}", e),
                start_time.elapsed().as_micros() as i64,
            ));
        }
        reserved_grant.disarm();

        // The catalog asset write above is the sole authoritative store for the
        // collection; the legacy metadata backend is no longer dual-written.

        info!(
            "✅ Collection created: {} (UUID: {}) with storage at: {} in {}μs",
            config.name,
            uuid,
            base_location,
            start_time.elapsed().as_micros()
        );

        // Use proto collection directly - no conversion needed in proto-first architecture

        // Generate storage path template
        let storage_path = format!("${{base_path}}/collections/{}", uuid);

        // Bump the corpus_version for (tenant, collection) so the
        // process-wide PlanCache invalidates on the first planner
        // lookup against the freshly-created collection. New
        // collections start at version 2 (default was 1), so any
        // entry that ended up in the cache during a race condition
        // — e.g. a search that arrived between catalog upsert and
        // this bump — gets superseded immediately.
        {
            let tenant = Self::version_tenant(tenant_context);
            let version = Self::bump_corpus_versions(tenant, &config.name).await;
            debug!(
                "🔄 corpus_version bumped after create: tenant={} collection={} version={}",
                tenant, config.name, version
            );
        }

        Ok(CollectionServiceResponse {
            success: true,
            collection: Some(proto_collection), // Direct proto usage - no conversion!
            storage_path: Some(storage_path),
            // error_message removed -  None,
            error_code: None,
            processing_time_us: start_time.elapsed().as_micros() as i64,
        })
    }

    /// Get the full proto collection with all metadata - direct access to deserialized object
    pub async fn collection(&self, identifier: &str) -> Result<Option<Collection>> {
        // Serve through the hot cache only on the path where invalidation is
        // provably coherent: single-tenant mode (writes bump corpus_version under
        // the `"default"` bucket, #435) AND a NAME key (not a UUID). A UUID lookup
        // is keyed in a version space the name-keyed write bump never touches, so
        // it would never invalidate — bypass it. Multi-tenant reads also bypass
        // until a tenant-aware caching slice lands. Bypass ⇒ today's direct read.
        if let Some(cache) = &self.syscat_cache
            && self.tenant_manager.is_none()
            && uuid::Uuid::parse_str(identifier).is_err()
        {
            return cache
                .resolve(Self::DEFAULT_VERSION_TENANT, identifier)
                .await;
        }
        self.get_native_proto(identifier).await
    }

    /// Get Collection by name or UUID
    async fn get_native_proto(&self, identifier: &str) -> Result<Option<Collection>> {
        if let Some(collection) = self.collection_from_catalog_asset(identifier).await? {
            return Ok(Some(collection));
        }

        // The catalog is the sole read authority: collection_from_catalog_asset
        // already resolves by both name and UUID, so a miss means the collection
        // does not exist.
        Ok(None)
    }

    /// ✅ RESOLVE COLLECTION NAME/ID TO COLLECTION ID
    /// This is the key method for collection identifier resolution
    /// - Input: Collection name OR collection ID
    /// - Output: Collection ID (base62) for internal use
    /// - Used by WAL, storage, and index path resolution
    pub async fn resolve_collection_id(&self, identifier: &str) -> Result<Option<String>> {
        tracing::debug!("🔍 Resolving collection identifier: '{}'", identifier);

        // Resolution is NAME-AUTHORITATIVE and shape-independent: `collection()`
        // looks up the catalog asset (by name) first, then the metadata backend
        // (by id). The historical base62-id "looks like an id" length heuristic is
        // gone — IDs are now opaque UUIDs (no overlap with user names), so name
        // length carries no meaning here. This is why short SQL/ANSI table names
        // are safe (TPC-H `part`/`orders`/`region`) and the 8-char floor was dropped.
        if let Some(collection) = self.collection(identifier).await? {
            let collection_id = collection.id;
            tracing::debug!(
                "✅ Resolved '{}' -> collection_id: '{}'",
                identifier,
                collection_id
            );
            Ok(Some(collection_id))
        } else {
            tracing::debug!("❌ Collection not found: '{}'", identifier);
            Ok(None)
        }
    }

    /// ✅ RESOLVE COLLECTION ID TO COLLECTION NAME  
    /// Reverse resolution for user-friendly displays
    pub async fn resolve_collection_name(&self, collection_id: &str) -> Result<Option<String>> {
        if let Some(collection) = self.collection(collection_id).await?
            && let Some(config) = &collection.config
        {
            return Ok(Some(config.name.clone()));
        }
        Ok(None)
    }

    /// Convert Collection to core Collection - direct proto to core mapping
    /// Get IndexConfig for a collection by name or UUID with caching
    pub async fn native_index_config(
        &self,
        identifier: &str,
    ) -> Result<Option<crate::index::config::RuntimeIndexConfig>> {
        debug!("🔍 Getting IndexConfig for collection: {}", identifier);

        // Check cache first
        if let Some(cached_config) = self.index_config_cache.get(identifier) {
            debug!(
                "📋 Retrieved IndexConfig from cache for collection: {}",
                identifier
            );
            return Ok(Some(cached_config.value().clone()));
        }

        if let Some(proto_collection) = self.get_native_proto(identifier).await? {
            let index_config = self.parse_index_config_from_proto(&proto_collection)?;

            // Cache the result
            self.index_config_cache
                .insert(identifier.to_string(), index_config.clone());
            self.index_config_cache
                .insert(proto_collection.id.clone(), index_config.clone()); // Cache by UUID too

            debug!("📋 Cached IndexConfig for collection: {}", identifier);
            Ok(Some(index_config))
        } else {
            Ok(None)
        }
    }

    /// Convert proto IndexConfig to internal IndexConfig
    fn convert_proto_index_config(
        &self,
        _proto_config: &crate::proto::proximadb_v1::IndexConfig,
    ) -> Result<crate::index::config::RuntimeIndexConfig> {
        // Extract algorithm name from proto config
        let _algorithm_name = match _proto_config.algorithm {
            1 => "HNSW",
            2 => "IVF",
            3 => "PQ",
            4 => "FLAT",
            5 => "ANNOY",
            _ => "HNSW", // Default to HNSW
        };

        // Use the from_proto method that handles all the config extraction
        crate::index::config::RuntimeIndexConfig::from_proto(_proto_config)
    }

    /// Parse IndexConfig from Collection
    fn parse_index_config_from_proto(
        &self,
        proto: &Collection,
    ) -> Result<crate::index::config::RuntimeIndexConfig> {
        // Check if proto has index_config field
        if let Some(config) = proto.config.as_ref()
            && !config.index_configs.is_empty()
        {
            // Take the first IndexConfig from proto (index_configs is a Vec)
            if let Some(first_config) = config.index_configs.first() {
                // Convert from proto IndexConfig to internal IndexConfig
                return self.convert_proto_index_config(first_config);
            }
        }

        // No IndexConfig found, create smart defaults based on algorithm
        let config = proto
            .config
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Collection has no config"))?;
        let indexing_algorithm: crate::core::IndexingAlgorithm =
            match config.primary_index.as_deref().unwrap_or("default") {
                "hnsw" => "hnsw".to_string(),
                "ivf" => "ivf".to_string(),
                "pq" => "pq".to_string(),
                "flat" => "flat".to_string(),
                "annoy" => "annoy".to_string(),
                "lsh" => "lsh".to_string(),
                _ => "hnsw".to_string(),
            };

        let algorithm_str = match indexing_algorithm.as_str() {
            "hnsw" => "HNSW",
            "ivf" => "IVF",
            "pq" => "PQ",
            "flat" => "FLAT",
            "annoy" => "ANNOY",
            "lsh" => "LSH",
            _ => "HNSW", // Default to HNSW
        };

        let smart_config = crate::index::config::RuntimeIndexConfig::create_smart_default(
            algorithm_str,
            config.dimension as usize,
            None, // Collection size hint not available
        );

        debug!(
            "📋 Created smart default IndexConfig for collection: {}",
            config.name
        );
        Ok(smart_config)
    }

    // Configuration parsing helper methods

    /// Get quantization configuration for a collection
    pub async fn native_quantization_config(
        &self,
        identifier: &str,
    ) -> Result<Option<crate::proto::proximadb_v1::QuantizationConfig>> {
        debug!(
            "🔍 Getting quantization config for collection: {}",
            identifier
        );

        if let Some(proto) = self.get_native_proto(identifier).await? {
            Ok(proto.config.and_then(|c| c.quantization))
        } else {
            Ok(None)
        }
    }

    /// Get search hints for a collection  
    pub async fn native_search_hints(&self, identifier: &str) -> Result<Option<serde_json::Value>> {
        debug!("🔍 Getting search hints for collection: {}", identifier);

        if let Some(_proto) = self.get_native_proto(identifier).await? {
            // Extract search hints from proto config
            if let Some(config) = _proto.config.as_ref() {
                // Build search hints from collection configuration
                let mut hints = serde_json::json!({
                    "ef_search": 200,
                    "max_candidates": 500,
                    "use_quantized": config.quantization.is_some(),
                    "enable_reranking": true
                });

                // Extract hints from index configs
                if let Some(first_index) = config.index_configs.first() {
                    // Override with algorithm-specific parameters
                    if let Some(hnsw_config) = &first_index.hnsw_config {
                        hints["ef_search"] = serde_json::json!(hnsw_config.ef_search);
                        hints["max_candidates"] =
                            serde_json::json!(hnsw_config.ef_search.unwrap_or(100) * 2);
                    }
                    if let Some(ivf_config) = &first_index.ivf_config {
                        hints["n_probe"] = serde_json::json!(ivf_config.n_probe);
                        hints["max_candidates"] =
                            serde_json::json!(ivf_config.n_probe.unwrap_or(10) * 100);
                    }
                }

                // Add storage engine specific hints
                hints["storage_engine"] =
                    serde_json::json!(crate::core::conversions::storage_engine_to_string(
                        config.storage_engine.unwrap_or(StorageEngine::Sst as i32)
                    ));

                Ok(Some(hints))
            } else {
                // Return default hints if no config
                Ok(Some(serde_json::json!({
                    "ef_search": 200,
                    "max_candidates": 500,
                    "use_quantized": false,
                    "enable_reranking": true
                })))
            }
        } else {
            Ok(None)
        }
    }

    /// Get index parameters for a collection
    pub async fn native_index_params(&self, identifier: &str) -> Result<Option<serde_json::Value>> {
        debug!("🔍 Getting index params for collection: {}", identifier);

        if let Some(proto) = self.get_native_proto(identifier).await? {
            if let Some(config) = proto.config {
                // Proto uses index_configs (plural) which is a vector of IndexConfig
                Ok(Some(serde_json::to_value(&config.index_configs)?))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Get storage configuration for a collection
    pub async fn native_storage_config(
        &self,
        identifier: &str,
    ) -> Result<Option<serde_json::Value>> {
        debug!("🔍 Getting storage config for collection: {}", identifier);

        if let Some(_proto) = self.get_native_proto(identifier).await? {
            if let Some(config) = _proto.config.as_ref() {
                // Build storage config from proto
                let engine_name = crate::core::conversions::storage_engine_to_string(
                    config.storage_engine.unwrap_or(StorageEngine::Sst as i32),
                );

                let mut storage_config = serde_json::json!({
                    "engine": engine_name,
                    "enable_compression": true,
                    "enable_deduplication": false, // Not exposed in proto yet
                    "enable_multi_tenancy": false,  // Not exposed in proto yet
                });

                // Add engine-specific configurations
                if engine_name == "VIPER" {
                    storage_config["parquet_config"] = serde_json::json!({
                        "row_group_size": 65536,
                        "compression": "snappy",
                        "enable_statistics": true,
                        "enable_bloom_filter": true,
                    });

                    // Add quantization config if present
                    if let Some(quant_config) = &config.quantization {
                        storage_config["quantization_enabled"] = serde_json::json!(true);
                        storage_config["quantization_level"] = serde_json::json!(quant_config);
                    }
                }

                Ok(Some(storage_config))
            } else {
                // Return default storage config
                Ok(Some(serde_json::json!({
                    "engine": "LSM",
                    "enable_compression": true,
                    "enable_deduplication": false,
                    "enable_multi_tenancy": false,
                })))
            }
        } else {
            Ok(None)
        }
    }

    /// List all collections - returns proto Collections directly (proto-first architecture)
    pub async fn list_collections(&self) -> Result<Vec<Collection>> {
        debug!("📋 Listing all collections");
        // The catalog is the sole read authority.
        self.list_collections_from_catalog().await
    }

    /// Delete collection with comprehensive cleanup across all storage components
    pub async fn delete_collection(
        &self,
        collection_identifier: &str,
    ) -> Result<CollectionServiceResponse> {
        info!("🗑️ Deleting collection: {}", collection_identifier);
        let start_time = std::time::Instant::now();

        // Get collection record first to retrieve UUID and other details. xCatalog is checked
        // first; the legacy backend is only a compatibility fallback.
        let collection_record = self.collection(collection_identifier).await?;

        if let Some(record) = collection_record {
            let collection_uuid = record.id.clone();
            let collection_name = record.config.as_ref().map(|c| c.name.clone()).clone();

            info!(
                "🔍 Found collection to delete: {} (UUID: {})",
                collection_name.as_deref().unwrap_or("<unnamed>"),
                collection_uuid
            );

            // Step 1: Clean up all storage directories and files
            let cleanup_results = self
                .cleanup_storage_directories(
                    collection_name.as_deref().unwrap_or(collection_identifier),
                    &collection_uuid,
                )
                .await;
            match cleanup_results {
                Ok(cleaned_components) => {
                    info!(
                        "🧹 Cleaned up {} storage components for collection {}",
                        cleaned_components,
                        collection_name.as_deref().unwrap_or("<unnamed>")
                    );
                }
                Err(e) => {
                    warn!(
                        "⚠️ Some storage cleanup failed for collection {}: {}",
                        collection_name.as_deref().unwrap_or("<unnamed>"),
                        e
                    );
                    // Continue with metadata deletion even if storage cleanup partially fails
                }
            }

            // Step 2: Assignment removal is no longer needed
            // Storage assignment is now part of collection metadata which gets deleted

            // Step 3: Delete from xCatalog and metadata backend
            if let Err(e) = self.drop_collection_catalog_asset(&record).await {
                return Ok(CollectionServiceResponse::error(
                    format!("CATALOG_DELETE_FAILED: {}", e),
                    start_time.elapsed().as_micros() as i64,
                ));
            }

            // TD-TIER-1 SC3-2 (review SF3): release the reservation HERE —
            // keyed by the record THIS delete actually removed, so every
            // caller (the tenant-aware wrapper, the embedded bindings) is
            // covered and a delete-vs-recreate race cannot key the wrong
            // incarnation. Both the record's owner and the default tenant are
            // released (idempotent): create folds its key the same way, and
            // the tail billing interval parks for the metering tick.
            if let Some(registry) = &self.reservations {
                let now = chrono::Utc::now().timestamp();
                let owner = record
                    .config
                    .as_ref()
                    .and_then(|cfg| cfg.owner.clone())
                    .filter(|owner| !owner.is_empty());
                if let Some(owner) = owner.as_deref() {
                    registry
                        .release_for_collection(owner, &collection_uuid, now)
                        .await;
                }
                registry
                    .release_for_collection(Self::DEFAULT_VERSION_TENANT, &collection_uuid, now)
                    .await;
            }

            let deleted = true;

            if deleted {
                info!(
                    "✅ Collection deleted: {} (UUID: {}) in {}μs",
                    collection_name.as_deref().unwrap_or("<unnamed>"),
                    collection_uuid,
                    start_time.elapsed().as_micros()
                );

                Ok(CollectionServiceResponse {
                    success: true,
                    collection: Some(record.clone()), // Include the deleted collection record
                    storage_path: None,
                    // error_message removed -  None,
                    error_code: None,
                    processing_time_us: start_time.elapsed().as_micros() as i64,
                })
            } else {
                Ok(CollectionServiceResponse {
                    success: false,
                    collection: Some(record.clone()), // Include the collection that failed to delete
                    storage_path: None,
                    error_code: Some("METADATA_DELETE_FAILED".to_string()),
                    processing_time_us: start_time.elapsed().as_micros() as i64,
                })
            }
        } else {
            Ok(CollectionServiceResponse {
                success: false,
                collection: None,
                storage_path: None,
                error_code: Some("COLLECTION_NOT_FOUND".to_string()),
                processing_time_us: start_time.elapsed().as_micros() as i64,
            })
        }
    }

    /// Update collection statistics (called by storage engine after vector operations)
    pub async fn update_stats(
        &self,
        collection_name: &str,
        vector_delta: i64,
        size_delta: i64,
    ) -> Result<()> {
        // SF2 (SC3-2b review): this is a full-record read-modify-write of the
        // catalog asset — without the RMW lock it interleaves with
        // update_collection's policy mutation and silently reverts a
        // persisted class change while its grant stays live.
        let _rmw = COLLECTION_CONFIG_RMW.lock().await;
        debug!(
            "📊 Updating stats for {}: vectors={:+}, size={:+}",
            collection_name, vector_delta, size_delta
        );

        // Get current record from the catalog, update stats, and save back.
        if let Some(mut record) = self.collection(collection_name).await? {
            // Update stats manually for Collection
            if let Some(stats) = record.stats.as_mut() {
                stats.vector_count += vector_delta;
                stats.data_size_bytes += size_delta;
            }
            record.updated_at = chrono::Utc::now().timestamp_millis();

            self.upsert_collection_catalog_asset(&record, None).await?;
        } else {
            warn!(
                "⚠️ Attempted to update stats for non-existent collection: {}",
                collection_name
            );
        }

        Ok(())
    }

    /// Get collection statistics for cost-based query optimization
    ///
    /// Returns canonical `CollectionStats` from the storage engine, enriched
    /// with metadata from the collection config (dimension, index type).
    /// Used by the query planner's CostModel.
    pub async fn get_collection_stats(
        &self,
        collection_name: &str,
        storage_engine: Option<&std::sync::Arc<dyn crate::storage::traits::UnifiedStorageEngine>>,
    ) -> Result<crate::storage::traits::CollectionStats> {
        // If a storage engine is provided, delegate to it for real stats
        if let Some(engine) = storage_engine {
            let mut stats = engine.collection_stats(collection_name).await?;

            // Enrich with metadata from collection config
            if let Some(collection) = self.collection(collection_name).await?
                && let Some(config) = &collection.config
            {
                stats.dimension = Some(config.dimension);
            }

            return Ok(stats);
        }

        // Fallback: return stats from proto collection metadata
        if let Some(collection) = self.collection(collection_name).await? {
            let mut stats = crate::storage::traits::CollectionStats::default();
            if let Some(proto_stats) = collection.stats {
                stats.row_count = proto_stats.vector_count as u64;
                stats.total_bytes = proto_stats.data_size_bytes as u64;
            }
            if let Some(config) = &collection.config {
                stats.dimension = Some(config.dimension);
            }
            return Ok(stats);
        }

        Ok(crate::storage::traits::CollectionStats::default())
    }

    /// Get collection UUID by name or UUID
    pub async fn uuid(&self, collection_id: &str) -> Result<Option<String>> {
        debug!("🔍 Getting UUID for collection: {}", collection_id);

        // First check if it's already a UUID
        if proximadb_kernel::uuid::Uuid::parse(collection_id).is_ok() {
            // Verify it exists
            if let Some(collection) = self.collection(collection_id).await? {
                return Ok(Some(collection.id));
            }
            return Ok(None);
        }

        // Otherwise look up by name
        if let Some(collection) = self.collection(collection_id).await? {
            Ok(Some(collection.id))
        } else {
            Ok(None)
        }
    }

    /// Update collection - type-safe method with native parameters
    pub async fn update_collection(
        &self,
        identifier: &str,
        config_update: Option<CollectionConfig>, // Use native proto type!
    ) -> Result<CollectionServiceResponse> {
        info!("📝 Updating collection: {}", identifier);
        let start_time = std::time::Instant::now();
        // One read-modify-write at a time (shared with the owner backfill), so
        // the recorded owner carried over below is the current one.
        let _rmw = COLLECTION_CONFIG_RMW.lock().await;

        // Get current record (supports both names and UUIDs) through xCatalog first.
        let mut record = match self.collection(identifier).await? {
            Some(record) => record,
            None => {
                return Ok(CollectionServiceResponse {
                    success: false,
                    collection: None,
                    storage_path: None,
                    error_code: Some("COLLECTION_NOT_FOUND".to_string()),
                    processing_time_us: start_time.elapsed().as_micros() as i64,
                });
            }
        };
        let previous_record = record.clone();

        // TD-TIER-1 SC3-2b: pending grant side effect of a class change —
        // `Some(true)` = reserve after persist, `Some(false)` = release.
        // Applied only after the durable upsert succeeds.
        let mut policy_side_effect: Option<bool> = None;
        // SF1: a post-persist reserve failure surfaces here (policy durable,
        // grant missing) instead of a quiet success.
        let mut reservation_pending: Option<String> = None;

        // Apply updates using native proto types
        if let Some(new_config) = config_update {
            // Merge the new config with existing one to preserve unchanged fields
            if let Some(existing_config) = record.config.as_mut() {
                // Only update fields that are provided in new_config
                if !new_config.name.is_empty() {
                    existing_config.name = new_config.name;
                }
                if new_config.dimension > 0 {
                    existing_config.dimension = new_config.dimension;
                }
                if new_config.distance_metric.unwrap_or(0) != 0 {
                    existing_config.distance_metric = new_config.distance_metric;
                }
                if new_config.storage_engine.unwrap_or(0) != 0 {
                    // TD-VIPER-1 S2 (ADR-093) + the MMAP/Hybrid/experimental-
                    // gated fail-loud guarantee: the same two checks
                    // `create_collection_with_tenant_context` applies — the
                    // guard travels with the field, not just the one call
                    // site that happens to be wired today.
                    if new_config.storage_engine
                        == Some(crate::proto::proximadb_v1::StorageEngine::Viper as i32)
                    {
                        return Err(anyhow::anyhow!(
                            "VIPER is deprecated (ADR-093) and can no longer be selected: \
                             use SST (default), NOVA (columnar analytics), HELIX, or the \
                             DataFusion/Parquet path for warehouse-shaped analytics"
                        ));
                    }
                    if let Ok(engine) = crate::proto::proximadb_v1::StorageEngine::try_from(
                        new_config.storage_engine.unwrap_or(0),
                    ) {
                        crate::storage::engines::factory::StorageFormatFactory::create_from_proto_async(
                            engine,
                        )
                        .await
                        .map_err(|error| {
                            anyhow::anyhow!("storage engine {engine:?} is unavailable: {error:#}")
                        })?;
                    }
                    existing_config.storage_engine = new_config.storage_engine;
                }
                if new_config.description.is_some() {
                    existing_config.description = new_config.description;
                }
                // TD-TIER-1 SC2: durable placement is resolved at create; a
                // later storage-path change is a migration, not a config
                // patch. It used to be silently dropped by this merge — now
                // it is an actionable rejection.
                if let Some(new_storage) = new_config.storage_config.as_ref()
                    && let Some(new_path) = new_storage.storage_path.as_deref()
                    && new_path
                        != existing_config
                            .storage_config
                            .as_ref()
                            .and_then(|s| s.storage_path.as_deref())
                            .unwrap_or_default()
                {
                    return Err(anyhow::anyhow!(
                        "storage_config.storage_path cannot be changed after create \
                         ('{new_path}'): durable placement is immutable in v1; recreate \
                         the collection at the new location to migrate it"
                    ));
                }
                // TD-TIER-1 SC3-2b: the policy-mutation surface. A different
                // class is RE-ADMITTED, not rejected: entry to PERFORMANCE
                // re-runs the SC2 placement floor on the EXISTING resolved
                // placement and PEEKS the reservation registry (default pool
                // resolvable + armed + kill switch off) BEFORE the config
                // persists — a refused update keeps the old policy unchanged.
                // The grant side effects (reserve/release) apply only AFTER
                // the catalog upsert succeeds, so a failed update can never
                // orphan a grant (the create-path lesson, applied here). A
                // same-value re-apply is an idempotent no-op (no generation
                // bump); an unknown enum number is rejected on every facade;
                // an omitted field (or UNSPECIFIED) preserves the recorded
                // class.
                if let Some(raw_class) = new_config.storage_class
                    && raw_class != 0
                {
                    let resolved =
                        crate::services::collection::storage_class::validate_storage_class_raw(
                            Some(raw_class),
                        )?;
                    let recorded =
                        crate::services::collection::storage_class::storage_class_or_balanced(
                            existing_config,
                        );
                    if resolved != recorded {
                        if resolved == crate::proto::proximadb_v1::StorageClass::Performance {
                            // SC2 floor on the existing placement: the same
                            // checks create runs, against the placement this
                            // collection already has (durable placement is
                            // immutable, so create-time admissibility carries).
                            let mut prospective_config = existing_config.clone();
                            prospective_config.storage_class = Some(resolved as i32);
                            let engine =
                                crate::services::collection::storage_class::record_storage_engine(
                                    previous_record
                                        .storage_assignment
                                        .as_ref()
                                        .map(|a| a.engine),
                                    existing_config.storage_engine,
                                );
                            let base = previous_record
                                .storage_assignment
                                .as_ref()
                                .map(|a| a.base_location.clone())
                                .unwrap_or_default();
                            let owner = previous_record
                                .config
                                .as_ref()
                                .and_then(|cfg| cfg.owner.clone())
                                .filter(|owner| !owner.is_empty());
                            let admission_tenant = owner.as_deref().map(str::to_string);
                            crate::services::collection::storage_class::check_performance_admission(
                                &prospective_config,
                                engine,
                                &base,
                                admission_tenant.as_deref(),
                            )?;
                            // Registry peek (no side effects): an unarmed or
                            // kill-switched pool refuses BEFORE the config
                            // persists. `self.reservations = None` (no pools
                            // declared) also refuses — PERFORMANCE without a
                            // reservation admission would be a silent lie.
                            match &self.reservations {
                                Some(registry) => registry.peek_performance_admission().await?,
                                None => {
                                    return Err(anyhow::anyhow!(
                                        "storage_class 'performance' requires a \
                                         configured reservation pool: none is declared \
                                         ([storage.reservation_pools]); the update is \
                                         refused and the recorded policy is unchanged"
                                    ));
                                }
                            }
                        }
                        // Commit the class change + monotonic generation bump
                        // in-memory; the grant side effect applies after the
                        // durable upsert below.
                        let generation =
                            crate::services::collection::storage_class::policy_generation_or_legacy(
                                existing_config,
                            );
                        existing_config.policy_generation =
                            Some(u32::try_from(generation.saturating_add(1)).unwrap_or(u32::MAX));
                        existing_config.storage_class = Some(resolved as i32);
                        policy_side_effect =
                            Some(resolved == crate::proto::proximadb_v1::StorageClass::Performance);
                    }
                }
                // TD-TIER-1 SC5: the tier governs FUTURE segment PUTs only
                // (nothing is rewritten), so a change needs no reservation
                // effects — just re-admission on the EXISTING placement,
                // refused before the config persists. Omitted/UNSPECIFIED
                // preserves the recorded tier; an unknown number is refused.
                if let Some(raw_tier) = new_config.default_access_tier
                    && raw_tier != 0
                {
                    let base = previous_record
                        .storage_assignment
                        .as_ref()
                        .map(|a| a.base_location.clone())
                        .unwrap_or_default();
                    let prospective = crate::proto::proximadb_v1::CollectionConfig {
                        default_access_tier: Some(raw_tier),
                        ..Default::default()
                    };
                    crate::services::collection::storage_class::check_access_tier_admission(
                        &prospective,
                        &base,
                    )?;
                    existing_config.default_access_tier = Some(raw_tier);
                }
                if !new_config.tags.is_empty() {
                    existing_config.tags = new_config.tags;
                }
                if new_config.owner.is_some() {
                    existing_config.owner = new_config.owner;
                }
                // Ownership is not client-updatable: keep the recorded owner
                // signals exactly as they were.
                if let Some(previous) = previous_record.config.as_ref() {
                    proximadb_tenant::ownership::carry_over_owner(previous, existing_config);
                }
                // TD-CATREPLACE-1: a schema update (marked by the record-schema
                // block `apply_schema_update` always sets) carries the
                // authoritative FULL list — replace it even when empty, so a
                // removed filterable column stops being served. A raw config
                // patch without that intent — including one that only flips
                // `enable_proxima_record` — keeps the legacy non-empty merge.
                let schema_update_intent = new_config.record_schema.is_some();
                if schema_update_intent || !new_config.filterable_columns.is_empty() {
                    existing_config.filterable_columns = new_config.filterable_columns.clone();
                }
                // TD-GRPCSCHEMA-2: a schema update carries the authoritative
                // record-mode block too — before, the merge DROPPED
                // `enable_proxima_record`/`record_schema` (and the text-column
                // projection), so record mode could never be enabled after
                // create, no matter how many schema updates ran.
                if schema_update_intent {
                    existing_config.enable_proxima_record = new_config.enable_proxima_record;
                    existing_config.record_schema = new_config.record_schema.clone();
                    existing_config.text_columns = new_config.text_columns.clone();
                    existing_config.text_storage_configs = new_config.text_storage_configs.clone();
                }
                // Add other fields as needed
            } else {
                // No existing config: use the new one, but a storage class
                // cannot ride this defensive branch — it would bypass the
                // entire policy surface (SC2 floor, registry peek,
                // generation, grant effects). Refuse it explicitly.
                if new_config.storage_class.is_some_and(|class| class != 0)
                    || new_config.default_access_tier.is_some_and(|tier| tier != 0)
                {
                    return Err(anyhow::anyhow!(
                        "storage_class / default_access_tier cannot be set on a \
                         record without a recorded config; recreate the collection \
                         instead"
                    ));
                }
                record.config = Some(new_config);
            }
        }

        // Update timestamp
        record.updated_at = chrono::Utc::now().timestamp_millis();

        if previous_record
            .config
            .as_ref()
            .zip(record.config.as_ref())
            .is_some_and(|(previous, current)| previous.name != current.name)
        {
            self.drop_collection_catalog_asset(&previous_record)
                .await
                .context("Failed to remove previous collection catalog asset")?;
        }

        self.upsert_collection_catalog_asset(&record, None)
            .await
            .context("Failed to update collection catalog metadata")?;

        // TD-TIER-1 SC3-2b: the policy is durable — now apply the grant side
        // effect. Entry's reserve failure here is a recorded SC4
        // reconciliation residual (policy persisted as performance without a
        // grant; status stays NOT_ACTIVE and re-admission heals it) — the
        // inverse ordering (grant before persist) could ORPHAN a grant,
        // which is strictly worse.
        if let Some(to_performance) = policy_side_effect
            && let Some(registry) = &self.reservations
        {
            let collection_id = previous_record.id.clone();
            let owner = previous_record
                .config
                .as_ref()
                .and_then(|cfg| cfg.owner.clone())
                .filter(|owner| !owner.is_empty());
            let now = chrono::Utc::now().timestamp();
            if to_performance {
                let tenant = owner.unwrap_or_else(|| Self::DEFAULT_VERSION_TENANT.to_string());
                // SC4-2: protection prefixes from the record's own storage
                // assignment (the exact base + typed triple flush uses).
                let protect_prefixes = previous_record
                    .storage_assignment
                    .as_ref()
                    .map(|sa| {
                        crate::services::collection::storage_class::residency_prefixes(
                            &sa.base_location,
                            &collection_id,
                            crate::storage::trait_components::path_resolver::typed_identity_from_storage_assignment(
                                Some(sa),
                            ),
                        )
                    })
                    .unwrap_or_default();
                if protect_prefixes.is_empty() {
                    warn!(
                        "TD-TIER-1 SC4: collection {collection_id} has no storage assignment; \
                         its grant charges capacity but protects no cached data"
                    );
                }
                if let Err(error) = registry
                    .reserve_for_performance(&tenant, &collection_id, now, &protect_prefixes)
                    .await
                {
                    // The policy is durable but unreserved (peek→reserve
                    // window): the caller must SEE it, not just the log —
                    // success stays true (the update persisted) with an
                    // explicit error_code.
                    warn!(
                        "TD-TIER-1 SC3: post-persist reserve failed for {collection_id}: \
                         {error} (policy recorded; SC4 reconciliation re-admits)"
                    );
                    reservation_pending = Some(error.to_string());
                } else {
                    // SC4-4: segments published before this class change
                    // become resident in the background.
                    crate::storage::engines::sst::segment_io::spawn_warm_prefixes(
                        self.filesystem_factory.clone(),
                        protect_prefixes.clone(),
                    );
                }
            } else {
                if let Some(owner) = owner.as_deref() {
                    registry
                        .release_for_collection(owner, &collection_id, now)
                        .await;
                }
                registry
                    .release_for_collection(Self::DEFAULT_VERSION_TENANT, &collection_id, now)
                    .await;
            }
        }

        info!(
            "✅ Collection updated: {} in {}μs",
            identifier,
            start_time.elapsed().as_micros()
        );

        // Bump the corpus_version so corpus-version-keyed caches invalidate after
        // the schema change. `update_collection` carries no tenant context, so it
        // keys by the default bucket (consistent with reads). A rename bumps both
        // the new and the previous name so a read of either key reloads.
        let new_name = record
            .config
            .as_ref()
            .map(|c| c.name.as_str())
            .unwrap_or(identifier);
        let _ = crate::catalog::CorpusVersionRegistry::global()
            .bump(Self::DEFAULT_VERSION_TENANT, new_name)
            .await;
        if let Some(prev_name) = previous_record.config.as_ref().map(|c| c.name.as_str())
            && prev_name != new_name
        {
            let _ = crate::catalog::CorpusVersionRegistry::global()
                .bump(Self::DEFAULT_VERSION_TENANT, prev_name)
                .await;
        }

        // NIT6: recompute the status projection so the response carries the
        // NEW class's status, not the read-time snapshot.
        if let Some(config) = record.config.as_ref() {
            record.storage_class_status =
                Some(crate::services::collection::storage_class::storage_class_status(config));
        }

        // Record is already a proto Collection, no conversion needed
        let collection = record;

        Ok(CollectionServiceResponse {
            success: true,
            collection: Some(collection),
            storage_path: None,
            // error_message removed -  None,
            error_code: reservation_pending
                .map(|message| format!("RESERVATION_ADMISSION_PENDING: {message}")),
            processing_time_us: start_time.elapsed().as_micros() as i64,
        })
    }

    /// The catalog this service reads/writes collection assets through, if wired.
    /// Used to make the catalog the WAL/recovery collection-resolution authority.
    pub fn catalog_manager(&self) -> Option<Arc<CatalogManager>> {
        self.catalog_manager.clone()
    }

    /// Resolve compression configuration based on SDK request and server defaults
    fn resolve_compression_config(
        &self,
        requested: Option<&crate::proto::proximadb_v1::CompressionConfig>,
        _storage_engine: i32,
    ) -> Option<crate::proto::proximadb_v1::CompressionConfig> {
        use crate::proto::proximadb_v1::{CompressionAlgorithm, CompressionConfig};

        // If compression explicitly requested, validate and use it
        if let Some(config) = requested {
            // Validate compression level if specified
            if let Some(level) = config.level
                && let Ok(CompressionAlgorithm::CompressionZstd) =
                    CompressionAlgorithm::try_from(config.algorithm)
                && !(1..=22).contains(&level)
            {
                warn!("Invalid ZSTD compression level {}, using default 3", level);
                return Some(CompressionConfig {
                    algorithm: config.algorithm,
                    level: Some(3),
                    adaptive: config.adaptive,
                    min_ratio: config.min_ratio,
                    enable_quantization: config.enable_quantization,
                    quantization_type: config.quantization_type.clone(),
                    normalization_method: config.normalization_method.clone(),
                    block_size_kb: config.block_size_kb,
                    dynamic_block_sizing: config.dynamic_block_sizing,
                });
            }
            return Some(config.clone());
        }

        // SDK-DRIVEN COMPRESSION (2025-08-06): No server defaults!
        // Compression must be specified by the SDK/client
        // Return None to indicate no compression if not specified by SDK
        None

        // SDK-DRIVEN: All compression config removed from server.
        // Compression is 100% controlled by SDK/client through collection metadata.
    }

    /// Update collection compression configuration
    pub async fn update_collection_compression(
        &self,
        identifier: &str,
        compression: &crate::proto::proximadb_v1::CompressionConfig,
    ) -> Result<CollectionServiceResponse> {
        let start_time = std::time::Instant::now();
        // SF2 (SC3-2b review): same full-record RMW discipline as
        // update_stats/update_collection — the policy mutation's ordering
        // contract rests on one writer at a time.
        let _rmw = COLLECTION_CONFIG_RMW.lock().await;

        // Get existing collection
        let collection = match self.collection(identifier).await? {
            Some(c) => c,
            None => {
                return Ok(CollectionServiceResponse::error(
                    format!(
                        "COLLECTION_NOT_FOUND: Collection '{}' not found",
                        identifier
                    ),
                    start_time.elapsed().as_micros() as i64,
                ));
            }
        };

        // Update compression config (now in storage_config)
        let mut updated_collection = collection.clone();
        if let Some(ref mut config) = updated_collection.config {
            // Ensure storage_config exists
            if config.storage_config.is_none() {
                config.storage_config = Some(crate::proto::proximadb_v1::StorageConfig::default());
            }
            // Set compression in storage_config
            if let Some(ref mut storage_config) = config.storage_config {
                storage_config.compression = Some(compression.algorithm);
            }
        }

        // Store the updated collection in the catalog (the sole authority).
        self.upsert_collection_catalog_asset(&updated_collection, None)
            .await
            .context("Failed to update collection compression in catalog")?;

        info!(
            "✅ Updated compression for collection {}: algorithm={}, level={:?}",
            identifier, compression.algorithm, compression.level
        );

        Ok(CollectionServiceResponse {
            success: true,
            collection: Some(updated_collection),
            storage_path: None,
            // error_message removed -  None,
            error_code: None,
            processing_time_us: start_time.elapsed().as_micros() as i64,
        })
    }

    /// Validate collection configuration
    #[allow(dead_code)]
    fn validate_collection_config(&self, config: &CollectionConfig) -> Result<()> {
        if config.name.is_empty() {
            return Err(anyhow::anyhow!("Collection name cannot be empty"));
        }

        if config.name.len() > 255 {
            return Err(anyhow::anyhow!(
                "Collection name too long (max 255 characters)"
            ));
        }

        if config.dimension == 0 {
            return Err(anyhow::anyhow!("Dimension must be positive"));
        }

        if config.dimension > 65536 {
            return Err(anyhow::anyhow!("Dimension too large (max 65536)"));
        }

        // Validate name contains only allowed characters
        if !config
            .name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
        {
            return Err(anyhow::anyhow!(
                "Collection name contains invalid characters (only alphanumeric, _, -, . allowed)"
            ));
        }

        Ok(())
    }

    /// ADR-031 Phase 4c: pre-mint the typed collection identity
    /// `(account_u32, namespace_u16, collection_u32)` for a collection being
    /// created, BEFORE storage-dir creation. The catalog mints the triple from
    /// the SAME `stable_ids` allocators that `create_table`→`mint_stable_identity`
    /// later hits, so the values are preserved (idempotent — no double-mint).
    ///
    /// Returns `None` when:
    /// * no `catalog_manager` is configured (no catalog to mint against), or
    /// * the tenant context carries no `account_id` (single-tenant / anonymous —
    ///   the typed path is account-rooted, so no account ⇒ no typed path), or
    /// * the catalog returns `None` (e.g. account registry lookup fails).
    ///
    /// `None` ⇒ the caller uses legacy byte-identical paths (mixed-read-safe).
    async fn mint_typed_identity_for_collection(
        &self,
        tenant_context: Option<&crate::storage::tenant::TenantContext>,
        collection_name: &str,
        _collection_id: &str,
    ) -> Option<CollectionIdentity> {
        let catalog_manager = self.catalog_manager.as_ref()?;
        // The account string is the SaaS billing/isolation tier — the top of the
        // typed path (`accounts/{base62}/…`). ADR-031 Phase 4 collapses the
        // tenant tier into the account, so the collection's owning TENANT (the
        // `tenant_context.tenant_id`, mirroring `collection_tenant_id` /
        // `tenant_id_of`) serves as the account string for typed-path minting.
        // `None` when no tenant context ⇒ single-tenant/anonymous ⇒ no typed
        // path (legacy, mixed-read-safe). NOTE: the network-layer
        // `MiddlewareTenantContext.account_id` is a separate (Phase 5) field not
        // threaded to the storage-layer `StorageTenantContext` yet; when it is,
        // prefer it here. Until then the tenant IS the account (Phase 4 collapse).
        // ADR-0083 rev2 D3: always mint — assign "default" account for
        // embedded/single-tenant paths (no tenant_context). The composite is
        // the sole collection identity, so it must be Some on every create.
        let account = tenant_context
            .map(|ctx| ctx.tenant_id.as_str())
            .filter(|a| !a.is_empty())
            .unwrap_or("default");
        // Derive the namespace_key the SAME way `upsert_collection_catalog_asset`
        // → `collection_table_identifier` scopes the asset: parse the qualified
        // name, default the namespace to `["default"]` when bare (mirrors
        // `collection_table_identifier`). Joined on '.' → the catalog namespace
        // key, so the pre-minted namespace u16 matches the schema's landing ns.
        let parsed = crate::catalog::TableIdentifier::parse(collection_name);
        let namespace_key = if parsed.namespace.is_empty() {
            "default".to_string()
        } else {
            parsed.namespace.join(".")
        };

        let catalog = catalog_manager.default_catalog().await.ok()?;
        let triple = catalog
            .mint_collection_typed_identity(account, &namespace_key)
            .await
            .ok()
            .flatten()?;
        Some(CollectionIdentity {
            account_id: triple.0,
            namespace_id: triple.1,
            collection_id: triple.2,
        })
    }

    /// Create storage directories for a new collection
    ///
    /// For multi-tenant deployments, paths are isolated under `{base}/tenants/{tenant_id}/`.
    /// ADR-031 Phase 4c: when `typed_identity` is `Some`, the data/wal/index
    /// dirs are created at the typed account-rooted path
    /// (`{base}/accounts/{base62}/…/{sub}`) instead of the tenant/legacy path.
    /// `None` keeps the legacy `StoragePath::collection_*_path_with_tenant`
    /// (byte-identical, mixed-read-safe).
    /// `tenant_base_location` is the TENANT-SCOPED base the storage
    /// assignment records (`{base}/tenants/{id}` for a tenant) — the base
    /// every reader and writer of the collection resolves paths from.
    async fn create_storage_directories(
        &self,
        tenant_base_location: &str,
        collection_name: &str,
        collection_uuid: &str,
        tenant_id: Option<&str>,
        typed_identity: Option<CollectionIdentity>,
    ) -> Result<Vec<StorageComponentType>> {
        let base_location = tenant_base_location;
        let tenant_info = tenant_id.unwrap_or("(default)");
        info!(
            "🏗️ Creating storage directories for collection {} (UUID: {}, tenant: {}) at base: {}",
            collection_name, collection_uuid, tenant_info, base_location
        );

        let mut created_components = Vec::new();

        // Typed layout `{base}/accounts/{a}/{n}/{c}/…` or legacy
        // `{base}/{uuid}/…`, both under the tenant-scoped base. The typed
        // branch used to take the RAW base while flush wrote under the
        // tenant-scoped one, so a tenant's typed directories were created
        // where no segment ever landed.
        let (write_buffer_dir, data_dir, indexes_dir) =
            collection_storage_dirs(base_location, collection_uuid, typed_identity);

        // Create directories
        if let Ok(filesystem) = self.filesystem_factory.get_filesystem(base_location) {
            // Create WAL directory
            if let Err(e) = filesystem.create_dir_all(&write_buffer_dir).await {
                warn!(
                    "⚠️ Failed to create WAL directory {}: {}",
                    write_buffer_dir, e
                );
            } else {
                debug!("Created WAL storage directory: {}", write_buffer_dir);
                created_components.push(StorageComponentType::Wal);
            }

            // Create data directory (flat structure for unified compaction framework)
            if let Err(e) = filesystem.create_dir_all(&data_dir).await {
                warn!("⚠️ Failed to create data directory {}: {}", data_dir, e);
            } else {
                debug!("Created data storage directory: {}", data_dir);
                created_components.push(StorageComponentType::Storage);
            }

            // Create index directory
            if let Err(e) = filesystem.create_dir_all(&indexes_dir).await {
                warn!("⚠️ Failed to create index directory {}: {}", indexes_dir, e);
            } else {
                debug!("Created index storage directory: {}", indexes_dir);
                created_components.push(StorageComponentType::Index);
            }
        } else {
            return Err(anyhow::anyhow!(
                "Failed to get filesystem for location: {}",
                base_location
            ));
        }

        info!(
            "🏗️ Created {} storage components for collection {} at location {}",
            created_components.len(),
            collection_name,
            base_location
        );
        Ok(created_components)
    }

    /// Clean up storage directories for a deleted collection
    async fn cleanup_storage_directories(
        &self,
        collection_name: &str,
        collection_uuid: &str,
    ) -> Result<usize> {
        info!(
            "🧹 Cleaning up storage directories for collection {} (UUID: {})",
            collection_name, collection_uuid
        );

        let mut cleaned_components = 0;

        // Get collection from the catalog to find storage assignment
        let collection = match self.collection(collection_uuid).await? {
            Some(col) => col,
            None => {
                warn!("Collection {} not found in metadata_info", collection_uuid);
                return Ok(0);
            }
        };

        if let Some(ref assignment) = collection.storage_assignment {
            let base_location = &assignment.base_location;
            // Every root the collection's storage may live under: the legacy
            // `{base}/{uuid}` and, when the assignment records a typed
            // identity, `{base}/accounts/{a}/{n}/{c}`. Dropping only the
            // legacy root left a typed collection's segments in place.
            let roots = collection_storage_roots(
                base_location,
                collection_uuid,
                crate::storage::trait_components::path_resolver::typed_identity_from_storage_assignment(
                    Some(assignment),
                ),
            );

            match self.filesystem_factory.get_filesystem(base_location) {
                Ok(filesystem) => {
                    let mut all_clean = true;
                    for collection_dir in &roots {
                        match filesystem.exists(collection_dir).await {
                            Ok(true) => match filesystem.remove_dir_all(collection_dir).await {
                                Ok(_) => {
                                    info!("✅ Deleted collection directory: {}", collection_dir)
                                }
                                Err(e) => {
                                    error!(
                                        "❌ Failed to delete collection directory {}: {}",
                                        collection_dir, e
                                    );
                                    return Err(anyhow::anyhow!(
                                        "Failed to delete collection directory: {}",
                                        e
                                    ));
                                }
                            },
                            Ok(false) => debug!(
                                "📂 Collection directory {} does not exist (already cleaned up)",
                                collection_dir
                            ),
                            Err(e) => {
                                all_clean = false;
                                warn!(
                                    "⚠️ Failed to check existence of collection directory {}: {}",
                                    collection_dir, e
                                );
                            }
                        }
                    }
                    if all_clean {
                        cleaned_components = 3; // All components deleted
                        for collection_dir in &roots {
                            let purged =
                                crate::storage::engines::sst::core::purge_warm_tier_prefix(
                                    collection_dir,
                                )
                                .await;
                            if purged > 0 {
                                debug!(
                                    collection_dir,
                                    purged,
                                    "Reclaimed retired collection entries from the PAX warm tier"
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("⚠️ Failed to get filesystem for {}: {}", base_location, e);
                }
            }
        } else {
            debug!(
                "📂 No storage assignment found for collection {} (may not have been created)",
                collection_name
            );
        }

        info!(
            "🧹 Cleaned up {} storage components for collection {}",
            cleaned_components, collection_name
        );
        Ok(cleaned_components)
    }

    async fn upsert_collection_catalog_asset(
        &self,
        collection: &Collection,
        typed_identity: Option<CollectionIdentity>,
    ) -> Result<()> {
        let Some(catalog_manager) = &self.catalog_manager else {
            return Ok(());
        };

        let Some(config) = collection.config.as_ref() else {
            return Ok(());
        };

        let catalog = catalog_manager.default_catalog().await?;
        // TD-CAT-2b (S3a): scope the asset under a tenant-prefixed namespace when
        // the gate is on, so two tenants' identically-named namespaces don't
        // collide on the bare key. Done once here, then every op below
        // (namespace_exists / create / table_exists / get / drop / create_table)
        // is consistently tenant-scoped.
        let identifier = Self::tenant_scoped_identifier(
            self.tenant_namespaces_on(),
            Self::collection_tenant_id(collection).as_deref(),
            crate::storage::metadata::collection_mapping::collection_table_identifier(config),
        );

        if !catalog.namespace_exists(&identifier.namespace).await? {
            // TD-CAT-2a (audit gap G6): record the owning tenant on the persisted
            // namespace via the tenant-aware constructor, instead of the
            // tenant-less `create_namespace` that left `CatalogNamespace.tenant_id`
            // = None. The tenant is derived the same way every storage/network
            // boundary derives it (`collection_tenant_id`: `tenant:` tag / owner).
            // Additive + inert until consumed: the only behavior-shifting reader,
            // DrPathBuilder index-location resolution, is gated default-OFF
            // (`PROXIMADB_INDEX_CATALOG_PATHS`), and catalog asset paths are still
            // flat/name-keyed — so this records identity now for the
            // tenant-prefixed-paths slice (TD-CAT-2b) without shifting any path.
            // Without tenant-prefixed namespaces (TD-CAT-2b gate off) a
            // namespace is shared by every tenant's collections, so it records
            // no single owner even though the collection itself does.
            let namespace_tenant = if self.tenant_namespaces_on() {
                Self::collection_tenant_id(collection)
            } else {
                None
            };
            catalog
                .create_namespace_for_tenant(
                    &identifier.namespace,
                    std::collections::HashMap::new(),
                    namespace_tenant.as_deref(),
                )
                .await?;
        }

        let mut schema =
            crate::storage::metadata::collection_mapping::catalog_schema_from_collection(
                collection,
            )?;
        // ADR-031 allocator unification: pre-set `schema.object_id` from the
        // collection's numeric id so `create_table` ADOPTS it
        // (`mint_object_id(Some)` raises the floor + returns the id) instead of
        // minting a separate one. Result: `schema.object_id == collection.id`'s
        // oid — one identity per collection, not two divergent ones.
        // Only the fresh-create path (`create_table(schema)` below) is affected;
        // the existing-table path preserves its already-minted object_id. Legacy
        // UUID collection.ids don't parse → leave None (create_table mints fresh,
        // un-unified — mixed-read-safe).
        if let Ok(oid) = collection.id.parse::<u64>() {
            schema.object_id = Some(oid);
        }
        // ADR-031 Phase 4c: stamp the pre-minted typed identity onto the schema
        // so the persisted `stable_namespace_id`/`stable_collection_id` match
        // the typed DATA path that `create_storage_directories` just created.
        // `create_table`→`mint_stable_identity` then preserves these (idempotent
        // via `resolve_typed_triple`'s `existing_*` short-circuit — no double-mint).
        // `None` (update/import paths, or env OFF) leaves them unset → legacy path
        // (mixed-read-safe). This is the ONLY channel from the manager's pre-mint
        // into the catalog schema (the proto `Collection` carries no properties map).
        if let Some(id) = typed_identity {
            schema.stable_namespace_id = Some(id.namespace_id);
            schema.stable_collection_id = Some(id.collection_id);
        }
        // ADR-047 / TD-TBL-1 + ADR-048 P1: the fresh `schema` above is rebuilt
        // from the narrow `Collection` config, which cannot carry the canonical
        // ProximaType columns. Capture any existing canonical schema — both the
        // legacy `collection.canonical_schema` property AND the typed 200+
        // columns (ADR-048) — so they can be re-attached below; otherwise an
        // unrelated `update_collection` (e.g. an index-param tweak) would
        // silently drop the canonical schema.
        let mut preserved_canonical_schema: Option<String> = None;
        let mut preserved_canonical_columns: Vec<_> = Vec::new();
        if catalog.table_exists(&identifier).await? {
            let mut existing = catalog.get_table(&identifier).await?;
            preserved_canonical_schema = existing
                .properties
                .get(crate::storage::metadata::collection_mapping::CANONICAL_SCHEMA_PROPERTY)
                .cloned();
            // TD-GRPCCREATE-1: include declarations folded into reserved columns,
            // which the rebuild below would otherwise lose.
            preserved_canonical_columns =
                crate::storage::metadata::collection_mapping::declared_canonical_columns(
                    &existing.columns,
                );
            if existing
                .properties
                .get("asset.kind")
                .is_none_or(|kind| kind != "collection")
            {
                existing
                    .properties
                    .insert("asset.kind".to_string(), "collection".to_string());
                existing.properties.insert(
                    proximadb_catalog::table_properties::COLLECTION_PROVENANCE_PROPERTY.to_string(),
                    proximadb_catalog::table_properties::COLLECTION_PROVENANCE_VALUE.to_string(),
                );
                existing
                    .properties
                    .insert("asset.capability.vector".to_string(), "true".to_string());
                existing
                    .properties
                    .insert("collection.id".to_string(), collection.id.clone());
                existing
                    .properties
                    .insert("collection.name".to_string(), config.name.clone());
                existing
                    .properties
                    .insert("vector.dimension".to_string(), config.dimension.to_string());
                existing.updated_at_ms = collection.updated_at / 1000;
                if existing.storage_layouts.is_empty() {
                    existing.storage_layouts = schema.storage_layouts.clone();
                }
                if existing.location.is_none() {
                    existing.location = schema.location.clone();
                }

                // TD-GRPCCREATE-1 validates first; TD-CATREPLACE-1 replaces in
                // place — no drop window in which the collection has no table.
                catalog.replace_table(&identifier, existing).await?;
                return Ok(());
            }
        }
        // TD-GRPCCREATE-1: the rebuilt reserved band can now hold a name the
        // declared schema also has (a column turned filterable). Merge with the
        // same rule as `set_collection_schema_columns`, and validate BEFORE
        // dropping the existing table: a rejected schema must never cost the
        // collection its table.
        if !preserved_canonical_columns.is_empty() {
            crate::storage::metadata::collection_mapping::merge_canonical_band(
                &mut schema,
                preserved_canonical_columns,
            )?;
        }
        if let Some(canonical) = preserved_canonical_schema {
            schema.properties.insert(
                crate::storage::metadata::collection_mapping::CANONICAL_SCHEMA_PROPERTY.to_string(),
                canonical,
            );
        }
        // TD-GRPCCREATE-1 validates first (inside replace_table);
        // TD-CATREPLACE-1 replaces in place — no drop window.
        catalog.replace_table(&identifier, schema).await?;
        Ok(())
    }

    async fn drop_collection_catalog_asset(&self, collection: &Collection) -> Result<()> {
        let Some(catalog_manager) = &self.catalog_manager else {
            return Ok(());
        };

        let Some(config) = collection.config.as_ref() else {
            return Ok(());
        };

        let catalog = catalog_manager.default_catalog().await?;
        // TD-CAT-2b (S3a): mirror the write-side scoping so a drop targets the
        // same tenant-prefixed asset the upsert created.
        let identifier = Self::tenant_scoped_identifier(
            self.tenant_namespaces_on(),
            Self::collection_tenant_id(collection).as_deref(),
            crate::storage::metadata::collection_mapping::collection_table_identifier(config),
        );
        if catalog.table_exists(&identifier).await? {
            let _ = catalog.drop_table(&identifier, false).await?;
        }
        Ok(())
    }

    async fn collection_from_catalog_asset(&self, identifier: &str) -> Result<Option<Collection>> {
        let Some(catalog_manager) = &self.catalog_manager else {
            return Ok(None);
        };
        read_collection_asset(catalog_manager, identifier).await
    }

    async fn list_collections_from_catalog(&self) -> Result<Vec<Collection>> {
        let Some(catalog_manager) = &self.catalog_manager else {
            return Ok(Vec::new());
        };
        read_collections_from_catalog(catalog_manager).await
    }

    /// Generate unique collection ID using UUIDs.
    ///
    /// Base62 timestamp IDs are still accepted as legacy identifiers by lookup paths, but new
    /// catalog assets use UUID strings so identity is opaque, non-time-leaking, and compatible
    /// with catalog/schema UUID fields across SDKs and embedded mode.
    async fn generate_unique_collection_id(&self) -> Result<String> {
        // ADR-031: the collection ID is the stable object_id (u64) as a decimal
        // string — NOT a UUID. When xCatalog is wired, its object-id sequence is
        // the one authority shared with relational DDL. This avoids the
        // time-of-check/time-of-use race inherent in sampling max_object_id and
        // then minting from an independent collection allocator.
        if let Some(catalog_manager) = &self.catalog_manager {
            let catalog = catalog_manager.default_catalog().await?;
            if let Some(object_id) = catalog.allocate_object_id().await? {
                return Ok(object_id.to_string());
            }

            // Compatibility for federated/external catalogs that do not own a
            // native object-id sequence. Refresh the legacy allocator from any
            // observable persisted floor before minting locally.
            if let Some(max_object_id) = catalog.max_object_id().await? {
                recover_collection_id_floor(max_object_id);
            }
        }

        // No native catalog authority: retain the legacy monotonic allocator.
        // base62 is reserved for path segments only (DrCollectionPath), not the
        // collection.id variable.
        Ok(generate_numeric_collection_id())
    }
}

/// System-wide monotonic allocator for collection object_ids (ADR-031).
/// One sequence for all collections across all tenants — globally unique,
/// never reused. Recovered on restart by scanning existing collection IDs.
static COLLECTION_ID_ALLOCATOR: std::sync::OnceLock<proximadb_catalog::id_allocator::IdAllocator> =
    std::sync::OnceLock::new();

/// Generate a stable, monotonic collection ID as the **decimal** `object_id`.
///
/// ADR-031 representation rule: the `object_id` is `u64` in-memory (numeric —
/// the canonical identity, never stringified for keying/lookup). Its
/// client-facing string form (the `collection.id` API field) is the **decimal**
/// `object_id` — numeric, opaque, JSON-safe. The **base62** encoding is reserved
/// strictly for object-store *path segments* (`DrCollectionPath`), where
/// zero-padded base62 gives lexicographic S3 LIST order; it is NOT used for the
/// `collection.id` variable.
fn generate_numeric_collection_id() -> String {
    let allocator =
        COLLECTION_ID_ALLOCATOR.get_or_init(proximadb_catalog::id_allocator::IdAllocator::default);
    allocator.allocate().to_string()
}

/// ADR-031: recover the collection ID allocator floor from existing collections.
///
/// Call at startup (after scanning existing collection IDs) with
/// `max(existing decimal object_id)` to prevent ID reuse after restart.
/// The `OnceLock` allocator is initialized (if not already) and its floor raised
/// so subsequent `generate_numeric_collection_id` calls never produce an ID
/// that's already on disk.
///
/// **Usage** (in server/embedded startup):
/// ```ignore
/// let max_existing = collections.iter()
///     .filter_map(|c| c.id.parse::<u64>().ok())
///     .max()
///     .unwrap_or(0);
/// recover_collection_id_floor(max_existing);
/// ```
pub fn recover_collection_id_floor(max_existing: u64) {
    let allocator =
        COLLECTION_ID_ALLOCATOR.get_or_init(proximadb_catalog::id_allocator::IdAllocator::default);
    allocator.raise_floor(max_existing + 1);
}

#[cfg(test)]
mod adr031_collection_id_tests {
    use super::{generate_numeric_collection_id, recover_collection_id_floor};

    #[test]
    fn collection_id_is_numeric_not_uuid() {
        let id = generate_numeric_collection_id();
        // ADR-031: collection.id is the decimal object_id (u64) — NOT a UUID.
        // UUIDs contain dashes (`-`); a decimal u64 is `[0-9]+` with no dashes.
        assert!(
            !id.contains('-'),
            "collection ID must not be a UUID (no dashes), got: {id}"
        );
        // Must parse as a valid decimal u64.
        let parsed: u64 = id
            .parse()
            .expect("collection ID must be a valid decimal object_id");
        assert!(parsed > 0, "object_id must be positive, got: {parsed}");
    }

    #[test]
    fn collection_ids_are_monotonic() {
        let a = generate_numeric_collection_id();
        let b = generate_numeric_collection_id();
        let oid_a: u64 = a.parse().expect("parse a");
        let oid_b: u64 = b.parse().expect("parse b");
        assert!(
            oid_b > oid_a,
            "consecutive object_ids must be monotonic: {oid_a} -> {oid_b}"
        );
    }

    #[test]
    fn recover_collection_id_floor_prevents_reuse() {
        // ADR-031: after restart, the allocator must not reuse IDs below the
        // recovered floor. Simulate: raise floor to 100000, then allocate —
        // the result must parse to > 100000.
        recover_collection_id_floor(100_000);
        let id = generate_numeric_collection_id();
        let oid: u64 = id.parse().expect("parse");
        assert!(
            oid > 100_000,
            "after recovery to floor=100001, allocated ID must be > 100000, got {oid}"
        );
    }
}

/// Per-tenant system-catalog hot-cache pool (ADR-035 D2 / TD-SC-1b): a shared
/// byte budget across tenants with a 1 MB/tenant ceiling. 256 MB ⇒ ~256
/// concurrently-active tenants at the ceiling; a single-tenant deployment uses
/// only the one `"default"` bucket (≤1 MB).
const SYSCAT_CACHE_POOL_BYTES: u64 = 256 * 1024 * 1024;

/// Read a single collection asset from the catalog by name-or-UUID. Free function
/// (depends only on the catalog manager) so it can back BOTH the
/// `CollectionService` method and the hot-cache's inner source **without a
/// self-referential `Arc` cycle** (the cache must not hold the service that holds
/// the cache).
async fn read_collection_asset(
    catalog_manager: &CatalogManager,
    identifier: &str,
) -> Result<Option<Collection>> {
    if let Ok((catalog, table_id)) = catalog_manager.resolve_table(identifier).await
        && catalog.table_exists(&table_id).await.unwrap_or(false)
    {
        let schema = catalog.get_table(&table_id).await?;
        match crate::storage::metadata::collection_mapping::collection_from_catalog_schema(
            &table_id, &schema,
        ) {
            Ok(Some(collection)) => return Ok(Some(collection)),
            Ok(None) => {}
            Err(error) => {
                tracing::error!(table = ?table_id, %error, "unreadable collection catalog entry; skipped");
            }
        }
    }

    for collection in read_collections_from_catalog(catalog_manager).await? {
        if collection.id == identifier {
            return Ok(Some(collection));
        }
        if let Some(config) = &collection.config
            && config.name == identifier
        {
            return Ok(Some(collection));
        }
    }

    Ok(None)
}

/// List every collection asset across namespaces (the expensive 1+N+M path the
/// hot cache amortises). Free function — catalog-manager-only, see
/// [`read_collection_asset`].
async fn read_collections_from_catalog(
    catalog_manager: &CatalogManager,
) -> Result<Vec<Collection>> {
    let catalog = match catalog_manager.default_catalog().await {
        Ok(catalog) => catalog,
        Err(_) => return Ok(Vec::new()),
    };

    let mut namespaces: Vec<Vec<String>> = catalog
        .list_namespaces(None)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|namespace| namespace.levels)
        .collect();
    if !namespaces.iter().any(|namespace| namespace == &["default"]) {
        namespaces.push(vec!["default".to_string()]);
    }

    let mut collections: Vec<Collection> = Vec::new();
    for namespace in namespaces {
        let table_ids = match catalog.list_tables(&namespace).await {
            Ok(table_ids) => table_ids,
            Err(_) => continue,
        };

        for table_id in table_ids {
            let schema = match catalog.get_table(&table_id).await {
                Ok(schema) => schema,
                Err(_) => continue,
            };
            // One unreadable entry never takes down the listing (or startup).
            let collection =
                match crate::storage::metadata::collection_mapping::collection_from_catalog_schema(
                    &table_id, &schema,
                ) {
                    Ok(Some(collection)) => collection,
                    Ok(None) => continue,
                    Err(error) => {
                        tracing::error!(table = ?table_id, %error, "unreadable collection catalog entry; skipped");
                        continue;
                    }
                };
            collections.push(collection);
        }
    }

    // Entries that claim the same collection id must agree on its owner; if
    // they do not, none is trusted. Otherwise the first is used.
    let mut owners: std::collections::HashMap<String, HashSet<Option<String>>> =
        std::collections::HashMap::new();
    for collection in &collections {
        owners
            .entry(collection.id.clone())
            .or_default()
            .insert(proximadb_tenant::effective_owner(collection));
    }
    let mut seen_ids = HashSet::new();
    collections.retain(|collection| {
        if owners.get(&collection.id).is_some_and(|set| set.len() > 1) {
            tracing::error!(
                collection = %collection.id,
                "catalog entries record this collection id with different owners; none is used"
            );
            return false;
        }
        seen_ids.insert(collection.id.clone())
    });

    Ok(collections)
}

/// TD-SC-4 (S4): idempotently pre-create a tenant's bare-minimum system-catalog
/// skeleton at signup — the tenant's system namespace (`[tenant, "default"]`,
/// the same tenant-prefixed shape S3a stores collections under), recorded with
/// its owning `tenant_id`. Pre-creating it means the first collection/list for a
/// freshly-signed-up tenant doesn't hit a cold/missing namespace.
///
/// Idempotent: a re-signup (namespace already present) is a no-op, and a partial
/// failure is safe to retry (the `namespace_exists` guard converges). Free
/// function (catalog-manager-only) so it can be called from any onboarding path
/// without holding the `CollectionService`. **No user tables are created.**
async fn provision_tenant_system_catalog(
    catalog_manager: &CatalogManager,
    tenant: &str,
) -> Result<()> {
    if tenant.is_empty() {
        return Ok(());
    }
    let catalog = catalog_manager.default_catalog().await?;
    // Bare-minimum skeleton: the tenant's default namespace, tenant-prefixed to
    // match where S3a stores this tenant's collections (no collision with other
    // tenants' identically-named namespaces).
    let namespace = vec![tenant.to_string(), "default".to_string()];
    if !catalog.namespace_exists(&namespace).await? {
        catalog
            .create_namespace_for_tenant(&namespace, std::collections::HashMap::new(), Some(tenant))
            .await?;
    }
    Ok(())
}

/// Hot-cache inner source: reads collection metadata straight from the catalog.
/// Holds `Arc<CatalogManager>` (not the `CollectionService`), so the cache and the
/// service do not form an `Arc` cycle.
struct CatalogAssetSource {
    catalog_manager: Arc<CatalogManager>,
}

#[async_trait::async_trait]
impl crate::catalog::syscat_cache::CatalogMetadataSource for CatalogAssetSource {
    async fn fetch(&self, _tenant_id: &str, name: &str) -> Result<Option<Collection>> {
        read_collection_asset(&self.catalog_manager, name).await
    }
}

impl std::fmt::Debug for CollectionService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollectionService")
            .field("assignment_service", &"AssignmentService")
            .field("filesystem_factory", &"FilesystemFactory")
            .field("index_config_cache_info", &"HashMap<String, IndexConfig>")
            .finish()
    }
}

/// Response for collection operations - includes the full collection data
#[derive(Debug, Clone)]
pub struct CollectionServiceResponse {
    /// Whether the operation completed successfully.
    pub success: bool,
    /// The collection affected by the operation, if applicable (proto-first architecture).
    pub collection: Option<Collection>,
    /// Filesystem path where the collection's data is stored.
    pub storage_path: Option<String>,
    /// Machine-readable error code when the operation fails.
    pub error_code: Option<String>,
    /// Wall-clock time taken to process the request, in microseconds.
    pub processing_time_us: i64,
}

impl CollectionServiceResponse {
    /// Create success response
    pub fn success(
        _collection_uuid: String,
        storage_path: String,
        processing_time_us: i64,
    ) -> Self {
        Self {
            success: true,
            collection: None, // Collection should be passed in if needed
            storage_path: Some(storage_path),
            error_code: None,
            processing_time_us,
        }
    }

    /// Create success response with collection
    pub fn success_with_collection(
        collection: Collection,
        storage_path: String,
        processing_time_us: i64,
    ) -> Self {
        Self {
            success: true,
            collection: Some(collection),
            storage_path: Some(storage_path),
            error_code: None,
            processing_time_us,
        }
    }

    /// Create error response
    pub fn error(error_code: String, processing_time_us: i64) -> Self {
        Self {
            success: false,
            collection: None,
            storage_path: None,
            error_code: Some(error_code),
            processing_time_us,
        }
    }
}

/// Builder for collection service with dependencies
pub struct CollectionServiceBuilder {
    /// Optional storage configuration to set during construction
    storage_config: Option<StorageConfig>,
}

impl CollectionServiceBuilder {
    /// Create a new builder with no dependencies configured.
    pub fn new() -> Self {
        Self {
            storage_config: None,
        }
    }

    /// Set the storage configuration (data paths, engine settings, etc.).
    pub fn with_storage_config(mut self, config: StorageConfig) -> Self {
        self.storage_config = Some(config);
        self
    }

    /// Consume the builder and construct a [`CollectionService`].
    pub async fn build(self) -> Result<CollectionService> {
        let storage_config = self.storage_config.unwrap_or_default();

        CollectionService::new(storage_config).await
    }
}

impl Default for CollectionServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Dependency-inversion seam (Slice D): storage holds an
/// `Arc<dyn CollectionMetadataPort>` and never references this service crate.
/// `CollectionService` is the concrete implementor; the composition root
/// injects it. Delegates to the inherent metadata-fetch path.
#[async_trait::async_trait]
impl proximadb_storage_ports::CollectionMetadataPort for CollectionService {
    async fn collection(&self, identifier: &str) -> Result<Option<Collection>> {
        self.get_native_proto(identifier).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default tenant's request context (tenant-less callers act as it).
    fn default_tenant_ctx() -> crate::storage::tenant::TenantContext {
        crate::storage::tenant::TenantContext::for_tenant_id(proximadb_tenant::DEFAULT_TENANT)
    }
    use crate::catalog::TableIdentifier;
    use crate::proto::proximadb_v1::CollectionConfig;
    use crate::proto::proximadb_v1::{FilterableDataType, IndexingAlgorithm};

    /// The standard valid-collection preamble (dimension 128, metric 1,
    /// engine SST, primary index "default") — the exact values the
    /// exhaustive literals used; tests override what they exercise.
    fn collection_config(name: &str) -> CollectionConfig {
        CollectionConfig {
            name: name.to_string(),
            embedding_config: None,
            dimension: 128,
            distance_metric: Some(1),
            storage_engine: Some(2), // SST — VIPER is no longer selectable for new collections (ADR-093 S2)
            filterable_columns: vec![],
            index_configs: vec![],
            quantization: None,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            storage_config: None,
            description: Some("Test collection".to_string()),
            tags: vec![],
            owner: Some("test".to_string()),
            embedding_models: vec![],
            record_schema: None,
            enable_proxima_record: None,
            text_columns: vec![],
            text_storage_configs: vec![],
            enable_dual_use_embeddings: None,
            canonical_embedding_precision: None,
            permitted_principals: vec![],
            index_policy: None,
            pax_vector_quant: None,
            storage_class: None,
            policy_generation: None,
            default_access_tier: None,
        }
    }

    /// TD-GRPCSCHEMA-2: a schema update persists the record-mode block.
    /// Before, `update_collection`'s field merge dropped `enable_proxima_record`
    /// and `record_schema`, so record mode could never be enabled after create.
    #[tokio::test]
    async fn schema_update_persists_record_mode() -> Result<()> {
        let (svc, mgr, _temp_dir) = service_in_tempdir().await?;

        let name = "grpcschema2_record";
        assert!(
            svc.create_collection(&collection_config(name))
                .await?
                .success
        );
        let before = svc.collection(name).await?.expect("present");
        assert_ne!(
            before.config.as_ref().unwrap().enable_proxima_record,
            Some(true),
            "precondition: record mode is off after create"
        );

        let mut update = collection_config(name);
        update.enable_proxima_record = Some(true);
        update.record_schema = Some(crate::proto::proximadb_v1::RecordSchemaConfig {
            schema_id: "s1".to_string(),
            schema_version: "1".to_string(),
            enforcement: 0,
            auto_evolve: false,
            columns: vec![],
        });
        assert!(svc.update_collection(name, Some(update)).await?.success);

        let after = svc.collection(name).await?.expect("present");
        let config = after.config.as_ref().unwrap();
        assert_eq!(
            config.enable_proxima_record,
            Some(true),
            "the record-mode switch must persist through the merge"
        );
        assert!(
            config.record_schema.is_some(),
            "the record-schema block must persist through the merge"
        );
        Ok(())
    }

    /// A collection service over a fresh native catalog in a tempdir.
    async fn service_in_tempdir()
    -> Result<(CollectionService, Arc<CatalogManager>, tempfile::TempDir)> {
        let temp_dir = tempfile::tempdir()?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let mgr = Arc::new(CatalogManager::new());
        mgr.create_native_catalog("default", &temp_path).await?;
        let svc = CollectionService::new(StorageConfig::default())
            .await?
            .with_catalog_manager(mgr.clone());
        Ok((svc, mgr, temp_dir))
    }

    /// TD-CATREPLACE-1: a schema update (the record-schema switch
    /// `apply_schema_update` always sets) carries the authoritative FULL
    /// filterable list — an empty list REMOVES the columns. Before, an empty
    /// list was ignored, so a removed filterable column kept being served and
    /// could not be retyped or removed afterwards.
    #[tokio::test]
    async fn schema_update_clears_filterable_columns_it_empties() -> Result<()> {
        use crate::proto::proximadb_v1::{FilterableColumnSpec, FilterableDataType};

        let (svc, mgr, _temp_dir) = service_in_tempdir().await?;

        // The schema-update intent marker `apply_schema_update` always writes.
        let record_schema = || crate::proto::proximadb_v1::RecordSchemaConfig {
            schema_id: "s1".to_string(),
            schema_version: "1".to_string(),
            enforcement: 0,
            auto_evolve: false,
            columns: vec![],
        };
        let int_x = || FilterableColumnSpec {
            name: "x".to_string(),
            data_type: FilterableDataType::FilterableInteger as i32,
            indexed: true,
            supports_range: true,
            estimated_cardinality: None,
        };
        let mut config = collection_config("catreplace_removal");
        config.filterable_columns = vec![int_x()];
        assert!(svc.create_collection(&config).await?.success);

        // A schema update that no longer declares `x` filterable — marked by
        // the record-schema block `apply_schema_update` always sets (NOT by
        // `enable_proxima_record`: a raw flag flip must not clear columns).
        let mut update = collection_config("catreplace_removal");
        update.filterable_columns = vec![];
        update.enable_proxima_record = Some(true);
        update.record_schema = Some(record_schema());
        assert!(
            svc.update_collection("catreplace_removal", Some(update))
                .await?
                .success
        );

        let after = svc
            .collection("catreplace_removal")
            .await?
            .expect("collection present");
        assert!(
            after
                .config
                .as_ref()
                .expect("config")
                .filterable_columns
                .is_empty(),
            "an emptied filterable list must remove the columns, got {:?}",
            after.config.as_ref().unwrap().filterable_columns
        );

        // A RAW flag flip (enable_proxima_record set, no record-schema block)
        // must NOT clear the columns — only a schema update carries the
        // authoritative list (review M3).
        let mut restore = collection_config("catreplace_removal");
        restore.filterable_columns = vec![int_x()];
        restore.enable_proxima_record = Some(true);
        assert!(
            svc.update_collection("catreplace_removal", Some(restore))
                .await?
                .success
        );
        let mut flag_flip = collection_config("catreplace_removal");
        flag_flip.filterable_columns = vec![];
        flag_flip.enable_proxima_record = Some(true);
        assert!(
            svc.update_collection("catreplace_removal", Some(flag_flip))
                .await?
                .success
        );
        let after_flip = svc
            .collection("catreplace_removal")
            .await?
            .expect("collection present");
        assert_eq!(
            after_flip.config.as_ref().unwrap().filterable_columns.len(),
            1,
            "a raw enable_proxima_record flip must keep the filterable columns"
        );

        // Clear it again via a schema update for the catalog-band assertions.
        let mut clear = collection_config("catreplace_removal");
        clear.filterable_columns = vec![];
        clear.enable_proxima_record = Some(true);
        clear.record_schema = Some(record_schema());
        assert!(
            svc.update_collection("catreplace_removal", Some(clear))
                .await?
                .success
        );

        // The catalog table's reserved filterable band follows.
        // `get_catalog` (not the counted `default_catalog`): the anonymous-
        // identity guard ratchets direct default-catalog uses in this file, and
        // a test-side lookup must not spend one.
        let catalog = mgr.get_catalog("default").await?;
        let identifier = crate::storage::metadata::collection_mapping::collection_table_identifier(
            after.config.as_ref().unwrap(),
        );
        let table = catalog.get_table(&identifier).await?;
        // The reserved filterable band is ids [100, 200): assert by RANGE, not by a
        // bound the column ids can never satisfy (round-2 review: the old `< 100`
        // was vacuous — filterable columns start at 100).
        let gone = !table.columns.iter().any(|c| {
            c.name == "x"
                && (100..crate::storage::metadata::collection_mapping::CANONICAL_COLUMN_ID_BASE)
                    .contains(&c.id)
        });
        assert!(
            gone,
            "the reserved filterable column must be gone, got {:?}",
            table
                .columns
                .iter()
                .map(|c| (c.id, c.name.clone()))
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_collection_validation() -> Result<()> {
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("Failed to create collection service for test")?;

        // Valid config
        let valid_config = collection_config("valid_collection");

        // Test create with valid config
        let result = service
            .create_collection(&valid_config)
            .await
            .context("Failed to create valid collection")?;
        assert!(result.success);

        // Test empty name
        let empty_name = CollectionConfig {
            name: "".to_string(),
            ..valid_config.clone()
        };
        let result = service
            .create_collection(&empty_name)
            .await
            .context("Failed to create collection with empty name")?;
        assert!(!result.success);
        assert!(
            result
                .error_code
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Error code missing"))?
                .contains("INVALID_NAME"),
            "Error code should contain INVALID_NAME, got: {:?}",
            result.error_code
        );

        // Short names are valid SQL/ANSI identifiers — the vestigial 8-char floor was
        // removed (TPC-H `part`/`orders` etc. need short table names). "short" now succeeds.
        let short_name = CollectionConfig {
            name: "short".to_string(),
            ..valid_config.clone()
        };
        let result = service
            .create_collection(&short_name)
            .await
            .context("Failed to create collection with short name")?;
        assert!(
            result.success,
            "short names are now valid; got error: {:?}",
            result.error_code
        );

        // Test exactly 8 characters (should pass)
        let eight_chars = CollectionConfig {
            name: "exactly8".to_string(),
            ..valid_config.clone()
        };
        let result = service
            .create_collection(&eight_chars)
            .await
            .context("Failed to create collection with 8-character name")?;
        assert!(result.success);

        // Test invalid dimension
        let invalid_dimension = CollectionConfig {
            name: "valid_dimension_test".to_string(),
            dimension: 0,
            ..valid_config.clone()
        };
        let result = service
            .create_collection(&invalid_dimension)
            .await
            .context("Failed to create collection with invalid dimension")?;
        assert!(!result.success);
        assert!(
            result
                .error_code
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Error code missing"))?
                .contains("INVALID_DIMENSION"),
            "Error code should contain INVALID_DIMENSION, got: {:?}",
            result.error_code
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_collection_name_length_validation() -> Result<()> {
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("Failed to create collection service for test")?;

        // Test cases for collection name length
        let test_cases = vec![
            ("", false, "INVALID_NAME"), // Empty name
            // Short names are valid SQL/ANSI identifiers (no artificial 8-char floor);
            // required for relational tables over pgwire (TPC-H `part`, `orders`, ...).
            ("a", true, ""),                              // 1 char
            ("abc", true, ""),                            // 3 chars
            ("seven77", true, ""),                        // 7 chars
            ("exactly8", true, ""),                       // 8 chars (valid)
            ("ninechars", true, ""),                      // 9 chars (valid)
            ("this_is_a_long_collection_name", true, ""), // Long name (valid)
        ];

        for (name, should_succeed, expected_error_code) in test_cases {
            let config = collection_config(name);

            let result = service
                .create_collection(&config)
                .await
                .context(format!("Failed to create collection with name '{}'", name))?;

            assert_eq!(
                result.success, should_succeed,
                "Name '{}' validation failed: expected success={}, got success={}",
                name, should_succeed, result.success
            );

            if !should_succeed {
                assert!(
                    result
                        .error_code
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Error code missing for name '{}'", name))?
                        .contains(expected_error_code),
                    "Name '{}' error code mismatch: expected to contain '{}', got '{:?}'",
                    name,
                    expected_error_code,
                    result.error_code
                );
            }
        }

        Ok(())
    }

    /// TDD: short SQL/ANSI table names (e.g. TPC-H `part`) create successfully,
    /// resolve to a DISTINCT opaque UUID id (not the name), and round-trip
    /// name<->id. Proves the redesign: name and id are separate, unambiguous
    /// namespaces with NO length dependence (the old 8-char floor is gone), and
    /// resolution is name-authoritative.
    /// Coherence (invariant #16b): create / update / delete must bump the
    /// corpus version **even without a threaded tenant context** (the common
    /// single-tenant/anonymous case), so corpus-version-keyed caches invalidate.
    /// Previously the bump was tenant-context-gated and thus inert in production.
    #[tokio::test]
    async fn writes_bump_corpus_version_without_tenant_context() -> Result<()> {
        use crate::catalog::CorpusVersionRegistry;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &temp_path)
            .await
            .context("Failed to create test xCatalog")?;
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("collection service")?
            .with_catalog_manager(catalog_manager.clone());

        // Unique name isolates the process-global CorpusVersionRegistry.
        let name = "cv_bump_probe";
        let tenant = CollectionService::DEFAULT_VERSION_TENANT;
        let v0 = CorpusVersionRegistry::global().current(tenant, name).await;

        // create (inherent, tenant-less → effective tenant "default")
        let config = CollectionConfig {
            name: name.to_string(),
            dimension: 8,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };
        assert!(
            service
                .create_collection(&config)
                .await
                .context("create")?
                .success
        );
        let v_create = CorpusVersionRegistry::global().current(tenant, name).await;
        assert!(
            v_create > v0,
            "create must bump corpus version without a tenant context ({v0} -> {v_create})"
        );

        // update (tenant-less)
        let update = CollectionConfig {
            description: Some("updated".to_string()),
            ..Default::default()
        };
        assert!(
            service
                .update_collection(name, Some(update))
                .await
                .context("update")?
                .success
        );
        let v_update = CorpusVersionRegistry::global().current(tenant, name).await;
        assert!(
            v_update > v_create,
            "update must bump ({v_create} -> {v_update})"
        );

        // delete (tenant-less)
        service
            .delete_collection_with_tenant_context(name, Some(&default_tenant_ctx()))
            .await
            .context("delete")?;
        let v_delete = CorpusVersionRegistry::global().current(tenant, name).await;
        assert!(
            v_delete > v_update,
            "delete must bump ({v_update} -> {v_delete})"
        );

        Ok(())
    }

    /// Regression for the pgwire graph failure where relational DDL and the
    /// backing collection each minted `object_id = 1` from independent
    /// allocators. A collection created after any catalog object must reserve
    /// from the live catalog sequence, even while tenant-prefixed collection
    /// namespaces remain default-OFF and the two logical table names differ.
    #[tokio::test]
    async fn collection_id_comes_from_live_catalog_object_id_authority() -> Result<()> {
        use crate::catalog::{Catalog, CatalogColumn, CatalogTableSchema};
        use crate::services::canonical_wal::MemoryTableWalAppender;
        use crate::services::system_catalog::SystemCatalog;
        use crate::services::system_catalog_state::SystemCatalogState;
        use proximadb_data_model::ProximaType;

        let catalog_manager = Arc::new(CatalogManager::new());
        let catalog: Arc<dyn Catalog> = Arc::new(SystemCatalog::new(
            "default",
            SystemCatalogState::new(),
            Arc::new(MemoryTableWalAppender::new()),
        ));
        catalog_manager.register(catalog.clone()).await?;

        // Model the pgwire DDL namespace. Use a deliberately high imported id
        // so this test deterministically fails if CollectionService falls back
        // to its independent process-global allocator, regardless of test order.
        let ddl_namespace = vec!["proximadb".to_string(), "default".to_string()];
        catalog
            .create_namespace(&ddl_namespace, std::collections::HashMap::new())
            .await?;
        let relational_id = 9_000_000_000_u64;
        let relational_identifier = TableIdentifier::new(ddl_namespace, "gperson");
        let mut relational_schema = CatalogTableSchema::new("gperson")
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64));
        relational_schema.object_id = Some(relational_id);
        catalog
            .create_table(&relational_identifier, relational_schema)
            .await?;

        let service = CollectionService::new(StorageConfig::default())
            .await?
            .with_catalog_manager(catalog_manager);
        service.set_tenant_namespaces_for_test(false);
        let config = CollectionConfig {
            name: "gperson".to_string(),
            dimension: 1,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };

        let response = service.create_collection(&config).await?;
        assert!(
            response.success,
            "backing collection create failed: {:?}",
            response.error_code
        );
        let collection = response.collection.context("created collection")?;
        let collection_id = collection.id.parse::<u64>().context("numeric object id")?;
        assert!(
            collection_id > relational_id,
            "collection id {collection_id} must follow the live catalog floor {relational_id}"
        );

        let collection_identifier =
            crate::storage::metadata::collection_mapping::collection_table_identifier(&config);
        let collection_schema = catalog.get_table(&collection_identifier).await?;
        assert_eq!(collection_schema.object_id, Some(collection_id));
        assert_eq!(
            catalog.get_table(&relational_identifier).await?.object_id,
            Some(relational_id),
            "collection registration must not overwrite relational oid identity"
        );
        Ok(())
    }

    /// TD-SC-1b: `collection()` is fronted by the hot cache, and a write
    /// invalidates it (no stale read). After caching a live collection, deleting
    /// it must make the next `collection()` return `None` — proving the
    /// corpus-version stamp on the cached entry self-invalidates on the write's
    /// bump (#435). Also exercises the UUID bypass path.
    #[tokio::test]
    async fn collection_cache_serves_then_invalidates_on_write() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &temp_path)
            .await
            .context("xcatalog")?;
        // No tenant manager → single-tenant → the cache path is active.
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("service")?
            .with_catalog_manager(catalog_manager.clone());

        let name = "sc1b_cache_probe";
        let config = CollectionConfig {
            name: name.to_string(),
            dimension: 8,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };
        assert!(
            service
                .create_collection(&config)
                .await
                .context("create")?
                .success
        );

        // Read through the cache (name key) — present, and a second read still
        // present (served from cache or catalog; both correct).
        let first = service.collection(name).await.context("read1")?;
        assert!(first.is_some(), "collection must be found after create");
        assert!(service.collection(name).await.context("read2")?.is_some());

        // UUID identifier bypasses the cache and reads through (non-existent id).
        assert!(
            service
                .collection("00000000-0000-0000-0000-000000000000")
                .await
                .context("uuid read")?
                .is_none()
        );

        // Delete bumps corpus_version("default", name); the cached entry's stamp
        // no longer matches → the next read must reflect the delete (None), not a
        // stale hit.
        service
            .delete_collection_with_tenant_context(name, Some(&default_tenant_ctx()))
            .await
            .context("delete")?;
        assert!(
            service
                .collection(name)
                .await
                .context("read after delete")?
                .is_none(),
            "cache must invalidate on write — a deleted collection must not be served stale"
        );

        Ok(())
    }

    /// TD-SC-2b: with a warm dir configured, the hot cache reads through the
    /// on-disk warm tier — a `collection()` read materializes a warm file on
    /// local disk (proving the `WarmDiskStore` is wired into the chain), and the
    /// result is correct.
    #[tokio::test]
    async fn warm_tier_wired_materializes_disk_entry() -> Result<()> {
        use tempfile::TempDir;

        let cat_dir = TempDir::new().context("cat dir")?;
        let warm_dir = TempDir::new().context("warm dir")?;
        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &format!("file://{}", cat_dir.path().display()))
            .await
            .context("xcatalog")?;
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("service")?
            .with_syscat_warm_dir(warm_dir.path().to_path_buf())
            .with_catalog_manager(catalog_manager.clone());

        let name = "sc2b_warm_wire_probe";
        let config = CollectionConfig {
            name: name.to_string(),
            dimension: 8,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };
        assert!(
            service
                .create_collection(&config)
                .await
                .context("create")?
                .success
        );

        // Read through hot → warm → canonical; the warm tier writes a file.
        assert!(service.collection(name).await.context("read")?.is_some());
        let warm_file = warm_dir
            .path()
            .join(CollectionService::DEFAULT_VERSION_TENANT)
            .join(format!("{name}.bin"));
        assert!(
            warm_file.exists(),
            "warm tier must materialize {warm_file:?} on a read (proves it's in the chain)"
        );

        Ok(())
    }

    /// TD-CAT-2a (audit gap G6): with tenant-prefixed namespaces (TD-CAT-2b
    /// gate on), creating a `tenant:`-tagged collection records the owning
    /// tenant on the freshly-created namespace. The upsert routes
    /// through `create_namespace_for_tenant`, so the persisted
    /// `CatalogNamespace.tenant_id` carries the tenant derived by
    /// `collection_tenant_id` (the `tenant:` tag) — instead of the `None` the
    /// old tenant-less `create_namespace` left behind.
    #[tokio::test]
    async fn tenant_tagged_collection_records_namespace_tenant() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &temp_path)
            .await
            .context("xcatalog")?;
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("service")?
            .with_catalog_manager(catalog_manager.clone());
        // Owner recording applies to tenant-prefixed namespaces (TD-CAT-2b);
        // a namespace shared by every tenant records no single owner.
        service.set_tenant_namespaces_for_test(true);

        // A namespaced name → the upsert creates a brand-new namespace
        // `tdc2a_ns` (the tenant is recorded only on namespace creation, so the
        // namespace must be fresh for the tenant-aware constructor to fire).
        let config = CollectionConfig {
            name: "tdc2a_ns.tbl".to_string(),
            dimension: 8,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            tags: vec!["tenant:acme".to_string()],
            ..Default::default()
        };
        assert!(
            service
                .create_collection(&config)
                .await
                .context("create")?
                .success
        );

        let catalog = catalog_manager
            .default_catalog()
            .await
            .context("default catalog")?;
        let ns = catalog
            .get_namespace(&["acme".to_string(), "tdc2a_ns".to_string()])
            .await
            .context("get_namespace")?;
        assert_eq!(
            ns.tenant_id.as_deref(),
            Some("acme"),
            "the freshly-created namespace must record the owning tenant (TD-CAT-2a)"
        );

        Ok(())
    }

    async fn service_with_catalog() -> Result<(CollectionService, tempfile::TempDir)> {
        let temp_dir = tempfile::TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &temp_path)
            .await
            .context("xcatalog")?;
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("service")?
            .with_catalog_manager(catalog_manager);
        Ok((service, temp_dir))
    }

    fn scope_config(name: &str) -> CollectionConfig {
        CollectionConfig {
            name: name.to_string(),
            dimension: 4,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        }
    }

    /// Collection resolution is scoped to the request tenant: a collection
    /// created through the port records its creator as owner, resolves for that
    /// tenant (by name and object id), and is absent for every other tenant —
    /// in get, list, update and delete alike.
    #[tokio::test]
    async fn port_resolution_is_scoped_to_the_request_tenant() -> Result<()> {
        use proximadb_runtime::CollectionPort;
        let (service, _dir) = service_with_catalog().await?;
        let created =
            CollectionPort::create_collection(&service, scope_config("scope_acme"), Some("acme"))
                .await
                .context("create as acme")?;
        assert_eq!(
            proximadb_tenant::collection_owner(&created),
            proximadb_tenant::CollectionOwner::Tenant("acme".into()),
            "create records its owner"
        );
        for reference in ["scope_acme", created.id.as_str()] {
            assert!(
                CollectionPort::get_collection(&service, reference, Some("acme"))
                    .await?
                    .is_some()
            );
            assert!(
                CollectionPort::get_collection(&service, reference, Some("globex"))
                    .await?
                    .is_none()
            );
            assert!(
                CollectionPort::get_collection(&service, reference, None)
                    .await?
                    .is_none(),
                "no tenant is denied"
            );
            assert!(
                CollectionPort::get_collection_internal(&service, reference)
                    .await?
                    .is_some(),
                "internal lookup"
            );
            assert!(
                CollectionPort::get_collection(&service, reference, Some(""))
                    .await?
                    .is_none()
            );
        }
        let names = |list: Vec<Collection>| {
            list.into_iter()
                .filter_map(|c| c.config.map(|cfg| cfg.name))
                .collect::<Vec<_>>()
        };
        assert!(
            names(CollectionPort::list_collections(&service, Some("acme")).await?)
                .contains(&"scope_acme".to_string())
        );
        assert!(
            !names(CollectionPort::list_collections(&service, Some("globex")).await?)
                .contains(&"scope_acme".to_string())
        );

        // Update by another tenant behaves like a missing collection; an update
        // by the owner cannot change the recorded owner.
        let foreign_update = CollectionPort::update_collection(
            &service,
            "scope_acme",
            CollectionConfig {
                description: Some("x".into()),
                ..Default::default()
            },
            Some("globex"),
        )
        .await;
        assert!(foreign_update.is_err());
        let updated = CollectionPort::update_collection(
            &service,
            "scope_acme",
            CollectionConfig {
                tags: vec!["tenant:globex".into()],
                owner: Some("globex".into()),
                ..Default::default()
            },
            Some("acme"),
        )
        .await?;
        assert_eq!(
            proximadb_tenant::collection_owner(&updated),
            proximadb_tenant::CollectionOwner::Tenant("acme".into())
        );
        assert!(!CollectionPort::delete_collection(&service, "scope_acme", Some("globex")).await?);
        assert!(
            CollectionPort::get_collection(&service, "scope_acme", Some("acme"))
                .await?
                .is_some()
        );
        Ok(())
    }

    /// A collection created before ownership was recorded belongs to the
    /// default tenant only, until the owner backfill records its owner — and
    /// the backfill never re-assigns an owned collection.
    #[tokio::test]
    async fn unattributed_collection_is_default_only_until_assigned() -> Result<()> {
        use proximadb_runtime::CollectionPort;
        let (service, _dir) = service_with_catalog().await?;
        let legacy = service
            .create_collection_with_tenant_context(&scope_config("scope_legacy"), None)
            .await?
            .collection
            .context("created")?;
        assert_eq!(
            proximadb_tenant::collection_owner(&legacy),
            proximadb_tenant::CollectionOwner::Unattributed
        );
        assert!(
            CollectionPort::get_collection(&service, "scope_legacy", Some("default"))
                .await?
                .is_some()
        );
        assert!(
            CollectionPort::get_collection(&service, "scope_legacy", Some("acme"))
                .await?
                .is_none()
        );

        assert!(
            service
                .assign_owner_if_unattributed("scope_legacy", "acme")
                .await?
        );
        assert!(
            CollectionPort::get_collection(&service, "scope_legacy", Some("acme"))
                .await?
                .is_some()
        );
        assert!(
            CollectionPort::get_collection(&service, "scope_legacy", Some("default"))
                .await?
                .is_none()
        );
        assert!(
            !service
                .assign_owner_if_unattributed("scope_legacy", "globex")
                .await?,
            "never re-assigns"
        );
        assert!(
            CollectionPort::get_collection(&service, "scope_legacy", Some("acme"))
                .await?
                .is_some()
        );
        Ok(())
    }

    /// TD-CAT-2b (S3a): the pure scoping function prepends the tenant as
    /// namespace level 0 only when enabled and a non-empty tenant is present.
    #[test]
    fn tenant_scoped_identifier_prepends_tenant_when_enabled() {
        let base = TableIdentifier::new(vec!["default".to_string()], "t".to_string());

        let scoped = CollectionService::tenant_scoped_identifier(true, Some("acme"), base.clone());
        assert_eq!(
            scoped.namespace,
            vec!["acme".to_string(), "default".to_string()]
        );
        assert_eq!(scoped.name, "t");

        // Off ⇒ unchanged; no tenant ⇒ unchanged; empty tenant ⇒ unchanged.
        assert_eq!(
            CollectionService::tenant_scoped_identifier(false, Some("acme"), base.clone())
                .namespace,
            vec!["default".to_string()]
        );
        assert_eq!(
            CollectionService::tenant_scoped_identifier(true, None, base.clone()).namespace,
            vec!["default".to_string()]
        );
        assert_eq!(
            CollectionService::tenant_scoped_identifier(true, Some(""), base).namespace,
            vec!["default".to_string()]
        );
    }

    /// TD-CAT-2b (S3a): with the gate ON, two tenants creating an
    /// identically-named namespace land under DISTINCT tenant-prefixed
    /// namespaces — no collision on the bare key — each recording its own tenant.
    #[tokio::test]
    async fn tenant_namespaces_isolate_same_named_namespace() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &temp_path)
            .await
            .context("xcatalog")?;
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("service")?
            .with_catalog_manager(catalog_manager.clone());
        service.set_tenant_namespaces_for_test(true);

        // Two collections in the SAME bare namespace `shared`, different tenants
        // (distinct collection names — the service dedupes by name; it's the
        // shared NAMESPACE that must not collide across tenants).
        for tenant in ["acme", "globex"] {
            let config = CollectionConfig {
                name: format!("shared.{tenant}_tbl"),
                dimension: 8,
                primary_index: Some("default".to_string()),
                auto_index_selection: Some(false),
                tags: vec![format!("tenant:{tenant}")],
                ..Default::default()
            };
            assert!(
                service
                    .create_collection(&config)
                    .await
                    .with_context(|| format!("create for {tenant}"))?
                    .success
            );
        }

        let catalog = catalog_manager
            .default_catalog()
            .await
            .context("default catalog")?;

        // Each tenant has its OWN `shared` namespace, recording its own tenant.
        for tenant in ["acme", "globex"] {
            let ns = catalog
                .get_namespace(&[tenant.to_string(), "shared".to_string()])
                .await
                .with_context(|| format!("get_namespace for {tenant}"))?;
            assert_eq!(
                ns.tenant_id.as_deref(),
                Some(tenant),
                "tenant-prefixed namespace records its owning tenant"
            );
        }

        // The bare `shared` namespace was never created — no cross-tenant
        // collision on the shared key.
        assert!(
            !catalog
                .namespace_exists(&["shared".to_string()])
                .await
                .context("bare namespace_exists")?,
            "no bare `shared` namespace ⇒ tenants did not collide on the shared key"
        );

        Ok(())
    }

    /// TD-SC-4 (S4): provisioning pre-creates the tenant's system namespace and
    /// is idempotent — a second call is a no-op, and the namespace records its
    /// owning tenant.
    #[tokio::test]
    async fn provision_tenant_system_catalog_is_idempotent() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &temp_path)
            .await
            .context("xcatalog")?;
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("service")?
            .with_catalog_manager(catalog_manager.clone());
        service.set_tenant_namespaces_for_test(true);

        // Provision twice — the second call must be a safe no-op.
        service
            .provision_tenant_system_catalog("acme")
            .await
            .context("provision 1")?;
        service
            .provision_tenant_system_catalog("acme")
            .await
            .context("provision 2 (idempotent)")?;

        let catalog = catalog_manager
            .default_catalog()
            .await
            .context("default catalog")?;
        let ns = catalog
            .get_namespace(&["acme".to_string(), "default".to_string()])
            .await
            .context("provisioned namespace present")?;
        assert_eq!(
            ns.tenant_id.as_deref(),
            Some("acme"),
            "provisioned system namespace records its owning tenant"
        );

        Ok(())
    }

    /// TD-SC-4 (S4): with tenant namespaces off (single-tenant deployments),
    /// provisioning is a no-op — no per-tenant skeleton is created.
    #[tokio::test]
    async fn provision_tenant_system_catalog_noop_when_gate_off() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &temp_path)
            .await
            .context("xcatalog")?;
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("service")?
            .with_catalog_manager(catalog_manager.clone());
        service.set_tenant_namespaces_for_test(false);

        service
            .provision_tenant_system_catalog("acme")
            .await
            .context("provision (gate off)")?;

        let catalog = catalog_manager
            .default_catalog()
            .await
            .context("default catalog")?;
        assert!(
            !catalog
                .namespace_exists(&["acme".to_string(), "default".to_string()])
                .await
                .context("namespace_exists")?,
            "gate off ⇒ provisioning creates nothing"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_short_name_resolves_name_authoritative() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &temp_path)
            .await
            .context("Failed to create test xCatalog")?;
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("collection service")?
            .with_catalog_manager(catalog_manager.clone());

        // A short, standard SQL identifier (4 chars) — would have been rejected by
        // the old 8-char floor.
        let name = "part";
        let config = CollectionConfig {
            dimension: 16,
            description: None,
            owner: None,
            ..collection_config(name)
        };
        let created = service.create_collection(&config).await.context("create")?;
        assert!(
            created.success,
            "short name should create: {:?}",
            created.error_code
        );

        // Resolve name -> id: must yield an opaque id DISTINCT from the name.
        let id = service
            .resolve_collection_id(name)
            .await
            .context("resolve id")?
            .ok_or_else(|| anyhow::anyhow!("name did not resolve to an id"))?;
        assert_ne!(id, name, "id must be opaque, not the name");

        // Round-trip id -> name.
        let back = service
            .resolve_collection_name(&id)
            .await
            .context("resolve name")?
            .ok_or_else(|| anyhow::anyhow!("id did not resolve back to a name"))?;
        assert_eq!(back, name, "id must round-trip to the original name");

        Ok(())
    }

    #[tokio::test]
    async fn test_create_collection_persists_explicit_cosine_metric() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("Failed to create temporary directory for test")?;
        let temp_path = format!("file://{}", temp_dir.path().display());

        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &temp_path)
            .await
            .context("Failed to create test xCatalog")?;
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("Failed to create collection service for test")?
            .with_catalog_manager(catalog_manager.clone());

        let config = CollectionConfig {
            distance_metric: None,
            storage_engine: Some(StorageEngine::Sst as i32), // VIPER is no longer selectable for new collections (ADR-093 S2)
            primary_index: None,
            ..collection_config("metric_default_test")
        };

        let result = service.create_collection(&config).await?;
        assert!(
            result.success,
            "create failed with error_code={:?}",
            result.error_code
        );

        let stored = service
            .collection("metric_default_test")
            .await?
            .expect("collection should exist");
        assert_eq!(
            stored.config.as_ref().and_then(|cfg| cfg.distance_metric),
            Some(crate::proto::proximadb_v1::DistanceMetric::Cosine as i32)
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_create_collection_preserves_exact_default_without_indexes() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("Failed to create temporary directory for test")?;
        let temp_path = format!("file://{}", temp_dir.path().display());

        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &temp_path)
            .await
            .context("Failed to create test xCatalog")?;
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("Failed to create collection service for test")?
            .with_catalog_manager(catalog_manager.clone());

        let config = CollectionConfig {
            name: "exact_default_case".to_string(),
            dimension: 384,
            storage_engine: Some(StorageEngine::Sst as i32),
            index_configs: vec![],
            auto_index_selection: Some(false),
            ..Default::default()
        };

        let result = service.create_collection(&config).await?;
        assert!(
            result.success,
            "create failed with error_code={:?}",
            result.error_code
        );

        let stored = service
            .collection("exact_default_case")
            .await?
            .expect("collection should exist");
        assert!(
            stored
                .config
                .as_ref()
                .is_some_and(|cfg| cfg.index_configs.is_empty())
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_collection_lifecycle_mirrors_to_xcatalog_with_numeric_id() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("Failed to create temporary directory for test")?;
        let temp_path = format!("file://{}", temp_dir.path().display());

        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &temp_path)
            .await
            .context("Failed to create test xCatalog")?;

        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("Failed to create collection service for test")?
            .with_catalog_manager(catalog_manager.clone());

        let config = CollectionConfig {
            name: "catalog_vector_assets".to_string(),
            dimension: 384,
            storage_engine: Some(StorageEngine::Sst as i32),
            filterable_columns: vec![crate::proto::proximadb_v1::FilterableColumnSpec {
                name: "category".to_string(),
                data_type: FilterableDataType::FilterableString as i32,
                indexed: true,
                supports_range: false,
                estimated_cardinality: Some(32),
            }],
            index_configs: vec![crate::proto::proximadb_v1::IndexConfig {
                index_name: "catalog_vector_assets_hnsw".to_string(),
                algorithm: IndexingAlgorithm::Hnsw as i32,
                enabled: Some(true),
                ..Default::default()
            }],
            auto_index_selection: Some(false),
            ..Default::default()
        };

        let result = service.create_collection(&config).await?;
        assert!(
            result.success,
            "create failed with error_code={:?}",
            result.error_code
        );
        let collection = result.collection.expect("collection should be returned");
        // ADR-031: collection.id is the decimal object_id (u64), not a UUID.
        assert!(
            collection.id.parse::<u64>().is_ok(),
            "collection.id must be a numeric object_id, got: {}",
            collection.id
        );

        let catalog = catalog_manager.default_catalog().await?;
        let table_id = TableIdentifier::new(
            vec!["default".to_string()],
            "catalog_vector_assets".to_string(),
        );
        let schema = catalog.get_table(&table_id).await?;
        assert_eq!(schema.properties.get("collection.id"), Some(&collection.id));
        // ADR-031 allocator unification: schema.object_id must equal collection.id's
        // oid (create_table adopted it) — one identity, not two divergent ones.
        assert_eq!(
            schema.object_id,
            collection.id.parse::<u64>().ok(),
            "schema.object_id must equal collection.id's oid (unified identity)"
        );
        assert_eq!(
            schema.properties.get("asset.capability.vector"),
            Some(&"true".to_string())
        );
        assert!(
            schema
                .columns
                .iter()
                .any(|column| column.name == "category")
        );
        assert_eq!(schema.projections.len(), 1);

        // The catalog is the sole store; reads reconstruct the collection from the
        // xCatalog asset (by name and by UUID).
        let catalog_backed_by_name = service
            .collection("catalog_vector_assets")
            .await?
            .expect("collection should be reconstructed from xCatalog by name");
        assert_eq!(catalog_backed_by_name.id, collection.id);
        let catalog_backed_by_id = service
            .collection(&collection.id)
            .await?
            .expect("collection should be reconstructed from xCatalog by UUID");
        assert_eq!(
            catalog_backed_by_id
                .config
                .as_ref()
                .map(|config| config.dimension),
            Some(384)
        );
        assert!(catalog_backed_by_id.config.as_ref().is_some_and(|config| {
            config.filterable_columns.iter().any(|column| {
                column.name == "category"
                    && column.data_type == FilterableDataType::FilterableString as i32
            })
        }));
        assert!(
            service
                .list_collections()
                .await?
                .iter()
                .any(|listed| listed.id == collection.id)
        );

        let duplicate = service.create_collection(&config).await?;
        assert!(!duplicate.success);
        assert_eq!(duplicate.error_code.as_deref(), Some("COLLECTION_EXISTS"));

        let delete = service.delete_collection("catalog_vector_assets").await?;
        assert!(delete.success);
        assert!(!catalog.table_exists(&table_id).await?);

        Ok(())
    }

    #[test]
    fn test_response_conversion() {
        let response = CollectionServiceResponse::success(
            "test-uuid".to_string(),
            "/path/to/storage".to_string(),
            1000,
        );

        assert!(response.success);
        assert_eq!(response.processing_time_us, 1000);
    }

    /// TD-122: the detailed per-index (HNSW m/ef, IVF n_lists/n_probe,
    /// is_primary) and quantization (enabled, strategy) config must survive a
    /// round-trip through the read-authoritative xCatalog table asset. Before
    /// the fix this reconstruction returned `m=0`, `is_primary=false`, and
    /// quantization disabled.
    #[test]
    fn catalog_asset_round_trips_detailed_index_and_quant_config() {
        use crate::proto::proximadb_v1::{
            Collection, HnswConfig, IndexConfig, IvfConfig, QuantizationConfig, StorageAssignment,
            quantization_config::Strategy,
        };

        let collection = Collection {
            id: "col-td122".to_string(),
            config: Some(CollectionConfig {
                name: "td122_round_trip".to_string(),
                dimension: 128,
                index_configs: vec![IndexConfig {
                    index_name: "primary_hnsw".to_string(),
                    hnsw_config: Some(HnswConfig {
                        m: Some(24),
                        ef_construction: Some(150),
                        ef_search: Some(64),
                        ..Default::default()
                    }),
                    ivf_config: Some(IvfConfig {
                        n_lists: Some(256),
                        n_probe: Some(16),
                        ..Default::default()
                    }),
                    is_primary: Some(true),
                    ..Default::default()
                }],
                quantization: Some(QuantizationConfig {
                    enabled: Some(true),
                    strategy: Some(Strategy::Aggressive as i32),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            storage_assignment: Some(StorageAssignment {
                primary_path: "file:///tmp/td122".to_string(),
                base_location: "file:///tmp/td122".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        let schema = crate::storage::metadata::collection_mapping::catalog_schema_from_collection(
            &collection,
        )
        .expect("schema from collection");
        let identifier = crate::storage::metadata::collection_mapping::collection_table_identifier(
            collection.config.as_ref().expect("config"),
        );
        let restored =
            crate::storage::metadata::collection_mapping::collection_from_catalog_schema(
                &identifier,
                &schema,
            )
            .expect("collection from schema")
            .expect("collection present");

        let config = restored.config.expect("restored config");
        assert_eq!(config.index_configs.len(), 1, "one ANN index retained");
        let ic = &config.index_configs[0];
        assert_eq!(ic.index_name, "primary_hnsw");
        assert_eq!(ic.is_primary, Some(true));
        let hnsw = ic.hnsw_config.as_ref().expect("hnsw config restored");
        assert_eq!(hnsw.m, Some(24));
        assert_eq!(hnsw.ef_construction, Some(150));
        assert_eq!(hnsw.ef_search, Some(64));
        let ivf = ic.ivf_config.as_ref().expect("ivf config restored");
        assert_eq!(ivf.n_lists, Some(256));
        assert_eq!(ivf.n_probe, Some(16));
        let quant = config.quantization.as_ref().expect("quantization restored");
        assert_eq!(quant.enabled, Some(true));
        assert_eq!(quant.strategy, Some(Strategy::Aggressive as i32));
    }

    /// Build a canonical schema column (non-filterable, non-indexed) for the
    /// round-trip tests. Fully-qualified types so it needs no test-fn imports.
    fn canonical_col(
        name: &str,
        data_type: proximadb_data_model::ProximaType,
    ) -> proximadb_runtime::CollectionSchemaColumn {
        proximadb_runtime::CollectionSchemaColumn {
            name: name.to_string(),
            data_type,
            nullable: true,
            indexed: false,
            filterable: false,
            text_storage: None,
            max_length: None,
        }
    }

    /// ADR-047 / TD-TBL-1: canonical ProximaType columns (incl. UInt/Struct/Map/
    /// Sparse/BinaryVector — none representable in the narrow v1 config) must
    /// survive create → set → catalog restart → read-back.
    #[tokio::test]
    async fn canonical_schema_columns_survive_catalog_restart() -> Result<()> {
        use proximadb_data_model::{ProximaType, TimeUnit, VectorElement};
        use proximadb_runtime::CollectionPort;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());

        let columns = vec![
            canonical_col("u", ProximaType::UInt64),
            canonical_col(
                "st",
                ProximaType::Struct {
                    fields: vec![("a".to_string(), ProximaType::Int64)],
                },
            ),
            canonical_col(
                "m",
                ProximaType::Map {
                    key: Box::new(ProximaType::String),
                    value: Box::new(ProximaType::Int64),
                },
            ),
            canonical_col(
                "sv",
                ProximaType::SparseVector {
                    element: VectorElement::Float32,
                },
            ),
            canonical_col("bv", ProximaType::BinaryVector { dim: 256 }),
            canonical_col(
                "dv",
                ProximaType::DenseVector {
                    element: VectorElement::Float32,
                    dim: 128,
                },
            ),
            canonical_col("ts", ProximaType::Timestamp(TimeUnit::Nanosecond)),
        ];

        let name = "canon_rt";
        // Instance A: create the collection, then persist the canonical sidecar.
        let mgr_a = Arc::new(CatalogManager::new());
        mgr_a
            .create_native_catalog("default", &temp_path)
            .await
            .context("create test catalog A")?;
        let svc_a = CollectionService::new(StorageConfig::default())
            .await
            .context("collection service A")?
            .with_catalog_manager(mgr_a);
        let config = CollectionConfig {
            name: name.to_string(),
            dimension: 8,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };
        assert!(
            svc_a
                .create_collection(&config)
                .await
                .context("create collection A")?
                .success
        );
        svc_a
            .set_collection_schema_columns(name, &columns, Some("default"))
            .await
            .context("set canonical columns")?;
        drop(svc_a);

        // Instance B: a fresh service + catalog manager on the SAME on-disk dir
        // simulates a restart (the canonical sidecar must be read from disk).
        let mgr_b = Arc::new(CatalogManager::new());
        mgr_b
            .create_native_catalog("default", &temp_path)
            .await
            .context("create test catalog B")?;
        let svc_b = CollectionService::new(StorageConfig::default())
            .await
            .context("collection service B")?
            .with_catalog_manager(mgr_b);

        let restored = svc_b
            .get_collection_schema_columns(name, Some("default"))
            .await
            .context("read canonical columns after restart")?
            .expect("canonical columns persisted across catalog restart");
        assert_eq!(restored.len(), columns.len());
        assert_eq!(
            restored, columns,
            "every canonical ProximaType variant round-trips through the catalog"
        );

        // Legacy collection (no canonical sidecar) → None (mixed-read-safe).
        let legacy = CollectionConfig {
            name: "canon_legacy".to_string(),
            dimension: 4,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };
        assert!(
            svc_b
                .create_collection(&legacy)
                .await
                .context("create legacy collection")?
                .success
        );
        assert_eq!(
            svc_b
                .get_collection_schema_columns("canon_legacy", Some("default"))
                .await?,
            None,
            "collection without a canonical sidecar returns None"
        );

        Ok(())
    }

    /// ADR-048 P1: canonical columns are stored as TYPED catalog columns (200+
    /// band), not a `collection.canonical_schema` property — and the reserved v1
    /// columns (oid/embedding) are preserved.
    #[tokio::test]
    async fn canonical_schema_columns_stored_as_typed_columns_not_sidecar() -> Result<()> {
        use proximadb_data_model::ProximaType;
        use proximadb_runtime::CollectionPort;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let mgr = Arc::new(CatalogManager::new());
        mgr.create_native_catalog("default", &temp_path)
            .await
            .context("catalog")?;
        let svc = CollectionService::new(StorageConfig::default())
            .await
            .context("service")?
            .with_catalog_manager(mgr.clone());
        let name = "canon_typed";
        let config = CollectionConfig {
            name: name.to_string(),
            dimension: 8,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };
        assert!(svc.create_collection(&config).await?.success);
        let columns = vec![
            canonical_col("u", ProximaType::UInt64),
            canonical_col("s", ProximaType::String),
        ];
        svc.set_collection_schema_columns(name, &columns, Some("default"))
            .await
            .context("set canonical columns")?;

        // Peek at the catalog table directly.
        let catalog = mgr.default_catalog().await?;
        let identifier =
            crate::storage::metadata::collection_mapping::collection_table_identifier(&config);
        let schema = catalog.get_table(&identifier).await?;

        // Canonical columns are typed, in the 200+ band.
        let typed: Vec<_> = schema.columns.iter().filter(|c| c.id >= 200).collect();
        assert_eq!(typed.len(), 2, "two canonical columns in the 200+ band");
        assert_eq!(typed[0].name, "u");
        assert!(matches!(typed[0].data_type, ProximaType::UInt64));
        assert_eq!(typed[1].name, "s");
        // The legacy sidecar property is GONE (typed columns are the sole authority).
        assert!(
            schema
                .properties
                .get(crate::storage::metadata::collection_mapping::CANONICAL_SCHEMA_PROPERTY)
                .is_none(),
            "ADR-048 P1 retires the collection.canonical_schema sidecar property"
        );
        // Reserved v1 columns are preserved (oid + embedding).
        assert!(
            schema.columns.iter().any(|c| c.name == "oid"),
            "reserved oid column preserved"
        );
        assert!(
            schema.columns.iter().any(|c| c.name == "embedding"),
            "reserved embedding column preserved"
        );

        // And the trait read returns them.
        let restored = svc
            .get_collection_schema_columns(name, Some("default"))
            .await?
            .expect("typed columns read back");
        assert_eq!(restored, columns);
        Ok(())
    }

    /// ADR-048 P1 mixed-read: a catalog written before P1 (canonical schema as a
    /// `collection.canonical_schema` property, no typed 200+ columns) is still
    /// readable via the transitional path, and a subsequent `set` migrates it to
    /// typed columns (removing the property).
    #[tokio::test]
    async fn canonical_schema_transitional_read_of_legacy_sidecar() -> Result<()> {
        use proximadb_data_model::ProximaType;
        use proximadb_runtime::CollectionPort;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let mgr = Arc::new(CatalogManager::new());
        mgr.create_native_catalog("default", &temp_path)
            .await
            .context("catalog")?;
        let svc = CollectionService::new(StorageConfig::default())
            .await
            .context("service")?
            .with_catalog_manager(mgr.clone());
        let name = "canon_legacy_prop";
        let config = CollectionConfig {
            name: name.to_string(),
            dimension: 8,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };
        assert!(svc.create_collection(&config).await?.success);

        // Simulate a pre-P1 catalog: write the canonical schema as the legacy
        // property directly (no typed 200+ columns).
        let columns = vec![canonical_col("u", ProximaType::UInt64)];
        let catalog = mgr.default_catalog().await?;
        let identifier =
            crate::storage::metadata::collection_mapping::collection_table_identifier(&config);
        let mut schema = catalog.get_table(&identifier).await?;
        schema.properties.insert(
            crate::storage::metadata::collection_mapping::CANONICAL_SCHEMA_PROPERTY.to_string(),
            serde_json::to_string(&columns).context("serialize legacy canonical sidecar")?,
        );
        let _ = catalog.drop_table(&identifier, false).await?;
        catalog.create_table(&identifier, schema).await?;

        // Transitional read returns the legacy sidecar.
        let restored = svc
            .get_collection_schema_columns(name, Some("default"))
            .await?
            .expect("transitional read of the legacy sidecar");
        assert_eq!(restored, columns);

        // A subsequent set migrates it to typed columns (property removed).
        let columns2 = vec![canonical_col("n", ProximaType::Int64)];
        svc.set_collection_schema_columns(name, &columns2, Some("default"))
            .await?;
        let restored2 = svc
            .get_collection_schema_columns(name, Some("default"))
            .await?
            .expect("post-migration read");
        assert_eq!(restored2, columns2);
        let schema2 = catalog.get_table(&identifier).await?;
        assert!(
            schema2
                .properties
                .get(crate::storage::metadata::collection_mapping::CANONICAL_SCHEMA_PROPERTY)
                .is_none(),
            "set migrated the legacy sidecar to typed columns"
        );
        Ok(())
    }

    /// ADR-047 / TD-TBL-1: an unrelated `update_collection` (e.g. a description
    /// tweak) must NOT drop the canonical sidecar — the sticky-preserve in
    /// `upsert_collection_catalog_asset` carries it across the narrow rebuild.
    #[tokio::test]
    async fn canonical_schema_columns_survive_unrelated_update_collection() -> Result<()> {
        use proximadb_data_model::ProximaType;
        use proximadb_runtime::CollectionPort;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let mgr = Arc::new(CatalogManager::new());
        mgr.create_native_catalog("default", &temp_path)
            .await
            .context("create test catalog")?;
        let svc = CollectionService::new(StorageConfig::default())
            .await
            .context("collection service")?
            .with_catalog_manager(mgr);

        let name = "canon_sticky";
        let config = CollectionConfig {
            name: name.to_string(),
            dimension: 8,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };
        assert!(
            svc.create_collection(&config)
                .await
                .context("create collection")?
                .success
        );

        let columns = vec![canonical_col(
            "s",
            ProximaType::Struct {
                fields: vec![("a".to_string(), ProximaType::Int64)],
            },
        )];
        svc.set_collection_schema_columns(name, &columns, Some("default"))
            .await
            .context("set canonical columns")?;

        // Unrelated config change — rebuilds the catalog asset from the narrow
        // config, which must re-attach the preserved canonical sidecar.
        let update = CollectionConfig {
            description: Some("tweaked".to_string()),
            ..Default::default()
        };
        assert!(
            svc.update_collection(name, Some(update))
                .await
                .context("unrelated update_collection")?
                .success
        );

        let restored = svc
            .get_collection_schema_columns(name, Some("default"))
            .await
            .context("read canonical after update")?
            .expect("canonical sidecar survived unrelated update_collection");
        assert_eq!(restored, columns);

        Ok(())
    }

    /// TD-TIER-1 SC1: absent storage class resolves BALANCED, persists
    /// explicitly, and round-trips through the catalog with an ACTIVE status;
    /// a pre-SC1 record (config_json without the field) decodes with the
    /// field absent and reports generation 0.
    #[tokio::test]
    async fn storage_class_absent_resolves_balanced_and_round_trips() -> Result<()> {
        use crate::proto::proximadb_v1::StorageClass;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let mgr = Arc::new(CatalogManager::new());
        mgr.create_native_catalog("default", &temp_path)
            .await
            .context("create test catalog")?;
        let svc = CollectionService::new(StorageConfig::default())
            .await
            .context("collection service")?
            .with_catalog_manager(mgr);

        let config = CollectionConfig {
            name: "sc_balanced_default".to_string(),
            dimension: 8,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };
        assert!(
            svc.create_collection(&config)
                .await
                .context("create collection")?
                .success
        );

        let read_back = svc
            .collection("sc_balanced_default")
            .await?
            .context("collection after create")?;
        let read_config = read_back.config.context("config on read")?;
        assert_eq!(
            crate::services::collection::storage_class::storage_class_or_balanced(&read_config),
            StorageClass::Balanced
        );
        assert_eq!(
            read_config.storage_class,
            Some(StorageClass::Balanced as i32)
        );
        let status = read_back
            .storage_class_status
            .context("status projection on read")?;
        assert_eq!(
            status.state,
            crate::proto::proximadb_v1::storage_class_status::State::Active as i32
        );
        assert_eq!(status.policy_generation, 1);

        // Restart: a NEW service instance over the SAME catalog URL replays
        // the recorded class and status (durable round-trip).
        let reopened_mgr = Arc::new(CatalogManager::new());
        reopened_mgr
            .create_native_catalog("default", &temp_path)
            .await
            .context("reopen test catalog")?;
        let restarted = CollectionService::new(StorageConfig::default())
            .await
            .context("restarted collection service")?
            .with_catalog_manager(reopened_mgr);
        let after_restart = restarted
            .collection("sc_balanced_default")
            .await?
            .context("collection after restart")?;
        assert_eq!(
            after_restart.config.context("config")?.storage_class,
            Some(StorageClass::Balanced as i32)
        );
        assert_eq!(
            after_restart
                .storage_class_status
                .context("status after restart")?
                .state,
            crate::proto::proximadb_v1::storage_class_status::State::Active as i32
        );

        // Pre-SC1 record: a config_json WITHOUT the field decodes with the
        // field absent (serde default) and reports generation 0 — old assets
        // serve unchanged.
        let pre_sc1_json = r#"{"name":"legacy","dimension":8}"#;
        let legacy: CollectionConfig =
            serde_json::from_str(pre_sc1_json).context("decode pre-SC1 config")?;
        assert_eq!(legacy.storage_class, None);
        let legacy_status =
            crate::services::collection::storage_class::storage_class_status(&legacy);
        assert_eq!(legacy_status.policy_generation, 0);
        assert_eq!(
            legacy_status.state,
            crate::proto::proximadb_v1::storage_class_status::State::Active as i32
        );

        Ok(())
    }

    /// TD-TIER-1 SC1: an unknown storage-class enum number is rejected at
    /// create (fail-closed on every facade — gRPC/Flight reach here with the
    /// raw number), never coerced to a default.
    #[tokio::test]
    async fn storage_class_unknown_number_rejected_on_create() -> Result<()> {
        use crate::proto::proximadb_v1::StorageClass;
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("collection service")?;
        let config = CollectionConfig {
            name: "sc_bad_number".to_string(),
            dimension: 8,
            storage_class: Some(99), // an unknown discriminant
            ..Default::default()
        };
        let error = service
            .create_collection(&config)
            .await
            .expect_err("unknown storage class number must be rejected");
        assert!(error.to_string().contains("99"), "{error}");
        Ok(())
    }

    /// A collection service over a fresh native catalog in a temp dir (kept
    /// alive by the returned guard), on the default local storage config.
    async fn catalog_backed_service() -> Result<(tempfile::TempDir, CollectionService)> {
        let temp_dir = tempfile::TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let mgr = Arc::new(CatalogManager::new());
        mgr.create_native_catalog("default", &temp_path)
            .await
            .context("create test catalog")?;
        let svc = CollectionService::new(StorageConfig::default())
            .await
            .context("collection service")?
            .with_catalog_manager(mgr);
        Ok((temp_dir, svc))
    }

    /// TD-TIER-1 SC1/SC2: economy persists intent and reports NOT_ACTIVE with
    /// a reason naming the missing dependency; performance is admitted only
    /// with an object-store-backed SST placement (rejected with an actionable
    /// message otherwise — never accepted and silently degraded).
    #[tokio::test]
    async fn economy_persists_intent_and_performance_requires_object_backed_sst() -> Result<()> {
        use crate::proto::proximadb_v1::StorageClass;

        let (_temp_dir, svc) = catalog_backed_service().await?;

        // Economy: accepted, persisted, NOT_ACTIVE with a dependency-naming reason.
        let economy = CollectionConfig {
            name: "sc_economy".to_string(),
            dimension: 8,
            storage_class: Some(StorageClass::Economy as i32),
            ..Default::default()
        };
        assert!(
            svc.create_collection(&economy)
                .await
                .context("create economy collection")?
                .success
        );
        let read_back = svc
            .collection("sc_economy")
            .await?
            .context("economy read")?;
        assert_eq!(
            read_back.config.context("config")?.storage_class,
            Some(StorageClass::Economy as i32)
        );
        let status = read_back.storage_class_status.context("status")?;
        assert_eq!(
            status.state,
            crate::proto::proximadb_v1::storage_class_status::State::NotActive as i32
        );
        assert!(
            status.reason.contains("SC4"),
            "reason names the missing dependency: {}",
            status.reason
        );

        // Performance without a storage path: rejected, naming the requirement.
        let perf_no_path = CollectionConfig {
            name: "sc_perf_no_path".to_string(),
            dimension: 8,
            storage_engine: Some(StorageEngine::Sst as i32),
            storage_class: Some(StorageClass::Performance as i32),
            ..Default::default()
        };
        let error = svc
            .create_collection(&perf_no_path)
            .await
            .expect_err("performance over the local default location is rejected");
        let message = error.to_string();
        assert!(message.contains("object-store-backed"), "{message}");
        assert!(message.contains("file://./data"), "{message}");

        // Performance on a non-SST engine: rejected, naming the engine.
        let perf_helix = CollectionConfig {
            name: "sc_perf_helix".to_string(),
            dimension: 8,
            storage_engine: Some(StorageEngine::Helix as i32),
            storage_class: Some(StorageClass::Performance as i32),
            ..Default::default()
        };
        let error = svc
            .create_collection(&perf_helix)
            .await
            .expect_err("performance on a non-PAX engine is rejected");
        assert!(error.to_string().contains("SST"), "{error}");

        // Performance with SST + an object-store path: ADMITTED. The
        // in-process create then fails at storage-directory creation (no S3
        // backend in the test process) — the assertion is that this failure
        // is the environmental one, never a storage_class rejection: an
        // admission failure would name the class/placement and fire BEFORE
        // any side effect.
        let perf_ok = CollectionConfig {
            name: "sc_perf_ok".to_string(),
            dimension: 8,
            storage_engine: Some(StorageEngine::Sst as i32),
            storage_class: Some(StorageClass::Performance as i32),
            storage_config: Some(crate::proto::proximadb_v1::StorageConfig {
                storage_path: Some("s3://bench/collections/sc_perf_ok".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let error = svc
            .create_collection(&perf_ok)
            .await
            .expect_err("no S3 backend in-process: directories cannot be created")
            .to_string();
        assert!(
            !error.contains("storage_class") && !error.contains("object-store-backed"),
            "admission must have passed; got: {error}"
        );
        // The environmental signature: storage-directory creation over the
        // object-store location (no S3 backend in-process).
        assert!(error.contains("storage directories"), "{error}");

        // Configured remote DEFAULT (no client path): admissible — the operator's
        // configured location is the resolved placement and is authorized by
        // construction (a bare default StorageConfig resolves to the local
        // "file://./data" and is NOT admissible, per the arm above).
        let remote_default_svc = CollectionService::new(StorageConfig {
            storage_locations: vec![crate::core::config::StorageLocation {
                url: "s3://bench/datasets".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        })
        .await
        .context("service with a configured remote location")?;
        let perf_default = CollectionConfig {
            name: "sc_perf_configured_default".to_string(),
            dimension: 8,
            storage_class: Some(StorageClass::Performance as i32),
            ..Default::default()
        };
        // Admitted: the create fails only at storage-directory creation over
        // the configured s3:// location (environmental), never at admission.
        let error = remote_default_svc
            .create_collection(&perf_default)
            .await
            .expect_err("no S3 backend in-process: directories cannot be created")
            .to_string();
        assert!(
            !error.contains("storage_class") && !error.contains("object-store-backed"),
            "admission must have passed; got: {error}"
        );

        // An explicit path under ANOTHER tenant's prefix is rejected for that
        // tenant, at admission, on the resolved output.
        let other_tenant_path = CollectionConfig {
            name: "sc_perf_other_tenant".to_string(),
            dimension: 8,
            storage_engine: Some(StorageEngine::Sst as i32),
            storage_class: Some(StorageClass::Performance as i32),
            storage_config: Some(crate::proto::proximadb_v1::StorageConfig {
                storage_path: Some("s3://bench/tenants/other-tenant/coll".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let ctx = crate::storage::tenant::TenantContext::for_tenant_id("acme");
        let error = svc
            .create_collection_with_tenant_context(&other_tenant_path, Some(&ctx))
            .await
            .expect_err("another tenant's prefix is rejected at admission");
        let message = error.to_string();
        assert!(message.contains("other-tenant"), "{message}");
        assert!(message.contains("acme"), "{message}");

        Ok(())
    }

    /// The typed + tenant create directories sit under the TENANT-scoped
    /// base the storage assignment records — the base flush, compaction and
    /// the residency grant resolve paths from — not the raw base (the
    /// directory a tenant's typed segments never landed in).
    #[test]
    fn create_directories_follow_the_tenant_scoped_base() {
        let identity = CollectionIdentity {
            account_id: 7,
            namespace_id: 5,
            collection_id: 9,
        };
        let tenant_base = StoragePath::tenant_root_path("s3://bucket/data", "acme");
        let (wal, data, indexes) = collection_storage_dirs(&tenant_base, "42", Some(identity));
        assert_eq!(
            data,
            "s3://bucket/data/tenants/acme/accounts/000007/005/000009/data"
        );
        assert!(
            wal.starts_with("s3://bucket/data/tenants/acme/accounts/"),
            "{wal}"
        );
        assert!(
            indexes.starts_with("s3://bucket/data/tenants/acme/accounts/"),
            "{indexes}"
        );
        assert_eq!(
            format!("{data}/"),
            crate::services::collection::storage_class::residency_prefixes(
                &tenant_base,
                "42",
                Some(identity)
            )[1],
            "the grant protects exactly the directory created"
        );
        // Legacy layout: byte-identical to the historical tenant-aware paths.
        let (wal, data, indexes) = collection_storage_dirs(&tenant_base, "42", None);
        assert_eq!(
            (wal, data, indexes),
            (
                StoragePath::collection_wal_path_with_tenant(
                    "s3://bucket/data",
                    Some("acme"),
                    "42"
                ),
                StoragePath::collection_data_path_with_tenant(
                    "s3://bucket/data",
                    Some("acme"),
                    "42"
                ),
                StoragePath::collection_index_path_with_tenant(
                    "s3://bucket/data",
                    Some("acme"),
                    "42"
                ),
            )
        );
    }

    #[test]
    fn storage_roots_cover_both_layouts() {
        let identity = CollectionIdentity {
            account_id: 7,
            namespace_id: 5,
            collection_id: 9,
        };
        assert_eq!(collection_storage_roots("/b/", "42", None), vec!["/b/42"]);
        assert_eq!(
            collection_storage_roots("/b", "42", Some(identity)),
            vec![
                "/b/42".to_string(),
                "/b/accounts/000007/005/000009".to_string()
            ]
        );
    }

    /// Dropping a typed collection deletes its account-rooted directory too
    /// (cleanup used to delete only the legacy `{base}/{uuid}` root, leaving
    /// every typed segment in storage).
    #[tokio::test]
    async fn drop_cleanup_deletes_the_typed_root() -> Result<()> {
        let temp_dir = tempfile::TempDir::new().context("temp dir")?;
        let catalog_url = format!("file://{}/catalog", temp_dir.path().display());
        let data_url = format!("file://{}/data", temp_dir.path().display());
        let mgr = Arc::new(CatalogManager::new());
        mgr.create_native_catalog("default", &catalog_url)
            .await
            .context("create test catalog")?;
        let svc = CollectionService::new(StorageConfig {
            storage_locations: vec![crate::core::config::StorageLocation {
                url: data_url.clone(),
                ..Default::default()
            }],
            ..Default::default()
        })
        .await
        .context("collection service")?
        .with_catalog_manager(mgr);
        let config = CollectionConfig {
            name: "typed_drop".to_string(),
            dimension: 8,
            ..Default::default()
        };
        assert!(
            svc.create_collection(&config)
                .await
                .context("create")?
                .success
        );
        let mut record = svc.collection("typed_drop").await?.context("record")?;
        let identity = CollectionIdentity {
            account_id: 7,
            namespace_id: 5,
            collection_id: 9,
        };
        let assignment = record.storage_assignment.as_mut().context("assignment")?;
        assignment.typed_account_id = Some(identity.account_id);
        assignment.typed_namespace_id = Some(u32::from(identity.namespace_id));
        assignment.typed_collection_id = Some(identity.collection_id);
        let base = assignment.base_location.clone();
        svc.upsert_collection_catalog_asset(&record, Some(identity))
            .await
            .context("record typed identity")?;
        let roots = collection_storage_roots(&base, &record.id, Some(identity));
        let typed_root = std::path::PathBuf::from(roots[1].trim_start_matches("file://"));
        std::fs::create_dir_all(typed_root.join("data")).context("typed dirs")?;
        std::fs::write(typed_root.join("data/L0_x.pax"), b"segment").context("segment")?;

        svc.cleanup_storage_directories("typed_drop", &record.id)
            .await
            .context("cleanup")?;
        assert!(
            !typed_root.exists(),
            "typed root must be deleted: {}",
            typed_root.display()
        );
        let legacy_root = std::path::PathBuf::from(roots[0].trim_start_matches("file://"));
        assert!(!legacy_root.exists(), "legacy root must be deleted");
        Ok(())
    }

    /// TD-TIER-1 SC5 (review S2): the access tier is admitted on the resolved
    /// placement at create and on the existing placement at update — refused
    /// on local storage (it would be dropped silently), refused for unknown
    /// numbers (incl. archive's raw 4), admitted on an object-store path.
    #[tokio::test]
    async fn access_tier_admission_on_create_and_update() -> Result<()> {
        use crate::proto::proximadb_v1::AccessTier;

        let (_temp_dir, svc) = catalog_backed_service().await?;

        // Local default placement: an explicit tier is refused, by name.
        let local_tiered = CollectionConfig {
            name: "tier_local".to_string(),
            dimension: 8,
            default_access_tier: Some(AccessTier::Cool as i32),
            ..Default::default()
        };
        let message = svc
            .create_collection(&local_tiered)
            .await
            .expect_err("a tier on local storage is refused")
            .to_string();
        assert!(message.contains("object-store placement"), "{message}");

        // Archive's provider number (4) is not an AccessTier: refused.
        let archive = CollectionConfig {
            name: "tier_archive".to_string(),
            dimension: 8,
            default_access_tier: Some(4),
            storage_config: Some(crate::proto::proximadb_v1::StorageConfig {
                storage_path: Some("s3://bench/collections/tier_archive".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let message = svc
            .create_collection(&archive)
            .await
            .expect_err("unknown tier number is refused")
            .to_string();
        assert!(message.contains("default_access_tier"), "{message}");

        // Object-store path: ADMITTED (the in-process create then fails only
        // at storage-directory creation — no S3 backend here).
        let s3_tiered = CollectionConfig {
            name: "tier_s3".to_string(),
            dimension: 8,
            default_access_tier: Some(AccessTier::Cold as i32),
            storage_config: Some(crate::proto::proximadb_v1::StorageConfig {
                storage_path: Some("s3://bench/collections/tier_s3".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let message = svc
            .create_collection(&s3_tiered)
            .await
            .expect_err("no S3 backend in-process")
            .to_string();
        assert!(
            !message.contains("default_access_tier") && !message.contains("object-store placement"),
            "admission must have passed; got: {message}"
        );

        // Update on a local collection: refused before persisting; the
        // recorded tier stays unset.
        let plain = CollectionConfig {
            name: "tier_update".to_string(),
            dimension: 8,
            ..Default::default()
        };
        assert!(
            svc.create_collection(&plain)
                .await
                .context("create")?
                .success
        );
        let update = CollectionConfig {
            default_access_tier: Some(AccessTier::Cool as i32),
            ..Default::default()
        };
        let message = svc
            .update_collection("tier_update", Some(update))
            .await
            .expect_err("a tier on a local placement is refused at update")
            .to_string();
        assert!(message.contains("object-store placement"), "{message}");
        let read_back = svc.collection("tier_update").await?.context("read")?;
        assert_eq!(
            read_back.config.context("config")?.default_access_tier,
            None
        );
        Ok(())
    }

    /// TD-TIER-1 SC1/SC2 update semantics: an unrelated update preserves the
    /// recorded class (no silent reset); an unknown number is rejected; a
    /// class change re-runs admission; and a storage-path mutation is
    /// rejected as an immutable-placement change (it used to be silently
    /// dropped by the merge).
    #[tokio::test]
    async fn update_collection_storage_class_semantics() -> Result<()> {
        use crate::proto::proximadb_v1::StorageClass;

        let (_temp_dir, svc) = catalog_backed_service().await?;

        let config = CollectionConfig {
            name: "sc_update".to_string(),
            dimension: 8,
            storage_class: Some(StorageClass::Economy as i32),
            ..Default::default()
        };
        assert!(
            svc.create_collection(&config)
                .await
                .context("create")?
                .success
        );

        // Unrelated update (description only): class preserved.
        let unrelated = CollectionConfig {
            description: Some("tweaked".to_string()),
            ..Default::default()
        };
        assert!(
            svc.update_collection("sc_update", Some(unrelated))
                .await?
                .success
        );
        let read_back = svc.collection("sc_update").await?.context("read")?;
        assert_eq!(
            read_back.config.context("config")?.storage_class,
            Some(StorageClass::Economy as i32),
            "an unrelated update must not reset the storage class"
        );

        // Unknown number on update: rejected.
        let bad = CollectionConfig {
            storage_class: Some(-7),
            ..Default::default()
        };
        let error = svc
            .update_collection("sc_update", Some(bad))
            .await
            .expect_err("unknown storage class number on update is rejected");
        assert!(error.to_string().contains("-7"), "{error}");

        // Same-value re-apply: an accepted idempotent no-op.
        let same = CollectionConfig {
            storage_class: Some(StorageClass::Economy as i32),
            ..Default::default()
        };
        assert!(
            svc.update_collection("sc_update", Some(same))
                .await?
                .success
        );
        let read_back = svc.collection("sc_update").await?.context("read")?;
        assert_eq!(
            read_back.config.context("config")?.storage_class,
            Some(StorageClass::Economy as i32)
        );

        // SC3-2b: a DIFFERENT class is re-admitted, not rejected. Entry to
        // PERFORMANCE re-runs the SC2 floor first: this collection's local
        // placement (file://) is inadmissible, so the update is refused
        // BEFORE the config persists — the recorded economy class survives.
        // (A pool-less deployment refuses one step later, at the registry
        // peek — covered by the registry's own gates.)
        let perf = CollectionConfig {
            storage_class: Some(StorageClass::Performance as i32),
            ..Default::default()
        };
        let error = svc
            .update_collection("sc_update", Some(perf))
            .await
            .expect_err("PERFORMANCE entry onto an inadmissible placement is refused");
        let message = error.to_string();
        assert!(message.contains("object-store-backed"), "{message}");
        let read_back = svc.collection("sc_update").await?.context("read")?;
        assert_eq!(
            read_back.config.context("config")?.storage_class,
            Some(StorageClass::Economy as i32),
            "a refused class change must not alter the recorded class"
        );

        // economy → balanced: a real mutation through the surface — the
        // class changes and the generation bumps monotonically (legacy
        // heuristic is 1 for a class persisted at create ⇒ now 2).
        let balanced = CollectionConfig {
            storage_class: Some(StorageClass::Balanced as i32),
            ..Default::default()
        };
        assert!(
            svc.update_collection("sc_update", Some(balanced))
                .await?
                .success
        );
        let read_back = svc.collection("sc_update").await?.context("read")?;
        let config = read_back.config.context("config")?;
        assert_eq!(config.storage_class, Some(StorageClass::Balanced as i32));
        assert_eq!(
            config.policy_generation,
            Some(2),
            "an actual class change bumps the monotonic generation"
        );

        // balanced → economy: bumps again; same-value re-apply does not.
        let economy = CollectionConfig {
            storage_class: Some(StorageClass::Economy as i32),
            ..Default::default()
        };
        assert!(
            svc.update_collection("sc_update", Some(economy))
                .await?
                .success
        );
        let read_back = svc.collection("sc_update").await?.context("read")?;
        assert_eq!(
            read_back.config.context("config")?.policy_generation,
            Some(3)
        );
        let same_again = CollectionConfig {
            storage_class: Some(StorageClass::Economy as i32),
            ..Default::default()
        };
        assert!(
            svc.update_collection("sc_update", Some(same_again))
                .await?
                .success
        );
        let read_back = svc.collection("sc_update").await?.context("read")?;
        assert_eq!(
            read_back.config.context("config")?.policy_generation,
            Some(3),
            "a same-value re-apply does not bump the generation"
        );

        // Storage-path mutation: rejected as an immutable-placement change.
        let move_config = CollectionConfig {
            storage_config: Some(crate::proto::proximadb_v1::StorageConfig {
                storage_path: Some("s3://other/place".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let error = svc
            .update_collection("sc_update", Some(move_config))
            .await
            .expect_err("storage path mutation is rejected");
        assert!(error.to_string().contains("cannot be changed"), "{error}");

        Ok(())
    }

    /// TD-VIPER-1 S2 (ADR-093) create-path negative control: the manager-level
    /// guard must refuse BOTH the deprecated VIPER selection AND any engine
    /// that cannot construct (MMAP/Hybrid are unimplemented) — before the
    /// collection is persisted, not silently at first flush/read.
    #[tokio::test]
    async fn create_collection_rejects_viper_and_unconstructible_engines() -> Result<()> {
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("collection service")?;
        for (engine, label) in [
            (StorageEngine::Viper, "viper (ADR-093 deprecation)"),
            (StorageEngine::Mmap, "mmap (unimplemented)"),
            (StorageEngine::Hybrid, "hybrid (unimplemented)"),
        ] {
            let config = CollectionConfig {
                name: format!(
                    "create_reject_{}",
                    label.split(' ').next().unwrap_or("engine")
                ),
                dimension: 8,
                storage_engine: Some(engine as i32),
                primary_index: Some("default".to_string()),
                auto_index_selection: Some(false),
                ..Default::default()
            };
            let result = service.create_collection(&config).await;
            assert!(
                result.is_err(),
                "create must reject {label}; got: {result:?}"
            );
        }

        Ok(())
    }

    /// TD-VIPER-1 S2 (ADR-093): `update_collection` must refuse a VIPER
    /// storage_engine the same way `create_collection` does — the guard
    /// travels with the field, not just the one wired creation call site.
    #[tokio::test]
    async fn update_collection_rejects_viper_storage_engine() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().context("temp dir")?;
        let temp_path = format!("file://{}", temp_dir.path().display());
        let mgr = Arc::new(CatalogManager::new());
        mgr.create_native_catalog("default", &temp_path)
            .await
            .context("create test catalog")?;
        let svc = CollectionService::new(StorageConfig::default())
            .await
            .context("collection service")?
            .with_catalog_manager(mgr);

        let name = "viper_update_reject";
        let config = CollectionConfig {
            name: name.to_string(),
            dimension: 8,
            storage_engine: Some(StorageEngine::Sst as i32),
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            ..Default::default()
        };
        assert!(
            svc.create_collection(&config)
                .await
                .context("create collection")?
                .success
        );

        let update = CollectionConfig {
            storage_engine: Some(StorageEngine::Viper as i32),
            ..Default::default()
        };
        let result = svc.update_collection(name, Some(update)).await;
        assert!(
            result.is_err(),
            "update_collection must reject a VIPER storage_engine, got: {result:?}"
        );

        Ok(())
    }
}

// ── CollectionPort impl ───────────────────────────────────────────────────────

#[async_trait::async_trait]
impl proximadb_runtime::CollectionPort for CollectionService {
    async fn get_collection(
        &self,
        identifier: &str,
        tenant_id: Option<&str>,
    ) -> anyhow::Result<Option<crate::proto::proximadb_v1::Collection>> {
        self.collection_for_tenant(identifier, tenant_id).await
    }

    async fn enrich_collection_status(
        &self,
        collection: &mut crate::proto::proximadb_v1::Collection,
    ) {
        self.overlay_live_residency(collection).await;
    }

    async fn create_collection(
        &self,
        mut config: crate::proto::proximadb_v1::CollectionConfig,
        tenant_id: Option<&str>,
    ) -> anyhow::Result<crate::proto::proximadb_v1::Collection> {
        let ctx = self.load_tenant_context(tenant_id)?;
        // Stamp the creating tenant as owner (the tenant-context path below
        // does the same when a context exists). Without a creating tenant the
        // owner is never taken from the config: client-supplied owner signals
        // are dropped.
        match tenant_id {
            Some(tenant) => proximadb_tenant::stamp_collection_owner(&mut config, tenant),
            None => proximadb_tenant::clear_collection_owner(&mut config),
        }
        let resp = self
            .create_collection_with_tenant_context(&config, ctx.as_ref())
            .await?;
        resp.collection.ok_or_else(|| {
            anyhow::anyhow!(
                "create_collection returned no collection: error_code={:?}",
                resp.error_code
            )
        })
    }

    async fn update_collection(
        &self,
        id: &str,
        config: crate::proto::proximadb_v1::CollectionConfig,
        tenant_id: Option<&str>,
    ) -> anyhow::Result<crate::proto::proximadb_v1::Collection> {
        // Scoped to the request tenant (no tenant ⇒ deny).
        if self.collection_for_tenant(id, tenant_id).await?.is_none() {
            return Err(
                proximadb_search_types::client_argument::CollectionNotFound::new(id).into(),
            );
        }
        let resp = CollectionService::update_collection(self, id, Some(config)).await?;
        resp.collection.ok_or_else(|| {
            anyhow::anyhow!(
                "update_collection returned no collection: error_code={:?}",
                resp.error_code
            )
        })
    }

    async fn delete_collection(&self, id: &str, tenant_id: Option<&str>) -> anyhow::Result<bool> {
        // Scoped to the request tenant (no tenant ⇒ deny).
        let ctx = self.request_tenant_context(tenant_id)?;
        let resp = self
            .delete_collection_with_tenant_context(id, ctx.as_ref())
            .await?;
        Ok(resp.success)
    }

    async fn list_collections(
        &self,
        tenant_id: Option<&str>,
    ) -> anyhow::Result<Vec<crate::proto::proximadb_v1::Collection>> {
        // Scoped to the request tenant (no tenant ⇒ an empty listing).
        let ctx = self.request_tenant_context(tenant_id)?;
        self.list_collections_with_tenant_context(ctx.as_ref())
            .await
    }

    async fn resolve_collection_id(&self, identifier: &str) -> anyhow::Result<Option<String>> {
        CollectionService::resolve_collection_id(self, identifier).await
    }

    async fn get_collection_internal(
        &self,
        identifier: &str,
    ) -> anyhow::Result<Option<crate::proto::proximadb_v1::Collection>> {
        self.collection_internal(identifier).await
    }

    async fn list_collections_internal(
        &self,
    ) -> anyhow::Result<Vec<crate::proto::proximadb_v1::Collection>> {
        self.list_collections().await
    }

    async fn set_collection_schema_columns(
        &self,
        id: &str,
        columns: &[proximadb_runtime::CollectionSchemaColumn],
        tenant_id: Option<&str>,
    ) -> anyhow::Result<()> {
        // ADR-047 / TD-TBL-1: persist the canonical ProximaType columns as a
        // catalog-asset sidecar. Resolve the SAME asset `upsert_collection_catalog_asset`
        // writes (config → table identifier → tenant-scoped) so the sidecar lands
        // on the exact table the narrow config lives on. Best-effort no-op when
        // there is no catalog or no such collection (preserves the trait default).
        let Some(catalog_manager) = &self.catalog_manager else {
            return Ok(());
        };
        let Some(collection) = self.get_collection(id, tenant_id).await? else {
            return Ok(());
        };
        let Some(config) = collection.config.as_ref() else {
            return Ok(());
        };
        let catalog = catalog_manager.default_catalog().await?;
        let identifier = Self::tenant_scoped_identifier(
            self.tenant_namespaces_on(),
            Self::collection_tenant_id(&collection).as_deref(),
            crate::storage::metadata::collection_mapping::collection_table_identifier(config),
        );
        // TD-CATREPLACE-1 (review m4): fail closed. The canonical columns are
        // the schema authority; silently dropping them while reporting success
        // (which the old best-effort no-op did once the caller reordered to
        // canonical-first) leaves the collection wearing a schema it does not
        // have. A collection without its catalog asset is already inconsistent.
        if !catalog.table_exists(&identifier).await? {
            anyhow::bail!(
                "collection '{}' has no catalog asset to hold its schema columns",
                id
            );
        }
        let mut schema = catalog.get_table(&identifier).await?;
        // ADR-048 P1: the canonical schema IS the typed catalog columns (200+
        // band), not a properties-bag sidecar. Preserve every reserved column
        // (id < 200 — system/embedding/v1 filterable) and replace only the
        // canonical band. Remove the legacy `collection.canonical_schema`
        // property so the typed columns are the sole authority (mixed-read-safe:
        // `get_collection_schema_columns` reads typed columns first, then the
        // transitional property for catalogs not yet migrated).
        let base = crate::storage::metadata::collection_mapping::CANONICAL_COLUMN_ID_BASE;
        schema.columns.retain(|c| c.id < base);
        schema
            .properties
            .remove(crate::storage::metadata::collection_mapping::CANONICAL_SCHEMA_PROPERTY);
        // TD-GRPCCREATE-1: a column that repeats a reserved column's name folds
        // into it when compatible (recorded on the reserved column, so the typed
        // columns stay the one authority) and is rejected otherwise. Runs even
        // for an empty list, to clear earlier fold markers.
        crate::storage::metadata::collection_mapping::merge_canonical_band(
            &mut schema,
            crate::storage::metadata::collection_mapping::collection_schema_columns_to_catalog_columns(
                columns,
            ),
        )?;
        // TD-GRPCCREATE-1: validate before touching the table (replace_table
        // does). TD-CATREPLACE-1: replace in place — no drop window.
        proximadb_catalog::schema::normalize_identity(&mut schema);
        proximadb_catalog::schema::validate_schema(&schema)?;
        catalog.replace_table(&identifier, schema).await?;
        Ok(())
    }

    async fn get_collection_schema_columns(
        &self,
        id: &str,
        tenant_id: Option<&str>,
    ) -> anyhow::Result<Option<Vec<proximadb_runtime::CollectionSchemaColumn>>> {
        let Some(catalog_manager) = &self.catalog_manager else {
            return Ok(None);
        };
        let Some(collection) = self.get_collection(id, tenant_id).await? else {
            return Ok(None);
        };
        let Some(config) = collection.config.as_ref() else {
            return Ok(None);
        };
        let catalog = catalog_manager.default_catalog().await?;
        let identifier = Self::tenant_scoped_identifier(
            self.tenant_namespaces_on(),
            Self::collection_tenant_id(&collection).as_deref(),
            crate::storage::metadata::collection_mapping::collection_table_identifier(config),
        );
        if !catalog.table_exists(&identifier).await.unwrap_or(false) {
            return Ok(None);
        }
        let schema = catalog.get_table(&identifier).await?;
        // ADR-048 P1: read the canonical schema from the typed 200+ columns.
        let typed = crate::storage::metadata::collection_mapping::catalog_columns_to_collection_schema_columns(&schema.columns);
        if !typed.is_empty() {
            return Ok(Some(typed));
        }
        // Transitional read (mixed-read-safe): a catalog written before ADR-048
        // P1 carries the canonical schema as a `collection.canonical_schema`
        // property. Serve it until the collection is next evolved (which writes
        // typed columns), after which the property is gone.
        Ok(schema
            .properties
            .get(crate::storage::metadata::collection_mapping::CANONICAL_SCHEMA_PROPERTY)
            .and_then(|json| {
                crate::storage::metadata::catalog_config::canonical_schema_columns_from_json(json)
            }))
    }
}
