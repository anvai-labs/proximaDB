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

        Ok(Self {
            filesystem_factory,
            index_config_cache: Arc::new(dashmap::DashMap::new()),
            storage_config,
            catalog_manager: None,
            syscat_cache: None,    // Built in `with_catalog_manager`.
            syscat_warm_dir: None, // Set via `with_syscat_warm_dir` / env.
            tenant_namespaces: std::sync::atomic::AtomicBool::new(Self::tenant_namespaces_enabled()),
            tenant_manager: None, // Will be set via with_tenant_manager()
            rbac_enforcer: None,  // Will be set via with_rbac_enforcer()
            #[cfg(feature = "experimental-turboquant")]
            turboquant_registry: None, // Will be set via with_turboquant_registry()
        })
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

        let Some(collection_tenant) = Self::collection_tenant_id(&collection) else {
            warn!(
                "🚨 Collection '{}' is missing tenant metadata; denying tenant-scoped access",
                collection_identifier
            );
            return Ok(None);
        };

        if collection_tenant != tenant_ctx.tenant_id {
            warn!(
                "🚨 Cross-tenant access attempt blocked: user tenant {} tried to access collection owned by tenant {}",
                tenant_ctx.tenant_id, collection_tenant
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

    /// Get collection with tenant validation
    pub async fn get_collection_with_tenant_context(
        &self,
        collection_name: &str,
        tenant_context: Option<&crate::storage::tenant::TenantContext>,
    ) -> Result<Option<crate::proto::proximadb_v1::Collection>> {
        if let Some(tenant_ctx) = tenant_context.filter(|_| self.tenant_manager.is_some()) {
            let collection = self
                .validate_tenant_collection_access(collection_name, tenant_ctx)
                .await?;

            if collection.is_some() {
                debug!(
                    "✅ Tenant ownership validation passed for collection access: tenant={}, collection={}",
                    tenant_ctx.tenant_id, collection_name
                );
            }

            return Ok(collection);
        }

        // Proceed with normal collection retrieval
        self.collection(collection_name).await
    }

    /// List all collections, filtered to the given tenant context if multi-tenant mode is active.
    pub async fn list_collections_with_tenant_context(
        &self,
        tenant_context: Option<&crate::storage::tenant::TenantContext>,
    ) -> Result<Vec<Collection>> {
        let collections = self.list_collections().await?;

        if let Some(tenant_ctx) = tenant_context.filter(|_| self.tenant_manager.is_some()) {
            if let Some(ref tenant_manager) = self.tenant_manager
                && !tenant_manager.is_tenant_active(&tenant_ctx.tenant_id)
            {
                warn!(
                    "🚨 Tenant '{}' is not active; returning empty collection list",
                    tenant_ctx.tenant_id
                );
                return Ok(Vec::new());
            }

            let filtered = collections
                .into_iter()
                .filter(|collection| {
                    Self::collection_tenant_id(collection).as_deref() == Some(&tenant_ctx.tenant_id)
                })
                .collect();
            return Ok(filtered);
        }

        Ok(collections)
    }

    /// Delete collection with tenant validation
    pub async fn delete_collection_with_tenant_context(
        &self,
        collection_name: &str,
        tenant_context: Option<&crate::storage::tenant::TenantContext>,
    ) -> Result<CollectionServiceResponse> {
        debug!("🗑️ Deleting collection: {}", collection_name);

        if let Some(tenant_ctx) = tenant_context.filter(|_| self.tenant_manager.is_some()) {
            let collection = self
                .validate_tenant_collection_access(collection_name, tenant_ctx)
                .await?;

            if collection.is_none() {
                return Ok(CollectionServiceResponse::error(
                    format!(
                        "TENANT_ACCESS_DENIED: collection {} is not accessible to tenant {}",
                        collection_name, tenant_ctx.tenant_id
                    ),
                    0,
                ));
            }

            debug!(
                "✅ Tenant ownership validation passed for collection deletion: tenant={}, collection={}",
                tenant_ctx.tenant_id, collection_name
            );
        }

        let response = self.delete_collection(collection_name).await?;

        // Bump the corpus_version for (tenant, collection) so corpus-version-keyed
        // caches (PlanCache, the per-tenant system-catalog cache) invalidate on
        // the next lookup. A delete definitionally invalidates any cached
        // metadata/plans. Bumped **unconditionally** via the effective tenant —
        // anonymous/single-tenant deletes (no threaded context) must invalidate
        // too, keyed by the default bucket reads use.
        if response.success {
            let tenant = Self::version_tenant(tenant_context);
            let version = crate::catalog::CorpusVersionRegistry::global()
                .bump(tenant, collection_name)
                .await;
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
        let config = CollectionConfig {
            name: name.to_string(),
            dimension,
            storage_engine: Some(StorageEngine::Sst as i32),
            enable_proxima_record: Some(true),
            filterable_columns,
            ..Default::default()
        };
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
            return Err(anyhow!(
                "VIPER is deprecated (ADR-093) and can no longer be selected for new \
                 collections: use SST (default), NOVA (columnar analytics), HELIX, or the \
                 DataFusion/Parquet path for warehouse-shaped analytics"
            ));
        }
        let previous_engine_field = enriched_config.storage_engine;
        enriched_config.storage_engine = Some(selected_engine as i32);
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

        // NEW: Add tenant metadata to collection if tenant context is provided
        if let Some(tenant_ctx) = tenant_context {
            enriched_config
                .tags
                .retain(|tag| !tag.starts_with("tenant:") && tag != "tenant_isolated:true");

            // Add tenant ID to collection tags for tenant isolation (metadata field doesn't exist)
            enriched_config
                .tags
                .push(format!("tenant:{}", tenant_ctx.tenant_id));
            enriched_config
                .tags
                .push("tenant_isolated:true".to_string());
            enriched_config
                .tags
                .push(format!("created_at:{}", chrono::Utc::now().to_rfc3339()));

            // Set owner field if available
            enriched_config.owner = Some(tenant_ctx.tenant_id.clone());

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

        // Create storage directories (tenant-isolated if multi-tenant mode)
        let tenant_id = tenant_context.map(|ctx| ctx.tenant_id.as_str());
        let _storage_created = self
            .create_storage_directories(
                &base_location,
                &enriched_config.name,
                &uuid,
                tenant_id,
                crate::storage::trait_components::path_resolver::typed_path_identity(
                    typed_identity,
                ),
            )
            .await
            .context("Failed to create storage directories")?;

        // Build tenant-prefixed base location for storage assignment
        let tenant_base_location = match tenant_id {
            Some(tid) => StoragePath::tenant_root_path(&base_location, tid),
            None => base_location.clone(),
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
            return Ok(CollectionServiceResponse::error(
                format!("CATALOG_CREATE_FAILED: {}", e),
                start_time.elapsed().as_micros() as i64,
            ));
        }

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
            let version = crate::catalog::CorpusVersionRegistry::global()
                .bump(tenant, &config.name)
                .await;
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
                    existing_config.storage_engine = new_config.storage_engine;
                }
                if new_config.description.is_some() {
                    existing_config.description = new_config.description;
                }
                if !new_config.tags.is_empty() {
                    existing_config.tags = new_config.tags;
                }
                if new_config.owner.is_some() {
                    existing_config.owner = new_config.owner;
                }
                if !new_config.filterable_columns.is_empty() {
                    existing_config.filterable_columns = new_config.filterable_columns;
                }
                // Add other fields as needed
            } else {
                // No existing config, use the new one
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

        // Record is already a proto Collection, no conversion needed
        let collection = record;

        Ok(CollectionServiceResponse {
            success: true,
            collection: Some(collection),
            storage_path: None,
            // error_message removed -  None,
            error_code: None,
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
    async fn create_storage_directories(
        &self,
        base_location: &str,
        collection_name: &str,
        collection_uuid: &str,
        tenant_id: Option<&str>,
        typed_identity: Option<CollectionIdentity>,
    ) -> Result<Vec<StorageComponentType>> {
        let tenant_info = tenant_id.unwrap_or("(default)");
        info!(
            "🏗️ Creating storage directories for collection {} (UUID: {}, tenant: {}) at base: {}",
            collection_name, collection_uuid, tenant_info, base_location
        );

        let mut created_components = Vec::new();

        // ADR-031 Phase 4c: typed path short-circuits the tenant/legacy layout.
        // The typed helper's `None` branch is byte-identical to the legacy
        // `StoragePath::collection_*_path` (non-tenant) — but the create path
        // here historically used the `_with_tenant` variant. To preserve that
        // exactly when no typed identity is set, fall through to the tenant-aware
        // StoragePath calls below for `None`. For `Some`, use the typed helpers
        // (the typed path has NO tenant slot — Phase 4 hierarchy collapse).
        let (write_buffer_dir, data_dir, indexes_dir) = match typed_identity {
            Some(id) => (
                collection_wal_path_typed(base_location, collection_uuid, Some(id)),
                collection_data_path_typed(base_location, collection_uuid, Some(id)),
                collection_index_path_typed(base_location, collection_uuid, Some(id)),
            ),
            None => (
                StoragePath::collection_wal_path_with_tenant(
                    base_location,
                    tenant_id,
                    collection_uuid,
                ),
                StoragePath::collection_data_path_with_tenant(
                    base_location,
                    tenant_id,
                    collection_uuid,
                ),
                StoragePath::collection_index_path_with_tenant(
                    base_location,
                    tenant_id,
                    collection_uuid,
                ),
            ),
        };

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

            // Delete the entire collection directory (includes write_buffer/, data/, indexes/)
            let collection_dir = format!(
                "{}/{}",
                base_location.trim_end_matches('/'),
                collection_uuid
            );

            match self.filesystem_factory.get_filesystem(base_location) {
                Ok(filesystem) => {
                    // Check if directory exists before attempting to delete
                    match filesystem.exists(&collection_dir).await {
                        Ok(true) => {
                            // Recursively delete the entire collection directory
                            match filesystem.remove_dir_all(&collection_dir).await {
                                Ok(_) => {
                                    info!(
                                        "✅ Deleted entire collection directory: {}",
                                        collection_dir
                                    );
                                    cleaned_components = 3; // All components deleted
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
                            }
                        }
                        Ok(false) => {
                            debug!(
                                "📂 Collection directory {} does not exist (already cleaned up)",
                                collection_dir
                            );
                            cleaned_components = 3; // Count as all cleaned
                        }
                        Err(e) => {
                            warn!(
                                "⚠️ Failed to check existence of collection directory {}: {}",
                                collection_dir, e
                            );
                        }
                    }
                    if cleaned_components == 3 {
                        let purged = crate::storage::engines::sst::core::purge_warm_tier_prefix(
                            &collection_dir,
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
            catalog
                .create_namespace_for_tenant(
                    &identifier.namespace,
                    std::collections::HashMap::new(),
                    Self::collection_tenant_id(collection).as_deref(),
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
            preserved_canonical_columns = existing
                .columns
                .iter()
                .filter(|c| {
                    c.id >= crate::storage::metadata::collection_mapping::CANONICAL_COLUMN_ID_BASE
                })
                .cloned()
                .collect();
            if existing
                .properties
                .get("asset.kind")
                .is_none_or(|kind| kind != "collection")
            {
                existing
                    .properties
                    .insert("asset.kind".to_string(), "collection".to_string());
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

                let _ = catalog.drop_table(&identifier, false).await?;
                catalog.create_table(&identifier, existing).await?;
                return Ok(());
            }

            let _ = catalog.drop_table(&identifier, false).await?;
        }
        if let Some(canonical) = preserved_canonical_schema {
            schema.properties.insert(
                crate::storage::metadata::collection_mapping::CANONICAL_SCHEMA_PROPERTY.to_string(),
                canonical,
            );
        }
        if !preserved_canonical_columns.is_empty() {
            schema.columns.extend(preserved_canonical_columns);
        }
        catalog.create_table(&identifier, schema).await?;
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
        if let Some(collection) =
            crate::storage::metadata::collection_mapping::collection_from_catalog_schema(
                &table_id, &schema,
            )?
        {
            return Ok(Some(collection));
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

    let mut collections = Vec::new();
    let mut seen_ids = HashSet::new();
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
            let Some(collection) =
                crate::storage::metadata::collection_mapping::collection_from_catalog_schema(
                    &table_id, &schema,
                )?
            else {
                continue;
            };
            if seen_ids.insert(collection.id.clone()) {
                collections.push(collection);
            }
        }
    }

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
    use crate::catalog::TableIdentifier;
    use crate::proto::proximadb_v1::CollectionConfig;
    use crate::proto::proximadb_v1::{FilterableDataType, IndexingAlgorithm};

    #[tokio::test]
    async fn test_collection_validation() -> Result<()> {
        let service = CollectionService::new(StorageConfig::default())
            .await
            .context("Failed to create collection service for test")?;

        // Valid config
        let valid_config = CollectionConfig {
            name: "valid_collection".to_string(),
            embedding_config: None,
            dimension: 128,
            distance_metric: Some(1),
            storage_engine: Some(1),
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
        };

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
            let config = CollectionConfig {
                name: name.to_string(),
                embedding_config: None,
                dimension: 128,
                distance_metric: Some(1),
                storage_engine: Some(1),
                filterable_columns: vec![],
                index_configs: vec![],
                quantization: None,
                primary_index: Some("default".to_string()),
                auto_index_selection: Some(false),
                description: Some("Test collection".to_string()),
                tags: vec![],
                owner: Some("test".to_string()),
                embedding_models: vec![],
                storage_config: None,
                record_schema: None,
                enable_proxima_record: None,
                text_columns: vec![],
                text_storage_configs: vec![],
                enable_dual_use_embeddings: None,
                canonical_embedding_precision: None,
                permitted_principals: vec![],
                index_policy: None,
                pax_vector_quant: None,
            };

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
            .delete_collection_with_tenant_context(name, None)
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
            .delete_collection_with_tenant_context(name, None)
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

    /// TD-CAT-2a (audit gap G6): creating a `tenant:`-tagged collection records
    /// the owning tenant on the freshly-created namespace. The upsert routes
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
            .get_namespace(&["tdc2a_ns".to_string()])
            .await
            .context("get_namespace")?;
        assert_eq!(
            ns.tenant_id.as_deref(),
            Some("acme"),
            "the freshly-created namespace must record the owning tenant (TD-CAT-2a)"
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
            name: name.to_string(),
            embedding_config: None,
            dimension: 16,
            distance_metric: Some(1),
            storage_engine: Some(1),
            filterable_columns: vec![],
            index_configs: vec![],
            quantization: None,
            primary_index: Some("default".to_string()),
            auto_index_selection: Some(false),
            description: None,
            tags: vec![],
            owner: None,
            embedding_models: vec![],
            storage_config: None,
            record_schema: None,
            enable_proxima_record: None,
            text_columns: vec![],
            text_storage_configs: vec![],
            enable_dual_use_embeddings: None,
            canonical_embedding_precision: None,
            permitted_principals: vec![],
            index_policy: None,
            pax_vector_quant: None,
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
            name: "metric_default_test".to_string(),
            embedding_config: None,
            dimension: 128,
            distance_metric: None,
            storage_engine: Some(StorageEngine::Viper as i32),
            filterable_columns: vec![],
            index_configs: vec![],
            quantization: None,
            primary_index: None,
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
            .set_collection_schema_columns(name, &columns, None)
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
            .get_collection_schema_columns(name, None)
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
                .get_collection_schema_columns("canon_legacy", None)
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
        svc.set_collection_schema_columns(name, &columns, None)
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
            .get_collection_schema_columns(name, None)
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
            .get_collection_schema_columns(name, None)
            .await?
            .expect("transitional read of the legacy sidecar");
        assert_eq!(restored, columns);

        // A subsequent set migrates it to typed columns (property removed).
        let columns2 = vec![canonical_col("n", ProximaType::Int64)];
        svc.set_collection_schema_columns(name, &columns2, None)
            .await?;
        let restored2 = svc
            .get_collection_schema_columns(name, None)
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
        svc.set_collection_schema_columns(name, &columns, None)
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
            .get_collection_schema_columns(name, None)
            .await
            .context("read canonical after update")?
            .expect("canonical sidecar survived unrelated update_collection");
        assert_eq!(restored, columns);

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
        let ctx = self.load_tenant_context(tenant_id)?;
        self.get_collection_with_tenant_context(identifier, ctx.as_ref())
            .await
    }

    async fn create_collection(
        &self,
        config: crate::proto::proximadb_v1::CollectionConfig,
        tenant_id: Option<&str>,
    ) -> anyhow::Result<crate::proto::proximadb_v1::Collection> {
        let ctx = self.load_tenant_context(tenant_id)?;
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
        _tenant_id: Option<&str>,
    ) -> anyhow::Result<crate::proto::proximadb_v1::Collection> {
        let resp = CollectionService::update_collection(self, id, Some(config)).await?;
        resp.collection.ok_or_else(|| {
            anyhow::anyhow!(
                "update_collection returned no collection: error_code={:?}",
                resp.error_code
            )
        })
    }

    async fn delete_collection(&self, id: &str, tenant_id: Option<&str>) -> anyhow::Result<bool> {
        let ctx = self.load_tenant_context(tenant_id)?;
        let resp = self
            .delete_collection_with_tenant_context(id, ctx.as_ref())
            .await?;
        Ok(resp.success)
    }

    async fn list_collections(
        &self,
        tenant_id: Option<&str>,
    ) -> anyhow::Result<Vec<crate::proto::proximadb_v1::Collection>> {
        let ctx = self.load_tenant_context(tenant_id)?;
        self.list_collections_with_tenant_context(ctx.as_ref())
            .await
    }

    async fn resolve_collection_id(&self, identifier: &str) -> anyhow::Result<Option<String>> {
        CollectionService::resolve_collection_id(self, identifier).await
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
        if !catalog.table_exists(&identifier).await.unwrap_or(false) {
            return Ok(());
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
        if !columns.is_empty() {
            schema.columns.extend(
                crate::storage::metadata::collection_mapping::collection_schema_columns_to_catalog_columns(
                    columns,
                ),
            );
        }
        let _ = catalog.drop_table(&identifier, false).await?;
        catalog.create_table(&identifier, schema).await?;
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
