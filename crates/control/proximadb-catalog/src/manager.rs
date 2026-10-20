//! Catalog runtime manager (`CatalogManager`) + the per-table operation-lock
//! registry, extracted from the root `src/catalog` (decomposition Slice 2).
//! `CatalogManager` holds `Arc<dyn Catalog>` backends (in this crate) and an
//! injected [`CatalogFilesystemResolver`] port for object-store catalog URLs,
//! dissolving the catalog->storage up-edge.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, RwLock, Semaphore};
use tracing::info;

use proximadb_storage_filesystem_types::FileSystem;

use crate::cache::CatalogCache;
use crate::{Catalog, TableIdentifier};

/// A table name that resolves, for the requesting tenant, to an entry it may
/// not see. Displays exactly like a missing table so the two are
/// indistinguishable; callers that treat "missing" specially (`DROP ... IF
/// EXISTS`, `CREATE`) downcast to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableNotVisible {
    /// The name as the caller supplied it.
    pub table: String,
}

impl std::fmt::Display for TableNotVisible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Table '{}' does not exist", self.table)
    }
}

impl std::error::Error for TableNotVisible {}

impl TableNotVisible {
    /// Whether `error` (or anything it wraps) is a [`TableNotVisible`].
    pub fn is(error: &anyhow::Error) -> bool {
        error.chain().any(|cause| cause.is::<TableNotVisible>())
    }
}

/// Resolves a [`FileSystem`] backend for an object-store catalog URL
/// (`s3://`, `gs://`, `az://`). Implemented by the root composition root (which
/// wraps `FilesystemFactory`); injected via
/// [`CatalogManager::set_filesystem_resolver`]. Dissolves the catalog->storage
/// up-edge so `CatalogManager` can live in this crate.
#[async_trait::async_trait]
pub trait CatalogFilesystemResolver: Send + Sync {
    /// Resolve the filesystem for `url`, or return an error if no backend is
    /// configured for the scheme.
    async fn get_filesystem(&self, url: &str) -> Result<Arc<dyn FileSystem>>;
}

/// TD-110 S1: per-table, non-reentrant, in-process DML operation lock registry.
///
/// The durable DML lock (`crate::cluster::partition_lease::DmlLockService`)
/// serializes DML *across pods* but is deliberately re-entrant within a pod: a
/// pod re-acquiring its own table lease *renews* it, and the in-memory check
/// skips same-pod/same-scope as compatible. Embedded and single-pod deployments
/// therefore need a separate, *nonreentrant* in-process lock to serialize
/// concurrent connections that touch the same table during a referential-action
/// critical section (ON DELETE CASCADE/RESTRICT child scan -> child mutation ->
/// parent tombstone). Without it, a child row inserted by connection B while
/// connection A's parent DELETE is mid-flight can be orphaned (B's row survives
/// referencing a now-deleted parent).
///
/// Each `(namespace, table)` key maps to a binary semaphore (1 permit);
/// `acquire_owned` yields an owned permit held across the critical section and
/// released on drop. Multi-table critical sections (a cascading DELETE) acquire
/// the whole set in a deterministic `(namespace, name)` order to avoid
/// self-deadlock -- single-table acquirers (INSERT/UPDATE) cannot form a cycle
/// with a sorted multi-table acquirer.
#[derive(Default)]
pub struct TableOpLockRegistry {
    locks: Mutex<HashMap<TableIdentifier, Arc<Semaphore>>>,
}

impl TableOpLockRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            locks: Mutex::new(HashMap::new()),
        }
    }

    /// Resolve (lazily creating) the binary semaphore for a table.
    fn semaphore_for(&self, table_id: &TableIdentifier) -> Arc<Semaphore> {
        // Sync lock held only for the map get/insert — never across an await.
        let mut locks = self.locks.lock();
        locks
            .entry(table_id.clone())
            .or_insert_with(|| Arc::new(Semaphore::new(1)))
            .clone()
    }

    /// Acquire the (non-reentrant) operation lock for one table, blocking until
    /// free. The returned permit is released on drop.
    pub async fn acquire(&self, table_id: &TableIdentifier) -> Result<OwnedSemaphorePermit> {
        self.semaphore_for(table_id)
            .acquire_owned()
            .await
            .map_err(|_| {
                anyhow!(
                    "operation lock for table '{}/{}' was closed unexpectedly",
                    table_id.namespace.join("."),
                    table_id.name
                )
            })
    }

    /// Acquire operation locks for a set of tables in deterministic `(namespace,
    /// name)` order (de-duplicated), blocking until all are held. The returned
    /// permits are released on drop, so bind them to a guard that outlives the
    /// critical section.
    pub async fn acquire_sorted(
        &self,
        mut tables: Vec<TableIdentifier>,
    ) -> Result<Vec<OwnedSemaphorePermit>> {
        tables.sort_by(|a, b| {
            a.namespace
                .cmp(&b.namespace)
                .then_with(|| a.name.cmp(&b.name))
        });
        tables.dedup();
        let mut permits = Vec::with_capacity(tables.len());
        for table_id in &tables {
            permits.push(self.acquire(table_id).await?);
        }
        Ok(permits)
    }
}

impl std::fmt::Debug for TableOpLockRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableOpLockRegistry")
            .finish_non_exhaustive()
    }
}

/// Catalog manager - manages multiple catalog instances
pub struct CatalogManager {
    /// Registered catalogs by name
    catalogs: RwLock<HashMap<String, Arc<dyn Catalog>>>,
    /// Default catalog name
    default_catalog: RwLock<Option<String>>,
    /// Catalog cache for metadata
    cache: Arc<CatalogCache>,
    /// TD-110 S1: per-table, non-reentrant in-process DML operation locks used
    /// to serialize referential-action critical sections within a single pod.
    /// See [`TableOpLockRegistry`].
    op_locks: TableOpLockRegistry,
    /// Object-store catalog filesystem resolver (injected by the root
    /// composition root; `None` for local-only setups). Dissolves the
    /// catalog->storage up-edge -- see [`CatalogFilesystemResolver`].
    fs_resolver: RwLock<Option<Arc<dyn CatalogFilesystemResolver>>>,
}

impl CatalogManager {
    /// Create a new catalog manager
    pub fn new() -> Self {
        Self {
            catalogs: RwLock::new(HashMap::new()),
            default_catalog: RwLock::new(None),
            cache: Arc::new(CatalogCache::new(10000, 300)), // 10K entries, 5min TTL
            op_locks: TableOpLockRegistry::new(),
            fs_resolver: RwLock::new(None),
        }
    }

    /// Create a new catalog manager with custom cache settings
    pub fn with_cache(max_entries: usize, ttl_seconds: u64) -> Self {
        Self {
            catalogs: RwLock::new(HashMap::new()),
            default_catalog: RwLock::new(None),
            cache: Arc::new(CatalogCache::new(max_entries, ttl_seconds)),
            op_locks: TableOpLockRegistry::new(),
            fs_resolver: RwLock::new(None),
        }
    }

    /// Inject the object-store filesystem resolver (root composition root
    /// wraps `FilesystemFactory` behind [`CatalogFilesystemResolver`]).
    /// Required only if `create_native_catalog` / `create_delta_catalog` is
    /// called with an object-store URL; local `file://` setups leave it `None`.
    pub async fn set_filesystem_resolver(&self, resolver: Arc<dyn CatalogFilesystemResolver>) {
        *self.fs_resolver.write().await = Some(resolver);
    }

    /// Resolve the injected `FileSystem` for an object-store catalog URL, or
    /// `None` for local paths (`file://`, bare paths) and `memory://` (which
    /// keeps its pre-existing non-resolver behavior).
    ///
    /// ANY other scheme is an object store — never enumerate schemes here:
    /// the old `s3://|gs://|az://` allowlist let the documented aliases
    /// (`adls://`, `abfs://`, `azure://`, `gcs://`) fall through to the
    /// local-path branch and silently stage catalog metadata in a temp cache
    /// (TD-OBJSTORE-1, #960).
    async fn resolve_object_store_fs(
        &self,
        storage_url: &str,
    ) -> Result<Option<Arc<dyn FileSystem>>> {
        let is_object_store = storage_url.contains("://")
            && !storage_url.starts_with("file://")
            && !storage_url.starts_with("memory://");
        if !is_object_store {
            return Ok(None);
        }
        let resolver = self.fs_resolver.read().await;
        let resolver = resolver.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "object-store catalog URL '{storage_url}' requires a filesystem \
                 resolver -- call CatalogManager::set_filesystem_resolver at startup"
            )
        })?;
        let fs = resolver.get_filesystem(storage_url).await.map_err(|e| {
            anyhow::anyhow!("no filesystem backend for catalog url '{storage_url}': {e}")
        })?;
        Ok(Some(fs))
    }

    /// TD-110 S1: per-table, non-reentrant in-process DML operation locks.
    /// Hosted on `CatalogManager` because it is the single object shared across
    /// all per-connection `DmlService` instances (pgwire builds a fresh
    /// `DmlService` per connection), so a registry here is the one place
    /// concurrent connections actually contend.
    pub fn op_locks(&self) -> &TableOpLockRegistry {
        &self.op_locks
    }

    /// Register a pre-created catalog
    pub async fn register(&self, catalog: Arc<dyn Catalog>) -> Result<()> {
        let name = catalog.name().to_string();
        info!(
            "Registering catalog: {} (type: {})",
            name,
            catalog.catalog_type()
        );

        let mut catalogs = self.catalogs.write().await;
        catalogs.insert(name.clone(), catalog);

        // Set as default if first catalog
        let mut default = self.default_catalog.write().await;
        if default.is_none() {
            *default = Some(name);
        }

        Ok(())
    }

    /// Create and register a native catalog
    pub async fn create_native_catalog(
        &self,
        name: &str,
        storage_url: &str,
    ) -> Result<Arc<dyn Catalog>> {
        use crate::native::NativeCatalogConfig;

        let config = NativeCatalogConfig {
            storage_url: storage_url.to_string(),
            ..Default::default()
        };

        // TD-CAT-1b (S0): object-store catalog URLs persist durably by routing
        // all catalog I/O through an injected `FileSystem` backend resolved
        // from the configured backends. `file://` (and bare local paths) keep
        // the local-`tokio::fs` path (`fs = None`) so the on-disk layout and
        // existing tests are byte-identical — the object-store branch only
        // *relaxes* `NativeCatalog::parse_storage_url`'s fail-closed bail, it
        // does not change the local path. A cloud scheme whose backend feature
        // isn't compiled in surfaces a clear `UnsupportedScheme` error
        // (still strictly safer than silently caching the catalog under /tmp).
        let catalog = if let Some(fs) = self.resolve_object_store_fs(storage_url).await? {
            crate::native::NativeCatalog::new_with_filesystem(
                name.to_string(),
                config,
                self.cache.clone(),
                Some(fs),
            )
            .await?
        } else {
            crate::native::NativeCatalog::new(name.to_string(), config, self.cache.clone()).await?
        };

        let catalog: Arc<dyn Catalog> = Arc::new(catalog);
        self.register(catalog.clone()).await?;
        Ok(catalog)
    }

    /// Create and register an Iceberg catalog
    pub async fn create_iceberg_catalog(
        &self,
        name: &str,
        uri: &str,
        warehouse: &str,
    ) -> Result<Arc<dyn Catalog>> {
        use crate::iceberg::IcebergCatalogConfig;

        let config = IcebergCatalogConfig {
            uri: uri.to_string(),
            warehouse: warehouse.to_string(),
            ..Default::default()
        };

        let catalog =
            crate::iceberg::IcebergCatalog::new(name.to_string(), config, self.cache.clone())
                .await?;

        let catalog: Arc<dyn Catalog> = Arc::new(catalog);
        self.register(catalog.clone()).await?;
        Ok(catalog)
    }

    /// Create and register an AWS Glue catalog
    ///
    /// Requires the `aws` feature flag to be enabled.
    ///
    /// # Arguments
    /// * `name` - Catalog name
    /// * `region` - AWS region (e.g., "us-east-1")
    /// * `catalog_id` - AWS account ID (optional, uses default account if empty)
    ///
    /// # Example
    /// ```ignore
    /// let catalog = manager.create_glue_catalog("glue", "us-east-1", "123456789012").await?;
    /// ```
    #[cfg(feature = "aws")]
    pub async fn create_glue_catalog(
        &self,
        name: &str,
        region: &str,
        catalog_id: &str,
    ) -> Result<Arc<dyn Catalog>> {
        use crate::glue::GlueCatalogConfig;

        let config = GlueCatalogConfig {
            region: region.to_string(),
            catalog_id: catalog_id.to_string(),
            ..Default::default()
        };

        let catalog =
            crate::glue::GlueCatalog::new(name.to_string(), config, self.cache.clone()).await?;

        let catalog: Arc<dyn Catalog> = Arc::new(catalog);
        self.register(catalog.clone()).await?;
        Ok(catalog)
    }

    /// Create and register an AWS Glue catalog (stub for non-AWS builds)
    #[cfg(not(feature = "aws"))]
    pub async fn create_glue_catalog(
        &self,
        _name: &str,
        _region: &str,
        _catalog_id: &str,
    ) -> Result<Arc<dyn Catalog>> {
        Err(anyhow!(
            "AWS Glue catalog requires the 'aws' feature flag. \
             Build with: cargo build --features aws"
        ))
    }

    /// Create and register a Databricks Unity catalog
    ///
    /// Requires the `unity-catalog` feature flag to be enabled.
    ///
    /// # Arguments
    /// * `name` - Catalog name
    /// * `workspace_url` - Databricks workspace URL (e.g., "https://xxx.cloud.databricks.com")
    /// * `token` - Personal access token or OAuth token
    /// * `catalog_name` - Unity catalog name within the workspace
    ///
    /// # Example
    /// ```ignore
    /// let catalog = manager.create_unity_catalog(
    ///     "unity",
    ///     "https://my-workspace.cloud.databricks.com",
    ///     "dapi123...",
    ///     "main"
    /// ).await?;
    /// ```
    #[cfg(feature = "unity-catalog")]
    pub async fn create_unity_catalog(
        &self,
        name: &str,
        workspace_url: &str,
        token: &str,
        catalog_name: &str,
    ) -> Result<Arc<dyn Catalog>> {
        use crate::unity::UnityCatalogConfig;

        let config = UnityCatalogConfig {
            workspace_url: workspace_url.to_string(),
            token: token.to_string(),
            catalog_name: catalog_name.to_string(),
            ..Default::default()
        };

        let catalog =
            crate::unity::UnityCatalog::new(name.to_string(), config, self.cache.clone()).await?;

        let catalog: Arc<dyn Catalog> = Arc::new(catalog);
        self.register(catalog.clone()).await?;
        Ok(catalog)
    }

    /// Create and register a Unity catalog (stub for non-Unity builds)
    #[cfg(not(feature = "unity-catalog"))]
    pub async fn create_unity_catalog(
        &self,
        _name: &str,
        _workspace_url: &str,
        _token: &str,
        _catalog_name: &str,
    ) -> Result<Arc<dyn Catalog>> {
        Err(anyhow!(
            "Databricks Unity catalog requires the 'unity-catalog' feature flag. \
             Build with: cargo build --features unity-catalog"
        ))
    }

    /// Create and register an Apache Polaris catalog (Iceberg REST)
    ///
    /// Requires the `polaris-catalog` feature flag to be enabled.
    ///
    /// # Arguments
    /// * `name` - Catalog name
    /// * `uri` - Polaris server URI
    /// * `warehouse` - Warehouse name
    /// * `credential` - OAuth credential (client_id:client_secret format)
    ///
    /// # Example
    /// ```ignore
    /// let catalog = manager.create_polaris_catalog(
    ///     "polaris",
    ///     "https://polaris.example.com",
    ///     "my_warehouse",
    ///     "client_id:client_secret"
    /// ).await?;
    /// ```
    #[cfg(feature = "polaris-catalog")]
    pub async fn create_polaris_catalog(
        &self,
        name: &str,
        uri: &str,
        warehouse: &str,
        credential: &str,
    ) -> Result<Arc<dyn Catalog>> {
        use crate::polaris::PolarisCatalogConfig;

        let config = PolarisCatalogConfig {
            uri: uri.to_string(),
            warehouse: warehouse.to_string(),
            credential: credential.to_string(),
            ..Default::default()
        };

        let catalog =
            crate::polaris::PolarisCatalog::new(name.to_string(), config, self.cache.clone())
                .await?;

        let catalog: Arc<dyn Catalog> = Arc::new(catalog);
        self.register(catalog.clone()).await?;
        Ok(catalog)
    }

    /// Create and register a Polaris catalog (stub for non-Polaris builds)
    #[cfg(not(feature = "polaris-catalog"))]
    pub async fn create_polaris_catalog(
        &self,
        _name: &str,
        _uri: &str,
        _warehouse: &str,
        _credential: &str,
    ) -> Result<Arc<dyn Catalog>> {
        Err(anyhow!(
            "Apache Polaris catalog requires the 'polaris-catalog' feature flag. \
             Build with: cargo build --features polaris-catalog"
        ))
    }

    /// Create and register a Delta Lake catalog
    ///
    /// Requires the `delta-lake` feature flag to be enabled.
    ///
    /// # Arguments
    /// * `name` - Catalog name
    /// * `storage_url` - Storage URL (s3://bucket/path, file:///path, etc.)
    ///
    /// # Example
    /// ```ignore
    /// let catalog = manager.create_delta_catalog(
    ///     "delta",
    ///     "s3://my-bucket/delta-tables"
    /// ).await?;
    /// ```
    #[cfg(feature = "delta-lake")]
    pub async fn create_delta_catalog(
        &self,
        name: &str,
        storage_url: &str,
    ) -> Result<Arc<dyn Catalog>> {
        use crate::delta::DeltaCatalogConfig;

        let config = DeltaCatalogConfig {
            storage_url: storage_url.to_string(),
            ..Default::default()
        };

        // Object-store URLs route all catalog I/O through the injected
        // `FileSystem` (durable); local paths keep the tokio::fs layout
        // (TD-OBJSTORE-1, #960).
        let filesystem = self.resolve_object_store_fs(storage_url).await?;
        let catalog = crate::delta::DeltaCatalog::new_with_filesystem(
            name.to_string(),
            config,
            self.cache.clone(),
            filesystem,
        )
        .await?;

        let catalog: Arc<dyn Catalog> = Arc::new(catalog);
        self.register(catalog.clone()).await?;
        Ok(catalog)
    }

    /// Create and register a Delta Lake catalog (stub for non-Delta builds)
    #[cfg(not(feature = "delta-lake"))]
    pub async fn create_delta_catalog(
        &self,
        _name: &str,
        _storage_url: &str,
    ) -> Result<Arc<dyn Catalog>> {
        Err(anyhow!(
            "Delta Lake catalog requires the 'delta-lake' feature flag. \
             Build with: cargo build --features delta-lake"
        ))
    }

    /// Create and register an OLTP catalog (PostgreSQL / Neon / Supabase / MariaDB / SQLite).
    ///
    /// The OLTP catalog stores ONLY catalog metadata — record data always stays in ProximaDB's
    /// internal engines (stacked durability mandate). Use for collections < 1 GB.
    ///
    /// Connection string formats:
    /// - `postgres://user:pw@host/db` — PostgreSQL, Neon, Supabase, CockroachDB
    /// - `mysql://user:pw@host/db` — MariaDB, MySQL, TiDB
    /// - `sqlite:///path/catalog.db` — SQLite
    ///
    /// **TD-CAT-10:** This function is gated behind `oltp-catalog` and is **not maintained**.
    #[cfg(feature = "oltp-catalog")]
    pub async fn create_oltp_catalog(
        &self,
        name: &str,
        connection_string: &str,
    ) -> Result<Arc<dyn Catalog>> {
        let config = crate::oltp::OltpCatalogConfig {
            connection_string: connection_string.to_string(),
            ..Default::default()
        };

        let catalog =
            crate::oltp::OltpCatalog::new(name.to_string(), config, self.cache.clone()).await?;

        let catalog: Arc<dyn Catalog> = Arc::new(catalog);
        self.register(catalog.clone()).await?;
        Ok(catalog)
    }

    /// Select the appropriate catalog based on estimated table size.
    ///
    /// - `size_bytes < threshold` → OLTP catalog (if registered)
    /// - `size_bytes >= threshold` → default catalog (lakehouse / native)
    pub async fn catalog_for_size(
        &self,
        size_bytes: u64,
        oltp_threshold_bytes: u64,
    ) -> Result<Arc<dyn Catalog>> {
        let oltp_candidate = if size_bytes < oltp_threshold_bytes {
            let catalogs = self.catalogs.read().await;
            catalogs
                .values()
                .find(|c| c.catalog_type().starts_with("oltp-"))
                .cloned()
        } else {
            None
        };

        if let Some(cat) = oltp_candidate {
            return Ok(cat);
        }
        self.default_catalog().await
    }

    /// Get a catalog by name
    pub async fn get_catalog(&self, name: &str) -> Result<Arc<dyn Catalog>> {
        let catalogs = self.catalogs.read().await;
        catalogs
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Catalog '{}' not found", name))
    }

    /// Get the default catalog
    pub async fn default_catalog(&self) -> Result<Arc<dyn Catalog>> {
        let default = self.default_catalog.read().await;
        let name = default
            .as_ref()
            .ok_or_else(|| anyhow!("No default catalog configured"))?;

        self.get_catalog(name).await
    }

    /// TD-TENANT-1 item 3: SYNC best-effort lookup of the account u32 via the
    /// default catalog's registry (no mint, no I/O) — for the request-hot
    /// `TenantStableIdResolver`. Uses `try_read` on the two backing RwLocks and
    /// returns `None` on (rare) contention — callers treat `None` as fail-closed
    /// deny (safe; the next request retries). Also `None` when no default
    /// catalog is configured or the account is unminted.
    pub fn account_id_u32_lookup(&self, account: &str) -> Option<u32> {
        let account = account.trim();
        if account.is_empty() {
            return None;
        }
        let name = self
            .default_catalog
            .try_read()
            .ok()
            .and_then(|n| n.as_ref().cloned())?;
        let catalogs = self.catalogs.try_read().ok()?;
        catalogs.get(&name)?.account_id_u32_lookup(account)
    }

    /// Mint-or-return a tenant's stable policy key in the default catalog.
    ///
    /// This is deliberately async and control-plane-only: request hot paths use
    /// [`Self::account_id_u32_lookup`], while bootstrap and ABAC provisioning
    /// call this method before publishing state keyed by the id. NativeCatalog
    /// persists its account registry sidecar; SystemCatalog commits the mapping
    /// to its canonical WAL/snapshot. Other catalog kinds return `None` and the
    /// caller must fail closed.
    pub async fn ensure_tenant_stable_id(&self, tenant: &str) -> Result<Option<u64>> {
        let tenant = tenant.trim();
        if tenant.is_empty() {
            return Ok(None);
        }
        Ok(self
            .default_catalog()
            .await?
            .account_id_u32(tenant)
            .await?
            .map(u64::from))
    }

    /// Set the default catalog
    pub async fn set_default_catalog(&self, name: &str) -> Result<()> {
        // Verify catalog exists
        let catalogs = self.catalogs.read().await;
        if !catalogs.contains_key(name) {
            return Err(anyhow!("Catalog '{}' not found", name));
        }
        drop(catalogs);

        let mut default = self.default_catalog.write().await;
        *default = Some(name.to_string());
        Ok(())
    }

    /// List all registered catalogs
    pub async fn list_catalogs(&self) -> Vec<String> {
        let catalogs = self.catalogs.read().await;
        catalogs.keys().cloned().collect()
    }

    /// Unregister a catalog
    pub async fn unregister_catalog(&self, name: &str) -> Result<bool> {
        let mut catalogs = self.catalogs.write().await;
        let removed = catalogs.remove(name).is_some();

        // Clear default if it was this catalog
        if removed {
            let mut default = self.default_catalog.write().await;
            if default.as_deref() == Some(name) {
                *default = catalogs.keys().next().cloned();
            }
        }

        Ok(removed)
    }

    /// Resolve a fully-qualified table name (catalog.namespace.table)
    pub async fn resolve_table(&self, fqn: &str) -> Result<(Arc<dyn Catalog>, TableIdentifier)> {
        let parts: Vec<&str> = fqn.split('.').collect();

        match parts.len() {
            1 => {
                // Just table name - use default catalog and namespace
                let catalog = self.default_catalog().await?;
                let id = TableIdentifier::new(vec!["default".to_string()], parts[0].to_string());
                Ok((catalog, id))
            }
            2 => {
                // namespace.table - use default catalog
                let catalog = self.default_catalog().await?;
                let id = TableIdentifier::new(vec![parts[0].to_string()], parts[1].to_string());
                Ok((catalog, id))
            }
            3 => {
                // catalog.namespace.table
                let catalog = self.get_catalog(parts[0]).await?;
                let id = TableIdentifier::new(vec![parts[1].to_string()], parts[2].to_string());
                Ok((catalog, id))
            }
            _ => {
                // catalog.ns1.ns2...nsN.table
                let catalog = self.get_catalog(parts[0]).await?;
                let namespace: Vec<String> = parts[1..parts.len() - 1]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                let table = parts
                    .last()
                    .ok_or_else(|| anyhow!("Invalid table name: missing table component"))?
                    .to_string();
                let id = TableIdentifier::new(namespace, table);
                Ok((catalog, id))
            }
        }
    }

    /// Resolve a table name within a tenant scope (TD-064). Delegates to
    /// [`Self::resolve_table`] for the catalog + base identifier, then
    /// tenant-prefixes the namespace so each tenant's schema row is distinct
    /// (`orders` → namespace `[tenant, "default"]`). An empty/None tenant
    /// resolves identically to [`Self::resolve_table`] (single-tenant).
    ///
    /// TD-OLAP-18: after the exact resolution, an id that names no registered
    /// table falls back to a *unique* case-insensitive match within the final
    /// namespace (declared-case always wins; ambiguous case-variants stay
    /// unresolved). This is the write-path mirror of the relational frontend's
    /// read-path fold — `INSERT INTO casetbl` after `CREATE TABLE CaseTbl`
    /// resolves. The gate env var is read inline (the canonical
    /// `ident_case_fold_enabled()` lives in the query-layer
    /// `proximadb-relational-frontend`, which this control-layer crate must not
    /// depend on); the name and semantics are those of the registered gate
    /// `PROXIMADB_IDENT_CASE_FOLD` (see ENV_GATE_REGISTRY).
    pub async fn resolve_table_scoped(
        &self,
        fqn: &str,
        tenant: Option<&str>,
    ) -> Result<(Arc<dyn Catalog>, TableIdentifier)> {
        // Structural system-catalog isolation (validate input first): the tenant
        // id becomes `namespace[0]`, so a tenant must never shadow a reserved
        // control-plane / per-tenant system subtree. Those are the
        // underscore-prefixed segments (`_operator`, `_metering`, `_trace`,
        // `_branches`, `_manifests` — see `DrPathBuilder::RESERVED_SYSTEM_SEGMENTS`).
        // The physical path resolver already rejects them via `validate_id`;
        // mirror that at the logical catalog-resolution boundary so DDL/DML cannot
        // address the system catalog by passing a `_`-prefixed tenant.
        if let Some(tenant) = tenant.filter(|tenant| !tenant.is_empty())
            && tenant.starts_with('_')
        {
            anyhow::bail!(
                "tenant '{tenant}' is invalid: tenant identifiers must not begin with '_' \
                 (reserved for system/control-plane catalog subtrees)"
            );
        }
        let (catalog, id) = self.resolve_table(fqn).await?;
        let base_id = id.clone();
        let id = match tenant {
            Some(tenant) if !tenant.is_empty() => {
                if id
                    .namespace
                    .first()
                    .is_some_and(|segment| segment == tenant)
                {
                    id
                } else {
                    let mut namespace = Vec::with_capacity(id.namespace.len() + 1);
                    namespace.push(tenant.to_string());
                    namespace.extend(id.namespace.iter().cloned());
                    TableIdentifier::new(namespace, id.name)
                }
            }
            _ => id,
        };
        // TD-OLAP-18 fold-on-miss (see doc comment). `table_exists` is a
        // CatalogCache hit on every path that would succeed exactly, so the
        // common case pays one cached lookup, not a listing.
        if ident_case_fold_env_enabled() && !catalog.table_exists(&id).await.unwrap_or(false) {
            let candidates: Vec<TableIdentifier> = catalog
                .list_tables(&id.namespace)
                .await
                .unwrap_or_default()
                .into_iter()
                // Only a table in exactly this namespace: a listing must never
                // hand back another tenant's table.
                .filter(|t| t.namespace == id.namespace && t.name.eq_ignore_ascii_case(&id.name))
                .collect();
            if let [declared] = candidates.as_slice() {
                let declared = declared.clone();
                Self::ensure_visible_to(catalog.as_ref(), &declared, fqn, tenant).await?;
                return Ok((catalog, declared));
            }
        }
        // A collection's catalog entry is recorded under the unprefixed
        // identifier; a tenant whose own namespace has no entry of that name
        // resolves to it when — and only when — it records a collection visible
        // to the tenant (its owner).
        if id != base_id
            && !catalog.table_exists(&id).await.unwrap_or(false)
            && Self::records_collection_visible_to(catalog.as_ref(), &base_id, tenant).await
        {
            return Ok((catalog, base_id));
        }
        Self::ensure_visible_to(catalog.as_ref(), &id, fqn, tenant).await?;
        Ok((catalog, id))
    }

    /// Whether `id` exists and records a collection visible to `tenant`.
    async fn records_collection_visible_to(
        catalog: &dyn Catalog,
        id: &TableIdentifier,
        tenant: Option<&str>,
    ) -> bool {
        let Some(tenant) = tenant.filter(|tenant| !tenant.is_empty()) else {
            return false;
        };
        if !catalog.table_exists(id).await.unwrap_or(false) {
            return false;
        }
        let Ok(schema) = catalog.get_table(id).await else {
            return false;
        };
        schema.properties.contains_key("collection.id")
            && proximadb_tenant::collection_entry_visible_to(
                schema.properties.get("owner").map(String::as_str),
                schema.properties.get("tags").map(String::as_str),
                Some(tenant),
            )
    }

    /// A request acting as `tenant` resolves a catalog entry that records a
    /// collection only when that collection is visible to it (the
    /// `proximadb_tenant` ownership rule — the same rule every collection
    /// surface applies). Other entries surface as [`TableNotVisible`], which
    /// reads exactly like a missing table. Internal resolution (`tenant` =
    /// `None`) is not filtered.
    async fn ensure_visible_to(
        catalog: &dyn Catalog,
        id: &TableIdentifier,
        fqn: &str,
        tenant: Option<&str>,
    ) -> Result<()> {
        let Some(tenant) = tenant.filter(|tenant| !tenant.is_empty()) else {
            return Ok(());
        };
        if !catalog.table_exists(id).await.unwrap_or(false) {
            return Ok(());
        }
        let schema = catalog.get_table(id).await?;
        let records_collection = schema.properties.contains_key("collection.id")
            && (schema
                .properties
                .get("asset.kind")
                .is_some_and(|kind| kind == "collection")
                || schema
                    .properties
                    .get("asset.capability.vector")
                    .is_some_and(|flag| flag == "true"));
        if records_collection
            && !proximadb_tenant::collection_entry_visible_to(
                schema.properties.get("owner").map(String::as_str),
                schema.properties.get("tags").map(String::as_str),
                Some(tenant),
            )
        {
            return Err(TableNotVisible {
                table: fqn.to_string(),
            }
            .into());
        }
        Ok(())
    }

    /// Get cache reference for direct access
    pub fn cache(&self) -> Arc<CatalogCache> {
        self.cache.clone()
    }
}

impl Default for CatalogManager {
    fn default() -> Self {
        Self::new()
    }
}

/// TD-OLAP-18 identifier case-fold gate (write path). Mirrors the canonical
/// `proximadb_relational_frontend::ident_case_fold_enabled()` — read inline
/// here because this control-layer crate must not depend on the query layer.
/// Default ON; `PROXIMADB_IDENT_CASE_FOLD=0|false|off|no` restores
/// case-exact resolution (see ENV_GATE_REGISTRY).
fn ident_case_fold_env_enabled() -> bool {
    match std::env::var("PROXIMADB_IDENT_CASE_FOLD") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

// `TableIdentifier` lives in `proximadb_catalog` and is re-exported via
// `pub use self::types::*` above. The previous duplicate definition
// here shadowed the canonical type and caused the in-flight catalog
// trait migration to keep "two TableIdentifier" type-identity errors.
// Removed as part of Option B consolidation.
