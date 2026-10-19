//! OLTP Catalog Backend
//!
//! Stores xCatalog metadata (namespaces, table schemas, snapshots, schema history, statistics)
//! in a transactional relational database. Supports:
//!
//! - **PostgreSQL / Neon / Supabase / CockroachDB** — via `oltp-catalog-postgres` feature
//! - **MariaDB / MySQL / TiDB / PlanetScale** — via `oltp-catalog-mysql` feature
//! - **SQLite** — via `oltp-catalog-sqlite` feature (embedded / dev)
//!
//! # TD-CAT-10: Retirement (2026-08-23)
//!
//! This module is gated behind the `oltp-catalog` feature and is **not maintained**.
//! It is unusable in both configurations:
//!
//! * **Feature OFF (default):** silent no-op, persists nothing, reports success
//! * **Feature ON:** does not compile (sqlx injection lint at `:291`, `:317`)
//!
//! See the TD for evidence and migration path for the three affected tests.
//!
#![cfg(feature = "oltp-catalog")]
//!
//! ## Stacked Durability Mandate
//!
//! The OLTP catalog stores ONLY catalog metadata (schema, stats, snapshots).
//! ProximaDB record data always stays in the internal storage engines (VIPER/NOVA/HELIX).
//! The OLTP DB is not a durable record store — it is a fast, ACID metadata authority.
//!
//! ## Size-Based Routing
//!
//! `CatalogManager::catalog_for_size(bytes)` selects between OLTP catalog (< 1 GB by default)
//! and lakehouse catalog (Delta/Iceberg/native) for larger collections.
//!
//! ## DDL (auto-migrated on startup)
//!
//! ```sql
//! xcatalog_namespaces  — namespace hierarchy with properties
//! xcatalog_tables      — table schema, format, location, size
//! xcatalog_snapshots   — Iceberg-compatible snapshot log
//! xcatalog_schema_history — full schema version history
//! xcatalog_statistics  — row count, size, column stats
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json;
use tracing::{debug, info, warn};

use crate::cache::CatalogCache;
use crate::schema::apply_evolution;
use crate::{
    Catalog, CatalogIndex, CatalogNamespace, CatalogSchemaEvolution, CatalogTableSchema,
    CatalogTableStatistics, TableIdentifier,
};

/// OLTP catalog backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OltpBackend {
    /// PostgreSQL-compatible: PostgreSQL, Neon, Supabase, CockroachDB
    Postgres,
    /// MySQL-compatible: MariaDB, MySQL, TiDB, PlanetScale
    Mysql,
    /// SQLite (embedded / development)
    Sqlite,
}

/// Configuration for the OLTP catalog
#[derive(Debug, Clone)]
pub struct OltpCatalogConfig {
    /// Connection string, e.g.:
    /// - `postgres://user:password@host/dbname`
    /// - `mysql://user:password@host/dbname`
    /// - `sqlite:///path/to/catalog.db` or `sqlite::memory:`
    pub connection_string: String,
    /// Maximum connections in the pool (default: 10)
    pub pool_max_connections: u32,
    /// Prefix for catalog tables (default: "xcatalog_")
    pub table_prefix: String,
    /// Run CREATE TABLE IF NOT EXISTS on startup (default: true)
    pub auto_migrate: bool,
    /// Tables larger than this (bytes) should use a lakehouse catalog instead
    pub size_threshold_bytes: u64,
}

impl Default for OltpCatalogConfig {
    fn default() -> Self {
        Self {
            connection_string: "sqlite::memory:".to_string(),
            pool_max_connections: 10,
            table_prefix: "xcatalog_".to_string(),
            auto_migrate: true,
            size_threshold_bytes: 1_073_741_824, // 1 GB
        }
    }
}

impl OltpCatalogConfig {
    pub fn postgres(connection_string: impl Into<String>) -> Self {
        Self {
            connection_string: connection_string.into(),
            ..Default::default()
        }
    }

    pub fn mysql(connection_string: impl Into<String>) -> Self {
        Self {
            connection_string: connection_string.into(),
            ..Default::default()
        }
    }

    pub fn sqlite(path: impl Into<String>) -> Self {
        Self {
            connection_string: path.into(),
            ..Default::default()
        }
    }
}

fn detect_backend(connection_string: &str) -> OltpBackend {
    if connection_string.starts_with("postgres://")
        || connection_string.starts_with("postgresql://")
    {
        OltpBackend::Postgres
    } else if connection_string.starts_with("mysql://")
        || connection_string.starts_with("mariadb://")
    {
        OltpBackend::Mysql
    } else {
        OltpBackend::Sqlite
    }
}

// ============================================================================
// Feature-gated pool types
// ============================================================================

// When no OLTP feature is enabled, provide a stub that returns errors.
// Each feature gate provides a concrete pool implementation.

#[cfg(not(any(
    feature = "oltp-catalog-postgres",
    feature = "oltp-catalog-mysql",
    feature = "oltp-catalog-sqlite"
)))]
#[allow(dead_code)] // Stub fallback when no oltp-catalog-* feature is enabled
mod pool_impl {
    use anyhow::{Result, anyhow};

    pub struct Pool;

    impl Pool {
        pub async fn connect(_url: &str, _max: u32) -> Result<Self> {
            Err(anyhow!(
                "OLTP catalog requires one of: oltp-catalog-postgres, oltp-catalog-mysql, \
                 oltp-catalog-sqlite feature flags. Build with: \
                 cargo build --features oltp-catalog-postgres"
            ))
        }

        pub async fn execute(&self, _sql: &str) -> Result<()> {
            Err(anyhow!("OLTP catalog not available"))
        }
    }
}

#[cfg(feature = "oltp-catalog")]
mod pool_impl {
    use anyhow::{Result, anyhow};
    use sqlx::AnyPool;

    pub async fn connect(url: &str, max_connections: u32) -> Result<AnyPool> {
        sqlx::any::install_default_drivers();
        // SQLite in-memory databases are per-connection. Cap to 1 so DDL and DML
        // always use the same connection and share the same in-memory schema.
        let effective_max = if url.contains(":memory:") {
            1
        } else {
            max_connections
        };
        let pool = sqlx::pool::PoolOptions::<sqlx::Any>::new()
            .max_connections(effective_max)
            .connect(url)
            .await
            .map_err(|e| anyhow!("OLTP catalog connection failed: {}", e))?;
        Ok(pool)
    }
}

// ============================================================================
// OltpCatalog
// ============================================================================

/// OLTP-backed xCatalog — stores catalog metadata in a relational database.
///
/// Record data is never stored here; it always lives in ProximaDB's storage engines.
/// This catalog is the fast metadata authority for schemas, namespaces, snapshots, and statistics.
pub struct OltpCatalog {
    name: String,
    backend: OltpBackend,
    config: OltpCatalogConfig,
    #[allow(dead_code)] // Wired in by upcoming catalog cache integration
    cache: Arc<CatalogCache>,
    /// In-memory write-through cache (populated from DB on startup)
    namespaces: tokio::sync::RwLock<HashMap<String, CatalogNamespace>>,
    tables: tokio::sync::RwLock<HashMap<String, CatalogTableSchema>>,
    /// Persistent connection pool — present when an oltp-catalog-* feature is enabled
    #[cfg(feature = "oltp-catalog")]
    pool: Option<sqlx::AnyPool>,
}

impl OltpCatalog {
    /// Create a new OLTP catalog. Detects backend from connection string.
    pub async fn new(
        name: impl Into<String>,
        config: OltpCatalogConfig,
        cache: Arc<CatalogCache>,
    ) -> Result<Self> {
        let name = name.into();
        let backend = detect_backend(&config.connection_string);

        info!(
            "Initializing OLTP catalog '{}' backend={:?} dsn={}",
            name,
            backend,
            &config.connection_string[..config.connection_string.find('@').unwrap_or(40).min(40)]
        );

        // Establish the persistent pool (feature-gated).
        #[cfg(feature = "oltp-catalog")]
        let pool_opt = {
            match pool_impl::connect(&config.connection_string, config.pool_max_connections).await {
                Ok(p) => Some(p),
                Err(e) => {
                    warn!(
                        "OLTP catalog pool connection failed, falling back to in-memory: {}",
                        e
                    );
                    None
                }
            }
        };

        let catalog = Self {
            name,
            backend,
            config,
            cache,
            namespaces: tokio::sync::RwLock::new(HashMap::new()),
            tables: tokio::sync::RwLock::new(HashMap::new()),
            #[cfg(feature = "oltp-catalog")]
            pool: pool_opt,
        };

        // Run DDL migrations using the persistent pool, then warm the in-memory cache.
        if catalog.config.auto_migrate {
            if let Err(e) = catalog.run_migrations().await {
                warn!(
                    "OLTP catalog migration warning (continuing with in-memory): {}",
                    e
                );
            } else {
                #[cfg(feature = "oltp-catalog")]
                if let Err(e) = catalog.load_from_db().await {
                    warn!("OLTP catalog load_from_db warning: {}", e);
                }
            }
        }

        Ok(catalog)
    }

    fn prefix(&self) -> &str {
        &self.config.table_prefix
    }

    async fn run_migrations(&self) -> Result<()> {
        debug!(
            "OLTP catalog: running migrations with prefix '{}' on backend {:?}",
            self.prefix(),
            self.backend
        );

        #[cfg(feature = "oltp-catalog")]
        {
            let pool = self
                .pool
                .as_ref()
                .ok_or_else(|| anyhow!("OLTP pool not connected"))?;
            let ddl = self.generate_ddl();
            for stmt in ddl {
                sqlx::query(&stmt).execute(pool).await.map_err(|e| {
                    anyhow!("DDL error: {} — SQL: {}", e, &stmt[..80.min(stmt.len())])
                })?;
            }
            info!("OLTP catalog '{}': migrations completed", self.name);
        }

        #[cfg(not(feature = "oltp-catalog"))]
        debug!("OLTP catalog: no feature enabled, skipping DDL");

        Ok(())
    }

    /// Load existing namespaces and tables from the DB into the in-memory cache.
    /// Called once after migrations to warm up so subsequent reads hit the cache.
    #[cfg(feature = "oltp-catalog")]
    async fn load_from_db(&self) -> Result<()> {
        use sqlx::Row as _;

        let pool = match &self.pool {
            Some(p) => p,
            None => return Ok(()),
        };
        let p = self.prefix();

        // Load namespaces
        let rows = sqlx::query(&format!(
            "SELECT namespace_path, properties, owner, location FROM {p}namespaces"
        ))
        .fetch_all(pool)
        .await
        .map_err(|e| anyhow!("load_from_db namespaces: {}", e))?;

        let mut ns_guard = self.namespaces.write().await;
        for row in rows {
            let path_json: String = row.try_get("namespace_path")?;
            let path: Vec<String> = serde_json::from_str(&path_json)
                .unwrap_or_else(|_| path_json.split('.').map(String::from).collect());
            let props_json: String = row
                .try_get("properties")
                .unwrap_or_else(|_| "{}".to_string());
            let properties: HashMap<String, String> =
                serde_json::from_str(&props_json).unwrap_or_default();
            let owner: Option<String> = row.try_get("owner").ok().and_then(|v: Option<String>| v);
            let location: Option<String> =
                row.try_get("location").ok().and_then(|v: Option<String>| v);
            let key = path.join(".");
            ns_guard.insert(
                key,
                CatalogNamespace {
                    levels: path,
                    properties,
                    owner,
                    location,
                    created_at_ms: 0,
                    updated_at_ms: 0,
                    ..CatalogNamespace::new(Vec::new())
                },
            );
        }
        let ns_count = ns_guard.len();
        drop(ns_guard);

        // Load tables
        let rows = sqlx::query(&format!(
            "SELECT namespace_path, table_name, schema_json FROM {p}tables"
        ))
        .fetch_all(pool)
        .await
        .map_err(|e| anyhow!("load_from_db tables: {}", e))?;

        let mut table_guard = self.tables.write().await;
        for row in rows {
            let ns_json: String = row.try_get("namespace_path")?;
            let ns: Vec<String> = serde_json::from_str(&ns_json)
                .unwrap_or_else(|_| ns_json.split('.').map(String::from).collect());
            let table_name: String = row.try_get("table_name")?;
            let schema_json: String = row.try_get("schema_json")?;
            let schema: CatalogTableSchema = serde_json::from_str(&schema_json)
                .unwrap_or_else(|_| CatalogTableSchema::new(&table_name));
            let key = format!("{}.{}", ns.join("."), table_name);
            table_guard.insert(key, schema);
        }
        let table_count = table_guard.len();
        drop(table_guard);

        info!(
            "OLTP catalog '{}': loaded {} namespaces, {} tables from DB",
            self.name, ns_count, table_count
        );
        Ok(())
    }

    #[cfg(feature = "oltp-catalog")]
    fn generate_ddl(&self) -> Vec<String> {
        let p = self.prefix();

        match self.backend {
            OltpBackend::Postgres => vec![
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}namespaces (\
                        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\
                        namespace_path TEXT NOT NULL UNIQUE,\
                        properties JSONB DEFAULT '{{}}',\
                        owner TEXT,\
                        location TEXT,\
                        created_at TIMESTAMPTZ DEFAULT NOW(),\
                        updated_at TIMESTAMPTZ DEFAULT NOW()\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}tables (\
                        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\
                        namespace_path TEXT[] NOT NULL,\
                        table_name TEXT NOT NULL,\
                        schema_json JSONB NOT NULL,\
                        properties JSONB DEFAULT '{{}}',\
                        format TEXT DEFAULT 'INTERNAL',\
                        location TEXT,\
                        size_bytes BIGINT DEFAULT 0,\
                        row_count BIGINT DEFAULT 0,\
                        created_at TIMESTAMPTZ DEFAULT NOW(),\
                        updated_at TIMESTAMPTZ DEFAULT NOW(),\
                        UNIQUE(namespace_path, table_name)\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}snapshots (\
                        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\
                        table_id BIGINT REFERENCES {p}tables(id) ON DELETE CASCADE,\
                        snapshot_id BIGINT NOT NULL UNIQUE,\
                        parent_snapshot_id BIGINT,\
                        timestamp_ms BIGINT NOT NULL,\
                        manifest_list TEXT,\
                        summary JSONB DEFAULT '{{}}',\
                        schema_id INT\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}schema_history (\
                        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\
                        table_id BIGINT REFERENCES {p}tables(id) ON DELETE CASCADE,\
                        schema_version INT NOT NULL,\
                        schema_json JSONB NOT NULL,\
                        applied_at TIMESTAMPTZ DEFAULT NOW()\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}statistics (\
                        table_id BIGINT REFERENCES {p}tables(id) ON DELETE CASCADE PRIMARY KEY,\
                        row_count BIGINT,\
                        size_bytes BIGINT,\
                        column_stats JSONB DEFAULT '{{}}',\
                        updated_at TIMESTAMPTZ DEFAULT NOW()\
                    )"
                ),
            ],

            OltpBackend::Mysql => vec![
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}namespaces (\
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,\
                        namespace_path JSON NOT NULL,\
                        properties JSON,\
                        owner VARCHAR(255),\
                        location TEXT,\
                        created_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3),\
                        updated_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3)\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}tables (\
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,\
                        namespace_path JSON NOT NULL,\
                        table_name VARCHAR(255) NOT NULL,\
                        schema_json LONGTEXT NOT NULL,\
                        properties JSON,\
                        format VARCHAR(64) DEFAULT 'INTERNAL',\
                        location TEXT,\
                        size_bytes BIGINT DEFAULT 0,\
                        row_count BIGINT DEFAULT 0,\
                        created_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3),\
                        updated_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3)\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}snapshots (\
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,\
                        table_id BIGINT,\
                        snapshot_id BIGINT NOT NULL UNIQUE,\
                        parent_snapshot_id BIGINT,\
                        timestamp_ms BIGINT NOT NULL,\
                        manifest_list TEXT,\
                        summary JSON,\
                        schema_id INT,\
                        FOREIGN KEY (table_id) REFERENCES {p}tables(id) ON DELETE CASCADE\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}schema_history (\
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,\
                        table_id BIGINT,\
                        schema_version INT NOT NULL,\
                        schema_json LONGTEXT NOT NULL,\
                        applied_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3),\
                        FOREIGN KEY (table_id) REFERENCES {p}tables(id) ON DELETE CASCADE\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}statistics (\
                        table_id BIGINT PRIMARY KEY,\
                        row_count BIGINT,\
                        size_bytes BIGINT,\
                        column_stats JSON,\
                        updated_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),\
                        FOREIGN KEY (table_id) REFERENCES {p}tables(id) ON DELETE CASCADE\
                    )"
                ),
            ],

            OltpBackend::Sqlite => vec![
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}namespaces (\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        namespace_path TEXT NOT NULL UNIQUE,\
                        properties TEXT DEFAULT '{{}}',\
                        owner TEXT,\
                        location TEXT,\
                        created_at INTEGER DEFAULT (unixepoch('now') * 1000),\
                        updated_at INTEGER DEFAULT (unixepoch('now') * 1000)\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}tables (\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        namespace_path TEXT NOT NULL,\
                        table_name TEXT NOT NULL,\
                        schema_json TEXT NOT NULL,\
                        properties TEXT DEFAULT '{{}}',\
                        format TEXT DEFAULT 'INTERNAL',\
                        location TEXT,\
                        size_bytes INTEGER DEFAULT 0,\
                        row_count INTEGER DEFAULT 0,\
                        created_at INTEGER DEFAULT (unixepoch('now') * 1000),\
                        updated_at INTEGER DEFAULT (unixepoch('now') * 1000),\
                        UNIQUE(namespace_path, table_name)\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}snapshots (\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        table_id INTEGER REFERENCES {p}tables(id) ON DELETE CASCADE,\
                        snapshot_id INTEGER NOT NULL UNIQUE,\
                        parent_snapshot_id INTEGER,\
                        timestamp_ms INTEGER NOT NULL,\
                        manifest_list TEXT,\
                        summary TEXT DEFAULT '{{}}',\
                        schema_id INTEGER\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}schema_history (\
                        id INTEGER PRIMARY KEY AUTOINCREMENT,\
                        table_id INTEGER REFERENCES {p}tables(id) ON DELETE CASCADE,\
                        schema_version INTEGER NOT NULL,\
                        schema_json TEXT NOT NULL,\
                        applied_at INTEGER DEFAULT (unixepoch('now') * 1000)\
                    )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {p}statistics (\
                        table_id INTEGER REFERENCES {p}tables(id) ON DELETE CASCADE PRIMARY KEY,\
                        row_count INTEGER,\
                        size_bytes INTEGER,\
                        column_stats TEXT DEFAULT '{{}}',\
                        updated_at INTEGER DEFAULT (unixepoch('now') * 1000)\
                    )"
                ),
            ],
        }
    }

    fn ns_key(namespace: &[String]) -> String {
        namespace.join(".")
    }

    fn table_key(identifier: &TableIdentifier) -> String {
        format!("{}.{}", identifier.namespace.join("."), identifier.name)
    }

    #[allow(dead_code)] // Used by upcoming sqlx row mapping for catalog persistence
    fn catalog_schema_to_json(schema: &CatalogTableSchema) -> serde_json::Value {
        serde_json::to_value(schema).unwrap_or_default()
    }

    #[allow(dead_code)] // Used by upcoming sqlx row mapping for catalog persistence
    fn json_to_catalog_schema(json: &serde_json::Value) -> CatalogTableSchema {
        serde_json::from_value(json.clone()).unwrap_or_else(|_| CatalogTableSchema::default())
    }
}

#[async_trait]
impl Catalog for OltpCatalog {
    fn name(&self) -> &str {
        &self.name
    }

    fn catalog_type(&self) -> &str {
        match self.backend {
            OltpBackend::Postgres => "oltp-postgres",
            OltpBackend::Mysql => "oltp-mysql",
            OltpBackend::Sqlite => "oltp-sqlite",
        }
    }

    async fn create_namespace(
        &self,
        namespace: &[String],
        properties: HashMap<String, String>,
    ) -> Result<CatalogNamespace> {
        let key = Self::ns_key(namespace);
        let ns = ns_with_properties(CatalogNamespace::new(namespace.to_vec()), properties);

        // SQL persistence (write-through, before in-memory so conflicts surface from DB).
        #[cfg(feature = "oltp-catalog")]
        if let Some(ref pool) = self.pool {
            let ns_json = serde_json::to_string(namespace).map_err(|e| anyhow!("{}", e))?;
            let props_json = serde_json::to_string(&ns.properties).map_err(|e| anyhow!("{}", e))?;
            sqlx::query(&format!(
                "INSERT INTO {}namespaces (namespace_path, properties) VALUES (?, ?)",
                self.prefix()
            ))
            .bind(&ns_json)
            .bind(&props_json)
            .execute(pool)
            .await
            .map_err(|e| anyhow!("Failed to persist namespace '{}': {}", key, e))?;
        }

        {
            let mut guard = self.namespaces.write().await;
            if guard.contains_key(&key) {
                return Err(anyhow!("Namespace '{}' already exists", key));
            }
            guard.insert(key, ns.clone());
        }

        debug!("OltpCatalog: created namespace '{}'", namespace.join("."));
        Ok(ns)
    }

    async fn drop_namespace(&self, namespace: &[String], _cascade: bool) -> Result<bool> {
        let key = Self::ns_key(namespace);

        #[cfg(feature = "oltp-catalog")]
        if let Some(ref pool) = self.pool {
            let ns_json = serde_json::to_string(namespace).map_err(|e| anyhow!("{}", e))?;
            sqlx::query(&format!(
                "DELETE FROM {}namespaces WHERE namespace_path = ?",
                self.prefix()
            ))
            .bind(&ns_json)
            .execute(pool)
            .await
            .map_err(|e| anyhow!("Failed to drop namespace '{}': {}", key, e))?;
        }

        let mut guard = self.namespaces.write().await;
        Ok(guard.remove(&key).is_some())
    }

    async fn list_namespaces(&self, _parent: Option<&[String]>) -> Result<Vec<CatalogNamespace>> {
        let guard = self.namespaces.read().await;
        Ok(guard.values().cloned().collect())
    }

    async fn namespace_exists(&self, namespace: &[String]) -> Result<bool> {
        let key = Self::ns_key(namespace);
        Ok(self.namespaces.read().await.contains_key(&key))
    }

    async fn get_namespace(&self, namespace: &[String]) -> Result<CatalogNamespace> {
        let key = Self::ns_key(namespace);
        self.namespaces
            .read()
            .await
            .get(&key)
            .cloned()
            .ok_or_else(|| anyhow!("Namespace '{}' not found", key))
    }

    async fn update_namespace_properties(
        &self,
        namespace: &[String],
        updates: HashMap<String, String>,
        removals: Vec<String>,
    ) -> Result<()> {
        let key = Self::ns_key(namespace);
        let mut guard = self.namespaces.write().await;
        let ns = guard
            .get_mut(&key)
            .ok_or_else(|| anyhow!("Namespace '{}' not found", key))?;

        for k in &removals {
            ns.properties.remove(k);
        }
        for (k, v) in updates {
            ns.properties.insert(k, v);
        }
        Ok(())
    }

    async fn create_table_inner(
        &self,
        identifier: &TableIdentifier,
        schema: CatalogTableSchema,
    ) -> Result<CatalogTableSchema> {
        // Ensure namespace exists (auto-create if missing)
        if !self.namespace_exists(&identifier.namespace).await? {
            let _ = self
                .create_namespace(&identifier.namespace, HashMap::new())
                .await;
        }

        let key = Self::table_key(identifier);

        #[cfg(feature = "oltp-catalog")]
        if let Some(ref pool) = self.pool {
            let ns_json =
                serde_json::to_string(&identifier.namespace).map_err(|e| anyhow!("{}", e))?;
            let schema_json = serde_json::to_string(&schema).map_err(|e| anyhow!("{}", e))?;
            sqlx::query(&format!(
                "INSERT INTO {}tables (namespace_path, table_name, schema_json) VALUES (?, ?, ?)",
                self.prefix()
            ))
            .bind(&ns_json)
            .bind(&identifier.name)
            .bind(&schema_json)
            .execute(pool)
            .await
            .map_err(|e| anyhow!("Failed to persist table '{}': {}", key, e))?;
        }

        {
            let mut guard = self.tables.write().await;
            if guard.contains_key(&key) {
                return Err(anyhow!("Table '{}' already exists", key));
            }
            guard.insert(key, schema.clone());
        }

        debug!("OltpCatalog: created table '{}'", identifier.name);
        Ok(schema)
    }

    async fn drop_table(&self, identifier: &TableIdentifier, _purge: bool) -> Result<bool> {
        let key = Self::table_key(identifier);

        #[cfg(feature = "oltp-catalog")]
        if let Some(ref pool) = self.pool {
            let ns_json =
                serde_json::to_string(&identifier.namespace).map_err(|e| anyhow!("{}", e))?;
            sqlx::query(&format!(
                "DELETE FROM {}tables WHERE namespace_path = ? AND table_name = ?",
                self.prefix()
            ))
            .bind(&ns_json)
            .bind(&identifier.name)
            .execute(pool)
            .await
            .map_err(|e| anyhow!("Failed to drop table '{}': {}", key, e))?;
        }

        Ok(self.tables.write().await.remove(&key).is_some())
    }

    async fn list_tables(&self, namespace: &[String]) -> Result<Vec<TableIdentifier>> {
        let ns_prefix = namespace.join(".");
        let guard = self.tables.read().await;
        Ok(guard
            .keys()
            .filter_map(|key| {
                if key.starts_with(&ns_prefix) {
                    let table_name = key[ns_prefix.len()..].trim_start_matches('.').to_string();
                    if !table_name.is_empty() && !table_name.contains('.') {
                        Some(TableIdentifier::new(namespace.to_vec(), table_name))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect())
    }

    async fn table_exists(&self, identifier: &TableIdentifier) -> Result<bool> {
        let key = Self::table_key(identifier);
        Ok(self.tables.read().await.contains_key(&key))
    }

    async fn get_table(&self, identifier: &TableIdentifier) -> Result<CatalogTableSchema> {
        let key = Self::table_key(identifier);
        self.tables
            .read()
            .await
            .get(&key)
            .cloned()
            .ok_or_else(|| anyhow!("Table '{}' not found", key))
    }

    async fn rename_table(&self, from: &TableIdentifier, to: &TableIdentifier) -> Result<()> {
        let from_key = Self::table_key(from);
        let to_key = Self::table_key(to);
        let mut guard = self.tables.write().await;
        let schema = guard
            .remove(&from_key)
            .ok_or_else(|| anyhow!("Table '{}' not found", from_key))?;
        guard.insert(to_key, schema);
        Ok(())
    }

    async fn evolve_schema(
        &self,
        identifier: &TableIdentifier,
        evolution: CatalogSchemaEvolution,
    ) -> Result<CatalogTableSchema> {
        let key = Self::table_key(identifier);
        let mut guard = self.tables.write().await;
        let schema = guard
            .get_mut(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", key))?;

        apply_evolution(schema, &evolution)?;
        Ok(schema.clone())
    }

    async fn get_schema_version(&self, identifier: &TableIdentifier) -> Result<i32> {
        Ok(self.get_table(identifier).await?.schema_version)
    }

    async fn get_schema_by_version(
        &self,
        identifier: &TableIdentifier,
        _version: i32,
    ) -> Result<CatalogTableSchema> {
        // Simplified: return current schema (full version history requires DB query)
        self.get_table(identifier).await
    }

    async fn create_index(
        &self,
        identifier: &TableIdentifier,
        index: CatalogIndex,
    ) -> Result<CatalogIndex> {
        let key = Self::table_key(identifier);
        let mut guard = self.tables.write().await;
        let schema = guard
            .get_mut(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", key))?;
        schema.indexes.push(index.clone());
        Ok(index)
    }

    async fn drop_index(&self, identifier: &TableIdentifier, index_name: &str) -> Result<bool> {
        let key = Self::table_key(identifier);
        let mut guard = self.tables.write().await;
        if let Some(schema) = guard.get_mut(&key) {
            let before = schema.indexes.len();
            schema.indexes.retain(|i| i.name != index_name);
            return Ok(schema.indexes.len() < before);
        }
        Ok(false)
    }

    async fn list_indexes(&self, identifier: &TableIdentifier) -> Result<Vec<CatalogIndex>> {
        Ok(self.get_table(identifier).await?.indexes)
    }

    async fn get_statistics(&self, identifier: &TableIdentifier) -> Result<CatalogTableStatistics> {
        let key = Self::table_key(identifier);
        let guard = self.tables.read().await;
        let schema = guard
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", key))?;
        Ok(CatalogTableStatistics {
            row_count: schema
                .properties
                .get("row_count")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            size_bytes: schema
                .properties
                .get("size_bytes")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            ..Default::default()
        })
    }

    async fn update_statistics(
        &self,
        identifier: &TableIdentifier,
        stats: CatalogTableStatistics,
    ) -> Result<()> {
        let key = Self::table_key(identifier);
        let mut guard = self.tables.write().await;
        if let Some(schema) = guard.get_mut(&key) {
            schema
                .properties
                .insert("row_count".to_string(), stats.row_count.to_string());
            schema
                .properties
                .insert("size_bytes".to_string(), stats.size_bytes.to_string());
        }
        Ok(())
    }

    async fn health_check(&self) -> Result<crate::CatalogHealth> {
        #[cfg(feature = "oltp-catalog")]
        if let Some(ref pool) = self.pool {
            // Ping the DB with a trivial query.
            if let Err(e) = sqlx::query("SELECT 1").execute(pool).await {
                return Ok(crate::CatalogHealth::unhealthy(format!(
                    "OLTP DB ping failed: {}",
                    e
                )));
            }
        }
        Ok(crate::CatalogHealth::healthy(
            self.tables.read().await.len() as u64,
        ))
    }

    async fn close(&self) -> Result<()> {
        #[cfg(feature = "oltp-catalog")]
        if let Some(ref pool) = self.pool {
            pool.close().await;
        }
        Ok(())
    }
}

// ============================================================================
// CatalogNamespace builder helper
// ============================================================================

fn ns_with_properties(
    mut ns: CatalogNamespace,
    props: HashMap<String, String>,
) -> CatalogNamespace {
    ns.properties = props;
    ns
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CatalogCache;

    async fn make_catalog() -> OltpCatalog {
        let config = OltpCatalogConfig::sqlite("sqlite::memory:");
        OltpCatalog::new("test", config, Arc::new(CatalogCache::new(1000, 60)))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_create_and_get_namespace() {
        let cat = make_catalog().await;
        let ns = cat
            .create_namespace(&["db".to_string()], HashMap::new())
            .await
            .unwrap();
        assert_eq!(ns.levels, vec!["db"]);
        assert!(cat.namespace_exists(&["db".to_string()]).await.unwrap());
    }

    #[tokio::test]
    async fn test_create_and_list_table() {
        let cat = make_catalog().await;
        cat.create_namespace(&["db".to_string()], HashMap::new())
            .await
            .unwrap();

        let id = TableIdentifier::new(vec!["db".to_string()], "orders".to_string());
        let schema = CatalogTableSchema::new("orders");
        cat.create_table(&id, schema).await.unwrap();

        let tables = cat.list_tables(&["db".to_string()]).await.unwrap();
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].name, "orders");
    }

    #[tokio::test]
    async fn test_drop_table() {
        let cat = make_catalog().await;
        let id = TableIdentifier::new(vec!["db".to_string()], "t".to_string());
        cat.create_table(&id, CatalogTableSchema::new("t"))
            .await
            .unwrap();
        assert!(cat.table_exists(&id).await.unwrap());
        cat.drop_table(&id, false).await.unwrap();
        assert!(!cat.table_exists(&id).await.unwrap());
    }

    #[tokio::test]
    async fn test_detect_backend() {
        assert_eq!(
            detect_backend("postgres://localhost/db"),
            OltpBackend::Postgres
        );
        assert_eq!(detect_backend("mysql://localhost/db"), OltpBackend::Mysql);
        assert_eq!(detect_backend("sqlite::memory:"), OltpBackend::Sqlite);
    }
}
