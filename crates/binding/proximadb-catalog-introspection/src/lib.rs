use std::sync::Arc;

use anyhow::Result;
use serde::Serialize;
use sqlparser::ast::{SelectItem, SetExpr, Statement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use proximadb_catalog::{CatalogIndexType, CatalogManager, CatalogTableSchema, TableIdentifier};

use proximadb_data_model::ProximaType;

// ---- inlined SQL alias helpers -------------------------------------------------
// `parse_select_aliases` + `projection_label` are thin wrappers over sqlparser-rs
// (the same crate the root `query::sql_frontend` wraps). Inlined here so this crate
// can answer psql-style catalog queries (`SELECT ... AS "Name" ... FROM
// pg_catalog.pg_class ...`) without taking a production dependency on the root
// `proximadb` crate (which would create a cycle, since the root re-exports this
// crate via a shim). Kept verbatim with `src/query/sql_frontend/parser.rs`; if that
// implementation changes, update this copy in lockstep.

/// Best-effort output-column label for a single SELECT projection item: the
/// explicit alias when present, else the trailing identifier of the expression
/// (`a.attname` → `attname`), else the raw expression text.
fn projection_label(item: &SelectItem) -> String {
    match item {
        SelectItem::ExprWithAlias { alias, .. } => alias.value.clone(),
        SelectItem::UnnamedExpr(expr) => {
            let text = expr.to_string();
            text.rsplit_once('.')
                .map(|(_, last)| last.trim().to_string())
                .unwrap_or(text)
        }
        SelectItem::Wildcard(_) => "*".to_string(),
        _ => item.to_string(),
    }
}

/// Parse a single SELECT's projection list and return the output-column labels
/// the client expects (explicit alias, else a label derived from the expression).
/// Returns `None` for anything that isn't a single SELECT, for any projection
/// containing a wildcard (whose width we cannot determine from our side), or on
/// parse failure — callers fall back to substring-based dispatch and the
/// width-safety net.
fn parse_select_aliases(sql: &str) -> Option<Vec<String>> {
    let statements = Parser::parse_sql(&GenericDialect {}, sql).ok()?;
    let [statement] = statements.as_slice() else {
        return None;
    };
    let Statement::Query(query) = statement else {
        return None;
    };
    let SetExpr::Select(select) = &*query.body else {
        return None;
    };
    if select
        .projection
        .iter()
        .any(|item| matches!(item, SelectItem::Wildcard(_)))
    {
        return None;
    }
    Some(select.projection.iter().map(projection_label).collect())
}

/// Tabular catalog metadata for SQL-compatible introspection surfaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CatalogIntrospectionResult {
    pub columns: Vec<String>,
    pub column_types: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl CatalogIntrospectionResult {
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
            rows: Vec::new(),
        }
    }
}

/// Projects the native xCatalog into pgwire/embedded SQL metadata tables.
///
/// This is intentionally read-only and table-shaped so JDBC/SQLAlchemy/dbt-style clients can
/// introspect schemas without depending on Rust structs or SDK-only APIs.
#[derive(Clone)]
pub struct CatalogIntrospectionService {
    catalog_manager: Arc<CatalogManager>,
    /// The tenant the request acts as; listings show only what it may see.
    /// `None` = internal (embedded) use, unfiltered.
    tenant: Option<String>,
}

impl CatalogIntrospectionService {
    pub fn new(catalog_manager: Arc<CatalogManager>) -> Self {
        Self {
            catalog_manager,
            tenant: None,
        }
    }

    /// Introspection for a request acting as `tenant`: only tables in the
    /// tenant's namespace, and collections visible to it, are listed.
    pub fn for_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    /// Whether the request tenant may see a cataloged table.
    fn visible_to_tenant(&self, table_id: &TableIdentifier, schema: &CatalogTableSchema) -> bool {
        let Some(tenant) = self.tenant.as_deref() else {
            return true;
        };
        let tenant = proximadb_tenant::resolve_request_tenant(Some(tenant));
        if schema.properties.contains_key("collection.id") {
            return proximadb_tenant::collection_entry_visible_to(
                schema.properties.get("owner").map(String::as_str),
                schema.properties.get("tags").map(String::as_str),
                Some(&tenant),
            );
        }
        table_id
            .namespace
            .first()
            .is_some_and(|first| *first == tenant)
    }

    pub fn is_catalog_query(sql: &str) -> bool {
        let normalized = normalize_sql(sql);
        normalized.contains(" from xcatalog.tables")
            || normalized.contains(" from xcatalog.columns")
            || normalized.contains(" from xcatalog.indexes")
            || normalized.contains(" from xcatalog.table_routing")
            || normalized.contains(" from xcatalog.namespaces")
            || normalized.contains(" from information_schema.tables")
            || normalized.contains(" from information_schema.columns")
            || normalized.contains(" from information_schema.table_routing")
            || normalized.contains(" from information_schema.schemata")
            || normalized.contains("proximadb_catalog.props_promotion_candidates")
            || (normalized.contains("pg_catalog.pg_class")
                && normalized.contains("pg_catalog.pg_namespace"))
            || (normalized.contains("pg_catalog.pg_attribute")
                && normalized.contains("pg_catalog.pg_class"))
            || normalized.contains("pg_catalog.pg_index")
            // Bare `FROM pg_class` (no qualifier) — common shape from
            // psql `\dt` and lightweight ORM connection probes.
            || normalized.contains(" from pg_class")
            // Canonical pg_catalog views (TD-120). These are caught only
            // when the more specific ORM-shaped joins above do not match,
            // so existing JDBC/SQLAlchemy routes are unaffected.
            || normalized.contains(" from pg_catalog.pg_type")
            || normalized.contains(" from pg_catalog.pg_namespace")
            || normalized.contains(" from pg_catalog.pg_class")
            || normalized.contains(" from pg_catalog.pg_attribute")
            || normalized.contains(" from pg_catalog.pg_constraint")
    }

    pub async fn execute_select(&self, sql: &str) -> Result<Option<CatalogIntrospectionResult>> {
        let normalized = normalize_sql(sql);
        let table_filter = extract_string_filter(sql, "table_name");

        // `information_schema.tables` returns the SQL-standard
        // 4-column shape `(table_catalog, table_schema, table_name,
        // table_type)`. Admin tools and ORMs probe for this exact
        // shape; returning ProximaDB-extended columns (as
        // `xcatalog.tables` does) breaks them. Keep the two views
        // distinct so each carries its own contract.
        if normalized.contains(" from information_schema.tables") {
            return Ok(Some(
                self.information_schema_tables(table_filter.as_deref())
                    .await?,
            ));
        }
        if normalized.contains(" from xcatalog.tables") {
            return Ok(Some(self.tables(table_filter.as_deref()).await?));
        }
        if normalized.contains(" from xcatalog.namespaces")
            || normalized.contains(" from information_schema.schemata")
        {
            return Ok(Some(self.namespaces().await?));
        }
        // Bare `FROM pg_class` (without joins) — return the
        // SQLAlchemy/psql-compatible shape used elsewhere for
        // pg_class + pg_namespace queries. The shape covers `relname`,
        // `relkind`, and the columns most lightweight probes ask
        // for; extending it is Phase 2's `pg_catalog` work.
        if normalized.contains(" from pg_class")
            && !normalized.contains("pg_attribute")
            && !normalized.contains("pg_namespace")
        {
            // A rich psql-style probe (`SELECT ... AS "Name" ... FROM pg_class`)
            // selects many columns; the narrow `sqlalchemy_table_names` builder
            // returns one and would trip libpq's "column number out of range".
            // Route multi-column queries to the shape-aware builder; keep the
            // 1-column builder for genuine ORM probes that select `relname` only.
            if let Some(aliases) = parse_select_aliases(sql).filter(|a| a.len() > 1) {
                let relname = extract_string_filter(sql, "relname")
                    .or_else(|| extract_string_filter(sql, "c.relname"))
                    .or(table_filter.clone());
                return Ok(Some(
                    self.psql_shaped_table_list(&aliases, relname.as_deref())
                        .await?,
                ));
            }
            return Ok(Some(self.sqlalchemy_table_names(None).await?));
        }
        if normalized.contains(" from xcatalog.columns")
            || normalized.contains(" from information_schema.columns")
        {
            return Ok(Some(self.columns(table_filter.as_deref()).await?));
        }
        if normalized.contains(" from xcatalog.indexes") {
            return Ok(Some(self.indexes(table_filter.as_deref()).await?));
        }
        if normalized.contains(" from xcatalog.table_routing")
            || normalized.contains(" from information_schema.table_routing")
        {
            return Ok(Some(self.table_routing(table_filter.as_deref()).await?));
        }
        if normalized.contains("proximadb_catalog.props_promotion_candidates") {
            let table_filter = extract_string_filter(sql, "table_name")
                .or_else(|| extract_function_arg(sql, "props_promotion_candidates"));
            return Ok(Some(
                self.props_promotion_candidates(table_filter.as_deref())
                    .await?,
            ));
        }
        if normalized.contains("pg_catalog.pg_index") {
            return Ok(Some(self.jdbc_indexes(None).await?));
        }
        if normalized.contains("pg_catalog.pg_attribute")
            && normalized.contains("pg_catalog.pg_class")
            && normalized.contains("attname")
        {
            let relname_filter = extract_string_filter(sql, "c.relname")
                .or_else(|| extract_string_filter(sql, "ct.relname"))
                .or_else(|| extract_string_filter(sql, "pg_class.relname"))
                .or(table_filter);
            // psql's `\d` column probe aliases columns (`AS "Column"`, `AS "Type"`,
            // ...); the fixed-shape `sqlalchemy_columns` builder returns different
            // names and psql fails to resolve them. Route multi-column queries to
            // the shape-aware column builder; keep the ORM builder otherwise.
            if let Some(aliases) = parse_select_aliases(sql).filter(|a| a.len() > 1) {
                return Ok(Some(
                    self.psql_shaped_columns(&aliases, relname_filter.as_deref())
                        .await?,
                ));
            }
            return Ok(Some(
                self.sqlalchemy_columns(relname_filter.as_deref()).await?,
            ));
        }
        if normalized.contains("pg_catalog.pg_attribute") && normalized.contains("column_name") {
            let relname_filter = extract_string_filter(sql, "c.relname")
                .or_else(|| extract_string_filter(sql, "ct.relname"))
                .or(table_filter);
            return Ok(Some(self.jdbc_columns(relname_filter.as_deref()).await?));
        }
        if normalized.contains("pg_catalog.pg_class")
            && normalized.contains("pg_catalog.pg_namespace")
            && normalized.contains("relname")
        {
            let relname_filter = extract_string_filter(sql, "c.relname")
                .or_else(|| extract_string_filter(sql, "pg_class.relname"))
                .or(table_filter);
            // psql `\dt` reaches this branch; route rich multi-column probes to
            // the shape-aware builder so the returned columns match the SELECT
            // list. The 1-column `sqlalchemy_table_names` path stays for ORM
            // probes that select only `relname`.
            if let Some(aliases) = parse_select_aliases(sql).filter(|a| a.len() > 1) {
                return Ok(Some(
                    self.psql_shaped_table_list(&aliases, relname_filter.as_deref())
                        .await?,
                ));
            }
            return Ok(Some(
                self.sqlalchemy_table_names(relname_filter.as_deref())
                    .await?,
            ));
        }
        if normalized.contains("pg_catalog.pg_class")
            && normalized.contains("pg_catalog.pg_namespace")
            && normalized.contains("table_name")
        {
            let relname_filter = extract_string_filter(sql, "c.relname").or(table_filter);
            return Ok(Some(self.jdbc_tables(relname_filter.as_deref()).await?));
        }

        // --- Canonical pg_catalog views (TD-120) ---
        // Placed last so the ORM-specific routes above keep precedence;
        // these only fire for forms that would otherwise return `None`.
        if normalized.contains(" from pg_catalog.pg_type") {
            return Ok(Some(Self::pg_type()));
        }
        if normalized.contains(" from pg_catalog.pg_namespace") {
            return Ok(Some(self.pg_namespace().await?));
        }
        if normalized.contains(" from pg_catalog.pg_class") {
            let relname = extract_string_filter(sql, "relname").or(table_filter);
            return Ok(Some(self.pg_class(relname.as_deref()).await?));
        }
        if normalized.contains(" from pg_catalog.pg_attribute") {
            let relname = extract_string_filter(sql, "relname").or(table_filter);
            return Ok(Some(self.pg_attribute(relname.as_deref()).await?));
        }
        if normalized.contains(" from pg_catalog.pg_constraint") {
            let relname = extract_string_filter(sql, "conrelid").or(table_filter);
            return Ok(Some(self.pg_constraint(relname.as_deref()).await?));
        }

        Ok(None)
    }

    /// `information_schema.tables` with the SQL-standard 4-column
    /// shape `(table_catalog, table_schema, table_name, table_type)`.
    /// Distinct from `xcatalog.tables` which carries ProximaDB-
    /// specific extras (schema_kind, storage_engine, layout, etc.).
    async fn information_schema_tables(
        &self,
        table_filter: Option<&str>,
    ) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();
        for (catalog_name, table_id, _schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            rows.push(vec![
                catalog_name,
                namespace_name(&table_id),
                table_id.name,
                // Per ANSI: BASE TABLE | VIEW | LOCAL TEMPORARY |
                // GLOBAL TEMPORARY. We always return BASE TABLE
                // until views land.
                "BASE TABLE".to_string(),
            ]);
        }
        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec![
                "table_catalog".to_string(),
                "table_schema".to_string(),
                "table_name".to_string(),
                "table_type".to_string(),
            ],
            column_types: vec![
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
            ],
            rows,
        })
    }

    /// `xcatalog.namespaces` (and `information_schema.schemata`) —
    /// the set of registered namespaces. Carries the engine
    /// authority fields added in P0.5 (namespace_id, tenant_id,
    /// region_home, storage_pool_class) so operator tools can
    /// inspect them via SQL.
    ///
    /// Iteration pattern mirrors `cataloged_tables`: walk every
    /// catalog registered on `CatalogManager`, then every namespace
    /// each catalog returns from `list_namespaces`.
    async fn namespaces(&self) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();
        for catalog_name in self.catalog_manager.list_catalogs().await {
            let catalog = match self.catalog_manager.get_catalog(&catalog_name).await {
                Ok(c) => c,
                Err(_) => continue,
            };
            let listed = match catalog.list_namespaces(None).await {
                Ok(ns) => ns,
                Err(_) => continue,
            };
            for ns in listed {
                // A request sees only its own tenant's namespaces.
                if let Some(tenant) = self.tenant.as_deref() {
                    let tenant = proximadb_tenant::resolve_request_tenant(Some(tenant));
                    if ns.levels.first() != Some(&tenant) {
                        continue;
                    }
                }
                rows.push(vec![
                    catalog_name.clone(),
                    ns.levels.join("."),
                    ns.namespace_id.clone().unwrap_or_default(),
                    ns.tenant_id.clone().unwrap_or_default(),
                    ns.region_home.clone().unwrap_or_default(),
                    format!("{:?}", ns.storage_pool_class).to_lowercase(),
                    ns.owner.clone().unwrap_or_default(),
                    ns.location.clone().unwrap_or_default(),
                ]);
            }
        }
        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec![
                "catalog_name".to_string(),
                "namespace_name".to_string(),
                "namespace_id".to_string(),
                "tenant_id".to_string(),
                "region_home".to_string(),
                "storage_pool_class".to_string(),
                "owner".to_string(),
                "location".to_string(),
            ],
            column_types: vec![
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
            ],
            rows,
        })
    }

    async fn tables(&self, table_filter: Option<&str>) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();

        for (catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            rows.push(vec![
                catalog_name,
                namespace_name(&table_id),
                table_id.name,
                schema
                    .properties
                    .get("schema_kind")
                    .cloned()
                    .unwrap_or_else(|| infer_schema_kind(&schema).to_string()),
                schema
                    .properties
                    .get("storage_engine")
                    .cloned()
                    .unwrap_or_default(),
                schema.properties.get("layout").cloned().unwrap_or_default(),
                schema
                    .properties
                    .get("table_format")
                    .cloned()
                    .unwrap_or_default(),
                schema
                    .properties
                    .get("xcatalog_namespace")
                    .cloned()
                    .unwrap_or_default(),
                schema.schema_version.to_string(),
                // Stable catalog identity is an operator-facing contract: ABAC
                // policy scopes, CDC, and object-id-keyed storage all consume
                // this value. Keep it in the extended xcatalog view (never the
                // SQL-standard information_schema.tables shape) so control-plane
                // clients can provision policy without guessing an allocator
                // result or reaching into catalog persistence.
                // TD-CAT-7: absence renders as an explicit `NULL`, not `""`.
                // A blank cell is what made TD-AUTHZ-2 read as "this view does
                // not populate object_id" rather than "the catalog minted none"
                // — the diagnosis cost, not a parse failure (consumers parse
                // this as u64 and fail on either spelling). #1680's
                // post-condition makes an authority reaching this branch
                // unreachable, so this is defence-in-depth and only the
                // identity column is changed.
                schema
                    .object_id
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "NULL".to_string()),
            ]);
        }

        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec![
                "table_catalog".to_string(),
                "table_schema".to_string(),
                "table_name".to_string(),
                "schema_kind".to_string(),
                "storage_engine".to_string(),
                "layout".to_string(),
                "table_format".to_string(),
                "xcatalog_namespace".to_string(),
                "schema_version".to_string(),
                "object_id".to_string(),
            ],
            column_types: vec![
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "int4".to_string(),
                "int8".to_string(),
            ],
            rows,
        })
    }

    async fn columns(&self, table_filter: Option<&str>) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();

        for (catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            for column in &schema.columns {
                rows.push(vec![
                    catalog_name.clone(),
                    namespace_name(&table_id),
                    table_id.name.clone(),
                    column.name.clone(),
                    column.id.to_string(),
                    catalog_type_name(&column.data_type, &column.properties).to_string(),
                    if column.nullable { "YES" } else { "NO" }.to_string(),
                    column.default_value.clone().unwrap_or_default(),
                    column
                        .properties
                        .get("json_encoding")
                        .cloned()
                        .unwrap_or_default(),
                    column
                        .properties
                        .get("dimension")
                        .cloned()
                        .unwrap_or_else(|| "0".to_string()),
                    if schema.primary_key.contains(&column.name) {
                        "YES"
                    } else {
                        "NO"
                    }
                    .to_string(),
                ]);
            }
        }

        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec![
                "table_catalog".to_string(),
                "table_schema".to_string(),
                "table_name".to_string(),
                "column_name".to_string(),
                "ordinal_position".to_string(),
                "data_type".to_string(),
                "is_nullable".to_string(),
                "column_default".to_string(),
                "json_encoding".to_string(),
                "vector_dimension".to_string(),
                "is_primary_key".to_string(),
            ],
            column_types: vec![
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "int4".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "int4".to_string(),
                "text".to_string(),
            ],
            rows,
        })
    }

    async fn indexes(&self, table_filter: Option<&str>) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();

        for (catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            let catalog = self.catalog_manager.get_catalog(&catalog_name).await?;
            let mut indexes = schema.indexes.clone();
            indexes.extend(catalog.list_indexes(&table_id).await?);
            indexes.sort_by(|left, right| left.name.cmp(&right.name));
            indexes.dedup_by(|left, right| left.name == right.name);

            for index in indexes {
                rows.push(vec![
                    catalog_name.clone(),
                    namespace_name(&table_id),
                    table_id.name.clone(),
                    index.name,
                    catalog_index_type_name(index.index_type).to_string(),
                    index.columns.join(","),
                    index.is_unique.to_string(),
                ]);
            }
        }

        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec![
                "table_catalog".to_string(),
                "table_schema".to_string(),
                "table_name".to_string(),
                "index_name".to_string(),
                "index_type".to_string(),
                "columns".to_string(),
                "is_unique".to_string(),
            ],
            column_types: vec![
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "bool".to_string(),
            ],
            rows,
        })
    }

    async fn table_routing(
        &self,
        table_filter: Option<&str>,
    ) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();

        for (catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            let primary_layout = primary_layout(&schema);
            rows.push(vec![
                catalog_name,
                namespace_name(&table_id),
                table_id.name,
                primary_layout
                    .map(|layout| layout.authority.ownership_mode_name().to_string())
                    .unwrap_or_else(|| "ProximaAuthoritative".to_string()),
                schema.workload_profile.as_str().to_string(),
                schema.storage_specialization.as_str().to_string(),
                primary_layout
                    .map(|layout| format!("{:?}", layout.physical_format))
                    .unwrap_or_default(),
                property_value(&schema, &["compute_route", "preferred_compute_route"])
                    .unwrap_or_default(),
                property_value(
                    &schema,
                    &["partitioning", "partition_key", "distribution_key"],
                )
                .unwrap_or_default(),
                schema
                    .relational_capabilities
                    .transaction_profile
                    .clone()
                    .or_else(|| property_value(&schema, &["isolation_profile", "isolation"]))
                    .unwrap_or_default(),
                property_value(&schema, &["freshness_sla", "projection_freshness"])
                    .unwrap_or_default(),
                policy_boundary(&schema),
                // ADR-059: the materialized-parquet location (storage_layout.location)
                // so the perf-ledger harness can register the SAME parquet with an
                // in-process DuckDB engine (apples-to-apples vs DataFusion).
                primary_layout
                    .and_then(|l| l.location.clone())
                    .unwrap_or_default(),
            ]);
        }

        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec![
                "table_catalog".to_string(),
                "table_schema".to_string(),
                "table_name".to_string(),
                "authority_mode".to_string(),
                "workload_profile".to_string(),
                "storage_specialization".to_string(),
                "primary_format".to_string(),
                "compute_route".to_string(),
                "partitioning".to_string(),
                "isolation_profile".to_string(),
                "freshness_sla".to_string(),
                "policy_boundary".to_string(),
                "location".to_string(),
            ],
            column_types: vec![
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
            ],
            rows,
        })
    }

    async fn jdbc_tables(&self, table_filter: Option<&str>) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();

        for (_catalog_name, table_id, _schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            rows.push(vec![
                String::new(),
                namespace_name(&table_id),
                table_id.name,
                "TABLE".to_string(),
            ]);
        }

        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec![
                "TABLE_CAT".to_string(),
                "TABLE_SCHEM".to_string(),
                "TABLE_NAME".to_string(),
                "TABLE_TYPE".to_string(),
            ],
            column_types: vec![
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
            ],
            rows,
        })
    }

    async fn jdbc_columns(&self, table_filter: Option<&str>) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();

        for (_catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            for column in &schema.columns {
                let (data_type, column_size) = jdbc_type_and_size(&column.data_type);
                rows.push(vec![
                    String::new(),
                    namespace_name(&table_id),
                    table_id.name.clone(),
                    column.name.clone(),
                    data_type.to_string(),
                    column_size.to_string(),
                    if column.nullable { "YES" } else { "NO" }.to_string(),
                ]);
            }
        }

        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec![
                "TABLE_CAT".to_string(),
                "TABLE_SCHEM".to_string(),
                "TABLE_NAME".to_string(),
                "COLUMN_NAME".to_string(),
                "DATA_TYPE".to_string(),
                "COLUMN_SIZE".to_string(),
                "IS_NULLABLE".to_string(),
            ],
            column_types: vec![
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "int4".to_string(),
                "int4".to_string(),
                "text".to_string(),
            ],
            rows,
        })
    }

    async fn sqlalchemy_table_names(
        &self,
        table_filter: Option<&str>,
    ) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();

        for (_catalog_name, table_id, _schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            rows.push(vec![table_id.name]);
        }

        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec!["relname".to_string()],
            column_types: vec!["text".to_string()],
            rows,
        })
    }

    async fn sqlalchemy_columns(
        &self,
        table_filter: Option<&str>,
    ) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();

        for (_catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            for column in &schema.columns {
                rows.push(vec![
                    column.name.clone(),
                    postgres_format_type(&column.data_type, &column.properties).to_string(),
                    column.default_value.clone().unwrap_or_default(),
                    (!column.nullable).to_string(),
                    table_id.name.clone(),
                    column.comment.clone().unwrap_or_default(),
                    String::new(),
                    String::new(),
                ]);
            }
        }

        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec![
                "name".to_string(),
                "format_type".to_string(),
                "default".to_string(),
                "not_null".to_string(),
                "table_name".to_string(),
                "comment".to_string(),
                "generated".to_string(),
                "identity_options".to_string(),
            ],
            column_types: vec![
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "bool".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
            ],
            rows,
        })
    }

    async fn jdbc_indexes(&self, table_filter: Option<&str>) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();

        for (_catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            for (idx, column_name) in schema.primary_key.iter().enumerate() {
                rows.push(vec![
                    String::new(),
                    namespace_name(&table_id),
                    table_id.name.clone(),
                    "false".to_string(),
                    String::new(),
                    format!("{}_pkey", table_id.name),
                    "3".to_string(),
                    (idx + 1).to_string(),
                    column_name.clone(),
                    "A".to_string(),
                ]);
            }
        }

        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec![
                "TABLE_CAT".to_string(),
                "TABLE_SCHEM".to_string(),
                "TABLE_NAME".to_string(),
                "NON_UNIQUE".to_string(),
                "INDEX_QUALIFIER".to_string(),
                "INDEX_NAME".to_string(),
                "TYPE".to_string(),
                "ORDINAL_POSITION".to_string(),
                "COLUMN_NAME".to_string(),
                "ASC_OR_DESC".to_string(),
            ],
            column_types: vec![
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "bool".to_string(),
                "text".to_string(),
                "text".to_string(),
                "int4".to_string(),
                "int4".to_string(),
                "text".to_string(),
                "text".to_string(),
            ],
            rows,
        })
    }

    /// Return props promotion state for all tables (or a single table when `table_filter` is set).
    ///
    /// Columns: table_name, props_key, promoted_column, promotion_enabled,
    ///          frequency_threshold, min_record_count, status
    async fn props_promotion_candidates(
        &self,
        table_filter: Option<&str>,
    ) -> Result<CatalogIntrospectionResult> {
        let mut rows: Vec<Vec<String>> = Vec::new();

        for (_catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }

            let policy = &schema.props_auto_promotion;
            let enabled_str = if policy.enabled { "true" } else { "false" };
            let threshold_str = format!("{:.2}", policy.frequency_threshold);
            let min_count_str = policy.min_record_count.to_string();

            // Rows for already-promoted keys.
            for (key, col_name) in &policy.promoted_keys {
                rows.push(vec![
                    table_id.name.clone(),
                    key.clone(),
                    col_name.clone(),
                    enabled_str.to_string(),
                    threshold_str.clone(),
                    min_count_str.clone(),
                    "promoted".to_string(),
                ]);
            }

            // If auto-promotion is enabled and no keys are promoted yet, emit one
            // summary row so the table is visible even before any promotions.
            if policy.enabled && policy.promoted_keys.is_empty() {
                rows.push(vec![
                    table_id.name.clone(),
                    String::new(),
                    String::new(),
                    enabled_str.to_string(),
                    threshold_str.clone(),
                    min_count_str.clone(),
                    "no_candidates_yet".to_string(),
                ]);
            }
        }

        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: vec![
                "table_name".to_string(),
                "props_key".to_string(),
                "promoted_column".to_string(),
                "promotion_enabled".to_string(),
                "frequency_threshold".to_string(),
                "min_record_count".to_string(),
                "status".to_string(),
            ],
            column_types: vec![
                "text".to_string(),
                "text".to_string(),
                "text".to_string(),
                "boolean".to_string(),
                "float8".to_string(),
                "int8".to_string(),
                "text".to_string(),
            ],
            rows,
        })
    }

    /// `pg_catalog.pg_type` — the supported base types with their canonical
    /// PostgreSQL OIDs. Static; ORMs read this to resolve column type OIDs.
    fn pg_type() -> CatalogIntrospectionResult {
        // (oid, typname, typlen, typbyval, typcategory)
        const TYPES: &[(i32, &str, i32, bool, &str)] = &[
            (16, "bool", 1, true, "B"),
            (17, "bytea", -1, false, "U"),
            (20, "int8", 8, true, "N"),
            (21, "int2", 2, true, "N"),
            (23, "int4", 4, true, "N"),
            (25, "text", -1, false, "S"),
            (114, "json", -1, false, "U"),
            (700, "float4", 4, true, "N"),
            (701, "float8", 8, true, "N"),
            (1043, "varchar", -1, false, "S"),
            (1082, "date", 4, true, "D"),
            (1114, "timestamp", 8, true, "D"),
            (1184, "timestamptz", 8, true, "D"),
            (1700, "numeric", -1, false, "N"),
            (2950, "uuid", 16, false, "U"),
            (3802, "jsonb", -1, false, "U"),
        ];
        let rows = TYPES
            .iter()
            .map(|(oid, name, len, byval, category)| {
                vec![
                    oid.to_string(),
                    name.to_string(),
                    PG_CATALOG_NAMESPACE_OID.to_string(),
                    len.to_string(),
                    bool_str(*byval).to_string(),
                    "b".to_string(),
                    category.to_string(),
                ]
            })
            .collect();
        CatalogIntrospectionResult {
            columns: str_cols(&[
                "oid",
                "typname",
                "typnamespace",
                "typlen",
                "typbyval",
                "typtype",
                "typcategory",
            ]),
            column_types: str_cols(&["int4", "text", "int4", "int2", "bool", "text", "text"]),
            rows,
        }
    }

    /// `pg_catalog.pg_namespace` — schemas. Includes the fixed system
    /// namespaces plus every distinct namespace present in the catalog.
    async fn pg_namespace(&self) -> Result<CatalogIntrospectionResult> {
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut rows: Vec<Vec<String>> = vec![
            vec!["11".to_string(), "pg_catalog".to_string(), "10".to_string()],
            vec!["2200".to_string(), "public".to_string(), "10".to_string()],
            vec![
                INFORMATION_SCHEMA_NAMESPACE_OID.to_string(),
                "information_schema".to_string(),
                "10".to_string(),
            ],
        ];
        for (_catalog, table_id, _schema) in self.cataloged_tables().await? {
            let ns = namespace_name(&table_id);
            if ns == "public" || !seen.insert(ns.clone()) {
                continue;
            }
            rows.push(vec![namespace_oid(&ns).to_string(), ns, "10".to_string()]);
        }
        Ok(CatalogIntrospectionResult {
            columns: str_cols(&["oid", "nspname", "nspowner"]),
            column_types: str_cols(&["int4", "text", "int4"]),
            rows,
        })
    }

    /// `pg_catalog.pg_class` — relations (tables) backed by the catalog.
    async fn pg_class(&self, relname_filter: Option<&str>) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();
        for (catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, relname_filter) {
                continue;
            }
            let ns = namespace_name(&table_id);
            rows.push(vec![
                table_oid(&catalog_name, &table_id).to_string(),
                table_id.name.clone(),
                namespace_oid(&ns).to_string(),
                "r".to_string(),
                schema.columns.len().to_string(),
                "0".to_string(),
                bool_str(!schema.indexes.is_empty()).to_string(),
            ]);
        }
        Ok(CatalogIntrospectionResult {
            columns: str_cols(&[
                "oid",
                "relname",
                "relnamespace",
                "relkind",
                "relnatts",
                "reltype",
                "relhasindex",
            ]),
            column_types: str_cols(&["int4", "text", "int4", "text", "int2", "int4", "bool"]),
            rows,
        })
    }

    /// `pg_catalog.pg_attribute` — columns. `attnum` is 1-based per
    /// PostgreSQL convention; `atttypid` is the canonical type OID.
    async fn pg_attribute(
        &self,
        relname_filter: Option<&str>,
    ) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();
        for (catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, relname_filter) {
                continue;
            }
            let attrelid = table_oid(&catalog_name, &table_id);
            for (idx, column) in schema.columns.iter().enumerate() {
                rows.push(vec![
                    attrelid.to_string(),
                    column.name.clone(),
                    proxima_type_oid(&column.data_type).to_string(),
                    proxima_type_len(&column.data_type).to_string(),
                    (idx as i32 + 1).to_string(),
                    bool_str(!column.nullable).to_string(),
                    "f".to_string(),
                    "f".to_string(),
                ]);
            }
        }
        Ok(CatalogIntrospectionResult {
            columns: str_cols(&[
                "attrelid",
                "attname",
                "atttypid",
                "attlen",
                "attnum",
                "attnotnull",
                "atthasdef",
                "attisdropped",
            ]),
            column_types: str_cols(&[
                "int4", "text", "int4", "int2", "int2", "bool", "bool", "bool",
            ]),
            rows,
        })
    }

    /// `pg_catalog.pg_constraint` — primary-key, unique, foreign-key and
    /// check constraints, synthesized from the catalog's relational metadata.
    async fn pg_constraint(
        &self,
        relname_filter: Option<&str>,
    ) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();
        for (catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, relname_filter) {
                continue;
            }
            let conrelid = table_oid(&catalog_name, &table_id);
            let mut push = |conname: String, contype: &str, cols: &[String]| {
                rows.push(vec![
                    stable_oid(&[&table_id.name, &conname]).to_string(),
                    conname,
                    contype.to_string(),
                    conrelid.to_string(),
                    cols.join(","),
                ]);
            };
            if !schema.primary_key.is_empty() {
                push(format!("{}_pkey", table_id.name), "p", &schema.primary_key);
            }
            for constraint in &schema.relational_capabilities.constraints {
                match constraint {
                    proximadb_catalog::ColumnConstraint::Unique { columns } => push(
                        format!("{}_{}_key", table_id.name, columns.join("_")),
                        "u",
                        columns,
                    ),
                    proximadb_catalog::ColumnConstraint::Check { expression } => push(
                        format!("{}_check", table_id.name),
                        "c",
                        std::slice::from_ref(expression),
                    ),
                    proximadb_catalog::ColumnConstraint::ForeignKey { columns, .. } => push(
                        format!("{}_{}_fkey", table_id.name, columns.join("_")),
                        "f",
                        columns,
                    ),
                }
            }
        }
        Ok(CatalogIntrospectionResult {
            columns: str_cols(&["oid", "conname", "contype", "conrelid", "conkey"]),
            column_types: str_cols(&["int4", "text", "text", "int4", "text"]),
            rows,
        })
    }

    /// Shape-aware `\dt`/relation-list builder: return exactly the columns the
    /// client's SELECT list named (the parsed aliases), one row per deduped
    /// table, populating the aliases we recognize and leaving the rest empty.
    /// This is what makes psql `\dt` show correct data instead of receiving a
    /// 1-column shape (from the narrow `sqlalchemy_table_names` builder) that
    /// triggers libpq's "column number N is out of range" and a client crash.
    /// All cells are text-encoded (psql/ORMs treat pg_catalog probe columns as
    /// text), so all `column_types` are `text`.
    async fn psql_shaped_table_list(
        &self,
        aliases: &[String],
        table_filter: Option<&str>,
    ) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();
        for (catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            let row = aliases
                .iter()
                .map(|label| {
                    let stripped = label.trim_matches('"');
                    match stripped.to_ascii_lowercase().as_str() {
                        "schema" | "nspname" | "table_schema" | "namespace" => {
                            namespace_name(&table_id)
                        }
                        "name" | "relname" | "table_name" | "relation" => table_id.name.clone(),
                        "owner" => "postgres".to_string(),
                        "type" => "ordinary table".to_string(),
                        "relkind" => "r".to_string(),
                        "relowner" => "10".to_string(),
                        "relnamespace" => namespace_oid(&namespace_name(&table_id)).to_string(),
                        "relnatts" | "natts" => schema.columns.len().to_string(),
                        "oid" | "relid" => table_oid(&catalog_name, &table_id).to_string(),
                        "relhasindex" => bool_str(!schema.indexes.is_empty()).to_string(),
                        "relpersistence" => "p".to_string(),
                        "relrowsecurity" | "relforcerowsecurity" => "f".to_string(),
                        _ => String::new(),
                    }
                })
                .collect();
            rows.push(row);
        }
        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: aliases.to_vec(),
            column_types: aliases.iter().map(|_| "text".to_string()).collect(),
            rows,
        })
    }

    /// Shape-aware `\d` (column) builder: return exactly the columns the
    /// client's SELECT list named, one row per column of matching tables. Maps
    /// the common pg_attribute aliases (`Column`/`Type`/`Default`/`Nullable`/
    /// ...) to catalog values and leaves unrecognized aliases empty.
    async fn psql_shaped_columns(
        &self,
        aliases: &[String],
        table_filter: Option<&str>,
    ) -> Result<CatalogIntrospectionResult> {
        let mut rows = Vec::new();
        for (_catalog_name, table_id, schema) in self.cataloged_tables().await? {
            if !matches_filter(&table_id.name, table_filter) {
                continue;
            }
            for (idx, column) in schema.columns.iter().enumerate() {
                let row = aliases
                    .iter()
                    .map(|label| {
                        let stripped = label.trim_matches('"');
                        match stripped.to_ascii_lowercase().as_str() {
                            "column" | "attname" | "field" | "column_name" => column.name.clone(),
                            "type" | "format_type" | "data_type" | "udt_name" => {
                                postgres_format_type(&column.data_type, &column.properties)
                                    .to_string()
                            }
                            "default" | "column_default" | "default_value" => {
                                column.default_value.clone().unwrap_or_default()
                            }
                            "atthasdef" => bool_str(column.default_value.is_some()).to_string(),
                            "nullable" | "is_nullable" => {
                                if column.nullable {
                                    "YES".to_string()
                                } else {
                                    "NO".to_string()
                                }
                            }
                            "not_null" | "notnull" | "attnotnull" => (!column.nullable).to_string(),
                            "ordinal" | "attnum" | "ordinal_position" | "attposition" => {
                                (idx as i32 + 1).to_string()
                            }
                            "comment" => column.comment.clone().unwrap_or_default(),
                            _ => String::new(),
                        }
                    })
                    .collect();
                rows.push(row);
            }
        }
        rows.sort();
        Ok(CatalogIntrospectionResult {
            columns: aliases.to_vec(),
            column_types: aliases.iter().map(|_| "text".to_string()).collect(),
            rows,
        })
    }

    async fn cataloged_tables(&self) -> Result<Vec<(String, TableIdentifier, CatalogTableSchema)>> {
        let mut tables = Vec::new();
        for catalog_name in self.catalog_manager.list_catalogs().await {
            let catalog = self.catalog_manager.get_catalog(&catalog_name).await?;
            for namespace in catalog.list_namespaces(None).await? {
                for table_id in catalog.list_tables(&namespace.levels).await? {
                    let schema = catalog.get_table(&table_id).await?;
                    if !self.visible_to_tenant(&table_id, &schema) {
                        continue;
                    }
                    tables.push((catalog_name.clone(), table_id, schema));
                }
            }
        }

        // Dedup dual-registered tables. A table created via SQL surfaces both
        // as an internal `proximadb.<ns>` collection namespace and as the
        // SQL-standard `<ns>` schema (observed: `b` listed twice as
        // `proximadb.default.b` and `public.b`). Collapse by normalized identity
        // (see `table_identity_key`), preferring:
        //   1. The catalog with a materialized Parquet layout (the CURRENT
        //      storage state — `set_storage_layouts` updates only the internal
        //      `proximadb.` catalog, so the SQL-standard form may be stale).
        //   2. The SQL-standard form (no `proximadb.` prefix) as a tiebreaker.
        // Same-name tables in different user namespaces stay distinct.
        tables.sort_by_key(|(_, table_id, schema)| {
            let has_parquet = schema
                .storage_layouts
                .iter()
                .any(|l| l.physical_format == proximadb_catalog::CatalogPhysicalFormat::Parquet);
            // Sort key: (!has_parquet, is_proximadb) — Parquet layouts first,
            // then non-proximadb (SQL-standard) within each group.
            (
                !has_parquet,
                table_id
                    .namespace
                    .first()
                    .is_some_and(|first| first.eq_ignore_ascii_case("proximadb")),
            )
        });
        let mut seen = std::collections::BTreeSet::new();
        tables.retain(|(_, table_id, _)| seen.insert(table_identity_key(table_id)));

        Ok(tables)
    }
}

fn normalize_sql(sql: &str) -> String {
    sql.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn extract_string_filter(sql: &str, column: &str) -> Option<String> {
    let normalized = normalize_sql(sql);
    let needle = format!("{column} = '");
    let start = normalized.find(&needle)? + needle.len();
    let rest = &normalized[start..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

/// Extract the first single-quoted string argument from a function call.
/// e.g. `SELECT * FROM proximadb_catalog.props_promotion_candidates('events')` → `Some("events")`
fn extract_function_arg(sql: &str, fn_name: &str) -> Option<String> {
    let lower = sql.to_lowercase();
    let needle = format!("{}('", fn_name.to_lowercase());
    let start = lower.find(&needle)? + needle.len();
    let rest = &sql[start..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

fn matches_filter(value: &str, filter: Option<&str>) -> bool {
    filter.is_none_or(|filter| value.eq_ignore_ascii_case(filter))
}

fn primary_layout(schema: &CatalogTableSchema) -> Option<&proximadb_catalog::CatalogStorageLayout> {
    // Prefer the materialized Parquet layout (ProjectionPublication) — it has
    // the correct `location` (the data dir where parquet is immediate, per
    // ADR-059). The original "primary" layout may have a stale/empty location.
    schema
        .storage_layouts
        .iter()
        .rev()
        .find(|l| l.physical_format == proximadb_catalog::CatalogPhysicalFormat::Parquet)
        .or_else(|| {
            schema
                .storage_layouts
                .iter()
                .rev()
                .find(|l| l.name == "primary")
        })
        .or_else(|| schema.storage_layouts.first())
}

fn property_value(schema: &CatalogTableSchema, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| schema.properties.get(*key).cloned())
}

fn policy_boundary(schema: &CatalogTableSchema) -> String {
    if let Some(boundary) = property_value(schema, &["policy_boundary", "rls_boundary"]) {
        return boundary;
    }

    match primary_layout(schema) {
        Some(layout) if layout.policy_enforced_in_proxima => "engine-enforced".to_string(),
        Some(layout) if layout.authority.is_external_authoritative() => {
            "external-policy".to_string()
        }
        Some(layout)
            if matches!(
                layout.authority,
                proximadb_catalog::CatalogAuthorityMode::FederatedRead
            ) =>
        {
            "connector-enforced".to_string()
        }
        Some(_) => "unsupported".to_string(),
        None => "engine-enforced".to_string(),
    }
}

fn namespace_name(table_id: &TableIdentifier) -> String {
    let namespace = table_id.namespace.join(".");
    if namespace.is_empty() || namespace == "default" {
        "public".to_string()
    } else {
        namespace
    }
}

/// Fully-qualified logical identity for a catalog table, used to dedup
/// `cataloged_tables()` across the catalog × namespace cross-product. Uses the
/// normalized [`identity_namespace`] (NOT the raw display namespace) so the
/// same logical table — which surfaces both as an internal `proximadb.<ns>`
/// collection namespace and as the SQL-standard `<ns>` schema (e.g.
/// `proximadb.default` and `public`) — collapses to one identity. Two same-name
/// tables in DIFFERENT user namespaces (`analytics.events` vs `audit.events`)
/// remain distinct.
fn table_identity_key(table_id: &TableIdentifier) -> String {
    format!("{}|{}", identity_namespace(table_id), table_id.name)
}

/// Normalized namespace used for dedup identity. Strips a leading `proximadb`
/// segment — the collection/relational duality surfaces one namespace both as
/// `proximadb.<ns>` and as `<ns>` — and maps `default`/empty → `public`, so
/// `proximadb.default` and `public` (and bare `default`) all identify the same
/// logical namespace.
fn identity_namespace(table_id: &TableIdentifier) -> String {
    let levels = &table_id.namespace;
    let levels = if levels
        .first()
        .is_some_and(|first| first.eq_ignore_ascii_case("proximadb"))
    {
        &levels[1..]
    } else {
        levels
    };
    let joined = levels
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(".");
    if joined.is_empty() || joined == "default" {
        "public".to_string()
    } else {
        joined
    }
}

fn infer_schema_kind(schema: &CatalogTableSchema) -> &'static str {
    let has_json = schema
        .columns
        .iter()
        .any(|column| matches!(column.data_type, ProximaType::Json));
    let has_vector = schema.columns.iter().any(|column| {
        matches!(
            column.data_type,
            ProximaType::DenseVector { .. }
                | ProximaType::SparseVector { .. }
                | ProximaType::BinaryVector { .. }
        )
    });

    match (has_json, has_vector) {
        (true, true) => "mixed_relational_document_vector",
        (true, false) => "relational_document",
        (false, true) => "relational_vector",
        (false, false) => "relational",
    }
}

fn catalog_type_name(
    data_type: &ProximaType,
    properties: &std::collections::HashMap<String, String>,
) -> &'static str {
    match data_type {
        ProximaType::Boolean => "boolean",
        ProximaType::Int8 => "int8",
        ProximaType::Int16 => "int16",
        ProximaType::Int32 => "int32",
        ProximaType::Int64 => "int64",
        ProximaType::Float32 => "float32",
        ProximaType::Float64 => "float64",
        ProximaType::String => "text",
        ProximaType::Binary => "bytea",
        ProximaType::Date => "date",
        ProximaType::Time(_) => "time",
        ProximaType::Timestamp(_) => "timestamp",
        ProximaType::TimestampTz(_) => "timestamptz",
        ProximaType::Decimal { .. } => "decimal",
        ProximaType::Uuid => "uuid",
        ProximaType::Json => match properties.get("json_encoding").map(String::as_str) {
            Some("jsonb") => "jsonb",
            _ => "json",
        },
        ProximaType::DenseVector { .. } => "vector",
        ProximaType::SparseVector { .. } => "sparse_vector",
        ProximaType::BinaryVector { .. } => "binary_vector",
        _ => "text",
    }
}

fn postgres_format_type(
    data_type: &ProximaType,
    properties: &std::collections::HashMap<String, String>,
) -> &'static str {
    match data_type {
        ProximaType::Boolean => "boolean",
        ProximaType::Int8 => "smallint",
        ProximaType::Int16 => "smallint",
        ProximaType::Int32 => "integer",
        ProximaType::Int64 => "bigint",
        ProximaType::Float32 => "real",
        ProximaType::Float64 => "double precision",
        ProximaType::String => "character varying",
        ProximaType::Binary => "bytea",
        ProximaType::Date => "date",
        ProximaType::Time(_) => "time without time zone",
        ProximaType::Timestamp(_) => "timestamp without time zone",
        ProximaType::TimestampTz(_) => "timestamp with time zone",
        ProximaType::Decimal { .. } => "numeric",
        ProximaType::Uuid => "uuid",
        ProximaType::Json => match properties.get("json_encoding").map(String::as_str) {
            Some("jsonb") => "jsonb",
            _ => "json",
        },
        ProximaType::DenseVector { .. } => "vector",
        ProximaType::SparseVector { .. } => "sparse_vector",
        ProximaType::BinaryVector { .. } => "binary_vector",
        _ => "character varying",
    }
}

fn jdbc_type_and_size(data_type: &ProximaType) -> (i32, i32) {
    match data_type {
        ProximaType::Boolean => (16, 1),
        ProximaType::Int8 => (-6, 3),
        ProximaType::Int16 => (5, 5),
        ProximaType::Int32 => (4, 10),
        ProximaType::Int64 => (-5, 19),
        ProximaType::Float32 => (6, 24),
        ProximaType::Float64 => (8, 53),
        ProximaType::String => (12, 255),
        ProximaType::Binary => (-2, 255),
        ProximaType::Date => (91, 13),
        ProximaType::Time(_) => (92, 15),
        ProximaType::Timestamp(_) | ProximaType::TimestampTz(_) => (93, 29),
        ProximaType::Decimal { .. } => (3, 38),
        ProximaType::Uuid => (1111, 36),
        ProximaType::Json => (1111, 0),
        ProximaType::DenseVector { .. }
        | ProximaType::SparseVector { .. }
        | ProximaType::BinaryVector { .. } => (1111, 0),
        // Richer ProximaType variants → OTHER JDBC type, text-ish size.
        _ => (1111, 0),
    }
}

fn catalog_index_type_name(index_type: CatalogIndexType) -> &'static str {
    match index_type {
        CatalogIndexType::BTree => "btree",
        CatalogIndexType::Hash => "hash",
        CatalogIndexType::FullText => "fulltext",
        CatalogIndexType::Gin => "gin",
        CatalogIndexType::Hnsw => "hnsw",
        CatalogIndexType::Ivf => "ivf",
        CatalogIndexType::Pq => "pq",
    }
}

// --- pg_catalog view helpers (TD-120) ---

/// `pg_catalog` schema OID (fixed in PostgreSQL).
const PG_CATALOG_NAMESPACE_OID: i32 = 11;
/// Stable synthetic OID for the `information_schema` namespace.
const INFORMATION_SCHEMA_NAMESPACE_OID: i32 = 13000;

/// PostgreSQL text-format boolean encoding (`t`/`f`), not `true`/`false`.
fn bool_str(value: bool) -> &'static str {
    if value { "t" } else { "f" }
}

fn str_cols(cols: &[&str]) -> Vec<String> {
    cols.iter().map(|c| c.to_string()).collect()
}

/// Deterministic, restart-stable OID in the user range (≥ 16384) derived from
/// `parts` via FNV-1a. ORMs join on these so they must be stable, not random.
fn stable_oid(parts: &[&str]) -> i32 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= u64::from(b'.');
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Land in [16384, 16384 + 0x3FFF_FFFF] so it is always a positive i32
    // above PostgreSQL's reserved system-OID range.
    (((hash & 0x3FFF_FFFF) as i32).wrapping_add(16384)).max(16384)
}

fn namespace_oid(ns_name: &str) -> i32 {
    match ns_name {
        "pg_catalog" => PG_CATALOG_NAMESPACE_OID,
        "public" | "default" => 2200,
        "information_schema" => INFORMATION_SCHEMA_NAMESPACE_OID,
        other => stable_oid(&["ns", other]),
    }
}

fn table_oid(catalog_name: &str, table_id: &TableIdentifier) -> i32 {
    stable_oid(&[catalog_name, &table_id.namespace.join("."), &table_id.name])
}

/// Canonical PostgreSQL type OID for a ProximaDB logical type.
fn proxima_type_oid(data_type: &ProximaType) -> i32 {
    match data_type {
        ProximaType::Boolean => 16,
        ProximaType::Int8 | ProximaType::Int16 => 21,
        ProximaType::Int32 => 23,
        ProximaType::Int64 => 20,
        ProximaType::Float32 => 700,
        ProximaType::Float64 => 701,
        ProximaType::String => 25,
        ProximaType::Binary => 17,
        ProximaType::Date => 1082,
        ProximaType::Time(_) => 1083,
        ProximaType::Timestamp(_) => 1114,
        ProximaType::TimestampTz(_) => 1184,
        ProximaType::Decimal { .. } => 1700,
        ProximaType::Uuid => 2950,
        ProximaType::Json => 3802,
        // Vector/other types have no standard pg OID.
        _ => 0,
    }
}

/// Physical length for a type OID (`-1` for variable-length).
fn proxima_type_len(data_type: &ProximaType) -> i32 {
    match data_type {
        ProximaType::Boolean => 1,
        ProximaType::Int8 | ProximaType::Int16 => 2,
        ProximaType::Int32 | ProximaType::Float32 | ProximaType::Date => 4,
        ProximaType::Int64
        | ProximaType::Float64
        | ProximaType::Timestamp(_)
        | ProximaType::TimestampTz(_) => 8,
        ProximaType::Uuid => 16,
        _ => -1,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use proximadb::query::sql_frontend::SqlFrontendParser;
    use proximadb::services::{DdlService, DdlStatement};

    /// The dedup key must collapse the SAME logical table surfacing under two
    /// namespace forms — the internal `proximadb.<ns>` collection namespace and
    /// the SQL-standard `<ns>` schema (the user's `b` listed twice by `\dt` as
    /// `proximadb.default.b` and `public.b`) — while keeping same-name tables in
    /// DIFFERENT user namespaces distinct.
    #[test]
    fn table_identity_key_collapses_duplicates_keeps_cross_namespace() {
        // The observed duality: `proximadb.default.b` and `public.b` are the
        // same logical table and must collapse.
        let via_proximadb_ns =
            TableIdentifier::new(vec!["proximadb".to_string(), "default".to_string()], "b");
        let via_public = TableIdentifier::new(vec!["public".to_string()], "b");
        assert_eq!(
            table_identity_key(&via_proximadb_ns),
            table_identity_key(&via_public)
        );

        // default/empty roots also normalize to "public".
        let via_default = TableIdentifier::new(vec!["default".to_string()], "b");
        let via_empty_root = TableIdentifier::new(Vec::new(), "b");
        assert_eq!(
            table_identity_key(&via_default),
            table_identity_key(&via_empty_root)
        );
        assert_eq!(
            table_identity_key(&via_default),
            table_identity_key(&via_public)
        );

        // Same name, different user namespaces → distinct (must NOT collapse).
        let events_analytics = TableIdentifier::new(vec!["analytics".to_string()], "events");
        let events_audit = TableIdentifier::new(vec!["audit".to_string()], "events");
        assert_ne!(
            table_identity_key(&events_analytics),
            table_identity_key(&events_audit)
        );
    }

    #[tokio::test]
    async fn test_catalog_introspection_projects_agentic_mixed_schema() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let manager = Arc::new(CatalogManager::new());
        manager
            .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
            .await
            .expect("native catalog");

        let ddl = DdlService::new(manager.clone());
        ddl.execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

        let parser = SqlFrontendParser::new();
        let table = parser
            .parse_ddl(
                "CREATE TABLE IF NOT EXISTS agent_store (
                    record_id TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    embedding VECTOR(16),
                    PRIMARY KEY (record_id)
                ) WITH (
                    storage_engine = 'SST',
                    workload = 'htap',
                    layout = 'hybrid',
                    xcatalog_namespace = 'agentic.langgraph',
                    schema_kind = 'agentic_mixed',
                    compute_route = 'datafusion-local',
                    partitioning = 'tenant_id,bucket',
                    isolation = 'snapshot-isolation',
                    freshness_sla = '5s',
                    policy_boundary = 'engine-enforced'
                );",
            )
            .expect("parse table")
            .expect("table ddl");
        ddl.execute(table).await.expect("create table");

        for sql in [
            "CREATE INDEX idx_payload ON agent_store USING GIN (payload);",
            "CREATE INDEX idx_embedding ON agent_store USING HNSW (embedding);",
        ] {
            let index = parser
                .parse_ddl(sql)
                .expect("parse index")
                .expect("index ddl");
            ddl.execute(index).await.expect("create index");
        }

        let introspection = CatalogIntrospectionService::new(manager);
        let tables = introspection
            .execute_select("SELECT * FROM xcatalog.tables WHERE table_name = 'agent_store'")
            .await
            .expect("tables query")
            .expect("tables result");
        assert_eq!(tables.rows.len(), 1);
        assert_eq!(tables.rows[0][2], "agent_store");
        assert_eq!(tables.rows[0][3], "agentic_mixed");
        assert_eq!(tables.rows[0][4], "SST");
        assert_eq!(tables.rows[0][5], "hybrid");
        assert_eq!(tables.rows[0][7], "agentic.langgraph");
        assert_eq!(tables.columns[9], "object_id");
        assert!(
            tables.rows[0][9].parse::<u64>().is_ok_and(|id| id > 0),
            "xcatalog.tables must expose the stable object id needed by policy provisioning"
        );

        let columns = introspection
            .execute_select("SELECT * FROM xcatalog.columns WHERE table_name = 'agent_store'")
            .await
            .expect("columns query")
            .expect("columns result");
        assert!(
            columns
                .rows
                .iter()
                .any(|row| row[3] == "payload" && row[5] == "jsonb")
        );
        assert!(
            columns
                .rows
                .iter()
                .any(|row| row[3] == "embedding" && row[5] == "vector" && row[9] == "16")
        );

        let indexes = introspection
            .execute_select("SELECT * FROM xcatalog.indexes WHERE table_name = 'agent_store'")
            .await
            .expect("indexes query")
            .expect("indexes result");
        assert!(
            indexes
                .rows
                .iter()
                .any(|row| row[3] == "idx_payload" && row[4] == "gin")
        );
        assert!(
            indexes
                .rows
                .iter()
                .any(|row| row[3] == "idx_embedding" && row[4] == "hnsw")
        );

        let routing = introspection
            .execute_select(
                "SELECT * FROM information_schema.table_routing WHERE table_name = 'agent_store'",
            )
            .await
            .expect("routing query")
            .expect("routing result");
        assert_eq!(routing.rows.len(), 1);
        assert_eq!(routing.rows[0][2], "agent_store");
        assert_eq!(routing.rows[0][3], "ProximaAuthoritative");
        assert_eq!(routing.rows[0][4], "htap");
        assert_eq!(routing.rows[0][5], "pax_row_family");
        assert_eq!(routing.rows[0][7], "datafusion-local");
        assert_eq!(routing.rows[0][8], "tenant_id,bucket");
        assert_eq!(routing.rows[0][9], "snapshot-isolation");
        assert_eq!(routing.rows[0][10], "5s");
        assert_eq!(routing.rows[0][11], "engine-enforced");
    }

    #[tokio::test]
    async fn test_props_promotion_candidates_reflects_promoted_keys() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let manager = Arc::new(CatalogManager::new());
        manager
            .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
            .await
            .expect("native catalog");

        let ddl = DdlService::new(manager.clone());
        ddl.execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

        let parser = SqlFrontendParser::new();
        let create = parser
            .parse_ddl("CREATE TABLE events (id TEXT NOT NULL, payload JSONB, PRIMARY KEY (id))")
            .expect("parse")
            .expect("ddl");
        ddl.execute(create).await.expect("create table");

        // Enable auto-promotion
        let set_opt = parser
            .parse_ddl("ALTER TABLE events SET (props_auto_promotion = 'enabled')")
            .expect("parse set option")
            .expect("ddl");
        ddl.execute(set_opt).await.expect("set option");

        // Promote a key
        let promote = parser
            .parse_ddl("ALTER TABLE events PROMOTE PROPS KEY user_id TYPE BIGINT")
            .expect("parse promote")
            .expect("ddl");
        ddl.execute(promote).await.expect("promote key");

        let introspection = CatalogIntrospectionService::new(manager.clone());

        // Query via function-call form
        let result = introspection
            .execute_select("SELECT * FROM proximadb_catalog.props_promotion_candidates('events')")
            .await
            .expect("query")
            .expect("result");

        assert_eq!(
            result.columns,
            [
                "table_name",
                "props_key",
                "promoted_column",
                "promotion_enabled",
                "frequency_threshold",
                "min_record_count",
                "status"
            ]
        );

        let promoted = result
            .rows
            .iter()
            .find(|r| r[1] == "user_id")
            .expect("expected user_id row");
        assert_eq!(promoted[0], "events");
        assert_eq!(promoted[2], "props__user_id");
        assert_eq!(promoted[3], "true");
        assert_eq!(promoted[6], "promoted");
    }

    #[tokio::test]
    async fn pg_catalog_views_project_catalog_metadata() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let manager = Arc::new(CatalogManager::new());
        manager
            .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
            .await
            .expect("native catalog");

        let ddl = DdlService::new(manager.clone());
        ddl.execute(DdlStatement::CreateNamespace {
            namespace: vec!["default".to_string()],
            if_not_exists: true,
            properties: HashMap::new(),
        })
        .await
        .expect("namespace");

        let parser = SqlFrontendParser::new();
        let create = parser
            .parse_ddl("CREATE TABLE pg_demo (id TEXT NOT NULL, score INT, PRIMARY KEY (id))")
            .expect("parse")
            .expect("ddl");
        ddl.execute(create).await.expect("create table");

        let introspection = CatalogIntrospectionService::new(manager);

        // Detection routes all canonical pg_catalog views.
        for sql in [
            "SELECT * FROM pg_catalog.pg_type",
            "SELECT * FROM pg_catalog.pg_namespace",
            "SELECT * FROM pg_catalog.pg_class WHERE relname = 'pg_demo'",
            "SELECT * FROM pg_catalog.pg_attribute WHERE relname = 'pg_demo'",
            "SELECT * FROM pg_catalog.pg_constraint",
        ] {
            assert!(
                CatalogIntrospectionService::is_catalog_query(sql),
                "should detect: {sql}"
            );
        }

        // pg_type: canonical OIDs for the base types ORMs resolve against.
        let types = introspection
            .execute_select("SELECT * FROM pg_catalog.pg_type")
            .await
            .expect("pg_type query")
            .expect("pg_type result");
        assert!(types.rows.iter().any(|r| r[0] == "23" && r[1] == "int4"));
        assert!(types.rows.iter().any(|r| r[0] == "25" && r[1] == "text"));

        // pg_namespace: fixed system schemas present.
        let namespaces = introspection
            .execute_select("SELECT * FROM pg_catalog.pg_namespace")
            .await
            .expect("pg_namespace query")
            .expect("pg_namespace result");
        assert!(namespaces.rows.iter().any(|r| r[1] == "pg_catalog"));
        assert!(namespaces.rows.iter().any(|r| r[1] == "public"));

        // pg_class: one relation row for the table, relkind 'r', 2 attrs.
        let classes = introspection
            .execute_select("SELECT * FROM pg_catalog.pg_class WHERE relname = 'pg_demo'")
            .await
            .expect("pg_class query")
            .expect("pg_class result");
        assert_eq!(classes.rows.len(), 1);
        assert_eq!(classes.rows[0][1], "pg_demo");
        assert_eq!(classes.rows[0][3], "r");
        assert_eq!(classes.rows[0][4], "2");
        let table_oid = classes.rows[0][0].clone();

        // pg_attribute: 1-based attnum, NOT NULL surfaced as 't', typed by OID.
        let attrs = introspection
            .execute_select("SELECT * FROM pg_catalog.pg_attribute WHERE relname = 'pg_demo'")
            .await
            .expect("pg_attribute query")
            .expect("pg_attribute result");
        let id_attr = attrs
            .rows
            .iter()
            .find(|r| r[1] == "id")
            .expect("id attribute");
        assert_eq!(id_attr[0], table_oid, "attrelid matches pg_class.oid");
        assert_eq!(id_attr[2], "25", "id is text (oid 25)");
        assert_eq!(id_attr[4], "1", "attnum is 1-based");
        assert_eq!(id_attr[5], "t", "NOT NULL → attnotnull 't'");

        // pg_constraint: the primary key surfaces as contype 'p'.
        let constraints = introspection
            .execute_select("SELECT * FROM pg_catalog.pg_constraint")
            .await
            .expect("pg_constraint query")
            .expect("pg_constraint result");
        let pk = constraints
            .rows
            .iter()
            .find(|r| r[1] == "pg_demo_pkey")
            .expect("primary key constraint");
        assert_eq!(pk[2], "p");
        assert_eq!(pk[4], "id");
    }
}
