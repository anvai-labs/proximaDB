//! Databricks Unity Catalog Integration
//!
//! Provides integration with Databricks Unity Catalog for metadata management.
//! Supports Unity's three-level namespace (catalog.schema.table) model.
//!
//! This module is feature-gated behind `unity-catalog` feature.

#![cfg(feature = "unity-catalog")]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::cache::CatalogCache;
use crate::schema::{apply_evolution, validate_schema};
use proximadb_data_model::{ProximaType, TimeUnit, VectorElement};

use crate::{
    Catalog, CatalogColumn, CatalogHealth, CatalogIndex, CatalogNamespace, CatalogSchemaEvolution,
    CatalogTableSchema, CatalogTableStatistics, TableIdentifier,
};

/// Plain Rust configuration for the Databricks Unity catalog.
///
/// Decoupled from `proximadb_proto::proximadb::v1::UnityCatalogConfig` so the
/// workspace contract crate doesn't depend on the heavy proto crate. The
/// network/API layer converts from the proto form when configuring the
/// catalog.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnityCatalogConfig {
    /// Workspace URL, e.g. `https://xxx.cloud.databricks.com`
    pub workspace_url: String,
    /// Personal access token or OAuth token
    pub token: String,
    /// Unity catalog name
    pub catalog_name: String,
    /// Metastore ID (optional)
    pub metastore_id: String,
    /// Use external storage locations
    pub use_external_locations: bool,
}

/// Databricks Unity Catalog implementation
///
/// Maps ProximaDB concepts to Unity:
/// - Catalog name -> Unity Catalog
/// - Namespace[0] -> Unity Schema
/// - Table -> Unity Table
pub struct UnityCatalog {
    /// Catalog name
    name: String,
    /// Configuration
    config: UnityCatalogConfig,
    /// HTTP client for Unity REST API
    http_client: reqwest::Client,
    /// Catalog cache
    cache: Arc<CatalogCache>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UnitySchemaInfo {
    name: String,
    catalog_name: String,
    comment: Option<String>,
    properties: Option<HashMap<String, String>>,
    created_at: Option<i64>,
    updated_at: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UnityTableInfo {
    name: String,
    catalog_name: String,
    schema_name: String,
    table_type: String,
    columns: Vec<UnityColumnInfo>,
    properties: Option<HashMap<String, String>>,
    storage_location: Option<String>,
    created_at: Option<i64>,
    updated_at: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UnityColumnInfo {
    name: String,
    type_name: String,
    type_text: String,
    position: i32,
    nullable: bool,
    comment: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UnityListResponse<T> {
    items: Option<Vec<T>>,
    next_page_token: Option<String>,
}

impl UnityCatalog {
    /// Create a new Unity Catalog
    pub async fn new(
        name: String,
        config: UnityCatalogConfig,
        cache: Arc<CatalogCache>,
    ) -> Result<Self> {
        info!(
            "Initializing Unity Catalog: {} at {}",
            name, config.workspace_url
        );

        // Build HTTP client with auth
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", config.token).parse()?,
        );
        headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse()?);

        let http_client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            name,
            config,
            http_client,
            cache,
        })
    }

    /// Get the Unity API base URL
    fn api_url(&self) -> String {
        format!(
            "{}/api/2.1/unity-catalog",
            self.config.workspace_url.trim_end_matches('/')
        )
    }

    /// Get the Unity catalog name (from config or default)
    fn unity_catalog_name(&self) -> &str {
        if self.config.catalog_name.is_empty() {
            "main"
        } else {
            &self.config.catalog_name
        }
    }

    /// Get the Unity schema name for a namespace
    fn schema_name(&self, namespace: &[String]) -> String {
        if namespace.is_empty() {
            "default".to_string()
        } else {
            namespace[0].clone()
        }
    }

    /// Convert Unity column to CatalogColumn
    fn unity_column_to_column(col: &UnityColumnInfo) -> CatalogColumn {
        let data_type = Self::unity_type_to_data_type(&col.type_name);

        let mut properties = HashMap::new();

        // Parse vector dimension from comment
        if let Some(comment) = &col.comment
            && comment.starts_with("vector:")
            && let Some(dim) = comment
                .strip_prefix("vector:")
                .and_then(|s| s.split(':').next())
            && let Ok(d) = dim.parse::<u32>()
        {
            properties.insert("dimension".to_string(), d.to_string());
        }

        CatalogColumn {
            id: col.position,
            object_id: None,
            name: col.name.clone(),
            data_type,
            nullable: col.nullable,
            default_value: None,
            comment: col.comment.clone(),
            properties,
            is_deleted: false,
            original_id: None,
        }
    }

    /// Convert Unity type to canonical [`ProximaType`].
    fn unity_type_to_data_type(unity_type: &str) -> ProximaType {
        let lower = unity_type.to_lowercase();
        match lower.as_str() {
            "boolean" => ProximaType::Boolean,
            "byte" | "tinyint" => ProximaType::Int32,
            "short" | "smallint" => ProximaType::Int32,
            "int" | "integer" => ProximaType::Int32,
            "long" | "bigint" => ProximaType::Int64,
            "float" | "real" => ProximaType::Float32,
            "double" => ProximaType::Float64,
            "string" => ProximaType::String,
            "binary" => ProximaType::Binary,
            "date" => ProximaType::Date,
            "timestamp" | "timestamp_ntz" => ProximaType::Timestamp(TimeUnit::Nanosecond),
            "decimal" => ProximaType::Decimal {
                precision: 38,
                scale: 10,
            },
            t if t.starts_with("array<float>") || t.contains("vector") => {
                ProximaType::DenseVector {
                    element: VectorElement::Float32,
                    dim: 0,
                }
            }
            t if t.starts_with("map<") => ProximaType::Json,
            t if t.starts_with("struct<") => ProximaType::Json,
            _ => ProximaType::String,
        }
    }

    /// Convert canonical [`ProximaType`] to Unity type
    fn data_type_to_unity_type(
        data_type: &ProximaType,
        _properties: &HashMap<String, String>,
    ) -> String {
        match data_type {
            ProximaType::Boolean => "BOOLEAN".to_string(),
            ProximaType::Int8 | ProximaType::Int16 | ProximaType::Int32 => "INT".to_string(),
            ProximaType::Int64 => "BIGINT".to_string(),
            ProximaType::Float32 => "FLOAT".to_string(),
            ProximaType::Float64 => "DOUBLE".to_string(),
            ProximaType::String => "STRING".to_string(),
            ProximaType::Binary | ProximaType::BinaryVector { .. } => "BINARY".to_string(),
            ProximaType::Date => "DATE".to_string(),
            ProximaType::Timestamp(_) | ProximaType::TimestampTz(_) => "TIMESTAMP".to_string(),
            ProximaType::Decimal { .. } => "DECIMAL(38,18)".to_string(),
            ProximaType::Json | ProximaType::Uuid => "STRING".to_string(), // Unity uses STRING for JSON/UUID
            ProximaType::DenseVector { .. } => {
                "ARRAY<FLOAT>".to_string() // Vector stored as array
            }
            ProximaType::SparseVector { .. } => "MAP<INT,FLOAT>".to_string(),
            _ => "STRING".to_string(),
        }
    }

    /// Make an API request
    async fn api_request<T: for<'de> Deserialize<'de>>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<T> {
        let url = format!("{}{}", self.api_url(), path);

        let mut request = self.http_client.request(method, &url);

        if let Some(b) = body {
            request = request.json(&b);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            return Err(anyhow!("Unity API error ({}): {}", status, error_body));
        }

        let result: T = response.json().await?;
        Ok(result)
    }

    /// Make an API request that returns no content
    async fn api_request_no_content(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<()> {
        let url = format!("{}{}", self.api_url(), path);

        let mut request = self.http_client.request(method, &url);

        if let Some(b) = body {
            request = request.json(&b);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            return Err(anyhow!("Unity API error ({}): {}", status, error_body));
        }

        Ok(())
    }
}

#[async_trait]
impl Catalog for UnityCatalog {
    fn name(&self) -> &str {
        &self.name
    }

    fn catalog_type(&self) -> &str {
        "unity"
    }

    // ========================
    // Namespace Operations
    // ========================

    async fn create_namespace(
        &self,
        namespace: &[String],
        properties: HashMap<String, String>,
    ) -> Result<CatalogNamespace> {
        let schema_name = self.schema_name(namespace);
        let catalog_name = self.unity_catalog_name();

        let body = serde_json::json!({
            "name": schema_name,
            "catalog_name": catalog_name,
            "comment": properties.get("description").cloned(),
            "properties": properties,
        });

        let schema: UnitySchemaInfo = self
            .api_request(reqwest::Method::POST, "/schemas", Some(body))
            .await?;

        info!("Created Unity schema: {}.{}", catalog_name, schema_name);

        Ok(CatalogNamespace {
            levels: namespace.to_vec(),
            properties: schema.properties.unwrap_or_default(),
            owner: None,
            location: None,
            created_at_ms: schema.created_at.unwrap_or(0),
            updated_at_ms: schema.updated_at.unwrap_or(0),
            ..CatalogNamespace::new(Vec::new())
        })
    }

    async fn drop_namespace(&self, namespace: &[String], cascade: bool) -> Result<bool> {
        let schema_name = self.schema_name(namespace);
        let catalog_name = self.unity_catalog_name();

        if !cascade {
            // Check if schema has tables
            let tables = self.list_tables(namespace).await?;
            if !tables.is_empty() {
                return Err(anyhow!(
                    "Schema '{}.{}' is not empty. Use cascade=true to force drop.",
                    catalog_name,
                    schema_name
                ));
            }
        }

        // Delete all tables if cascade
        if cascade {
            let tables = self.list_tables(namespace).await?;
            for table_id in tables {
                self.drop_table(&table_id, true).await?;
            }
        }

        let path = format!("/schemas/{}.{}", catalog_name, schema_name);

        self.api_request_no_content(reqwest::Method::DELETE, &path, None)
            .await?;

        info!("Deleted Unity schema: {}.{}", catalog_name, schema_name);
        Ok(true)
    }

    async fn list_namespaces(&self, _parent: Option<&[String]>) -> Result<Vec<CatalogNamespace>> {
        let catalog_name = self.unity_catalog_name();
        let mut all_namespaces = Vec::new();
        let mut next_page_token = None;

        loop {
            let mut path = format!("/schemas?catalog_name={}", catalog_name);
            if let Some(token) = &next_page_token {
                path.push_str(&format!("&page_token={}", token));
            }

            let response: UnityListResponse<UnitySchemaInfo> =
                self.api_request(reqwest::Method::GET, &path, None).await?;

            if let Some(items) = response.items {
                all_namespaces.extend(items.into_iter().map(|s| CatalogNamespace {
                    levels: vec![s.name],
                    properties: s.properties.unwrap_or_default(),
                    owner: None,
                    location: None,
                    created_at_ms: s.created_at.unwrap_or(0),
                    updated_at_ms: s.updated_at.unwrap_or(0),
                    ..CatalogNamespace::new(Vec::new())
                }));
            }

            next_page_token = response.next_page_token;
            if next_page_token.is_none() {
                break;
            }
        }

        Ok(all_namespaces)
    }

    async fn namespace_exists(&self, namespace: &[String]) -> Result<bool> {
        let schema_name = self.schema_name(namespace);
        let catalog_name = self.unity_catalog_name();

        let path = format!("/schemas/{}.{}", catalog_name, schema_name);

        match self
            .api_request::<UnitySchemaInfo>(reqwest::Method::GET, &path, None)
            .await
        {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn get_namespace(&self, namespace: &[String]) -> Result<CatalogNamespace> {
        let schema_name = self.schema_name(namespace);
        let catalog_name = self.unity_catalog_name();

        let path = format!("/schemas/{}.{}", catalog_name, schema_name);

        let schema: UnitySchemaInfo = self.api_request(reqwest::Method::GET, &path, None).await?;

        Ok(CatalogNamespace {
            levels: namespace.to_vec(),
            properties: schema.properties.unwrap_or_default(),
            owner: None,
            location: None,
            created_at_ms: schema.created_at.unwrap_or(0),
            updated_at_ms: schema.updated_at.unwrap_or(0),
            ..CatalogNamespace::new(Vec::new())
        })
    }

    async fn update_namespace_properties(
        &self,
        namespace: &[String],
        updates: HashMap<String, String>,
        _removals: Vec<String>,
    ) -> Result<()> {
        let schema_name = self.schema_name(namespace);
        let catalog_name = self.unity_catalog_name();

        let path = format!("/schemas/{}.{}", catalog_name, schema_name);

        let body = serde_json::json!({
            "properties": updates,
        });

        self.api_request_no_content(reqwest::Method::PATCH, &path, Some(body))
            .await?;

        Ok(())
    }

    // ========================
    // Table Operations
    // ========================

    async fn create_table_inner(
        &self,
        identifier: &TableIdentifier,
        schema: CatalogTableSchema,
    ) -> Result<CatalogTableSchema> {
        // ADR-077 M1: normalize BEFORE validating. A creation path may legitimately
        // hand us a UNIQUE recorded only in the projection — `CREATE TABLE … UNIQUE(x)`
        // lowers to `relational_capabilities.unique_indexes` — and rejecting that would
        // break table creation. Folding it into the canonical field first is what makes
        // the invariant hold by construction rather than by rejecting real schemas.
        let mut schema = schema;
        crate::schema::normalize_identity(&mut schema);
        validate_schema(&schema)?;

        let catalog_name = self.unity_catalog_name();
        let schema_name = self.schema_name(&identifier.namespace);

        // Convert columns
        let columns: Vec<serde_json::Value> = schema
            .columns
            .iter()
            .enumerate()
            .map(|(pos, col)| {
                let type_name = Self::data_type_to_unity_type(&col.data_type, &col.properties);

                let comment = if matches!(col.data_type, ProximaType::DenseVector { .. }) {
                    Some(format!(
                        "vector:{}:metric={}",
                        col.properties.get("dimension").unwrap_or(&"0".to_string()),
                        col.properties
                            .get("metric")
                            .unwrap_or(&"cosine".to_string())
                    ))
                } else {
                    col.comment.clone()
                };

                serde_json::json!({
                    "name": col.name,
                    "type_name": type_name,
                    "type_text": type_name,
                    "position": pos,
                    "nullable": col.nullable,
                    "comment": comment,
                })
            })
            .collect();

        let mut properties = schema.properties.clone();
        properties.insert("proximadb.version".to_string(), "1".to_string());
        properties.insert(
            "proximadb.schema_version".to_string(),
            schema.schema_version.to_string(),
        );

        let body = serde_json::json!({
            "name": identifier.name,
            "catalog_name": catalog_name,
            "schema_name": schema_name,
            "table_type": "EXTERNAL",
            "columns": columns,
            "properties": properties,
        });

        let _table: UnityTableInfo = self
            .api_request(reqwest::Method::POST, "/tables", Some(body))
            .await?;

        info!(
            "Created Unity table: {}.{}.{}",
            catalog_name, schema_name, identifier.name
        );

        Ok(schema)
    }

    async fn drop_table(&self, identifier: &TableIdentifier, _purge: bool) -> Result<bool> {
        let catalog_name = self.unity_catalog_name();
        let schema_name = self.schema_name(&identifier.namespace);

        let path = format!(
            "/tables/{}.{}.{}",
            catalog_name, schema_name, identifier.name
        );

        self.api_request_no_content(reqwest::Method::DELETE, &path, None)
            .await?;

        // Invalidate cache
        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);

        info!(
            "Deleted Unity table: {}.{}.{}",
            catalog_name, schema_name, identifier.name
        );

        Ok(true)
    }

    async fn list_tables(&self, namespace: &[String]) -> Result<Vec<TableIdentifier>> {
        let catalog_name = self.unity_catalog_name();
        let schema_name = self.schema_name(namespace);

        let mut all_tables = Vec::new();
        let mut next_page_token = None;

        loop {
            let mut path = format!(
                "/tables?catalog_name={}&schema_name={}",
                catalog_name, schema_name
            );
            if let Some(token) = &next_page_token {
                path.push_str(&format!("&page_token={}", token));
            }

            let response: UnityListResponse<UnityTableInfo> =
                self.api_request(reqwest::Method::GET, &path, None).await?;

            if let Some(items) = response.items {
                for t in items {
                    all_tables.push(TableIdentifier::new(namespace.to_vec(), t.name));
                }
            }

            next_page_token = response.next_page_token;
            if next_page_token.is_none() {
                break;
            }
        }

        Ok(all_tables)
    }

    async fn table_exists(&self, identifier: &TableIdentifier) -> Result<bool> {
        let catalog_name = self.unity_catalog_name();
        let schema_name = self.schema_name(&identifier.namespace);

        let path = format!(
            "/tables/{}.{}.{}",
            catalog_name, schema_name, identifier.name
        );

        match self
            .api_request::<UnityTableInfo>(reqwest::Method::GET, &path, None)
            .await
        {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn get_table(&self, identifier: &TableIdentifier) -> Result<CatalogTableSchema> {
        crate::cached_table(&self.cache, &self.name, identifier, async {
            let catalog_name = self.unity_catalog_name();
            let schema_name = self.schema_name(&identifier.namespace);

            let path = format!(
                "/tables/{}.{}.{}",
                catalog_name, schema_name, identifier.name
            );

            let table: UnityTableInfo = self.api_request(reqwest::Method::GET, &path, None).await?;

            let columns: Vec<CatalogColumn> = table
                .columns
                .iter()
                .map(Self::unity_column_to_column)
                .collect();

            let schema_version = table
                .properties
                .as_ref()
                .and_then(|p| p.get("proximadb.schema_version"))
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);

            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64;

            Ok(CatalogTableSchema {
                name: identifier.name.clone(),
                columns,
                primary_key: vec![],
                indexes: vec![],
                schema_version,
                properties: table.properties.unwrap_or_default(),
                location: table.storage_location,
                created_at_ms: table.created_at.unwrap_or(now),
                updated_at_ms: table.updated_at.unwrap_or(now),
                ..Default::default()
            })
        })
        .await
    }

    async fn rename_table(&self, from: &TableIdentifier, to: &TableIdentifier) -> Result<()> {
        // Unity doesn't support direct rename across schemas
        // For same schema, use PATCH
        let catalog_name = self.unity_catalog_name();
        let from_schema = self.schema_name(&from.namespace);
        let to_schema = self.schema_name(&to.namespace);

        if from_schema != to_schema {
            // Cross-schema rename: copy and delete
            let schema = self.get_table(from).await?;
            self.create_table(to, schema).await?;
            self.drop_table(from, false).await?;
        } else {
            // Same schema: use PATCH
            let path = format!("/tables/{}.{}.{}", catalog_name, from_schema, from.name);

            let body = serde_json::json!({
                "name": to.name,
            });

            self.api_request_no_content(reqwest::Method::PATCH, &path, Some(body))
                .await?;
        }

        // Invalidate cache
        self.cache.invalidate_table_in_catalog(&self.name, from);

        info!("Renamed Unity table: {} -> {}", from, to);
        Ok(())
    }

    // ========================
    // Schema Evolution
    // ========================

    async fn evolve_schema(
        &self,
        identifier: &TableIdentifier,
        evolution: CatalogSchemaEvolution,
    ) -> Result<CatalogTableSchema> {
        let current_schema = self.get_table(identifier).await?;
        let new_schema = apply_evolution(&current_schema, &evolution)?;

        // Update table columns
        let catalog_name = self.unity_catalog_name();
        let schema_name = self.schema_name(&identifier.namespace);

        let columns: Vec<serde_json::Value> = new_schema
            .columns
            .iter()
            .enumerate()
            .map(|(pos, col)| {
                let type_name = Self::data_type_to_unity_type(&col.data_type, &col.properties);

                serde_json::json!({
                    "name": col.name,
                    "type_name": type_name,
                    "type_text": type_name,
                    "position": pos,
                    "nullable": col.nullable,
                    "comment": col.comment.clone().unwrap_or_default(),
                })
            })
            .collect();

        let path = format!(
            "/tables/{}.{}.{}",
            catalog_name, schema_name, identifier.name
        );

        let mut properties = new_schema.properties.clone();
        properties.insert(
            "proximadb.schema_version".to_string(),
            new_schema.schema_version.to_string(),
        );

        let body = serde_json::json!({
            "columns": columns,
            "properties": properties,
        });

        self.api_request_no_content(reqwest::Method::PATCH, &path, Some(body))
            .await?;

        // Invalidate cache
        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);

        info!(
            "Evolved Unity table schema: {} (v{})",
            identifier, new_schema.schema_version
        );

        Ok(new_schema)
    }

    async fn get_schema_version(&self, identifier: &TableIdentifier) -> Result<i32> {
        let schema = self.get_table(identifier).await?;
        Ok(schema.schema_version)
    }

    async fn get_schema_by_version(
        &self,
        identifier: &TableIdentifier,
        version: i32,
    ) -> Result<CatalogTableSchema> {
        // Unity doesn't keep historical versions
        let schema = self.get_table(identifier).await?;
        if schema.schema_version == version {
            Ok(schema)
        } else {
            Err(anyhow!(
                "Historical schema version {} not available for '{}' (current: {})",
                version,
                identifier,
                schema.schema_version
            ))
        }
    }

    // ========================
    // Index Operations
    // ========================

    async fn create_index(
        &self,
        _identifier: &TableIdentifier,
        index: CatalogIndex,
    ) -> Result<CatalogIndex> {
        // Unity doesn't have native index support
        // Store as table property
        warn!("Unity Catalog: indexes stored as metadata only, not enforced");
        Ok(index)
    }

    async fn drop_index(&self, _identifier: &TableIdentifier, _index_name: &str) -> Result<bool> {
        // No-op for Unity
        Ok(true)
    }

    async fn list_indexes(&self, identifier: &TableIdentifier) -> Result<Vec<CatalogIndex>> {
        let schema = self.get_table(identifier).await?;
        Ok(schema.indexes)
    }

    // ========================
    // Statistics
    // ========================

    async fn get_statistics(&self, identifier: &TableIdentifier) -> Result<CatalogTableStatistics> {
        if let Some(stats) = self.cache.get_statistics(&self.name, identifier) {
            return Ok(stats);
        }

        let stats = CatalogTableStatistics::default();
        self.cache
            .put_statistics(&self.name, identifier, stats.clone());
        Ok(stats)
    }

    async fn update_statistics(
        &self,
        identifier: &TableIdentifier,
        stats: CatalogTableStatistics,
    ) -> Result<()> {
        self.cache.put_statistics(&self.name, identifier, stats);
        Ok(())
    }

    // ========================
    // Health & Connectivity
    // ========================

    async fn health_check(&self) -> Result<CatalogHealth> {
        let start = Instant::now();

        match self
            .api_request::<serde_json::Value>(reqwest::Method::GET, "/catalogs", None)
            .await
        {
            Ok(_) => {
                let latency = start.elapsed().as_millis() as u64;
                Ok(CatalogHealth::healthy(latency)
                    .with_detail("workspace_url", &self.config.workspace_url)
                    .with_detail("catalog_name", self.unity_catalog_name())
                    .with_detail("catalog_type", "unity"))
            }
            Err(e) => Ok(CatalogHealth::unhealthy(e.to_string())),
        }
    }

    async fn close(&self) -> Result<()> {
        debug!("Closing Unity Catalog: {}", self.name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unity_type_conversion() {
        assert_eq!(
            UnityCatalog::unity_type_to_data_type("BIGINT"),
            ProximaType::Int64
        );
        assert_eq!(
            UnityCatalog::unity_type_to_data_type("STRING"),
            ProximaType::String
        );
        assert_eq!(
            UnityCatalog::unity_type_to_data_type("ARRAY<FLOAT>"),
            ProximaType::DenseVector {
                element: VectorElement::Float32,
                dim: 0
            }
        );
    }

    #[test]
    fn test_data_type_to_unity() {
        let props = HashMap::new();
        assert_eq!(
            UnityCatalog::data_type_to_unity_type(&ProximaType::Int64, &props),
            "BIGINT"
        );
        assert_eq!(
            UnityCatalog::data_type_to_unity_type(&ProximaType::String, &props),
            "STRING"
        );
        assert_eq!(
            UnityCatalog::data_type_to_unity_type(
                &ProximaType::DenseVector {
                    element: VectorElement::Float32,
                    dim: 0,
                },
                &props
            ),
            "ARRAY<FLOAT>"
        );
    }

    #[test]
    fn test_unity_column_to_column() {
        let unity_col = UnityColumnInfo {
            name: "embedding".to_string(),
            type_name: "ARRAY<FLOAT>".to_string(),
            type_text: "ARRAY<FLOAT>".to_string(),
            position: 0,
            nullable: true,
            comment: Some("vector:768:metric=cosine".to_string()),
        };

        let col = UnityCatalog::unity_column_to_column(&unity_col);
        assert_eq!(col.name, "embedding");
        assert_eq!(col.properties.get("dimension"), Some(&"768".to_string()));
    }
}
