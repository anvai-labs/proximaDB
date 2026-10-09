//! HTTP/REST client for ProximaDB
//!
//! This module provides the `ProximaClient` for connecting to a remote
//! ProximaDB server over HTTP/REST.

use crate::collection::CollectionHandle;
use crate::error::{ConfigError, NetworkError, ProximaError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Configuration for the ProximaDB client
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Server URL (e.g., "http://localhost:5678")
    pub url: String,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Number of retries for failed requests
    pub max_retries: u32,
    /// Optional API key for authentication
    pub api_key: Option<String>,
    /// Enable connection pooling
    pub pool_connections: bool,
    /// Maximum idle connections in pool
    pub max_idle_connections: usize,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:5678".to_string(),
            timeout_ms: 30000,
            max_retries: 3,
            api_key: None,
            pool_connections: true,
            max_idle_connections: 10,
        }
    }
}

/// Builder for creating a ProximaClient
#[derive(Debug, Clone)]
pub struct ClientBuilder {
    config: ClientConfig,
}

impl ClientBuilder {
    /// Create a new client builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ClientConfig::default(),
        }
    }

    /// Set the server URL
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.config.url = url.into();
        self
    }

    /// Set the request timeout in milliseconds
    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.config.timeout_ms = timeout_ms;
        self
    }

    /// Set the maximum number of retries
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.config.max_retries = max_retries;
        self
    }

    /// Set the API key for authentication
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.config.api_key = Some(api_key.into());
        self
    }

    /// Enable or disable connection pooling
    pub fn pool_connections(mut self, enable: bool) -> Self {
        self.config.pool_connections = enable;
        self
    }

    /// Set maximum idle connections in pool
    pub fn max_idle_connections(mut self, max: usize) -> Self {
        self.config.max_idle_connections = max;
        self
    }

    /// Build the ProximaClient
    pub fn build(self) -> Result<ProximaClient> {
        ProximaClient::with_config(self.config)
    }

    /// Connect to the server (alias for build)
    pub fn connect(self) -> Result<ProximaClient> {
        self.build()
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// HTTP/REST client for ProximaDB
///
/// Provides methods for connecting to a remote ProximaDB server and
/// performing vector database operations.
///
/// # Example
///
/// ```rust,ignore
/// use proximadb_sdk::ProximaClient;
///
/// let client = ProximaClient::connect("http://localhost:5678")?;
///
/// // Create a collection
/// client.create_collection("embeddings")
///     .dimension(768)
///     .engine("sst")
///     .execute()?;
///
/// // Search with fluent API
/// let results = client.collection("embeddings")
///     .search()
///     .vector(&query_vector)
///     .top_k(10)
///     .execute()?;
/// ```
#[derive(Clone)]
pub struct ProximaClient {
    inner: Arc<ProximaClientInner>,
}

struct ProximaClientInner {
    config: ClientConfig,
    /// Generated REST transport (TD-126 Phase 4). The facade owns the
    /// `reqwest::Client` configuration, bearer auth, error mapping, and the
    /// permissive response DTOs; the generated `genrest::Client` owns the wire
    /// plumbing (the configured `reqwest::Client` + base URL). This mirrors the
    /// Go pilot, where the facade calls into the generated raw `Client`. The
    /// transport helpers below route through `transport.client()` / `transport.baseurl()`.
    #[cfg(feature = "client")]
    transport: crate::genrest::Client,
}

impl ProximaClient {
    /// Connect to a ProximaDB server at the given URL
    pub fn connect(url: impl Into<String>) -> Result<Self> {
        ClientBuilder::new().url(url).build()
    }

    /// Create a client with custom configuration
    pub fn with_config(config: ClientConfig) -> Result<Self> {
        // Validate URL
        let _parsed_url = url::Url::parse(&config.url).map_err(|_| {
            ProximaError::Config(ConfigError::InvalidValue {
                field: "url".to_string(),
                reason: format!("Invalid URL: {}", config.url),
            })
        })?;

        #[cfg(feature = "client")]
        let transport = {
            let mut builder = reqwest::Client::builder()
                .timeout(Duration::from_millis(config.timeout_ms))
                .pool_max_idle_per_host(config.max_idle_connections);

            if !config.pool_connections {
                builder = builder.pool_max_idle_per_host(0);
            }

            let http_client = builder.build().map_err(|e| {
                ProximaError::Network(NetworkError::ConnectionFailed {
                    url: config.url.clone(),
                    reason: e.to_string(),
                })
            })?;

            // Hand the configured reqwest client to the generated transport.
            crate::genrest::Client::new_with_client(&config.url, http_client)
        };

        Ok(Self {
            inner: Arc::new(ProximaClientInner {
                config,
                #[cfg(feature = "client")]
                transport,
            }),
        })
    }

    /// Get a builder for creating a client
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Get a handle to a collection for fluent operations
    pub fn collection(&self, name: &str) -> CollectionHandle<'_> {
        CollectionHandle::new(self, name)
    }

    /// Get the server URL
    pub fn url(&self) -> &str {
        &self.inner.config.url
    }

    /// Get the client configuration
    pub fn config(&self) -> &ClientConfig {
        &self.inner.config
    }

    /// Check if the server is healthy
    #[cfg(feature = "client")]
    pub async fn health(&self) -> Result<HealthStatus> {
        let url = format!("{}/health", self.inner.config.url);
        let response = self.get(&url).await?;
        Ok(response)
    }

    /// Kubernetes liveness probe — `GET /health/live`. Mirrors the OpenAPI
    /// `getLiveness` operation and the Python SDK `.live()` method.
    #[cfg(feature = "client")]
    pub async fn health_live(&self) -> Result<ProbeStatus> {
        let url = format!("{}/health/live", self.inner.config.url);
        self.get(&url).await
    }

    /// Kubernetes readiness probe — `GET /health/ready`. Mirrors the OpenAPI
    /// `getReadiness` operation and the Python SDK `.ready()` method.
    #[cfg(feature = "client")]
    pub async fn health_ready(&self) -> Result<ProbeStatus> {
        let url = format!("{}/health/ready", self.inner.config.url);
        self.get(&url).await
    }

    /// Get a collection's schema — `GET /api/v2/collections/{id}/schema`.
    /// Mirrors the OpenAPI `getCollectionSchema` operation.
    #[cfg(feature = "client")]
    pub async fn get_collection_schema(&self, collection_id: &str) -> Result<SchemaResponse> {
        let url = format!(
            "{}/api/v2/collections/{}/schema",
            self.inner.config.url, collection_id
        );
        self.get(&url).await
    }

    /// Update a collection's schema — `PUT /api/v2/collections/{id}/schema`.
    /// Mirrors the OpenAPI `updateCollectionSchema` operation. The request
    /// body is the [`SchemaDefinition`] extended with an optional `force`
    /// flag (see [`UpdateSchemaRequest`]).
    #[cfg(feature = "client")]
    pub async fn update_collection_schema(
        &self,
        collection_id: &str,
        body: &UpdateSchemaRequest,
    ) -> Result<UpdateSchemaResponse> {
        let url = format!(
            "{}/api/v2/collections/{}/schema",
            self.inner.config.url, collection_id
        );
        self.put(&url, body).await
    }

    /// Execute an AQL/UQL/federated query — `POST /api/v2/query`. Mirrors
    /// the OpenAPI `executeQuery` operation. The shared query facade lowers
    /// the textual query through ProximaDB's logical query layer; SQL still
    /// lands on pgwire.
    #[cfg(feature = "client")]
    pub async fn execute_query(&self, req: &QueryRequest) -> Result<serde_json::Value> {
        let url = format!("{}/api/v2/query", self.inner.config.url);
        self.post(&url, req).await
    }

    /// Explain an AQL/UQL query — `POST /api/v2/query/explain`. Mirrors the
    /// OpenAPI `explainQuery` operation. Returns the plan and lowering
    /// details as a free-form JSON document (the OpenAPI `QueryResponse`
    /// schema is `additionalProperties: true`).
    #[cfg(feature = "client")]
    pub async fn explain_query(&self, req: &ExplainQueryRequest) -> Result<serde_json::Value> {
        let url = format!("{}/api/v2/query/explain", self.inner.config.url);
        self.post(&url, req).await
    }

    /// Create a collection builder
    pub fn create_collection(&self, name: &str) -> crate::collection::CollectionBuilder<'_> {
        crate::collection::CollectionBuilder::new(self, name)
    }

    /// Delete a collection
    #[cfg(feature = "client")]
    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        let url = format!("{}/api/v2/collections/{}", self.inner.config.url, name);
        self.delete::<serde_json::Value>(&url).await?;
        Ok(())
    }

    /// Get a handle to a graph for fluent operations
    #[cfg(feature = "client")]
    pub fn graph(&self, name: &str) -> crate::graph::GraphHandle<'_> {
        crate::graph::GraphHandle::new(self, name)
    }

    /// Create a graph builder
    #[cfg(feature = "client")]
    pub fn create_graph(&self, name: &str) -> crate::graph::GraphBuilder<'_> {
        crate::graph::GraphBuilder::new(self, name)
    }

    /// Delete a graph via the generated `delete_graph` operation's
    /// path/verb (`DELETE /api/v2/graphs/{graph_id}`), through the shared
    /// generated transport connection (see `graph::GraphBuilder::execute`
    /// for why the fully-typed builder is not used).
    #[cfg(feature = "client")]
    pub async fn delete_graph(&self, name: &str) -> Result<()> {
        let url = format!("{}/api/v2/graphs/{}", self.inner.config.url, name);
        self.delete::<serde_json::Value>(&url).await?;
        Ok(())
    }

    /// List all graphs (`GET /api/v2/graphs`, the generated `list_graphs`
    /// operation's path/verb).
    ///
    /// Kept on the facade's permissive `GraphInfo` DTO + hand-formatted path
    /// rather than the typed `list_graphs().send()`: the facade `GraphInfo`
    /// exposes a required `name` plus node/edge counts, while the generated
    /// `GraphCollectionResponse` makes every field optional and names them
    /// differently (`graph_id`), so the typed path would need a lossy
    /// field-by-field lowering. The wire connection flows through the shared
    /// transport via `self.get`.
    #[cfg(feature = "client")]
    pub async fn list_graphs(&self) -> Result<Vec<GraphInfo>> {
        let url = format!("{}/api/v2/graphs", self.inner.config.url);
        let envelope: GraphListEnvelope = self.get(&url).await?;
        Ok(envelope.data.into_iter().map(GraphInfo::from).collect())
    }

    /// List all collections
    #[cfg(feature = "client")]
    pub async fn list_collections(&self) -> Result<Vec<CollectionInfo>> {
        let url = format!("{}/api/v2/collections", self.inner.config.url);
        let response: ListCollectionsResponse = self.get(&url).await?;
        Ok(response.collections)
    }

    // Internal HTTP methods.
    //
    // These route through the generated transport's configured `reqwest::Client`
    // (`self.inner.transport.client()`, exposed via the progenitor `ClientInfo` trait).
    // The generated client owns the wire plumbing (reqwest config + base URL);
    // the facade keeps the permissive serde DTOs, bearer auth, and error mapping
    // — the same division of labour as the Go pilot's facade over the raw
    // generated `Client`.

    /// Borrow the generated transport's configured reqwest client.
    #[cfg(feature = "client")]
    pub(crate) fn http(&self) -> &reqwest::Client {
        use crate::genrest::ClientInfo;
        self.inner.transport.client()
    }

    #[cfg(feature = "client")]
    pub(crate) async fn get<T: for<'de> Deserialize<'de>>(&self, url: &str) -> Result<T> {
        let mut request = self.http().get(url);

        if let Some(ref api_key) = self.inner.config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = self.send_with_retries(request).await?;
        self.handle_response(response).await
    }

    #[cfg(feature = "client")]
    pub(crate) async fn post<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        url: &str,
        body: &B,
    ) -> Result<T> {
        let mut request = self.http().post(url).json(body);

        if let Some(ref api_key) = self.inner.config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = self.send_with_retries(request).await?;
        self.handle_response(response).await
    }

    #[cfg(feature = "client")]
    pub(crate) async fn put<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        url: &str,
        body: &B,
    ) -> Result<T> {
        let mut request = self.http().put(url).json(body);

        if let Some(ref api_key) = self.inner.config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = self.send_with_retries(request).await?;
        self.handle_response(response).await
    }

    #[cfg(feature = "client")]
    pub(crate) async fn delete<T: for<'de> Deserialize<'de>>(&self, url: &str) -> Result<T> {
        let mut request = self.http().delete(url);

        if let Some(ref api_key) = self.inner.config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = self.send_with_retries(request).await?;
        self.handle_response(response).await
    }

    #[cfg(feature = "client")]
    async fn send_with_retries(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response> {
        let max_retries = self.inner.config.max_retries;
        let mut attempts = 0;
        let mut next_request = Some(request);

        loop {
            let Some(request) = next_request.take() else {
                return Err(ProximaError::Network(NetworkError::ConnectionFailed {
                    url: self.inner.config.url.clone(),
                    reason: "request body could not be cloned for retry".to_string(),
                }));
            };
            let retry_request = request.try_clone();

            match request.send().await {
                Ok(response)
                    if attempts < max_retries
                        && retry_request.is_some()
                        && Self::is_retryable_status(response.status()) =>
                {
                    attempts += 1;
                    Self::sleep_before_retry(attempts).await;
                    next_request = retry_request;
                }
                Ok(response) => return Ok(response),
                Err(err)
                    if attempts < max_retries
                        && retry_request.is_some()
                        && Self::is_retryable_error(&err) =>
                {
                    attempts += 1;
                    Self::sleep_before_retry(attempts).await;
                    next_request = retry_request;
                }
                Err(err) => return Err(err.into()),
            }
        }
    }

    #[cfg(feature = "client")]
    fn is_retryable_status(status: reqwest::StatusCode) -> bool {
        status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
    }

    #[cfg(feature = "client")]
    fn is_retryable_error(err: &reqwest::Error) -> bool {
        err.is_timeout() || err.is_connect() || err.is_request()
    }

    #[cfg(feature = "client")]
    async fn sleep_before_retry(attempt: u32) {
        let delay_ms = 25u64.saturating_mul(1u64 << attempt.saturating_sub(1).min(6));
        tokio::time::sleep(Duration::from_millis(delay_ms.min(1_000))).await;
    }

    #[cfg(feature = "client")]
    async fn handle_response<T: for<'de> Deserialize<'de>>(
        &self,
        response: reqwest::Response,
    ) -> Result<T> {
        let status = response.status();

        if status.is_success() {
            let body = response.json::<T>().await.map_err(|e| {
                ProximaError::Network(NetworkError::Deserialization {
                    reason: e.to_string(),
                })
            })?;
            Ok(body)
        } else {
            let status_code = status.as_u16();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            match status_code {
                401 => Err(ProximaError::Network(NetworkError::AuthenticationFailed {
                    reason: message,
                })),
                429 => Err(ProximaError::Network(NetworkError::RateLimited {
                    retry_after_ms: 1000,
                })),
                _ => Err(ProximaError::Network(NetworkError::HttpError {
                    status: status_code,
                    message,
                })),
            }
        }
    }
}

impl std::fmt::Debug for ProximaClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProximaClient")
            .field("url", &self.inner.config.url)
            .field("timeout_ms", &self.inner.config.timeout_ms)
            .finish()
    }
}

#[cfg(all(test, feature = "client"))]
impl ProximaClient {
    pub(crate) fn for_tests(url: &str) -> Self {
        let config = ClientConfig {
            url: url.to_string(),
            ..ClientConfig::default()
        };
        let http_client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(config.timeout_ms))
            .pool_max_idle_per_host(config.max_idle_connections)
            .build()
            .expect("test client should build without system proxy lookup");
        let transport = crate::genrest::Client::new_with_client(&config.url, http_client);

        Self {
            inner: Arc::new(ProximaClientInner { config, transport }),
        }
    }
}

/// Health status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Server status
    pub status: String,
    /// Server version
    #[serde(default)]
    pub version: Option<String>,
    /// Uptime in seconds
    #[serde(default)]
    pub uptime_seconds: Option<u64>,
}

/// Collection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    /// Collection ID
    #[serde(default)]
    pub collection_id: Option<String>,
    /// Collection name
    pub name: String,
    /// Vector dimension
    pub dimension: u32,
    /// Number of vectors. The v2 list surface
    /// (`CollectionV2Summary.record_count`) emits `null` when
    /// `include_stats=false`, so tolerate an explicit JSON `null` (→ 0)
    /// in addition to an absent field.
    #[serde(
        default,
        alias = "record_count",
        deserialize_with = "crate::serde_compat::null_as_default"
    )]
    pub vector_count: u64,
    /// Storage engine type
    #[serde(default)]
    pub engine: Option<String>,
    /// Nested v2 collection statistics.
    #[serde(default)]
    pub stats: Option<CollectionStats>,
}

/// v2 collection statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    /// Total number of records.
    #[serde(default)]
    pub record_count: u64,
    /// Total storage size in bytes.
    #[serde(default)]
    pub storage_size_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct ListCollectionsResponse {
    collections: Vec<CollectionInfo>,
}

/// Graph information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphInfo {
    /// Graph name
    pub name: String,
    /// Number of nodes
    #[serde(default)]
    pub node_count: u64,
    /// Number of edges
    #[serde(default)]
    pub edge_count: u64,
    /// Graph description
    #[serde(default)]
    pub description: Option<String>,
}

/// The wave-6 envelope truth: `GET /api/v2/graphs` returns
/// `{success, data: [...]}` (the old flat `{graphs: [...]}` shape never
/// existed on the wire — the pre-wave-6 published spec was wrong).
#[derive(Debug, Deserialize)]
struct GraphListEnvelope {
    data: Vec<GraphRawInfo>,
}

/// One serialized graph-collection record (open field set; the facade
/// lowers it best-effort into [`GraphInfo`]).
#[derive(Debug, Deserialize)]
struct GraphRawInfo {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    graph_id: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    node_count: Option<u64>,
    #[serde(default)]
    edge_count: Option<u64>,
    #[serde(default)]
    stats: Option<GraphRawStats>,
}

#[derive(Debug, Deserialize)]
struct GraphRawStats {
    #[serde(default)]
    total_nodes: Option<u64>,
    #[serde(default)]
    total_edges: Option<u64>,
}

impl From<GraphRawInfo> for GraphInfo {
    fn from(raw: GraphRawInfo) -> Self {
        let stats = raw.stats.as_ref();
        Self {
            name: raw
                .name
                .filter(|name| !name.trim().is_empty())
                .or(raw.graph_id)
                .unwrap_or_default(),
            node_count: raw
                .node_count
                .or_else(|| stats.and_then(|value| value.total_nodes))
                .unwrap_or(0),
            edge_count: raw
                .edge_count
                .or_else(|| stats.and_then(|value| value.total_edges))
                .unwrap_or(0),
            description: raw.description,
        }
    }
}

/// Liveness / readiness probe payload.
///
/// Mirrors the OpenAPI `ProbeResponse` schema returned from
/// `/health/live` and `/health/ready`. `status` is the only required
/// field; servers may add additional diagnostic fields, which the SDK
/// preserves via `extra` for forward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeStatus {
    /// Probe status string (typically `"ok"`, `"ready"`, or
    /// `"not_ready"`).
    pub status: String,
    /// Forward-compat capture of any additional fields the server may
    /// emit. The OpenAPI schema only declares `status`, but new
    /// diagnostic fields shouldn't break older SDKs.
    #[serde(flatten, default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Column declaration within a [`SchemaDefinition`].
///
/// Mirrors the OpenAPI `ColumnDefinition` schema. Field names and
/// allowed `data_type` values match the spec exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDefinition {
    /// Column name.
    pub name: String,
    /// One of the OpenAPI-declared data types (`text`, `integer`,
    /// `float`, `boolean`, `timestamp`, `vector`, etc.). Strings are
    /// passed through verbatim; validation lives on the server side.
    pub data_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nullable: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filterable: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_dimension: Option<u32>,
}

/// Collection schema definition.
///
/// Mirrors the OpenAPI `SchemaDefinition` schema. Used as the body of
/// schema-update requests and as the nested `schema` payload returned
/// from [`SchemaResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDefinition {
    pub columns: Vec<ColumnDefinition>,
    /// One of `strict`, `flexible`, `hybrid`. Defaults to `hybrid`
    /// server-side; omitted from the wire payload when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enforcement: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_additional_fields: Option<bool>,
}

/// Response from `GET /api/v2/collections/{id}/schema`.
///
/// Mirrors the OpenAPI `SchemaResponse` schema. Forward-compat fields
/// are captured in `extra` so the SDK keeps deserialising successfully
/// if the server starts emitting additional diagnostic fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaResponse {
    pub schema_id: String,
    pub schema_version: String,
    pub collection_id: String,
    pub schema: SchemaDefinition,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_schema_id: Option<String>,
    #[serde(flatten, default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Request body for `PUT /api/v2/collections/{id}/schema`.
///
/// Mirrors the OpenAPI `UpdateSchemaRequest` schema, which is
/// `SchemaDefinition` extended with an optional `force` flag. Flattened
/// serialization so the wire payload matches the spec's `allOf` shape
/// (columns/enforcement/allow_additional_fields at the top level, not
/// nested under `schema`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateSchemaRequest {
    #[serde(flatten)]
    pub schema: SchemaDefinition,
    /// When `true`, server bypasses backward-compatibility checks for
    /// breaking schema changes. Defaults to `false` server-side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,
}

/// Response from `PUT /api/v2/collections/{id}/schema`.
///
/// Mirrors the OpenAPI `UpdateSchemaResponse` schema. `changes` is
/// `Vec<serde_json::Value>` because the OpenAPI spec declares each
/// change entry as a free-form object with `additionalProperties: true`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateSchemaResponse {
    pub schema_id: String,
    pub schema_version: String,
    pub previous_schema_id: String,
    pub changes: Vec<serde_json::Value>,
    pub warnings: Vec<String>,
    pub updated_at: String,
    #[serde(flatten, default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Request body for `POST /api/v2/query`.
///
/// Mirrors the OpenAPI `QueryRequest` schema. `language` is one of
/// `uql`, `aql`, `federated`. `parameters` is left as
/// `Vec<serde_json::Value>` because the OpenAPI `ProximaValue` union
/// admits null, bool, number, string, array, object, and tagged
/// `{type, value}` forms — all expressible as raw JSON values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// One of `uql`, `aql`, `federated`.
    pub language: String,
    /// Query text (non-empty).
    pub query: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
}

impl QueryRequest {
    /// Convenience constructor for the most common path (language +
    /// query, no extras).
    pub fn new(language: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            query: query.into(),
            parameters: None,
            collection: None,
            limit: None,
        }
    }
}

/// Request body for `POST /api/v2/query/explain`.
///
/// Mirrors the OpenAPI `ExplainQueryRequest` schema. Strict subset of
/// [`QueryRequest`] — no parameters, no limit (explain doesn't execute).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainQueryRequest {
    /// One of `uql`, `aql`, `federated`.
    pub language: String,
    pub query: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
}

impl ExplainQueryRequest {
    /// Convenience constructor for the most common path.
    pub fn new(language: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            query: query.into(),
            collection: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_client_builder() {
        let builder = ClientBuilder::new()
            .url("http://localhost:5678")
            .timeout_ms(5000)
            .max_retries(5)
            .api_key("test-key");

        assert_eq!(builder.config.url, "http://localhost:5678");
        assert_eq!(builder.config.timeout_ms, 5000);
        assert_eq!(builder.config.max_retries, 5);
        assert_eq!(builder.config.api_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_client_connect() {
        let client = if std::env::var_os("LLVM_PROFILE_FILE").is_some() {
            ProximaClient::for_tests("http://localhost:5678")
        } else {
            ProximaClient::connect("http://localhost:5678").unwrap()
        };
        assert_eq!(client.url(), "http://localhost:5678");
    }

    #[test]
    fn test_invalid_url() {
        let result = ProximaClient::connect("not-a-valid-url");
        assert!(result.is_err());
    }

    #[test]
    fn default_client_config_targets_local_server_with_pooling() {
        let config = ClientConfig::default();

        assert_eq!(config.url, "http://localhost:5678");
        assert_eq!(config.timeout_ms, 30000);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.api_key, None);
        assert!(config.pool_connections);
        assert_eq!(config.max_idle_connections, 10);
    }

    #[test]
    fn builder_applies_pooling_and_retry_options_to_client_config() {
        let client = ProximaClient::builder()
            .url("https://db.example.com")
            .timeout_ms(1500)
            .max_retries(7)
            .api_key("secret")
            .pool_connections(false)
            .max_idle_connections(2)
            .connect()
            .unwrap();

        assert_eq!(client.url(), "https://db.example.com");
        assert_eq!(client.config().timeout_ms, 1500);
        assert_eq!(client.config().max_retries, 7);
        assert_eq!(client.config().api_key.as_deref(), Some("secret"));
        assert!(!client.config().pool_connections);
        assert_eq!(client.config().max_idle_connections, 2);
    }

    #[cfg(feature = "client")]
    #[test]
    fn retry_status_classification_matches_transient_failures() {
        assert!(ProximaClient::is_retryable_status(
            reqwest::StatusCode::TOO_MANY_REQUESTS
        ));
        assert!(ProximaClient::is_retryable_status(
            reqwest::StatusCode::BAD_GATEWAY
        ));
        assert!(!ProximaClient::is_retryable_status(
            reqwest::StatusCode::BAD_REQUEST
        ));
        assert!(!ProximaClient::is_retryable_status(reqwest::StatusCode::OK));
    }

    #[test]
    fn invalid_url_returns_config_error_with_url_field() {
        let result = ProximaClient::with_config(ClientConfig {
            url: "not a url".to_string(),
            ..ClientConfig::default()
        });

        assert!(matches!(
            result.unwrap_err(),
            ProximaError::Config(ConfigError::InvalidValue { field, .. }) if field == "url"
        ));
    }

    #[test]
    fn debug_output_exposes_url_and_timeout_only() {
        let client = ProximaClient::builder()
            .url("https://db.example.com")
            .timeout_ms(2500)
            .api_key("secret")
            .build()
            .unwrap();

        let debug = format!("{client:?}");

        assert!(debug.contains("https://db.example.com"));
        assert!(debug.contains("2500"));
        assert!(!debug.contains("secret"));
    }

    #[test]
    fn health_status_deserializes_optional_fields() {
        let status: HealthStatus = serde_json::from_value(json!({"status": "ok"})).unwrap();

        assert_eq!(status.status, "ok");
        assert_eq!(status.version, None);
        assert_eq!(status.uptime_seconds, None);

        let status: HealthStatus =
            serde_json::from_value(json!({"status": "ok", "version": "1.0", "uptime_seconds": 42}))
                .unwrap();
        assert_eq!(status.version.as_deref(), Some("1.0"));
        assert_eq!(status.uptime_seconds, Some(42));
    }

    #[test]
    fn collection_info_accepts_record_count_alias_and_nested_stats() {
        let info: CollectionInfo = serde_json::from_value(json!({
            "collection_id": "uuid-1",
            "name": "items",
            "dimension": 384,
            "record_count": 17,
            "engine": "sst",
            "stats": {
                "record_count": 19,
                "storage_size_bytes": 1024
            }
        }))
        .unwrap();

        assert_eq!(info.collection_id.as_deref(), Some("uuid-1"));
        assert_eq!(info.name, "items");
        assert_eq!(info.dimension, 384);
        assert_eq!(info.vector_count, 17);
        assert_eq!(info.engine.as_deref(), Some("sst"));
        assert_eq!(info.stats.as_ref().unwrap().record_count, 19);
        assert_eq!(info.stats.as_ref().unwrap().storage_size_bytes, 1024);
    }

    /// Regression (TD-126 Phase 4): the v2 list surface
    /// (`CollectionV2Summary.record_count`) is `Option<u64>` server-side
    /// and emits an explicit JSON `null` when `include_stats=false`. The
    /// facade `CollectionInfo.vector_count` is a non-`Option` `u64` for
    /// callers, so a plain `#[serde(default)]` errored on `null` and broke
    /// `list_collections` decode. `null_as_default` must map `null` → 0.
    #[test]
    fn collection_info_tolerates_null_record_count() {
        let info: CollectionInfo = serde_json::from_value(json!({
            "collection_id": "uuid-1",
            "name": "items",
            "dimension": 384,
            "record_count": null,
            "engine": "sst"
        }))
        .unwrap();

        assert_eq!(info.name, "items");
        assert_eq!(info.vector_count, 0);
        assert_eq!(info.engine.as_deref(), Some("sst"));
    }

    /// The full wrapped list payload the server sends for
    /// `GET /api/v2/collections` without stats — every summary carries a
    /// `null` `record_count`. Exercises the same decode path
    /// `list_collections` uses.
    #[test]
    fn list_collections_response_tolerates_null_record_counts() {
        let response: ListCollectionsResponse = serde_json::from_value(json!({
            "collections": [
                {
                    "collection_id": "uuid-1",
                    "name": "items",
                    "dimension": 128,
                    "engine": "sst",
                    "proxima_record_enabled": true,
                    "record_count": null
                }
            ],
            "total": 1,
            "limit": 50,
            "offset": 0,
            "has_more": false
        }))
        .unwrap();

        assert_eq!(response.collections.len(), 1);
        assert_eq!(response.collections[0].name, "items");
        assert_eq!(response.collections[0].vector_count, 0);
    }

    #[test]
    fn list_response_dtos_deserialize_wrapped_collections_and_graphs() {
        let collections: ListCollectionsResponse = serde_json::from_value(json!({
            "collections": [
                {"name": "items", "dimension": 128, "vector_count": 3}
            ]
        }))
        .unwrap();
        assert_eq!(collections.collections.len(), 1);
        assert_eq!(collections.collections[0].name, "items");

        let graphs: GraphListEnvelope = serde_json::from_value(json!({
            "success": true,
            "data": [
                {
                    "graph_id": "kg",
                    "name": "",
                    "description": "test graph",
                    "stats": {"total_nodes": 2, "total_edges": 1}
                }
            ]
        }))
        .unwrap();
        let lowered: Vec<GraphInfo> = graphs.data.into_iter().map(GraphInfo::from).collect();
        assert_eq!(lowered.len(), 1);
        assert_eq!(lowered[0].name, "kg");
        assert_eq!(lowered[0].node_count, 2);
        assert_eq!(lowered[0].edge_count, 1);
        assert_eq!(lowered[0].description.as_deref(), Some("test graph"));
    }
}
