//! Collection management and fluent API for ProximaDB
//!
//! This module provides the `CollectionHandle` for fluent operations on
//! collections, and `CollectionBuilder` for creating collections.

use crate::error::{CollectionError, ProximaError, Result, VectorError};
use crate::search::SearchBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Storage engine types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StorageEngine {
    /// SST - Write-optimized, real-time workloads
    #[default]
    Sst,
    /// HELIX - Locality-optimized with Hilbert curves
    Helix,
    /// VIPER - Columnar Parquet for analytics
    Viper,
    /// SWIFT - Ultra-low latency for small datasets
    Swift,
    /// NOVA - Progressive columnar for mixed workloads
    Nova,
    /// RAPTOR - Adaptive row-group for dynamic workloads
    Raptor,
    /// TST - Time-series optimized storage
    Tst,
}

impl StorageEngine {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            StorageEngine::Sst => "sst",
            StorageEngine::Helix => "helix",
            StorageEngine::Viper => "viper",
            StorageEngine::Swift => "swift",
            StorageEngine::Nova => "nova",
            StorageEngine::Raptor => "raptor",
            StorageEngine::Tst => "tst",
        }
    }
}

impl std::str::FromStr for StorageEngine {
    type Err = ProximaError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "sst" => Ok(StorageEngine::Sst),
            "helix" => Ok(StorageEngine::Helix),
            "viper" => Ok(StorageEngine::Viper),
            "swift" => Ok(StorageEngine::Swift),
            "nova" => Ok(StorageEngine::Nova),
            "raptor" => Ok(StorageEngine::Raptor),
            "tst" => Ok(StorageEngine::Tst),
            _ => Err(ProximaError::Collection(CollectionError::UnknownEngine {
                engine: s.to_string(),
            })),
        }
    }
}

/// Index type for collections
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum IndexType {
    /// HNSW - Hierarchical Navigable Small World
    #[default]
    Hnsw,
    /// IVF - Inverted File Index
    Ivf,
    /// LSH - Locality Sensitive Hashing
    Lsh,
    /// Flat - Brute force (no index)
    Flat,
}

impl IndexType {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            IndexType::Hnsw => "hnsw",
            IndexType::Ivf => "ivf",
            IndexType::Lsh => "lsh",
            IndexType::Flat => "flat",
        }
    }
}

/// Canonical embedding precision for a collection.
///
/// Mirrors the server's proto `EmbeddingPrecision` enum and the
/// SQL DDL `WITH (canonical_embedding_precision = '...')` syntax.
/// Set once at collection-create time via
/// [`CollectionBuilder::precision`]; controls the on-disk + in-memory
/// scalar type for the embedding column. See
/// `docs/05-concepts/embedding-precision.adoc` for the operator guide.
///
/// `Fp32` is the default and matches pre-precision-rollout behavior —
/// existing SDK callers that never touch `.precision()` see no change
/// in the wire payload or server-side semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingPrecision {
    /// 32-bit float (legacy default, byte-identical with pre-rollout
    /// SDK requests).
    #[default]
    Fp32,
    /// 16-bit float (IEEE 754 half). ~50% storage vs fp32.
    Fp16,
    /// Brain float 16 (fp32 dynamic range with fp16 width).
    Bf16,
    /// Signed 8-bit scalar quantization (~25% of fp32).
    Int8,
    /// Unsigned 8-bit scalar quantization with zero-point.
    Uint8,
}

impl EmbeddingPrecision {
    /// String form matching the server's `apply_proto_enum_workarounds`
    /// and SQL DDL parser. Lowercase, no prefix.
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbeddingPrecision::Fp32 => "fp32",
            EmbeddingPrecision::Fp16 => "fp16",
            EmbeddingPrecision::Bf16 => "bf16",
            EmbeddingPrecision::Int8 => "int8",
            EmbeddingPrecision::Uint8 => "uint8",
        }
    }
}

impl std::str::FromStr for EmbeddingPrecision {
    type Err = ProximaError;

    fn from_str(s: &str) -> Result<Self> {
        // Normalize: lowercase, strip the proto SCREAMING prefix if present.
        let normalised = s.trim().to_ascii_lowercase();
        let stripped = normalised
            .strip_prefix("embedding_precision_")
            .unwrap_or(&normalised);
        match stripped {
            "fp32" | "f32" | "float32" => Ok(EmbeddingPrecision::Fp32),
            "fp16" | "f16" | "half" | "float16" => Ok(EmbeddingPrecision::Fp16),
            "bf16" | "bfloat16" => Ok(EmbeddingPrecision::Bf16),
            "int8" | "i8" | "int8_scalar" => Ok(EmbeddingPrecision::Int8),
            "uint8" | "u8" | "uint8_scalar" => Ok(EmbeddingPrecision::Uint8),
            _ => Err(ProximaError::Collection(CollectionError::InvalidConfig {
                reason: format!(
                    "unrecognised canonical_embedding_precision '{}'; \
                     accepted: fp32, fp16, bf16, int8, uint8 (case-insensitive)",
                    s
                ),
            })),
        }
    }
}

/// Distance metric for similarity search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    /// L2 (Euclidean) distance
    L2,
    /// Cosine similarity
    #[default]
    Cosine,
    /// Dot product
    DotProduct,
    /// Inner product (alias for dot product)
    InnerProduct,
}

/// Builder for creating collections
///
/// # Example
///
/// ```rust,ignore
/// client.create_collection("embeddings")
///     .dimension(768)
///     .engine(StorageEngine::Sst)
///     .index(IndexType::Hnsw)
///     .metric(DistanceMetric::Cosine)
///     .execute()
///     .await?;
/// ```
pub struct CollectionBuilder<'a> {
    #[cfg(feature = "client")]
    client: Option<&'a crate::client::ProximaClient>,
    name: String,
    dimension: Option<u32>,
    engine: StorageEngine,
    index: IndexType,
    metric: DistanceMetric,
    pub(crate) precision: EmbeddingPrecision,
}

impl<'a> CollectionBuilder<'a> {
    /// Create a new collection builder (client mode)
    #[cfg(feature = "client")]
    pub fn new(client: &'a crate::client::ProximaClient, name: &str) -> Self {
        Self {
            client: Some(client),
            name: name.to_string(),
            dimension: None,
            engine: StorageEngine::default(),
            index: IndexType::default(),
            metric: DistanceMetric::default(),
            precision: EmbeddingPrecision::default(),
        }
    }

    /// Set the vector dimension (required)
    pub fn dimension(mut self, dim: u32) -> Self {
        self.dimension = Some(dim);
        self
    }

    /// Set the storage engine
    pub fn engine(mut self, engine: StorageEngine) -> Self {
        self.engine = engine;
        self
    }

    /// Set the storage engine from string
    pub fn engine_str(mut self, engine: &str) -> Result<Self> {
        self.engine = engine.parse()?;
        Ok(self)
    }

    /// Set the index type
    pub fn index(mut self, index: IndexType) -> Self {
        self.index = index;
        self
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the canonical embedding precision (default
    /// [`EmbeddingPrecision::Fp32`]).
    ///
    /// Non-fp32 values trade a small recall delta for ~50% storage
    /// reduction (`Fp16` / `Bf16`) or up to ~75% (`Int8` / `Uint8`).
    /// Immutable after collection creation — set at build time.
    pub fn precision(mut self, precision: EmbeddingPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Set the canonical embedding precision from a string (mirrors
    /// [`Self::engine_str`]).
    ///
    /// Accepts the canonical labels (`"fp32"`, `"fp16"`, `"bf16"`,
    /// `"int8"`, `"uint8"`), case-insensitive variants, the proto
    /// SCREAMING form (`"EMBEDDING_PRECISION_FP16"`), and common
    /// aliases (`"half"`, `"float16"`, `"bfloat16"`, etc.).
    pub fn precision_str(mut self, precision: &str) -> Result<Self> {
        self.precision = precision.parse()?;
        Ok(self)
    }

    /// Execute the collection creation (async, client mode)
    #[cfg(feature = "client")]
    pub async fn execute(self) -> Result<()> {
        let client = self
            .client
            .ok_or_else(|| ProximaError::Internal("No client reference".to_string()))?;

        let dimension = self.dimension.ok_or_else(|| {
            ProximaError::Collection(CollectionError::InvalidConfig {
                reason: "dimension is required".to_string(),
            })
        })?;

        // Omit canonical_embedding_precision when fp32 (default) so the
        // wire payload is byte-identical with pre-precision-rollout SDK
        // requests. Servers that don't yet know the field stay happy;
        // newer servers see the field for non-fp32 callers and persist
        // it on the catalog row.
        let precision_payload = match self.precision {
            EmbeddingPrecision::Fp32 => None,
            other => Some(other.as_str().to_string()),
        };

        let (index_type, index_configs, index_policy) = index_request_fields(self.index);
        let request = CreateCollectionRequest {
            name: self.name,
            dimension,
            engine: Some(self.engine.as_str().to_string()),
            index_type,
            index_configs,
            index_policy,
            canonical_embedding_precision: precision_payload,
        };

        let url = format!("{}/api/v2/collections", client.url());
        let _response: CreateCollectionResponse = client.post(&url, &request).await?;
        Ok(())
    }
}

/// Handle to a collection for fluent operations
///
/// # Example
///
/// ```rust,ignore
/// let results = client.collection("embeddings")
///     .search()
///     .vector(&query)
///     .top_k(10)
///     .filter("category = 'tech'")
///     .execute()
///     .await?;
/// ```
pub struct CollectionHandle<'a> {
    #[cfg(feature = "client")]
    client: Option<&'a crate::client::ProximaClient>,
    name: String,
}

impl<'a> CollectionHandle<'a> {
    /// Create a new collection handle (client mode)
    #[cfg(feature = "client")]
    pub fn new(client: &'a crate::client::ProximaClient, name: &str) -> Self {
        Self {
            client: Some(client),
            name: name.to_string(),
        }
    }

    /// Get the collection name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Start building a search query
    #[cfg(feature = "client")]
    pub fn search(&self) -> SearchBuilder<'a> {
        SearchBuilder::new_client(self.client.expect("Client reference required"), &self.name)
    }

    /// Start building an insert operation
    pub fn insert(&self) -> InsertBuilder<'a> {
        #[cfg(feature = "client")]
        {
            InsertBuilder::new_client(self.client.expect("Client reference required"), &self.name)
        }
    }

    /// Get collection info
    #[cfg(feature = "client")]
    pub async fn info(&self) -> Result<CollectionInfo> {
        let client = self.client.expect("Client reference required");
        let url = format!("{}/api/v2/collections/{}", client.url(), self.name);
        client.get(&url).await
    }

    /// Get vector count
    #[cfg(feature = "client")]
    pub async fn count(&self) -> Result<u64> {
        let info = self.info().await?;
        Ok(info
            .stats
            .as_ref()
            .map_or(info.vector_count, |stats| stats.record_count))
    }

    /// Delete the collection
    #[cfg(feature = "client")]
    pub async fn delete(self) -> Result<()> {
        let client = self.client.expect("Client reference required");
        client.delete_collection(&self.name).await
    }

    /// Start building an update operation for a vector
    pub fn update(&self, id: &str) -> UpdateBuilder<'a> {
        #[cfg(feature = "client")]
        {
            UpdateBuilder::new_client(
                self.client.expect("Client reference required"),
                &self.name,
                id,
            )
        }
    }

    /// Delete a vector by ID
    #[cfg(feature = "client")]
    pub async fn delete_vector(&self, id: &str) -> Result<()> {
        let client = self.client.expect("Client reference required");
        let url = format!(
            "{}/api/v2/collections/{}/records/{}",
            client.url(),
            self.name,
            id
        );
        client.delete::<serde_json::Value>(&url).await?;
        Ok(())
    }

    /// Delete multiple vectors by IDs
    #[cfg(feature = "client")]
    pub async fn delete_vectors(&self, ids: Vec<String>) -> Result<usize> {
        let mut deleted = 0usize;
        for id in ids {
            self.delete_vector(&id).await?;
            deleted += 1;
        }
        Ok(deleted)
    }

    /// Get a vector by ID
    #[cfg(feature = "client")]
    #[allow(deprecated)]
    pub async fn get_vector(&self, id: &str) -> Result<Option<VectorRecord>> {
        let client = self.client.expect("Client reference required");
        let url = format!(
            "{}/api/v2/collections/{}/records/{}",
            client.url(),
            self.name,
            id
        );
        match client.get::<VectorRecord>(&url).await {
            Ok(record) => Ok(Some(record)),
            Err(ProximaError::Network(crate::error::NetworkError::HttpError {
                status: 404,
                ..
            })) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Check if a vector exists
    #[cfg(feature = "client")]
    pub async fn exists(&self, id: &str) -> Result<bool> {
        Ok(self.get_vector(id).await?.is_some())
    }
}

/// Builder for update operations
///
/// # Example
///
/// ```rust,ignore
/// collection.update("vec_123")
///     .vector(&new_embedding)
///     .meta("updated_at", "2024-01-15")
///     .execute()
///     .await?;
/// ```
pub struct UpdateBuilder<'a> {
    #[cfg(feature = "client")]
    client: Option<&'a crate::client::ProximaClient>,
    collection: String,
    id: String,
    vector: Option<Vec<f32>>,
    metadata: HashMap<String, serde_json::Value>,
    replace_metadata: bool,
}

impl<'a> UpdateBuilder<'a> {
    /// Create a new update builder (client mode)
    #[cfg(feature = "client")]
    pub fn new_client(
        client: &'a crate::client::ProximaClient,
        collection: &str,
        id: &str,
    ) -> Self {
        Self {
            client: Some(client),
            collection: collection.to_string(),
            id: id.to_string(),
            vector: None,
            metadata: HashMap::new(),
            replace_metadata: false,
        }
    }

    /// Set a new vector
    pub fn vector(mut self, vector: &[f32]) -> Self {
        self.vector = Some(vector.to_vec());
        self
    }

    /// Set metadata from JSON value (merges with existing)
    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        if let serde_json::Value::Object(map) = metadata {
            for (k, v) in map {
                self.metadata.insert(k, v);
            }
        }
        self
    }

    /// Set a single metadata field
    pub fn meta(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Replace all metadata instead of merging
    pub fn replace_metadata(mut self, replace: bool) -> Self {
        self.replace_metadata = replace;
        self
    }

    /// Execute the update (async, client mode)
    #[cfg(feature = "client")]
    pub async fn execute(self) -> Result<()> {
        let client = self
            .client
            .ok_or_else(|| ProximaError::Internal("No client reference".to_string()))?;

        let request = ProximaRecordBatchRequest {
            records: vec![ProximaRecord {
                id: self.id,
                vector: self.vector.unwrap_or_default(),
                props: self.metadata,
                text_fields: Vec::new(),
                source: None,
            }],
            validate_schema: true,
            upsert: true,
        };

        let url = format!(
            "{}/api/v2/collections/{}/records/batch",
            client.url(),
            self.collection
        );
        let _response: InsertResponse = client.post(&url, &request).await?;
        Ok(())
    }
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
    /// Number of vectors. The v2 collection surfaces may report
    /// `record_count` as `null` (e.g. list without stats), so tolerate
    /// an explicit JSON `null` (→ 0) as well as an absent field.
    #[serde(
        default,
        alias = "record_count",
        deserialize_with = "crate::serde_compat::null_as_default"
    )]
    pub vector_count: u64,
    /// Storage engine
    #[serde(default)]
    pub engine: Option<String>,
    /// Disk usage in bytes
    #[serde(default)]
    pub disk_usage_bytes: Option<u64>,
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

/// Builder for insert operations
///
/// # Example
///
/// ```rust,ignore
/// collection.insert()
///     .id("vec_123")
///     .vector(&embedding)
///     .metadata(json!({"type": "article"}))
///     .execute()
///     .await?;
/// ```
pub struct InsertBuilder<'a> {
    #[cfg(feature = "client")]
    client: Option<&'a crate::client::ProximaClient>,
    collection: String,
    records: Vec<InsertRecord>,
}

/// A single record to insert
#[derive(Debug, Clone)]
struct InsertRecord {
    id: String,
    vector: Vec<f32>,
    metadata: HashMap<String, serde_json::Value>,
    text_fields: Vec<TextFieldInput>,
}

/// TEXT field payload mirroring the v2 REST `TextFieldInput` shape at
/// `src/network/rest/v2/records.rs`. Each entry attaches a named TEXT
/// payload to the record alongside the vector — used by full-text-search
/// indexing (BM25) and rerank features. Public so the SDK builder API
/// can take typed values instead of stringly-typed JSON. TD-083 closure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFieldInput {
    /// Field name. Must match a TEXT column declared on the collection schema.
    pub name: String,
    /// Field content. v0.2 supports plain UTF-8 strings; richer encodings
    /// (chunks, language hints) are tracked post-v0.2.
    pub content: String,
}

impl TextFieldInput {
    /// Construct a new TEXT field input.
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
        }
    }
}

impl<'a> InsertBuilder<'a> {
    /// Create a new insert builder (client mode)
    #[cfg(feature = "client")]
    pub fn new_client(client: &'a crate::client::ProximaClient, collection: &str) -> Self {
        Self {
            client: Some(client),
            collection: collection.to_string(),
            records: Vec::new(),
        }
    }

    /// Add a single vector with ID
    pub fn id(self, id: impl Into<String>) -> InsertBuilderWithId<'a> {
        InsertBuilderWithId {
            builder: self,
            id: id.into(),
            vector: None,
            metadata: HashMap::new(),
            text_fields: Vec::new(),
        }
    }

    /// Add multiple vectors in batch
    pub fn batch(
        mut self,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<InsertBuilderBatch<'a>> {
        if ids.len() != vectors.len() {
            return Err(ProximaError::Vector(VectorError::BatchSizeMismatch {
                ids: ids.len(),
                vectors: vectors.len(),
            }));
        }

        for (id, vector) in ids.into_iter().zip(vectors) {
            self.records.push(InsertRecord {
                id,
                vector,
                metadata: HashMap::new(),
                text_fields: Vec::new(),
            });
        }

        Ok(InsertBuilderBatch { builder: self })
    }

    /// Execute the insert (internal)
    #[cfg(feature = "client")]
    async fn execute_internal(self) -> Result<InsertResponse> {
        let client = self
            .client
            .ok_or_else(|| ProximaError::Internal("No client reference".to_string()))?;

        if self.records.is_empty() {
            return Ok(InsertResponse { inserted_count: 0 });
        }

        let collection = self.collection;
        let request = ProximaRecordBatchRequest {
            records: self
                .records
                .into_iter()
                .map(ProximaRecord::from_insert_record)
                .collect(),
            validate_schema: true,
            upsert: false,
        };

        let url = format!(
            "{}/api/v2/collections/{}/records/batch",
            client.url(),
            collection
        );
        client.post(&url, &request).await
    }
}

/// Insert builder with ID set
pub struct InsertBuilderWithId<'a> {
    builder: InsertBuilder<'a>,
    id: String,
    vector: Option<Vec<f32>>,
    metadata: HashMap<String, serde_json::Value>,
    text_fields: Vec<TextFieldInput>,
}

impl<'a> InsertBuilderWithId<'a> {
    /// Set the vector
    pub fn vector(mut self, vector: &[f32]) -> Self {
        self.vector = Some(vector.to_vec());
        self
    }

    /// Set metadata from JSON value
    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        if let serde_json::Value::Object(map) = metadata {
            for (k, v) in map {
                self.metadata.insert(k, v);
            }
        }
        self
    }

    /// Set a single metadata field
    pub fn meta(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Attach a single TEXT field to the record. v2 ProximaRecord supports
    /// multiple named TEXT fields per record alongside the vector and props
    /// (the REST shape lives in `src/network/rest/v2/records.rs::TextFieldInput`).
    /// TD-083 closure — was previously inaccessible from the Rust SDK.
    pub fn text_field(mut self, name: impl Into<String>, content: impl Into<String>) -> Self {
        self.text_fields.push(TextFieldInput::new(name, content));
        self
    }

    /// Replace any previously-set TEXT fields with `text_fields`.
    pub fn text_fields(mut self, text_fields: Vec<TextFieldInput>) -> Self {
        self.text_fields = text_fields;
        self
    }

    /// Execute the insert (async, client mode)
    #[cfg(feature = "client")]
    pub async fn execute(mut self) -> Result<()> {
        let vector = self.vector.ok_or_else(|| {
            ProximaError::Vector(VectorError::InvalidFormat {
                reason: "vector is required".to_string(),
            })
        })?;

        self.builder.records.push(InsertRecord {
            id: self.id,
            vector,
            metadata: self.metadata,
            text_fields: self.text_fields,
        });

        self.builder.execute_internal().await?;
        Ok(())
    }
}

/// Insert builder for batch operations
pub struct InsertBuilderBatch<'a> {
    builder: InsertBuilder<'a>,
}

impl<'a> InsertBuilderBatch<'a> {
    /// Add metadata for all vectors
    pub fn with_metadata(
        mut self,
        metadata: Vec<HashMap<String, serde_json::Value>>,
    ) -> Result<Self> {
        if metadata.len() != self.builder.records.len() {
            return Err(ProximaError::Vector(VectorError::BatchSizeMismatch {
                ids: self.builder.records.len(),
                vectors: metadata.len(),
            }));
        }

        for (record, meta) in self.builder.records.iter_mut().zip(metadata) {
            record.metadata = meta;
        }

        Ok(self)
    }

    /// Execute the batch insert (async, client mode)
    #[cfg(feature = "client")]
    pub async fn execute(self) -> Result<usize> {
        let response = self.builder.execute_internal().await?;
        Ok(response.inserted_count)
    }
}

// Request/Response types for HTTP API

#[derive(Debug, Serialize)]
struct CreateCollectionRequest {
    name: String,
    dimension: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index_configs: Option<Vec<IndexConfigRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index_policy: Option<IndexPolicyRequest>,
    /// `None` for the legacy fp32 default — omitted from the wire
    /// payload to keep requests byte-identical with pre-precision-
    /// rollout SDKs. `Some("fp16")` etc. when the caller asks for a
    /// non-fp32 collection.
    #[serde(skip_serializing_if = "Option::is_none")]
    canonical_embedding_precision: Option<String>,
}

/// TD-RESTGET-1: the server ignores `index_type` "hnsw" (index selection is
/// cost-derived) and rejects any other value, which it never honoured. Send
/// what each choice means: Hnsw keeps the byte-identical default payload; Flat
/// forces brute force through the routing policy; the others declare that
/// primary index.
fn index_request_fields(
    index: IndexType,
) -> (
    Option<String>,
    Option<Vec<IndexConfigRequest>>,
    Option<IndexPolicyRequest>,
) {
    match index {
        IndexType::Hnsw => (Some("hnsw".to_string()), None, None),
        IndexType::Flat => (
            None,
            None,
            Some(IndexPolicyRequest {
                mode: "exact".to_string(),
            }),
        ),
        other => (
            None,
            Some(vec![IndexConfigRequest {
                algorithm: other.as_str().to_string(),
                is_primary: true,
            }]),
            None,
        ),
    }
}

#[cfg(test)]
mod index_request_tests {
    use super::*;

    fn body(index: IndexType) -> serde_json::Value {
        let (index_type, index_configs, index_policy) = index_request_fields(index);
        serde_json::to_value(CreateCollectionRequest {
            name: "c".to_string(),
            dimension: 4,
            engine: None,
            index_type,
            index_configs,
            index_policy,
            canonical_embedding_precision: None,
        })
        .unwrap_or_default()
    }

    #[test]
    fn default_hnsw_payload_is_unchanged() {
        assert_eq!(
            body(IndexType::Hnsw),
            serde_json::json!({"name": "c", "dimension": 4, "index_type": "hnsw"})
        );
    }

    #[test]
    fn flat_is_the_exact_routing_policy() {
        let b = body(IndexType::Flat);
        assert_eq!(b["index_policy"], serde_json::json!({"mode": "exact"}));
        assert!(b.get("index_type").is_none());
    }

    #[test]
    fn other_indexes_are_declared_not_sent_as_index_type() {
        let b = body(IndexType::Ivf);
        assert_eq!(
            b["index_configs"],
            serde_json::json!([{"algorithm": "ivf", "is_primary": true}])
        );
        assert!(b.get("index_type").is_none());
    }
}

#[derive(Debug, Serialize)]
struct IndexConfigRequest {
    algorithm: String,
    is_primary: bool,
}

#[derive(Debug, Serialize)]
struct IndexPolicyRequest {
    mode: String,
}

#[derive(Debug, Deserialize)]
struct CreateCollectionResponse {
    #[allow(dead_code)]
    #[serde(default)]
    success: bool,
}

#[derive(Debug, Serialize)]
struct ProximaRecordBatchRequest {
    records: Vec<ProximaRecord>,
    validate_schema: bool,
    #[serde(default)]
    upsert: bool,
}

/// Canonical record payload with optional vector embedding and rich properties.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProximaRecord {
    /// Record ID
    pub id: String,
    /// Dense vector embedding. The v2 `GET .../records/{id}` surface
    /// (`RecordV2Response.vector`) emits `null` when `include_vector=false`,
    /// so tolerate an explicit JSON `null` (→ empty) as well as an absent
    /// field.
    #[serde(
        default,
        deserialize_with = "crate::serde_compat::null_as_default",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub vector: Vec<f32>,
    /// Rich record properties
    #[serde(default, alias = "metadata", skip_serializing_if = "HashMap::is_empty")]
    pub props: HashMap<String, serde_json::Value>,
    /// TEXT field payloads attached to the record (BM25 indexing / rerank
    /// inputs). v0.2 supports plain UTF-8 strings; richer encodings tracked
    /// post-v0.2. See `src/network/rest/v2/records.rs::TextFieldInput`.
    /// The v2 `GET .../records/{id}` surface emits `text_fields: null`
    /// when `include_text=false`; `null_as_default` maps that to empty.
    /// `TextFieldInput` decodes the server's richer `TextFieldOutput`
    /// shape by ignoring its extra `chunk_count` / `truncated` fields.
    #[serde(
        default,
        deserialize_with = "crate::serde_compat::null_as_default",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub text_fields: Vec<TextFieldInput>,
    /// Original source text or external reference
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl ProximaRecord {
    fn from_insert_record(record: InsertRecord) -> Self {
        Self {
            id: record.id,
            vector: record.vector,
            props: record.metadata,
            text_fields: record.text_fields,
            source: None,
        }
    }
}

/// Deprecated compatibility alias for vector-shaped SDK callers.
#[deprecated(note = "use ProximaRecord; VectorRecord is a compatibility alias")]
pub type VectorRecord = ProximaRecord;

#[derive(Debug, Deserialize)]
struct InsertResponse {
    #[serde(default, alias = "success_count")]
    inserted_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ProximaClient;
    use serde_json::json;

    #[test]
    fn test_storage_engine_parse() {
        assert_eq!("sst".parse::<StorageEngine>().unwrap(), StorageEngine::Sst);
        assert_eq!(
            "helix".parse::<StorageEngine>().unwrap(),
            StorageEngine::Helix
        );
        assert_eq!("tst".parse::<StorageEngine>().unwrap(), StorageEngine::Tst);
        assert!("invalid".parse::<StorageEngine>().is_err());
    }

    #[test]
    fn test_storage_engine_as_str() {
        assert_eq!(StorageEngine::Sst.as_str(), "sst");
        assert_eq!(StorageEngine::Helix.as_str(), "helix");
        assert_eq!(StorageEngine::Tst.as_str(), "tst");
    }

    #[test]
    fn all_storage_engines_parse_case_insensitively_and_serialize_lowercase() {
        let cases = [
            (StorageEngine::Sst, "sst"),
            (StorageEngine::Helix, "helix"),
            (StorageEngine::Viper, "viper"),
            (StorageEngine::Swift, "swift"),
            (StorageEngine::Nova, "nova"),
            (StorageEngine::Raptor, "raptor"),
            (StorageEngine::Tst, "tst"),
        ];

        for (engine, name) in cases {
            assert_eq!(engine.as_str(), name);
            assert_eq!(
                name.to_uppercase().parse::<StorageEngine>().unwrap(),
                engine
            );
            assert_eq!(serde_json::to_value(engine).unwrap(), json!(name));
        }
    }

    #[test]
    fn index_types_and_metrics_serialize_as_api_names() {
        assert_eq!(IndexType::Hnsw.as_str(), "hnsw");
        assert_eq!(IndexType::Ivf.as_str(), "ivf");
        assert_eq!(IndexType::Lsh.as_str(), "lsh");
        assert_eq!(IndexType::Flat.as_str(), "flat");

        assert_eq!(
            serde_json::to_value(IndexType::Flat).unwrap(),
            json!("flat")
        );
        assert_eq!(
            serde_json::to_value(DistanceMetric::Cosine).unwrap(),
            json!("cosine")
        );
    }

    #[test]
    fn collection_builder_records_fluent_configuration_before_execute() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let builder = CollectionBuilder::new(&client, "items")
            .dimension(384)
            .engine_str("viper")
            .unwrap()
            .index(IndexType::Flat)
            .metric(DistanceMetric::DotProduct);

        assert_eq!(builder.name, "items");
        assert_eq!(builder.dimension, Some(384));
        assert_eq!(builder.engine, StorageEngine::Viper);
        assert_eq!(builder.index, IndexType::Flat);
        assert_eq!(builder.metric, DistanceMetric::DotProduct);
    }

    #[tokio::test]
    async fn collection_builder_validates_dimension_before_network_call() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let result = CollectionBuilder::new(&client, "items").execute().await;

        assert!(matches!(
            result.unwrap_err(),
            ProximaError::Collection(CollectionError::InvalidConfig { reason })
                if reason == "dimension is required"
        ));
    }

    #[test]
    fn collection_handle_exposes_name_and_creates_child_builders() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let handle = CollectionHandle::new(&client, "items");

        assert_eq!(handle.name(), "items");
        let _search = handle.search();
        assert_eq!(handle.insert().collection, "items");
        assert_eq!(handle.update("vec_1").collection, "items");
    }

    #[test]
    fn insert_batch_rejects_mismatched_ids_and_vectors() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let result = InsertBuilder::new_client(&client, "items")
            .batch(vec!["a".to_string(), "b".to_string()], vec![vec![1.0]]);

        match result {
            Err(ProximaError::Vector(VectorError::BatchSizeMismatch { ids: 2, vectors: 1 })) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("mismatched batch should fail"),
        }
    }

    #[test]
    fn insert_batch_attaches_per_record_metadata() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let batch = InsertBuilder::new_client(&client, "items")
            .batch(
                vec!["a".to_string(), "b".to_string()],
                vec![vec![1.0], vec![2.0]],
            )
            .unwrap()
            .with_metadata(vec![
                HashMap::from([("category".to_string(), json!("alpha"))]),
                HashMap::from([("category".to_string(), json!("beta"))]),
            ])
            .unwrap();

        assert_eq!(batch.builder.records.len(), 2);
        assert_eq!(
            batch.builder.records[0].metadata["category"],
            json!("alpha")
        );
        assert_eq!(batch.builder.records[1].metadata["category"], json!("beta"));
    }

    #[test]
    fn insert_batch_rejects_mismatched_metadata_count() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let batch = InsertBuilder::new_client(&client, "items")
            .batch(vec!["a".to_string()], vec![vec![1.0]])
            .unwrap();

        match batch.with_metadata(Vec::new()) {
            Err(ProximaError::Vector(VectorError::BatchSizeMismatch { ids: 1, vectors: 0 })) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("mismatched metadata count should fail"),
        }
    }

    #[tokio::test]
    async fn insert_single_record_validates_vector_before_network_call() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let result = InsertBuilder::new_client(&client, "items")
            .id("vec_1")
            .execute()
            .await;

        assert!(matches!(
            result.unwrap_err(),
            ProximaError::Vector(VectorError::InvalidFormat { reason })
                if reason == "vector is required"
        ));
    }

    #[tokio::test]
    async fn empty_insert_builder_returns_zero_without_network_call() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let response = InsertBuilder::new_client(&client, "items")
            .execute_internal()
            .await
            .unwrap();

        assert_eq!(response.inserted_count, 0);
    }

    #[test]
    fn update_builder_merges_object_metadata_and_tracks_replace_flag() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let builder = UpdateBuilder::new_client(&client, "items", "vec_1")
            .vector(&[1.0, 2.0])
            .metadata(json!({"category": "tech"}))
            .metadata(json!("ignored"))
            .meta("source", "sdk")
            .replace_metadata(true);

        assert_eq!(builder.collection, "items");
        assert_eq!(builder.id, "vec_1");
        assert_eq!(builder.vector, Some(vec![1.0, 2.0]));
        assert_eq!(builder.metadata["category"], json!("tech"));
        assert_eq!(builder.metadata["source"], json!("sdk"));
        assert!(builder.replace_metadata);
    }

    #[test]
    fn proxima_record_uses_props_alias_and_skips_empty_optional_fields() {
        let record: ProximaRecord = serde_json::from_value(json!({
            "id": "vec_1",
            "vector": [1.0, 2.0],
            "metadata": {"category": "tech"},
            "source": "doc"
        }))
        .unwrap();

        assert_eq!(record.id, "vec_1");
        assert_eq!(record.vector, vec![1.0, 2.0]);
        assert_eq!(record.props["category"], json!("tech"));
        assert_eq!(record.source.as_deref(), Some("doc"));

        let serialized = serde_json::to_value(ProximaRecord {
            id: "vec_2".to_string(),
            vector: Vec::new(),
            props: HashMap::new(),
            source: None,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(serialized, json!({"id": "vec_2"}));
    }

    /// TD-083: the Rust SDK now exposes v2 TEXT field payloads. Asserts the
    /// JSON shape lines up with the server-side `TextFieldInput` at
    /// `src/network/rest/v2/records.rs` so a regression in either direction
    /// shows up on the SDK side too.
    #[test]
    fn proxima_record_serializes_text_fields_in_v2_shape() {
        let record = ProximaRecord {
            id: "rec_1".to_string(),
            vector: Vec::new(),
            props: HashMap::new(),
            text_fields: vec![
                TextFieldInput::new("title", "ProximaDB"),
                TextFieldInput::new("body", "Vector + relational storage."),
            ],
            source: None,
        };
        let serialized = serde_json::to_value(&record).unwrap();
        assert_eq!(
            serialized,
            json!({
                "id": "rec_1",
                "text_fields": [
                    {"name": "title", "content": "ProximaDB"},
                    {"name": "body", "content": "Vector + relational storage."}
                ]
            })
        );
    }

    /// Regression (TD-126 Phase 4): `get_vector` decodes the server's
    /// `RecordV2Response`, whose `vector` and `text_fields` are
    /// `Option<...>` and emit an explicit JSON `null` when
    /// `include_vector` / `include_text` are false. The facade
    /// `ProximaRecord` keeps non-`Option` `Vec` fields, so a plain
    /// `#[serde(default)]` errored on `null` and broke `get_vector`
    /// decode. `null_as_default` must map `null` → empty, and the
    /// props-shaped record (rich `props`, server-only `version` /
    /// `timestamp`) must round-trip.
    #[test]
    fn proxima_record_tolerates_null_vector_and_text_fields() {
        let record: ProximaRecord = serde_json::from_value(json!({
            "id": "rec_1",
            "vector": null,
            "props": {"category": "tech", "score": 0.9},
            "text_fields": null,
            "version": 3,
            "timestamp": 1718000000
        }))
        .unwrap();

        assert_eq!(record.id, "rec_1");
        assert!(record.vector.is_empty());
        assert!(record.text_fields.is_empty());
        assert_eq!(record.props["category"], json!("tech"));
        assert_eq!(record.props["score"], json!(0.9));
    }

    /// The server's `RecordV2Response` populates `text_fields` with the
    /// richer `TextFieldOutput` shape (`name`, `content`, `chunk_count`,
    /// `truncated`). `get_vector` decodes those into the SDK's
    /// `TextFieldInput` by ignoring the extra fields, and a present
    /// `vector` decodes normally.
    #[test]
    fn proxima_record_decodes_record_v2_response_shape() {
        let record: ProximaRecord = serde_json::from_value(json!({
            "id": "rec_2",
            "vector": [0.1, 0.2, 0.3],
            "props": {"k": "v"},
            "text_fields": [
                {"name": "title", "content": "ProximaDB", "chunk_count": 1, "truncated": false}
            ],
            "version": 1,
            "timestamp": 42
        }))
        .unwrap();

        assert_eq!(record.id, "rec_2");
        assert_eq!(record.vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(record.text_fields.len(), 1);
        assert_eq!(record.text_fields[0].name, "title");
        assert_eq!(record.text_fields[0].content, "ProximaDB");
        assert_eq!(record.props["k"], json!("v"));
    }

    /// TD-083: chained text_field builder calls accumulate. Matches the
    /// behaviour of `meta(k, v)` for relational props.
    #[test]
    fn insert_builder_chains_text_fields() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let with_id = InsertBuilder::new_client(&client, "items")
            .id("rec_1")
            .vector(&[0.0])
            .text_field("title", "ProximaDB")
            .text_field("body", "Vector + relational storage.");
        assert_eq!(with_id.text_fields.len(), 2);
        assert_eq!(with_id.text_fields[0].name, "title");
        assert_eq!(
            with_id.text_fields[1].content,
            "Vector + relational storage."
        );
    }

    #[test]
    fn request_and_response_dtos_use_expected_api_field_names() {
        let create = CreateCollectionRequest {
            name: "items".to_string(),
            dimension: 384,
            engine: Some("sst".to_string()),
            index_type: Some("hnsw".to_string()),
            index_configs: None,
            index_policy: None,
            canonical_embedding_precision: None,
        };
        assert_eq!(
            serde_json::to_value(create).unwrap(),
            json!({
                "name": "items",
                "dimension": 384,
                "engine": "sst",
                "index_type": "hnsw"
            })
        );

        let response: InsertResponse = serde_json::from_value(json!({"success_count": 7})).unwrap();
        assert_eq!(response.inserted_count, 7);

        let info: CollectionInfo = serde_json::from_value(json!({
            "collection_id": "uuid-1",
            "name": "items",
            "dimension": 384,
            "record_count": 9,
            "engine": "sst",
            "disk_usage_bytes": 2048,
            "stats": {"record_count": 11, "storage_size_bytes": 4096}
        }))
        .unwrap();
        assert_eq!(info.collection_id.as_deref(), Some("uuid-1"));
        assert_eq!(info.vector_count, 9);
        assert_eq!(info.stats.as_ref().unwrap().record_count, 11);
    }

    // ── EmbeddingPrecision (mirrors proto EmbeddingPrecision) ────────────────
    //
    // The SDK accepts the same string-or-int shape the server's
    // `apply_proto_enum_workarounds` accepts. Set on a collection at
    // create time; immutable after creation; controls the on-disk +
    // in-memory scalar type for the embedding column. See
    // `docs/05-concepts/embedding-precision.adoc` for the operator
    // guide.

    #[test]
    fn embedding_precision_as_str_matches_proto_screaming_label() {
        assert_eq!(EmbeddingPrecision::Fp32.as_str(), "fp32");
        assert_eq!(EmbeddingPrecision::Fp16.as_str(), "fp16");
        assert_eq!(EmbeddingPrecision::Bf16.as_str(), "bf16");
        assert_eq!(EmbeddingPrecision::Int8.as_str(), "int8");
        assert_eq!(EmbeddingPrecision::Uint8.as_str(), "uint8");
    }

    #[test]
    fn embedding_precision_parses_canonical_lowercase() {
        assert_eq!(
            "fp16".parse::<EmbeddingPrecision>().unwrap(),
            EmbeddingPrecision::Fp16
        );
        assert_eq!(
            "FP16".parse::<EmbeddingPrecision>().unwrap(),
            EmbeddingPrecision::Fp16
        );
        assert_eq!(
            "EMBEDDING_PRECISION_FP16"
                .parse::<EmbeddingPrecision>()
                .unwrap(),
            EmbeddingPrecision::Fp16
        );
    }

    #[test]
    fn embedding_precision_accepts_common_aliases() {
        // Same alias set the server's apply_proto_enum_workarounds takes
        // so SDK round-trip matches what curl/manual requests look like.
        for fp16_alias in ["fp16", "f16", "half", "float16"] {
            assert_eq!(
                fp16_alias.parse::<EmbeddingPrecision>().unwrap(),
                EmbeddingPrecision::Fp16
            );
        }
        for bf16_alias in ["bf16", "bfloat16"] {
            assert_eq!(
                bf16_alias.parse::<EmbeddingPrecision>().unwrap(),
                EmbeddingPrecision::Bf16
            );
        }
        for int8_alias in ["int8", "i8", "int8_scalar"] {
            assert_eq!(
                int8_alias.parse::<EmbeddingPrecision>().unwrap(),
                EmbeddingPrecision::Int8
            );
        }
        for uint8_alias in ["uint8", "u8", "uint8_scalar"] {
            assert_eq!(
                uint8_alias.parse::<EmbeddingPrecision>().unwrap(),
                EmbeddingPrecision::Uint8
            );
        }
    }

    #[test]
    fn embedding_precision_unknown_label_errors_with_recognisable_message() {
        let err = "definitely_not_a_precision"
            .parse::<EmbeddingPrecision>()
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.to_lowercase().contains("precision"),
            "error message should mention 'precision' so SDK users \
             know what field they typo'd; got: {msg}"
        );
    }

    #[test]
    fn embedding_precision_serializes_as_lowercase_string() {
        // Matches what the server's REST workaround accepts so a
        // ProximaRecord-style direct JSON build (without the SDK)
        // sees the same wire format.
        assert_eq!(
            serde_json::to_value(EmbeddingPrecision::Fp16).unwrap(),
            json!("fp16")
        );
    }

    #[test]
    fn collection_builder_default_precision_is_fp32() {
        // Backward compatibility: existing callers that never touch
        // .precision() must continue to land fp32 collections.
        let client = ProximaClient::for_tests("http://localhost:5678");
        let builder = CollectionBuilder::new(&client, "items").dimension(8);
        assert_eq!(builder.precision, EmbeddingPrecision::Fp32);
    }

    #[test]
    fn collection_builder_precision_setter_records_target() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let builder = CollectionBuilder::new(&client, "items")
            .dimension(8)
            .precision(EmbeddingPrecision::Fp16);
        assert_eq!(builder.precision, EmbeddingPrecision::Fp16);
    }

    #[test]
    fn collection_builder_precision_str_setter_accepts_aliases() {
        // Mirrors `engine_str` — lets callers spec the precision from
        // a string without importing the enum (handy for CLI / config-
        // driven builds).
        let client = ProximaClient::for_tests("http://localhost:5678");
        let builder = CollectionBuilder::new(&client, "items")
            .dimension(8)
            .precision_str("fp16")
            .unwrap();
        assert_eq!(builder.precision, EmbeddingPrecision::Fp16);
    }

    #[test]
    fn create_collection_request_includes_precision_when_non_fp32() {
        // The REST POST body must carry canonical_embedding_precision
        // when set, so the server's create-handler path materializes
        // the right catalog row. Fp32 (default) is omitted from the
        // payload to keep the wire shape byte-identical with
        // pre-precision-rollout SDK requests.
        let req = CreateCollectionRequest {
            name: "rust_sdk_fp16".to_string(),
            dimension: 384,
            engine: None,
            index_type: None,
            index_configs: None,
            index_policy: None,
            canonical_embedding_precision: Some("fp16".to_string()),
        };
        let body = serde_json::to_value(&req).unwrap();
        assert_eq!(body["canonical_embedding_precision"], json!("fp16"));
    }

    #[test]
    fn create_collection_request_omits_precision_when_fp32_default() {
        let req = CreateCollectionRequest {
            name: "rust_sdk_fp32".to_string(),
            dimension: 384,
            engine: None,
            index_type: None,
            index_configs: None,
            index_policy: None,
            canonical_embedding_precision: None,
        };
        let body = serde_json::to_value(&req).unwrap();
        assert!(
            body.get("canonical_embedding_precision").is_none(),
            "fp32 default must not appear in the wire payload — \
             keeps requests byte-identical with pre-rollout SDKs. Got: {body}"
        );
    }
}
