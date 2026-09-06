// Language-binding API methods may be unused without features + have many args
// (matching the root crate allowances).
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

//! # Embedded Mode - In-Process ProximaDB Database
//!
//! This module provides an embedded mode for using ProximaDB as an in-process
//! database without the network layer. It enables direct access to all ProximaDB
//! functionality with full multi-disk support.
//!
//! ## Architecture
//!
//! ```text
//! Application (Rust / Python / Java / Go / Node.js)
//!       |
//!       v
//! ┌─────────────────────────────────────────────────────────────┐
//! │         EmbeddedProximaDB (this module)                 │
//! │   - Direct in-process API                               │
//! │   - No network overhead                                 │
//! │   - Multi-disk configuration                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │   Language Bindings (feature-gated)                     │
//! │   - Python: PyO3 bindings (feature = "python")          │
//! │   - Java: JNI bindings (feature = "java")               │
//! │   - Go: C FFI bindings (feature = "c_ffi")              │
//! │   - Node.js: NAPI-RS (feature = "nodejs")               │
//! ├─────────────────────────────────────────────────────────────┤
//! │         ProximaDB Core (Rust)                           │
//! │   - Storage Engines (SST, VIPER, NOVA, SWIFT, etc.)     │
//! │   - Multi-disk support via StorageLocation              │
//! │   - WAL persistence                                     │
//! │   - Compute (SIMD-accelerated)                          │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **No network overhead**: Direct in-process API calls
//! - **Multi-disk support**: Configure multiple storage locations with weights
//! - **Automatic hardware acceleration**: SIMD detection and usage
//! - **Full persistence**: WAL, snapshots, and recovery
//! - **Thread-safe**: Safe for concurrent access
//!
//! ## Protocol Boundary
//!
//! Embedded language bindings are in-process entry points. They should call the
//! shared Rust services, catalog/query facades, and storage/runtime contracts
//! directly instead of starting loopback REST, gRPC, Arrow Flight, or pgwire
//! servers. Those protocols remain the right surfaces for remote clients and
//! explicit protocol benchmarks, but embedded mode should keep ports out of the
//! normal hot path.
//!
//! ## Build Features
//!
//! Enable language-specific bindings with Cargo features:
//! - `--features python` - Python PyO3 bindings
//! - `--features java` - Java JNI bindings
//! - `--features c_ffi` - C FFI for Go CGO
//! - `--features nodejs` - Node.js NAPI-RS bindings
//! - `--features embedded-all` - All language bindings

// Multi-process coordination
pub mod coordination;

// Re-export embedded support types for public API compatibility.
pub use proximadb_embedded_common::{
    EmbeddedMetrics, EmbeddedMetricsCollector, HistogramStats, LatencyStats, LatencyTimer,
    RollingWindow,
};

// Re-export coordination types for public API
pub use coordination::{
    AccessMode, CoordinationError, FileLockManager, FileLockSet, LeaderElection, LeaderStatus,
};

// Language-specific bindings - compiled when corresponding feature is enabled

// Python bindings via PyO3
#[cfg(feature = "python")]
pub mod python;

// Python DataFrame API via DataFusion
#[cfg(feature = "python")]
pub mod python_dataframe;

// Java bindings via JNI
#[cfg(feature = "java")]
pub mod java;

// C FFI for Go CGO and other C-compatible languages
#[cfg(feature = "c_ffi")]
pub mod c_ffi;

// Node.js bindings via NAPI-RS
#[cfg(feature = "nodejs")]
pub mod nodejs;

// Streaming search infrastructure
pub mod spark;
pub mod streaming;

// Agent memory management with checkpoints
pub mod agent_memory;

// Re-export agent memory types
pub use agent_memory::{
    CheckpointInfo, CheckpointManager, CollectionCheckpointState, DeltaEntry, DeltaHeader,
    DeltaInfo,
};

// Re-export streaming types
pub use streaming::{EmbeddedSearchIterator, StreamingSearchConfig, StreamingSearchResult};

use proximadb::core::config::{AdvancedPruneConfig, PruneModeConfig};
// FA-b: only referenced on the `#[cfg(feature = "abac-policy")]` trailing arg of
// `unified_search_native`. Embedded mode is abac-OFF by default; this import is
// gated to match (the `proximadb-abac` dep is optional, behind the same feature).
#[cfg(feature = "abac-policy")]
use proximadb_abac::{ReadContext, SystemReadReason};

/// Embedded database configuration for multi-disk support
#[derive(Debug, Clone)]
pub struct EmbeddedConfig {
    /// Storage locations with weights for data distribution
    pub storage_locations: Vec<StorageLocationConfig>,
    /// Metadata storage path (should be on fast storage)
    pub metadata_path: String,
    /// Total cache size in MB
    pub cache_size_mb: usize,
    /// Default storage engine type
    pub default_engine: String,
    /// Enable WAL for durability
    pub enable_wal: bool,
    /// WAL sync mode: "immediate", "batch", or "async"
    pub wal_sync_mode: String,
    /// Block prune mode for approximate search: "sqrt" (default), "ratio", "fixed", or "exact" (disabled)
    pub block_prune_mode: String,
    /// Block prune ratio (0.0-1.0) when mode is "ratio"
    pub block_prune_ratio: f32,
    /// Minimum blocks to keep (used with sqrt/ratio modes)
    pub block_prune_min_keep: usize,
    /// Maximum blocks to keep (0 = no cap)
    pub block_prune_max_keep: usize,
    /// Enable RL-based adaptive query planner
    pub enable_rl_planner: bool,
    /// Path for RL policy persistence (default: data_dir/rl_policy.json)
    pub rl_policy_path: Option<String>,
    /// Access mode for multi-process coordination
    /// - Exclusive: Single writer, exclusive access (default)
    /// - SharedRead: Multiple readers, no writers
    /// - LeaderFollower: One leader (write), many followers (read)
    pub access_mode: AccessMode,
    /// Node ID for leader election (only used in LeaderFollower mode)
    /// If not set, a random UUID will be generated
    pub node_id: Option<String>,
    /// Logical tenant for the boundary governance contract (co-design tenet 3).
    /// `None` (default) keeps the legacy single-tenant behavior byte-identical —
    /// records are written with an empty `tenant_id` and the vector on-disk layout
    /// (collection-keyed) is unchanged. When set, inserts are routed through the
    /// same tenant-scoped path the networked server uses, so the boundary contract
    /// is uniform and multi-tenant embedding becomes opt-in (no migration: embedded
    /// vector paths are not tenant-keyed).
    pub tenant_id: Option<String>,
}

impl Default for EmbeddedConfig {
    fn default() -> Self {
        Self {
            storage_locations: vec![StorageLocationConfig {
                path: "./data".to_string(),
                weight: 1,
                tags: vec![],
            }],
            metadata_path: "./data/metadata".to_string(),
            cache_size_mb: 512,
            default_engine: "sst".to_string(),
            enable_wal: true,
            wal_sync_mode: "batch".to_string(),
            // Block pruning defaults (sqrt mode with sensible bounds)
            block_prune_mode: "sqrt".to_string(),
            block_prune_ratio: 0.2,
            block_prune_min_keep: 1,
            block_prune_max_keep: 0, // No cap
            // RL planner defaults (enabled by default for adaptive query optimization)
            enable_rl_planner: true,
            rl_policy_path: None, // Default to data_dir/rl_policy.json
            // Multi-process coordination defaults
            access_mode: AccessMode::Exclusive, // Default to exclusive access
            node_id: None,                      // Auto-generate if needed
            tenant_id: None,                    // Single-tenant default (legacy behavior)
        }
    }
}

impl EmbeddedConfig {
    /// Create an optimized configuration for embedded benchmarks
    ///
    /// This configuration is tuned for:
    /// - Maximum write throughput
    /// - Low-latency reads
    /// - Efficient memory usage with adaptive bloom filters
    /// - Aggressive compaction for smaller footprint
    pub fn for_benchmarks(data_path: impl Into<String>) -> Self {
        let path = data_path.into();
        Self {
            storage_locations: vec![StorageLocationConfig {
                path: path.clone(),
                weight: 1,
                tags: vec!["benchmark".to_string()],
            }],
            metadata_path: format!("{}/metadata", path),
            cache_size_mb: 1024, // 1GB cache for benchmarks
            default_engine: "sst".to_string(),
            enable_wal: true,
            wal_sync_mode: "batch".to_string(), // Batch for better throughput
            block_prune_mode: "sqrt".to_string(),
            block_prune_ratio: 0.2,
            block_prune_min_keep: 1,
            block_prune_max_keep: 0,
            enable_rl_planner: true, // Enable RL for benchmark optimization
            rl_policy_path: Some(format!("{}/rl_policy.json", path)),
            access_mode: AccessMode::Exclusive, // Benchmarks use exclusive access
            node_id: None,
            tenant_id: None,
        }
    }

    /// Cold-path benchmark config (TD-096 S2): identical to [`Self::for_benchmarks`]
    /// but with the block cache **disabled** (`cache_size_mb: 0`) and the RL planner
    /// off (deterministic). Pair with a flush-then-**reopen** so the WAL memtable and
    /// cache start empty — this forces searches to read from the `FileSystem`-backed
    /// SST files so `CountingFileSystem` measures real disk GETs (the warm-cache +
    /// WAL-retention confound that masked S2's measurement is removed).
    pub fn for_benchmarks_cold(data_path: impl Into<String>) -> Self {
        let mut cfg = Self::for_benchmarks(data_path);
        cfg.cache_size_mb = 0;
        cfg.enable_rl_planner = false;
        cfg.rl_policy_path = None;
        cfg
    }

    /// Create an optimized configuration for memory-constrained environments
    ///
    /// Uses minimal cache and aggressive pruning
    pub fn for_low_memory(data_path: impl Into<String>) -> Self {
        let path = data_path.into();
        Self {
            storage_locations: vec![StorageLocationConfig {
                path: path.clone(),
                weight: 1,
                tags: vec![],
            }],
            metadata_path: format!("{}/metadata", path),
            cache_size_mb: 128, // Minimal cache
            default_engine: "sst".to_string(),
            enable_wal: true,
            wal_sync_mode: "batch".to_string(),
            block_prune_mode: "sqrt".to_string(),
            block_prune_ratio: 0.2,
            block_prune_min_keep: 1,
            block_prune_max_keep: 0,
            enable_rl_planner: false, // Disable RL for memory-constrained environments
            rl_policy_path: None,
            access_mode: AccessMode::Exclusive, // Default to exclusive access
            node_id: None,
            tenant_id: None,
        }
    }

    /// Set the access mode for multi-process coordination
    ///
    /// # Arguments
    /// * `mode` - Access mode to use
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = EmbeddedConfig::default()
    ///     .with_access_mode(AccessMode::SharedRead);
    /// ```
    pub fn with_access_mode(mut self, mode: AccessMode) -> Self {
        self.access_mode = mode;
        self
    }

    /// Set the node ID for leader election
    ///
    /// Only used in LeaderFollower mode.
    ///
    /// # Arguments
    /// * `node_id` - Unique identifier for this node
    pub fn with_node_id(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    /// Set the logical tenant for the boundary governance contract.
    ///
    /// `None` (the default) keeps legacy single-tenant behavior; setting a tenant
    /// opts the embedded database into the same tenant-scoped write path the
    /// networked server uses. Embedded vector storage is collection-keyed (not
    /// tenant-keyed), so this is non-breaking — existing data resolves unchanged.
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = EmbeddedConfig::default().with_tenant("acme");
    /// ```
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }
}

/// Storage location configuration for multi-disk support
#[derive(Debug, Clone)]
pub struct StorageLocationConfig {
    /// Path to storage directory (will be converted to file:// URL)
    pub path: String,
    /// Weight for data distribution (higher = more data)
    pub weight: u32,
    /// Tags for storage tier identification (e.g., "hot", "cold")
    pub tags: Vec<String>,
}

impl StorageLocationConfig {
    /// Create a new storage location
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            weight: 1,
            tags: vec![],
        }
    }

    /// Set the weight for this storage location
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    /// Add a tag to this storage location
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Convert to file:// URL format for internal use
    pub fn to_url(&self) -> String {
        if self.path.starts_with("file://") {
            self.path.clone()
        } else {
            let abs_path = if self.path.starts_with('/') {
                self.path.clone()
            } else {
                // Make relative path absolute
                std::env::current_dir().map_or_else(
                    |_| self.path.clone(),
                    |p| p.join(&self.path).to_string_lossy().to_string(),
                )
            };
            format!("file://{}", abs_path)
        }
    }
}

/// Backwards-compat alias for [`EmbeddedSearchResult`].
pub type SearchResult = EmbeddedSearchResult;

/// Search result from embedded database
#[derive(Debug, Clone)]
pub struct EmbeddedSearchResult {
    /// Vector ID
    pub id: String,
    /// Similarity score (lower is more similar for distance metrics)
    pub score: f32,
    /// Associated metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Backwards-compat alias for [`EmbeddedCollectionInfo`].
pub type CollectionInfo = EmbeddedCollectionInfo;

/// Collection information
#[derive(Debug, Clone)]
pub struct EmbeddedCollectionInfo {
    /// Collection name
    pub name: String,
    /// Vector dimension
    pub dimension: u32,
    /// Number of vectors
    pub vector_count: u64,
    /// Storage engine type
    pub engine: String,
    /// Disk usage in bytes for this collection
    pub disk_usage_bytes: u64,
}

/// Convert a ProximaValue to a plain String for the Python-exposed EmbeddedSearchResult.metadata map.
/// Rich types (Map, Array, Json, Jsonb) are JSON-serialised so no data is lost.
pub(crate) fn proxima_value_to_string(v: proximadb_data_model::ProximaValue) -> String {
    use proximadb_data_model::ProximaValue;
    match v {
        ProximaValue::String(s) | ProximaValue::Symbol(s) => s,
        ProximaValue::Float32(f) => f.to_string(),
        ProximaValue::Float64(f) => f.to_string(),
        ProximaValue::Int8(i) => i.to_string(),
        ProximaValue::Int16(i) => i.to_string(),
        ProximaValue::Int32(i) => i.to_string(),
        ProximaValue::Int64(i) => i.to_string(),
        ProximaValue::UInt8(i) => i.to_string(),
        ProximaValue::UInt16(i) => i.to_string(),
        ProximaValue::UInt32(i) => i.to_string(),
        ProximaValue::UInt64(i) => i.to_string(),
        ProximaValue::Boolean(b) => b.to_string(),
        ProximaValue::Json(v) | ProximaValue::Jsonb(v) => v.to_string(),
        ProximaValue::Null => String::new(),
        // Everything else — containers AND exotics (dashed uuid, base64
        // binary, temporals) — renders through the shared canonical JSON
        // surface: strings come back as bare text, structured values as
        // their JSON text. The old arms here included a '<N bytes>'
        // Binary placeholder (content unrecoverable) and the Rust-Debug
        // fallback for exotics.
        other => match proximadb_embedded_common::proxima_value_to_json(other) {
            serde_json::Value::String(s) => s,
            json => json.to_string(),
        },
    }
}

pub(crate) use proximadb_embedded_common::proxima_value_to_json;

pub(crate) fn json_to_proxima_value(
    value: serde_json::Value,
) -> proximadb_data_model::ProximaValue {
    use proximadb_data_model::ProximaValue;

    match value {
        serde_json::Value::Null => ProximaValue::Null,
        serde_json::Value::Bool(v) => ProximaValue::Boolean(v),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ProximaValue::Int64(i)
            } else if let Some(u) = n.as_u64() {
                ProximaValue::UInt64(u)
            } else if let Some(f) = n.as_f64() {
                ProximaValue::Float64(f)
            } else {
                ProximaValue::String(n.to_string())
            }
        }
        serde_json::Value::String(v) => ProximaValue::String(v),
        serde_json::Value::Array(values) => {
            ProximaValue::Array(values.into_iter().map(json_to_proxima_value).collect())
        }
        serde_json::Value::Object(fields) => ProximaValue::Map(
            fields
                .into_iter()
                .map(|(key, value)| (key, json_to_proxima_value(value)))
                .collect(),
        ),
    }
}

fn collection_engine_name(storage_engine: Option<i32>) -> String {
    proximadb::core::conversions::storage_engine_to_string(
        storage_engine.unwrap_or(proximadb::proto::proximadb_v1::StorageEngine::Sst as i32),
    )
    .to_string()
}

/// Backwards-compat alias for [`EmbeddedStorageStats`].
pub type StorageStats = EmbeddedStorageStats;

/// Storage statistics
#[derive(Debug, Clone)]
pub struct EmbeddedStorageStats {
    /// Total number of vectors across all collections
    pub total_vectors: u64,
    /// Total number of collections
    pub total_collections: u64,
    /// Total disk usage in bytes
    pub disk_usage_bytes: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
}

/// Cache statistics snapshot (internal type for metrics integration)
#[derive(Debug, Clone)]
struct CacheStatsSnapshot {
    /// Number of entries in cache
    entries: u64,
    /// Memory used by cache in bytes
    memory_bytes: u64,
}

// ============================================================================
// Generic Graph Database Types - Tool Agnostic API
// ============================================================================
//
// These types provide a generic, flexible graph API that can be used for
// any domain: social graphs, knowledge graphs, code graphs, etc.
// Domain-specific fields (like code's "signature", "docstring", "line")
// should be stored in the `properties` map.
//
// For code intelligence use cases, the consuming application (e.g., Victor)
// should build an adapter layer that maps its domain types to these generic types.

/// Generic graph node with flexible property storage
///
/// This is a domain-agnostic node type. All domain-specific attributes
/// should be stored in the `properties` map.
///
/// # Example
/// ```rust,ignore
/// // For a code symbol:
/// let node = EmbeddedGraphNode::new("fn_main")
///     .with_label("function")
///     .with_property("name", "main")
///     .with_property("file", "main.py")
///     .with_property("line", "42");
///
/// // For a social network:
/// let node = EmbeddedGraphNode::new("user_123")
///     .with_label("Person")
///     .with_property("name", "Alice")
///     .with_property("email", "alice@example.com");
/// ```
///
/// Backwards-compat alias `GraphNode` is provided below.
#[derive(Debug, Clone)]
pub struct EmbeddedGraphNode {
    /// Unique node identifier
    pub id: String,
    /// Node labels/types (e.g., "Person", "function", "Document")
    pub labels: Vec<String>,
    /// Typed property storage. Was `HashMap<String, String>`, which forced
    /// every value through stringification on write (`line: 42` stored as
    /// `StringValue("42")`, leaving the engine's numeric index permanently
    /// empty) and silently DROPPED Bytes/Array/Object/Vector on read
    /// (proximaDB#1698). `PropertyValue` has `From<&str>/String/i64/i32/f64/
    /// bool`, so `with_property("name", "Alice")` still reads the same.
    pub properties: std::collections::HashMap<String, proximadb::graph::model::PropertyValue>,
}

impl EmbeddedGraphNode {
    /// Create a new node with the given ID
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            labels: Vec::new(),
            properties: std::collections::HashMap::new(),
        }
    }

    /// Add a label to this node
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.labels.push(label.into());
        self
    }

    /// Add a property to this node (typed; `&str`/`String`/`i64`/`i32`/`f64`/
    /// `bool` convert implicitly, or pass a `PropertyValue` for the rest)
    pub fn with_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<proximadb::graph::model::PropertyValue>,
    ) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Convert to proto Node for storage (lossless: properties are typed)
    pub fn to_proto(&self) -> proximadb::graph::Node {
        use proximadb::graph::Node;

        let properties = self.properties.clone();

        Node {
            id: self.id.clone(),
            labels: self.labels.clone(),
            properties,
            embedding: None,
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            updated_at_ms: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Create from proto Node (lossless: every variant survives, where the
    /// stringly version dropped Bytes/Array/Object/Vector outright)
    pub fn from_proto(node: &proximadb::graph::Node) -> Self {
        let properties = node.properties.clone();

        Self {
            id: node.id.clone(),
            labels: node.labels.clone(),
            properties,
        }
    }
}

/// Backwards-compat alias for [`EmbeddedGraphNode`]. Matches the alias
/// promise in the doc comment on `EmbeddedGraphNode` above.
pub type GraphNode = EmbeddedGraphNode;

/// Backwards-compat alias for [`EmbeddedGraphEdge`].
pub type GraphEdge = EmbeddedGraphEdge;

/// Generic graph edge with flexible property storage
#[derive(Debug, Clone)]
pub struct EmbeddedGraphEdge {
    /// Optional edge ID (auto-generated if not provided)
    pub id: Option<String>,
    /// Source node ID
    pub from_node_id: String,
    /// Destination node ID
    pub to_node_id: String,
    /// Edge type/relationship name
    pub edge_type: String,
    /// Optional weight for weighted traversal
    pub weight: Option<f64>,
    /// Typed property storage (see the node-side comment; proximaDB#1698)
    pub properties: std::collections::HashMap<String, proximadb::graph::model::PropertyValue>,
}

impl EmbeddedGraphEdge {
    /// Create a new edge
    pub fn new(
        from_node_id: impl Into<String>,
        to_node_id: impl Into<String>,
        edge_type: impl Into<String>,
    ) -> Self {
        Self {
            id: None,
            from_node_id: from_node_id.into(),
            to_node_id: to_node_id.into(),
            edge_type: edge_type.into(),
            weight: None,
            properties: std::collections::HashMap::new(),
        }
    }

    /// Set edge ID explicitly
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set edge weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = Some(weight);
        self
    }

    /// Add a property (typed; scalars convert implicitly)
    pub fn with_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<proximadb::graph::model::PropertyValue>,
    ) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Generate edge ID from components
    fn generated_id(&self) -> String {
        format!(
            "{}->{}:{}",
            self.from_node_id, self.to_node_id, self.edge_type
        )
    }

    /// Convert to proto Edge
    pub fn to_proto(&self) -> proximadb::graph::Edge {
        use proximadb::graph::Edge;

        let properties = self.properties.clone();

        Edge {
            id: self.id.clone().unwrap_or_else(|| self.generated_id()),
            from_node_id: self.from_node_id.clone(),
            to_node_id: self.to_node_id.clone(),
            edge_type: self.edge_type.clone(),
            properties,
            weight: self.weight,
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            updated_at_ms: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Create from proto Edge
    pub fn from_proto(edge: &proximadb::graph::Edge) -> Self {
        let properties = edge.properties.clone();

        Self {
            id: Some(edge.id.clone()),
            from_node_id: edge.from_node_id.clone(),
            to_node_id: edge.to_node_id.clone(),
            edge_type: edge.edge_type.clone(),
            weight: edge.weight,
            properties,
        }
    }
}

/// Backwards-compat alias for [`EmbeddedGraphStats`].
pub type GraphStats = EmbeddedGraphStats;

/// Graph statistics
#[derive(Debug, Clone)]
pub struct EmbeddedGraphStats {
    /// Total number of nodes
    pub total_nodes: u64,
    /// Total number of edges
    pub total_edges: u64,
}

/// Embedded ProximaDB instance without network layer
///
/// This provides direct in-process access to ProximaDB functionality
/// without any network overhead. All operations are performed directly
/// against the storage engines.
///
/// # Example
///
/// ```rust,ignore
/// use proximadb::embedded::{EmbeddedProximaDB, EmbeddedConfig, StorageLocationConfig};
///
/// let config = EmbeddedConfig {
///     storage_locations: vec![
///         StorageLocationConfig::new("/nvme/data").with_weight(2),
///         StorageLocationConfig::new("/hdd/data").with_weight(1),
///     ],
///     ..Default::default()
/// };
///
/// let db = EmbeddedProximaDB::new(config)?;
/// db.create_collection("embeddings", 768, None)?;
/// db.insert("embeddings", vec!["id1".into()], vec![vec![0.1; 768]], None)?;
/// let results = db.search("embeddings", vec![0.1; 768], 10, None)?;
/// // Call close() to persist RL policy before dropping
/// db.close();
/// ```
/// TD-DSEFF-2: `true` when a non-SST-declared collection's data directory
/// holds files and every one of them carries the legacy SST extension —
/// the residue of the pre-#1743 flush-routing bug (`FlushExecutionCoordinator::
/// engine_for` unconditionally used a caller-supplied SST fallback, so every
/// embedded collection flushed as SST regardless of its declared engine).
/// Pure and unit-testable: callers already gate this on the declared engine
/// being non-SST before listing the directory, so this only needs the
/// resulting file names.
fn engine_data_mismatch(file_names: &[String]) -> bool {
    !file_names.is_empty() && file_names.iter().all(|name| name.ends_with(".sst"))
}

pub struct EmbeddedProximaDB {
    /// Configuration
    config: EmbeddedConfig,
    /// Tokio runtime for async operations
    runtime: tokio::runtime::Runtime,
    /// Shared services containing all ProximaDB functionality
    shared_services: proximadb::network::multi_server::SharedServices,
    /// Embedded catalog manager for pgwire-compatible schema DDL and metadata
    catalog_manager: std::sync::Arc<proximadb::catalog::CatalogManager>,
    /// Collection port for collection management (Phase 9: port-typed; concrete still available via `self.shared_services.collection_service` when port surface insufficient — see Task #76 deferral notes)
    collection_port: std::sync::Arc<dyn proximadb_runtime::CollectionPort>,
    /// Path where RL planner policy is persisted (None if RL disabled)
    rl_policy_path: Option<String>,
    /// Metrics collector for embedded mode observability
    metrics_collector: std::sync::Arc<EmbeddedMetricsCollector>,
    /// Checkpoint manager for incremental persistence
    checkpoint_manager: std::sync::Arc<CheckpointManager>,
    /// One ownership lease spanning every configured local writable root.
    lock_manager: Option<FileLockSet>,
    /// Leader election for leader/follower mode
    leader_election: Option<LeaderElection>,
    /// Per-open fence for cursors. An embedded close/reopen is a database
    /// incarnation change even when the host process stays alive.
    content_revision_incarnation: String,
    /// Makes close idempotent and prevents Drop from flushing after ownership
    /// has already been handed to a newly opened instance.
    closed: std::sync::atomic::AtomicBool,
}

impl EmbeddedProximaDB {
    /// Create a new embedded ProximaDB instance
    ///
    /// This initializes the database with the given configuration,
    /// including multi-disk support and WAL settings.
    pub fn new(config: EmbeddedConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let base_path = config
            .storage_locations
            .first()
            .map_or_else(|| "./data".to_string(), |loc| loc.path.clone());
        let mut ownership_roots: Vec<std::path::PathBuf> = config
            .storage_locations
            .iter()
            .filter_map(|location| {
                location
                    .path
                    .strip_prefix("file://")
                    .map(std::path::PathBuf::from)
                    .or_else(|| {
                        (!location.path.contains("://"))
                            .then(|| std::path::PathBuf::from(&location.path))
                    })
            })
            .collect();
        if let Some(path) = config.metadata_path.strip_prefix("file://") {
            ownership_roots.push(std::path::PathBuf::from(path));
        } else if !config.metadata_path.contains("://") {
            ownership_roots.push(std::path::PathBuf::from(&config.metadata_path));
        }

        // Ownership is the first side effect. A losing opener must not reset
        // process globals, construct services, or begin recovery before it
        // learns that this database is already in use.
        let (lock_manager, leader_election) = match config.access_mode {
            AccessMode::Exclusive => {
                let lock = FileLockSet::acquire(&ownership_roots, AccessMode::Exclusive).map_err(
                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(format!(
                            "Failed to acquire exclusive lock: {}",
                            e
                        )))
                    },
                )?;
                (Some(lock), None)
            }
            AccessMode::SharedRead => {
                let lock = FileLockSet::acquire(&ownership_roots, AccessMode::SharedRead).map_err(
                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(format!(
                            "Failed to acquire shared read lock: {}",
                            e
                        )))
                    },
                )?;
                (Some(lock), None)
            }
            AccessMode::LeaderFollower => {
                let access_lock = FileLockSet::acquire(&ownership_roots, AccessMode::SharedRead)
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(format!(
                            "Failed to acquire leader/follower access lock: {}",
                            e
                        )))
                    })?;
                let node_id = config
                    .node_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                let election = LeaderElection::new(&base_path, &node_id).map_err(
                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(format!(
                            "Failed to initialize leader election: {}",
                            e
                        )))
                    },
                )?;
                (Some(access_lock), Some(election))
            }
        };

        // [AGENT_FIX]: Forcefully reset global static state to allow multiple
        // embedded instances within the same process, which is critical for tests
        // and benchmarks. This is an unsafe workaround for a design limitation
        // where the engine relies on `OnceLock` globals.
        unsafe {
            proximadb::storage::persistence::write_ahead_log::reset_global_wal_state_for_tests();
            tracing::info!(
                "🧹 EMBEDDED: Unsafe reset of global state (manifest, write buffer, registry) complete."
            );
        }

        // Create tokio runtime for async operations.
        // 8MB worker stack: correlated subqueries (TPC-H Q2/Q20-Q22) produce
        // deep-but-finite async call chains through join/predicate evaluation
        // that overflow the default ~2MB stack. 8MB passes all 22 TPC-H queries.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_cpus::get().min(4))
            .thread_stack_size(8 * 1024 * 1024)
            .enable_all()
            .build()?;

        // Convert EmbeddedConfig to StorageConfig
        let storage_config = Self::to_storage_config(&config);

        // Initialize SharedServices using the runtime
        let (shared_services, collection_port) =
            runtime.block_on(async { Self::init_services(storage_config).await })?;

        // Recover graphs from snapshots + WAL before serving any request.
        //
        // The other half of #1524. `ProximaDB::start` has always done this,
        // but — exactly like the shutdown-side flush — its call is gated on
        // `self.multi_server`, the NETWORK object. Port-free embedded never
        // constructs one, so it started with an empty ORION engine no matter
        // what the WAL held: reopening a directory containing 177k durable
        // edges answered `edges=0`, indistinguishable from data loss.
        //
        // Both halves of durability were attached to the network server, so a
        // transport choice silently decided whether data survived. Recovery is
        // best-effort here for the same reason the server treats it that way:
        // a corrupt or partial WAL should degrade to an empty graph with a
        // warning, not prevent the database from opening at all.
        runtime.block_on(async {
            match shared_services.graph_service.recover_all_graphs().await {
                Ok(()) => tracing::info!("EMBEDDED: graphs recovered from persistent storage"),
                Err(e) => tracing::warn!("EMBEDDED: graph recovery failed (continuing): {}", e),
            }
        });

        // Initialize RL planner if enabled
        let rl_policy_path = if config.enable_rl_planner {
            let policy_path = config.rl_policy_path.clone().unwrap_or_else(|| {
                // Default to first storage location + /rl_policy.json
                if let Some(loc) = config.storage_locations.first() {
                    format!("{}/rl_policy.json", loc.path.trim_end_matches('/'))
                } else {
                    "./data/rl_policy.json".to_string()
                }
            });

            // Initialize RL planner with default config
            let rl_config = proximadb::query::rl_planner::RLPlannerConfig::default();
            proximadb::query::rl_planner::init_rl_planner(rl_config);

            // Try to load existing policy
            if let Some(planner) = proximadb::query::rl_planner::get_rl_planner()
                && std::path::Path::new(&policy_path).exists()
            {
                runtime.block_on(async {
                    match planner.load_policy(&policy_path).await {
                        Ok(()) => {
                            tracing::info!("🎯 EMBEDDED: RL policy loaded from {}", policy_path);
                        }
                        Err(e) => {
                            tracing::debug!(
                                "EMBEDDED: No existing RL policy (starting fresh): {}",
                                e
                            );
                        }
                    }
                });
            }

            tracing::info!("🎯 EMBEDDED: RL Query Planner initialized");
            Some(policy_path)
        } else {
            tracing::debug!("EMBEDDED: RL Query Planner disabled");
            None
        };

        // Initialize metrics collector for embedded observability
        let metrics_collector = std::sync::Arc::new(EmbeddedMetricsCollector::new());
        tracing::debug!("EMBEDDED: Metrics collector initialized");

        // Initialize checkpoint manager for incremental persistence
        let catalog_manager = shared_services.catalog_manager.clone();
        runtime
            .block_on(async {
                let ddl_service = proximadb::services::DdlService::new(catalog_manager.clone());
                ddl_service
                    .execute(proximadb::services::DdlStatement::CreateNamespace {
                        namespace: vec!["default".to_string()],
                        if_not_exists: true,
                        properties: std::collections::HashMap::new(),
                    })
                    .await?;
                anyhow::Ok(())
            })
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(format!(
                    "Failed to initialize embedded catalog: {}",
                    e
                )))
            })?;

        let checkpoint_manager = std::sync::Arc::new(CheckpointManager::new(&base_path));
        runtime.block_on(async {
            if let Err(e) = checkpoint_manager.init().await {
                tracing::warn!("EMBEDDED: Failed to initialize checkpoint manager: {}", e);
            }
        });
        tracing::debug!("EMBEDDED: Checkpoint manager initialized");

        // TD-DSEFF-2: pre-#1743, `FlushExecutionCoordinator::engine_for`
        // unconditionally used a caller-supplied SST fallback, so every
        // embedded collection's flushes were routed through SST regardless
        // of its declared engine. Post-fix, new flushes correctly write the
        // declared engine's own format — but a pre-existing deployment's
        // non-SST-declared collection may have OLD SST-formatted data that
        // its declared engine's extension-filtered reader will never see.
        // One-shot, best-effort, log-only diagnostic on open: never blocks
        // startup or fails construction.
        runtime.block_on(async {
            let Ok(collections) = collection_port.list_collections(None).await else {
                return;
            };
            for collection in collections {
                let Ok(plan) =
                    proximadb::storage::flush_materializer::flush_plan_from_collection_meta(
                        &collection,
                    )
                else {
                    continue;
                };
                if plan.engine_type == proximadb::proto::proximadb_v1::StorageEngine::Sst as i32 {
                    continue;
                }
                let data_dir = format!("{}/{}/data", plan.base_location, plan.collection_object_id);
                let Ok(fs) = shared_services.filesystem_factory.get_filesystem(&data_dir) else {
                    continue;
                };
                let Ok(entries) = fs.list(&data_dir).await else {
                    continue; // no flushes yet — nothing to migrate
                };
                let file_names: Vec<String> = entries
                    .into_iter()
                    .filter(|e| !e.metadata.is_directory)
                    .map(|e| e.name)
                    .collect();
                if engine_data_mismatch(&file_names) {
                    let declared =
                        proximadb::proto::proximadb_v1::StorageEngine::try_from(plan.engine_type);
                    tracing::warn!(
                        "⚠️ EMBEDDED: collection '{}' declares engine {:?} but its data directory \
                         contains only .sst-formatted files (pre-#1743 flush) — this data is \
                         invisible to the declared engine's reader; re-ingest to migrate",
                        plan.collection_name,
                        declared,
                    );
                }
            }
        });

        Ok(Self {
            config,
            runtime,
            shared_services,
            catalog_manager,
            collection_port,
            rl_policy_path,
            metrics_collector,
            checkpoint_manager,
            lock_manager,
            leader_election,
            content_revision_incarnation: uuid::Uuid::new_v4().simple().to_string(),
            closed: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Convert EmbeddedConfig to the internal StorageConfig
    fn to_storage_config(config: &EmbeddedConfig) -> proximadb::core::config::StorageConfig {
        use proximadb::core::config::{StorageConfig, StorageLocation};

        // Convert storage locations
        let storage_locations: Vec<StorageLocation> = config
            .storage_locations
            .iter()
            .map(|loc| StorageLocation {
                url: loc.to_url(),
                weight: loc.weight,
                tags: loc.tags.clone(),
                io_budget: None,
            })
            .collect();

        // Convert metadata path to URL
        let metadata_url = if config.metadata_path.starts_with("file://") {
            config.metadata_path.clone()
        } else if config.metadata_path.starts_with('/') {
            format!("file://{}", config.metadata_path)
        } else {
            let abs_path = std::env::current_dir().map_or_else(
                |_| config.metadata_path.clone(),
                |p| p.join(&config.metadata_path).to_string_lossy().to_string(),
            );
            format!("file://{}", abs_path)
        };

        // Create optimized SST config for embedded mode
        // Adaptive bloom filters are automatically used by the SST writer
        let sst_config = proximadb::core::config::SstConfig {
            // Use default block size (256KB) optimized for NVMe
            block_size_kb: 256,
            // Enable bloom filters with adaptive sizing
            bloom_filter_config: Some(proximadb::core::bloom::BloomFilterConfig {
                enabled: true,
                strategy: proximadb::core::bloom::BloomStrategy::ByteAligned,
                bits_per_key: 10, // Base value, adaptive sizing will optimize this
                false_positive_rate: Some(0.01), // 1% FPR target
                expected_items: 10000,
                hash_algorithm: proximadb::core::bloom::HashAlgorithm::XXHash,
            }),
            // Enable LZ4 compression for better throughput
            compression: "lz4".to_string(),
            compression_level: 3,
            // Aggressive compaction for embedded mode
            compaction_threshold: 4, // Compact more frequently than server (5)
            compaction_strategy: "leveled".to_string(),
            // Optimize for embedded workloads
            cache_size_mb: (config.cache_size_mb / 4).max(32) as u64, // Reserve some cache for SST
            mmap_enabled: true,
            prefetch_enabled: true,
            prefetch_size_kb: 64,
            ..Default::default()
        };

        let prune_mode = match config.block_prune_mode.as_str() {
            "exact" => None,
            "sqrt" => Some(PruneModeConfig::Simple("sqrt".to_string())),
            "ratio" => Some(PruneModeConfig::Advanced(AdvancedPruneConfig {
                r#type: "ratio".to_string(),
                min_keep: Some(config.block_prune_min_keep),
                max_keep: Some(config.block_prune_max_keep),
                ratio: Some(config.block_prune_ratio),
            })),
            other => Some(PruneModeConfig::Simple(other.to_string())),
        };

        StorageConfig {
            storage_locations,
            metadata_url,
            cache_size_mb: config.cache_size_mb as u64,
            sst_config: Some(sst_config),
            // Enable bloom filters globally
            bloom_filter_config: Some(proximadb::core::bloom::BloomFilterConfig {
                enabled: true,
                strategy: proximadb::core::bloom::BloomStrategy::ByteAligned,
                bits_per_key: 10,
                false_positive_rate: Some(0.01),
                expected_items: 10000,
                hash_algorithm: proximadb::core::bloom::HashAlgorithm::XXHash,
            }),
            wal_config: proximadb::core::config::WriteBufferUserConfig {
                enable_wal: config.enable_wal,
                ..Default::default()
            },
            prune_mode,
            ..Default::default()
        }
    }

    /// Initialize the internal services asynchronously
    async fn init_services(
        storage_config: proximadb::core::config::StorageConfig,
    ) -> Result<
        (
            proximadb::network::multi_server::SharedServices,
            std::sync::Arc<dyn proximadb_runtime::CollectionPort>,
        ),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        use proximadb::storage::cache::orchestrator::CrossCacheOrchestrator;
        use std::sync::Arc;

        // Initialize hardware capabilities
        let _ = proximadb_hardware::hardware_capabilities();

        // Log hardware capabilities summary
        let _hw_summary =
            proximadb::core::hardware_capabilities::log_hardware_capabilities_summary();

        // Initialize global WAL manifest for proper WAL file cleanup
        // This is critical for embedded mode to avoid duplicate data
        Self::init_global_manifest(&storage_config).await?;

        // Create cache orchestrator
        let cache_budget_bytes = (storage_config.cache_size_mb * 1024 * 1024) as usize;
        let orchestrator = Arc::new(CrossCacheOrchestrator::new(cache_budget_bytes));

        // Create SharedServices - this sets up all the internal components
        let (shared_services, collection_service) =
            proximadb::network::multi_server::SharedServices::new(
                None, // No metrics collector needed for embedded
                &storage_config,
                Some(orchestrator),
                None, // No full config needed
                // Fused in-process: no Prometheus/billing/scrape background work.
                proximadb::network::multi_server::ServiceProfile::Embedded,
            )
            .await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            })?;

        // Set the global catalog for WAL path resolution. The catalog is the
        // sole authority, so WAL operations resolve collection paths through it.
        if let Some(catalog_manager) = collection_service.catalog_manager() {
            proximadb::storage::persistence::write_ahead_log::set_global_catalog(catalog_manager)
                .await;
            tracing::debug!("✅ Embedded: Global catalog set for WAL/recovery resolution");
        }

        let collection_port: std::sync::Arc<dyn proximadb_runtime::CollectionPort> =
            collection_service;
        Ok((shared_services, collection_port))
    }

    /// Initialize the global WAL manifest for proper WAL file cleanup
    async fn init_global_manifest(
        storage_config: &proximadb::core::config::StorageConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::storage::persistence::write_ahead_log::config::WALConfig;
        use proximadb::storage::persistence::write_ahead_log::manifest;

        // In embedded mode, always reset the manifest to support multiple database instances
        // with different storage locations in the same process
        if let Err(e) = manifest::reset().await {
            tracing::debug!(
                "Note: manifest reset returned error (may not have been initialized): {}",
                e
            );
        }

        // Build WAL config from storage config
        let mut wal_config = WALConfig::default();

        // Set up data directories from storage locations
        wal_config.multi_disk.data_directories = storage_config
            .storage_locations
            .iter()
            .map(|loc| loc.url.clone())
            .collect();

        // Set the manifest URL to the first storage location + /manifest
        if let Some(first_loc) = storage_config.storage_locations.first() {
            let base_url = first_loc.url.trim_end_matches('/');
            wal_config.global_manifest_url = Some(format!("{}/manifest", base_url));
        }

        // Initialize the global manifest
        match manifest::init(&wal_config).await {
            Ok(_) => {
                tracing::info!("🗂️  Initialized global WAL manifest for embedded mode");
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    "⚠️  Failed to initialize global WAL manifest: {}. WAL files may not be cleaned up after flush.",
                    e
                );
                // Don't fail - embedded mode can still work, just with duplicate data
                Ok(())
            }
        }
    }

    // ========================================================================
    // Multi-Process Coordination Methods
    // ========================================================================

    /// Check if write operations are allowed based on access mode
    ///
    /// Returns `Ok(())` if writes are allowed, or an error if not.
    fn check_write_access(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match self.config.access_mode {
            AccessMode::Exclusive => match &self.lock_manager {
                Some(lock) if lock.is_locked() => Ok(()),
                _ => Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "Database is closed: exclusive access lock is not held.",
                ))),
            },
            AccessMode::SharedRead => {
                // SharedRead mode never allows writes
                Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "Write operation not allowed in SharedRead mode. Open database with Exclusive or LeaderFollower mode for write access.",
                )))
            }
            AccessMode::LeaderFollower => {
                // LeaderFollower mode allows writes only if we are the leader
                match &self.leader_election {
                    Some(election) if election.is_leader() => Ok(()),
                    Some(_) => Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "Write operation not allowed: this node is a follower. Only the leader can write.",
                    ))),
                    None => Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Leader election not initialized in LeaderFollower mode.",
                    ))),
                }
            }
        }
    }

    /// Check if this database instance can perform write operations
    ///
    /// Returns `true` if the current access mode and leader status allow writes.
    pub fn can_write(&self) -> bool {
        self.check_write_access().is_ok()
    }

    /// Get the current access mode
    pub fn access_mode(&self) -> AccessMode {
        self.config.access_mode
    }

    /// Check if this node is the leader (only relevant in LeaderFollower mode)
    ///
    /// Returns `true` if in LeaderFollower mode and this node is the leader,
    /// or if in Exclusive mode. Returns `false` in SharedRead mode.
    pub fn is_leader(&self) -> bool {
        match self.config.access_mode {
            AccessMode::Exclusive => true,
            AccessMode::SharedRead => false,
            AccessMode::LeaderFollower => {
                self.leader_election.as_ref().is_some_and(|e| e.is_leader())
            }
        }
    }

    /// Get the current leader ID (only relevant in LeaderFollower mode)
    ///
    /// Returns `None` if not in LeaderFollower mode or if leader is unknown.
    pub fn leader_id(&self) -> Option<String> {
        self.leader_election.as_ref().and_then(|e| e.leader_id())
    }

    // ========================================================================
    // Collection Management Methods
    // ========================================================================

    /// Create a new collection
    pub fn create_collection(
        &self,
        name: &str,
        dimension: u32,
        engine: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before creating collection
        self.check_write_access()?;
        use proximadb::proto::proximadb_v1::{
            CollectionConfig, CompressionAlgorithm, StorageConfig,
        };

        let requested_engine = engine.unwrap_or(&self.config.default_engine);
        let storage_engine = proximadb::core::conversions::parse_storage_engine(requested_engine)
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
            format!("Unknown storage engine: {}", requested_engine).into()
        })?;

        let collection_config = CollectionConfig {
            name: name.to_string(),
            dimension,
            distance_metric: Some(proximadb::proto::proximadb_v1::DistanceMetric::Cosine as i32),
            storage_engine: Some(storage_engine as i32),
            storage_config: Some(StorageConfig {
                compression: Some(CompressionAlgorithm::CompressionLz4 as i32),
                ..Default::default()
            }),
            index_configs: vec![],
            ..Default::default()
        };

        self.runtime.block_on(async {
            let response = self
                .shared_services
                .collection_service
                .create_collection(&collection_config)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            // Check if the collection service returned an error in the response
            if !response.success {
                let error_msg = response
                    .error_code
                    .unwrap_or_else(|| "Unknown error".to_string());
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Failed to create collection '{}': {}", name, error_msg),
                ))
                    as Box<dyn std::error::Error + Send + Sync>);
            }

            // Register the collection in the global cache for EventLog consumer
            // This enables AXIS index building when flush events occur
            if let Some(collection) = response.collection {
                proximadb::services::events::log::register_collection_in_cache(
                    std::sync::Arc::new(collection),
                );
                tracing::info!(
                    "📦 EMBEDDED: Registered collection '{}' in global cache for AXIS indexing",
                    name
                );
            }

            Ok(())
        })
    }

    /// Create a collection with explicit index configuration
    ///
    /// # Arguments
    /// * `name` - Collection name
    /// * `dimension` - Vector dimension
    /// * `engine` - Optional storage engine ("sst", "helix", "viper", "swift", "nova", "raptor", "tst")
    /// * `index_type` - Index type: "hnsw", "ivf", "flat", "lsh", or "none"
    ///
    /// # Example
    /// ```rust,ignore
    /// db.create_collection_with_index("my_collection", 768, Some("sst"), "hnsw")?;
    /// ```
    pub fn create_collection_with_index(
        &self,
        name: &str,
        dimension: u32,
        engine: Option<&str>,
        index_type: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::proto::proximadb_v1::{
            CollectionConfig, CompressionAlgorithm, HnswConfig, IndexConfig, IndexingAlgorithm,
            IvfConfig, LshConfig, StorageConfig,
        };

        let requested_engine = engine.unwrap_or(&self.config.default_engine);
        let storage_engine = proximadb::core::conversions::parse_storage_engine(requested_engine)
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
            format!("Unknown storage engine: {}", requested_engine).into()
        })?;

        // Build index config based on requested type
        let index_configs = match index_type.to_lowercase().as_str() {
            "hnsw" => {
                vec![IndexConfig {
                    index_name: "hnsw_index".to_string(),
                    algorithm: IndexingAlgorithm::Hnsw as i32,
                    enabled: Some(true),
                    hnsw_config: Some(HnswConfig {
                        m: Some(16),
                        ef_construction: Some(200),
                        ef_search: Some(50),
                        max_partition_size: Some(100_000),
                        adaptive_parameters: Some(true),
                        use_simd: Some(true),
                        memory_limit_mb: Some(512),
                        lazy_loading: Some(false),
                    }),
                    ..Default::default()
                }]
            }
            "ivf" => {
                // Use sqrt(n) clusters as default for IVF
                let n_clusters = ((dimension as f64).sqrt() * 4.0) as u32;
                vec![IndexConfig {
                    index_name: "ivf_index".to_string(),
                    algorithm: IndexingAlgorithm::Ivf as i32,
                    enabled: Some(true),
                    ivf_config: Some(IvfConfig {
                        n_lists: Some(n_clusters.max(64)),
                        n_probe: Some(10),
                        ..Default::default()
                    }),
                    ..Default::default()
                }]
            }
            "lsh" => {
                vec![IndexConfig {
                    index_name: "lsh_index".to_string(),
                    algorithm: IndexingAlgorithm::Lsh as i32,
                    enabled: Some(true),
                    lsh_config: Some(LshConfig {
                        n_hash_tables: Some(10),
                        n_hash_functions: Some(8),
                        ..Default::default()
                    }),
                    ..Default::default()
                }]
            }
            "flat" | "none" => {
                // No index - brute force search
                vec![]
            }
            other => {
                return Err(format!(
                    "Unknown index type: {}. Supported: hnsw, ivf, lsh, flat, none",
                    other
                )
                .into());
            }
        };

        let collection_config = CollectionConfig {
            name: name.to_string(),
            dimension,
            storage_engine: Some(storage_engine as i32),
            storage_config: Some(StorageConfig {
                compression: Some(CompressionAlgorithm::CompressionLz4 as i32),
                ..Default::default()
            }),
            index_configs,
            ..Default::default()
        };

        self.runtime.block_on(async {
            let response = self.shared_services.collection_service
                .create_collection(&collection_config)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            if !response.success {
                let error_msg = response.error_code.unwrap_or_else(|| "Unknown error".to_string());
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Failed to create collection '{}': {}", name, error_msg)
                )) as Box<dyn std::error::Error + Send + Sync>);
            }

            // Register the collection in the global cache for EventLog consumer
            if let Some(collection) = response.collection {
                proximadb::services::events::log::register_collection_in_cache(
                    std::sync::Arc::new(collection)
                );
                tracing::info!(
                    "📦 EMBEDDED: Registered collection '{}' with index type '{}' for AXIS indexing",
                    name, index_type
                );
            }

            Ok(())
        })
    }

    /// Delete a collection
    pub fn delete_collection(
        &self,
        name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before deleting collection
        self.check_write_access()?;

        self.runtime.block_on(async {
            let response = self
                .shared_services
                .collection_service
                .delete_collection(name)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            // Check if the collection service returned an error in the response
            if !response.success {
                let error_msg = response
                    .error_code
                    .unwrap_or_else(|| "Unknown error".to_string());
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Failed to delete collection '{}': {}", name, error_msg),
                ))
                    as Box<dyn std::error::Error + Send + Sync>);
            }
            Ok(())
        })
    }

    /// Build the optional governance `TenantContext` from the configured tenant.
    ///
    /// `None` (the default) means legacy single-tenant behavior — inserts take the
    /// plain path and records keep an empty `tenant_id`. When a tenant is configured
    /// (`EmbeddedConfig::with_tenant`), inserts route through the same tenant-scoped
    /// path the networked server uses, making the boundary contract uniform
    /// (co-design tenet 3). Non-breaking: embedded vector storage is collection-keyed.
    fn embedded_tenant_context(
        &self,
    ) -> Option<proximadb::storage::tenant::context::TenantContext> {
        self.config
            .tenant_id
            .clone()
            .map(proximadb::storage::tenant::context::TenantContext::for_tenant_id)
    }

    /// Insert vectors into a collection
    ///
    /// Supports batched inserts for better performance. Internally, vectors are
    /// processed in batches of up to 1024 for optimal throughput.
    pub fn insert(
        &self,
        collection: &str,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadata: Option<Vec<std::collections::HashMap<String, serde_json::Value>>>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before inserting
        self.check_write_access()?;

        // Start timing for metrics
        let start = std::time::Instant::now();
        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);

        let records = Self::build_embedded_vector_records(ids, vectors, metadata, now_ns);

        let count = records.len();

        // Governance contract (co-design tenet 3): when a tenant is configured,
        // route through the tenant-scoped insert the networked path uses; otherwise
        // keep the legacy plain path byte-identical (single-tenant default).
        let tenant_ctx = self.embedded_tenant_context();
        let result = self.runtime.block_on(async {
            let vos = &self.shared_services.vector_operations_service;
            let result = match tenant_ctx.as_ref() {
                Some(ctx) => {
                    vos.insert_batch_with_tenant_context(collection, records, Some(ctx))
                        .await
                }
                None => vos.insert_batch(collection, records).await,
            }
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            })?;

            if !result.success {
                return Err(Box::new(std::io::Error::other(
                    result
                        .errors
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "Vector batch insert failed".to_string()),
                ))
                    as Box<dyn std::error::Error + Send + Sync>);
            }

            Ok(result.metrics.successful_count.max(0) as usize)
        });

        // Record insert latency and count
        let elapsed_us = start.elapsed().as_micros() as u64;
        self.metrics_collector
            .record_insert_us(elapsed_us, count as u64);

        // Record error if insert failed
        if result.is_err() {
            self.metrics_collector.record_error();
        }

        result
    }

    fn build_embedded_vector_records(
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadata: Option<Vec<std::collections::HashMap<String, serde_json::Value>>>,
        now_ns: i64,
    ) -> Vec<proximadb_records::ProximaRecord> {
        ids.into_iter()
            .zip(vectors)
            .enumerate()
            .map(|(i, (oid, values))| {
                let mut props = proximadb_records::ProximaTree::new();
                if let Some(meta_slice) = metadata.as_ref().and_then(|m| m.get(i)) {
                    for (k, v) in meta_slice {
                        props.insert(
                            k.clone(),
                            proximadb_records::ProximaTreeNode::Value(json_to_proxima_value(
                                v.clone(),
                            )),
                        );
                    }
                }
                let dim = values.len() as u32;
                proximadb_records::ProximaRecord {
                    oid,
                    embeddings: vec![proximadb_records::EmbeddingCell {
                        model_id: "default".to_string(),
                        modality: "vector".to_string(),
                        dim,
                        values: proximadb_records::EmbeddingValues::Fp32(values),
                        ..Default::default()
                    }],
                    props,
                    created_at_ns: now_ns,
                    updated_at_ns: now_ns,
                    record_version: 1,
                    ..Default::default()
                }
            })
            .collect()
    }

    /// Insert canonical records into a collection without lowering through
    /// legacy ids/vectors/metadata transport at the language binding boundary.
    pub fn insert_proxima_records(
        &self,
        collection: &str,
        records: Vec<proximadb_records::ProximaRecord>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        self.check_write_access()?;

        let start = std::time::Instant::now();
        let count = records.len();

        // Governance contract (co-design tenet 3): when a tenant is configured,
        // route through the tenant-scoped insert the networked path uses; otherwise
        // keep the legacy plain path byte-identical (single-tenant default).
        let tenant_ctx = self.embedded_tenant_context();
        let result = self.runtime.block_on(async {
            let vos = &self.shared_services.vector_operations_service;
            let result = match tenant_ctx.as_ref() {
                Some(ctx) => {
                    vos.insert_batch_with_tenant_context(collection, records, Some(ctx))
                        .await
                }
                None => vos.insert_batch(collection, records).await,
            }
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            })?;

            if !result.success {
                return Err(Box::new(std::io::Error::other(
                    result
                        .errors
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "Record batch insert failed".to_string()),
                ))
                    as Box<dyn std::error::Error + Send + Sync>);
            }

            Ok(result.metrics.successful_count.max(0) as usize)
        });

        let elapsed_us = start.elapsed().as_micros() as u64;
        self.metrics_collector
            .record_insert_us(elapsed_us, count as u64);

        if result.is_err() {
            self.metrics_collector.record_error();
        }

        result
    }

    /// Search for similar vectors
    pub fn search(
        &self,
        collection: &str,
        query_vector: Vec<f32>,
        top_k: usize,
        _filter: Option<&str>,
    ) -> Result<Vec<EmbeddedSearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        self.search_with_mode(collection, query_vector, top_k, _filter, None)
    }

    /// Search for similar vectors with explicit search mode
    ///
    /// # Arguments
    /// * `collection` - Name of the collection to search
    /// * `query_vector` - Query vector
    /// * `top_k` - Number of results to return
    /// * `filter` - Optional filter expression
    /// * `search_mode` - Search mode: "exact", "approximate", or "adaptive"
    ///   - "exact": 100% recall, searches all partitions (default)
    ///   - "approximate": Faster search using IVF-style partition pruning
    ///   - "approximate:N": Approximate with explicit nprobe value
    ///   - "adaptive": Auto-select based on dataset size
    ///   - "adaptive:N": Adaptive with explicit threshold
    ///
    /// # Example
    /// ```rust,ignore
    /// // Exact search (100% recall)
    /// let results = db.search_with_mode("my_collection", vec![0.1; 768], 10, None, Some("exact"))?;
    ///
    /// // Approximate search (faster, ~95% recall)
    /// let results = db.search_with_mode("my_collection", vec![0.1; 768], 10, None, Some("approximate"))?;
    ///
    /// // Approximate with custom nprobe
    /// let results = db.search_with_mode("my_collection", vec![0.1; 768], 10, None, Some("approximate:5"))?;
    /// ```
    pub fn search_with_mode(
        &self,
        collection: &str,
        query_vector: Vec<f32>,
        top_k: usize,
        filter: Option<&str>,
        search_mode: Option<&str>,
    ) -> Result<Vec<EmbeddedSearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        // Parse filter into (field, value) predicate pairs and push them to the storage
        // layer via VectorSearchRequest.filters — the SST engine applies predicate pushdown
        // during ANN search, so top_k already reflects post-filter cardinality.
        let predicates = filter
            .map(proximadb_embedded_common::parse_vector_filter)
            .unwrap_or_default();
        let fetch_k = top_k; // No over-fetch: filter is enforced at data layer by SST engine

        if matches!(search_mode, None | Some("exact")) {
            let start = std::time::Instant::now();

            // Build SqlValue filter map for the query adapter.
            let filter_map: std::collections::HashMap<
                String,
                proximadb::proto::proximadb_v1::SqlValue,
            > = predicates
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        proximadb::proto::proximadb_v1::SqlValue {
                            value: Some(
                                proximadb::proto::proximadb_v1::sql_value::Value::StringValue(
                                    v.clone(),
                                ),
                            ),
                        },
                    )
                })
                .collect();

            let result = self.runtime.block_on(async {
                let request = proximadb::proto::proximadb_v1::VectorSearchRequest {
                    collection_id: collection.to_string(),
                    queries: vec![proximadb::proto::proximadb_v1::SearchQuery {
                        vector: query_vector,
                        filters: filter_map,
                        advanced_filter: None,
                    }],
                    top_k: fetch_k as u32,
                    include_fields: None,
                    search_params: None,
                    distance_metric_override: None,
                    search_optimization: None,
                };

                let response = self
                    .shared_services
                    .query_adapter()
                    .vector_search(request)
                    .await
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(e.to_string()))
                    })?;

                let results = response
                    .results
                    .map(|result_set| {
                        result_set
                            .results
                            .into_iter()
                            .map(|r| EmbeddedSearchResult {
                                id: r.id,
                                score: r.score as f32,
                                metadata: r
                                    .metadata
                                    .into_iter()
                                    .map(|(k, v)| {
                                        // gRPC path: v is SqlValue — convert to ProximaValue first
                                        use proximadb::core::search::results::sql_value_to_proxima_value;
                                        let val_str = proxima_value_to_string(
                                            sql_value_to_proxima_value(v),
                                        );
                                        (k, val_str)
                                    })
                                    .collect(),
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                Ok(results)
            });

            let elapsed_us = start.elapsed().as_micros() as u64;
            self.metrics_collector.record_search_us(elapsed_us);
            if result.is_err() {
                self.metrics_collector.record_error();
            }
            return result;
        }

        use proximadb::core::search::SearchMode;
        use proximadb::services::operations::vectors::UnifiedSearchConfig;

        // Start timing for metrics
        let start = std::time::Instant::now();

        // Parse search mode string into SearchMode enum
        let mode = match search_mode {
            None | Some("exact") => SearchMode::Exact,
            Some("approximate") => SearchMode::Approximate { nprobe: None },
            Some(s) if s.starts_with("approximate:") => {
                let nprobe_str = s.strip_prefix("approximate:").unwrap_or("0");
                let nprobe = nprobe_str.parse::<usize>().ok();
                SearchMode::Approximate { nprobe }
            }
            Some("adaptive") => SearchMode::Adaptive { threshold: 10000 }, // Default threshold
            Some(s) if s.starts_with("adaptive:") => {
                let threshold_str = s.strip_prefix("adaptive:").unwrap_or("10000");
                let threshold = threshold_str.parse::<usize>().unwrap_or(10000);
                SearchMode::Adaptive { threshold }
            }
            Some(_) => SearchMode::Exact, // Default fallback
        };

        // Create config with the search mode
        let config = UnifiedSearchConfig {
            search_mode: mode,
            ..Default::default()
        };

        let result = self.runtime.block_on(async {
            let results = self
                .shared_services
                .vector_operations_service
                .unified_search_native(
                    collection,
                    query_vector,
                    fetch_k,
                    None,
                    Some(config),
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(
                        SystemReadReason::Statistics,
                        "embedded [CLIENT-PLACEHOLDER]",
                    ),
                )
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            // Convert to embedded EmbeddedSearchResult format
            Ok(results
                .into_iter()
                .map(|r| EmbeddedSearchResult {
                    id: r.id,
                    score: r.score,
                    metadata: r
                        .metadata
                        .into_iter()
                        .map(|(k, v)| {
                            let val_str = proxima_value_to_string(v);
                            (k, val_str)
                        })
                        .collect(),
                })
                .collect())
        });

        // Record search latency
        let elapsed_us = start.elapsed().as_micros() as u64;
        self.metrics_collector.record_search_us(elapsed_us);

        // Record error if search failed
        if result.is_err() {
            self.metrics_collector.record_error();
        }

        result
    }

    /// Create a streaming search iterator for memory-efficient large result set processing
    ///
    /// This method returns an iterator that yields results in batches, allowing for
    /// memory-efficient processing of large result sets. Results are fetched in
    /// configurable batch sizes, providing backpressure control.
    ///
    /// # Arguments
    /// * `collection` - Name of the collection to search
    /// * `query_vector` - Query vector for similarity search
    /// * `top_k` - Total number of results to return
    /// * `batch_size` - Number of results per batch (default: 100)
    ///
    /// # Returns
    /// An iterator that yields `Result<Vec<StreamingSearchResult>, Error>` for each batch
    ///
    /// # Example
    /// ```rust,ignore
    /// // Create streaming iterator for large result set
    /// let iterator = db.search_streaming("my_collection", query_vector, 10000, 100)?;
    ///
    /// // Process results in batches
    /// for batch_result in iterator {
    ///     let batch = batch_result?;
    ///     for result in batch {
    ///         println!("Found: {} (score: {})", result.id, result.score);
    ///     }
    /// }
    /// ```
    pub fn search_streaming(
        &self,
        collection: &str,
        query_vector: Vec<f32>,
        top_k: usize,
        batch_size: usize,
    ) -> Result<EmbeddedSearchIterator, Box<dyn std::error::Error + Send + Sync>> {
        self.search_streaming_with_config(
            collection,
            query_vector,
            top_k,
            StreamingSearchConfig::default().with_batch_size(batch_size),
        )
    }

    /// Create a streaming search iterator with full configuration options
    ///
    /// # Arguments
    /// * `collection` - Name of the collection to search
    /// * `query_vector` - Query vector for similarity search
    /// * `top_k` - Total number of results to return
    /// * `config` - Streaming search configuration
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = StreamingSearchConfig::default()
    ///     .with_batch_size(50)
    ///     .with_buffer_size(500)
    ///     .with_search_mode("approximate");
    ///
    /// let iterator = db.search_streaming_with_config("my_collection", query_vector, 10000, config)?;
    ///
    /// for batch_result in iterator {
    ///     let batch = batch_result?;
    ///     // Process batch...
    /// }
    /// ```
    pub fn search_streaming_with_config(
        &self,
        collection: &str,
        query_vector: Vec<f32>,
        top_k: usize,
        config: StreamingSearchConfig,
    ) -> Result<EmbeddedSearchIterator, Box<dyn std::error::Error + Send + Sync>> {
        use streaming::StreamingSearchExecutor;
        use tokio::sync::mpsc;

        // Calculate buffer size based on config
        let buffer_size = (config.buffer_size / config.batch_size).max(2);

        // Create channel for result batches
        let (sender, receiver) = mpsc::channel(buffer_size);

        // Create executor
        let executor = StreamingSearchExecutor::new(
            collection.to_string(),
            query_vector,
            top_k,
            config.clone(),
        );

        // Get vector operations service
        let vector_operations =
            std::sync::Arc::clone(&self.shared_services.vector_operations_service);

        // Spawn async task to execute search and send results
        let runtime_handle = self.runtime.handle().clone();
        runtime_handle.spawn(async move {
            executor.execute(vector_operations, sender).await;
        });

        // Create and return the iterator
        Ok(EmbeddedSearchIterator::new(
            receiver,
            config,
            top_k,
            runtime_handle,
        ))
    }

    /// Get collection information
    pub fn get_collection(
        &self,
        name: &str,
    ) -> Result<Option<EmbeddedCollectionInfo>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let collection = self
                .collection_port
                .get_collection(name, None)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(collection.map(|c| {
                let config = c.config.unwrap_or_default();
                EmbeddedCollectionInfo {
                    name: config.name,
                    dimension: config.dimension,
                    vector_count: c.stats.map_or(0, |s| s.vector_count as u64),
                    engine: collection_engine_name(config.storage_engine),
                    disk_usage_bytes: 0, // Deferred: Calculate actual disk usage
                }
            }))
        })
    }

    /// Get the shared services
    pub fn shared_services(&self) -> &proximadb::network::multi_server::SharedServices {
        &self.shared_services
    }

    /// Get the Tokio runtime
    pub fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }

    /// List all collections
    pub fn list_collections(
        &self,
    ) -> Result<Vec<EmbeddedCollectionInfo>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let collections = self.collection_port.list_collections(None).await.map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                },
            )?;

            Ok(collections
                .into_iter()
                .map(|c| {
                    let config = c.config.unwrap_or_default();
                    EmbeddedCollectionInfo {
                        name: config.name,
                        dimension: config.dimension,
                        vector_count: c.stats.map_or(0, |s| s.vector_count as u64),
                        engine: collection_engine_name(config.storage_engine),
                        disk_usage_bytes: 0, // Deferred: Calculate actual disk usage
                    }
                })
                .collect())
        })
    }

    // ========================================================================
    // Connector-binding surfaces (TD-097 Spark JNI + TD-099 cursor scan)
    //
    // These three methods (`get_collection_schema`, `scan_records`,
    // `plan_partitions`) are the in-process equivalent of the
    // REST/Flight handlers the other connectors dial. The Spark JNI
    // cdylib calls them directly; future NodeJS/C-FFI DataFrame surfaces
    // can reuse the same methods. They are intentionally sync (each
    // wraps `self.runtime.block_on`) so JNI / FFI callers don't need to
    // know about tokio.
    // ========================================================================

    /// Return the named collection's schema as a JSON value matching
    /// the OpenAPI `SchemaDefinition` shape. Used by Spark JNI's
    /// `getTableSchema` and by future embedded callers that need to
    /// introspect a collection without going through the REST handler.
    /// Returns `Ok(None)` when the collection doesn't exist.
    pub fn get_collection_schema(
        &self,
        collection: &str,
    ) -> Result<Option<serde_json::Value>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let opt = self
                .collection_port
                .get_collection(collection, None)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;
            Ok(opt.map(|c| {
                let cfg = c.config.unwrap_or_default();
                serde_json::json!({
                    "name": cfg.name,
                    "dimension": cfg.dimension,
                    "distance_metric": cfg.distance_metric,
                    "storage_engine": cfg.storage_engine,
                    "vector_count": c.stats.map_or(0, |s| s.vector_count),
                })
            }))
        })
    }

    /// Scan one page of records via the same canonical pipeline used
    /// by the REST `scanRecords` handler — `UnifiedHandlers::handle_record_scan_for_tenant`
    /// then [`apply_scan_cursor`](proximadb::services::scan_cursor::apply_scan_cursor)
    /// for stable-sorted, cursor-paginated output. Returns
    /// `(records, next_cursor)`: the cursor is `None` when the page is
    /// short (definite end-of-scan).
    ///
    /// `cursor`: opaque continuation token from the previous call; pass
    /// `None` to start at the beginning. Stale cursors (>24h) and
    /// collection-mismatched cursors error out via
    /// [`ScanCursorDecodeError`](proximadb::services::scan_cursor::ScanCursorDecodeError).
    pub fn scan_records(
        &self,
        collection: &str,
        cursor: Option<String>,
        limit: usize,
    ) -> Result<
        (Vec<proximadb_records::ProximaRecord>, Option<String>),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        self.scan_records_filtered(collection, cursor, limit, None)
    }

    /// Paginated scan with an optional pushed-down `filter` (e.g. a lowered Spark
    /// predicate) applied INSIDE the WAL scan before the limit, via the same
    /// `scan_records_paginated` seam. `filter` is evaluated against each record's
    /// property tree.
    pub fn scan_records_filtered(
        &self,
        collection: &str,
        cursor: Option<String>,
        limit: usize,
        filter: Option<&proximadb::core::search::FilterExpression>,
    ) -> Result<
        (Vec<proximadb_records::ProximaRecord>, Option<String>),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        self.runtime.block_on(async {
            let tenant_id = self
                .config
                .tenant_id
                .as_deref()
                .filter(|tenant| !tenant.is_empty());
            let collection_record = self
                .collection_port
                .get_collection(collection, tenant_id)
                .await
                .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(error.to_string()))
                })?
                .ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("Collection '{collection}' not found"),
                    ))
                })?;
            let canonical_collection_id = collection_record.id;
            let revision_tenant =
                tenant_id.unwrap_or(proximadb::catalog::DEFAULT_CORPUS_VERSION_TENANT);
            let revision = proximadb::catalog::CorpusVersionRegistry::global()
                .content_snapshot(
                    revision_tenant,
                    &canonical_collection_id,
                    &self.content_revision_incarnation,
                )
                .await;
            let inbound_cursor = match cursor.as_deref() {
                Some(raw) if !raw.is_empty() => {
                    let decoded = proximadb::services::scan_cursor::ScanCursor::decode(
                        raw,
                        &canonical_collection_id,
                        now_ns,
                    )?;
                    decoded.validate_content_revision(revision.revision, &revision.token)?;
                    Some(decoded)
                }
                _ => None,
            };

            // TD-099(3d): push cursor + limit into the WAL streaming layer via
            // the same VectorOperationsService pathway used by
            // `insert_proxima_records`, so writes and reads agree on the
            // collection-id key. Embedded mode is single-tenant (no tenant
            // predicate); the optional `filter` rides the same predicate seam.
            let (page, next) = self
                .shared_services
                .vector_operations_service
                .scan_records_paginated(
                    &canonical_collection_id,
                    &canonical_collection_id,
                    inbound_cursor.as_ref(),
                    limit,
                    true,
                    true,
                    None,
                    filter,
                    now_ns,
                )
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            let revision_after = proximadb::catalog::CorpusVersionRegistry::global()
                .content_snapshot(
                    revision_tenant,
                    &canonical_collection_id,
                    &self.content_revision_incarnation,
                )
                .await;
            if revision_after != revision {
                return Err(Box::new(std::io::Error::other(
                    "collection changed while scanning; restart pagination",
                ))
                    as Box<dyn std::error::Error + Send + Sync>);
            }

            let next_str = match next {
                Some(mut c) => {
                    c.stamp_content_revision(revision.revision, revision.token);
                    Some(
                        c.encode()
                            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                Box::new(std::io::Error::other(e))
                            })?,
                    )
                }
                None => None,
            };

            Ok((page, next_str))
        })
    }

    /// Plan input partitions for parallel reads. Returns a list of
    /// [`SparkInputPartition`](crate::spark::SparkInputPartition)
    /// — each carries `FileSplit`s that subsequent
    /// `create_partition_reader` calls consume.
    ///
    /// **Single-partition fallback** (TD-097 first-slice): this
    /// implementation returns exactly ONE partition spanning the whole
    /// collection, regardless of the requested `num_partitions`. It's
    /// correct-but-not-parallel — readers will still drain the entire
    /// collection via cursor pagination inside the single partition.
    /// Real AXIS/SST shard-aware split generation (so Spark gets actual
    /// parallelism per executor) is a follow-up TD that wires this
    /// method through the existing `FileSplit` infrastructure that
    /// Trino already uses.
    pub fn plan_partitions(
        &self,
        collection: &str,
        _num_partitions: u32,
    ) -> Result<Vec<crate::spark::SparkInputPartition>, Box<dyn std::error::Error + Send + Sync>>
    {
        use crate::spark::SparkInputPartition;
        use proximadb::storage::formats::FileSplit;

        // Validate the collection exists before producing splits, so
        // a typo on the JNI side fails fast (vs. emitting a "ghost"
        // split that yields zero rows).
        match self.get_collection_schema(collection)? {
            None => Err(Box::new(std::io::Error::other(format!(
                "collection '{collection}' not found"
            )))),
            Some(_) => Ok(vec![SparkInputPartition::from_splits(
                0,
                vec![FileSplit::whole_collection(collection)],
            )]),
        }
    }

    /// Flush all pending writes to disk
    ///
    /// This forces all in-memory data (memtable/WAL) to be persisted to storage engine files.
    /// It also triggers compaction to consolidate data into SST files for durability.
    pub fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before flushing
        self.check_write_access()?;

        // Start timing for metrics
        let start = std::time::Instant::now();

        // Explicitly typed as Result<u64, E> to capture bytes written for metrics
        let result: Result<u64, Box<dyn std::error::Error + Send + Sync>> =
            self.runtime.block_on(async {
                use proximadb::storage::persistence::write_ahead_log::get_global_write_buffer_behavior;

                tracing::info!("🛑 EMBEDDED: Flushing all unflushed data to storage engines...");

                // Get the global write buffer to access unflushed data
                let write_buffer = match get_global_write_buffer_behavior() {
                    Some(wb) => wb,
                    None => {
                        tracing::info!(
                            "📋 EMBEDDED: No global write buffer initialized, nothing to flush"
                        );
                        return Ok(0);
                    }
                };

                // Get list of collections with unflushed data
                let collections_to_flush =
                    write_buffer.list_collections_with_unflushed_data().await;
                if collections_to_flush.is_empty() {
                    tracing::info!("📋 EMBEDDED: No collections have unflushed data");
                    return Ok(0);
                }

                tracing::info!(
                    "🔄 EMBEDDED: Found {} collections with unflushed data: {:?}",
                    collections_to_flush.len(),
                    collections_to_flush
                );

                let mut total_vectors_flushed = 0u64;
                let mut total_bytes_written = 0u64;
                let mut collections_flushed = 0usize;
                let mut failed_collections: Vec<(String, String)> = Vec::new();

                // Flush each collection directly using its configured storage engine
                // Idempotency is handled at the batch level by get_unflushed_batches() and clear_flushed()
                for collection_id in &collections_to_flush {
                    tracing::info!("🔄 EMBEDDED: Flushing collection '{}'", collection_id);

                    // Resolve the catalog-owned L1 object id, L2 addressing
                    // composite, engine, and dimension once. Flush fails closed
                    // when the WAL key has no catalog object; silently inventing
                    // a path would retire WAL without publishing readable data.
                    let collection_metadata = match self
                        .collection_port
                        .get_collection(collection_id, None)
                        .await
                    {
                        Ok(Some(metadata)) => metadata,
                        Ok(None) => {
                            failed_collections.push((
                                collection_id.clone(),
                                "no catalog metadata".to_string(),
                            ));
                            continue;
                        }
                        Err(error) => {
                            failed_collections.push((collection_id.clone(), error.to_string()));
                            continue;
                        }
                    };

                    // Materialize this collection's unflushed batches via the shared
                    // helper (engine factory + flush() funnel + WAL cleanup + A6 fence).
                    // Embedded is single-pod: it passes no fence and reuses the running
                    // unified SST engine as the create-failure fallback.
                    let plan = match proximadb::storage::flush_materializer::flush_plan_from_collection_meta(
                        &collection_metadata,
                    ) {
                        Ok(plan) => plan,
                        Err(error) => {
                            failed_collections.push((collection_id.clone(), error.to_string()));
                            continue;
                        }
                    };
                    let collection_name = plan.collection_name.clone();
                    let fallback_engine = self
                        .shared_services
                        .vector_operations_service
                        .unified_engine();
                    // free_wal=true preserves embedded's existing behavior (delete the
                    // WAL after flush — no 2× overhead). Its cold-read recall gap is the
                    // same dependent TD as the server's.
                    #[cfg(feature = "axis")]
                    let axis_manager = self
                        .shared_services
                        .vector_operations_service
                        .axis_index_manager();
                    #[cfg(feature = "axis")]
                    let axis_arg: Option<&proximadb::index::AxisManager> = Some(&axis_manager);
                    // `axis` off (exact-scan embedded): no in-memory projection to reap
                    // — the durable segment is the source of truth.
                    #[cfg(not(feature = "axis"))]
                    let axis_arg: Option<&()> = None;
                    match proximadb::storage::flush_materializer::materialize_collection(
                        &write_buffer,
                        &plan,
                        None,
                        Some(fallback_engine),
                        true,
                        axis_arg,
                    )
                    .await
                    {
                        Ok(Some(outcome)) => {
                            total_vectors_flushed += outcome.entries_flushed;
                            total_bytes_written += outcome.bytes;
                            collections_flushed += 1;

                            // Update collection stats (the optimizer skips index lookup if
                            // it sees 0 vectors) + invalidate the cache so the next search
                            // loads fresh stats. Stats use the submitted count (parity with
                            // the prior embedded behavior).
                            if let Err(e) = self
                                .shared_services
                                .collection_service
                                .update_stats(
                                    &collection_name,
                                    outcome.vectors_submitted as i64,
                                    outcome.bytes as i64,
                                )
                                .await
                            {
                                tracing::warn!(
                                    "⚠️ EMBEDDED: Failed to update stats for '{}': {}",
                                    collection_id,
                                    e
                                );
                            } else {
                                self.shared_services
                                    .vector_operations_service
                                    .invalidate_collection_cache(&collection_name);
                            }

                            tracing::info!(
                                "✅ EMBEDDED: Flushed collection '{}': {} vectors, {} bytes",
                                collection_id,
                                outcome.entries_flushed,
                                outcome.bytes
                            );
                        }
                        Ok(None) => {
                            tracing::debug!(
                                "📋 EMBEDDED: Collection '{}' has no unflushed batches",
                                collection_id
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                "❌ EMBEDDED: Failed to flush collection '{}': {}",
                                collection_id,
                                e
                            );
                            failed_collections.push((collection_id.clone(), e.to_string()));
                        }
                    }
                }

                tracing::info!(
                    "🛑 EMBEDDED: Flush complete - {} collections, {} vectors, {} bytes{}",
                    collections_flushed,
                    total_vectors_flushed,
                    total_bytes_written,
                    if failed_collections.is_empty() {
                        String::new()
                    } else {
                        format!(", {} failures", failed_collections.len())
                    }
                );

                // NOTE: We removed the redundant force_flush_all() call here.
                // The per-collection storage_engine.flush() above already flushes all data.
                // Calling force_flush_all() caused duplicate SST files (2x data overhead).

                if failed_collections.is_empty() {
                    Ok(total_bytes_written)
                } else {
                    Err(Box::new(std::io::Error::other(format!(
                        "Failed to flush {} collections: {:?}",
                        failed_collections.len(),
                        failed_collections
                    )))
                        as Box<dyn std::error::Error + Send + Sync>)
                }
            });

        // Record flush latency and bytes
        let elapsed_us = start.elapsed().as_micros() as u64;
        let bytes = result.as_ref().ok().copied().unwrap_or(0);
        self.metrics_collector.record_flush_us(elapsed_us, bytes);

        // Record error if flush failed
        if result.is_err() {
            self.metrics_collector.record_error();
        }

        // Convert Result<u64, E> to Result<(), E>
        result.map(|_| ())
    }

    // ========================================================================
    // Vector CRUD Operations - Phase 1: GET
    // ========================================================================

    /// Get a vector by ID from a collection
    ///
    /// Returns the vector record with id, vector data, and metadata if found.
    /// Searches both unflushed data (WAL/memtable) and flushed storage (SST files).
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `vector_id` - ID of the vector to retrieve
    ///
    /// # Returns
    /// * `Ok(Some(ProximaRecord))` - Vector found
    /// * `Ok(None)` - Vector not found
    /// * `Err` - Error occurred during lookup
    ///
    /// # Example
    /// ```rust,ignore
    /// let record = db.get_vector("embeddings", "vec_123")?;
    /// if let Some(vec) = record {
    ///     println!("Found vector: {:?}", vec.vector);
    /// }
    /// ```
    pub fn get_vector(
        &self,
        collection: &str,
        vector_id: &str,
    ) -> Result<Option<proximadb_records::ProximaRecord>, Box<dyn std::error::Error + Send + Sync>>
    {
        use proximadb::storage::persistence::write_ahead_log::get_global_write_buffer_behavior;

        self.runtime.block_on(async {
            // Step 1: Check WAL/memtable for unflushed data using MVCC-aware lookup.
            // vector_by_id selects the latest version across all batches (version then
            // timestamp), so an UPDATE for the same ID always wins over an INSERT.
            // Tombstones (expires_at=0) are returned as None automatically.
            if let Some(write_buffer) = get_global_write_buffer_behavior() {
                match write_buffer.vector_by_id(collection, vector_id).await {
                    Ok(Some(record)) => return Ok(Some(record)),
                    Ok(None) => {} // not in memtable; fall through to storage
                    Err(_) => {}   // ignore memtable errors; fall through to storage
                }
            }

            // Step 2: Search in flushed storage using the unified storage engine
            // Use a filter-based search to find the specific vector by ID
            let results = self
                .shared_services
                .vector_operations_service
                .unified_search_by_id(collection, vector_id)
                .await;

            match results {
                Ok(Some(record)) => {
                    // Tombstones have valid_to_ns = Some(0)
                    if record.valid_to_ns == Some(0) {
                        Ok(None)
                    } else {
                        Ok(Some(record))
                    }
                }
                Ok(None) => Ok(None),
                Err(e) => Err(Box::new(std::io::Error::other(format!(
                    "Failed to get vector: {}",
                    e
                )))
                    as Box<dyn std::error::Error + Send + Sync>),
            }
        })
    }

    /// Check if a vector exists in a collection
    ///
    /// This is a fast existence check that uses bloom filters when available.
    /// More efficient than `get_vector` when you only need to check existence.
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `vector_id` - ID of the vector to check
    ///
    /// # Returns
    /// * `true` - Vector exists
    /// * `false` - Vector does not exist
    pub fn vector_exists(
        &self,
        collection: &str,
        vector_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        // For now, use get_vector and check if Some
        // In the future, this could use bloom filter for faster negative checks
        Ok(self.get_vector(collection, vector_id)?.is_some())
    }

    /// Delete a single vector by ID (tombstone-based)
    ///
    /// This uses tombstone markers to logically delete the vector. The tombstone
    /// will be written to the WAL and will shadow the original vector in searches.
    /// Physical deletion happens during compaction.
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `vector_id` - ID of the vector to delete
    ///
    /// # Returns
    /// * `true` - Tombstone was written (doesn't guarantee vector existed)
    /// * `false` - Failed to write tombstone
    ///
    /// # Example
    /// ```rust,ignore
    /// let deleted = db.delete_vector("embeddings", "vec_123")?;
    /// ```
    pub fn delete_vector(
        &self,
        collection: &str,
        vector_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        self.check_write_access()?;
        use std::sync::Arc;

        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let tombstone = proximadb_records::ProximaRecord {
            oid: vector_id.to_string(),
            created_at_ns: now_ns,
            updated_at_ns: now_ns,
            valid_to_ns: Some(0), // Expired at epoch = tombstone
            origin: Some("delete".to_string()),
            ..Default::default()
        };

        let records = Arc::new(vec![tombstone]);

        self.runtime.block_on(async {
            self.shared_services
                .vector_operations_service
                .insert_vectors_direct(collection, records)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;
            Ok(true)
        })
    }

    /// Delete multiple vectors by IDs (batch tombstone operation)
    ///
    /// More efficient than calling `delete_vector` multiple times.
    /// All tombstones are written in a single batch operation.
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `vector_ids` - Vector of IDs to delete
    ///
    /// # Returns
    /// Number of tombstones written (equals input count)
    ///
    /// # Example
    /// ```rust,ignore
    /// let count = db.delete_vectors("embeddings", vec!["vec_1".to_string(), "vec_2".to_string()])?;
    /// assert_eq!(count, 2);
    /// ```
    pub fn delete_vectors(
        &self,
        collection: &str,
        vector_ids: Vec<String>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before deleting vectors
        self.check_write_access()?;

        use std::sync::Arc;

        if vector_ids.is_empty() {
            return Ok(0);
        }

        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);

        let tombstones: Vec<proximadb_records::ProximaRecord> = vector_ids
            .iter()
            .map(|id| proximadb_records::ProximaRecord {
                oid: id.clone(),
                created_at_ns: now_ns,
                updated_at_ns: now_ns,
                valid_to_ns: Some(0), // Expired at epoch = tombstone
                origin: Some("delete".to_string()),
                ..Default::default()
            })
            .collect();

        let count = tombstones.len();
        let records = Arc::new(tombstones);

        self.runtime.block_on(async {
            self.shared_services
                .vector_operations_service
                .insert_vectors_direct(collection, records)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;
            Ok(count)
        })
    }

    /// Upsert vectors (insert or update)
    ///
    /// This is an atomic operation that:
    /// 1. Checks which IDs already exist
    /// 2. Deletes existing vectors (creates tombstones)
    /// 3. Inserts all vectors as new records
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `ids` - Vector IDs
    /// * `vectors` - Vector data
    /// * `metadata` - Optional metadata for each vector
    ///
    /// # Returns
    /// Tuple of (inserted_count, updated_count)
    ///
    /// # Example
    /// ```rust,ignore
    /// let (inserted, updated) = db.upsert(
    ///     "embeddings",
    ///     vec!["vec_1".to_string(), "vec_2".to_string()],
    ///     vec![vec![0.1; 768], vec![0.2; 768]],
    ///     None,
    /// )?;
    /// println!("Inserted: {}, Updated: {}", inserted, updated);
    /// ```
    pub fn upsert(
        &self,
        collection: &str,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadata: Option<Vec<std::collections::HashMap<String, serde_json::Value>>>,
    ) -> Result<(usize, usize), Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before upserting
        self.check_write_access()?;

        if ids.is_empty() {
            return Ok((0, 0));
        }

        if ids.len() != vectors.len() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "IDs count ({}) must match vectors count ({})",
                    ids.len(),
                    vectors.len()
                ),
            )) as Box<dyn std::error::Error + Send + Sync>);
        }

        // Check which IDs already exist
        let mut existing_ids = Vec::new();
        let mut inserted = 0;
        let mut updated = 0;

        for id in &ids {
            if self.vector_exists(collection, id)? {
                existing_ids.push(id.clone());
                updated += 1;
            } else {
                inserted += 1;
            }
        }

        // Delete existing vectors (creates tombstones)
        if !existing_ids.is_empty() {
            self.delete_vectors(collection, existing_ids)?;
        }

        // Insert all vectors as new records. Some storage engines can detect
        // insert-only conflicts for flushed records that the read-side
        // vector_exists probe misses, so recover by tombstoning the requested
        // IDs and retrying as a true upsert.
        match self.insert(collection, ids.clone(), vectors.clone(), metadata.clone()) {
            Ok(_) => Ok((inserted, updated)),
            Err(err) if err.to_string().contains("INSERT_CONFLICT") => {
                self.delete_vectors(collection, ids.clone())?;
                let count = ids.len();
                let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
                let records = Self::build_embedded_vector_records(ids, vectors, metadata, now_ns);

                self.runtime.block_on(async {
                    self.shared_services
                        .vector_operations_service
                        .insert_vectors_direct(collection, std::sync::Arc::new(records))
                        .await
                        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                            Box::new(std::io::Error::other(e.to_string()))
                        })?;
                    Ok((0, count))
                })
            }
            Err(err) => Err(err),
        }
    }

    /// Get storage statistics
    pub fn stats(&self) -> Result<EmbeddedStorageStats, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let collections = self.collection_port.list_collections(None).await.ok();
            let total_collections = collections.as_ref().map_or(0, |c| c.len() as u64);
            let total_vectors: u64 = collections.map_or(0, |c| {
                c.iter()
                    .filter_map(|col| col.stats.as_ref())
                    .map(|s| s.vector_count as u64)
                    .sum()
            });

            Ok(EmbeddedStorageStats {
                total_vectors,
                total_collections,
                disk_usage_bytes: 0, // Deferred: Calculate actual disk usage
                cache_hit_rate: 0.0, // Deferred: Get from cache orchestrator
            })
        })
    }

    // ========================================================================
    // Observability Metrics API
    // ========================================================================

    /// Get current metrics snapshot
    ///
    /// Returns comprehensive metrics including latency histograms, operation
    /// counters, cache statistics, and WAL statistics.
    ///
    /// # Arguments
    /// * `window` - Rolling window for latency statistics:
    ///   - `RollingWindow::OneMinute` - Last 1 minute
    ///   - `RollingWindow::FiveMinutes` - Last 5 minutes
    ///   - `RollingWindow::OneHour` - Last 1 hour
    ///   - `RollingWindow::AllTime` - All time (default)
    ///
    /// # Example
    /// ```rust,ignore
    /// let metrics = db.metrics(RollingWindow::AllTime);
    /// println!("p99 search latency: {:.2}ms", metrics.search_latency.p99_ms);
    /// println!("Cache hit rate: {:.1}%", metrics.cache_hit_rate * 100.0);
    /// ```
    pub fn metrics(&self, window: RollingWindow) -> EmbeddedMetrics {
        // Update cache stats from system before snapshot
        self.update_cache_stats();
        self.update_wal_stats();

        self.metrics_collector.snapshot(window)
    }

    /// Get current metrics with default window (all time)
    pub fn metrics_default(&self) -> EmbeddedMetrics {
        self.metrics(RollingWindow::AllTime)
    }

    /// Reset all metrics counters and histograms
    ///
    /// This clears all accumulated metrics data. Useful for:
    /// - Starting fresh measurement periods
    /// - Clearing test data
    /// - Benchmarking specific operations
    ///
    /// # Example
    /// ```rust,ignore
    /// db.reset_metrics();
    /// // Run benchmark...
    /// let metrics = db.metrics(RollingWindow::AllTime);
    /// ```
    pub fn reset_metrics(&self) {
        self.metrics_collector.reset();
        tracing::debug!("EMBEDDED: Metrics reset");
    }

    /// Export metrics in Prometheus text format
    ///
    /// Returns a string in Prometheus exposition format suitable for
    /// scraping by Prometheus or compatible monitoring systems.
    ///
    /// # Example
    /// ```rust,ignore
    /// let prometheus = db.export_prometheus();
    /// // Save to file or serve via HTTP endpoint
    /// std::fs::write("/metrics/embedded.prom", prometheus)?;
    /// ```
    pub fn export_prometheus(&self) -> String {
        self.metrics(RollingWindow::AllTime).to_prometheus()
    }

    /// Get metrics collector for manual instrumentation
    ///
    /// Advanced usage for custom metrics tracking.
    pub fn metrics_collector(&self) -> std::sync::Arc<EmbeddedMetricsCollector> {
        self.metrics_collector.clone()
    }

    /// Update cache statistics from the cache orchestrator
    fn update_cache_stats(&self) {
        // Try to get cache stats from the shared services
        // This is best-effort - if we can't get stats, we just skip updating
        if let Some(cache_stats) = self.try_get_cache_stats() {
            self.metrics_collector
                .set_cache_entries(cache_stats.entries);
            self.metrics_collector
                .set_cache_memory_bytes(cache_stats.memory_bytes);
        }
    }

    /// Update WAL statistics
    fn update_wal_stats(&self) {
        // Try to get WAL stats from the global write buffer
        self.runtime.block_on(async {
            if let Some(write_buffer) =
                proximadb::storage::persistence::write_ahead_log::get_global_write_buffer_behavior()
            {
                let collections = write_buffer.list_collections_with_unflushed_data().await;
                let mut total_pending_bytes = 0u64;

                for collection_id in collections {
                    if let Ok(batches) = write_buffer.get_unflushed_batches(&collection_id).await {
                        for batch in batches {
                            // Estimate bytes per batch (rough approximation)
                            total_pending_bytes += batch.vector_records.len() as u64 * 4096; // ~4KB per vector
                        }
                    }
                }

                self.metrics_collector
                    .set_wal_pending_bytes(total_pending_bytes);
            }
        });
    }

    /// Try to get cache statistics (best-effort)
    fn try_get_cache_stats(&self) -> Option<CacheStatsSnapshot> {
        // For now, return None - will be implemented when cache integration is available
        // The cache orchestrator in SharedServices doesn't expose stats directly yet
        None
    }

    /// Close the embedded database gracefully
    ///
    /// This method should be called before dropping the database to:
    /// 1. Persist the RL planner policy to disk
    /// 2. Log final statistics
    ///
    /// Note: Calling `flush()` before `close()` is recommended to ensure
    /// all vector data is persisted.
    ///
    /// # Example
    /// ```rust,ignore
    /// let db = EmbeddedProximaDB::new(config)?;
    /// // ... use the database ...
    /// db.flush()?;  // Persist vector data
    /// db.close();   // Persist RL policy and cleanup
    /// ```
    /// Flush every graph's WAL buffer to disk. Idempotent and best-effort.
    ///
    /// Graph writes land in `UnifiedWALWriter::current_segment_data`, an
    /// in-memory buffer that reaches disk only on an explicit flush or a 64 MB
    /// segment rotation. `UnifiedWALEntry::new` never sets `requires_fsync`, so
    /// no graph write triggers the former, and a realistic code graph (93k
    /// nodes / 177k edges) never reaches the latter. Nothing on the in-process
    /// path called flush, so the buffer died with the process: measured before
    /// this change, ingest reported 177,046 edges, `graph_stats` agreed,
    /// `close()` returned cleanly, the WAL file on disk was 0 bytes, and a
    /// fresh process saw `edges=0`. Total silent data loss (#1524).
    ///
    /// The server has always flushed here (`ProximaDB::stop`), but its loop is
    /// gated on `self.multi_server` — the NETWORK object. Embedded is
    /// deliberately port-free and never constructs one, so it inherited none of
    /// the durability. Attaching a durability step to the network layer is what
    /// let a transport choice decide whether data survives.
    ///
    /// Errors are logged rather than propagated: this runs from `close()` and
    /// from `Drop`, where raising is worse than warning, and one unflushable
    /// graph must not stop the others from flushing.
    fn flush_graph_wals(&self) {
        self.runtime.block_on(async {
            let graph_service = &self.shared_services.graph_service;
            match graph_service.list_graphs().await {
                Ok(graph_ids) => {
                    let total = graph_ids.len();
                    let mut flushed = 0usize;
                    for graph_id in graph_ids {
                        match graph_service.flush_wal(&graph_id).await {
                            Ok(()) => flushed += 1,
                            Err(e) => tracing::warn!(
                                "EMBEDDED: failed to flush WAL for graph {}: {}",
                                graph_id,
                                e
                            ),
                        }
                    }
                    if total > 0 {
                        tracing::info!("EMBEDDED: flushed WAL for {flushed}/{total} graph(s)");
                    }
                }
                Err(e) => tracing::warn!("EMBEDDED: could not list graphs to flush WAL: {}", e),
            }
        });
    }

    pub fn close(&self) {
        if self.closed.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        // Persist RL planner policy if enabled
        if let Some(ref policy_path) = self.rl_policy_path
            && let Some(planner) = proximadb::query::rl_planner::get_rl_planner()
        {
            self.runtime.block_on(async {
                match planner.save_policy(policy_path).await {
                    Ok(()) => {
                        tracing::info!("🎯 EMBEDDED: RL policy persisted to {}", policy_path);
                    }
                    Err(e) => {
                        tracing::warn!("EMBEDDED: Failed to persist RL policy: {}", e);
                    }
                }

                // Log final stats
                let stats = planner.get_action_stats().await;
                if !stats.is_empty() {
                    tracing::info!(
                        "🎯 EMBEDDED: RL Planner final stats - {} actions tracked",
                        stats.len()
                    );
                }
            });
        }

        self.flush_graph_wals();

        self.release_coordination();

        tracing::info!("🛑 EMBEDDED: Database closed");
    }

    fn release_coordination(&self) {
        if let Some(election) = &self.leader_election
            && let Err(e) = election.release()
        {
            tracing::warn!("EMBEDDED: failed to release leadership: {}", e);
        }
        if let Some(lock) = &self.lock_manager
            && let Err(e) = lock.release()
        {
            tracing::warn!("EMBEDDED: failed to release access lock: {}", e);
        }
    }

    // ========================================================================
    // Checkpoint and Delta Persistence API
    // ========================================================================

    /// Create a named checkpoint of the current database state
    ///
    /// This captures the current state of all collections and allows restoration
    /// to this point later. Checkpoints are persisted to disk and survive restarts.
    ///
    /// # Arguments
    /// * `name` - Name for the checkpoint (must be unique)
    ///
    /// # Returns
    /// * `CheckpointInfo` with details about the created checkpoint
    ///
    /// # Example
    /// ```rust,ignore
    /// let info = db.checkpoint("before_experiment")?;
    /// println!("Checkpoint created at LSN {}", info.checkpoint_lsn);
    ///
    /// // Make changes...
    /// db.insert("vectors", ids, vectors, None)?;
    ///
    /// // Restore to checkpoint
    /// db.restore_checkpoint("before_experiment")?;
    /// ```
    pub fn checkpoint(
        &self,
        name: &str,
    ) -> Result<CheckpointInfo, Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before creating checkpoint
        self.check_write_access()?;

        self.runtime.block_on(async {
            // First, flush all pending writes to ensure checkpoint captures current state
            tracing::info!(
                "EMBEDDED: Creating checkpoint '{}' - flushing pending writes...",
                name
            );

            // Get current LSN from global manifest
            let current_lsn =
                match proximadb::storage::persistence::write_ahead_log::manifest::get_service() {
                    Some(svc) => svc.current_lsn().await,
                    None => 0,
                };

            // Gather collection states
            let collections = self.collection_port.list_collections(None).await.map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                },
            )?;

            let collection_states: Vec<CollectionCheckpointState> = collections
                .into_iter()
                .map(|c| {
                    let config = c.config.unwrap_or_default();
                    let stats = c.stats.unwrap_or_default();
                    CollectionCheckpointState {
                        name: config.name,
                        vector_count: stats.vector_count as u64,
                        last_lsn: current_lsn, // Approximate with global LSN
                        dimension: config.dimension,
                        engine: format!("{:?}", config.storage_engine.unwrap_or(0)),
                    }
                })
                .collect();

            // Create the checkpoint
            let info = self
                .checkpoint_manager
                .create_checkpoint(name, current_lsn, collection_states)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            tracing::info!(
                "EMBEDDED: Checkpoint '{}' created at LSN {} with {} collections",
                name,
                info.checkpoint_lsn,
                info.collections.len()
            );

            Ok(info)
        })
    }

    /// Restore the database to a named checkpoint
    ///
    /// This restores all collections to the state they were in when the checkpoint
    /// was created. Any changes made after the checkpoint are discarded.
    ///
    /// WARNING: This is a destructive operation. All data added after the checkpoint
    /// will be lost.
    ///
    /// # Arguments
    /// * `name` - Name of the checkpoint to restore
    ///
    /// # Example
    /// ```rust,ignore
    /// db.checkpoint("backup")?;
    /// db.insert("vectors", ids, vectors, None)?;  // Add data
    /// db.restore_checkpoint("backup")?;  // Restore - new data is discarded
    /// ```
    pub fn restore_checkpoint(
        &self,
        name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before restoring checkpoint (this modifies data)
        self.check_write_access()?;

        self.runtime.block_on(async {
            // Get the checkpoint info
            let checkpoint_info = self.checkpoint_manager
                .get_checkpoint(name)
                .await
                .ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("Checkpoint '{}' not found", name),
                    ))
                })?;

            tracing::info!(
                "EMBEDDED: Restoring checkpoint '{}' (LSN {})",
                name, checkpoint_info.checkpoint_lsn
            );

            // For a full restore, we would need to:
            // 1. Clear all data after the checkpoint LSN
            // 2. Rebuild collection state from storage engine data at checkpoint time
            //
            // For now, we implement a simplified restore that uses PITR if available,
            // or logs a warning if full PITR is not set up

            if let Some(manifest_service) = proximadb::storage::persistence::write_ahead_log::manifest::get_service() {
                // Use the manifest to mark entries after checkpoint as rolled back
                match manifest_service.mark_entries_after_lsn_rolled_back(checkpoint_info.checkpoint_lsn).await {
                    Ok(rolled_back) => {
                        tracing::info!(
                            "EMBEDDED: Rolled back {} entries after checkpoint LSN {}",
                            rolled_back, checkpoint_info.checkpoint_lsn
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            "EMBEDDED: Failed to rollback entries: {}. Manual cleanup may be required.",
                            e
                        );
                    }
                }
            }

            // Clear the write buffer for affected collections
            if let Some(write_buffer) = proximadb::storage::persistence::write_ahead_log::get_global_write_buffer_behavior() {
                for collection_name in &checkpoint_info.collections {
                    if let Err(e) = write_buffer.clear_flushed(collection_name).await {
                        tracing::warn!(
                            "EMBEDDED: Failed to clear write buffer for '{}': {}",
                            collection_name, e
                        );
                    }
                }
            }

            tracing::info!("EMBEDDED: Checkpoint '{}' restored", name);
            Ok(())
        })
    }

    /// Save incremental changes since the last checkpoint to a delta file
    ///
    /// Delta files contain only the changes made since the last checkpoint,
    /// making them much smaller and faster to create than full checkpoints.
    ///
    /// # Arguments
    /// * `path` - Path where the delta file will be saved
    ///
    /// # Returns
    /// * `DeltaInfo` with details about the saved delta
    ///
    /// # Example
    /// ```rust,ignore
    /// db.checkpoint("baseline")?;
    /// db.insert("vectors", ids1, vectors1, None)?;
    /// db.insert("vectors", ids2, vectors2, None)?;
    ///
    /// // Save only the changes since checkpoint
    /// let delta = db.save_delta("/backup/delta_001.delta")?;
    /// println!("Delta saved: {} entries, {} bytes", delta.entry_count, delta.size_bytes);
    /// ```
    pub fn save_delta(
        &self,
        path: &str,
    ) -> Result<DeltaInfo, Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before saving delta (requires flushing)
        self.check_write_access()?;

        use proximadb::storage::persistence::write_ahead_log::get_global_write_buffer_behavior;

        self.runtime.block_on(async {
            // Get the last checkpoint LSN
            let base_checkpoint = self.checkpoint_manager.current_checkpoint_name().await;
            let start_lsn = self.checkpoint_manager.last_checkpoint_lsn().await;

            // Get current LSN
            let end_lsn =
                match proximadb::storage::persistence::write_ahead_log::manifest::get_service() {
                    Some(svc) => svc.current_lsn().await,
                    None => start_lsn,
                };

            if end_lsn <= start_lsn {
                // No changes since checkpoint
                let info = DeltaInfo {
                    path: path.to_string(),
                    timestamp: chrono::Utc::now(),
                    size_bytes: 0,
                    entry_count: 0,
                    base_checkpoint,
                    start_lsn,
                    end_lsn,
                    affected_collections: vec![],
                };
                tracing::info!("EMBEDDED: No changes to save in delta");
                return Ok(info);
            }

            // Collect changes from the write buffer
            let mut entries = Vec::new();

            if let Some(write_buffer) = get_global_write_buffer_behavior() {
                let collections = write_buffer.list_collections_with_unflushed_data().await;

                for collection_id in collections {
                    if let Ok(batches) = write_buffer.get_unflushed_batches(&collection_id).await {
                        for batch in batches {
                            // Serialize canonical records - dereference Arc to get the Vec.
                            let records: &Vec<proximadb_records::ProximaRecord> =
                                batch.vector_records.as_ref();
                            let vector_data = bincode::serialize(records).map_err(
                                |e| -> Box<dyn std::error::Error + Send + Sync> {
                                    Box::new(std::io::Error::other(e.to_string()))
                                },
                            )?;

                            entries.push(DeltaEntry {
                                lsn: end_lsn, // Use current LSN for all entries
                                collection_id: collection_id.clone(),
                                operation: "upsert".to_string(),
                                vector_data: Some(vector_data),
                                vector_ids: None,
                                collection_config: None,
                            });
                        }
                    }
                }
            }

            // Save the delta file
            let info = self
                .checkpoint_manager
                .save_delta(path, entries, base_checkpoint, start_lsn, end_lsn)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            tracing::info!(
                "EMBEDDED: Delta saved to {} ({} entries, {} bytes)",
                path,
                info.entry_count,
                info.size_bytes
            );

            Ok(info)
        })
    }

    /// Load changes from a delta file
    ///
    /// Applies the changes from a delta file to the current database state.
    /// This is typically used to replay changes after restoring from a checkpoint.
    ///
    /// # Arguments
    /// * `path` - Path to the delta file to load
    ///
    /// # Example
    /// ```rust,ignore
    /// // Restore to checkpoint, then apply delta
    /// db.restore_checkpoint("baseline")?;
    /// db.load_delta("/backup/delta_001.delta")?;
    /// ```
    pub fn load_delta(&self, path: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before loading delta (this modifies data)
        self.check_write_access()?;

        self.runtime.block_on(async {
            // Load the delta file
            let (header, entries) = self.checkpoint_manager.load_delta(path).await.map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                },
            )?;

            tracing::info!(
                "EMBEDDED: Loading delta from {} ({} entries, LSN {}..{})",
                path,
                entries.len(),
                header.start_lsn,
                header.end_lsn
            );

            // Apply each entry
            for entry in entries {
                match entry.operation.as_str() {
                    "upsert" => {
                        if let Some(vector_data) = entry.vector_data {
                            // Deserialize legacy VectorRecord delta data and convert to canonical
                            // ProximaRecord envelopes at this storage migration boundary.
                            let vr_records: Vec<proximadb::proto::proximadb_v1::VectorRecord> =
                                bincode::deserialize(&vector_data).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        Box::new(std::io::Error::other(e.to_string()))
                                    },
                                )?;
                            let records: Vec<proximadb_records::ProximaRecord> = vr_records
                                .into_iter()
                                .map(proximadb_records::ProximaRecord::from)
                                .collect();

                            let records = std::sync::Arc::new(records);
                            self.shared_services
                                .vector_operations_service
                                .insert_vectors_direct(&entry.collection_id, records)
                                .await
                                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                    Box::new(std::io::Error::other(e.to_string()))
                                })?;
                        }
                    }
                    "delete" => {
                        if let Some(vector_ids) = entry.vector_ids {
                            let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
                            let tombstones: Vec<proximadb_records::ProximaRecord> = vector_ids
                                .into_iter()
                                .map(|id| proximadb_records::ProximaRecord {
                                    oid: id,
                                    created_at_ns: now_ns,
                                    updated_at_ns: now_ns,
                                    valid_to_ns: Some(0),
                                    origin: Some("delete".to_string()),
                                    ..Default::default()
                                })
                                .collect();
                            let records = std::sync::Arc::new(tombstones);
                            self.shared_services
                                .vector_operations_service
                                .insert_vectors_direct(&entry.collection_id, records)
                                .await
                                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                    Box::new(std::io::Error::other(e.to_string()))
                                })?;
                        }
                    }
                    op => {
                        tracing::warn!("EMBEDDED: Unknown delta operation: {}", op);
                    }
                }
            }

            tracing::info!("EMBEDDED: Delta loaded successfully");
            Ok(())
        })
    }

    /// List all available checkpoints
    ///
    /// Returns a list of all checkpoints that have been created, sorted by
    /// creation timestamp (oldest first).
    ///
    /// # Returns
    /// * `Vec<CheckpointInfo>` with details about each checkpoint
    ///
    /// # Example
    /// ```rust,ignore
    /// let checkpoints = db.list_checkpoints()?;
    /// for cp in checkpoints {
    ///     println!("{}: {} collections at LSN {}", cp.name, cp.collections.len(), cp.checkpoint_lsn);
    /// }
    /// ```
    pub fn list_checkpoints(
        &self,
    ) -> Result<Vec<CheckpointInfo>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime
            .block_on(async { Ok(self.checkpoint_manager.list_checkpoints().await) })
    }

    // ========================================================================
    // Generic Graph Operations API
    // ========================================================================

    /// Create a graph collection
    ///
    /// A graph must be created before nodes and edges can be added.
    pub fn create_graph(
        &self,
        graph_id: &str,
        engine: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before creating graph
        self.check_write_access()?;

        self.runtime.block_on(async {
            let graph_service = &self.shared_services.graph_service;

            let engine_config = engine.map(|e| proximadb::proto::proximadb_v1::GraphEngineConfig {
                engine_type: e.to_string(),
                memory_pool_size_mb: 0,
                csr_cache_size_mb: 0,
                enable_parallel_operations: true,
                max_traversal_depth: 10,
                advanced_config: std::collections::HashMap::new(),
            });

            let request = proximadb::proto::proximadb_v1::CreateGraphRequest {
                graph_id: graph_id.to_string(),
                name: Some(graph_id.to_string()),
                description: None,
                schema: None,
                storage_config: None,
                engine_config,
                access_control: None,
            };

            graph_service
                .create_graph_collection(request)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(())
        })
    }

    /// Create nodes in the graph
    ///
    /// Inserts nodes with their properties. Use labels and properties
    /// for domain-specific categorization and attributes.
    pub fn create_nodes(
        &self,
        graph_id: &str,
        nodes: Vec<EmbeddedGraphNode>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before creating nodes
        self.check_write_access()?;

        self.runtime.block_on(async {
            let graph_service = &self.shared_services.graph_service;

            // Use batch API for optimal performance (100-500x faster than individual inserts)
            let proto_nodes: Vec<_> = nodes.into_iter().map(|n| n.to_proto()).collect();
            graph_service
                .batch_create_nodes(graph_id, proto_nodes)
                .await
                .map(|inserted| inserted.len())
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })
        })
    }

    /// Create edges in the graph
    pub fn create_edges(
        &self,
        graph_id: &str,
        edges: Vec<EmbeddedGraphEdge>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before creating edges
        self.check_write_access()?;

        let _count = edges.len(); // Tracking original count for potential logging

        self.runtime.block_on(async {
            let graph_service = &self.shared_services.graph_service;

            let proto_edges: Vec<_> = edges.into_iter().map(|e| e.to_proto()).collect();
            graph_service
                .batch_create_edges(graph_id, proto_edges)
                .await
                .map(|outcome| outcome.created.len())
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })
        })
    }

    /// Create a single node in the graph.
    pub fn create_node(
        &self,
        graph_id: &str,
        node_id: &str,
        labels: Vec<String>,
        properties: std::collections::HashMap<String, proximadb::graph::model::PropertyValue>,
    ) -> Result<EmbeddedGraphNode, Box<dyn std::error::Error + Send + Sync>> {
        let node = EmbeddedGraphNode {
            id: node_id.to_string(),
            labels,
            properties,
        };
        self.create_nodes(graph_id, vec![node.clone()])?;
        Ok(node)
    }

    /// Create a single edge in the graph.
    pub fn create_edge(
        &self,
        graph_id: &str,
        edge_id: Option<&str>,
        from_node_id: &str,
        to_node_id: &str,
        edge_type: &str,
        weight: Option<f64>,
        properties: std::collections::HashMap<String, proximadb::graph::model::PropertyValue>,
    ) -> Result<EmbeddedGraphEdge, Box<dyn std::error::Error + Send + Sync>> {
        let edge = EmbeddedGraphEdge {
            id: edge_id.map(str::to_string),
            from_node_id: from_node_id.to_string(),
            to_node_id: to_node_id.to_string(),
            edge_type: edge_type.to_string(),
            weight,
            properties,
        };
        self.create_edges(graph_id, vec![edge.clone()])?;
        Ok(edge)
    }

    /// Get a node by ID
    pub fn get_node(
        &self,
        graph_id: &str,
        node_id: &str,
    ) -> Result<Option<EmbeddedGraphNode>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let graph_service = &self.shared_services.graph_service;
            let node_id_string = node_id.to_string();

            let proto_node = graph_service
                .get_node(graph_id, &node_id_string)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(proto_node.map(|n| EmbeddedGraphNode::from_proto(&n)))
        })
    }

    /// Query nodes by labels
    pub fn query_nodes_by_labels(
        &self,
        graph_id: &str,
        labels: Vec<String>,
    ) -> Result<Vec<EmbeddedGraphNode>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let graph_service = &self.shared_services.graph_service;

            let node_query = proximadb::graph::NodeQuery {
                labels,
                ..Default::default()
            };

            let proto_nodes = graph_service
                .query_nodes(graph_id, node_query)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(proto_nodes
                .into_iter()
                .map(|n| EmbeddedGraphNode::from_proto(&n))
                .collect())
        })
    }

    /// Query graph nodes by labels and exact-match properties.
    pub fn query_nodes(
        &self,
        graph_id: &str,
        labels: Option<Vec<String>>,
        properties: Option<
            std::collections::HashMap<String, proximadb::graph::model::PropertyValue>,
        >,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<EmbeddedGraphNode>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let graph_service = &self.shared_services.graph_service;

            let proto_nodes = graph_service
                .query_nodes(
                    graph_id,
                    proximadb::graph::NodeQuery {
                        graph_id: graph_id.to_string(),
                        labels: labels.unwrap_or_default(),
                        filters: properties
                            .as_ref()
                            .map_or_else(Vec::new, Self::property_filters_from_map),
                        limit,
                        offset,
                        continuation_token: None,
                    },
                )
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(proto_nodes
                .into_iter()
                .map(|n| EmbeddedGraphNode::from_proto(&n))
                .collect())
        })
    }

    /// Get outgoing edges from a node
    pub fn get_outgoing_edges(
        &self,
        graph_id: &str,
        node_id: &str,
        edge_types: Option<Vec<String>>,
    ) -> Result<Vec<EmbeddedGraphEdge>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let graph_service = &self.shared_services.graph_service;

            let edge_query = proximadb::graph::EdgeQuery {
                from_node_id: Some(node_id.to_string()),
                to_node_id: None,
                edge_types: edge_types.unwrap_or_default(),
                ..Default::default()
            };

            let proto_edges = graph_service
                .query_edges(graph_id, edge_query)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(proto_edges
                .into_iter()
                .map(|e| EmbeddedGraphEdge::from_proto(&e))
                .collect())
        })
    }

    /// Get incoming edges to a node
    pub fn get_incoming_edges(
        &self,
        graph_id: &str,
        node_id: &str,
        edge_types: Option<Vec<String>>,
    ) -> Result<Vec<EmbeddedGraphEdge>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let graph_service = &self.shared_services.graph_service;

            let edge_query = proximadb::graph::EdgeQuery {
                from_node_id: None,
                to_node_id: Some(node_id.to_string()),
                edge_types: edge_types.unwrap_or_default(),
                ..Default::default()
            };

            let proto_edges = graph_service
                .query_edges(graph_id, edge_query)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(proto_edges
                .into_iter()
                .map(|e| EmbeddedGraphEdge::from_proto(&e))
                .collect())
        })
    }

    /// Delete a node and its edges
    pub fn delete_node(
        &self,
        graph_id: &str,
        node_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let graph_service = &self.shared_services.graph_service;
            let node_id_string = node_id.to_string();

            let deleted = graph_service
                .delete_node(graph_id, &node_id_string)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(deleted.is_some())
        })
    }

    /// Get graph statistics
    pub fn graph_stats(
        &self,
        graph_id: &str,
    ) -> Result<EmbeddedGraphStats, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let graph_service = &self.shared_services.graph_service;

            let proto_stats = graph_service.get_stats(graph_id).await.map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                },
            )?;

            Ok(EmbeddedGraphStats {
                total_nodes: proto_stats.total_nodes,
                total_edges: proto_stats.total_edges,
            })
        })
    }

    /// Traverse the graph starting from a given node.
    pub fn traverse_graph(
        &self,
        graph_id: &str,
        start_node_id: &str,
        max_depth: u32,
        edge_types: Option<Vec<String>>,
        limit: Option<u32>,
    ) -> Result<EmbeddedGraphTraversalResult, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let response = self
                .shared_services
                .graph_service
                .traverse(
                    graph_id,
                    proximadb::graph::TraversalRequest {
                        graph_id: graph_id.to_string(),
                        start_node_id: start_node_id.to_string(),
                        max_depth,
                        edge_types: edge_types.unwrap_or_default(),
                        node_labels: Vec::new(),
                        filters: Vec::new(),
                        algorithm: proximadb::graph::TraversalAlgorithm::Bfs as i32,
                        limit,
                        timeout_ms: None,
                        max_frontier: None,
                    },
                )
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(Self::traversal_response_to_embedded(response))
        })
    }

    /// Impact analysis (TD-131): forward blast radius (outgoing edges — "what does X impact") or
    /// backward (incoming edges — "what impacts X"). `direction` = `"forward"` | `"backward"`.
    /// Delegates to the server graph service's `impact_analysis`, so the embedded result is the
    /// server result (the embedded-parity guarantee). Reuses [`Self::traversal_response_to_embedded`].
    pub fn impact_analysis(
        &self,
        graph_id: &str,
        start_node_id: &str,
        direction: &str,
        edge_types: Option<Vec<String>>,
        max_depth: u32,
        limit: usize,
    ) -> Result<EmbeddedGraphTraversalResult, Box<dyn std::error::Error + Send + Sync>> {
        let dir = match direction {
            "backward" => proximadb::graph::ImpactDirection::Backward,
            _ => proximadb::graph::ImpactDirection::Forward,
        };
        self.runtime.block_on(async {
            let response = self
                .shared_services
                .graph_service
                .impact_analysis(
                    graph_id,
                    start_node_id,
                    dir,
                    edge_types.unwrap_or_default(),
                    max_depth,
                    limit,
                )
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;
            Ok(Self::traversal_response_to_embedded(response))
        })
    }

    /// Vector-seed → graph-expand → fuse-by-`oid` (TD-137/TD-131). Constructs the SAME `FusionService`
    /// core the REST endpoint uses (`FusionService::new(vector, graph)`), so embedded results are
    /// identical to the server path by construction. `rrf` selects the rank-based fallback policy;
    /// `grain` = `"nodes"` | `"edges"` | `"both"`. Returns `(FusedItem, FusionStats)`.
    pub fn fusion_search(
        &self,
        graph_id: &str,
        vector_collection: &str,
        query_vector: Vec<f32>,
        max_depth: u32,
        edge_types: Vec<String>,
        max_seeds: usize,
        limit: usize,
        vector_weight: f32,
        graph_weight: f32,
        rrf: bool,
        grain: &str,
        text_query: Option<String>,
        document_collection: Option<String>,
        document_weight: Option<f32>,
    ) -> Result<
        (
            Vec<proximadb::core::search::cross_modal_fusion::FusedItem>,
            proximadb::core::search::cross_modal_fusion::FusionStats,
        ),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        self.runtime.block_on(async {
            use proximadb::core::search::cross_modal_fusion::FusionPolicy;
            use proximadb::services::fusion_service::{
                DocumentFusionSpec, FusionOidKey, FusionService, GraphFusionParams, GraphGrain,
            };

            let service = FusionService::new(
                self.shared_services.vector_operations_service.clone(),
                self.shared_services.graph_service.clone(),
                // TD-138: share the live full-text index map so the document modality is powered
                // when a caller supplies a document spec (the params below pass `document: None`
                // for now; the shared map is wired so a future embedded text_query works directly).
                self.shared_services.fulltext_indexes.clone(),
            );
            let params = GraphFusionParams {
                route_policy: None,
                graph_id: graph_id.to_string(),
                // Embedded stays single-tenant by default (unscoped, byte-identical legacy behavior).
                tenant: None,
                tenant_stable_id: None, // FA-2 PR-D3: embedded path — no client subject
                vector_collection: vector_collection.to_string(),
                query_vector,
                max_depth,
                edge_types,
                max_seeds,
                limit,
                vector_weight,
                graph_weight,
                grain: match grain {
                    "edges" => GraphGrain::Edges,
                    "both" => GraphGrain::Both,
                    _ => GraphGrain::Nodes,
                },
                // Embedded = single-repo/local; within-tenant `permitted_principals` RBAC does not
                // apply (structural isolation only).
                principal: None,
                policy: if rrf {
                    FusionPolicy::rrf()
                } else {
                    FusionPolicy::default()
                },
                oid_key: FusionOidKey::Canonical,
                // TD-138 document modality (embedded parity with REST #508): enable iff a
                // non-empty `text_query` was supplied. The service above shares the live
                // full-text index map, so the DocumentExpander runs the BM25 search.
                document: text_query
                    .as_ref()
                    .filter(|t| !t.trim().is_empty())
                    .map(|t| DocumentFusionSpec {
                        text_query: t.clone(),
                        collection: document_collection.clone(),
                        weight: document_weight.unwrap_or(1.0),
                        k: None,
                    }),
            };
            service.graph_fusion_search(params).await.map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                },
            )
        })
    }

    /// Delete entire graph
    pub fn delete_graph(
        &self,
        graph_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // remove_graph returns Option, not Result - it's always successful
        let _ = self.shared_services.graph_service.remove_graph(graph_id);
        Ok(())
    }

    // ========================================================================
    // Document Store Operations (MongoDB-like JSON documents)
    // ========================================================================

    /// Create a document collection for storing JSON documents
    ///
    /// # Arguments
    /// * `name` - Collection name
    /// * `indexes` - Optional list of JSON path expressions to index for faster queries
    ///
    /// # Example
    /// ```rust,ignore
    /// db.create_document_collection("products", Some(vec!["$.category", "$.price"]))?;
    /// ```
    pub fn create_document_collection(
        &self,
        name: &str,
        indexes: Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::proto::proximadb_v1::{
            DocIndexType, DocumentCollectionConfig, IndexDefinition,
        };

        self.runtime.block_on(async {
            let doc_service = self.shared_services.document_service.clone();

            // Build index definitions from paths
            let index_defs: Vec<IndexDefinition> = indexes
                .unwrap_or_default()
                .into_iter()
                .map(|path| IndexDefinition {
                    name: Some(format!("idx_{}", path.replace("$.", "").replace('.', "_"))),
                    path: path.clone(),
                    index_type: DocIndexType::Btree as i32, // B-tree for range queries
                    unique: false,
                    ..Default::default()
                })
                .collect();

            let config = DocumentCollectionConfig {
                name: name.to_string(),
                indexes: index_defs,
                ..Default::default()
            };

            doc_service.create_collection(name, config).await.map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                },
            )?;

            tracing::info!("📄 EMBEDDED: Created document collection '{}'", name);
            Ok(())
        })
    }

    /// Insert a JSON document into a collection
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `id` - Optional document ID (auto-generated UUID if not provided)
    /// * `document` - JSON document as serde_json::Value
    ///
    /// # Returns
    /// The document ID
    ///
    /// # Example
    /// ```rust,ignore
    /// let doc = serde_json::json!({
    ///     "name": "Widget",
    ///     "price": 99.99,
    ///     "tags": ["electronics", "sale"]
    /// });
    /// let id = db.insert_document("products", Some("prod_001"), doc)?;
    /// ```
    pub fn insert_document(
        &self,
        collection: &str,
        id: Option<&str>,
        document: serde_json::Value,
    ) -> Result<(String, u64), Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let doc_service = self.shared_services.document_service.clone();

            // Convert serde_json::Value to SqlObject
            let sql_object = Self::json_to_sql_object(&document);

            let record = doc_service
                .insert_document(collection, id, sql_object)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok((record.id, record.version))
        })
    }

    /// Get a document by ID
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `id` - Document ID
    ///
    /// # Returns
    /// The document as serde_json::Value if found
    ///
    /// # Example
    /// ```rust,ignore
    /// if let Some(doc) = db.get_document("products", "prod_001")? {
    ///     println!("Product: {}", doc["name"]);
    /// }
    /// ```
    pub fn get_document(
        &self,
        collection: &str,
        id: &str,
    ) -> Result<Option<serde_json::Value>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let doc_service = self.shared_services.document_service.clone();

            let record = doc_service
                .get_document(collection, id, None)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(record.map(|r| {
                Self::sql_object_to_json(&proximadb::storage::document::proxima_tree_to_sql_object(
                    &r.props,
                ))
            }))
        })
    }

    /// Query documents with a filter expression
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `filter` - JSON path filter expression (e.g., "$.price > 50 AND $.category = 'electronics'")
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    /// List of matching documents as serde_json::Value
    ///
    /// # Example
    /// ```rust,ignore
    /// let results = db.query_documents("products", "$.price > 50", 100)?;
    /// for doc in results {
    ///     println!("Found: {}", doc["name"]);
    /// }
    /// ```
    pub fn query_documents(
        &self,
        collection: &str,
        filter: Option<&str>,
        limit: u32,
    ) -> Result<Vec<(String, serde_json::Value)>, Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::proto::proximadb_v1::DocumentFilter;
        use proximadb::storage::document::DocumentQueryParams;

        self.runtime.block_on(async {
            let doc_service = self.shared_services.document_service.clone();

            // Parse filter expression into DocumentFilter
            let doc_filter = if let Some(filter_str) = filter {
                let conditions = Self::parse_document_filter(filter_str);
                if conditions.is_empty() {
                    None
                } else {
                    Some(DocumentFilter {
                        conditions,
                        or_filters: vec![],
                        and_filters: vec![],
                    })
                }
            } else {
                None
            };

            let params = DocumentQueryParams {
                filter: doc_filter,
                limit,
                ..Default::default()
            };

            let result = doc_service
                .query_documents(collection, params)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(result
                .documents
                .into_iter()
                .map(|r| {
                    let obj = proximadb::storage::document::proxima_tree_to_sql_object(&r.props);
                    (r.id, Self::sql_object_to_json(&obj))
                })
                .collect())
        })
    }

    /// Update a document by ID with patch operations
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `id` - Document ID
    /// * `updates` - Map of field paths to new values
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut updates = HashMap::new();
    /// updates.insert("$.price".to_string(), serde_json::json!(79.99));
    /// updates.insert("$.sale".to_string(), serde_json::json!(true));
    /// db.update_document("products", "prod_001", updates)?;
    /// ```
    pub fn update_document(
        &self,
        collection: &str,
        id: &str,
        updates: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check write access before updating document
        self.check_write_access()?;

        use proximadb::proto::proximadb_v1::{DocumentUpdate, UpdateOperation};

        self.runtime.block_on(async {
            let doc_service = self.shared_services.document_service.clone();

            // Convert updates to DocumentUpdate operations
            let doc_updates: Vec<DocumentUpdate> = updates
                .into_iter()
                .map(|(path, value)| DocumentUpdate {
                    operation: UpdateOperation::Set as i32,
                    path,
                    value: Some(Self::json_to_sql_value(&value)),
                })
                .collect();

            doc_service
                .update_document(collection, id, doc_updates, None)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(())
        })
    }

    /// Delete a document by ID
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `id` - Document ID
    ///
    /// # Returns
    /// true if document was deleted, false if not found
    ///
    /// # Example
    /// ```rust,ignore
    /// let deleted = db.delete_document("products", "prod_001")?;
    /// ```
    pub fn delete_document(
        &self,
        collection: &str,
        id: &str,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let doc_service = self.shared_services.document_service.clone();

            doc_service.delete_document(collection, id).await.map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                },
            )
        })
    }

    /// Delete a document collection
    ///
    /// # Arguments
    /// * `name` - Collection name
    ///
    /// # Returns
    /// true if collection was deleted, false if not found
    pub fn delete_document_collection(
        &self,
        name: &str,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let doc_service = self.shared_services.document_service.clone();

            doc_service.delete_collection(name).await.map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                },
            )
        })
    }

    /// List all document collections
    ///
    /// # Returns
    /// List of collection names
    ///
    /// # Example
    /// ```rust,ignore
    /// let collections = db.list_document_collections()?;
    /// for name in collections {
    ///     println!("Collection: {}", name);
    /// }
    /// ```
    pub fn list_document_collections(
        &self,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let doc_service = self.shared_services.document_service.clone();

            let collections = doc_service.list_collections().await.map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                },
            )?;

            Ok(collections.into_iter().map(|c| c.name).collect())
        })
    }

    // ========================================================================
    // Unified Query Methods
    // ========================================================================

    /// Execute SQL through the embedded unified handlers.
    pub fn execute_sql(
        &self,
        query: &str,
        parameters: Option<Vec<serde_json::Value>>,
        collection: Option<&str>,
    ) -> Result<EmbeddedSqlQueryResult, Box<dyn std::error::Error + Send + Sync>> {
        let trimmed = query.trim();
        let upper = trimmed.to_ascii_uppercase();
        if upper.starts_with("COMMENT ON ") {
            return Ok(EmbeddedSqlQueryResult {
                rows: vec![serde_json::json!({
                    "status": "ok",
                    "message": "COMMENT accepted for catalog-compatible SDK DDL"
                })],
                columns: vec!["status".to_string(), "message".to_string()],
                column_types: vec!["text".to_string(), "text".to_string()],
                row_count: 1,
                rows_scanned: 0,
                execution_time_ms: 0,
            });
        }

        if proximadb::services::CatalogIntrospectionService::is_catalog_query(trimmed) {
            let start_time = std::time::Instant::now();
            let catalog_manager = self.catalog_manager.clone();
            if let Some(result) = self
                .runtime
                .block_on(async {
                    proximadb::services::CatalogIntrospectionService::new(catalog_manager)
                        .execute_select(trimmed)
                        .await
                })
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?
            {
                let rows = result
                    .rows
                    .iter()
                    .map(|row| {
                        let mut object = serde_json::Map::new();
                        for (idx, value) in row.iter().enumerate() {
                            if let Some(column) = result.columns.get(idx) {
                                object.insert(
                                    column.clone(),
                                    serde_json::Value::String(value.clone()),
                                );
                            }
                        }
                        serde_json::Value::Object(object)
                    })
                    .collect::<Vec<_>>();

                return Ok(EmbeddedSqlQueryResult {
                    row_count: rows.len() as u64,
                    rows,
                    columns: result.columns,
                    column_types: result.column_types,
                    rows_scanned: 0,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                });
            }
        }

        let ddl_parser = proximadb::query::sql_frontend::SqlFrontendParser::new();
        match ddl_parser.parse_ddl(trimmed) {
            Ok(Some(statement)) => {
                let start_time = std::time::Instant::now();
                let catalog_manager = self.catalog_manager.clone();
                let statement_for_backing_store = statement.clone();
                let result = self
                    .runtime
                    .block_on(async {
                        let ddl_service = proximadb::services::DdlService::new(catalog_manager);
                        ddl_service.execute(statement).await
                    })
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(e.to_string()))
                    })?;
                if result.success {
                    self.ensure_catalog_vector_backing_collection(&statement_for_backing_store)?;
                }

                return Ok(EmbeddedSqlQueryResult {
                    rows: vec![serde_json::json!({
                        "success": result.success,
                        "message": result.message,
                        "affected_count": result.affected_count,
                        "warnings": result.warnings
                    })],
                    columns: vec![
                        "success".to_string(),
                        "message".to_string(),
                        "affected_count".to_string(),
                        "warnings".to_string(),
                    ],
                    column_types: vec![
                        "bool".to_string(),
                        "text".to_string(),
                        "int4".to_string(),
                        "jsonb".to_string(),
                    ],
                    row_count: 1,
                    rows_scanned: 0,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                });
            }
            Ok(None) => {}
            Err(error)
                if upper.starts_with("CREATE ")
                    || upper.starts_with("ALTER ")
                    || upper.starts_with("DROP ") =>
            {
                return Err(Box::new(std::io::Error::other(error.to_string())));
            }
            Err(_) => {}
        }

        let dml_parser = proximadb::query::sql_frontend::SqlFrontendParser::new();
        match dml_parser.parse_dml(trimmed) {
            Ok(Some(statement)) => {
                let start_time = std::time::Instant::now();
                self.ensure_dml_vector_backing_collection(&statement)?;
                let catalog_manager = self.catalog_manager.clone();
                let vector_ops = self.shared_services.vector_operations_service.clone();
                let result = self
                    .runtime
                    .block_on(async {
                        let dml_service =
                            proximadb::services::DmlService::new(catalog_manager, vector_ops);
                        dml_service.execute(statement).await
                    })
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(e.to_string()))
                    })?;

                return Ok(EmbeddedSqlQueryResult {
                    rows: vec![serde_json::json!({
                        "success": result.success,
                        "message": result.message,
                        "rows_affected": result.rows_affected,
                        "inserted_ids": result.inserted_ids,
                        "warnings": result.warnings
                    })],
                    columns: vec![
                        "success".to_string(),
                        "message".to_string(),
                        "rows_affected".to_string(),
                        "inserted_ids".to_string(),
                        "warnings".to_string(),
                    ],
                    column_types: vec![
                        "bool".to_string(),
                        "text".to_string(),
                        "int8".to_string(),
                        "jsonb".to_string(),
                        "jsonb".to_string(),
                    ],
                    row_count: 1,
                    rows_scanned: 0,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                });
            }
            Ok(None) => {}
            Err(error)
                if upper.starts_with("INSERT ")
                    || upper.starts_with("UPDATE ")
                    || upper.starts_with("DELETE ") =>
            {
                return Err(Box::new(std::io::Error::other(error.to_string())));
            }
            Err(_) => {}
        }

        self.runtime.block_on(async {
            let proto_params: Option<Vec<proximadb::proto::proximadb_v1::SqlValue>> = parameters
                .map(|values| {
                    values
                        .into_iter()
                        .map(|value| Self::json_to_sql_value(&value))
                        .collect()
                });

            let response = self
                .shared_services
                .api_handlers
                .execute_sql_v1(
                    query.to_string(),
                    proto_params.as_ref().map(|values| {
                        values
                            .iter()
                            .map(proximadb_records::conversions::sql_value_to_proxima)
                            .collect()
                    }),
                    collection.map(str::to_string),
                    proximadb_runtime::PortIdentity::anonymous(),
                )
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(Self::sql_response_to_embedded(response))
        })
    }

    /// Insert or upsert Arrow IPC stream bytes through the embedded vector batch path.
    ///
    /// This is the in-process equivalent of Arrow Flight vector bulk_insert/bulk_upsert:
    /// Arrow IPC stream bytes are decoded to RecordBatches and delegated to
    /// [`Self::insert_arrow_batches`]. This entry point exists for byte-stream
    /// sources; in-process callers should prefer the batch path (Arrow C Data
    /// Interface) which avoids the IPC serialize/deserialize round-trip.
    pub fn insert_arrow_ipc(
        &self,
        collection: &str,
        ipc_stream: &[u8],
        insert_only: bool,
        tenant_id: Option<&str>,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let cursor = std::io::Cursor::new(ipc_stream);
        let reader = arrow_ipc::reader::StreamReader::try_new(cursor, None).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(format!(
                    "Failed to read Arrow IPC stream: {}",
                    e
                )))
            },
        )?;

        let mut batches = Vec::new();
        for batch in reader {
            batches.push(
                batch.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(format!(
                        "Failed to decode Arrow record batch: {}",
                        e
                    )))
                })?,
            );
        }

        self.insert_arrow_batches(collection, batches, insert_only, tenant_id)
    }

    /// Zero-copy sibling of [`Self::insert_arrow_ipc`]: ingest already-decoded
    /// Arrow `RecordBatch`es. The Python embedding hands these across the FFI
    /// boundary via the **Arrow C Data Interface** (`arrow::pyarrow`), so there is
    /// no IPC serialize on the Python side and no IPC deserialize here — only the
    /// unavoidable Arrow→ProximaRecord model conversion remains. Co-design Pillar C
    /// (move the dominant cost term of the boundary that replaced the deleted
    /// network dimension). Routed to the same embedded vector service; no ports.
    pub fn insert_arrow_batches(
        &self,
        collection: &str,
        batches: Vec<arrow::record_batch::RecordBatch>,
        insert_only: bool,
        tenant_id: Option<&str>,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        self.check_write_access()?;

        let start = std::time::Instant::now();

        let mut records =
            proximadb::network::arrow_ipc::codec::ArrowProtoCodec::batches_to_proxima_records(
                batches,
            )
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(format!(
                    "Failed to convert Arrow batches to ProximaRecords: {}",
                    e
                )))
            })?;

        if let Some(tid) = tenant_id {
            for record in &mut records {
                if record.tenant_id.is_empty() {
                    record.tenant_id = tid.to_string();
                }
            }
        }

        let record_count = records.len() as u64;

        if !insert_only {
            let mut existing_ids = Vec::new();
            for record in &records {
                if self.vector_exists(collection, &record.oid)? {
                    existing_ids.push(record.oid.clone());
                }
            }
            if !existing_ids.is_empty() {
                self.delete_vectors(collection, existing_ids)?;
            }
        }

        let result = self
            .runtime
            .block_on(async {
                self.shared_services
                    .vector_operations_service
                    .insert_batch(collection, records)
                    .await
            })
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            })?;

        if !result.success {
            return Err(Box::new(std::io::Error::other(format!(
                "Arrow batch insert failed: {}",
                result
                    .errors
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "unknown error".to_string())
            ))));
        }

        self.metrics_collector
            .record_insert_us(start.elapsed().as_micros() as u64, record_count);

        Ok(record_count)
    }

    fn ensure_catalog_vector_backing_collection(
        &self,
        statement: &proximadb::services::DdlStatement,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let proximadb::services::DdlStatement::CreateTable { table_name, .. } = statement else {
            return Ok(());
        };
        self.ensure_vector_backing_collection_for_table(table_name)
    }

    fn ensure_dml_vector_backing_collection(
        &self,
        statement: &proximadb::services::DmlStatement,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.ensure_vector_backing_collection_for_table(statement.target_table_name())
    }

    fn ensure_vector_backing_collection_for_table(
        &self,
        table_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let Ok((catalog, table_id)) = self
            .runtime
            .block_on(self.catalog_manager.resolve_table(table_name))
        else {
            return Ok(());
        };
        let schema = self
            .runtime
            .block_on(catalog.get_table(&table_id))
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            })?;
        let Some(vector_column) = schema.columns.iter().find(|column| {
            matches!(
                column.data_type,
                proximadb_data_model::ProximaType::DenseVector { .. }
            )
        }) else {
            return Ok(());
        };
        let dimension = vector_column
            .properties
            .get("dimension")
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        if dimension == 0 {
            return Ok(());
        }

        let exists = self
            .runtime
            .block_on(self.collection_port.get_collection(&table_id.name, None))
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            })?
            .is_some();
        if exists {
            return Ok(());
        }

        self.create_collection(
            &table_id.name,
            dimension,
            schema.properties.get("storage_engine").map(String::as_str),
        )
    }

    /// Execute a unified multi-model query
    ///
    /// Supports SQL-like syntax with extensions for vector similarity,
    /// document queries, graph traversal, and observability data.
    ///
    /// # Arguments
    /// * `query` - SQL-like query string with model-specific extensions
    /// * `query_vector` - Optional query vector for similarity search
    /// * `fusion_strategy` - Optional fusion strategy: "intersection", "union", "rrf", "weighted"
    ///
    /// # Returns
    /// List of query records from matching models
    ///
    /// # Example
    /// ```rust,ignore
    /// let results = db.execute_unified_query(
    ///     "SELECT * FROM products WHERE $.category = 'electronics'",
    ///     Some(vec![0.1; 384]),
    ///     Some("intersection"),
    /// )?;
    /// ```
    pub fn execute_unified_query(
        &self,
        query: &str,
        query_vector: Option<Vec<f32>>,
        fusion_strategy: Option<&str>,
    ) -> Result<Vec<UnifiedQueryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let models = self.detect_query_models(query);
        let fusion = fusion_strategy.unwrap_or("rrf");
        let parameters = query_vector.map(|vector| {
            vec![serde_json::Value::Array(
                vector
                    .into_iter()
                    .map(|value| serde_json::Value::from(value as f64))
                    .collect(),
            )]
        });
        let sql_result = self.execute_sql(
            query,
            parameters,
            self.extract_collection_from_query(query).as_deref(),
        )?;
        let source_model = if models.len() == 1 {
            models[0].to_string()
        } else {
            "unified".to_string()
        };

        let mut all_results: Vec<UnifiedQueryRecord> = sql_result
            .rows
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let id = row
                    .get("id")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("row_{}", index));
                let score = row
                    .get("score")
                    .and_then(|value| value.as_f64())
                    .or_else(|| row.get("similarity").and_then(|value| value.as_f64()))
                    .unwrap_or(1.0);

                UnifiedQueryRecord::new(id, source_model.clone(), score)
                    .with_data(row.to_string())
                    .with_metadata("fusion_strategy", fusion)
                    .with_metadata("models", models.join(","))
            })
            .collect();

        // Apply fusion strategy if multiple result sets
        if fusion == "rrf" && !all_results.is_empty() {
            // Reciprocal Rank Fusion - already sorted by score
            all_results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(all_results)
    }

    /// Explain a unified query's execution plan
    ///
    /// Returns the query decomposition and execution plan without executing.
    ///
    /// # Arguments
    /// * `query` - SQL-like query string
    ///
    /// # Returns
    /// Query execution plan
    pub fn explain_unified_query(
        &self,
        query: &str,
    ) -> Result<UnifiedQueryPlan, Box<dyn std::error::Error + Send + Sync>> {
        let models = self.detect_query_models(query);

        let components: Vec<QueryComponent> = models
            .iter()
            .map(|model| {
                QueryComponent {
                    model: model.to_string(),
                    parallelizable: *model != "graph", // Graph queries often need sequential execution
                    estimated_cost: match *model {
                        "vector" => 1.0,
                        "document" => 0.5,
                        "graph" => 2.0,
                        "logs" => 0.3,
                        "metrics" => 0.2,
                        _ => 1.0,
                    },
                }
            })
            .collect();

        let fusion_strategy = if components.len() > 1 {
            "rrf".to_string() // Default to Reciprocal Rank Fusion for multi-model
        } else {
            "none".to_string()
        };

        Ok(UnifiedQueryPlan {
            fusion_strategy,
            component_count: components.len(),
            components,
        })
    }

    /// Detect which data models are involved in a query
    fn detect_query_models(&self, query: &str) -> Vec<&'static str> {
        let query_upper = query.to_uppercase();
        let mut models = Vec::new();

        if query_upper.contains("VECTOR_SIMILAR")
            || query_upper.contains("VECTOR_SEARCH")
            || query_upper.contains("<->")
            || query_upper.contains("EMBEDDING")
        {
            models.push("vector");
        }

        if query_upper.contains("$.")
            || query_upper.contains("DOCUMENT")
            || query_upper.contains("JSON_")
        {
            models.push("document");
        }

        if query_upper.contains("GRAPH_QUERY")
            || query_upper.contains("MATCH")
            || query_upper.contains("TRAVERSE")
        {
            models.push("graph");
        }

        if query_upper.contains("LOGS(") || query_upper.contains("LOG_SEARCH") {
            models.push("logs");
        }

        if query_upper.contains("METRICS(") || query_upper.contains("METRIC_") {
            models.push("metrics");
        }

        // Default to document if no specific model detected
        if models.is_empty() {
            models.push("document");
        }

        models
    }

    /// Extract collection name from query (simplified)
    fn extract_collection_from_query(&self, query: &str) -> Option<String> {
        // Handle SQL extensions such as:
        //   FROM VECTOR_SEARCH('collection_name', '[...]', 10)
        let query_upper = query.to_uppercase();
        if let Some(from_pos) = query_upper.find("FROM ") {
            let rest = query[from_pos + 5..].trim_start();
            let rest_upper = rest.to_uppercase();

            if rest_upper.starts_with("VECTOR_SEARCH(") {
                let args = &rest["VECTOR_SEARCH(".len()..];
                if let Some(first_quote) = args.find('\'') {
                    let quoted = &args[first_quote + 1..];
                    if let Some(second_quote) = quoted.find('\'') {
                        let collection = quoted[..second_quote].trim();
                        if !collection.is_empty() {
                            return Some(collection.to_string());
                        }
                    }
                }
            }

            let collection: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !collection.is_empty() {
                return Some(collection);
            }
        }
        None
    }

    // ========================================================================
    // Document Store Helper Methods
    // ========================================================================

    /// Convert serde_json::Value to SqlObject
    fn json_to_sql_object(value: &serde_json::Value) -> proximadb::proto::proximadb_v1::SqlObject {
        use proximadb::proto::proximadb_v1::SqlObject;

        let mut fields = std::collections::HashMap::new();
        if let serde_json::Value::Object(map) = value {
            for (key, val) in map {
                fields.insert(key.clone(), Self::json_to_sql_value(val));
            }
        }
        SqlObject { fields }
    }

    /// Convert serde_json::Value to SqlValue
    fn json_to_sql_value(value: &serde_json::Value) -> proximadb::proto::proximadb_v1::SqlValue {
        proximadb::core::search::results::proxima_value_to_sql_value(json_to_proxima_value(
            value.clone(),
        ))
    }

    /// Convert SqlObject to serde_json::Value
    fn sql_object_to_json(obj: &proximadb::proto::proximadb_v1::SqlObject) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for (key, value) in &obj.fields {
            map.insert(
                key.clone(),
                proximadb_records::conversions::sql_value_to_json(value),
            );
        }
        serde_json::Value::Object(map)
    }

    /// Parse a simple filter expression into DocumentFilter conditions
    fn parse_document_filter(
        filter: &str,
    ) -> Vec<proximadb::proto::proximadb_v1::DocFilterCondition> {
        use proximadb::proto::proximadb_v1::{DocFilterCondition, DocFilterOperator};

        let mut conditions = Vec::new();
        if filter.trim().is_empty() {
            return conditions;
        }

        // Split by AND for now (simplified parser)
        let parts: Vec<&str> = filter.split(" AND ").collect();
        for part in parts {
            let part = part.trim();

            // Try to parse operators: >=, <=, !=, =, >, <, CONTAINS
            let operators = [">=", "<=", "!=", "=", ">", "<", "CONTAINS"];
            for op_str in operators {
                if let Some(op_pos) = part.find(op_str) {
                    let path = part[..op_pos].trim().to_string();
                    let value_str = part[op_pos + op_str.len()..].trim();

                    let operator = match op_str {
                        "=" => DocFilterOperator::Eq as i32,
                        "!=" => DocFilterOperator::Ne as i32,
                        ">" => DocFilterOperator::Gt as i32,
                        ">=" => DocFilterOperator::Gte as i32,
                        "<" => DocFilterOperator::Lt as i32,
                        "<=" => DocFilterOperator::Lte as i32,
                        "CONTAINS" => DocFilterOperator::Contains as i32,
                        _ => DocFilterOperator::Eq as i32,
                    };

                    // Parse value
                    let value = Self::parse_filter_value(value_str);

                    conditions.push(DocFilterCondition {
                        path,
                        operator,
                        value: Some(value),
                        values: vec![], // Empty for non-IN operators
                    });
                    break;
                }
            }
        }

        conditions
    }

    /// Parse a value string into SqlValue
    fn parse_filter_value(value_str: &str) -> proximadb::proto::proximadb_v1::SqlValue {
        use proximadb::proto::proximadb_v1::{SqlValue, sql_value::Value};

        let value_str = value_str.trim();

        // Check for quoted string
        if (value_str.starts_with('\'') && value_str.ends_with('\''))
            || (value_str.starts_with('"') && value_str.ends_with('"'))
        {
            return SqlValue {
                value: Some(Value::StringValue(
                    value_str[1..value_str.len() - 1].to_string(),
                )),
            };
        }

        // Check for boolean
        if value_str == "true" {
            return SqlValue {
                value: Some(Value::BoolValue(true)),
            };
        }
        if value_str == "false" {
            return SqlValue {
                value: Some(Value::BoolValue(false)),
            };
        }

        // Check for integer
        if let Ok(i) = value_str.parse::<i64>() {
            return SqlValue {
                value: Some(Value::Int64Value(i)),
            };
        }

        // Check for float
        if let Ok(f) = value_str.parse::<f64>() {
            return SqlValue {
                value: Some(Value::NumberValue(f)),
            };
        }

        // Default to string
        SqlValue {
            value: Some(Value::StringValue(value_str.to_string())),
        }
    }

    fn property_filters_from_map(
        properties: &std::collections::HashMap<String, proximadb::graph::model::PropertyValue>,
    ) -> Vec<proximadb::graph::PropertyFilter> {
        use proximadb::graph::{PropertyFilter, PropertyFilterOperator};

        // Typed filters against typed stored values. When both sides were
        // stringly this happened to match; once storage is typed (see
        // #1698), a StringValue("42") filter would never equal IntValue(42),
        // so the filter map carries PropertyValue end to end.
        properties
            .iter()
            .map(|(key, value)| PropertyFilter {
                key: key.clone(),
                operator: PropertyFilterOperator::Equals as i32,
                value: Some(value.clone()),
            })
            .collect()
    }

    fn observability_namespace_config(
        name: &str,
        retention_hours: Option<u64>,
    ) -> proximadb::proto::proximadb_v1::ObservabilityNamespaceConfig {
        use proximadb::proto::proximadb_v1::{ObservabilityNamespaceConfig, RetentionConfig};

        let retention_hours_val = retention_hours.unwrap_or(720);
        let retention_days = retention_hours_val / 24;

        ObservabilityNamespaceConfig {
            name: name.to_string(),
            retention: Some(RetentionConfig {
                hot_retention_hours: retention_hours_val.min(24),
                warm_retention_days: retention_days.min(7),
                cold_retention_days: retention_days,
                archive_retention_days: 0,
            }),
            ..Default::default()
        }
    }

    fn traversal_response_to_embedded(
        response: proximadb::graph::TraversalResponse,
    ) -> EmbeddedGraphTraversalResult {
        let nodes = response
            .nodes
            .into_iter()
            .map(|node| EmbeddedGraphNode::from_proto(&node))
            .collect();
        let edges = response
            .edges
            .into_iter()
            .map(|edge| EmbeddedGraphEdge::from_proto(&edge))
            .collect();
        let paths = response
            .paths
            .into_iter()
            .map(|path| path.node_ids)
            .collect();
        let stats = response.stats.map(|stats| EmbeddedTraversalStats {
            nodes_visited: stats.nodes_visited,
            edges_traversed: stats.edges_traversed,
            max_depth_reached: stats.max_depth_reached,
            execution_time_microseconds: stats.execution_time_microseconds,
        });

        EmbeddedGraphTraversalResult {
            nodes,
            edges,
            paths,
            stats,
        }
    }

    fn trace_kind_to_string(kind: i32) -> String {
        match proximadb::proto::proximadb_v1::SpanKind::try_from(kind)
            .unwrap_or(proximadb::proto::proximadb_v1::SpanKind::Unspecified)
        {
            proximadb::proto::proximadb_v1::SpanKind::Internal => "INTERNAL",
            proximadb::proto::proximadb_v1::SpanKind::Server => "SERVER",
            proximadb::proto::proximadb_v1::SpanKind::Client => "CLIENT",
            proximadb::proto::proximadb_v1::SpanKind::Producer => "PRODUCER",
            proximadb::proto::proximadb_v1::SpanKind::Consumer => "CONSUMER",
            proximadb::proto::proximadb_v1::SpanKind::Unspecified => "UNSPECIFIED",
        }
        .to_string()
    }

    fn trace_kind_from_string(kind: &str) -> i32 {
        match kind.to_uppercase().as_str() {
            "INTERNAL" => proximadb::proto::proximadb_v1::SpanKind::Internal as i32,
            "SERVER" => proximadb::proto::proximadb_v1::SpanKind::Server as i32,
            "CLIENT" => proximadb::proto::proximadb_v1::SpanKind::Client as i32,
            "PRODUCER" => proximadb::proto::proximadb_v1::SpanKind::Producer as i32,
            "CONSUMER" => proximadb::proto::proximadb_v1::SpanKind::Consumer as i32,
            _ => proximadb::proto::proximadb_v1::SpanKind::Unspecified as i32,
        }
    }

    fn span_status_code_to_string(code: i32) -> String {
        match proximadb::proto::proximadb_v1::SpanStatusCode::try_from(code)
            .unwrap_or(proximadb::proto::proximadb_v1::SpanStatusCode::Unset)
        {
            proximadb::proto::proximadb_v1::SpanStatusCode::Ok => "OK",
            proximadb::proto::proximadb_v1::SpanStatusCode::Error => "ERROR",
            proximadb::proto::proximadb_v1::SpanStatusCode::Unset => "UNSET",
        }
        .to_string()
    }

    fn span_status_code_from_string(code: &str) -> i32 {
        match code.to_uppercase().as_str() {
            "OK" => proximadb::proto::proximadb_v1::SpanStatusCode::Ok as i32,
            "ERROR" => proximadb::proto::proximadb_v1::SpanStatusCode::Error as i32,
            _ => proximadb::proto::proximadb_v1::SpanStatusCode::Unset as i32,
        }
    }

    fn embedded_trace_to_proto(
        trace: EmbeddedTraceSpan,
    ) -> proximadb::proto::proximadb_v1::TraceData {
        let attributes = trace
            .attributes
            .into_iter()
            .map(|(key, value)| (key, Self::json_to_sql_value(&value)))
            .collect();
        let status = Some(proximadb::proto::proximadb_v1::SpanStatus {
            code: Self::span_status_code_from_string(&trace.status_code),
            message: trace.status_message.filter(|msg| !msg.is_empty()),
        });

        proximadb::proto::proximadb_v1::TraceData {
            trace_id: trace.trace_id,
            span_id: trace.span_id,
            parent_span_id: trace.parent_span_id.filter(|id| !id.is_empty()),
            name: trace.name,
            kind: Self::trace_kind_from_string(&trace.kind),
            start_time_ns: trace.start_time_ns,
            end_time_ns: trace.end_time_ns,
            status,
            attributes,
            events: vec![],
            links: vec![],
        }
    }

    fn proto_trace_to_embedded(
        trace: proximadb::proto::proximadb_v1::TraceData,
    ) -> EmbeddedTraceSpan {
        let service = trace
            .attributes
            .get("service.name")
            .map(proximadb_records::conversions::sql_value_to_json)
            .and_then(|value| value.as_str().map(|v| v.to_string()));
        let (status_code, status_message) = trace.status.map_or_else(
            || ("UNSET".to_string(), None),
            |status| {
                (
                    Self::span_status_code_to_string(status.code),
                    status.message.filter(|msg| !msg.is_empty()),
                )
            },
        );

        EmbeddedTraceSpan {
            trace_id: trace.trace_id,
            span_id: trace.span_id,
            parent_span_id: trace.parent_span_id.filter(|id| !id.is_empty()),
            name: trace.name,
            kind: Self::trace_kind_to_string(trace.kind),
            start_time_ns: trace.start_time_ns,
            end_time_ns: trace.end_time_ns,
            service,
            status_code,
            status_message,
            attributes: trace
                .attributes
                .into_iter()
                .map(|(key, value)| {
                    (
                        key,
                        proximadb_records::conversions::sql_value_to_json(&value),
                    )
                })
                .collect(),
        }
    }

    fn sql_response_to_embedded(
        response: proximadb::proto::proximadb_v1::ExecuteQueryResponse,
    ) -> EmbeddedSqlQueryResult {
        let rows = response
            .rows
            .into_iter()
            .map(|row| {
                let mut json_row = serde_json::Map::new();
                for field in row.fields {
                    if let Some(value) = field.value {
                        json_row.insert(
                            field.key,
                            proximadb_records::conversions::sql_value_to_json(&value),
                        );
                    } else {
                        json_row.insert(field.key, serde_json::Value::Null);
                    }
                }
                serde_json::Value::Object(json_row)
            })
            .collect();

        EmbeddedSqlQueryResult {
            rows,
            columns: response.columns,
            column_types: response.column_types,
            row_count: response.rows_returned,
            rows_scanned: response.rows_scanned,
            execution_time_ms: response.execution_time_ms,
        }
    }

    // ========================================================================
    // Observability Operations (Cloud SIEM / Datadog-like logs & metrics)
    // ========================================================================

    /// Create an observability namespace for logs and metrics
    ///
    /// # Arguments
    /// * `name` - Namespace name
    /// * `retention_hours` - Optional data retention period in hours (default: 720 = 30 days)
    ///
    /// # Example
    /// ```rust,ignore
    /// db.create_observability_namespace("production", Some(720))?; // 30 days retention
    /// ```
    pub fn create_observability_namespace(
        &self,
        name: &str,
        retention_hours: Option<u64>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            self.shared_services
                .observability_service
                .create_namespace(Self::observability_namespace_config(name, retention_hours))
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            tracing::info!("📊 EMBEDDED: Created observability namespace '{}'", name);
            Ok(())
        })
    }

    /// Ingest log entries into a namespace
    ///
    /// # Arguments
    /// * `namespace` - Namespace name
    /// * `logs` - Log entries to ingest
    ///
    /// # Returns
    /// Number of logs ingested
    ///
    /// # Example
    /// ```rust,ignore
    /// let logs = vec![
    ///     EmbeddedLogEntry::new("User logged in", "INFO")
    ///         .with_service("auth")
    ///         .with_source("api-gateway"),
    /// ];
    /// let count = db.ingest_logs("production", logs)?;
    /// ```
    pub fn ingest_logs(
        &self,
        namespace: &str,
        logs: Vec<EmbeddedLogEntry>,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::proto::proximadb_v1::{LogEntry, Severity};

        self.runtime.block_on(async {
            let proto_logs: Vec<LogEntry> = logs
                .into_iter()
                .map(|log| {
                    let severity = match log.severity.to_uppercase().as_str() {
                        "TRACE" => Severity::Trace,
                        "DEBUG" => Severity::Debug,
                        "INFO" => Severity::Info,
                        "WARN" | "WARNING" => Severity::Warn,
                        "ERROR" => Severity::Error,
                        "FATAL" | "CRITICAL" => Severity::Fatal,
                        _ => Severity::Unspecified,
                    };

                    LogEntry {
                        timestamp_ns: log.timestamp_ns,
                        severity: severity as i32,
                        message: log.message,
                        fields: log
                            .fields
                            .into_iter()
                            .map(|(k, v)| (k, Self::json_to_sql_value(&v)))
                            .collect(),
                        source: log.source,
                        service: log.service,
                    }
                })
                .collect();

            let result = self
                .shared_services
                .observability_service
                .ingest_logs(namespace, proto_logs, None)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(result.ingested)
        })
    }

    /// Query logs from a namespace
    ///
    /// # Arguments
    /// * `namespace` - Namespace name
    /// * `start_time_ns` - Start of time range (nanoseconds since Unix epoch)
    /// * `end_time_ns` - End of time range (nanoseconds since Unix epoch)
    /// * `query` - Optional query string (text search in message)
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    /// List of matching log entries
    ///
    /// # Example
    /// ```rust,ignore
    /// let logs = db.query_logs(
    ///     "production",
    ///     1703000000000000000,  // start time in nanos
    ///     1703100000000000000,  // end time in nanos
    ///     Some("error"),
    ///     100,
    /// )?;
    /// ```
    pub fn query_logs(
        &self,
        namespace: &str,
        start_time_ns: i64,
        end_time_ns: i64,
        query: Option<&str>,
        limit: u32,
    ) -> Result<Vec<EmbeddedLogEntry>, Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::observability::LogQueryParams;
        use proximadb::proto::proximadb_v1::Severity;

        self.runtime.block_on(async {
            let params = LogQueryParams {
                start_time_ns,
                end_time_ns,
                query: query.map(|s| s.to_string()),
                severities: vec![],
                services: vec![],
                sources: vec![],
                limit,
                cursor: None,
            };

            let result = self
                .shared_services
                .observability_service
                .query_logs(namespace, params)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            let logs = result
                .logs
                .into_iter()
                .map(|log| {
                    let severity_str =
                        match Severity::try_from(log.severity).unwrap_or(Severity::Unspecified) {
                            Severity::Trace => "TRACE",
                            Severity::Debug => "DEBUG",
                            Severity::Info => "INFO",
                            Severity::Warn => "WARN",
                            Severity::Error => "ERROR",
                            Severity::Fatal => "FATAL",
                            Severity::Unspecified => "UNKNOWN",
                        };

                    EmbeddedLogEntry {
                        timestamp_ns: log.timestamp_ns,
                        message: log.message,
                        severity: severity_str.to_string(),
                        service: log.service,
                        source: log.source,
                        fields: log
                            .fields
                            .into_iter()
                            .map(|(k, v)| {
                                (k, proximadb_records::conversions::sql_value_to_json(&v))
                            })
                            .collect(),
                    }
                })
                .collect();

            Ok(logs)
        })
    }

    /// Ingest metric samples into a namespace
    ///
    /// # Arguments
    /// * `namespace` - Namespace name
    /// * `metrics` - Metric samples to ingest
    ///
    /// # Returns
    /// Number of metrics ingested
    ///
    /// # Example
    /// ```rust,ignore
    /// let metrics = vec![
    ///     EmbeddedMetricSample::with_timestamp("http_requests_total", 1703000000000000000, 1234.0)
    ///         .with_label("endpoint", "/api/v1/users"),
    /// ];
    /// let count = db.ingest_metrics("production", metrics)?;
    /// ```
    pub fn ingest_metrics(
        &self,
        namespace: &str,
        metrics: Vec<EmbeddedMetricSample>,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::proto::proximadb_v1::MetricSample;

        self.runtime.block_on(async {
            let proto_metrics: Vec<MetricSample> = metrics
                .into_iter()
                .map(|m| MetricSample {
                    name: m.metric_name,
                    timestamp_ns: m.timestamp_ns,
                    value: m.value,
                    labels: m.labels,
                })
                .collect();

            let result = self
                .shared_services
                .observability_service
                .ingest_metrics(namespace, proto_metrics)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(result.ingested)
        })
    }

    /// Aggregate metrics from a namespace
    ///
    /// # Arguments
    /// * `namespace` - Namespace name
    /// * `metric_name` - Name of the metric
    /// * `aggregation` - Aggregation function ("avg", "sum", "min", "max", "count", "p50", "p90", "p95", "p99")
    /// * `start_time` - Start of time range
    /// * `end_time` - End of time range
    /// * `step_seconds` - Resolution/step in seconds
    ///
    /// # Returns
    /// List of aggregated data points
    ///
    /// # Example
    /// ```rust,ignore
    /// let results = db.aggregate_metrics(
    ///     "production",
    ///     "http_latency_ms",
    ///     "avg",
    ///     Some("2024-01-01T00:00:00Z"),
    ///     None,
    ///     60, // 1 minute buckets
    /// )?;
    /// ```
    pub fn aggregate_metrics(
        &self,
        namespace: &str,
        metric_name: &str,
        aggregation: &str,
        start_time: Option<&str>,
        end_time: Option<&str>,
        step_seconds: u32,
    ) -> Result<Vec<EmbeddedDataPoint>, Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::observability::{MetricAggParams, MetricAggregation};

        self.runtime.block_on(async {
            let start_ns = Self::parse_time_to_nanos(start_time).unwrap_or(0);
            let end_ns = Self::parse_time_to_nanos(end_time).unwrap_or(i64::MAX);

            let agg = match aggregation.to_lowercase().as_str() {
                "avg" | "average" => MetricAggregation::Avg,
                "sum" => MetricAggregation::Sum,
                "min" => MetricAggregation::Min,
                "max" => MetricAggregation::Max,
                "count" => MetricAggregation::Count,
                "rate" => MetricAggregation::Rate,
                "p50" => MetricAggregation::P50,
                "p90" => MetricAggregation::P90,
                "p95" => MetricAggregation::P95,
                "p99" => MetricAggregation::P99,
                _ => MetricAggregation::Avg,
            };

            let params = MetricAggParams {
                metric_name: metric_name.to_string(),
                start_time_ns: start_ns,
                end_time_ns: end_ns,
                aggregation: agg,
                step_seconds,
                label_filters: std::collections::HashMap::new(),
                group_by: vec![],
            };

            let result = self
                .shared_services
                .observability_service
                .aggregate_metrics(namespace, params)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            // Flatten all series into data points
            let points: Vec<EmbeddedDataPoint> = result
                .series
                .into_iter()
                .flat_map(|series| {
                    series.points.into_iter().map(|p| EmbeddedDataPoint {
                        timestamp_ns: p.timestamp_ns,
                        value: p.value,
                    })
                })
                .collect();

            Ok(points)
        })
    }

    /// Ingest trace spans into an observability namespace.
    pub fn ingest_traces(
        &self,
        namespace: &str,
        traces: Vec<EmbeddedTraceSpan>,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let proto_traces = traces
                .into_iter()
                .map(Self::embedded_trace_to_proto)
                .collect();

            let result = self
                .shared_services
                .observability_service
                .ingest_traces(namespace, proto_traces)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(result.ingested)
        })
    }

    /// Query trace spans across one or more traces in a namespace.
    pub fn query_traces(
        &self,
        namespace: &str,
        start_time_ns: i64,
        end_time_ns: i64,
        trace_id: Option<&str>,
        service: Option<&str>,
        operation: Option<&str>,
        min_duration_ns: Option<i64>,
        status: Option<&str>,
        limit: u32,
    ) -> Result<Vec<EmbeddedTraceSpan>, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let params = proximadb::observability::TraceQueryParams {
                start_time_ns,
                end_time_ns,
                trace_id: trace_id.map(str::to_string),
                service: service.map(str::to_string),
                operation: operation.map(str::to_string),
                min_duration_ns,
                status: status.map(Self::span_status_code_from_string),
                limit,
                cursor: None,
            };

            let result = self
                .shared_services
                .observability_service
                .query_traces(namespace, params)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(result
                .traces
                .into_iter()
                .map(Self::proto_trace_to_embedded)
                .collect())
        })
    }

    /// Get a full trace by ID.
    pub fn get_trace(
        &self,
        namespace: &str,
        trace_id: &str,
    ) -> Result<EmbeddedTraceResult, Box<dyn std::error::Error + Send + Sync>> {
        self.runtime.block_on(async {
            let result = self
                .shared_services
                .observability_service
                .get_trace(namespace, trace_id)
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;

            Ok(EmbeddedTraceResult {
                spans: result
                    .spans
                    .into_iter()
                    .map(Self::proto_trace_to_embedded)
                    .collect(),
                complete: result.complete,
            })
        })
    }

    // ========================================================================
    // Observability Helper Methods
    // ========================================================================

    /// Parse time string to nanoseconds
    fn parse_time_to_nanos(time_str: Option<&str>) -> Option<i64> {
        let s = time_str?;

        // Try ISO 8601 format
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
            return Some(dt.timestamp_nanos_opt().unwrap_or(0));
        }

        // Try epoch milliseconds
        if let Ok(ms) = s.parse::<i64>() {
            return Some(ms * 1_000_000); // Convert ms to ns
        }

        // Try relative time (e.g., "now-1h")
        if s.starts_with("now") {
            let now = chrono::Utc::now();
            if s == "now" {
                return Some(now.timestamp_nanos_opt().unwrap_or(0));
            }
            // Parse "now-1h", "now-30m", etc.
            if let Some(offset_str) = s.strip_prefix("now-")
                && let Some(duration) = Self::parse_duration(offset_str)
            {
                return Some((now - duration).timestamp_nanos_opt().unwrap_or(0));
            }
        }

        None
    }

    /// Parse duration string (e.g., "1h", "30m", "1d")
    fn parse_duration(s: &str) -> Option<chrono::Duration> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }

        let (num_str, unit) = s.split_at(s.len() - 1);
        let num: i64 = num_str.parse().ok()?;

        match unit {
            "s" => Some(chrono::Duration::seconds(num)),
            "m" => Some(chrono::Duration::minutes(num)),
            "h" => Some(chrono::Duration::hours(num)),
            "d" => Some(chrono::Duration::days(num)),
            "w" => Some(chrono::Duration::weeks(num)),
            _ => None,
        }
    }

    // ========================================================================
    // Prepared Statements API
    // ========================================================================

    /// Prepare a SQL statement for repeated execution
    ///
    /// This parses and optimizes the SQL query once, caching the result for
    /// efficient repeated execution with different parameters. Use `$1`, `$2`,
    /// etc. as parameter placeholders.
    ///
    /// # Arguments
    /// * `sql` - SQL query with parameter placeholders ($1, $2, etc.)
    ///
    /// # Returns
    /// * Statement ID that can be used with `execute_prepared()`
    ///
    /// # Example
    /// ```rust,ignore
    /// let stmt_id = db.prepare_statement("SELECT * FROM VECTOR_SEARCH($1, $2, 10)")?;
    /// let results1 = db.execute_prepared(&stmt_id, &["embeddings", "[0.1, 0.2]"])?;
    /// let results2 = db.execute_prepared(&stmt_id, &["products", "[0.3, 0.4]"])?;
    /// db.drop_prepared(&stmt_id)?;
    /// ```
    pub fn prepare_statement(
        &self,
        sql: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::query::prepared::{PreparedStatementCache, PreparedStatementConfig};

        // Use a global cache stored in the shared services
        // For now, create a cache per call (in production, this would be stored in shared_services)
        let cache = PreparedStatementCache::new(PreparedStatementConfig::default());

        cache
            .prepare(sql)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            })
    }

    /// Prepare a SQL statement with a custom TTL
    ///
    /// # Arguments
    /// * `sql` - SQL query with parameter placeholders
    /// * `ttl_seconds` - Time-to-live for the prepared statement in seconds
    ///
    /// # Returns
    /// * Statement ID
    pub fn prepare_statement_with_ttl(
        &self,
        sql: &str,
        ttl_seconds: u64,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::query::prepared::{PreparedStatementCache, PreparedStatementConfig};
        use std::time::Duration;

        let cache = PreparedStatementCache::new(PreparedStatementConfig::default());

        cache
            .prepare_with_ttl(sql, Duration::from_secs(ttl_seconds))
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            })
    }

    /// Execute a prepared statement with parameters
    ///
    /// Substitutes the provided parameters into the prepared statement and
    /// executes the query, returning the results.
    ///
    /// # Arguments
    /// * `statement_id` - ID returned from `prepare_statement()`
    /// * `params` - Parameter values to bind (as strings)
    ///
    /// # Returns
    /// * Query results as UnifiedQueryRecord list
    ///
    /// # Example
    /// ```rust,ignore
    /// let stmt_id = db.prepare_statement("SELECT * FROM products WHERE category = $1")?;
    /// let results = db.execute_prepared(&stmt_id, &["electronics"])?;
    /// for record in results {
    ///     println!("Found: {} (score: {})", record.id, record.score);
    /// }
    /// ```
    pub fn execute_prepared(
        &self,
        statement_id: &str,
        params: &[&str],
    ) -> Result<Vec<UnifiedQueryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::query::prepared::{
            ParameterValue, PreparedStatementCache, PreparedStatementConfig,
        };

        let cache = PreparedStatementCache::new(PreparedStatementConfig::default());

        // Convert string params to ParameterValue
        let param_values: Vec<ParameterValue> = params
            .iter()
            .map(|s| ParameterValue::String(s.to_string()))
            .collect();

        // Get the substituted SQL
        let sql = cache.execute_sql(statement_id, &param_values).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            },
        )?;

        // Execute the query using the existing unified query method
        self.execute_unified_query(&sql, None, None)
    }

    /// Execute a prepared statement with typed parameters
    ///
    /// Like `execute_prepared`, but accepts serde_json::Value for parameters,
    /// allowing for typed values (numbers, booleans, arrays, etc.)
    ///
    /// # Arguments
    /// * `statement_id` - ID returned from `prepare_statement()`
    /// * `params` - Parameter values as JSON values
    ///
    /// # Returns
    /// * Query results as UnifiedQueryRecord list
    pub fn execute_prepared_typed(
        &self,
        statement_id: &str,
        params: &[serde_json::Value],
    ) -> Result<Vec<UnifiedQueryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::query::prepared::{
            ParameterValue, PreparedStatementCache, PreparedStatementConfig,
        };

        let cache = PreparedStatementCache::new(PreparedStatementConfig::default());

        // Convert JSON values to ParameterValue
        let param_values: Vec<ParameterValue> =
            params.iter().map(Self::json_to_parameter_value).collect();

        // Get the substituted SQL
        let sql = cache.execute_sql(statement_id, &param_values).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            },
        )?;

        // Execute the query using the existing unified query method
        self.execute_unified_query(&sql, None, None)
    }

    /// Drop a prepared statement from the cache
    ///
    /// Frees resources associated with the prepared statement.
    /// The statement ID becomes invalid after this call.
    ///
    /// # Arguments
    /// * `statement_id` - ID of the statement to drop
    ///
    /// # Example
    /// ```rust,ignore
    /// let stmt_id = db.prepare_statement("SELECT 1")?;
    /// // ... use the statement ...
    /// db.drop_prepared(&stmt_id)?;
    /// ```
    pub fn drop_prepared(
        &self,
        statement_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use proximadb::query::prepared::{PreparedStatementCache, PreparedStatementConfig};

        let cache = PreparedStatementCache::new(PreparedStatementConfig::default());

        cache.drop_statement(statement_id).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            },
        )
    }

    /// Convert JSON value to ParameterValue
    fn json_to_parameter_value(
        v: &serde_json::Value,
    ) -> proximadb::query::prepared::ParameterValue {
        use proximadb::query::prepared::ParameterValue;

        match v {
            serde_json::Value::String(s) => ParameterValue::String(s.clone()),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ParameterValue::Int(i)
                } else if let Some(f) = n.as_f64() {
                    ParameterValue::Float(f)
                } else {
                    ParameterValue::String(n.to_string())
                }
            }
            serde_json::Value::Bool(b) => ParameterValue::Bool(*b),
            serde_json::Value::Null => ParameterValue::Null,
            serde_json::Value::Array(arr) => {
                // Try to parse as vector of f32
                let floats: Vec<f32> = arr
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                if floats.len() == arr.len() {
                    ParameterValue::Vector(floats)
                } else {
                    ParameterValue::Json(v.clone())
                }
            }
            serde_json::Value::Object(_) => ParameterValue::Json(v.clone()),
        }
    }
}

// ============================================================================
// Embedded Mode Types for Observability
// ============================================================================

/// Log entry for embedded mode
#[derive(Debug, Clone)]
pub struct EmbeddedLogEntry {
    /// Timestamp in nanoseconds since Unix epoch
    pub timestamp_ns: i64,
    /// Log message
    pub message: String,
    /// Severity level (DEBUG, INFO, WARN, ERROR, FATAL)
    pub severity: String,
    /// Service name
    pub service: Option<String>,
    /// Source (hostname, component)
    pub source: Option<String>,
    /// Additional fields
    pub fields: std::collections::HashMap<String, serde_json::Value>,
}

impl EmbeddedLogEntry {
    /// Create a new log entry with current timestamp
    pub fn new(message: impl Into<String>, severity: impl Into<String>) -> Self {
        Self {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0),
            message: message.into(),
            severity: severity.into(),
            service: None,
            source: None,
            fields: std::collections::HashMap::new(),
        }
    }

    /// Create a new log entry with specific timestamp
    pub fn with_timestamp(
        timestamp_ns: i64,
        message: impl Into<String>,
        severity: impl Into<String>,
    ) -> Self {
        Self {
            timestamp_ns,
            message: message.into(),
            severity: severity.into(),
            service: None,
            source: None,
            fields: std::collections::HashMap::new(),
        }
    }

    /// Set service name
    pub fn with_service(mut self, service: impl Into<String>) -> Self {
        self.service = Some(service.into());
        self
    }

    /// Set source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add a field
    pub fn with_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.fields.insert(key.into(), value);
        self
    }
}

/// Metric sample for embedded mode
#[derive(Debug, Clone)]
pub struct EmbeddedMetricSample {
    /// Metric name
    pub metric_name: String,
    /// Timestamp in nanoseconds since Unix epoch
    pub timestamp_ns: i64,
    /// Metric value
    pub value: f64,
    /// Labels (dimensions)
    pub labels: std::collections::HashMap<String, String>,
}

impl EmbeddedMetricSample {
    /// Create a new metric sample with current timestamp
    pub fn new(metric_name: impl Into<String>, value: f64) -> Self {
        Self {
            metric_name: metric_name.into(),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0),
            value,
            labels: std::collections::HashMap::new(),
        }
    }

    /// Create a new metric sample with specific timestamp
    pub fn with_timestamp(metric_name: impl Into<String>, timestamp_ns: i64, value: f64) -> Self {
        Self {
            metric_name: metric_name.into(),
            timestamp_ns,
            value,
            labels: std::collections::HashMap::new(),
        }
    }

    /// Add a label
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }
}

/// Data point from metric aggregation
#[derive(Debug, Clone)]
pub struct EmbeddedDataPoint {
    /// Timestamp in nanoseconds since epoch
    pub timestamp_ns: i64,
    /// Aggregated value
    pub value: f64,
}

/// SQL query result for embedded mode
#[derive(Debug, Clone)]
pub struct EmbeddedSqlQueryResult {
    /// Result rows as JSON objects
    pub rows: Vec<serde_json::Value>,
    /// Column names returned by the query
    pub columns: Vec<String>,
    /// Column type names when available
    pub column_types: Vec<String>,
    /// Number of rows returned
    pub row_count: u64,
    /// Number of rows scanned by the planner
    pub rows_scanned: u64,
    /// Total execution time in milliseconds
    pub execution_time_ms: u64,
}

/// Traversal statistics for embedded graph queries
#[derive(Debug, Clone)]
pub struct EmbeddedTraversalStats {
    /// Number of nodes visited
    pub nodes_visited: u32,
    /// Number of edges traversed
    pub edges_traversed: u32,
    /// Maximum depth reached
    pub max_depth_reached: u32,
    /// Execution time in microseconds
    pub execution_time_microseconds: u64,
}

/// Graph traversal result for embedded mode
#[derive(Debug, Clone)]
pub struct EmbeddedGraphTraversalResult {
    /// Nodes visited during traversal
    pub nodes: Vec<EmbeddedGraphNode>,
    /// Edges traversed
    pub edges: Vec<EmbeddedGraphEdge>,
    /// Paths returned by the traversal engine
    pub paths: Vec<Vec<String>>,
    /// Optional traversal statistics
    pub stats: Option<EmbeddedTraversalStats>,
}

/// Trace span result for embedded mode
#[derive(Debug, Clone)]
pub struct EmbeddedTraceSpan {
    /// Distributed trace identifier
    pub trace_id: String,
    /// Span identifier
    pub span_id: String,
    /// Parent span identifier for non-root spans
    pub parent_span_id: Option<String>,
    /// Span/operation name
    pub name: String,
    /// Span kind (server/client/internal/etc.)
    pub kind: String,
    /// Span start time in nanoseconds
    pub start_time_ns: i64,
    /// Span end time in nanoseconds
    pub end_time_ns: i64,
    /// Service name when available
    pub service: Option<String>,
    /// Span status code
    pub status_code: String,
    /// Optional status message
    pub status_message: Option<String>,
    /// Arbitrary span attributes
    pub attributes: std::collections::HashMap<String, serde_json::Value>,
}

/// Full trace assembly result for embedded mode
#[derive(Debug, Clone)]
pub struct EmbeddedTraceResult {
    /// Spans that belong to the trace
    pub spans: Vec<EmbeddedTraceSpan>,
    /// Whether the trace is complete (root span present)
    pub complete: bool,
}

// ============================================================================
// Unified Query Types
// ============================================================================

/// Record returned from a unified multi-model query
#[derive(Debug, Clone)]
pub struct UnifiedQueryRecord {
    /// Record ID
    pub id: String,
    /// Source model (vector, document, graph, logs, metrics)
    pub source_model: String,
    /// Relevance/similarity score
    pub score: f64,
    /// Record data as JSON string
    pub data: String,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl UnifiedQueryRecord {
    /// Create a new query record
    pub fn new(id: impl Into<String>, source_model: impl Into<String>, score: f64) -> Self {
        Self {
            id: id.into(),
            source_model: source_model.into(),
            score,
            data: "{}".to_string(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the data field
    pub fn with_data(mut self, data: impl Into<String>) -> Self {
        self.data = data.into();
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Component of a unified query plan
#[derive(Debug, Clone)]
pub struct QueryComponent {
    /// Model type (vector, document, graph, logs, metrics)
    pub model: String,
    /// Whether this component can run in parallel
    pub parallelizable: bool,
    /// Estimated cost (relative units)
    pub estimated_cost: f64,
}

/// Execution plan for a unified query
#[derive(Debug, Clone)]
pub struct UnifiedQueryPlan {
    /// Fusion strategy used (intersection, union, rrf, weighted)
    pub fusion_strategy: String,
    /// Number of query components
    pub component_count: usize,
    /// Individual query components
    pub components: Vec<QueryComponent>,
}

#[cfg(test)]
mod tests;

/// Flush graph WALs if the handle is dropped without an explicit `close()`.
///
/// `close()` is the documented path, but a Rust caller doing `drop(db)` — or a
/// Python object collected without `close()` — must not silently lose data;
/// that asymmetry is exactly how #1524 stayed invisible (the existing
/// `embedded_flush_persists_and_recovers` test drops the handle, and covers
/// only VECTOR durability, so the graph gap sat under a passing suite).
///
/// Guarded on not already being inside a Tokio runtime: `block_on` from within
/// a runtime thread panics, and Drop can run on odd threads (interpreter
/// teardown, nested runtimes). In that case the flush is skipped with a
/// warning rather than taking the process down during cleanup — callers who
/// need the guarantee should call `close()` explicitly.
impl Drop for EmbeddedProximaDB {
    fn drop(&mut self) {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        if tokio::runtime::Handle::try_current().is_ok() {
            tracing::warn!(
                "EMBEDDED: dropped inside a Tokio runtime; skipping graph WAL flush. \
                 Call close() explicitly to guarantee durability."
            );
            self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
            self.release_coordination();
            return;
        }
        self.close();
    }
}
