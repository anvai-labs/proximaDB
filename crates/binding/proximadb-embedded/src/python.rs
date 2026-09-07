// PyO3 generates impl blocks for #[pymethods] macros that trigger the
// non_local_definitions lint. This is a known issue with pyo3_macros.
#![allow(non_local_definitions)]

//! PyO3 Python Bindings for ProximaDB Embedded Mode
//!
//! This module provides the Python interface for ProximaDB embedded mode.
//! It uses PyO3 with the numpy crate for ZERO-COPY NumPy array handling.
//!
//! ## Performance
//!
//! The numpy crate provides direct access to NumPy array buffers without copying:
//! - `insert_numpy()`: Zero-copy insert from numpy.ndarray (fastest)
//! - `insert()`: Python list conversion (fallback, slower)
//!
//! ## Example
//!
//! ```python
//! import numpy as np
//! from proximadb import ProximaDB
//!
//! db = ProximaDB("./data")
//! db.create_collection("vectors", dimension=384)
//!
//! # Fast: Zero-copy numpy insert
//! vectors = np.random.rand(10000, 384).astype(np.float32)
//! ids = [f"vec_{i}" for i in range(10000)]
//! db.insert_numpy("vectors", ids, vectors)  # Zero-copy!
//!
//! # Slower: Python list (converts via .tolist())
//! db.insert("vectors", ids, vectors.tolist())
//! ```

use pyo3::exceptions::{PyRuntimeError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule, PyTuple};
use pyo3::{Bound, IntoPyObject, PyErr};
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

// Zero-copy numpy support
use numpy::{PyReadonlyArray1, PyReadonlyArray2};

use super::{
    AccessMode, EmbeddedConfig, EmbeddedProximaDB, EmbeddedSqlQueryResult, StorageLocationConfig,
    json_to_proxima_value,
};
use proximadb::core::config::{AdvancedPruneConfig, PruneModeConfig};
use proximadb_embedded_common::proxima_value_to_json;

/// Python wrapper for disk configuration
#[pyclass(name = "DiskConfig", from_py_object)]
#[derive(Clone)]
pub struct PyDiskConfig {
    /// Path to storage directory
    #[pyo3(get, set)]
    pub path: String,
    /// Weight for data distribution (higher = more data)
    #[pyo3(get, set)]
    pub weight: u32,
    /// Tags for storage tier identification
    #[pyo3(get, set)]
    pub tags: Vec<String>,
}

#[pymethods]
impl PyDiskConfig {
    /// Create a new disk configuration
    ///
    /// Args:
    ///     path: Path to the storage directory
    ///     weight: Weight for data distribution (default: 1)
    ///     tags: Optional list of tags (e.g., ["hot", "ssd"])
    #[new]
    #[pyo3(signature = (path, weight=1, tags=None))]
    fn new(path: String, weight: u32, tags: Option<Vec<String>>) -> Self {
        PyDiskConfig {
            path,
            weight,
            tags: tags.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DiskConfig(path='{}', weight={}, tags={:?})",
            self.path, self.weight, self.tags
        )
    }
}

impl From<PyDiskConfig> for StorageLocationConfig {
    fn from(config: PyDiskConfig) -> Self {
        StorageLocationConfig {
            path: config.path,
            weight: config.weight,
            tags: config.tags,
        }
    }
}

/// Python wrapper for search results
#[pyclass(name = "SearchResult")]
pub struct PySearchResult {
    /// Vector ID
    #[pyo3(get)]
    pub id: String,
    /// Similarity score
    #[pyo3(get)]
    pub score: f32,
    /// Associated metadata as dictionary
    metadata_map: HashMap<String, String>,
}

#[pymethods]
impl PySearchResult {
    /// Get metadata as a Python dictionary
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.metadata_map {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id='{}', score={:.4}, metadata={{...}})",
            self.id, self.score
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python wrapper for collection information
#[pyclass(name = "CollectionInfo")]
pub struct PyCollectionInfo {
    /// Collection name
    #[pyo3(get)]
    pub name: String,
    /// Vector dimension
    #[pyo3(get)]
    pub dimension: u32,
    /// Number of vectors in the collection
    #[pyo3(get)]
    pub vector_count: u64,
    /// Storage engine type
    #[pyo3(get)]
    pub engine: String,
    /// Disk usage in bytes for this collection
    #[pyo3(get)]
    pub disk_usage_bytes: u64,
}

#[pymethods]
impl PyCollectionInfo {
    fn __repr__(&self) -> String {
        format!(
            "CollectionInfo(name='{}', dimension={}, vector_count={}, engine='{}', disk_usage_bytes={})",
            self.name, self.dimension, self.vector_count, self.engine, self.disk_usage_bytes
        )
    }
}

/// Python wrapper for storage statistics
#[pyclass(name = "StorageStats")]
pub struct PyStorageStats {
    /// Total vectors across all collections
    #[pyo3(get)]
    pub total_vectors: u64,
    /// Total number of collections
    #[pyo3(get)]
    pub total_collections: u64,
    /// Disk usage in bytes
    #[pyo3(get)]
    pub disk_usage_bytes: u64,
    /// Cache hit rate (0.0 to 1.0)
    #[pyo3(get)]
    pub cache_hit_rate: f64,
}

// ============================================================================
// Checkpoint and Delta Python Bindings
// ============================================================================

/// Python wrapper for checkpoint information
#[pyclass(name = "CheckpointInfo", from_py_object)]
#[derive(Clone)]
pub struct PyCheckpointInfo {
    /// Name of the checkpoint
    #[pyo3(get)]
    pub name: String,
    /// Timestamp when checkpoint was created (ISO 8601 format)
    #[pyo3(get)]
    pub timestamp: String,
    /// Total size of the checkpoint in bytes
    #[pyo3(get)]
    pub size_bytes: u64,
    /// Collections included in this checkpoint
    #[pyo3(get)]
    pub collections: Vec<String>,
    /// Global LSN at checkpoint time
    #[pyo3(get)]
    pub checkpoint_lsn: u64,
}

#[pymethods]
impl PyCheckpointInfo {
    fn __repr__(&self) -> String {
        format!(
            "CheckpointInfo(name='{}', timestamp='{}', size_bytes={}, collections={:?}, checkpoint_lsn={})",
            self.name, self.timestamp, self.size_bytes, self.collections, self.checkpoint_lsn
        )
    }
}

impl From<super::CheckpointInfo> for PyCheckpointInfo {
    fn from(info: super::CheckpointInfo) -> Self {
        Self {
            name: info.name,
            timestamp: info.timestamp.to_rfc3339(),
            size_bytes: info.size_bytes,
            collections: info.collections,
            checkpoint_lsn: info.checkpoint_lsn,
        }
    }
}

/// Python wrapper for delta save information
#[pyclass(name = "DeltaInfo", from_py_object)]
#[derive(Clone)]
pub struct PyDeltaInfo {
    /// Path where delta was saved
    #[pyo3(get)]
    pub path: String,
    /// Timestamp when delta was created (ISO 8601 format)
    #[pyo3(get)]
    pub timestamp: String,
    /// Size of the delta file in bytes
    #[pyo3(get)]
    pub size_bytes: u64,
    /// Number of entries in the delta
    #[pyo3(get)]
    pub entry_count: u64,
    /// Base checkpoint name (if any)
    #[pyo3(get)]
    pub base_checkpoint: Option<String>,
    /// Starting LSN of the delta
    #[pyo3(get)]
    pub start_lsn: u64,
    /// Ending LSN of the delta (inclusive)
    #[pyo3(get)]
    pub end_lsn: u64,
    /// Collections with changes in this delta
    #[pyo3(get)]
    pub affected_collections: Vec<String>,
}

#[pymethods]
impl PyDeltaInfo {
    fn __repr__(&self) -> String {
        format!(
            "DeltaInfo(path='{}', size_bytes={}, entry_count={}, start_lsn={}, end_lsn={})",
            self.path, self.size_bytes, self.entry_count, self.start_lsn, self.end_lsn
        )
    }
}

impl From<super::DeltaInfo> for PyDeltaInfo {
    fn from(info: super::DeltaInfo) -> Self {
        Self {
            path: info.path,
            timestamp: info.timestamp.to_rfc3339(),
            size_bytes: info.size_bytes,
            entry_count: info.entry_count,
            base_checkpoint: info.base_checkpoint,
            start_lsn: info.start_lsn,
            end_lsn: info.end_lsn,
            affected_collections: info.affected_collections,
        }
    }
}

// ============================================================================
// Generic Graph Database Python Bindings
// ============================================================================
//
// These bindings provide a generic, tool-agnostic graph API. Domain-specific
// behavior (like code intelligence for Victor) should be built as an adapter
// layer in the consuming application.

/// Generic graph node with flexible property storage
///
/// This is a domain-agnostic node type. All domain-specific attributes
/// should be stored in the `properties` dict.
///
/// Example:
///     # For a code symbol:
///     node = GraphNode("fn_main", labels=["function"], properties={"name": "main", "file": "main.py"})
///
///     # For a social network:
///     node = GraphNode("user_123", labels=["Person"], properties={"name": "Alice"})
/// Python object -> typed graph `PropertyValue`.
///
/// Order matters: Python `bool` is a subclass of `int`, so it must be probed
/// first or `True` stores as `IntValue(1)`. The final arm keeps the old
/// permissiveness — an arbitrary object still round-trips via `str()` — but
/// scalars, lists, dicts and bytes now keep their types. Before this, EVERY
/// value went through `str()`: `line: 42` persisted as `StringValue("42")`,
/// the engine's numeric property index stayed permanently empty for
/// PyO3-written graphs, and numeric range filters compared lexicographically
/// ("9" > "42"). See proximaDB#1698.
type GraphPropertyValue = proximadb::graph::model::PropertyValue;

fn py_to_property_value(v: &Bound<'_, PyAny>) -> PyResult<GraphPropertyValue> {
    use proximadb::graph::model::property_value::Value;
    use proximadb::graph::model::{PropertyArray, PropertyObject};

    if v.is_none() {
        return Ok(GraphPropertyValue { value: None });
    }
    if let Ok(b) = v.cast::<pyo3::types::PyBool>() {
        return Ok(GraphPropertyValue {
            value: Some(Value::BoolValue(b.is_true())),
        });
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(GraphPropertyValue {
            value: Some(Value::IntValue(i)),
        });
    }
    if let Ok(f) = v.extract::<f64>()
        && v.cast::<pyo3::types::PyFloat>().is_ok()
    {
        return Ok(GraphPropertyValue {
            value: Some(Value::DoubleValue(f)),
        });
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(GraphPropertyValue {
            value: Some(Value::StringValue(s)),
        });
    }
    if let Ok(b) = v.extract::<Vec<u8>>()
        && v.cast::<pyo3::types::PyBytes>().is_ok()
    {
        return Ok(GraphPropertyValue {
            value: Some(Value::BytesValue(b)),
        });
    }
    if let Ok(list) = v.cast::<pyo3::types::PyList>() {
        let mut values = Vec::with_capacity(list.len());
        for item in list.iter() {
            values.push(py_to_property_value(&item)?);
        }
        return Ok(GraphPropertyValue {
            value: Some(Value::ArrayValue(PropertyArray { values })),
        });
    }
    if let Ok(dict) = v.cast::<PyDict>() {
        let mut fields = HashMap::new();
        for (k, val) in dict.iter() {
            fields.insert(k.extract::<String>()?, py_to_property_value(&val)?);
        }
        return Ok(GraphPropertyValue {
            value: Some(Value::ObjectValue(PropertyObject { fields })),
        });
    }
    // Arbitrary objects (Path, UUID, ...) keep the historical str() behaviour.
    Ok(GraphPropertyValue {
        value: Some(Value::StringValue(v.str()?.to_string())),
    })
}

/// Typed graph `PropertyValue` -> Python object (inverse of the above; every
/// variant survives, where the stringly layer dropped four of them).
fn property_value_to_py(py: Python<'_>, v: &GraphPropertyValue) -> PyResult<Py<PyAny>> {
    use proximadb::graph::model::property_value::Value;

    Ok(match &v.value {
        None => py.None(),
        Some(Value::StringValue(s)) => s.into_pyobject(py)?.into_any().unbind(),
        Some(Value::IntValue(i)) => i.into_pyobject(py)?.into_any().unbind(),
        Some(Value::DoubleValue(d)) => d.into_pyobject(py)?.into_any().unbind(),
        Some(Value::BoolValue(b)) => pyo3::types::PyBool::new(py, *b)
            .to_owned()
            .into_any()
            .unbind(),
        Some(Value::BytesValue(b)) => pyo3::types::PyBytes::new(py, b).into_any().unbind(),
        Some(Value::ArrayValue(arr)) => {
            let list = pyo3::types::PyList::empty(py);
            for item in &arr.values {
                list.append(property_value_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        Some(Value::ObjectValue(obj)) => {
            let dict = PyDict::new(py);
            for (k, val) in &obj.fields {
                dict.set_item(k, property_value_to_py(py, val)?)?;
            }
            dict.into_any().unbind()
        }
        Some(Value::VectorValue(vec)) => {
            let list = pyo3::types::PyList::empty(py);
            for f in vec {
                list.append(*f)?;
            }
            list.into_any().unbind()
        }
    })
}

fn py_dict_to_properties(
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<HashMap<String, GraphPropertyValue>> {
    let mut map = HashMap::new();
    if let Some(dict) = dict {
        for (k, v) in dict.iter() {
            map.insert(k.extract::<String>()?, py_to_property_value(&v)?);
        }
    }
    Ok(map)
}

#[pyclass(name = "GraphNode", from_py_object)]
#[derive(Clone)]
pub struct PyGraphNode {
    /// Unique node ID
    #[pyo3(get, set)]
    pub id: String,
    /// Node labels/types
    labels_vec: Vec<String>,
    /// Typed property storage (proximaDB#1698: was HashMap<String, String>)
    properties_map: HashMap<String, GraphPropertyValue>,
}

#[pymethods]
impl PyGraphNode {
    /// Create a new graph node
    ///
    /// Args:
    ///     id: Unique node identifier
    ///     labels: List of labels/types for this node
    ///     properties: Dictionary of properties
    #[new]
    #[pyo3(signature = (id, labels=None, properties=None))]
    fn new(
        id: String,
        labels: Option<Vec<String>>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let properties_map = py_dict_to_properties(properties)?;

        Ok(PyGraphNode {
            id,
            labels_vec: labels.unwrap_or_default(),
            properties_map,
        })
    }

    /// Get labels as a list
    #[getter]
    fn labels(&self) -> Vec<String> {
        self.labels_vec.clone()
    }

    /// Set labels
    #[setter]
    fn set_labels(&mut self, labels: Vec<String>) {
        self.labels_vec = labels;
    }

    /// Get properties as a dictionary (typed: int/float/bool/str/bytes/list/dict)
    #[getter]
    fn properties(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.properties_map {
            dict.set_item(k, property_value_to_py(py, v)?)?;
        }
        Ok(dict.into())
    }

    /// Set properties from a dictionary
    #[setter]
    fn set_properties(&mut self, properties: &Bound<'_, PyDict>) -> PyResult<()> {
        self.properties_map = py_dict_to_properties(Some(properties))?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("GraphNode(id='{}', labels={:?})", self.id, self.labels_vec)
    }
}

impl From<super::GraphNode> for PyGraphNode {
    fn from(node: super::GraphNode) -> Self {
        PyGraphNode {
            id: node.id,
            labels_vec: node.labels,
            properties_map: node.properties,
        }
    }
}

impl From<PyGraphNode> for super::GraphNode {
    fn from(node: PyGraphNode) -> Self {
        super::GraphNode {
            id: node.id,
            labels: node.labels_vec,
            properties: node.properties_map,
        }
    }
}

/// Generic graph edge with flexible property storage
#[pyclass(name = "GraphEdge", from_py_object)]
#[derive(Clone)]
pub struct PyGraphEdge {
    /// Optional edge ID
    #[pyo3(get, set)]
    pub id: Option<String>,
    /// Source node ID
    #[pyo3(get, set)]
    pub from_node_id: String,
    /// Destination node ID
    #[pyo3(get, set)]
    pub to_node_id: String,
    /// Edge type
    #[pyo3(get, set)]
    pub edge_type: String,
    /// Optional weight
    #[pyo3(get, set)]
    pub weight: Option<f64>,
    /// Flexible property storage
    properties_map: HashMap<String, GraphPropertyValue>,
}

#[pymethods]
impl PyGraphEdge {
    /// Create a new graph edge
    ///
    /// Args:
    ///     from_node_id: Source node ID
    ///     to_node_id: Destination node ID
    ///     edge_type: Type of relationship
    ///     id: Optional explicit edge ID
    ///     weight: Optional edge weight
    ///     properties: Optional property dictionary
    #[new]
    #[pyo3(signature = (from_node_id, to_node_id, edge_type, id=None, weight=None, properties=None))]
    fn new(
        from_node_id: String,
        to_node_id: String,
        edge_type: String,
        id: Option<String>,
        weight: Option<f64>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let properties_map = py_dict_to_properties(properties)?;

        Ok(PyGraphEdge {
            id,
            from_node_id,
            to_node_id,
            edge_type,
            weight,
            properties_map,
        })
    }

    /// Get properties as a dictionary (typed: int/float/bool/str/bytes/list/dict)
    #[getter]
    fn properties(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.properties_map {
            dict.set_item(k, property_value_to_py(py, v)?)?;
        }
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphEdge(from='{}', to='{}', type='{}')",
            self.from_node_id, self.to_node_id, self.edge_type
        )
    }
}

impl From<super::GraphEdge> for PyGraphEdge {
    fn from(edge: super::GraphEdge) -> Self {
        PyGraphEdge {
            id: edge.id,
            from_node_id: edge.from_node_id,
            to_node_id: edge.to_node_id,
            edge_type: edge.edge_type,
            weight: edge.weight,
            properties_map: edge.properties,
        }
    }
}

impl From<PyGraphEdge> for super::GraphEdge {
    fn from(edge: PyGraphEdge) -> Self {
        super::GraphEdge {
            id: edge.id,
            from_node_id: edge.from_node_id,
            to_node_id: edge.to_node_id,
            edge_type: edge.edge_type,
            weight: edge.weight,
            properties: edge.properties_map,
        }
    }
}

/// Graph statistics
#[pyclass(name = "GraphStats")]
pub struct PyGraphStats {
    /// Total number of nodes
    #[pyo3(get)]
    pub total_nodes: u64,
    /// Total number of edges
    #[pyo3(get)]
    pub total_edges: u64,
}

// ============================================================================
// Observability Metrics Python Bindings
// ============================================================================

/// Latency statistics for an operation type
#[pyclass(name = "LatencyStats", from_py_object)]
#[derive(Clone)]
pub struct PyLatencyStats {
    /// Number of operations recorded
    #[pyo3(get)]
    pub count: u64,
    /// Minimum latency in milliseconds
    #[pyo3(get)]
    pub min_ms: f64,
    /// Maximum latency in milliseconds
    #[pyo3(get)]
    pub max_ms: f64,
    /// Mean latency in milliseconds
    #[pyo3(get)]
    pub mean_ms: f64,
    /// 50th percentile latency in milliseconds
    #[pyo3(get)]
    pub p50_ms: f64,
    /// 95th percentile latency in milliseconds
    #[pyo3(get)]
    pub p95_ms: f64,
    /// 99th percentile latency in milliseconds
    #[pyo3(get)]
    pub p99_ms: f64,
}

#[pymethods]
impl PyLatencyStats {
    fn __repr__(&self) -> String {
        format!(
            "LatencyStats(count={}, p50={:.2}ms, p95={:.2}ms, p99={:.2}ms)",
            self.count, self.p50_ms, self.p95_ms, self.p99_ms
        )
    }
}

impl From<super::LatencyStats> for PyLatencyStats {
    fn from(stats: super::LatencyStats) -> Self {
        PyLatencyStats {
            count: stats.count,
            min_ms: stats.min_ms,
            max_ms: stats.max_ms,
            mean_ms: stats.mean_ms,
            p50_ms: stats.p50_ms,
            p95_ms: stats.p95_ms,
            p99_ms: stats.p99_ms,
        }
    }
}

/// Comprehensive embedded metrics snapshot
///
/// Contains latency histograms, operation counters, cache statistics,
/// and WAL statistics for embedded mode observability.
///
/// Example:
///     ```python
///     metrics = db.metrics()
///     print(f"p99 search latency: {metrics.search_latency.p99_ms}ms")
///     print(f"Cache hit rate: {metrics.cache_hit_rate * 100}%")
///     print(f"Total searches: {metrics.total_searches}")
///     ```
#[pyclass(name = "EmbeddedMetrics", from_py_object)]
#[derive(Clone)]
pub struct PyEmbeddedMetrics {
    // Latency histograms
    search_latency_inner: super::LatencyStats,
    insert_latency_inner: super::LatencyStats,
    flush_latency_inner: super::LatencyStats,
    delete_latency_inner: super::LatencyStats,
    get_latency_inner: super::LatencyStats,

    // Operation counters
    /// Total search operations
    #[pyo3(get)]
    pub total_searches: u64,
    /// Total insert operations
    #[pyo3(get)]
    pub total_inserts: u64,
    /// Total delete operations
    #[pyo3(get)]
    pub total_deletes: u64,
    /// Total flush operations
    #[pyo3(get)]
    pub total_flushes: u64,
    /// Total get operations
    #[pyo3(get)]
    pub total_gets: u64,
    /// Total upsert operations
    #[pyo3(get)]
    pub total_upserts: u64,
    /// Total vectors inserted
    #[pyo3(get)]
    pub total_vectors_inserted: u64,
    /// Total vectors deleted
    #[pyo3(get)]
    pub total_vectors_deleted: u64,
    /// Total bytes written
    #[pyo3(get)]
    pub total_bytes_written: u64,
    /// Total errors
    #[pyo3(get)]
    pub total_errors: u64,

    // Cache statistics
    /// Cache hit rate (0.0 to 1.0)
    #[pyo3(get)]
    pub cache_hit_rate: f64,
    /// Total cache hits
    #[pyo3(get)]
    pub cache_hits: u64,
    /// Total cache misses
    #[pyo3(get)]
    pub cache_misses: u64,
    /// Number of entries in cache
    #[pyo3(get)]
    pub cache_entries: u64,
    /// Memory used by cache in bytes
    #[pyo3(get)]
    pub cache_memory_bytes: u64,
    /// Total cache evictions
    #[pyo3(get)]
    pub cache_evictions: u64,

    // WAL statistics
    /// Pending bytes in WAL
    #[pyo3(get)]
    pub wal_pending_bytes: u64,
    /// Number of WAL segments
    #[pyo3(get)]
    pub wal_segments_count: u64,
    /// Total bytes written to WAL
    #[pyo3(get)]
    pub wal_total_bytes_written: u64,

    // Timing
    /// Database uptime in seconds
    #[pyo3(get)]
    pub uptime_secs: u64,
}

#[pymethods]
impl PyEmbeddedMetrics {
    /// Get search operation latency statistics
    #[getter]
    fn search_latency(&self) -> PyLatencyStats {
        self.search_latency_inner.clone().into()
    }

    /// Get insert operation latency statistics
    #[getter]
    fn insert_latency(&self) -> PyLatencyStats {
        self.insert_latency_inner.clone().into()
    }

    /// Get flush operation latency statistics
    #[getter]
    fn flush_latency(&self) -> PyLatencyStats {
        self.flush_latency_inner.clone().into()
    }

    /// Get delete operation latency statistics
    #[getter]
    fn delete_latency(&self) -> PyLatencyStats {
        self.delete_latency_inner.clone().into()
    }

    /// Get get operation latency statistics
    #[getter]
    fn get_latency(&self) -> PyLatencyStats {
        self.get_latency_inner.clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddedMetrics(searches={}, inserts={}, cache_hit_rate={:.1}%, uptime={}s)",
            self.total_searches,
            self.total_inserts,
            self.cache_hit_rate * 100.0,
            self.uptime_secs
        )
    }
}

impl From<super::EmbeddedMetrics> for PyEmbeddedMetrics {
    fn from(m: super::EmbeddedMetrics) -> Self {
        PyEmbeddedMetrics {
            search_latency_inner: m.search_latency,
            insert_latency_inner: m.insert_latency,
            flush_latency_inner: m.flush_latency,
            delete_latency_inner: m.delete_latency,
            get_latency_inner: m.get_latency,

            total_searches: m.total_searches,
            total_inserts: m.total_inserts,
            total_deletes: m.total_deletes,
            total_flushes: m.total_flushes,
            total_gets: m.total_gets,
            total_upserts: m.total_upserts,
            total_vectors_inserted: m.total_vectors_inserted,
            total_vectors_deleted: m.total_vectors_deleted,
            total_bytes_written: m.total_bytes_written,
            total_errors: m.total_errors,

            cache_hit_rate: m.cache_hit_rate,
            cache_hits: m.cache_hits,
            cache_misses: m.cache_misses,
            cache_entries: m.cache_entries,
            cache_memory_bytes: m.cache_memory_bytes,
            cache_evictions: m.cache_evictions,

            wal_pending_bytes: m.wal_pending_bytes,
            wal_segments_count: m.wal_segments_count,
            wal_total_bytes_written: m.wal_total_bytes_written,

            uptime_secs: m.uptime_secs,
        }
    }
}

#[pymethods]
impl PyGraphStats {
    fn __repr__(&self) -> String {
        format!(
            "GraphStats(total_nodes={}, total_edges={})",
            self.total_nodes, self.total_edges
        )
    }
}

impl From<super::GraphStats> for PyGraphStats {
    fn from(stats: super::GraphStats) -> Self {
        PyGraphStats {
            total_nodes: stats.total_nodes,
            total_edges: stats.total_edges,
        }
    }
}

/// Python wrapper for streaming search results
#[pyclass(name = "StreamingSearchResult", from_py_object)]
#[derive(Clone)]
pub struct PyStreamingSearchResult {
    /// Vector ID
    #[pyo3(get)]
    pub id: String,
    /// Similarity score
    #[pyo3(get)]
    pub score: f32,
    /// Associated metadata as dictionary
    metadata_map: HashMap<String, String>,
}

#[pymethods]
impl PyStreamingSearchResult {
    /// Get metadata as a Python dictionary
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.metadata_map {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingSearchResult(id='{}', score={:.4})",
            self.id, self.score
        )
    }
}

/// Python iterator for streaming search results
///
/// This iterator yields batches of search results, providing memory-efficient
/// access to large result sets. Results are fetched in configurable batch sizes.
///
/// Example:
///     ```python
///     # Create streaming search iterator
///     for batch in db.search_streaming("my_collection", query, top_k=10000, batch_size=100):
///         for result in batch:
///             print(f"Found: {result.id} (score: {result.score})")
///     ```
#[pyclass(name = "SearchStreamIterator")]
pub struct PySearchStreamIterator {
    /// The underlying Rust iterator wrapped in an Option for take semantics
    inner: Option<super::EmbeddedSearchIterator>,
}

#[pymethods]
impl PySearchStreamIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<Vec<PyStreamingSearchResult>>> {
        let inner = match self.inner.as_mut() {
            Some(it) => it,
            None => return Ok(None),
        };

        match inner.next() {
            Some(Ok(batch)) => {
                let py_batch: Vec<PyStreamingSearchResult> = batch
                    .into_iter()
                    .map(|r| PyStreamingSearchResult {
                        id: r.id,
                        score: r.score,
                        metadata_map: r.metadata,
                    })
                    .collect();

                if py_batch.is_empty() {
                    // Empty batch signals completion
                    self.inner = None;
                    Ok(None)
                } else {
                    Ok(Some(py_batch))
                }
            }
            Some(Err(e)) => {
                self.inner = None;
                Err(PyRuntimeError::new_err(format!(
                    "Streaming search error: {}",
                    e
                )))
            }
            None => {
                self.inner = None;
                Ok(None)
            }
        }
    }

    /// Get the current batch size configuration
    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.as_ref().map(|i| i.batch_size()).unwrap_or(0)
    }

    /// Get the number of results returned so far
    #[getter]
    fn results_returned(&self) -> usize {
        self.inner
            .as_ref()
            .map(|i| i.results_returned())
            .unwrap_or(0)
    }

    /// Get the total results requested (top_k)
    #[getter]
    fn top_k(&self) -> usize {
        self.inner.as_ref().map(|i| i.top_k()).unwrap_or(0)
    }

    /// Check if the iterator is complete
    #[getter]
    fn is_complete(&self) -> bool {
        self.inner.as_ref().map(|i| i.is_complete()).unwrap_or(true)
    }

    fn __repr__(&self) -> String {
        if let Some(inner) = &self.inner {
            format!(
                "SearchStreamIterator(batch_size={}, returned={}, top_k={}, complete={})",
                inner.batch_size(),
                inner.results_returned(),
                inner.top_k(),
                inner.is_complete()
            )
        } else {
            "SearchStreamIterator(exhausted)".to_string()
        }
    }
}

#[pymethods]
impl PyStorageStats {
    /// Get disk usage in human-readable format
    fn disk_usage_human(&self) -> String {
        let bytes = self.disk_usage_bytes as f64;
        if bytes >= 1e12 {
            format!("{:.2} TB", bytes / 1e12)
        } else if bytes >= 1e9 {
            format!("{:.2} GB", bytes / 1e9)
        } else if bytes >= 1e6 {
            format!("{:.2} MB", bytes / 1e6)
        } else if bytes >= 1e3 {
            format!("{:.2} KB", bytes / 1e3)
        } else {
            format!("{} B", self.disk_usage_bytes)
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "StorageStats(vectors={}, collections={}, disk={}, cache_hit={:.1}%)",
            self.total_vectors,
            self.total_collections,
            self.disk_usage_human(),
            self.cache_hit_rate * 100.0
        )
    }
}

/// ProximaDB embedded database instance
///
/// This class provides a Python interface to ProximaDB's embedded mode,
/// allowing direct in-process access to the vector database without
/// network overhead.
///
/// Example:
///     ```python
///     from proximadb_embedded import ProximaDB, DiskConfig
///
///     # Multi-disk configuration
///     db = ProximaDB(
///         data_dirs=[
///             DiskConfig("/nvme/data", weight=2),  # Fast SSD
///             DiskConfig("/hdd/data", weight=1),   # Slower HDD
///         ],
///         metadata_dir="/nvme/metadata",
///         cache_size_mb=2048
///     )
///
///     # Create collection
///     db.create_collection("embeddings", dimension=768)
///
///     # Insert vectors (supports NumPy arrays)
///     import numpy as np
///     vectors = np.random.rand(1000, 768).astype(np.float32)
///     db.insert("embeddings", ids=[f"v{i}" for i in range(1000)], vectors=vectors)
///
///     # Search
///     results = db.search("embeddings", query=vectors[0], top_k=10)
///     ```
#[pyclass(name = "ProximaDB")]
pub struct PyProximaDB {
    inner: Mutex<Option<Arc<EmbeddedProximaDB>>>,
}

impl PyProximaDB {
    fn db(&self) -> PyResult<Arc<EmbeddedProximaDB>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Database mutex poisoned"))?
            .as_ref()
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("Database has been closed"))
    }

    fn db_opt(&self) -> Option<Arc<EmbeddedProximaDB>> {
        self.inner.lock().ok()?.as_ref().cloned()
    }
}

fn set_approx_defaults(mode_str: &str) -> (String, f32, usize, usize) {
    match mode_str {
        "ratio" => ("ratio".to_string(), 0.1, 1, 0),
        _ => ("sqrt".to_string(), 0.2, 1, 0),
    }
}

#[pymethods]
impl PyProximaDB {
    /// Create a new embedded ProximaDB instance
    ///
    /// Args:
    ///     data_dirs: Path string, list of paths, or list of DiskConfig
    ///     metadata_dir: Optional metadata directory path
    ///     cache_size_mb: Cache size in megabytes (default: 512)
    ///     default_engine: Default storage engine (default: "sst")
    ///     enable_wal: Enable write-ahead log (default: true)
    ///     prune_mode: Prune mode for approximate search
    ///     mode: Access mode for multi-process coordination:
    ///           - "exclusive": Single writer, exclusive access (default)
    ///           - "leader" or "leader_follower": Leader/follower mode
    ///           - "follower" or "shared_read": Read-only follower
    ///     node_id: Node ID for leader election (optional, auto-generated if not set)
    ///
    /// Example:
    ///     ```python
    ///     # Single process, exclusive access (default)
    ///     db = ProximaDB("/data/vectors")
    ///
    ///     # Read-only follower mode
    ///     db = ProximaDB("/data/vectors", mode="follower")
    ///
    ///     # Leader/follower mode with explicit node ID
    ///     db = ProximaDB("/data/vectors", mode="leader", node_id="node1")
    ///     ```
    #[new]
    #[pyo3(signature = (
        data_dirs=None,
        metadata_dir=None,
        cache_size_mb=512,
        default_engine="sst",
        enable_wal=true,
        prune_mode=None,
        mode="exclusive",
        node_id=None
    ))]
    fn new(
        py: Python,
        data_dirs: Option<&Bound<'_, PyAny>>,
        metadata_dir: Option<String>,
        cache_size_mb: usize,
        default_engine: &str,
        enable_wal: bool,
        prune_mode: Option<&Bound<'_, PyAny>>,
        mode: &str,
        node_id: Option<String>,
    ) -> PyResult<Self> {
        let mut final_prune_config = None;

        if let Some(mode_arg) = prune_mode {
            if let Ok(s) = mode_arg.extract::<String>() {
                let s_lower = s.to_lowercase();
                if s_lower == "exact" {
                    // None implies exact; leave as default
                } else if s_lower == "approximate" || s_lower == "sqrt" || s_lower == "ratio" {
                    final_prune_config = Some(PruneModeConfig::Simple(s_lower));
                } else {
                    warn_user(
                        py,
                        "Invalid 'prune_mode' string provided. Falling back to 'exact' mode.",
                        1,
                    )?;
                }
            } else if let Ok(dict) = mode_arg.cast::<PyDict>() {
                let prune_type = dict
                    .get_item("type")?
                    .and_then(|t| t.extract::<String>().ok())
                    .unwrap_or_else(|| "sqrt".to_string());

                final_prune_config = Some(PruneModeConfig::Advanced(AdvancedPruneConfig {
                    r#type: prune_type,
                    min_keep: dict.get_item("min_keep")?.and_then(|v| v.extract().ok()),
                    max_keep: dict.get_item("max_keep")?.and_then(|v| v.extract().ok()),
                    ratio: dict.get_item("ratio")?.and_then(|v| v.extract().ok()),
                }));
            } else {
                return Err(PyValueError::new_err(
                    "'prune_mode' must be a string, a dictionary, or None.",
                ));
            }
        }

        // Parse data directories
        let storage_locations = if let Some(dirs) = data_dirs {
            if let Ok(path) = dirs.extract::<String>() {
                // Single path string
                vec![StorageLocationConfig::new(path)]
            } else if let Ok(configs) = dirs.extract::<Vec<PyDiskConfig>>() {
                // List of DiskConfig
                configs.into_iter().map(|c| c.into()).collect()
            } else if let Ok(paths) = dirs.extract::<Vec<String>>() {
                // List of path strings
                paths.into_iter().map(StorageLocationConfig::new).collect()
            } else {
                return Err(PyValueError::new_err(
                    "data_dirs must be a path string, list of paths, or list of DiskConfig",
                ));
            }
        } else {
            vec![StorageLocationConfig::new("./data")]
        };

        // Determine metadata path
        let metadata_path = metadata_dir.unwrap_or_else(|| {
            let first_dir = &storage_locations[0].path;
            if first_dir.starts_with("file://") {
                format!("{}/metadata", first_dir)
            } else {
                format!("{}/metadata", first_dir)
            }
        });

        // Parse access mode
        let access_mode = match mode.to_lowercase().as_str() {
            "exclusive" => AccessMode::Exclusive,
            "leader" | "leader_follower" | "writer" => AccessMode::LeaderFollower,
            "follower" | "shared_read" | "shared" | "reader" => AccessMode::SharedRead,
            _ => {
                warn_user(
                    py,
                    &format!("Invalid 'mode' '{}'. Using 'exclusive' mode.", mode),
                    1,
                )?;
                AccessMode::Exclusive
            }
        };

        let config = EmbeddedConfig {
            storage_locations,
            metadata_path,
            cache_size_mb,
            default_engine: default_engine.to_string(),
            enable_wal,
            // [AGENT_MODIFICATION]: The wal_sync_mode is not exposed in the new signature, so use a smart default.
            wal_sync_mode: "batch".to_string(),
            block_prune_mode: "exact".to_string(),
            block_prune_ratio: 0.0,
            block_prune_min_keep: 0,
            block_prune_max_keep: 0,
            // RL planner is enabled by default for adaptive query optimization
            enable_rl_planner: true,
            rl_policy_path: None, // Will use default path based on data_dir
            // Multi-process coordination
            access_mode,
            node_id,
            tenant_id: None,
        };

        let mut config = config;

        // Translate PruneModeConfig to EmbeddedConfig fields
        if let Some(prune_cfg) = final_prune_config {
            match prune_cfg {
                PruneModeConfig::Simple(s) => {
                    let (mode, ratio, min_k, max_k) = set_approx_defaults(&s);
                    config.block_prune_mode = mode;
                    config.block_prune_ratio = ratio;
                    config.block_prune_min_keep = min_k;
                    config.block_prune_max_keep = max_k;
                }
                PruneModeConfig::Advanced(adv) => {
                    let (_mode, def_ratio, def_min_k, def_max_k) = set_approx_defaults(&adv.r#type);
                    config.block_prune_mode = adv.r#type;
                    config.block_prune_ratio = adv.ratio.unwrap_or(def_ratio);
                    config.block_prune_min_keep = adv.min_keep.unwrap_or(def_min_k);
                    config.block_prune_max_keep = adv.max_keep.unwrap_or(def_max_k);
                }
            }
        } else {
            config.block_prune_mode = "exact".to_string();
            // leave ratio/min/max at safe exact defaults
        }

        let db = EmbeddedProximaDB::new(config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create database: {}", e)))?;

        Ok(PyProximaDB {
            inner: Mutex::new(Some(Arc::new(db))),
        })
    }

    /// Create a new collection
    ///
    /// Args:
    ///     name: Collection name
    ///     dimension: Vector dimension
    ///     engine: Storage engine type (optional, uses default if not specified)
    ///
    /// Examples:
    ///     db.create_collection("vectors", 768, "sst")
    ///     db.create_collection("vectors", 768, "helix")
    ///
    /// Raises:
    ///     RuntimeError: If collection creation fails
    #[pyo3(signature = (name, dimension, engine=None))]
    fn create_collection(&self, name: &str, dimension: u32, engine: Option<&str>) -> PyResult<()> {
        self.db()?
            .create_collection(name, dimension, engine)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create collection: {}", e)))
    }

    /// Delete a collection
    ///
    /// Args:
    ///     name: Collection name to delete
    ///
    /// Raises:
    ///     RuntimeError: If collection deletion fails
    fn delete_collection(&self, name: &str) -> PyResult<()> {
        self.db()?
            .delete_collection(name)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete collection: {}", e)))
    }

    /// Get collection information
    ///
    /// Args:
    ///     name: Collection name
    ///
    /// Returns:
    ///     CollectionInfo or None if collection doesn't exist
    fn get_collection(&self, name: &str) -> PyResult<Option<PyCollectionInfo>> {
        self.db()?
            .get_collection(name)
            .map(|opt| {
                opt.map(|info| PyCollectionInfo {
                    name: info.name,
                    dimension: info.dimension,
                    vector_count: info.vector_count,
                    engine: info.engine,
                    disk_usage_bytes: info.disk_usage_bytes,
                })
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get collection: {}", e)))
    }

    /// List all collections
    ///
    /// Returns:
    ///     List of CollectionInfo
    fn list_collections(&self) -> PyResult<Vec<PyCollectionInfo>> {
        self.db()?
            .list_collections()
            .map(|collections| {
                collections
                    .into_iter()
                    .map(|info| PyCollectionInfo {
                        name: info.name,
                        dimension: info.dimension,
                        vector_count: info.vector_count,
                        engine: info.engine,
                        disk_usage_bytes: info.disk_usage_bytes,
                    })
                    .collect()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to list collections: {}", e)))
    }

    #[pyo3(signature = (collection, requested_partitions=1))]
    fn plan_partitions(
        &self,
        py: Python<'_>,
        collection: &str,
        requested_partitions: u32,
    ) -> PyResult<Py<PyAny>> {
        let requested = requested_partitions.max(1);
        let partitions = self
            .db()?
            .plan_partitions(collection, requested)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to plan partitions: {}", e)))?;

        let partition_items = PyList::empty(py);
        for partition in &partitions {
            let partition_dict = PyDict::new(py);
            partition_dict.set_item("partition_id", partition.partition_id)?;
            partition_dict.set_item("preferred_locations", &partition.preferred_locations)?;
            partition_dict.set_item("estimated_rows", partition.estimated_rows)?;
            partition_dict.set_item("estimated_bytes", partition.estimated_bytes)?;

            let splits = PyList::empty(py);
            for split in &partition.splits {
                let split_dict = PyDict::new(py);
                split_dict.set_item("split_id", &split.split_id)?;
                split_dict.set_item("file_path", &split.file_path)?;
                split_dict.set_item("offset", split.offset)?;
                split_dict.set_item("length", split.length)?;
                split_dict.set_item("estimated_rows", split.statistics.row_count)?;
                split_dict.set_item("estimated_bytes", split.statistics.byte_size)?;
                splits.append(split_dict)?;
            }
            partition_dict.set_item("splits", splits)?;
            partition_items.append(partition_dict)?;
        }

        let dict = PyDict::new(py);
        dict.set_item("collection", collection)?;
        dict.set_item("requested_partitions", requested)?;
        dict.set_item("planned_partitions", partitions.len())?;
        dict.set_item("effective_read_partitions", partitions.len())?;
        dict.set_item("planner", "whole_collection_fallback")?;
        dict.set_item("execution_scope", "local_process")?;
        dict.set_item("safe_parallelism", partitions.len())?;
        dict.set_item("partitions", partition_items)?;
        if requested as usize > partitions.len() {
            dict.set_item(
                "rejected_parallelism_reason",
                "split-bound readers are not wired yet; returning more collection:// partitions would duplicate scans",
            )?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Insert vectors into a collection
    ///
    /// Args:
    ///     collection: Collection name
    ///     ids: List of vector IDs
    ///     vectors: 2D array of vectors (can be NumPy array or list of lists)
    ///     metadata: Optional list of metadata dictionaries
    ///
    /// Returns:
    ///     Number of vectors inserted
    ///
    /// Example:
    ///     ```python
    ///     import numpy as np
    ///     vectors = np.random.rand(100, 768).astype(np.float32)
    ///     ids = [f"vec_{i}" for i in range(100)]
    ///     metadata = [{"category": "A", "score": 0.9} for _ in range(100)]
    ///     count = db.insert("my_collection", ids, vectors, metadata)
    ///     ```
    #[pyo3(signature = (collection, ids, vectors, metadata=None))]
    fn insert(
        &self,
        py: Python<'_>,
        collection: &str,
        ids: Vec<String>,
        vectors: &Bound<'_, PyAny>,
        metadata: Option<&Bound<'_, PyList>>,
    ) -> PyResult<usize> {
        // Convert vectors from Python to Rust
        let rust_vectors: Vec<Vec<f32>> = if vectors.hasattr("tolist")? {
            // NumPy array - call tolist() to convert
            vectors.call_method0("tolist")?.extract()?
        } else {
            // Already a Python list
            vectors.extract()?
        };

        // Convert metadata
        let rust_metadata: Option<Vec<HashMap<String, serde_json::Value>>> =
            if let Some(meta_list) = metadata {
                let mut result = Vec::with_capacity(meta_list.len());
                for item in meta_list.iter() {
                    let dict = item.cast::<PyDict>()?;
                    let mut map = HashMap::new();
                    for (k, v) in dict.iter() {
                        let key: String = k.extract()?;
                        let value = python_to_json(&v)?;
                        map.insert(key, value);
                    }
                    result.push(map);
                }
                Some(result)
            } else {
                None
            };

        let inner = self.db()?;
        py.detach(move || inner.insert(collection, ids, rust_vectors, rust_metadata))
            .map_err(|e| PyRuntimeError::new_err(format!("Insert failed: {}", e)))
    }

    /// Search for similar vectors
    ///
    /// Args:
    ///     collection: Collection name
    ///     query: Query vector (can be NumPy array or list)
    ///     top_k: Number of results to return (default: 10)
    ///     filter: Optional filter expression
    ///     search_mode: Optional search mode for accuracy vs speed tradeoff
    ///         - "exact": 100% recall, searches all partitions (default)
    ///         - "approximate": Faster search using IVF-style partition pruning (~95% recall)
    ///         - "approximate:N": Approximate with explicit nprobe value N
    ///         - "adaptive": Auto-select based on dataset size
    ///         - "adaptive:N": Adaptive with explicit threshold N
    ///
    /// Returns:
    ///     List of SearchResult objects
    ///
    /// Example:
    ///     ```python
    ///     import numpy as np
    ///     query = np.random.rand(768).astype(np.float32)
    ///
    ///     # Exact search (100% recall, default)
    ///     results = db.search("my_collection", query, top_k=10)
    ///
    ///     # Approximate search (faster, ~95% recall)
    ///     results = db.search("my_collection", query, top_k=10, search_mode="approximate")
    ///
    ///     # Approximate with custom nprobe
    ///     results = db.search("my_collection", query, top_k=10, search_mode="approximate:5")
    ///     ```
    #[pyo3(signature = (collection, query, top_k=10, filter=None, search_mode=None))]
    fn search(
        &self,
        py: Python<'_>,
        collection: &str,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        filter: Option<&str>,
        search_mode: Option<&str>,
    ) -> PyResult<Vec<PySearchResult>> {
        // Convert query vector
        let query_vec: Vec<f32> = if query.hasattr("tolist")? {
            query.call_method0("tolist")?.extract()?
        } else {
            query.extract()?
        };

        let inner = self.db()?;
        py.detach(move || inner.search_with_mode(collection, query_vec, top_k, filter, search_mode))
            .map(|results| {
                results
                    .into_iter()
                    .map(|r| PySearchResult {
                        id: r.id,
                        score: r.score,
                        metadata_map: r.metadata,
                    })
                    .collect()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {}", e)))
    }

    /// Insert vectors from a NumPy array with ZERO-COPY transfer
    ///
    /// This is the fastest way to insert vectors from Python. The numpy array
    /// buffer is accessed directly without copying to a Python list first.
    ///
    /// Args:
    ///     collection: Collection name
    ///     ids: List of vector IDs
    ///     vectors: 2D numpy array of shape (n_vectors, dimension) with dtype=float32
    ///     metadata: Optional list of metadata dictionaries
    ///
    /// Returns:
    ///     Number of vectors inserted
    ///
    /// Performance:
    ///     ~3-5x faster than insert() for large arrays (>10K vectors)
    ///
    /// Example:
    ///     ```python
    ///     import numpy as np
    ///     vectors = np.random.rand(100000, 384).astype(np.float32)
    ///     ids = [f"vec_{i}" for i in range(100000)]
    ///     count = db.insert_numpy("my_collection", ids, vectors)  # Zero-copy!
    ///     ```
    #[pyo3(signature = (collection, ids, vectors, metadata=None))]
    fn insert_numpy(
        &self,
        py: Python<'_>,
        collection: &str,
        ids: Vec<String>,
        vectors: PyReadonlyArray2<f32>,
        metadata: Option<&Bound<'_, PyList>>,
    ) -> PyResult<usize> {
        // Zero-copy access to numpy buffer
        let array = vectors.as_array();
        let shape = array.shape();
        let n_vectors = shape[0];

        // Validate dimensions
        if ids.len() != n_vectors {
            return Err(PyValueError::new_err(format!(
                "Number of IDs ({}) doesn't match number of vectors ({})",
                ids.len(),
                n_vectors
            )));
        }

        // Convert to Vec<Vec<f32>> - we still need this format for the internal API
        // but at least we avoided the Python .tolist() overhead
        let rust_vectors: Vec<Vec<f32>> = if let Some(slice) = array.as_slice() {
            let dimension = shape[1];
            slice
                .chunks(dimension)
                .map(|chunk| chunk.to_vec())
                .collect()
        } else {
            array.rows().into_iter().map(|row| row.to_vec()).collect()
        };

        // Convert metadata (same as before)
        let rust_metadata: Option<Vec<HashMap<String, serde_json::Value>>> =
            if let Some(meta_list) = metadata {
                let mut result = Vec::with_capacity(meta_list.len());
                for item in meta_list.iter() {
                    let dict = item.cast::<PyDict>()?;
                    let mut map = HashMap::new();
                    for (k, v) in dict.iter() {
                        let key: String = k.extract()?;
                        let value = python_to_json(&v)?;
                        map.insert(key, value);
                    }
                    result.push(map);
                }
                Some(result)
            } else {
                None
            };

        let inner = self.db()?;
        py.detach(move || inner.insert(collection, ids, rust_vectors, rust_metadata))
            .map_err(|e| PyRuntimeError::new_err(format!("Insert failed: {}", e)))
    }

    /// Insert canonical ProximaRecord payloads without lowering through the
    /// legacy ids/vectors/metadata Python transport.
    #[pyo3(signature = (collection, records))]
    fn _insert_proxima_records_native(
        &self,
        py: Python<'_>,
        collection: &str,
        records: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let record_list = records.cast::<PyList>()?;
        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let rust_records = record_list
            .iter()
            .map(|record| python_to_proxima_record(&record, now_ns))
            .collect::<PyResult<Vec<_>>>()?;

        let inner = self.db()?;
        py.detach(move || inner.insert_proxima_records(collection, rust_records))
            .map_err(|e| PyRuntimeError::new_err(format!("Record insert failed: {}", e)))
    }

    /// Insert canonical ProximaRecord payloads from a columnar Python batch:
    /// ids + dense embedding matrix + props. This preserves the canonical
    /// record/storage boundary while avoiding per-record vector extraction.
    #[pyo3(signature = (collection, ids, vectors, props=None))]
    fn _insert_proxima_record_batch_native(
        &self,
        py: Python<'_>,
        collection: &str,
        ids: Vec<String>,
        vectors: PyReadonlyArray2<f32>,
        props: Option<&Bound<'_, PyList>>,
    ) -> PyResult<usize> {
        let array = vectors.as_array();
        let shape = array.shape();
        let n_records = shape[0];
        if ids.len() != n_records {
            return Err(PyValueError::new_err(format!(
                "Number of IDs ({}) doesn't match number of records ({})",
                ids.len(),
                n_records
            )));
        }
        if let Some(prop_list) = props
            && prop_list.len() != n_records
        {
            return Err(PyValueError::new_err(format!(
                "Number of props ({}) doesn't match number of records ({})",
                prop_list.len(),
                n_records
            )));
        }

        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let dimension = shape[1];
        let mut records = Vec::with_capacity(n_records);

        if let Some(slice) = array.as_slice() {
            for (index, values) in slice.chunks(dimension).enumerate() {
                let props = python_props_list_item(props, index)?;
                records.push(proxima_record_from_batch_parts(
                    ids[index].clone(),
                    values.to_vec(),
                    props,
                    now_ns,
                ));
            }
        } else {
            for (index, row) in array.rows().into_iter().enumerate() {
                let props = python_props_list_item(props, index)?;
                records.push(proxima_record_from_batch_parts(
                    ids[index].clone(),
                    row.to_vec(),
                    props,
                    now_ns,
                ));
            }
        }

        let inner = self.db()?;
        py.detach(move || inner.insert_proxima_records(collection, records))
            .map_err(|e| PyRuntimeError::new_err(format!("Record insert failed: {}", e)))
    }

    /// Profiled variant of `_insert_proxima_record_batch_native` used by the
    /// benchmark harness to locate native-side hot spots.
    #[pyo3(signature = (collection, ids, vectors, props=None))]
    fn _insert_proxima_record_batch_native_profiled(
        &self,
        py: Python<'_>,
        collection: &str,
        ids: Vec<String>,
        vectors: PyReadonlyArray2<f32>,
        props: Option<&Bound<'_, PyList>>,
    ) -> PyResult<(usize, Py<PyAny>)> {
        let total_started = std::time::Instant::now();
        let array = vectors.as_array();
        let shape = array.shape();
        let n_records = shape[0];
        if ids.len() != n_records {
            return Err(PyValueError::new_err(format!(
                "Number of IDs ({}) doesn't match number of records ({})",
                ids.len(),
                n_records
            )));
        }
        if let Some(prop_list) = props
            && prop_list.len() != n_records
        {
            return Err(PyValueError::new_err(format!(
                "Number of props ({}) doesn't match number of records ({})",
                prop_list.len(),
                n_records
            )));
        }

        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let dimension = shape[1];
        let mut records = Vec::with_capacity(n_records);
        let mut props_conversion = std::time::Duration::ZERO;
        let mut vector_copy = std::time::Duration::ZERO;
        let mut record_assembly = std::time::Duration::ZERO;

        if let Some(slice) = array.as_slice() {
            for (index, values) in slice.chunks(dimension).enumerate() {
                let props_started = std::time::Instant::now();
                let props = python_props_list_item(props, index)?;
                props_conversion += props_started.elapsed();

                let vector_started = std::time::Instant::now();
                let values = values.to_vec();
                vector_copy += vector_started.elapsed();

                let record_started = std::time::Instant::now();
                records.push(proxima_record_from_batch_parts(
                    ids[index].clone(),
                    values,
                    props,
                    now_ns,
                ));
                record_assembly += record_started.elapsed();
            }
        } else {
            for (index, row) in array.rows().into_iter().enumerate() {
                let props_started = std::time::Instant::now();
                let props = python_props_list_item(props, index)?;
                props_conversion += props_started.elapsed();

                let vector_started = std::time::Instant::now();
                let values = row.to_vec();
                vector_copy += vector_started.elapsed();

                let record_started = std::time::Instant::now();
                records.push(proxima_record_from_batch_parts(
                    ids[index].clone(),
                    values,
                    props,
                    now_ns,
                ));
                record_assembly += record_started.elapsed();
            }
        }

        let build_elapsed = total_started.elapsed();
        let inner = self.db()?;
        let insert_started = std::time::Instant::now();
        let result = py
            .detach(move || inner.insert_proxima_records(collection, records))
            .map_err(|e| PyRuntimeError::new_err(format!("Record insert failed: {}", e)))?;
        let insert_elapsed = insert_started.elapsed();
        let total_elapsed = total_started.elapsed();

        let profile = PyDict::new(py);
        profile.set_item(
            "native_props_conversion_seconds",
            props_conversion.as_secs_f64(),
        )?;
        profile.set_item("native_vector_copy_seconds", vector_copy.as_secs_f64())?;
        profile.set_item(
            "native_record_assembly_seconds",
            record_assembly.as_secs_f64(),
        )?;
        profile.set_item("native_build_records_seconds", build_elapsed.as_secs_f64())?;
        profile.set_item(
            "native_storage_insert_seconds",
            insert_elapsed.as_secs_f64(),
        )?;
        profile.set_item("native_total_seconds", total_elapsed.as_secs_f64())?;
        Ok((result, profile.into()))
    }

    /// Search with a NumPy query vector (zero-copy)
    ///
    /// Args:
    ///     collection: Collection name
    ///     query: 1D numpy array of shape (dimension,) with dtype=float32
    ///     top_k: Number of results to return (default: 10)
    ///     filter: Optional filter expression
    ///     search_mode: Optional search mode for accuracy vs speed tradeoff
    ///         - "exact": 100% recall, searches all partitions (default)
    ///         - "approximate": Faster search using IVF-style partition pruning (~95% recall)
    ///         - "approximate:N": Approximate with explicit nprobe value N
    ///         - "adaptive": Auto-select based on dataset size
    ///
    /// Returns:
    ///     List of SearchResult objects
    #[pyo3(signature = (collection, query, top_k=10, filter=None, search_mode=None))]
    fn search_numpy(
        &self,
        py: Python<'_>,
        collection: &str,
        query: PyReadonlyArray1<f32>,
        top_k: usize,
        filter: Option<&str>,
        search_mode: Option<&str>,
    ) -> PyResult<Vec<PySearchResult>> {
        // Zero-copy access to query vector
        let query_vec: Vec<f32> = query
            .as_slice()
            .map(|slice| slice.to_vec())
            .unwrap_or_else(|_| query.as_array().to_vec());

        let inner = self.db()?;
        py.detach(move || inner.search_with_mode(collection, query_vec, top_k, filter, search_mode))
            .map(|results| {
                results
                    .into_iter()
                    .map(|r| PySearchResult {
                        id: r.id,
                        score: r.score,
                        metadata_map: r.metadata,
                    })
                    .collect()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {}", e)))
    }

    /// Profiled variant of `search_numpy` used by the benchmark harness to
    /// locate Python/native search boundary hot spots.
    #[pyo3(signature = (collection, query, top_k=10, filter=None, search_mode=None))]
    fn _search_numpy_profiled(
        &self,
        py: Python<'_>,
        collection: &str,
        query: PyReadonlyArray1<f32>,
        top_k: usize,
        filter: Option<&str>,
        search_mode: Option<&str>,
    ) -> PyResult<(Vec<PySearchResult>, Py<PyAny>)> {
        let total_started = std::time::Instant::now();
        let copy_started = std::time::Instant::now();
        let query_vec: Vec<f32> = query
            .as_slice()
            .map(|slice| slice.to_vec())
            .unwrap_or_else(|_| query.as_array().to_vec());
        let query_vector_copy = copy_started.elapsed();

        let inner = self.db()?;
        let search_started = std::time::Instant::now();
        let results = py
            .detach(move || {
                inner.search_with_mode(collection, query_vec, top_k, filter, search_mode)
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {}", e)))?;
        let native_search = search_started.elapsed();

        let conversion_started = std::time::Instant::now();
        let py_results = results
            .into_iter()
            .map(|r| PySearchResult {
                id: r.id,
                score: r.score,
                metadata_map: r.metadata,
            })
            .collect();
        let result_conversion = conversion_started.elapsed();
        let total_elapsed = total_started.elapsed();

        let profile = PyDict::new(py);
        profile.set_item(
            "native_query_vector_copy_seconds",
            query_vector_copy.as_secs_f64(),
        )?;
        profile.set_item("native_search_seconds", native_search.as_secs_f64())?;
        profile.set_item(
            "native_search_result_conversion_seconds",
            result_conversion.as_secs_f64(),
        )?;
        profile.set_item("native_search_total_seconds", total_elapsed.as_secs_f64())?;
        Ok((py_results, profile.into()))
    }

    /// Batch search with multiple query vectors (zero-copy)
    ///
    /// Args:
    ///     collection: Collection name
    ///     queries: 2D numpy array of shape (n_queries, dimension) with dtype=float32
    ///     top_k: Number of results per query (default: 10)
    ///     search_mode: Optional search mode for accuracy vs speed tradeoff
    ///         - "exact": 100% recall, searches all partitions (default)
    ///         - "approximate": Faster search using IVF-style partition pruning (~95% recall)
    ///         - "approximate:N": Approximate with explicit nprobe value N
    ///
    /// Returns:
    ///     List of lists of SearchResult objects
    #[pyo3(signature = (collection, queries, top_k=10, search_mode=None))]
    fn batch_search_numpy(
        &self,
        collection: &str,
        queries: PyReadonlyArray2<f32>,
        top_k: usize,
        search_mode: Option<&str>,
    ) -> PyResult<Vec<Vec<PySearchResult>>> {
        let array = queries.as_array();
        let mut all_results = Vec::with_capacity(array.nrows());

        for row in array.rows() {
            let query_vec: Vec<f32> = row.to_vec();
            let results = self
                .db()?
                .search_with_mode(collection, query_vec, top_k, None, search_mode)
                .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {}", e)))?;

            all_results.push(
                results
                    .into_iter()
                    .map(|r| PySearchResult {
                        id: r.id,
                        score: r.score,
                        metadata_map: r.metadata,
                    })
                    .collect(),
            );
        }

        Ok(all_results)
    }

    /// Create a streaming search iterator for memory-efficient large result sets
    ///
    /// This method returns an iterator that yields batches of search results,
    /// allowing for memory-efficient processing of large result sets without
    /// loading all results into memory at once.
    ///
    /// Args:
    ///     collection: Collection name
    ///     query: Query vector (list or numpy array)
    ///     top_k: Total number of results to return (default: 1000)
    ///     batch_size: Number of results per batch (default: 100)
    ///     search_mode: Optional search mode for accuracy vs speed tradeoff
    ///         - "exact": 100% recall, searches all partitions (default)
    ///         - "approximate": Faster search using IVF-style partition pruning
    ///         - "adaptive": Auto-select based on dataset size
    ///
    /// Returns:
    ///     SearchStreamIterator that yields batches of StreamingSearchResult
    ///
    /// Example:
    ///     ```python
    ///     # Process 10,000 results in batches of 100
    ///     for batch in db.search_streaming("my_collection", query, top_k=10000, batch_size=100):
    ///         for result in batch:
    ///             print(f"Found: {result.id} (score: {result.score})")
    ///
    ///     # Use with search mode
    ///     for batch in db.search_streaming("my_collection", query, top_k=10000,
    ///                                       batch_size=100, search_mode="approximate"):
    ///         process_batch(batch)
    ///     ```
    #[pyo3(signature = (collection, query, top_k=1000, batch_size=100, search_mode=None))]
    fn search_streaming(
        &self,
        collection: &str,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        batch_size: usize,
        search_mode: Option<&str>,
    ) -> PyResult<PySearchStreamIterator> {
        // Convert query vector
        let query_vec: Vec<f32> = if query.hasattr("tolist")? {
            query.call_method0("tolist")?.extract()?
        } else {
            query.extract()?
        };

        // Build streaming config
        let mut config = super::StreamingSearchConfig::default().with_batch_size(batch_size);

        if let Some(mode) = search_mode {
            config = config.with_search_mode(mode);
        }

        // Create the streaming iterator
        let iterator = self
            .db()?
            .search_streaming_with_config(collection, query_vec, top_k, config)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create streaming search: {}", e))
            })?;

        Ok(PySearchStreamIterator {
            inner: Some(iterator),
        })
    }

    /// Create a streaming search iterator with zero-copy numpy query
    ///
    /// Same as `search_streaming` but with zero-copy access to numpy arrays.
    ///
    /// Args:
    ///     collection: Collection name
    ///     query: Query vector as numpy array with dtype=float32
    ///     top_k: Total number of results to return (default: 1000)
    ///     batch_size: Number of results per batch (default: 100)
    ///     search_mode: Optional search mode for accuracy vs speed tradeoff
    ///
    /// Returns:
    ///     SearchStreamIterator that yields batches of StreamingSearchResult
    ///
    /// Example:
    ///     ```python
    ///     import numpy as np
    ///     query = np.array([0.1, 0.2, ...], dtype=np.float32)
    ///     for batch in db.search_streaming_numpy("my_collection", query, top_k=10000):
    ///         for result in batch:
    ///             print(f"Found: {result.id}")
    ///     ```
    #[pyo3(signature = (collection, query, top_k=1000, batch_size=100, search_mode=None))]
    fn search_streaming_numpy(
        &self,
        collection: &str,
        query: PyReadonlyArray1<f32>,
        top_k: usize,
        batch_size: usize,
        search_mode: Option<&str>,
    ) -> PyResult<PySearchStreamIterator> {
        // Zero-copy access to query vector
        let query_vec: Vec<f32> = query.as_array().to_vec();

        // Build streaming config
        let mut config = super::StreamingSearchConfig::default().with_batch_size(batch_size);

        if let Some(mode) = search_mode {
            config = config.with_search_mode(mode);
        }

        // Create the streaming iterator
        let iterator = self
            .db()?
            .search_streaming_with_config(collection, query_vec, top_k, config)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create streaming search: {}", e))
            })?;

        Ok(PySearchStreamIterator {
            inner: Some(iterator),
        })
    }

    // ========================================================================
    // Vector Lookup Operations - GET by ID
    // ========================================================================

    /// Get a vector by its ID
    ///
    /// Args:
    ///     collection: Collection name
    ///     vector_id: The unique ID of the vector to retrieve
    ///
    /// Returns:
    ///     Dictionary with {id, vector, metadata} or None if not found
    ///
    /// Example:
    ///     ```python
    ///     result = db.get_vector("my_collection", "vec_123")
    ///     if result:
    ///         print(f"Vector ID: {result['id']}")
    ///         print(f"Vector: {result['vector'][:5]}...")  # First 5 dims
    ///         print(f"Metadata: {result['metadata']}")
    ///     ```
    fn get_vector(
        &self,
        py: Python<'_>,
        collection: &str,
        vector_id: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        match self.db()?.get_vector(collection, vector_id) {
            Ok(Some(mut record)) => {
                let dict = PyDict::new(py);
                dict.set_item("id", &record.oid)?;
                let vector = record
                    .embeddings
                    .first()
                    .map(|embedding| embedding.values.to_fp32_owned())
                    .unwrap_or_default();
                dict.set_item("vector", vector)?;

                let props_dict = PyDict::new(py);
                // props are MOVED out (the record is owned and props are
                // dead after this loop) — each leaf flows into the owned
                // canonical converter without the per-field deep clone.
                for (key, node) in std::mem::take(&mut record.props) {
                    let value = proxima_tree_node_to_json(node);
                    let py_value = json_to_python(py, &value)?;
                    props_dict.set_item(key, py_value)?;
                }
                dict.set_item("props", &props_dict)?;
                dict.set_item("metadata", &props_dict)?;

                dict.set_item("timestamp", record.created_at_ns / 1_000_000)?;
                dict.set_item("updated_at", record.updated_at_ns / 1_000_000)?;

                Ok(Some(dict.into()))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get vector: {}",
                e
            ))),
        }
    }

    /// Check if a vector exists in the collection
    ///
    /// This is a fast existence check using bloom filters when available.
    ///
    /// Args:
    ///     collection: Collection name
    ///     vector_id: The unique ID of the vector to check
    ///
    /// Returns:
    ///     True if the vector exists, False otherwise
    ///
    /// Example:
    ///     ```python
    ///     if db.vector_exists("my_collection", "vec_123"):
    ///         print("Vector exists!")
    ///     else:
    ///         print("Vector not found")
    ///     ```
    fn vector_exists(&self, collection: &str, vector_id: &str) -> PyResult<bool> {
        self.db()?
            .vector_exists(collection, vector_id)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to check vector existence: {}", e))
            })
    }

    /// Delete a single vector by ID (tombstone-based)
    ///
    /// This uses tombstone markers to logically delete the vector. The tombstone
    /// will be written to the WAL and will shadow the original vector in searches.
    /// Physical deletion happens during compaction.
    ///
    /// Args:
    ///     collection: Name of the collection
    ///     vector_id: ID of the vector to delete
    ///
    /// Returns:
    ///     True if tombstone was written (doesn't guarantee vector existed)
    ///
    /// Example:
    ///     ```python
    ///     deleted = db.delete_vector("embeddings", "vec_123")
    ///     if deleted:
    ///         print("Vector marked for deletion")
    ///     ```
    fn delete_vector(&self, collection: &str, vector_id: &str) -> PyResult<bool> {
        self.db()?
            .delete_vector(collection, vector_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete vector: {}", e)))
    }

    /// Delete multiple vectors by IDs (batch tombstone operation)
    ///
    /// More efficient than calling `delete_vector` multiple times.
    /// All tombstones are written in a single batch operation.
    ///
    /// Args:
    ///     collection: Name of the collection
    ///     vector_ids: List of vector IDs to delete
    ///
    /// Returns:
    ///     Number of tombstones written (equals input count)
    ///
    /// Example:
    ///     ```python
    ///     count = db.delete_vectors("embeddings", ["vec_1", "vec_2", "vec_3"])
    ///     print(f"Marked {count} vectors for deletion")
    ///     ```
    fn delete_vectors(&self, collection: &str, vector_ids: Vec<String>) -> PyResult<usize> {
        self.db()?
            .delete_vectors(collection, vector_ids)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete vectors: {}", e)))
    }

    /// Upsert vectors (insert or update)
    ///
    /// This is an atomic operation that inserts new vectors and updates existing ones.
    /// Existing vectors are deleted (via tombstone) before the new vectors are inserted.
    ///
    /// Args:
    ///     collection: Collection name
    ///     ids: List of vector IDs
    ///     vectors: List of vectors (each is a list of floats), or NumPy array
    ///     metadata: Optional list of metadata dicts for each vector
    ///
    /// Returns:
    ///     Tuple of (inserted_count, updated_count)
    ///
    /// Example:
    ///     ```python
    ///     # Insert new and update existing vectors in one operation
    ///     inserted, updated = db.upsert(
    ///         "embeddings",
    ///         ["vec_1", "vec_2", "vec_3"],
    ///         [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]],
    ///         [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
    ///     )
    ///     print(f"Inserted {inserted}, updated {updated}")
    ///     ```
    #[pyo3(signature = (collection, ids, vectors, metadata=None))]
    fn upsert(
        &self,
        py: Python<'_>,
        collection: &str,
        ids: Vec<String>,
        vectors: &Bound<'_, PyAny>,
        metadata: Option<&Bound<'_, PyList>>,
    ) -> PyResult<(usize, usize)> {
        // Convert vectors from Python to Rust
        let rust_vectors: Vec<Vec<f32>> = if vectors.hasattr("tolist")? {
            // NumPy array - call tolist() to convert
            vectors.call_method0("tolist")?.extract()?
        } else {
            // Already a Python list
            vectors.extract()?
        };

        // Convert metadata (same pattern as insert)
        let rust_metadata: Option<Vec<HashMap<String, serde_json::Value>>> =
            if let Some(meta_list) = metadata {
                let mut result = Vec::with_capacity(meta_list.len());
                for item in meta_list.iter() {
                    let dict = item.cast::<PyDict>()?;
                    let mut map = HashMap::new();
                    for (k, v) in dict.iter() {
                        let key: String = k.extract()?;
                        let value = python_to_json(&v)?;
                        map.insert(key, value);
                    }
                    result.push(map);
                }
                Some(result)
            } else {
                None
            };

        let inner = self.db()?;
        py.detach(move || inner.upsert(collection, ids, rust_vectors, rust_metadata))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to upsert vectors: {}", e)))
    }

    /// Upsert vectors using NumPy arrays (zero-copy)
    ///
    /// This is the high-performance variant of upsert() that accepts NumPy arrays directly.
    ///
    /// Args:
    ///     collection: Collection name
    ///     ids: List of vector IDs
    ///     vectors: 2D NumPy array of shape (n_vectors, dimension)
    ///     metadata: Optional list of metadata dicts for each vector
    ///
    /// Returns:
    ///     Tuple of (inserted_count, updated_count)
    ///
    /// Example:
    ///     ```python
    ///     import numpy as np
    ///     vectors = np.random.randn(1000, 768).astype(np.float32)
    ///     ids = [f"vec_{i}" for i in range(1000)]
    ///     inserted, updated = db.upsert_numpy("embeddings", ids, vectors)
    ///     ```
    #[pyo3(signature = (collection, ids, vectors, metadata=None))]
    fn upsert_numpy(
        &self,
        py: Python<'_>,
        collection: &str,
        ids: Vec<String>,
        vectors: PyReadonlyArray2<f32>,
        metadata: Option<&Bound<'_, PyList>>,
    ) -> PyResult<(usize, usize)> {
        // Zero-copy access to numpy buffer
        let array = vectors.as_array();
        let rust_vectors: Vec<Vec<f32>> = if let Some(slice) = array.as_slice() {
            let dimension = array.shape()[1];
            slice
                .chunks(dimension)
                .map(|chunk| chunk.to_vec())
                .collect()
        } else {
            array.rows().into_iter().map(|row| row.to_vec()).collect()
        };

        // Convert metadata (same pattern as insert_numpy)
        let rust_metadata: Option<Vec<HashMap<String, serde_json::Value>>> =
            if let Some(meta_list) = metadata {
                let mut result = Vec::with_capacity(meta_list.len());
                for item in meta_list.iter() {
                    let dict = item.cast::<PyDict>()?;
                    let mut map = HashMap::new();
                    for (k, v) in dict.iter() {
                        let key: String = k.extract()?;
                        let value = python_to_json(&v)?;
                        map.insert(key, value);
                    }
                    result.push(map);
                }
                Some(result)
            } else {
                None
            };

        let inner = self.db()?;
        py.detach(move || inner.upsert(collection, ids, rust_vectors, rust_metadata))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to upsert vectors: {}", e)))
    }

    /// Flush all pending writes to disk
    ///
    /// This ensures all data is persisted to disk. Called automatically
    /// when the database is closed via close() or context manager exit.
    fn flush(&self) -> PyResult<()> {
        if let Some(db) = self.db_opt() {
            db.flush()
                .map_err(|e| PyRuntimeError::new_err(format!("Flush failed: {}", e)))
        } else {
            Ok(())
        }
    }

    /// Close the database, flushing all pending writes to disk
    ///
    /// This ensures all data is persisted to SST files before the database
    /// is closed. This is important for:
    /// - Creating SST files with centroid indexes (enables approximate search)
    /// - Avoiding WAL replay overhead on subsequent opens
    /// - Ensuring data durability
    ///
    /// Example:
    ///     db = ProximaDB("/tmp/mydb")
    ///     db.insert("collection", vectors, ids)
    ///     db.close()  # Flushes and closes gracefully
    fn close(&self) -> PyResult<()> {
        // Take the Arc out of the Option, dropping our reference.
        // If this was the last Arc clone (all allow_threads closures have returned),
        // EmbeddedProximaDB is dropped here → Tokio runtime shuts down → file handles closed.
        let arc = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Database mutex poisoned"))?
            .take();
        if let Some(arc) = arc {
            let flush_result = arc
                .flush()
                .map_err(|e| PyRuntimeError::new_err(format!("Flush failed: {}", e)));
            arc.close();
            drop(arc);
            flush_result
        } else {
            Ok(()) // already closed — idempotent
        }
    }

    fn __del__(&self) {
        // Best-effort cleanup when Python GC collects this object without an explicit close().
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(arc) = guard.take() {
                let _ = arc.flush();
                arc.close();
                drop(arc);
            }
        }
    }

    // ========================================================================
    // Multi-Process Coordination API
    // ========================================================================

    /// Check if this database instance can perform write operations
    ///
    /// Returns:
    ///     True if writes are allowed based on access mode and leader status
    ///
    /// Example:
    ///     ```python
    ///     if db.can_write():
    ///         db.insert("vectors", ids, vectors)
    ///     else:
    ///         print("Read-only mode")
    ///     ```
    fn can_write(&self) -> bool {
        self.db_opt().map(|db| db.can_write()).unwrap_or(false)
    }

    /// Get the current access mode
    ///
    /// Returns:
    ///     Access mode string: "exclusive", "shared_read", or "leader_follower"
    fn access_mode(&self) -> String {
        self.db_opt()
            .map(|db| format!("{}", db.access_mode()))
            .unwrap_or_default()
    }

    /// Check if this node is the leader (only relevant in leader/follower mode)
    ///
    /// Returns:
    ///     True if this node is the leader or if in exclusive mode
    ///
    /// Example:
    ///     ```python
    ///     db = ProximaDB("/data", mode="leader")
    ///     if db.is_leader():
    ///         print("This process is the leader")
    ///     ```
    fn is_leader(&self) -> bool {
        self.db_opt().map(|db| db.is_leader()).unwrap_or(false)
    }

    /// Get the current leader ID (only relevant in leader/follower mode)
    ///
    /// Returns:
    ///     Leader node ID or None if not in leader/follower mode
    fn leader_id(&self) -> Option<String> {
        self.db_opt().and_then(|db| db.leader_id())
    }

    /// Get storage statistics
    ///
    /// Returns:
    ///     StorageStats with information about disk usage, vector counts, etc.
    fn stats(&self) -> PyResult<PyStorageStats> {
        self.db()?
            .stats()
            .map(|s| PyStorageStats {
                total_vectors: s.total_vectors,
                total_collections: s.total_collections,
                disk_usage_bytes: s.disk_usage_bytes,
                cache_hit_rate: s.cache_hit_rate,
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get stats: {}", e)))
    }

    // ========================================================================
    // Checkpoint and Delta Persistence API
    // ========================================================================

    /// Create a named checkpoint of the current database state
    ///
    /// This captures the current state of all collections and allows restoration
    /// to this point later. Checkpoints are persisted to disk and survive restarts.
    ///
    /// Args:
    ///     name: Name for the checkpoint (must be unique)
    ///
    /// Returns:
    ///     CheckpointInfo with details about the created checkpoint
    ///
    /// Example:
    ///     ```python
    ///     info = db.checkpoint("before_experiment")
    ///     print(f"Checkpoint created at LSN {info.checkpoint_lsn}")
    ///
    ///     # Make changes...
    ///     db.insert("vectors", ids, vectors)
    ///
    ///     # Restore to checkpoint
    ///     db.restore_checkpoint("before_experiment")
    ///     ```
    fn checkpoint(&self, name: &str) -> PyResult<PyCheckpointInfo> {
        self.db()?
            .checkpoint(name)
            .map(PyCheckpointInfo::from)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create checkpoint: {}", e)))
    }

    /// Restore the database to a named checkpoint
    ///
    /// This restores all collections to the state they were in when the checkpoint
    /// was created. Any changes made after the checkpoint are discarded.
    ///
    /// WARNING: This is a destructive operation. All data added after the checkpoint
    /// will be lost.
    ///
    /// Args:
    ///     name: Name of the checkpoint to restore
    ///
    /// Example:
    ///     ```python
    ///     db.checkpoint("backup")
    ///     db.insert("vectors", ids, vectors)  # Add data
    ///     db.restore_checkpoint("backup")  # Restore - new data is discarded
    ///     ```
    fn restore_checkpoint(&self, name: &str) -> PyResult<()> {
        self.db()?
            .restore_checkpoint(name)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to restore checkpoint: {}", e)))
    }

    /// Save incremental changes since the last checkpoint to a delta file
    ///
    /// Delta files contain only the changes made since the last checkpoint,
    /// making them much smaller and faster to create than full checkpoints.
    ///
    /// Args:
    ///     path: Path where the delta file will be saved
    ///
    /// Returns:
    ///     DeltaInfo with details about the saved delta
    ///
    /// Example:
    ///     ```python
    ///     db.checkpoint("baseline")
    ///     db.insert("vectors", ids1, vectors1)
    ///     db.insert("vectors", ids2, vectors2)
    ///
    ///     # Save only the changes since checkpoint
    ///     delta = db.save_delta("/backup/delta_001.delta")
    ///     print(f"Delta saved: {delta.entry_count} entries, {delta.size_bytes} bytes")
    ///     ```
    fn save_delta(&self, path: &str) -> PyResult<PyDeltaInfo> {
        self.db()?
            .save_delta(path)
            .map(PyDeltaInfo::from)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to save delta: {}", e)))
    }

    /// Load changes from a delta file
    ///
    /// Applies the changes from a delta file to the current database state.
    /// This is typically used to replay changes after restoring from a checkpoint.
    ///
    /// Args:
    ///     path: Path to the delta file to load
    ///
    /// Example:
    ///     ```python
    ///     # Restore to checkpoint, then apply delta
    ///     db.restore_checkpoint("baseline")
    ///     db.load_delta("/backup/delta_001.delta")
    ///     ```
    fn load_delta(&self, path: &str) -> PyResult<()> {
        self.db()?
            .load_delta(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load delta: {}", e)))
    }

    /// List all available checkpoints
    ///
    /// Returns a list of all checkpoints that have been created, sorted by
    /// creation timestamp (oldest first).
    ///
    /// Returns:
    ///     List of CheckpointInfo objects with details about each checkpoint
    ///
    /// Example:
    ///     ```python
    ///     checkpoints = db.list_checkpoints()
    ///     for cp in checkpoints:
    ///         print(f"{cp.name}: {len(cp.collections)} collections at LSN {cp.checkpoint_lsn}")
    ///     ```
    fn list_checkpoints(&self) -> PyResult<Vec<PyCheckpointInfo>> {
        self.db()?
            .list_checkpoints()
            .map(|cps| cps.into_iter().map(PyCheckpointInfo::from).collect())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to list checkpoints: {}", e)))
    }

    /// Context manager entry
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager exit - flushes and closes the database
    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false) // Don't suppress exceptions
    }

    fn __repr__(&self) -> PyResult<String> {
        let stats = self.stats()?;
        Ok(format!(
            "ProximaDB(collections={}, vectors={}, disk={})",
            stats.total_collections,
            stats.total_vectors,
            stats.disk_usage_human()
        ))
    }

    // ========================================================================
    // Generic Graph Operations - Tool Agnostic API
    // ========================================================================
    //
    // These methods provide a generic graph API. Domain-specific behavior
    // (like code intelligence) should be built as an adapter in the consuming
    // application (e.g., Victor).

    /// Create a graph collection
    ///
    /// A graph must be created before nodes and edges can be added.
    ///
    /// Args:
    ///     graph_id: Unique identifier for the graph
    ///     engine: Graph engine type ("orion") (optional)
    ///
    /// Example:
    ///     ```python
    ///     db.create_graph("my_knowledge_graph", engine="orion")
    ///     db.create_nodes("my_knowledge_graph", nodes)
    ///     ```
    #[pyo3(signature = (graph_id, engine=None))]
    fn create_graph(&self, graph_id: &str, engine: Option<&str>) -> PyResult<()> {
        self.db()?
            .create_graph(graph_id, engine)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create graph: {}", e)))
    }

    /// Create nodes in the graph
    ///
    /// Inserts nodes with their properties. Use labels and properties
    /// for domain-specific categorization and attributes.
    ///
    /// Args:
    ///     graph_id: Graph identifier
    ///     nodes: List of GraphNode objects
    ///
    /// Returns:
    ///     Number of nodes created
    ///
    /// Example:
    ///     ```python
    ///     # For code intelligence:
    ///     nodes = [
    ///         GraphNode("fn_main", labels=["function"], properties={"name": "main", "file": "main.py"}),
    ///         GraphNode("class_A", labels=["class"], properties={"name": "A", "file": "models.py"}),
    ///     ]
    ///     count = db.create_nodes("my_graph", nodes)
    ///
    ///     # For social network:
    ///     nodes = [
    ///         GraphNode("user_1", labels=["Person"], properties={"name": "Alice"}),
    ///         GraphNode("user_2", labels=["Person"], properties={"name": "Bob"}),
    ///     ]
    ///     count = db.create_nodes("social_graph", nodes)
    ///     ```
    fn create_nodes(&self, graph_id: &str, nodes: Vec<PyGraphNode>) -> PyResult<usize> {
        let rust_nodes: Vec<super::GraphNode> = nodes.into_iter().map(|n| n.into()).collect();
        self.db()?
            .create_nodes(graph_id, rust_nodes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create nodes: {}", e)))
    }

    /// Create edges in the graph
    ///
    /// Args:
    ///     graph_id: Graph identifier
    ///     edges: List of GraphEdge objects
    ///
    /// Returns:
    ///     Number of edges created
    ///
    /// Example:
    ///     ```python
    ///     edges = [
    ///         GraphEdge("fn_main", "fn_helper", "CALLS"),
    ///         GraphEdge("class_A", "class_B", "INHERITS"),
    ///     ]
    ///     count = db.create_edges("my_graph", edges)
    ///     ```
    fn create_edges(&self, graph_id: &str, edges: Vec<PyGraphEdge>) -> PyResult<usize> {
        let rust_edges: Vec<super::GraphEdge> = edges.into_iter().map(|e| e.into()).collect();
        self.db()?
            .create_edges(graph_id, rust_edges)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create edges: {}", e)))
    }

    #[pyo3(signature = (graph_id, node_id, labels=None, properties=None))]
    fn create_node(
        &self,
        graph_id: &str,
        node_id: &str,
        labels: Option<Vec<String>>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyGraphNode> {
        let property_map = py_dict_to_properties(properties)?;

        self.db()?
            .create_node(graph_id, node_id, labels.unwrap_or_default(), property_map)
            .map(PyGraphNode::from)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create node: {}", e)))
    }

    #[pyo3(signature = (graph_id, from_node_id, to_node_id, edge_type, id=None, weight=None, properties=None))]
    fn create_edge(
        &self,
        graph_id: &str,
        from_node_id: &str,
        to_node_id: &str,
        edge_type: &str,
        id: Option<&str>,
        weight: Option<f64>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyGraphEdge> {
        let property_map = py_dict_to_properties(properties)?;

        self.db()?
            .create_edge(
                graph_id,
                id,
                from_node_id,
                to_node_id,
                edge_type,
                weight,
                property_map,
            )
            .map(PyGraphEdge::from)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create edge: {}", e)))
    }

    /// Get a node by its ID
    ///
    /// Args:
    ///     graph_id: Graph identifier
    ///     node_id: Node ID to retrieve
    ///
    /// Returns:
    ///     GraphNode if found, None otherwise
    fn get_node(&self, graph_id: &str, node_id: &str) -> PyResult<Option<PyGraphNode>> {
        self.db()?
            .get_node(graph_id, node_id)
            .map(|opt| opt.map(|n| n.into()))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get node: {}", e)))
    }

    /// Query nodes by labels
    ///
    /// Args:
    ///     graph_id: Graph identifier
    ///     labels: List of labels to filter by
    ///
    /// Returns:
    ///     List of matching GraphNode objects
    ///
    /// Example:
    ///     ```python
    ///     # Get all functions
    ///     functions = db.query_nodes_by_labels("code_graph", ["function"])
    ///
    ///     # Get all Person nodes
    ///     people = db.query_nodes_by_labels("social_graph", ["Person"])
    ///     ```
    fn query_nodes_by_labels(
        &self,
        graph_id: &str,
        labels: Vec<String>,
    ) -> PyResult<Vec<PyGraphNode>> {
        self.db()?
            .query_nodes_by_labels(graph_id, labels)
            .map(|nodes| nodes.into_iter().map(|n| n.into()).collect())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to query nodes: {}", e)))
    }

    #[pyo3(signature = (graph_id, labels=None, properties=None, limit=None, offset=None))]
    fn query_nodes(
        &self,
        graph_id: &str,
        labels: Option<Vec<String>>,
        properties: Option<&Bound<'_, PyDict>>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> PyResult<Vec<PyGraphNode>> {
        let property_map = py_dict_to_properties(properties)?;

        self.db()?
            .query_nodes(
                graph_id,
                labels,
                if property_map.is_empty() {
                    None
                } else {
                    Some(property_map)
                },
                limit,
                offset,
            )
            .map(|nodes| nodes.into_iter().map(PyGraphNode::from).collect())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to query nodes: {}", e)))
    }

    /// Get outgoing edges from a node
    ///
    /// Args:
    ///     graph_id: Graph identifier
    ///     node_id: Node ID to get outgoing edges for
    ///     edge_types: Optional list of edge types to filter by
    ///
    /// Returns:
    ///     List of GraphEdge objects representing outgoing edges
    #[pyo3(signature = (graph_id, node_id, edge_types=None))]
    fn get_outgoing_edges(
        &self,
        graph_id: &str,
        node_id: &str,
        edge_types: Option<Vec<String>>,
    ) -> PyResult<Vec<PyGraphEdge>> {
        self.db()?
            .get_outgoing_edges(graph_id, node_id, edge_types)
            .map(|edges| edges.into_iter().map(|e| e.into()).collect())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get outgoing edges: {}", e)))
    }

    /// Get incoming edges to a node
    ///
    /// Args:
    ///     graph_id: Graph identifier
    ///     node_id: Node ID to get incoming edges for
    ///     edge_types: Optional list of edge types to filter by
    ///
    /// Returns:
    ///     List of GraphEdge objects representing incoming edges
    #[pyo3(signature = (graph_id, node_id, edge_types=None))]
    fn get_incoming_edges(
        &self,
        graph_id: &str,
        node_id: &str,
        edge_types: Option<Vec<String>>,
    ) -> PyResult<Vec<PyGraphEdge>> {
        self.db()?
            .get_incoming_edges(graph_id, node_id, edge_types)
            .map(|edges| edges.into_iter().map(|e| e.into()).collect())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get incoming edges: {}", e)))
    }

    /// Delete a node and its edges
    ///
    /// Args:
    ///     graph_id: Graph identifier
    ///     node_id: Node ID to delete
    ///
    /// Returns:
    ///     True if node was deleted, False if not found
    fn delete_node(&self, graph_id: &str, node_id: &str) -> PyResult<bool> {
        self.db()?
            .delete_node(graph_id, node_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete node: {}", e)))
    }

    /// Get graph statistics
    ///
    /// Args:
    ///     graph_id: Graph identifier
    ///
    /// Returns:
    ///     GraphStats with total_nodes and total_edges
    fn graph_stats(&self, graph_id: &str) -> PyResult<PyGraphStats> {
        self.db()?
            .graph_stats(graph_id)
            .map(|s| s.into())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get graph stats: {}", e)))
    }

    #[pyo3(signature = (graph_id, start_node_id, max_depth=3, edge_types=None, limit=None))]
    fn traverse_graph(
        &self,
        py: Python<'_>,
        graph_id: &str,
        start_node_id: &str,
        max_depth: u32,
        edge_types: Option<Vec<String>>,
        limit: Option<u32>,
    ) -> PyResult<Py<PyAny>> {
        let result = self
            .db()?
            .traverse_graph(graph_id, start_node_id, max_depth, edge_types, limit)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to traverse graph: {}", e)))?;

        let dict = PyDict::new(py);
        let nodes: Vec<PyGraphNode> = result.nodes.into_iter().map(PyGraphNode::from).collect();
        let edges: Vec<PyGraphEdge> = result.edges.into_iter().map(PyGraphEdge::from).collect();
        dict.set_item("nodes", nodes)?;
        dict.set_item("edges", edges)?;
        dict.set_item("paths", result.paths)?;

        let stats = PyDict::new(py);
        if let Some(stat_values) = result.stats {
            stats.set_item("nodes_visited", stat_values.nodes_visited)?;
            stats.set_item("edges_traversed", stat_values.edges_traversed)?;
            stats.set_item("max_depth_reached", stat_values.max_depth_reached)?;
            stats.set_item(
                "execution_time_microseconds",
                stat_values.execution_time_microseconds,
            )?;
        }
        dict.set_item("stats", stats)?;

        Ok(dict.into())
    }

    /// Impact analysis (TD-131): forward blast radius (outgoing edges — "what does X impact") or
    /// backward (incoming edges — "what impacts X"). Delegates to the same server graph service
    /// method the REST `POST /api/v2/graphs/{id}/impact-analysis` endpoint uses.
    #[pyo3(signature = (graph_id, node_id, direction="forward", max_depth=3, edge_types=None, limit=100))]
    fn impact_analysis(
        &self,
        py: Python<'_>,
        graph_id: &str,
        node_id: &str,
        direction: &str,
        max_depth: u32,
        edge_types: Option<Vec<String>>,
        limit: usize,
    ) -> PyResult<Py<PyAny>> {
        let result = self
            .db()?
            .impact_analysis(graph_id, node_id, direction, edge_types, max_depth, limit)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed impact analysis: {}", e)))?;

        let dict = PyDict::new(py);
        let nodes: Vec<PyGraphNode> = result.nodes.into_iter().map(PyGraphNode::from).collect();
        let edges: Vec<PyGraphEdge> = result.edges.into_iter().map(PyGraphEdge::from).collect();
        dict.set_item("nodes", nodes)?;
        dict.set_item("edges", edges)?;

        let stats = PyDict::new(py);
        if let Some(stat_values) = result.stats {
            stats.set_item("edges_traversed", stat_values.edges_traversed)?;
            stats.set_item("max_depth_reached", stat_values.max_depth_reached)?;
        }
        dict.set_item("stats", stats)?;

        Ok(dict.into())
    }

    /// Vector-seed → graph-expand → fuse-by-`oid` (TD-137/TD-131). Runs the SAME `FusionService`
    /// core as the REST endpoint in-process, so embedded results match the server path by
    /// construction (the embedded-parity guarantee). Returns `{results: [{oid, score,
    /// source_count}], stats: {...}}`.
    #[pyo3(signature = (graph_id, vector_collection, query_vector, max_depth=1, edge_types=None, max_seeds=5, limit=10, vector_weight=1.0, graph_weight=1.0, rrf=false, grain="nodes", text_query=None, document_collection=None, document_weight=None))]
    fn fusion_search(
        &self,
        py: Python<'_>,
        graph_id: &str,
        vector_collection: &str,
        query_vector: Vec<f32>,
        max_depth: u32,
        edge_types: Option<Vec<String>>,
        max_seeds: usize,
        limit: usize,
        vector_weight: f32,
        graph_weight: f32,
        rrf: bool,
        grain: &str,
        text_query: Option<String>,
        document_collection: Option<String>,
        document_weight: Option<f32>,
    ) -> PyResult<Py<PyAny>> {
        let (items, stats) = self
            .db()?
            .fusion_search(
                graph_id,
                vector_collection,
                query_vector,
                max_depth,
                edge_types.unwrap_or_default(),
                max_seeds,
                limit,
                vector_weight,
                graph_weight,
                rrf,
                grain,
                text_query,
                document_collection,
                document_weight,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed fusion search: {}", e)))?;

        let dict = PyDict::new(py);
        let mut results: Vec<Py<PyAny>> = Vec::with_capacity(items.len());
        for item in items {
            let hit = PyDict::new(py);
            hit.set_item("oid", item.oid)?;
            hit.set_item("score", item.score)?;
            hit.set_item("source_count", item.source_count)?;
            results.push(hit.into());
        }
        dict.set_item("results", results)?;

        let stats_dict = PyDict::new(py);
        stats_dict.set_item("sources_fused", stats.sources_fused)?;
        stats_dict.set_item("sources_skipped", stats.sources_skipped)?;
        stats_dict.set_item("candidates_in", stats.candidates_in)?;
        stats_dict.set_item("items_out", stats.items_out)?;
        dict.set_item("stats", stats_dict)?;

        Ok(dict.into())
    }

    /// Execute a read-only Cypher query through the embedded GRAPH_QUERY SQL
    /// extension. The durable authority remains the shared graph/query stack.
    #[pyo3(signature = (cypher, graph_id=None))]
    fn execute_cypher(
        &self,
        py: Python<'_>,
        cypher: &str,
        graph_id: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let cypher = inject_graph_target_into_cypher(graph_id, cypher);
        let sql = format!(
            "SELECT * FROM GRAPH_QUERY('{}')",
            cypher.replace('\'', "''")
        );
        self.execute_sql(py, &sql, None, None)
    }

    /// Delete entire graph
    ///
    /// Removes all nodes and edges in the graph.
    ///
    /// Args:
    ///     graph_id: Graph identifier
    fn delete_graph(&self, graph_id: &str) -> PyResult<()> {
        self.db()?
            .delete_graph(graph_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete graph: {}", e)))
    }

    // ========================================================================
    // Document Storage Operations
    // ========================================================================

    /// Create a new document collection
    ///
    /// Args:
    ///     name: Collection name
    ///     indexed_paths: Optional list of JSON paths to index (e.g., ["$.email", "$.profile.name"])
    ///
    /// Example:
    ///     ```python
    ///     db.create_document_collection("users", indexed_paths=["$.email", "$.profile.name"])
    ///     ```
    #[pyo3(signature = (name, indexed_paths=None))]
    fn create_document_collection(
        &self,
        name: &str,
        indexed_paths: Option<Vec<String>>,
    ) -> PyResult<()> {
        self.db()?
            .create_document_collection(name, indexed_paths)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create document collection: {}", e))
            })
    }

    /// Insert a document into a collection
    ///
    /// Args:
    ///     collection: Collection name
    ///     document: Dictionary representing the JSON document
    ///     doc_id: Optional document ID (auto-generated if not provided)
    ///
    /// Returns:
    ///     Tuple of (doc_id, version)
    ///
    /// Example:
    ///     ```python
    ///     doc_id, version = db.insert_document("users", {
    ///         "name": "John",
    ///         "email": "john@example.com",
    ///         "profile": {"age": 30, "city": "NYC"}
    ///     })
    ///     ```
    #[pyo3(signature = (collection, document, doc_id=None))]
    fn insert_document(
        &self,
        collection: &str,
        document: &Bound<'_, PyDict>,
        doc_id: Option<&str>,
    ) -> PyResult<(String, u64)> {
        // Convert Python dict to serde_json::Value
        let json_doc = python_to_json(document.as_any())?;

        self.db()?
            .insert_document(collection, doc_id, json_doc)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to insert document: {}", e)))
    }

    /// Get a document by ID
    ///
    /// Args:
    ///     collection: Collection name
    ///     doc_id: Document ID
    ///
    /// Returns:
    ///     Document as dictionary, or None if not found
    ///
    /// Example:
    ///     ```python
    ///     doc = db.get_document("users", "user_123")
    ///     if doc:
    ///         print(f"Name: {doc['name']}")
    ///     ```
    fn get_document(
        &self,
        py: Python<'_>,
        collection: &str,
        doc_id: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        match self.db()?.get_document(collection, doc_id) {
            Ok(Some(doc)) => json_to_python(py, &doc).map(Some),
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get document: {}",
                e
            ))),
        }
    }

    /// Query documents with optional filter
    ///
    /// Args:
    ///     collection: Collection name
    ///     filter: Optional filter expression (e.g., "$.profile.city = 'NYC'")
    ///     limit: Maximum number of documents to return (default: 100)
    ///
    /// Returns:
    ///     List of (doc_id, document) tuples
    ///
    /// Example:
    ///     ```python
    ///     results = db.query_documents("users", filter="$.profile.age > 25", limit=10)
    ///     for doc_id, doc in results:
    ///         print(f"{doc_id}: {doc['name']}")
    ///     ```
    #[pyo3(signature = (collection, filter=None, limit=100))]
    fn query_documents(
        &self,
        py: Python<'_>,
        collection: &str,
        filter: Option<&str>,
        limit: u32,
    ) -> PyResult<Vec<(String, Py<PyAny>)>> {
        self.db()?
            .query_documents(collection, filter, limit)
            .map(|results| {
                results
                    .into_iter()
                    .filter_map(|(id, doc)| {
                        json_to_python(py, &doc).ok().map(|py_doc| (id, py_doc))
                    })
                    .collect()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to query documents: {}", e)))
    }

    fn update_document(
        &self,
        collection: &str,
        doc_id: &str,
        updates: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let mut rust_updates = HashMap::new();
        for (k, v) in updates.iter() {
            let key: String = k.extract()?;
            let value = python_to_json(&v)?;
            rust_updates.insert(key, value);
        }

        self.db()?
            .update_document(collection, doc_id, rust_updates)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update document: {}", e)))
    }

    /// Delete a document by ID
    ///
    /// Args:
    ///     collection: Collection name
    ///     doc_id: Document ID
    ///
    /// Returns:
    ///     True if deleted, False if not found
    fn delete_document(&self, collection: &str, doc_id: &str) -> PyResult<bool> {
        self.db()?
            .delete_document(collection, doc_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete document: {}", e)))
    }

    /// List all document collections
    ///
    /// Returns:
    ///     List of collection names
    fn list_document_collections(&self) -> PyResult<Vec<String>> {
        self.db()?.list_document_collections().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to list document collections: {}", e))
        })
    }

    fn delete_document_collection(&self, name: &str) -> PyResult<bool> {
        self.db()?.delete_document_collection(name).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to delete document collection: {}", e))
        })
    }

    // ========================================================================
    // Observability Operations (Logs, Metrics, Traces)
    // ========================================================================

    /// Create an observability namespace
    ///
    /// Args:
    ///     name: Namespace name
    ///     retention_days: Optional retention period in days (default: 30)
    ///
    /// Example:
    ///     ```python
    ///     db.create_observability_namespace("production", retention_days=90)
    ///     ```
    #[pyo3(signature = (name, retention_days=None))]
    fn create_observability_namespace(
        &self,
        name: &str,
        retention_days: Option<u32>,
    ) -> PyResult<()> {
        // Convert retention_days to retention_hours for the inner API
        let retention_hours = retention_days.map(|d| d as u64 * 24);
        self.db()?
            .create_observability_namespace(name, retention_hours)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create observability namespace: {}", e))
            })
    }

    /// Ingest log entries
    ///
    /// Args:
    ///     namespace: Observability namespace
    ///     logs: List of log entry dicts with keys: timestamp_ns, severity, message, source, service, fields
    ///
    /// Returns:
    ///     Number of logs successfully ingested
    ///
    /// Example:
    ///     ```python
    ///     logs = [
    ///         {"timestamp_ns": 1703000000000000000, "severity": "INFO", "message": "Server started", "source": "main", "service": "api"},
    ///         {"timestamp_ns": 1703000001000000000, "severity": "ERROR", "message": "Connection failed", "source": "db", "service": "api"}
    ///     ]
    ///     count = db.ingest_logs("production", logs)
    ///     ```
    fn ingest_logs(
        &self,
        py: Python<'_>,
        namespace: &str,
        logs: &Bound<'_, PyList>,
    ) -> PyResult<u64> {
        use super::EmbeddedLogEntry;

        let mut rust_logs = Vec::with_capacity(logs.len());
        for item in logs.iter() {
            let dict = item.cast::<PyDict>()?;

            let timestamp_ns: i64 = dict
                .get_item("timestamp_ns")?
                .ok_or_else(|| PyValueError::new_err("Log entry missing 'timestamp_ns'"))?
                .extract()?;
            let severity: String = dict
                .get_item("severity")?
                .ok_or_else(|| PyValueError::new_err("Log entry missing 'severity'"))?
                .extract()?;
            let message: String = dict
                .get_item("message")?
                .ok_or_else(|| PyValueError::new_err("Log entry missing 'message'"))?
                .extract()?;
            let source: Option<String> = dict
                .get_item("source")?
                .and_then(|v| v.extract::<String>().ok())
                .filter(|s| !s.is_empty());
            let service: Option<String> = dict
                .get_item("service")?
                .and_then(|v| v.extract::<String>().ok())
                .filter(|s| !s.is_empty());

            let mut fields = std::collections::HashMap::new();
            if let Some(fields_dict) = dict.get_item("fields")? {
                if let Ok(d) = fields_dict.cast::<PyDict>() {
                    for (k, v) in d.iter() {
                        let key: String = k.extract()?;
                        let value = python_to_json(&v)?;
                        fields.insert(key, value);
                    }
                }
            }

            rust_logs.push(EmbeddedLogEntry {
                timestamp_ns,
                message,
                severity,
                service,
                source,
                fields,
            });
        }

        let inner = self.db()?;
        py.detach(move || inner.ingest_logs(namespace, rust_logs))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to ingest logs: {}", e)))
    }

    /// Query logs
    ///
    /// Args:
    ///     namespace: Observability namespace
    ///     start_time_ns: Start timestamp in nanoseconds
    ///     end_time_ns: End timestamp in nanoseconds
    ///     query: Optional search query string
    ///     limit: Maximum number of logs to return (default: 100)
    ///
    /// Returns:
    ///     List of log entry dicts
    ///
    /// Example:
    ///     ```python
    ///     import time
    ///     now = int(time.time() * 1e9)
    ///     hour_ago = now - 3600_000_000_000
    ///     logs = db.query_logs("production", hour_ago, now, query="error", limit=50)
    ///     ```
    #[pyo3(signature = (namespace, start_time_ns, end_time_ns, query=None, limit=100))]
    fn query_logs(
        &self,
        py: Python<'_>,
        namespace: &str,
        start_time_ns: i64,
        end_time_ns: i64,
        query: Option<&str>,
        limit: u32,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let inner = self.db()?;
        py.detach(move || inner.query_logs(namespace, start_time_ns, end_time_ns, query, limit))
            .map(|logs| {
                logs.into_iter()
                    .map(|log| {
                        let dict = PyDict::new(py);
                        dict.set_item("timestamp_ns", log.timestamp_ns).ok();
                        dict.set_item("severity", &log.severity).ok();
                        dict.set_item("message", &log.message).ok();
                        // Handle Option<String> for source and service
                        if let Some(ref source) = log.source {
                            dict.set_item("source", source).ok();
                        } else {
                            dict.set_item("source", py.None()).ok();
                        }
                        if let Some(ref service) = log.service {
                            dict.set_item("service", service).ok();
                        } else {
                            dict.set_item("service", py.None()).ok();
                        }

                        let fields_dict = PyDict::new(py);
                        for (k, v) in &log.fields {
                            // Convert serde_json::Value to Python
                            if let Ok(py_val) = json_to_python(py, v) {
                                fields_dict.set_item(k, py_val).ok();
                            }
                        }
                        dict.set_item("fields", fields_dict).ok();

                        dict.into()
                    })
                    .collect()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to query logs: {}", e)))
    }

    /// Ingest metric samples
    ///
    /// Args:
    ///     namespace: Observability namespace
    ///     samples: List of metric sample dicts with keys: metric_name, timestamp_ns, value, labels
    ///
    /// Returns:
    ///     Number of samples successfully ingested
    ///
    /// Example:
    ///     ```python
    ///     samples = [
    ///         {"metric_name": "http_latency", "timestamp_ns": 1703000000000000000, "value": 0.123, "labels": {"endpoint": "/api/search"}},
    ///         {"metric_name": "cpu_usage", "timestamp_ns": 1703000001000000000, "value": 65.5, "labels": {"host": "server1"}}
    ///     ]
    ///     count = db.ingest_metrics("production", samples)
    ///     ```
    fn ingest_metrics(
        &self,
        py: Python<'_>,
        namespace: &str,
        samples: &Bound<'_, PyList>,
    ) -> PyResult<u64> {
        use super::EmbeddedMetricSample;

        let mut rust_samples = Vec::with_capacity(samples.len());
        for item in samples.iter() {
            let dict = item.cast::<PyDict>()?;

            let metric_name: String = dict
                .get_item("metric_name")?
                .ok_or_else(|| PyValueError::new_err("Metric sample missing 'metric_name'"))?
                .extract()?;
            let timestamp_ns: i64 = dict
                .get_item("timestamp_ns")?
                .ok_or_else(|| PyValueError::new_err("Metric sample missing 'timestamp_ns'"))?
                .extract()?;
            let value: f64 = dict
                .get_item("value")?
                .ok_or_else(|| PyValueError::new_err("Metric sample missing 'value'"))?
                .extract()?;

            let mut labels = HashMap::new();
            if let Some(labels_dict) = dict.get_item("labels")? {
                if let Ok(d) = labels_dict.cast::<PyDict>() {
                    for (k, v) in d.iter() {
                        let key: String = k.extract()?;
                        let val: String = v
                            .extract()
                            .unwrap_or_else(|_| v.str().map(|s| s.to_string()).unwrap_or_default());
                        labels.insert(key, val);
                    }
                }
            }

            rust_samples.push(EmbeddedMetricSample {
                metric_name,
                timestamp_ns,
                value,
                labels,
            });
        }

        let inner = self.db()?;
        py.detach(move || inner.ingest_metrics(namespace, rust_samples))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to ingest metrics: {}", e)))
    }

    #[pyo3(signature = (namespace, metric_name, aggregation="avg", start_time=None, end_time=None, step_seconds=60))]
    fn aggregate_metrics(
        &self,
        py: Python<'_>,
        namespace: &str,
        metric_name: &str,
        aggregation: &str,
        start_time: Option<&str>,
        end_time: Option<&str>,
        step_seconds: u32,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let inner = self.db()?;
        py.detach(move || {
            inner.aggregate_metrics(
                namespace,
                metric_name,
                aggregation,
                start_time,
                end_time,
                step_seconds,
            )
        })
        .and_then(|points| {
            points
                .into_iter()
                .map(|point| {
                    let dict = PyDict::new(py);
                    dict.set_item("timestamp_ns", point.timestamp_ns)?;
                    dict.set_item("value", point.value)?;
                    Ok(dict.into())
                })
                .collect()
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to aggregate metrics: {}", e)))
    }

    fn ingest_traces(
        &self,
        py: Python<'_>,
        namespace: &str,
        traces: &Bound<'_, PyList>,
    ) -> PyResult<u64> {
        use super::EmbeddedTraceSpan;

        let mut rust_traces = Vec::with_capacity(traces.len());
        for item in traces.iter() {
            let dict = item.cast::<PyDict>()?;
            let trace_id: String = dict
                .get_item("trace_id")?
                .ok_or_else(|| PyValueError::new_err("Trace span missing 'trace_id'"))?
                .extract()?;
            let span_id: String = dict
                .get_item("span_id")?
                .ok_or_else(|| PyValueError::new_err("Trace span missing 'span_id'"))?
                .extract()?;
            let name: String = dict
                .get_item("name")?
                .ok_or_else(|| PyValueError::new_err("Trace span missing 'name'"))?
                .extract()?;
            let kind: String = dict
                .get_item("kind")?
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_else(|| "INTERNAL".to_string());
            let start_time_ns: i64 = dict
                .get_item("start_time_ns")?
                .ok_or_else(|| PyValueError::new_err("Trace span missing 'start_time_ns'"))?
                .extract()?;
            let end_time_ns: i64 = dict
                .get_item("end_time_ns")?
                .ok_or_else(|| PyValueError::new_err("Trace span missing 'end_time_ns'"))?
                .extract()?;
            let parent_span_id = dict
                .get_item("parent_span_id")?
                .and_then(|v| v.extract::<String>().ok());
            let service = dict
                .get_item("service")?
                .and_then(|v| v.extract::<String>().ok());
            let status_code = dict
                .get_item("status_code")?
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_else(|| "UNSET".to_string());
            let status_message = dict
                .get_item("status_message")?
                .and_then(|v| v.extract::<String>().ok());

            let mut attributes = HashMap::new();
            if let Some(attr_dict) = dict.get_item("attributes")?
                && let Ok(py_dict) = attr_dict.cast::<PyDict>()
            {
                for (k, v) in py_dict.iter() {
                    let key: String = k.extract()?;
                    attributes.insert(key, python_to_json(&v)?);
                }
            }

            rust_traces.push(EmbeddedTraceSpan {
                trace_id,
                span_id,
                parent_span_id,
                name,
                kind,
                start_time_ns,
                end_time_ns,
                service,
                status_code,
                status_message,
                attributes,
            });
        }

        let inner = self.db()?;
        py.detach(move || inner.ingest_traces(namespace, rust_traces))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to ingest traces: {}", e)))
    }

    #[pyo3(signature = (namespace, start_time_ns, end_time_ns, trace_id=None, service=None, operation=None, min_duration_ns=None, status=None, limit=100))]
    fn query_traces(
        &self,
        py: Python<'_>,
        namespace: &str,
        start_time_ns: i64,
        end_time_ns: i64,
        trace_id: Option<&str>,
        service: Option<&str>,
        operation: Option<&str>,
        min_duration_ns: Option<i64>,
        status: Option<&str>,
        limit: u32,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let inner = self.db()?;
        py.detach(move || {
            inner.query_traces(
                namespace,
                start_time_ns,
                end_time_ns,
                trace_id,
                service,
                operation,
                min_duration_ns,
                status,
                limit,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to query traces: {}", e)))
        .and_then(|spans| {
            spans
                .into_iter()
                .map(|span| trace_span_to_python(py, span))
                .collect::<PyResult<Vec<_>>>()
        })
    }

    fn get_trace(&self, py: Python<'_>, namespace: &str, trace_id: &str) -> PyResult<Py<PyAny>> {
        let inner = self.db()?;
        py.detach(move || inner.get_trace(namespace, trace_id))
            .and_then(|trace| {
                let dict = PyDict::new(py);
                let spans: Vec<Py<PyAny>> = trace
                    .spans
                    .into_iter()
                    .map(|span| trace_span_to_python(py, span))
                    .collect::<PyResult<_>>()?;
                dict.set_item("spans", spans)?;
                dict.set_item("complete", trace.complete)?;
                Ok(dict.into())
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trace: {}", e)))
    }

    // ============================================
    // Unified Multi-Model Query API
    // ============================================

    #[pyo3(signature = (query, parameters=None, collection=None))]
    fn execute_sql(
        &self,
        py: Python<'_>,
        query: &str,
        parameters: Option<&Bound<'_, PyAny>>,
        collection: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let rust_params = if let Some(params) = parameters {
            let json_value = python_to_json(params)?;
            Some(match json_value {
                serde_json::Value::Array(values) => values,
                other => vec![other],
            })
        } else {
            None
        };

        let inner = self.db()?;
        py.detach(move || inner.execute_sql(query, rust_params, collection))
            .and_then(|result| {
                let dict = PyDict::new(py);
                let rows = PyList::empty(py);
                for row in result.rows {
                    rows.append(json_to_python(py, &row)?)?;
                }
                dict.set_item("rows", rows)?;
                dict.set_item("columns", result.columns)?;
                dict.set_item("column_types", result.column_types)?;
                dict.set_item("row_count", result.row_count)?;
                dict.set_item("rows_scanned", result.rows_scanned)?;
                dict.set_item("execution_time_ms", result.execution_time_ms)?;
                Ok(dict.into())
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to execute SQL: {}", e)))
    }

    #[pyo3(signature = (query, parameters=None, collection=None))]
    fn execute_sql_arrow_ipc(
        &self,
        py: Python<'_>,
        query: &str,
        parameters: Option<&Bound<'_, PyAny>>,
        collection: Option<&str>,
    ) -> PyResult<Vec<u8>> {
        let rust_params = if let Some(params) = parameters {
            let json_value = python_to_json(params)?;
            Some(match json_value {
                serde_json::Value::Array(values) => values,
                other => vec![other],
            })
        } else {
            None
        };

        let inner = self.db()?;
        py.detach(move || inner.execute_sql(query, rust_params, collection))
            .and_then(|result| sql_result_to_arrow_ipc(&result))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to execute SQL as Arrow: {}", e)))
    }

    fn explain_notebook_plan(
        &self,
        py: Python<'_>,
        plan: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let plan_json = python_to_json(plan)?;
        let explanation = notebook_plan_explain(self.db()?.as_ref(), &plan_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid notebook plan: {}", e)))?;
        json_to_python(py, &explanation)
    }

    fn execute_notebook_plan_arrow_ipc(
        &self,
        py: Python<'_>,
        plan: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<u8>> {
        let plan_json = python_to_json(plan)?;
        let sql = compile_notebook_plan_sql(&plan_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid notebook plan: {}", e)))?;
        let inner = self.db()?;
        py.detach(move || inner.execute_sql(&sql, None, None))
            .and_then(|result| sql_result_to_arrow_ipc(&result))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to execute notebook plan as Arrow: {}", e))
            })
    }

    /// Create a DataFusion session for distributed execution (Proxima-Spark)
    #[cfg(feature = "datafusion-integration")]
    fn dataframe_session(&self) -> PyResult<crate::python_dataframe::PyDataFusionSession> {
        let db = self.db()?;
        let vector_ops = db.shared_services().vector_operations_service.clone();

        let ctx = proximadb::datafusion::create_session_context_with_vector_ops(vector_ops)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create DataFusion context: {}", e))
            })?;

        Ok(crate::python_dataframe::PyDataFusionSession::new(ctx, db))
    }

    #[pyo3(signature = (collection, ipc_stream, mode="insert", tenant_id=None))]
    fn insert_arrow_ipc(
        &self,
        py: Python<'_>,
        collection: &str,
        ipc_stream: Vec<u8>,
        mode: &str,
        tenant_id: Option<&str>,
    ) -> PyResult<u64> {
        let insert_only = match mode.to_ascii_lowercase().as_str() {
            "insert" | "bulk_insert" | "batch_insert" => true,
            "upsert" | "bulk_upsert" | "batch_upsert" => false,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid Arrow write mode '{}'. Use 'insert' or 'upsert'",
                    other
                )));
            }
        };

        let inner = self.db()?;
        py.detach(move || inner.insert_arrow_ipc(collection, &ipc_stream, insert_only, tenant_id))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to insert Arrow batch: {}", e)))
    }

    #[pyo3(signature = (collection, ipc_stream, tenant_id=None))]
    fn upsert_arrow_ipc(
        &self,
        py: Python<'_>,
        collection: &str,
        ipc_stream: Vec<u8>,
        tenant_id: Option<&str>,
    ) -> PyResult<u64> {
        let inner = self.db()?;
        py.detach(move || inner.insert_arrow_ipc(collection, &ipc_stream, false, tenant_id))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to upsert Arrow batch: {}", e)))
    }

    /// Zero-copy Arrow insert via the **Arrow C Data Interface**. Accepts a list of
    /// pyarrow `RecordBatch`es whose buffers are shared (not serialized) across the
    /// FFI boundary, and routes them straight into the embedded vector service —
    /// the zero-copy replacement for the pyarrow-IPC serialize/deserialize
    /// round-trip of `insert_arrow_ipc`. Co-design Pillar C: this is the dominant
    /// cost term of the boundary that replaced the deleted network dimension.
    #[pyo3(signature = (collection, batches, mode="insert", tenant_id=None))]
    fn insert_arrow_batches(
        &self,
        py: Python<'_>,
        collection: &str,
        batches: Vec<arrow::pyarrow::PyArrowType<arrow::record_batch::RecordBatch>>,
        mode: &str,
        tenant_id: Option<&str>,
    ) -> PyResult<u64> {
        let insert_only = match mode.to_ascii_lowercase().as_str() {
            "insert" | "bulk_insert" | "batch_insert" => true,
            "upsert" | "bulk_upsert" | "batch_upsert" => false,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid Arrow write mode '{}'. Use 'insert' or 'upsert'",
                    other
                )));
            }
        };
        // The Arrow C Data Interface import happens during argument extraction
        // (GIL held); the batches are then owned Rust RecordBatches, so the GIL is
        // released for the actual ingest.
        let batches: Vec<arrow::record_batch::RecordBatch> =
            batches.into_iter().map(|b| b.0).collect();
        let inner = self.db()?;
        py.detach(move || inner.insert_arrow_batches(collection, batches, insert_only, tenant_id))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to insert Arrow batch: {}", e)))
    }

    /// Execute a unified multi-model query
    ///
    /// This executes a SQL-like query that can span multiple data models
    /// (vector, document, graph, observability).
    ///
    /// Args:
    ///     query: SQL-like query string (e.g., "SELECT * FROM products WHERE VECTOR_SIMILAR(embedding, ?, 0.8)")
    ///     query_vector: Optional query vector for VECTOR_SIMILAR clauses
    ///     fusion_strategy: Strategy for combining results ("intersection", "union", "rrf", "ranked")
    ///
    /// Returns:
    ///     List of unified record dicts with keys: id, source_model, data, score, metadata
    ///
    /// Example:
    ///     ```python
    ///     # Hybrid vector + document query
    ///     results = db.execute_unified_query(
    ///         "SELECT * FROM products WHERE $.category = 'electronics' AND VECTOR_SIMILAR(embedding, ?, 0.8)",
    ///         query_vector=[0.1] * 384,
    ///         fusion_strategy="intersection"
    ///     )
    ///     for r in results:
    ///         print(f"{r['id']}: {r['score']}")
    ///     ```
    #[pyo3(signature = (query, query_vector=None, fusion_strategy=None))]
    fn execute_unified_query(
        &self,
        py: Python<'_>,
        query: &str,
        query_vector: Option<Vec<f32>>,
        fusion_strategy: Option<&str>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let inner = self.db()?;
        py.detach(move || inner.execute_unified_query(query, query_vector, fusion_strategy))
            .map(|records| {
                records
                    .into_iter()
                    .map(|r| {
                        let dict = PyDict::new(py);
                        dict.set_item("id", &r.id).ok();
                        dict.set_item("source_model", &r.source_model).ok();
                        dict.set_item("score", r.score).ok();

                        // Convert data from JSON string to Python
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&r.data) {
                            if let Ok(data_py) = json_to_python(py, &parsed) {
                                dict.set_item("data", data_py).ok();
                            }
                        } else {
                            // Fallback: use raw string
                            dict.set_item("data", &r.data).ok();
                        }

                        // Convert metadata
                        let meta_dict = PyDict::new(py);
                        for (k, v) in &r.metadata {
                            meta_dict.set_item(k, v).ok();
                        }
                        dict.set_item("metadata", meta_dict).ok();

                        dict.into()
                    })
                    .collect()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to execute unified query: {}", e)))
    }

    /// Explain a unified query's execution plan
    ///
    /// Returns the decomposition and execution plan for a multi-model query
    /// without actually executing it.
    ///
    /// Args:
    ///     query: SQL-like query string
    ///
    /// Returns:
    ///     Dict with plan details: components, fusion_strategy, component_count
    ///
    /// Example:
    ///     ```python
    ///     plan = db.explain_unified_query(
    ///         "SELECT * FROM products WHERE VECTOR_SIMILAR(embedding, ?, 0.8)"
    ///     )
    ///     print(f"Components: {plan['component_count']}")
    ///     for comp in plan['components']:
    ///         print(f"  {comp['model']}: cost={comp['estimated_cost']}")
    ///     ```
    fn explain_unified_query(&self, py: Python<'_>, query: &str) -> PyResult<Py<PyAny>> {
        self.db()?
            .explain_unified_query(query)
            .map(|plan| {
                let dict = PyDict::new(py);
                dict.set_item("fusion_strategy", &plan.fusion_strategy).ok();
                dict.set_item("component_count", plan.component_count).ok();

                // Convert components
                let components_list = PyList::empty(py);
                for comp in plan.components {
                    let comp_dict = PyDict::new(py);
                    comp_dict.set_item("model", &comp.model).ok();
                    comp_dict
                        .set_item("parallelizable", comp.parallelizable)
                        .ok();
                    comp_dict
                        .set_item("estimated_cost", comp.estimated_cost)
                        .ok();
                    components_list.append(comp_dict).ok();
                }
                dict.set_item("components", components_list).ok();

                dict.into()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to explain query: {}", e)))
    }

    // ============================================
    // Embedded Observability Metrics API
    // ============================================

    /// Get embedded database metrics snapshot
    ///
    /// Returns comprehensive metrics including latency histograms (p50, p95, p99),
    /// operation counters, cache statistics, and WAL statistics.
    ///
    /// Args:
    ///     window: Rolling window for latency stats - "1min", "5min", "1hr", or "all" (default: "all")
    ///
    /// Returns:
    ///     EmbeddedMetrics object with all statistics
    ///
    /// Example:
    ///     ```python
    ///     metrics = db.metrics()
    ///     print(f"p99 search latency: {metrics.search_latency.p99_ms}ms")
    ///     print(f"Cache hit rate: {metrics.cache_hit_rate * 100}%")
    ///     print(f"Total searches: {metrics.total_searches}")
    ///
    ///     # Get 1-minute rolling window
    ///     recent = db.metrics(window="1min")
    ///     ```
    #[pyo3(signature = (window=None))]
    fn metrics(&self, window: Option<&str>) -> PyResult<PyEmbeddedMetrics> {
        let rolling_window = match window.unwrap_or("all") {
            "1min" | "1m" | "one_minute" => super::RollingWindow::OneMinute,
            "5min" | "5m" | "five_minutes" => super::RollingWindow::FiveMinutes,
            "1hr" | "1h" | "one_hour" => super::RollingWindow::OneHour,
            "all" | "alltime" | "all_time" => super::RollingWindow::AllTime,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid window '{}'. Use: '1min', '5min', '1hr', or 'all'",
                    other
                )));
            }
        };

        Ok(self.db()?.metrics(rolling_window).into())
    }

    /// Reset all metrics to zero
    ///
    /// Clears all latency histograms, operation counters, and cache hit/miss counts.
    /// Useful for benchmarking or starting fresh measurements.
    ///
    /// Example:
    ///     ```python
    ///     db.reset_metrics()
    ///     # Run benchmark...
    ///     metrics = db.metrics()
    ///     print(f"Benchmark results: {metrics}")
    ///     ```
    fn reset_metrics(&self) {
        if let Some(db) = self.db_opt() {
            db.reset_metrics();
        }
    }

    /// Export metrics in Prometheus text format
    ///
    /// Returns metrics formatted for Prometheus scraping. Can be saved to a file
    /// or served via an HTTP endpoint for monitoring.
    ///
    /// Returns:
    ///     String in Prometheus exposition format
    ///
    /// Example:
    ///     ```python
    ///     prometheus_text = db.export_prometheus()
    ///
    ///     # Save to file for node_exporter textfile collector
    ///     with open("/var/lib/prometheus/embedded.prom", "w") as f:
    ///         f.write(prometheus_text)
    ///
    ///     # Or print to stdout
    ///     print(prometheus_text)
    ///     ```
    fn export_prometheus(&self) -> String {
        self.db_opt()
            .map(|db| db.export_prometheus())
            .unwrap_or_default()
    }
}

/// Convert Python value to serde_json::Value
fn warn_user(py: Python<'_>, message: &str, stacklevel: i32) -> PyResult<()> {
    let warning = CString::new(message)
        .map_err(|_| PyValueError::new_err("warning message contains embedded NUL byte"))?;
    let category = py.get_type::<PyUserWarning>();
    PyErr::warn(py, category.as_any(), &warning, stacklevel)
}

fn python_to_json(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if value.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = value.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(serde_json::Value::Number(i.into()))
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = value.cast::<PyList>() {
        let arr: Vec<serde_json::Value> = list
            .iter()
            .map(|item| python_to_json(&item))
            .collect::<PyResult<_>>()?;
        Ok(serde_json::Value::Array(arr))
    } else if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, python_to_json(&v)?);
        }
        Ok(serde_json::Value::Object(map))
    } else {
        // Fallback: convert to string representation
        Ok(serde_json::Value::String(value.str()?.to_string()))
    }
}

fn python_to_json_record_value(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if value.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = value.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(serde_json::Value::Number(i.into()))
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = value.cast::<PyList>() {
        let arr: Vec<serde_json::Value> = list
            .iter()
            .map(|item| python_to_json_record_value(&item))
            .collect::<PyResult<_>>()?;
        Ok(serde_json::Value::Array(arr))
    } else if let Ok(tuple) = value.cast::<PyTuple>() {
        let arr: Vec<serde_json::Value> = tuple
            .iter()
            .map(|item| python_to_json_record_value(&item))
            .collect::<PyResult<_>>()?;
        Ok(serde_json::Value::Array(arr))
    } else if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, python_to_json_record_value(&v)?);
        }
        Ok(serde_json::Value::Object(map))
    } else if value.hasattr("to_dict")? {
        let converted = value.call_method0("to_dict")?;
        python_to_json_record_value(&converted)
    } else {
        Ok(serde_json::Value::String(value.str()?.to_string()))
    }
}

fn python_record_field<'py>(
    record: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(dict) = record.cast::<PyDict>() {
        dict.get_item(name)
    } else if record.hasattr(name)? {
        Ok(Some(record.getattr(name)?))
    } else {
        Ok(None)
    }
}

fn required_python_record_field<'py>(
    record: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    python_record_field(record, name)?
        .ok_or_else(|| PyValueError::new_err(format!("ProximaRecord is missing {name:?}")))
}

fn python_record_vector(value: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if value.hasattr("tolist")? {
        value.call_method0("tolist")?.extract()
    } else {
        value.extract()
    }
}

fn python_record_props(record: &Bound<'_, PyAny>) -> PyResult<proximadb_records::ProximaTree> {
    let mut props = proximadb_records::ProximaTree::new();
    if let Some(props_value) = python_record_field(record, "props")?
        && !props_value.is_none()
    {
        let props_dict = props_value.cast::<PyDict>()?;
        for (key, value) in props_dict.iter() {
            let key: String = key.extract()?;
            props.insert(
                key,
                proximadb_records::ProximaTreeNode::Value(json_to_proxima_value(
                    python_to_json_record_value(&value)?,
                )),
            );
        }
    }

    for (field_name, prop_name) in [
        ("text_fields", "_text_fields"),
        ("source", "_source"),
        ("schema_id", "_schema_id"),
    ] {
        if let Some(value) = python_record_field(record, field_name)?
            && !value.is_none()
        {
            props.insert(
                prop_name.to_string(),
                proximadb_records::ProximaTreeNode::Value(json_to_proxima_value(
                    python_to_json_record_value(&value)?,
                )),
            );
        }
    }

    Ok(props)
}

fn python_dict_to_proxima_tree(
    dict: &Bound<'_, PyDict>,
) -> PyResult<proximadb_records::ProximaTree> {
    let mut props = proximadb_records::ProximaTree::new();
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        props.insert(
            key,
            proximadb_records::ProximaTreeNode::Value(json_to_proxima_value(
                python_to_json_record_value(&value)?,
            )),
        );
    }
    Ok(props)
}

fn python_props_list_item(
    props: Option<&Bound<'_, PyList>>,
    index: usize,
) -> PyResult<proximadb_records::ProximaTree> {
    if let Some(prop_list) = props {
        let item = prop_list.get_item(index)?;
        let dict = item.cast::<PyDict>()?;
        python_dict_to_proxima_tree(dict)
    } else {
        Ok(proximadb_records::ProximaTree::new())
    }
}

fn proxima_record_modality(props: &proximadb_records::ProximaTree) -> String {
    props
        .get("_modality")
        .and_then(|node| match node {
            proximadb_records::ProximaTreeNode::Value(
                proximadb_data_model::ProximaValue::String(value),
            ) => Some(value.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "vector".to_string())
}

fn proxima_record_from_batch_parts(
    oid: String,
    values: Vec<f32>,
    props: proximadb_records::ProximaTree,
    now_ns: i64,
) -> proximadb_records::ProximaRecord {
    let dim = values.len() as u32;
    let modality = proxima_record_modality(&props);
    proximadb_records::ProximaRecord {
        oid,
        embeddings: vec![proximadb_records::EmbeddingCell {
            model_id: "default".to_string(),
            modality,
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
}

fn python_to_proxima_record(
    record: &Bound<'_, PyAny>,
    now_ns: i64,
) -> PyResult<proximadb_records::ProximaRecord> {
    let oid_value = python_record_field(record, "id")?
        .or_else(|| python_record_field(record, "oid").ok().flatten())
        .ok_or_else(|| PyValueError::new_err("ProximaRecord is missing 'id'"))?;
    let oid: String = oid_value.extract()?;
    let vector_value = required_python_record_field(record, "vector")?;
    let values = python_record_vector(&vector_value)?;
    if values.is_empty() {
        return Err(PyValueError::new_err(
            "ProximaRecord vector must not be empty",
        ));
    }

    let props = python_record_props(record)?;
    let modality = python_record_field(record, "modality")?
        .and_then(|value| value.extract::<String>().ok())
        .unwrap_or_else(|| proxima_record_modality(&props));
    let dim = values.len() as u32;

    Ok(proximadb_records::ProximaRecord {
        oid,
        embeddings: vec![proximadb_records::EmbeddingCell {
            model_id: "default".to_string(),
            modality,
            dim,
            values: proximadb_records::EmbeddingValues::Fp32(values),
            ..Default::default()
        }],
        props,
        created_at_ns: now_ns,
        updated_at_ns: now_ns,
        record_version: 1,
        ..Default::default()
    })
}

/// Convert a props tree node to serde_json (OWNED — leaves flow into the
/// owned canonical converter without a deep clone).
fn proxima_tree_node_to_json(node: proximadb_records::ProximaTreeNode) -> serde_json::Value {
    match node {
        proximadb_records::ProximaTreeNode::Value(value) => proxima_value_to_json(value),
        proximadb_records::ProximaTreeNode::Object(tree) => serde_json::Value::Object(
            tree.into_iter()
                .map(|(key, value)| (key, proxima_tree_node_to_json(value)))
                .collect(),
        ),
    }
}

fn json_to_python(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(n.to_string().into_pyobject(py)?.into_any().unbind())
            }
        }
        serde_json::Value::String(s) => Ok(s.as_str().into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_python(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (k, v) in obj {
                dict.set_item(k, json_to_python(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

fn sql_result_to_arrow_ipc(
    result: &EmbeddedSqlQueryResult,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    use arrow::array::{ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;

    let fields = result
        .columns
        .iter()
        .enumerate()
        .map(|(idx, column)| {
            Field::new(
                column.clone(),
                sql_column_arrow_type(result.column_types.get(idx).map(String::as_str)),
                true,
            )
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let mut arrays = Vec::with_capacity(result.columns.len());

    for (idx, column) in result.columns.iter().enumerate() {
        let data_type = schema.field(idx).data_type().clone();
        match data_type {
            DataType::Boolean => {
                let mut builder = BooleanBuilder::new();
                for row in &result.rows {
                    match sql_row_value(row, column).and_then(json_bool_value) {
                        Some(value) => builder.append_value(value),
                        None => builder.append_null(),
                    }
                }
                arrays.push(Arc::new(builder.finish()) as ArrayRef);
            }
            DataType::Int64 => {
                let mut builder = Int64Builder::new();
                for row in &result.rows {
                    match sql_row_value(row, column).and_then(json_i64_value) {
                        Some(value) => builder.append_value(value),
                        None => builder.append_null(),
                    }
                }
                arrays.push(Arc::new(builder.finish()) as ArrayRef);
            }
            DataType::Float64 => {
                let mut builder = Float64Builder::new();
                for row in &result.rows {
                    match sql_row_value(row, column).and_then(json_f64_value) {
                        Some(value) => builder.append_value(value),
                        None => builder.append_null(),
                    }
                }
                arrays.push(Arc::new(builder.finish()) as ArrayRef);
            }
            _ => {
                let mut builder = StringBuilder::new();
                for row in &result.rows {
                    match sql_row_value(row, column) {
                        Some(serde_json::Value::String(value)) => builder.append_value(value),
                        Some(serde_json::Value::Null) | None => builder.append_null(),
                        Some(value) => builder.append_value(value.to_string()),
                    }
                }
                arrays.push(Arc::new(builder.finish()) as ArrayRef);
            }
        }
    }

    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let mut bytes = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut bytes, &schema)?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    Ok(bytes)
}

fn notebook_plan_explain(
    db: &EmbeddedProximaDB,
    plan: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let sql = compile_notebook_plan_sql(plan).ok();
    let source_kind = notebook_plan_source_kind(plan)?;
    let source = notebook_plan_source(plan)?;
    let workers = notebook_plan_workers(plan);
    let partition_plan = if source_kind == "table" {
        match db.plan_partitions(source, workers.max(1)) {
            Ok(partitions) => notebook_partitions_to_json(source, workers, &partitions),
            Err(error) => notebook_fallback_partition_plan(
                source,
                workers,
                &format!(
                    "native partition planning rejected this source; using one safe logical partition: {error}"
                ),
            ),
        }
    } else {
        serde_json::json!({
            "requested_partitions": workers.max(1),
            "planned_partitions": 1,
            "effective_read_partitions": 1,
            "planner": "sql_source_fallback",
            "execution_scope": "local_process"
        })
    };
    let effective_parallelism = partition_plan
        .get("effective_read_partitions")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1);

    Ok(serde_json::json!({
        "source_surface": "python_notebook",
        "execution_scope": "local_process",
        "master": notebook_plan_master(plan),
        "workers": workers,
        "memory_limit": notebook_plan_session_value(plan, "memory_limit").cloned().unwrap_or(serde_json::Value::Null),
        "batch_size": notebook_plan_session_value(plan, "batch_size").and_then(serde_json::Value::as_u64).unwrap_or(10_000),
        "authority_mode": "ProximaAuthoritative",
        "policy_boundary": "engine-enforced",
        "compute_route": if sql.is_some() { "DataFusionLocal" } else { "Native" },
        "status": "phase1_rust_plan_boundary",
        "compiled_sql": sql,
        "partition_plan": partition_plan,
        "effective_parallelism": effective_parallelism,
        "unsupported_operations": notebook_plan_unsupported_operations(plan),
        "plan": plan.get("plan").cloned().unwrap_or_else(|| plan.clone()),
    }))
}

fn compile_notebook_plan_sql(plan: &serde_json::Value) -> Result<String, String> {
    let source_kind = notebook_plan_source_kind(plan)?;
    let source = notebook_plan_source(plan)?;
    let operations = notebook_plan_operations(plan)?;

    if source_kind == "sql" {
        if !operations.is_empty() {
            return Err("operations on raw SQL frames are not supported yet".to_string());
        }
        return Ok(source.trim().to_string());
    }
    if source_kind != "table" {
        return Err(format!("unsupported source kind {source_kind:?}"));
    }

    let mut select_columns: Option<Vec<String>> = None;
    let mut predicates = Vec::new();
    let mut limit_count: Option<u64> = None;
    let mut group_columns: Option<Vec<String>> = None;
    let mut aggregate_count = false;

    for op in operations {
        let op_type = op
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "operation is missing string field 'type'".to_string())?;
        match op_type {
            "select" => {
                select_columns = Some(notebook_string_array(op, "columns")?);
            }
            "where" => {
                let predicate = op
                    .get("predicate")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        "where operation is missing string field 'predicate'".to_string()
                    })?;
                predicates.push(predicate.to_string());
            }
            "limit" => {
                limit_count = Some(
                    op.get("count")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or_else(|| {
                            "limit operation is missing numeric field 'count'".to_string()
                        })?,
                );
            }
            "group_count" => {
                group_columns = Some(notebook_string_array(op, "columns")?);
                aggregate_count = true;
            }
            other => return Err(format!("{other} is not SQL-backed in phase 1")),
        }
    }

    let (select_clause, group_clause) = if aggregate_count {
        let group_columns =
            group_columns.ok_or_else(|| "group_count operation is missing columns".to_string())?;
        let projection = group_columns
            .iter()
            .map(|column| notebook_quote_ident(column))
            .collect::<Vec<_>>()
            .join(", ");
        (
            format!("{projection}, COUNT(*) AS count"),
            format!(" GROUP BY {projection}"),
        )
    } else {
        let select_clause = match select_columns {
            Some(columns) if !columns.is_empty() => columns
                .iter()
                .map(|column| notebook_quote_ident(column))
                .collect::<Vec<_>>()
                .join(", "),
            _ => "*".to_string(),
        };
        (select_clause, String::new())
    };

    let mut sql = format!(
        "SELECT {select_clause} FROM {}",
        notebook_table_ident(source)?
    );
    if !predicates.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(
            &predicates
                .iter()
                .map(|predicate| format!("({predicate})"))
                .collect::<Vec<_>>()
                .join(" AND "),
        );
    }
    sql.push_str(&group_clause);
    if let Some(limit_count) = limit_count {
        sql.push_str(&format!(" LIMIT {limit_count}"));
    }
    Ok(sql)
}

fn notebook_plan_root(plan: &serde_json::Value) -> &serde_json::Value {
    plan.get("plan").unwrap_or(plan)
}

fn notebook_plan_source_kind(plan: &serde_json::Value) -> Result<&str, String> {
    notebook_plan_root(plan)
        .get("source_kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "plan is missing string field 'source_kind'".to_string())
}

fn notebook_plan_source(plan: &serde_json::Value) -> Result<&str, String> {
    notebook_plan_root(plan)
        .get("source")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "plan is missing string field 'source'".to_string())
}

fn notebook_plan_operations(plan: &serde_json::Value) -> Result<&[serde_json::Value], String> {
    match notebook_plan_root(plan).get("operations") {
        Some(serde_json::Value::Array(values)) => Ok(values),
        None => Ok(&[]),
        _ => Err("plan field 'operations' must be an array".to_string()),
    }
}

fn notebook_plan_session_value<'a>(
    plan: &'a serde_json::Value,
    key: &str,
) -> Option<&'a serde_json::Value> {
    plan.get("session").and_then(|session| session.get(key))
}

fn notebook_plan_workers(plan: &serde_json::Value) -> u32 {
    notebook_plan_session_value(plan, "workers")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

fn notebook_plan_master(plan: &serde_json::Value) -> serde_json::Value {
    notebook_plan_session_value(plan, "master")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::String("proxima-local[1]".to_string()))
}

fn notebook_plan_unsupported_operations(plan: &serde_json::Value) -> Vec<serde_json::Value> {
    notebook_plan_operations(plan)
        .unwrap_or(&[])
        .iter()
        .filter_map(|op| op.get("type").and_then(serde_json::Value::as_str))
        .filter(|op_type| !matches!(*op_type, "select" | "where" | "limit" | "group_count"))
        .map(|op_type| serde_json::Value::String(op_type.to_string()))
        .collect()
}

fn notebook_string_array(value: &serde_json::Value, field: &str) -> Result<Vec<String>, String> {
    let values = value
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("operation is missing array field '{field}'"))?;
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToString::to_string)
                .ok_or_else(|| format!("field '{field}' must contain only strings"))
        })
        .collect()
}

fn notebook_quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn notebook_table_ident(name: &str) -> Result<String, String> {
    if name.is_empty() {
        return Err("table name must not be empty".to_string());
    }
    let parts = name.split('.').collect::<Vec<_>>();
    if parts.iter().any(|part| part.is_empty()) {
        return Err("table name contains an empty identifier segment".to_string());
    }
    Ok(parts
        .iter()
        .map(|part| {
            if notebook_is_simple_ident(part) {
                (*part).to_string()
            } else {
                notebook_quote_ident(part)
            }
        })
        .collect::<Vec<_>>()
        .join("."))
}

fn notebook_is_simple_ident(value: &str) -> bool {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) if first == '_' || first.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn notebook_partitions_to_json(
    collection: &str,
    requested: u32,
    // `SparkInputPartition` lives in THIS crate's `spark` module, not in the root
    // crate's `connectors` — #1021 moved the bindings here. The stale path only
    // failed to surface because the `#[pymodule] fn proximadb` shadowing above
    // made every `proximadb::…` path in this file fail first.
    partitions: &[crate::spark::SparkInputPartition],
) -> serde_json::Value {
    serde_json::json!({
        "collection": collection,
        "requested_partitions": requested.max(1),
        "planned_partitions": partitions.len(),
        "effective_read_partitions": partitions.len(),
        "planner": "whole_collection_fallback",
        "execution_scope": "local_process",
        "safe_parallelism": partitions.len(),
        "rejected_parallelism_reason": if requested as usize > partitions.len() {
            Some("split-bound readers are not wired yet; returning more collection:// partitions would duplicate scans")
        } else {
            None
        },
        "partitions": partitions.iter().map(notebook_partition_to_json).collect::<Vec<_>>(),
    })
}

fn notebook_partition_to_json(partition: &crate::spark::SparkInputPartition) -> serde_json::Value {
    serde_json::json!({
        "partition_id": partition.partition_id,
        "preferred_locations": &partition.preferred_locations,
        "estimated_rows": partition.estimated_rows,
        "estimated_bytes": partition.estimated_bytes,
        "splits": partition.splits.iter().map(notebook_split_to_json).collect::<Vec<_>>(),
    })
}

fn notebook_split_to_json(split: &proximadb::storage::formats::FileSplit) -> serde_json::Value {
    serde_json::json!({
        "split_id": &split.split_id,
        "file_path": &split.file_path,
        "offset": split.offset,
        "length": split.length,
        "estimated_rows": split.statistics.row_count,
        "estimated_bytes": split.statistics.byte_size,
    })
}

fn notebook_fallback_partition_plan(
    collection: &str,
    requested: u32,
    reason: &str,
) -> serde_json::Value {
    serde_json::json!({
        "collection": collection,
        "requested_partitions": requested.max(1),
        "planned_partitions": 1,
        "effective_read_partitions": 1,
        "planner": "whole_collection_fallback",
        "execution_scope": "local_process",
        "safe_parallelism": 1,
        "rejected_parallelism_reason": reason,
        "partitions": [{
            "partition_id": 0,
            "preferred_locations": [],
            "estimated_rows": null,
            "estimated_bytes": null,
            "splits": [{
                "split_id": format!("collection:{collection}:whole"),
                "file_path": format!("collection://{collection}"),
                "offset": 0,
                "length": 0,
                "estimated_rows": null,
                "estimated_bytes": null,
            }],
        }],
    })
}

fn sql_column_arrow_type(column_type: Option<&str>) -> arrow::datatypes::DataType {
    match column_type
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "bool" | "boolean" => arrow::datatypes::DataType::Boolean,
        "int2" | "int4" | "int8" | "integer" | "bigint" | "smallint" => {
            arrow::datatypes::DataType::Int64
        }
        "float4" | "float8" | "float" | "double" | "real" | "decimal" | "numeric" => {
            arrow::datatypes::DataType::Float64
        }
        _ => arrow::datatypes::DataType::Utf8,
    }
}

fn sql_row_value<'a>(row: &'a serde_json::Value, column: &str) -> Option<&'a serde_json::Value> {
    row.as_object().and_then(|object| object.get(column))
}

fn json_bool_value(value: &serde_json::Value) -> Option<bool> {
    match value {
        serde_json::Value::Bool(value) => Some(*value),
        serde_json::Value::String(value) => value.parse().ok(),
        _ => None,
    }
}

fn json_i64_value(value: &serde_json::Value) -> Option<i64> {
    match value {
        serde_json::Value::Number(value) => value.as_i64(),
        serde_json::Value::String(value) => value.parse().ok(),
        _ => None,
    }
}

fn json_f64_value(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::Number(value) => value.as_f64(),
        serde_json::Value::String(value) => value.parse().ok(),
        _ => None,
    }
}

fn trace_span_to_python(py: Python<'_>, span: super::EmbeddedTraceSpan) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("trace_id", span.trace_id)?;
    dict.set_item("span_id", span.span_id)?;
    dict.set_item("parent_span_id", span.parent_span_id)?;
    dict.set_item("name", span.name)?;
    dict.set_item("kind", span.kind)?;
    dict.set_item("start_time_ns", span.start_time_ns)?;
    dict.set_item("end_time_ns", span.end_time_ns)?;
    dict.set_item("service", span.service)?;
    dict.set_item("status_code", span.status_code)?;
    dict.set_item("status_message", span.status_message)?;

    let attributes = PyDict::new(py);
    for (key, value) in span.attributes {
        attributes.set_item(key, json_to_python(py, &value)?)?;
    }
    dict.set_item("attributes", attributes)?;

    Ok(dict.into_any().unbind())
}

fn inject_graph_target_into_cypher(graph_id: Option<&str>, cypher: &str) -> String {
    let cypher = cypher.trim().trim_end_matches(';').trim();
    let Some(graph_id) = graph_id.map(str::trim).filter(|value| !value.is_empty()) else {
        return cypher.to_string();
    };

    let upper = cypher.to_ascii_uppercase();
    if upper.contains(" FROM ") {
        return cypher.to_string();
    }

    if let Some(return_index) = upper.find(" RETURN ") {
        format!(
            "{} FROM {}{}",
            &cypher[..return_index],
            graph_id,
            &cypher[return_index..]
        )
    } else {
        format!("{cypher} FROM {graph_id}")
    }
}

/// Initialize tracing/logging for the embedded ProximaDB module.
///
/// This function sets up a tracing subscriber that outputs to stderr.
/// Call this at the beginning of your Python script to enable debug logging.
///
/// Args:
///     level: Log level - "error", "warn", "info", "debug", or "trace" (default: "info")
///
/// Example:
///     import proximadb
///     proximadb.init_logging("debug")  # Enable debug logging
///     db = proximadb.ProximaDB("/tmp/test")
#[pyfunction]
#[pyo3(signature = (level = "info"))]
fn init_logging(level: &str) -> PyResult<()> {
    use std::sync::Once;
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    static INIT: Once = Once::new();
    let mut initialized = false;

    INIT.call_once(|| {
        // Build filter from environment or provided level
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

        // Create a simple stderr layer
        let fmt_layer = fmt::layer()
            .with_target(true)
            .with_line_number(false)
            .with_file(false)
            .with_thread_ids(false)
            .with_ansi(true)
            .with_writer(std::io::stderr);

        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .init();

        initialized = true;
    });

    if !initialized {
        eprintln!("Warning: Logging already initialized, ignoring init_logging() call");
    }

    Ok(())
}

fn register_python_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core classes
    m.add_class::<PyProximaDB>()?;
    m.add_class::<PyDiskConfig>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyCollectionInfo>()?;
    m.add_class::<PyStorageStats>()?;

    // Streaming search classes
    m.add_class::<PyStreamingSearchResult>()?;
    m.add_class::<PySearchStreamIterator>()?;

    // Checkpoint and delta classes
    m.add_class::<PyCheckpointInfo>()?;
    m.add_class::<PyDeltaInfo>()?;

    // Graph classes - Victor Framework compatible
    m.add_class::<PyGraphNode>()?;
    m.add_class::<PyGraphEdge>()?;
    m.add_class::<PyGraphStats>()?;

    // Add logging initialization function
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // DataFusion DataFrame API (Proxima-Spark)
    #[cfg(feature = "datafusion-integration")]
    crate::python_dataframe::register_dataframe_module(m)?;

    Ok(())
}

/// Python module definition
/// This exports as "proximadb" which is the module name used by benchmarks
///
/// The Rust function is deliberately NOT named `proximadb`. `#[pymodule]` defines
/// an item in this module's scope, and a function named `proximadb` shadows the
/// `proximadb` *crate* — every `proximadb::…` path in this file then resolves
/// against the pymodule instead and fails (`E0659: ambiguous`, plus one
/// `cannot find X in proximadb` per use site). The `name = "proximadb"` attribute
/// keeps the Python-visible module name identical, so benchmarks are unaffected.
#[pymodule(name = "proximadb")]
fn proximadb_legacy_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_python_module(m)
}

/// Packaged embedded wheel entry point.
#[pymodule(name = "_proximadb_embedded")]
fn proximadb_embedded_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_python_module(m)
}
