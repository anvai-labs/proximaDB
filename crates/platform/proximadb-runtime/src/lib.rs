//! Platform runtime composition helpers.
//!
//! This crate owns host/runtime policy that composes lower contract crates without
//! pushing system inspection, tracing, or bootstrap behavior into foundation crates.

pub mod batch_result;
pub mod bm25_port;
pub mod bootstrap_config;
pub mod cluster;
pub mod cluster_port;
pub mod cluster_rpc;
pub mod composition;
pub mod document_port;
pub mod entity_port;
pub mod graph_port;
pub mod handlers;
pub mod hardware;
pub mod hybrid_port;
pub mod observability_port;
pub mod port;
pub mod proto_defaults;
pub mod record_ops_port;
pub mod record_route_port;
pub mod record_search_port;
pub mod resources;
pub mod rich_record;
pub mod rich_search;
pub mod security_port;
pub mod service_ports;
pub mod streaming_port;
pub mod unified_query_port;

// Re-exports
pub use batch_result::{BatchOperationMetrics, BatchOperationResult, OperationMetrics};
pub use bm25_port::{BM25Document, BM25IndexPort, BM25IndexResult};
pub use cluster_port::{ClusterHealthStatus, ClusterPort};
pub use composition::{DIContainer, ServiceComposer};
pub use document_port::DocumentPort;
pub use entity_port::EntityPort;
pub use graph_port::GraphPort;
pub use handlers::{CollectionIdCache, UnifiedHandlers};
pub use hardware::{HardwareCapabilities, SimdLevel, best_simd_level, hardware_capabilities};
pub use hybrid_port::HybridPort;
pub use observability_port::ObservabilityPort;
pub use port::{
    ApiHandlersPort, CollectionSchemaColumn, CollectionSchemaEnforcement, CollectionSchemaMetadata,
    CollectionSchemaUpdate, CollectionTextStorage,
};
pub use record_ops_port::RecordOpsPort;
pub use record_route_port::{PaxColumnDesc, PaxScanInputs, RecordRoutePort};
pub use record_search_port::RecordSearchPort;
pub use resources::{MemoryBudget, ResourceManager};
pub use rich_record::{RichRecordBatchRequest, RichRecordDeleteBatchRequest, RichRecordGetRequest};
pub use rich_search::{
    RichFilterCondition, RichFilterOperator, RichSearchRequest, RichSearchResponse,
    RichSearchResult,
};
pub use security_port::{PortAuthCredential, PortUserContext, SecurityPort};
pub use service_ports::{
    CollectionPort, OwnedPortIdentity, PortIdentity, QueryAdapterPort, SqlExecutionResult,
    VectorOpsPort,
};
pub use streaming_port::StreamingPort;
pub use unified_query_port::{InvalidQueryInput, UnifiedQueryPort};
