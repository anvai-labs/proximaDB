//! # DataSource Connectors Module
//!
//! This module provides Spark DataSource V2-style connector interfaces for external system
//! integration with ProximaDB. It enables Hadoop-style storage-compute separation by defining
//! standardized traits for data readers, writers, and pushdown optimization.
//!
//! ## Overview
//!
//! The connectors module implements a plugin architecture for integrating ProximaDB with
//! external data processing systems like Apache Spark, Apache Flink, or other query engines.
//! This enables ProximaDB to act as both a data source and data sink in larger data pipelines.
//!
//! ## Key Components
//!
//! - **DataSourceConnector**: Main trait for table discovery and reader/writer creation
//! - **DataReader**: Streaming interface for reading Arrow RecordBatches from tables
//! - **DataWriter**: Transactional interface for writing Arrow RecordBatches to tables
//! - **Pushdown Protocol**: Negotiation protocol for predicate and projection pushdown
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        External Query Engine                             │
//! │                    (Spark, Flink, Trino, etc.)                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//!                                     │
//!                        ┌────────────┼────────────┐
//!                        ▼            ▼            ▼
//!              ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
//!              │ list_tables │ │ get_table   │ │ negotiate_  │
//!              │             │ │             │ │ pushdown    │
//!              └─────────────┘ └─────────────┘ └─────────────┘
//!                        │            │            │
//!                        └────────────┼────────────┘
//!                                     ▼
//!                        ┌─────────────────────────┐
//!                        │   DataSourceConnector   │
//!                        └─────────────────────────┘
//!                           │                   │
//!                           ▼                   ▼
//!               ┌─────────────────┐   ┌─────────────────┐
//!               │   DataReader    │   │   DataWriter    │
//!               │  (streaming)    │   │ (transactional) │
//!               └─────────────────┘   └─────────────────┘
//!                           │                   │
//!                           ▼                   ▼
//!               ┌─────────────────────────────────────┐
//!               │     ProximaDB Storage Engines       │
//!               │    (SST, HELIX, etc.)                │
//!               └─────────────────────────────────────┘
//! ```
//!
//! ## Pushdown Optimization
//!
//! The connector supports aggressive pushdown of operations to ProximaDB:
//!
//! - **Filter Pushdown**: SQL predicates are pushed down to storage engines
//! - **Projection Pushdown**: Only requested columns are read
//! - **Aggregate Pushdown**: Aggregations can be computed at the storage layer
//! - **Vector Search Pushdown**: KNN queries are executed natively
//! - **Graph Traversal Pushdown**: Graph queries are executed by the ORION runtime
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use proximadb::connectors::{DataSourceConnector, ReadContext, WriteContext};
//!
//! // Connect to ProximaDB as a data source
//! let connector = ProximaDBConnector::new(config);
//!
//! // Discover available tables
//! let tables = connector.list_tables().await?;
//!
//! // Create a reader with filter pushdown
//! let ctx = ReadContext::new()
//!     .with_projection(vec!["id", "embedding", "metadata"])
//!     .with_filter("category = 'science'");
//!
//! let mut reader = connector.create_reader("vectors", &ctx)?;
//!
//! // Stream data in Arrow batches
//! while let Some(batch) = reader.next_batch().await? {
//!     process_batch(batch);
//! }
//! ```
//!
//! ## Integration Points
//!
//! - **Storage Layer**: Connectors delegate to `UnifiedStorageFormat` implementations
//! - **Index Layer**: AXIS engine handles vector search pushdown
//! - **Graph Layer**: ORION handles graph traversal pushdown
//! - **WAL System**: Writers integrate with WAL for transactional guarantees

pub use proximadb_pushdown::pushdown;
pub mod traits;
pub use proximadb_connectors_types::types;

// Compute engine connectors
pub mod duckdb;
pub mod trino;

// Legacy Hadoop compatibility
pub mod hadoop;

// Re-export main types for convenient access
pub use pushdown::{
    AggExpr, Expr, GraphTraversalPushdown, PushdownRequest, PushdownResponse, VectorSearchPushdown,
};
pub use traits::{DataReader, DataSourceConnector, DataWriter, ReadContext, WriteContext};
pub use types::{ColumnStatistics, Statistics, TableInfo, TableStatistics, WriteResult};

pub use duckdb::{
    DuckDBBindData, DuckDBColumnRef, DuckDBColumnStats, DuckDBConnectorConfig, DuckDBCopy,
    DuckDBCopyResult, DuckDBError, DuckDBErrorType, DuckDBFilter, DuckDBFilterType,
    DuckDBGlobalState, DuckDBInitData, DuckDBInsert, DuckDBInsertResult, DuckDBLocalState,
    DuckDBScanStatistics, DuckDBTableScan, DuckDBVectorSearch, DuckDBVectorSearchParams,
    DuckDBWriteMode,
};
pub use hadoop::{
    HadoopError, HadoopErrorCode, HadoopInputSplit, HadoopShimConfig, HadoopWritable, HiveType,
    ProximaInputFormat, ProximaOutputCommitter, ProximaOutputFormat, ProximaRecordReader,
    ProximaRecordWriter, ProximaSerDe,
};
pub use trino::{
    TrinoBlock, TrinoColumnMetadata, TrinoConnectorConfig, TrinoConnectorSession, TrinoDomain,
    TrinoError, TrinoErrorCode, TrinoHostAddress, TrinoPage, TrinoPageSink, TrinoPageSource,
    TrinoRange, TrinoSchema, TrinoSplit, TrinoSplitManager, TrinoTable, TrinoTableLayout,
    TrinoTupleDomain, TrinoWriteSummary,
};
