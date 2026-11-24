//! # Multi-Model Storage Module
//!
//! This module provides a unified storage facade that orchestrates multiple specialized
//! storage engines for different data models: Vector, Document, Graph, RDBMS, and Observability.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │               MultiModelStorageFacade                            │
//! │  Unified entry point for all multi-model storage operations     │
//! └─────────────────────────────────────────────────────────────────┘
//!                               │
//!         ┌─────────────────────┼─────────────────────┐
//!         ▼                     ▼                     ▼
//! ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
//! │  VectorStore  │    │  GraphStore   │    │  RDBMSStore   │
//! │  (Helix+SST)  │    │   (Orion)     │    │  (SST+column) │
//! └───────────────┘    └───────────────┘    └───────────────┘
//!         │                     │                     │
//! ┌───────────────────────────────────────────────────────────────┐
//! │              SHARED INFRASTRUCTURE                             │
//! │  UnifiedCacheOrchestrator | Quantization | WAL | Catalog      │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Design Philosophy
//!
//! **Dedicated Storage + Unified Query**: Each data model has its own optimized storage
//! engine, but all are accessed through a single unified interface.
//!
//! ## Storage Engine Mapping
//!
//! | Model Type | Primary Engine | Secondary Engine | Rationale |
//! |------------|----------------|------------------|-----------|
//! | Vector | SST | HELIX | Real-time writes with optional high-dimensional routing |
//! | Document | DocumentService (SST-backed) | Parquet/Tantivy integration in progress | JSON CRUD with explicit hot/cold tiering still being hardened |
//! | Graph | ORION | - | Native CSR format |
//! | RDBMS | SST (OLTP) | columnar (OLAP) | HTAP separation |
//! | Observability | ObservabilityService | Tantivy | WAL-backed logs/metrics/traces with indexed log search |

pub mod facade;
pub mod htap;
pub mod observability;
pub mod stores;
pub mod traits;
pub mod transaction;

// Re-exports
pub use facade::MultiModelStorageFacade;
pub use htap::{
    QueryCharacteristics, ReplicationConfig, ReplicationCoordinator, ReplicationStats,
    RoutingDecision, WorkloadRouter, WorkloadType,
};
pub use observability::{
    AggregationFunction, CardinalityConfig, CardinalityLimiter, CheckResult, LabelStats,
    LimitAction, Partition, PartitionConfig, PartitionGranularity, PartitionRange, RollupConfig,
    RollupInterval, RollupManager, RollupView, TimePartitioner,
};
pub use stores::{DocumentStore, GraphStore, ObservabilityStore, RDBMSStore, VectorStore};
pub use traits::{ModelType, MultiModelStorageEngine, StoreCapabilities};
pub use transaction::{
    CommitResult, IsolationLevel, IsolationManager, PrepareResult, ReadSnapshot, Transaction,
    TransactionConfig, TransactionCoordinator, TransactionState, TransactionStats,
    TwoPhaseCommitProtocol, WriteSet,
};
