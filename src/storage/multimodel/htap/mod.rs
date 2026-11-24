//! # HTAP (Hybrid Transactional/Analytical Processing) Module
//!
//! Provides async replication from OLTP (SST) to OLAP stores,
//! with intelligent workload-aware query routing.
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │                    HTAP Coordinator                           │
//! │  ┌─────────────────────────────────────────────────────────┐ │
//! │  │              Replication Manager                         │ │
//! │  │  - Change Data Capture (CDC) from SST                   │ │
//! │  │  - Async batch replication to the OLAP store            │ │
//! │  │  - LSN tracking for consistency                         │ │
//! │  └─────────────────────────────────────────────────────────┘ │
//! │                                                               │
//! │  ┌─────────────────────────────────────────────────────────┐ │
//! │  │              Query Router                                │ │
//! │  │  - Workload classification (OLTP/OLAP)                  │ │
//! │  │  - Freshness-aware routing                              │ │
//! │  │  - Adaptive learning from query patterns                │ │
//! │  └─────────────────────────────────────────────────────────┘ │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Workload Classification
//!
//! | Characteristic | OLTP Route | OLAP Route |
//! |----------------|------------|------------|
//! | Row count | < 10K | >= 10K |
//! | Has aggregation | No | Yes |
//! | Point lookup | Yes | No |
//! | Full table scan | No | Yes |
//! | GROUP BY | No | Yes |

pub mod replication;
pub mod router;

// Re-exports
pub use replication::{ReplicationConfig, ReplicationCoordinator, ReplicationStats};
pub use router::{QueryCharacteristics, RoutingDecision, WorkloadRouter, WorkloadType};
