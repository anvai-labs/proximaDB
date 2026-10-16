/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Cluster Module for Distributed ProximaDB
//!
//! This module provides the foundational components for distributed operation:
//! - **Metadata Service**: Cluster-wide metadata management
//! - **Node Registry**: Node discovery and health tracking
//! - **Consensus**: Raft-based distributed consensus
//! - **Routing**: Shard-aware request routing
//! - **Shard**: Data sharding and placement
//! - **RPC**: SOLID-compliant RPC abstractions for inter-node communication
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Cluster Manager                       │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
//! │  │   Metadata   │  │    Node      │  │  Consensus   │  │
//! │  │   Service    │  │   Registry   │  │    (Raft)    │  │
//! │  └──────────────┘  └──────────────┘  └──────────────┘  │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐                    │
//! │  │   Routing    │  │    Shard     │                    │
//! │  │   Service    │  │   Manager    │                    │
//! │  └──────────────┘  └──────────────┘                    │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌──────────────────────────────────────────────────┐  │
//! │  │                 RPC Layer                         │  │
//! │  │  ConsensusTransport | ReplicationSink |          │  │
//! │  │  SearchFanout | HealthChecker                    │  │
//! │  └──────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────┘
//! ```

// Cluster-core subsystem pieces (cache_affinity/metadata_service/node_registry/shard)
// now live in `proximadb-runtime::cluster` (TD-DECOMP-5) — self-contained, no upward
// edge. Re-exported here so `crate::cluster::*` paths (incl. distributed_ops/
// replication/consensus) resolve unchanged.
pub use proximadb_runtime::cluster::{cache_affinity, metadata_service, node_registry, shard};
// `consensus` now lives in `proximadb-runtime::cluster` (TD-DECOMP-6); re-exported so
// `crate::cluster::consensus::*` paths (grpc_server/distributed_ops/replication/external
// callers) resolve unchanged.
pub use proximadb_runtime::cluster::consensus;
pub mod distributed_ops;
/// Generation-fenced per-partition write leases over object storage (Phase 7):
/// a peer holds a durable, fenced lease over one (tenant, collection) for
/// contention-free local writes. Backs [`primary_pod_registry`]'s in-memory
/// routing with a split-brain-safe durable authority.
pub mod partition_lease;
/// Write-side affinity for (tenant_id, collection_id) → primary_pod
/// bindings. Complementary to [`cache_affinity`]: durable, authoritative,
/// drives write routing. See module doc for the read/write split.
pub mod primary_pod_registry;
pub mod replication;
pub mod routing;
/// ADR-095 D2: object-store writer-epoch log fencing a reattachable WAL volume.
pub mod writer_epoch;

// RPC abstraction layer for inter-node communication
// Provides SOLID-compliant traits for consensus, replication, search fanout, and health
pub mod rpc;

use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;

pub use cache_affinity::{AffinityEntry, CacheAffinityRegistry};
pub use consensus::{ConsensusConfig, ConsensusState, RaftConsensus};
/// Backwards-compat alias for the per-shard SearchResult re-exported from distributed_ops.
pub use distributed_ops::ShardSearchResult as SearchResult;
pub use distributed_ops::{
    ConsistencyLevel, DistributedCollectionOps, DistributedOpsConfig, DistributedSearchRequest,
    DistributedSearchResult, DistributedWriteRequest, DistributedWriteResult, QueryContext,
    RetryConfig, ShardSearchResult, WriteRecord,
};
pub use metadata_service::{ClusterMetadata, MetadataService, MetadataServiceConfig};
pub use node_registry::{
    NodeHealth, NodeInfo, NodeRegistry, NodeRegistryConfig, NodeRole, NodeStatus,
};
pub use partition_lease::{
    LeaseOutcome, PartitionLease, PartitionLeaseManager, PartitionLeaseStore,
};
pub use primary_pod_registry::{
    AssignmentReason, PrimaryPod, PrimaryPodRegistry, WriteRoutingDecision, consult_for_write,
    consult_for_write_leased, resolve_self_pod_id,
};
pub use replication::{
    EngineReplication, ReplicaState, ReplicationAck, ReplicationConfig, ReplicationEntry,
    ReplicationHealth, ReplicationOperation, ReplicationRetryConfig, ReplicationStatsSummary,
};
pub use routing::{RouteContext, RouteDecision, RoutingConfig, RoutingService};
pub use shard::{
    MetadataBounds, PartitionConfig, PartitionStrategy, Shard, ShardConfig, ShardId, ShardManager,
    ShardPlacement, ShardState,
};

// HMGI distributed partitioning surfaces used by cluster-aware AXIS routing.
#[cfg(feature = "axis")]
pub use crate::index::axis::hmgi::{
    ClusterMembership as HmgiClusterMembership, ClusterNode as HmgiClusterNode,
    ClusterNodeId as HmgiClusterNodeId, DistributedPartitionLocator as HmgiPartitionLocator,
    HmgiQueryCoordinator, HmgiRouteStats, HmgiSearchRequest, NetworkService as HmgiNetworkService,
    NodeState as HmgiNodeState,
};

// Re-export RPC abstractions for inter-node communication
pub use rpc::{
    // Traits
    ConsensusTransport,
    HealthChecker,
    NodeClient,
    // Common types
    NodeEndpoint,
    ReplicationSink,
    RpcError,
    RpcResult,
    SearchFanout,
};

// Config consolidated into proximadb-config (TD-107, seam S4); re-exported
// so existing `crate::cluster::...` import paths keep resolving.
pub use proximadb_config::cluster_config::ClusterConfig;

/// Main cluster manager that coordinates all distributed components
pub struct ClusterManager {
    config: ClusterConfig,
    metadata_service: Arc<MetadataService>,
    node_registry: Arc<NodeRegistry>,
    consensus: Arc<RwLock<RaftConsensus>>,
    routing_service: Arc<RoutingService>,
    shard_manager: Arc<ShardManager>,
    is_leader: Arc<AtomicBool>,
}

impl ClusterManager {
    /// Create a new cluster manager
    pub async fn new(config: ClusterConfig) -> Result<Self> {
        let metadata_service = Arc::new(MetadataService::new(config.metadata.clone())?);
        let node_registry = Arc::new(NodeRegistry::new(config.node_registry.clone())?);
        let consensus = Arc::new(RwLock::new(RaftConsensus::new(config.consensus.clone())?));
        let routing_service = Arc::new(RoutingService::new(config.routing.clone())?);
        let shard_manager = Arc::new(ShardManager::new(config.shard.clone())?);

        Ok(Self {
            config,
            metadata_service,
            node_registry,
            consensus,
            routing_service,
            shard_manager,
            is_leader: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Start the cluster manager and all components
    pub async fn start(&self) -> Result<()> {
        tracing::info!(
            cluster_id = %self.config.cluster_id,
            node_id = %self.config.node_id,
            "Starting cluster manager"
        );

        // Register this node
        let node_info = NodeInfo {
            node_id: self.config.node_id.clone(),
            address: self.config.advertise_addr.clone(),
            role: NodeRole::Follower,
            status: NodeStatus::Starting,
            health: NodeHealth::Unknown,
            ..Default::default()
        };
        self.node_registry.register_node(node_info).await?;

        // Start consensus
        {
            let mut consensus = self.consensus.write().await;
            consensus.start().await?;
        }

        // Update node status
        self.node_registry
            .update_status(&self.config.node_id, NodeStatus::Running)
            .await?;

        tracing::info!("Cluster manager started successfully");
        Ok(())
    }

    /// Stop the cluster manager gracefully
    pub async fn stop(&self) -> Result<()> {
        tracing::info!("Stopping cluster manager");

        // Update node status
        self.node_registry
            .update_status(&self.config.node_id, NodeStatus::Stopping)
            .await?;

        // Stop consensus
        {
            let mut consensus = self.consensus.write().await;
            consensus.stop().await?;
        }

        // Deregister this node
        self.node_registry
            .deregister_node(&self.config.node_id)
            .await?;

        tracing::info!("Cluster manager stopped");
        Ok(())
    }

    /// Check if this node is the current leader
    pub async fn is_leader(&self) -> bool {
        self.is_leader.load(Ordering::Acquire)
    }

    /// Get the metadata service
    pub fn metadata_service(&self) -> &Arc<MetadataService> {
        &self.metadata_service
    }

    /// Get the node registry
    pub fn node_registry(&self) -> &Arc<NodeRegistry> {
        &self.node_registry
    }

    /// Get the routing service
    pub fn routing_service(&self) -> &Arc<RoutingService> {
        &self.routing_service
    }

    /// Get the shard manager
    pub fn shard_manager(&self) -> &Arc<ShardManager> {
        &self.shard_manager
    }

    /// Get cluster health summary
    pub async fn health(&self) -> ClusterHealth {
        let nodes = self.node_registry.list_nodes().await;
        let healthy_nodes = nodes
            .iter()
            .filter(|n| n.health == NodeHealth::Healthy)
            .count();
        let total_nodes = nodes.len();

        ClusterHealth {
            cluster_id: self.config.cluster_id.clone(),
            is_leader: self.is_leader().await,
            total_nodes,
            healthy_nodes,
            unhealthy_nodes: total_nodes - healthy_nodes,
            shard_count: self.shard_manager.shard_count().await,
        }
    }
}

/// Cluster health summary
#[derive(Debug, Clone)]
pub struct ClusterHealth {
    /// Unique identifier for this cluster
    pub cluster_id: String,
    /// Whether this node is the current Raft leader
    pub is_leader: bool,
    /// Total number of nodes registered in the cluster
    pub total_nodes: usize,
    /// Number of nodes currently passing health checks
    pub healthy_nodes: usize,
    /// Number of nodes failing health checks or unreachable
    pub unhealthy_nodes: usize,
    /// Total number of shards distributed across the cluster
    pub shard_count: usize,
}

// ── ClusterPort ───────────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl proximadb_runtime::ClusterPort for ClusterManager {
    async fn is_leader(&self) -> bool {
        ClusterManager::is_leader(self).await
    }

    async fn health(&self) -> proximadb_runtime::ClusterHealthStatus {
        let h = ClusterManager::health(self).await;
        proximadb_runtime::ClusterHealthStatus {
            cluster_id: h.cluster_id,
            is_leader: h.is_leader,
            total_nodes: h.total_nodes,
            healthy_nodes: h.healthy_nodes,
            unhealthy_nodes: h.unhealthy_nodes,
            shard_count: h.shard_count,
        }
    }

    async fn start(&self) -> anyhow::Result<()> {
        ClusterManager::start(self).await
    }

    async fn stop(&self) -> anyhow::Result<()> {
        ClusterManager::stop(self).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cluster_config_default() {
        let config = ClusterConfig::default();
        assert_eq!(config.cluster_id, "proximadb-cluster");
        assert!(!config.node_id.is_empty());
    }

    #[tokio::test]
    async fn test_cluster_manager_creation() {
        let config = ClusterConfig::default();
        let manager = ClusterManager::new(config).await;
        assert!(manager.is_ok());
    }
}
