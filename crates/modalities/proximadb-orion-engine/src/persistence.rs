/*
 * Copyright 2025 Vijaykumar Singh
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

//! # ORION Persistence Module
//!
//! Provides persistence capabilities for the ORION graph engine using:
//! - Snapshots with compression
//! - Write-Ahead Logging (WAL)
//! - Cloud storage integration via IntelligentFilesystem
//!
//! ## Error type note
//!
//! This module returns `proximadb_kernel::error::ProximaDBError`. Storage
//! failures are wrapped via `ProximaDBError::Storage(kernel::StorageError)`.
//! The 2026-05-20 proliferation audit (P2) flagged the ~30 kernel
//! `StorageError` constructions here as a "migration candidate" toward the
//! richer `proximadb_storage_common::StorageError`, but a deeper review on
//! 2026-05-26 found this is not a clean migration: `ProximaDBError::Storage`
//! is defined to wrap kernel `StorageError` specifically, and the `From`
//! bridge is one-directional (kernel → storage_common, not the reverse).
//!
//! Migrating would require either:
//! - Adding a new `ProximaDBError::StorageCommon(...)` variant (introduces
//!   a third error path, not consolidation), or
//! - Migrating every caller of `ProximaDBError::Storage` to the richer
//!   type (massive blast radius across the codebase).
//!
//! Neither is consolidation in the reuse-first sense. Leave kernel
//! `StorageError` here until a broader VectorDBError shape decision is made.

use crate::OrionGraphEngine;
use proximadb_compression_types::CompressionAlgorithm;
use proximadb_graph_model::{Edge, EdgeId, Node, NodeId};
use proximadb_kernel::error::ProximaDBError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

type Result<T> = std::result::Result<T, ProximaDBError>;

/// Whether the cold-payload tier is enabled. Mirrors `graph::service::cold_payloads_enabled`
/// (reads `PROXIMADB_GRAPH_COLD_PAYLOADS`) — inlined here so this module has no root
/// `crate::graph::service` dependency (ORION crate extraction).
fn cold_payloads_enabled() -> bool {
    std::env::var("PROXIMADB_GRAPH_COLD_PAYLOADS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Serializable snapshot of the ORION engine state
#[derive(Serialize, Deserialize)]
pub struct OrionSnapshot {
    /// Version for backward compatibility
    version: u32,

    /// All nodes in the graph
    nodes: Vec<Node>,

    /// All edges in the graph
    edges: Vec<Edge>,

    /// CSR format for outgoing edges
    csr_outgoing_offsets: Vec<usize>,
    csr_outgoing_targets: Vec<usize>,

    /// CSR format for incoming edges
    csr_incoming_offsets: Vec<usize>,
    csr_incoming_sources: Vec<usize>,

    /// Node to CSR index mapping
    node_to_index: HashMap<NodeId, usize>,

    /// Timestamp of snapshot creation
    timestamp: i64,

    /// TD-066 (c) Part 2: the canonical checkpoint LSN (edge-epoch) this
    /// snapshot covers. Recovery finds the matching `CanonicalEmission` marker
    /// in the engine WAL and replays only frames after it (those frames have
    /// edge-epoch > this LSN, so they are NOT already in the snapshot → no
    /// over-replay). v1 snapshots (written before this field existed; see
    /// [`OrionSnapshotLegacy`]) decode with `0`, which recovery treats as
    /// "uncorrelated → full replay."
    checkpoint_lsn: u64,
}

/// Legacy v1 on-disk snapshot shape (no `checkpoint_lsn`). Used only to
/// mixed-read-safely decode snapshots written before TD-066 Part 2 — the v2
/// [`OrionSnapshot`] deserializer fails on v1 bytes (missing trailing field),
/// so `load_snapshot` falls back to this shape and maps `checkpoint_lsn = 0`.
#[derive(Serialize, Deserialize)]
struct OrionSnapshotLegacy {
    version: u32,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    csr_outgoing_offsets: Vec<usize>,
    csr_outgoing_targets: Vec<usize>,
    csr_incoming_offsets: Vec<usize>,
    csr_incoming_sources: Vec<usize>,
    node_to_index: HashMap<NodeId, usize>,
    timestamp: i64,
}

// Use the unified GraphOperation from graph_memtable
use proximadb_graph_model::GraphOperation;

/// Persistence manager for ORION engine
pub struct OrionPersistence {
    /// Graph ID for multi-graph support
    graph_id: String,

    /// TD-066 (c) Part 2: number of graph-op frames applied by the most recent
    /// `replay_wal` call. Observable (`last_replay_applied`) so recovery-scoping
    /// tests can prove only post-checkpoint frames were replayed (and so
    /// operators/metrics can see the truncation working).
    last_replay_applied: std::sync::atomic::AtomicU64,

    /// TD-066 (d): number of engine-WAL segments reclaimed by the most recent
    /// `truncate_wal_through_checkpoint` call. Observable
    /// (`last_truncate_reclaimed`) so size-bounding/crash-safety tests can prove
    /// segments were actually reclaimed (and operators can see truncation work).
    last_truncate_reclaimed: std::sync::atomic::AtomicU64,

    /// True while `replay_wal` is applying recovered frames through the engine's
    /// public insert/delete paths. Those paths unconditionally append to the WAL,
    /// so without this gate every restart re-logged the whole history — doubling
    /// the WAL per restart and, worse, poisoning it with duplicate `CreateEdge`
    /// frames whose replay aborted the NEXT recovery on the CSR's duplicate-edge
    /// check (#1524). All `write_*_operation` methods no-op while this is set.
    replaying: std::sync::atomic::AtomicBool,

    /// Base URL for storage (e.g., "file:///data", "s3://bucket", etc.)
    base_url: String,

    /// Filesystem port for snapshot I/O (read/list), injected via the
    /// `GraphWalFactory` so the engine never names the concrete filesystem
    /// factory.
    filesystem_factory: Arc<dyn proximadb_storage_ports::FilesystemPort>,

    /// WAL path for future implementation
    wal_path: Option<PathBuf>,

    /// Optional path to the shared canonical WAL
    /// (`<data_dir>/pgwire/canonical-records.wal`) held by
    /// `SharedServices`. Used by [`Self::canonical_checkpoint_lsn`] for
    /// read-side observability of TD-066 checkpoint emission on
    /// recovery — `None` falls back to today's engine-WAL-only
    /// recovery behavior. Behavior of `replay_wal` is unchanged either
    /// way (Part 1 of TD-066 (c) is read-side observability; behavior
    /// changes are the Part 2 follow-up slice).
    canonical_wal_path: Option<PathBuf>,

    /// Canonical-WAL reader (TD-066 checkpoint-LSN correlation), obtained
    /// through the injected `GraphWalFactory` so the engine never names the
    /// concrete appender. Only used when `canonical_wal_path` is `Some`.
    canonical_wal_reader: Arc<dyn proximadb_storage_ports::CanonicalWalReaderPort>,

    /// WAL sink for graph operations. The engine appends through the
    /// [`GraphWalPort`] dependency-inversion port (injected by the composition
    /// root as the unified WAL writer), so it never names the concrete writer
    /// or the unified operation type — a prerequisite for extracting ORION into
    /// its own crate without a cyclic dependency on the root storage layer.
    wal_writer: Option<Arc<tokio::sync::Mutex<dyn proximadb_storage_ports::GraphWalPort>>>,

    /// WAL reader for recovery. The engine replays through the
    /// [`GraphWalReaderPort`] dependency-inversion port (injected by the
    /// composition root as the unified WAL reader), so it never names the
    /// concrete reader or the unified entry/operation types — the read-side
    /// counterpart to the writer port.
    wal_reader: Option<Arc<dyn proximadb_storage_ports::GraphWalReaderPort>>,

    /// Compression configuration
    compression: CompressionAlgorithm,
    compression_level: i32,

    /// Snapshot configuration
    max_snapshots: usize,
    incremental_snapshots: bool,
}

impl std::fmt::Debug for OrionPersistence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrionPersistence")
            .field("graph_id", &self.graph_id)
            .field("base_url", &self.base_url)
            .field("wal_path", &self.wal_path)
            .field("wal_writer", &"<dyn GraphWalPort>")
            .field("wal_reader", &"<dyn GraphWalReaderPort>")
            .field("compression", &self.compression)
            .field("compression_level", &self.compression_level)
            .field("max_snapshots", &self.max_snapshots)
            .field("incremental_snapshots", &self.incremental_snapshots)
            .finish()
    }
}

impl OrionPersistence {
    fn read_lock<'a, T>(
        lock: &'a std::sync::RwLock<T>,
        lock_name: &str,
    ) -> Result<std::sync::RwLockReadGuard<'a, T>> {
        lock.read()
            .map_err(|_| ProximaDBError::Internal(format!("{lock_name} read lock poisoned")))
    }

    fn write_lock<'a, T>(
        lock: &'a std::sync::RwLock<T>,
        lock_name: &str,
    ) -> Result<std::sync::RwLockWriteGuard<'a, T>> {
        lock.write()
            .map_err(|_| ProximaDBError::Internal(format!("{lock_name} write lock poisoned")))
    }

    /// Create a new persistence manager for a specific graph.
    ///
    /// The constructor does not take a canonical WAL path — that's set
    /// post-construction via [`Self::with_canonical_wal_path`] so the
    /// 24+ existing test/bench callers don't need updating. Production
    /// wiring (`src/graph/service_engine_factory.rs` →
    /// `OrionGraphEngine::with_persistence_for_graph`) reaches into
    /// the underlying persistence and sets the path via the builder.
    pub async fn new(
        graph_id: String,
        base_url: String,
        enable_wal: bool,
        wal_factory: Arc<dyn proximadb_storage_ports::GraphWalFactory>,
    ) -> Result<Self> {
        // Obtain the filesystem port (for snapshot I/O) through the injected
        // GraphWalFactory — the engine never names the concrete filesystem
        // factory.
        let filesystem_factory = wal_factory.make_filesystem().await.map_err(|e| {
            ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                e.to_string(),
            ))
        })?;
        let canonical_wal_reader = wal_factory.make_canonical_wal_reader().await.map_err(|e| {
            ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                e.to_string(),
            ))
        })?;

        // Build graph-specific path: {base_url}/graphs/{graph_id}/data
        let graph_path = format!(
            "{}/graphs/{}/data",
            base_url.trim_end_matches('/'),
            graph_id
        );

        // Create graph directory
        filesystem_factory
            .create_dir_all(&graph_path)
            .await
            .map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::SstEngine(
                    e.to_string(),
                ))
            })?;

        // Store WAL path and initialize WAL writer
        let (wal_path, wal_writer, wal_reader) = if enable_wal {
            // Build WAL URL (keep as URL for filesystem operations)
            let wal_url = format!("{}/wal", graph_path);

            // Extract path component for WAL writer (strip file:// prefix if present)
            let wal_path_str = if wal_url.starts_with("file://") {
                wal_url
                    .strip_prefix("file://")
                    .map_or_else(|| wal_url.clone(), |s| s.to_string())
            } else {
                wal_url.clone()
            };
            let wal_path = PathBuf::from(&wal_path_str);

            // Create WAL directory using URL
            filesystem_factory
                .create_dir_all(&wal_url)
                .await
                .map_err(|e| {
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::SstEngine(
                        e.to_string(),
                    ))
                })?;

            // Obtain the WAL writer + reader through the injected GraphWalFactory
            // port — the engine never names the concrete UnifiedWAL* types (the
            // factory is the single composition-root seam that does). Both are
            // tolerant of an absent/empty WAL (the reader opens no files until a
            // read; recovery is a no-op before the first write).
            tracing::debug!("Creating WAL writer with path: {}", wal_path_str);
            let wal_writer = wal_factory.make_writer(&wal_path_str).await.map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                    e.to_string(),
                ))
            })?;
            let wal_reader = wal_factory.make_reader(&wal_path_str).await.map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                    e.to_string(),
                ))
            })?;
            tracing::debug!("WAL writer + reader created successfully");

            (Some(wal_path), Some(wal_writer), Some(wal_reader))
        } else {
            (None, None, None)
        };

        Ok(Self {
            graph_id,
            base_url,
            filesystem_factory,
            wal_path,
            canonical_wal_path: None,
            canonical_wal_reader,
            wal_writer,
            wal_reader,
            compression: CompressionAlgorithm::Zstd,
            compression_level: 3,
            max_snapshots: 10,
            incremental_snapshots: false,
            last_replay_applied: std::sync::atomic::AtomicU64::new(0),
            last_truncate_reclaimed: std::sync::atomic::AtomicU64::new(0),
            replaying: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// True while `replay_wal` is re-applying recovered frames (see the
    /// `replaying` field). WAL appends are suppressed for the duration so
    /// recovery never re-logs what it reads.
    fn is_replaying(&self) -> bool {
        self.replaying.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Inject the path to the shared canonical WAL
    /// (`<data_dir>/pgwire/canonical-records.wal`) for read-side
    /// observability on recovery (TD-066 (c) Part 1). Production wiring
    /// derives this from `SharedServices.canonical_wal_appender.path()`.
    /// Test/bench paths can leave this unset; recovery falls back to
    /// engine-WAL-only behavior.
    pub fn with_canonical_wal_path(mut self, path: PathBuf) -> Self {
        self.canonical_wal_path = Some(path);
        self
    }

    /// The graph this persistence manager is bound to (used by
    /// `canonical_checkpoint_lsn` and recovery observability).
    pub fn graph_id(&self) -> &str {
        &self.graph_id
    }

    /// Scan the shared canonical WAL for the latest
    /// `CanonicalOperation::Checkpoint(SnapshotManifest)` entry whose
    /// `manifest.collection_ids` contains this persistence manager's
    /// `graph_id`, and return the manifest's `sequence_number` (the
    /// engine-side checkpoint LSN at the time of emission).
    ///
    /// Returns `None` when:
    /// - `canonical_wal_path` is `None` (recovery falls back to
    ///   engine-WAL-only behavior), OR
    /// - the canonical WAL file doesn't exist yet (fresh deployment),
    ///   OR
    /// - no Checkpoint entry for this `graph_id` has been persisted.
    ///
    /// This is the **read-side** half of TD-066 (c) Part 1:
    /// observability only — `replay_wal` doesn't use this value yet.
    /// Per ADR-020 the canonical WAL is the durability authority; this
    /// method makes that authority visible to recovery so operators
    /// can confirm "did my graph checkpoint actually land durably?"
    /// before the follow-up slice changes recovery behavior to use the
    /// LSN as a replay bound.
    pub async fn canonical_checkpoint_lsn(&self) -> Option<u64> {
        let path = self.canonical_wal_path.as_ref()?;
        if !tokio::fs::try_exists(path).await.unwrap_or(false) {
            return None;
        }
        let entries = match self.canonical_wal_reader.read_entries(path).await {
            Ok(entries) => entries,
            Err(err) => {
                tracing::warn!(
                    graph_id = %self.graph_id,
                    canonical_wal_path = %path.display(),
                    error = %err,
                    "ORION canonical_checkpoint_lsn: failed to read canonical WAL; \
                     returning None and falling back to engine-WAL-only recovery"
                );
                return None;
            }
        };

        let graph_id = self.graph_id.as_str();
        entries
            .iter()
            .filter_map(|entry| match &entry.operation {
                proximadb_storage_common::CanonicalOperation::Checkpoint(manifest)
                    if manifest.collection_ids.iter().any(|id| id == graph_id) =>
                {
                    Some(manifest.sequence_number)
                }
                _ => None,
            })
            .max()
    }

    /// Companion to [`Self::canonical_checkpoint_lsn`] that also returns
    /// the manifest's `timestamp_ms` for the same matching Checkpoint.
    /// `OrionGraphEngine::recover` uses this to feed the
    /// `orion_recovery_canonical_checkpoint_age_seconds` gauge per the
    /// TD-066 (c) Part 2 design Option E (`docs/12-design/TD_066_PART2_LSN_CORRELATION_DESIGN_2026_05_28.adoc`).
    ///
    /// Returns `None` under the same conditions as
    /// `canonical_checkpoint_lsn`; returns `Some((lsn, timestamp_ms))`
    /// for the Checkpoint with the maximum `sequence_number` whose
    /// `collection_ids` contains this graph.
    pub async fn canonical_checkpoint_with_timestamp(&self) -> Option<(u64, u64)> {
        let path = self.canonical_wal_path.as_ref()?;
        if !tokio::fs::try_exists(path).await.unwrap_or(false) {
            return None;
        }
        let entries = match self.canonical_wal_reader.read_entries(path).await {
            Ok(entries) => entries,
            Err(err) => {
                tracing::warn!(
                    graph_id = %self.graph_id,
                    canonical_wal_path = %path.display(),
                    error = %err,
                    "ORION canonical_checkpoint_with_timestamp: failed to read canonical WAL; \
                     returning None and falling back to engine-WAL-only recovery"
                );
                return None;
            }
        };

        let graph_id = self.graph_id.as_str();
        entries
            .iter()
            .filter_map(|entry| match &entry.operation {
                proximadb_storage_common::CanonicalOperation::Checkpoint(manifest)
                    if manifest.collection_ids.iter().any(|id| id == graph_id) =>
                {
                    Some((manifest.sequence_number, manifest.timestamp_ms))
                }
                _ => None,
            })
            .max_by_key(|(lsn, _)| *lsn)
    }

    /// Save a snapshot of the engine state
    pub async fn save_snapshot(
        &self,
        engine: &OrionGraphEngine,
        checkpoint_lsn: u64,
    ) -> Result<PathBuf> {
        info!("Creating ORION snapshot");

        // TD-168 Phase 1b: when the cold-payload tier is ON, write a **topology-only**
        // (v3) snapshot — CSR + node_to_index only, NO node/edge payloads — so a graph
        // whose payloads exceed RAM need not materialize every payload to snapshot, and
        // cold-start loads topology only (payloads are served lazily from the canonical
        // record store via the service cold-fetch path). When OFF (default), the full
        // v2 snapshot is written exactly as before. The same `OrionSnapshot` struct
        // carries both; a v3 is simply a v2 with empty `nodes`/`edges` + `version = 3`.
        let topology_only = cold_payloads_enabled();

        // Collect node/edge payloads ONLY for the full (v2) snapshot. The clones are the
        // expensive part a huge graph wants to avoid, so they are skipped entirely for v3.
        let nodes: Vec<Node> = if topology_only {
            Vec::new()
        } else {
            engine
                .memory_pool
                .nodes
                .iter()
                .map(|entry| entry.value().as_ref().clone())
                .collect()
        };
        let edges: Vec<Edge> = if topology_only {
            Vec::new()
        } else {
            engine
                .edge_metadata
                .iter()
                .map(|entry| entry.value().as_ref().clone())
                .collect()
        };

        // Build node_to_index mapping
        let node_to_index = engine
            .node_to_index
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect::<HashMap<_, _>>();

        // Clone data needed for snapshot - guards are dropped when block ends
        let (
            csr_outgoing_offsets,
            csr_outgoing_targets,
            csr_incoming_offsets,
            csr_incoming_sources,
        ) = {
            let csr_outgoing = Self::read_lock(&engine.csr_outgoing, "CSR outgoing")?;
            let csr_incoming = Self::read_lock(&engine.csr_incoming, "CSR incoming")?;
            let csr_outgoing_offsets = csr_outgoing.offsets.clone();
            let csr_outgoing_targets = csr_outgoing.targets.clone();
            let csr_incoming_offsets = csr_incoming.offsets.clone();
            let csr_incoming_sources = csr_incoming.targets.clone();
            // Guards dropped here when block ends
            Ok::<(Vec<_>, Vec<_>, Vec<_>, Vec<_>), ProximaDBError>((
                csr_outgoing_offsets,
                csr_outgoing_targets,
                csr_incoming_offsets,
                csr_incoming_sources,
            ))
        }?;

        // Create snapshot. Same v2 shape either way (nothing is released, so no new
        // on-disk version is warranted): a topology-only snapshot is simply a v2 with
        // empty `nodes`/`edges`. Load detects it from that shape — a full snapshot
        // always has `nodes.len() == node_to_index.len()`, so empty payloads + non-empty
        // topology is unambiguously topology-only.
        let snapshot = OrionSnapshot {
            version: 2,
            nodes,
            edges,
            csr_outgoing_offsets,
            csr_outgoing_targets,
            csr_incoming_offsets,
            csr_incoming_sources, // Note: 'targets' stores sources for incoming
            node_to_index,
            timestamp: chrono::Utc::now().timestamp(),
            checkpoint_lsn,
        };

        // Serialize snapshot
        let serialized = bincode::serialize(&snapshot).map_err(|e| {
            ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                e.to_string(),
            ))
        })?;

        // Apply compression based on algorithm
        let compressed = self.compress_data(&serialized)?;

        // Generate snapshot filename with graph ID
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("graph_{}_snapshot_{}.bin.zst", self.graph_id, timestamp);
        let snapshot_url = format!(
            "{}/graphs/{}/snapshots/{}",
            self.base_url.trim_end_matches('/'),
            self.graph_id,
            filename
        );

        // Ensure snapshots directory exists
        let snapshots_dir = format!(
            "{}/graphs/{}/snapshots",
            self.base_url.trim_end_matches('/'),
            self.graph_id
        );
        self.filesystem_factory
            .create_dir_all(&snapshots_dir)
            .await
            .map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::SstEngine(
                    e.to_string(),
                ))
            })?;

        // Write compressed snapshot
        self.filesystem_factory
            .write(&snapshot_url, &compressed, None)
            .await
            .map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::SstEngine(
                    e.to_string(),
                ))
            })?;

        info!(
            "ORION graph {} snapshot saved: {} ({}MB compressed)",
            self.graph_id,
            filename,
            compressed.len() / 1_048_576
        );

        // Clean up old snapshots if needed
        self.cleanup_old_snapshots().await?;

        Ok(PathBuf::from(snapshot_url))
    }

    /// Load a snapshot and restore engine state
    pub async fn load_snapshot(
        &self,
        engine: &OrionGraphEngine,
        snapshot_path: impl AsRef<Path>,
    ) -> Result<u64> {
        info!("Loading ORION snapshot from {:?}", snapshot_path.as_ref());

        // Read compressed snapshot
        let path_str = snapshot_path.as_ref().to_str().ok_or_else(|| {
            ProximaDBError::InvalidInput("Snapshot path contains invalid UTF-8".to_string())
        })?;
        let compressed = self.filesystem_factory.read(path_str).await.map_err(|e| {
            ProximaDBError::Storage(proximadb_kernel::error::StorageError::SstEngine(
                e.to_string(),
            ))
        })?;

        // Decompress data
        let decompressed = self.decompress_data(&compressed)?;

        // Deserialize. Mixed-read-safe: v2 snapshots carry `checkpoint_lsn`;
        // v1 (legacy) snapshots lack the field and fail to deserialize as the
        // v2 struct, so fall back to the legacy shape with checkpoint_lsn = 0
        // (recovery treats 0 as "uncorrelated → full replay").
        let snapshot: OrionSnapshot = match bincode::deserialize::<OrionSnapshot>(&decompressed) {
            Ok(s) => s,
            Err(_) => {
                let legacy =
                    bincode::deserialize::<OrionSnapshotLegacy>(&decompressed).map_err(|e| {
                        ProximaDBError::Storage(
                            proximadb_kernel::error::StorageError::Serialization(e.to_string()),
                        )
                    })?;
                OrionSnapshot {
                    version: legacy.version,
                    nodes: legacy.nodes,
                    edges: legacy.edges,
                    csr_outgoing_offsets: legacy.csr_outgoing_offsets,
                    csr_outgoing_targets: legacy.csr_outgoing_targets,
                    csr_incoming_offsets: legacy.csr_incoming_offsets,
                    csr_incoming_sources: legacy.csr_incoming_sources,
                    node_to_index: legacy.node_to_index,
                    timestamp: legacy.timestamp,
                    checkpoint_lsn: 0,
                }
            }
        };
        // Capture the watermark before the restore loops move out of `snapshot`.
        let checkpoint_lsn = snapshot.checkpoint_lsn;
        // A topology-only (TD-168) snapshot is a v2 snapshot written with the cold-payload
        // gate ON: it carries the full topology but EMPTY payloads. A full snapshot always
        // has `nodes.len() == node_to_index.len()`, so empty payloads alongside a non-empty
        // topology is unambiguous — no on-disk version bump needed.
        let topology_only_snapshot =
            snapshot.nodes.is_empty() && !snapshot.node_to_index.is_empty();

        // Mixed-read-safety (fail-closed): a topology-only snapshot's reads depend on the
        // cold-fetch path, which is the SAME `PROXIMADB_GRAPH_COLD_PAYLOADS` gate. Loading
        // one with the gate OFF would leave the engine with topology but no payloads AND no
        // cold-fetch, so get_node/get_edge would silently return None (apparent data loss).
        // Refuse the load loudly instead, so flipping the gate off after enabling it yields
        // a clear error, not silent corruption.
        if topology_only_snapshot && !cold_payloads_enabled() {
            return Err(ProximaDBError::InvalidInput(format!(
                "graph '{}': snapshot is topology-only (empty payloads) but the cold-payload \
                 tier (PROXIMADB_GRAPH_COLD_PAYLOADS) is OFF — enable it to load this snapshot, \
                 otherwise node/edge payloads are unreachable",
                self.graph_id
            )));
        }

        // Clear existing data
        engine.memory_pool.nodes.clear();
        engine.memory_pool.edges.clear();
        engine.memory_pool.edge_composite_index.clear();
        engine.edge_metadata.clear();
        engine.node_to_index.clear();

        // Restore nodes
        for node in snapshot.nodes {
            engine
                .memory_pool
                .nodes
                .insert(node.id.clone(), Arc::new(node));
        }

        // Restore edges into BOTH stores: `edge_metadata` (id lookup) AND the
        // memory pool — `get_all_edges`/the composite-uniqueness index read the
        // pool, so restoring only metadata left every full-snapshot load
        // edge-blind on the read path (the #1549 read-blindness bug, snapshot
        // edition; latent while snapshot loading was gated off).
        for edge in snapshot.edges {
            let edge_id = edge.id.clone();
            let edge_arc = engine.memory_pool.insert_edge(edge);
            engine.edge_metadata.insert(edge_id, edge_arc);
        }

        // Restore CSR structures
        {
            let mut csr_out = Self::write_lock(&engine.csr_outgoing, "CSR outgoing")?;
            csr_out.offsets = snapshot.csr_outgoing_offsets;
            csr_out.targets = snapshot.csr_outgoing_targets;
        }

        {
            let mut csr_in = Self::write_lock(&engine.csr_incoming, "CSR incoming")?;
            csr_in.offsets = snapshot.csr_incoming_offsets;
            csr_in.targets = snapshot.csr_incoming_sources;
        }

        // Restore mappings
        for (node_id, index) in snapshot.node_to_index {
            engine.node_to_index.insert(node_id.clone(), index);
        }

        // Rebuild index_to_node
        {
            let mut index_to_node = Self::write_lock(&engine.index_to_node, "index_to_node")?;
            index_to_node.clear();
            let mut node_indices: Vec<(NodeId, usize)> = engine
                .node_to_index
                .iter()
                .map(|entry| (entry.key().clone(), *entry.value()))
                .collect();
            node_indices.sort_by_key(|&(_, idx)| idx);

            for (node_id, _) in node_indices {
                index_to_node.push(node_id);
            }
        }

        // Counts come from the TOPOLOGY (node_to_index + CSR), not the payload maps:
        // for a topology-only (v3) load the payload maps are empty by design, so reading
        // their lengths would report 0. node_to_index has every node; the outgoing CSR
        // `targets` has one entry per directed edge — both correct for v2 and v3.
        let node_count = engine.node_to_index.len() as u64;
        let edge_count = {
            let csr_out = Self::read_lock(&engine.csr_outgoing, "CSR outgoing")?;
            csr_out.targets.len() as u64
        };

        // Update stats
        {
            let mut stats = engine
                .stats
                .write()
                .map_err(|_| ProximaDBError::Internal("stats write lock poisoned".to_string()))?;
            stats.nodes_created = node_count;
            stats.edges_created = edge_count;
        }

        info!(
            "ORION snapshot loaded ({}): {} nodes, {} edges{}",
            if topology_only_snapshot {
                "topology-only / cold payloads"
            } else {
                "full payloads"
            },
            node_count,
            edge_count,
            if topology_only_snapshot {
                " (payloads served lazily from the record store)"
            } else {
                ""
            }
        );

        Ok(checkpoint_lsn)
    }

    /// TD-066 (c) Part 2: discover the newest on-disk snapshot for this graph,
    /// if any. Snapshot files are named `graph_{id}_snapshot_{YYYYMMDD_HHMMSS}.bin.zst`,
    /// so a lexical sort by URL yields chronological order. Returns `None` when
    /// the snapshots directory is absent or empty (recovery then falls back to a
    /// full engine-WAL replay).
    pub async fn latest_snapshot_path(&self) -> Result<Option<String>> {
        let snapshots_dir = format!(
            "{}/graphs/{}/snapshots",
            self.base_url.trim_end_matches('/'),
            self.graph_id
        );
        let entries = match self.filesystem_factory.list(&snapshots_dir).await {
            Ok(e) => e,
            Err(_) => return Ok(None),
        };
        let mut paths: Vec<String> = entries
            .into_iter()
            .map(|e| e.url)
            .filter(|u| u.ends_with(".bin.zst"))
            .collect();
        paths.sort();
        Ok(paths.into_iter().last())
    }

    /// TD-066 (c) Part 2: number of graph-op frames the most recent
    /// `replay_wal` applied (0 when no WAL is configured, or when scoping
    /// skipped everything). Lets tests and observers confirm the truncation.
    pub fn last_replay_applied(&self) -> u64 {
        self.last_replay_applied
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// TD-066 (c) Part 2: append a non-data `CanonicalEmission(lsn)` marker
    /// frame to the engine WAL. Called by `GraphOperationsService::flush_wal`
    /// right after the canonical checkpoint at `lsn` is persisted, so recovery
    /// can truncate engine-WAL replay at the latest marker whose LSN ≤ the
    /// recovered canonical checkpoint LSN (frames at/before it are already
    /// covered by the durable canonical snapshot). No-op when no WAL writer is
    /// configured.
    pub async fn append_canonical_emission_marker(&self, lsn: u64) -> Result<()> {
        if let Some(ref wal_writer) = self.wal_writer {
            use proximadb_graph_model::MarkerKind;

            wal_writer
                .lock()
                .await
                .append_graph_marker(MarkerKind::CanonicalEmission(lsn))
                .await
                .map_err(|e| {
                    tracing::error!(
                        lsn, error = ?e,
                        "canonical-emission marker WAL append failed"
                    );
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;
            tracing::debug!(lsn, "canonical-emission marker appended to engine WAL");
        }
        Ok(())
    }

    /// TD-066 (d): number of engine-WAL segments reclaimed by the most recent
    /// `truncate_wal_through_checkpoint` call. Lets tests and observers confirm
    /// the WAL is actually being bounded.
    pub fn last_truncate_reclaimed(&self) -> u64 {
        self.last_truncate_reclaimed
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// TD-066 (d): after a durable snapshot at `checkpoint_lsn`, reclaim engine
    /// WAL segments fully covered by it (every segment whose frames all precede
    /// the `CanonicalEmission(checkpoint_lsn)` marker).
    ///
    /// CRASH-SAFETY CONTRACT: callers MUST invoke this only AFTER
    /// `save_snapshot` returns, so the deleted frames are already covered by a
    /// durable snapshot. The underlying primitive deletes lowest-segment-first,
    /// so a crash mid-truncation still leaves a recoverable contiguous suffix.
    /// No-op (returns 0) when no WAL writer is configured or the marker is
    /// absent.
    pub async fn truncate_wal_through_checkpoint(&self, checkpoint_lsn: u64) -> Result<u64> {
        if let Some(ref wal_writer) = self.wal_writer {
            let reclaimed = wal_writer
                .lock()
                .await
                .truncate_through_canonical_marker(checkpoint_lsn)
                .await
                .map_err(|e| {
                    tracing::error!(
                        checkpoint_lsn, error = ?e,
                        "engine WAL truncation through checkpoint failed"
                    );
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;
            self.last_truncate_reclaimed
                .store(reclaimed, std::sync::atomic::Ordering::Relaxed);
            if reclaimed > 0 {
                tracing::debug!(
                    checkpoint_lsn,
                    reclaimed,
                    "TD-066 (d): engine WAL truncated to checkpoint LSN"
                );
            }
            return Ok(reclaimed);
        }
        Ok(0)
    }

    /// Write node operation to WAL
    pub async fn write_node_operation(&self, node: Node) -> Result<()> {
        if self.is_replaying() {
            return Ok(());
        }
        if let Some(ref wal_writer) = self.wal_writer {
            tracing::debug!("Writing node operation to WAL for node: {}", node.id);

            let graph_op = GraphOperation::CreateNode {
                graph_id: self.graph_id.clone(),
                node,
            };

            wal_writer
                .lock()
                .await
                .append_graph_op(graph_op)
                .await
                .map_err(|e| {
                    tracing::error!("WAL append failed: {:?}", e);
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;

            tracing::debug!("WAL write completed successfully");
        } else {
            tracing::warn!("No WAL writer available - operations will not be persisted!");
        }

        Ok(())
    }

    /// Write edge operation to WAL
    pub async fn write_edge_operation(&self, edge: Edge) -> Result<()> {
        if self.is_replaying() {
            return Ok(());
        }
        if let Some(ref wal_writer) = self.wal_writer {
            let graph_op = GraphOperation::CreateEdge {
                graph_id: self.graph_id.clone(),
                edge,
            };

            wal_writer
                .lock()
                .await
                .append_graph_op(graph_op)
                .await
                .map_err(|e| {
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;
        }

        Ok(())
    }

    /// Write a batch of edge operations to WAL in a single record
    pub async fn write_edge_batch_operation(&self, edges: &[Edge]) -> Result<()> {
        if self.is_replaying() {
            return Ok(());
        }
        if edges.is_empty() {
            return Ok(());
        }

        if let Some(ref wal_writer) = self.wal_writer {
            // Wrap all CreateEdge operations in a single batch GraphOp to reduce WAL traffic
            let batch = GraphOperation::BatchOperation {
                operations: edges
                    .iter()
                    .cloned()
                    .map(|edge| GraphOperation::CreateEdge {
                        graph_id: self.graph_id.clone(),
                        edge,
                    })
                    .collect(),
            };

            wal_writer
                .lock()
                .await
                .append_graph_op(batch)
                .await
                .map_err(|e| {
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;
        }

        Ok(())
    }

    /// Write a batch of node operations to WAL in a single record
    pub async fn write_node_batch_operation(&self, nodes: &[Node]) -> Result<()> {
        if self.is_replaying() {
            return Ok(());
        }
        if nodes.is_empty() {
            return Ok(());
        }

        if let Some(ref wal_writer) = self.wal_writer {
            // Wrap all CreateNode operations in a single batch GraphOp to reduce WAL traffic
            let batch = GraphOperation::BatchOperation {
                operations: nodes
                    .iter()
                    .cloned()
                    .map(|node| GraphOperation::CreateNode {
                        graph_id: self.graph_id.clone(),
                        node,
                    })
                    .collect(),
            };

            wal_writer
                .lock()
                .await
                .append_graph_op(batch)
                .await
                .map_err(|e| {
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;
        }

        Ok(())
    }

    /// Write node update operation to WAL
    /// For updates, we write the full updated node (upsert semantic)
    pub async fn write_update_node_operation(&self, node: Node) -> Result<()> {
        if self.is_replaying() {
            return Ok(());
        }
        if let Some(ref wal_writer) = self.wal_writer {
            tracing::debug!("Writing update node operation to WAL for node: {}", node.id);

            // Use CreateNode with upsert semantic - during recovery, this will overwrite existing node
            let graph_op = GraphOperation::CreateNode {
                graph_id: self.graph_id.clone(),
                node,
            };

            wal_writer
                .lock()
                .await
                .append_graph_op(graph_op)
                .await
                .map_err(|e| {
                    tracing::error!("WAL append failed for update node: {:?}", e);
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;

            tracing::debug!("Update node WAL write completed");
        } else {
            tracing::warn!(
                "No WAL writer available - update node operation will not be persisted!"
            );
        }

        Ok(())
    }

    /// Write node delete operation to WAL
    pub async fn write_delete_node_operation(&self, node_id: &NodeId) -> Result<()> {
        if self.is_replaying() {
            return Ok(());
        }
        if let Some(ref wal_writer) = self.wal_writer {
            tracing::debug!("Writing delete node operation to WAL for node: {}", node_id);

            let graph_op = GraphOperation::DeleteNode {
                graph_id: self.graph_id.clone(),
                node_id: node_id.clone(),
            };

            wal_writer
                .lock()
                .await
                .append_graph_op(graph_op)
                .await
                .map_err(|e| {
                    tracing::error!("WAL append failed for delete node: {:?}", e);
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;

            tracing::debug!("Delete node WAL write completed");
        } else {
            tracing::warn!(
                "No WAL writer available - delete node operation will not be persisted!"
            );
        }

        Ok(())
    }

    /// Write edge update operation to WAL
    /// For updates, we write the full updated edge (upsert semantic)
    pub async fn write_update_edge_operation(&self, edge: Edge) -> Result<()> {
        if self.is_replaying() {
            return Ok(());
        }
        if let Some(ref wal_writer) = self.wal_writer {
            tracing::debug!("Writing update edge operation to WAL for edge: {}", edge.id);

            // Use CreateEdge with upsert semantic - during recovery, this will overwrite existing edge
            let graph_op = GraphOperation::CreateEdge {
                graph_id: self.graph_id.clone(),
                edge,
            };

            wal_writer
                .lock()
                .await
                .append_graph_op(graph_op)
                .await
                .map_err(|e| {
                    tracing::error!("WAL append failed for update edge: {:?}", e);
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;

            tracing::debug!("Update edge WAL write completed");
        } else {
            tracing::warn!(
                "No WAL writer available - update edge operation will not be persisted!"
            );
        }

        Ok(())
    }

    /// Write edge delete operation to WAL
    pub async fn write_delete_edge_operation(&self, edge_id: &EdgeId) -> Result<()> {
        if self.is_replaying() {
            return Ok(());
        }
        if let Some(ref wal_writer) = self.wal_writer {
            tracing::debug!("Writing delete edge operation to WAL for edge: {}", edge_id);

            let graph_op = GraphOperation::DeleteEdge {
                graph_id: self.graph_id.clone(),
                edge_id: edge_id.clone(),
            };

            wal_writer
                .lock()
                .await
                .append_graph_op(graph_op)
                .await
                .map_err(|e| {
                    tracing::error!("WAL append failed for delete edge: {:?}", e);
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;

            tracing::debug!("Delete edge WAL write completed");
        } else {
            tracing::warn!(
                "No WAL writer available - delete edge operation will not be persisted!"
            );
        }

        Ok(())
    }

    /// Log a generic graph operation to WAL
    pub async fn log_operation(&self, op: GraphOperation) -> Result<()> {
        if let Some(ref wal_writer) = self.wal_writer {
            wal_writer
                .lock()
                .await
                .append_graph_op(op)
                .await
                .map_err(|e| {
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;
        }

        Ok(())
    }

    /// Flush WAL buffer to disk
    /// This ensures all pending operations are persisted
    pub async fn flush_wal(&self) -> Result<()> {
        if let Some(ref wal_writer) = self.wal_writer {
            wal_writer.lock().await.flush().await.map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                    e.to_string(),
                ))
            })?;
            tracing::debug!("WAL flushed to disk for graph: {}", self.graph_id);
        }
        Ok(())
    }

    /// Replay WAL operations from all segments.
    ///
    /// `snapshot_lsn`: when recovery has loaded a snapshot tagged with canonical
    /// checkpoint LSN X (TD-066 (c) Part 2), pass `Some(X)` to scope replay to
    /// frames after the matching `CanonicalEmission(X)` marker (those frames are
    /// NOT in the snapshot → no over-replay). Pass `None` for full replay.
    pub async fn replay_wal(
        &self,
        engine: &OrionGraphEngine,
        snapshot_lsn: Option<u64>,
    ) -> Result<()> {
        if let Some(ref wal_reader) = self.wal_reader {
            use proximadb_graph_model::{GraphWalRecord, MarkerKind};

            tracing::debug!("Attempting WAL recovery for graph {}", self.graph_id);

            let entries = wal_reader.read_all_graph().await.map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                    e.to_string(),
                ))
            })?;

            tracing::info!(
                "Replaying {} WAL entries for graph {}",
                entries.len(),
                self.graph_id
            );

            // TD-066 (c) Part 2: if recovery loaded a snapshot tagged with
            // checkpoint_lsn X (`snapshot_lsn = Some(X)`), find the matching
            // `CanonicalEmission(X)` marker and replay only frames AFTER it —
            // those frames have edge-epoch > X, so they are NOT in the snapshot,
            // so there is no over-replay (and therefore no need for replay
            // idempotency). The marker-before-snapshot write order in `flush_wal`
            // guarantees "snapshot(X) exists ⟹ marker(X) is durable". Fallback
            // (no snapshot / snapshot_lsn == 0 [v1, uncorrelated] / marker not
            // found in the WAL) is full replay — today's behavior.
            let mut start_index = 0usize;
            if let Some(snapshot_lsn) = snapshot_lsn
                && snapshot_lsn > 0
            {
                let mut marker_idx: Option<usize> = None;
                for (i, entry) in entries.iter().enumerate() {
                    if let GraphWalRecord::Marker(MarkerKind::CanonicalEmission(lsn)) =
                        &entry.record
                        && *lsn == snapshot_lsn
                    {
                        marker_idx = Some(i);
                    }
                }
                if let Some(idx) = marker_idx {
                    start_index = idx + 1;
                    tracing::info!(
                        graph_id = %self.graph_id,
                        snapshot_lsn,
                        marker_index = idx,
                        total_entries = entries.len(),
                        replay_from = start_index,
                        "TD-066 Part 2: truncating engine-WAL replay at snapshot's canonical-emission marker"
                    );
                } else {
                    tracing::warn!(
                        graph_id = %self.graph_id,
                        snapshot_lsn,
                        "snapshot loaded but its canonical-emission marker not found in engine WAL; full replay"
                    );
                }
            }

            // Replay drives the engine's public insert/delete paths, which
            // append to this same WAL. Suppress those appends for the duration
            // — otherwise every restart re-logs the whole history (#1524).
            // Cleared before every return path below; replay runs during
            // startup recovery, before the engine serves traffic.
            self.replaying
                .store(true, std::sync::atomic::Ordering::Release);

            // Replay applies frames through `create_edge`, which rebuilt BOTH
            // CSRs per edge. A rebuild is O(V+E), so recovering E edges cost
            // O(E^2): measured insert throughput inside one replay decayed
            // 2,526/s -> ~60/s as the graph filled, and a 177k-edge graph never
            // finished reopening (#1673). Replay has the whole frame set up
            // front and serves no reads until it returns, so it accumulates
            // into the temp buffer and rebuilds exactly once below.
            engine.set_csr_rebuild_deferred(true);

            let mut replayed: u64 = 0;
            let mut skipped_failed: u64 = 0;
            for entry in entries.into_iter().skip(start_index) {
                // `entries` is already projected to graph records; apply the
                // data ops and skip the canonical-sync markers (they carry no
                // engine state to reapply).
                if let GraphWalRecord::Op(graph_op) = entry.record {
                    // Apply per-entry, warn-and-continue on failure: aborting
                    // the whole replay on one bad frame leaves the graph EMPTY
                    // (strictly worse than a partial recovery), and WALs
                    // written before the re-append fix above legitimately
                    // contain duplicate CreateEdge frames that must not poison
                    // recovery.
                    match self.apply_graph_operation(engine, *graph_op).await {
                        Ok(()) => replayed += 1,
                        Err(e) => {
                            skipped_failed += 1;
                            tracing::warn!(
                                graph_id = %self.graph_id,
                                error = %e,
                                "WAL replay: frame failed to apply; continuing with remaining frames"
                            );
                        }
                    }
                }
            }
            // Clear both flags before the rebuild so an error cannot leave the
            // engine stuck in replay mode, then commit the accumulated edits.
            // Until this lands the deferred edges are invisible to traversal,
            // so it must happen before recovery returns — propagate a failure
            // rather than serve a silently incomplete graph.
            engine.set_csr_rebuild_deferred(false);
            self.replaying
                .store(false, std::sync::atomic::Ordering::Release);
            engine.rebuild_csr()?;
            tracing::info!(
                graph_id = %self.graph_id,
                skipped = start_index,
                replayed,
                skipped_failed,
                "WAL replay completed for graph"
            );
            self.last_replay_applied
                .store(replayed, std::sync::atomic::Ordering::Relaxed);
        }

        Ok(())
    }

    /// Apply a graph operation to the engine during WAL replay
    fn apply_graph_operation<'a>(
        &'a self,
        engine: &'a OrionGraphEngine,
        op: GraphOperation,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            match op {
                GraphOperation::CreateNode { graph_id: _, node } => {
                    engine.create_node(node).await?;
                }
                GraphOperation::CreateEdge { graph_id: _, edge } => {
                    // Idempotent: WALs written before replay suppressed its own
                    // re-appends carry duplicate CreateEdge frames; re-applying
                    // one would trip the CSR's duplicate-edge check and (before
                    // the per-frame tolerance in `replay_wal`) abort recovery.
                    if engine.memory_pool.get_edge(&edge.id).is_some() {
                        debug!(
                            "WAL replay: edge {} already applied; skipping duplicate frame",
                            edge.id
                        );
                    } else {
                        engine.create_edge(edge).await?;
                    }
                }
                GraphOperation::UpdateNode {
                    graph_id: _,
                    node_id,
                    update,
                } => {
                    // Get existing node from memory pool and apply update
                    if let Some(existing_node) = engine.memory_pool.get_node(&node_id) {
                        let mut updated_node = (*existing_node).clone();

                        // Apply updates
                        if let Some(labels) = update.labels {
                            updated_node.labels = labels;
                        }
                        if let Some(properties) = update.properties {
                            updated_node.properties = properties;
                        }
                        if let Some(embedding) = update.embedding {
                            updated_node.embedding = Some(embedding);
                        }

                        // Update the node (without re-logging to WAL during replay)
                        engine.memory_pool.remove_node(&node_id);
                        engine.memory_pool.insert_node(updated_node);
                        debug!("Replayed UpdateNode for node_id: {}", node_id);
                    } else {
                        warn!("UpdateNode replay: node {} not found, skipping", node_id);
                    }
                }
                GraphOperation::DeleteNode {
                    graph_id: _,
                    node_id,
                } => {
                    engine.delete_node(&node_id).await?;
                }
                GraphOperation::DeleteEdge {
                    graph_id: _,
                    edge_id,
                } => {
                    engine.delete_edge(&edge_id).await?;
                }
                GraphOperation::CreateEdgeIndex {
                    graph_id: _,
                    index_config: _,
                } => {
                    // Index operations need to be implemented in OrionGraphEngine
                    warn!("Create edge index operation not yet implemented in ORION engine");
                }
                GraphOperation::DropEdgeIndex {
                    graph_id: _,
                    index_name: _,
                } => {
                    // Index operations need to be implemented in OrionGraphEngine
                    warn!("Drop edge index operation not yet implemented in ORION engine");
                }
                GraphOperation::UpdateEdge {
                    graph_id: _,
                    edge_id,
                    update,
                } => {
                    // Get existing edge from memory pool and apply update
                    if let Some(existing_edge) = engine.memory_pool.get_edge(&edge_id) {
                        let mut updated_edge = (*existing_edge).clone();

                        // Apply updates
                        if let Some(properties) = update.properties {
                            updated_edge.properties = properties;
                        }
                        if let Some(weight) = update.weight {
                            updated_edge.weight = Some(weight);
                        }

                        // Update the edge (without re-logging to WAL during replay)
                        engine.memory_pool.remove_edge(&edge_id);
                        engine.memory_pool.insert_edge(updated_edge);
                        debug!("Replayed UpdateEdge for edge_id: {}", edge_id);
                    } else {
                        warn!("UpdateEdge replay: edge {} not found, skipping", edge_id);
                    }
                }
                GraphOperation::BatchOperation { operations } => {
                    // Apply each operation in the batch
                    for op in operations {
                        self.apply_graph_operation(engine, op).await?;
                    }
                }
            }

            Ok(())
        })
    }

    /// Create a checkpoint (snapshot + truncate WAL)
    ///
    /// Saves a full snapshot, then truncates the WAL directory so that
    /// replay after a restart starts from this checkpoint.
    pub async fn checkpoint(&self, engine: &OrionGraphEngine) -> Result<PathBuf> {
        let snapshot_path = self.save_snapshot(engine, 0).await?;

        // Truncate WAL: remove all segment files so replay is a no-op after
        // a clean checkpoint.  The next write re-creates segment files.
        if let Some(ref wal_path) = self.wal_path
            && wal_path.exists()
        {
            match std::fs::read_dir(wal_path) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_file()
                            && let Err(e) = std::fs::remove_file(&path)
                        {
                            tracing::warn!("Failed to truncate WAL segment {:?}: {}", path, e);
                        }
                    }
                    info!("WAL truncated after checkpoint for graph {}", self.graph_id);
                }
                Err(e) => {
                    tracing::warn!("Failed to read WAL directory for truncation: {}", e);
                }
            }
        }

        Ok(snapshot_path)
    }

    /// Compress data using the configured algorithm
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use lz4_flex::compress_prepend_size;
        use snap::raw::Encoder as SnapEncoder;
        use zstd::encode_all;

        match self.compression {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Zstd => encode_all(data, self.compression_level).map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                    e.to_string(),
                ))
            }),
            CompressionAlgorithm::Lz4 => Ok(compress_prepend_size(data)),
            CompressionAlgorithm::Snappy => {
                let mut encoder = SnapEncoder::new();
                encoder.compress_vec(data).map_err(|e| {
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })
            }
            _ => {
                // For other algorithms, default to Zstd
                encode_all(data, self.compression_level).map_err(|e| {
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })
            }
        }
    }

    /// Decompress data based on the configured algorithm
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use lz4_flex::decompress_size_prepended;
        use snap::raw::Decoder as SnapDecoder;
        use zstd::decode_all;

        match self.compression {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Zstd => decode_all(data).map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                    e.to_string(),
                ))
            }),
            CompressionAlgorithm::Lz4 => decompress_size_prepended(data).map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                    e.to_string(),
                ))
            }),
            CompressionAlgorithm::Snappy => {
                let mut decoder = SnapDecoder::new();
                decoder.decompress_vec(data).map_err(|e| {
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })
            }
            _ => {
                // For other algorithms, default to Zstd
                decode_all(data).map_err(|e| {
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })
            }
        }
    }

    /// Clean up old snapshots keeping only max_snapshots
    async fn cleanup_old_snapshots(&self) -> Result<()> {
        let snapshots_dir = format!(
            "{}/graphs/{}/snapshots",
            self.base_url.trim_end_matches('/'),
            self.graph_id
        );

        // List all snapshot files
        let mut snapshots = Vec::new();
        let entries = self
            .filesystem_factory
            .list(&snapshots_dir)
            .await
            .map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::SstEngine(
                    e.to_string(),
                ))
            })?;

        for entry in entries {
            if entry.url.ends_with(".bin.zst") {
                snapshots.push(entry.url);
            }
        }

        // Sort by filename (which includes timestamp)
        snapshots.sort();

        // Remove old snapshots if we exceed max_snapshots
        if snapshots.len() > self.max_snapshots {
            let to_remove = snapshots.len() - self.max_snapshots;
            for snapshot in snapshots.iter().take(to_remove) {
                self.filesystem_factory
                    .delete(snapshot)
                    .await
                    .map_err(|e| {
                        ProximaDBError::Storage(proximadb_kernel::error::StorageError::SstEngine(
                            e.to_string(),
                        ))
                    })?;
                debug!("Removed old snapshot: {:?}", snapshot);
            }
        }

        Ok(())
    }

    /// Export graph to a portable format
    pub async fn export(&self, engine: &OrionGraphEngine, path: impl AsRef<Path>) -> Result<()> {
        // Create snapshot
        let nodes: Vec<Node> = engine
            .memory_pool
            .nodes
            .iter()
            .map(|entry| (**entry.value()).clone())
            .collect();

        let edges: Vec<Edge> = engine
            .edge_metadata
            .iter()
            .map(|entry| (**entry.value()).clone())
            .collect();

        // Export as JSON for portability
        let export_data = serde_json::json!({
            "version": 1,
            "engine": "ORION",
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": nodes.len(),
                "edge_count": edges.len(),
                "timestamp": chrono::Utc::now().timestamp(),
            }
        });

        let json = serde_json::to_string_pretty(&export_data).map_err(|e| {
            ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                e.to_string(),
            ))
        })?;

        let export_url = path
            .as_ref()
            .to_str()
            .ok_or_else(|| ProximaDBError::InvalidInput("Invalid export path".to_string()))?;
        self.filesystem_factory
            .write(export_url, json.as_bytes(), None)
            .await
            .map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::SstEngine(
                    e.to_string(),
                ))
            })?;
        info!("Graph {} exported to {:?}", self.graph_id, path.as_ref());

        Ok(())
    }

    /// Import graph from portable format
    pub async fn import(&self, engine: &OrionGraphEngine, path: impl AsRef<Path>) -> Result<()> {
        let import_url = path
            .as_ref()
            .to_str()
            .ok_or_else(|| ProximaDBError::InvalidInput("Invalid import path".to_string()))?;
        let data = self
            .filesystem_factory
            .read(import_url)
            .await
            .map_err(|e| {
                ProximaDBError::Storage(proximadb_kernel::error::StorageError::SstEngine(
                    e.to_string(),
                ))
            })?;

        let import_data: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
            ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                e.to_string(),
            ))
        })?;

        // Clear existing data
        engine.memory_pool.nodes.clear();
        engine.edge_metadata.clear();

        // Import nodes
        if let Some(nodes) = import_data["nodes"].as_array() {
            for node_val in nodes {
                let node: Node = serde_json::from_value(node_val.clone()).map_err(|e| {
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;
                engine.create_node(node).await?;
            }
        }

        // Import edges
        if let Some(edges) = import_data["edges"].as_array() {
            for edge_val in edges {
                let edge: Edge = serde_json::from_value(edge_val.clone()).map_err(|e| {
                    ProximaDBError::Storage(proximadb_kernel::error::StorageError::Serialization(
                        e.to_string(),
                    ))
                })?;
                engine.create_edge(edge).await?;
            }
        }

        info!("Graph imported from {:?}", path.as_ref());
        Ok(())
    }
}
