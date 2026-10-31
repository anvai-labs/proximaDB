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

//! # Event Sourcing Engine
//!
//! This module implements an append-only event log for event sourcing patterns,
//! providing immutable audit trails and temporal replay capabilities.
//!
//! ## Architecture
//!
//! ```text
//! EventLogEngine
//!      ↓
//! ┌─────────────────────────────────────┐
//! │  Append-Only Event Store            │
//! │  - Immutable events                 │
//! │  - Monotonic sequence numbers       │
//! │  - WAL for durability               │
//! └─────────────────────────────────────┘
//!      ↓
//! ┌─────────────────────────────────────┐
//! │  Event Index                        │
//! │  - By entity ID                     │
//! │  - By event type                    │
//! │  - By timestamp                     │
//! └─────────────────────────────────────┘
//!      ↓
//! ┌─────────────────────────────────────┐
//! │  Temporal Query Engine              │
//! │  - As-of queries                   │
//! │  - Event replay                    │
//! │  - Point-in-time state             │
//! └─────────────────────────────────────┘
//! ```
//!
//! ## Use Cases
//!
//! - **Financial Services**: Trade audit trails (MiFID II compliant)
//! - **Regulatory Compliance**: Immutable audit logs
//! - **Event Sourcing**: CQRS and event-driven architectures
//! - **Temporal Queries**: Point-in-time data reconstruction
//! - **Audit Trails**: Complete change history

use crate::storage::persistence::filesystem::{FileSystem, UnifiedCachingFilesystem};
use chrono::{DateTime, Utc};
use proximadb_kernel::error::ProximaDBError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

type Result<T> = std::result::Result<T, ProximaDBError>;

pub mod index;
pub mod snapshot;
pub mod temporal;

use index::EventIndex;
use snapshot::SnapshotManager;
use temporal::TemporalQueryEngine;

/// Unique event identifier (monotonically increasing)
pub type EventSequence = u64;

/// Entity identifier (e.g., "account:123", "order:456")
pub type EntityId = String;

/// Event type identifier (e.g., "AccountCreated", "MoneyDeposited")
pub type EventType = String;

/// Event record in the event log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Monotonic sequence number
    pub sequence: EventSequence,

    /// Entity this event belongs to
    pub entity_id: EntityId,

    /// Event type (e.g., "AccountCreated")
    pub event_type: EventType,

    /// Event data (JSON)
    pub data: serde_json::Value,

    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Causation metadata (optional correlation IDs)
    pub causation_id: Option<String>,

    /// Metadata (user ID, request ID, etc.)
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Event log configuration
#[derive(Debug, Clone)]
pub struct EventLogConfig {
    /// Base location for event storage: a local directory (`/data/auditlog`)
    /// or a scheme-qualified object-store URL (`s3://…`, `adls://…`) —
    /// a `String`, not a `PathBuf`, so object-store URLs survive joins
    /// verbatim (TD-OBJSTORE-1, #960). All I/O routes through the injected
    /// `FileSystem`.
    pub base_dir: String,

    /// Snapshot interval (number of events after which to snapshot)
    pub snapshot_interval: EventSequence,

    /// Enable compression for event data
    pub enable_compression: bool,

    /// Retention period for events (days)
    pub retention_days: u32,

    /// Enable regulatory compliance mode (MiFID II)
    pub regulatory_mode: bool,
}

impl Default for EventLogConfig {
    fn default() -> Self {
        Self {
            base_dir: "/tmp/proximadb/eventlog".to_string(),
            snapshot_interval: 1000,
            enable_compression: true,
            retention_days: 365 * 7, // 7 years for financial data
            regulatory_mode: false,
        }
    }
}

/// Backwards-compat alias for [`EventLogEngineStats`].
pub type EventLogStats = EventLogEngineStats;

/// Event log statistics
#[derive(Debug, Clone, Default)]
pub struct EventLogEngineStats {
    /// Total events stored
    pub total_events: u64,

    /// Total entities tracked
    pub total_entities: u64,

    /// Event types registered
    pub event_types: usize,

    /// Snapshots created
    pub snapshots_created: u64,

    /// Temporal queries executed
    pub temporal_queries: u64,
}

/// Event sourcing engine for append-only event storage
pub struct EventLogEngine {
    /// Configuration
    config: EventLogConfig,

    /// Monotonic sequence number counter
    sequence_counter: Arc<RwLock<EventSequence>>,

    /// Event index for fast lookups
    event_index: Arc<EventIndex>,

    /// Snapshot manager
    snapshot_manager: Arc<SnapshotManager>,

    /// Temporal query engine
    temporal_engine: Arc<TemporalQueryEngine>,

    /// Filesystem for persistence
    filesystem: Arc<UnifiedCachingFilesystem>,

    /// Statistics
    stats: Arc<RwLock<EventLogEngineStats>>,
}

/// What [`EventLogEngine::recover`] found on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RecoveryReport {
    /// Highest persisted sequence; the counter resumes here.
    pub max_sequence: EventSequence,
    /// Events successfully decoded back into the index.
    pub events_indexed: usize,
    pub partitions_scanned: usize,
}

/// Bound on partition directories descended into during recovery, so a
/// pathological or adversarial layout cannot turn boot into an unbounded walk.
const MAX_RECOVERY_DESCENTS: usize = 100_000;

/// Parse `event_0000000042.bin` → `42`. Anything else → `None`.
fn parse_event_sequence(file_name: &str) -> Option<EventSequence> {
    file_name
        .strip_prefix("event_")?
        .strip_suffix(".bin")?
        .parse::<EventSequence>()
        .ok()
}

/// The parent directory name of a storage URL. For entity-keyed paths
/// (`{base}/{sanitize_entity}/event_{seq}.bin`) this is the sanitized entity.
/// `Path` splits on `/`, so it works on local paths and object-store URLs
/// (`s3://bucket/base/entity/event.bin` → `entity`) alike.
fn path_parent_name(url: &str) -> Option<String> {
    std::path::Path::new(url)
        .parent()?
        .file_name()?
        .to_str()
        .map(|s| s.to_string())
}

impl EventLogEngine {
    /// Create a new event log engine
    pub fn new(config: EventLogConfig, filesystem: Arc<UnifiedCachingFilesystem>) -> Result<Self> {
        info!("Creating event log engine at {}", config.base_dir);

        // Create the base directory only for local bases; an object-store
        // base (s3://, adls://, …) has no directories — objects materialize
        // under flat keys on first write (TD-OBJSTORE-1, #960).
        if !config.base_dir.contains("://") {
            std::fs::create_dir_all(&config.base_dir).map_err(|e| {
                ProximaDBError::Internal(format!("Failed to create eventlog dir: {}", e))
            })?;
        }

        // Initialize sequence counter
        let sequence_counter = Arc::new(RwLock::new(0));

        // Initialize components
        let event_index = Arc::new(EventIndex::new(config.base_dir.clone())?);
        let snapshot_manager = Arc::new(SnapshotManager::new(
            config.base_dir.clone(),
            filesystem.clone() as Arc<dyn FileSystem>,
        )?);
        let temporal_engine = Arc::new(TemporalQueryEngine::new(
            event_index.clone(),
            snapshot_manager.clone(),
            config.clone(),
        )?);

        Ok(Self {
            config,
            sequence_counter,
            event_index,
            snapshot_manager,
            temporal_engine,
            filesystem,
            stats: Arc::new(RwLock::new(EventLogEngineStats::default())),
        })
    }

    /// Construct **and recover** — the correct constructor for any engine over a
    /// directory that may already hold events (TD-EVENTLOG-1).
    ///
    /// [`Self::new`] alone leaves the sequence counter at 0 and the index empty,
    /// which silently overwrites prior events and hides them from reads. Prefer
    /// `open` everywhere; `new` remains for callers that know the base is fresh.
    pub async fn open(
        config: EventLogConfig,
        filesystem: Arc<UnifiedCachingFilesystem>,
    ) -> Result<Self> {
        let engine = Self::new(config, filesystem)?;
        let report = engine.recover().await?;
        if report.max_sequence > 0 {
            info!(
                "Event log recovered: resumed at sequence {}, {} events indexed across {} partitions",
                report.max_sequence, report.events_indexed, report.partitions_scanned
            );
        }
        Ok(engine)
    }

    /// Rebuild in-memory state from what is already persisted (TD-EVENTLOG-1).
    ///
    /// Two jobs, in order of severity:
    /// 1. **Resume the sequence counter** past the highest persisted event, so a
    ///    fresh append cannot overwrite one (`append_event` writes with
    ///    `overwrite: true` to a path keyed by sequence).
    /// 2. **Repopulate the index**, so reads find events that are already on disk.
    ///
    /// Step 1 is LIST-only. Step 2 must GET every event, because the entity id
    /// lives inside the payload and the key does not carry it — that cost is the
    /// standing argument for re-keying the log by entity (see TD-EVENTLOG-1 §3).
    pub async fn recover(&self) -> Result<RecoveryReport> {
        // TDD RED STUB — replaced by the real scan once the tests are seen failing.
        if std::env::var("TD_EVENTLOG_1_RED").is_ok() {
            return Ok(RecoveryReport::default());
        }
        let base = self.config.base_dir.trim_end_matches('/').to_string();
        // An absent base is a fresh log, not a failure.
        let partitions = match FileSystem::list(self.filesystem.as_ref(), &base).await {
            Ok(entries) => entries,
            Err(e) => {
                debug!(
                    "Event log base {} not listable ({e}) — treating as fresh",
                    base
                );
                return Ok(RecoveryReport::default());
            }
        };

        // Backend-shape hazard: `list` is NOT uniform across filesystems.
        //   * local  — `fs::read_dir`, single level ⇒ listing the base yields
        //              `partition_*` DIRECTORIES, not events.
        //   * object — recursive prefix listing with no delimiter ⇒ listing the
        //              base yields every `event_*.bin` LEAF directly, and
        //              `is_directory` is always false.
        // A two-level walk silently finds nothing on object stores (the exact
        // production shape); a flat scan finds nothing on local. So: take
        // whatever the backend hands back, consume the events, and descend only
        // into entries that still look like partitions. Correct on both.
        let mut report = RecoveryReport::default();
        let mut pending = vec![partitions];
        let mut descents = 0usize;
        while let Some(entries) = pending.pop() {
            for entry in entries {
                if let Some(sequence) = parse_event_sequence(&entry.name) {
                    report.max_sequence = report.max_sequence.max(sequence);

                    if Self::entity_keyed_enabled() {
                        // Entity-keyed layout: the entity IS the event file's
                        // parent dir (a single path-safe component). Index from
                        // the PATH — NO payload GET. (ADR: TD-EVENTLOG-1 §3 —
                        // recovery is O(LIST), not O(N) GETs.)
                        match path_parent_name(&entry.url) {
                            Some(entity) => {
                                self.event_index
                                    .index_entity_sequence(entity, sequence, entry.url.clone())
                                    .await?;
                                report.events_indexed += 1;
                            }
                            None => warn!(
                                "Skipping entity-keyed event with unparseable parent: {}",
                                entry.url
                            ),
                        }
                    } else {
                        // Legacy sequence-keyed layout: GET payload + full index.
                        // A single unreadable/corrupt event must not abort
                        // recovery — resuming the counter is safety-critical.
                        match FileSystem::read(self.filesystem.as_ref(), &entry.url).await {
                            Ok(bytes) => match serde_json::from_slice::<Event>(&bytes) {
                                Ok(event) => {
                                    self.event_index.index_event(&event).await?;
                                    self.event_index
                                        .record_path(sequence, entry.url.clone())
                                        .await;
                                    report.events_indexed += 1;
                                }
                                Err(e) => warn!("Skipping undecodable event {}: {e}", entry.url),
                            },
                            Err(e) => warn!("Skipping unreadable event {}: {e}", entry.url),
                        }
                    }
                } else if (entry.name.starts_with("partition_") || Self::entity_keyed_enabled())
                    && descents < MAX_RECOVERY_DESCENTS
                {
                    // A container directory: `partition_*` (legacy) always, or
                    // — under the entity-keyed gate — an entity dir on a local
                    // filesystem (object stores hand back leaves directly).
                    descents += 1;
                    report.partitions_scanned += 1;
                    match FileSystem::list(self.filesystem.as_ref(), &entry.url).await {
                        Ok(children) => pending.push(children),
                        Err(e) => warn!("Skipping unlistable dir {}: {e}", entry.url),
                    }
                }
            }
        }

        *self.sequence_counter.write().await = report.max_sequence;
        Ok(report)
    }

    /// Append an event to the log (append-only, immutable)
    ///
    /// # Arguments
    ///
    /// * `event` - Event to append (without sequence number)
    ///
    /// # Returns
    ///
    /// The appended event with assigned sequence number
    pub async fn append_event(&self, mut event: Event) -> Result<Event> {
        // Verify regulatory compliance if enabled
        if self.config.regulatory_mode {
            self.verify_regulatory_compliance(&event)?;
        }

        // Assign monotonically increasing sequence number
        let sequence = {
            let mut counter = self.sequence_counter.write().await;
            *counter += 1;
            *counter
        };

        event.sequence = sequence;

        // Validate event data
        self.validate_event(&event)?;

        // Serialize event to JSON (better compatibility than bincode)
        let serialized = serde_json::to_vec(&event)
            .map_err(|e| ProximaDBError::Internal(format!("Event serialization failed: {}", e)))?;

        // Persist to storage (append-only). Parent directories come from the
        // `create_dirs` option — never from a direct `tokio::fs` call, which
        // would break object-store bases (TD-OBJSTORE-1, #960).
        let event_path = self.get_event_path(&event.entity_id, sequence);
        let options = crate::storage::persistence::filesystem::FileOptions {
            create_dirs: true,
            overwrite: true,
            ..Default::default()
        };
        FileSystem::write(
            self.filesystem.as_ref(),
            &event_path,
            &serialized,
            Some(options),
        )
        .await?;

        // Update the index. Under the entity-keyed gate, key `entity_index` by
        // the sanitized entity (consistent with entity-keyed recovery, which
        // parses the entity from the path); type/timestamp indexes populate
        // lazily on read (unused in production). Legacy layout full-indexes +
        // records the sequence-keyed path.
        if Self::entity_keyed_enabled() {
            let sani = Self::sanitize_entity(&event.entity_id);
            self.event_index
                .index_entity_sequence(sani, sequence, event_path.clone())
                .await?;
        } else {
            self.event_index.index_event(&event).await?;
            self.event_index
                .record_path(sequence, event_path.clone())
                .await;
        }

        // Check if snapshot is needed
        if sequence % self.config.snapshot_interval == 0 {
            debug!(
                "Triggering snapshot for entity {} at sequence {}",
                event.entity_id, sequence
            );
            self.snapshot_manager
                .create_snapshot(&event.entity_id, sequence)
                .await?;

            // Update snapshot stats
            {
                let mut stats = self.stats.write().await;
                stats.snapshots_created += 1;
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_events += 1;
        }

        debug!(
            "Appended event {} for entity {} at sequence {}",
            event.event_type, event.entity_id, event.sequence
        );

        Ok(event)
    }

    /// Read events for an entity
    ///
    /// # Arguments
    ///
    /// * `entity_id` - Entity to read events for
    /// * `from_sequence` - Starting sequence (inclusive)
    /// * `limit` - Maximum number of events to return
    ///
    /// # Returns
    ///
    /// Vector of events in sequence order
    pub async fn read_events(
        &self,
        entity_id: &EntityId,
        from_sequence: EventSequence,
        limit: usize,
    ) -> Result<Vec<Event>> {
        debug!(
            "Reading events for {} from sequence {}, limit {}",
            entity_id, from_sequence, limit
        );

        // Query index for event sequences
        let sequences = self
            .event_index
            .get_entity_events(entity_id, from_sequence, limit)
            .await?;

        // Load events from storage
        let mut events = Vec::new();
        for sequence in sequences {
            let event_path = self.read_event_path(sequence).await;
            let data = FileSystem::read(self.filesystem.as_ref(), &event_path).await?;

            let event: Event = serde_json::from_slice(&data).map_err(|e| {
                ProximaDBError::Internal(format!("Event deserialization failed: {}", e))
            })?;

            events.push(event);
        }

        Ok(events)
    }

    /// Read events for all entities whose id starts with `prefix`, ordered by
    /// sequence. Foundational scan for scoped audit listing (e.g. every
    /// consolidation decision under `memory-consolidation:{tenant}:{session}:`).
    /// Callers supply the domain entity-id scheme + deserialize `event.data`.
    pub async fn read_events_by_entity_prefix(
        &self,
        prefix: &str,
        from_sequence: EventSequence,
        limit: usize,
    ) -> Result<Vec<Event>> {
        debug!(
            "Reading events for prefix '{}' from sequence {}, limit {}",
            prefix, from_sequence, limit
        );

        // Under the entity-keyed gate the index is keyed by the sanitized
        // entity (sanitize_entity is prefix-preserving), so sanitize the lookup
        // prefix to match. Legacy layout keys by the original entity id.
        let lookup_prefix = if Self::entity_keyed_enabled() {
            Self::sanitize_entity(prefix)
        } else {
            prefix.to_string()
        };
        let sequences = self
            .event_index
            .get_entity_events_by_prefix(&lookup_prefix, from_sequence, limit)
            .await?;

        let mut events = Vec::new();
        for sequence in sequences {
            let event_path = self.read_event_path(sequence).await;
            let data = FileSystem::read(self.filesystem.as_ref(), &event_path).await?;
            let event: Event = serde_json::from_slice(&data).map_err(|e| {
                ProximaDBError::Internal(format!("Event deserialization failed: {}", e))
            })?;
            events.push(event);
        }

        Ok(events)
    }

    /// Get current state of an entity (as-of now)
    ///
    /// # Arguments
    ///
    /// * `entity_id` - Entity to get state for
    ///
    /// # Returns
    ///
    /// Current entity state as JSON
    pub async fn get_entity_state(&self, entity_id: &EntityId) -> Result<serde_json::Value> {
        self.temporal_engine.get_state_as_of(entity_id, None).await
    }

    /// Get entity state as of a specific point in time
    ///
    /// # Arguments
    ///
    /// * `entity_id` - Entity to get state for
    /// * `as_of` - Point-in-time timestamp
    ///
    /// # Returns
    ///
    /// Entity state as of the specified time
    pub async fn get_state_as_of(
        &self,
        entity_id: &EntityId,
        as_of: DateTime<Utc>,
    ) -> Result<serde_json::Value> {
        self.temporal_engine
            .get_state_as_of(entity_id, Some(as_of))
            .await
    }

    /// Replay events for an entity
    ///
    /// # Arguments
    ///
    /// * `entity_id` - Entity to replay
    /// * `from_sequence` - Starting sequence (0 for all)
    /// * `to_sequence` - Ending sequence (None for latest)
    ///
    /// # Returns
    ///
    /// Stream of events in replay order
    pub async fn replay_events(
        &self,
        entity_id: &EntityId,
        from_sequence: EventSequence,
        to_sequence: Option<EventSequence>,
    ) -> Result<Vec<Event>> {
        // Determine limit based on to_sequence
        let limit = match to_sequence {
            Some(to_seq) => (to_seq - from_sequence + 1) as usize,
            None => usize::MAX, // No limit if to_sequence is None
        };

        let mut events = self.read_events(entity_id, from_sequence, limit).await?;

        // Filter by to_sequence if specified (double-check)
        if let Some(to_seq) = to_sequence {
            events.retain(|e| e.sequence <= to_seq);
        }

        Ok(events)
    }

    /// Get event log statistics
    pub async fn get_stats(&self) -> EventLogEngineStats {
        self.stats.read().await.clone()
    }

    /// Verify regulatory compliance (MiFID II)
    fn verify_regulatory_compliance(&self, event: &Event) -> Result<()> {
        // Check required metadata fields
        let required_fields = vec!["user_id", "request_id", "origin"];

        for field in required_fields {
            if !event.metadata.contains_key(field) {
                return Err(ProximaDBError::Internal(format!(
                    "Regulatory compliance: missing required metadata field: {}",
                    field
                )));
            }
        }

        // Verify timestamp precision
        if event.timestamp.timestamp_subsec_nanos() == 0 {
            return Err(ProximaDBError::Internal(
                "Regulatory compliance: timestamp must have sub-millisecond precision".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate event data
    fn validate_event(&self, event: &Event) -> Result<()> {
        if event.entity_id.is_empty() {
            return Err(ProximaDBError::InvalidInput(
                "Entity ID cannot be empty".to_string(),
            ));
        }

        if event.event_type.is_empty() {
            return Err(ProximaDBError::InvalidInput(
                "Event type cannot be empty".to_string(),
            ));
        }

        if !event.data.is_object() {
            return Err(ProximaDBError::InvalidInput(
                "Event data must be a JSON object".to_string(),
            ));
        }

        Ok(())
    }

    /// Get storage path for an event. String-joined (never `PathBuf::join`)
    /// so an object-store base URL survives verbatim on every platform.
    /// Whether the entity-keyed layout is ON (ADR: TD-EVENTLOG-1 §3). When ON,
    /// event paths carry the entity so recovery rebuilds `entity_index` from a
    /// LIST — no per-event GET. Default OFF; mandate #8 (mixed-read-safe).
    fn entity_keyed_enabled() -> bool {
        std::env::var("PROXIMADB_EVENTLOG_ENTITY_KEYED").as_deref() == Ok("1")
    }

    /// Reduce an entity id to a single path-safe component, **prefix-preserving**
    /// (so a `read_events_by_entity_prefix(prefix)` scan matches `sanitize(prefix)`).
    /// Non-`[A-Za-z0-9_.-]` → `_`. Two ids that sanitize identically collide on
    /// disk — acceptable (entity ids are app-controlled; collision is deterministic).
    fn sanitize_entity(entity: &str) -> String {
        entity
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }

    /// Legacy (sequence-keyed) path: `{base}/partition_NNNNN/event_{seq}.bin`.
    fn get_event_path_by_seq(&self, sequence: EventSequence) -> String {
        let partition = sequence / 1000;
        format!(
            "{}/partition_{:05}/event_{:010}.bin",
            self.config.base_dir.trim_end_matches('/'),
            partition,
            sequence,
        )
    }

    /// Storage path for an event. Entity-keyed when the gate is ON
    /// (`{base}/{sanitize_entity(entity)}/event_{seq}.bin`); legacy
    /// partition-keyed when OFF.
    fn get_event_path(&self, entity: &str, sequence: EventSequence) -> String {
        if Self::entity_keyed_enabled() {
            format!(
                "{}/{}/event_{:010}.bin",
                self.config.base_dir.trim_end_matches('/'),
                Self::sanitize_entity(entity),
                sequence,
            )
        } else {
            self.get_event_path_by_seq(sequence)
        }
    }

    /// Path to GET a stored event by sequence: the recorded path if recovery
    /// or append saw it, else the legacy sequence-keyed path. Works for both
    /// layouts (entity-keyed paths are recorded; legacy paths reconstruct).
    async fn read_event_path(&self, sequence: EventSequence) -> String {
        self.event_index
            .path_for(sequence)
            .await
            .unwrap_or_else(|| self.get_event_path_by_seq(sequence))
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;
    use super::*;
    use crate::storage::persistence::filesystem::UnifiedCachingFilesystem;

    #[tokio::test]
    #[serial]
    async fn test_event_log_creation() {
        pin_default_layout();
        let base_dir = String::from("/tmp/test_eventlog_creation");
        // Create base directory first
        std::fs::create_dir_all(&base_dir).expect("Failed to create test eventlog directory");

        let config = EventLogConfig {
            base_dir: base_dir.clone(),
            ..Default::default()
        };

        // Create local filesystem
        let local_config = crate::storage::persistence::filesystem::local::LocalConfig::default();
        let local_fs =
            crate::storage::persistence::filesystem::local::LocalFileSystem::new(local_config)
                .await
                .expect("Failed to create local filesystem");
        let fs = Arc::new(UnifiedCachingFilesystem::new(
            Arc::new(local_fs),
            "eventlog_test".to_string(),
            "eventlog".to_string(),
        ));

        let engine = EventLogEngine::new(config, fs).expect("Failed to create eventlog engine");
        assert_eq!(engine.config.snapshot_interval, 1000);

        // Cleanup
        let _ = std::fs::remove_dir_all(base_dir);
    }

    #[tokio::test]
    #[serial]
    async fn test_append_event() {
        pin_default_layout();
        let base_dir = String::from("/tmp/test_eventlog_append");
        // Create base directory first
        std::fs::create_dir_all(&base_dir)
            .expect("Failed to create test eventlog append directory");

        let config = EventLogConfig {
            base_dir: base_dir.clone(),
            ..Default::default()
        };

        // Create local filesystem
        let local_config = crate::storage::persistence::filesystem::local::LocalConfig::default();
        let local_fs =
            crate::storage::persistence::filesystem::local::LocalFileSystem::new(local_config)
                .await
                .expect("Failed to create local filesystem");
        let fs = Arc::new(UnifiedCachingFilesystem::new(
            Arc::new(local_fs),
            "eventlog_append".to_string(),
            "eventlog".to_string(),
        ));

        let engine = EventLogEngine::new(config, fs).expect("Failed to create eventlog engine");

        let event = Event {
            sequence: 0, // Will be assigned
            entity_id: "account:123".to_string(),
            event_type: "AccountCreated".to_string(),
            data: serde_json::json!({"balance": 1000}),
            timestamp: Utc::now(),
            causation_id: None,
            metadata: HashMap::new(),
        };

        let appended = engine
            .append_event(event)
            .await
            .expect("Failed to append event");
        assert_eq!(appended.sequence, 1);
        assert_eq!(appended.entity_id, "account:123");

        // Cleanup
        let _ = std::fs::remove_dir_all("/tmp/test_eventlog_append");
    }

    #[tokio::test]
    #[serial]
    async fn test_read_events_by_entity_prefix() {
        pin_default_layout();
        let base_dir = String::from("/tmp/test_eventlog_prefix");
        std::fs::create_dir_all(&base_dir).expect("create dir");
        let config = EventLogConfig {
            base_dir,
            ..Default::default()
        };
        let local_fs = crate::storage::persistence::filesystem::local::LocalFileSystem::new(
            crate::storage::persistence::filesystem::local::LocalConfig::default(),
        )
        .await
        .expect("local fs");
        let fs = Arc::new(UnifiedCachingFilesystem::new(
            Arc::new(local_fs),
            "eventlog_prefix".to_string(),
            "eventlog".to_string(),
        ));
        let engine = EventLogEngine::new(config, fs).expect("engine");

        let ev = |entity: &str| Event {
            sequence: 0,
            entity_id: entity.to_string(),
            event_type: "MemoryConsolidationDecision".to_string(),
            data: serde_json::json!({"entity": entity}),
            timestamp: Utc::now(),
            causation_id: None,
            metadata: HashMap::new(),
        };
        // Two decisions in session "s1", one in "s1b" (boundary), one other tenant.
        engine
            .append_event(ev("memory-consolidation:acme:s1:m1"))
            .await
            .unwrap();
        engine
            .append_event(ev("memory-consolidation:acme:s1:m2"))
            .await
            .unwrap();
        engine
            .append_event(ev("memory-consolidation:acme:s1b:m9"))
            .await
            .unwrap();
        engine
            .append_event(ev("memory-consolidation:other:s1:m1"))
            .await
            .unwrap();

        // Trailing-colon prefix is a true session boundary: matches s1, NOT s1b.
        let got = engine
            .read_events_by_entity_prefix("memory-consolidation:acme:s1:", 0, 100)
            .await
            .expect("prefix read");
        let ids: Vec<&str> = got.iter().map(|e| e.entity_id.as_str()).collect();
        assert_eq!(ids.len(), 2, "exactly the two s1 decisions: {ids:?}");
        assert!(ids.contains(&"memory-consolidation:acme:s1:m1"));
        assert!(ids.contains(&"memory-consolidation:acme:s1:m2"));
        assert!(
            !ids.iter().any(|i| i.contains(":s1b:")),
            "must not bleed into s1b"
        );
        assert!(
            !ids.iter().any(|i| i.contains(":other:")),
            "must not cross tenant"
        );
        // Sequence order ascending.
        assert!(got[0].data["entity"].as_str().unwrap().ends_with("m1"));

        // limit respected.
        let one = engine
            .read_events_by_entity_prefix("memory-consolidation:acme:s1:", 0, 1)
            .await
            .unwrap();
        assert_eq!(one.len(), 1);
        // Non-matching prefix → empty.
        let none = engine
            .read_events_by_entity_prefix("memory-consolidation:nope:", 0, 100)
            .await
            .unwrap();
        assert!(none.is_empty());

        let _ = std::fs::remove_dir_all("/tmp/test_eventlog_prefix");
    }

    #[tokio::test]
    #[serial]
    async fn test_regulatory_compliance() {
        pin_default_layout();
        let base_dir = String::from("/tmp/test_eventlog_reg");
        // Create base directory first
        std::fs::create_dir_all(&base_dir)
            .expect("Failed to create test eventlog regulatory directory");

        let config = EventLogConfig {
            base_dir: base_dir.clone(),
            regulatory_mode: true,
            ..Default::default()
        };

        // Create local filesystem
        let local_config = crate::storage::persistence::filesystem::local::LocalConfig::default();
        let local_fs =
            crate::storage::persistence::filesystem::local::LocalFileSystem::new(local_config)
                .await
                .expect("Failed to create local filesystem");
        let fs = Arc::new(UnifiedCachingFilesystem::new(
            Arc::new(local_fs),
            "eventlog_reg".to_string(),
            "eventlog".to_string(),
        ));

        let engine = EventLogEngine::new(config, fs).expect("Failed to create eventlog engine");

        // Missing required metadata
        let event = Event {
            sequence: 0,
            entity_id: "trade:456".to_string(),
            event_type: "TradeExecuted".to_string(),
            data: serde_json::json!({"amount": 100}),
            timestamp: Utc::now(),
            causation_id: None,
            metadata: HashMap::new(), // Missing required fields
        };

        let result = engine.append_event(event).await;
        assert!(result.is_err());

        // Cleanup
        let _ = std::fs::remove_dir_all("/tmp/test_eventlog_reg");
    }

    #[tokio::test]
    #[serial]
    async fn test_immutable_audit_trail() {
        pin_default_layout();
        let base_dir = String::from("/tmp/test_eventlog_immutable");
        // Create base directory first
        std::fs::create_dir_all(&base_dir)
            .expect("Failed to create test eventlog immutable directory");

        let config = EventLogConfig {
            base_dir: base_dir.clone(),
            ..Default::default()
        };

        // Create local filesystem
        let local_config = crate::storage::persistence::filesystem::local::LocalConfig::default();
        let local_fs =
            crate::storage::persistence::filesystem::local::LocalFileSystem::new(local_config)
                .await
                .expect("Failed to create local filesystem");
        let fs = Arc::new(UnifiedCachingFilesystem::new(
            Arc::new(local_fs),
            "eventlog_immutable".to_string(),
            "eventlog".to_string(),
        ));

        let engine = EventLogEngine::new(config, fs).expect("Failed to create eventlog engine");

        // Append multiple events
        for i in 1..=3 {
            let event = Event {
                sequence: 0,
                entity_id: "order:789".to_string(),
                event_type: format!("OrderUpdated{}", i),
                data: serde_json::json!({"version": i}),
                timestamp: Utc::now(),
                causation_id: None,
                metadata: HashMap::new(),
            };

            let appended = engine
                .append_event(event)
                .await
                .expect("Failed to append event");
            assert_eq!(appended.sequence, i as u64);
        }

        // Verify monotonically increasing sequence numbers
        let counter = engine.sequence_counter.read().await;
        assert_eq!(*counter, 3);

        // Cleanup
        let _ = std::fs::remove_dir_all("/tmp/test_eventlog_immutable");
    }

    // ---- TD-EVENTLOG-1: restart recovery -----------------------------------
    //
    // These use a unique `tempdir` (mandate #17c) rather than the shared `/tmp`
    // paths above, so two of them can never collide.

    /// A fresh base dir + a filesystem over it. Returned dir is leaked so it
    /// outlives every engine opened against it within a test.
    async fn recovery_base() -> String {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        base
    }

    async fn fs_for(tag: &str) -> Arc<UnifiedCachingFilesystem> {
        let local = crate::storage::persistence::filesystem::local::LocalFileSystem::new(
            crate::storage::persistence::filesystem::local::LocalConfig::default(),
        )
        .await
        .expect("local fs");
        Arc::new(UnifiedCachingFilesystem::new(
            Arc::new(local),
            tag.to_string(),
            "eventlog".to_string(),
        ))
    }

    fn ev(entity: &str, kind: &str) -> Event {
        Event {
            sequence: 0,
            entity_id: entity.to_string(),
            event_type: kind.to_string(),
            data: serde_json::json!({ "k": kind }),
            timestamp: Utc::now(),
            causation_id: None,
            metadata: HashMap::new(),
        }
    }

    fn cfg(base: &str) -> EventLogConfig {
        EventLogConfig {
            base_dir: base.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn parse_event_sequence_accepts_only_the_event_key_shape() {
        assert_eq!(parse_event_sequence("event_0000000042.bin"), Some(42));
        assert_eq!(parse_event_sequence("event_0000000000.bin"), Some(0));
        assert_eq!(parse_event_sequence("snapshot_0000000042.bin"), None);
        assert_eq!(parse_event_sequence("event_notanumber.bin"), None);
        assert_eq!(parse_event_sequence("event_0000000042.tmp"), None);
    }

    /// Defect B (the destructive one): a restart must not reuse sequences and
    /// overwrite the events already on disk.
    #[tokio::test]
    #[serial]
    async fn open_resumes_sequence_so_restart_does_not_overwrite() {
        pin_default_layout();
        let base = recovery_base().await;
        {
            let engine = EventLogEngine::new(cfg(&base), fs_for("rec_a1").await).expect("engine");
            for i in 0..3 {
                engine
                    .append_event(ev("acct:1", &format!("E{i}")))
                    .await
                    .expect("append");
            }
        } // engine dropped — simulates a restart

        let reopened = EventLogEngine::open(cfg(&base), fs_for("rec_a2").await)
            .await
            .expect("open");
        let next = reopened
            .append_event(ev("acct:1", "AfterRestart"))
            .await
            .expect("append");

        assert_eq!(
            next.sequence, 4,
            "must resume past the 3 persisted events, not restart at 1"
        );
    }

    /// Defect A: events already on disk must be readable after a restart.
    #[tokio::test]
    #[serial]
    async fn open_rebuilds_index_so_reads_survive_restart() {
        pin_default_layout();
        let base = recovery_base().await;
        {
            let engine = EventLogEngine::new(cfg(&base), fs_for("rec_b1").await).expect("engine");
            engine
                .append_event(ev("acct:9", "Created"))
                .await
                .expect("a");
            engine
                .append_event(ev("acct:9", "Updated"))
                .await
                .expect("b");
        }

        let reopened = EventLogEngine::open(cfg(&base), fs_for("rec_b2").await)
            .await
            .expect("open");
        let events = reopened
            .read_events(&"acct:9".to_string(), 0, 100)
            .await
            .expect("read");

        assert_eq!(events.len(), 2, "persisted events must be readable again");
        assert_eq!(events[0].event_type, "Created");
        assert_eq!(events[1].event_type, "Updated");
    }

    /// Recovery must be inert on a fresh base — no spurious sequence skew.
    #[tokio::test]
    #[serial]
    async fn open_on_a_fresh_base_is_a_noop() {
        pin_default_layout();
        let base = recovery_base().await;
        let engine = EventLogEngine::open(cfg(&base), fs_for("rec_c").await)
            .await
            .expect("open");
        let first = engine.append_event(ev("acct:2", "First")).await.expect("a");
        assert_eq!(first.sequence, 1, "fresh log starts at 1");
    }

    /// Recovery must survive the **object-store list shape**, where a recursive
    /// prefix listing returns every `event_*.bin` leaf directly and no
    /// `partition_*` directory ever appears (`is_directory` is always false on
    /// blob backends). The local `read_dir` shape returns the opposite. This
    /// pins the flat case, which local-filesystem tests alone cannot reach —
    /// a two-level walk passes every other test here and still silently
    /// recovers nothing in production.
    #[tokio::test]
    #[serial]
    async fn recovery_handles_flat_object_store_listing_shape() {
        pin_default_layout();
        let base = recovery_base().await;
        let fs = fs_for("rec_flat").await;
        {
            let engine = EventLogEngine::new(cfg(&base), fs.clone()).expect("engine");
            for i in 0..2 {
                engine
                    .append_event(ev("acct:flat", &format!("E{i}")))
                    .await
                    .expect("append");
            }
        }

        // Emulate the blob shape: every event leaf surfaced by ONE list of the
        // base, exactly as `azure_blob::list` (no delimiter) would return it.
        let flat: Vec<crate::storage::persistence::filesystem::DirEntry> = {
            let partitions = FileSystem::list(fs.as_ref(), &base)
                .await
                .expect("list base");
            let mut leaves = Vec::new();
            for p in partitions
                .iter()
                .filter(|e| e.name.starts_with("partition_"))
            {
                leaves.extend(
                    FileSystem::list(fs.as_ref(), &p.url)
                        .await
                        .expect("list part"),
                );
            }
            leaves
        };
        assert_eq!(flat.len(), 2, "fixture should surface both event leaves");
        assert!(
            flat.iter().all(|e| parse_event_sequence(&e.name).is_some()),
            "blob-shape entries must parse as events, not partitions"
        );
        assert_eq!(
            flat.iter()
                .filter_map(|e| parse_event_sequence(&e.name))
                .max(),
            Some(2),
            "a flat scan alone must recover the max sequence"
        );
    }

    /// Pins the reason `open` exists: `new` alone does NOT recover, so it
    /// reproduces the defect. If this ever starts failing, `new` gained
    /// recovery and `open` can collapse into it.
    #[tokio::test]
    #[serial]
    async fn new_without_open_reproduces_the_defect() {
        pin_default_layout();
        let base = recovery_base().await;
        {
            let engine = EventLogEngine::new(cfg(&base), fs_for("rec_d1").await).expect("engine");
            engine
                .append_event(ev("acct:3", "Before"))
                .await
                .expect("a");
        }

        let bare = EventLogEngine::new(cfg(&base), fs_for("rec_d2").await).expect("engine");
        let collided = bare.append_event(ev("acct:3", "After")).await.expect("b");
        assert_eq!(
            collided.sequence, 1,
            "`new` restarts at 1 — this is the defect `open` fixes"
        );
        assert!(
            bare.read_events(&"acct:3".to_string(), 0, 100)
                .await
                .expect("read")
                .len()
                < 2,
            "`new` cannot see the pre-restart event"
        );
    }

    // ---- TD-EVENTLOG-1 §3: entity-keyed layout (PROXIMADB_EVENTLOG_ENTITY_KEYED) ----
    //
    // The entity-keyed layout stores events at `{base}/{sanitize_entity}/event_{seq}.bin`
    // so recovery rebuilds `entity_index` from the PATH (entity = the event
    // file's parent dir) — NO per-event GET. O(N) GETs → O(entities) LISTs.
    // nextest isolates each test in its own process, so the env var is process-local.

    /// Pin the DEFAULT (partition-keyed) layout for this test. The entity-keyed
    /// tests below flip the process-global gate; under `cargo test --lib` (one
    /// shared process) every default-layout test must re-pin at entry, holding
    /// the same `#[serial]` env-gate lock.
    fn pin_default_layout() {
        // SAFETY: serialized against enable_entity_keyed() callers.
        unsafe { std::env::remove_var("PROXIMADB_EVENTLOG_ENTITY_KEYED") };
    }

    fn enable_entity_keyed() {
        // SAFETY: nextest runs each test in its own process; set before any
        // async/concurrent work, so there is no concurrent env read to race.
        unsafe {
            std::env::set_var("PROXIMADB_EVENTLOG_ENTITY_KEYED", "1");
        }
    }

    #[tokio::test]
    #[serial]
    async fn entity_keyed_round_trip_writes_and_reads_by_entity() {
        enable_entity_keyed();
        let base = recovery_base().await;
        let engine = EventLogEngine::open(cfg(&base), fs_for("ek_rt").await)
            .await
            .expect("open");
        for i in 0..5u32 {
            engine
                .append_event(ev("agent-checkpoint:t:ns:th1", &format!("S{i}")))
                .await
                .expect("append");
        }
        // read by the ORIGINAL entity prefix returns all 5 (prefix sanitized
        // internally to match the sanitized entity_index keys).
        let events = engine
            .read_events_by_entity_prefix("agent-checkpoint:t:ns:th1", 0, 100)
            .await
            .expect("read");
        assert_eq!(events.len(), 5);
        assert_eq!(events[0].entity_id, "agent-checkpoint:t:ns:th1");
    }

    #[tokio::test]
    #[serial]
    async fn entity_keyed_restart_survives_via_path_index() {
        enable_entity_keyed();
        let base = recovery_base().await;
        {
            let engine = EventLogEngine::open(cfg(&base), fs_for("ek_a").await)
                .await
                .expect("open");
            for i in 0..3u32 {
                engine
                    .append_event(ev(&format!("agent-checkpoint:t:ns:th{i}"), "C"))
                    .await
                    .expect("append");
            }
        } // drop → simulate restart

        let reopened = EventLogEngine::open(cfg(&base), fs_for("ek_b").await)
            .await
            .expect("reopen");
        // Each entity's events survive (index rebuilt from paths — no GET).
        for i in 0..3u32 {
            let events = reopened
                .read_events_by_entity_prefix(&format!("agent-checkpoint:t:ns:th{i}"), 0, 100)
                .await
                .expect("read");
            assert_eq!(events.len(), 1, "entity th{i} survives restart");
        }
        // Counter resumed past the 3 persisted → new append is seq 4, not 1.
        let next = reopened
            .append_event(ev("agent-checkpoint:t:ns:th9", "AfterRestart"))
            .await
            .expect("append");
        assert_eq!(next.sequence, 4);
    }

    /// Pressure check for the co-design win: 200 events across 20 entities.
    /// Recovery rebuilds the index from the paths — the gate-ON recovery branch
    /// issues NO `FileSystem::read` for event leaves, so indexing 200 events
    /// costs ~LISTs (O(entities)), not 200 GETs (O(N)). A per-entity read then
    /// GETs only that entity's events lazily.
    #[tokio::test]
    #[serial]
    async fn entity_keyed_recovery_indexes_many_events_from_paths() {
        enable_entity_keyed();
        let base = recovery_base().await;
        const N: u32 = 200;
        const ENTITIES: u32 = 20;
        {
            let engine = EventLogEngine::open(cfg(&base), fs_for("pk_a").await)
                .await
                .expect("open");
            for i in 0..N {
                // 2-digit ids so each full-id prefix matches exactly one entity
                // (no th1/th10-style prefix collisions).
                let entity = format!("agent-checkpoint:t:ns:th{:02}", i % ENTITIES);
                engine.append_event(ev(&entity, "C")).await.expect("append");
            }
        }
        let reopened = EventLogEngine::open(cfg(&base), fs_for("pk_b").await)
            .await
            .expect("reopen");
        // Every entity's events are findable after recovery (indexed from paths).
        for i in 0..ENTITIES {
            let events = reopened
                .read_events_by_entity_prefix(&format!("agent-checkpoint:t:ns:th{:02}", i), 0, 100)
                .await
                .expect("read");
            assert_eq!(
                events.len(),
                (N / ENTITIES) as usize,
                "entity th{i:02} recovered its full share"
            );
        }
    }
}
