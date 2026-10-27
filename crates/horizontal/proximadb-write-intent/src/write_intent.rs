// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Shared write-intent and write-lane routing contract.
//!
//! Protocol handlers are adapters. SQL/pgwire, REST, gRPC, Arrow Flight, and
//! embedded clients should decode their protocol-specific payloads and then
//! lower writes into this contract before touching durable storage. This keeps
//! ADR-009/ADR-010 semantics centralized: xCatalog/schema validation, tenant
//! policy, canonical WAL/current-state writes, direct batch commits, and
//! rebuildable projection writes are explicit routing decisions instead of
//! hidden protocol-specific optimizations.

/// Minimum row hint before append-style writes are considered bulk writes.
pub const DEFAULT_BULK_ROW_THRESHOLD: u64 = 500;

/// Minimum byte hint before append-style writes are considered bulk writes.
pub const DEFAULT_BULK_BYTES_THRESHOLD: u64 = 2 * 1024 * 1024;

/// Logical mutation requested by a protocol or internal executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WriteOperationKind {
    Insert,
    Upsert,
    Update,
    Delete,
    Merge,
    Append,
    OverwriteTable,
    OverwritePartitions,
    ProjectionRefresh,
}

impl WriteOperationKind {
    /// Returns true when the operation needs prior-version checks or tombstones.
    pub fn requires_row_level_mvcc(self) -> bool {
        matches!(
            self,
            Self::Upsert | Self::Update | Self::Delete | Self::Merge
        )
    }

    /// Returns true when the operation can be modeled as an append-only load.
    pub fn is_append_like(self) -> bool {
        matches!(self, Self::Insert | Self::Append)
    }

    /// Returns true when the operation swaps table or partition snapshots.
    pub fn is_overwrite_like(self) -> bool {
        matches!(self, Self::OverwriteTable | Self::OverwritePartitions)
    }
}

/// Authoritative write lane selected below protocol adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WriteLane {
    /// Default OLTP/HTAP lane: canonical WAL plus current-state storage.
    WalCurrentState,
    /// Append-only durable segment/manifest commit, valid only with guards.
    BulkAppendCommit,
    /// Table/partition snapshot swap, valid only with snapshot guards.
    OverwriteSnapshotCommit,
    /// Rebuildable projection write after another authoritative commit.
    ProjectionOnly,
}

/// Durability authority requested by the decoded write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WriteDurabilityRequirement {
    /// The write must be recovered from the canonical WAL.
    WalRequired,
    /// A direct durable segment/manifest commit may replace per-row WAL.
    DirectCommitAllowed,
    /// The write is a rebuildable projection refresh, not authoritative state.
    ProjectionOnly,
    /// External open-table metadata is authoritative for the committed snapshot.
    ExternalAuthoritative,
}

/// Isolation profile requested by the write.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WriteIsolationRequirement {
    #[default]
    ReadCommitted,
    Snapshot,
    Serializable,
}

/// Freshness expectation for projections populated as part of the write.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionFreshnessRequirement {
    #[default]
    None,
    BestEffort,
    ReadYourWrites,
    Synchronous,
}

/// Guard that must be satisfied by the executor selected for a write lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WriteGuard {
    CatalogSchemaValidation,
    TenantRlsCoercion,
    ConstraintValidation,
    CanonicalWalAppend,
    CurrentStateUpdate,
    RowLevelMvcc,
    BatchLocalConstraintCheck,
    AtomicSnapshotManifestCommit,
    IdempotencyKey,
    ProjectionFreshnessMetadata,
    ExternalAuthorityBoundary,
}

/// Protocol-neutral write intent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteIntent {
    pub target_table: String,
    pub operation_kind: WriteOperationKind,
    pub durability: WriteDurabilityRequirement,
    pub isolation: WriteIsolationRequirement,
    pub projection_freshness: ProjectionFreshnessRequirement,
    pub tenant_id: Option<String>,
    pub actor: Option<String>,
    pub idempotency_key: Option<String>,
    pub catalog_schema_version: Option<u64>,
    pub row_count_hint: Option<u64>,
    pub estimated_bytes: Option<u64>,
    pub requires_row_level_semantics: bool,
    pub batch_local_constraints_sufficient: bool,

    /// The table's published (Iceberg) parent snapshot carries deletion
    /// vectors. The bulk-append lane takes a successor slot without the
    /// publish lock and never records the catalog layout flip, so running it
    /// above a DV-carrying parent strands every later publication's
    /// (version, LSN) pairing — the visibility fence contract's
    /// forbid-or-fence decision for the bulk lane is FORBID
    /// (PUBLICATION_WAL_VISIBILITY_FENCE_2026_10_27.adoc). The DML service
    /// sets this from the metadata log; `false` when the table has no
    /// published snapshot.
    pub published_parent_has_deletes: bool,

    /// Fencing generation from the DML lock guard that authorizes this
    /// write (A6 boundary seam). `None` for non-DML/legacy writes. The
    /// storage writer validates this against the current durable lease
    /// generation before committing.
    pub fencing_generation: Option<u64>,
}

impl WriteIntent {
    pub fn new(target_table: impl Into<String>, operation_kind: WriteOperationKind) -> Self {
        let requires_row_level_semantics = operation_kind.requires_row_level_mvcc();

        Self {
            target_table: target_table.into(),
            operation_kind,
            durability: WriteDurabilityRequirement::WalRequired,
            isolation: WriteIsolationRequirement::default(),
            projection_freshness: ProjectionFreshnessRequirement::default(),
            tenant_id: None,
            actor: None,
            idempotency_key: None,
            catalog_schema_version: None,
            row_count_hint: None,
            estimated_bytes: None,
            requires_row_level_semantics,
            batch_local_constraints_sufficient: false,
            published_parent_has_deletes: false,
            fencing_generation: None,
        }
    }

    pub fn with_durability(mut self, durability: WriteDurabilityRequirement) -> Self {
        self.durability = durability;
        self
    }

    pub fn with_isolation(mut self, isolation: WriteIsolationRequirement) -> Self {
        self.isolation = isolation;
        self
    }

    pub fn with_projection_freshness(mut self, freshness: ProjectionFreshnessRequirement) -> Self {
        self.projection_freshness = freshness;
        self
    }

    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    pub fn with_actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = Some(actor.into());
        self
    }

    pub fn with_idempotency_key(mut self, idempotency_key: impl Into<String>) -> Self {
        self.idempotency_key = Some(idempotency_key.into());
        self
    }

    pub fn with_catalog_schema_version(mut self, version: u64) -> Self {
        self.catalog_schema_version = Some(version);
        self
    }

    pub fn with_row_count_hint(mut self, row_count: u64) -> Self {
        self.row_count_hint = Some(row_count);
        self
    }

    pub fn with_estimated_bytes(mut self, bytes: u64) -> Self {
        self.estimated_bytes = Some(bytes);
        self
    }

    pub fn with_row_level_semantics(mut self, requires_row_level_semantics: bool) -> Self {
        self.requires_row_level_semantics = requires_row_level_semantics;
        self
    }

    pub fn with_batch_local_constraints_sufficient(mut self, sufficient: bool) -> Self {
        self.batch_local_constraints_sufficient = sufficient;
        self
    }

    pub fn with_published_parent_has_deletes(mut self, has_deletes: bool) -> Self {
        self.published_parent_has_deletes = has_deletes;
        self
    }

    /// Attach the fencing generation from a DML lock guard (A6 seam).
    /// The storage writer validates this against the current durable
    /// lease generation before committing the mutation.
    pub fn with_fencing_generation(mut self, generation: u64) -> Self {
        self.fencing_generation = Some(generation);
        self
    }

    fn has_bulk_hint(&self, config: &WriteLaneRouterConfig) -> bool {
        self.row_count_hint
            .is_some_and(|rows| rows >= config.bulk_row_threshold)
            || self
                .estimated_bytes
                .is_some_and(|bytes| bytes >= config.bulk_bytes_threshold)
    }
}

/// Explains why a lane was rejected for a decoded write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedWriteLane {
    pub lane: WriteLane,
    pub reason: String,
}

impl RejectedWriteLane {
    pub fn new(lane: WriteLane, reason: impl Into<String>) -> Self {
        Self {
            lane,
            reason: reason.into(),
        }
    }
}

/// Routing decision plus executor guard requirements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteLaneDecision {
    pub lane: WriteLane,
    pub reason: String,
    pub required_guards: Vec<WriteGuard>,
    pub rejected_lanes: Vec<RejectedWriteLane>,
}

impl WriteLaneDecision {
    fn new(
        lane: WriteLane,
        reason: impl Into<String>,
        required_guards: Vec<WriteGuard>,
        rejected_lanes: Vec<RejectedWriteLane>,
    ) -> Self {
        Self {
            lane,
            reason: reason.into(),
            required_guards,
            rejected_lanes,
        }
    }

    /// Assert that the selected lane is `WalCurrentState`.
    ///
    /// Protocol adapters (gRPC, Arrow Flight) that route through legacy vector
    /// handlers rather than `NativeTableWriteExecutor` must call this after
    /// computing the lane decision so that non-WAL intents are rejected
    /// explicitly rather than silently falling through to WAL.
    pub fn require_wal_lane(&self, context: &str) -> anyhow::Result<()> {
        if self.lane == WriteLane::WalCurrentState {
            return Ok(());
        }
        let rejected: Vec<&str> = self
            .rejected_lanes
            .iter()
            .map(|r| r.reason.as_str())
            .collect();
        Err(anyhow::anyhow!(
            "{}: write-lane {:?} is not yet committed by this adapter \
             (selected reason: {}; rejected lane notes: {:?}). \
             Only WalCurrentState is supported until dedicated commit protocols are wired.",
            context,
            self.lane,
            self.reason,
            rejected,
        ))
    }
}

/// Tunables for the shared write-lane router.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteLaneRouterConfig {
    pub bulk_row_threshold: u64,
    pub bulk_bytes_threshold: u64,
}

impl Default for WriteLaneRouterConfig {
    fn default() -> Self {
        Self {
            bulk_row_threshold: DEFAULT_BULK_ROW_THRESHOLD,
            bulk_bytes_threshold: DEFAULT_BULK_BYTES_THRESHOLD,
        }
    }
}

/// Shared router used after protocol-specific decoding/parsing.
#[derive(Debug, Clone, Default)]
pub struct WriteLaneRouter {
    config: WriteLaneRouterConfig,
}

impl WriteLaneRouter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: WriteLaneRouterConfig) -> Self {
        Self { config }
    }

    pub fn route(&self, intent: &WriteIntent) -> WriteLaneDecision {
        match intent.operation_kind {
            WriteOperationKind::ProjectionRefresh => self.route_projection(intent),
            kind if kind.requires_row_level_mvcc() || intent.requires_row_level_semantics => {
                self.route_row_level(intent)
            }
            kind if kind.is_overwrite_like() => self.route_overwrite(intent),
            kind if kind.is_append_like() => self.route_append(intent),
            WriteOperationKind::Insert
            | WriteOperationKind::Upsert
            | WriteOperationKind::Update
            | WriteOperationKind::Delete
            | WriteOperationKind::Merge
            | WriteOperationKind::Append
            | WriteOperationKind::OverwriteTable
            | WriteOperationKind::OverwritePartitions => self.route_wal_default(
                "operation does not satisfy a specialized fast-lane rule",
                self.default_wal_guards(intent),
                Vec::new(),
            ),
        }
    }

    fn route_projection(&self, intent: &WriteIntent) -> WriteLaneDecision {
        let mut guards = vec![WriteGuard::ProjectionFreshnessMetadata];
        if matches!(
            intent.durability,
            WriteDurabilityRequirement::ExternalAuthoritative
        ) {
            guards.push(WriteGuard::ExternalAuthorityBoundary);
        }

        WriteLaneDecision::new(
            WriteLane::ProjectionOnly,
            "projection refresh is rebuildable and not authoritative current state",
            guards,
            vec![
                RejectedWriteLane::new(
                    WriteLane::WalCurrentState,
                    "projection refresh does not require canonical row mutation",
                ),
                RejectedWriteLane::new(
                    WriteLane::BulkAppendCommit,
                    "projection refresh cannot create authoritative append state",
                ),
            ],
        )
    }

    fn route_row_level(&self, intent: &WriteIntent) -> WriteLaneDecision {
        self.route_wal_default(
            "row-level mutation requires WAL ordering, MVCC, and current-state update",
            self.default_wal_guards(intent),
            vec![
                RejectedWriteLane::new(
                    WriteLane::BulkAppendCommit,
                    "append direct commit cannot close prior versions or write tombstones",
                ),
                RejectedWriteLane::new(
                    WriteLane::OverwriteSnapshotCommit,
                    "snapshot overwrite is not valid for row-level mutation semantics",
                ),
            ],
        )
    }

    fn route_overwrite(&self, intent: &WriteIntent) -> WriteLaneDecision {
        if matches!(
            intent.durability,
            WriteDurabilityRequirement::DirectCommitAllowed
                | WriteDurabilityRequirement::ExternalAuthoritative
        ) {
            let mut guards = vec![
                WriteGuard::CatalogSchemaValidation,
                WriteGuard::TenantRlsCoercion,
                WriteGuard::ConstraintValidation,
                WriteGuard::AtomicSnapshotManifestCommit,
                WriteGuard::IdempotencyKey,
            ];
            if matches!(
                intent.durability,
                WriteDurabilityRequirement::ExternalAuthoritative
            ) {
                guards.push(WriteGuard::ExternalAuthorityBoundary);
            }

            return WriteLaneDecision::new(
                WriteLane::OverwriteSnapshotCommit,
                "overwrite can commit through an atomic table or partition snapshot swap",
                guards,
                vec![RejectedWriteLane::new(
                    WriteLane::BulkAppendCommit,
                    "overwrite replaces a snapshot instead of appending into one",
                )],
            );
        }

        self.route_wal_default(
            "overwrite lacks direct snapshot-commit authority, so it must be replayable from WAL",
            self.default_wal_guards(intent),
            vec![RejectedWriteLane::new(
                WriteLane::OverwriteSnapshotCommit,
                "direct snapshot commit requires explicit direct or external authority",
            )],
        )
    }

    fn route_append(&self, intent: &WriteIntent) -> WriteLaneDecision {
        let direct_allowed = matches!(
            intent.durability,
            WriteDurabilityRequirement::DirectCommitAllowed
                | WriteDurabilityRequirement::ExternalAuthoritative
        );
        let can_bulk_append = direct_allowed
            && intent.batch_local_constraints_sufficient
            && intent.has_bulk_hint(&self.config)
            && !intent.published_parent_has_deletes;

        if can_bulk_append {
            let mut guards = vec![
                WriteGuard::CatalogSchemaValidation,
                WriteGuard::TenantRlsCoercion,
                WriteGuard::BatchLocalConstraintCheck,
                WriteGuard::AtomicSnapshotManifestCommit,
                WriteGuard::IdempotencyKey,
            ];
            if matches!(
                intent.durability,
                WriteDurabilityRequirement::ExternalAuthoritative
            ) {
                guards.push(WriteGuard::ExternalAuthorityBoundary);
            }

            return WriteLaneDecision::new(
                WriteLane::BulkAppendCommit,
                "append-like write has bulk hints and direct commit authority",
                guards,
                vec![RejectedWriteLane::new(
                    WriteLane::OverwriteSnapshotCommit,
                    "append preserves the active snapshot instead of replacing it",
                )],
            );
        }

        let mut rejected = Vec::new();
        if !direct_allowed {
            rejected.push(RejectedWriteLane::new(
                WriteLane::BulkAppendCommit,
                "bulk append direct commit requires explicit direct or external authority",
            ));
        }
        if !intent.batch_local_constraints_sufficient {
            rejected.push(RejectedWriteLane::new(
                WriteLane::BulkAppendCommit,
                "bulk append requires constraints that can be validated batch-locally",
            ));
        }
        if !intent.has_bulk_hint(&self.config) {
            rejected.push(RejectedWriteLane::new(
                WriteLane::BulkAppendCommit,
                "write is below configured bulk row and byte thresholds",
            ));
        }
        if intent.published_parent_has_deletes {
            rejected.push(RejectedWriteLane::new(
                WriteLane::BulkAppendCommit,
                "published parent snapshot carries deletion vectors; the bulk lane \
                 cannot record the layout flip the visibility fence requires",
            ));
        }

        self.route_wal_default(
            "append-like write does not satisfy all bulk direct-commit guards",
            self.default_wal_guards(intent),
            rejected,
        )
    }

    fn route_wal_default(
        &self,
        reason: impl Into<String>,
        guards: Vec<WriteGuard>,
        rejected_lanes: Vec<RejectedWriteLane>,
    ) -> WriteLaneDecision {
        WriteLaneDecision::new(WriteLane::WalCurrentState, reason, guards, rejected_lanes)
    }

    fn default_wal_guards(&self, intent: &WriteIntent) -> Vec<WriteGuard> {
        let mut guards = vec![
            WriteGuard::CatalogSchemaValidation,
            WriteGuard::TenantRlsCoercion,
            WriteGuard::ConstraintValidation,
            WriteGuard::CanonicalWalAppend,
            WriteGuard::CurrentStateUpdate,
        ];
        if intent.operation_kind.requires_row_level_mvcc() || intent.requires_row_level_semantics {
            guards.push(WriteGuard::RowLevelMvcc);
        }
        if matches!(
            intent.projection_freshness,
            ProjectionFreshnessRequirement::BestEffort
                | ProjectionFreshnessRequirement::ReadYourWrites
                | ProjectionFreshnessRequirement::Synchronous
        ) {
            guards.push(WriteGuard::ProjectionFreshnessMetadata);
        }
        if matches!(
            intent.durability,
            WriteDurabilityRequirement::ExternalAuthoritative
        ) {
            guards.push(WriteGuard::ExternalAuthorityBoundary);
        }
        guards
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_intent_router_keeps_upsert_on_wal_current_state() {
        let decision = WriteLaneRouter::new().route(&WriteIntent::new(
            "public.accounts",
            WriteOperationKind::Upsert,
        ));

        assert_eq!(decision.lane, WriteLane::WalCurrentState);
        assert!(decision.required_guards.contains(&WriteGuard::RowLevelMvcc));
        assert!(
            decision
                .rejected_lanes
                .iter()
                .any(|rejected| rejected.lane == WriteLane::BulkAppendCommit)
        );
    }

    #[test]
    fn write_intent_router_allows_guarded_bulk_append_commit() {
        let intent = WriteIntent::new("public.events", WriteOperationKind::Append)
            .with_durability(WriteDurabilityRequirement::DirectCommitAllowed)
            .with_row_count_hint(DEFAULT_BULK_ROW_THRESHOLD)
            .with_batch_local_constraints_sufficient(true)
            .with_idempotency_key("load-42");

        let decision = WriteLaneRouter::new().route(&intent);

        assert_eq!(decision.lane, WriteLane::BulkAppendCommit);
        assert!(
            decision
                .required_guards
                .contains(&WriteGuard::AtomicSnapshotManifestCommit)
        );
        assert!(
            decision
                .required_guards
                .contains(&WriteGuard::BatchLocalConstraintCheck)
        );
    }

    #[test]
    fn write_intent_router_refuses_bulk_above_deletion_vector_parent() {
        // The visibility fence contract's forbid decision (review finding 1):
        // the bulk lane takes a successor slot without the publish lock and
        // never records the layout flip, so it must not run above a
        // DV-carrying published parent.
        let intent = WriteIntent::new("public.events", WriteOperationKind::Append)
            .with_durability(WriteDurabilityRequirement::DirectCommitAllowed)
            .with_row_count_hint(DEFAULT_BULK_ROW_THRESHOLD)
            .with_batch_local_constraints_sufficient(true)
            .with_idempotency_key("load-43")
            .with_published_parent_has_deletes(true);

        let decision = WriteLaneRouter::new().route(&intent);

        assert_eq!(decision.lane, WriteLane::WalCurrentState);
        assert!(decision.rejected_lanes.iter().any(|rejected| {
            rejected.lane == WriteLane::BulkAppendCommit
                && rejected.reason.contains("deletion vectors")
        }));
    }

    #[test]
    fn write_intent_router_rejects_bulk_append_without_batch_local_constraints() {
        let intent = WriteIntent::new("public.events", WriteOperationKind::Append)
            .with_durability(WriteDurabilityRequirement::DirectCommitAllowed)
            .with_row_count_hint(DEFAULT_BULK_ROW_THRESHOLD);

        let decision = WriteLaneRouter::new().route(&intent);

        assert_eq!(decision.lane, WriteLane::WalCurrentState);
        assert!(decision.rejected_lanes.iter().any(|rejected| {
            rejected.lane == WriteLane::BulkAppendCommit
                && rejected.reason.contains("batch-locally")
        }));
    }

    #[test]
    fn write_intent_router_routes_delete_to_wal_current_state() {
        let decision = WriteLaneRouter::new().route(&WriteIntent::new(
            "public.accounts",
            WriteOperationKind::Delete,
        ));

        assert_eq!(decision.lane, WriteLane::WalCurrentState);
        assert!(
            decision
                .required_guards
                .contains(&WriteGuard::CanonicalWalAppend)
        );
        assert!(
            decision
                .required_guards
                .contains(&WriteGuard::CurrentStateUpdate)
        );
        assert!(decision.required_guards.contains(&WriteGuard::RowLevelMvcc));
    }

    #[test]
    fn write_intent_router_allows_guarded_overwrite_snapshot_commit() {
        let intent = WriteIntent::new("public.fact_sales", WriteOperationKind::OverwritePartitions)
            .with_durability(WriteDurabilityRequirement::DirectCommitAllowed)
            .with_idempotency_key("overwrite-2026-05-20");

        let decision = WriteLaneRouter::new().route(&intent);

        assert_eq!(decision.lane, WriteLane::OverwriteSnapshotCommit);
        assert!(
            decision
                .required_guards
                .contains(&WriteGuard::AtomicSnapshotManifestCommit)
        );
        assert!(
            decision
                .required_guards
                .contains(&WriteGuard::IdempotencyKey)
        );
    }

    #[test]
    fn write_intent_router_routes_projection_refresh_to_projection_only() {
        let intent = WriteIntent::new("public.events_pax", WriteOperationKind::ProjectionRefresh)
            .with_durability(WriteDurabilityRequirement::ProjectionOnly);

        let decision = WriteLaneRouter::new().route(&intent);

        assert_eq!(decision.lane, WriteLane::ProjectionOnly);
        assert!(
            decision
                .required_guards
                .contains(&WriteGuard::ProjectionFreshnessMetadata)
        );
        assert!(
            !decision
                .required_guards
                .contains(&WriteGuard::CanonicalWalAppend)
        );
    }

    #[test]
    fn require_wal_lane_passes_for_wal_current_state() {
        let decision =
            WriteLaneRouter::new().route(&WriteIntent::new("t", WriteOperationKind::Insert));
        assert_eq!(decision.lane, WriteLane::WalCurrentState);
        assert!(decision.require_wal_lane("test adapter").is_ok());
    }

    #[test]
    fn require_wal_lane_rejects_projection_only() {
        let intent = WriteIntent::new("t", WriteOperationKind::ProjectionRefresh)
            .with_durability(WriteDurabilityRequirement::ProjectionOnly);
        let decision = WriteLaneRouter::new().route(&intent);
        assert_eq!(decision.lane, WriteLane::ProjectionOnly);
        let result = decision.require_wal_lane("test adapter");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("ProjectionOnly"),
            "error should name the lane: {msg}"
        );
    }

    #[test]
    fn require_wal_lane_rejects_overwrite_snapshot_commit() {
        let intent = WriteIntent::new("t", WriteOperationKind::OverwriteTable)
            .with_durability(WriteDurabilityRequirement::DirectCommitAllowed)
            .with_idempotency_key("snap-001");
        let decision = WriteLaneRouter::new().route(&intent);
        assert_eq!(decision.lane, WriteLane::OverwriteSnapshotCommit);
        let result = decision.require_wal_lane("Arrow Flight DoPut");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("OverwriteSnapshotCommit")
        );
    }
}
