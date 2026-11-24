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

//! # Auto-Tiering Policy Engine
//!
//! **Status**: Policy engine implemented. SST integration module available (opt-in).
//!
//! The tiering policy engine provides data movement logic between storage tiers.
//! Storage engine integration is available through the SST tiering integration module.
//!
//! ## What Works
//! - Policy definition and evaluation
//! - Access pattern tracking
//! - Migration task generation
//! - Retention policy management
//! - **SST Integration Module** (`storage::engines::sst::tiering_integration`)
//!
//! ## SST Tiering Integration
//!
//! The SST engine has a tiering integration module that provides:
//! - Access tracking hooks for the search path
//! - Flush tier determination for new data placement
//! - Compaction-time tier evaluation
//! - Policy-based migration task generation
//!
//! Enable via configuration:
//! ```toml
//! [storage.sst.tiering]
//! enabled = true
//! evaluation_interval_secs = 300
//! cold_age_threshold_days = 7
//! hot_access_threshold = 100
//! ```
//!
//! ## Remaining Integration Work
//!
//! 1. ✅ Access tracking wired into SST search path
//!    (`src/storage/engines/sst/search/coordinator.rs:78-100` — calls
//!    `record_access` on every result; landed 2026-05-27)
//! 2. ✅ Tier determination wired into SST flush path
//!    (`src/storage/engines/sst/flush/coordinator.rs:47-63` — calls
//!    `determine_flush_tier` pre-flush, decision currently advisory;
//!    landed 2026-05-27)
//! 3. ✅ Tier evaluation wired into SST compaction trigger
//!    (`src/storage/engines/sst/flush/coordinator.rs:130-170` — calls
//!    `evaluate_collection` when flush triggers compaction; landed
//!    2026-05-27)
//! 4. ✅ Migration executor for physical byte-movement between tier
//!    paths (`src/storage/tiering/executor.rs`). Routes through
//!    `FilesystemFactory::move_atomic`, so file://↔s3://↔gs:// all
//!    work via the same code path. Includes idempotency check (crash
//!    after copy / before delete → next attempt completes the delete)
//!    and a bounded-concurrency batch executor.
//! 5. ✅ Policy engine wired to executor (`TieringPolicyEngine.with_executor`).
//!    The background eval loop now invokes `executor.execute_batch`
//!    on every cycle and records each `MigrationResult` via
//!    `record_migration_complete`. `SharedServices` bootstrap
//!    constructs the executor from the same `SstTieringConfig` and
//!    attaches it before `start()`, so an operator who sets
//!    `enabled = true` in TOML gets actual byte movement — no
//!    follow-up code change required.
//! 6. ✅ Prometheus metrics (`src/metrics/tier_migration_metrics.rs`):
//!    `proximadb_tier_migrations_total`,
//!    `proximadb_tier_migration_bytes_total`,
//!    `proximadb_tier_migration_duration_seconds`, and
//!    `proximadb_tier_migration_in_flight`. Emitted from
//!    `TierMigrationExecutor::execute` on every call with RAII-guarded
//!    in-flight gauge — survives panics. See the module doc for
//!    label cardinality and bucket layout.
//!
//! When `SstEngine.with_tiering_integration` is set, items 1–3 fire on
//! their natural cadences. When unset (the default), all three become
//! cheap `if-let-Some` no-ops — no behavior change for legacy callers.
//!
//! See `src/storage/engines/sst/tiering_integration.rs` for the detailed
//! deferred-design comments around physical migration.
//!
//! ---
//!
//! ## Legacy Documentation (Archived)
//!
//! Provides automatic data movement between storage tiers based on configurable policies.
//! Supports hot/warm/cold/archive tiers with rules based on:
//! - Age (time since last modification)
//! - Access patterns (frequency, recency)
//! - Size thresholds
//! - Cost optimization
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     TieringPolicyEngine                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
//! │  │   Policies  │  │  Evaluator  │  │  Migration Coordinator  │ │
//! │  │   (Rules)   │  │  (Scoring)  │  │     (Async Moves)       │ │
//! │  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘ │
//! ├─────────┼─────────────────┼─────────────────────┼───────────────┤
//! │         │                 │                     │               │
//! │   ┌─────▼─────┐     ┌─────▼─────┐         ┌─────▼─────┐        │
//! │   │ AccessLog │     │ TierCost  │         │ EnginePool │       │
//! │   │  Tracker  │     │ Calculator│         │ (SST)      │       │
//! │   └───────────┘     └───────────┘         └───────────┘        │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use proximadb::storage::tiering::{TieringPolicyEngine, TieringPolicy, TieringRule};
//!
//! let engine = TieringPolicyEngine::new(config)
//!     .with_policy(TieringPolicy::age_based("default", Duration::days(7), PerformanceTier::Cold))
//!     .with_policy(TieringPolicy::access_based("hot-keep", 100, PerformanceTier::Hot));
//!
//! engine.start().await?;
//! ```

pub mod engine;
/// Tier migration executor — physical byte-movement between tier paths.
/// Closes the last deferred item from the tier-migration pipeline.
pub mod executor;
pub mod migration;
pub mod policy;
pub mod retention;
pub mod tracker;

pub use engine::{TieringEngineConfig, TieringPolicyEngine, TieringStats};
pub use executor::{MigrationExecutionError, TierMigrationExecutor, apply_result_to_task};
pub use migration::{MigrationResult, MigrationStatus, MigrationTask};
pub use policy::{PerformanceTier, PolicyAction, PolicyCondition, TieringPolicy, TieringRule};
pub use retention::{
    ArchiveConfig, ArchiveDestinationType, RetentionAction, RetentionCondition, RetentionManager,
    RetentionManagerConfig, RetentionMetadata, RetentionPolicy, RetentionRule, RetentionStats,
};
pub use tracker::{AccessEvent, AccessPattern, AccessTracker};
