//! Prometheus bridge for the catalog crate's `DrMetrics` trait.
//!
//! Lives in the root crate because the catalog crate stays free of
//! the `prometheus` dependency (per the OSS/SaaS split — catalog is
//! lower in the layering graph). This module ships
//! [`PrometheusDrMetrics`], the concrete sink the server wires into
//! `DrReconcilerShard::with_metrics` at startup.
//!
//! Metric families exposed at `/metrics/prometheus`:
//! - `proximadb_dr_ticks_total{tier, provider, outcome}` — one
//!   increment per `observe_tick` call. The `outcome` label has
//!   bounded cardinality (Reconciled-*, DeferredBackoff,
//!   DeferredRateLimit, DeferredLeaseHeldElsewhere, SkippedShardPaused,
//!   EscalatedAfterMaxAttempts).
//! - `proximadb_dr_drift_total{provider, reason}` — drift detections
//!   and repairs grouped by reason (DriftReason enum variants).
//! - `proximadb_dr_reconciler_errors_total{provider, reason}` —
//!   transient + non-retryable adapter errors and max-attempt
//!   escalations.
//! - `proximadb_dr_reconciler_paused_total{reason}` — shard pause
//!   transitions (currently always `quota_exceeded`).
//!
//! Cardinality stays bounded by the contract: 5 tier strings, 4
//! `ObjectProvider` variants, ~10 outcome strings, ~5 reason strings.
//! Total per-family cardinality is single-digit thousands worst case.

use crate::{register_counter_vec_safe, register_gauge_vec_safe};
use lazy_static::lazy_static;
use prometheus::{CounterVec, GaugeVec};
use proximadb_catalog::collection_dr_policy::ObjectProvider;
use proximadb_catalog::dr_reconciler::{
    BlockReason, DrMetrics, DriftReason, IdleReason, PolicyLabels, ReconcileOutcome,
    ShardPauseReason, TickOutcome,
};

lazy_static! {
    /// Per-tick outcome counter. The `outcome` label captures every
    /// `TickOutcome` variant; the `tier` and `provider` labels mirror
    /// the policy. Reading: rate(proximadb_dr_ticks_total[5m]) by
    /// (outcome) shows reconciler throughput broken down by what
    /// actually happened to each policy.
    pub static ref DR_TICKS_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_dr_ticks_total",
        "Total DR reconciler ticks by outcome.",
        &["tier", "provider", "outcome"]
    );

    /// Drift detections and repairs. Bumped by `MarkedDrifted` and
    /// `RepairedDrift` outcomes.
    pub static ref DR_DRIFT_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_dr_drift_total",
        "Total DR drift detections / repairs by reason.",
        &["provider", "reason"]
    );

    /// Adapter errors and max-attempt escalations. Use this for SLO
    /// burn-rate alerting on the reconciler.
    pub static ref DR_RECONCILER_ERRORS_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_dr_reconciler_errors_total",
        "Total DR reconciler errors by reason.",
        &["provider", "reason"]
    );

    /// Shard pause transitions. Fires once per pause event (not on
    /// every tick during the pause window).
    pub static ref DR_RECONCILER_PAUSED_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_dr_reconciler_paused_total",
        "Total DR shard pause events by reason.",
        &["reason"]
    );

    /// Last observed provider lag in seconds, keyed on the matching
    /// `{provider, region_pair}` label set per contract
    /// §"Observability". The reconciler updates this gauge whenever
    /// `fetch_state` returns a non-None `observed_lag_seconds`. Use
    /// `histogram_quantile`/`max_over_time` on this family for SLO
    /// alerts.
    pub static ref DR_PROVIDER_LAG_SECONDS: GaugeVec = register_gauge_vec_safe(
        "proximadb_dr_provider_lag_seconds",
        "Last observed DR replication lag in seconds.",
        &["provider", "region_pair"]
    );
}

/// Prometheus-backed `DrMetrics`. Wire one at startup and hand it to
/// every `DrReconcilerShard::with_metrics`. Cloning is cheap — the
/// struct is zero-sized; all state lives in the static `CounterVec`
/// registry.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrometheusDrMetrics;

impl PrometheusDrMetrics {
    pub const fn new() -> Self {
        Self
    }
}

impl DrMetrics for PrometheusDrMetrics {
    fn observe_tick(&self, labels: &PolicyLabels, outcome: &TickOutcome) {
        let provider = provider_label(labels.provider);
        let outcome_label = tick_outcome_label(outcome);
        DR_TICKS_TOTAL
            .with_label_values(&[labels.tier.as_str(), provider, outcome_label])
            .inc();

        match outcome {
            TickOutcome::Reconciled(ReconcileOutcome::MarkedDrifted(reason))
            | TickOutcome::Reconciled(ReconcileOutcome::RepairedDrift(reason)) => {
                DR_DRIFT_TOTAL
                    .with_label_values(&[provider, drift_reason_label(*reason)])
                    .inc();
            }
            TickOutcome::Reconciled(ReconcileOutcome::AdapterTransient(_)) => {
                DR_RECONCILER_ERRORS_TOTAL
                    .with_label_values(&[provider, "transient"])
                    .inc();
            }
            TickOutcome::Reconciled(ReconcileOutcome::AdapterEscalated(reason))
            | TickOutcome::Reconciled(ReconcileOutcome::MarkedProviderBlocked(reason)) => {
                DR_RECONCILER_ERRORS_TOTAL
                    .with_label_values(&[provider, block_reason_label(*reason)])
                    .inc();
            }
            TickOutcome::EscalatedAfterMaxAttempts => {
                DR_RECONCILER_ERRORS_TOTAL
                    .with_label_values(&[provider, "max_attempts_exceeded"])
                    .inc();
            }
            _ => {}
        }
    }

    fn observe_shard_paused(&self, reason: ShardPauseReason) {
        DR_RECONCILER_PAUSED_TOTAL
            .with_label_values(&[shard_pause_reason_label(reason)])
            .inc();
    }

    fn observe_lag(&self, labels: &PolicyLabels, lag_seconds: u32) {
        DR_PROVIDER_LAG_SECONDS
            .with_label_values(&[
                provider_label(labels.provider),
                labels.region_pair_id.as_str(),
            ])
            .set(lag_seconds as f64);
    }
}

// ---------------------------------------------------------------------------
// Label helpers — all return &'static str so Prometheus label sets
// stay deduplicated.
// ---------------------------------------------------------------------------

fn provider_label(p: ObjectProvider) -> &'static str {
    match p {
        ObjectProvider::AwsS3 => "aws_s3",
        ObjectProvider::AzureBlob => "azure_blob",
        ObjectProvider::AzureAdlsHns => "azure_adls_hns",
        ObjectProvider::GcsFuture => "gcs_future",
    }
}

fn tick_outcome_label(o: &TickOutcome) -> &'static str {
    match o {
        TickOutcome::DeferredBackoff => "deferred_backoff",
        TickOutcome::DeferredRateLimit => "deferred_rate_limit",
        TickOutcome::DeferredLeaseHeldElsewhere { .. } => "deferred_lease",
        TickOutcome::SkippedShardPaused => "skipped_shard_paused",
        TickOutcome::EscalatedAfterMaxAttempts => "escalated_after_max_attempts",
        TickOutcome::Reconciled(r) => reconcile_outcome_label(r),
    }
}

fn reconcile_outcome_label(o: &ReconcileOutcome) -> &'static str {
    match o {
        ReconcileOutcome::Idle(reason) => match reason {
            IdleReason::Disabled => "idle_disabled",
            IdleReason::AwaitingBillingApproval => "idle_awaiting_billing",
            IdleReason::SuspendedByOps => "idle_suspended",
            IdleReason::Retired => "idle_retired",
            IdleReason::HealthyActive => "idle_healthy",
        },
        ReconcileOutcome::EnsuredRule { .. } => "ensured_rule",
        ReconcileOutcome::Retired => "retired",
        ReconcileOutcome::RepairedDrift(_) => "repaired_drift",
        ReconcileOutcome::MarkedDrifted(_) => "marked_drifted",
        ReconcileOutcome::MarkedProviderBlocked(_) => "marked_provider_blocked",
        ReconcileOutcome::MarkedBillingBlocked(_) => "marked_billing_blocked",
        ReconcileOutcome::AdapterTransient(_) => "adapter_transient",
        ReconcileOutcome::AdapterEscalated(_) => "adapter_escalated",
    }
}

fn drift_reason_label(r: DriftReason) -> &'static str {
    match r {
        DriftReason::RuleMissing => "rule_missing",
        DriftReason::RuleDisabled => "rule_disabled",
        DriftReason::PrefixMismatch => "prefix_mismatch",
        DriftReason::DestinationMismatch => "destination_mismatch",
        DriftReason::UnknownProviderRule => "unknown_provider_rule",
        DriftReason::SourceVersioningDisabled => "source_versioning_disabled",
        DriftReason::DestinationWritable => "destination_writable",
        DriftReason::KmsBindingChanged => "kms_binding_changed",
    }
}

fn block_reason_label(r: BlockReason) -> &'static str {
    match r {
        BlockReason::BillingApprovalMissing => "billing_approval_missing",
        BlockReason::CostOwnerMissing => "cost_owner_missing",
        BlockReason::ProviderMisconfiguration => "provider_misconfiguration",
        BlockReason::ProviderQuotaExceeded => "provider_quota_exceeded",
        BlockReason::ProviderAuthDenied => "provider_auth_denied",
    }
}

fn shard_pause_reason_label(r: ShardPauseReason) -> &'static str {
    match r {
        ShardPauseReason::QuotaExceeded => "quota_exceeded",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_catalog::StoragePoolClass;
    use proximadb_catalog::collection_dr_policy::CollectionDrPolicy;
    use proximadb_catalog::collection_dr_policy::{
        DrBillingBinding, DrHealth, DrPlacement, DrReplicationBehavior, DrState,
    };
    use proximadb_catalog::dr_reconciler::PolicyLabels;

    fn sample_labels() -> PolicyLabels {
        let p = CollectionDrPolicy {
            policy_id: "drp_1".into(),
            tenant_id: "tnt_acme".into(),
            namespace_id: "ns_1".into(),
            collection_id: "col_orders".into(),
            tier: "business".into(),
            state: DrState::Active,
            provider: ObjectProvider::AwsS3,
            source_region: "us-east-1".into(),
            destination_region: "us-west-2".into(),
            region_pair_id: "aws:us-east-1:us-west-2".into(),
            placement: DrPlacement {
                source_pool_class: StoragePoolClass::Standard,
                destination_pool_class: StoragePoolClass::Standard,
                source_bucket_or_account: "src".into(),
                destination_bucket_or_account: "dst".into(),
                source_container: None,
                destination_container: None,
                source_prefix: "data/tnt_acme/ns_1/col_orders/".into(),
                destination_prefix: "data/tnt_acme/ns_1/col_orders/".into(),
            },
            replication: DrReplicationBehavior::default(),
            billing: DrBillingBinding {
                cost_binding_ref: "dr-standard-binding".into(),
                cost_owner_tenant_id: "tnt_acme".into(),
                billing_approval_id: Some("a".into()),
                operator_estimate_cents: None,
            },
            provider_binding: None,
            health: DrHealth::default(),
            requested_by: "u".into(),
            approved_by: None,
            created_at_ns: 0,
            updated_at_ns: 0,
            policy_version: 1,
        };
        PolicyLabels::from_policy(&p)
    }

    fn ticks_count(tier: &str, provider: &str, outcome: &str) -> f64 {
        DR_TICKS_TOTAL
            .with_label_values(&[tier, provider, outcome])
            .get()
    }

    fn drift_count(provider: &str, reason: &str) -> f64 {
        DR_DRIFT_TOTAL.with_label_values(&[provider, reason]).get()
    }

    fn error_count(provider: &str, reason: &str) -> f64 {
        DR_RECONCILER_ERRORS_TOTAL
            .with_label_values(&[provider, reason])
            .get()
    }

    fn pause_count(reason: &str) -> f64 {
        DR_RECONCILER_PAUSED_TOTAL
            .with_label_values(&[reason])
            .get()
    }

    #[test]
    fn registry_populates_without_panic() {
        // Touching the lazy_statics is the registration test. If
        // anything mis-registered, the lazy_init would have panicked
        // when first accessed.
        let _ = &*DR_TICKS_TOTAL;
        let _ = &*DR_DRIFT_TOTAL;
        let _ = &*DR_RECONCILER_ERRORS_TOTAL;
        let _ = &*DR_RECONCILER_PAUSED_TOTAL;
    }

    #[test]
    fn observe_tick_increments_outcome_counter() {
        let m = PrometheusDrMetrics::new();
        let l = sample_labels();
        let before = ticks_count("business", "aws_s3", "deferred_backoff");
        m.observe_tick(&l, &TickOutcome::DeferredBackoff);
        let after = ticks_count("business", "aws_s3", "deferred_backoff");
        assert!((after - before - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn observe_tick_drift_increments_drift_counter() {
        let m = PrometheusDrMetrics::new();
        let l = sample_labels();
        let before = drift_count("aws_s3", "destination_mismatch");
        let outcome = TickOutcome::Reconciled(ReconcileOutcome::MarkedDrifted(
            DriftReason::DestinationMismatch,
        ));
        m.observe_tick(&l, &outcome);
        let after = drift_count("aws_s3", "destination_mismatch");
        assert!((after - before - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn observe_tick_repair_increments_both_tick_and_drift() {
        let m = PrometheusDrMetrics::new();
        let l = sample_labels();
        let ticks_before = ticks_count("business", "aws_s3", "repaired_drift");
        let drift_before = drift_count("aws_s3", "rule_missing");
        let outcome =
            TickOutcome::Reconciled(ReconcileOutcome::RepairedDrift(DriftReason::RuleMissing));
        m.observe_tick(&l, &outcome);
        let ticks_after = ticks_count("business", "aws_s3", "repaired_drift");
        let drift_after = drift_count("aws_s3", "rule_missing");
        assert!((ticks_after - ticks_before - 1.0).abs() < f64::EPSILON);
        assert!((drift_after - drift_before - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn observe_tick_transient_increments_error_counter() {
        let m = PrometheusDrMetrics::new();
        let l = sample_labels();
        let before = error_count("aws_s3", "transient");
        let outcome = TickOutcome::Reconciled(ReconcileOutcome::AdapterTransient("blip".into()));
        m.observe_tick(&l, &outcome);
        let after = error_count("aws_s3", "transient");
        assert!((after - before - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn observe_tick_provider_blocked_increments_error_with_reason() {
        let m = PrometheusDrMetrics::new();
        let l = sample_labels();
        let before = error_count("aws_s3", "provider_quota_exceeded");
        let outcome = TickOutcome::Reconciled(ReconcileOutcome::AdapterEscalated(
            BlockReason::ProviderQuotaExceeded,
        ));
        m.observe_tick(&l, &outcome);
        let after = error_count("aws_s3", "provider_quota_exceeded");
        assert!((after - before - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn observe_tick_escalated_max_attempts_increments_error() {
        let m = PrometheusDrMetrics::new();
        let l = sample_labels();
        let before = error_count("aws_s3", "max_attempts_exceeded");
        m.observe_tick(&l, &TickOutcome::EscalatedAfterMaxAttempts);
        let after = error_count("aws_s3", "max_attempts_exceeded");
        assert!((after - before - 1.0).abs() < f64::EPSILON);
    }

    fn lag_gauge_value(provider: &str, region_pair: &str) -> f64 {
        DR_PROVIDER_LAG_SECONDS
            .with_label_values(&[provider, region_pair])
            .get()
    }

    #[test]
    fn observe_lag_sets_gauge_with_provider_and_region_pair_labels() {
        let m = PrometheusDrMetrics::new();
        let l = sample_labels();
        m.observe_lag(&l, 73);
        assert_eq!(lag_gauge_value("aws_s3", "aws:us-east-1:us-west-2"), 73.0,);
        // Subsequent observations overwrite — this is a gauge, not
        // a counter, so the "last observed value" semantics apply.
        m.observe_lag(&l, 5);
        assert_eq!(lag_gauge_value("aws_s3", "aws:us-east-1:us-west-2"), 5.0,);
    }

    #[test]
    fn observe_shard_paused_increments_pause_counter() {
        let m = PrometheusDrMetrics::new();
        let before = pause_count("quota_exceeded");
        m.observe_shard_paused(ShardPauseReason::QuotaExceeded);
        let after = pause_count("quota_exceeded");
        assert!((after - before - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn idle_outcomes_only_increment_ticks_not_drift_or_error() {
        let m = PrometheusDrMetrics::new();
        let l = sample_labels();
        let ticks_before = ticks_count("business", "aws_s3", "idle_healthy");
        let drift_before = drift_count("aws_s3", "rule_missing");
        let err_before = error_count("aws_s3", "transient");
        let outcome = TickOutcome::Reconciled(ReconcileOutcome::Idle(IdleReason::HealthyActive));
        m.observe_tick(&l, &outcome);
        let ticks_after = ticks_count("business", "aws_s3", "idle_healthy");
        let drift_after = drift_count("aws_s3", "rule_missing");
        let err_after = error_count("aws_s3", "transient");
        assert!((ticks_after - ticks_before - 1.0).abs() < f64::EPSILON);
        assert_eq!(drift_after, drift_before);
        assert_eq!(err_after, err_before);
    }
}
