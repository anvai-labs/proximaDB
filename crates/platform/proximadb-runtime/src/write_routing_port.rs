//! Write-routing + recall-probe ports (ADR-094) — the cluster-side gates the
//! v2 record handlers consult, expressed as runtime ports so the converged
//! handlers carry no cluster/catalog dependencies.
//!
//! Single-node builds run with `None`: the ADR-084 semantics are that the
//! primary-pod registry and the recall probe are multi-node/operational
//! affordances — `consult_for_write` absent ⇒ the write is allowed (the
//! historical single-node behavior), the probe absent ⇒ `None` (probe closed,
//! reported in route-explain rather than erroring).

use async_trait::async_trait;

/// Outcome of the primary-pod write-routing consultation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteRoutingDecision {
    /// This pod serves the `(tenant, collection)` write.
    Allow,
    /// Another pod is primary; the client SDK should retry against
    /// `target_pod`.
    Misrouted { target_pod: String },
}

/// Primary-pod write-routing gate (tenant-pod-affinity Slice 4).
pub trait WriteRoutingPort: Send + Sync {
    /// Consult the registry for `(tenant_id, collection)`. Implementations
    /// record the allowed/bounded/misrouted metrics.
    fn consult_for_write(&self, tenant_id: &str, collection: &str) -> WriteRoutingDecision;

    /// This pod's identity (echoed in misroute logs).
    fn self_pod_id(&self) -> String;
}

/// Recall-probe gate (experimental AXIS diagnostics): `Some(open)` when the
/// gate is registered for the `(tenant, collection)` scope, `None` when closed
/// for the scope — surfaced via route-explain's `RECALL_PROBE_CLOSED` hint on
/// debug requests only.
#[async_trait]
pub trait RecallProbePort: Send + Sync {
    async fn probe_open(&self, tenant_id: &str, collection_id: &str) -> Option<bool>;
}
