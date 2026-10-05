//! Shared REST application state.
//!
//! `RestAppState` is the Axum state type injected into every platform REST handler.
//! It holds only port traits from `proximadb-runtime` so no root-crate concrete types
//! cross the crate boundary.

use std::sync::Arc;

use proximadb_runtime::{ApiHandlersPort, RecallProbePort, WriteRoutingPort};

/// Tenant context extracted from request headers/JWT and injected as an Axum Extension.
///
/// Handlers receive this via `Extension(tenant): Extension<TenantContext>`.  The
/// `tenant_id` is passed as `Option<&str>` to port methods which resolve it internally.
#[derive(Debug, Clone)]
pub struct TenantContext {
    pub tenant_id: String,
    /// FA-2 PR-D3: the ABAC binding-filter key, populated by the root tenant
    /// middleware at injection time. `None` on unauthenticated/test paths.
    pub tenant_stable_id: Option<u64>,
}

impl TenantContext {
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            tenant_stable_id: None,
        }
    }

    /// Convenience for tests and middleware that need a default tenant.
    pub fn default_tenant() -> Self {
        Self::new("default")
    }

    /// Set the ABAC stable tenant id (root middleware injects this from the
    /// resolved identity).
    pub fn with_tenant_stable_id(mut self, id: u64) -> Self {
        self.tenant_stable_id = Some(id);
        self
    }
}

/// Axum application state shared by all platform REST v1 handlers.
///
/// All service dependencies are expressed as port traits so the handlers can
/// be compiled independently of root-crate concrete service types.
#[derive(Clone)]
pub struct RestAppState {
    /// Primary API port — collection, vector, hybrid, and SQL operations.
    pub handlers: Arc<dyn ApiHandlersPort>,
    /// Primary-pod write-routing gate. `None` ⇒ single-node: writes allowed
    /// unconditionally (ADR-084 — the registry is a multi-node affordance).
    pub write_routing: Option<Arc<dyn WriteRoutingPort>>,
    /// Recall-probe gate (experimental AXIS diagnostics). `None` ⇒ closed.
    pub recall_probe: Option<Arc<dyn RecallProbePort>>,
}

impl RestAppState {
    pub fn new(handlers: Arc<dyn ApiHandlersPort>) -> Self {
        Self {
            handlers,
            write_routing: None,
            recall_probe: None,
        }
    }

    /// Wire the primary-pod write-routing gate (multi-node deployments).
    pub fn with_write_routing(mut self, gate: Arc<dyn WriteRoutingPort>) -> Self {
        self.write_routing = Some(gate);
        self
    }

    /// Wire the recall-probe gate (experimental AXIS diagnostics).
    pub fn with_recall_probe(mut self, gate: Arc<dyn RecallProbePort>) -> Self {
        self.recall_probe = Some(gate);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenant_context_preserves_supplied_tenant_id() {
        let tenant = TenantContext::new("tenant-a");

        assert_eq!(tenant.tenant_id, "tenant-a");
    }

    #[test]
    fn default_tenant_context_uses_default_authority_scope() {
        let tenant = TenantContext::default_tenant();

        assert_eq!(tenant.tenant_id, "default");
    }
}
