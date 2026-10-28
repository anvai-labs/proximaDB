// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! The shared per-collection write-lease authority (ADR-032).
//!
//! One pod may write or publish a collection at a time: the partition lease
//! when the lease system is on, else the in-memory primary-pod registry. DDL
//! checks it before executing; the live delete publisher checks it
//! immediately before its successor CAS (the visibility fence contract's
//! phase-3 lease — gate 4: a pod that lost its lease cannot publish). Both
//! publication lanes consult the SAME authority value, wired once in
//! `shared_services`, so they cannot disagree about who may publish.

use std::sync::Arc;

use anyhow::{Result, anyhow};

/// An owned snapshot of the lease triple: the manager and registry are
/// `Arc`s, the pod id a `String`, so cloning is cheap and the value can be
/// handed to services that never see each other's handles.
#[derive(Clone, Default)]
pub struct WriteLeaseAuthority {
    partition_lease_manager: Option<Arc<crate::cluster::partition_lease::PartitionLeaseManager>>,
    primary_pod_registry: Option<Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>>,
    self_pod_id: Option<String>,
}

impl WriteLeaseAuthority {
    /// The full authority: lease manager (optional — absent means in-memory
    /// routing only), primary-pod registry, and this pod's identity.
    pub fn new(
        partition_lease_manager: Option<
            Arc<crate::cluster::partition_lease::PartitionLeaseManager>,
        >,
        primary_pod_registry: Option<Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>>,
        self_pod_id: Option<String>,
    ) -> Self {
        Self {
            partition_lease_manager,
            primary_pod_registry,
            self_pod_id,
        }
    }

    /// Inert: every check allows. Embedded and single-pod deployments, and
    /// tests that do not exercise routing.
    pub fn inert() -> Self {
        Self::default()
    }

    /// Whether the wiring is inert (no registry + pod id): inert wiring
    /// allows everything, by design — there is no other pod to defer to.
    pub fn is_inert(&self) -> bool {
        !matches!(
            (&self.primary_pod_registry, &self.self_pod_id),
            (Some(_), Some(_))
        )
    }

    /// Check that THIS pod currently holds write authority for the
    /// (tenant, collection). `caller` names the checking lane in error
    /// messages ("DDL", "live delete") so a misroute names its refuser.
    ///
    /// Inert unless a registry + pod id are wired (the essentials for any
    /// routing decision). The lease manager is optional: present ⇒ lease
    /// acquire/renew; absent ⇒ in-memory registry lookup only.
    pub async fn check(
        &self,
        caller: &str,
        tenant: Option<&str>,
        collection_id: &str,
    ) -> Result<()> {
        use crate::cluster::primary_pod_registry::{
            WriteRoutingDecision, consult_for_write_leased,
        };

        let (registry, pod_id) = match (&self.primary_pod_registry, &self.self_pod_id) {
            (Some(r), Some(p)) => (r, p),
            _ => return Ok(()),
        };
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let decision = consult_for_write_leased(
            registry,
            self.partition_lease_manager.as_deref(),
            pod_id,
            tenant.unwrap_or("default"),
            collection_id,
            now_ms,
        )
        .await
        .map_err(|e| anyhow!("{caller} lease check failed: {e}"))?;
        match decision {
            WriteRoutingDecision::Allow => Ok(()),
            WriteRoutingDecision::Misrouted { target_pod } => Err(anyhow!(
                "{caller} misrouted: write for collection '{collection_id}' must go to pod \
                 '{target_pod}'"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the DDL lease tests: inert wiring allows; a registry whose
    /// primary is another pod refuses; the refusal names the caller lane.
    #[tokio::test]
    async fn write_lease_allows_when_inert() {
        let authority = WriteLeaseAuthority::inert();
        assert!(authority.is_inert());
        authority
            .check("live delete", Some("tenant-a"), "t")
            .await
            .expect("inert wiring allows every write");
    }

    #[tokio::test]
    async fn write_lease_refuses_when_primary_is_another_pod() {
        use crate::cluster::primary_pod_registry::{AssignmentReason, PrimaryPodRegistry};

        let registry = Arc::new(PrimaryPodRegistry::new());
        registry.assign("tenant-a", "t", "other-pod", AssignmentReason::Create);
        let authority =
            WriteLeaseAuthority::new(None, Some(registry), Some("this-pod".to_string()));
        assert!(!authority.is_inert());
        let err = authority
            .check("live delete", Some("tenant-a"), "t")
            .await
            .expect_err("misrouted write must refuse");
        let msg = format!("{err:#}");
        assert!(msg.contains("live delete misrouted"), "{msg}");
        assert!(msg.contains("other-pod"), "{msg}");
    }
}
