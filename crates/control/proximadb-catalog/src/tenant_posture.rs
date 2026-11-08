// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-SEC-2 Slice C — **per-tenant security posture**.
//!
//! ADR-090 L1.2 shipped grant enforcement behind ONE process-global env gate
//! (`PROXIMADB_AUTHZ_REQUIRE_GRANTS`). That is the wrong shape for SaaS: a
//! multi-tenant operator cannot flag-day every customer onto strict
//! authorization simultaneously. Tenants must be onboarded individually, and
//! they need a *rehearsal* step before enforcement can break their traffic.
//!
//! This module makes the posture a per-tenant, durable record:
//!
//! | mode | grants evaluated? | effect on a read |
//! |---|---|---|
//! | [`GrantEnforcement::Off`] | no | admitted (policy-only admission) |
//! | [`GrantEnforcement::Audit`] | **yes** | **admitted**, and a would-be denial is reported |
//! | [`GrantEnforcement::Enforce`] | yes | denied when no grant admits |
//!
//! `Audit` is the ramp that makes onboarding safe: the operator provisions
//! grants, watches the would-be-denial signal fall to zero for that tenant, and
//! only then flips to `Enforce`. Without it the only options are "unenforced"
//! and "possibly break production", which is why an env flag alone was
//! insufficient.
//!
//! **The env gate is retained as the DEFAULT for tenants with no explicit
//! record** — so an existing deployment behaves exactly as before, and the
//! per-tenant record is a targeted override on top.
//!
//! Persistence deliberately reuses the mechanism proven by
//! [`crate::principal_registry`] and [`crate::grants`]: a JSON snapshot,
//! load-on-open, atomic tmp+rename+fsync, held behind a shared `Arc` so an
//! admin write is visible to the live enforcer without a restart. The posture
//! is a *record*, not a bool, so later fields (mfa_required,
//! allowed_auth_methods, max_key_age) have a home that already exists.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::fc_metamodel::TenantStableId;

const POSTURE_FILE: &str = "tenant-posture.json";

/// How grant entitlements gate reads for one tenant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GrantEnforcement {
    /// Grants are not consulted; admission is policy-only (pre-ADR-090 L1.2).
    #[default]
    Off,
    /// Grants ARE evaluated and a missing entitlement is *reported*, but the
    /// read is still admitted. The safe onboarding rehearsal.
    Audit,
    /// Grants are required: no applicable grant ⇒ deny.
    Enforce,
}

/// The per-tenant security posture record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TenantSecurityPosture {
    pub tenant_stable_id: TenantStableId,
    #[serde(default)]
    pub grant_enforcement: GrantEnforcement,
    pub updated_at_ms: i64,
}

/// The outcome of consulting the posture at an enforcement seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostureDecision {
    /// Do not consult grants at all.
    Skip,
    /// Consult grants; on "no applicable grant", ADMIT but report.
    AuditOnly,
    /// Consult grants; on "no applicable grant", DENY.
    Enforce,
}

impl GrantEnforcement {
    pub fn decision(self) -> PostureDecision {
        match self {
            Self::Off => PostureDecision::Skip,
            Self::Audit => PostureDecision::AuditOnly,
            Self::Enforce => PostureDecision::Enforce,
        }
    }
}

#[derive(Debug)]
pub enum PostureStoreError {
    Io(String),
    Serde(String),
}

impl fmt::Display for PostureStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "posture store io error: {e}"),
            Self::Serde(e) => write!(f, "posture store serde error: {e}"),
        }
    }
}

impl std::error::Error for PostureStoreError {}

/// Durable per-tenant posture store.
pub struct FileSystemTenantPostureStore {
    dir: PathBuf,
    state: RwLock<BTreeMap<TenantStableId, TenantSecurityPosture>>,
}

impl FileSystemTenantPostureStore {
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, PostureStoreError> {
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(|e| PostureStoreError::Io(e.to_string()))?;
        let state: BTreeMap<TenantStableId, TenantSecurityPosture> =
            match fs::read(dir.join(POSTURE_FILE)) {
                Ok(bytes) => serde_json::from_slice(&bytes)
                    .map_err(|e| PostureStoreError::Serde(e.to_string()))?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
                Err(e) => return Err(PostureStoreError::Io(e.to_string())),
            };
        Ok(Self {
            dir,
            state: RwLock::new(state),
        })
    }

    /// Set (or replace) a tenant's posture.
    pub fn set(
        &self,
        tenant_stable_id: TenantStableId,
        grant_enforcement: GrantEnforcement,
        now_ms: i64,
    ) -> Result<(), PostureStoreError> {
        let mut state = write_lock(&self.state);
        state.insert(
            tenant_stable_id,
            TenantSecurityPosture {
                tenant_stable_id,
                grant_enforcement,
                updated_at_ms: now_ms,
            },
        );
        self.persist(&state)
    }

    /// The tenant's explicit posture, if one has been set.
    pub fn get(&self, tenant_stable_id: TenantStableId) -> Option<TenantSecurityPosture> {
        read_lock(&self.state).get(&tenant_stable_id).cloned()
    }

    pub fn list(&self) -> Vec<TenantSecurityPosture> {
        read_lock(&self.state).values().cloned().collect()
    }

    /// Resolve the enforcement mode for a tenant: the explicit record if there
    /// is one, else `default_mode` (which the caller derives from the process
    /// env gate, preserving existing deployment behavior exactly).
    pub fn resolve(
        &self,
        tenant_stable_id: TenantStableId,
        default_mode: GrantEnforcement,
    ) -> GrantEnforcement {
        self.get(tenant_stable_id)
            .map(|p| p.grant_enforcement)
            .unwrap_or(default_mode)
    }

    fn persist(
        &self,
        state: &BTreeMap<TenantStableId, TenantSecurityPosture>,
    ) -> Result<(), PostureStoreError> {
        let bytes = serde_json::to_vec_pretty(state)
            .map_err(|e| PostureStoreError::Serde(e.to_string()))?;
        let tmp = self.dir.join(format!("{POSTURE_FILE}.tmp"));
        {
            let mut f = fs::File::create(&tmp).map_err(|e| PostureStoreError::Io(e.to_string()))?;
            f.write_all(&bytes)
                .map_err(|e| PostureStoreError::Io(e.to_string()))?;
            f.sync_all()
                .map_err(|e| PostureStoreError::Io(e.to_string()))?;
        }
        fs::rename(&tmp, self.dir.join(POSTURE_FILE))
            .map_err(|e| PostureStoreError::Io(e.to_string()))?;
        Ok(())
    }
}

use crate::lock_util::{read_lock, write_lock};

// ---------------------------------------------------------------------------
// Specification (TD-SEC-2 Slice C). Written before the seam wiring.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    const A: TenantStableId = 1;
    const B: TenantStableId = 2;

    fn store() -> (tempfile::TempDir, FileSystemTenantPostureStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = FileSystemTenantPostureStore::open(dir.path()).expect("open");
        (dir, s)
    }

    /// A tenant with no record inherits the process default — this is what
    /// keeps existing deployments behaving EXACTLY as before when the posture
    /// store is introduced.
    #[test]
    fn absent_record_falls_back_to_the_process_default() {
        let (_d, s) = store();
        assert_eq!(s.resolve(A, GrantEnforcement::Off), GrantEnforcement::Off);
        assert_eq!(
            s.resolve(A, GrantEnforcement::Enforce),
            GrantEnforcement::Enforce,
            "an operator who armed the env gate keeps enforcement for \
             tenants that have no explicit posture"
        );
    }

    /// The whole point of the slice: posture is PER TENANT, so onboarding one
    /// tenant to Enforce leaves every other tenant untouched.
    #[test]
    fn posture_is_per_tenant_not_process_global() {
        let (_d, s) = store();
        s.set(A, GrantEnforcement::Enforce, 1).expect("set");
        assert_eq!(
            s.resolve(A, GrantEnforcement::Off),
            GrantEnforcement::Enforce
        );
        assert_eq!(
            s.resolve(B, GrantEnforcement::Off),
            GrantEnforcement::Off,
            "tenant B must NOT be dragged into enforcement by tenant A"
        );
    }

    /// An explicit record OVERRIDES the process default in both directions —
    /// including opting a tenant OUT while the env gate is armed globally.
    #[test]
    fn explicit_record_overrides_the_default_in_both_directions() {
        let (_d, s) = store();
        s.set(A, GrantEnforcement::Off, 1).expect("set");
        assert_eq!(
            s.resolve(A, GrantEnforcement::Enforce),
            GrantEnforcement::Off,
            "a tenant can be held back from a globally-armed rollout"
        );
        s.set(A, GrantEnforcement::Enforce, 2).expect("set");
        assert_eq!(
            s.resolve(A, GrantEnforcement::Off),
            GrantEnforcement::Enforce
        );
    }

    /// The three modes map to distinct seam behaviors; `Audit` evaluates grants
    /// but does NOT deny — that is what makes it a safe rehearsal.
    #[test]
    fn audit_mode_evaluates_without_denying() {
        assert_eq!(GrantEnforcement::Off.decision(), PostureDecision::Skip);
        assert_eq!(
            GrantEnforcement::Audit.decision(),
            PostureDecision::AuditOnly
        );
        assert_eq!(
            GrantEnforcement::Enforce.decision(),
            PostureDecision::Enforce
        );
    }

    /// Posture survives a restart (it is a durable security decision, not a
    /// process-lifetime toggle).
    #[test]
    fn posture_survives_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let s = FileSystemTenantPostureStore::open(dir.path()).expect("open");
            s.set(A, GrantEnforcement::Enforce, 1).expect("set");
            s.set(B, GrantEnforcement::Audit, 1).expect("set");
        }
        let s = FileSystemTenantPostureStore::open(dir.path()).expect("reopen");
        assert_eq!(
            s.resolve(A, GrantEnforcement::Off),
            GrantEnforcement::Enforce
        );
        assert_eq!(s.resolve(B, GrantEnforcement::Off), GrantEnforcement::Audit);
        assert_eq!(s.list().len(), 2);
    }
}
