// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! ADR-090 L1.1 — the `Grant` catalog object: the ENTITLEMENT half of
//! authorization, and the one place cross-tenant sharing is expressible.
//!
//! Division of labour with the existing FC metamodel:
//! * [`crate::fc_metamodel::PolicyBinding`] is the *constraint* layer —
//!   subjectless, deny-biased restrictions scoped to a tenant's own resources.
//! * [`GrantRecord`] is the *entitlement* layer — "this grantee may perform
//!   these actions on this resource", where the grantee may be a **foreign
//!   tenant or a foreign tenant's user**. Ownership stays structural (the
//!   resource lives under its owner's path and bill); access is granted.
//!
//! Multi-modality falls out of resource choice, not code: grants scope over
//! [`crate::fc_metamodel::Scope`] — the same `Namespace` / `Table(collection)`
//! / `Column` lattice policy uses — and a collection id is the modality-
//! agnostic unit (vector, graph, document, timeseries, relational alike).
//! `scope_covers` is REUSED from the metamodel, not reimplemented, so grant
//! and policy coverage can never drift.
//!
//! Contract (the tests below are the specification):
//! * **Fail-closed**: no applicable grant ⇒ `Deny(NoApplicableGrant)`.
//!   Revoked and expired grants are not applicable.
//! * **Deny > absence > grant**: [`compose_with_policy`] makes a policy deny
//!   veto any grant (ADR-090 L1 composition rule).
//! * **Foreign grantees admit** — the invariant the pre-ADR-090 model made
//!   impossible by construction at four layers. A grant naming
//!   `Grantee::User { tenant_stable_id: B, .. }` admits that user of tenant B
//!   to tenant A's resource; every *other* subject of tenant B stays denied.
//! * Persistence is the OSS mechanism shared with the ABAC stores and the
//!   principal registry: JSON snapshot, atomic tmp+rename+fsync,
//!   load-on-open, synchronous persist, partitioned by **owner** tenant.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::RwLock;

use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::fc_metamodel::{FieldMask, ObjectId, Scope, SubjectId, Target, TenantStableId};

const GRANTS_FILE: &str = "grants.json";

/// Who a grant admits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grantee {
    /// Every subject of the named tenant.
    Tenant(TenantStableId),
    /// One specific user of the named tenant (which may be — and for sharing,
    /// is — a different tenant from the resource owner).
    User {
        tenant_stable_id: TenantStableId,
        subject: SubjectId,
    },
}

/// What a grant permits. Deliberately coarse; row-level nuance rides the
/// attached `predicate_ref`, not extra verbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GrantAction {
    Read,
    Write,
    Ddl,
    /// Permission to further delegate (re-grant). Not consulted by
    /// [`evaluate_grants`] for data access; the admin surface checks it.
    Grant,
}

/// A durable entitlement, owned (and revocable) by the resource's tenant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrantRecord {
    pub grant_id: String,
    /// The RESOURCE OWNER — the partition key. Only the owner's store slice
    /// is consulted when their resource is opened, mirroring the structural
    /// tenant wall everywhere else.
    pub owner_tenant_stable_id: TenantStableId,
    pub resource: Scope,
    pub grantee: Grantee,
    pub actions: BTreeSet<GrantAction>,
    /// Optional row-filter predicate attached to the share (same predicate-
    /// object space the policy layer dereferences).
    pub predicate_ref: Option<ObjectId>,
    pub field_mask: Option<FieldMask>,
    pub created_at_ms: i64,
    pub expires_at_ms: Option<i64>,
    pub revoked_at_ms: Option<i64>,
}

/// The acting subject, as resolved by L0 identity: a user of some tenant.
#[derive(Debug, Clone, PartialEq)]
pub struct GrantSubject<'a> {
    pub tenant_stable_id: TenantStableId,
    pub subject: &'a str,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GrantDecision {
    /// Admitted. Predicate refs / masks from every applicable grant are
    /// surfaced for the read path to AND into the plan (same compile path as
    /// policy predicates).
    Permit {
        predicate_refs: Vec<ObjectId>,
        field_masks: Vec<FieldMask>,
    },
    Deny(GrantDenyReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrantDenyReason {
    /// Fail-closed default: nothing applicable admitted the subject.
    NoApplicableGrant,
    /// A policy-layer deny vetoed the grant (deny > absence > grant).
    PolicyDeny,
}

/// Pure decision: which of `grants` admit `subject` to `target` for `action`
/// at time `now_ms`. Fail-closed on absence; revoked/expired never apply.
pub fn evaluate_grants(
    grants: &[GrantRecord],
    subject: &GrantSubject<'_>,
    target: &Target,
    action: GrantAction,
    now_ms: i64,
) -> GrantDecision {
    let mut predicate_refs = Vec::new();
    let mut field_masks = Vec::new();
    let mut applicable = 0usize;

    for grant in grants {
        if grant.revoked_at_ms.is_some() {
            continue;
        }
        if let Some(exp) = grant.expires_at_ms
            && now_ms >= exp
        {
            continue;
        }
        if !grant.actions.contains(&action) {
            continue;
        }
        if !crate::fc_metamodel::scope_covers(&grant.resource, target) {
            continue;
        }
        let admits = match &grant.grantee {
            Grantee::Tenant(t) => *t == subject.tenant_stable_id,
            Grantee::User {
                tenant_stable_id,
                subject: s,
            } => *tenant_stable_id == subject.tenant_stable_id && s.0 == subject.subject,
        };
        if !admits {
            continue;
        }
        applicable += 1;
        if let Some(p) = grant.predicate_ref {
            predicate_refs.push(p);
        }
        if let Some(m) = grant.field_mask {
            field_masks.push(m);
        }
    }

    if applicable == 0 {
        GrantDecision::Deny(GrantDenyReason::NoApplicableGrant)
    } else {
        GrantDecision::Permit {
            predicate_refs,
            field_masks,
        }
    }
}

/// The decision at the **collection-open seam** when the acting identity does not
/// own the collection (ADR-090 L1.2 / TD-AUTHZ-1 item 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossTenantOpen {
    /// The acting tenant OWNS the collection — this path does not apply, and the
    /// caller must use its ordinary owner-scoped resolution. Kept distinct from
    /// `Admitted` so an owner can never be mistaken for a grantee in an audit
    /// record.
    NotApplicable,
    /// A grant admits the acting identity to the owner's collection.
    Admitted,
    /// No applicable grant. The caller MUST fail closed exactly as it does today.
    Denied,
}

/// Decide whether `acting` may open a collection owned by `owner`.
///
/// This is the seam that makes cross-tenant sharing observable: today a foreign
/// grantee is rejected by the owner-scoped catalog lookup before any row filter
/// runs, so a grant that the catalog correctly evaluates has no visible effect.
///
/// # The escalation rule this function exists to enforce
///
/// The collection-open seam guards **reads and writes alike** — in the vector
/// service the same validator fronts `search`/`get` AND
/// `insert_batch`/`insert_records_only`. So the action is a REQUIRED parameter,
/// never defaulted: a `Read` grant must not admit an INSERT. Callers pass the
/// action they are actually performing, and a grant admits only if it carries
/// that exact action.
///
/// # Cost
///
/// Only reached when the owner-scoped lookup already MISSED, so the ordinary
/// same-tenant path pays nothing. Cross-tenant opens cost one grant-slice scan,
/// evaluated once per open rather than per row — the ADR-090 L1 rule that
/// sharing is a catalog decision, not a query predicate.
pub fn cross_tenant_open_decision(
    owner_grants: &[GrantRecord],
    acting: &GrantSubject<'_>,
    owner: TenantStableId,
    resource: &Target,
    action: GrantAction,
    now_ms: i64,
) -> CrossTenantOpen {
    if acting.tenant_stable_id == owner {
        return CrossTenantOpen::NotApplicable;
    }
    match evaluate_grants(owner_grants, acting, resource, action, now_ms) {
        GrantDecision::Permit { .. } => CrossTenantOpen::Admitted,
        GrantDecision::Deny(_) => CrossTenantOpen::Denied,
    }
}

/// ADR-090 L1 composition: **deny > absence-of-grant > grant**. A policy-layer
/// deny vetoes any entitlement.
pub fn compose_with_policy(policy_denied: bool, grants: GrantDecision) -> GrantDecision {
    if policy_denied {
        GrantDecision::Deny(GrantDenyReason::PolicyDeny)
    } else {
        grants
    }
}

#[derive(Debug)]
pub enum GrantStoreError {
    UnknownGrant { grant_id: String },
    Io(String),
    Serde(String),
}

impl fmt::Display for GrantStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownGrant { grant_id } => write!(f, "no grant with id '{grant_id}'"),
            Self::Io(e) => write!(f, "grant store io error: {e}"),
            Self::Serde(e) => write!(f, "grant store serde error: {e}"),
        }
    }
}

impl std::error::Error for GrantStoreError {}

/// Durable grant store, partitioned by owner tenant. Same OSS persistence
/// mechanism as the ABAC stores and the principal registry.
pub struct FileSystemGrantStore {
    dir: PathBuf,
    state: RwLock<BTreeMap<TenantStableId, Vec<GrantRecord>>>,
}

impl FileSystemGrantStore {
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, GrantStoreError> {
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(|e| GrantStoreError::Io(e.to_string()))?;
        let state: BTreeMap<TenantStableId, Vec<GrantRecord>> =
            match fs::read(dir.join(GRANTS_FILE)) {
                Ok(bytes) => serde_json::from_slice(&bytes)
                    .map_err(|e| GrantStoreError::Serde(e.to_string()))?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
                Err(e) => return Err(GrantStoreError::Io(e.to_string())),
            };
        Ok(Self {
            dir,
            state: RwLock::new(state),
        })
    }

    /// Mint and persist a grant on behalf of `owner`. Returns the grant id.
    #[allow(clippy::too_many_arguments)]
    pub fn grant(
        &self,
        owner: TenantStableId,
        resource: Scope,
        grantee: Grantee,
        actions: BTreeSet<GrantAction>,
        predicate_ref: Option<ObjectId>,
        field_mask: Option<FieldMask>,
        expires_at_ms: Option<i64>,
    ) -> Result<String, GrantStoreError> {
        let mut id_bytes = [0u8; 8];
        rand::rngs::OsRng.fill_bytes(&mut id_bytes);
        let grant_id: String = {
            let mut s = String::with_capacity(16);
            for b in id_bytes {
                use fmt::Write as _;
                let _ = write!(s, "{b:02x}");
            }
            s
        };
        let record = GrantRecord {
            grant_id: grant_id.clone(),
            owner_tenant_stable_id: owner,
            resource,
            grantee,
            actions,
            predicate_ref,
            field_mask,
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            expires_at_ms,
            revoked_at_ms: None,
        };
        let mut state = write_lock(&self.state);
        state.entry(owner).or_default().push(record);
        self.persist(&state)?;
        Ok(grant_id)
    }

    /// Revoke by owner + grant id. Only the owner's slice is searched — a
    /// tenant cannot revoke (or even name) another owner's grants.
    pub fn revoke(&self, owner: TenantStableId, grant_id: &str) -> Result<(), GrantStoreError> {
        let mut state = write_lock(&self.state);
        let found = state
            .get_mut(&owner)
            .and_then(|v| v.iter_mut().find(|g| g.grant_id == grant_id));
        match found {
            Some(g) => g.revoked_at_ms = Some(chrono::Utc::now().timestamp_millis()),
            None => {
                return Err(GrantStoreError::UnknownGrant {
                    grant_id: grant_id.to_string(),
                });
            }
        }
        self.persist(&state)
    }

    /// The owner's grant slice — the input to [`evaluate_grants`] when one of
    /// the owner's resources is opened.
    pub fn grants_for_owner(&self, owner: TenantStableId) -> Vec<GrantRecord> {
        read_lock(&self.state)
            .get(&owner)
            .cloned()
            .unwrap_or_default()
    }

    fn persist(
        &self,
        state: &BTreeMap<TenantStableId, Vec<GrantRecord>>,
    ) -> Result<(), GrantStoreError> {
        let bytes =
            serde_json::to_vec_pretty(state).map_err(|e| GrantStoreError::Serde(e.to_string()))?;
        let tmp = self.dir.join(format!("{GRANTS_FILE}.tmp"));
        {
            let mut f = fs::File::create(&tmp).map_err(|e| GrantStoreError::Io(e.to_string()))?;
            f.write_all(&bytes)
                .map_err(|e| GrantStoreError::Io(e.to_string()))?;
            f.sync_all()
                .map_err(|e| GrantStoreError::Io(e.to_string()))?;
        }
        fs::rename(&tmp, self.dir.join(GRANTS_FILE))
            .map_err(|e| GrantStoreError::Io(e.to_string()))?;
        Ok(())
    }
}

use crate::lock_util::{read_lock, write_lock};

// ---------------------------------------------------------------------------
// The specification (ADR-090 L1.1 / TD-AUTHZ-1). Written before the
// implementation; each test names the contract clause it pins.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    const OWNER: TenantStableId = 7;
    const FOREIGN: TenantStableId = 9;

    fn target(ns: u16, table: u32) -> Target {
        Target {
            namespace: ns,
            table,
            column: None,
        }
    }

    fn subject(tenant: TenantStableId, s: &'static str) -> GrantSubject<'static> {
        GrantSubject {
            tenant_stable_id: tenant,
            subject: s,
        }
    }

    fn read_only() -> BTreeSet<GrantAction> {
        BTreeSet::from([GrantAction::Read])
    }

    fn now() -> i64 {
        chrono::Utc::now().timestamp_millis()
    }

    fn store() -> (tempfile::TempDir, FileSystemGrantStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = FileSystemGrantStore::open(dir.path()).expect("open");
        (dir, s)
    }

    /// Fail-closed default: an empty grant slice denies.
    #[test]
    fn absence_of_grant_denies() {
        let d = evaluate_grants(
            &[],
            &subject(OWNER, "alice"),
            &target(1, 10),
            GrantAction::Read,
            now(),
        );
        assert_eq!(d, GrantDecision::Deny(GrantDenyReason::NoApplicableGrant));
    }

    /// ⭐ THE ADR-090 invariant the old model made impossible: a grant naming a
    /// FOREIGN tenant's user admits exactly that user — and no other subject
    /// of the foreign tenant.
    #[test]
    fn foreign_grantee_admits_and_only_that_grantee() {
        let (_d, s) = store();
        s.grant(
            OWNER,
            Scope::Table(10),
            Grantee::User {
                tenant_stable_id: FOREIGN,
                subject: SubjectId("bob".into()),
            },
            read_only(),
            Some(42), // shared with a row-filter attached
            None,
            None,
        )
        .expect("grant");
        let grants = s.grants_for_owner(OWNER);

        // bob@FOREIGN is admitted, and the share's predicate ref surfaces.
        match evaluate_grants(
            &grants,
            &subject(FOREIGN, "bob"),
            &target(1, 10),
            GrantAction::Read,
            now(),
        ) {
            GrantDecision::Permit { predicate_refs, .. } => assert_eq!(predicate_refs, vec![42]),
            other => panic!("foreign grantee must admit, got {other:?}"),
        }
        // mallory@FOREIGN is NOT.
        assert_eq!(
            evaluate_grants(
                &grants,
                &subject(FOREIGN, "mallory"),
                &target(1, 10),
                GrantAction::Read,
                now()
            ),
            GrantDecision::Deny(GrantDenyReason::NoApplicableGrant)
        );
        // and bob is admitted to THIS table only.
        assert_eq!(
            evaluate_grants(
                &grants,
                &subject(FOREIGN, "bob"),
                &target(1, 11),
                GrantAction::Read,
                now()
            ),
            GrantDecision::Deny(GrantDenyReason::NoApplicableGrant)
        );
    }

    /// A tenant-wide grant admits every subject of the grantee tenant.
    #[test]
    fn tenant_grant_covers_all_its_subjects() {
        let (_d, s) = store();
        s.grant(
            OWNER,
            Scope::Table(10),
            Grantee::Tenant(FOREIGN),
            read_only(),
            None,
            None,
            None,
        )
        .expect("grant");
        let grants = s.grants_for_owner(OWNER);
        for who in ["bob", "carol"] {
            assert!(matches!(
                evaluate_grants(
                    &grants,
                    &subject(FOREIGN, who),
                    &target(1, 10),
                    GrantAction::Read,
                    now()
                ),
                GrantDecision::Permit { .. }
            ));
        }
        // but not subjects of an unrelated tenant
        assert!(matches!(
            evaluate_grants(
                &grants,
                &subject(11, "bob"),
                &target(1, 10),
                GrantAction::Read,
                now()
            ),
            GrantDecision::Deny(_)
        ));
    }

    /// Lifecycle: grant → permit; revoke → deny. Revocation persists.
    #[test]
    fn lifecycle_grant_then_revoke_then_deny_and_it_persists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let id = {
            let s = FileSystemGrantStore::open(dir.path()).expect("open");
            let id = s
                .grant(
                    OWNER,
                    Scope::Table(10),
                    Grantee::Tenant(FOREIGN),
                    read_only(),
                    None,
                    None,
                    None,
                )
                .expect("grant");
            assert!(matches!(
                evaluate_grants(
                    &s.grants_for_owner(OWNER),
                    &subject(FOREIGN, "bob"),
                    &target(1, 10),
                    GrantAction::Read,
                    now()
                ),
                GrantDecision::Permit { .. }
            ));
            s.revoke(OWNER, &id).expect("revoke");
            id
        };
        // reopen from disk: revocation survived
        let s = FileSystemGrantStore::open(dir.path()).expect("reopen");
        let grants = s.grants_for_owner(OWNER);
        assert_eq!(grants.len(), 1, "revoked grants stay listed");
        assert!(grants[0].revoked_at_ms.is_some());
        assert_eq!(grants[0].grant_id, id);
        assert!(matches!(
            evaluate_grants(
                &grants,
                &subject(FOREIGN, "bob"),
                &target(1, 10),
                GrantAction::Read,
                now()
            ),
            GrantDecision::Deny(_)
        ));
        // revoking under the WRONG owner cannot touch it
        assert!(matches!(
            s.revoke(FOREIGN, &id),
            Err(GrantStoreError::UnknownGrant { .. })
        ));
    }

    /// Expiry is honored at evaluation time.
    #[test]
    fn expired_grant_denies() {
        let (_d, s) = store();
        s.grant(
            OWNER,
            Scope::Table(10),
            Grantee::Tenant(FOREIGN),
            read_only(),
            None,
            None,
            Some(now() - 1_000),
        )
        .expect("grant");
        assert!(matches!(
            evaluate_grants(
                &s.grants_for_owner(OWNER),
                &subject(FOREIGN, "bob"),
                &target(1, 10),
                GrantAction::Read,
                now()
            ),
            GrantDecision::Deny(GrantDenyReason::NoApplicableGrant)
        ));
    }

    /// Actions are scoped: Read does not smuggle Write or DDL.
    #[test]
    fn actions_are_scoped() {
        let (_d, s) = store();
        s.grant(
            OWNER,
            Scope::Table(10),
            Grantee::Tenant(FOREIGN),
            read_only(),
            None,
            None,
            None,
        )
        .expect("grant");
        let grants = s.grants_for_owner(OWNER);
        for denied in [GrantAction::Write, GrantAction::Ddl, GrantAction::Grant] {
            assert!(matches!(
                evaluate_grants(
                    &grants,
                    &subject(FOREIGN, "bob"),
                    &target(1, 10),
                    denied,
                    now()
                ),
                GrantDecision::Deny(_)
            ));
        }
    }

    /// A namespace-scoped grant covers every collection in it — the same
    /// `scope_covers` lattice policy uses, hence uniformly multi-modal (a
    /// Table id is a collection of ANY modality).
    #[test]
    fn namespace_grant_covers_its_collections_only() {
        let (_d, s) = store();
        s.grant(
            OWNER,
            Scope::Namespace(2),
            Grantee::Tenant(FOREIGN),
            read_only(),
            None,
            None,
            None,
        )
        .expect("grant");
        let grants = s.grants_for_owner(OWNER);
        assert!(matches!(
            evaluate_grants(
                &grants,
                &subject(FOREIGN, "bob"),
                &target(2, 10),
                GrantAction::Read,
                now()
            ),
            GrantDecision::Permit { .. }
        ));
        assert!(matches!(
            evaluate_grants(
                &grants,
                &subject(FOREIGN, "bob"),
                &target(3, 10),
                GrantAction::Read,
                now()
            ),
            GrantDecision::Deny(_)
        ));
    }

    /// The acting tenant OWNS the collection ⇒ this seam does not apply, even
    /// when a grant would also admit. Owners must never be recorded as grantees.
    #[test]
    fn owner_is_not_applicable_not_admitted() {
        let (_d, s) = store();
        s.grant(
            OWNER,
            Scope::Table(10),
            Grantee::Tenant(OWNER),
            read_only(),
            None,
            None,
            None,
        )
        .expect("grant");
        assert_eq!(
            cross_tenant_open_decision(
                &s.grants_for_owner(OWNER),
                &subject(OWNER, "alice"),
                OWNER,
                &target(1, 10),
                GrantAction::Read,
                now(),
            ),
            CrossTenantOpen::NotApplicable
        );
    }

    /// ⭐ THE ESCALATION TEST. The collection-open seam fronts reads AND writes
    /// (search/get alongside insert_batch/insert_records_only), so a Read grant
    /// must NOT admit a Write open. If this ever passes as Admitted, a shared
    /// read has silently become shared write access.
    #[test]
    fn read_grant_never_admits_a_write_open() {
        let (_d, s) = store();
        s.grant(
            OWNER,
            Scope::Table(10),
            Grantee::User {
                tenant_stable_id: FOREIGN,
                subject: SubjectId("bob".into()),
            },
            read_only(),
            None,
            None,
            None,
        )
        .expect("grant");
        let grants = s.grants_for_owner(OWNER);
        let bob = subject(FOREIGN, "bob");

        assert_eq!(
            cross_tenant_open_decision(
                &grants,
                &bob,
                OWNER,
                &target(1, 10),
                GrantAction::Read,
                now()
            ),
            CrossTenantOpen::Admitted,
            "the granted action opens"
        );
        for escalated in [GrantAction::Write, GrantAction::Ddl, GrantAction::Grant] {
            assert_eq!(
                cross_tenant_open_decision(&grants, &bob, OWNER, &target(1, 10), escalated, now()),
                CrossTenantOpen::Denied,
                "a Read grant must never admit {escalated:?}"
            );
        }
    }

    /// Absent, revoked, and expired grants all leave the seam CLOSED — the
    /// caller's existing rejection is preserved unchanged.
    #[test]
    fn seam_stays_closed_without_a_live_grant() {
        let (_d, s) = store();
        let bob = subject(FOREIGN, "bob");
        let t = target(1, 10);

        assert_eq!(
            cross_tenant_open_decision(&[], &bob, OWNER, &t, GrantAction::Read, now()),
            CrossTenantOpen::Denied,
            "no grants at all"
        );

        let id = s
            .grant(
                OWNER,
                Scope::Table(10),
                Grantee::Tenant(FOREIGN),
                read_only(),
                None,
                None,
                None,
            )
            .expect("grant");
        assert_eq!(
            cross_tenant_open_decision(
                &s.grants_for_owner(OWNER),
                &bob,
                OWNER,
                &t,
                GrantAction::Read,
                now()
            ),
            CrossTenantOpen::Admitted
        );
        s.revoke(OWNER, &id).expect("revoke");
        assert_eq!(
            cross_tenant_open_decision(
                &s.grants_for_owner(OWNER),
                &bob,
                OWNER,
                &t,
                GrantAction::Read,
                now()
            ),
            CrossTenantOpen::Denied,
            "revocation closes the seam"
        );

        let (_d2, s2) = store();
        s2.grant(
            OWNER,
            Scope::Table(10),
            Grantee::Tenant(FOREIGN),
            read_only(),
            None,
            None,
            Some(now() - 1),
        )
        .expect("grant");
        assert_eq!(
            cross_tenant_open_decision(
                &s2.grants_for_owner(OWNER),
                &bob,
                OWNER,
                &t,
                GrantAction::Read,
                now()
            ),
            CrossTenantOpen::Denied,
            "expiry closes the seam"
        );
    }

    /// Deny > absence > grant: a policy deny vetoes an otherwise-valid grant.
    #[test]
    fn policy_deny_overrides_grant() {
        let permit = GrantDecision::Permit {
            predicate_refs: vec![],
            field_masks: vec![],
        };
        assert_eq!(
            compose_with_policy(true, permit),
            GrantDecision::Deny(GrantDenyReason::PolicyDeny)
        );
        // and composition is a no-op when policy does not deny
        let permit = GrantDecision::Permit {
            predicate_refs: vec![1],
            field_masks: vec![],
        };
        assert_eq!(compose_with_policy(false, permit.clone()), permit);
    }
}
