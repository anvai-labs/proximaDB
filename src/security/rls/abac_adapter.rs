// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! The ABAC-to-scan-predicate adapter: turns compiled `ObjectId` refs into a
//! per-row `Fn(&ProximaRecord) -> bool` that `scan_records_filtered` ANDs into
//! its existing `RecordScanPredicate`.
//!
//! This is the **scan-path integration point** (FA-c Phase 2b/2c). It compiles
//! the `AuthorizedReadContext`'s `row_predicate_refs` once (at request-scope),
//! then the returned closure evaluates each record under the strict 3-valued
//! walker via [`admits_with_security`](super::filter_lattice::admits_with_security).

#[cfg(feature = "abac-policy")]
use std::sync::Arc;

#[cfg(feature = "abac-policy")]
use crate::core::search::{
    FilterExpression, OptimizedSearchRecord, sql_value_filter::proxima_value_to_filter_literal,
};
#[cfg(feature = "abac-policy")]
use crate::security::rls::filter_lattice::admits_with_security;
#[cfg(feature = "abac-policy")]
use proximadb_abac::{
    AttributeAuthority, AuthorizedReadContext, DenyReason, PolicyBindingStore, PolicyEpochSource,
    PredicateObjectStore, compile_security_filter,
};
#[cfg(feature = "abac-policy")]
use proximadb_catalog::fc_metamodel::{ObjectId, PolicyBinding, SubjectId, Target};
#[cfg(feature = "abac-policy")]
use proximadb_records::{ProximaRecord, ProximaTreeNode};

/// Compile `refs` into a per-row predicate for `scan_records_filtered`.
///
/// Returns `None` when the refs are empty (no row restriction) or when the
/// compiled expression is `None`. Returns `Some(closure)` that evaluates each
/// `ProximaRecord` under the strict security walker. A missing predicate ref
/// (fail-closed in `compile_security_filter`) yields an unsatisfiable closure
/// that denies every row.
///
/// The closure owns the compiled `FilterExpression`; it is `Send + Sync` and
/// can be passed as a `RecordScanPredicate` to `scan_records_filtered`.
/// FA-c Phase 2c wires this into `scan_records_filtered`'s predicate param; until
/// then it has no production caller (the feature is default-OFF and the function is
/// the integration point the wiring step consumes).
#[cfg(feature = "abac-policy")]
#[allow(dead_code)]
pub fn abac_scan_predicate(
    refs: &[ObjectId],
    store: &dyn PredicateObjectStore,
    subject: &proximadb_catalog::fc_metamodel::SubjectAttributes,
) -> Option<Box<dyn Fn(&ProximaRecord) -> bool + Send + Sync>> {
    let security = compile_security_filter(refs, store, subject)?;

    Some(Box::new(move |record: &ProximaRecord| {
        // Resolve each field the security expression references from the record's
        // props. Using `proxima_value_to_filter_literal` (not the whole-tree map) avoids a
        // per-row HashMap allocation — only the fields the expression actually
        // touches are extracted, lazily.
        let resolve = |field: &str| -> Option<serde_json::Value> {
            record.props.get(field).and_then(|node| match node {
                ProximaTreeNode::Value(pv) => Some(proxima_value_to_filter_literal(pv)),
                _ => None,
            })
        };
        admits_with_security(None, Some(&security), &resolve)
    }))
}

/// The outcome of an ABAC resolution for a scan: how to handle the rows.
#[cfg(feature = "abac-policy")]
#[allow(dead_code)]
pub enum AbacScanResult {
    /// No row restriction (the subject is permitted with no row predicates).
    Unrestricted,
    /// A per-row predicate that `scan_records_filtered` ANDs into its hook.
    Restricted(Box<dyn Fn(&ProximaRecord) -> bool + Send + Sync>),
    /// The subject is denied entirely — return zero rows.
    Denied(DenyReason),
}

/// The service-facing ABAC enforcement API. Holds the three substrate stores
/// and provides one method a scan call site calls: resolve the subject's
/// authorization, compile it, and return a scan predicate (or a deny).
///
/// FA-c Phase 2c constructs this at the scan boundary; Phase 4 (FA-b) threads
/// it (or its `AuthorizedReadContext`) as a required parameter.
#[cfg(feature = "abac-policy")]
#[allow(dead_code)]
pub struct AbacEnforcer {
    authority: Arc<dyn AttributeAuthority + Send + Sync>,
    store: Arc<dyn PredicateObjectStore + Send + Sync>,
    epochs: Arc<dyn PolicyEpochSource + Send + Sync>,
    /// The policy bindings this enforcer governs. Holding them makes the enforcer
    /// a self-contained policy a service can store once and call per-read
    /// (`predicate_for`); the per-call `scan_predicate_for(.., bindings)` variant
    /// remains for substrate unit tests.
    bindings: Vec<PolicyBinding>,
    /// The durable policy source (Phase 5b). When set, `predicate_for` loads the
    /// tenant's bindings from here per read — restart-surviving policy. When
    /// unset, it falls back to the held `bindings` (the in-memory/test path).
    ///
    /// Held as a shared `Arc<dyn PolicyBindingStore>` (not `Box`) so the same
    /// durable store instance is shared between this enforcer and an
    /// admin-provisioning writer: a write through the writer's `Arc` handle is
    /// visible to the enforcer's next read without a restart (hot-reload,
    /// TD-ABAC control-plane).
    binding_store: Option<Arc<dyn PolicyBindingStore + Send + Sync>>,
    /// ADR-090 L1.2: the durable grant store (entitlement layer). Shared `Arc`
    /// for the same hot-reload reason as `binding_store`. Consulted only when
    /// `PROXIMADB_AUTHZ_REQUIRE_GRANTS` is armed.
    grant_store: Option<Arc<proximadb_catalog::grants::FileSystemGrantStore>>,
    /// TD-SEC-2 Slice C: per-tenant security posture. When present it decides
    /// each tenant's enforcement mode; the env gate below is only the DEFAULT
    /// for tenants with no explicit record.
    posture_store: Option<Arc<proximadb_catalog::tenant_posture::FileSystemTenantPostureStore>>,
}

#[cfg(feature = "abac-policy")]
#[allow(dead_code)]
impl AbacEnforcer {
    /// Construct from the three substrate stores. In production these are the
    /// durable-backed impls (shared `Arc` handles so an admin writer and the
    /// enforcer observe the same instance); in tests, the in-memory ones.
    pub fn new(
        authority: Arc<dyn AttributeAuthority + Send + Sync>,
        store: Arc<dyn PredicateObjectStore + Send + Sync>,
        epochs: Arc<dyn PolicyEpochSource + Send + Sync>,
    ) -> Self {
        Self {
            authority,
            store,
            epochs,
            bindings: Vec::new(),
            binding_store: None,
            grant_store: None,
            posture_store: None,
        }
    }

    /// Install the policy bindings this enforcer governs (builder). After this,
    /// [`predicate_for`](Self::predicate_for) resolves against the held bindings
    /// — the one call a read-serving service makes per scan.
    pub fn with_bindings(mut self, bindings: Vec<PolicyBinding>) -> Self {
        self.bindings = bindings;
        self
    }

    /// Install the durable policy source (builder, Phase 5b). When set,
    /// [`predicate_for`](Self::predicate_for) loads the tenant's bindings from the
    /// store on every read — a restart-surviving policy. This is the production
    /// path; [`with_bindings`](Self::with_bindings) remains for tests.
    ///
    /// Takes a shared `Arc` so the caller (boot wiring) can retain its own clone
    /// of the same durable store for the admin-provisioning writer — a provision
    /// is then visible to this enforcer without a restart.
    pub fn with_posture_store(
        mut self,
        store: Arc<proximadb_catalog::tenant_posture::FileSystemTenantPostureStore>,
    ) -> Self {
        self.posture_store = Some(store);
        self
    }

    pub fn with_grant_store(
        mut self,
        store: Arc<proximadb_catalog::grants::FileSystemGrantStore>,
    ) -> Self {
        self.grant_store = Some(store);
        self
    }

    pub fn with_binding_store(mut self, store: Arc<dyn PolicyBindingStore + Send + Sync>) -> Self {
        self.binding_store = Some(store);
        self
    }

    /// Resolve `subject`'s authorization for `target` and compile to a scan
    /// predicate. When a durable [`PolicyBindingStore`] is installed, the
    /// tenant's bindings are loaded from it per read (the production path);
    /// otherwise the enforcer's held `bindings` are used (the in-memory/test
    /// path). This is the one call a read-serving service makes per scan.
    pub fn predicate_for(
        &self,
        subject: &SubjectId,
        tenant: u64,
        target: Target,
    ) -> AbacScanResult {
        let owned;
        let bindings: &[PolicyBinding] = match &self.binding_store {
            Some(store) => {
                owned = store.bindings_for(tenant);
                &owned
            }
            None => &self.bindings,
        };
        self.scan_predicate_for(subject, tenant, target, bindings)
    }

    /// Resolve `subject`'s read authorization for `target`, returning the
    /// admitted [`AuthorizedReadContext`] for the vector push-down path: the
    /// caller wraps it in `ReadContext::Client(ctx)` and hands it to
    /// `unified_search_native`. `Err(reason)` ⇒ DENY — the caller MUST fail
    /// closed (return empty results) and MUST NOT collapse it to `None`/`System`
    /// (that would be a fail-open hole). Mirrors `predicate_for`'s binding-store
    /// selection so a durable store is consulted per read when present.
    pub fn resolve_read_context(
        &self,
        subject: &SubjectId,
        tenant: u64,
        target: Target,
    ) -> Result<AuthorizedReadContext, DenyReason> {
        let owned;
        let bindings: &[PolicyBinding] = match &self.binding_store {
            Some(store) => {
                owned = store.bindings_for(tenant);
                &owned
            }
            None => &self.bindings,
        };
        match AuthorizedReadContext::resolve(
            self.authority.as_ref(),
            self.epochs.as_ref(),
            bindings,
            subject,
            tenant,
            target,
        ) {
            proximadb_abac::ReadDecision::Deny(reason) => Err(reason),
            proximadb_abac::ReadDecision::Admit(ctx) => {
                // ADR-090 L1.2 (deny > absence-of-grant > grant): with grant
                // enforcement armed, a policy admit is necessary but not
                // sufficient — an applicable GRANT must also admit the subject.
                // Armed with NO store attached fails closed too: "required"
                // cannot degrade to "optional" because wiring is incomplete.
                // Grant predicate-ref composition into the row filter is L2
                // (today only admit/deny is enforced here).
                // TD-SEC-2 Slice C: the ENFORCEMENT MODE IS PER TENANT.
                // ADR-090 L1.2 gated this on one process-global env var, which
                // is the wrong shape for SaaS — an operator cannot flag-day
                // every customer onto strict authorization at once. The env
                // gate now supplies only the DEFAULT for tenants that have no
                // explicit posture, so existing deployments are unchanged.
                use proximadb_catalog::tenant_posture::PostureDecision;
                match self.posture_for(tenant) {
                    PostureDecision::Skip => {}
                    PostureDecision::AuditOnly => {
                        // Rehearsal: evaluate, report, ADMIT. This is the ramp
                        // that makes onboarding a tenant to Enforce safe —
                        // without it the only choices are "unenforced" and
                        // "possibly break production", so nobody ever flips it.
                        if !grant_admits_read(
                            self.grant_store.as_deref(),
                            // owner == acting tenant: L1.2 enforces same-tenant
                            // reads only, so this is bit-for-bit today's
                            // behavior. Cross-tenant opens supply the real
                            // owner (ADR-090 item 3), which is why the
                            // parameter exists rather than being derived here.
                            tenant,
                            tenant,
                            &subject.0,
                            &target,
                        ) {
                            tracing::warn!(
                                target: "proximadb.authz.audit",
                                tenant,
                                subject = %subject.0,
                                table = target.table,
                                "GRANT AUDIT: this read would be DENIED under Enforce \
                                 (no applicable grant); admitted because the tenant's \
                                 posture is Audit"
                            );
                        }
                    }
                    PostureDecision::Enforce => {
                        if !grant_admits_read(
                            self.grant_store.as_deref(),
                            // owner == acting tenant: L1.2 enforces same-tenant
                            // reads only, so this is bit-for-bit today's
                            // behavior. Cross-tenant opens supply the real
                            // owner (ADR-090 item 3), which is why the
                            // parameter exists rather than being derived here.
                            tenant,
                            tenant,
                            &subject.0,
                            &target,
                        ) {
                            return Err(DenyReason::NoApplicableGrant);
                        }
                    }
                }
                Ok(ctx)
            }
        }
    }

    /// Resolve `subject`'s authorization for `target` and compile to a scan
    /// predicate. The caller provides the `bindings` (loaded from the policy
    /// store — Phase 5 makes that durable; today they are supplied directly).
    ///
    /// This is the one call a scan call site makes: it ties resolve → compile
    /// → `abac_scan_predicate` into a single `AbacScanResult`.
    pub fn scan_predicate_for(
        &self,
        subject: &SubjectId,
        tenant: u64,
        target: Target,
        bindings: &[PolicyBinding],
    ) -> AbacScanResult {
        match AuthorizedReadContext::resolve(
            self.authority.as_ref(),
            self.epochs.as_ref(),
            bindings,
            subject,
            tenant,
            target,
        ) {
            proximadb_abac::ReadDecision::Deny(reason) => AbacScanResult::Denied(reason),
            proximadb_abac::ReadDecision::Admit(ctx) => {
                match abac_scan_predicate(
                    ctx.row_predicate_refs(),
                    self.store.as_ref(),
                    ctx.subject(),
                ) {
                    Some(predicate) => AbacScanResult::Restricted(predicate),
                    None => AbacScanResult::Unrestricted,
                }
            }
        }
    }

    /// Resolve `subject`'s authorization and return the compiled security
    /// `FilterExpression` directly — for read paths that take a `FilterExpression`
    /// natively (e.g. `unified_search_native`'s `filter` param), rather than a
    /// per-record closure.
    ///
    /// Returns `Ok(None)` when the subject is permitted with no row predicates;
    /// `Ok(Some(expr))` when a security filter applies; `Err(reason)` when denied.
    /// The caller ANDs `expr` with the user's own filter before the search.
    pub fn security_expression_for(
        &self,
        subject: &SubjectId,
        tenant: u64,
        target: Target,
        bindings: &[PolicyBinding],
    ) -> Result<Option<FilterExpression>, DenyReason> {
        match AuthorizedReadContext::resolve(
            self.authority.as_ref(),
            self.epochs.as_ref(),
            bindings,
            subject,
            tenant,
            target,
        ) {
            proximadb_abac::ReadDecision::Deny(reason) => Err(reason),
            // The subject comes from the ADMITTED context, not the `&SubjectId`
            // parameter: `ctx.subject()` carries the server-RESOLVED attribute
            // bag, which is the only source a `$subject.<attr>` placeholder may
            // read from (ADR-090 L2.1).
            proximadb_abac::ReadDecision::Admit(ctx) => Ok(compile_security_filter(
                ctx.row_predicate_refs(),
                self.store.as_ref(),
                ctx.subject(),
            )),
        }
    }

    /// Compile an **already-admitted** [`AuthorizedReadContext`]'s row predicates
    /// into a security `FilterExpression` (no re-resolution). For the vector
    /// push-down path: the caller (a network handler) resolves the subject →
    /// [`AuthorizedReadContext::resolve`], handles `Deny` fail-closed (→ empty
    /// results) THERE, and passes the admitted `Client` context here. Returns
    /// `Option`, not `Result`: deny is impossible at this stage because the
    /// context is already admitted — the caller MUST NOT collapse a `DenyReason`
    /// into `None` (that would be a fail-open hole). `None` = admitted with no
    /// row predicate.
    pub fn security_filter_for_context(
        &self,
        ctx: &AuthorizedReadContext,
    ) -> Option<FilterExpression> {
        compile_security_filter(ctx.row_predicate_refs(), self.store.as_ref(), ctx.subject())
    }

    /// Resolve `subject`'s authorization and **post-filter** vector search
    /// results. The ANN search runs first (over its own metadata filter), then
    /// this removes inadmissible results via the strict 3-valued walker.
    ///
    /// This is the **vector-path integration** (FA-c Phase 3, Option A): the
    /// search kernel is untouched; the security filter is a post-processing step
    /// on the results. Less efficient than a pre-filter (the search evaluates
    /// more rows than returned) but correct — the permissive search filter and
    /// the strict security filter are evaluated independently and ANDed, so the
    /// fail-open the `combine_filters` merge caused is structurally avoided.
    ///
    /// Returns `Ok(filtered)` on success, `Err(reason)` if the subject is denied
    /// entirely.
    #[allow(dead_code)]
    pub fn filter_search_results(
        &self,
        results: Vec<OptimizedSearchRecord>,
        subject: &SubjectId,
        tenant: u64,
        target: Target,
        bindings: &[PolicyBinding],
    ) -> Result<Vec<OptimizedSearchRecord>, DenyReason> {
        let security = self.security_expression_for(subject, tenant, target, bindings)?;
        Ok(match security {
            None => results,
            Some(expr) => post_filter_search_results(results, &expr),
        })
    }
}

/// Post-filter vector search results by a compiled security `FilterExpression`.
///
/// Each result's `metadata` is resolved against the expression under the strict
/// 3-valued walker. Inadmissible results are removed. This is the vector-path
/// analogue of `abac_scan_predicate` (relational): both evaluate
/// `admits_with_security` per record, but this operates on the *output* of the
/// search rather than as a *predicate* during the scan.
#[cfg(feature = "abac-policy")]
#[allow(dead_code)]
pub fn post_filter_search_results(
    results: Vec<OptimizedSearchRecord>,
    security: &FilterExpression,
) -> Vec<OptimizedSearchRecord> {
    results
        .into_iter()
        .filter(|record| {
            let resolve = |field: &str| -> Option<serde_json::Value> {
                record
                    .metadata
                    .get(field)
                    .map(proxima_value_to_filter_literal)
            };
            admits_with_security(None, Some(security), &resolve)
        })
        .collect()
}

#[cfg(all(test, feature = "abac-policy"))]
mod tests {
    use super::*;
    use crate::core::search::{ComparisonOperator, FilterExpression};
    use proximadb_abac::{
        FileSystemPolicyBindingStore, InMemoryAttributeAuthority, InMemoryPolicyEpochs,
        InMemoryPredicateObjectStore,
    };
    use proximadb_catalog::fc_metamodel::{AttrValue, Effect, Scope, SubjectAttributes};
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{ProximaRecord, ProximaTreeNode};
    use serde_json::json;

    /// A subject with no attributes — these tests exercise the ref-resolution
    /// path, not `$subject.<attr>` substitution (covered in `proximadb-abac`).
    fn test_subject() -> SubjectAttributes {
        SubjectAttributes::new("alice", 1)
    }

    fn record_with(dept: &str) -> ProximaRecord {
        let mut rec = ProximaRecord::default();
        rec.props.insert(
            "dept".to_string(),
            ProximaTreeNode::Value(ProximaValue::String(dept.to_string())),
        );
        rec
    }

    #[test]
    fn predicate_filters_rows_by_dept() {
        let mut store = InMemoryPredicateObjectStore::new();
        store.register(
            42,
            FilterExpression::Comparison {
                field: "dept".to_string(),
                operator: ComparisonOperator::Equals,
                value: json!("eng"),
            },
        );

        let predicate = abac_scan_predicate(&[42], &store, &test_subject())
            .expect("refs non-empty → predicate");

        assert!(predicate(&record_with("eng")), "dept=eng must be admitted");
        assert!(!predicate(&record_with("hr")), "dept=hr must be denied");
    }

    #[test]
    fn empty_refs_produce_no_predicate() {
        let store = InMemoryPredicateObjectStore::new();
        assert!(abac_scan_predicate(&[], &store, &test_subject()).is_none());
    }

    #[test]
    fn missing_ref_denies_every_row() {
        let store = InMemoryPredicateObjectStore::new();
        let predicate = abac_scan_predicate(&[999], &store, &test_subject())
            .expect("non-empty refs → unsatisfiable predicate");
        assert!(!predicate(&record_with("eng")), "missing ref denies all");
    }

    // --- AbacEnforcer: the service-facing one-call API ---

    fn enforcer_with_alice(dept: &str) -> (AbacEnforcer, Vec<PolicyBinding>) {
        let mut authority = InMemoryAttributeAuthority::new();
        authority.upsert(
            proximadb_abac::AttributeBinding::new("alice", 7)
                .with_attr("dept", AttrValue::Str(dept.into())),
        );
        let mut store = InMemoryPredicateObjectStore::new();
        store.register(
            42,
            FilterExpression::Comparison {
                field: "dept".to_string(),
                operator: ComparisonOperator::Equals,
                value: json!(dept),
            },
        );
        let bindings = vec![PolicyBinding {
            object_id: 1,
            tenant_stable_id: 7,
            scope: Scope::Table(200),
            effect: Effect::Permit,
            predicate_ref: Some(42),
            field_mask: None,
        }];
        let enforcer = AbacEnforcer::new(
            Arc::new(authority),
            Arc::new(store),
            Arc::new(InMemoryPolicyEpochs::new()),
        );
        (enforcer, bindings)
    }

    #[test]
    fn enforcer_returns_restricted_predicate_for_dept_eng() {
        let (enforcer, bindings) = enforcer_with_alice("eng");
        let result = enforcer.scan_predicate_for(
            &SubjectId("alice".into()),
            7,
            Target {
                namespace: 3,
                table: 200,
                column: None,
            },
            &bindings,
        );
        match result {
            AbacScanResult::Restricted(pred) => {
                assert!(pred(&record_with("eng")));
                assert!(!pred(&record_with("hr")));
            }
            _ => panic!("alice (dept=eng, binding present) must be Restricted"),
        }
    }

    #[test]
    fn enforcer_denies_an_unbound_subject() {
        let (enforcer, bindings) = enforcer_with_alice("eng");
        let result = enforcer.scan_predicate_for(
            &SubjectId("mallory".into()),
            7,
            Target {
                namespace: 3,
                table: 200,
                column: None,
            },
            &bindings,
        );
        assert!(matches!(result, AbacScanResult::Denied(_)));
    }

    #[test]
    fn enforcer_returns_security_expression_for_vector_path() {
        // The vector search path (unified_search_native) takes a FilterExpression,
        // not a closure. security_expression_for returns it directly.
        let (enforcer, bindings) = enforcer_with_alice("eng");
        let result = enforcer
            .security_expression_for(
                &SubjectId("alice".into()),
                7,
                Target {
                    namespace: 3,
                    table: 200,
                    column: None,
                },
                &bindings,
            )
            .expect("alice is admitted");

        let expr = result.expect("alice has a predicate ref → Some(expr)");
        // The expression should be the dept=eng filter.
        match expr {
            FilterExpression::Comparison {
                field,
                operator: ComparisonOperator::Equals,
                value,
            } => {
                assert_eq!(field, "dept");
                assert_eq!(value, json!("eng"));
            }
            _ => panic!("expected a single Eq(dept, eng) comparison"),
        }
    }

    #[test]
    fn enforcer_resolves_admitted_read_context_for_vector_path() {
        // resolve_read_context is the vector push-down seam: it returns the
        // admitted AuthorizedReadContext for the caller to wrap as
        // ReadContext::Client(ctx). Alice (dept=eng, binding present) is admitted
        // with her row-predicate ref (42) intact — so security_filter_for_context
        // would later compile the dept=eng filter for the ANN push-down.
        let (enforcer, bindings) = enforcer_with_alice("eng");
        let enforcer = enforcer.with_bindings(bindings);
        let ctx = enforcer
            .resolve_read_context(
                &SubjectId("alice".into()),
                7,
                Target {
                    namespace: 3,
                    table: 200,
                    column: None,
                },
            )
            .expect("alice (dept=eng, binding present) must be admitted");
        assert!(
            ctx.row_predicate_refs().contains(&42),
            "admitted context carries alice's row-predicate ref"
        );
    }

    #[test]
    fn enforcer_resolve_read_context_denies_unbound_subject() {
        // Fail-closed contract: an unbound subject (no attribute binding) denies,
        // and the caller must map Err ⇒ empty results — never collapse to None.
        let (enforcer, bindings) = enforcer_with_alice("eng");
        let enforcer = enforcer.with_bindings(bindings);
        let deny = enforcer.resolve_read_context(
            &SubjectId("mallory".into()),
            7,
            Target {
                namespace: 3,
                table: 200,
                column: None,
            },
        );
        assert!(
            deny.is_err(),
            "mallory (unbound) must be denied, not admitted"
        );
    }

    #[test]
    fn enforcer_expression_denies_unbound_subject() {
        let (enforcer, bindings) = enforcer_with_alice("eng");
        let result = enforcer.security_expression_for(
            &SubjectId("mallory".into()),
            7,
            Target {
                namespace: 3,
                table: 200,
                column: None,
            },
            &bindings,
        );
        assert!(result.is_err(), "unbound subject must be denied");
    }

    // --- Phase 5b: durable policy-binding store (the enforcer loads from it) ---

    /// Shared store/authority/predicate-store config for the durable-policy tests.
    /// `dept=eng` is the only admitted dept; tenant 7's policy permits table 200
    /// under predicate ref 42.
    fn predicate_store_for_dept() -> InMemoryPredicateObjectStore {
        let mut store = InMemoryPredicateObjectStore::new();
        store.register(
            42,
            FilterExpression::Comparison {
                field: "dept".to_string(),
                operator: ComparisonOperator::Equals,
                value: json!("eng"),
            },
        );
        store
    }

    #[test]
    fn enforcer_loads_bindings_from_the_store_not_the_held_set() {
        // The enforcer is built with EMPTY held bindings + a durable store holding
        // tenant 7's permit. predicate_for must still resolve — proof it consulted
        // the store, not the (empty) held set.
        let dir = std::env::temp_dir().join(format!(
            "proximadb-abac-enforcer-store-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("policy.json");

        let mut authority = InMemoryAttributeAuthority::new();
        authority.upsert(
            proximadb_abac::AttributeBinding::new("alice", 7)
                .with_attr("dept", AttrValue::Str("eng".into())),
        );
        let store = FileSystemPolicyBindingStore::open(&path).expect("open");
        store.replace_tenant(
            7,
            vec![PolicyBinding {
                object_id: 1,
                tenant_stable_id: 7,
                scope: Scope::Table(200),
                effect: Effect::Permit,
                predicate_ref: Some(42),
                field_mask: None,
            }],
        );

        let enforcer = AbacEnforcer::new(
            Arc::new(authority),
            Arc::new(predicate_store_for_dept()),
            Arc::new(InMemoryPolicyEpochs::new()),
        )
        // NOTE: no with_bindings — held bindings are empty by design.
        .with_binding_store(Arc::new(store));

        match enforcer.predicate_for(
            &SubjectId("alice".into()),
            7,
            Target {
                namespace: 3,
                table: 200,
                column: None,
            },
        ) {
            AbacScanResult::Restricted(pred) => {
                assert!(
                    pred(&record_with("eng")),
                    "alice (dept=eng) admits eng rows"
                );
                assert!(!pred(&record_with("hr")), "dept=hr is denied");
            }
            _ => panic!("alice must be Restricted via the store, not Denied/Unrestricted"),
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn enforcer_reconstitutes_the_same_result_after_a_restart() {
        // The Phase-5b ratchet at the enforcer level: write tenant 7's policy via
        // the durable store, drop it, reopen it, rebuild the enforcer — and the
        // compiled AbacScanResult for alice is byte-identical. The policy survived
        // the restart; the enforcement outcome did too.
        let dir = std::env::temp_dir().join(format!(
            "proximadb-abac-enforcer-restart-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("policy.json");
        let target = Target {
            namespace: 3,
            table: 200,
            column: None,
        };

        // Write phase: persist tenant 7's permit, then drop everything.
        {
            let store = FileSystemPolicyBindingStore::open(&path).expect("open");
            store.replace_tenant(
                7,
                vec![PolicyBinding {
                    object_id: 1,
                    tenant_stable_id: 7,
                    scope: Scope::Table(200),
                    effect: Effect::Permit,
                    predicate_ref: Some(42),
                    field_mask: None,
                }],
            );
        }

        // Read phase: reopen the durable store and build a fresh enforcer over it
        // (authority/predicate-store/epochs are reconstructed identically — only
        // the policy source is what persisted).
        fn fresh_enforcer(store: FileSystemPolicyBindingStore) -> AbacEnforcer {
            let mut authority = InMemoryAttributeAuthority::new();
            authority.upsert(
                proximadb_abac::AttributeBinding::new("alice", 7)
                    .with_attr("dept", AttrValue::Str("eng".into())),
            );
            AbacEnforcer::new(
                Arc::new(authority),
                Arc::new(predicate_store_for_dept()),
                Arc::new(InMemoryPolicyEpochs::new()),
            )
            .with_binding_store(Arc::new(store))
        }

        let store = FileSystemPolicyBindingStore::open(&path).expect("reopen");
        let enforcer = fresh_enforcer(store);
        let result = enforcer.predicate_for(&SubjectId("alice".into()), 7, target);

        match result {
            AbacScanResult::Restricted(pred) => {
                assert!(pred(&record_with("eng")));
                assert!(!pred(&record_with("hr")));
            }
            _ => panic!("alice must be Restricted after restart"),
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn enforcer_observes_a_binding_provisioned_after_construction_hot_reload() {
        // PR-A hot-reload gate (TD-ABAC control-plane): the admin-provisioning
        // path writes through a SHARED `Arc<FileSystemPolicyBindingStore>` — the
        // SAME instance the live enforcer reads. A binding written AFTER the
        // enforcer is constructed must be visible to the very next read, with no
        // restart. This is the property that makes runtime policy provisioning
        // usable in a running server.
        let dir = std::env::temp_dir().join(format!(
            "proximadb-abac-enforcer-hotreload-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("policy.json");
        let target = Target {
            namespace: 3,
            table: 200,
            column: None,
        };

        let mut authority = InMemoryAttributeAuthority::new();
        authority.upsert(
            proximadb_abac::AttributeBinding::new("alice", 7)
                .with_attr("dept", AttrValue::Str("eng".into())),
        );

        // Build the durable store EMPTY, wrap it in an `Arc`, and hand a CLONE
        // (the same instance) to the enforcer. The caller retains its own clone —
        // the admin-provisioning handle.
        let store = Arc::new(FileSystemPolicyBindingStore::open(&path).expect("open"));
        let enforcer = AbacEnforcer::new(
            Arc::new(authority),
            Arc::new(predicate_store_for_dept()),
            Arc::new(InMemoryPolicyEpochs::new()),
        )
        .with_binding_store(store.clone());

        // Before provisioning, alice has no applicable policy ⇒ deny-biased ⇒
        // fail-closed DENY (also the unprovisioned-tenant behavior the admin API
        // must not let through).
        assert!(
            matches!(
                enforcer.predicate_for(&SubjectId("alice".into()), 7, target),
                AbacScanResult::Denied(_)
            ),
            "no binding provisioned yet ⇒ alice is denied (fail-closed)"
        );

        // The admin provisions tenant 7's permit through the SAME shared handle —
        // no restart, no re-open. This is the write the enforcer must observe.
        store.replace_tenant(
            7,
            vec![PolicyBinding {
                object_id: 1,
                tenant_stable_id: 7,
                scope: Scope::Table(200),
                effect: Effect::Permit,
                predicate_ref: Some(42),
                field_mask: None,
            }],
        );

        // The live enforcer observes the provisioned binding on the next read:
        // alice is now admitted with her dept=eng row predicate.
        match enforcer.predicate_for(&SubjectId("alice".into()), 7, target) {
            AbacScanResult::Restricted(pred) => {
                assert!(pred(&record_with("eng")), "dept=eng row admitted");
                assert!(!pred(&record_with("hr")), "dept=hr row denied");
            }
            AbacScanResult::Unrestricted => {
                panic!("alice must be Restricted (predicate ref 42), not Unrestricted")
            }
            AbacScanResult::Denied(_) => {
                panic!("hot-reload failed: provisioned binding not visible to enforcer")
            }
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    // --- Phase 3: vector-path post-filter ---

    use proximadb_search_types::results::OptimizedSearchRecord;
    use std::collections::HashMap;

    fn search_result(id: &str, dept: &str) -> OptimizedSearchRecord {
        let mut metadata = HashMap::new();
        metadata.insert("dept".to_string(), ProximaValue::String(dept.to_string()));
        OptimizedSearchRecord {
            id: id.to_string(),
            metadata,
            ..Default::default()
        }
    }

    #[test]
    fn post_filter_removes_inadmissible_search_results() {
        let mut store = InMemoryPredicateObjectStore::new();
        store.register(
            42,
            FilterExpression::Comparison {
                field: "dept".to_string(),
                operator: ComparisonOperator::Equals,
                value: json!("eng"),
            },
        );

        let results = vec![
            search_result("r1", "eng"),
            search_result("r2", "hr"),
            search_result("r3", "eng"),
            search_result("r4", "legal"),
        ];

        let security = compile_security_filter(&[42], &store, &test_subject()).expect("compiled");
        let filtered = post_filter_search_results(results, &security);

        assert_eq!(filtered.len(), 2, "only dept=eng results survive");
        assert_eq!(filtered[0].id, "r1");
        assert_eq!(filtered[1].id, "r3");
    }

    #[test]
    fn enforcer_filter_search_results_denies_unbound() {
        let (enforcer, bindings) = enforcer_with_alice("eng");
        let results = vec![search_result("r1", "eng")];
        let outcome = enforcer.filter_search_results(
            results,
            &SubjectId("mallory".into()),
            7,
            Target {
                namespace: 3,
                table: 200,
                column: None,
            },
            &bindings,
        );
        assert!(outcome.is_err(), "unbound subject denied");
    }
}

impl AbacEnforcer {
    /// The enforcement mode for `tenant`: its explicit posture record when one
    /// exists, else the process default derived from the env gate.
    fn posture_for(&self, tenant: u64) -> proximadb_catalog::tenant_posture::PostureDecision {
        use proximadb_catalog::tenant_posture::GrantEnforcement;
        let default_mode = if grants_required() {
            GrantEnforcement::Enforce
        } else {
            GrantEnforcement::Off
        };
        match &self.posture_store {
            Some(store) => store.resolve(tenant, default_mode).decision(),
            None => default_mode.decision(),
        }
    }
}

/// ADR-090 L1.2 opt-in gate: when `PROXIMADB_AUTHZ_REQUIRE_GRANTS` is truthy,
/// a policy admit additionally requires an applicable grant (deny > absence >
/// grant). Default OFF — absent means today's behavior, unchanged.
#[cfg(feature = "abac-policy")]
fn grants_required() -> bool {
    match std::env::var("PROXIMADB_AUTHZ_REQUIRE_GRANTS") {
        Ok(v) => {
            let v = v.trim();
            v == "1"
                || v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("on")
                || v.eq_ignore_ascii_case("yes")
        }
        Err(_) => false,
    }
}

/// ADR-090 L1.2 pure decision: does an applicable grant admit `subject` (a
/// user of `tenant`) to read `target`? `None` store ⇒ **false** — when
/// enforcement is armed, "required" cannot degrade to "optional" because
/// wiring is incomplete.
#[cfg(feature = "abac-policy")]
/// Does a live grant admit `subject` (a user of `acting_tenant`) to read `target`?
///
/// B2 FIX: `owner` and `acting_tenant` are SEPARATE parameters. A grant is
/// issued BY the resource owner, and `FileSystemGrantStore` is partitioned by
/// `owner_tenant_stable_id` — so the OWNER's slice is the one that must be
/// loaded. This previously passed the acting tenant for both, which asks "what
/// has this tenant granted itself": correct by accident when owner == acting
/// (the only case L1.2 enforced), and structurally unable to find any
/// cross-tenant share. Callers that genuinely mean same-tenant pass the acting
/// tenant for both, which reduces to the previous expression bit-for-bit.
fn grant_admits_read(
    store: Option<&proximadb_catalog::grants::FileSystemGrantStore>,
    owner: u64,
    acting_tenant: u64,
    subject: &str,
    target: &Target,
) -> bool {
    let Some(store) = store else {
        return false;
    };
    matches!(
        proximadb_catalog::grants::evaluate_grants(
            &store.grants_for_owner(owner),
            &proximadb_catalog::grants::GrantSubject {
                tenant_stable_id: acting_tenant,
                subject,
            },
            target,
            proximadb_catalog::grants::GrantAction::Read,
            chrono::Utc::now().timestamp_millis(),
        ),
        proximadb_catalog::grants::GrantDecision::Permit { .. }
    )
}

// ADR-090 L1.2 specification for the seam's NEW logic. The full
// provision→permit→deny→revoke e2e across live transports is the gate's
// flip-precondition (registry row) and lands with the admin surface; these
// pin the decision semantics the seam composes.
#[cfg(all(test, feature = "abac-policy"))]
mod grant_gate_tests {
    use super::*;
    use proximadb_catalog::grants::{FileSystemGrantStore, GrantAction, Grantee};
    use std::collections::BTreeSet;

    fn target(table: u32) -> Target {
        Target {
            namespace: 0,
            table,
            column: None,
        }
    }

    /// B2 REGRESSION: a grant is issued BY the resource owner, and the store is
    /// partitioned by `owner_tenant_stable_id`. Loading the ACTING tenant's
    /// slice asks "what has this tenant granted itself" — so a cross-tenant
    /// share is structurally unreachable: the grant written by owner A never
    /// appears in tenant B's slice.
    ///
    /// This test FAILS before the fix (the owner's slice is never loaded) and
    /// passes after. Same-tenant behavior is unchanged because owner == acting
    /// there, which is why the defect stayed invisible.
    #[test]
    fn a_grant_from_another_owner_is_found() {
        use proximadb_catalog::grants::{FileSystemGrantStore, GrantAction, Grantee};
        use std::collections::BTreeSet;

        const OWNER: u64 = 7;
        const GRANTEE: u64 = 9;

        let dir = tempfile::tempdir().expect("tempdir");
        let store = FileSystemGrantStore::open(dir.path()).expect("open");

        // Owner 7 shares its collection with tenant 9.
        store
            .grant(
                OWNER,
                proximadb_catalog::fc_metamodel::Scope::Table(10),
                Grantee::Tenant(GRANTEE),
                BTreeSet::from([GrantAction::Read]),
                None,
                None,
                None,
            )
            .expect("grant");

        let target = Target {
            namespace: 0,
            table: 10,
            column: None,
        };

        assert!(
            grant_admits_read(Some(&store), OWNER, GRANTEE, "bob", &target),
            "a grant written by owner {OWNER} for tenant {GRANTEE} must be found \
             when the OWNER's slice is consulted"
        );

        // And a tenant with no grant from this owner still gets nothing.
        assert!(
            !grant_admits_read(Some(&store), OWNER, 11, "bob", &target),
            "an ungranted tenant must stay denied"
        );
    }

    /// Armed with NO store ⇒ deny (fail-closed on incomplete wiring).
    #[test]
    fn no_store_never_admits() {
        assert!(!grant_admits_read(None, 7, 7, "alice", &target(10)));
    }

    /// No applicable grant ⇒ deny; an applicable grant ⇒ admit; revoke ⇒ deny.
    #[test]
    fn grant_lifecycle_drives_admission() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FileSystemGrantStore::open(dir.path()).expect("open");
        assert!(!grant_admits_read(Some(&store), 7, 7, "alice", &target(10)));

        let id = store
            .grant(
                7,
                proximadb_catalog::fc_metamodel::Scope::Table(10),
                Grantee::User {
                    tenant_stable_id: 7,
                    subject: proximadb_catalog::fc_metamodel::SubjectId("alice".into()),
                },
                BTreeSet::from([GrantAction::Read]),
                None,
                None,
                None,
            )
            .expect("grant");
        assert!(grant_admits_read(Some(&store), 7, 7, "alice", &target(10)));
        assert!(
            !grant_admits_read(Some(&store), 7, 7, "mallory", &target(10)),
            "another subject must not ride alice's grant"
        );

        store.revoke(7, &id).expect("revoke");
        assert!(!grant_admits_read(Some(&store), 7, 7, "alice", &target(10)));
    }

    /// The env gate parses the standard truthy set and defaults OFF.
    #[test]
    fn gate_parsing_defaults_off() {
        // nextest process-per-test isolation makes set_var safe here (the same
        // justification as compaction_tests::RecordVersionGate).
        unsafe { std::env::remove_var("PROXIMADB_AUTHZ_REQUIRE_GRANTS") };
        assert!(!grants_required());
        for v in ["1", "true", "ON", "yes"] {
            unsafe { std::env::set_var("PROXIMADB_AUTHZ_REQUIRE_GRANTS", v) };
            assert!(grants_required(), "{v} must arm the gate");
        }
        unsafe { std::env::set_var("PROXIMADB_AUTHZ_REQUIRE_GRANTS", "0") };
        assert!(!grants_required());
        unsafe { std::env::remove_var("PROXIMADB_AUTHZ_REQUIRE_GRANTS") };
    }
}
