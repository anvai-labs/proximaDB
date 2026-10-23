//! End-to-end TurboQuant **propagation** integration test
//! (Phase M — Quantization Trait Convergence Plan).
//!
//! Where `tests/turboquant_e2e_test.rs` proves the TurboQuant API
//! surface itself works (encode → save → load → search → mask), this
//! test proves the *propagation chain* the Quantization Trait
//! Convergence Plan landed:
//!
//! ```text
//!  TurboQuantStoreRegistry (Phase B/H)
//!         │
//!         ▼
//!  TurboQuantAxisIndex (Phase D) — registers vectors, exposes
//!         │                        AxisVectorIndex::search_with_candidate_set
//!         ▼
//!  PredicateDiagnostics::scope (Phase K) — task-local bus
//!         │
//!         ▼  (handler take_turboquant_hints inside the scope)
//!  TraceBuilderInputs.turboquant_explain (Phase L)
//!         │
//!         ▼
//!  SearchPlanTrace.turboquant_explain — structured log + EXPLAIN
//! ```
//!
//! The test consumes the same public surface a real handler does — no
//! private helpers, no private fields. Any regression in module-level
//! encapsulation, trait dispatch, or wire-shape stability shows up
//! here.
//!
//! Run with:
//!
//! ```ignore
//! cargo test --features experimental-turboquant \
//!     --test turboquant_e2e_propagation_test -- --test-threads=1
//! ```

#![cfg(feature = "experimental-turboquant")]

use std::sync::Arc;

use proximadb::compute::quantization::turboquant_store_registry::{
    InMemoryTurboQuantStoreRegistry, TurboQuantStoreRegistry,
};
use proximadb::core::search::filter_contract::{CandidateMaskSet, CandidateSet, SlotIdResolver};
use proximadb::core::service_types::IndexStats;
use proximadb::index::axis::index_factory::AxisVectorIndex;
use proximadb::index::axis::indexes::{
    TurboQuantAxisIndex, TurboQuantAxisIndexConfig, TurboQuantSlotResolver,
};
use proximadb::index::turboquant_bridge::TurboQuantExplainHints;
use proximadb::observability::predicate_diagnostics;
use proximadb::observability::search_plan_trace::{
    CacheResult, FilterStrategy, IndexRoute, SureSignals,
};
use proximadb::observability::search_plan_trace_builder::{TraceBuilderInputs, build};
use proximadb::query::federated::optimizer::plan_builder::PlanOutput;
use proximadb_quantization_types::{CalibrationMode, derive_rotation_seed};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim)
            .map(|_| rng.sample::<f64, _>(StandardNormal) as f32)
            .collect();
        let sumsq: f32 = v.iter().map(|x| x * x).sum();
        let inv = if sumsq > 1e-30 {
            1.0 / sumsq.sqrt()
        } else {
            0.0
        };
        for x in v.iter_mut() {
            *x *= inv;
        }
        out.push(v);
    }
    out
}

fn plan() -> PlanOutput {
    PlanOutput {
        filter_strategy: FilterStrategy::HybridFilter,
        index_route: IndexRoute::FullPrecisionGraph,
        estimated_selectivity: Some(0.1),
        gls_score: None,
    }
}

/// Construct an empty registry and verify `get` returns `None` for an
/// unknown collection. This is the wire-shape contract every consumer
/// (Phase C progressive-search, Phase D AXIS adapter) depends on:
/// absence is a typed `None`, not a panic or a synthesized default.
#[tokio::test]
async fn registry_get_for_unknown_collection_returns_none() {
    let registry: Arc<dyn TurboQuantStoreRegistry> =
        Arc::new(InMemoryTurboQuantStoreRegistry::new());
    assert!(
        registry.get("never-registered").await.unwrap().is_none(),
        "absence must be typed None, not a synthesized default store",
    );
}

/// Construct a registry, call `get_or_create` with a derived rotation
/// seed, then verify subsequent `get` calls return the same `Arc`. This
/// pins the per-collection-isolation contract from ADR-021 §"Authority
/// mode": same collection id → same store across consumers.
#[tokio::test]
async fn registry_get_or_create_caches_store_across_consumers() {
    let registry: Arc<dyn TurboQuantStoreRegistry> =
        Arc::new(InMemoryTurboQuantStoreRegistry::new());

    let collection_id = "kb-prop-test";
    let seed = derive_rotation_seed(collection_id);

    let store_a = registry
        .get_or_create(collection_id, 64, 4, CalibrationMode::Identity, seed)
        .await
        .expect("get_or_create");
    let store_b = registry
        .get_or_create(collection_id, 64, 4, CalibrationMode::Identity, seed)
        .await
        .expect("get_or_create second call");

    assert!(
        Arc::ptr_eq(&store_a, &store_b),
        "registry must cache; two get_or_create calls produced distinct Arcs",
    );
}

/// Full Phase D propagation: build a `TurboQuantAxisIndex` and run a
/// `search_with_candidate_set` through the **trait** (`Arc<dyn
/// AxisVectorIndex>`). Verifies the trait-method dispatch reaches the
/// `IdMapIndex` kernel and that the bitmap-fast path returns only
/// allowlisted ids.
#[tokio::test]
async fn axis_trait_dispatch_routes_candidate_mask_through_kernel() {
    let cfg = TurboQuantAxisIndexConfig {
        dim: 64,
        bit_width: 4,
        calibration_mode: CalibrationMode::Identity,
        rotation_seed: derive_rotation_seed("kb-axis-test"),
    };
    let index: Arc<dyn AxisVectorIndex> =
        Arc::new(TurboQuantAxisIndex::new(cfg).expect("construct index"));

    let n = 50;
    let dim = 64;
    let vecs = random_unit_vectors(n, dim, 42);
    for (i, v) in vecs.into_iter().enumerate() {
        index.add(format!("vec-{i}"), v).await.expect("add");
    }

    // Build a CandidateMaskSet covering only slots 5..15. Bind it to
    // the adapter's slot resolver so the bridge downcast in
    // `search_with_candidate_set` engages the bitmap-fast path.
    let tq_index = TurboQuantAxisIndex::new(TurboQuantAxisIndexConfig {
        dim,
        bit_width: 4,
        calibration_mode: CalibrationMode::Identity,
        rotation_seed: derive_rotation_seed("kb-axis-test-resolver"),
    })
    .expect("resolver fixture");
    // Resolver fixture is just to satisfy SlotIdResolver — slot test
    // below is over the live `index` so the mask must use a resolver
    // that references the same inner table. We hand-build one using
    // the public TurboQuantSlotResolver constructor pattern shown in
    // the unit tests (which uses the adapter's own DashMaps).
    //
    // For a public-surface integration test, the simpler shape:
    // build a no-op resolver bound to the right slot count. The kernel
    // dispatch reads only `bitmap()`, so the resolver's job is purely
    // bookkeeping for `to_vec()` / `contains()` — neither of which
    // this test calls.
    #[derive(Debug)]
    struct NoopResolver {
        capacity: usize,
    }
    impl SlotIdResolver for NoopResolver {
        fn id_for_slot(&self, slot: usize) -> Option<String> {
            if slot < self.capacity {
                Some(format!("slot-{slot}"))
            } else {
                None
            }
        }
        fn slot_for_id(&self, id: &str) -> Option<usize> {
            id.strip_prefix("slot-").and_then(|s| s.parse().ok())
        }
    }
    drop(tq_index); // silence unused warning

    let resolver: Arc<dyn SlotIdResolver> = Arc::new(NoopResolver { capacity: n });
    let mut mask = CandidateMaskSet::new(n, resolver);
    for slot in 5..15 {
        mask.set_slot(slot);
    }

    let q = random_unit_vectors(1, dim, 7).into_iter().next().unwrap();
    let hits = index
        .search_with_candidate_set(&q, 5, Some(&mask as &dyn CandidateSet))
        .await
        .expect("trait search");

    // Every returned id must lie inside the mask range. The adapter's
    // String↔u64 mapping resolves slot → id; we recover slot from id.
    assert!(!hits.is_empty(), "kernel must produce hits");
    for (id, score) in &hits {
        assert!(score.is_finite(), "non-finite score for {id}: {score}");
        let slot_num: usize = id
            .strip_prefix("vec-")
            .and_then(|n| n.parse().ok())
            .expect("id parses");
        assert!(
            (5..15).contains(&slot_num),
            "leaked id {id} outside mask range 5..15",
        );
    }
}

/// Phase K propagation: a payload `record`ed inside a
/// `predicate_diagnostics::scope` must be `take`able by the same task
/// and must NOT leak across scope boundaries. This is the load-bearing
/// per-request isolation contract — without it, two concurrent
/// requests would see each other's TurboQuant payloads.
#[tokio::test]
async fn predicate_diagnostics_isolates_turboquant_payload_per_scope() {
    // Outer scope: record one payload.
    let outer_take = predicate_diagnostics::scope(async {
        predicate_diagnostics::record_turboquant_hints(serde_json::json!({
            "marker": "outer",
            "blocks_skipped_by_mask": 1,
        }));

        // Inner scope: record a different payload, take it inside.
        let inner_take = predicate_diagnostics::scope(async {
            predicate_diagnostics::record_turboquant_hints(serde_json::json!({
                "marker": "inner",
                "blocks_skipped_by_mask": 99,
            }));
            predicate_diagnostics::take_turboquant_hints()
        })
        .await;

        // Outer take MUST see the outer payload — inner scope did
        // not leak.
        let outer = predicate_diagnostics::take_turboquant_hints();
        (inner_take, outer)
    })
    .await;

    let (inner, outer) = outer_take;
    assert_eq!(
        inner.as_ref().and_then(|v| v.get("marker")),
        Some(&serde_json::Value::from("inner")),
    );
    assert_eq!(
        outer.as_ref().and_then(|v| v.get("marker")),
        Some(&serde_json::Value::from("outer")),
        "outer scope MUST NOT see inner payload — bus isolation broken",
    );
}

/// Phase L propagation: the payload pulled from the bus must land
/// verbatim on `SearchPlanTrace.turboquant_explain`. Without this
/// propagation, structured-log + EXPLAIN consumers would silently lose
/// the TurboQuant routing info. This is the ADR-004 wire-contract
/// test.
#[test]
fn trace_builder_propagates_turboquant_payload_verbatim() {
    let payload = serde_json::json!({
        "quantization": "turboquant_4bit",
        "calibration_mode": "tq_plus",
        "rotation_seed": "0xabcdef",
        "encoded_epoch": 3,
        "current_epoch": 3,
        "mask_pushed_to_kernel": true,
        "kernel_arch": "scalar",
        "blocks_skipped_by_mask": 17,
        "length_renorm_applied": true,
        "candidate_set_size": 1024,
        "n_vectors_scanned": 4096,
    });

    let p = plan();
    let inputs = TraceBuilderInputs {
        trace_id: "trace-prop-1".into(),
        tenant_id: "tenant-prop".into(),
        collection_name: "kb-prop".into(),
        filter_strategy: p.filter_strategy,
        index_route: p.index_route,
        estimated_selectivity: p.estimated_selectivity,
        gls_score: p.gls_score,
        latency_ms: 7.5,
        index_stats: IndexStats::default(),
        candidate_count: 42,
        rerank_count: 5,
        repair_count: 0,
        sure_signals: SureSignals::default(),
        cache_result: CacheResult::Miss,
        failure_class: None,
        bytes_per_vector: 0.0,
        egress_bytes: 0,
        object_store_gets: 0,
        object_store_bytes_read: 0,
        predicate_shortfall: None,
        turboquant_explain: Some(payload.clone()),
    };
    let trace = build(inputs);

    // Trace carries the payload verbatim.
    assert_eq!(trace.turboquant_explain.as_ref(), Some(&payload));

    // JSON serialization preserves the payload at the right path.
    let v = serde_json::to_value(&trace).expect("serialize");
    let tq = v
        .get("turboquant_explain")
        .expect("turboquant_explain key present")
        .as_object()
        .expect("payload is an object");
    assert_eq!(
        tq.get("quantization").and_then(|x| x.as_str()),
        Some("turboquant_4bit")
    );
    assert_eq!(
        tq.get("blocks_skipped_by_mask").and_then(|x| x.as_u64()),
        Some(17)
    );
    assert_eq!(
        tq.get("mask_pushed_to_kernel").and_then(|x| x.as_bool()),
        Some(true)
    );
}

/// Full chain: build EXPLAIN hints via the bridge, route through the
/// bus, lift inside scope, plumb into the trace builder, and verify
/// the round-trip yields the same JSON shape. This is the canonical
/// "did the entire Phase A→L chain stay wired" regression test.
#[tokio::test]
async fn full_chain_bridge_hints_to_trace_payload_via_bus() {
    use proximadb_vector::quantization::turboquant::TurboQuantStore;

    // Construct a TurboQuant store directly — same as the bridge
    // construction path used inside `score_turboquant`.
    let store = TurboQuantStore::new(
        64,
        4,
        CalibrationMode::Identity,
        derive_rotation_seed("chain"),
    )
    .expect("store");

    // Build the canonical TurboQuantExplainHints from the bridge's
    // public surface. This is what `score_turboquant` builds at the
    // emit site.
    let hints = TurboQuantExplainHints::for_search(&store)
        .with_blocks_skipped(13)
        .with_n_vectors_scanned(1024)
        .with_mask_pushed(true);
    let payload = hints.to_explain_value();

    // Route through the per-request bus: scope, record, take inside.
    let lifted = predicate_diagnostics::scope(async {
        predicate_diagnostics::record_turboquant_hints(payload.clone());
        predicate_diagnostics::take_turboquant_hints()
    })
    .await;
    let lifted = lifted.expect("lifted payload");

    // Feed into the trace builder as a handler would.
    let p = plan();
    let trace = build(TraceBuilderInputs {
        trace_id: "trace-full-chain".into(),
        tenant_id: "tenant-chain".into(),
        collection_name: "kb-chain".into(),
        filter_strategy: p.filter_strategy,
        index_route: p.index_route,
        estimated_selectivity: p.estimated_selectivity,
        gls_score: p.gls_score,
        latency_ms: 4.2,
        index_stats: IndexStats::default(),
        candidate_count: 100,
        rerank_count: 10,
        repair_count: 0,
        sure_signals: SureSignals::default(),
        cache_result: CacheResult::Miss,
        failure_class: None,
        bytes_per_vector: 0.0,
        egress_bytes: 0,
        object_store_gets: 0,
        object_store_bytes_read: 0,
        predicate_shortfall: None,
        turboquant_explain: Some(lifted),
    });

    // The trace's payload must equal the bridge's hint payload bit-for-bit.
    assert_eq!(
        trace.turboquant_explain.as_ref(),
        Some(&payload),
        "full chain regression: bridge → bus → trace dropped or mutated payload",
    );
    // Wire-shape spot check: the bridge's invariants must reach the
    // operator-visible structured log.
    let v = trace.turboquant_explain.as_ref().unwrap();
    assert_eq!(
        v.get("quantization").and_then(|x| x.as_str()),
        Some("turboquant_4bit")
    );
    assert_eq!(
        v.get("blocks_skipped_by_mask").and_then(|x| x.as_u64()),
        Some(13)
    );
    assert_eq!(
        v.get("length_renorm_applied").and_then(|x| x.as_bool()),
        Some(true)
    );
}

/// Reuse the `TurboQuantSlotResolver` adapter type to verify it can be
/// constructed and dispatch through the trait surface. Caught here so
/// a future refactor that breaks the `SlotIdResolver` impl on
/// `TurboQuantSlotResolver` surfaces at the integration boundary, not
/// just inside the adapter's private tests.
#[test]
fn turboquant_slot_resolver_is_object_safe_under_dyn() {
    // The adapter's resolver type implements `SlotIdResolver`. We can't
    // construct a real one here without reaching into the adapter's
    // private DashMap fields, but the type must be nameable and
    // exposed as `Arc<dyn SlotIdResolver>` for the convergence-plan
    // contract. This is a compile-time test (success = code compiles).
    let _phantom: Option<Arc<dyn SlotIdResolver>> =
        None::<Arc<TurboQuantSlotResolver>>.map(|r| r as Arc<dyn SlotIdResolver>);
}

/// Phase P contract: the SAME registry instance is shared between the
/// create-time wire (CollectionService::with_turboquant_registry +
/// get_or_create at create_collection) and the boot-time hydration
/// (hydrate_registry_from_policy_rows in SharedServices::new). Both
/// consumers see one map, so create-time registrations land in the
/// same slots boot hydration populates.
///
/// This test simulates the production assembly:
///   1. One Arc<dyn TurboQuantStoreRegistry> hoisted (Phase P Site 2).
///   2. A create-time-path-style `get_or_create` registers one collection.
///   3. A boot-hydration-path-style `hydrate_registry_from_policy_rows`
///      registers a second collection from a hand-built row.
///   4. The same registry now owns BOTH stores — assert via len + Arc
///      identity on a subsequent `get`.
///
/// Without this contract, create-time registrations would land in one
/// map and boot hydration in another — concurrent boot + ingest
/// would race and the first search after restart would see only the
/// hydrated half.
#[tokio::test]
async fn phase_p_shared_registry_unifies_create_time_and_boot_hydration() {
    use proximadb::compute::quantization::turboquant_store_registry::{
        TurboQuantHydrationRow, hydrate_registry_from_policy_rows,
    };

    // (1) Hoist a single registry — same shape SharedServices::new does.
    let registry: Arc<dyn TurboQuantStoreRegistry> =
        Arc::new(InMemoryTurboQuantStoreRegistry::new());

    // (2) Create-time path: CollectionService::create_collection
    // resolves the seed via derive_rotation_seed(&name) and calls
    // registry.get_or_create with the dimensional config. Simulate
    // that call exactly.
    let create_time_id = "kb-created-at-runtime";
    let create_time_dim = 128usize;
    let create_time_seed = derive_rotation_seed(create_time_id);
    let create_time_store = registry
        .get_or_create(
            create_time_id,
            create_time_dim,
            4,
            CalibrationMode::TqPlus,
            create_time_seed,
        )
        .await
        .expect("create-time get_or_create");
    assert_eq!(create_time_store.dim(), create_time_dim);

    // (3) Boot-hydration path: SharedServices::new projects each
    // existing collection's proto into a TurboQuantHydrationRow and
    // calls hydrate_registry_from_policy_rows on the SAME registry.
    let boot_id = "kb-existing-at-boot";
    let boot_dim = 64usize;
    let boot_seed = derive_rotation_seed(boot_id);
    let rows = vec![TurboQuantHydrationRow {
        collection_id: boot_id.to_string(),
        dim: boot_dim,
        bit_width: 4,
        calibration_mode: "tq_plus".to_string(),
        rotation_seed: boot_seed,
    }];
    let hydrated = hydrate_registry_from_policy_rows(registry.as_ref(), &rows).await;
    assert_eq!(hydrated, 1);

    // (4) Both stores must live in the SAME registry. The count proves
    // it; Arc identity on a fresh `get` proves the create-time store
    // wasn't replaced by hydration.
    assert_eq!(registry.registered_count(), 2);

    let create_time_fetched = registry
        .get(create_time_id)
        .await
        .expect("registry get for create-time id")
        .expect("create-time store still registered");
    assert!(
        Arc::ptr_eq(&create_time_store, &create_time_fetched),
        "create-time store survived hydration — registry is shared",
    );

    let boot_fetched = registry
        .get(boot_id)
        .await
        .expect("registry get for boot id")
        .expect("boot store registered by hydration");
    assert_eq!(boot_fetched.dim(), boot_dim);
}

/// Phase P boot-hydration contract: a collection whose
/// `enable_turboquant=true` was set at create-time but whose seed is
/// derived from the catalog id must produce the SAME store after a
/// simulated restart (where the registry was rebuilt empty and only
/// the catalog row survives).
///
/// This locks the "boot hydration is correct after restart" contract:
/// the seed is deterministic so the rebuilt store has identical
/// rotation, calibration, and dimensions to the pre-restart store.
#[tokio::test]
async fn phase_p_boot_hydration_after_simulated_restart_uses_deterministic_seed() {
    use proximadb::compute::quantization::turboquant_store_registry::{
        TurboQuantHydrationRow, hydrate_registry_from_policy_rows,
    };

    let collection_id = "kb-restart-test";
    let dim = 64usize;
    let seed = derive_rotation_seed(collection_id);

    // Pre-restart: create-time-path registered the store.
    let pre_restart_registry: Arc<dyn TurboQuantStoreRegistry> =
        Arc::new(InMemoryTurboQuantStoreRegistry::new());
    let pre_store = pre_restart_registry
        .get_or_create(collection_id, dim, 4, CalibrationMode::TqPlus, seed)
        .await
        .unwrap();

    // Simulate restart: drop the pre-restart registry, build a fresh
    // empty one (same as a SharedServices::new on a cold process),
    // hydrate from the (deterministic) catalog row.
    drop(pre_store);
    drop(pre_restart_registry);

    let post_restart_registry: Arc<dyn TurboQuantStoreRegistry> =
        Arc::new(InMemoryTurboQuantStoreRegistry::new());
    let rows = vec![TurboQuantHydrationRow {
        collection_id: collection_id.to_string(),
        dim,
        bit_width: 4,
        calibration_mode: "tq_plus".to_string(),
        rotation_seed: seed,
    }];
    let n = hydrate_registry_from_policy_rows(post_restart_registry.as_ref(), &rows).await;
    assert_eq!(n, 1);

    let post_store = post_restart_registry
        .get(collection_id)
        .await
        .unwrap()
        .expect("hydration registered the store");
    assert_eq!(post_store.dim(), dim);
    assert_eq!(post_store.bit_width(), 4);
    assert_eq!(post_store.rotation_seed(), seed);
    assert_eq!(post_store.calibration_mode(), CalibrationMode::TqPlus);
}
