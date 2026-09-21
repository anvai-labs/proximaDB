// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Offline route-cost **regret** eval — the "prove offline first" gate (mandate #6)
//! for closing the routing loop, and a CI-gated eval of a decision surface
//! (mandate #13).
//!
//! It drives the **real** [`RouteCostModel`] — `observe` / `recommend_override` /
//! `exploration_choice`, the exact code a live override would run — over a
//! calibrated per-`(shape-class, backend)` cost surface, and measures the gated
//! controller's **cumulative regret vs. the static heuristic** against the
//! per-shape oracle. The claim under test is the controller *mechanism*: "given
//! measured per-arm costs, the gated cost router learns to pick the cheaper arm,
//! with sublinear regret, and never does worse where the static rule is already
//! right." That is the evidence required to later flip
//! `PROXIMADB_ROUTE_COST_OVERRIDE` — this test itself ships **no** production
//! routing change (it constructs an isolated model, never the process-global one).
//!
//! Honest framing (mandate #6): this is a controller-convergence proof on a cost
//! surface **calibrated to measured cost ratios**, not an end-to-end latency
//! claim. Cost is carried through the model's own `score()` via `range_gets` (the
//! dominant per-request term, `per_get = 20`), so the surface exercises the real
//! scoring path; a small deterministic `compute_ms` jitter makes repeated
//! observations of one arm vary (realistic EWMA learning) without crossing the
//! inter-arm gap. Fully deterministic (the model has no internal RNG; jitter and
//! workload order are functions of the round index) → a stable CI ratchet.

use std::collections::BTreeMap;

use proximadb::observability::io_trace::IoTraceSnapshot;
use proximadb::query::compute_scheduler::{CardinalityClass, PartitionFanout, QueryShape};
use proximadb::query::route_cost_model::{RouteCostModel, shape_class};
use proximadb::query::table_write_plan::ComputeBackend;

/// Per-arm mean cost in `range_gets` (× `per_get`=20 → score units).
struct Arm {
    backend: ComputeBackend,
    mean_gets: u64,
}

/// One workload shape: its `QueryShape` (→ real shape-class key), the backend the
/// **static** heuristic picks for it, and the freshness-safe candidate arms with
/// their ground-truth mean costs.
struct Case {
    shape: QueryShape,
    static_backend: ComputeBackend,
    arms: Vec<Arm>,
}

impl Case {
    fn class(&self) -> String {
        shape_class(&self.shape)
    }
    fn candidates(&self) -> Vec<ComputeBackend> {
        self.arms.iter().map(|a| a.backend.clone()).collect()
    }
    fn mean_gets(&self, backend: &ComputeBackend) -> u64 {
        self.arms
            .iter()
            .find(|a| std::mem::discriminant(&a.backend) == std::mem::discriminant(backend))
            .map(|a| a.mean_gets)
            .unwrap_or(0)
    }
    /// Noise-free expected score for decision-regret (per_get=20). The constant
    /// jitter mean cancels in any regret difference, so it is omitted here.
    fn expected_cost(&self, backend: &ComputeBackend) -> f64 {
        20.0 * self.mean_gets(backend) as f64
    }
    fn oracle_cost(&self) -> f64 {
        self.arms
            .iter()
            .map(|a| 20.0 * a.mean_gets as f64)
            .fold(f64::INFINITY, f64::min)
    }
}

/// A measured trace for `mean_gets` GETs plus a small deterministic compute jitter
/// (5..=15 score units) — well inside the inter-arm gap, so the controller can
/// still separate arms but each observation differs (exercises the EWMA fold).
fn snapshot_for(mean_gets: u64, round: u64) -> IoTraceSnapshot {
    let jitter = 5 + (round.wrapping_mul(7) % 11); // deterministic 5..=15
    let mut compute_ms = BTreeMap::new();
    compute_ms.insert("sim".to_string(), jitter);
    IoTraceSnapshot {
        range_gets: mean_gets,
        compute_ms,
        ..Default::default()
    }
}

/// One controller decision for `case`, mirroring `route_select_advised`'s order:
/// (1) bounded/rate-limited exploration of an under-warmed freshness-safe arm,
/// (2) else a confident cost override of the static choice, (3) else static.
fn controller_choice(model: &RouteCostModel, case: &Case) -> ComputeBackend {
    let class = case.class();
    let candidates = case.candidates();
    if let Some(b) = model.exploration_choice(&class, &candidates) {
        return b;
    }
    if let Some(rec) = model.recommend_override(&class, &case.static_backend, &candidates) {
        return rec.backend;
    }
    case.static_backend.clone()
}

struct Regret {
    controller: f64,
    static_: f64,
}

/// Run the simulation and return per-case + windowed regret.
fn simulate(rounds: u64) -> (Vec<(String, Regret)>, f64, f64) {
    // Two shapes that share the static rule (relational + parquet → DataFusion)
    // but differ in the *right* answer:
    //   A: plain OLAP/parquet — DataFusion really is cheaper (static is correct).
    //   B: large + many-partition OLAP/parquet — DataFusion's per-partition GET
    //      fan-out makes it the *more* expensive arm; Native wins (static wrong).
    let case_a = Case {
        shape: QueryShape {
            engages_relational: true,
            parquet_backed: true,
            ..Default::default()
        },
        static_backend: ComputeBackend::DataFusionLocal,
        arms: vec![
            Arm {
                backend: ComputeBackend::DataFusionLocal,
                mean_gets: 5,
            }, // 100
            Arm {
                backend: ComputeBackend::Native,
                mean_gets: 8,
            }, // 160
        ],
    };
    let case_b = Case {
        shape: QueryShape {
            engages_relational: true,
            parquet_backed: true,
            cardinality: CardinalityClass::Large,
            partition_fanout: PartitionFanout::Many,
            pax_backed: false,
            native_registerable: false,
            // TD-ROUTE-3: a native-eligible (single-scan scalar-aggregate) shape,
            // so the controller flipping this case to Native reflects a route the
            // real `route_select_advised` would actually take (Native is not an
            // eligible flip target for a non-servable shape).
            operation_class: proximadb::query::compute_scheduler::OperationClass::ScalarAggregate,
            geometry: Default::default(),
            join_bearing: false,
        },
        static_backend: ComputeBackend::DataFusionLocal,
        arms: vec![
            Arm {
                backend: ComputeBackend::DataFusionLocal,
                mean_gets: 11,
            }, // 220 (fan-out)
            Arm {
                backend: ComputeBackend::Native,
                mean_gets: 6,
            }, // 120
        ],
    };

    // Fresh, isolated model (NOT the process-global). Warm fast and explore often
    // so a CI-sized round count converges; override default 15% advantage kept.
    let model = RouteCostModel::new()
        .with_min_samples(3)
        .with_exploration_interval(4);
    model.set_override_enabled(true);

    let mut reg_a = Regret {
        controller: 0.0,
        static_: 0.0,
    };
    let mut reg_b = Regret {
        controller: 0.0,
        static_: 0.0,
    };
    let mut ctrl_first_q = 0.0;
    let mut ctrl_last_q = 0.0;

    for round in 0..rounds {
        let case = if round % 2 == 0 { &case_a } else { &case_b };
        let reg = if round % 2 == 0 {
            &mut reg_a
        } else {
            &mut reg_b
        };

        let chosen = controller_choice(&model, case);
        model.observe(
            &case.class(),
            &chosen,
            &snapshot_for(case.mean_gets(&chosen), round),
        );

        let oracle = case.oracle_cost();
        let ctrl_step = case.expected_cost(&chosen) - oracle;
        reg.controller += ctrl_step;
        reg.static_ += case.expected_cost(&case.static_backend) - oracle;

        // Learning signal: controller regret in the first vs last quarter.
        if round < rounds / 4 {
            ctrl_first_q += ctrl_step;
        } else if round >= rounds - rounds / 4 {
            ctrl_last_q += ctrl_step;
        }
    }

    let per_case = vec![(case_a.class(), reg_a), (case_b.class(), reg_b)];
    (per_case, ctrl_first_q, ctrl_last_q)
}

#[test]
fn gated_cost_router_beats_static_heuristic_offline() {
    let rounds = 4_000;
    let (per_case, ctrl_first_q, ctrl_last_q) = simulate(rounds);

    let mut ctrl_total = 0.0;
    let mut static_total = 0.0;
    eprintln!("\n=== offline route-cost regret (rounds={rounds}) ===");
    eprintln!(
        "{:<32} {:>14} {:>14}",
        "shape-class", "ctrl-regret", "static-regret"
    );
    for (class, r) in &per_case {
        eprintln!("{class:<32} {:>14.0} {:>14.0}", r.controller, r.static_);
        ctrl_total += r.controller;
        static_total += r.static_;

        if class.contains("part=m") {
            // Static is SUBOPTIMAL here (always picks the costlier DataFusion arm),
            // so its regret grows ~linearly; the controller must learn to flip.
            assert!(
                r.static_ > 1_000.0,
                "static heuristic should accumulate large regret on the static-suboptimal \
                 shape-class, got {:.0}",
                r.static_
            );
            assert!(
                r.controller < r.static_ * 0.25,
                "controller should cut >=75% of static's regret where static is wrong: \
                 ctrl={:.0} static={:.0}",
                r.controller,
                r.static_
            );
        } else {
            // Static is OPTIMAL here — the controller must do no meaningful harm.
            // Some bounded regret from forced exploration warming the worse arm is
            // expected, but it stays a small slice of what a wrong static would cost.
            assert!(
                r.controller < 6_000.0,
                "controller should add only bounded exploration regret where static is \
                 already optimal, got {:.0}",
                r.controller
            );
        }
    }

    eprintln!(
        "{:<32} {:>14.0} {:>14.0}",
        "TOTAL", ctrl_total, static_total
    );
    eprintln!(
        "learning: first-quarter ctrl-regret={ctrl_first_q:.0}, last-quarter={ctrl_last_q:.0}\n"
    );

    // Overall: the controller cuts total regret well below the static heuristic.
    assert!(
        ctrl_total < static_total * 0.6,
        "controller total regret ({ctrl_total:.0}) should be well under static ({static_total:.0})"
    );
    // Learning: regret per round falls sharply once arms are warm (sublinear).
    assert!(
        ctrl_last_q < ctrl_first_q * 0.5,
        "controller regret should fall as it learns: first-q={ctrl_first_q:.0}, last-q={ctrl_last_q:.0}"
    );
}
