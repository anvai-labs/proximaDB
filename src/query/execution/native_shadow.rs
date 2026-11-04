// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! # Native-vs-Volcano shadow comparison + auto-demotion (ADR-054 §7 Phase 0.5)
//!
//! The correctness safety-net for the experimental native vectorized/join path
//! (TD-OLAP-11 §Gate). When `PROXIMADB_NATIVE_JOIN_SHADOW` is set,
//! `NativeVolcanoEngine::execute_physical` runs BOTH the native path and the
//! Volcano reference for a query, compares their result multisets, logs any
//! discrepancy, and — per ADR-054 §7 / TD-OLAP-11 §Gate — **auto-demotes the
//! query shape** when the row-level mismatch rate exceeds
//! [`MISMATCH_DEMOTE_THRESHOLD`] (0.1%). A demoted shape skips the native path on
//! subsequent runs ([`is_shape_demoted`], consulted by `try_vectorized`), and the
//! diverging run itself fails safe to the Volcano result.
//!
//! Comparison is **order-insensitive** (a multiset over per-row canonical
//! strings): join/aggregate output order is not guaranteed to match between the
//! two engines, so only row *content and multiplicity* are compared.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use proximadb_relational_planner::PhysicalPlan;

use super::engine::ExecutionPipelineResult;

/// Per-query-shape divergence rate above which the native path is auto-demoted.
/// ADR-054 §7 Phase 0.5 / TD-OLAP-11 §Gate: "mismatch rate >0.1% → auto-demote".
pub const MISMATCH_DEMOTE_THRESHOLD: f64 = 0.001;

proximadb_config::env_gate! {
    /// Gate: is native-vs-Volcano shadow comparison opted in? Default OFF — shadow
    /// mode runs every native-handled query twice (native + Volcano), so it is a
    /// validation/CI switch, not a production default (TD-OLAP-11 §Gate flag,
    /// distinct from `PROXIMADB_NATIVE_JOIN`).
    pub fn native_join_shadow_enabled() { gate: "PROXIMADB_NATIVE_JOIN_SHADOW", default: false }
}

/// The outcome of one shadow comparison.
#[derive(Debug, Clone, Copy)]
pub struct ShadowVerdict {
    /// Row-level mismatch rate in `[0, 1]` (symmetric multiset difference / rows).
    pub mismatch_rate: f64,
    /// `true` when `mismatch_rate` exceeded the threshold — the shape is now
    /// demoted and the caller MUST fail safe to the reference (Volcano) result.
    pub diverged: bool,
}

/// Process-global set of demoted query-shape signatures.
fn demoted_shapes() -> &'static Mutex<HashSet<u64>> {
    static REG: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Has a prior shadow run demoted this plan's shape? Consulted by
/// `try_vectorized` to skip the native path for a known-divergent shape.
pub fn is_shape_demoted(plan: &PhysicalPlan) -> bool {
    let sig = plan_shape_signature(plan);
    demoted_shapes()
        .lock()
        .map(|g| g.contains(&sig))
        .unwrap_or(false)
}

/// Compare a native result against the Volcano reference, log any discrepancy,
/// and demote the shape when the mismatch rate exceeds the threshold. Returns the
/// verdict; on divergence the caller MUST return the reference result.
pub fn shadow_compare_and_record(
    plan: &PhysicalPlan,
    native: &ExecutionPipelineResult,
    reference: &ExecutionPipelineResult,
) -> ShadowVerdict {
    let mismatch_rate = mismatch_rate(native, reference);
    let diverged = mismatch_rate > MISMATCH_DEMOTE_THRESHOLD;
    let sig = plan_shape_signature(plan);
    if diverged {
        if let Ok(mut g) = demoted_shapes().lock() {
            g.insert(sig);
        }
        tracing::warn!(
            target: "proximadb::native_shadow",
            mismatch_rate,
            shape = sig,
            native_rows = native.rows.len(),
            reference_rows = reference.rows.len(),
            "native path diverged from Volcano reference; auto-demoting shape and failing safe to reference"
        );
    } else {
        tracing::debug!(
            target: "proximadb::native_shadow",
            mismatch_rate,
            shape = sig,
            rows = native.rows.len(),
            "native path matched Volcano reference within tolerance"
        );
    }
    ShadowVerdict {
        mismatch_rate,
        diverged,
    }
}

/// Order-insensitive row-multiset mismatch rate in `[0, 1]`. A schema-shape
/// mismatch (different column count) is a total mismatch (`1.0`).
fn mismatch_rate(a: &ExecutionPipelineResult, b: &ExecutionPipelineResult) -> f64 {
    if a.schema.columns.len() != b.schema.columns.len() {
        return 1.0;
    }
    let total = a.rows.len().max(b.rows.len());
    if total == 0 {
        return 0.0;
    }
    let mut counts: HashMap<String, i64> = HashMap::new();
    for row in &a.rows {
        *counts.entry(canonical_row(row)).or_default() += 1;
    }
    for row in &b.rows {
        *counts.entry(canonical_row(row)).or_default() -= 1;
    }
    // Each surplus/deficit row on either side counts once toward the mismatch.
    let unmatched: i64 = counts.values().map(|c| c.abs()).sum();
    ((unmatched as f64) / (total as f64)).min(1.0)
}

/// Canonical, order-stable string for one result row (its column values in
/// order). Uses `Debug` — good enough to detect content divergence; two
/// semantically-equal-but-differently-typed cells count as a mismatch, which is
/// the conservative direction for a correctness net.
fn canonical_row(row: &[proximadb_data_model::ProximaValue]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for cell in row {
        // Unit-separator between cells so column boundaries can't blur.
        let _ = write!(s, "{cell:?}\u{1f}");
    }
    s
}

/// A structural signature of a plan that is invariant to literal DATA (Values
/// payloads, literal values, predicate constants) but captures operator shape,
/// join kind, arity, and column widths — the "query shape" the demotion registry
/// keys on (ADR-054 §7: "auto-demote native for that query shape").
pub fn plan_shape_signature(plan: &PhysicalPlan) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut buf = String::new();
    shape_into(plan, &mut buf);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    buf.hash(&mut h);
    h.finish()
}

fn shape_into(plan: &PhysicalPlan, out: &mut String) {
    use std::fmt::Write;
    match plan {
        PhysicalPlan::Scan {
            table,
            output_schema,
            projection,
            predicate,
            ..
        } => {
            let _ = write!(
                out,
                "Scan[t={table:?},cols={},proj={},pred={}];",
                output_schema.columns.len(),
                projection.is_some(),
                predicate.is_some()
            );
        }
        PhysicalPlan::Filter { input, .. } => {
            out.push_str("Filter(");
            shape_into(input, out);
            out.push_str(");");
        }
        PhysicalPlan::Project { input, outputs } => {
            let _ = write!(out, "Project[{}](", outputs.len());
            shape_into(input, out);
            out.push_str(");");
        }
        PhysicalPlan::Join {
            left,
            right,
            kind,
            on,
            strategy,
        } => {
            let _ = write!(out, "Join[{kind:?},{strategy:?},on={}](", on.is_some());
            shape_into(left, out);
            out.push(',');
            shape_into(right, out);
            out.push_str(");");
        }
        PhysicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
            having,
            strategy,
        } => {
            let _ = write!(
                out,
                "Agg[gb={},ag={},hav={},{strategy:?}](",
                group_by.len(),
                aggregates.len(),
                having.is_some()
            );
            shape_into(input, out);
            out.push_str(");");
        }
        PhysicalPlan::Sort {
            input,
            keys,
            strategy,
        } => {
            let _ = write!(out, "Sort[{},{strategy:?}](", keys.len());
            shape_into(input, out);
            out.push_str(");");
        }
        PhysicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            let _ = write!(out, "Limit[{limit:?},{offset}](");
            shape_into(input, out);
            out.push_str(");");
        }
        PhysicalPlan::Distinct { input, strategy } => {
            let _ = write!(out, "Distinct[{strategy:?}](");
            shape_into(input, out);
            out.push_str(");");
        }
        PhysicalPlan::AssertMaxOneRow { input } => {
            out.push_str("AssertMax1(");
            shape_into(input, out);
            out.push_str(");");
        }
        PhysicalPlan::Union { inputs, all } => {
            let _ = write!(out, "Union[all={all}](");
            for i in inputs {
                shape_into(i, out);
                out.push(',');
            }
            out.push_str(");");
        }
        PhysicalPlan::SetOp {
            op,
            left,
            right,
            all,
        } => {
            let _ = write!(out, "SetOp[{op:?},all={all}](");
            shape_into(left, out);
            out.push(',');
            shape_into(right, out);
            out.push_str(");");
        }
        PhysicalPlan::Values {
            rows,
            output_schema,
        } => {
            // DATA-INVARIANT: only the row COUNT + column width, never payloads.
            let _ = write!(
                out,
                "Values[rows={},cols={}];",
                rows.len(),
                output_schema.columns.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_data_model::{ProximaType, ProximaValue};
    use proximadb_relational_types::{ColumnInfo, RelationalSchema};

    fn result(rows: Vec<Vec<ProximaValue>>) -> ExecutionPipelineResult {
        let schema = RelationalSchema::new(vec![
            ColumnInfo::new("k", ProximaType::Int64, true),
            ColumnInfo::new("v", ProximaType::String, true),
        ]);
        ExecutionPipelineResult { schema, rows }
    }

    fn row(k: i64, v: &str) -> Vec<ProximaValue> {
        vec![ProximaValue::Int64(k), ProximaValue::String(v.to_string())]
    }

    #[test]
    fn identical_results_match_regardless_of_order() {
        let a = result(vec![row(1, "a"), row(2, "b"), row(3, "c")]);
        // Same multiset, permuted order → must NOT be flagged as divergence.
        let b = result(vec![row(3, "c"), row(1, "a"), row(2, "b")]);
        assert_eq!(mismatch_rate(&a, &b), 0.0);
    }

    #[test]
    fn one_wrong_row_exceeds_threshold() {
        let a = result(vec![row(1, "a"), row(2, "b")]);
        let b = result(vec![row(1, "a"), row(2, "WRONG")]);
        // Two surplus/deficit rows over a 2-row set → well above 0.1%.
        assert!(mismatch_rate(&a, &b) > MISMATCH_DEMOTE_THRESHOLD);
    }

    #[test]
    fn column_count_mismatch_is_total() {
        let a = result(vec![row(1, "a")]);
        let mut b = a.clone();
        b.schema = RelationalSchema::new(vec![ColumnInfo::new("k", ProximaType::Int64, true)]);
        assert_eq!(mismatch_rate(&a, &b), 1.0);
    }

    #[test]
    fn shape_signature_ignores_values_payload_but_tracks_shape() {
        use proximadb_relational_planner::PhysicalPlan;
        let schema = RelationalSchema::new(vec![ColumnInfo::new("k", ProximaType::Int64, true)]);
        let mk = |vals: &[i64]| PhysicalPlan::Values {
            rows: vals
                .iter()
                .map(|&n| {
                    vec![proximadb_relational_types::Expr::Literal {
                        value: ProximaValue::Int64(n),
                        ty: ProximaType::Int64,
                    }]
                })
                .collect(),
            output_schema: schema.clone(),
        };
        // Same shape (row count + width), different payload → same signature.
        assert_eq!(
            plan_shape_signature(&mk(&[1, 2, 3])),
            plan_shape_signature(&mk(&[7, 8, 9]))
        );
        // Different row count → different signature (shape changed).
        assert_ne!(
            plan_shape_signature(&mk(&[1, 2, 3])),
            plan_shape_signature(&mk(&[1, 2]))
        );
    }
}
