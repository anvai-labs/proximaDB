//! Stack-based interpreter for expression bytecode.
//!
//! The hot loop is a single `match` on [`Op`] with a `pc` cursor and a
//! [`SmallVec`]-backed operand stack. For typical R-2 / R-3 expressions
//! (≤ 16 stack slots) the stack stays inline — no heap allocation per
//! `execute()` call. Larger expressions spill silently to the heap.
//!
//! NaN policy: `if(cond, then, else)` treats any non-zero cond — including
//! NaN — as truthy. The else branch fires only on exact `cond == 0.0`.
//! Users that want NaN → else should explicitly compose `min/max` guards
//! around the cond.

use crate::bytecode::{Code, Op};
use proximadb_rank_core::{DocHandle, FeatureExecutor, FeatureLookup, ScoreCtx};
use smallvec::SmallVec;

/// Run the bytecode and return the final stack value. Panics if the
/// program is malformed (e.g. underflow or wrong arity at emit-time —
/// both are lowering bugs, not user input).
pub fn execute(
    code: &Code,
    sub_features: &mut [Box<dyn FeatureExecutor>],
    doc: DocHandle,
    lookup: &mut dyn FeatureLookup,
    ctx: &mut ScoreCtx<'_>,
) -> f32 {
    // SmallVec inline capacity sized to cover typical R-3 expressions.
    // The bench gate in R-3 §6.4 (`vm_no_alloc_per_doc`) asserts this stays
    // in steady state.
    let mut stack: SmallVec<[f32; 32]> = SmallVec::with_capacity(code.max_stack as usize);
    let mut pc: usize = 0;
    while pc < code.ops.len() {
        let op = code.ops[pc];
        pc += 1;
        match op {
            Op::PushConst(v) => stack.push(v),
            Op::PushSubFeature(idx) => {
                let v = sub_features[idx as usize].execute(doc, lookup, ctx);
                stack.push(v);
            }
            Op::Add => binop(&mut stack, |a, b| a + b),
            Op::Sub => binop(&mut stack, |a, b| a - b),
            Op::Mul => binop(&mut stack, |a, b| a * b),
            Op::Div => binop(&mut stack, |a, b| a / b),
            Op::Pow => binop(&mut stack, |a, b| a.powf(b)),
            Op::Neg => {
                let v = stack.pop().expect("vm underflow at Neg");
                stack.push(-v);
            }
            Op::Min => binop(&mut stack, f32::min),
            Op::Max => binop(&mut stack, f32::max),
            Op::Clamp => {
                let hi = stack.pop().expect("vm underflow at Clamp[hi]");
                let lo = stack.pop().expect("vm underflow at Clamp[lo]");
                let v = stack.pop().expect("vm underflow at Clamp[val]");
                stack.push(v.clamp(lo, hi));
            }
            Op::Abs => unary(&mut stack, f32::abs),
            Op::Sqrt => unary(&mut stack, f32::sqrt),
            Op::Log => unary(&mut stack, f32::ln),
            Op::Exp => unary(&mut stack, f32::exp),
            Op::Sigmoid => unary(&mut stack, |x| 1.0 / (1.0 + (-x).exp())),
            Op::Relu => unary(&mut stack, |x| x.max(0.0)),
            Op::Tanh => unary(&mut stack, f32::tanh),
            Op::JumpIfZero(target) => {
                let cond = stack.pop().expect("vm underflow at JumpIfZero");
                // Non-zero (including NaN) = truthy → fall through to then.
                if cond == 0.0 {
                    pc = target as usize;
                }
            }
            Op::Jump(target) => {
                pc = target as usize;
            }
        }
    }
    stack.pop().expect("vm produced empty stack — lowering bug")
}

#[inline(always)]
fn binop(stack: &mut SmallVec<[f32; 32]>, f: impl FnOnce(f32, f32) -> f32) {
    let b = stack.pop().expect("vm underflow at binop[rhs]");
    let a = stack.pop().expect("vm underflow at binop[lhs]");
    stack.push(f(a, b));
}

#[inline(always)]
fn unary(stack: &mut SmallVec<[f32; 32]>, f: impl FnOnce(f32) -> f32) {
    let v = stack.pop().expect("vm underflow at unary");
    stack.push(f(v));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Code;
    use crate::lowering::lower;
    use crate::parser::parse;
    use proximadb_rank_core::test_support::MapAttributeAccess;
    use proximadb_rank_core::{
        Blueprint, BlueprintFactory, FeatureArena, NoopAttributeAccess, NoopCandidateData,
        NoopMetricsSink, NoopModelCache, OutputSpec, PhaseConfig, QueryContext, RankResult,
    };
    use std::collections::HashMap;
    use std::sync::Arc;

    struct NullLookup;
    impl FeatureLookup for NullLookup {
        fn force(
            &mut self,
            _idx: proximadb_rank_core::ExecutorIdx,
            _doc: DocHandle,
            _ctx: &mut ScoreCtx<'_>,
        ) -> f32 {
            0.0
        }
    }

    fn run(code: &Code, subs: &mut [Box<dyn FeatureExecutor>]) -> f32 {
        let q = QueryContext::default();
        let arena = FeatureArena::new();
        let (a, c, m, met) = (
            NoopAttributeAccess,
            NoopCandidateData,
            NoopModelCache,
            NoopMetricsSink,
        );
        let mut ctx = ScoreCtx::new(&q, &arena, &a, &c, &m, &met);
        let mut lk = NullLookup;
        execute(code, subs, DocHandle(0), &mut lk, &mut ctx)
    }

    fn lower_str(s: &str, factory: &BlueprintFactory) -> (Code, Vec<Box<dyn FeatureExecutor>>) {
        let ast = parse(s).unwrap();
        let q = QueryContext::default();
        lower(&ast, factory, &q).unwrap()
    }

    #[test]
    fn vm_evaluates_arithmetic() {
        let f = BlueprintFactory::new();
        let (code, mut subs) = lower_str("1 + 2 * 3", &f);
        assert_eq!(run(&code, &mut subs), 7.0);
    }

    #[test]
    fn vm_handles_unary_neg_and_pow() {
        let f = BlueprintFactory::new();
        let (code, mut subs) = lower_str("-2 ^ 3", &f);
        // -(2^3) = -8 per spec parsing.
        assert!((run(&code, &mut subs) - (-8.0)).abs() < 1e-5);
    }

    #[test]
    fn vm_handles_division_by_zero_as_inf() {
        let f = BlueprintFactory::new();
        let (code, mut subs) = lower_str("1 / 0", &f);
        assert!(run(&code, &mut subs).is_infinite());
    }

    #[test]
    fn vm_evaluates_min_max() {
        let f = BlueprintFactory::new();
        let (code, mut subs) = lower_str("max(min(5, 10), 3)", &f);
        assert_eq!(run(&code, &mut subs), 5.0);
    }

    #[test]
    fn vm_evaluates_clamp() {
        let f = BlueprintFactory::new();
        // clamp(7, 0, 5) → 5
        let (code, mut subs) = lower_str("clamp(7, 0, 5)", &f);
        assert_eq!(run(&code, &mut subs), 5.0);
        // clamp(-3, 0, 5) → 0
        let (code, mut subs) = lower_str("clamp(-3, 0, 5)", &f);
        assert_eq!(run(&code, &mut subs), 0.0);
        // clamp(2, 0, 5) → 2
        let (code, mut subs) = lower_str("clamp(2, 0, 5)", &f);
        assert_eq!(run(&code, &mut subs), 2.0);
    }

    #[test]
    fn vm_evaluates_abs_sqrt_log_exp() {
        let f = BlueprintFactory::new();
        let (code, mut subs) = lower_str("abs(-3)", &f);
        assert_eq!(run(&code, &mut subs), 3.0);
        let (code, mut subs) = lower_str("sqrt(9)", &f);
        assert_eq!(run(&code, &mut subs), 3.0);
        let (code, mut subs) = lower_str("log(1)", &f);
        assert_eq!(run(&code, &mut subs), 0.0);
        let (code, mut subs) = lower_str("exp(0)", &f);
        assert_eq!(run(&code, &mut subs), 1.0);
    }

    #[test]
    fn vm_evaluates_sigmoid_relu_tanh() {
        let f = BlueprintFactory::new();
        let (code, mut subs) = lower_str("sigmoid(0)", &f);
        assert!((run(&code, &mut subs) - 0.5).abs() < 1e-6);
        let (code, mut subs) = lower_str("relu(-1)", &f);
        assert_eq!(run(&code, &mut subs), 0.0);
        let (code, mut subs) = lower_str("relu(2.5)", &f);
        assert_eq!(run(&code, &mut subs), 2.5);
        let (code, mut subs) = lower_str("tanh(0)", &f);
        assert!(run(&code, &mut subs).abs() < 1e-6);
    }

    #[test]
    fn vm_if_truthy_picks_then_branch() {
        let f = BlueprintFactory::new();
        let (code, mut subs) = lower_str("if(1, 100, 999)", &f);
        assert_eq!(run(&code, &mut subs), 100.0);
    }

    #[test]
    fn vm_if_zero_cond_picks_else_branch() {
        let f = BlueprintFactory::new();
        let (code, mut subs) = lower_str("if(0, 100, 999)", &f);
        assert_eq!(run(&code, &mut subs), 999.0);
    }

    #[test]
    fn vm_if_short_circuits_else_branch() {
        // Build an "else" branch that pushes a sub-feature whose execute
        // would explode (we'll use a "must-not-fire" executor). When
        // cond is truthy, the else branch must not run.
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct MustNotFireBp(Arc<AtomicUsize>);
        struct MustNotFireEx(Arc<AtomicUsize>);
        impl FeatureExecutor for MustNotFireEx {
            fn execute(
                &mut self,
                _doc: DocHandle,
                _l: &mut dyn FeatureLookup,
                _c: &mut ScoreCtx<'_>,
            ) -> f32 {
                self.0.fetch_add(1, Ordering::SeqCst);
                0.0
            }
        }
        impl Blueprint for MustNotFireBp {
            fn name(&self) -> &str {
                "must_not_fire"
            }
            fn declared_outputs(&self) -> &[OutputSpec] {
                const OUT: &[OutputSpec] = &[];
                OUT
            }
            fn build_executor(
                &self,
                _cfg: &PhaseConfig,
                _q: &QueryContext,
            ) -> RankResult<Box<dyn FeatureExecutor>> {
                Ok(Box::new(MustNotFireEx(self.0.clone())))
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let f = BlueprintFactory::new();
        f.register(Arc::new(MustNotFireBp(counter.clone())));

        // cond = 1 (truthy) → must_not_fire's execute should not run.
        let (code, mut subs) = lower_str("if(1, 42, must_not_fire())", &f);
        assert_eq!(run(&code, &mut subs), 42.0);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "else-branch sub-feature must NOT have executed"
        );

        // cond = 0 → must_not_fire should run exactly once.
        let (code, mut subs) = lower_str("if(0, 42, must_not_fire())", &f);
        let _ = run(&code, &mut subs);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn vm_evaluates_with_feature_refs() {
        struct AttrLikeBp;
        struct AttrLikeEx {
            field: String,
        }
        impl FeatureExecutor for AttrLikeEx {
            fn execute(
                &mut self,
                doc: DocHandle,
                _l: &mut dyn FeatureLookup,
                ctx: &mut ScoreCtx<'_>,
            ) -> f32 {
                ctx.attributes.read_f32(doc, &self.field).unwrap_or(0.0)
            }
        }
        impl Blueprint for AttrLikeBp {
            fn name(&self) -> &str {
                "attr"
            }
            fn declared_outputs(&self) -> &[OutputSpec] {
                const OUT: &[OutputSpec] = &[];
                OUT
            }
            fn build_executor(
                &self,
                cfg: &PhaseConfig,
                _q: &QueryContext,
            ) -> RankResult<Box<dyn FeatureExecutor>> {
                let f = cfg.literal_args[0].clone();
                Ok(Box::new(AttrLikeEx { field: f }))
            }
        }

        let f = BlueprintFactory::new();
        f.register(Arc::new(AttrLikeBp));
        let (code, mut subs) = lower_str("attr(\"a\") * 2 + attr(\"b\")", &f);

        let attrs = MapAttributeAccess {
            values: HashMap::from([((0, "a".into()), 3.0), ((0, "b".into()), 4.0)]),
        };
        let q = QueryContext::default();
        let arena = FeatureArena::new();
        let (c, m, met) = (NoopCandidateData, NoopModelCache, NoopMetricsSink);
        let mut ctx = ScoreCtx::new(&q, &arena, &attrs, &c, &m, &met);
        let mut lk = NullLookup;
        // 3*2 + 4 = 10
        assert_eq!(
            execute(&code, &mut subs, DocHandle(0), &mut lk, &mut ctx),
            10.0
        );
    }
}
