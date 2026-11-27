//! Integration tests for the selector benchmark harness (TD-SELECTOR-1).
//!
//! These assert *properties of the harness*, not production quality:
//!
//! - the candidate pool really is preserved — adding selectors to a run cannot
//!   change what another selector measures;
//! - `NoOp` reproduces the pool's own recall ceiling, so the ceiling is measurable;
//! - a selector with a finite rerank window scores BELOW that ceiling, and widening
//!   the window recovers it — the mechanism by which PR #1726's MiniLM selector
//!   turned a 0.3734 pool into 0.3136 delivered recall;
//! - the statistics are real: a selector compared against itself is never
//!   significant, and a recommendation is withheld when nothing is significant.
//!
//! The data is synthetic (see `selector_benchmark_data`), so no absolute number
//! here is evidence about ProximaDB's production retrieval quality.

#[path = "../benches/answer_quality_data.rs"]
mod answer_quality_data;
#[path = "../benches/bench_answer_quality.rs"]
mod bench_answer_quality;
#[path = "../benches/bench_selector_benchmark.rs"]
mod bench_selector_benchmark;
#[path = "../benches/selector_benchmark_data.rs"]
mod selector_benchmark_data;

use bench_selector_benchmark::{
    GroundTruth, ResultSource, SelectorBenchmark, SelectorConfig, SelectorResult, SelectorStrategy,
    recall_at_ks, stats,
};
use selector_benchmark_data::{
    DataGenConfig, RelevanceDistribution, ScoreDistribution, generate_benchmark_data,
    pool_coverage_ceiling, pool_recall_ceiling,
};

/// Fewer resamples than the 1000 default: these tests assert direction and
/// ordering, not p-value precision, and the permutation loop is O(resamples × n).
const TEST_RESAMPLES: usize = 200;

/// One seeded bed shared by every test, so results are comparable across them.
///
/// Deliberately *under-covered*: each query has ~140-200 relevant documents but each
/// retriever surfaces at most 300 candidates of which 25-30% are relevant, so the
/// union pool holds only a fraction of the relevant set. A bed where the pool
/// already contains everything makes recall saturate near 1.0 and stops
/// discriminating between selectors — which is exactly the regime PR #1726 is *not*
/// in (a 0.3734 pool ceiling).
fn bed() -> DataGenConfig {
    DataGenConfig {
        num_queries: 40,
        num_documents: 2_000,
        num_relevant_per_query: 200,
        candidate_pool_size: 300,
        relevance_distribution: RelevanceDistribution::SkewedLow,
        score_distribution: ScoreDistribution::Normal {
            mean: 0.0,
            std: 0.15,
        },
        vector_relevant_rate: 0.30,
        bm25_relevant_rate: 0.25,
        score_separation: 0.25,
        seed: 0x0DEF_ACED_0000_0001,
    }
}

fn run(configs: Vec<SelectorConfig>) -> bench_selector_benchmark::BenchmarkResults {
    let (ground_truth, pools, queries) = generate_benchmark_data(&bed());
    SelectorBenchmark::new(ground_truth, pools, queries)
        .with_resamples(TEST_RESAMPLES)
        .run(configs)
        .expect("benchmark runs")
}

#[test]
fn noop_reproduces_the_pools_own_raw_score_ordering() {
    let (ground_truth, pools, queries) = generate_benchmark_data(&bed());
    let baseline = pool_recall_ceiling(&pools, &ground_truth, 100);
    assert!(
        baseline > 0.0,
        "the bed must put some relevant documents in the pool"
    );

    let results = SelectorBenchmark::new(ground_truth, pools, queries)
        .with_resamples(TEST_RESAMPLES)
        .run(vec![SelectorConfig::noop("noop")])
        .expect("benchmark runs");

    let measured = results.mean_recall_at("noop", 100).expect("Recall@100");
    assert!(
        (measured - baseline).abs() < 1e-9,
        "NoOp must reproduce the pool's own ordering exactly: {measured} vs {baseline}"
    );
}

#[test]
fn no_selector_can_exceed_the_pool_coverage_ceiling() {
    // The only true bound: a selector can drop candidates, never invent them. The
    // *ordering* baseline (`NoOp`) is beatable — a better-ranking selector legitimately
    // exceeds it — so coverage, not NoOp, is what bounds every strategy.
    let (ground_truth, pools, queries) = generate_benchmark_data(&bed());
    let ceiling = pool_coverage_ceiling(&pools, &ground_truth);
    assert!(
        ceiling < 0.95,
        "an under-covered bed is the point: ceiling {ceiling:.4} is too saturated to \
         discriminate between selectors"
    );

    let results = SelectorBenchmark::new(ground_truth, pools, queries)
        .with_resamples(TEST_RESAMPLES)
        .run(vec![
            SelectorConfig::noop("noop"),
            SelectorConfig::fusion("weighted", SelectorStrategy::WeightedFusion, 0.6),
            SelectorConfig::fusion("rrf", SelectorStrategy::ReciprocalRankFusion, 0.5),
            SelectorConfig::fusion("cascade_vec", SelectorStrategy::CascadeVectorFirst, 0.6),
            SelectorConfig::reranker("cross_encoder", "minilm-l12-v2", 100, 0.8),
        ])
        .expect("benchmark runs");

    for name in ["noop", "weighted", "rrf", "cascade_vec", "cross_encoder"] {
        let recall = results.mean_recall_at(name, 100).expect("Recall@100");
        assert!(
            recall <= ceiling + 1e-12,
            "{name} reported Recall@100 {recall:.4} above the pool's coverage ceiling \
             {ceiling:.4} — the harness is measuring something the pool does not contain"
        );
    }
}

#[test]
fn adding_selectors_to_a_run_cannot_change_what_another_measures() {
    // The pool-preservation guarantee, stated as an observable: `noop` scores
    // identically whether it runs alone or beside four rival selectors.
    let alone = run(vec![SelectorConfig::noop("noop")]);
    let together = run(vec![
        SelectorConfig::noop("noop"),
        SelectorConfig::fusion("weighted", SelectorStrategy::WeightedFusion, 0.6),
        SelectorConfig::fusion("rrf", SelectorStrategy::ReciprocalRankFusion, 0.5),
        SelectorConfig::fusion("cascade", SelectorStrategy::CascadeVectorFirst, 0.6),
        SelectorConfig::reranker("cross_encoder", "minilm-l12-v2", 100, 0.8),
    ]);

    assert_eq!(together.aggregate_metrics.len(), 5);
    for k in [1, 10, 100] {
        let a = alone.mean_recall_at("noop", k).expect("alone");
        let b = together.mean_recall_at("noop", k).expect("together");
        assert_eq!(a, b, "Recall@{k} for noop must not depend on its rivals");
    }
    assert_eq!(
        alone.aggregate("noop").expect("a").mean_mrr,
        together.aggregate("noop").expect("b").mean_mrr
    );
}

#[test]
fn fusion_selectors_reach_the_whole_pool_but_a_cascade_does_not() {
    // A cascade is gated on one retriever's top-`rerank_window`, so most of the
    // preserved pool is unreachable to it. That structural truncation — not model
    // quality — is what costs recall.
    let results = run(vec![
        SelectorConfig::fusion("weighted", SelectorStrategy::WeightedFusion, 0.6),
        SelectorConfig::fusion("rrf", SelectorStrategy::ReciprocalRankFusion, 0.5),
        SelectorConfig::fusion("cascade_vec", SelectorStrategy::CascadeVectorFirst, 0.6),
        SelectorConfig::fusion("cascade_bm25", SelectorStrategy::CascadeBM25First, 0.6),
    ]);

    for name in ["weighted", "rrf"] {
        let utilization = results
            .aggregate(name)
            .expect("aggregate")
            .mean_pool_utilization;
        assert!(
            utilization > 99.0,
            "{name} fuses both retrievers and should reach the whole pool, got {utilization:.2}%"
        );
    }
    for name in ["cascade_vec", "cascade_bm25"] {
        let utilization = results
            .aggregate(name)
            .expect("aggregate")
            .mean_pool_utilization;
        assert!(
            utilization < 50.0,
            "{name} is gated on a 100-deep first stage and must reach far less of the \
             pool, got {utilization:.2}%"
        );
    }
}

#[test]
fn a_truncating_reranker_scores_below_its_own_upstream() {
    // The PR #1726 failure mode, stated so the comparison is airtight: the reranker
    // is measured against the exact ranking it consumes. A 50-deep window cannot
    // emit 100 results, so Recall@100 is structurally capped at the upstream's
    // Recall@50 — no amount of model quality recovers it.
    let results = run(vec![
        SelectorConfig::fusion("weighted", SelectorStrategy::WeightedFusion, 0.6),
        SelectorConfig::reranker("reranker_w50", "minilm-l12-v2", 50, 0.9),
    ]);

    let upstream = results.mean_recall_at("weighted", 100).expect("upstream");
    let reranked = results
        .mean_recall_at("reranker_w50", 100)
        .expect("reranked");

    assert!(
        reranked < upstream,
        "a window-truncating reranker must lose recall its upstream already held: \
         {reranked:.4} vs upstream {upstream:.4}"
    );
    assert_eq!(
        reranked,
        results.mean_recall_at("weighted", 50).expect("upstream@50"),
        "the loss is exactly truncation: Recall@100 of a 50-window reranker IS the \
         upstream's Recall@50"
    );

    let comparison = results
        .comparison("weighted", "reranker_w50", "Recall@100")
        .expect("paired comparison exists");
    assert!(
        comparison.statistically_significant,
        "the regression should be significant (p = {:.4})",
        comparison.p_value
    );
    assert_eq!(comparison.winner.as_deref(), Some("weighted"));
}

#[test]
fn widening_the_rerank_window_monotonically_recovers_recall() {
    let results = run(vec![
        SelectorConfig::reranker("w25", "minilm-l12-v2", 25, 1.0),
        SelectorConfig::reranker("w50", "minilm-l12-v2", 50, 1.0),
        SelectorConfig::reranker("w100", "minilm-l12-v2", 100, 1.0),
        SelectorConfig::reranker("w400", "minilm-l12-v2", 400, 1.0),
    ]);

    let mut previous = f64::NEG_INFINITY;
    for name in ["w25", "w50", "w100", "w400"] {
        let recall = results.mean_recall_at(name, 100).expect("Recall@100");
        assert!(
            recall >= previous - 1e-12,
            "Recall@100 must not fall as the rerank window widens: {name} = {recall:.4}, \
             previous = {previous:.4}"
        );
        previous = recall;
    }
    // Past k the window stops mattering: w100 and w400 both emit the full top-100.
    assert_eq!(
        results.mean_recall_at("w100", 100).expect("w100"),
        results.mean_recall_at("w400", 100).expect("w400")
    );
}

#[test]
fn an_untruncated_perfect_reranker_preserves_its_upstream_exactly() {
    // Isolates truncation as the cause of the regression above: with an unbounded
    // window and perfect fidelity the reranker reproduces its upstream ranking, so
    // it matches weighted fusion to the digit.
    let results = run(vec![
        SelectorConfig::fusion("weighted", SelectorStrategy::WeightedFusion, 0.6),
        SelectorConfig::reranker("perfect", "oracle", usize::MAX, 1.0),
    ]);

    for k in [1, 10, 100] {
        assert_eq!(
            results.mean_recall_at("weighted", k).expect("weighted"),
            results.mean_recall_at("perfect", k).expect("perfect"),
            "an untruncated perfect reranker must not move Recall@{k}"
        );
    }
}

#[test]
fn a_lower_fidelity_reranker_ranks_worse_within_the_same_window() {
    // Same window for both, so any difference is scoring quality, not truncation.
    let results = run(vec![
        SelectorConfig::reranker("high_quality", "bge-reranker-large", 100, 1.0),
        SelectorConfig::reranker("low_quality", "minilm-l12-v2", 100, 0.1),
    ]);

    let high = results
        .aggregate("high_quality")
        .expect("high")
        .mean_ndcg_at_10;
    let low = results
        .aggregate("low_quality")
        .expect("low")
        .mean_ndcg_at_10;
    assert!(
        high > low,
        "a higher-fidelity reranker must order the window better: NDCG@10 {high:.4} vs {low:.4}"
    );

    // Truncation is identical, so Recall@100 — which only cares about set membership
    // in the emitted 100 — is unchanged by fidelity.
    assert_eq!(
        results.mean_recall_at("high_quality", 100).expect("high"),
        results.mean_recall_at("low_quality", 100).expect("low"),
        "fidelity reorders the window; it does not change which documents survive it"
    );
}

#[test]
fn recall_at_k_is_recall_not_precision() {
    // 4 relevant documents; the selector returns exactly one of them at rank 1.
    // Recall@1 = 1/4 = 0.25. Precision@1 would be 1.0 — the bug this guards.
    let ground_truth = GroundTruth {
        query_id: "q".to_string(),
        relevant_docs: vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ],
    };
    let results: Vec<SelectorResult> = ["a", "z", "b", "y"]
        .iter()
        .enumerate()
        .map(|(rank, id)| SelectorResult {
            doc_id: id.to_string(),
            score: 1.0 - rank as f64 * 0.1,
            rank,
            source: ResultSource::Hybrid,
        })
        .collect();

    let recall = recall_at_ks(&results, &ground_truth, &[1, 4]);
    assert_eq!(recall.get(&1).copied(), Some(0.25));
    assert_eq!(recall.get(&4).copied(), Some(0.5));
}

#[test]
fn a_selector_compared_against_itself_is_never_significant() {
    let results = run(vec![
        SelectorConfig::fusion("rrf_a", SelectorStrategy::ReciprocalRankFusion, 0.5),
        SelectorConfig::fusion("rrf_b", SelectorStrategy::ReciprocalRankFusion, 0.5),
    ]);

    for metric in ["Recall@100", "MRR", "NDCG@10"] {
        let comparison = results
            .comparison("rrf_a", "rrf_b", metric)
            .expect("comparison exists");
        assert_eq!(
            comparison.mean_difference, 0.0,
            "{metric} difference must be zero"
        );
        assert_eq!(comparison.p_value, 1.0, "{metric} p-value must be 1.0");
        assert!(!comparison.statistically_significant);
        assert!(comparison.winner.is_none());
    }

    // Nothing beat anything, so nothing is promotable (plan gate 3).
    let recommendation = results.recommendations.first().expect("one recommendation");
    assert_eq!(recommendation.confidence, 0.0);
    assert!(
        recommendation.reasoning.contains("Not promotable"),
        "reasoning must say why: {}",
        recommendation.reasoning
    );
}

#[test]
fn a_significant_winner_is_promoted_with_its_p_value() {
    let results = run(vec![
        SelectorConfig::fusion("weighted", SelectorStrategy::WeightedFusion, 0.6),
        SelectorConfig::reranker("reranker_w25", "minilm-l12-v2", 25, 0.9),
    ]);

    let recommendation = results.recommendations.first().expect("one recommendation");
    assert_eq!(recommendation.selector, "weighted");
    assert!(
        recommendation.confidence > 0.9,
        "confidence is 1 - p of the deciding comparison, got {}",
        recommendation.confidence
    );
    assert!(recommendation.reasoning.contains("significantly better"));
}

#[test]
fn bootstrap_intervals_bracket_the_mean_and_widen_with_variance() {
    let tight: Vec<f64> = (0..50).map(|i| 0.50 + (i % 2) as f64 * 0.01).collect();
    let loose: Vec<f64> = (0..50).map(|i| (i % 10) as f64 / 10.0).collect();

    let (lo_t, hi_t) = stats::bootstrap_ci(&tight, 500, 7);
    let (lo_l, hi_l) = stats::bootstrap_ci(&loose, 500, 7);

    assert!(lo_t <= stats::mean(&tight) && stats::mean(&tight) <= hi_t);
    assert!(lo_l <= stats::mean(&loose) && stats::mean(&loose) <= hi_l);
    assert!(
        (hi_l - lo_l) > (hi_t - lo_t),
        "the higher-variance sample must produce the wider interval"
    );

    // Degenerate inputs must not panic.
    assert_eq!(stats::bootstrap_ci(&[], 500, 7), (0.0, 0.0));
    assert_eq!(stats::bootstrap_ci(&[0.42], 500, 7), (0.42, 0.42));
}

#[test]
fn sign_flip_p_separates_a_real_shift_from_noise() {
    // A consistent +0.1 on every query: the sign-flip null almost never reaches it.
    let shifted: Vec<f64> = vec![0.1; 40];
    assert!(stats::sign_flip_p(&shifted, 1000, 11) < 0.01);
    // p is never exactly zero — finite resampling cannot support that certainty.
    assert!(stats::sign_flip_p(&shifted, 1000, 11) > 0.0);

    // Symmetric alternating differences: no shift, no significance.
    let noise: Vec<f64> = (0..40)
        .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
        .collect();
    assert!(stats::sign_flip_p(&noise, 1000, 11) > 0.05);

    assert_eq!(stats::sign_flip_p(&[], 1000, 11), 1.0);
}

/// The plan-step-2 sweep: every strategy, one preserved pool, paired statistics.
/// Run with `cargo nextest run --no-capture selector_sweep_report` to read the table.
///
/// The assertions here are the ones the protocol guarantees; the printed numbers are
/// synthetic-bed measurements and are not evidence about production retrieval.
#[test]
fn selector_sweep_report() {
    let (ground_truth, pools, queries) = generate_benchmark_data(&bed());
    let ceiling = pool_coverage_ceiling(&pools, &ground_truth);
    let results = SelectorBenchmark::new(ground_truth, pools, queries)
        .with_resamples(TEST_RESAMPLES)
        .run(vec![
            SelectorConfig::noop("noop_raw_union"),
            SelectorConfig::fusion("weighted_60_40", SelectorStrategy::WeightedFusion, 0.6),
            SelectorConfig::fusion("weighted_50_50", SelectorStrategy::WeightedFusion, 0.5),
            SelectorConfig::fusion("rrf_k60", SelectorStrategy::ReciprocalRankFusion, 0.5),
            SelectorConfig::fusion(
                "cascade_vector_first",
                SelectorStrategy::CascadeVectorFirst,
                0.6,
            ),
            SelectorConfig::fusion(
                "cascade_bm25_first",
                SelectorStrategy::CascadeBM25First,
                0.6,
            ),
            SelectorConfig::reranker("xenc_w50_q90", "minilm-l12-v2", 50, 0.9),
            SelectorConfig::reranker("xenc_w100_q90", "minilm-l12-v2", 100, 0.9),
            SelectorConfig::reranker("xenc_w400_q90", "bge-reranker-large", 400, 0.9),
        ])
        .expect("benchmark runs");

    println!("pool coverage ceiling = {ceiling:.4}\n{}", results.render());
    for comparison in results
        .pairwise_comparisons
        .iter()
        .filter(|c| c.metric == "Recall@100" && c.statistically_significant)
    {
        println!(
            "{:<22} vs {:<22} d={:+.4} CI[{:+.4},{:+.4}] p={:.4}",
            comparison.selector_a,
            comparison.selector_b,
            comparison.mean_difference,
            comparison.difference_ci.0,
            comparison.difference_ci.1,
            comparison.p_value
        );
    }
    for recommendation in &results.recommendations {
        println!(
            "recommendation: {} (confidence {:.4}) — {}",
            recommendation.selector, recommendation.confidence, recommendation.reasoning
        );
    }

    // Every strategy stays under the coverage bound.
    for config in &results.selector_configs {
        let recall = results
            .mean_recall_at(&config.name, 100)
            .expect("Recall@100");
        assert!(
            recall <= ceiling + 1e-12,
            "{} exceeded coverage",
            config.name
        );
    }
    // The narrow-window reranker is the worst strategy in the sweep — truncation
    // costs more recall than any fusion choice gains.
    let worst = results
        .selector_configs
        .iter()
        .map(|c| {
            (
                c.name.clone(),
                results.mean_recall_at(&c.name, 100).unwrap_or(0.0),
            )
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("at least one selector");
    assert_eq!(worst.0, "xenc_w50_q90");
}

#[test]
fn duplicate_selector_names_are_rejected() {
    let (ground_truth, pools, queries) = generate_benchmark_data(&bed());
    let error = SelectorBenchmark::new(ground_truth, pools, queries)
        .with_resamples(TEST_RESAMPLES)
        .run(vec![
            SelectorConfig::noop("same"),
            SelectorConfig::noop("same"),
        ])
        .expect_err("duplicate names must be rejected");
    assert!(error.to_string().contains("duplicate selector name"));
}

// ===========================================================================
// Gate 4 — span / answer quality, scored on the selectors' real rankings
// ===========================================================================

use answer_quality_data::{QaBed, QaBedConfig, generate_qa_bed, reachable_fraction};
use bench_answer_quality::{
    AnswerExtractor, AnswerQualityEvaluator, AnswerQualityResults, BestSentence, SelectorRankings,
    WholeTopDocument,
};

/// Run gate 2's selectors over the QA bed and hand their *actual* rankings to gate 4.
/// Nothing is re-simulated: the answer scores below are scored on the same lists the
/// selector benchmark measures.
fn answer_quality_over(
    bed: &QaBed,
    configs: &[SelectorConfig],
    extractors: Vec<Box<dyn AnswerExtractor>>,
) -> AnswerQualityResults {
    let benchmark = SelectorBenchmark::new(
        bed.ground_truth.clone(),
        bed.candidate_pools.clone(),
        bed.queries.clone(),
    );
    let rankings: Vec<(String, SelectorRankings)> = configs
        .iter()
        .map(|config| {
            (
                config.name.clone(),
                benchmark
                    .rankings(config)
                    .expect("selector produces rankings"),
            )
        })
        .collect();

    AnswerQualityEvaluator::new(extractors)
        .with_resamples(TEST_RESAMPLES)
        .evaluate(&bed.qa_pairs, &bed.corpus, &rankings)
        .expect("answer-quality evaluation runs")
}

#[test]
fn answer_quality_collapses_when_the_selector_drops_the_gold_document() {
    // The end-to-end statement of PR #1726's finding: a selector that truncates does
    // not merely lose Recall@100 — it removes the only document that carries the
    // answer, and no extractor downstream can recover it.
    let bed = generate_qa_bed(&QaBedConfig::default());
    let configs = vec![
        SelectorConfig::noop("keeps_everything"),
        SelectorConfig::reranker("truncates_at_25", "minilm-l12-v2", 25, 1.0),
    ];
    let results = answer_quality_over(&bed, &configs, vec![Box::new(BestSentence::new(200))]);

    let full = results
        .get("keeps_everything", "best_sentence")
        .expect("full");
    let truncated = results
        .get("truncates_at_25", "best_sentence")
        .expect("cut");

    assert!(
        full.mean_gold_doc_retrieved > truncated.mean_gold_doc_retrieved,
        "truncation must cost gold-document retrieval: {:.4} vs {:.4}",
        full.mean_gold_doc_retrieved,
        truncated.mean_gold_doc_retrieved
    );
    assert!(
        full.mean_f1 > truncated.mean_f1 && full.mean_exact_match > truncated.mean_exact_match,
        "and that cost must reach the answer: F1 {:.4} vs {:.4}, EM {:.4} vs {:.4}",
        full.mean_f1,
        truncated.mean_f1,
        full.mean_exact_match,
        truncated.mean_exact_match
    );

    let comparison = results
        .comparison("best_sentence", "keeps_everything", "truncates_at_25", "F1")
        .expect("paired F1 comparison");
    assert!(
        comparison.statistically_significant,
        "the answer-quality regression must be significant (p = {:.4})",
        comparison.p_value
    );
    assert_eq!(comparison.winner.as_deref(), Some("keeps_everything"));
}

#[test]
fn exact_match_never_exceeds_gold_document_retrieval() {
    // The bound that makes gate 4 interpretable: an extractor cannot produce the gold
    // answer from a ranking that does not contain the gold document. If this fails,
    // the bed is leaking the answer through a distractor.
    let bed = generate_qa_bed(&QaBedConfig::default());
    let configs = vec![
        SelectorConfig::noop("keeps_everything"),
        SelectorConfig::reranker("truncates_at_10", "minilm-l12-v2", 10, 1.0),
        SelectorConfig::reranker("truncates_at_50", "minilm-l12-v2", 50, 1.0),
    ];
    let results = answer_quality_over(&bed, &configs, vec![Box::new(BestSentence::new(200))]);

    for config in &configs {
        let aggregate = results
            .get(&config.name, "best_sentence")
            .expect("aggregate");
        assert!(
            aggregate.mean_exact_match <= aggregate.mean_gold_doc_retrieved + 1e-12,
            "{}: EM {:.4} exceeded gold-doc retrieval {:.4}",
            config.name,
            aggregate.mean_exact_match,
            aggregate.mean_gold_doc_retrieved
        );
    }
}

#[test]
fn the_extractors_read_depth_caps_answer_quality_independently_of_the_selector() {
    // Two failure modes look identical in the aggregate — the selector dropped the
    // document, or the extractor never read that far. Holding the selector fixed and
    // varying only read depth separates them.
    let bed = generate_qa_bed(&QaBedConfig::default());
    let configs = vec![SelectorConfig::noop("keeps_everything")];
    let shallow = answer_quality_over(&bed, &configs, vec![Box::new(BestSentence::new(5))]);
    let deep = answer_quality_over(&bed, &configs, vec![Box::new(BestSentence::new(200))]);

    let shallow_f1 = shallow
        .get("keeps_everything", "best_sentence")
        .expect("shallow")
        .mean_f1;
    let deep_f1 = deep
        .get("keeps_everything", "best_sentence")
        .expect("deep")
        .mean_f1;
    assert!(
        deep_f1 > shallow_f1,
        "reading deeper into an unchanged ranking must help: {deep_f1:.4} vs {shallow_f1:.4}"
    );

    // Gold-document retrieval is a property of the selector, so it is identical for
    // both extractors — the difference above is extraction, not retrieval.
    assert_eq!(
        shallow
            .get("keeps_everything", "best_sentence")
            .expect("shallow")
            .mean_gold_doc_retrieved,
        deep.get("keeps_everything", "best_sentence")
            .expect("deep")
            .mean_gold_doc_retrieved
    );
}

#[test]
fn returning_a_whole_document_trades_precision_for_recall() {
    let bed = generate_qa_bed(&QaBedConfig::default());
    let configs = vec![SelectorConfig::noop("keeps_everything")];
    let results = answer_quality_over(
        &bed,
        &configs,
        vec![Box::new(WholeTopDocument), Box::new(BestSentence::new(200))],
    );

    let whole = results
        .get("keeps_everything", "whole_top_document")
        .expect("whole");
    let sentence = results
        .get("keeps_everything", "best_sentence")
        .expect("sentence");

    assert!(
        sentence.mean_precision > whole.mean_precision,
        "a sentence answer must be more precise than a whole document: {:.4} vs {:.4}",
        sentence.mean_precision,
        whole.mean_precision
    );
    assert!(
        sentence.mean_f1 > whole.mean_f1,
        "and better on F1: {:.4} vs {:.4}",
        sentence.mean_f1,
        whole.mean_f1
    );
    assert!(
        whole.mean_exact_match < 0.05,
        "returning an entire document is essentially never an exact match, got {:.4}",
        whole.mean_exact_match
    );
}

#[test]
fn a_correct_span_scores_exactly_on_both_boundaries_and_iou() {
    // When the extractor finds the gold sentence it must agree with gold on the
    // character offsets, not merely on the text — otherwise span metrics are decorative.
    let bed = generate_qa_bed(&QaBedConfig::default());
    let configs = vec![SelectorConfig::noop("keeps_everything")];
    let results = answer_quality_over(&bed, &configs, vec![Box::new(BestSentence::new(200))]);
    let aggregate = results
        .get("keeps_everything", "best_sentence")
        .expect("aggregate");

    assert!(
        (aggregate.mean_span_exact - aggregate.mean_exact_match).abs() < 1e-12,
        "span-exact {:.4} and EM {:.4} must agree: the extractor reports the span it \
         quoted from",
        aggregate.mean_span_exact,
        aggregate.mean_exact_match
    );
    assert!(aggregate.mean_span_iou >= aggregate.mean_span_exact - 1e-12);
}

#[test]
fn the_qa_bed_puts_gold_documents_out_of_reach_of_shallow_selectors() {
    // Guards the bed itself: if every gold document ranked in the top 10, gate 4
    // would report "truncation is free" and be measuring nothing.
    let bed = generate_qa_bed(&QaBedConfig::default());
    assert!(reachable_fraction(&bed, 10) < 0.35);
    assert!(reachable_fraction(&bed, 25) < 0.55);
    assert!(reachable_fraction(&bed, 200) > 0.95);
}

/// The gate-4 sweep. Run with
/// `cargo nextest run --no-capture answer_quality_report` to read the table.
#[test]
fn answer_quality_report() {
    let bed = generate_qa_bed(&QaBedConfig::default());
    let configs = vec![
        SelectorConfig::noop("keeps_everything"),
        SelectorConfig::fusion("weighted_60_40", SelectorStrategy::WeightedFusion, 0.6),
        SelectorConfig::fusion("rrf_k60", SelectorStrategy::ReciprocalRankFusion, 0.5),
        SelectorConfig::reranker("xenc_w10", "minilm-l12-v2", 10, 1.0),
        SelectorConfig::reranker("xenc_w25", "minilm-l12-v2", 25, 1.0),
        SelectorConfig::reranker("xenc_w100", "bge-reranker-large", 100, 1.0),
    ];
    let results = answer_quality_over(
        &bed,
        &configs,
        vec![Box::new(BestSentence::new(200)), Box::new(WholeTopDocument)],
    );

    println!(
        "gold-document reachability: top-10 {:.4}, top-25 {:.4}, top-100 {:.4}\n{}",
        reachable_fraction(&bed, 10),
        reachable_fraction(&bed, 25),
        reachable_fraction(&bed, 100),
        results.render()
    );
    for comparison in results
        .comparisons
        .iter()
        .filter(|c| c.metric == "F1" && c.statistically_significant)
    {
        println!(
            "{:<14} {:<18} vs {:<18} d={:+.4} CI[{:+.4},{:+.4}] p={:.4}",
            comparison.extractor,
            comparison.selector_a,
            comparison.selector_b,
            comparison.mean_difference,
            comparison.difference_ci.0,
            comparison.difference_ci.1,
            comparison.p_value
        );
    }

    // Answer quality must be monotone in how much of the ranking the selector kept.
    let f1 = |name: &str| {
        results
            .get(name, "best_sentence")
            .expect("aggregate")
            .mean_f1
    };
    assert!(f1("xenc_w10") < f1("xenc_w25"));
    assert!(f1("xenc_w25") < f1("xenc_w100"));
    assert!(f1("xenc_w100") <= f1("keeps_everything") + 1e-12);
}

// ===========================================================================
// Gate 5 — real ONNX reranking under the selector protocol
// ===========================================================================
//
// The `SelectorStrategy::CrossEncoderRerank` above is a *modelled* reranker
// (window + quality parameters) — deliberately, so the default build needs no
// model artifacts. This section runs the REAL production scorer
// (`BertPairTokenizingDocFeatureExtractor` → `OrtTokenizedScorerSession`) as one
// more selector over the same preserved pools, through the same answer-quality
// protocol and paired statistics. Feature-gated on the root `onnx` feature
// (which forwards rank-onnx's real-onnx + bert-tokenizer), env-gated on the
// registered model-path gates, so default CI compiles nothing here.

#[cfg(feature = "onnx")]
mod onnx_selector {
    //! The REAL production scorer as one more selector under the gate-4
    //! protocol: same preserved pools, same `AnswerQualityEvaluator`, same
    //! paired statistics as the modelled strategies — the comparison isolates
    //! real-vs-modelled *scoring*, not window policy. Feature-gated on the root
    //! `onnx` feature (which forwards rank-onnx's real-onnx + bert-tokenizer)
    //! and env-gated on the registered `PROXIMADB_TEST_BERT_ONNX_PATH`, so
    //! default CI compiles nothing from this module.

    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::{Arc, RwLock};

    use anyhow::{Result, anyhow};

    use super::answer_quality_data::{QaBed, QaBedConfig, generate_qa_bed};
    use super::bench_answer_quality::{
        AnswerQualityEvaluator, BestSentence, SelectorRankings, WholeTopDocument, sentence_bounds,
    };
    use super::bench_selector_benchmark::{
        ResultSource, SelectorBenchmark, SelectorConfig, SelectorResult,
    };

    /// Shared doc-text source: ONE stateless extractor serves every request,
    /// with per-call window texts swapped in (the R-5b.1.3 production shape).
    /// This also keeps the `tokenizers` type inside rank-onnx — the root crate
    /// does not depend on it directly.
    struct SharedTexts(RwLock<HashMap<u32, String>>);

    impl proximadb_rank_onnx::bert_pair_tokenizing_extractor::DocTextSource for SharedTexts {
        fn doc_text(
            &self,
            doc: proximadb_rank_core::DocHandle,
        ) -> proximadb_rank_core::RankResult<Option<String>> {
            Ok(self
                .0
                .read()
                .ok()
                .and_then(|texts| texts.get(&doc.0).cloned()))
        }
    }

    /// The real production scorer: one extractor + one session behind a
    /// per-(query, doc) call. Docs longer than the model budget are split into
    /// sentence windows (the #1726 zero-truncation protocol) and scored in ONE
    /// batched session call — max-over-window is the doc score.
    struct RealScorer {
        extractor: proximadb_rank_onnx::bert_pair_tokenizing_extractor::BertPairTokenizingDocFeatureExtractor,
        texts: Arc<SharedTexts>,
        session: proximadb_rank_onnx::ort_tokenized_scorer_session::OrtTokenizedScorerSession,
    }

    impl RealScorer {
        fn load(model_path: &Path, input_slots: usize) -> Result<Self> {
            use proximadb_rank_onnx::descriptor::{
                DType, ModelDescriptor, ModelFramework, ModelKey, TensorIoSpec,
            };

            let max_seq_len = 512usize;
            let emits_tti = input_slots == 3;
            let slot_names: &[&str] = if emits_tti {
                &["input_ids", "attention_mask", "token_type_ids"]
            } else {
                &["input_ids", "attention_mask"]
            };
            // The tokenizer loads through the extractor's own file constructor —
            // the root crate never names the `tokenizers` types directly.
            let texts = Arc::new(SharedTexts(RwLock::new(HashMap::new())));
            let extractor =
                proximadb_rank_onnx::bert_pair_tokenizing_extractor::BertPairTokenizingDocFeatureExtractor::from_tokenizer_file(
                    &model_path.with_file_name("tokenizer.json"),
                    texts.clone(),
                    max_seq_len,
                    emits_tti,
                )
                .map_err(|e| anyhow!("extractor construction: {e}"))?;
            let descriptor = ModelDescriptor {
                key: ModelKey::new("selector-bench-reranker", "gate5"),
                tenant: None,
                uri: format!("file://{}", model_path.display()),
                sha256: [0; 32],
                size_bytes: 0,
                framework: ModelFramework::Onnx,
                dtype: DType::Fp32,
                input_spec: slot_names
                    .iter()
                    .map(|name| TensorIoSpec {
                        name: (*name).into(),
                        shape: vec![None, Some(max_seq_len as i64)],
                        dtype: DType::Fp32,
                    })
                    .collect(),
                output_spec: vec![TensorIoSpec {
                    name: "logits".into(),
                    shape: vec![None],
                    dtype: DType::Fp32,
                }],
                max_batch_size: 64,
                seq: 0,
                created_at_ms: 0,
            };
            let session =
                proximadb_rank_onnx::ort_tokenized_scorer_session::OrtTokenizedScorerSession::load_from_file(
                    descriptor,
                    model_path,
                )?;
            Ok(Self {
                extractor,
                texts,
                session,
            })
        }

        /// Score one (query, doc) pair. Windows that still overflow the budget
        /// are dropped (zero-truncation), not scored truncated; a doc with
        /// nothing scoreable keeps a finite tail-of-window slot (0.0).
        fn score_pair(&self, query: &str, doc: &str) -> f64 {
            use proximadb_rank_core::{DocHandle, QueryContext};
            use proximadb_rank_onnx::tokenized_doc_feature_extractor::TokenizedDocFeatureExtractor;
            use proximadb_rank_onnx::tokenized_scorer_session::TokenizedScorerSession;

            let windows: Vec<String> = sentence_bounds(doc)
                .into_iter()
                .map(|(start, end)| doc[start..end].to_string())
                .collect();
            if windows.is_empty() {
                return 0.0;
            }

            // Swap this doc's windows into the shared source; the extractor
            // itself stays stateless across requests by construction.
            if let Ok(mut texts) = self.texts.0.write() {
                texts.clear();
                for (i, window) in windows.iter().enumerate() {
                    texts.insert(i as u32, window.clone());
                }
            }
            let qctx = QueryContext {
                query_text: Some(Arc::from(query)),
                ..QueryContext::default()
            };
            let docs: Vec<DocHandle> = (0..windows.len() as u32).map(DocHandle).collect();
            // The extractor rejects the WHOLE batch if any window overflows, so
            // fall back to per-window scoring when the batch fails — batched for
            // the common case, per-window for the rare long tail.
            let scores: Vec<f32> = match self.extractor.extract_batch(&docs, &qctx) {
                Ok(batch) => self.session.score(&batch).unwrap_or_default(),
                Err(_) => docs
                    .iter()
                    .filter_map(|doc| {
                        let batch = self
                            .extractor
                            .extract_batch(std::slice::from_ref(doc), &qctx)
                            .ok()?;
                        self.session.score(&batch).ok()?.first().copied()
                    })
                    .collect(),
            };
            if scores.is_empty() {
                0.0
            } else {
                scores
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, |acc, s| acc.max(s as f64))
            }
        }
    }

    /// Rerank the head of `upstream` with the real cross-encoder. Docs outside
    /// the window are dropped — the same truncation contract as the modelled
    /// reranker, so the comparison isolates real-vs-modelled SCORING, not
    /// window policy. Deterministic tie-break on doc id.
    fn onnx_rerank(
        query: &str,
        upstream: &[SelectorResult],
        corpus: &HashMap<String, String>,
        window: usize,
        scorer: &RealScorer,
    ) -> Vec<SelectorResult> {
        let mut scored: Vec<(String, f64)> = upstream
            .iter()
            .take(window)
            .map(|result| {
                let score = corpus
                    .get(&result.doc_id)
                    .map(|text| scorer.score_pair(query, text))
                    // Unresolvable text cannot be scored; keep a finite tail slot.
                    .unwrap_or(f64::NEG_INFINITY);
                (result.doc_id.clone(), score)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored
            .into_iter()
            .enumerate()
            .map(|(rank, (doc_id, score))| SelectorResult {
                doc_id,
                score,
                rank,
                source: ResultSource::Reranked,
            })
            .collect()
    }

    #[test]
    fn real_onnx_reranker_scores_under_the_selector_protocol() {
        let Ok(model) = std::env::var("PROXIMADB_TEST_BERT_ONNX_PATH") else {
            eprintln!("skipping: PROXIMADB_TEST_BERT_ONNX_PATH not set");
            return;
        };
        let model_path = Path::new(&model);
        if !model_path.exists() {
            eprintln!("skipping: {model_path:?} does not exist");
            return;
        }
        let scorer = RealScorer::load(model_path, 3).expect("real scorer loads");

        // Real inference is paid per (pair, window); keep the bed small enough
        // for a manual run on a laptop while spanning all answer types.
        let bed: QaBed = generate_qa_bed(&QaBedConfig {
            num_questions: 20,
            ..QaBedConfig::default()
        });
        let benchmark = SelectorBenchmark::new(
            bed.ground_truth.clone(),
            bed.candidate_pools.clone(),
            bed.queries.clone(),
        );

        let noop_rankings = benchmark
            .rankings(&SelectorConfig::noop("noop"))
            .expect("noop rankings");

        // The control: noop truncated to the SAME window. Real-vs-noop_w10 then
        // isolates pure re-ORDERING inside an identical 10-document set —
        // membership, window policy, and the extractor are all held fixed.
        let mut noop_w10: SelectorRankings = HashMap::new();
        for (query_id, ranking) in &noop_rankings {
            noop_w10.insert(query_id.clone(), ranking.iter().take(10).cloned().collect());
        }

        let mut real_rankings: SelectorRankings = HashMap::new();
        for qa in &bed.qa_pairs {
            let upstream = noop_rankings
                .get(&qa.query_id)
                .expect("noop ranked every query");
            let reranked = onnx_rerank(&qa.query_text, upstream, &bed.corpus, 10, &scorer);

            // Membership preservation under REAL inference: a reranker reorders
            // its window; it must never change which documents the window holds.
            let upstream_set: Vec<&str> = upstream
                .iter()
                .take(10)
                .map(|r| r.doc_id.as_str())
                .collect();
            let reranked_set: Vec<&str> = reranked.iter().map(|r| r.doc_id.as_str()).collect();
            assert_eq!(
                upstream_set.len(),
                reranked_set.len(),
                "the real reranker must emit exactly its window"
            );
            for doc in &upstream_set {
                assert!(
                    reranked_set.contains(doc),
                    "real reranker dropped '{doc}' from its window — reorder only"
                );
            }

            real_rankings.insert(qa.query_id.clone(), reranked);
        }

        // Ordering quality, gate-2 metrics: does REAL cross-encoder scoring rank
        // the gold document higher within the window than raw score order did?
        use super::bench_selector_benchmark::reciprocal_rank;
        let mrr_of = |rankings: &SelectorRankings| -> f64 {
            bed.ground_truth
                .iter()
                .map(|gt| {
                    rankings
                        .get(&gt.query_id)
                        .map(|r| reciprocal_rank(r, gt))
                        .unwrap_or(0.0)
                })
                .sum::<f64>()
                / bed.ground_truth.len() as f64
        };
        let noop_mrr = mrr_of(&noop_w10);
        let real_mrr = mrr_of(&real_rankings);
        println!("MRR within window 10: noop {noop_mrr:.4} vs real-onnx {real_mrr:.4}");
        assert!(
            real_mrr >= noop_mrr,
            "real cross-encoder scoring must not order the window worse than raw              order: real {real_mrr:.4} vs noop {noop_mrr:.4}"
        );

        // Gate-4 view of the same two rankings (same window, so any difference
        // is ordering reaching the answer extractors).
        let results = AnswerQualityEvaluator::new(vec![
            Box::new(BestSentence::new(10)),
            Box::new(WholeTopDocument),
        ])
        .with_resamples(200)
        .evaluate(
            &bed.qa_pairs,
            &bed.corpus,
            &[
                ("noop_w10".to_string(), noop_w10),
                ("real_onnx_w10".to_string(), real_rankings),
            ],
        )
        .expect("evaluation runs");
        println!("{}", results.render());
    }
}
