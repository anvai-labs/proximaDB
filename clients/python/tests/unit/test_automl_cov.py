"""Offline unit tests for proximadb_sdk.automl.

Pure module — no network, no server. All logic is heuristic/in-memory.
The optimizer's only client touchpoint (test_queries search) is exercised
via a hand fake client, never a live connection.
"""

import random

from proximadb_sdk.automl import (
    AutoML,
    EngineRecommendation,
    EngineSelector,
    HyperparameterConfig,
    HyperparameterOptimizer,
    OptimizationGoal,
    OptimizationResult,
    WorkloadCharacteristics,
    WorkloadPredictor,
    WorkloadType,
)

# --------------------------------------------------------------------------
# Dataclasses / enums
# --------------------------------------------------------------------------


def test_workload_characteristics_to_dict():
    wc = WorkloadCharacteristics(
        read_ratio=0.8,
        write_ratio=0.2,
        vector_count=1000,
        vector_dimension=768,
        target_latency_ms=10,
        target_throughput=5000,
        memory_budget_mb=512,
    )
    d = wc.to_dict()
    assert d["read_ratio"] == 0.8
    assert d["vector_dimension"] == 768
    assert d["target_latency_ms"] == 10
    assert set(d.keys()) >= {
        "read_ratio",
        "write_ratio",
        "query_complexity",
        "vector_count",
        "vector_dimension",
        "metadata_cardinality",
        "temporal_locality",
        "spatial_locality",
        "hot_data_ratio",
        "target_latency_ms",
        "target_throughput",
        "memory_budget_mb",
    }


def test_workload_characteristics_defaults():
    wc = WorkloadCharacteristics()
    assert wc.read_ratio == 0.5
    assert wc.target_latency_ms is None


def test_engine_recommendation_to_dict():
    rec = EngineRecommendation(
        engine="sst",
        confidence=0.9,
        reasoning="because",
        estimated_latency_ms=5.0,
        estimated_throughput=100000,
        estimated_memory_mb=45,
        config_overrides={"compression": "lz4"},
    )
    d = rec.to_dict()
    assert d["engine"] == "sst"
    assert d["config_overrides"] == {"compression": "lz4"}
    assert d["estimated_throughput"] == 100000


def test_hyperparameter_config_defaults():
    hp = HyperparameterConfig(name="m", current_value=16)
    assert hp.min_value is None
    assert hp.allowed_values is None


def test_optimization_result_defaults():
    r = OptimizationResult(best_config={"m": 16}, best_score=0.9, iterations=5)
    assert r.search_history == []
    assert r.improvement_ratio == 0


def test_enums_are_str():
    assert WorkloadType.READ_HEAVY.value == "read_heavy"
    assert OptimizationGoal.LATENCY.value == "latency"
    assert OptimizationGoal("balanced") == OptimizationGoal.BALANCED


# --------------------------------------------------------------------------
# WorkloadPredictor
# --------------------------------------------------------------------------


def test_predictor_empty_predict():
    p = WorkloadPredictor()
    wc = p.predict()
    assert isinstance(wc, WorkloadCharacteristics)
    assert wc.read_ratio == 0.5


def test_predictor_observe_and_predict():
    p = WorkloadPredictor()
    p.observe_operation("search", latency_ms=5.0, vector_count=100)
    p.observe_operation("get", latency_ms=3.0, vector_count=50)
    p.observe_operation("insert", latency_ms=2.0, vector_count=10, metadata={"x": 1})
    wc = p.predict()
    assert abs(wc.read_ratio - (2 / 3)) < 1e-6
    assert abs(wc.write_ratio - (1 / 3)) < 1e-6
    assert wc.vector_count == 160
    assert 0 <= wc.query_complexity <= 1


def test_predictor_single_op_temporal_default():
    p = WorkloadPredictor()
    p.observe_operation("search", latency_ms=50.0)
    wc = p.predict()
    assert wc.temporal_locality == 0.5


def test_predictor_window_eviction():
    p = WorkloadPredictor(window_size=3)
    for _ in range(10):
        p.observe_operation("search", latency_ms=1.0)
    assert len(p._operations) == 3
    assert p._operation_counts["search"] == 10


def test_predictor_high_latency_complexity_capped():
    p = WorkloadPredictor()
    p.observe_operation("search", latency_ms=10000.0)
    wc = p.predict()
    assert wc.query_complexity == 1.0


def test_predictor_get_workload_type_write_heavy():
    p = WorkloadPredictor()
    for _ in range(8):
        p.observe_operation("insert", latency_ms=1.0)
    for _ in range(2):
        p.observe_operation("search", latency_ms=1.0)
    assert p.get_workload_type() == WorkloadType.WRITE_HEAVY


def test_predictor_get_workload_type_read_heavy():
    p = WorkloadPredictor()
    for _ in range(8):
        p.observe_operation("search", latency_ms=1.0)
    for _ in range(2):
        p.observe_operation("insert", latency_ms=1.0)
    assert p.get_workload_type() == WorkloadType.READ_HEAVY


def test_predictor_get_workload_type_mixed():
    p = WorkloadPredictor()
    for _ in range(5):
        p.observe_operation("search", latency_ms=1.0)
    for _ in range(5):
        p.observe_operation("insert", latency_ms=1.0)
    assert p.get_workload_type() == WorkloadType.MIXED


# --------------------------------------------------------------------------
# EngineSelector
# --------------------------------------------------------------------------


def test_selector_recommend_basic():
    s = EngineSelector()
    wc = WorkloadCharacteristics(read_ratio=0.8, write_ratio=0.2, vector_count=100000)
    rec = s.recommend(wc, OptimizationGoal.LATENCY)
    assert isinstance(rec, EngineRecommendation)
    assert rec.engine in EngineSelector.ENGINE_PROFILES
    assert 0 <= rec.confidence <= 1
    assert rec.estimated_latency_ms > 0
    assert rec.estimated_throughput >= 1000
    assert rec.estimated_memory_mb >= 10
    assert rec.reasoning


def test_selector_recommend_all_goals():
    s = EngineSelector()
    wc = WorkloadCharacteristics(read_ratio=0.6, write_ratio=0.4, vector_count=5000)
    for goal in OptimizationGoal:
        rec = s.recommend(wc, goal)
        assert rec.engine in EngineSelector.ENGINE_PROFILES


def test_selector_swift_skipped_for_large_datasets():
    s = EngineSelector()
    wc = WorkloadCharacteristics(vector_count=1_000_000)
    rec = s.recommend(wc, OptimizationGoal.BALANCED)
    assert rec.engine != "swift"


def test_selector_never_recommends_deprecated_or_experimental():
    # TD-SDKENG-1: compare_engines must not hand back VIPER (deprecated,
    # ADR-093) or SWIFT/RAPTOR (experimental) — following the documented
    # quickstart with such a rec would ask the server for an engine it
    # rejects.
    s = EngineSelector()
    wc = WorkloadCharacteristics(
        read_ratio=0.95,
        write_ratio=0.05,
        vector_count=500,
    )
    recs = s.compare_engines(wc)
    engines = {r.engine for r in recs}
    assert engines, "some selectable engine must be recommended"
    assert engines <= {"sst", "nova", "helix"}, engines


def test_selector_target_latency_penalty():
    s = EngineSelector()
    wc = WorkloadCharacteristics(vector_count=1000, target_latency_ms=1)
    rec = s.recommend(wc, OptimizationGoal.LATENCY)
    assert rec.engine != "viper"


def test_selector_target_throughput_penalty():
    s = EngineSelector()
    wc = WorkloadCharacteristics(vector_count=1000, target_throughput=200000)
    rec = s.recommend(wc, OptimizationGoal.THROUGHPUT)
    assert rec.engine in EngineSelector.ENGINE_PROFILES


def test_selector_compare_engines_sorted():
    s = EngineSelector()
    wc = WorkloadCharacteristics(read_ratio=0.5, write_ratio=0.5, vector_count=20000)
    recs = s.compare_engines(wc)
    assert all(r.engine != "swift" for r in recs)
    confidences = [r.confidence for r in recs]
    assert confidences == sorted(confidences, reverse=True)


def test_selector_get_engine_recommendation_unknown():
    s = EngineSelector()
    wc = WorkloadCharacteristics()
    assert s._get_engine_recommendation("does_not_exist", wc) is None


def test_selector_get_engine_recommendation_swift_constraint():
    s = EngineSelector()
    wc = WorkloadCharacteristics(vector_count=999999)
    assert s._get_engine_recommendation("swift", wc) is None


def test_infer_workload_type_branches():
    s = EngineSelector()
    assert (
        s._infer_workload_type(WorkloadCharacteristics(write_ratio=0.9))
        == "write_heavy"
    )
    assert (
        s._infer_workload_type(WorkloadCharacteristics(read_ratio=0.9, write_ratio=0.1))
        == "read_heavy"
    )
    assert (
        s._infer_workload_type(
            WorkloadCharacteristics(
                read_ratio=0.5, write_ratio=0.5, query_complexity=0.9
            )
        )
        == "analytics"
    )
    assert (
        s._infer_workload_type(
            WorkloadCharacteristics(
                read_ratio=0.5,
                write_ratio=0.5,
                query_complexity=0.1,
                temporal_locality=0.9,
            )
        )
        == "streaming"
    )
    assert (
        s._infer_workload_type(
            WorkloadCharacteristics(
                read_ratio=0.5,
                write_ratio=0.5,
                query_complexity=0.1,
                temporal_locality=0.1,
            )
        )
        == "mixed"
    )


def test_generate_config_nova_empty():
    s = EngineSelector()
    cfg = s._generate_config(
        "nova", s.ENGINE_PROFILES["nova"], WorkloadCharacteristics()
    )
    assert cfg == {}


def test_generate_reasoning_all_engines():
    s = EngineSelector()
    for engine in EngineSelector.ENGINE_PROFILES:
        wc = WorkloadCharacteristics(
            write_ratio=0.8,
            read_ratio=0.2,
            spatial_locality=0.9,
            query_complexity=0.9,
            hot_data_ratio=0.5,
        )
        r = s._generate_reasoning(engine, wc, OptimizationGoal.LATENCY)
        assert isinstance(r, str) and r


def test_generate_reasoning_balanced_no_goal_suffix():
    s = EngineSelector()
    r = s._generate_reasoning(
        "nova", WorkloadCharacteristics(), OptimizationGoal.BALANCED
    )
    assert "Optimized for" not in r


# --------------------------------------------------------------------------
# HyperparameterOptimizer
# --------------------------------------------------------------------------


def test_optimizer_searchable_params():
    opt = HyperparameterOptimizer()
    params = opt._get_searchable_params()
    names = {p.name for p in params}
    assert names == {
        "ef_search",
        "ef_construction",
        "m",
        "bloom_filter_fpp",
        "block_size",
    }


def test_optimizer_optimize_heuristic_no_client():
    random.seed(42)
    opt = HyperparameterOptimizer()
    result = opt.optimize("col", goal=OptimizationGoal.RECALL, max_iterations=10)
    assert isinstance(result, OptimizationResult)
    assert result.iterations == 10
    assert len(result.search_history) == 10
    assert result.best_config
    assert result.best_score > float("-inf")


def test_optimizer_optimize_latency_goal():
    random.seed(1)
    opt = HyperparameterOptimizer()
    result = opt.optimize("col", goal=OptimizationGoal.LATENCY, max_iterations=5)
    assert result.iterations == 5
    assert isinstance(result.improvement_ratio, float)


def test_optimizer_generate_candidate_explore(monkeypatch):
    opt = HyperparameterOptimizer()
    params = opt._get_searchable_params()
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "randint", lambda a, b: 0)
    monkeypatch.setattr(random, "uniform", lambda a, b: a)
    cfg = opt._generate_candidate(params, iteration=0, max_iterations=10)
    assert cfg["block_size"] == 16384
    assert cfg["ef_search"] == 16
    assert cfg["bloom_filter_fpp"] == 0.001


def test_optimizer_generate_candidate_exploit(monkeypatch):
    opt = HyperparameterOptimizer()
    params = opt._get_searchable_params()
    monkeypatch.setattr(random, "random", lambda: 0.99)
    cfg = opt._generate_candidate(params, iteration=10, max_iterations=10)
    assert cfg["ef_search"] == 64
    assert cfg["m"] == 16


def test_optimizer_evaluate_heuristic_recall():
    opt = HyperparameterOptimizer()
    score = opt._evaluate_config(
        "col", {"ef_search": 512, "m": 64}, OptimizationGoal.RECALL, None
    )
    assert score > 0.5


def test_optimizer_evaluate_heuristic_latency_and_memory():
    opt = HyperparameterOptimizer()
    lat = opt._heuristic_score({"ef_search": 16}, OptimizationGoal.LATENCY)
    mem = opt._heuristic_score({"m": 4}, OptimizationGoal.MEMORY)
    assert lat > 0.5
    assert mem > 0.5


def test_optimizer_evaluate_heuristic_default_goal():
    opt = HyperparameterOptimizer()
    score = opt._heuristic_score({}, OptimizationGoal.BALANCED)
    assert score == 0.5


def test_optimizer_evaluate_with_client_and_queries():
    calls = {"n": 0}

    class FakeClient:
        def search(self, collection, query, top_k=10):
            calls["n"] += 1
            return {"results": []}

    opt = HyperparameterOptimizer(client=FakeClient())
    queries = [[0.1, 0.2], [0.3, 0.4]]
    score = opt._evaluate_config(
        "col", {"ef_search": 64}, OptimizationGoal.LATENCY, queries
    )
    assert calls["n"] == 2
    assert score > 0


def test_optimizer_evaluate_with_client_throughput_and_error():
    class BoomClient:
        def search(self, collection, query, top_k=10):
            raise RuntimeError("boom")

    opt = HyperparameterOptimizer(client=BoomClient())
    queries = [[0.1], [0.2]]
    score = opt._evaluate_config("col", {}, OptimizationGoal.THROUGHPUT, queries)
    assert score > 0


def test_optimizer_evaluate_with_client_default_goal():
    class FakeClient:
        def search(self, collection, query, top_k=10):
            return {}

    opt = HyperparameterOptimizer(client=FakeClient())
    score = opt._evaluate_config("col", {}, OptimizationGoal.MEMORY, [[0.1]])
    assert score > 0


def test_optimizer_optimize_with_client_and_test_queries():
    random.seed(7)

    class FakeClient:
        def search(self, collection, query, top_k=10):
            return {}

    opt = HyperparameterOptimizer(client=FakeClient())
    result = opt.optimize(
        "col",
        goal=OptimizationGoal.LATENCY,
        max_iterations=3,
        test_queries=[[0.1, 0.2], [0.3, 0.4]],
    )
    assert result.iterations == 3
    assert len(result.search_history) == 3


# --------------------------------------------------------------------------
# AutoML facade
# --------------------------------------------------------------------------


def test_automl_init_components():
    am = AutoML()
    assert isinstance(am.predictor, WorkloadPredictor)
    assert isinstance(am.selector, EngineSelector)
    assert isinstance(am.optimizer, HyperparameterOptimizer)


def test_automl_recommend_engine_default_write_ratio():
    am = AutoML()
    rec = am.recommend_engine(
        vector_count=50000,
        vector_dimension=768,
        read_ratio=0.7,
        goal=OptimizationGoal.LATENCY,
    )
    assert isinstance(rec, EngineRecommendation)
    assert rec.engine in EngineSelector.ENGINE_PROFILES


def test_automl_recommend_engine_explicit_write_ratio():
    am = AutoML()
    rec = am.recommend_engine(
        vector_count=100,
        read_ratio=0.3,
        write_ratio=0.7,
        target_latency_ms=20,
        target_throughput=10000,
        memory_budget_mb=256,
        goal=OptimizationGoal.THROUGHPUT,
    )
    assert rec.engine in EngineSelector.ENGINE_PROFILES


def test_automl_auto_configure():
    random.seed(3)
    am = AutoML()
    cfg = am.auto_configure("col", goal=OptimizationGoal.RECALL, max_iterations=4)
    assert isinstance(cfg, dict)
    assert cfg


def test_automl_observe_and_analyze():
    am = AutoML()
    am.observe("search", latency_ms=4.0, vector_count=10)
    am.observe("insert", latency_ms=2.0, vector_count=5)
    wc = am.analyze_workload()
    assert isinstance(wc, WorkloadCharacteristics)
    assert wc.vector_count == 15


def test_generate_config_emits_only_config_fields():
    """TD-SDKENG-1: overrides feed create_collection(**rec.config_overrides),
    which now REJECTS unknown keywords — so only CollectionConfig fields
    (or known aliases) may be emitted. The old tuning knobs
    (flush_threshold_mb, pca_dimensions, ...) were silently dropped."""
    from proximadb_sdk.automl import WorkloadCharacteristics
    from proximadb_sdk.models import (
        CollectionConfig,
        normalize_collection_config_kwargs,
    )

    recommender = EngineSelector()
    chars = WorkloadCharacteristics(
        vector_count=100_000, vector_dimension=768, write_ratio=0.8
    )
    for engine in ("sst", "helix", "nova"):
        overrides = recommender._generate_config(
            engine, recommender.ENGINE_PROFILES[engine], chars
        )
        config_kwargs, passthrough = normalize_collection_config_kwargs(overrides)
        CollectionConfig(name="t", dimension=3, **config_kwargs)
        assert not passthrough
