"""Offline unit tests for proximadb_sdk.multimodal_query.

Pure builders / serialization / mocked-execution coverage. Fully offline:
the executor's client is a plain object / hand fake; no network or DB boot.
"""

import time

import pytest

from proximadb_sdk.multimodal_query import (
    CrossModalReranker,
    DocumentQueryComponent,
    FeatureExtractor,
    FeedbackSignal,
    FeedbackType,
    FusionFeatures,
    FusionModelType,
    FusionStrategy,
    GraphQueryComponent,
    JoinType,
    LearnedFusion,
    LearnedFusionConfig,
    LogQueryComponent,
    MetricQueryComponent,
    MultiModalQuery,
    MultiModalQueryBuilder,
    MultiModalQueryExecutor,
    MultiModalQueryResult,
    QueryContext,
    QueryIntent,
    QueryType,
    RerankConfig,
    RerankedResult,
    RerankExplanation,
    ScoreComponent,
    SemanticJoin,
    TemporalPreference,
    TimeDecayFunction,
    TrainingMetrics,
    TrainingSample,
    VectorQueryComponent,
    knowledge_graph_search,
    logs_with_context,
    semantic_search_with_graph,
)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------
def test_enums_values():
    assert QueryType.VECTOR.value == "vector"
    assert FusionStrategy.RRF.value == "rrf"
    assert JoinType.SEMANTIC.value == "semantic"
    assert TimeDecayFunction.GAUSSIAN.value == "gaussian"
    assert QueryIntent.ANALYTICAL.value == "analytical"
    assert TemporalPreference.RECENT.value == "recent"
    assert FusionModelType.LINEAR.value == "linear"
    assert FeedbackType.CLICK.value == "click"


# --------------------------------------------------------------------------
# Dataclass component to_dict
# --------------------------------------------------------------------------
def test_rerank_config_defaults_and_to_dict():
    cfg = RerankConfig()
    assert cfg.model_weights["vector"] == 1.0
    d = cfg.to_dict()
    assert d["rerank_top_k"] == 100
    assert d["model_weights"]["graph"] == 0.9


def test_rerank_config_explicit_weights_preserved():
    cfg = RerankConfig(model_weights={"vector": 0.5})
    assert cfg.model_weights == {"vector": 0.5}


def test_query_context_to_dict():
    ctx = QueryContext(
        query_text="hi",
        query_embedding=[1.0, 2.0],
        intent=QueryIntent.NAVIGATIONAL,
        temporal_preference=TemporalPreference.RECENT,
        required_models=["vector"],
        user_preferences={"a": 1.0},
    )
    d = ctx.to_dict()
    assert d["intent"] == "navigational"
    assert d["temporal_preference"] == "recent"
    assert d["required_models"] == ["vector"]


def test_query_context_defaults_to_dict():
    d = QueryContext().to_dict()
    assert d["intent"] is None
    assert d["required_models"] == []
    assert d["user_preferences"] == {}


def test_score_component_to_dict():
    sc = ScoreComponent(name="n", value=0.5, weight=1.0, contribution=0.5)
    assert sc.to_dict()["name"] == "n"


def test_rerank_explanation_to_dict():
    exp = RerankExplanation(
        record_id="r1",
        original_rank=2,
        new_rank=0,
        score_components=[ScoreComponent("a", 1.0, 1.0, 1.0)],
        explanation_text="x",
        confidence=0.9,
    )
    d = exp.to_dict()
    assert d["record_id"] == "r1"
    assert d["score_components"][0]["name"] == "a"


def test_reranked_result_to_dict():
    rr = RerankedResult(
        records=[{"id": "1"}], explanations=[], quality_score=0.8, diversity_score=0.4
    )
    d = rr.to_dict()
    assert d["quality_score"] == 0.8
    assert d["records"] == [{"id": "1"}]


def test_vector_query_component_to_dict():
    c = VectorQueryComponent(collection="c", query_vector=[0.1])
    d = c.to_dict()
    assert d["type"] == "vector"
    assert d["top_k"] == 10


def test_graph_query_component_to_dict():
    c = GraphQueryComponent(graph_id="g", edge_types=["E"])
    d = c.to_dict()
    assert d["type"] == "graph"
    assert d["max_depth"] == 2


def test_document_query_component_to_dict():
    c = DocumentQueryComponent(collection="docs", text_query="t")
    d = c.to_dict()
    assert d["type"] == "document"
    assert d["sort_order"] == "asc"


def test_log_query_component_to_dict():
    c = LogQueryComponent(namespace="ns", services=["s"])
    d = c.to_dict()
    assert d["type"] == "logs"
    assert d["limit"] == 1000


def test_metric_query_component_to_dict():
    c = MetricQueryComponent(namespace="ns", metric_names=["m"], time_range=(0, 1))
    d = c.to_dict()
    assert d["type"] == "metrics"
    assert d["aggregation"] == "avg"


def test_semantic_join_to_dict():
    j = SemanticJoin(left_field="a", right_field="b")
    d = j.to_dict()
    assert d["join_type"] == "semantic"
    assert d["similarity_threshold"] == 0.7


# --------------------------------------------------------------------------
# MultiModalQueryResult
# --------------------------------------------------------------------------
def test_query_result_iter_len():
    r = MultiModalQueryResult(
        records=[{"id": 1}, {"id": 2}],
        total_count=2,
        query_time_ms=1.0,
        component_times={},
        fusion_strategy="rrf",
    )
    assert len(r) == 2
    assert list(r) == [{"id": 1}, {"id": 2}]


def test_query_result_to_dataframe():
    pd = pytest.importorskip("pandas")
    r = MultiModalQueryResult(
        records=[{"id": 1}],
        total_count=1,
        query_time_ms=1.0,
        component_times={},
        fusion_strategy="rrf",
    )
    df = r.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------
def test_builder_full_chain_build():
    q = (
        MultiModalQueryBuilder()
        .vector("c", [0.1, 0.2], top_k=5, weight=2.0)
        .graph("g", start_label="L", edge_types=["E"], weight=1.5)
        .document("d", text_query="t")
        .logs("ns", time_range=(0, 1))
        .metrics("ns", ["m"], (0, 1))
        .join_semantic("a", "b", similarity_threshold=0.9)
        .join_by_id("x", "y")
        .join_graph_path("p", "q", graph_id="g", max_path_length=2)
        .fuse(FusionStrategy.WEIGHTED, weights={"extra": 1.0})
        .with_time_decay(TimeDecayFunction.EXPONENTIAL, halflife_hours=12)
        .with_custom_scorer(lambda r: 1.0)
        .limit(20)
        .offset(2)
        .timeout(5000)
        .include_scores(False)
        .include_metadata(False)
        .build()
    )
    assert isinstance(q, MultiModalQuery)
    assert len(q.components) == 5
    assert len(q.joins) == 3
    assert q.fusion_strategy == "weighted"
    assert q.limit == 20
    assert q.offset == 2
    assert q.timeout_ms == 5000
    assert q.include_scores is False
    assert q.custom_scorer is not None
    assert q.fusion_weights["extra"] == 1.0


def test_builder_graph_from_vector_results_marks_previous():
    b = MultiModalQueryBuilder().graph_from_vector_results(
        "g", id_field="key", edge_types=["E"]
    )
    comp = b._components[0]
    assert comp._from_previous is True
    assert comp._id_field == "key"


def test_builder_with_time_decay_default_reference_time():
    b = MultiModalQueryBuilder().with_time_decay()
    func, params = b._time_decay
    assert func == TimeDecayFunction.EXPONENTIAL
    assert params["reference_time"] is not None


def test_multimodal_query_to_dict_with_and_without_decay():
    q = MultiModalQueryBuilder().vector("c", [0.1]).build()
    d = q.to_dict()
    assert "time_decay" not in d
    assert d["fusion_strategy"] == "rrf"

    q2 = (
        MultiModalQueryBuilder()
        .vector("c", [0.1])
        .with_time_decay(TimeDecayFunction.GAUSSIAN, halflife_hours=2)
        .build()
    )
    d2 = q2.to_dict()
    assert d2["time_decay"]["function"] == "gaussian"
    assert d2["time_decay"]["halflife_hours"] == 2


def test_multimodal_query_to_dict_time_decay_string_func():
    q = MultiModalQuery(
        components=[],
        joins=[],
        fusion_strategy="rrf",
        fusion_weights={},
        time_decay=("exponential", {"halflife_hours": 1}),
        limit=1,
        offset=0,
        timeout_ms=1,
        include_scores=True,
        include_metadata=True,
    )
    d = q.to_dict()
    assert d["time_decay"]["function"] == "exponential"


# --------------------------------------------------------------------------
# CrossModalReranker
# --------------------------------------------------------------------------
def test_reranker_empty_records():
    rr = CrossModalReranker().rerank([])
    assert rr.records == []
    assert rr.quality_score == 1.0


def test_reranker_basic_pipeline():
    records = [
        {"id": "a", "score": 0.9, "_source_type": "vector", "embedding": [1.0, 0.0]},
        {"id": "b", "score": 0.5, "_source_type": "document", "content": "hello world"},
        {"id": "c", "score": 0.7, "_source_type": "graph", "timestamp": 1},
    ]
    ctx = QueryContext(
        query_text="hello",
        query_embedding=[1.0, 0.0],
        intent=QueryIntent.SIMILARITY_SEARCH,
        temporal_preference=TemporalPreference.RECENT,
        required_models=["vector"],
    )
    cfg = RerankConfig(generate_explanations=True)
    rr = CrossModalReranker(cfg).rerank(records, ctx)
    assert len(rr.records) == 3
    assert len(rr.explanations) == 3
    assert 0.0 <= rr.quality_score <= 1.0
    assert 0.0 <= rr.diversity_score <= 1.0
    for rec in rr.records:
        assert "_rerank_score" in rec


def test_reranker_no_semantic_no_context_no_diversity():
    cfg = RerankConfig(
        semantic_rerank=False,
        context_aware=False,
        diversity_optimization=False,
        generate_explanations=False,
    )
    records = [{"id": "a", "score": 0.5}, {"id": "b", "score": 0.9}]
    rr = CrossModalReranker(cfg).rerank(records)
    assert rr.records[0]["id"] == "b"
    assert rr.explanations == []


def test_reranker_semantic_text_fallback_and_default():
    cfg = RerankConfig(diversity_optimization=False, context_aware=False)
    ctx = QueryContext(query_embedding=[1.0], query_text="alpha beta")
    records = [
        {"id": "a", "score": 0.5, "content": "alpha gamma"},
        {"id": "b", "score": 0.5},
    ]
    rr = CrossModalReranker(cfg).rerank(records, ctx)
    assert len(rr.records) == 2


def test_reranker_context_intents_and_temporal_historical():
    cfg = RerankConfig(semantic_rerank=False, diversity_optimization=False)
    now_ns = int(time.time() * 1e9)
    records = [
        {"id": "a", "score": 0.95, "_source_type": "vector", "timestamp": now_ns},
        {
            "id": "b",
            "score": 0.5,
            "_source_type": "graph",
            "created_at": now_ns - 10**12,
        },
        {"id": "c", "score": 0.5, "_source_type": "logs"},
    ]
    for intent in (
        QueryIntent.NAVIGATIONAL,
        QueryIntent.INFORMATIONAL,
        QueryIntent.RELATIONSHIP_EXPLORATION,
        QueryIntent.ANALYTICAL,
    ):
        ctx = QueryContext(
            intent=intent, temporal_preference=TemporalPreference.HISTORICAL
        )
        rr = CrossModalReranker(cfg).rerank([dict(r) for r in records], ctx)
        assert len(rr.records) == 3


def test_reranker_mmr_single_record():
    cfg = RerankConfig(semantic_rerank=False, context_aware=False)
    rr = CrossModalReranker(cfg).rerank([{"id": "only", "score": 0.5}])
    assert len(rr.records) == 1


def test_reranker_helpers_directly():
    r = CrossModalReranker()
    assert r._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert r._cosine_similarity([], []) == 0.0
    assert r._cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert r._cosine_similarity([0.0], [0.0]) == 0.0
    assert r._text_similarity("a b", "b c") == pytest.approx(1 / 3)
    assert r._text_similarity("", "") == 0.0
    now_ns = int(time.time() * 1e9)
    assert r._compute_temporal_boost(now_ns, TemporalPreference.RECENT) > 0
    assert r._compute_temporal_boost(0, TemporalPreference.HISTORICAL) > 0
    assert r._compute_temporal_boost(0, TemporalPreference.NEUTRAL) == 0.0
    sim = r._record_similarity(
        {"_source_type": "vector", "score": 0.5, "x": 1},
        {"_source_type": "vector", "score": 0.5, "x": 2},
    )
    assert 0.0 <= sim <= 1.0
    assert r._compute_diversity_score([{"id": 1}]) == 1.0
    assert r._compute_quality_score([]) == 1.0
    assert r._compute_explanation_confidence({"score_components": []}) == 0.5


def test_reranker_explanations_text_branches():
    cfg = RerankConfig(
        generate_explanations=True,
        semantic_rerank=False,
        context_aware=False,
        diversity_optimization=False,
    )
    records = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.1}]
    rr = CrossModalReranker(cfg).rerank(records, QueryContext())
    texts = [e.explanation_text for e in rr.explanations]
    assert any(
        "unchanged" in t.lower() or "Promoted" in t or "Demoted" in t for t in texts
    )


# --------------------------------------------------------------------------
# MultiModalQueryExecutor (hand-fake client)
# --------------------------------------------------------------------------
class _Hit:
    def __init__(self, id, score, metadata=None):
        self.id = id
        self.score = score
        self.metadata = metadata or {}


class _FakeClient:
    def __init__(self):
        self.graph = self

    # ProximaDBClient.search's REAL signature (TD-SDKDOC-1): the fake used to
    # copy the executor's call shape (collection=/filter=), so the vector leg
    # raised TypeError against every real client while this fake accepted it.
    def search(
        self,
        collection_id,
        vector,
        top_k=10,
        metadata_filter=None,
        include_metadata=True,
        include_vectors=False,
        **kwargs,
    ):
        self.search_calls = getattr(self, "search_calls", [])
        self.search_calls.append(
            {"collection_id": collection_id, "metadata_filter": metadata_filter}
        )
        return [_Hit("v1", 0.9, {"k": 1}), _Hit("v2", 0.5)]

    def traverse(self, graph_id, start_nodes, edge_types, max_depth, limit):
        return [{"id": "v1", "depth": 1, "labels": ["L"], "properties": {}}]

    def query_documents(self, collection, filter, text_query, limit):
        return [{"id": "d1"}, {"id": "v1"}]

    def query_logs(self, namespace, time_range, services, text_query, limit):
        return [{"id": "l1", "timestamp": 5}]

    def aggregate_metrics(self, namespace, metric_names, time_range, aggregation):
        return [{"name": "m1", "value": 1.0, "timestamp": 5}]


def test_executor_execute_all_components_rrf():
    q = (
        MultiModalQueryBuilder()
        .vector("c", [0.1], top_k=2)
        .graph("g", edge_types=["E"])
        .document("d")
        .logs("ns")
        .metrics("ns", ["m"], (0, 1))
        .fuse(FusionStrategy.RRF)
        .build()
    )
    res = MultiModalQueryExecutor(_FakeClient()).execute(q)
    assert isinstance(res, MultiModalQueryResult)
    assert res.metadata["component_count"] == 5
    assert res.total_count >= 1
    # The other legs return rows, so total_count alone cannot see a dead
    # vector leg — assert it ran.
    assert not any("vector" in k for k in res.component_errors), res.component_errors


def test_executor_vector_leg_calls_search_with_the_real_signature():
    client = _FakeClient()
    q = (
        MultiModalQueryBuilder()
        .vector("c", [0.1], top_k=2, filter={"k": 1})
        .fuse(FusionStrategy.RRF)
        .build()
    )
    res = MultiModalQueryExecutor(client).execute(q)
    assert not res.component_errors, res.component_errors
    assert {r["id"] for r in res.records} == {"v1", "v2"}
    assert client.search_calls == [{"collection_id": "c", "metadata_filter": {"k": 1}}]


def test_executor_with_joins_and_custom_scorer_and_decay():
    q = (
        MultiModalQueryBuilder()
        .vector("c", [0.1])
        .document("d")
        .join_by_id("id", "id")
        .fuse(FusionStrategy.UNION)
        .with_time_decay(
            TimeDecayFunction.EXPONENTIAL, halflife_hours=1, reference_time=1000
        )
        .with_custom_scorer(lambda r: r.get("score", 0))
        .limit(5)
        .build()
    )
    res = MultiModalQueryExecutor(_FakeClient()).execute(q)
    assert res.metadata["join_count"] == 1


def test_executor_component_exceptions_return_empty():
    class Bad:
        def search(self, **kw):
            raise RuntimeError("boom")

    q = MultiModalQueryBuilder().vector("c", [0.1]).build()
    res = MultiModalQueryExecutor(Bad()).execute(q)
    assert res.total_count == 0


def test_executor_missing_methods_return_empty():
    class NoMethods:
        pass

    q = (
        MultiModalQueryBuilder()
        .graph("g")
        .document("d")
        .logs("ns")
        .metrics("ns", ["m"], (0, 1))
        .build()
    )
    res = MultiModalQueryExecutor(NoMethods()).execute(q)
    assert res.total_count == 0


def test_executor_graph_from_previous_fills_start_nodes():
    client = _FakeClient()
    q = (
        MultiModalQueryBuilder()
        .vector("c", [0.1])
        .graph_from_vector_results("g", id_field="id")
        .build()
    )
    q.components[1]["_from_previous"] = True
    q.components[1]["_id_field"] = "id"
    res = MultiModalQueryExecutor(client).execute(q)
    assert isinstance(res, MultiModalQueryResult)


def test_executor_fuse_strategies_direct():
    ex = MultiModalQueryExecutor(_FakeClient())
    a = [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.4}]
    b = [{"id": "2", "score": 0.8}, {"id": "3", "score": 0.1}]
    assert ex._fuse_results([], "rrf", {}) == []
    assert ex._fuse_results([a], "rrf", {}) == a
    inter = ex._fuse_results([a, b], "intersection", {})
    assert [r["id"] for r in inter] == ["2"]
    union = ex._fuse_results([a, b], "union", {})
    assert {r["id"] for r in union} == {"1", "2", "3"}
    rrf = ex._fuse_results([a, b], "rrf", {"component_0": 1.0})
    assert all("_rrf_score" in r for r in rrf)
    weighted = ex._fuse_results([a, b], "weighted", {})
    assert all("_weighted_score" in r for r in weighted)
    default = ex._fuse_results([a, b], "unknown_strategy", {})
    assert all("_rrf_score" in r for r in default)


def test_executor_fuse_union_anonymous_records():
    ex = MultiModalQueryExecutor(_FakeClient())
    a = [{"score": 0.5}]
    union = ex._fuse_results([a, a], "union", {})
    assert len(union) == 2


def test_executor_fuse_intersection_empty():
    ex = MultiModalQueryExecutor(_FakeClient())
    assert ex._fuse_intersection([]) == []


def test_executor_extract_field():
    ex = MultiModalQueryExecutor(_FakeClient())
    assert ex._extract_field({"a": {"b": 5}}, "a.b") == "5"
    assert ex._extract_field({"a": 1}, "a.b") is None
    assert ex._extract_field({"a": None}, "a") is None


def test_executor_apply_joins_single_component_passthrough():
    ex = MultiModalQueryExecutor(_FakeClient())
    comp = [[{"id": "1"}]]
    assert ex._apply_joins(comp, [{"join_type": "inner"}]) == comp


def test_executor_apply_joins_inner_merges():
    ex = MultiModalQueryExecutor(_FakeClient())
    left = [{"id": "1", "a": 1}, {"id": "2"}]
    right = [{"id": "1", "b": 2}]
    out = ex._apply_joins(
        [left, right], [{"join_type": "inner", "left_field": "id", "right_field": "id"}]
    )
    assert out[0][0]["a"] == 1 and out[0][0]["b"] == 2


def test_executor_apply_time_decay_functions():
    ex = MultiModalQueryExecutor(_FakeClient())
    recs = [{"id": "1", "score": 1.0, "timestamp": 0}, {"id": "2"}]
    for func in (
        TimeDecayFunction.LINEAR,
        TimeDecayFunction.EXPONENTIAL,
        TimeDecayFunction.GAUSSIAN,
        TimeDecayFunction.NONE,
    ):
        out = ex._apply_time_decay(
            [dict(r) for r in recs],
            (
                func,
                {
                    "reference_time": 10**18,
                    "halflife_hours": 24,
                    "time_field": "timestamp",
                },
            ),
        )
        assert any("_decayed_score" in r for r in out)


def test_executor_apply_time_decay_negative_age_clamped():
    ex = MultiModalQueryExecutor(_FakeClient())
    out = ex._apply_time_decay(
        [{"id": "1", "score": 1.0, "timestamp": 10**18}],
        (TimeDecayFunction.LINEAR, {"reference_time": 0, "halflife_hours": 1}),
    )
    assert out[0]["_time_decay"] == 1.0


# --------------------------------------------------------------------------
# Convenience functions
# --------------------------------------------------------------------------
def test_convenience_functions():
    client = _FakeClient()
    r1 = semantic_search_with_graph(client, "c", [0.1], "g", edge_types=["E"], top_k=3)
    r2 = knowledge_graph_search(client, "g", "L", [0.1], "c", max_depth=1, top_k=3)
    r3 = logs_with_context(client, "ns", "error", "g", (0, 1), top_k=5)
    for r in (r1, r2, r3):
        assert isinstance(r, MultiModalQueryResult)


# --------------------------------------------------------------------------
# FusionFeatures / FeatureExtractor
# --------------------------------------------------------------------------
def test_fusion_features_to_flat_vector():
    f = FusionFeatures(
        query_features=[1.0, 2.0],
        model_features={"vector": [0.5, 0.5], "document": [0.1, 0.1]},
        interaction_features=[9.0],
    )
    flat = f.to_flat_vector()
    assert len(flat) == 2 + 8 + 1
    assert flat[0] == 1.0


def test_feature_extractor_extract():
    fe = FeatureExtractor(num_features=8)
    results = [
        {
            "source": "vector",
            "records": [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.5}],
        },
        {"source": "document", "records": [{"id": "2", "score": 0.8}]},
        {"source": "graph", "records": []},
        {"source": "other", "records": [{"id": "3"}]},
    ]
    feats = fe.extract(results)
    assert isinstance(feats, FusionFeatures)
    assert "vector" in feats.model_features
    assert feats.query_features[0] > 0
    assert len(feats.to_flat_vector()) > 0


def test_feature_extractor_empty():
    fe = FeatureExtractor()
    feats = fe.extract([])
    assert feats.query_features[0] == 0.0


# --------------------------------------------------------------------------
# LearnedFusion
# --------------------------------------------------------------------------
def test_learned_fusion_config_defaults():
    cfg = LearnedFusionConfig()
    assert cfg.model_type == FusionModelType.GRADIENT_BOOSTING
    assert cfg.num_features == 32


def test_learned_fusion_fuse_empty_and_single():
    lf = LearnedFusion()
    assert lf.fuse([]) == []
    single = [{"source": "vector", "records": [{"id": "1"}]}]
    assert lf.fuse(single) == [{"id": "1"}]


def test_learned_fusion_fuse_untrained_rrf_fallback():
    lf = LearnedFusion()
    results = [
        {
            "source": "vector",
            "records": [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.4}],
        },
        {
            "source": "document",
            "records": [{"id": "2", "score": 0.8}, {"id": "3", "score": 0.1}],
        },
    ]
    fused = lf.fuse(results)
    assert all("_fusion_score" in r for r in fused)
    assert {r["id"] for r in fused} == {"1", "2", "3"}


def test_learned_fusion_record_feedback_and_train_linear():
    cfg = LearnedFusionConfig(
        model_type=FusionModelType.LINEAR,
        min_samples_for_training=3,
        num_features=2,
    )
    lf = LearnedFusion(cfg)
    feats = FusionFeatures(query_features=[0.1, 0.2], interaction_features=[0.0])
    lf.record_feedback(
        feats, FeedbackSignal(FeedbackType.CLICK, record_id="r", position=0)
    )
    lf.record_feedback(
        feats,
        FeedbackSignal(FeedbackType.RELEVANCE_JUDGMENT, record_id="r", relevant=True),
    )
    lf.record_feedback(
        feats,
        FeedbackSignal(FeedbackType.RELEVANCE_JUDGMENT, record_id="r", relevant=False),
    )
    assert lf.training_buffer_size == 3
    metrics = lf.train()
    assert isinstance(metrics, TrainingMetrics)
    assert lf.is_trained is True
    assert lf.get_feature_importance() is not None


def test_learned_fusion_train_not_enough_samples():
    lf = LearnedFusion(LearnedFusionConfig(min_samples_for_training=5))
    with pytest.raises(ValueError):
        lf.train()


def test_learned_fusion_gradient_boosting_train_and_predict():
    cfg = LearnedFusionConfig(
        model_type=FusionModelType.GRADIENT_BOOSTING,
        min_samples_for_training=3,
        num_features=8,
    )
    lf = LearnedFusion(cfg)
    for i in range(4):
        feats = FusionFeatures(
            query_features=[float(i), 0.5], interaction_features=[0.1]
        )
        lf.add_training_sample(
            TrainingSample(features=feats, target_scores={"r": float(i % 2)})
        )
    metrics = lf.train()
    assert metrics.num_samples == 4
    assert lf.is_trained
    results = [
        {"source": "vector", "records": [{"id": "1", "score": 0.9}]},
        {"source": "document", "records": [{"id": "2", "score": 0.8}]},
    ]
    fused = lf.fuse(results)
    assert all("_fusion_score" in r for r in fused)
    assert lf.get_feature_importance() is not None


def test_learned_fusion_feature_importance_untrained():
    lf = LearnedFusion()
    assert lf.get_feature_importance() is None


def test_learned_fusion_add_sample_buffer_cap():
    cfg = LearnedFusionConfig(max_training_samples=2)
    lf = LearnedFusion(cfg)
    feats = FusionFeatures(query_features=[0.1])
    for _ in range(5):
        lf.add_training_sample(TrainingSample(features=feats, target_scores={}))
    assert lf.training_buffer_size == 2


def test_learned_fusion_no_collect_training_data():
    cfg = LearnedFusionConfig(collect_training_data=False)
    lf = LearnedFusion(cfg)
    lf.record_feedback(
        FusionFeatures(query_features=[0.1]),
        FeedbackSignal(FeedbackType.CLICK, record_id="r", position=0),
    )
    lf.add_training_sample(TrainingSample(features=FusionFeatures(), target_scores={}))
    assert lf.training_buffer_size == 0


def test_learned_fusion_online_learning_triggers_train():
    cfg = LearnedFusionConfig(
        model_type=FusionModelType.LINEAR,
        enable_online_learning=True,
        online_update_frequency=1,
        min_samples_for_training=1,
        num_features=8,
    )
    lf = LearnedFusion(cfg)
    lf.add_training_sample(
        TrainingSample(
            features=FusionFeatures(query_features=[0.1, 0.2]), target_scores={"r": 1.0}
        )
    )
    results = [
        {"source": "vector", "records": [{"id": "1", "score": 0.9}]},
        {"source": "document", "records": [{"id": "2", "score": 0.8}]},
    ]
    lf.fuse(results)
    assert lf.is_trained


def test_learned_fusion_stump_predict_branches():
    lf = LearnedFusion()
    stump = {
        "feature_index": 0,
        "threshold": 1.0,
        "left_value": -1.0,
        "right_value": 2.0,
    }
    assert lf._stump_predict(stump, [0.5]) == -1.0
    assert lf._stump_predict(stump, [5.0]) == 2.0
    assert lf._stump_predict(stump, []) == -1.0


def test_learned_fusion_fit_stump_empty():
    lf = LearnedFusion()
    assert lf._fit_stump([], []) is None
    assert lf._fit_stump([[]], [0.0]) is None
