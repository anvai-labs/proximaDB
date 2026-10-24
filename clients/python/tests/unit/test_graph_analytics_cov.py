"""Offline unit tests for proximadb_sdk.graph_analytics.

Fully offline: a fake client whose ``_make_request`` returns crafted dicts is
injected into GraphAnalytics. No network, no server, no model downloads.
"""

import pytest

from proximadb_sdk.graph_analytics import (
    AlgorithmConfig,
    AlgorithmResult,
    GraphAlgorithm,
    GraphAnalytics,
    GraphPattern,
    PatternElement,
    PatternMatchMode,
    PatternMatchResult,
    RelationshipPattern,
    SemanticTraversalConfig,
    SemanticTraversalResult,
    TraversalDirection,
    node,
    relationship,
)


class FakeClient:
    """Client exposing ``_make_request``; records calls and returns canned responses."""

    def __init__(self, response=None, url="http://testserver"):
        self.url = url
        self._base_url = url
        self._response = response if response is not None else {}
        self.calls = []

    def _make_request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return self._response


class ShortestPathClient:
    """Client exposing graph_shortest_path (no _make_request route taken)."""

    def __init__(self, result, url="http://testserver"):
        self._base_url = url
        self.url = url
        self._result = result
        self.calls = []

    # The REAL client signature, not **kwargs: a **kwargs mock is how the
    # start_node=/end_node= call shape (TypeError on every real client)
    # passed these tests (TD-SDKDOC-1).
    def graph_shortest_path(
        self,
        start_node_id,
        target_node_id,
        max_depth=None,
        edge_types=None,
        algorithm="DIJKSTRA",
        k=None,
        enable_prefetch=None,
        prefetch_budget=None,
        timeout=None,
        graph_id="default",
    ):
        self.calls.append(
            {
                "start_node_id": start_node_id,
                "target_node_id": target_node_id,
                "edge_types": edge_types,
                "graph_id": graph_id,
            }
        )
        return self._result


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


def test_enum_values():
    assert GraphAlgorithm.PAGERANK.value == "pagerank"
    assert GraphAlgorithm.LOUVAIN.value == "louvain"
    assert TraversalDirection.OUTGOING.value == "outgoing"
    assert TraversalDirection.INCOMING.value == "incoming"
    assert TraversalDirection.BOTH.value == "both"
    assert PatternMatchMode.EXACT.value == "exact"
    assert PatternMatchMode.FUZZY.value == "fuzzy"


# --------------------------------------------------------------------------
# Config dataclasses
# --------------------------------------------------------------------------


def test_algorithm_config_defaults_to_dict():
    cfg = AlgorithmConfig()
    d = cfg.to_dict()
    assert d["damping_factor"] == 0.85
    assert d["max_iterations"] == 100
    assert d["normalized"] is True
    assert d["random_seed"] is None
    assert d["weight_property"] is None
    assert d["max_depth"] is None


def test_algorithm_config_custom():
    cfg = AlgorithmConfig(
        damping_factor=0.5,
        max_iterations=10,
        resolution=2.0,
        random_seed=7,
        normalized=False,
        weight_property="weight",
        max_depth=4,
    )
    d = cfg.to_dict()
    assert d["damping_factor"] == 0.5
    assert d["resolution"] == 2.0
    assert d["random_seed"] == 7
    assert d["normalized"] is False
    assert d["weight_property"] == "weight"
    assert d["max_depth"] == 4


def test_semantic_traversal_config_to_dict():
    cfg = SemanticTraversalConfig(
        similarity_threshold=0.9,
        vector_field="vec",
        max_depth=2,
        direction=TraversalDirection.BOTH,
        edge_types=["A", "B"],
        limit=5,
        include_scores=False,
        include_paths=True,
        node_label_filter=["X"],
        property_filters={"k": 1},
    )
    d = cfg.to_dict()
    assert d["similarity_threshold"] == 0.9
    assert d["vector_field"] == "vec"
    assert d["direction"] == "both"
    assert d["edge_types"] == ["A", "B"]
    assert d["include_paths"] is True
    assert d["node_label_filter"] == ["X"]
    assert d["property_filters"] == {"k": 1}


# --------------------------------------------------------------------------
# Pattern elements / cypher rendering
# --------------------------------------------------------------------------


def test_pattern_element_to_cypher_minimal():
    el = PatternElement("a")
    assert el.to_cypher() == "(a)"


def test_pattern_element_to_cypher_label_and_props():
    el = PatternElement("u", "User", {"name": "alice", "age": 30})
    cy = el.to_cypher()
    assert cy.startswith("(u:User {")
    assert '"alice"' in cy
    assert cy.endswith("})")


def test_relationship_pattern_outgoing():
    rp = RelationshipPattern(
        source=PatternElement("a"),
        target=PatternElement("b"),
        relationship_type="FOLLOWS",
        relationship_var="r",
    )
    cy = rp.to_cypher()
    assert cy.startswith("(a)-[r:FOLLOWS]->")
    assert cy.endswith("(b)")


def test_relationship_pattern_incoming():
    rp = RelationshipPattern(
        source=PatternElement("a"),
        target=PatternElement("b"),
        relationship_type="KNOWS",
        direction=TraversalDirection.INCOMING,
    )
    cy = rp.to_cypher()
    assert cy.startswith("<(a)-[:KNOWS]-")


def test_relationship_pattern_both_no_arrow():
    rp = RelationshipPattern(
        source=PatternElement("a"),
        target=PatternElement("b"),
        direction=TraversalDirection.BOTH,
    )
    cy = rp.to_cypher()
    # BOTH -> arrow is empty (not OUTGOING, not INCOMING branch)
    assert "->" not in cy
    assert "<(a)" not in cy


def test_relationship_pattern_variable_hops_equal():
    rp = RelationshipPattern(
        source=PatternElement("a"),
        target=PatternElement("b"),
        min_hops=2,
        max_hops=2,
    )
    assert "*2" in rp.to_cypher()


def test_relationship_pattern_variable_hops_range():
    rp = RelationshipPattern(
        source=PatternElement("a"),
        target=PatternElement("b"),
        min_hops=1,
        max_hops=3,
    )
    assert "*1..3" in rp.to_cypher()


def test_relationship_pattern_with_properties():
    rp = RelationshipPattern(
        source=PatternElement("a"),
        target=PatternElement("b"),
        properties={"since": 2020},
    )
    cy = rp.to_cypher()
    assert "since" in cy


# --------------------------------------------------------------------------
# GraphPattern builder
# --------------------------------------------------------------------------


def test_graph_pattern_builder_chain():
    p = (
        GraphPattern()
        .match(PatternElement("a", "User"))
        .relationship(
            PatternElement("a"),
            PatternElement("b"),
            rel_type="FOLLOWS",
            direction=TraversalDirection.OUTGOING,
            min_hops=1,
            max_hops=2,
        )
        .where("a.age > 18")
        .returns("a", "b")
        .order("a.name")
        .with_limit(50)
    )
    d = p.to_dict()
    assert len(d["patterns"]) == 2
    assert d["where"] == ["a.age > 18"]
    assert d["return"] == ["a", "b"]
    assert d["order_by"] == "a.name"
    assert d["limit"] == 50


def test_graph_pattern_to_dict_str_fallback():
    p = GraphPattern()
    # Insert an object without to_cypher to exercise the str() fallback.
    p.patterns.append("RAW_PATTERN")
    d = p.to_dict()
    assert d["patterns"] == ["RAW_PATTERN"]


# --------------------------------------------------------------------------
# Result dataclasses
# --------------------------------------------------------------------------


def test_algorithm_result_from_dict_full():
    r = AlgorithmResult.from_dict(
        {
            "algorithm": "louvain",
            "node_scores": {"a": 1.0},
            "communities": {"a": 0},
            "paths": [["a", "b"]],
            "components": [["a"]],
            "statistics": {"global_coefficient": 0.3},
            "execution_time_ms": 12.5,
        }
    )
    assert r.algorithm == GraphAlgorithm.LOUVAIN
    assert r.node_scores == {"a": 1.0}
    assert r.execution_time_ms == 12.5


def test_algorithm_result_from_dict_defaults():
    r = AlgorithmResult.from_dict({})
    assert r.algorithm == GraphAlgorithm.PAGERANK
    assert r.node_scores is None
    assert r.execution_time_ms == 0


def test_semantic_traversal_result_dataclass():
    r = SemanticTraversalResult(nodes=[{"id": "x"}], total_count=1)
    assert r.nodes[0]["id"] == "x"
    assert r.edges is None


def test_pattern_match_result_dataclass():
    r = PatternMatchResult(matches=[{"m": 1}], total_count=1, execution_time_ms=2.0)
    assert r.matches == [{"m": 1}]


# --------------------------------------------------------------------------
# GraphAnalytics.__init__
# --------------------------------------------------------------------------


def test_init_with_base_url():
    c = FakeClient()
    a = GraphAnalytics(c)
    assert a._base_url == "http://testserver"


def test_init_with_url_only():
    class UrlOnly:
        url = "http://example"

    a = GraphAnalytics(UrlOnly())
    assert a._base_url == "http://example"


def test_init_default_base_url():
    class Bare:
        pass

    a = GraphAnalytics(Bare())
    assert a._base_url == "http://localhost:5678"


# --------------------------------------------------------------------------
# run_algorithm and wrappers
# --------------------------------------------------------------------------


def test_run_algorithm_make_request_path_and_payload():
    c = FakeClient({"algorithm": "pagerank", "node_scores": {"a": 0.9}})
    a = GraphAnalytics(c)
    result = a.run_algorithm(
        "g1",
        GraphAlgorithm.PAGERANK,
        AlgorithmConfig(damping_factor=0.7),
        node_subset=["a", "b"],
    )
    assert isinstance(result, AlgorithmResult)
    assert result.node_scores == {"a": 0.9}
    method, path, kwargs = c.calls[0]
    assert method == "POST"
    assert path == "/v1/graphs/g1/algorithms/pagerank"
    assert kwargs["json"]["node_subset"] == ["a", "b"]
    assert kwargs["json"]["config"]["damping_factor"] == 0.7


def test_run_algorithm_default_config():
    c = FakeClient({"algorithm": "pagerank"})
    a = GraphAnalytics(c)
    a.run_algorithm("g", GraphAlgorithm.PAGERANK)
    _, _, kwargs = c.calls[0]
    assert "node_subset" not in kwargs["json"]


def test_pagerank_returns_scores():
    c = FakeClient({"algorithm": "pagerank", "node_scores": {"a": 0.5, "b": 0.3}})
    a = GraphAnalytics(c)
    scores = a.pagerank("g", damping_factor=0.9, max_iterations=5)
    assert scores == {"a": 0.5, "b": 0.3}


def test_pagerank_empty_scores():
    c = FakeClient({"algorithm": "pagerank"})
    a = GraphAnalytics(c)
    assert a.pagerank("g") == {}


@pytest.mark.parametrize(
    "ctype,expected_algo",
    [
        ("betweenness", "betweenness_centrality"),
        ("closeness", "closeness_centrality"),
        ("degree", "degree_centrality"),
        ("eigenvector", "eigenvector_centrality"),
    ],
)
def test_centrality_types(ctype, expected_algo):
    c = FakeClient({"algorithm": expected_algo, "node_scores": {"n": 1.0}})
    a = GraphAnalytics(c)
    scores = a.centrality("g", ctype, normalized=False, weight_property="w")
    assert scores == {"n": 1.0}
    _, path, _ = c.calls[0]
    assert path.endswith(expected_algo)


def test_centrality_unknown_raises():
    a = GraphAnalytics(FakeClient())
    with pytest.raises(ValueError):
        a.centrality("g", "bogus")


def test_centrality_empty_scores():
    c = FakeClient({"algorithm": "degree_centrality"})
    a = GraphAnalytics(c)
    assert a.centrality("g", "degree") == {}


def test_community_detection_louvain():
    c = FakeClient({"algorithm": "louvain", "communities": {"a": 0, "b": 1}})
    a = GraphAnalytics(c)
    out = a.community_detection("g", "louvain", resolution=1.5, random_seed=3)
    assert out == {"a": 0, "b": 1}
    _, path, _ = c.calls[0]
    assert path.endswith("louvain")


def test_community_detection_label_propagation():
    c = FakeClient({"algorithm": "label_propagation", "communities": {}})
    a = GraphAnalytics(c)
    out = a.community_detection("g", "label_propagation")
    assert out == {}
    _, path, _ = c.calls[0]
    assert path.endswith("label_propagation")


def test_connected_components_default():
    c = FakeClient(
        {"algorithm": "connected_components", "components": [["a", "b"], ["c"]]}
    )
    a = GraphAnalytics(c)
    out = a.connected_components("g")
    assert out == [["a", "b"], ["c"]]
    _, path, _ = c.calls[0]
    assert path.endswith("connected_components")


def test_connected_components_strongly():
    c = FakeClient({"algorithm": "strongly_connected"})
    a = GraphAnalytics(c)
    out = a.connected_components("g", strongly_connected=True)
    assert out == []
    _, path, _ = c.calls[0]
    assert path.endswith("strongly_connected")


# --------------------------------------------------------------------------
# shortest_path: both routes
# --------------------------------------------------------------------------


def test_shortest_path_via_client_method():
    c = ShortestPathClient({"path": ["a", "x", "b"]})
    a = GraphAnalytics(c)
    out = a.shortest_path("g", "a", "b", edge_types=["E"])
    assert out == ["a", "x", "b"]
    assert c.calls[0] == {
        "start_node_id": "a",
        "target_node_id": "b",
        "edge_types": ["E"],
        "graph_id": "g",
    }


def test_shortest_path_via_client_method_none():
    c = ShortestPathClient(None)
    a = GraphAnalytics(c)
    assert a.shortest_path("g", "a", "b") is None


def test_shortest_path_via_run_algorithm():
    c = FakeClient({"algorithm": "shortest_path", "paths": [["a", "b", "c"]]})
    a = GraphAnalytics(c)
    out = a.shortest_path("g", "a", "c", weight_property="w")
    assert out == ["a", "b", "c"]


def test_shortest_path_via_run_algorithm_no_paths():
    c = FakeClient({"algorithm": "shortest_path"})
    a = GraphAnalytics(c)
    assert a.shortest_path("g", "a", "c") is None


# --------------------------------------------------------------------------
# semantic_traverse / semantic_neighbors
# --------------------------------------------------------------------------


def test_semantic_traverse_full():
    c = FakeClient(
        {
            "nodes": [{"id": "n1"}],
            "edges": [{"s": "n1"}],
            "paths": [["n1"]],
            "scores": {"n1": 0.9},
            "total_count": 1,
            "execution_time_ms": 4.0,
        }
    )
    a = GraphAnalytics(c)
    cfg = SemanticTraversalConfig(max_depth=2)
    res = a.semantic_traverse("g", "start", [0.1, 0.2], cfg, collection_id="vecs")
    assert isinstance(res, SemanticTraversalResult)
    assert res.nodes == [{"id": "n1"}]
    assert res.scores == {"n1": 0.9}
    _, path, kwargs = c.calls[0]
    assert path == "/v1/graphs/g/semantic-traverse"
    assert kwargs["json"]["collection_id"] == "vecs"
    assert kwargs["json"]["query_vector"] == [0.1, 0.2]


def test_semantic_traverse_default_config_no_collection():
    c = FakeClient({"nodes": []})
    a = GraphAnalytics(c)
    res = a.semantic_traverse("g", "s", [0.0])
    assert res.nodes == []
    assert res.total_count == 0
    _, _, kwargs = c.calls[0]
    assert "collection_id" not in kwargs["json"]


def test_semantic_neighbors_with_vector():
    c = FakeClient({"nodes": [{"id": "nb"}]})
    a = GraphAnalytics(c)
    out = a.semantic_neighbors(
        "g", "node1", query_vector=[0.5], similarity_threshold=0.8, edge_types=["E"]
    )
    assert out == [{"id": "nb"}]
    _, _, kwargs = c.calls[0]
    assert kwargs["json"]["query_vector"] == [0.5]
    assert kwargs["json"]["config"]["max_depth"] == 1


def test_semantic_neighbors_no_vector_uses_empty():
    c = FakeClient({"nodes": []})
    a = GraphAnalytics(c)
    out = a.semantic_neighbors("g", "node1")
    assert out == []
    _, _, kwargs = c.calls[0]
    assert kwargs["json"]["query_vector"] == []


# --------------------------------------------------------------------------
# pattern_match
# --------------------------------------------------------------------------


def test_pattern_match():
    c = FakeClient(
        {"matches": [{"a": 1, "b": 2}], "total_count": 1, "execution_time_ms": 3.0}
    )
    a = GraphAnalytics(c)
    pattern = GraphPattern().match(PatternElement("a", "User")).returns("a")
    res = a.pattern_match("g", pattern, mode=PatternMatchMode.FUZZY, limit=5)
    assert isinstance(res, PatternMatchResult)
    assert res.matches == [{"a": 1, "b": 2}]
    assert res.total_count == 1
    assert pattern.limit == 5
    _, path, kwargs = c.calls[0]
    assert path == "/v1/graphs/g/pattern-match"
    assert kwargs["json"]["mode"] == "fuzzy"


def test_pattern_match_no_limit():
    c = FakeClient({"matches": []})
    a = GraphAnalytics(c)
    pattern = GraphPattern()
    res = a.pattern_match("g", pattern)
    assert res.matches == []
    assert pattern.limit is None


# --------------------------------------------------------------------------
# find_triangles / clustering_coefficient
# --------------------------------------------------------------------------


def test_find_triangles_all():
    c = FakeClient({"algorithm": "triangle_count", "paths": [["a", "b", "c"]]})
    a = GraphAnalytics(c)
    out = a.find_triangles("g")
    assert out == [["a", "b", "c"]]
    _, _, kwargs = c.calls[0]
    assert "node_subset" not in kwargs["json"]


def test_find_triangles_for_node():
    c = FakeClient({"algorithm": "triangle_count"})
    a = GraphAnalytics(c)
    out = a.find_triangles("g", node_id="a")
    assert out == []
    _, _, kwargs = c.calls[0]
    assert kwargs["json"]["node_subset"] == ["a"]


def test_clustering_coefficient_global():
    c = FakeClient(
        {
            "algorithm": "clustering_coefficient",
            "statistics": {"global_coefficient": 0.42},
        }
    )
    a = GraphAnalytics(c)
    assert a.clustering_coefficient("g") == 0.42


def test_clustering_coefficient_global_no_stats():
    c = FakeClient({"algorithm": "clustering_coefficient"})
    a = GraphAnalytics(c)
    assert a.clustering_coefficient("g") == 0.0


def test_clustering_coefficient_node():
    c = FakeClient({"algorithm": "clustering_coefficient", "node_scores": {"a": 0.75}})
    a = GraphAnalytics(c)
    assert a.clustering_coefficient("g", node_id="a") == 0.75


def test_clustering_coefficient_node_no_scores():
    c = FakeClient({"algorithm": "clustering_coefficient"})
    a = GraphAnalytics(c)
    assert a.clustering_coefficient("g", node_id="a") == 0.0


def test_clustering_coefficient_node_missing_key():
    c = FakeClient(
        {"algorithm": "clustering_coefficient", "node_scores": {"other": 0.5}}
    )
    a = GraphAnalytics(c)
    assert a.clustering_coefficient("g", node_id="a") == 0.0


# --------------------------------------------------------------------------
# Fallback HTTP route (no _make_request) via requests monkeypatch
# --------------------------------------------------------------------------


class NoMakeRequestClient:
    def __init__(self, url="http://testserver"):
        self._base_url = url
        self.url = url


def test_run_algorithm_requests_fallback(monkeypatch):
    import requests as _requests

    captured = {}

    class FakeResp:
        def json(self_inner):
            return {"algorithm": "pagerank", "node_scores": {"z": 1.0}}

    def fake_post(url, json=None, **kw):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    monkeypatch.setattr(_requests, "post", fake_post)
    a = GraphAnalytics(NoMakeRequestClient())
    res = a.run_algorithm("g", GraphAlgorithm.PAGERANK)
    assert res.node_scores == {"z": 1.0}
    assert captured["url"].endswith("/v1/graphs/g/algorithms/pagerank")


def test_semantic_traverse_requests_fallback(monkeypatch):
    import requests as _requests

    class FakeResp:
        def json(self_inner):
            return {"nodes": [{"id": "f"}], "total_count": 1}

    monkeypatch.setattr(_requests, "post", lambda url, json=None, **kw: FakeResp())
    a = GraphAnalytics(NoMakeRequestClient())
    res = a.semantic_traverse("g", "s", [0.1])
    assert res.nodes == [{"id": "f"}]


def test_pattern_match_requests_fallback(monkeypatch):
    import requests as _requests

    class FakeResp:
        def json(self_inner):
            return {"matches": [{"m": 1}], "total_count": 1}

    monkeypatch.setattr(_requests, "post", lambda url, json=None, **kw: FakeResp())
    a = GraphAnalytics(NoMakeRequestClient())
    res = a.pattern_match("g", GraphPattern())
    assert res.matches == [{"m": 1}]


# --------------------------------------------------------------------------
# Convenience functions
# --------------------------------------------------------------------------


def test_node_helper():
    el = node("u", "User", active=True)
    assert el.variable == "u"
    assert el.label == "User"
    assert el.properties == {"active": True}


def test_node_helper_no_props():
    el = node("u")
    assert el.properties is None


def test_relationship_helper():
    rel = relationship(node("a"), node("b"), "FOLLOWS", since=2020)
    assert rel.relationship_type == "FOLLOWS"
    assert rel.properties == {"since": 2020}


def test_relationship_helper_no_props():
    rel = relationship(node("a"), node("b"))
    assert rel.properties is None
    assert rel.direction == TraversalDirection.OUTGOING
