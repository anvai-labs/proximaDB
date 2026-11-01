"""TD-SDK-3 batch 1: the graph-CRUD family ported onto the v2 unified client.

Covers four layers, mirroring the batch contract in the TD:

1. Facade delegation — every ported method reaches the adapter with the
   canonical kwargs and passes the result through untouched.
2. Facade validation — the protocol-independent TypeError/ValueError checks
   carried over from v1.
3. Error translation — adapter ``NotImplementedError`` surfaces as
   ``ProximaDBError`` (one error vocabulary for the namespace).
4. Wire equivalence — the REST adapter forwards to the transport with
   exactly the kwargs v1's ``_invoke_graph_method`` dispatch produces
   (``graph_id=None`` omitted → the transport's "default"), plus the two
   deliberate fixes: ``graph_id`` is honored by ``graph_traverse``, and
   ``delete_node`` is bool-coerced.
5. Parity census ratchet — the ported set is pinned; it may only grow.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import proximadb_sdk.unified_client_v2 as uc
from proximadb_sdk.adapters.base import BaseProtocolAdapter
from proximadb_sdk.config import Protocol
from proximadb_sdk.exceptions import ProximaDBError
from proximadb_sdk.unified_client_v2 import ProximaDBClient

# ---------------------------------------------------------------------------
# Parity census (TD-SDK-3 program step 2 — the ratchet)
# ---------------------------------------------------------------------------

#: The graph-CRUD methods ported in TD-SDK-3 batch 1. Batches append here;
#: the count is pinned so a removal is a deliberate, visible edit.
PORTED_GRAPH_METHODS = frozenset(
    {
        "create_node",
        "create_edge",
        "batch_create_nodes",
        "batch_create_edges",
        "get_node",
        "get_outgoing_edges",
        "get_incoming_edges",
        "delete_node",
        "query_nodes",
        "traverse_graph",
        "create_graph",
        "delete_graph",
        "get_graph",
        "list_graphs",
        "get_graph_stats",
        "graph_shortest_path",
        "graph_traverse",
    }
)
assert len(PORTED_GRAPH_METHODS) == 17, "TD-SDK-3 batch 1 ported exactly 17"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.protocol_name = "rest"
    return adapter


@pytest.fixture
def client(mock_adapter):
    with patch.object(uc, "create_adapter", return_value=mock_adapter):
        c = ProximaDBClient(url="http://testserver:5678", protocol=Protocol.REST)
    return c


# ---------------------------------------------------------------------------
# 1. Facade delegation
# ---------------------------------------------------------------------------


class TestGraphDelegation:
    def test_create_node(self, client, mock_adapter):
        mock_adapter.create_node.return_value = {"id": "n1"}
        result = client.create_node(
            "n1", ["Person"], properties={"name": "A"}, embedding=[0.1]
        )
        mock_adapter.create_node.assert_called_once_with(
            node_id="n1",
            labels=["Person"],
            properties={"name": "A"},
            embedding=[0.1],
            graph_id=None,
        )
        assert result == {"id": "n1"}

    def test_create_edge(self, client, mock_adapter):
        mock_adapter.create_edge.return_value = {"id": "e1"}
        result = client.create_edge(
            "e1", "a", "b", "KNOWS", properties={"since": 2020}, weight=1.0
        )
        mock_adapter.create_edge.assert_called_once_with(
            edge_id="e1",
            from_node_id="a",
            to_node_id="b",
            edge_type="KNOWS",
            properties={"since": 2020},
            weight=1.0,
            graph_id=None,
        )
        assert result == {"id": "e1"}

    def test_batch_create_nodes(self, client, mock_adapter):
        mock_adapter.batch_create_nodes.return_value = {"created_count": 2}
        result = client.batch_create_nodes([{"id": "n1"}], graph_id="g")
        mock_adapter.batch_create_nodes.assert_called_once_with(
            nodes=[{"id": "n1"}], graph_id="g"
        )
        assert result == {"created_count": 2}

    def test_batch_create_edges(self, client, mock_adapter):
        mock_adapter.batch_create_edges.return_value = {"created_count": 1}
        result = client.batch_create_edges([{"id": "e1"}])
        mock_adapter.batch_create_edges.assert_called_once_with(
            edges=[{"id": "e1"}], graph_id=None
        )
        assert result == {"created_count": 1}

    def test_get_node_none_passthrough(self, client, mock_adapter):
        mock_adapter.get_node.return_value = None
        assert client.get_node("n1", graph_id="g") is None
        mock_adapter.get_node.assert_called_once_with(node_id="n1", graph_id="g")

    def test_edge_readers(self, client, mock_adapter):
        mock_adapter.get_outgoing_edges.return_value = [{"id": "e1"}]
        mock_adapter.get_incoming_edges.return_value = []
        assert client.get_outgoing_edges("n1", edge_types=["KNOWS"]) == [{"id": "e1"}]
        assert client.get_incoming_edges("n1") == []
        mock_adapter.get_outgoing_edges.assert_called_once_with(
            node_id="n1", edge_types=["KNOWS"], graph_id=None
        )
        mock_adapter.get_incoming_edges.assert_called_once_with(
            node_id="n1", edge_types=None, graph_id=None
        )

    def test_delete_node_bool_coerced(self, client, mock_adapter):
        mock_adapter.delete_node.return_value = {"success": True}
        assert client.delete_node("n1") is True
        mock_adapter.delete_node.assert_called_once_with(node_id="n1", graph_id=None)

    def test_query_nodes(self, client, mock_adapter):
        mock_adapter.query_nodes.return_value = {"nodes": []}
        client.query_nodes(labels=["P"], limit=5, offset=2, graph_id="g")
        mock_adapter.query_nodes.assert_called_once_with(
            labels=["P"], properties=None, limit=5, offset=2, graph_id="g"
        )

    def test_traverse_graph(self, client, mock_adapter):
        mock_adapter.traverse_graph.return_value = {"nodes": [], "edges": []}
        client.traverse_graph(
            "n1", max_depth=2, edge_types=["KNOWS"], algorithm="DFS", limit=10
        )
        mock_adapter.traverse_graph.assert_called_once_with(
            start_node_id="n1",
            max_depth=2,
            edge_types=["KNOWS"],
            node_labels=None,
            algorithm="DFS",
            limit=10,
            graph_id=None,
        )

    def test_graph_collection_management(self, client, mock_adapter):
        mock_adapter.create_graph.return_value = {"graph_id": "g"}
        mock_adapter.delete_graph.return_value = {"success": True}
        mock_adapter.get_graph.return_value = {"name": "g"}
        mock_adapter.list_graphs.return_value = {"data": []}
        mock_adapter.get_graph_stats.return_value = {"node_count": 0}

        assert client.create_graph("g", name="G", schema={"k": 1}) == {"graph_id": "g"}
        mock_adapter.create_graph.assert_called_once_with(
            graph_id="g", name="G", description=None, schema={"k": 1}
        )
        client.delete_graph("g")
        mock_adapter.delete_graph.assert_called_once_with(graph_id="g")
        client.get_graph("g")
        mock_adapter.get_graph.assert_called_once_with(graph_id="g")
        client.list_graphs()
        mock_adapter.list_graphs.assert_called_once_with()
        client.get_graph_stats("g")
        mock_adapter.get_graph_stats.assert_called_once_with(graph_id="g")

    def test_graph_shortest_path(self, client, mock_adapter):
        mock_adapter.graph_shortest_path.return_value = {"found": True}
        client.graph_shortest_path(
            "a", "b", algorithm="BELLMAN_FORD", k=3, graph_id="social"
        )
        mock_adapter.graph_shortest_path.assert_called_once_with(
            "a",
            "b",
            None,
            None,
            "BELLMAN_FORD",
            3,
            None,
            None,
            timeout=None,
            graph_id="social",
        )

    def test_graph_traverse_forwards_graph_id(self, client, mock_adapter):
        """v1 dropped graph_id here (always ran on default); v2 forwards it."""
        mock_adapter.graph_traverse.return_value = {"nodes": []}
        client.graph_traverse("a", max_depth=4, graph_id="social")
        mock_adapter.graph_traverse.assert_called_once_with(
            "a",
            4,
            None,
            "BFS",
            None,
            None,
            None,
            None,
            None,
            timeout=None,
            graph_id="social",
        )


# ---------------------------------------------------------------------------
# 2. Facade validation (carried over from v1)
# ---------------------------------------------------------------------------


class TestGraphValidation:
    def test_create_node_type_checks(self, client):
        with pytest.raises(TypeError, match="node_id must be str"):
            client.create_node(123, ["P"])
        with pytest.raises(TypeError, match="labels must be list"):
            client.create_node("n1", "P")
        with pytest.raises(TypeError, match="properties must be dict"):
            client.create_node("n1", ["P"], properties=[("k", 1)])

    def test_create_edge_type_checks(self, client):
        with pytest.raises(TypeError, match="edge_id must be str"):
            client.create_edge(1, "a", "b", "KNOWS")
        with pytest.raises(TypeError, match="weight must be number"):
            client.create_edge("e1", "a", "b", "KNOWS", weight="heavy")

    def test_traverse_graph_validation(self, client):
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            client.traverse_graph("n1", max_depth=0)
        with pytest.raises(ValueError, match="BFS/DFS/PARALLEL_BFS"):
            client.traverse_graph("n1", algorithm="GREEDY")
        with pytest.raises(TypeError, match="edge_types must be list"):
            client.traverse_graph("n1", edge_types="KNOWS")


# ---------------------------------------------------------------------------
# 3. Error translation
# ---------------------------------------------------------------------------


def test_adapter_not_implemented_becomes_proximadb_error(client, mock_adapter):
    mock_adapter.delete_graph.side_effect = NotImplementedError(
        "RestProtocolAdapter does not support graph collection deletion"
    )
    with pytest.raises(ProximaDBError, match="does not support delete_graph"):
        client.delete_graph("g")


# ---------------------------------------------------------------------------
# 4. Wire equivalence: the REST adapter vs the v1 dispatch shape
# ---------------------------------------------------------------------------


class _Recorder:
    """Stands in for protocols.rest_sync.ProximaDBClient; records calls."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return {"success": True}

        return _call


@pytest.fixture
def rest_adapter():
    from proximadb_sdk.adapters.rest_adapter import RestProtocolAdapter

    adapter = RestProtocolAdapter(url="http://testserver:5678")
    recorder = _Recorder()
    adapter._client = recorder
    adapter._recorder = recorder
    return adapter


class TestRestWireEquivalence:
    """Same adapter config → same transport calls the v1 dispatch makes.

    v1's ``_invoke_graph_method`` omits ``graph_id`` when None (letting the
    transport default apply) and passes it as ``graph_id=`` otherwise; the
    adapter normalizes ``None → "default"`` explicitly — same wire shape.
    """

    def test_create_node_default_graph(self, rest_adapter):
        rest_adapter.create_node("n1", ["P"])
        name, args, kwargs = rest_adapter._recorder.calls[0]
        assert (name, kwargs) == (
            "create_node",
            {
                "node_id": "n1",
                "labels": ["P"],
                "properties": None,
                "embedding": None,
                "graph_id": "default",
            },
        )

    def test_create_node_explicit_graph(self, rest_adapter):
        rest_adapter.create_node("n1", ["P"], graph_id="social")
        _, _, kwargs = rest_adapter._recorder.calls[0]
        assert kwargs["graph_id"] == "social"

    def test_traverse_graph_full_kwargs(self, rest_adapter):
        rest_adapter.traverse_graph(
            "n1",
            max_depth=2,
            edge_types=["KNOWS"],
            node_labels=["Person"],
            algorithm="DFS",
            limit=7,
            graph_id="social",
        )
        name, _, kwargs = rest_adapter._recorder.calls[0]
        assert name == "traverse_graph"
        assert kwargs == {
            "start_node_id": "n1",
            "max_depth": 2,
            "edge_types": ["KNOWS"],
            "node_labels": ["Person"],
            "algorithm": "DFS",
            "limit": 7,
            "graph_id": "social",
        }

    def test_query_nodes_and_reads(self, rest_adapter):
        rest_adapter.query_nodes(labels=["P"], limit=3, offset=1, graph_id="g")
        _, _, kwargs = rest_adapter._recorder.calls[0]
        assert kwargs == {
            "labels": ["P"],
            "properties": None,
            "limit": 3,
            "offset": 1,
            "graph_id": "g",
        }
        rest_adapter.get_node("n1")
        _, _, kwargs = rest_adapter._recorder.calls[1]
        assert kwargs == {"node_id": "n1", "graph_id": "default"}

    def test_graph_collection_calls_positional_shape(self, rest_adapter):
        rest_adapter.delete_graph("g")
        name, args, kwargs = rest_adapter._recorder.calls[0]
        assert (name, args, kwargs) == ("delete_graph", (), {"graph_id": "g"})
        rest_adapter.get_graph_stats("g")
        name, args, kwargs = rest_adapter._recorder.calls[1]
        assert (name, kwargs) == ("get_graph_stats", {"graph_id": "g"})

    def test_analytics_positional_v1_shape(self, rest_adapter):
        rest_adapter.graph_shortest_path(
            "a",
            "b",
            None,
            ["KNOWS"],
            "DIJKSTRA",
            2,
            True,
            64,
            timeout=1.5,
            graph_id="social",
        )
        name, args, kwargs = rest_adapter._recorder.calls[0]
        assert name == "graph_shortest_path"
        assert args == ("a", "b", None, ["KNOWS"], "DIJKSTRA", 2, True, 64)
        assert kwargs == {"timeout": 1.5, "graph_id": "social"}
        rest_adapter.graph_traverse(
            "a",
            3,
            None,
            "BFS",
            10,
            100,
            50,
            True,
            64,
            timeout=2.0,
            graph_id="social",
        )
        name, args, kwargs = rest_adapter._recorder.calls[1]
        assert name == "graph_traverse"
        assert kwargs["graph_id"] == "social"


# ---------------------------------------------------------------------------
# 4b. Embedded adapter: canonical kwarg compatibility + batch degradation
# ---------------------------------------------------------------------------


class TestEmbeddedGraphCompatibility:
    def _embedded_adapter(self, db):
        from proximadb_sdk.adapters.embedded_adapter import EmbeddedProtocolAdapter

        adapter = EmbeddedProtocolAdapter.__new__(EmbeddedProtocolAdapter)
        adapter._db = db
        return adapter

    def test_existing_methods_accept_canonical_graph_id_kwarg(self):
        """The pre-existing embedded graph methods take ``graph=`` but pop
        ``graph_id`` from kwargs — the canonical spelling must work unchanged."""
        seen = {}

        class Db:
            def create_node(self, graph_id, node_id, labels, properties):
                seen["args"] = (graph_id, node_id, labels, properties)
                return {"ok": True}

        adapter = self._embedded_adapter(Db())
        result = adapter.create_node(
            node_id="n1", labels=["P"], properties={"k": 1}, graph_id="social"
        )
        assert result["success"] is True
        assert seen["args"][0] == "social"

    def test_batch_degrades_without_engine_classes(self):
        class Db:  # has the method but the module exposes no GraphNode class
            def create_nodes(self, graph_id, nodes):
                return len(nodes)

        adapter = self._embedded_adapter(Db())
        with pytest.raises(NotImplementedError):
            adapter.batch_create_nodes([{"id": "n1"}], graph_id="g")

    def test_batch_maps_dicts_to_engine_objects(self):
        import sys as _sys
        import types as _types

        made_nodes, made_edges = {}, {}

        class GraphNode:
            def __init__(self, id, labels=None, properties=None):
                made_nodes[id] = (labels, properties)

        class GraphEdge:
            def __init__(
                self,
                from_node_id,
                to_node_id,
                edge_type,
                id=None,
                weight=None,
                properties=None,
            ):
                made_edges[id] = (from_node_id, to_node_id, edge_type, weight)

        # Model the real layout: the PyO3 engine exposes GraphNode/GraphEdge
        # at MODULE level of the db object's module — not as class attributes.
        fake_module = _types.ModuleType("fake_embedded_engine")
        fake_module.GraphNode = GraphNode
        fake_module.GraphEdge = GraphEdge
        _sys.modules["fake_embedded_engine"] = fake_module

        class Db:
            def create_nodes(self, graph_id, nodes):
                made_nodes["__graph__"] = graph_id
                return len(nodes)

            def create_edges(self, graph_id, edges):
                made_edges["__graph__"] = graph_id
                return len(edges)

        Db.__module__ = "fake_embedded_engine"

        adapter = self._embedded_adapter(Db())
        result = adapter.batch_create_nodes(
            [{"id": "n1", "labels": ["P"], "properties": {"k": 1}}], graph_id="social"
        )
        assert result == {
            "success": True,
            "created_count": 1,
            "nodes": [{"id": "n1", "labels": ["P"], "properties": {"k": 1}}],
        }
        assert made_nodes["n1"] == (["P"], {"k": 1})
        assert made_nodes["__graph__"] == "social"

        result = adapter.batch_create_edges(
            [{"from_node_id": "a", "to_node_id": "b", "edge_type": "KNOWS"}],
            graph_id="social",
        )
        assert result["created_count"] == 1
        assert made_edges["__graph__"] == "social"


# ---------------------------------------------------------------------------
# 5. Parity census ratchet
# ---------------------------------------------------------------------------


class TestParityCensus:
    def test_every_ported_method_exists_on_v2(self):
        for name in PORTED_GRAPH_METHODS:
            assert callable(
                getattr(ProximaDBClient, name, None)
            ), f"TD-SDK-3 ratchet: {name} missing from the v2 client"

    def test_base_adapter_defines_the_whole_family(self):
        """The canonical protocol surface exists on the base class, so every
        adapter (real or mocked) accepts the full family."""
        for name in PORTED_GRAPH_METHODS:
            assert hasattr(
                BaseProtocolAdapter, name
            ), f"TD-SDK-3 ratchet: {name} missing from BaseProtocolAdapter"

    def test_ported_set_matches_v1_public_graph_surface(self):
        """The pinned set IS v1's public graph family (the 2026-10-31 AST
        census: 17 public methods + the private ``_invoke_graph_method``
        dispatcher). If this fails, v1's surface changed — re-census and
        update the pin deliberately."""
        from proximadb_sdk.unified_client import ProximaDBClient as V1Client

        v1_graph_names = {
            name
            for name in dir(V1Client)
            if not name.startswith("_")
            and any(
                k in name
                for k in ("graph", "node", "edge", "traverse", "neighbor", "vertex")
            )
        }
        # v1-only names outside batch 1's scope would appear here as extras;
        # batch 1 claims the ENTIRE graph family, so the sets must be equal.
        assert v1_graph_names == set(PORTED_GRAPH_METHODS), (
            f"v1 graph surface drifted: extra={v1_graph_names - set(PORTED_GRAPH_METHODS)} "
            f"missing={set(PORTED_GRAPH_METHODS) - v1_graph_names}"
        )

    def test_v2_does_not_invent_graph_names(self):
        from proximadb_sdk.unified_client import ProximaDBClient as V1Client

        v1_names = {name for name in dir(V1Client) if not name.startswith("_")}
        invented = set(PORTED_GRAPH_METHODS) - v1_names
        assert not invented, f"v2 invented graph names not on v1: {invented}"
