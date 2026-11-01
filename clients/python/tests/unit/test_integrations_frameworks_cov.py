"""Offline unit tests for proximadb_sdk.integrations framework adapters.

Covers: langchain, langgraph, mcp_tools, graph_walk_client, dual_use_store,
mlops, agentic_ddl.

All heavy/optional deps (langchain_core, langchain) are stubbed via
sys.modules BEFORE importing the targets. The ProximaDB client is mocked.
No network, no model downloads, no real server.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Stub langchain_core / langchain before any target import.
# ---------------------------------------------------------------------------
def _install_langchain_stubs() -> None:
    # Always evict ProximaDB adapter modules that may have captured the *real*
    # langchain symbols in an earlier test (order-dependent sys.modules pollution),
    # so a subsequent fresh import rebinds against the stubs below.
    for _adapter in (
        "proximadb_sdk.integrations.langchain",
        "proximadb_sdk.integrations.langgraph",
    ):
        sys.modules.pop(_adapter, None)

    # Note: we deliberately do NOT short-circuit when a stub is already present.
    # A concurrently-collected test module can re-import the *real*
    # langchain_core.tools.retriever into sys.modules between calls; rebuilding
    # the stubs unconditionally keeps the offline assertions order-independent.

    lc_core = types.ModuleType("langchain_core")
    lc_core._proximadb_stub = True  # type: ignore[attr-defined]

    # langchain_core.documents.Document
    docs_mod = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content: str = "", metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

        def __eq__(self, other: object) -> bool:
            return (
                isinstance(other, Document)
                and other.page_content == self.page_content
                and other.metadata == self.metadata
            )

    docs_mod.Document = Document

    # langchain_core.embeddings.Embeddings
    emb_mod = types.ModuleType("langchain_core.embeddings")

    class Embeddings:
        def embed_documents(self, texts):  # pragma: no cover - overridden
            raise NotImplementedError

        def embed_query(self, text):  # pragma: no cover - overridden
            raise NotImplementedError

    emb_mod.Embeddings = Embeddings

    # langchain_core.vectorstores.VectorStore
    vs_mod = types.ModuleType("langchain_core.vectorstores")

    class _Retriever:
        def __init__(self, store, search_kwargs):
            self.store = store
            self.search_kwargs = search_kwargs

    class VectorStore:
        def as_retriever(self, **kwargs: Any) -> Any:
            return _Retriever(self, kwargs.get("search_kwargs", {}))

    vs_mod.VectorStore = VectorStore
    vs_mod._Retriever = _Retriever

    # langchain_core.tools.BaseTool + create_retriever_tool
    tools_mod = types.ModuleType("langchain_core.tools")

    class BaseTool:
        def __init__(
            self, name: str = "", description: str = "", retriever: Any = None
        ):
            self.name = name
            self.description = description
            self.retriever = retriever

    def create_retriever_tool(retriever, *, name, description):
        return BaseTool(name=name, description=description, retriever=retriever)

    tools_mod.BaseTool = BaseTool
    tools_mod.create_retriever_tool = create_retriever_tool

    # langchain_core.tools.retriever submodule (canonical home of
    # create_retriever_tool in LangChain 0.3/1.x; the langgraph adapter imports
    # it from here rather than the re-export on langchain_core.tools).
    tools_retriever_mod = types.ModuleType("langchain_core.tools.retriever")
    tools_retriever_mod.create_retriever_tool = create_retriever_tool
    tools_mod.retriever = tools_retriever_mod  # type: ignore[attr-defined]

    lc_core.documents = docs_mod  # type: ignore[attr-defined]
    lc_core.embeddings = emb_mod  # type: ignore[attr-defined]
    lc_core.vectorstores = vs_mod  # type: ignore[attr-defined]
    lc_core.tools = tools_mod  # type: ignore[attr-defined]

    # Give each stub a valid ModuleSpec. Without it, `__spec__` is None, and a
    # later test that does `importlib.util.find_spec("langchain_core")` (e.g. the
    # SDK's optional-export availability probe) raises
    # `ValueError: langchain_core.__spec__ is None`. The stub leaks into
    # sys.modules for the rest of the session, so this isolates the pollution.
    import importlib.machinery

    for _name, _mod in (
        ("langchain_core", lc_core),
        ("langchain_core.documents", docs_mod),
        ("langchain_core.embeddings", emb_mod),
        ("langchain_core.vectorstores", vs_mod),
        ("langchain_core.tools", tools_mod),
        ("langchain_core.tools.retriever", tools_retriever_mod),
    ):
        _mod.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)

    sys.modules["langchain_core"] = lc_core
    sys.modules["langchain_core.documents"] = docs_mod
    sys.modules["langchain_core.embeddings"] = emb_mod
    sys.modules["langchain_core.vectorstores"] = vs_mod
    sys.modules["langchain_core.tools"] = tools_mod
    sys.modules["langchain_core.tools.retriever"] = tools_retriever_mod


_install_langchain_stubs()


# Now safe to import targets.
from proximadb_sdk.integrations import (  # noqa: E402
    agentic_ddl,
    dual_use_store,
    graph_walk_client,
    mcp_tools,
    mlops,
)
from proximadb_sdk.integrations.dual_use_store import (  # noqa: E402
    DualUseModel,
    DualUseRetrievalResult,
    DualUseStore,
)
from proximadb_sdk.integrations.graph_walk_client import (  # noqa: E402
    GraphWalkClient,
    GraphWalkError,
    StubTransport,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class FakeEmbeddings:
    """Deterministic stub of langchain Embeddings."""

    def embed_documents(self, texts):
        return [[float(len(t)), 1.0, 2.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 2.0]


class FakeSearchResult:
    def __init__(self, *, id="r1", score=0.9, metadata=None, source=None, vector=None):
        self.id = id
        self.score = score
        self.metadata = metadata
        self.source = source
        self.vector = vector


class FakeClient:
    """Captures insert/search/delete calls."""

    def __init__(self, search_results=None, has_insert_records=True):
        self.search_results = search_results or []
        self.inserted = []
        self.deleted = []
        self.searches = []
        if has_insert_records:
            self.insert_records = self._insert_records  # type: ignore[assignment]

    def _insert_records(self, collection_name, records):
        self.inserted.append((collection_name, records))
        return {"inserted": len(records)}

    def insert_vectors(self, collection_name, records=None):
        self.inserted.append((collection_name, records))
        return {"inserted": len(records or [])}

    def search(
        self,
        collection_name,
        vector=None,
        top_k=None,
        metadata_filter=None,
        include_vectors=None,
    ):
        self.searches.append(
            dict(
                collection_name=collection_name,
                vector=vector,
                top_k=top_k,
                metadata_filter=metadata_filter,
                include_vectors=include_vectors,
            )
        )
        return self.search_results

    def delete_vectors(self, collection_name, ids):
        self.deleted.append((collection_name, ids))


class FakeDualModel:
    def embed(self, text: str):
        return [float(len(text)), 0.5]

    def decompress(self, embedding) -> str:
        return f"decompressed:{list(embedding)}"


# ---------------------------------------------------------------------------
# langchain.ProximaDBVectorStore
# ---------------------------------------------------------------------------
def _import_langchain():
    from proximadb_sdk.integrations import langchain as lc

    return lc


def test_langchain_add_texts_generates_ids_and_inserts():
    lc = _import_langchain()
    client = FakeClient()
    store = lc.ProximaDBVectorStore(client, "docs", FakeEmbeddings())
    ids = store.add_texts(["hello", "world"], metadatas=[{"a": 1}, {"b": 2}])
    assert len(ids) == 2
    assert len(client.inserted) == 1
    coll, records = client.inserted[0]
    assert coll == "docs"
    assert records[0]["id"] == ids[0]
    # TD-REC-1: text rides in text_fields, not a top-level source.
    assert records[0]["text_fields"][0]["content"] == "hello"
    assert records[0]["props"] == {"a": 1}


def test_langchain_add_texts_with_explicit_ids():
    lc = _import_langchain()
    client = FakeClient()
    store = lc.ProximaDBVectorStore(client, "docs", FakeEmbeddings())
    ids = store.add_texts(["a", "b"], ids=["x", "y"])
    assert ids == ["x", "y"]


def test_langchain_embeddings_property():
    lc = _import_langchain()
    emb = FakeEmbeddings()
    store = lc.ProximaDBVectorStore(FakeClient(), "docs", emb)
    assert store.embeddings is emb


def test_langchain_delete_empty_and_nonempty():
    lc = _import_langchain()
    client = FakeClient()
    store = lc.ProximaDBVectorStore(client, "docs", FakeEmbeddings())
    assert store.delete(None) is False
    assert store.delete([]) is False
    assert store.delete(["id1", "id2"]) is True
    assert client.deleted == [("docs", ["id1", "id2"])]


def test_langchain_similarity_search_uses_source():
    lc = _import_langchain()
    results = [FakeSearchResult(score=0.8, source="content text", metadata={"m": 1})]
    client = FakeClient(search_results=results)
    store = lc.ProximaDBVectorStore(client, "docs", FakeEmbeddings())
    docs = store.similarity_search("query", k=2)
    assert len(docs) == 1
    assert docs[0].page_content == "content text"
    assert docs[0].metadata == {"m": 1}
    assert client.searches[0]["top_k"] == 2


def test_langchain_similarity_search_falls_back_to_text_key():
    lc = _import_langchain()
    results = [FakeSearchResult(score=0.7, source=None, metadata={"text": "from meta"})]
    client = FakeClient(search_results=results)
    store = lc.ProximaDBVectorStore(client, "docs", FakeEmbeddings())
    pairs = store.similarity_search_with_score("q", k=1)
    assert pairs[0][0].page_content == "from meta"
    # text key popped out of metadata
    assert "text" not in pairs[0][0].metadata
    assert pairs[0][1] == 0.7


def test_langchain_similarity_search_no_metadata():
    lc = _import_langchain()
    results = [FakeSearchResult(score=0.5, source=None, metadata=None)]
    client = FakeClient(search_results=results)
    store = lc.ProximaDBVectorStore(client, "docs", FakeEmbeddings())
    docs = store.similarity_search_by_vector([0.1, 0.2], k=1)
    assert docs[0].page_content == ""


def test_langchain_similarity_search_by_vector_with_score_filter():
    lc = _import_langchain()
    client = FakeClient(search_results=[])
    store = lc.ProximaDBVectorStore(client, "docs", FakeEmbeddings())
    store.similarity_search_by_vector_with_score([0.1], k=3, filter={"k": "v"})
    assert client.searches[0]["metadata_filter"] == {"k": "v"}


def test_langchain_from_texts_requires_client():
    lc = _import_langchain()
    with pytest.raises(ValueError, match="requires a 'client'"):
        lc.ProximaDBVectorStore.from_texts(["a"], FakeEmbeddings())


def test_langchain_from_texts_creates_store():
    lc = _import_langchain()
    client = FakeClient()
    store = lc.ProximaDBVectorStore.from_texts(
        ["a", "b"],
        FakeEmbeddings(),
        metadatas=[{}, {}],
        client=client,
        collection_name="c",
        text_key="text",
    )
    assert isinstance(store, lc.ProximaDBVectorStore)
    assert len(client.inserted) == 1


def test_langchain_from_documents():
    lc = _import_langchain()
    Document = sys.modules["langchain_core.documents"].Document
    client = FakeClient()
    docs = [Document(page_content="x", metadata={"a": 1})]
    store = lc.ProximaDBVectorStore.from_documents(
        docs, FakeEmbeddings(), client=client
    )
    assert isinstance(store, lc.ProximaDBVectorStore)


def test_langchain_insert_records_fallback():
    lc = _import_langchain()
    client = FakeClient(has_insert_records=False)
    store = lc.ProximaDBVectorStore(client, "docs", FakeEmbeddings())
    store.add_texts(["hello"])
    assert len(client.inserted) == 1


# ---------------------------------------------------------------------------
# langgraph.create_retriever_tool
# ---------------------------------------------------------------------------
def test_langgraph_create_retriever_tool():
    # Another test module may have imported the *real* langgraph adapter earlier
    # in the session, caching the real create_retriever_tool. _install_langchain_stubs
    # re-stubs langchain_core and evicts the cached adapter modules, so a fresh
    # import below rebinds against the stubs and the assertions are order-independent.
    _install_langchain_stubs()

    from proximadb_sdk.integrations import langgraph as lg

    client = FakeClient()
    tool = lg.create_retriever_tool(
        client=client,
        collection_name="docs",
        embedding=FakeEmbeddings(),
        k=5,
        name="search_docs",
        description="Search docs.",
    )
    assert tool.name == "search_docs"
    assert tool.description == "Search docs."
    # The stub tool exposes the wrapped retriever directly, so we can assert the
    # ``k`` search kwarg was forwarded. If the *real* langchain tool leaked into
    # this session (collection-time sys.modules pollution from a sibling module),
    # it stores the retriever privately and this probe is not meaningful — the
    # name/description assertions above still cover the adapter's wiring.
    if hasattr(tool, "retriever"):
        assert tool.retriever.search_kwargs["k"] == 5


# ---------------------------------------------------------------------------
# mcp_tools
# ---------------------------------------------------------------------------
def test_mcp_list_tools_returns_deep_copies():
    tools = mcp_tools.list_tools()
    assert {t["name"] for t in tools} == {"graph_walk", "graph_step"}
    # mutate copy; canonical untouched
    tools[0]["name"] = "MUTATED"
    assert mcp_tools.GRAPH_WALK_TOOL["name"] == "graph_walk"


def test_mcp_render_invocation_walk():
    method, path, body = mcp_tools.render_invocation(
        "graph_walk",
        {"graph_id": "g1", "start_node_id": "n1", "max_depth": 3, "limit": 50},
    )
    assert method == "POST"
    assert path == "/api/v2/graphs/g1/walk"
    assert "graph_id" not in body
    assert body == {"start_node_id": "n1", "max_depth": 3, "limit": 50}


def test_mcp_render_invocation_step_strips_hallucinated_keys():
    method, path, body = mcp_tools.render_invocation(
        "graph_step",
        {"graph_id": "g1", "node_id": "n1", "bogus": "drop me", "limit": 5},
    )
    assert path == "/api/v2/graphs/g1/step"
    assert "bogus" not in body
    assert body == {"node_id": "n1", "limit": 5}


def test_mcp_render_invocation_unknown_tool():
    with pytest.raises(ValueError, match="unknown tool"):
        mcp_tools.render_invocation("nope", {})


def test_mcp_render_invocation_missing_required():
    with pytest.raises(ValueError, match="missing required argument"):
        mcp_tools.render_invocation("graph_walk", {"graph_id": "g1"})


# ---------------------------------------------------------------------------
# graph_walk_client
# ---------------------------------------------------------------------------
def test_graph_walk_client_walk_with_stub_transport():
    transport = StubTransport(response={"nodes": [1, 2]})
    client = GraphWalkClient("http://host:5678/", transport=transport)
    out = client.walk(graph_id="g1", start_node_id="n1", max_depth=2, limit=10)
    assert out == {"nodes": [1, 2]}
    method, url, body = transport.calls[0]
    assert method == "POST"
    assert url == "http://host:5678/api/v2/graphs/g1/walk"
    assert body == {"start_node_id": "n1", "max_depth": 2, "limit": 10}


def test_graph_walk_client_step_with_edge_type():
    transport = StubTransport(response={"neighbors": []})
    client = GraphWalkClient("http://host:5678", transport=transport)
    client.step(graph_id="g1", node_id="n1", edge_type="CALLS", limit=7)
    _, url, body = transport.calls[0]
    assert url == "http://host:5678/api/v2/graphs/g1/step"
    assert body == {"node_id": "n1", "limit": 7, "edge_type": "CALLS"}


def test_graph_walk_client_step_omits_empty_edge_type():
    transport = StubTransport()
    client = GraphWalkClient("http://host:5678", transport=transport)
    client.step(graph_id="g1", node_id="n1", edge_type="")
    _, _, body = transport.calls[0]
    assert "edge_type" not in body
    client.step(graph_id="g1", node_id="n2", edge_type=None)
    _, _, body2 = transport.calls[1]
    assert "edge_type" not in body2


def test_graph_walk_client_no_transport_configured():
    client = GraphWalkClient("http://host:5678")
    with pytest.raises(GraphWalkError, match="no transport configured"):
        client.walk(graph_id="g1", start_node_id="n1")


def test_graph_walk_client_wraps_transport_error():
    def boom(method, url, json):
        raise RuntimeError("connection refused")

    client = GraphWalkClient("http://host:5678", transport=boom)
    with pytest.raises(GraphWalkError, match="connection refused"):
        client.walk(graph_id="g1", start_node_id="n1")


def test_graph_walk_client_passes_through_graphwalk_error():
    def raiser(method, url, json):
        raise GraphWalkError("already wrapped")

    client = GraphWalkClient("http://host:5678", transport=raiser)
    with pytest.raises(GraphWalkError, match="already wrapped"):
        client.invoke("graph_walk", {"graph_id": "g1", "start_node_id": "n1"})


def test_graph_walk_client_invoke_validation_error_propagates():
    transport = StubTransport()
    client = GraphWalkClient("http://host:5678", transport=transport)
    with pytest.raises(ValueError):
        client.invoke("graph_walk", {"graph_id": "g1"})  # missing start_node_id


# ---------------------------------------------------------------------------
# dual_use_store
# ---------------------------------------------------------------------------
def test_dual_use_add_generates_id():
    client = FakeClient()
    store = DualUseStore(client, "coll", FakeDualModel())
    doc_id = store.add("some text")
    assert isinstance(doc_id, str) and len(doc_id) == 32  # uuid4 hex
    coll, records = client.inserted[0]
    assert coll == "coll"
    # no raw text stored
    assert "source" not in records[0]


def test_dual_use_add_with_explicit_id():
    client = FakeClient()
    store = DualUseStore(client, "coll", FakeDualModel())
    assert store.add("t", doc_id="my-id") == "my-id"


def test_dual_use_add_many_empty_is_noop():
    client = FakeClient()
    store = DualUseStore(client, "coll", FakeDualModel())
    assert store.add_many([]) == []
    assert client.inserted == []


def test_dual_use_add_many_with_ids():
    client = FakeClient()
    store = DualUseStore(client, "coll", FakeDualModel())
    out = store.add_many(["a", "b"], ids=["i1", "i2"])
    assert out == ["i1", "i2"]
    assert len(client.inserted[0][1]) == 2


def test_dual_use_add_many_generates_ids():
    client = FakeClient()
    store = DualUseStore(client, "coll", FakeDualModel())
    out = store.add_many(["a", "b"])
    assert len(out) == 2
    assert all(len(i) == 32 for i in out)


def test_dual_use_add_many_id_length_mismatch():
    store = DualUseStore(FakeClient(), "coll", FakeDualModel())
    with pytest.raises(ValueError, match="must match texts length"):
        store.add_many(["a", "b"], ids=["only-one"])


def test_dual_use_retrieve_forces_include_vectors_and_decompresses():
    results = [
        FakeSearchResult(id="r1", score=0.9, vector=[1.0, 2.0]),
        FakeSearchResult(id="r2", score=0.8, vector=None),  # skipped
    ]
    client = FakeClient(search_results=results)
    store = DualUseStore(client, "coll", FakeDualModel())
    out = store.retrieve("query", top_k=5)
    assert client.searches[0]["include_vectors"] is True
    assert client.searches[0]["top_k"] == 5
    assert len(out) == 1
    assert isinstance(out[0], DualUseRetrievalResult)
    assert out[0].id == "r1"
    assert out[0].text == "decompressed:[1.0, 2.0]"


def test_dual_use_delete_empty_and_nonempty():
    client = FakeClient()
    store = DualUseStore(client, "coll", FakeDualModel())
    store.delete([])
    assert client.deleted == []
    store.delete(["a", "b"])
    assert client.deleted == [("coll", ["a", "b"])]


def test_dual_use_model_protocol_runtime_checkable():
    assert isinstance(FakeDualModel(), DualUseModel)


# ---------------------------------------------------------------------------
# mlops
# ---------------------------------------------------------------------------
def test_mlops_feature_table_ddl_basic():
    ddl = mlops.FeatureTableDDL(name="My Features")
    sql = ddl.create_table_sql()
    assert "CREATE TABLE IF NOT EXISTS" in sql
    assert '"entity_id"' in sql
    assert "VECTOR(" not in sql  # no embedding dimension


def test_mlops_feature_table_ddl_with_embedding():
    ddl = mlops.FeatureTableDDL(name="feat", embedding_dimension=768)
    sql = ddl.create_table_sql()
    assert "VECTOR(768)" in sql


def test_mlops_feature_table_xcatalog_and_statements():
    ddl = mlops.FeatureTableDDL(name="feat", catalog_namespace="ns.x")
    xc = ddl.xcatalog_sql()
    assert any("xcatalog.namespace=ns.x" in s for s in xc)
    all_stmts = ddl.statements()
    assert len(all_stmts) == 2
    assert ddl.statements(include_xcatalog=False) == [ddl.create_table_sql()]


def test_mlops_feature_table_default_namespace():
    ddl = mlops.FeatureTableDDL(name="feat")
    xc = ddl.xcatalog_sql()
    assert any("xcatalog.namespace=features.feat" in s for s in xc)


def test_mlops_experiment_tracker_ddl():
    ddl = mlops.ExperimentTrackerDDL(prefix="exp")
    stmts = ddl.statements()
    assert len(stmts) == 6
    assert any('"exp_runs"' in s for s in stmts)
    assert any('"exp_metrics"' in s for s in stmts)
    assert any("COMMENT ON TABLE" in s for s in stmts)


def test_mlops_automl_run_spec_metadata():
    spec = mlops.AutoMLRunSpec(
        framework="autogluon",
        target="churn",
        feature_table="feat",
        label_column="y",
        problem_type="binary",
        extra={"seed": 1},
    )
    meta = spec.metadata()
    assert meta["framework"] == "autogluon"
    assert meta["problem_type"] == "binary"
    assert meta["extra"] == {"seed": 1}
    # extra is copied
    meta["extra"]["seed"] = 99
    assert spec.extra == {"seed": 1}


def test_mlops_optional_integrations():
    result = mlops.optional_integrations()
    assert set(result.keys()) == {"mlflow", "autogluon", "pycaret"}
    assert all(isinstance(v, bool) for v in result.values())


def test_mlops_ident_sanitizes():
    assert mlops._ident("My Table!") == "my_table"
    assert mlops._ident("!!!") == "mlops"


def test_mlops_quote_escapes():
    assert mlops._q('a"b') == '"a""b"'


# ---------------------------------------------------------------------------
# agentic_ddl
# ---------------------------------------------------------------------------
def test_agentic_ddl_default_factory():
    ddl = agentic_ddl.AgenticDDL.default("My Store", embedding_dimension=512)
    assert ddl.table == "my_store_agent_store"
    assert ddl.catalog_namespace == "agentic.my_store"
    assert ddl.vectors[0].dimension == 512


def test_agentic_ddl_create_table_sql():
    ddl = agentic_ddl.AgenticDDL.default("store")
    sql = ddl.create_table_sql()
    assert "CREATE TABLE IF NOT EXISTS" in sql
    assert "VECTOR(1536)" in sql
    assert "PRIMARY KEY" in sql
    assert "storage_engine = 'SST'" in sql
    assert "schema_kind = 'agentic_mixed'" in sql


def test_agentic_ddl_index_sql():
    ddl = agentic_ddl.AgenticDDL.default("store")
    idx = ddl.index_sql()
    # several indexed fields + jsonb GIN + vector HNSW
    assert any("USING HNSW" in s for s in idx)
    assert any("USING GIN" in s for s in idx)
    assert any("idx_store_agent_store_tenant_id" in s for s in idx)


def test_agentic_ddl_xcatalog_sql():
    ddl = agentic_ddl.AgenticDDL.default("store")
    xc = ddl.xcatalog_sql()
    assert any("xcatalog.namespace=agentic.store" in s for s in xc)
    assert any("xcatalog.graph.label=Symbol" in s for s in xc)
    assert any("xcatalog.event.stream_prefix=agent" in s for s in xc)


def test_agentic_ddl_statements():
    ddl = agentic_ddl.AgenticDDL.default("store")
    stmts = ddl.statements()
    assert stmts[0] == ddl.create_table_sql()
    no_xc = ddl.statements(include_xcatalog=False)
    assert all("COMMENT ON" not in s for s in no_xc)


def test_agentic_ddl_xcatalog_default_namespace_branch():
    # Construct with catalog_namespace=None to hit the `or` fallback branches.
    ddl = agentic_ddl.AgenticDDL(
        store="s",
        table="t",
        fields=(agentic_ddl.AgenticField("record_id", "TEXT", required=True),),
        vectors=(agentic_ddl.VectorProjection("embedding", 4),),
        graph=(agentic_ddl.GraphProjection("L", "record_id", ("E",)),),
        events=(agentic_ddl.EventProjection("ev"),),
        catalog_namespace=None,
    )
    sql = ddl.create_table_sql()
    assert "xcatalog_namespace = 'agentic.s'" in sql
    xc = ddl.xcatalog_sql()
    assert any("xcatalog.namespace=agentic.s" in s for s in xc)


def test_agentic_ddl_dataclasses():
    f = agentic_ddl.AgenticField("n", "TEXT", required=True, indexed=True)
    assert f.required and f.indexed
    j = agentic_ddl.JsonbProjection("col", "$.path", indexed=True)
    assert j.column == "col"
    v = agentic_ddl.VectorProjection("emb", 8)
    assert v.metadata_column == "metadata"
    g = agentic_ddl.GraphProjection("Label", "id")
    assert g.edge_types == ()
    e = agentic_ddl.EventProjection("pre")
    assert e.partition_fields == ("tenant_id", "thread_id")


def test_agentic_ddl_ident_helpers():
    assert agentic_ddl._ident("Foo Bar") == "foo_bar"
    assert agentic_ddl._ident("***") == "agent"
    assert agentic_ddl._q('x"y') == '"x""y"'
