"""Embedded smoke tests for agentic SDK contracts."""

from __future__ import annotations

import pytest

from proximadb_sdk.adapters.embedded_adapter import EmbeddedProtocolAdapter
from proximadb_sdk.integrations.agentic_io import ProximaEventStore
from proximadb_sdk.integrations.agentic_store import (
    ProximaBaseStore,
    ProximaCheckpointSaver,
)
from proximadb_sdk.integrations.victor_embedded import (
    ProximaDBEmbeddedVictorProvider,
    VictorSymbolRecord,
)

pytest.importorskip("proximadb_embedded")


def _adapter(tmp_path) -> EmbeddedProtocolAdapter:
    return EmbeddedProtocolAdapter(data_dir=str(tmp_path / "embedded_data"))


def test_agentic_store_checkpoint_and_event_use_real_embedded_documents(
    tmp_path,
) -> None:
    adapter = _adapter(tmp_path)

    store = ProximaBaseStore(adapter)
    store.put(("tenant", "thread"), "profile", {"role": "planner"})
    assert store.get(("tenant", "thread"), "profile").value == {"role": "planner"}

    saver = ProximaCheckpointSaver(adapter)
    config = {"configurable": {"thread_id": "thread-embedded"}}
    saved_config = saver.put(config, {"id": "checkpoint-1"}, {"source": "test"})
    saver.put_writes(saved_config, [("messages", {"content": "hello"})], "task-1")
    checkpoint = saver.get_tuple(config)
    assert checkpoint is not None
    assert checkpoint.checkpoint["id"] == "checkpoint-1"
    assert checkpoint.pending_writes[0]["channel"] == "messages"

    events = ProximaEventStore(adapter)
    first = events.append("stream-embedded", "Started", {}, expected_version=0)
    second = events.snapshot("stream-embedded", {"done": True})
    assert [event.event_id for event in events.read_stream("stream-embedded")] == [
        first.event_id,
        second.event_id,
    ]


def test_unified_query_cross_modal_path_uses_embedded_unified_entrypoint(
    tmp_path,
) -> None:
    adapter = _adapter(tmp_path)
    adapter.create_document_collection(
        "agent_docs", config={"indexed_paths": ["$.role"]}
    )
    adapter.insert_document(
        "agent_docs",
        {"id": "doc-main", "role": "planner", "embedding": [1.0, 0.0, 0.0, 0.0]},
        id="doc-main",
    )
    adapter.create_collection("agent_vectors", dimension=4)
    adapter.insert_vectors(
        "agent_vectors",
        [
            {
                "id": "doc-main",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "metadata": {"role": "planner"},
            }
        ],
    )
    adapter.create_graph("agent_graph")
    adapter.create_node(
        graph="agent_graph",
        node_id="doc-main",
        labels=["Symbol"],
        properties={"name": "main"},
    )

    query = (
        "SELECT * "
        "FROM DOCUMENT_QUERY('agent_docs', '$.role = \"planner\"') d "
        "JOIN LATERAL VECTOR_SEARCH('agent_vectors', d.document.embedding, 1) v ON true "
        "JOIN LATERAL GRAPH_QUERY('MATCH (s:Symbol) RETURN s') g ON true"
    )

    plan = adapter._db.explain_unified_query(query)
    assert plan["fusion_strategy"] == "rrf"
    assert {component["model"] for component in plan["components"]} == {
        "document",
        "vector",
        "graph",
    }

    results = adapter.execute_unified_query(query, fusion_strategy="rrf")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_victor_embedded_provider_uses_mapper_graph_and_events(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    provider = ProximaDBEmbeddedVictorProvider(adapter, workspace="agentic")

    await provider.upsert_symbol(
        VictorSymbolRecord(
            id="symbol-main",
            name="main",
            kind="Function",
            file_path="app.py",
            language="python",
            line=1,
            content="def main(): pass",
        ),
        vector=[1.0, 0.0, 0.0, 0.0],
    )
    await provider.upsert_symbol(
        VictorSymbolRecord(
            id="symbol-helper",
            name="helper",
            kind="Function",
            file_path="app.py",
            language="python",
            line=4,
            content="def helper(): pass",
        )
    )

    functions = await provider.find_symbols(kind="Function", file_path="app.py")
    assert {symbol.id for symbol in functions} == {"symbol-main", "symbol-helper"}

    linked = await provider.link_symbols(
        "symbol-main",
        "CALLS",
        "symbol-helper",
        properties={"line": 2},
    )
    assert linked["success"] is True

    semantic = await provider.semantic_search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert semantic[0].id == "symbol-main"

    history = await provider.event_history()
    assert [event["event_type"] for event in history] == [
        "SymbolUpserted",
        "SymbolUpserted",
        "SymbolLinked",
    ]


@pytest.mark.asyncio
async def test_victor_embedded_provider_matches_codingagent_unified_symbol_protocol(
    tmp_path,
) -> None:
    pytest.importorskip("victor.storage.unified.protocol")
    from victor.storage.unified.protocol import SearchParams, UnifiedEdge, UnifiedSymbol

    adapter = _adapter(tmp_path)
    provider = ProximaDBEmbeddedVictorProvider(adapter, workspace="victor_protocol")
    await provider.initialize(tmp_path)

    symbol = UnifiedSymbol(
        unified_id=provider.make_symbol_id("app.py", "main"),
        name="main",
        type="function",
        file_path="app.py",
        line=1,
        lang="python",
        signature="def main()",
        docstring="Entry point",
    )
    await provider.index_symbol(symbol, "def main(): return helper()")
    await provider.index_edge(
        UnifiedEdge(
            src_id=symbol.unified_id,
            dst_id=provider.make_symbol_id("app.py", "helper"),
            type="CALLS",
        )
    )

    loaded = await provider.get_symbol(symbol.unified_id)
    assert loaded is not None
    assert loaded.unified_id == symbol.unified_id

    by_file = await provider.get_symbols_in_file("app.py")
    assert [item.unified_id for item in by_file] == [symbol.unified_id]

    keyword = await provider.search_keyword("main", limit=5)
    assert keyword[0].symbol.unified_id == symbol.unified_id

    hybrid = await provider.search(SearchParams(query="main", limit=5))
    assert hybrid[0].symbol.unified_id == symbol.unified_id

    stats = await provider.stats()
    assert stats["symbol_count"] == 1
