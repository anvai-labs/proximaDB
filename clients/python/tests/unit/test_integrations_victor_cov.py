"""Offline coverage for proximadb_sdk.integrations.victor.

ProximaDBEmbeddingProvider.__init__ builds a REAL unified ProximaDBClient
(which connects on construction) and the storage path calls insert_records /
client.search / client.delete_vectors. A prior attempt hung because it left the
real client in place. The fix here: patch victor.ProximaDBClient,
EmbeddingModelConfig, create_embedding_model and insert_records BEFORE
constructing the provider — all via monkeypatch (auto-revert, no sys.modules
mutation), so the file is order-independent w.r.t. sibling *_cov.py files.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# This module exercises the SDK's adapter for the external `victor` (victor-ai)
# package, so importing it requires `victor` installed (../victor editable, or
# PyPI `victor-ai`). Skip cleanly when absent, and tag `live` so the offline SDK
# coverage ratchet filters it out (run it explicitly in an env that has victor).
pytest.importorskip("victor")
pytestmark = pytest.mark.live

import proximadb_sdk.integrations.victor as victor_mod  # noqa: E402


def _cfg(**over):
    base = dict(
        extra_config={
            "server_url": "http://testserver",
            "collection_name": "code_embeddings",
            "dimension": 4,
            "batch_size": 8,
        },
        embedding_model_type="bert",
        embedding_api_key="key",
        distance_metric="cosine",
        embedding_model_name="m1",
        embedding_model="m1",
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def victor(monkeypatch):
    """A provider with the real client + model + storage fully mocked."""
    client = MagicMock()
    monkeypatch.setattr(victor_mod, "ProximaDBClient", lambda url=None, **k: client)
    monkeypatch.setattr(victor_mod, "EmbeddingModelConfig", MagicMock())
    model = MagicMock()
    model.initialize = AsyncMock()
    model.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
    model.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    model.close = AsyncMock()
    monkeypatch.setattr(victor_mod, "create_embedding_model", lambda cfg: model)
    monkeypatch.setattr(victor_mod, "insert_records", MagicMock())
    p = victor_mod.ProximaDBEmbeddingProvider(_cfg())
    return SimpleNamespace(
        p=p, client=client, model=model, insert=victor_mod.insert_records
    )


def _hit(rid="d1", score=0.9, source="text", metadata=None):
    return SimpleNamespace(id=rid, score=score, source=source, metadata=metadata or {})


# ── construction / helpers ────────────────────────────────────────────────
def test_construction_defaults_and_client_mocked(victor):
    assert victor.p._collection_name == "code_embeddings"
    assert victor.p._dimension == 4
    assert victor.p._client is victor.client
    assert victor.p._initialized is False


def test_get_embedding_model_name_and_fallback(victor):
    assert victor.p._get_embedding_model_name() == "m1"
    victor.p.config = _cfg(embedding_model_name=None, embedding_model="fallback")
    # attribute missing -> getattr fallback chain
    delattr(victor.p.config, "embedding_model_name")
    assert victor.p._get_embedding_model_name() == "fallback"


# ── initialize ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_initialize_creates_model_and_collection(victor):
    await victor.p.initialize()
    assert victor.p._initialized is True
    victor.model.initialize.assert_awaited_once()
    victor.client.create_collection.assert_called_once()
    # idempotent: second call is a no-op
    victor.model.initialize.reset_mock()
    await victor.p.initialize()
    victor.model.initialize.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialize_swallows_collection_exists(victor):
    victor.client.create_collection.side_effect = RuntimeError("exists")
    await victor.p.initialize()  # must not raise
    assert victor.p._initialized is True


# ── embedding delegation ─────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_embed_text_and_batch(victor):
    await victor.p.initialize()
    assert await victor.p.embed_text("x") == [0.1, 0.2, 0.3, 0.4]
    assert await victor.p.embed_batch(["a", "b"]) == [[0.1, 0.2, 0.3, 0.4]]


@pytest.mark.asyncio
async def test_embed_requires_initialized_model(victor):
    with pytest.raises(RuntimeError, match="not initialized"):
        await victor.p.embed_text("x")
    with pytest.raises(RuntimeError, match="not initialized"):
        await victor.p.embed_batch(["x"])


# ── indexing ───────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_index_document_inserts_record(victor):
    await victor.p.initialize()
    await victor.p.index_document("d1", "hello", {"file_path": "m.py"})
    victor.insert.assert_called_once()
    args = victor.insert.call_args[0]
    assert args[0] is victor.client and args[1] == "code_embeddings"


@pytest.mark.asyncio
async def test_index_documents_batch_and_empty(victor):
    await victor.p.initialize()
    await victor.p.index_documents([])  # early return, no insert
    victor.insert.assert_not_called()
    await victor.p.index_documents(
        [
            {"id": "a", "content": "c1", "metadata": {"k": 1}},
            {"id": "b", "content": "c2"},
        ]
    )
    victor.insert.assert_called_once()


# ── search ───────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_search_similar_maps_results(victor):
    await victor.p.initialize()
    victor.client.search.return_value = [
        _hit(
            "d1",
            0.8,
            "src text",
            {"file_path": "a.py", "symbol_name": "f", "line_number": 3, "extra": "v"},
        ),
        _hit("d2", 0.5, None, {"content": "fallback"}),
    ]
    res = victor.p_search = await victor.p.search_similar("q", limit=5)
    assert len(res) == 2
    assert res[0].file_path == "a.py" and res[0].symbol_name == "f"
    assert res[0].content == "src text" and res[0].metadata == {"extra": "v"}
    assert res[1].content == "fallback"  # source None -> metadata content


# ── deletion / lifecycle ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_delete_document(victor):
    await victor.p.delete_document("d1")
    victor.client.delete_vectors.assert_called_once_with("code_embeddings", ["d1"])


@pytest.mark.asyncio
async def test_delete_by_file_hits_and_deletes(victor):
    victor.client.search.return_value = [_hit("d1"), _hit("d2")]
    n = await victor.p.delete_by_file("a.py")
    assert n == 2
    victor.client.delete_vectors.assert_called_once_with(
        "code_embeddings", ["d1", "d2"]
    )


@pytest.mark.asyncio
async def test_delete_by_file_no_hits_and_search_error(victor):
    victor.client.search.return_value = []
    assert await victor.p.delete_by_file("a.py") == 0
    # Since #1717 a failed read must be distinguishable from an empty one:
    # delete_by_file RAISES on a failed search instead of returning 0 (0
    # means "nothing matched", which a caller treats as success). This test
    # still asserted the pre-#1717 swallow — #1717 swept the sibling cov
    # files (test_code_knowledge_cov, test_graph_cov) and missed this one.
    victor.client.search.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="failed to delete records for file: boom"):
        await victor.p.delete_by_file("a.py")


@pytest.mark.asyncio
async def test_clear_index(victor):
    await victor.p.clear_index()
    victor.client.delete_collection.assert_called_once()
    victor.client.create_collection.assert_called_once()
    # exceptions are swallowed
    victor.client.delete_collection.side_effect = RuntimeError("x")
    victor.client.create_collection.side_effect = RuntimeError("y")
    await victor.p.clear_index()  # must not raise


@pytest.mark.asyncio
async def test_get_stats(victor):
    stats = await victor.p.get_stats()
    assert stats["provider"] == "proximadb"
    assert stats["collection_name"] == "code_embeddings"
    assert stats["dimension"] == 4
    assert stats["embedding_model"] == "m1"
