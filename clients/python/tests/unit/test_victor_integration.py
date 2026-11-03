"""Unit tests for the Victor BaseEmbeddingProvider adapter.

These tests mock the ProximaDB client and the Victor embedding model so they
can run without a live server, embedding weights, or the victor package itself.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip entire module if Victor or its transitive optional contracts are absent.
victor_base = pytest.importorskip("victor.storage.vector_stores.base")
EmbeddingConfig = victor_base.EmbeddingConfig
EmbeddingSearchResult = victor_base.EmbeddingSearchResult

from proximadb_sdk.integrations.victor import ProximaDBEmbeddingProvider  # noqa: E402
from proximadb_sdk.models import SearchResult  # noqa: E402


def _make_config(**overrides: object) -> EmbeddingConfig:
    defaults = {
        "vector_store": "proximadb",
        "embedding_model_type": "sentence-transformers",
        "embedding_model_name": "BAAI/bge-small-en-v1.5",
        "extra_config": {
            "server_url": "http://localhost:5678",
            "collection_name": "test_coll",
            "dimension": 3,
        },
    }
    defaults.update(overrides)
    return EmbeddingConfig(**defaults)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.create_collection = MagicMock()
    client.insert_records = MagicMock()
    client.delete_vectors = MagicMock()
    client.delete_collection = MagicMock()
    client.search = MagicMock(return_value=[])
    return client


@pytest.fixture
def mock_embedding_model():
    model = AsyncMock()
    model.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
    model.embed_batch = AsyncMock(
        side_effect=lambda texts: [[float(i)] * 3 for i in range(len(texts))]
    )
    model.initialize = AsyncMock()
    model.close = AsyncMock()
    return model


@pytest.fixture
def provider(mock_client, mock_embedding_model):
    config = _make_config()
    with (
        patch(
            "proximadb_sdk.integrations.victor.ProximaDBClient",
            return_value=mock_client,
        ),
        patch(
            "proximadb_sdk.integrations.victor.create_embedding_model",
            return_value=mock_embedding_model,
        ),
    ):
        p = ProximaDBEmbeddingProvider(config)
    # Inject mocks directly
    p._client = mock_client
    p.embedding_model = mock_embedding_model
    p._initialized = True
    return p


class TestIndexDocument:
    @pytest.mark.asyncio
    async def test_index_document(self, provider, mock_client):
        await provider.index_document(
            "doc1", "hello world", {"file_path": "src/main.py"}
        )
        mock_client.insert_records.assert_called_once()
        call_kwargs = mock_client.insert_records.call_args
        records = call_kwargs.args[1]
        assert len(records) == 1
        assert records[0]["id"] == "doc1"
        # TD-REC-1: text rides in text_fields, not a top-level source.
        assert records[0]["text_fields"][0]["content"] == "hello world"
        assert records[0]["props"]["file_path"] == "src/main.py"

    @pytest.mark.asyncio
    async def test_index_document_no_metadata(self, provider, mock_client):
        await provider.index_document("doc2", "content")
        records = mock_client.insert_records.call_args.args[1]
        assert records[0]["props"] == {}


class TestIndexDocumentsBatch:
    @pytest.mark.asyncio
    async def test_index_documents(self, provider, mock_client):
        docs = [
            {"id": "a", "content": "alpha", "metadata": {"file_path": "a.py"}},
            {"id": "b", "content": "beta", "metadata": {"file_path": "b.py"}},
        ]
        await provider.index_documents(docs)
        mock_client.insert_records.assert_called_once()
        records = mock_client.insert_records.call_args.args[1]
        assert len(records) == 2
        assert records[0]["id"] == "a"
        # TD-REC-1: text rides in text_fields, not a top-level source.
        assert records[0]["text_fields"][0]["content"] == "alpha"
        assert records[1]["id"] == "b"

    @pytest.mark.asyncio
    async def test_index_documents_empty(self, provider, mock_client):
        await provider.index_documents([])
        mock_client.insert_records.assert_not_called()


class TestSearchSimilar:
    @pytest.mark.asyncio
    async def test_search_similar(self, provider, mock_client):
        mock_client.search.return_value = [
            SearchResult(
                id="r1",
                score=0.95,
                source="hello",
                metadata={
                    "file_path": "src/main.py",
                    "symbol_name": "greet",
                    "line_number": 10,
                    "extra": "val",
                },
            ),
            SearchResult(
                id="r2",
                score=0.80,
                source="world",
                metadata={"file_path": "src/util.py"},
            ),
        ]

        results = await provider.search_similar("test query", limit=2)
        assert len(results) == 2
        assert isinstance(results[0], EmbeddingSearchResult)

        # First result
        assert results[0].file_path == "src/main.py"
        assert results[0].symbol_name == "greet"
        assert results[0].content == "hello"
        assert results[0].score == 0.95
        assert results[0].line_number == 10
        # Consumed keys removed from metadata
        assert "file_path" not in results[0].metadata
        assert results[0].metadata["extra"] == "val"

        # Second result - source used for content
        assert results[1].content == "world"
        assert results[1].file_path == "src/util.py"

    @pytest.mark.asyncio
    async def test_search_similar_empty(self, provider, mock_client):
        mock_client.search.return_value = []
        results = await provider.search_similar("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_source_fallback_to_content_meta(self, provider, mock_client):
        """When source is None, fall back to content in metadata."""
        mock_client.search.return_value = [
            SearchResult(
                id="r1",
                score=0.9,
                source=None,
                metadata={"content": "from metadata", "file_path": "f.py"},
            ),
        ]
        results = await provider.search_similar("q")
        assert results[0].content == "from metadata"


class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete_document(self, provider, mock_client):
        await provider.delete_document("doc1")
        mock_client.delete_vectors.assert_called_once_with("test_coll", ["doc1"])


class TestClearIndex:
    @pytest.mark.asyncio
    async def test_clear_index(self, provider, mock_client):
        await provider.clear_index()
        mock_client.delete_collection.assert_called_once_with("test_coll")
        mock_client.create_collection.assert_called_once_with("test_coll", dimension=3)


class TestGetStats:
    @pytest.mark.asyncio
    async def test_get_stats(self, provider):
        stats = await provider.get_stats()
        assert stats["provider"] == "proximadb"
        assert stats["engine"] == "SST"
        assert stats["collection_name"] == "test_coll"
        assert stats["dimension"] == 3
        assert stats["embedding_model"] == "BAAI/bge-small-en-v1.5"


class TestClose:
    @pytest.mark.asyncio
    async def test_close(self, provider, mock_embedding_model):
        await provider.close()
        mock_embedding_model.close.assert_awaited_once()
        assert provider.embedding_model is None
        assert provider._initialized is False
