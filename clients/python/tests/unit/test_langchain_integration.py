"""Unit tests for the LangChain VectorStore adapter.

These tests mock the ProximaDB client and LangChain embeddings so they can
run without a live server or real embedding model.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Skip entire module if langchain-core is not installed
langchain_core = pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from proximadb_sdk.integrations.langchain import ProximaDBVectorStore  # noqa: E402
from proximadb_sdk.models import SearchResult  # noqa: E402


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] * 3 for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))] * 3


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.insert_records = MagicMock()
    client.insert_vectors = MagicMock()
    client.delete_vectors = MagicMock()
    client.search = MagicMock(return_value=[])
    return client


@pytest.fixture
def store(mock_client):
    return ProximaDBVectorStore(
        client=mock_client,
        collection_name="test_collection",
        embedding=FakeEmbeddings(),
    )


class TestAddTexts:
    def test_add_texts_basic(self, store, mock_client):
        ids = store.add_texts(["hello", "world"])
        assert len(ids) == 2
        mock_client.insert_records.assert_called_once()
        records = mock_client.insert_records.call_args.args[1]
        assert len(records) == 2
        # TD-REC-1: text rides in text_fields, not a top-level source.
        assert records[0]["text_fields"][0]["content"] == "hello"
        assert records[1]["text_fields"][0]["content"] == "world"
        assert records[0]["text_fields"] == [{"name": "text", "content": "hello"}]

    def test_add_texts_with_metadata(self, store, mock_client):
        metadatas = [{"key": "val1"}, {"key": "val2"}]
        ids = store.add_texts(["a", "b"], metadatas=metadatas)
        assert len(ids) == 2
        records = mock_client.insert_records.call_args.args[1]
        assert records[0]["props"]["key"] == "val1"

    def test_add_texts_with_ids(self, store, mock_client):
        ids = store.add_texts(["x"], ids=["custom_id"])
        assert ids == ["custom_id"]
        records = mock_client.insert_records.call_args.args[1]
        assert records[0]["id"] == "custom_id"


class TestDelete:
    def test_delete(self, store, mock_client):
        result = store.delete(ids=["id1", "id2"])
        assert result is True
        mock_client.delete_vectors.assert_called_once_with(
            "test_collection", ["id1", "id2"]
        )

    def test_delete_no_ids(self, store, mock_client):
        result = store.delete(ids=None)
        assert result is False
        mock_client.delete_vectors.assert_not_called()


class TestSimilaritySearch:
    def test_similarity_search(self, store, mock_client):
        mock_client.search.return_value = [
            SearchResult(id="r1", score=0.95, source="hello", metadata={"k": "v"}),
            SearchResult(id="r2", score=0.80, source="world", metadata={}),
        ]
        docs = store.similarity_search("test query", k=2)
        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "hello"
        assert docs[0].metadata == {"k": "v"}
        mock_client.search.assert_called_once()

    def test_similarity_search_with_score(self, store, mock_client):
        mock_client.search.return_value = [
            SearchResult(id="r1", score=0.95, source="hello", metadata={}),
        ]
        results = store.similarity_search_with_score("test", k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert doc.page_content == "hello"
        assert score == 0.95

    def test_similarity_search_by_vector(self, store, mock_client):
        mock_client.search.return_value = [
            SearchResult(id="r1", score=0.9, source="content", metadata={}),
        ]
        docs = store.similarity_search_by_vector([1.0, 2.0, 3.0], k=1)
        assert len(docs) == 1
        assert docs[0].page_content == "content"
        # Verify we called search with the raw vector, not re-embedding
        call_kwargs = mock_client.search.call_args
        assert call_kwargs.kwargs.get("vector") == [1.0, 2.0, 3.0]

    def test_text_key_fallback(self, store, mock_client):
        """When source is None, fall back to text_key in metadata."""
        mock_client.search.return_value = [
            SearchResult(
                id="r1", score=0.9, source=None, metadata={"text": "from metadata"}
            ),
        ]
        docs = store.similarity_search("q", k=1)
        assert docs[0].page_content == "from metadata"


class TestClassMethods:
    def test_from_texts(self, mock_client):
        store = ProximaDBVectorStore.from_texts(
            texts=["a", "b"],
            embedding=FakeEmbeddings(),
            client=mock_client,
            collection_name="test_ft",
        )
        assert isinstance(store, ProximaDBVectorStore)
        mock_client.insert_records.assert_called_once()

    def test_from_texts_requires_client(self):
        with pytest.raises(ValueError, match="requires a 'client'"):
            ProximaDBVectorStore.from_texts(
                texts=["a"],
                embedding=FakeEmbeddings(),
            )

    def test_from_documents(self, mock_client):
        docs = [
            Document(page_content="doc1", metadata={"src": "test"}),
            Document(page_content="doc2", metadata={}),
        ]
        store = ProximaDBVectorStore.from_documents(
            documents=docs,
            embedding=FakeEmbeddings(),
            client=mock_client,
            collection_name="test_fd",
        )
        assert isinstance(store, ProximaDBVectorStore)
        records = mock_client.insert_records.call_args.args[1]
        # TD-REC-1: text rides in text_fields, not a top-level source.
        assert records[0]["text_fields"][0]["content"] == "doc1"
        assert records[0]["props"]["src"] == "test"


class TestNativeAsyncClient:
    """The async read path uses a native-async client when one is supplied."""

    @pytest.mark.asyncio
    async def test_async_search_uses_native_async_client(self, mock_client):
        from unittest.mock import AsyncMock

        async_client = MagicMock()
        async_client.search = AsyncMock(
            return_value=[
                SearchResult(id="a", score=0.9, metadata={"text": "hello"}),
            ]
        )
        store = ProximaDBVectorStore(
            client=mock_client,
            collection_name="test_collection",
            embedding=FakeEmbeddings(),
            async_client=async_client,
        )
        out = await store.asimilarity_search_by_vector_with_score([0.1, 0.2], k=3)
        # Native async client awaited; sync client.search NOT used.
        async_client.search.assert_awaited_once()
        mock_client.search.assert_not_called()
        assert len(out) == 1
        doc, score = out[0]
        assert isinstance(doc, Document)
        assert doc.page_content == "hello"
        assert score == 0.9

    @pytest.mark.asyncio
    async def test_async_search_falls_back_to_thread_without_async_client(
        self, store, mock_client
    ):
        # No async_client supplied -> offloads the sync client on a worker thread.
        out = await store.asimilarity_search_by_vector_with_score([0.1, 0.2], k=3)
        mock_client.search.assert_called_once()
        assert out == []
