"""LlamaIndex VectorStore adapter for ProximaDB.

Provides a ``ProximaDBVectorStore`` class that implements LlamaIndex's
``VectorStore`` interface, allowing ProximaDB to be used as a drop-in
vector store for RAG pipelines, agents, and LLM applications.

Requires: ``pip install proximadb-python[llama_index]`` or
            ``pip install llama-index-core proximadb-python``

Example::

    from proximadb_sdk.integrations.llama_index import ProximaDBVectorStore
    from proximadb_sdk import ProximaDBClient

    client = ProximaDBClient(url="http://localhost:5678")
    store = ProximaDBVectorStore(
        client=client,
        collection_name="docs",
    )

    # Add documents
    from llama_index.core import Document
    docs = [Document(text="Hello world", metadata={"source": "test"})]
    store.add(docs)

    # Query
    results = store.query("What is ProximaDB?")
"""

from __future__ import annotations

import uuid
from typing import Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQueryMode,
)

from proximadb_sdk.integrations._records import insert_records, record_payload


class ProximaDBVectorStore(BasePydanticVectorStore):
    """LlamaIndex VectorStore backed by ProximaDB.

    Args:
        client: An existing ``ProximaDBClient`` instance.
        collection_name: Name of the ProximaDB collection.
        text_key: Metadata key used to store the original document text.
            Defaults to ``"text"``.
    """

    stores_text: bool = True
    flat_metadata: bool = True

    _client: Any = PrivateAttr()
    _collection_name: str = PrivateAttr()
    _text_key: str = PrivateAttr()
    _async_client: Any = PrivateAttr(default=None)

    def __init__(
        self,
        client: Any,
        collection_name: str,
        *,
        text_key: str = "text",
        async_client: Any | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._collection_name = collection_name
        self._text_key = text_key
        # Optional native-async client (ProximaDBAsyncUnified). When provided +
        # started, aquery awaits real async I/O over the generated REST
        # transport instead of thread-offloading the sync client. Sync
        # construction/methods are unaffected when omitted.
        self._async_client = async_client

    @property
    def client(self) -> Any:
        return self._client

    def add(
        self,
        nodes: list[BaseNode],
        **add_kwargs: Any,
    ) -> list[str]:
        """Add nodes to ProximaDB.

        Args:
            nodes: List of nodes with embeddings to add.

        Returns:
            List of node IDs that were added.
        """
        ids: list[str] = []
        records: list[dict[str, Any]] = []

        for node in nodes:
            doc_id = node.node_id or str(uuid.uuid4())
            ids.append(doc_id)

            content = node.get_content()
            # Build metadata from node
            metadata = dict(node.metadata) if node.metadata else {}
            metadata[self._text_key] = content

            # Use the embedding from the node if available
            if node.embedding is None:
                raise ValueError(
                    f"Node {doc_id} has no embedding. Please embed before adding."
                )

            records.append(
                record_payload(
                    record_id=doc_id,
                    vector=node.embedding,
                    text=content,
                    metadata=metadata,
                )
            )

        insert_records(self._client, self._collection_name, records)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes from ProximaDB.

        Args:
            ref_doc_id: The document ID to delete.
            **delete_kwargs: Additional delete arguments (not used).
        """
        self._client.delete_vectors(self._collection_name, ids=[ref_doc_id])

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query ProximaDB for similar nodes.

        Args:
            query: The vector store query.
            **kwargs: Additional query arguments.

        Returns:
            Vector store query result with matching nodes.
        """
        search_results = self._client.search(
            self._collection_name,
            vector=self._query_vector(query),
            top_k=query.similarity_top_k,
            metadata_filter=self._query_filter(query),
        )
        return self._results_to_query_result(search_results)

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Asynchronously query ProximaDB for similar nodes.

        When a native-async client was supplied at construction, this awaits
        real async I/O over the generated REST transport; otherwise it offloads
        the synchronous ``search`` onto a worker thread to keep the event loop
        responsive and satisfy LlamaIndex's async query contract.
        """
        if self._async_client is not None:
            search_results = await self._async_client.search(
                self._collection_name,
                vector=self._query_vector(query),
                top_k=query.similarity_top_k,
                metadata_filter=self._query_filter(query),
            )
            return self._results_to_query_result(search_results)

        import asyncio

        search_results = await asyncio.to_thread(
            self._client.search,
            self._collection_name,
            vector=self._query_vector(query),
            top_k=query.similarity_top_k,
            metadata_filter=self._query_filter(query),
        )
        return self._results_to_query_result(search_results)

    def _query_vector(self, query: VectorStoreQuery) -> list[float]:
        """Extract a plain ``list[float]`` query vector from a ``VectorStoreQuery``."""
        embedding = query.query_embedding
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    def _query_filter(self, query: VectorStoreQuery) -> dict[str, Any] | None:
        """Convert a query's ``MetadataFilters`` to the ProximaDB filter format."""
        if query.filters and query.filters.filters:
            return self._convert_filters(query.filters)
        return None

    def _results_to_query_result(self, search_results: Any) -> VectorStoreQueryResult:
        """Convert ProximaDB search results into a ``VectorStoreQueryResult``."""
        nodes: list[TextNode] = []
        similarities: list[float] = []
        ids: list[str] = []

        for result in search_results:
            metadata = dict(result.metadata) if result.metadata else {}
            text_content = result.source or metadata.pop(self._text_key, "")

            nodes.append(
                TextNode(
                    text=text_content,
                    embedding=None,  # Embeddings not returned by default
                    metadata=metadata,
                    id_=result.id,
                )
            )
            similarities.append(result.score)
            ids.append(result.id)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def _convert_filters(self, filters: MetadataFilters) -> dict[str, Any] | None:
        """Convert LlamaIndex MetadataFilters to ProximaDB filter format.

        Args:
            filters: LlamaIndex metadata filters.

        Returns:
            ProximaDB-compatible filter dictionary.
        """
        if not filters.filters:
            return None

        proximadb_filter: dict[str, Any] = {"and": []}

        for f in filters.filters:
            if f.operator == "==" or f.operator == "eq":
                proximadb_filter["and"].append({f.key: f.value})
            elif f.operator == "!=" or f.operator == "ne":
                proximadb_filter["and"].append({f.key: {"$ne": f.value}})
            elif f.operator == ">" or f.operator == "gt":
                proximadb_filter["and"].append({f.key: {"$gt": f.value}})
            elif f.operator == "<" or f.operator == "lt":
                proximadb_filter["and"].append({f.key: {"$lt": f.value}})
            elif f.operator == ">=" or f.operator == "gte":
                proximadb_filter["and"].append({f.key: {"$gte": f.value}})
            elif f.operator == "<=" or f.operator == "lte":
                proximadb_filter["and"].append({f.key: {"$lte": f.value}})
            elif f.operator == "in":
                proximadb_filter["and"].append({f.key: {"$in": f.value}})
            elif f.operator == "nin" or f.operator == "not_in":
                proximadb_filter["and"].append({f.key: {"$nin": f.value}})
            else:
                # Unknown operator, include as-is
                proximadb_filter["and"].append({f.key: f.value})

        return proximadb_filter


class ProximaDBRetriever(BaseRetriever):
    """LlamaIndex Retriever for ProximaDB.

    Args:
        vector_store: The ProximaDBVectorStore to use.
        similarity_top_k: Number of results to return. Defaults to 10.
        filters: Optional metadata filters.
    """

    def __init__(
        self,
        vector_store: ProximaDBVectorStore,
        similarity_top_k: int = 10,
        filters: MetadataFilters | None = None,
    ) -> None:
        super().__init__()
        self._vector_store = vector_store
        self._similarity_top_k = similarity_top_k
        self._filters = filters

    def _retrieve(self, query_bundle: Any) -> list[TextNode]:
        """Retrieve nodes for a query bundle.

        Args:
            query_bundle: The query bundle containing the query embedding.

        Returns:
            List of retrieved nodes.
        """
        query = VectorStoreQuery(
            query_embedding=query_bundle.get_embedding(),
            similarity_top_k=self._similarity_top_k,
            filters=self._filters,
            mode=VectorStoreQueryMode.DEFAULT,
        )

        result = self._vector_store.query(query)
        return list(result.nodes)
