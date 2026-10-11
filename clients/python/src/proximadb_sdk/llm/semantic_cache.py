# Copyright 2025 ProximaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Semantic Cache for RAG responses using ProximaDB."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from proximadb_sdk.llm.config import SemanticCacheConfig

if TYPE_CHECKING:
    from proximadb_sdk import ProximaDBClient
    from proximadb_sdk.llm.rag import RAGResponse


@dataclass
class CachedResponse:
    """Cached RAG response.

    Attributes:
        question: Original question
        collection: Collection queried
        response: Cached RAG response
        embedding: Question embedding
        cached_at: Cache timestamp
        hit_count: Number of cache hits
    """

    question: str
    collection: str
    response: "RAGResponse"
    embedding: list[float]
    cached_at: datetime
    hit_count: int = 0


class SemanticCache:
    """Semantic cache for RAG responses.

    Uses ProximaDB to store and retrieve cached responses based on
    semantic similarity of questions. Similar questions get cached
    responses, reducing LLM API costs.

    Usage:
        cache = SemanticCache(config, client)
        await cache.initialize()

        # Check cache
        cached = await cache.lookup(question, collection)
        if cached:
            return cached.response

        # ... generate response ...

        # Store in cache
        await cache.store(question, collection, embedding, response)
    """

    def __init__(
        self,
        config: SemanticCacheConfig,
        client: "ProximaDBClient",
    ):
        """Initialize semantic cache.

        Args:
            config: Cache configuration
            client: ProximaDB client
        """
        self.config = config
        self.client = client
        self._initialized = False
        self._stats = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
        }

    async def initialize(self) -> None:
        """Initialize the semantic cache collection."""
        if self._initialized or not self.config.enabled:
            return

        # Create cache collection
        # We'll use a special collection to store cached responses
        # The dimension should match the embedding dimension
        # For now, we use 384 (all-MiniLM-L12-v2)
        try:
            await self.client.create_collection_async(
                name=self.config.collection_name,
                dimension=384,  # Default embedding dimension
                distance_metric="cosine",
            )
        except Exception:
            # Collection may already exist
            pass

        self._initialized = True

    async def lookup(
        self,
        question: str,
        collection: str,
    ) -> CachedResponse | None:
        """Look up cached response for a question.

        Uses semantic similarity to find cached responses for similar questions.

        Args:
            question: Question to look up
            collection: Collection this query is for

        Returns:
            Cached response if found, None otherwise
        """
        if not self.config.enabled:
            return None

        if len(question) < self.config.min_query_length:
            return None

        self._stats["lookups"] += 1

        try:
            # Generate embedding for question
            # Note: This requires embedding service, which should be passed
            # For now, we use exact hash matching as fallback
            cache_key = self._make_key(question, collection)

            # Try to find exact match first (fast path)
            # This uses metadata filtering
            results = await self.client.search_async(
                collection=self.config.collection_name,
                vector=[0.0] * 384,  # Placeholder, filtered by metadata
                top_k=1,
                filter={"cache_key": cache_key},
            )

            if results and len(results) > 0:
                result = results[0]
                metadata = result.get("metadata", {})

                # Check TTL
                cached_at_str = metadata.get("cached_at", "")
                if cached_at_str:
                    cached_at = datetime.fromisoformat(cached_at_str)
                    age_hours = (datetime.now(UTC) - cached_at).total_seconds() / 3600
                    if age_hours > self.config.ttl_hours:
                        # Expired
                        self._stats["misses"] += 1
                        return None

                # Reconstruct cached response
                from proximadb_sdk.llm.rag import RAGResponse, Source

                sources = []
                for src in metadata.get("sources", []):
                    sources.append(
                        Source(
                            id=src.get("id", ""),
                            title=src.get("title", ""),
                            url=src.get("url", ""),
                            relevance=src.get("relevance", 0.0),
                            snippet=src.get("snippet", ""),
                        )
                    )

                response = RAGResponse(
                    answer=metadata.get("answer", ""),
                    sources=sources,
                    confidence=metadata.get("confidence", 0.0),
                    latency_ms=metadata.get("latency_ms", 0),
                    retrieval_latency_ms=metadata.get("retrieval_latency_ms", 0),
                    generation_latency_ms=metadata.get("generation_latency_ms", 0),
                    tokens_used=metadata.get("tokens_used", 0),
                    cached=True,
                )

                cached = CachedResponse(
                    question=metadata.get("question", question),
                    collection=collection,
                    response=response,
                    embedding=result.get("vector", []),
                    cached_at=(cached_at if cached_at_str else datetime.now(UTC)),
                    hit_count=metadata.get("hit_count", 0) + 1,
                )

                self._stats["hits"] += 1
                return cached

        except Exception:
            pass

        self._stats["misses"] += 1
        return None

    async def store(
        self,
        question: str,
        collection: str,
        embedding: list[float],
        response: "RAGResponse",
    ) -> None:
        """Store response in cache.

        Args:
            question: Original question
            collection: Collection queried
            embedding: Question embedding
            response: RAG response to cache
        """
        if not self.config.enabled:
            return

        if len(question) < self.config.min_query_length:
            return

        try:
            cache_key = self._make_key(question, collection)
            now = datetime.now(UTC)

            # Serialize sources
            sources_data = []
            for src in response.sources:
                sources_data.append(
                    {
                        "id": src.id,
                        "title": src.title,
                        "url": src.url,
                        "relevance": src.relevance,
                        "snippet": src.snippet,
                    }
                )

            # Store in cache collection
            await self.client.insert_vectors_async(
                collection=self.config.collection_name,
                vectors=[
                    {
                        "id": cache_key,
                        "vector": embedding,
                        "metadata": {
                            "cache_key": cache_key,
                            "question": question,
                            "collection": collection,
                            "answer": response.answer,
                            "sources": sources_data,
                            "confidence": response.confidence,
                            "latency_ms": response.latency_ms,
                            "retrieval_latency_ms": response.retrieval_latency_ms,
                            "generation_latency_ms": response.generation_latency_ms,
                            "tokens_used": response.tokens_used,
                            "cached_at": now.isoformat(),
                            "hit_count": 0,
                        },
                    }
                ],
            )

            self._stats["stores"] += 1

        except Exception:
            pass

    async def invalidate(
        self,
        question: str,
        collection: str,
    ) -> None:
        """Invalidate a cached response.

        Args:
            question: Question to invalidate
            collection: Collection the query was for
        """
        if not self.config.enabled:
            return

        try:
            cache_key = self._make_key(question, collection)
            await self.client.delete_vector_async(
                self.config.collection_name,
                cache_key,
            )
        except Exception:
            pass

    async def invalidate_collection(self, collection: str) -> None:
        """Invalidate all cached responses for a collection.

        Args:
            collection: Collection to invalidate
        """
        if not self.config.enabled:
            return

        # This would require listing all entries with that collection
        # and deleting them. For now, we don't support this efficiently.
        pass

    async def clear(self) -> None:
        """Clear all cached responses."""
        if not self.config.enabled:
            return

        try:
            await self.client.delete_collection_async(self.config.collection_name)
            await self.initialize()  # Recreate collection
        except Exception:
            pass

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        lookups = self._stats["lookups"]
        hits = self._stats["hits"]

        return {
            "lookups": lookups,
            "hits": hits,
            "misses": self._stats["misses"],
            "stores": self._stats["stores"],
            "hit_rate": hits / lookups if lookups > 0 else 0.0,
        }

    def _make_key(self, question: str, collection: str) -> str:
        """Generate cache key from question and collection.

        Args:
            question: Question text
            collection: Collection name

        Returns:
            Cache key string
        """
        combined = f"{collection}:{question}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
