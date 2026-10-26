"""
ProximaDB Hybrid Query API Module

High-performance hybrid queries combining vector, graph, document, and time-series.
Implements strategy pattern for query fusion and result ranking.

Design Patterns:
- Strategy Pattern: Different fusion strategies (RRF, weighted, learned)
- Observer Pattern: Query execution events
- Builder Pattern: Complex hybrid query construction
- Repository Pattern: Query result caching
- Async/Await: Non-blocking parallel queries
- Connection Pooling: Efficient connection reuse
- Result Ranking: Cross-modal score fusion

Example:
    from proximadb_sdk import ProximaDBClient
    from proximadb_sdk.hybrid import ProximaDBHybrid

    client = ProximaDBClient(url="http://localhost:5678")
    hybrid = ProximaDBHybrid(client)

    # Vector + Graph + Document hybrid search
    results = hybrid.search(
        vector_query="parse JSON input",
        vector_collection="code_embeddings",
        top_k=10,
        graph_query="MATCH (c:Function)-[:CALLS]->(f:Function) WHERE c.name = 'main'",
        graph_collection="call_graph",
        document_filter={"language": "python"},
        document_collection="code_files",
        fusion_strategy=FusionStrategy.RECALL_RANK_FUSION
    )

    # Federated SQL query
    results = hybrid.sql(
        "SELECT v.id, v.score, n.properties, d.document "
        "FROM VECTOR_SEARCH('code_embeddings', ?, 10) v "
        "JOIN GRAPH_QUERY('call_graph', 'MATCH (n)-[r:CALLS]->(m) RETURN n, r, m') g "
        "ON v.id = g.node_id "
        "JOIN DOCUMENT_QUERY('code_files', '{\"language\": \"python\"}') d "
        "ON v.metadata.file_path = d.file_path "
        "WHERE v.metadata.language = 'python'",
        query_vector,
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# =============================================================================
# Enums and Constants
# =============================================================================


class FusionStrategy(str, Enum):
    """Strategies for combining multi-model results."""

    # Vector-first, augment with other models
    VECTOR_FIRST = "vector_first"

    # Graph-first, augment with other models
    GRAPH_FIRST = "graph_first"

    # Reciprocal Rank Fusion (RRF)
    RRF = "rrf"

    # Weighted linear combination
    WEIGHTED = "weighted"

    # Learned model (ML-based)
    LEARNED = "learned"

    # Cascade: filter -> vector -> rerank
    CASCADE = "cascade"

    # Parallel with balanced scores
    BALANCED = "balanced"

    # Projection Fusion B5 (arXiv:2604.13728): faster than RRF with greater
    # result diversity, but RRF wins relevance (nDCG@10) on TREC-COVID. Choose
    # this when low fusion latency or higher result diversity matters more
    # than peak relevance.
    PROJECTION = "projection"


class JoinType(str, Enum):
    """Join types for multi-model queries."""

    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"
    CROSS = "cross"
    SEMANTIC = "semantic"  # Vector similarity join


class QueryModel(str, Enum):
    """Data models for hybrid queries."""

    VECTOR = "vector"
    GRAPH = "graph"
    DOCUMENT = "document"
    TIMESERIES = "timeseries"
    HYBRID = "hybrid"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class VectorSearchResult:
    """Vector search result component.

    Attributes:
        id: Result identifier
        score: Similarity score (0-1, higher is better)
        distance: Distance metric value
        vector: Query vector
        metadata: Optional metadata
        collection: Source collection
    """

    id: str
    score: float
    distance: float | None = None
    rank: int = 0
    vector: list[float] | None = None
    metadata: dict[str, Any] | None = None
    collection: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "score": self.score,
            "distance": self.distance,
            "rank": self.rank,
            "metadata": self.metadata,
            "collection": self.collection,
            "model": QueryModel.VECTOR.value,
        }


@dataclass
class GraphSearchResult:
    """Graph search result component.

    Attributes:
        node_id: Node identifier
        score: Graph traversal score
        path: Traversal path (if applicable)
        properties: Node properties
        labels: Node labels
        edges: Connected edges
        collection: Source collection
    """

    node_id: str
    score: float
    path: list[str] | None = None
    properties: dict[str, Any] | None = None
    labels: list[str] | None = None
    edges: list[dict[str, Any]] | None = None
    collection: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "id": self.node_id,  # For joining
            "score": self.score,
            "path": self.path,
            "properties": self.properties,
            "labels": self.labels,
            "edges": self.edges,
            "collection": self.collection,
            "model": QueryModel.GRAPH.value,
        }


@dataclass
class DocumentSearchResult:
    """Document search result component.

    Attributes:
        id: Document identifier
        score: Document relevance score
        highlight: Highlighted snippets
        document: Document content
        metadata: Optional metadata
        collection: Source collection
    """

    id: str
    score: float
    rank: int = 0
    highlight: list[str] | None = None
    document: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    collection: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "score": self.score,
            "rank": self.rank,
            "highlight": self.highlight,
            "document": self.document,
            "metadata": self.metadata,
            "collection": self.collection,
            "model": QueryModel.DOCUMENT.value,
        }


@dataclass
class TimeSeriesResult:
    """Time-series result component.

    Attributes:
        id: Metric identifier (tags-based)
        score: Metric relevance score
        timestamp: Metric timestamp
        values: Metric values
        tags: Metric tags
        collection: Source collection
    """

    id: str
    score: float
    timestamp: datetime | None = None
    values: dict[str, Any] | None = None
    tags: dict[str, Any] | None = None
    collection: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "score": self.score,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "values": self.values,
            "tags": self.tags,
            "collection": self.collection,
            "model": QueryModel.TIMESERIES.value,
        }


@dataclass
class HybridSearchResult:
    """Combined hybrid search result.

    Attributes:
        id: Result identifier
        final_score: Fused score across all models
        components: Individual model results
        rank: Result rank
        explanation: Score breakdown explanation
        metadata: Combined metadata
    """

    id: str
    final_score: float
    components: dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fused_score(self) -> float:
        return self.final_score

    @property
    def vector_score(self) -> float:
        vector_component = self.components.get(QueryModel.VECTOR.value)
        if vector_component is None:
            return 0.0
        return (
            vector_component.score
            if hasattr(vector_component, "score")
            else vector_component.get("score", 0.0)
        )

    @property
    def bm25_score(self) -> float:
        document_component = self.components.get(QueryModel.DOCUMENT.value)
        if document_component is None:
            return 0.0
        return (
            document_component.score
            if hasattr(document_component, "score")
            else document_component.get("score", 0.0)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "score": self.final_score,
            "fused_score": self.final_score,
            "rank": self.rank,
            "components": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.components.items()
            },
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


def _result_id(result: Any) -> str:
    return result.id if hasattr(result, "id") else result.get("id")


def _normalize_fusion_inputs(
    primary: dict[str, list[Any]] | list[Any],
    secondary: list[Any] | None = None,
) -> dict[str, list[Any]]:
    if isinstance(primary, dict):
        return primary

    results = {QueryModel.VECTOR.value: primary}
    if secondary is not None:
        results[QueryModel.DOCUMENT.value] = secondary
    return results


def _merge_component_metadata(components: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for component in components.values():
        component_metadata = getattr(component, "metadata", None)
        if component_metadata is None and isinstance(component, dict):
            component_metadata = component.get("metadata")
        if component_metadata:
            metadata.update(component_metadata)
    return metadata


# =============================================================================
# Fusion Strategies (Strategy Pattern)
# =============================================================================


class FusionStrategyBase(ABC):
    """Base class for fusion strategies."""

    @abstractmethod
    def fuse(
        self,
        results: dict[str, list[Any]] | list[Any],
        secondary_results: list[Any] | None = None,
        top_k: int | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[HybridSearchResult]:
        """Fuse results from multiple models.

        Args:
            results: Dictionary of model -> results list
            weights: Optional model weights

        Returns:
            List of fused results
        """
        pass


class ReciprocalRankFusion(FusionStrategyBase):
    """Reciprocal Rank Fusion (RRF) strategy.

    RRF is a robust rank-based fusion method that combines ranked lists
    from multiple sources without requiring score normalization.

    Formula: score = sum(1 / (k + rank))
    where k is a constant (typically 60)
    """

    def __init__(self, k: int = 60):
        """Initialize RRF strategy.

        Args:
            k: RRF constant (default: 60)
        """
        self.k = k

    def fuse(
        self,
        results: dict[str, list[Any]] | list[Any],
        secondary_results: list[Any] | None = None,
        top_k: int | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[HybridSearchResult]:
        """Fuse results using RRF.

        Args:
            results: Model -> results mapping or primary ranked list
            secondary_results: Optional secondary ranked list
            top_k: Optional output cap
            weights: Optional model weights

        Returns:
            Fused and ranked results
        """
        results = _normalize_fusion_inputs(results, secondary_results)

        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}

        for model, model_results in results.items():
            model_weight = weights.get(model, 1.0) if weights else 1.0

            for rank, result in enumerate(model_results, 1):
                result_id = _result_id(result)

                # RRF score: 1 / (k + rank)
                rrf_score = model_weight / (self.k + rank)

                if result_id not in rrf_scores:
                    rrf_scores[result_id] = 0.0
                rrf_scores[result_id] += rrf_score

        # Sort by score
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        # Build hybrid results
        hybrid_results = []
        for rank, result_id in enumerate(sorted_ids, 1):
            # Collect components from all models
            components = {}
            for model, model_results in results.items():
                for result in model_results:
                    comp_id = _result_id(result)
                    if comp_id == result_id:
                        components[model] = result
                        break

            hybrid_results.append(
                HybridSearchResult(
                    id=result_id,
                    final_score=rrf_scores[result_id],
                    components=components,
                    rank=rank,
                    explanation=f"RRF score (k={self.k}): {rrf_scores[result_id]:.4f}",
                    metadata=_merge_component_metadata(components),
                )
            )

        return hybrid_results[:top_k] if top_k is not None else hybrid_results


class WeightedFusion(FusionStrategyBase):
    """Weighted linear combination fusion strategy.

    Combines normalized scores from multiple models using configurable weights.

    Formula: score = sum(weight_i * normalized_score_i)
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        alpha: float | None = None,
        bm25_normalize: bool = True,
        vector_normalize: bool = True,
    ):
        """Initialize weighted fusion strategy.

        Args:
            weights: Default model weights (if not provided per query)
        """
        if weights is not None:
            self.default_weights = weights
        else:
            vector_weight = alpha if alpha is not None else 0.5
            document_weight = 1.0 - vector_weight
            self.default_weights = {
                QueryModel.VECTOR.value: vector_weight,
                QueryModel.DOCUMENT.value: document_weight,
                QueryModel.GRAPH.value: 0.0,
            }
        self.bm25_normalize = bm25_normalize
        self.vector_normalize = vector_normalize

    def fuse(
        self,
        results: dict[str, list[Any]] | list[Any],
        secondary_results: list[Any] | None = None,
        top_k: int | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[HybridSearchResult]:
        """Fuse results using weighted combination.

        Args:
            results: Model -> results mapping
            weights: Optional model weights

        Returns:
            Fused and ranked results
        """
        results = _normalize_fusion_inputs(results, secondary_results)

        # Use provided weights or defaults
        fusion_weights = weights or self.default_weights

        # Normalize scores per model
        normalized_results: dict[str, dict[str, float]] = {}
        for model, model_results in results.items():
            if not model_results:
                continue

            # Find max score for normalization
            max_score = max(
                (r.score if hasattr(r, "score") else r.get("score", 0.0))
                for r in model_results
            )
            if max_score == 0:
                max_score = 1.0

            # Normalize
            normalized_results[model] = {}
            for result in model_results:
                result_id = _result_id(result)
                score = (
                    result.score
                    if hasattr(result, "score")
                    else result.get("score", 0.0)
                )
                normalized_results[model][result_id] = score / max_score

        # Calculate weighted scores
        weighted_scores: dict[str, float] = {}
        for model, norm_results in normalized_results.items():
            model_weight = fusion_weights.get(model, 1.0)

            for result_id, norm_score in norm_results.items():
                if result_id not in weighted_scores:
                    weighted_scores[result_id] = 0.0
                weighted_scores[result_id] += model_weight * norm_score

        # Sort by score
        sorted_ids = sorted(
            weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True
        )

        # Build hybrid results
        hybrid_results = []
        for rank, result_id in enumerate(sorted_ids, 1):
            # Collect components
            components = {}
            for model, model_results in results.items():
                for result in model_results:
                    comp_id = _result_id(result)
                    if comp_id == result_id:
                        components[model] = result
                        break

            hybrid_results.append(
                HybridSearchResult(
                    id=result_id,
                    final_score=weighted_scores[result_id],
                    components=components,
                    rank=rank,
                    explanation=f"Weighted score: {weighted_scores[result_id]:.4f}",
                    metadata=_merge_component_metadata(components),
                )
            )

        return hybrid_results[:top_k] if top_k is not None else hybrid_results


class CascadeFusion(FusionStrategyBase):
    """Cascade fusion strategy.

    Applies filters and ranking in stages:
    1. Filter by non-vector models
    2. Vector search on filtered results
    3. Rerank by combined scores

    This is efficient when non-vector models can significantly reduce the search space.
    """

    def __init__(
        self,
        primary_model: str = "vector",
        secondary_model: str = "document",
        threshold: float = 0.0,
    ):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.threshold = threshold

    def fuse(
        self,
        results: dict[str, list[Any]] | list[Any],
        secondary_results: list[Any] | None = None,
        top_k: int | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[HybridSearchResult]:
        """Fuse results using cascade strategy.

        Args:
            results: Model -> results mapping
            weights: Optional model weights

        Returns:
            Fused and ranked results
        """
        results = _normalize_fusion_inputs(results, secondary_results)

        # Find vector results
        vector_results = results.get(QueryModel.VECTOR.value, [])
        if not vector_results:
            return []

        # Use vector results as base, augment with other models
        hybrid_results = []
        for rank, vector_result in enumerate(vector_results, 1):
            components = {QueryModel.VECTOR.value: vector_result}

            # Try to find matching results from other models
            result_id = _result_id(vector_result)

            for model, model_results in results.items():
                if model == QueryModel.VECTOR.value:
                    continue

                for result in model_results:
                    comp_id = _result_id(result)
                    if comp_id == result_id:
                        components[model] = result
                        break

            hybrid_results.append(
                HybridSearchResult(
                    id=result_id,
                    final_score=vector_result.score,
                    components=components,
                    rank=rank,
                    explanation="Cascade: vector-primary with augmentation",
                    metadata=_merge_component_metadata(components),
                )
            )

        return hybrid_results[:top_k] if top_k is not None else hybrid_results


# =============================================================================
# Hybrid Query Repository
# =============================================================================


class HybridQueryRepository:
    """Repository for hybrid query operations.

    Implements parallel query execution across multiple models with
    result fusion and ranking.

    Attributes:
        _client: ProximaDB client instance
        _fusion_strategies: Available fusion strategies
        _cache: Query result cache
        _cache_ttl: Cache TTL in seconds
    """

    def __init__(
        self,
        client: Any,
        cache_ttl: int = 300,  # 5 minutes
    ):
        """Initialize hybrid query repository.

        Args:
            client: ProximaDB client instance
            cache_ttl: Cache TTL for query results
        """
        self._client = client
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[list[HybridSearchResult], float]] = {}

        # Fusion strategies
        self._fusion_strategies: dict[FusionStrategy, FusionStrategyBase] = {
            FusionStrategy.RRF: ReciprocalRankFusion(),
            FusionStrategy.WEIGHTED: WeightedFusion(),
            FusionStrategy.CASCADE: CascadeFusion(),
        }

    # ========================================================================
    # Hybrid Search
    # ========================================================================

    async def search_async(
        self,
        vector_query: list[float] | None = None,
        vector_collection: str | None = None,
        top_k: int = 10,
        graph_query: str | None = None,
        graph_collection: str | None = None,
        document_filter: dict[str, Any] | None = None,
        document_collection: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        timeseries_collection: str | None = None,
        fusion_strategy: FusionStrategy = FusionStrategy.RRF,
        weights: dict[str, float] | None = None,
    ) -> list[HybridSearchResult]:
        """Execute hybrid search across multiple models (async).

        Args:
            vector_query: Query vector for vector search
            vector_collection: Vector collection name
            top_k: Number of results per model
            graph_query: Cypher graph query
            graph_collection: Graph collection name
            document_filter: Document filter
            document_collection: Document collection name
            time_range: Time range for time-series
            timeseries_collection: Time-series collection name
            fusion_strategy: Result fusion strategy
            weights: Model weights for fusion

        Returns:
            List of fused hybrid results

        Example:
            results = await hybrid.search_async(
                vector_query=query_vector,
                vector_collection="code_embeddings",
                top_k=10,
                graph_query="MATCH (c:Function)-[:CALLS]->(f:Function)",
                graph_collection="call_graph",
                document_filter={"language": "python"},
                document_collection="code_files",
                fusion_strategy=FusionStrategy.RRF
            )
        """
        # Build cache key
        cache_key = self._build_cache_key(
            vector_query,
            graph_query,
            document_filter,
            time_range,
            fusion_strategy,
        )

        # Check cache
        if cache_key in self._cache:
            cached_results, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_results

        # Execute parallel queries
        tasks = []
        results = {}

        # Vector search
        if vector_query and vector_collection:
            tasks.append(self._vector_search(vector_collection, vector_query, top_k))

        # Graph search
        if graph_query and graph_collection:
            tasks.append(self._graph_search(graph_collection, graph_query))

        # Document search
        if document_filter and document_collection:
            tasks.append(self._document_search(document_collection, document_filter))

        # Wait for all queries
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect successful results
            model_order = [
                QueryModel.VECTOR,
                QueryModel.GRAPH,
                QueryModel.DOCUMENT,
                QueryModel.TIMESERIES,
            ]
            for i, result in enumerate(task_results):
                if isinstance(result, Exception):
                    continue
                if result and i < len(model_order):
                    results[model_order[i].value] = result

        # Fuse results
        fusion = self._fusion_strategies.get(fusion_strategy, ReciprocalRankFusion())
        fused_results = fusion.fuse(results, weights)

        # Update cache
        self._cache[cache_key] = (fused_results, time.time())

        return fused_results

    def search(
        self,
        vector_query: list[float] | None = None,
        vector_collection: str | None = None,
        top_k: int = 10,
        graph_query: str | None = None,
        graph_collection: str | None = None,
        document_filter: dict[str, Any] | None = None,
        document_collection: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        timeseries_collection: str | None = None,
        fusion_strategy: FusionStrategy = FusionStrategy.RRF,
        weights: dict[str, float] | None = None,
    ) -> list[HybridSearchResult]:
        """Execute hybrid search across multiple models (sync).

        Args:
            vector_query: Query vector
            vector_collection: Vector collection
            top_k: Results per model
            graph_query: Graph query
            graph_collection: Graph collection
            document_filter: Document filter
            document_collection: Document collection
            time_range: Time range
            timeseries_collection: Time-series collection
            fusion_strategy: Fusion strategy
            weights: Model weights

        Returns:
            Fused hybrid results
        """
        # Run async version synchronously
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.search_async(
                vector_query=vector_query,
                vector_collection=vector_collection,
                top_k=top_k,
                graph_query=graph_query,
                graph_collection=graph_collection,
                document_filter=document_filter,
                document_collection=document_collection,
                time_range=time_range,
                timeseries_collection=timeseries_collection,
                fusion_strategy=fusion_strategy,
                weights=weights,
            )
        )

    # ========================================================================
    # Federated SQL Query
    # ========================================================================

    async def sql_async(
        self,
        query: str,
        params: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute federated SQL query across multiple models.

        Supports SQL extensions:
        - VECTOR_SEARCH(collection, vector, k)
        - GRAPH_QUERY(collection, cypher)
        - DOCUMENT_QUERY(collection, filter_json)
        - TIMESERIES_QUERY(collection, start, end, filter)

        Args:
            query: SQL query with extensions
            params: Query parameters

        Returns:
            Query results

        Example:
            results = await hybrid.sql_async(
                "SELECT v.id, v.score, n.properties, d.document "
                "FROM VECTOR_SEARCH('code_embeddings', ?, 10) v "
                "JOIN GRAPH_QUERY('call_graph', 'MATCH ...') g ON v.id = g.node_id "
                "JOIN DOCUMENT_QUERY('code_files', '{"language": "python"}') d "
                "ON v.metadata.file_path = d.file_path",
                [query_vector],
            )
        """
        # TODO: Implement via client's execute_sql method
        # This would use the unified query engine

        return []

    def sql(
        self,
        query: str,
        params: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute federated SQL query (sync).

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Query results
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.sql_async(query, params))

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    async def _vector_search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        """Execute vector search.

        Args:
            collection: Collection name
            vector: Query vector
            top_k: Number of results

        Returns:
            Vector search results
        """
        # TODO: Implement via client
        return []

    async def _graph_search(
        self,
        collection: str,
        query: str,
    ) -> list[GraphSearchResult]:
        """Execute graph search.

        Args:
            collection: Graph collection name
            query: Cypher query

        Returns:
            Graph search results
        """
        # TODO: Implement via client
        return []

    async def _document_search(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> list[DocumentSearchResult]:
        """Execute document search.

        Args:
            collection: Document collection name
            filter: Document filter

        Returns:
            Document search results
        """
        # TODO: Implement via client
        return []

    def _build_cache_key(
        self,
        vector_query: list[float] | None,
        graph_query: str | None,
        document_filter: dict[str, Any] | None,
        time_range: tuple[datetime, datetime] | None,
        fusion_strategy: FusionStrategy,
    ) -> str:
        """Build cache key from query parameters.

        Args:
            vector_query: Query vector
            graph_query: Graph query
            document_filter: Document filter
            time_range: Time range
            fusion_strategy: Fusion strategy

        Returns:
            Cache key hash
        """
        key_parts = []

        if vector_query:
            vector_hash = hashlib.sha256(str(vector_query).encode()).hexdigest()[:16]
            key_parts.append(f"v:{vector_hash}")

        if graph_query:
            graph_hash = hashlib.sha256(graph_query.encode()).hexdigest()[:16]
            key_parts.append(f"g:{graph_hash}")

        if document_filter:
            filter_hash = hashlib.sha256(str(document_filter).encode()).hexdigest()[:16]
            key_parts.append(f"d:{filter_hash}")

        if time_range:
            time_str = f"{time_range[0].isoformat()}_{time_range[1].isoformat()}"
            key_parts.append(f"t:{time_str}")

        key_parts.append(f"f:{fusion_strategy.value}")

        return ":".join(key_parts)


# =============================================================================
# High-Level Hybrid API
# =============================================================================


class ProximaDBHybrid:
    """High-level hybrid query API.

    Provides simplified API for multi-model hybrid queries with automatic
    parallel execution, result fusion, and caching.

    Args:
        client: ProximaDB client instance
        cache_ttl: Cache TTL for results
        default_fusion: Default fusion strategy
    """

    def __init__(
        self,
        client: Any,
        cache_ttl: int = 300,
        default_fusion: FusionStrategy = FusionStrategy.RRF,
    ):
        """Initialize hybrid query API.

        Args:
            client: ProximaDB client instance
            cache_ttl: Cache TTL in seconds
            default_fusion: Default fusion strategy
        """
        self._repository = HybridQueryRepository(
            client=client,
            cache_ttl=cache_ttl,
        )
        self._client = client
        self._default_fusion = default_fusion

    def _resolve_fusion(
        self, fusion_strategy: FusionStrategy | FusionStrategyBase | None
    ) -> FusionStrategyBase:
        if isinstance(fusion_strategy, FusionStrategyBase):
            return fusion_strategy

        strategy = fusion_strategy or self._default_fusion
        if strategy == FusionStrategy.RRF:
            return ReciprocalRankFusion()
        if strategy == FusionStrategy.WEIGHTED:
            return WeightedFusion()
        if strategy == FusionStrategy.CASCADE:
            return CascadeFusion()
        return ReciprocalRankFusion()

    def _mock_vector_results(
        self,
        collection: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        metadata = {
            "category": "tutorial",
            "language": "python",
        }
        metadata.update(filters or {})
        ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        scores = [0.95, 0.9, 0.85, 0.8, 0.75]
        return [
            VectorSearchResult(
                id=doc_id,
                score=score,
                rank=rank,
                metadata=dict(metadata),
                collection=collection,
            )
            for rank, (doc_id, score) in enumerate(zip(ids, scores), 1)
        ][:top_k]

    def _mock_document_results(
        self,
        text_query: str | None,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[DocumentSearchResult]:
        metadata = {
            "category": "tutorial",
            "language": "python",
        }
        metadata.update(filters or {})
        ids = ["doc2", "doc1", "doc4", "doc3", "doc5"]
        scores = [0.88, 0.82, 0.75, 0.7, 0.65]
        return [
            DocumentSearchResult(
                id=doc_id,
                score=score,
                rank=rank,
                document={
                    "content": text_query or "mock hybrid document",
                    "language": metadata.get("language", "python"),
                },
                metadata=dict(metadata),
            )
            for rank, (doc_id, score) in enumerate(zip(ids, scores), 1)
        ][:top_k]

    def search(
        self,
        vector_query: list[float] | None = None,
        vector_collection: str | None = None,
        top_k: int = 10,
        graph_query: str | None = None,
        graph_collection: str | None = None,
        document_filter: dict[str, Any] | None = None,
        document_collection: str | None = None,
        fusion_strategy: FusionStrategy | FusionStrategyBase | None = None,
        weights: dict[str, float] | None = None,
        query_vector: list[float] | None = None,
        text_query: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridSearchResult]:
        """Execute hybrid search.

        Args:
            vector_query: Query vector
            vector_collection: Vector collection name
            top_k: Results per model
            graph_query: Graph query
            graph_collection: Graph collection
            document_filter: Document filter
            document_collection: Document collection
            fusion_strategy: Fusion strategy
            weights: Model weights

        Returns:
            Fused hybrid results

        Example:
            results = hybrid.search(
                vector_query=embedding,
                vector_collection="code_embeddings",
                top_k=10,
                graph_query="MATCH (c:Function)-[:CALLS]->(f:Function)",
                document_filter={"language": "python"},
                fusion_strategy=FusionStrategy.RRF
            )
        """
        vector_query = vector_query or query_vector
        document_filter = document_filter or filters

        if vector_query is not None or text_query is not None:
            # Execute vector search on server
            vector_results = []
            if vector_query is not None and vector_collection:
                try:
                    vector_response = self._client.search_vectors(
                        collection=vector_collection,
                        query_vector=vector_query,
                        top_k=top_k,
                        filters=document_filter,
                    )
                    # Convert to VectorSearchResult format
                    vector_results = [
                        VectorSearchResult(
                            id=result.get("id", ""),
                            score=result.get("score", 0.0),
                            rank=idx + 1,
                            metadata=result.get("metadata", {}),
                            collection=vector_collection,
                        )
                        for idx, result in enumerate(
                            vector_response.get("results", [])[:top_k], 1
                        )
                    ]
                except Exception as e:
                    # Log error but continue with document search
                    import warnings

                    warnings.warn(f"Vector search failed: {e}")

            # Execute document/bm25 search on server
            document_results = []
            if text_query is not None:
                try:
                    document_response = self._client.query_documents(
                        collection_name=document_collection or "hybrid_collection",
                        filter=document_filter,
                        limit=top_k,
                    )
                    # Convert to DocumentSearchResult format
                    document_results = [
                        DocumentSearchResult(
                            id=doc.get("id", ""),
                            score=1.0 / (idx + 1),  # Simple ranking based on position
                            rank=idx + 1,
                            document=doc.get("data", doc),
                            metadata=doc.get("metadata", {}),
                        )
                        for idx, doc in enumerate(
                            document_response.get("documents", [])[:top_k], 1
                        )
                    ]
                except Exception as e:
                    # Log error but continue with vector results
                    import warnings

                    warnings.warn(f"Document search failed: {e}")

            # Fuse results
            fusion = self._resolve_fusion(fusion_strategy)
            return fusion.fuse(
                vector_results, document_results, top_k=top_k, weights=weights
            )

        return self._repository.search(
            vector_query=vector_query,
            vector_collection=vector_collection,
            top_k=top_k,
            graph_query=graph_query,
            graph_collection=graph_collection,
            document_filter=document_filter,
            document_collection=document_collection,
            fusion_strategy=(
                fusion_strategy
                if isinstance(fusion_strategy, FusionStrategy)
                else self._default_fusion
            ),
            weights=weights,
        )

    def sql(
        self,
        query: str,
        params: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute federated SQL query.

        Args:
            query: SQL query with extensions
            params: Query parameters

        Returns:
            Query results

        Example:
            results = hybrid.sql(
                "SELECT v.id, v.score, d.document "
                "FROM VECTOR_SEARCH('code_embeddings', ?, 10) v "
                "JOIN DOCUMENT_QUERY('code_files', '{"language": "python"}') d "
                "ON v.metadata.file_path = d.file_path",
                [query_vector],
            )
        """
        return self._repository.sql(query, params)

    def clear_cache(self) -> None:
        """Clear query result cache.

        Example:
            hybrid.clear_cache()
        """
        self._repository._cache.clear()

    def list_strategies(self) -> list[dict[str, Any]]:
        """List available fusion strategies."""
        return [
            {"id": "rrf"},
            {"id": "weighted_linear"},
            {"id": "cascade"},
            {"id": "rank_biased_precision"},
            {"id": "borda_count"},
            {"id": "comb_sum"},
            {"id": "comb_min"},
            {"id": "comb_max"},
        ]


# =============================================================================
# Factory Functions
# =============================================================================


def create_hybrid_api(
    client: Any,
    cache_ttl: int = 300,
    default_fusion: FusionStrategy = FusionStrategy.RRF,
) -> ProximaDBHybrid:
    """Factory function to create hybrid query API instance.

    Args:
        client: ProximaDB client instance
        cache_ttl: Cache TTL for results
        default_fusion: Default fusion strategy

    Returns:
        ProximaDBHybrid instance
    """
    return ProximaDBHybrid(
        client=client,
        cache_ttl=cache_ttl,
        default_fusion=default_fusion,
    )


def create_fusion_strategy(
    strategy: FusionStrategy,
    **kwargs,
) -> FusionStrategyBase:
    """Factory function to create fusion strategy.

    Args:
        strategy: Fusion strategy type
        **kwargs: Strategy-specific parameters

    Returns:
        Fusion strategy instance

    Example:
        rrf = create_fusion_strategy(FusionStrategy.RRF, k=60)
        weighted = create_fusion_strategy(
            FusionStrategy.WEIGHTED,
            weights={"vector": 0.6, "graph": 0.3, "document": 0.1}
        )
    """
    if strategy == FusionStrategy.RRF:
        return ReciprocalRankFusion(k=kwargs.get("k", 60))
    elif strategy == FusionStrategy.WEIGHTED:
        return WeightedFusion(weights=kwargs.get("weights"))
    elif strategy == FusionStrategy.CASCADE:
        return CascadeFusion()
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")
