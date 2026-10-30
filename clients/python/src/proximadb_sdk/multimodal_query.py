"""
ProximaDB Multi-Modal Query API

Unified query builder for combining vector, graph, document, and observability
queries into a single cohesive query interface.

Features:
- Unified query builder with fluent API
- Semantic joins based on vector similarity
- Graph-vector fusion queries
- Cross-model result aggregation
- Time-decay scoring functions

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries that can be combined."""

    VECTOR = "vector"
    GRAPH = "graph"
    DOCUMENT = "document"
    LOGS = "logs"
    METRICS = "metrics"
    TRACES = "traces"


class FusionStrategy(Enum):
    """Strategies for combining results from multiple query types."""

    INTERSECTION = "intersection"  # Only results matching all queries
    UNION = "union"  # All results from any query
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"  # Weighted combination
    SEQUENTIAL = "sequential"  # Pipeline: output of one feeds next


class JoinType(Enum):
    """Types of joins between query results."""

    INNER = "inner"  # Only matching records
    LEFT = "left"  # All from left, matches from right
    SEMANTIC = "semantic"  # Join by vector similarity
    GRAPH_PATH = "graph_path"  # Join by graph connectivity


class TimeDecayFunction(Enum):
    """Time decay functions for scoring."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    NONE = "none"


class QueryIntent(Enum):
    """Query intent classification for context-aware reranking."""

    NAVIGATIONAL = "navigational"  # Looking for a specific item
    INFORMATIONAL = "informational"  # Seeking information
    TRANSACTIONAL = "transactional"  # Action-oriented
    SIMILARITY_SEARCH = "similarity_search"  # Vector similarity
    RELATIONSHIP_EXPLORATION = "relationship_exploration"  # Graph traversal
    ANALYTICAL = "analytical"  # Analytics/aggregation


class TemporalPreference(Enum):
    """Temporal preference for recency-aware reranking."""

    RECENT = "recent"  # Prefer recent results
    HISTORICAL = "historical"  # Prefer established results
    NEUTRAL = "neutral"  # No temporal preference


@dataclass
class RerankConfig:
    """Configuration for cross-modal reranking.

    Attributes:
        semantic_rerank: Enable cross-modal semantic similarity reranking
        diversity_optimization: Enable MMR-based diversity optimization
        diversity_weight: Weight for diversity (0.0 to 1.0)
        mmr_lambda: MMR lambda (0.0 = max diversity, 1.0 = max relevance)
        context_aware: Enable context-aware scoring
        model_weights: Weights per query type for scoring
        generate_explanations: Generate human-readable explanations
        rerank_top_k: Number of top results to consider for reranking
    """

    semantic_rerank: bool = True
    diversity_optimization: bool = True
    diversity_weight: float = 0.3
    mmr_lambda: float = 0.7
    context_aware: bool = True
    model_weights: dict[str, float] | None = None
    generate_explanations: bool = False
    rerank_top_k: int = 100

    def __post_init__(self):
        if self.model_weights is None:
            self.model_weights = {
                "vector": 1.0,
                "document": 0.8,
                "graph": 0.9,
                "logs": 0.7,
                "metrics": 0.7,
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_rerank": self.semantic_rerank,
            "diversity_optimization": self.diversity_optimization,
            "diversity_weight": self.diversity_weight,
            "mmr_lambda": self.mmr_lambda,
            "context_aware": self.context_aware,
            "model_weights": self.model_weights,
            "generate_explanations": self.generate_explanations,
            "rerank_top_k": self.rerank_top_k,
        }


@dataclass
class QueryContext:
    """Query context for context-aware reranking.

    Attributes:
        query_text: Original query text
        query_embedding: Query embedding vector
        intent: Classified query intent
        temporal_preference: Recency preference
        required_models: Query types that must appear in results
        user_preferences: User-specific preference weights
    """

    query_text: str | None = None
    query_embedding: list[float] | None = None
    intent: QueryIntent | None = None
    temporal_preference: TemporalPreference = TemporalPreference.NEUTRAL
    required_models: list[str] | None = None
    user_preferences: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_text": self.query_text,
            "query_embedding": self.query_embedding,
            "intent": self.intent.value if self.intent else None,
            "temporal_preference": self.temporal_preference.value,
            "required_models": self.required_models or [],
            "user_preferences": self.user_preferences or {},
        }


@dataclass
class ScoreComponent:
    """Component of a reranking score with explanation."""

    name: str
    value: float
    weight: float
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "contribution": self.contribution,
        }


@dataclass
class RerankExplanation:
    """Explanation for a reranking decision."""

    record_id: str
    original_rank: int
    new_rank: int
    score_components: list[ScoreComponent]
    explanation_text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "original_rank": self.original_rank,
            "new_rank": self.new_rank,
            "score_components": [c.to_dict() for c in self.score_components],
            "explanation_text": self.explanation_text,
            "confidence": self.confidence,
        }


@dataclass
class RerankedResult:
    """Result of cross-modal reranking."""

    records: list[dict[str, Any]]
    explanations: list[RerankExplanation]
    quality_score: float
    diversity_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "explanations": [e.to_dict() for e in self.explanations],
            "quality_score": self.quality_score,
            "diversity_score": self.diversity_score,
        }


@dataclass
class VectorQueryComponent:
    """Vector similarity search component."""

    collection: str
    query_vector: list[float]
    top_k: int = 10
    min_similarity: float = 0.0
    filter: dict[str, Any] | None = None
    include_metadata: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "vector",
            "collection": self.collection,
            "query_vector": self.query_vector,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "filter": self.filter,
            "include_metadata": self.include_metadata,
        }


@dataclass
class GraphQueryComponent:
    """Graph traversal query component."""

    graph_id: str
    start_nodes: list[str] | None = None
    start_label: str | None = None
    edge_types: list[str] | None = None
    max_depth: int = 2
    direction: str = "outgoing"  # outgoing, incoming, both
    node_filter: dict[str, Any] | None = None
    edge_filter: dict[str, Any] | None = None
    limit: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "graph",
            "graph_id": self.graph_id,
            "start_nodes": self.start_nodes,
            "start_label": self.start_label,
            "edge_types": self.edge_types,
            "max_depth": self.max_depth,
            "direction": self.direction,
            "node_filter": self.node_filter,
            "edge_filter": self.edge_filter,
            "limit": self.limit,
        }


@dataclass
class DocumentQueryComponent:
    """Document search query component."""

    collection: str
    filter: dict[str, Any] | None = None
    text_query: str | None = None
    json_path_filters: dict[str, Any] | None = None
    limit: int = 100
    offset: int = 0
    sort_by: str | None = None
    sort_order: str = "asc"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "document",
            "collection": self.collection,
            "filter": self.filter,
            "text_query": self.text_query,
            "json_path_filters": self.json_path_filters,
            "limit": self.limit,
            "offset": self.offset,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
        }


@dataclass
class LogQueryComponent:
    """Log search query component."""

    namespace: str
    time_range: tuple[int, int] | None = None  # (start_ns, end_ns)
    services: list[str] | None = None
    severities: list[str] | None = None
    text_query: str | None = None
    field_filters: dict[str, Any] | None = None
    limit: int = 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "logs",
            "namespace": self.namespace,
            "time_range": self.time_range,
            "services": self.services,
            "severities": self.severities,
            "text_query": self.text_query,
            "field_filters": self.field_filters,
            "limit": self.limit,
        }


@dataclass
class MetricQueryComponent:
    """Metric aggregation query component."""

    namespace: str
    metric_names: list[str]
    time_range: tuple[int, int]  # (start_ns, end_ns)
    aggregation: str = "avg"  # avg, sum, min, max, count, p50, p95, p99
    group_by: list[str] | None = None
    bucket_size_ms: int = 60000  # 1 minute default

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "metrics",
            "namespace": self.namespace,
            "metric_names": self.metric_names,
            "time_range": self.time_range,
            "aggregation": self.aggregation,
            "group_by": self.group_by,
            "bucket_size_ms": self.bucket_size_ms,
        }


@dataclass
class SemanticJoin:
    """Semantic join configuration."""

    left_field: str
    right_field: str
    similarity_threshold: float = 0.7
    join_type: JoinType = JoinType.SEMANTIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_field": self.left_field,
            "right_field": self.right_field,
            "similarity_threshold": self.similarity_threshold,
            "join_type": self.join_type.value,
        }


@dataclass
class MultiModalQueryResult:
    """Result from a multi-modal query."""

    records: list[dict[str, Any]]
    total_count: int
    query_time_ms: float
    component_times: dict[str, float]
    fusion_strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    #: Component name -> the error that component failed with.
    #:
    #: A fused query runs several legs and merges what they return. Each leg used
    #: to swallow any exception and contribute `[]`, so a leg that FAILED and a
    #: leg that legitimately MATCHED NOTHING were indistinguishable, and the
    #: caller received a partial answer presented as complete. For a database
    #: those are different facts; this is where the difference lives.
    #:
    #: The parallel to `component_times` is deliberate: per-component timing was
    #: already worth reporting, and per-component failure is strictly more so.
    component_errors: dict[str, str] = field(default_factory=dict)

    @property
    def partial(self) -> bool:
        """True when at least one component failed, so `records` is incomplete.

        Check this before treating an empty or short result as an answer.
        """
        return bool(self.component_errors)

    def raise_if_partial(self) -> None:
        """Convert a partial result into an error, for callers that want one.

        Offered rather than imposed: some callers genuinely want best-effort
        fusion across legs, which is why the legs still return `[]`. What they
        cannot be allowed to do is *not know*.
        """
        if self.component_errors:
            detail = "; ".join(
                f"{k}: {v}" for k, v in sorted(self.component_errors.items())
            )
            raise RuntimeError(
                f"multi-modal query was partial -- {len(self.component_errors)} of "
                f"{len(self.component_times) or len(self.component_errors)} components "
                f"failed and contributed no records ({detail})"
            )

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def to_dataframe(self):
        """Convert to pandas DataFrame if pandas is available."""
        try:
            import pandas as pd

            return pd.DataFrame(self.records)
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")


class MultiModalQueryBuilder:
    """
    Fluent builder for multi-modal queries.

    Allows combining vector, graph, document, and observability queries
    with various fusion strategies and semantic joins.

    Example:
        query = (MultiModalQueryBuilder()
            .vector("products", query_embedding, top_k=20)
            .graph("knowledge", start_label="Category", edge_types=["CONTAINS"])
            .join_semantic("vector.embedding", "graph.node.embedding", threshold=0.8)
            .fuse(FusionStrategy.RRF)
            .with_time_decay(TimeDecayFunction.EXPONENTIAL, halflife_hours=24)
            .limit(10)
            .build())

        results = client.execute_multimodal(query)
    """

    def __init__(self):
        self._components: list[Any] = []
        self._joins: list[SemanticJoin] = []
        self._fusion_strategy: FusionStrategy = FusionStrategy.RRF
        self._fusion_weights: dict[str, float] = {}
        self._time_decay: tuple[TimeDecayFunction, dict[str, Any]] | None = None
        self._limit: int = 100
        self._offset: int = 0
        self._timeout_ms: int = 30000
        self._include_scores: bool = True
        self._include_metadata: bool = True
        self._custom_scorer: Callable | None = None

    def vector(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        min_similarity: float = 0.0,
        filter: dict[str, Any] | None = None,
        weight: float = 1.0,
    ) -> "MultiModalQueryBuilder":
        """Add a vector similarity search component."""
        component = VectorQueryComponent(
            collection=collection,
            query_vector=query_vector,
            top_k=top_k,
            min_similarity=min_similarity,
            filter=filter,
        )
        self._components.append(component)
        self._fusion_weights[f"vector_{len(self._components)}"] = weight
        return self

    def graph(
        self,
        graph_id: str,
        start_nodes: list[str] | None = None,
        start_label: str | None = None,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        direction: str = "outgoing",
        node_filter: dict[str, Any] | None = None,
        weight: float = 1.0,
    ) -> "MultiModalQueryBuilder":
        """Add a graph traversal component."""
        component = GraphQueryComponent(
            graph_id=graph_id,
            start_nodes=start_nodes,
            start_label=start_label,
            edge_types=edge_types,
            max_depth=max_depth,
            direction=direction,
            node_filter=node_filter,
        )
        self._components.append(component)
        self._fusion_weights[f"graph_{len(self._components)}"] = weight
        return self

    def graph_from_vector_results(
        self,
        graph_id: str,
        id_field: str = "id",
        edge_types: list[str] | None = None,
        max_depth: int = 1,
        weight: float = 1.0,
    ) -> "MultiModalQueryBuilder":
        """
        Add a graph traversal starting from vector search results.

        This creates a sequential pipeline where vector results
        feed into graph traversal.
        """
        component = GraphQueryComponent(
            graph_id=graph_id,
            start_nodes=None,  # Will be filled from previous results
            edge_types=edge_types,
            max_depth=max_depth,
        )
        component._from_previous = True
        component._id_field = id_field
        self._components.append(component)
        self._fusion_weights[f"graph_{len(self._components)}"] = weight
        return self

    def document(
        self,
        collection: str,
        filter: dict[str, Any] | None = None,
        text_query: str | None = None,
        json_path_filters: dict[str, Any] | None = None,
        weight: float = 1.0,
    ) -> "MultiModalQueryBuilder":
        """Add a document search component."""
        component = DocumentQueryComponent(
            collection=collection,
            filter=filter,
            text_query=text_query,
            json_path_filters=json_path_filters,
        )
        self._components.append(component)
        self._fusion_weights[f"document_{len(self._components)}"] = weight
        return self

    def logs(
        self,
        namespace: str,
        time_range: tuple[int, int] | None = None,
        services: list[str] | None = None,
        severities: list[str] | None = None,
        text_query: str | None = None,
        weight: float = 1.0,
    ) -> "MultiModalQueryBuilder":
        """Add a log search component."""
        component = LogQueryComponent(
            namespace=namespace,
            time_range=time_range,
            services=services,
            severities=severities,
            text_query=text_query,
        )
        self._components.append(component)
        self._fusion_weights[f"logs_{len(self._components)}"] = weight
        return self

    def metrics(
        self,
        namespace: str,
        metric_names: list[str],
        time_range: tuple[int, int],
        aggregation: str = "avg",
        group_by: list[str] | None = None,
        weight: float = 1.0,
    ) -> "MultiModalQueryBuilder":
        """Add a metric aggregation component."""
        component = MetricQueryComponent(
            namespace=namespace,
            metric_names=metric_names,
            time_range=time_range,
            aggregation=aggregation,
            group_by=group_by,
        )
        self._components.append(component)
        self._fusion_weights[f"metrics_{len(self._components)}"] = weight
        return self

    def join_semantic(
        self,
        left_field: str,
        right_field: str,
        similarity_threshold: float = 0.7,
    ) -> "MultiModalQueryBuilder":
        """Add a semantic join between components based on vector similarity."""
        self._joins.append(
            SemanticJoin(
                left_field=left_field,
                right_field=right_field,
                similarity_threshold=similarity_threshold,
                join_type=JoinType.SEMANTIC,
            )
        )
        return self

    def join_by_id(
        self,
        left_field: str,
        right_field: str,
    ) -> "MultiModalQueryBuilder":
        """Add an inner join between components by ID field."""
        self._joins.append(
            SemanticJoin(
                left_field=left_field,
                right_field=right_field,
                similarity_threshold=1.0,
                join_type=JoinType.INNER,
            )
        )
        return self

    def join_graph_path(
        self,
        left_field: str,
        right_field: str,
        graph_id: str,
        max_path_length: int = 3,
    ) -> "MultiModalQueryBuilder":
        """Add a join based on graph path connectivity."""
        join = SemanticJoin(
            left_field=left_field,
            right_field=right_field,
            similarity_threshold=0.0,
            join_type=JoinType.GRAPH_PATH,
        )
        join._graph_id = graph_id
        join._max_path_length = max_path_length
        self._joins.append(join)
        return self

    def fuse(
        self,
        strategy: FusionStrategy = FusionStrategy.RRF,
        weights: dict[str, float] | None = None,
    ) -> "MultiModalQueryBuilder":
        """Set the fusion strategy for combining results."""
        self._fusion_strategy = strategy
        if weights:
            self._fusion_weights.update(weights)
        return self

    def with_time_decay(
        self,
        function: TimeDecayFunction = TimeDecayFunction.EXPONENTIAL,
        halflife_hours: float = 24.0,
        reference_time: int | None = None,
        time_field: str = "timestamp",
    ) -> "MultiModalQueryBuilder":
        """Apply time decay to scoring."""
        self._time_decay = (
            function,
            {
                "halflife_hours": halflife_hours,
                "reference_time": reference_time or int(time.time() * 1e9),
                "time_field": time_field,
            },
        )
        return self

    def with_custom_scorer(
        self,
        scorer: Callable[[dict[str, Any]], float],
    ) -> "MultiModalQueryBuilder":
        """Apply a custom scoring function to results."""
        self._custom_scorer = scorer
        return self

    def limit(self, limit: int) -> "MultiModalQueryBuilder":
        """Set the maximum number of results."""
        self._limit = limit
        return self

    def offset(self, offset: int) -> "MultiModalQueryBuilder":
        """Set the offset for pagination."""
        self._offset = offset
        return self

    def timeout(self, timeout_ms: int) -> "MultiModalQueryBuilder":
        """Set the query timeout in milliseconds."""
        self._timeout_ms = timeout_ms
        return self

    def include_scores(self, include: bool = True) -> "MultiModalQueryBuilder":
        """Include component scores in results."""
        self._include_scores = include
        return self

    def include_metadata(self, include: bool = True) -> "MultiModalQueryBuilder":
        """Include metadata in results."""
        self._include_metadata = include
        return self

    def build(self) -> "MultiModalQuery":
        """Build the query object."""
        return MultiModalQuery(
            components=[c.to_dict() for c in self._components],
            joins=[j.to_dict() for j in self._joins],
            fusion_strategy=self._fusion_strategy.value,
            fusion_weights=self._fusion_weights,
            time_decay=self._time_decay,
            limit=self._limit,
            offset=self._offset,
            timeout_ms=self._timeout_ms,
            include_scores=self._include_scores,
            include_metadata=self._include_metadata,
            custom_scorer=self._custom_scorer,
        )


@dataclass
class MultiModalQuery:
    """Compiled multi-modal query ready for execution."""

    components: list[dict[str, Any]]
    joins: list[dict[str, Any]]
    fusion_strategy: str
    fusion_weights: dict[str, float]
    time_decay: tuple[TimeDecayFunction, dict[str, Any]] | None
    limit: int
    offset: int
    timeout_ms: int
    include_scores: bool
    include_metadata: bool
    custom_scorer: Callable | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API transmission."""
        result = {
            "components": self.components,
            "joins": self.joins,
            "fusion_strategy": self.fusion_strategy,
            "fusion_weights": self.fusion_weights,
            "limit": self.limit,
            "offset": self.offset,
            "timeout_ms": self.timeout_ms,
            "include_scores": self.include_scores,
            "include_metadata": self.include_metadata,
        }
        if self.time_decay:
            func, params = self.time_decay
            result["time_decay"] = {
                "function": func.value if isinstance(func, TimeDecayFunction) else func,
                **params,
            }
        return result


class CrossModalReranker:
    """
    Cross-Modal Reranker for multi-modal query results.

    Provides advanced reranking capabilities including:
    - Semantic similarity across modalities
    - Context-aware scoring based on query intent
    - Diversity optimization (MMR)
    - Explanation generation for transparency

    Usage:
        reranker = CrossModalReranker(config=RerankConfig(
            diversity_optimization=True,
            generate_explanations=True
        ))

        context = QueryContext(
            intent=QueryIntent.SIMILARITY_SEARCH,
            temporal_preference=TemporalPreference.RECENT
        )

        reranked = reranker.rerank(records, context)
        print(f"Quality: {reranked.quality_score}, Diversity: {reranked.diversity_score}")
    """

    def __init__(self, config: RerankConfig | None = None):
        """Initialize with optional configuration."""
        self.config = config or RerankConfig()

    def rerank(
        self,
        records: list[dict[str, Any]],
        context: QueryContext | None = None,
    ) -> RerankedResult:
        """
        Rerank query results using cross-modal signals.

        Args:
            records: List of result records to rerank
            context: Query context for context-aware scoring

        Returns:
            RerankedResult with reranked records and explanations
        """
        if not records:
            return RerankedResult(
                records=[],
                explanations=[],
                quality_score=1.0,
                diversity_score=1.0,
            )

        context = context or QueryContext()

        # Limit to top-k for reranking
        records_to_rerank = records[: self.config.rerank_top_k]

        # Step 1: Compute base scores
        scored_records = self._compute_base_scores(records_to_rerank, context)

        # Step 2: Apply semantic reranking
        if self.config.semantic_rerank:
            scored_records = self._apply_semantic_reranking(scored_records, context)

        # Step 3: Apply context-aware scoring
        if self.config.context_aware:
            scored_records = self._apply_context_aware_scoring(scored_records, context)

        # Step 4: Apply diversity optimization (MMR)
        if self.config.diversity_optimization:
            scored_records = self._apply_mmr_diversity(scored_records)

        # Step 5: Generate explanations
        explanations = []
        if self.config.generate_explanations:
            explanations = self._generate_explanations(
                records_to_rerank, scored_records
            )

        # Step 6: Sort and extract final records
        scored_records.sort(key=lambda x: x["final_score"], reverse=True)
        final_records = [sr["record"] for sr in scored_records]

        # Update scores in records
        for sr in scored_records:
            sr["record"]["_rerank_score"] = sr["final_score"]

        # Compute quality and diversity scores
        diversity_score = self._compute_diversity_score(final_records)
        quality_score = self._compute_quality_score(final_records)

        return RerankedResult(
            records=final_records,
            explanations=explanations,
            quality_score=quality_score,
            diversity_score=diversity_score,
        )

    def _compute_base_scores(
        self,
        records: list[dict[str, Any]],
        context: QueryContext,
    ) -> list[dict[str, Any]]:
        """Compute base scores for all records."""
        scored = []
        for idx, record in enumerate(records):
            source_type = record.get("_source_type", "vector")
            model_weight = self.config.model_weights.get(source_type, 1.0)

            base_score = record.get("score", record.get("_score", 0.5))
            weighted_score = base_score * model_weight

            scored.append(
                {
                    "record": record,
                    "original_rank": idx,
                    "base_score": base_score,
                    "semantic_score": 0.0,
                    "context_score": 0.0,
                    "diversity_penalty": 0.0,
                    "final_score": weighted_score,
                    "score_components": [
                        ScoreComponent(
                            name="base_score",
                            value=base_score,
                            weight=model_weight,
                            contribution=weighted_score,
                        )
                    ],
                }
            )

        return scored

    def _apply_semantic_reranking(
        self,
        records: list[dict[str, Any]],
        context: QueryContext,
    ) -> list[dict[str, Any]]:
        """Apply semantic similarity reranking."""
        if not context.query_embedding:
            return records

        for record in records:
            # Try to get record embedding
            embedding = record["record"].get("embedding")
            if embedding:
                semantic_score = self._cosine_similarity(
                    context.query_embedding, embedding
                )
            else:
                # Fall back to text similarity
                record_text = record["record"].get(
                    "content", record["record"].get("text", "")
                )
                if context.query_text and record_text:
                    semantic_score = self._text_similarity(
                        context.query_text, record_text
                    )
                else:
                    semantic_score = 0.5

            record["semantic_score"] = semantic_score
            record["score_components"].append(
                ScoreComponent(
                    name="semantic_score",
                    value=semantic_score,
                    weight=0.3,
                    contribution=semantic_score * 0.3,
                )
            )
            record["final_score"] = record["final_score"] * 0.7 + semantic_score * 0.3

        return records

    def _apply_context_aware_scoring(
        self,
        records: list[dict[str, Any]],
        context: QueryContext,
    ) -> list[dict[str, Any]]:
        """Apply context-aware scoring based on query intent."""
        for record in records:
            context_score = 0.0
            components = []
            source_type = record["record"].get("_source_type", "vector")

            # Intent-based scoring
            if context.intent:
                intent_boost = 0.0
                if (
                    context.intent == QueryIntent.SIMILARITY_SEARCH
                    and source_type == "vector"
                ):
                    intent_boost = 0.2
                elif (
                    context.intent == QueryIntent.RELATIONSHIP_EXPLORATION
                    and source_type == "graph"
                ):
                    intent_boost = 0.2
                elif (
                    context.intent == QueryIntent.NAVIGATIONAL
                    and record["base_score"] > 0.9
                ):
                    intent_boost = 0.15
                elif context.intent == QueryIntent.INFORMATIONAL:
                    intent_boost = 0.05
                elif context.intent == QueryIntent.ANALYTICAL and source_type in (
                    "logs",
                    "metrics",
                ):
                    intent_boost = 0.15

                context_score += intent_boost
                if intent_boost > 0:
                    components.append(
                        ScoreComponent(
                            name="intent_boost",
                            value=intent_boost,
                            weight=1.0,
                            contribution=intent_boost,
                        )
                    )

            # Temporal preference
            timestamp = record["record"].get(
                "timestamp", record["record"].get("created_at")
            )
            if timestamp:
                temporal_boost = self._compute_temporal_boost(
                    timestamp, context.temporal_preference
                )
                context_score += temporal_boost
                if temporal_boost > 0:
                    components.append(
                        ScoreComponent(
                            name="temporal_boost",
                            value=temporal_boost,
                            weight=1.0,
                            contribution=temporal_boost,
                        )
                    )

            # Required models boost
            if context.required_models and source_type in context.required_models:
                context_score += 0.1
                components.append(
                    ScoreComponent(
                        name="required_model_boost",
                        value=0.1,
                        weight=1.0,
                        contribution=0.1,
                    )
                )

            record["context_score"] = context_score
            record["score_components"].extend(components)
            record["final_score"] += context_score

        return records

    def _apply_mmr_diversity(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply Maximum Marginal Relevance for diversity."""
        if len(records) <= 1:
            return records

        lambda_param = self.config.mmr_lambda
        selected = []
        remaining = list(records)

        # Select first record (highest score)
        remaining.sort(key=lambda x: x["final_score"], reverse=True)
        if remaining:
            selected.append(remaining.pop(0))

        # Iteratively select remaining records using MMR
        while remaining:
            best_idx = 0
            best_mmr = float("-inf")

            for idx, candidate in enumerate(remaining):
                # Relevance term
                relevance = candidate["final_score"]

                # Diversity term: max similarity to any selected record
                max_similarity = max(
                    self._record_similarity(candidate["record"], s["record"])
                    for s in selected
                )

                # MMR score
                mmr_score = (
                    lambda_param * relevance - (1 - lambda_param) * max_similarity
                )

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            # Add diversity penalty
            selected_record = remaining.pop(best_idx)
            diversity_penalty = (1 - lambda_param) * max(
                0, 1 - best_mmr / max(selected_record["final_score"], 0.001)
            )
            selected_record["diversity_penalty"] = diversity_penalty
            selected_record["score_components"].append(
                ScoreComponent(
                    name="diversity_adjustment",
                    value=-diversity_penalty,
                    weight=1.0,
                    contribution=-diversity_penalty,
                )
            )
            selected.append(selected_record)

        # Update final scores with diversity penalty
        for record in selected:
            record["final_score"] -= (
                record["diversity_penalty"] * self.config.diversity_weight
            )

        return selected

    def _generate_explanations(
        self,
        original_records: list[dict[str, Any]],
        reranked: list[dict[str, Any]],
    ) -> list[RerankExplanation]:
        """Generate explanations for reranking decisions."""
        original_ranks = {
            r.get("id", str(i)): i for i, r in enumerate(original_records)
        }

        explanations = []
        for new_rank, scored in enumerate(reranked):
            record_id = scored["record"].get("id", str(scored["original_rank"]))
            original_rank = original_ranks.get(record_id, scored["original_rank"])
            rank_change = original_rank - new_rank

            if rank_change > 0:
                explanation_text = (
                    f"Promoted {rank_change} positions due to: "
                    + ", ".join(
                        f"{c.name} (+{c.contribution:.2f})"
                        for c in scored["score_components"]
                        if c.contribution > 0
                    )
                )
            elif rank_change < 0:
                explanation_text = (
                    f"Demoted {-rank_change} positions due to: "
                    + ", ".join(
                        f"{c.name} ({c.contribution:.2f})"
                        for c in scored["score_components"]
                        if c.contribution < 0
                    )
                )
            else:
                explanation_text = "Rank unchanged"

            confidence = self._compute_explanation_confidence(scored)

            explanations.append(
                RerankExplanation(
                    record_id=record_id,
                    original_rank=original_rank,
                    new_rank=new_rank,
                    score_components=scored["score_components"],
                    explanation_text=explanation_text,
                    confidence=confidence,
                )
            )

        return explanations

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a > 0 and norm_b > 0:
            return dot / (norm_a * norm_b)
        return 0.0

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0

    def _compute_temporal_boost(
        self,
        timestamp: int,
        preference: TemporalPreference,
    ) -> float:
        """Compute temporal boost based on preference."""
        now_ns = int(time.time() * 1e9)
        age_hours = (now_ns - timestamp) / (3600 * 1e9)

        if preference == TemporalPreference.RECENT:
            return math.exp(-age_hours / 24) * 0.1
        elif preference == TemporalPreference.HISTORICAL:
            return (1 - math.exp(-age_hours / 720)) * 0.1
        return 0.0

    def _record_similarity(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> float:
        """Compute similarity between two records."""
        similarity = 0.0

        # Source type similarity
        if a.get("_source_type") == b.get("_source_type"):
            similarity += 0.3

        # Metadata key overlap
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        key_overlap = len(keys_a & keys_b)
        key_union = len(keys_a | keys_b)
        if key_union > 0:
            similarity += 0.3 * (key_overlap / key_union)

        # Score similarity
        score_a = a.get("score", a.get("_score"))
        score_b = b.get("score", b.get("_score"))
        if score_a is not None and score_b is not None:
            similarity += 0.4 * (1 - abs(score_a - score_b))

        return min(similarity, 1.0)

    def _compute_diversity_score(self, records: list[dict[str, Any]]) -> float:
        """Compute diversity score for result set."""
        if len(records) <= 1:
            return 1.0

        # Count unique source types
        source_types = set(r.get("_source_type", "unknown") for r in records)
        model_diversity = len(source_types) / 5.0  # Max 5 types

        # Average pairwise dissimilarity
        total_dissimilarity = 0.0
        count = 0
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                total_dissimilarity += 1 - self._record_similarity(
                    records[i], records[j]
                )
                count += 1

        avg_dissimilarity = total_dissimilarity / count if count > 0 else 0.0

        return min((model_diversity * 0.5 + avg_dissimilarity * 0.5), 1.0)

    def _compute_quality_score(self, records: list[dict[str, Any]]) -> float:
        """Compute quality score for result set."""
        if not records:
            return 1.0

        scores = [r.get("_rerank_score", r.get("score", 0.5)) for r in records]
        return min(sum(scores) / len(scores), 1.0)

    def _compute_explanation_confidence(self, scored: dict[str, Any]) -> float:
        """Compute confidence in the explanation."""
        components = scored["score_components"]
        positive = sum(1 for c in components if c.contribution > 0)
        total = len(components)
        return positive / total if total > 0 else 0.5


class MultiModalQueryExecutor:
    """
    Client-side executor for multi-modal queries.

    Handles query decomposition, parallel execution, result fusion,
    and semantic join operations.
    """

    def __init__(self, client):
        """
        Initialize with a ProximaDB client.

        Args:
            client: ProximaDBClient instance
        """
        self._client = client

    def execute(self, query: MultiModalQuery) -> MultiModalQueryResult:
        """
        Execute a multi-modal query.

        Args:
            query: Compiled MultiModalQuery object

        Returns:
            MultiModalQueryResult with fused results
        """
        start_time = time.time()
        component_times = {}
        component_errors: dict[str, str] = {}
        component_results = []

        # Execute each component
        for i, component in enumerate(query.components):
            comp_start = time.time()

            comp_type = component.get("type")
            comp_key = f"{comp_type}_{i}"
            try:
                results = self._dispatch_component(
                    comp_type, component, component_results
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not discarded
                # Best-effort fusion is the intended behaviour: one failing leg
                # should not lose the others. What is NOT acceptable is the
                # caller being unable to tell. The leg contributes nothing and
                # the failure is named on the result.
                logger.warning(
                    "multi-modal component %s failed: %s", comp_key, exc, exc_info=True
                )
                component_errors[comp_key] = f"{type(exc).__name__}: {exc}"
                results = []

            component_results.append(results)
            component_times[comp_key] = (time.time() - comp_start) * 1000

        return self._finish_execute(
            query, component_results, component_times, component_errors, start_time
        )

    def _dispatch_component(
        self,
        comp_type: str | None,
        component: dict[str, Any],
        component_results: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Run one query leg. Raises; the caller decides what a failure means."""
        if comp_type == "vector":
            return self._execute_vector(component)
        if comp_type == "graph":
            # Check if this depends on previous results
            if hasattr(component, "_from_previous") and component.get("_from_previous"):
                if component_results:
                    prev_results = component_results[-1]
                    id_field = component.get("_id_field", "id")
                    start_nodes = [
                        r.get(id_field) for r in prev_results if r.get(id_field)
                    ]
                    component["start_nodes"] = start_nodes
            return self._execute_graph(component)
        if comp_type == "document":
            return self._execute_document(component)
        if comp_type == "logs":
            return self._execute_logs(component)
        if comp_type == "metrics":
            return self._execute_metrics(component)
        return []

    def _finish_execute(
        self,
        query: Any,
        component_results: list[list[dict[str, Any]]],
        component_times: dict[str, float],
        component_errors: dict[str, str],
        start_time: float,
    ) -> "MultiModalQueryResult":
        """Join, fuse and package. Split out so `execute` reads as its policy."""
        # Apply joins if specified
        if query.joins:
            component_results = self._apply_joins(component_results, query.joins)

        # Fuse results
        fused = self._fuse_results(
            component_results,
            query.fusion_strategy,
            query.fusion_weights,
        )

        # Apply time decay if specified
        if query.time_decay:
            fused = self._apply_time_decay(fused, query.time_decay)

        # Apply custom scorer if specified
        if query.custom_scorer:
            for record in fused:
                record["_custom_score"] = query.custom_scorer(record)
            fused.sort(key=lambda x: x.get("_custom_score", 0), reverse=True)

        # Apply limit and offset
        fused = fused[query.offset : query.offset + query.limit]

        total_time = (time.time() - start_time) * 1000

        return MultiModalQueryResult(
            records=fused,
            total_count=len(fused),
            query_time_ms=total_time,
            component_times=component_times,
            fusion_strategy=query.fusion_strategy,
            component_errors=component_errors,
            metadata={
                "component_count": len(query.components),
                "join_count": len(query.joins),
            },
        )

    def _execute_vector(self, component: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute vector search component.

        Searches for similar vectors in the specified collection and returns
        results with score and metadata.
        """
        try:
            # TD-SDKDOC-1: this passed collection=/filter=, which search()
            # does not declare — every vector leg raised TypeError and the
            # dispatch loop reduced it to a warning (empty results).
            results = self._client.search(
                component["collection"],
                vector=component["query_vector"],
                top_k=component.get("top_k", 10),
                metadata_filter=component.get("filter"),
            )
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "metadata": r.metadata,
                    "_source_type": "vector",
                }
                for r in results
            ]
        except Exception as exc:  # noqa: BLE001 - re-raised; the caller records it
            # Deliberately NOT swallowed. A leg that failed and a leg that
            # matched nothing both used to return []. The dispatch loop in
            # `execute` owns the best-effort policy and records which
            # component failed, so a partial answer says so.
            raise exc

    def _execute_graph(self, component: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute graph traversal component.

        Performs graph traversal starting from specified nodes or nodes
        with a given label, following edges up to max_depth hops.
        """
        try:
            # Use graph analytics if available
            if hasattr(self._client, "graph"):
                results = self._client.graph.traverse(
                    graph_id=component["graph_id"],
                    start_nodes=component.get("start_nodes"),
                    edge_types=component.get("edge_types"),
                    max_depth=component.get("max_depth", 2),
                    limit=component.get("limit", 100),
                )
                return [
                    {
                        "id": r.get("id"),
                        "node": r,
                        "depth": r.get("depth", 0),
                        "labels": r.get("labels", []),
                        "properties": r.get("properties", {}),
                        "_source_type": "graph",
                    }
                    for r in results
                ]
            return []
        except Exception as exc:  # noqa: BLE001 - re-raised; the caller records it
            # Deliberately NOT swallowed. A leg that failed and a leg that
            # matched nothing both used to return []. The dispatch loop in
            # `execute` owns the best-effort policy and records which
            # component failed, so a partial answer says so.
            raise exc

    def _execute_document(self, component: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute document query component.

        Queries documents from a collection with optional filters and text search.
        """
        try:
            if hasattr(self._client, "query_documents"):
                results = self._client.query_documents(
                    collection=component["collection"],
                    filter=component.get("filter"),
                    text_query=component.get("text_query"),
                    limit=component.get("limit", 100),
                )
                return [
                    {
                        "id": r.get("id"),
                        "document": r,
                        "_source_type": "document",
                    }
                    for r in results
                ]
            return []
        except Exception as exc:  # noqa: BLE001 - re-raised; the caller records it
            # Deliberately NOT swallowed. A leg that failed and a leg that
            # matched nothing both used to return []. The dispatch loop in
            # `execute` owns the best-effort policy and records which
            # component failed, so a partial answer says so.
            raise exc

    def _execute_logs(self, component: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute log query component.

        Searches logs within a time range with optional service and text filters.
        """
        try:
            if hasattr(self._client, "query_logs"):
                results = self._client.query_logs(
                    namespace=component["namespace"],
                    time_range=component.get("time_range"),
                    services=component.get("services"),
                    text_query=component.get("text_query"),
                    limit=component.get("limit", 1000),
                )
                return [
                    {
                        "id": r.get("id"),
                        "log": r,
                        "timestamp": r.get("timestamp"),
                        "_source_type": "logs",
                    }
                    for r in results
                ]
            return []
        except Exception as exc:  # noqa: BLE001 - re-raised; the caller records it
            # Deliberately NOT swallowed. A leg that failed and a leg that
            # matched nothing both used to return []. The dispatch loop in
            # `execute` owns the best-effort policy and records which
            # component failed, so a partial answer says so.
            raise exc

    def _execute_metrics(self, component: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute metric aggregation component.

        Aggregates metrics over a time range with optional grouping.
        """
        try:
            if hasattr(self._client, "aggregate_metrics"):
                results = self._client.aggregate_metrics(
                    namespace=component["namespace"],
                    metric_names=component["metric_names"],
                    time_range=component["time_range"],
                    aggregation=component.get("aggregation", "avg"),
                )
                return [
                    {
                        "metric_name": r.get("name"),
                        "value": r.get("value"),
                        "timestamp": r.get("timestamp"),
                        "_source_type": "metrics",
                    }
                    for r in results
                ]
            return []
        except Exception as exc:  # noqa: BLE001 - re-raised; the caller records it
            # Deliberately NOT swallowed. A leg that failed and a leg that
            # matched nothing both used to return []. The dispatch loop in
            # `execute` owns the best-effort policy and records which
            # component failed, so a partial answer says so.
            raise exc

    def _apply_joins(
        self,
        component_results: list[list[dict[str, Any]]],
        joins: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Apply joins between component results."""
        if len(component_results) < 2:
            return component_results

        for join in joins:
            join_type = join.get("join_type", "inner")
            left_field = join.get("left_field", "id")
            right_field = join.get("right_field", "id")
            threshold = join.get("similarity_threshold", 1.0)

            # For now, implement simple ID-based inner join
            # More sophisticated semantic joins would require embedding comparison
            if join_type == "inner" or join_type == "semantic":
                left_results = component_results[0]
                right_results = component_results[1]

                # Build index of right results
                right_index = {}
                for r in right_results:
                    key = self._extract_field(r, right_field)
                    if key:
                        right_index[key] = r

                # Join
                joined = []
                for left in left_results:
                    left_key = self._extract_field(left, left_field)
                    if left_key and left_key in right_index:
                        merged = {**left, **right_index[left_key]}
                        merged["_join_type"] = join_type
                        joined.append(merged)

                component_results = [joined] + component_results[2:]

        return component_results

    def _extract_field(self, record: dict[str, Any], field_path: str) -> str | None:
        """Extract a field value from a nested record."""
        parts = field_path.split(".")
        current = record
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return str(current) if current is not None else None

    def _fuse_results(
        self,
        component_results: list[list[dict[str, Any]]],
        strategy: str,
        weights: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Fuse results from multiple components."""
        if not component_results:
            return []

        if len(component_results) == 1:
            return component_results[0]

        if strategy == "intersection":
            return self._fuse_intersection(component_results)
        elif strategy == "union":
            return self._fuse_union(component_results)
        elif strategy == "rrf":
            return self._fuse_rrf(component_results, weights)
        elif strategy == "weighted":
            return self._fuse_weighted(component_results, weights)
        else:
            # Default to RRF
            return self._fuse_rrf(component_results, weights)

    def _fuse_intersection(
        self,
        component_results: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Return only records present in all components."""
        if not component_results:
            return []

        # Get IDs from first component
        common_ids = set(r.get("id") for r in component_results[0] if r.get("id"))

        # Intersect with other components
        for results in component_results[1:]:
            ids = set(r.get("id") for r in results if r.get("id"))
            common_ids &= ids

        # Return records with common IDs
        return [r for r in component_results[0] if r.get("id") in common_ids]

    def _fuse_union(
        self,
        component_results: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Return all records from any component (deduplicated)."""
        seen_ids = set()
        result = []

        for results in component_results:
            for r in results:
                record_id = r.get("id")
                if record_id and record_id not in seen_ids:
                    seen_ids.add(record_id)
                    result.append(r)
                elif not record_id:
                    result.append(r)

        return result

    def _fuse_rrf(
        self,
        component_results: list[list[dict[str, Any]]],
        weights: dict[str, float],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion."""
        scores = {}
        records = {}

        for comp_idx, results in enumerate(component_results):
            weight = weights.get(f"component_{comp_idx}", 1.0)

            for rank, record in enumerate(results):
                record_id = record.get("id", f"_anon_{comp_idx}_{rank}")
                rrf_score = weight / (k + rank + 1)

                if record_id in scores:
                    scores[record_id] += rrf_score
                else:
                    scores[record_id] = rrf_score
                    records[record_id] = record

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        result = []
        for record_id in sorted_ids:
            record = records[record_id].copy()
            record["_rrf_score"] = scores[record_id]
            result.append(record)

        return result

    def _fuse_weighted(
        self,
        component_results: list[list[dict[str, Any]]],
        weights: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Weighted score combination."""
        scores = {}
        records = {}

        for comp_idx, results in enumerate(component_results):
            weight = weights.get(f"component_{comp_idx}", 1.0)

            for record in results:
                record_id = record.get("id", id(record))
                component_score = record.get("score", 1.0) * weight

                if record_id in scores:
                    scores[record_id] += component_score
                else:
                    scores[record_id] = component_score
                    records[record_id] = record

        # Sort by weighted score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        result = []
        for record_id in sorted_ids:
            record = records[record_id].copy()
            record["_weighted_score"] = scores[record_id]
            result.append(record)

        return result

    def _apply_time_decay(
        self,
        records: list[dict[str, Any]],
        time_decay: tuple[TimeDecayFunction, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply time decay to record scores."""
        func, params = time_decay
        reference_time = params.get("reference_time", int(time.time() * 1e9))
        halflife_ns = params.get("halflife_hours", 24) * 3600 * 1e9
        time_field = params.get("time_field", "timestamp")

        for record in records:
            timestamp = record.get(time_field)
            if timestamp is None:
                continue

            age_ns = reference_time - timestamp
            if age_ns < 0:
                age_ns = 0

            if func == TimeDecayFunction.LINEAR or func == "linear":
                decay = max(0, 1 - (age_ns / (halflife_ns * 2)))
            elif func == TimeDecayFunction.EXPONENTIAL or func == "exponential":
                decay = math.exp(-0.693 * age_ns / halflife_ns)  # ln(2) ≈ 0.693
            elif func == TimeDecayFunction.GAUSSIAN or func == "gaussian":
                decay = math.exp(-0.5 * (age_ns / halflife_ns) ** 2)
            else:
                decay = 1.0

            # Apply decay to existing score
            current_score = record.get("score", record.get("_rrf_score", 1.0))
            record["_decayed_score"] = current_score * decay
            record["_time_decay"] = decay

        # Re-sort by decayed score
        records.sort(key=lambda x: x.get("_decayed_score", 0), reverse=True)

        return records


# Convenience functions for common query patterns


def semantic_search_with_graph(
    client,
    collection: str,
    query_vector: list[float],
    graph_id: str,
    edge_types: list[str] | None = None,
    top_k: int = 10,
) -> MultiModalQueryResult:
    """
    Perform semantic search and expand results via graph traversal.

    Finds similar vectors, then explores their graph neighborhood.
    """
    query = (
        MultiModalQueryBuilder()
        .vector(collection, query_vector, top_k=top_k * 2)
        .graph_from_vector_results(graph_id, edge_types=edge_types)
        .fuse(FusionStrategy.RRF)
        .limit(top_k)
        .build()
    )

    executor = MultiModalQueryExecutor(client)
    return executor.execute(query)


def knowledge_graph_search(
    client,
    graph_id: str,
    start_label: str,
    query_vector: list[float],
    vector_collection: str,
    max_depth: int = 2,
    top_k: int = 10,
) -> MultiModalQueryResult:
    """
    Search knowledge graph with semantic similarity ranking.

    Traverses graph from nodes of a given label, ranking by vector similarity.
    """
    query = (
        MultiModalQueryBuilder()
        .graph(graph_id, start_label=start_label, max_depth=max_depth)
        .vector(vector_collection, query_vector, top_k=top_k * 3)
        .join_by_id("graph.id", "vector.id")
        .fuse(FusionStrategy.WEIGHTED, weights={"graph_1": 0.3, "vector_2": 0.7})
        .limit(top_k)
        .build()
    )

    executor = MultiModalQueryExecutor(client)
    return executor.execute(query)


def logs_with_context(
    client,
    namespace: str,
    error_query: str,
    context_graph_id: str,
    time_range: tuple[int, int],
    top_k: int = 20,
) -> MultiModalQueryResult:
    """
    Find error logs and their service context via graph.

    Searches logs for errors, then traverses service dependency graph.
    """
    query = (
        MultiModalQueryBuilder()
        .logs(
            namespace,
            time_range=time_range,
            text_query=error_query,
            severities=["ERROR", "FATAL"],
        )
        .graph(
            context_graph_id, start_label="Service", edge_types=["DEPENDS_ON", "CALLS"]
        )
        .join_by_id("logs.service", "graph.node.name")
        .fuse(FusionStrategy.SEQUENTIAL)
        .with_time_decay(TimeDecayFunction.EXPONENTIAL, halflife_hours=1)
        .limit(top_k)
        .build()
    )

    executor = MultiModalQueryExecutor(client)
    return executor.execute(query)


# =============================================================================
# Learned Fusion - ML-based Result Fusion
# =============================================================================


class FusionModelType(Enum):
    """Model type for learned fusion."""

    LINEAR = "linear"  # Linear model with learned weights
    GRADIENT_BOOSTING = "gradient_boosting"  # LightGBM-style
    NEURAL_NETWORK = "neural_network"  # Simple MLP
    ENSEMBLE = "ensemble"  # Ensemble of models


@dataclass
class LearnedFusionConfig:
    """Configuration for learned fusion.

    Attributes:
        model_type: Type of ML model to use
        num_features: Number of features to extract
        learning_rate: Learning rate for online updates
        regularization: L2 regularization strength
        collect_training_data: Whether to collect training data
        max_training_samples: Maximum training samples to keep
        min_samples_for_training: Minimum samples before training
        enable_online_learning: Enable continuous learning
        online_update_frequency: How often to update (in queries)
    """

    model_type: FusionModelType = FusionModelType.GRADIENT_BOOSTING
    num_features: int = 32
    learning_rate: float = 0.01
    regularization: float = 0.001
    collect_training_data: bool = True
    max_training_samples: int = 10000
    min_samples_for_training: int = 100
    enable_online_learning: bool = True
    online_update_frequency: int = 100


@dataclass
class FusionFeatures:
    """Features extracted for fusion learning.

    Attributes:
        query_features: Query-level features
        model_features: Per-model result features
        record_features: Per-record features
        interaction_features: Cross-model interaction features
    """

    query_features: list[float] = field(default_factory=list)
    model_features: dict[str, list[float]] = field(default_factory=dict)
    record_features: dict[str, list[float]] = field(default_factory=dict)
    interaction_features: list[float] = field(default_factory=list)

    def to_flat_vector(self) -> list[float]:
        """Convert to flat feature vector for model input."""
        features = list(self.query_features)

        # Add model features in deterministic order
        for model_type in ["vector", "document", "graph", "observability"]:
            if model_type in self.model_features:
                features.extend(self.model_features[model_type])
            else:
                features.extend([0.0] * len(self.query_features))

        features.extend(self.interaction_features)
        return features


class FeedbackType(Enum):
    """Types of user feedback for learning."""

    CLICK = "click"  # User clicked on result
    RATING = "rating"  # User rated results
    QUERY_REFINEMENT = "query_refinement"  # Negative signal
    RELEVANCE_JUDGMENT = "relevance_judgment"  # Explicit judgment


@dataclass
class FeedbackSignal:
    """User feedback signal for learning.

    Attributes:
        feedback_type: Type of feedback
        record_id: Record ID (for click/relevance)
        position: Click position (for click feedback)
        score: Rating score (for rating feedback)
        relevant: Relevance judgment (True/False)
    """

    feedback_type: FeedbackType
    record_id: str | None = None
    position: int | None = None
    score: float | None = None
    relevant: bool | None = None


@dataclass
class TrainingSample:
    """Training sample for learned fusion.

    Attributes:
        features: Extracted features
        target_scores: Target scores per record
        feedback: Optional user feedback
        timestamp_ms: Sample timestamp
    """

    features: FusionFeatures
    target_scores: dict[str, float]
    feedback: FeedbackSignal | None = None
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class TrainingMetrics:
    """Metrics from model training.

    Attributes:
        num_samples: Number of training samples
        loss: Training loss
        validation_loss: Validation loss (if applicable)
        training_time_ms: Training time in milliseconds
        iterations: Number of training iterations
    """

    num_samples: int = 0
    loss: float = 0.0
    validation_loss: float | None = None
    training_time_ms: int = 0
    iterations: int = 0


class LearnedFusion:
    """
    Learned Fusion - ML-based Result Fusion

    Uses machine learning models to learn optimal fusion strategies from
    training data and user feedback. Supports multiple model types including
    linear models, gradient boosting, and neural networks.

    Features:
    - Feature extraction from query and results
    - Online learning from user feedback
    - Multiple model types (Linear, GradientBoosting, NN)
    - Training data collection and management

    Usage:
        config = LearnedFusionConfig(
            model_type=FusionModelType.GRADIENT_BOOSTING,
            enable_online_learning=True
        )
        fusion = LearnedFusion(config)

        # Fuse results from multiple sources
        fused = fusion.fuse(vector_results, document_results, graph_results)

        # Record user feedback for learning
        fusion.record_feedback(features, FeedbackSignal(
            feedback_type=FeedbackType.CLICK,
            record_id="doc_123",
            position=0
        ))

        # Train model on accumulated data
        metrics = fusion.train()
    """

    def __init__(self, config: LearnedFusionConfig | None = None):
        """Initialize learned fusion engine.

        Args:
            config: Configuration for learned fusion
        """
        self.config = config or LearnedFusionConfig()
        self._training_buffer: list[TrainingSample] = []
        self._query_count = 0
        self._is_trained = False
        self._model_weights: list[float] = []
        self._feature_extractor = FeatureExtractor(self.config.num_features)

        # Initialize model weights based on type
        num_total_features = self.config.num_features * 5
        if self.config.model_type == FusionModelType.LINEAR:
            self._model_weights = [0.0] * num_total_features
        elif self.config.model_type == FusionModelType.GRADIENT_BOOSTING:
            self._model_weights = [0.0] * num_total_features
            self._trees: list[dict] = []  # Decision stumps for boosting

    def fuse(
        self,
        results: list[dict[str, list[dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        """
        Fuse results from multiple query sources using learned model.

        Args:
            results: List of result sets, each with source type and records
                     Format: [{"source": "vector", "records": [...]}, ...]

        Returns:
            Fused and ranked list of records
        """
        if not results:
            return []

        # Single source - no fusion needed
        if len(results) == 1:
            return results[0].get("records", [])

        # Extract features
        features = self._feature_extractor.extract(results)

        # Collect all unique records
        all_records: dict[str, dict[str, Any]] = {}
        for result_set in results:
            for record in result_set.get("records", []):
                record_id = record.get("id", str(id(record)))
                if record_id not in all_records:
                    all_records[record_id] = record.copy()
                else:
                    # Merge record data
                    existing = all_records[record_id]
                    for key, value in record.items():
                        if key not in existing:
                            existing[key] = value

        record_ids = list(all_records.keys())

        # Get fusion scores
        if self._is_trained:
            scores = self._predict(features, record_ids)
        else:
            # Fallback to RRF
            scores = self._fallback_rrf_scores(results, record_ids)

        # Apply scores and sort
        fused_records = []
        for record_id, score in zip(record_ids, scores):
            record = all_records[record_id]
            record["_fusion_score"] = score
            fused_records.append(record)

        fused_records.sort(key=lambda x: x.get("_fusion_score", 0), reverse=True)

        # Increment query count for online learning
        self._query_count += 1
        if (
            self.config.enable_online_learning
            and self._query_count % self.config.online_update_frequency == 0
            and len(self._training_buffer) >= self.config.min_samples_for_training
        ):
            self._maybe_train()

        return fused_records

    def record_feedback(
        self,
        features: FusionFeatures,
        feedback: FeedbackSignal,
    ) -> None:
        """
        Record user feedback for learning.

        Args:
            features: Features from the query
            feedback: User feedback signal
        """
        if not self.config.collect_training_data:
            return

        # Convert feedback to target scores
        target_scores: dict[str, float] = {}

        if feedback.feedback_type == FeedbackType.CLICK:
            if feedback.record_id and feedback.position is not None:
                # Position-based target
                target = 1.0 / (feedback.position + 1.0)
                target_scores[feedback.record_id] = target

        elif feedback.feedback_type == FeedbackType.RELEVANCE_JUDGMENT:
            if feedback.record_id and feedback.relevant is not None:
                target_scores[feedback.record_id] = 1.0 if feedback.relevant else 0.0

        sample = TrainingSample(
            features=features,
            target_scores=target_scores,
            feedback=feedback,
        )

        self.add_training_sample(sample)

    def add_training_sample(self, sample: TrainingSample) -> None:
        """
        Add a training sample to the buffer.

        Args:
            sample: Training sample to add
        """
        if not self.config.collect_training_data:
            return

        # Maintain max buffer size
        if len(self._training_buffer) >= self.config.max_training_samples:
            self._training_buffer.pop(0)

        self._training_buffer.append(sample)

    def train(self) -> TrainingMetrics:
        """
        Train the model on accumulated samples.

        Returns:
            Training metrics

        Raises:
            ValueError: If not enough training samples
        """
        if len(self._training_buffer) < self.config.min_samples_for_training:
            raise ValueError(
                f"Not enough training samples: {len(self._training_buffer)} < "
                f"{self.config.min_samples_for_training}"
            )

        start_time = time.time()

        if self.config.model_type == FusionModelType.LINEAR:
            metrics = self._train_linear()
        elif self.config.model_type == FusionModelType.GRADIENT_BOOSTING:
            metrics = self._train_gradient_boosting()
        else:
            # Default to linear
            metrics = self._train_linear()

        metrics.training_time_ms = int((time.time() - start_time) * 1000)
        self._is_trained = True

        return metrics

    def _train_linear(self) -> TrainingMetrics:
        """Train linear fusion model."""
        total_loss = 0.0
        iterations = 0

        for sample in self._training_buffer:
            flat_features = sample.features.to_flat_vector()

            # Ensure weights match dimension
            if len(flat_features) > len(self._model_weights):
                self._model_weights.extend(
                    [0.0] * (len(flat_features) - len(self._model_weights))
                )

            # Forward pass
            score = sum(w * f for w, f in zip(self._model_weights, flat_features))
            predicted = 1.0 / (1.0 + math.exp(-score))  # Sigmoid

            # Target
            if sample.target_scores:
                target = sum(sample.target_scores.values()) / len(sample.target_scores)
            else:
                target = 0.5

            # Loss
            loss = -target * math.log(predicted + 1e-10) - (1 - target) * math.log(
                1 - predicted + 1e-10
            )
            total_loss += loss

            # Gradient descent
            error = predicted - target
            gradient_scale = error * predicted * (1.0 - predicted)

            for i in range(min(len(flat_features), len(self._model_weights))):
                gradient = (
                    gradient_scale * flat_features[i]
                    + self.config.regularization * self._model_weights[i]
                )
                self._model_weights[i] -= self.config.learning_rate * gradient

            iterations += 1

        return TrainingMetrics(
            num_samples=len(self._training_buffer),
            loss=(
                total_loss / len(self._training_buffer) if self._training_buffer else 0
            ),
            iterations=iterations,
        )

    def _train_gradient_boosting(self) -> TrainingMetrics:
        """Train gradient boosting fusion model."""
        if not self._training_buffer:
            return TrainingMetrics()

        # Prepare features and targets
        feature_vecs = [s.features.to_flat_vector() for s in self._training_buffer]
        targets = []
        for sample in self._training_buffer:
            if sample.target_scores:
                targets.append(
                    sum(sample.target_scores.values()) / len(sample.target_scores)
                )
            else:
                targets.append(0.5)

        # Initialize residuals
        residuals = [t - 0.5 for t in targets]  # Start from base prediction 0.5

        iterations = 0
        total_loss = 0.0
        max_trees = 10

        while len(self._trees) < max_trees:
            # Fit stump to residuals
            best_stump = self._fit_stump(feature_vecs, residuals)
            if best_stump is None:
                break

            self._trees.append(best_stump)

            # Update residuals
            for i, features in enumerate(feature_vecs):
                pred = self._stump_predict(best_stump, features)
                residuals[i] -= pred

            iterations += 1

            # Check convergence
            mse = sum(r**2 for r in residuals) / len(residuals)
            total_loss = mse
            if mse < 1e-6:
                break

        return TrainingMetrics(
            num_samples=len(self._training_buffer),
            loss=total_loss,
            iterations=iterations,
        )

    def _fit_stump(
        self,
        feature_vecs: list[list[float]],
        residuals: list[float],
    ) -> dict | None:
        """Fit a decision stump to residuals."""
        if not feature_vecs or not feature_vecs[0]:
            return None

        num_features = len(feature_vecs[0])
        best_stump = None
        best_loss = float("inf")

        for feature_idx in range(num_features):
            # Get feature values with residuals
            values = [
                (fv[feature_idx] if feature_idx < len(fv) else 0.0, r)
                for fv, r in zip(feature_vecs, residuals)
            ]
            values.sort(key=lambda x: x[0])

            # Try different thresholds
            for i in range(len(values) - 1):
                threshold = (values[i][0] + values[i + 1][0]) / 2.0

                # Compute left and right means
                left = values[: i + 1]
                right = values[i + 1 :]

                left_mean = sum(v[1] for v in left) / len(left) if left else 0.0
                right_mean = sum(v[1] for v in right) / len(right) if right else 0.0

                # Compute MSE
                loss = sum((v[1] - left_mean) ** 2 for v in left)
                loss += sum((v[1] - right_mean) ** 2 for v in right)

                if loss < best_loss:
                    best_loss = loss
                    best_stump = {
                        "feature_index": feature_idx,
                        "threshold": threshold,
                        "left_value": left_mean * self.config.learning_rate,
                        "right_value": right_mean * self.config.learning_rate,
                    }

        return best_stump

    def _stump_predict(self, stump: dict, features: list[float]) -> float:
        """Get prediction from a decision stump."""
        feature_idx = stump["feature_index"]
        feature_val = features[feature_idx] if feature_idx < len(features) else 0.0

        if feature_val <= stump["threshold"]:
            return stump["left_value"]
        else:
            return stump["right_value"]

    def _predict(
        self,
        features: FusionFeatures,
        record_ids: list[str],
    ) -> list[float]:
        """Get predictions from the trained model."""
        flat_features = features.to_flat_vector()

        if self.config.model_type == FusionModelType.LINEAR:
            # Ensure weights match
            if len(flat_features) > len(self._model_weights):
                self._model_weights.extend(
                    [0.0] * (len(flat_features) - len(self._model_weights))
                )

            score = sum(w * f for w, f in zip(self._model_weights, flat_features))
            base_score = 1.0 / (1.0 + math.exp(-score))

            return [base_score] * len(record_ids)

        elif self.config.model_type == FusionModelType.GRADIENT_BOOSTING:
            base_score = 0.5

            for stump in self._trees:
                base_score += self._stump_predict(stump, flat_features)

            # Clamp to [0, 1]
            base_score = max(0.0, min(1.0, base_score))

            return [base_score] * len(record_ids)

        return [0.5] * len(record_ids)

    def _fallback_rrf_scores(
        self,
        results: list[dict[str, list[dict[str, Any]]]],
        record_ids: list[str],
    ) -> list[float]:
        """Fallback RRF scoring when model not trained."""
        k = 60.0
        rrf_scores: dict[str, float] = dict.fromkeys(record_ids, 0.0)

        for result_set in results:
            records = result_set.get("records", [])
            # Sort by score
            sorted_records = sorted(
                records, key=lambda x: x.get("score", 0), reverse=True
            )

            for rank, record in enumerate(sorted_records):
                record_id = record.get("id", str(id(record)))
                if record_id in rrf_scores:
                    rrf_scores[record_id] += 1.0 / (k + rank + 1)

        return [rrf_scores.get(rid, 0.0) for rid in record_ids]

    def _maybe_train(self) -> None:
        """Maybe train if conditions are met."""
        if len(self._training_buffer) >= self.config.min_samples_for_training:
            try:
                self.train()
            except Exception:
                pass  # Ignore errors in background training

    def get_feature_importance(self) -> list[float] | None:
        """Get feature importance from trained model."""
        if not self._is_trained:
            return None

        if self.config.model_type == FusionModelType.LINEAR:
            # Return absolute weights as importance
            total = sum(abs(w) for w in self._model_weights)
            if total > 0:
                return [abs(w) / total for w in self._model_weights]
            return None

        elif self.config.model_type == FusionModelType.GRADIENT_BOOSTING:
            # Count feature usage in trees
            if not self._trees:
                return None

            importance = [0.0] * self.config.num_features * 5
            for stump in self._trees:
                idx = stump.get("feature_index", 0)
                if idx < len(importance):
                    importance[idx] += 1.0

            total = sum(importance)
            if total > 0:
                return [i / total for i in importance]
            return None

        return None

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained

    @property
    def training_buffer_size(self) -> int:
        """Get number of training samples in buffer."""
        return len(self._training_buffer)


class FeatureExtractor:
    """Feature extractor for learned fusion."""

    def __init__(self, num_features: int = 32):
        """Initialize feature extractor.

        Args:
            num_features: Number of features per component
        """
        self.num_features = num_features

    def extract(
        self,
        results: list[dict[str, list[dict[str, Any]]]],
    ) -> FusionFeatures:
        """
        Extract features from query results.

        Args:
            results: List of result sets from different sources

        Returns:
            FusionFeatures for model input
        """
        features = FusionFeatures(
            query_features=[0.0] * self.num_features,
            interaction_features=[0.0] * self.num_features,
        )

        # Query-level features
        features.query_features = self._extract_query_features(results)

        # Per-model features
        for result_set in results:
            source = result_set.get("source", "unknown")
            model_features = self._extract_model_features(result_set)
            features.model_features[source] = model_features

        # Per-record features
        features.record_features = self._extract_record_features(results)

        # Interaction features
        features.interaction_features = self._extract_interaction_features(results)

        return features

    def _extract_query_features(
        self,
        results: list[dict[str, list[dict[str, Any]]]],
    ) -> list[float]:
        """Extract query-level features."""
        features = [0.0] * self.num_features

        # Number of modalities
        features[0] = len(results) / 4.0

        # Total results
        total_results = sum(len(r.get("records", [])) for r in results)
        features[1] = math.log(total_results + 1) / 10.0

        # Average results per modality
        if results:
            features[2] = (total_results / len(results)) / 100.0

        # Score statistics
        all_scores = []
        for result_set in results:
            for record in result_set.get("records", []):
                if "score" in record:
                    all_scores.append(record["score"])

        if all_scores:
            mean = sum(all_scores) / len(all_scores)
            variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)
            std_dev = math.sqrt(variance)

            features[3] = mean
            features[4] = std_dev
            features[5] = max(all_scores)
            features[6] = min(all_scores)

        return features

    def _extract_model_features(
        self,
        result_set: dict[str, list[dict[str, Any]]],
    ) -> list[float]:
        """Extract per-model features."""
        features = [0.0] * self.num_features

        records = result_set.get("records", [])
        features[0] = math.log(len(records) + 1) / 10.0

        # Score statistics
        scores = [r.get("score", 0) for r in records if "score" in r]
        if scores:
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)

            features[1] = mean
            features[2] = math.sqrt(variance)
            features[3] = max(scores)
            features[4] = min(scores)

        # Model type encoding
        source = result_set.get("source", "")
        if source == "vector":
            features[7] = 0.25
        elif source == "document":
            features[7] = 0.50
        elif source == "graph":
            features[7] = 0.75
        else:
            features[7] = 1.0

        return features

    def _extract_record_features(
        self,
        results: list[dict[str, list[dict[str, Any]]]],
    ) -> dict[str, list[float]]:
        """Extract per-record features."""
        record_features: dict[str, list[float]] = {}

        for result_set in results:
            records = result_set.get("records", [])
            for rank, record in enumerate(records):
                record_id = record.get("id", str(id(record)))

                if record_id not in record_features:
                    record_features[record_id] = [0.0] * self.num_features

                features = record_features[record_id]

                # Reciprocal rank
                features[0] += 1.0 / (rank + 1.0)

                # Score
                if "score" in record:
                    features[1] += record["score"]

                # Appearance count
                features[2] += 1.0

        # Normalize
        for features in record_features.values():
            appearances = max(features[2], 1.0)
            features[0] /= appearances
            features[1] /= appearances
            features[2] /= len(results)

        return record_features

    def _extract_interaction_features(
        self,
        results: list[dict[str, list[dict[str, Any]]]],
    ) -> list[float]:
        """Extract cross-model interaction features."""
        features = [0.0] * self.num_features

        # Compute overlap between result sets
        id_sets = []
        for result_set in results:
            ids = {r.get("id", str(id(r))) for r in result_set.get("records", [])}
            id_sets.append(ids)

        # Pairwise Jaccard similarity
        overlap_sum = 0.0
        pair_count = 0

        for i in range(len(id_sets)):
            for j in range(i + 1, len(id_sets)):
                intersection = id_sets[i] & id_sets[j]
                union = id_sets[i] | id_sets[j]

                if union:
                    overlap_sum += len(intersection) / len(union)
                pair_count += 1

        features[0] = overlap_sum / pair_count if pair_count > 0 else 0.0

        return features
