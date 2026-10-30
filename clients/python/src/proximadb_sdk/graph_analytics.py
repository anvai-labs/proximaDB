"""
Graph Analytics Module for ProximaDB Python SDK

Provides graph algorithms (PageRank, centrality, community detection),
semantic traversal with vector similarity, and pattern matching queries.

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _path_node_ids(result: Any) -> list[str] | None:
    """Node ids of a shortest-path answer, or None when there is no path.

    REST answers ``{"data": {"path": [...], "found": ...}}``; gRPC answers a
    ``GraphShortestPathResponse`` (``node_ids``, ``found``).
    """
    if result is None:
        return None
    if isinstance(result, dict):
        payload = result.get("data", result)
        path = payload.get("path") if isinstance(payload, dict) else None
    else:
        path = list(getattr(result, "node_ids", None) or [])
        if not getattr(result, "found", bool(path)):
            return None
    return list(path) if path else None


class GraphAlgorithm(str, Enum):
    """Supported graph algorithms"""

    PAGERANK = "pagerank"
    BETWEENNESS_CENTRALITY = "betweenness_centrality"
    CLOSENESS_CENTRALITY = "closeness_centrality"
    DEGREE_CENTRALITY = "degree_centrality"
    EIGENVECTOR_CENTRALITY = "eigenvector_centrality"
    LOUVAIN = "louvain"
    LABEL_PROPAGATION = "label_propagation"
    CONNECTED_COMPONENTS = "connected_components"
    STRONGLY_CONNECTED = "strongly_connected"
    TRIANGLE_COUNT = "triangle_count"
    CLUSTERING_COEFFICIENT = "clustering_coefficient"
    SHORTEST_PATH = "shortest_path"
    ALL_PAIRS_SHORTEST = "all_pairs_shortest"
    MINIMUM_SPANNING_TREE = "minimum_spanning_tree"


class TraversalDirection(str, Enum):
    """Direction of graph traversal"""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


class PatternMatchMode(str, Enum):
    """Pattern matching mode"""

    EXACT = "exact"
    PARTIAL = "partial"
    FUZZY = "fuzzy"


@dataclass
class AlgorithmConfig:
    """Configuration for graph algorithms"""

    # PageRank settings
    damping_factor: float = 0.85
    max_iterations: int = 100
    convergence_threshold: float = 1e-6

    # Community detection settings
    resolution: float = 1.0
    random_seed: int | None = None

    # Centrality settings
    normalized: bool = True
    weight_property: str | None = None

    # Path settings
    max_depth: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "damping_factor": self.damping_factor,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "resolution": self.resolution,
            "random_seed": self.random_seed,
            "normalized": self.normalized,
            "weight_property": self.weight_property,
            "max_depth": self.max_depth,
        }


@dataclass
class SemanticTraversalConfig:
    """Configuration for semantic graph traversal"""

    # Vector similarity settings
    similarity_threshold: float = 0.7
    vector_field: str = "embedding"

    # Traversal settings
    max_depth: int = 3
    direction: TraversalDirection = TraversalDirection.OUTGOING
    edge_types: list[str] | None = None

    # Result settings
    limit: int = 100
    include_scores: bool = True
    include_paths: bool = False

    # Filtering
    node_label_filter: list[str] | None = None
    property_filters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "vector_field": self.vector_field,
            "max_depth": self.max_depth,
            "direction": self.direction.value,
            "edge_types": self.edge_types,
            "limit": self.limit,
            "include_scores": self.include_scores,
            "include_paths": self.include_paths,
            "node_label_filter": self.node_label_filter,
            "property_filters": self.property_filters,
        }


@dataclass
class PatternElement:
    """Element in a graph pattern"""

    variable: str
    label: str | None = None
    properties: dict[str, Any] | None = None

    def to_cypher(self) -> str:
        """Convert to Cypher-like pattern notation"""
        parts = [f"({self.variable}"]
        if self.label:
            parts.append(f":{self.label}")
        if self.properties:
            props = ", ".join(
                f"{k}: {json.dumps(v)}" for k, v in self.properties.items()
            )
            parts.append(f" {{{props}}}")
        parts.append(")")
        return "".join(parts)


@dataclass
class RelationshipPattern:
    """Relationship pattern in graph matching"""

    source: PatternElement
    target: PatternElement
    relationship_type: str | None = None
    relationship_var: str | None = None
    direction: TraversalDirection = TraversalDirection.OUTGOING
    min_hops: int = 1
    max_hops: int = 1
    properties: dict[str, Any] | None = None

    def to_cypher(self) -> str:
        """Convert to Cypher-like pattern notation"""
        rel_parts = ["-["]
        if self.relationship_var:
            rel_parts.append(self.relationship_var)
        if self.relationship_type:
            rel_parts.append(f":{self.relationship_type}")
        if self.min_hops != 1 or self.max_hops != 1:
            if self.min_hops == self.max_hops:
                rel_parts.append(f"*{self.min_hops}")
            else:
                rel_parts.append(f"*{self.min_hops}..{self.max_hops}")
        if self.properties:
            props = ", ".join(
                f"{k}: {json.dumps(v)}" for k, v in self.properties.items()
            )
            rel_parts.append(f" {{{props}}}")
        rel_parts.append("]-")

        arrow = ">" if self.direction == TraversalDirection.OUTGOING else ""
        if self.direction == TraversalDirection.INCOMING:
            return f"<{self.source.to_cypher()}{''.join(rel_parts)}{self.target.to_cypher()}"
        return f"{self.source.to_cypher()}{''.join(rel_parts)}{arrow}{self.target.to_cypher()}"


@dataclass
class GraphPattern:
    """Complete graph pattern for matching"""

    patterns: list[PatternElement | RelationshipPattern] = field(default_factory=list)
    where_clauses: list[str] = field(default_factory=list)
    return_variables: list[str] = field(default_factory=list)
    order_by: str | None = None
    limit: int | None = None

    def match(self, element: PatternElement) -> "GraphPattern":
        """Add a node pattern to match"""
        self.patterns.append(element)
        return self

    def relationship(
        self,
        source: PatternElement,
        target: PatternElement,
        rel_type: str | None = None,
        direction: TraversalDirection = TraversalDirection.OUTGOING,
        min_hops: int = 1,
        max_hops: int = 1,
    ) -> "GraphPattern":
        """Add a relationship pattern"""
        self.patterns.append(
            RelationshipPattern(
                source=source,
                target=target,
                relationship_type=rel_type,
                direction=direction,
                min_hops=min_hops,
                max_hops=max_hops,
            )
        )
        return self

    def where(self, clause: str) -> "GraphPattern":
        """Add a where clause"""
        self.where_clauses.append(clause)
        return self

    def returns(self, *variables: str) -> "GraphPattern":
        """Specify return variables"""
        self.return_variables.extend(variables)
        return self

    def order(self, by: str) -> "GraphPattern":
        """Add ORDER BY clause"""
        self.order_by = by
        return self

    def with_limit(self, n: int) -> "GraphPattern":
        """Add LIMIT clause"""
        self.limit = n
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert pattern to dictionary for API"""
        return {
            "patterns": [
                p.to_cypher() if hasattr(p, "to_cypher") else str(p)
                for p in self.patterns
            ],
            "where": self.where_clauses,
            "return": self.return_variables,
            "order_by": self.order_by,
            "limit": self.limit,
        }


@dataclass
class AlgorithmResult:
    """Result from a graph algorithm execution"""

    algorithm: GraphAlgorithm
    node_scores: dict[str, float] | None = None
    communities: dict[str, int] | None = None
    paths: list[list[str]] | None = None
    components: list[list[str]] | None = None
    statistics: dict[str, Any] | None = None
    execution_time_ms: float = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlgorithmResult":
        return cls(
            algorithm=GraphAlgorithm(data.get("algorithm", "pagerank")),
            node_scores=data.get("node_scores"),
            communities=data.get("communities"),
            paths=data.get("paths"),
            components=data.get("components"),
            statistics=data.get("statistics"),
            execution_time_ms=data.get("execution_time_ms", 0),
        )


@dataclass
class SemanticTraversalResult:
    """Result from semantic graph traversal"""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]] | None = None
    paths: list[list[str]] | None = None
    scores: dict[str, float] | None = None
    total_count: int = 0
    execution_time_ms: float = 0


@dataclass
class PatternMatchResult:
    """Result from pattern matching"""

    matches: list[dict[str, Any]]
    total_count: int = 0
    execution_time_ms: float = 0


class GraphAnalytics:
    """
    Graph analytics client for ProximaDB.

    Provides graph algorithms, semantic traversal, and pattern matching.

    Example:
        >>> from proximadb_sdk import ProximaDBClient
        >>> from proximadb_sdk.graph_analytics import GraphAnalytics, GraphAlgorithm
        >>>
        >>> client = ProximaDBClient("http://localhost:5678")
        >>> analytics = GraphAnalytics(client)
        >>>
        >>> # Run PageRank
        >>> result = analytics.run_algorithm(
        ...     "social_network",
        ...     GraphAlgorithm.PAGERANK,
        ...     AlgorithmConfig(damping_factor=0.85, max_iterations=20)
        ... )
        >>> print(result.node_scores)
        >>>
        >>> # Semantic traversal with vector similarity
        >>> nodes = analytics.semantic_traverse(
        ...     "knowledge_graph",
        ...     start_node="concept_ai",
        ...     query_vector=[0.1, 0.2, ...],
        ...     config=SemanticTraversalConfig(
        ...         similarity_threshold=0.8,
        ...         max_depth=2
        ...     )
        ... )
    """

    def __init__(self, client):
        """
        Initialize graph analytics with a ProximaDB client.

        Args:
            client: ProximaDBClient instance
        """
        self._client = client
        self._base_url = getattr(
            client,
            "_base_url",
            client.url if hasattr(client, "url") else "http://localhost:5678",
        )

    def run_algorithm(
        self,
        graph_id: str,
        algorithm: GraphAlgorithm,
        config: AlgorithmConfig | None = None,
        node_subset: list[str] | None = None,
    ) -> AlgorithmResult:
        """
        Run a graph algorithm on the specified graph.

        Args:
            graph_id: ID of the graph
            algorithm: Algorithm to run
            config: Algorithm configuration
            node_subset: Optional subset of nodes to run on

        Returns:
            AlgorithmResult with scores, communities, or paths

        Example:
            >>> result = analytics.run_algorithm(
            ...     "social",
            ...     GraphAlgorithm.PAGERANK,
            ...     AlgorithmConfig(damping_factor=0.85)
            ... )
            >>> top_nodes = sorted(
            ...     result.node_scores.items(),
            ...     key=lambda x: x[1],
            ...     reverse=True
            ... )[:10]
        """
        config = config or AlgorithmConfig()

        payload = {
            "graph_id": graph_id,
            "algorithm": algorithm.value,
            "config": config.to_dict(),
        }
        if node_subset:
            payload["node_subset"] = node_subset

        # Use client's internal request method if available
        if hasattr(self._client, "_make_request"):
            response = self._client._make_request(
                "POST",
                f"/v1/graphs/{graph_id}/algorithms/{algorithm.value}",
                json=payload,
            )
        else:
            # Fallback to direct HTTP
            import requests

            response = requests.post(
                f"{self._base_url}/v1/graphs/{graph_id}/algorithms/{algorithm.value}",
                json=payload,
            ).json()

        return AlgorithmResult.from_dict(response)

    def pagerank(
        self,
        graph_id: str,
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
    ) -> dict[str, float]:
        """
        Compute PageRank scores for all nodes in the graph.

        Args:
            graph_id: ID of the graph
            damping_factor: Probability of following an edge (default 0.85)
            max_iterations: Maximum number of iterations
            convergence_threshold: Stop when scores change less than this

        Returns:
            Dictionary mapping node IDs to PageRank scores

        Example:
            >>> scores = analytics.pagerank("social_network")
            >>> for node, score in sorted(scores.items(), key=lambda x: -x[1])[:5]:
            ...     print(f"{node}: {score:.4f}")
        """
        result = self.run_algorithm(
            graph_id,
            GraphAlgorithm.PAGERANK,
            AlgorithmConfig(
                damping_factor=damping_factor,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
            ),
        )
        return result.node_scores or {}

    def centrality(
        self,
        graph_id: str,
        centrality_type: str = "betweenness",
        normalized: bool = True,
        weight_property: str | None = None,
    ) -> dict[str, float]:
        """
        Compute centrality scores for nodes.

        Args:
            graph_id: ID of the graph
            centrality_type: Type of centrality ("betweenness", "closeness", "degree", "eigenvector")
            normalized: Whether to normalize scores
            weight_property: Edge property to use as weight

        Returns:
            Dictionary mapping node IDs to centrality scores

        Example:
            >>> betweenness = analytics.centrality("network", "betweenness")
            >>> closeness = analytics.centrality("network", "closeness")
        """
        algorithm_map = {
            "betweenness": GraphAlgorithm.BETWEENNESS_CENTRALITY,
            "closeness": GraphAlgorithm.CLOSENESS_CENTRALITY,
            "degree": GraphAlgorithm.DEGREE_CENTRALITY,
            "eigenvector": GraphAlgorithm.EIGENVECTOR_CENTRALITY,
        }

        algorithm = algorithm_map.get(centrality_type)
        if not algorithm:
            raise ValueError(f"Unknown centrality type: {centrality_type}")

        result = self.run_algorithm(
            graph_id,
            algorithm,
            AlgorithmConfig(
                normalized=normalized,
                weight_property=weight_property,
            ),
        )
        return result.node_scores or {}

    def community_detection(
        self,
        graph_id: str,
        algorithm: str = "louvain",
        resolution: float = 1.0,
        random_seed: int | None = None,
    ) -> dict[str, int]:
        """
        Detect communities in the graph.

        Args:
            graph_id: ID of the graph
            algorithm: Algorithm to use ("louvain" or "label_propagation")
            resolution: Resolution parameter for Louvain (higher = more communities)
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary mapping node IDs to community IDs

        Example:
            >>> communities = analytics.community_detection("social", "louvain")
            >>> community_sizes = {}
            >>> for node, comm in communities.items():
            ...     community_sizes[comm] = community_sizes.get(comm, 0) + 1
        """
        algo = (
            GraphAlgorithm.LOUVAIN
            if algorithm == "louvain"
            else GraphAlgorithm.LABEL_PROPAGATION
        )

        result = self.run_algorithm(
            graph_id,
            algo,
            AlgorithmConfig(
                resolution=resolution,
                random_seed=random_seed,
            ),
        )
        return result.communities or {}

    def connected_components(
        self,
        graph_id: str,
        strongly_connected: bool = False,
    ) -> list[list[str]]:
        """
        Find connected components in the graph.

        Args:
            graph_id: ID of the graph
            strongly_connected: If True, find strongly connected components

        Returns:
            List of components, each component is a list of node IDs

        Example:
            >>> components = analytics.connected_components("network")
            >>> print(f"Found {len(components)} components")
            >>> largest = max(components, key=len)
        """
        algo = (
            GraphAlgorithm.STRONGLY_CONNECTED
            if strongly_connected
            else GraphAlgorithm.CONNECTED_COMPONENTS
        )

        result = self.run_algorithm(graph_id, algo)
        return result.components or []

    def shortest_path(
        self,
        graph_id: str,
        source: str,
        target: str,
        weight_property: str | None = None,
        edge_types: list[str] | None = None,
    ) -> list[str] | None:
        """
        Find the shortest path between two nodes.

        Args:
            graph_id: ID of the graph
            source: Source node ID
            target: Target node ID
            weight_property: Edge property to use as weight
            edge_types: Only consider these edge types

        Returns:
            List of node IDs in the path, or None if no path exists

        Example:
            >>> path = analytics.shortest_path("social", "alice", "bob")
            >>> if path:
            ...     print(" -> ".join(path))
        """
        # Use client's existing shortest_path if available. TD-SDKDOC-1: this
        # used to pass start_node=/end_node= (the client takes positional
        # start/target ids), so it raised TypeError on every real client.
        if hasattr(self._client, "graph_shortest_path"):
            result = self._client.graph_shortest_path(
                source, target, edge_types=edge_types, graph_id=graph_id
            )
            return _path_node_ids(result)

        result = self.run_algorithm(
            graph_id,
            GraphAlgorithm.SHORTEST_PATH,
            AlgorithmConfig(weight_property=weight_property),
        )
        paths = result.paths or []
        return paths[0] if paths else None

    def semantic_traverse(
        self,
        graph_id: str,
        start_node: str,
        query_vector: list[float],
        config: SemanticTraversalConfig | None = None,
        collection_id: str | None = None,
    ) -> SemanticTraversalResult:
        """
        Traverse the graph using vector similarity to guide traversal.

        Combines graph structure with vector embeddings to find semantically
        relevant nodes through relationship paths.

        Args:
            graph_id: ID of the graph to traverse
            start_node: Starting node ID
            query_vector: Query embedding vector
            config: Traversal configuration
            collection_id: Vector collection ID (if vectors stored separately)

        Returns:
            SemanticTraversalResult with matched nodes and optional paths

        Example:
            >>> # Find related concepts similar to a query
            >>> result = analytics.semantic_traverse(
            ...     "knowledge_graph",
            ...     "concept_machine_learning",
            ...     query_vector=embedding_model.encode("neural networks"),
            ...     config=SemanticTraversalConfig(
            ...         similarity_threshold=0.75,
            ...         max_depth=2,
            ...         edge_types=["RELATED_TO", "SIMILAR_TO"]
            ...     )
            ... )
            >>> for node in result.nodes:
            ...     print(f"{node['id']}: {node.get('similarity', 0):.3f}")
        """
        config = config or SemanticTraversalConfig()

        payload = {
            "graph_id": graph_id,
            "start_node": start_node,
            "query_vector": query_vector,
            "config": config.to_dict(),
        }
        if collection_id:
            payload["collection_id"] = collection_id

        if hasattr(self._client, "_make_request"):
            response = self._client._make_request(
                "POST", f"/v1/graphs/{graph_id}/semantic-traverse", json=payload
            )
        else:
            import requests

            response = requests.post(
                f"{self._base_url}/v1/graphs/{graph_id}/semantic-traverse", json=payload
            ).json()

        return SemanticTraversalResult(
            nodes=response.get("nodes", []),
            edges=response.get("edges"),
            paths=response.get("paths"),
            scores=response.get("scores"),
            total_count=response.get("total_count", 0),
            execution_time_ms=response.get("execution_time_ms", 0),
        )

    def semantic_neighbors(
        self,
        graph_id: str,
        node_id: str,
        query_vector: list[float] | None = None,
        similarity_threshold: float = 0.7,
        max_neighbors: int = 10,
        edge_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find neighbors of a node filtered by vector similarity.

        Args:
            graph_id: ID of the graph
            node_id: Node to find neighbors for
            query_vector: Optional query vector (uses node's embedding if not provided)
            similarity_threshold: Minimum similarity score
            max_neighbors: Maximum number of neighbors to return
            edge_types: Only consider these edge types

        Returns:
            List of neighbor nodes with similarity scores

        Example:
            >>> neighbors = analytics.semantic_neighbors(
            ...     "knowledge",
            ...     "concept_ai",
            ...     similarity_threshold=0.8
            ... )
        """
        config = SemanticTraversalConfig(
            similarity_threshold=similarity_threshold,
            max_depth=1,
            limit=max_neighbors,
            edge_types=edge_types,
            include_scores=True,
        )

        # If no query vector, we'll let the server use the node's embedding
        if query_vector is None:
            query_vector = []  # Server will use node's embedding

        result = self.semantic_traverse(
            graph_id,
            node_id,
            query_vector,
            config,
        )
        return result.nodes

    def pattern_match(
        self,
        graph_id: str,
        pattern: GraphPattern,
        mode: PatternMatchMode = PatternMatchMode.EXACT,
        limit: int | None = None,
    ) -> PatternMatchResult:
        """
        Find subgraphs matching a pattern.

        Args:
            graph_id: ID of the graph
            pattern: GraphPattern to match
            mode: Matching mode (exact, partial, fuzzy)
            limit: Maximum number of matches

        Returns:
            PatternMatchResult with matching subgraphs

        Example:
            >>> # Find users who follow each other (mutual connections)
            >>> pattern = (
            ...     GraphPattern()
            ...     .relationship(
            ...         PatternElement("a", "User"),
            ...         PatternElement("b", "User"),
            ...         rel_type="FOLLOWS"
            ...     )
            ...     .relationship(
            ...         PatternElement("b"),
            ...         PatternElement("a"),
            ...         rel_type="FOLLOWS"
            ...     )
            ...     .returns("a", "b")
            ... )
            >>> result = analytics.pattern_match("social", pattern)
        """
        if limit:
            pattern.limit = limit

        payload = {
            "graph_id": graph_id,
            "pattern": pattern.to_dict(),
            "mode": mode.value,
        }

        if hasattr(self._client, "_make_request"):
            response = self._client._make_request(
                "POST", f"/v1/graphs/{graph_id}/pattern-match", json=payload
            )
        else:
            import requests

            response = requests.post(
                f"{self._base_url}/v1/graphs/{graph_id}/pattern-match", json=payload
            ).json()

        return PatternMatchResult(
            matches=response.get("matches", []),
            total_count=response.get("total_count", 0),
            execution_time_ms=response.get("execution_time_ms", 0),
        )

    def find_triangles(
        self,
        graph_id: str,
        node_id: str | None = None,
    ) -> list[list[str]]:
        """
        Find triangles in the graph.

        Args:
            graph_id: ID of the graph
            node_id: If provided, only find triangles containing this node

        Returns:
            List of triangles, each a list of 3 node IDs

        Example:
            >>> triangles = analytics.find_triangles("social")
            >>> print(f"Found {len(triangles)} triangles")
        """
        result = self.run_algorithm(
            graph_id,
            GraphAlgorithm.TRIANGLE_COUNT,
            node_subset=[node_id] if node_id else None,
        )
        return result.paths or []

    def clustering_coefficient(
        self,
        graph_id: str,
        node_id: str | None = None,
    ) -> float | dict[str, float]:
        """
        Compute clustering coefficient.

        Args:
            graph_id: ID of the graph
            node_id: If provided, compute for this node only

        Returns:
            Global coefficient or dictionary of node coefficients

        Example:
            >>> global_cc = analytics.clustering_coefficient("social")
            >>> print(f"Global clustering: {global_cc:.4f}")
        """
        result = self.run_algorithm(
            graph_id,
            GraphAlgorithm.CLUSTERING_COEFFICIENT,
            node_subset=[node_id] if node_id else None,
        )
        if node_id:
            return result.node_scores.get(node_id, 0.0) if result.node_scores else 0.0
        return (
            result.statistics.get("global_coefficient", 0.0)
            if result.statistics
            else 0.0
        )


# Convenience function for creating pattern elements
def node(variable: str, label: str | None = None, **properties) -> PatternElement:
    """
    Create a pattern element for graph matching.

    Args:
        variable: Variable name for the node
        label: Optional node label
        **properties: Node properties to match

    Returns:
        PatternElement instance

    Example:
        >>> pattern = GraphPattern().match(node("u", "User", active=True))
    """
    return PatternElement(variable, label, properties if properties else None)


def relationship(
    source: PatternElement,
    target: PatternElement,
    rel_type: str | None = None,
    direction: TraversalDirection = TraversalDirection.OUTGOING,
    **properties,
) -> RelationshipPattern:
    """
    Create a relationship pattern for graph matching.

    Args:
        source: Source node pattern
        target: Target node pattern
        rel_type: Relationship type
        direction: Direction of the relationship
        **properties: Relationship properties to match

    Returns:
        RelationshipPattern instance

    Example:
        >>> rel = relationship(node("a"), node("b"), "FOLLOWS")
    """
    return RelationshipPattern(
        source=source,
        target=target,
        relationship_type=rel_type,
        direction=direction,
        properties=properties if properties else None,
    )


__all__ = [
    # Main class
    "GraphAnalytics",
    # Configuration
    "AlgorithmConfig",
    "SemanticTraversalConfig",
    # Patterns
    "GraphPattern",
    "PatternElement",
    "RelationshipPattern",
    # Results
    "AlgorithmResult",
    "SemanticTraversalResult",
    "PatternMatchResult",
    # Enums
    "GraphAlgorithm",
    "TraversalDirection",
    "PatternMatchMode",
    # Convenience functions
    "node",
    "relationship",
]
