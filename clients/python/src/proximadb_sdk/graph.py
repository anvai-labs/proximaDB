"""
ProximaDB Graph API Module

High-level graph operations for Python SDK.
Provides Cypher-like query interface, batch operations, and pattern matching.

This module fills the gap between low-level node/edge CRUD and Victor's needs
for call graph and dependency tracking.

Example:
    from proximadb_sdk import ProximaDBClient
    from proximadb_sdk.graph import ProximaDBGraph

    client = ProximaDBClient(url="http://localhost:5678")
    graph = ProximaDBGraph(client, "myrepo_graph")

    # Batch operations (performance-critical for code indexing)
    graph.batch_create_nodes([
        {"id": "func:main", "labels": ["Function"], "properties": {"name": "main"}},
        {"id": "func:parse", "labels": ["Function"], "properties": {"name": "parse"}},
    ])

    # Create edges
    graph.batch_create_edges([
        {"from": "func:main", "to": "func:parse", "type": "CALLS", "properties": {"line": 42}},
    ])

    # Cypher-like query
    results = graph.query_cypher(
        "MATCH (c:Function)-[:CALLS]->(f:Function) WHERE c.name = 'main' RETURN c, f"
    )

    # Find callers (reverse traversal)
    callers = graph.find_callers("func:parse_json")
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ProximaDBNotSupportedError


@dataclass
class GraphNode:
    """Graph node representation.

    Attributes:
        id: Unique node identifier
        labels: List of node labels/types
        properties: Node properties as key-value pairs
        embedding: Optional embedding vector for semantic search
    """

    id: str
    labels: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API calls."""
        return {
            "id": self.id,
            "labels": self.labels,
            "properties": self.properties,
        }


@dataclass
class GraphEdge:
    """Graph edge representation.

    Attributes:
        id: Unique edge identifier
        from_node: Source node ID
        to_node: Target node ID
        edge_type: Relationship type (e.g., "CALLS", "IMPORTS")
        properties: Edge properties as key-value pairs
        weight: Optional weight for weighted graphs
    """

    id: str
    from_node: str
    to_node: str
    edge_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    weight: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API calls."""
        result = {
            "id": self.id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "edge_type": self.edge_type,
            "properties": self.properties,
        }
        if self.weight is not None:
            result["weight"] = self.weight
        return result


@dataclass
class GraphPath:
    """Path representation for traversal results.

    Attributes:
        nodes: List of node IDs in path order
        edges: List of edge IDs in path order
        total_weight: Sum of edge weights
    """

    nodes: list[str] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)
    total_weight: float = 0.0


@dataclass
class GraphQueryResult:
    """Result from graph query operations.

    Attributes:
        nodes: List of nodes in result
        edges: List of edges in result
        paths: List of traversal paths (if applicable)
        stats: Query execution statistics
    """

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class ProximaDBGraph:
    """High-level ProximaDB graph operations interface.

    Provides simplified API for common graph operations needed by Victor:
    - Batch node/edge creation (performance-critical)
    - Cypher-like query interface
    - Reverse traversal (find callers)
    - Pattern matching

    Args:
        client: ProximaDB client instance
        graph_id: Graph collection identifier
    """

    def __init__(self, client: Any, graph_id: str):
        """Initialize graph interface.

        Args:
            client: ProximaDBClient instance
            graph_id: Graph collection ID
        """
        self._client = client
        self._graph_id = graph_id

    @staticmethod
    def _is_internal_label(label: str) -> bool:
        return label.startswith("__")

    @staticmethod
    def _normalize_node(node: Any) -> GraphNode | None:
        if node is None:
            return None
        if isinstance(node, GraphNode):
            return node
        if isinstance(node, dict):
            return GraphNode(
                id=node.get("id", ""),
                labels=list(node.get("labels", []) or []),
                properties=dict(node.get("properties", {}) or {}),
            )
        return GraphNode(
            id=getattr(node, "id", ""),
            labels=list(getattr(node, "labels", []) or []),
            properties=dict(getattr(node, "properties", {}) or {}),
        )

    @staticmethod
    def _normalize_edge(edge: Any) -> GraphEdge | None:
        if edge is None:
            return None
        if isinstance(edge, GraphEdge):
            return edge
        if isinstance(edge, dict):
            return GraphEdge(
                id=edge.get("id", ""),
                from_node=edge.get("from_node_id") or edge.get("from_node") or "",
                to_node=edge.get("to_node_id") or edge.get("to_node") or "",
                edge_type=edge.get("edge_type", edge.get("type", "")),
                properties=dict(edge.get("properties", {}) or {}),
                weight=edge.get("weight"),
            )
        return GraphEdge(
            id=getattr(edge, "id", ""),
            from_node=getattr(edge, "from_node_id", None)
            or getattr(edge, "from_node", ""),
            to_node=getattr(edge, "to_node_id", None) or getattr(edge, "to_node", ""),
            edge_type=getattr(edge, "edge_type", ""),
            properties=dict(getattr(edge, "properties", {}) or {}),
            weight=getattr(edge, "weight", None),
        )

    def _query_nodes_raw(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        result = self._client.query_nodes(
            graph_id=self._graph_id,
            labels=labels,
            properties=properties,
            limit=limit,
            offset=offset,
        )
        if isinstance(result, dict):
            return list(result.get("nodes", []) or [])
        return []

    def _get_node_raw(self, node_id: str) -> dict[str, Any] | None:
        try:
            result = self._client.get_node(node_id=node_id, graph_id=self._graph_id)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

        for node in self._query_nodes_raw(limit=100000):
            if node.get("id") == node_id:
                return node
        return None

    def _get_outgoing_edges_raw(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            edges = self._client.get_outgoing_edges(
                node_id=node_id,
                edge_types=edge_types,
                graph_id=self._graph_id,
            )
            return [
                dict(edge) if isinstance(edge, dict) else edge for edge in (edges or [])
            ]
        except Exception as direct_exc:  # noqa: BLE001 - deliberate fallback below
            # Legitimate: not every server exposes get_outgoing_edges, so fall
            # back to a depth-1 traversal. Kept for the chain on the raise below.
            first_error = direct_exc

        try:
            traversal = self._client.traverse_graph(
                graph_id=self._graph_id,
                start_node_id=node_id,
                max_depth=1,
                edge_types=edge_types,
                limit=None,
            )
        except Exception as exc:
            # BOTH ways of reading edges failed. Returning [] here silently
            # TRUNCATED every traversal built on this primitive -- three callers
            # walk it -- and "this node has no outgoing edges" is a perfectly
            # ordinary answer, so nothing downstream could tell the difference.
            # A private primitive should not decide that policy for its callers.
            raise RuntimeError(
                f"could not read outgoing edges for node {node_id!r} in graph "
                f"{self._graph_id!r}: get_outgoing_edges failed ({first_error!r}) "
                f"and the traverse_graph fallback failed too"
            ) from exc

        edges = []
        for edge in traversal.get("edges", []) or []:
            normalized = self._normalize_edge(edge)
            if normalized and normalized.from_node == node_id:
                edges.append(normalized.to_dict())
        return edges

    def _get_incoming_edges_raw(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            edges = self._client.get_incoming_edges(
                node_id=node_id,
                edge_types=edge_types,
                graph_id=self._graph_id,
            )
            return [
                dict(edge) if isinstance(edge, dict) else edge for edge in (edges or [])
            ]
        except Exception:
            pass

        incoming: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for node in self.get_all_nodes(include_internal=True):
            for edge in self._get_outgoing_edges_raw(node.id, edge_types=edge_types):
                normalized = self._normalize_edge(edge)
                if normalized is None or normalized.to_node != node_id:
                    continue
                signature = (
                    normalized.from_node,
                    normalized.to_node,
                    normalized.edge_type,
                )
                if signature in seen:
                    continue
                seen.add(signature)
                incoming.append(normalized.to_dict())
        return incoming

    @staticmethod
    def _normalize_json_node(raw_node: dict[str, Any]) -> dict[str, Any] | None:
        node_id = raw_node.get("id") or raw_node.get("node_id") or raw_node.get("key")
        if not node_id:
            return None

        labels = raw_node.get("labels")
        if isinstance(labels, str):
            labels = [labels]
        if not labels:
            node_type = raw_node.get("type") or raw_node.get("kind")
            labels = [str(node_type)] if node_type else []

        properties = dict(raw_node.get("properties", {}) or {})
        for key, value in raw_node.items():
            if key in {"id", "node_id", "key", "labels", "properties"}:
                continue
            properties.setdefault(key, value)

        return {
            "id": str(node_id),
            "labels": [str(label) for label in labels],
            "properties": properties,
        }

    @staticmethod
    def _normalize_json_edge(
        raw_edge: dict[str, Any], index: int
    ) -> dict[str, Any] | None:
        from_node = (
            raw_edge.get("from_node_id")
            or raw_edge.get("from_node")
            or raw_edge.get("source")
            or raw_edge.get("src")
            or raw_edge.get("from")
        )
        to_node = (
            raw_edge.get("to_node_id")
            or raw_edge.get("to_node")
            or raw_edge.get("target")
            or raw_edge.get("dst")
            or raw_edge.get("to")
        )
        if not from_node or not to_node:
            return None

        edge_type = (
            raw_edge.get("edge_type")
            or raw_edge.get("type")
            or raw_edge.get("label")
            or "RELATED_TO"
        )
        edge_id = (
            raw_edge.get("id") or f"edge_{index}_{from_node}_{to_node}_{edge_type}"
        )

        properties = dict(raw_edge.get("properties", {}) or {})
        for key, value in raw_edge.items():
            if key in {
                "id",
                "from_node_id",
                "from_node",
                "source",
                "src",
                "from",
                "to_node_id",
                "to_node",
                "target",
                "dst",
                "to",
                "edge_type",
                "type",
                "label",
                "properties",
                "weight",
            }:
                continue
            properties.setdefault(key, value)

        result = {
            "id": str(edge_id),
            "from_node_id": str(from_node),
            "to_node_id": str(to_node),
            "edge_type": str(edge_type),
            "properties": properties,
        }
        if raw_edge.get("weight") is not None:
            result["weight"] = raw_edge.get("weight")
        return result

    # ========================================================================
    # Batch Operations (Performance-Critical for Victor)
    # ========================================================================

    def batch_create_nodes(
        self,
        nodes: list[GraphNode | dict[str, Any]],
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """Create multiple nodes in a single operation.

        This is **critical for Victor performance** when indexing codebases
        with thousands of functions.

        Args:
            nodes: List of nodes to create (GraphNode or dict)
            batch_size: Number of nodes per batch request

        Returns:
            Creation result with success status and statistics

        Example:
            graph.batch_create_nodes([
                {"id": "func:main", "labels": ["Function"], "properties": {"name": "main"}},
                {"id": "func:parse", "labels": ["Function"], "properties": {"name": "parse"}},
            ])
        """
        if not nodes:
            return {"success": True, "created": 0}

        # Convert GraphNode objects to dicts if needed
        node_dicts = []
        for node in nodes:
            if isinstance(node, GraphNode):
                node_dicts.append(node.to_dict())
            else:
                node_dicts.append(node)

        # Process in batches. Prefer the true batch endpoint (one request per
        # page) — the per-item loop cost 1,000 round trips per 1,000 nodes and
        # dominated repo-scale ingest (5.2x slower than the sqlite baseline).
        total_created = 0
        failed_count = 0
        failed: list[dict[str, Any]] = []
        batch_method = getattr(self._client, "batch_create_nodes", None)

        for i in range(0, len(node_dicts), batch_size):
            batch = node_dicts[i : i + batch_size]
            if callable(batch_method):
                try:
                    resp = batch_method(nodes=batch, graph_id=self._graph_id)
                    data = resp.get("data", resp) if isinstance(resp, dict) else {}
                    created = data.get("created_count")
                    if created is None:
                        created = len(data.get("results", []) or data.get("nodes", []))
                    total_created += int(created or 0)
                    page_failed = int(data.get("failed_count") or 0)
                    failed_count += page_failed
                    for err in data.get("errors", []) or []:
                        failed.append({"batch": i // batch_size, "error": err})
                    continue
                except (AttributeError, ProximaDBNotSupportedError):
                    batch_method = None  # transport has no batch surface
                except Exception as e:
                    failed_count += len(batch)
                    failed.append({"batch": i // batch_size, "error": str(e)})
                    continue

            # Per-item fallback for transports without a batch endpoint.
            try:
                for node_dict in batch:
                    self._client.create_node(
                        graph_id=self._graph_id,
                        node_id=node_dict["id"],
                        labels=node_dict.get("labels", []),
                        properties=node_dict.get("properties", {}),
                    )
                    total_created += 1
            except Exception as e:
                failed_count += 1
                failed.append({"batch": i // batch_size, "error": str(e)})

        return {
            "success": failed_count == 0 and not failed,
            "created": total_created,
            "failed": failed_count if failed_count else len(failed),
            "errors": failed,
        }

    def batch_create_edges(
        self,
        edges: list[GraphEdge | dict[str, Any]],
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """Create multiple edges in a single operation.

        Critical for Victor when creating call graphs and import relationships.

        Args:
            edges: List of edges to create (GraphEdge or dict)
            batch_size: Number of edges per batch request

        Returns:
            Creation result with success status and statistics

        Example:
            graph.batch_create_edges([
                {"from": "func:main", "to": "func:parse", "type": "CALLS"},
                {"from": "func:main", "to": "func:validate", "type": "CALLS"},
            ])
        """
        if not edges:
            return {"success": True, "created": 0}

        # Convert GraphEdge objects to dicts if needed
        edge_dicts = []
        for edge in edges:
            if isinstance(edge, GraphEdge):
                edge_dicts.append(edge.to_dict())
            else:
                edge_dicts.append(edge)

        # Process in batches. Prefer the true batch endpoint — the server
        # applies per-edge admission (a bad edge rejects only itself; the
        # response's errors[] carries per-edge reasons) and one request
        # replaces 1,000.
        total_created = 0
        failed_count = 0
        failed: list[dict[str, Any]] = []
        batch_method = getattr(self._client, "batch_create_edges", None)

        for i in range(0, len(edge_dicts), batch_size):
            batch = edge_dicts[i : i + batch_size]

            if callable(batch_method):
                try:
                    normalized = []
                    for edge_offset, edge_dict in enumerate(batch):
                        from_node_id = (
                            edge_dict.get("from_node_id")
                            or edge_dict.get("from_node")
                            or edge_dict.get("from")
                        )
                        to_node_id = (
                            edge_dict.get("to_node_id")
                            or edge_dict.get("to_node")
                            or edge_dict.get("to")
                        )
                        edge_type = edge_dict.get("edge_type") or edge_dict.get("type")
                        if not from_node_id or not to_node_id or not edge_type:
                            raise ValueError(
                                "Edge batch items must include from/to node IDs and edge type"
                            )
                        item = {
                            "id": edge_dict.get("id", f"edge_{i + edge_offset}"),
                            "from_node_id": from_node_id,
                            "to_node_id": to_node_id,
                            "edge_type": edge_type,
                            "properties": edge_dict.get("properties", {}),
                        }
                        if edge_dict.get("weight") is not None:
                            item["weight"] = edge_dict.get("weight")
                        normalized.append(item)
                    resp = batch_method(edges=normalized, graph_id=self._graph_id)
                    data = resp.get("data", resp) if isinstance(resp, dict) else {}
                    created = data.get("created_count")
                    if created is None:
                        created = len(data.get("results", []) or data.get("edges", []))
                    total_created += int(created or 0)
                    failed_count += int(data.get("failed_count") or 0)
                    for err in data.get("errors", []) or []:
                        failed.append({"batch": i // batch_size, "error": err})
                    continue
                except (AttributeError, ProximaDBNotSupportedError):
                    batch_method = None
                except Exception as e:
                    failed_count += len(batch)
                    failed.append({"batch": i // batch_size, "error": str(e)})
                    continue

            try:
                for edge_offset, edge_dict in enumerate(batch):
                    from_node_id = (
                        edge_dict.get("from_node_id")
                        or edge_dict.get("from_node")
                        or edge_dict.get("from")
                    )
                    to_node_id = (
                        edge_dict.get("to_node_id")
                        or edge_dict.get("to_node")
                        or edge_dict.get("to")
                    )
                    edge_type = edge_dict.get("edge_type") or edge_dict.get("type")
                    if not from_node_id or not to_node_id or not edge_type:
                        raise ValueError(
                            "Edge batch items must include from/to node IDs and edge type"
                        )
                    self._client.create_edge(
                        graph_id=self._graph_id,
                        edge_id=edge_dict.get("id", f"edge_{i + edge_offset}"),
                        from_node_id=from_node_id,
                        to_node_id=to_node_id,
                        edge_type=edge_type,
                        properties=edge_dict.get("properties", {}),
                        weight=edge_dict.get("weight"),
                    )
                    total_created += 1
            except Exception as e:
                failed_count += 1
                failed.append({"batch": i // batch_size, "error": str(e)})

        return {
            "success": failed_count == 0 and not failed,
            "created": total_created,
            "failed": failed_count if failed_count else len(failed),
            "errors": failed,
        }

    # ========================================================================
    # Query Operations
    # ========================================================================

    def query_cypher(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> GraphQueryResult:
        """Execute Cypher-like graph query.

        Supports a subset of openCypher:
        - MATCH patterns with nodes and relationships
        - WHERE clauses with property filters
        - RETURN projections
        - Basic traversals

        Args:
            query: Cypher query string
            params: Optional query parameters

        Returns:
            Graph query result with nodes, edges, and paths

        Example:
            results = graph.query_cypher(
                "MATCH (c:Function)-[:CALLS]->(f:Function) WHERE c.name = 'main' RETURN c, f"
            )
        """
        # Parse Cypher query
        parsed = self._parse_cypher(query)

        # Execute using available client methods
        # For now, use graph traversal and node query
        if parsed["match"]:
            # Extract starting conditions
            start_labels = parsed.get("start_labels", [])
            start_props = parsed.get("start_properties", {})

            # Query starting nodes
            start_nodes_result = self._client.query_nodes(
                graph_id=self._graph_id,
                labels=start_labels,
                properties=start_props,
            )

            # Traverse from start nodes
            if start_nodes_result.get("nodes"):
                start_node_ids = [n["id"] for n in start_nodes_result["nodes"]]
                return self._execute_traversal(start_node_ids, parsed)
            else:
                return GraphQueryResult()

        return GraphQueryResult()

    def _parse_cypher(self, query: str) -> dict[str, Any]:
        """Parse Cypher query into components.

        This is a simplified parser for common patterns.
        Full Cypher support would require a proper parser.
        """
        # Remove extra whitespace
        query = " ".join(query.split())

        result = {
            "match": False,
            "where": False,
            "return": False,
            "start_labels": [],
            "start_properties": {},
            "traversal": None,
        }

        # Check for MATCH clause
        match_match = re.search(
            r"MATCH\s+(.+?)(?:\s+WHERE|\s+RETURN|$)", query, re.IGNORECASE
        )
        if match_match:
            result["match"] = True
            match_pattern = match_match.group(1)

            # Parse node pattern: (variable:Label {property: value})
            node_pattern = re.search(
                r"\((\w+)(?::(\w+))?(?:\s*\{(.+)\})?\)", match_pattern
            )
            if node_pattern:
                if node_pattern.group(2):  # Label
                    result["start_labels"] = [node_pattern.group(2)]

                if node_pattern.group(3):  # Properties
                    # Parse properties: key: value, key2: value2
                    props_str = node_pattern.group(3)
                    for prop_match in re.finditer(r'(\w+)\s*:\s*"?(\w+)"?', props_str):
                        result["start_properties"][prop_match.group(1)] = (
                            prop_match.group(2)
                        )

            # Parse relationship pattern: -[r:TYPE]->
            rel_pattern = re.search(r"-?\[(\w+)(?::(\w+))?\]->", match_pattern)
            if rel_pattern:
                result["traversal"] = {
                    "variable": rel_pattern.group(1),
                    "type": rel_pattern.group(2),
                }

        # Check for WHERE clause
        where_match = re.search(r"WHERE\s+(.+?)(?:\s+RETURN|$)", query, re.IGNORECASE)
        if where_match:
            result["where"] = True
            where_clause = where_match.group(1)
            # Parse WHERE conditions
            # Simplified: handles "property = value" patterns
            prop_match = re.search(r"(\w+)\s*=\s*['\"]?([^'\"]+)['\"]?", where_clause)
            if prop_match:
                result["start_properties"][prop_match.group(1)] = prop_match.group(2)

        return result

    def _execute_traversal(
        self, start_node_ids: list[str], parsed_query: dict[str, Any]
    ) -> GraphQueryResult:
        """Execute graph traversal from start nodes."""
        result = GraphQueryResult()

        # Traverse from each start node
        for start_id in start_node_ids[:10]:  # Limit for now
            try:
                traversal_result = self._client.traverse_graph(
                    graph_id=self._graph_id,
                    start_node_id=start_id,
                    max_depth=1,  # Default depth
                    edge_types=(
                        [parsed_query["traversal"]["type"]]
                        if parsed_query.get("traversal")
                        else None
                    ),
                )

                if traversal_result.get("nodes"):
                    for node_data in traversal_result["nodes"]:
                        node = self._normalize_node(node_data)
                        if node is not None:
                            result.nodes.append(node)

                if traversal_result.get("edges"):
                    for edge_data in traversal_result["edges"]:
                        edge = self._normalize_edge(edge_data)
                        if edge is not None:
                            result.edges.append(edge)
            except Exception:
                # Log and continue
                pass

        return result

    # ========================================================================
    # Reverse Traversal (Find Callers)
    # ========================================================================

    def find_callers(
        self,
        node_id: str,
        edge_type: str = "CALLS",
        max_depth: int = 1,
    ) -> list[GraphNode]:
        """Find all nodes that have edges pointing to the target node.

        This is **critical for Victor** to find:
        - All functions that call a given function
        - All modules that import a given module
        - All classes that inherit from a given class

        Args:
            node_id: Target node ID
            edge_type: Edge type to traverse (default: "CALLS")
            max_depth: Maximum depth to traverse (default: 1)

        Returns:
            List of nodes that point to the target

        Example:
            # Find all functions that call parse_json
            callers = graph.find_callers("func:parse_json", edge_type="CALLS")
            for caller in callers:
                print(f"{caller.properties['name']}() calls parse_json()")
        """
        try:
            callers: list[GraphNode] = []
            seen_ids: set[str] = set()

            if max_depth <= 1:
                for edge in self._get_incoming_edges_raw(
                    node_id,
                    edge_types=[edge_type] if edge_type else None,
                ):
                    normalized = self._normalize_edge(edge)
                    if normalized is None:
                        continue
                    caller = self.get_node_by_id(normalized.from_node)
                    if caller is None or caller.id in seen_ids:
                        continue
                    seen_ids.add(caller.id)
                    callers.append(caller)
                return callers

            frontier = {node_id}
            for _ in range(max_depth):
                next_frontier: set[str] = set()
                for target_id in frontier:
                    for edge in self._get_incoming_edges_raw(
                        target_id,
                        edge_types=[edge_type] if edge_type else None,
                    ):
                        normalized = self._normalize_edge(edge)
                        if normalized is None:
                            continue
                        caller_id = normalized.from_node
                        if caller_id in seen_ids:
                            continue
                        caller = self.get_node_by_id(caller_id)
                        if caller is None:
                            continue
                        seen_ids.add(caller.id)
                        callers.append(caller)
                        next_frontier.add(caller.id)
                frontier = next_frontier
                if not frontier:
                    break

            return callers

        except Exception:
            # Log error and return empty list
            return []

    def get_node_by_id(self, node_id: str) -> GraphNode | None:
        """Get a graph node by ID."""
        return self._normalize_node(self._get_node_raw(node_id))

    def get_all_nodes(
        self,
        labels: list[str] | None = None,
        batch_size: int = 1000,
        include_internal: bool = False,
    ) -> list[GraphNode]:
        """Get all graph nodes with optional label filtering."""
        nodes: list[GraphNode] = []
        offset = 0

        while True:
            page = self._query_nodes_raw(
                labels=labels,
                properties={},
                limit=batch_size,
                offset=offset,
            )
            if not page:
                break

            for raw_node in page:
                node = self._normalize_node(raw_node)
                if node is None:
                    continue
                if not include_internal and any(
                    self._is_internal_label(label) for label in node.labels
                ):
                    continue
                nodes.append(node)

            if len(page) < batch_size:
                break
            offset += batch_size

        return nodes

    def get_nodes_by_file(self, file_path: str) -> list[GraphNode]:
        """Return nodes associated with a file path."""
        result = []
        for node in self.get_all_nodes():
            candidate_file = (
                node.properties.get("file")
                or node.properties.get("file_path")
                or node.properties.get("path")
            )
            if candidate_file == file_path:
                result.append(node)
        return result

    def find_nodes(
        self,
        *,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
    ) -> list[GraphNode]:
        """Find nodes by exact symbol metadata."""
        labels = [type] if type else None
        nodes = [
            node
            for node in (
                self._normalize_node(raw_node)
                for raw_node in self._query_nodes_raw(
                    labels=labels,
                    properties={"name": name} if name else {},
                    limit=1000,
                    offset=0,
                )
            )
            if node is not None
        ]

        if file:
            nodes = [
                node
                for node in nodes
                if (
                    node.properties.get("file")
                    or node.properties.get("file_path")
                    or node.properties.get("path")
                )
                == file
            ]
        if name:
            nodes = [
                node
                for node in nodes
                if node.properties.get("name") == name
                or node.properties.get("qualified_name") == name
            ]
        return nodes

    def search_symbols(
        self,
        query: str,
        limit: int = 20,
        symbol_types: Iterable[str] | None = None,
    ) -> list[GraphNode]:
        """Search symbol-like nodes using exact, prefix, and substring ranking."""
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []

        allowed_types = {str(symbol_type) for symbol_type in symbol_types or []}
        ranked: list[tuple[int, GraphNode]] = []

        for node in self.get_all_nodes():
            if allowed_types and not any(
                label in allowed_types for label in node.labels
            ):
                continue

            props = node.properties
            candidates = [
                str(props.get("qualified_name", "")),
                str(props.get("name", "")),
                str(props.get("signature", "")),
                str(props.get("docstring", "")),
                str(props.get("file", props.get("file_path", ""))),
            ]
            haystack = " ".join(value for value in candidates if value).lower()
            if not haystack:
                continue

            score = 0
            if str(props.get("qualified_name", "")).lower() == normalized_query:
                score = 120
            elif str(props.get("name", "")).lower() == normalized_query:
                score = 110
            elif (
                str(props.get("qualified_name", ""))
                .lower()
                .startswith(normalized_query)
            ):
                score = 100
            elif str(props.get("name", "")).lower().startswith(normalized_query):
                score = 95
            elif normalized_query in str(props.get("qualified_name", "")).lower():
                score = 90
            elif normalized_query in str(props.get("name", "")).lower():
                score = 85
            elif normalized_query in str(props.get("signature", "")).lower():
                score = 70
            elif normalized_query in str(props.get("docstring", "")).lower():
                score = 60
            elif normalized_query in haystack:
                score = 40

            if score > 0:
                ranked.append((score, node))

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1].properties.get("file", item[1].properties.get("file_path", "")),
                item[1].properties.get("line", item[1].properties.get("line_start", 0))
                or 0,
                item[1].properties.get("name", ""),
            )
        )
        return [node for _, node in ranked[:limit]]

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Iterable[str] | None = None,
        *,
        direction: str = "both",
        max_depth: int = 1,
    ) -> list[GraphEdge]:
        """Get neighboring edges around a node."""
        allowed_edge_types = list(edge_types or [])
        seen_edges: set[tuple[str, str, str]] = set()
        collected: list[GraphEdge] = []
        frontier = {node_id}
        visited_nodes = {node_id}

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for current_node_id in frontier:
                raw_edges: list[dict[str, Any]] = []
                if direction in {"out", "both"}:
                    raw_edges.extend(
                        self._get_outgoing_edges_raw(
                            current_node_id, allowed_edge_types or None
                        )
                    )
                if direction in {"in", "both"}:
                    raw_edges.extend(
                        self._get_incoming_edges_raw(
                            current_node_id, allowed_edge_types or None
                        )
                    )

                for raw_edge in raw_edges:
                    edge = self._normalize_edge(raw_edge)
                    if edge is None:
                        continue
                    signature = (edge.from_node, edge.to_node, edge.edge_type)
                    if signature in seen_edges:
                        continue
                    seen_edges.add(signature)
                    collected.append(edge)

                    neighbor_id = (
                        edge.to_node
                        if edge.from_node == current_node_id
                        else edge.from_node
                    )
                    if neighbor_id and neighbor_id not in visited_nodes:
                        visited_nodes.add(neighbor_id)
                        next_frontier.add(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

        return collected

    def get_all_edges(
        self,
        edge_types: Iterable[str] | None = None,
    ) -> list[GraphEdge]:
        """Get all edges in the graph via outgoing-edge scans."""
        seen_edges: set[tuple[str, str, str]] = set()
        edges: list[GraphEdge] = []

        for node in self.get_all_nodes(include_internal=True):
            for raw_edge in self._get_outgoing_edges_raw(
                node.id,
                list(edge_types) if edge_types else None,
            ):
                edge = self._normalize_edge(raw_edge)
                if edge is None:
                    continue
                signature = (edge.from_node, edge.to_node, edge.edge_type)
                if signature in seen_edges:
                    continue
                seen_edges.add(signature)
                edges.append(edge)

        return edges

    def import_graph_json(
        self,
        graph_data: str | Path | dict[str, Any],
    ) -> dict[str, Any]:
        """Import a Graphify-like or generic graph.json artifact into the graph."""
        if isinstance(graph_data, (str, Path)):
            with Path(graph_data).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = graph_data

        raw_nodes = payload.get("nodes") or payload.get("graph", {}).get("nodes") or []
        raw_edges = payload.get("edges") or payload.get("graph", {}).get("edges") or []

        nodes = [
            normalized
            for normalized in (
                self._normalize_json_node(raw_node) for raw_node in raw_nodes
            )
            if normalized is not None
        ]
        edges = [
            normalized
            for normalized in (
                self._normalize_json_edge(raw_edge, index)
                for index, raw_edge in enumerate(raw_edges)
            )
            if normalized is not None
        ]

        node_result = self.batch_create_nodes(nodes)
        edge_result = self.batch_create_edges(edges)
        return {
            "success": bool(node_result.get("success"))
            and bool(edge_result.get("success")),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": node_result,
            "edges": edge_result,
        }

    # ========================================================================
    # Pattern Matching
    # ========================================================================

    def match_pattern(
        self,
        pattern: str,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, GraphNode]]:
        """Match a graph pattern and return matching subgraphs.

        Pattern syntax:
        - (n) matches any node
        - (n:Label) matches nodes with label
        - (n {key: value}) matches nodes with properties
        - (n)-[r:TYPE]->(m) matches relationships

        Args:
            pattern: Graph pattern string
            filters: Optional additional filters

        Returns:
            List of matched subgraphs (each is a dict of variable -> node)

        Example:
            # Find all functions that call other functions
            matches = graph.match_pattern("(f1:Function)-[:CALLS]->(f2:Function)")

            # Find test functions that call functions with "parse" in name
            matches = graph.match_pattern(
                "(t1:Function)-[:CALLS]->(f2:Function)",
                filters={"t1.name": "*test*", "f2.name": "*parse*"}
            )
        """
        # This is a simplified implementation
        # Full pattern matching would require a proper graph pattern engine

        # Parse pattern for nodes and relationships
        nodes = {}
        relationships = []

        # Find all node patterns: (variable:Label)
        node_matches = re.finditer(r"\((\w+)(?::(\w+))?(?:\s*\{(.+)\})?\)", pattern)
        for match in node_matches:
            var_name = match.group(1)
            label = match.group(2)
            props_str = match.group(3)

            props = {}
            if props_str:
                # Parse properties
                for prop_match in re.finditer(r'(\w+)\s*:\s*"?(\w+)"?', props_str):
                    props[prop_match.group(1)] = prop_match.group(2)

            nodes[var_name] = {"label": label, "properties": props}

        # Find relationship patterns: -[r:TYPE]->
        rel_matches = re.finditer(r"-?\[(\w+)(?::(\w+))?\]->", pattern)
        for match in rel_matches:
            relationships.append(
                {
                    "variable": match.group(1),
                    "type": match.group(2),
                }
            )

        # If no relationships, just return matching nodes
        if not relationships:
            if nodes:
                # Get first node as starting point
                first_var = list(nodes.keys())[0]
                first_node = nodes[first_var]

                node_results = self._client.query_nodes(
                    graph_id=self._graph_id,
                    labels=[first_node["label"]] if first_node["label"] else [],
                    properties=first_node["properties"],
                )

                return [
                    {
                        first_var: GraphNode(
                            id=n["id"],
                            labels=n.get("labels", []),
                            properties=n.get("properties", {}),
                        )
                    }
                    for n in node_results.get("nodes", [])
                ]

        return []

    # ========================================================================
    # Graph Statistics
    # ========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary with node_count, edge_count, etc.
        """
        return self._client.get_graph_stats(self._graph_id)


def create_graph_api(client: Any, graph_id: str) -> ProximaDBGraph:
    """Factory function to create a graph API instance.

    Args:
        client: ProximaDB client instance
        graph_id: Graph collection identifier

    Returns:
        ProximaDBGraph instance
    """
    return ProximaDBGraph(client, graph_id)
