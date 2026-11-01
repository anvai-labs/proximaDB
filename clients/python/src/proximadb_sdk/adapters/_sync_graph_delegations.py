"""Shared graph-CRUD delegations for sync transports (TD-SDK-3 batch 1).

``protocols.rest_sync`` and ``protocols.grpc_sync`` expose byte-compatible
graph-method signatures (the canonical surface ``BaseProtocolAdapter``
declares), so their adapters delegate identically. Housing those delegations
here once — instead of once per adapter — keeps the dedup program's own
duplication ratchet honest: two hand-maintained copies of the same ten
forwarding bodies is exactly the drifted-twin failure mode this program
exists to cure.

Scope: ONLY the ten methods BOTH transports implement. Anything a transport
lacks stays on ``BaseProtocolAdapter``'s ``NotImplementedError`` defaults
(gRPC: graph-collection management and the tuned ``graph_traverse``;
``graph_shortest_path`` differs per transport — REST forwards ``timeout``,
gRPC renames to ``shortest_path`` and drops it — so it is also absent here
and implemented per adapter).

Usage: list the mixin BEFORE ``BaseProtocolAdapter`` so its concrete methods
take precedence over the base defaults::

    class RestProtocolAdapter(SyncGraphDelegations, BaseProtocolAdapter): ...
"""

from typing import Any


class SyncGraphDelegations:
    """Ten canonical graph delegations over ``self._client``.

    Expects ``self._client`` — the sync transport client — whose graph
    methods accept the canonical kwargs (positional-or-keyword) plus
    ``graph_id: str`` (``None`` is normalized to the transport's ``"default"``
    here so a literal None never reaches a URL builder or proto conversion).
    """

    def create_node(
        self,
        node_id: str,
        labels: list[str],
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        graph_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a graph node via the sync transport."""
        return self._client.create_node(
            node_id=node_id,
            labels=labels,
            properties=properties,
            embedding=embedding,
            graph_id=graph_id or "default",
        )

    def create_edge(
        self,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
        weight: float | None = None,
        graph_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a graph edge via the sync transport."""
        return self._client.create_edge(
            edge_id=edge_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            edge_type=edge_type,
            properties=properties,
            weight=weight,
            graph_id=graph_id or "default",
        )

    def batch_create_nodes(
        self, nodes: list[dict[str, Any]], graph_id: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create many nodes in one request via the batch endpoint."""
        return self._client.batch_create_nodes(
            nodes=nodes, graph_id=graph_id or "default"
        )

    def batch_create_edges(
        self, edges: list[dict[str, Any]], graph_id: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create many edges in one request (per-edge admission server-side)."""
        return self._client.batch_create_edges(
            edges=edges, graph_id=graph_id or "default"
        )

    def get_node(
        self, node_id: str, graph_id: str | None = None, **kwargs
    ) -> dict[str, Any] | None:
        """Get a graph node by ID via the sync transport."""
        return self._client.get_node(node_id=node_id, graph_id=graph_id or "default")

    def get_outgoing_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        graph_id: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Get a node's outgoing edges via the sync transport."""
        return self._client.get_outgoing_edges(
            node_id=node_id,
            edge_types=edge_types,
            graph_id=graph_id or "default",
        )

    def get_incoming_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        graph_id: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Get a node's incoming edges via the sync transport."""
        return self._client.get_incoming_edges(
            node_id=node_id,
            edge_types=edge_types,
            graph_id=graph_id or "default",
        )

    def delete_node(self, node_id: str, graph_id: str | None = None, **kwargs) -> bool:
        """Delete a graph node by ID via the sync transport."""
        result = self._client.delete_node(
            node_id=node_id, graph_id=graph_id or "default"
        )
        return bool(result)

    def query_nodes(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        graph_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Query nodes by labels and properties via the sync transport."""
        return self._client.query_nodes(
            labels=labels,
            properties=properties,
            limit=limit,
            offset=offset,
            graph_id=graph_id or "default",
        )

    def traverse_graph(
        self,
        start_node_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
        node_labels: list[str] | None = None,
        algorithm: str = "BFS",
        limit: int | None = None,
        graph_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Traverse the graph from a starting node via the sync transport."""
        return self._client.traverse_graph(
            start_node_id=start_node_id,
            max_depth=max_depth,
            edge_types=edge_types,
            node_labels=node_labels,
            algorithm=algorithm,
            limit=limit,
            graph_id=graph_id or "default",
        )
