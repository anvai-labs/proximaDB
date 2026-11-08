"""Shared graph-call input validation (TD-SDK-3 batch 1).

One source of truth for the protocol-independent type/value checks the
graph-CRUD family performs, used by BOTH the v1 unified client and the v2
facade — copied inline blocks would drift apart exactly the way the dedup
program exists to prevent. Error messages are load-bearing: tests on both
clients pin them.
"""

from typing import Any


def validate_create_node_args(
    node_id: str,
    labels: list[str],
    properties: dict[str, Any] | None,
    embedding: list[float] | None,
) -> None:
    if not isinstance(node_id, str):
        raise TypeError(f"node_id must be str, got {type(node_id).__name__}")
    if not isinstance(labels, list):
        raise TypeError(f"labels must be list, got {type(labels).__name__}")
    if properties is not None and not isinstance(properties, dict):
        raise TypeError(
            f"properties must be dict or None, got {type(properties).__name__}"
        )
    if embedding is not None and not isinstance(embedding, (list, type(None))):
        raise TypeError(
            f"embedding must be list or None, got {type(embedding).__name__}"
        )


def validate_create_edge_args(
    edge_id: str,
    from_node_id: str,
    to_node_id: str,
    edge_type: str,
    properties: dict[str, Any] | None,
    weight: float | None,
) -> None:
    if not isinstance(edge_id, str):
        raise TypeError(f"edge_id must be str, got {type(edge_id).__name__}")
    if not isinstance(from_node_id, str):
        raise TypeError(f"from_node_id must be str, got {type(from_node_id).__name__}")
    if not isinstance(to_node_id, str):
        raise TypeError(f"to_node_id must be str, got {type(to_node_id).__name__}")
    if not isinstance(edge_type, str):
        raise TypeError(f"edge_type must be str, got {type(edge_type).__name__}")
    if properties is not None and not isinstance(properties, dict):
        raise TypeError(
            f"properties must be dict or None, got {type(properties).__name__}"
        )
    if weight is not None and not isinstance(weight, (int, float)):
        raise TypeError(f"weight must be number or None, got {type(weight).__name__}")


def validate_traverse_graph_args(
    start_node_id: str,
    max_depth: int,
    edge_types: list[str] | None,
    node_labels: list[str] | None,
    algorithm: str,
    limit: int | None,
) -> None:
    if not isinstance(start_node_id, str):
        raise TypeError(
            f"start_node_id must be str, got {type(start_node_id).__name__}"
        )
    if not isinstance(max_depth, int):
        raise TypeError(f"max_depth must be int, got {type(max_depth).__name__}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    if edge_types is not None and not isinstance(edge_types, list):
        raise TypeError(
            f"edge_types must be list or None, got {type(edge_types).__name__}"
        )
    if node_labels is not None and not isinstance(node_labels, list):
        raise TypeError(
            f"node_labels must be list or None, got {type(node_labels).__name__}"
        )
    if not isinstance(algorithm, str):
        raise TypeError(f"algorithm must be str, got {type(algorithm).__name__}")
    if algorithm not in ["BFS", "DFS", "PARALLEL_BFS"]:
        raise ValueError(
            f"algorithm must be one of BFS/DFS/PARALLEL_BFS, got {algorithm}"
        )
    if limit is not None and not isinstance(limit, int):
        raise TypeError(f"limit must be int or None, got {type(limit).__name__}")
