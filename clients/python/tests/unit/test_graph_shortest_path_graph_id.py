"""TD-SDKDOC-1: running the guide's graph example found two SDK bugs.

1. ``ProximaDBClient.graph_shortest_path`` dropped ``graph_id`` although both
   transports take it — every call ran on the ``default`` graph.
2. ``GraphAnalytics.shortest_path`` called it with ``graph_id=`` /
   ``start_node=`` / ``end_node=`` (TypeError on every real client) and read a
   top-level ``path`` the REST answer nests under ``data``.

The tests go through the REAL unified method with only the transport stubbed:
the shared conftest mock accepts ``**kwargs``, which is how bug 2 hid.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import proximadb_sdk

if not proximadb_sdk.__file__.startswith(str(Path(__file__).resolve().parents[2])):
    pytest.skip(
        f"tests the worktree SDK, but {proximadb_sdk.__file__} won the import; "
        "run with PYTHONPATH=<worktree>/clients/python/src",
        allow_module_level=True,
    )

from proximadb_sdk import GraphAnalytics, Protocol, ProximaDBClient  # noqa: E402


class _RestTransport:
    """Mirrors rest_sync.graph_shortest_path's signature and answer shape."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def graph_shortest_path(
        self,
        start_node_id: str,
        target_node_id: str,
        max_depth: int | None = None,
        edge_types: list[str] | None = None,
        algorithm: str = "DIJKSTRA",
        k: int | None = None,
        enable_prefetch: bool | None = None,
        prefetch_budget: int | None = None,
        timeout: float | None = None,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        self.calls.append(
            {"start": start_node_id, "target": target_node_id, "graph_id": graph_id}
        )
        return {
            "success": True,
            "data": {"path": [start_node_id, target_node_id], "found": True},
        }


class _GrpcTransport:
    """Mirrors grpc_sync.shortest_path's signature and proto answer shape."""

    def __init__(self, found: bool = True) -> None:
        self.calls: list[dict[str, Any]] = []
        self.found = found

    def shortest_path(
        self,
        start_node_id: str,
        target_node_id: str,
        max_depth: int | None = None,
        edge_types: list[str] | None = None,
        algorithm: str = "DIJKSTRA",
        k: int | None = None,
        enable_prefetch: bool | None = None,
        prefetch_budget: int | None = None,
        graph_id: str = "default",
    ) -> Any:
        self.calls.append(
            {"start": start_node_id, "target": target_node_id, "graph_id": graph_id}
        )
        ids = [start_node_id, target_node_id] if self.found else []
        return SimpleNamespace(node_ids=ids, found=self.found, total_weight=1.0)


def _client(protocol: Protocol, transport: Any) -> ProximaDBClient:
    client = ProximaDBClient.__new__(ProximaDBClient)
    client._active_protocol = protocol
    client._client = transport
    return client


@pytest.mark.parametrize(
    "protocol,transport",
    [(Protocol.REST, _RestTransport()), (Protocol.GRPC, _GrpcTransport())],
    ids=["rest", "grpc"],
)
def test_unified_shortest_path_forwards_graph_id(protocol, transport):
    client = _client(protocol, transport)
    client.graph_shortest_path("a", "b", graph_id="knowledge")
    assert transport.calls == [{"start": "a", "target": "b", "graph_id": "knowledge"}]


@pytest.mark.parametrize(
    "protocol,transport",
    [(Protocol.REST, _RestTransport()), (Protocol.GRPC, _GrpcTransport())],
    ids=["rest", "grpc"],
)
def test_analytics_shortest_path_returns_node_ids_on_the_named_graph(
    protocol, transport
):
    analytics = GraphAnalytics(_client(protocol, transport))
    assert analytics.shortest_path("knowledge", "a", "b") == ["a", "b"]
    assert transport.calls[-1]["graph_id"] == "knowledge"


class _AsyncRestTransport(_RestTransport):
    async def graph_shortest_path(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return _RestTransport.graph_shortest_path(self, *args, **kwargs)


def test_async_shortest_path_forwards_graph_id():
    import asyncio

    from proximadb_sdk.unified_client_async import ProximaDBAsyncUnified

    transport = _AsyncRestTransport()
    client = ProximaDBAsyncUnified.__new__(ProximaDBAsyncUnified)
    client._grpc = None
    client._rest = transport
    asyncio.run(client.graph_shortest_path("a", "b", graph_id="knowledge"))
    assert transport.calls == [{"start": "a", "target": "b", "graph_id": "knowledge"}]


def test_analytics_shortest_path_no_path_is_none():
    analytics = GraphAnalytics(_client(Protocol.GRPC, _GrpcTransport(found=False)))
    assert analytics.shortest_path("knowledge", "a", "b") is None
