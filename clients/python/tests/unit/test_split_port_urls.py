"""Split-port (MULTI) mode: gRPC dials the configured gRPC port.

get_protocol_url hardcoded gRPC to :5679, and the v1 gRPC *adapter* (which
most operations go through) dialled the REST URL's port, so the CLI's
--grpc-port never took effect.
"""

import pytest
from pydantic import ValidationError

from proximadb_sdk.config import ClientConfig, PortMode, Protocol, load_config


def _multi(url="http://db.example:5678", **kw):
    return ClientConfig(url=url, port_mode=PortMode.MULTI, **kw)


def test_configured_grpc_port_wins():
    assert _multi(grpc_port=50051).get_protocol_url(Protocol.GRPC) == "db.example:50051"


def test_grpc_url_port_is_used_when_grpc_port_unset():
    assert (
        _multi(url="grpc://db.example:50051").get_protocol_url(Protocol.GRPC)
        == "db.example:50051"
    )


def test_multi_defaults_are_unchanged():
    cfg = ClientConfig(url="http://db.example", port_mode=PortMode.MULTI)
    assert cfg.get_protocol_url(Protocol.REST) == "http://db.example:5678"
    assert cfg.get_protocol_url(Protocol.GRPC) == "db.example:5679"
    assert cfg.get_protocol_url(Protocol.ARROW_FLIGHT) == "db.example:5680"


def test_unified_mode_ignores_grpc_port():
    assert (
        ClientConfig(url="http://db.example:5678", grpc_port=50051).get_protocol_url(
            Protocol.GRPC
        )
        == "db.example:5678"
    )


@pytest.mark.parametrize("bad", [0, -1, 70000])
def test_grpc_port_is_validated(bad):
    with pytest.raises(ValidationError):
        ClientConfig(url="http://h:5678", grpc_port=bad)


def test_cli_passes_grpc_port_only_in_multi_mode(monkeypatch):
    import httpx

    from proximadb_sdk import cli as cli_mod

    class _Resp:
        status_code = 200

    monkeypatch.setattr(httpx, "get", lambda url, **kw: _Resp())
    seen = []

    class FakeUnified:
        def __init__(self, url=None, protocol=None, port_mode=None, **kw):
            seen.append((str(port_mode), kw.get("grpc_port")))

    monkeypatch.setattr(cli_mod, "UnifiedProximaDBClient", FakeUnified)
    cli_mod.get_client("h", 5678, 50051, "grpc", 5.0)
    cli_mod.get_client("h", 5678, 5678, "rest", 5.0)
    assert seen[0] == (str(PortMode.MULTI), 50051)
    assert seen[1] == (str(PortMode.UNIFIED), None)


def _capture_grpc(monkeypatch):
    """Record every gRPC client construction (adapter-internal and direct)."""
    from proximadb_sdk.protocols import grpc_sync

    seen = []

    class FakeGrpc:
        def __init__(self, server_address=None, timeout=None, **kw):
            seen.append((server_address, timeout))

        def __getattr__(self, name):  # tolerate setup-time calls
            return lambda *a, **k: None

    monkeypatch.setattr(grpc_sync, "ProximaDBSyncGrpcClient", FakeGrpc)
    import proximadb_sdk.adapters.grpc_adapter as ga

    monkeypatch.setattr(ga, "ProximaDBSyncGrpcClient", FakeGrpc, raising=False)
    return seen


def test_real_client_both_grpc_paths_dial_grpc_port(monkeypatch):
    """Through the REAL constructor: the adapter (most operations) and the
    direct gRPC client must both dial the configured port and deadline."""
    from proximadb_sdk.unified_client import ProximaDBClient

    seen = _capture_grpc(monkeypatch)
    ProximaDBClient(
        url="http://127.0.0.1:5678",
        protocol="grpc",
        port_mode="multi",
        grpc_port=50051,
        timeout=7.5,
    )
    assert seen, "no gRPC client was constructed"
    assert all(addr == "127.0.0.1:50051" for addr, _ in seen), seen
    assert all(t == 7.5 for _, t in seen), seen


def test_real_client_unset_timeout_keeps_60s(monkeypatch):
    from proximadb_sdk.unified_client import ProximaDBClient

    seen = _capture_grpc(monkeypatch)
    ProximaDBClient(
        url="http://127.0.0.1:5678", protocol="grpc", port_mode="multi", grpc_port=50051
    )
    assert seen and all(t == 60.0 for _, t in seen), seen


def test_unified_adapter_keeps_bare_host_port_url(monkeypatch):
    """Unified mode must not route the adapter through get_protocol_url,
    which loses the host of a bare host:port URL."""
    from proximadb_sdk.unified_client import ProximaDBClient

    seen = _capture_grpc(monkeypatch)
    ProximaDBClient(url="db.example:5678", protocol="grpc")
    assert any(addr == "db.example:5678" for addr, _ in seen), seen


def test_load_config_carries_grpc_port():
    cfg = load_config(url="http://db.example:5678", port_mode="multi", grpc_port=50051)
    assert cfg.get_protocol_url(Protocol.GRPC) == "db.example:50051"
