"""Split-port (MULTI) mode honours the configured ports.

get_protocol_url used to hardcode REST :5678, gRPC :5679 and Flight :5680 in
multi-port mode, so a REST URL on another port and a configured gRPC port
were silently ignored (the CLI's --grpc-port never reached the client).
"""

from proximadb_sdk.config import ClientConfig, PortMode, Protocol


def _multi(**kw):
    return ClientConfig(url="http://db.example:15678", port_mode=PortMode.MULTI, **kw)


def test_multi_mode_honours_url_port_for_rest():
    assert _multi().get_protocol_url(Protocol.REST) == "http://db.example:15678"


def test_multi_mode_honours_configured_grpc_and_flight_ports():
    cfg = _multi(grpc_port=50051, flight_port=50052)
    assert cfg.get_protocol_url(Protocol.GRPC) == "db.example:50051"
    assert cfg.get_protocol_url(Protocol.ARROW_FLIGHT) == "db.example:50052"


def test_multi_mode_defaults_are_unchanged():
    cfg = ClientConfig(url="http://db.example", port_mode=PortMode.MULTI)
    assert cfg.get_protocol_url(Protocol.REST) == "http://db.example:5678"
    assert cfg.get_protocol_url(Protocol.GRPC) == "db.example:5679"
    assert cfg.get_protocol_url(Protocol.ARROW_FLIGHT) == "db.example:5680"


def test_grpc_scheme_url_port_does_not_leak_into_rest():
    """A grpc:// URL's port is the gRPC port; REST keeps its default."""
    cfg = ClientConfig(url="grpc://db.example:5679", port_mode=PortMode.MULTI)
    assert cfg.get_protocol_url(Protocol.REST).endswith(":5678")


def test_unified_mode_ignores_split_ports():
    cfg = ClientConfig(url="http://db.example:5678", grpc_port=50051)
    assert cfg.get_protocol_url(Protocol.GRPC) == "db.example:5678"


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


def test_real_client_config_carries_grpc_port():
    """End to end through the constructor: kwargs -> load_config -> ClientConfig."""
    from proximadb_sdk.config import load_config

    cfg = load_config(url="http://db.example:5678", port_mode="multi", grpc_port=50051)
    assert cfg.get_protocol_url(Protocol.GRPC) == "db.example:50051"


def test_grpc_client_uses_explicit_timeout_and_keeps_60s_default(monkeypatch):
    """The gRPC client's deadline was hardcoded to 60 s."""
    from proximadb_sdk import unified_client as uc
    from proximadb_sdk.config import load_config
    from proximadb_sdk.protocols import grpc_sync

    seen = []

    class FakeGrpc:
        def __init__(self, server_address=None, timeout=None, **kw):
            seen.append((server_address, timeout))

    monkeypatch.setattr(grpc_sync, "ProximaDBSyncGrpcClient", FakeGrpc)

    def build(**cfg_kw):
        client = uc.ProximaDBClient.__new__(uc.ProximaDBClient)
        client.config = load_config(url="http://h:5678", port_mode="multi", **cfg_kw)
        client._auth = None
        client._create_grpc_client()

    build(grpc_port=50051, timeout=7.5)
    build(grpc_port=50051)
    assert seen[0] == ("h:50051", 7.5)
    assert seen[1] == ("h:50051", 60.0)
