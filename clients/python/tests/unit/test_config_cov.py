"""Offline unit tests for proximadb_sdk.config.

Pure tests: construct/validate/serialize ClientConfig and helpers, exercise
env-override (from_env), file loading (load_config_file), and all the URL /
protocol / port-mode branches. No network, no server, no heavy deps.
"""

import json
import os

import pytest

from proximadb_sdk.config import (
    DEFAULT_CONFIG,
    ClientConfig,
    CompressionConfig,
    ConnectionConfig,
    LogLevel,
    PortMode,
    Protocol,
    RetryConfig,
    TLSConfig,
    load_config,
    load_config_file,
)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
def test_enum_values():
    assert Protocol.AUTO.value == "auto"
    assert Protocol.GRPC.value == "grpc"
    assert Protocol.REST.value == "rest"
    assert Protocol.EMBEDDED.value == "embedded"
    assert Protocol.ARROW_FLIGHT.value == "arrow_flight"
    assert PortMode.MULTI.value == "multi"
    assert PortMode.UNIFIED.value == "unified"
    assert LogLevel.DEBUG.value == "DEBUG"
    assert LogLevel.CRITICAL.value == "CRITICAL"


# --------------------------------------------------------------------------- #
# Sub-config models + defaults
# --------------------------------------------------------------------------- #
def test_connection_config_defaults():
    c = ConnectionConfig()
    assert c.pool_size == 10
    assert c.pool_maxsize == 100
    assert c.read_timeout == 30.0


def test_connection_config_bounds():
    with pytest.raises(Exception):
        ConnectionConfig(pool_size=0)
    with pytest.raises(Exception):
        ConnectionConfig(pool_size=101)


def test_compression_config_defaults_and_level():
    c = CompressionConfig()
    assert c.enabled is False
    assert c.algorithm == "gzip"
    assert c.threshold_bytes == 1024
    assert c.level is None
    c2 = CompressionConfig(enabled=True, level=5)
    assert c2.level == 5
    with pytest.raises(Exception):
        CompressionConfig(level=10)


def test_tls_config_defaults():
    t = TLSConfig()
    assert t.verify is True
    assert t.ca_bundle is None


def test_retry_config_defaults():
    r = RetryConfig()
    assert r.max_retries == 3
    assert r.backoff_factor == 2.0


# --------------------------------------------------------------------------- #
# ClientConfig construction + defaults
# --------------------------------------------------------------------------- #
def test_client_config_defaults():
    cfg = ClientConfig(url="http://localhost:5678")
    # use_enum_values=True => stored as plain strings
    assert cfg.protocol == "auto"
    assert cfg.port_mode == "unified"
    assert cfg.timeout == 30.0
    assert cfg.log_level == "INFO"
    assert isinstance(cfg.retry, RetryConfig)
    assert isinstance(cfg.connection, ConnectionConfig)
    assert isinstance(cfg.compression, CompressionConfig)
    assert isinstance(cfg.tls, TLSConfig)
    assert cfg.custom_headers == {}


def test_default_config_instance():
    assert DEFAULT_CONFIG.url == "http://localhost:5678"
    assert DEFAULT_CONFIG.timeout == 30.0


def test_serialization_roundtrip():
    cfg = ClientConfig(url="https://example.com:5678", api_key="abcdefghijk")
    dumped = cfg.model_dump()
    assert dumped["url"] == "https://example.com:5678"
    assert dumped["api_key"] == "abcdefghijk"
    rebuilt = ClientConfig(**dumped)
    assert rebuilt.url == cfg.url
    js = cfg.model_dump_json()
    # Parse and assert the exact url field rather than a substring match: a
    # substring check is satisfied by an arbitrary position in the JSON
    # (CodeQL py/incomplete-url-substring-sanitization).
    assert json.loads(js)["url"] == "https://example.com:5678"


# --------------------------------------------------------------------------- #
# URL validator branches
# --------------------------------------------------------------------------- #
def test_url_empty_raises():
    with pytest.raises(Exception):
        ClientConfig(url="")


def test_url_grpc_hostport_kept_asis():
    # urlparse gives no scheme for an IP-style host:port, so the validator
    # takes the "looks like host:port for gRPC" branch and keeps it as-is.
    cfg = ClientConfig(url="127.0.0.1:5679")
    assert cfg.url == "127.0.0.1:5679"


def test_url_no_scheme_gets_https():
    # "example.com" has no scheme and no ':' host:port form -> https prepended
    cfg = ClientConfig(url="example.com")
    assert cfg.url == "https://example.com"


def test_url_grpc_scheme_allowed():
    cfg = ClientConfig(url="grpc://localhost:5679")
    assert cfg.url == "grpc://localhost:5679"


def test_url_embedded_scheme_allowed():
    cfg = ClientConfig(url="embedded://local")
    assert cfg.url == "embedded://local"


def test_url_bad_scheme_raises():
    with pytest.raises(Exception):
        ClientConfig(url="ftp://example.com")


def test_url_scheme_no_hostname_raises():
    with pytest.raises(Exception):
        ClientConfig(url="http://")


# --------------------------------------------------------------------------- #
# api_key validator
# --------------------------------------------------------------------------- #
def test_api_key_too_short_raises():
    with pytest.raises(Exception):
        ClientConfig(url="http://localhost:5678", api_key="short")


def test_api_key_none_ok():
    cfg = ClientConfig(url="http://localhost:5678", api_key=None)
    assert cfg.api_key is None


def test_api_key_valid():
    cfg = ClientConfig(url="http://localhost:5678", api_key="0123456789")
    assert cfg.api_key == "0123456789"


# --------------------------------------------------------------------------- #
# get_base_headers
# --------------------------------------------------------------------------- #
def test_base_headers_default():
    cfg = ClientConfig(url="http://localhost:5678")
    h = cfg.get_base_headers()
    assert h["Accept"] == "application/json"
    assert h["Content-Type"] == "application/json"
    assert h["User-Agent"].startswith("proximadb-python/")
    assert "Authorization" not in h
    assert "Accept-Encoding" not in h


def test_base_headers_with_api_key_compression_custom_useragent():
    cfg = ClientConfig(
        url="http://localhost:5678",
        api_key="0123456789",
        user_agent="my-agent/1.0",
        compression=CompressionConfig(enabled=True),
        custom_headers={"X-Trace": "abc"},
    )
    h = cfg.get_base_headers()
    assert h["Authorization"] == "Bearer 0123456789"
    assert h["User-Agent"] == "my-agent/1.0"
    assert "gzip" in h["Accept-Encoding"]
    assert h["X-Trace"] == "abc"


# --------------------------------------------------------------------------- #
# get_grpc_metadata
# --------------------------------------------------------------------------- #
def test_grpc_metadata_default():
    cfg = ClientConfig(url="http://localhost:5678")
    md = cfg.get_grpc_metadata()
    keys = dict(md)
    assert "user-agent" in keys
    assert "authorization" not in keys


def test_grpc_metadata_with_key_and_custom():
    cfg = ClientConfig(
        url="http://localhost:5678",
        api_key="0123456789",
        custom_headers={"X-Trace": "abc"},
    )
    md = cfg.get_grpc_metadata()
    d = dict(md)
    assert d["authorization"] == "Bearer 0123456789"
    assert d["x-trace"] == "abc"


# --------------------------------------------------------------------------- #
# _get_version
# --------------------------------------------------------------------------- #
def test_get_version_returns_string():
    cfg = ClientConfig(url="http://localhost:5678")
    v = cfg._get_version()
    assert isinstance(v, str)


# --------------------------------------------------------------------------- #
# is_secure / get_host_port
# --------------------------------------------------------------------------- #
def test_is_secure_https():
    assert ClientConfig(url="https://example.com").is_secure() is True


def test_is_secure_http():
    assert ClientConfig(url="http://example.com").is_secure() is False


def test_host_port_explicit():
    host, port = ClientConfig(url="http://example.com:5678").get_host_port()
    assert host == "example.com"
    assert port == 5678


def test_host_port_default_http():
    host, port = ClientConfig(url="http://example.com").get_host_port()
    assert host == "example.com"
    assert port == 80


def test_host_port_default_https():
    host, port = ClientConfig(url="https://example.com").get_host_port()
    assert port == 443


# --------------------------------------------------------------------------- #
# should_use_grpc
# --------------------------------------------------------------------------- #
def test_should_use_grpc_grpc():
    assert (
        ClientConfig(url="http://l:5678", protocol=Protocol.GRPC).should_use_grpc()
        is True
    )


def test_should_use_grpc_embedded():
    assert (
        ClientConfig(url="http://l:5678", protocol=Protocol.EMBEDDED).should_use_grpc()
        is False
    )


def test_should_use_grpc_rest():
    assert (
        ClientConfig(url="http://l:5678", protocol=Protocol.REST).should_use_grpc()
        is False
    )


def test_should_use_grpc_auto():
    assert (
        ClientConfig(url="http://l:5678", protocol=Protocol.AUTO).should_use_grpc()
        is True
    )


# --------------------------------------------------------------------------- #
# get_protocol_url
# --------------------------------------------------------------------------- #
def test_protocol_url_embedded_returns_url():
    cfg = ClientConfig(url="http://localhost:5678")
    assert cfg.get_protocol_url(Protocol.EMBEDDED) == "http://localhost:5678"


def test_protocol_url_unified_rest():
    cfg = ClientConfig(url="http://localhost:5678", port_mode=PortMode.UNIFIED)
    assert cfg.get_protocol_url(Protocol.REST) == "http://localhost:5678"


def test_protocol_url_unified_grpc_hostport():
    cfg = ClientConfig(url="http://localhost:5678", port_mode=PortMode.UNIFIED)
    assert cfg.get_protocol_url(Protocol.GRPC) == "localhost:5678"


def test_protocol_url_unified_default_port():
    # URL with no port -> defaults to 5678 in unified mode
    cfg = ClientConfig(url="http://localhost", port_mode=PortMode.UNIFIED)
    assert cfg.get_protocol_url(Protocol.ARROW_FLIGHT) == "localhost:5678"
    assert cfg.get_protocol_url(Protocol.REST) == "http://localhost:5678"


def test_protocol_url_multi_rest():
    cfg = ClientConfig(url="http://localhost:9999", port_mode=PortMode.MULTI)
    assert cfg.get_protocol_url(Protocol.REST) == "http://localhost:5678"


def test_protocol_url_multi_grpc():
    cfg = ClientConfig(url="http://localhost:9999", port_mode=PortMode.MULTI)
    assert cfg.get_protocol_url(Protocol.GRPC) == "localhost:5679"


def test_protocol_url_multi_arrow_flight():
    cfg = ClientConfig(url="http://localhost:9999", port_mode=PortMode.MULTI)
    assert cfg.get_protocol_url(Protocol.ARROW_FLIGHT) == "localhost:5680"


def test_protocol_url_multi_auto_keeps_port():
    cfg = ClientConfig(url="http://localhost:9999", port_mode=PortMode.MULTI)
    assert cfg.get_protocol_url(Protocol.AUTO) == "http://localhost:9999"


def test_protocol_url_multi_auto_https_default_port():
    cfg = ClientConfig(url="https://localhost", port_mode=PortMode.MULTI)
    assert cfg.get_protocol_url(Protocol.AUTO) == "https://localhost:443"


def test_is_unified_mode():
    assert ClientConfig(url="http://l:5678").is_unified_mode() is True
    assert (
        ClientConfig(url="http://l:5678", port_mode=PortMode.MULTI).is_unified_mode()
        is False
    )


# --------------------------------------------------------------------------- #
# from_env
# --------------------------------------------------------------------------- #
@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("PROXIMADB_"):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_from_env_requires_url(clean_env):
    with pytest.raises(ValueError, match="URL must be provided"):
        ClientConfig.from_env()


def test_from_env_override_url(clean_env):
    cfg = ClientConfig.from_env(url="http://override:5678")
    assert cfg.url == "http://override:5678"


def test_from_env_all_settings(clean_env):
    clean_env.setenv("PROXIMADB_URL", "http://envhost:5678")
    clean_env.setenv("PROXIMADB_API_KEY", "envapikey123")
    clean_env.setenv("PROXIMADB_PROTOCOL", "GRPC")
    clean_env.setenv("PROXIMADB_PORT_MODE", "MULTI")
    clean_env.setenv("PROXIMADB_TIMEOUT", "45.5")
    clean_env.setenv("PROXIMADB_MAX_RETRIES", "7")
    clean_env.setenv("PROXIMADB_BACKOFF_FACTOR", "3.0")
    clean_env.setenv("PROXIMADB_POOL_SIZE", "20")
    clean_env.setenv("PROXIMADB_READ_TIMEOUT", "55.0")
    clean_env.setenv("PROXIMADB_TLS_VERIFY", "false")
    clean_env.setenv("PROXIMADB_CA_BUNDLE", "/path/ca.pem")
    clean_env.setenv("PROXIMADB_CERT_FILE", "/path/cert.pem")
    clean_env.setenv("PROXIMADB_KEY_FILE", "/path/key.pem")
    clean_env.setenv("PROXIMADB_LOG_LEVEL", "debug")
    clean_env.setenv("PROXIMADB_DEBUG", "yes")
    clean_env.setenv("PROXIMADB_BATCH_SIZE", "500")
    clean_env.setenv("PROXIMADB_MAX_CONCURRENT", "50")

    cfg = ClientConfig.from_env()
    assert cfg.url == "http://envhost:5678"
    assert cfg.api_key == "envapikey123"
    assert cfg.protocol == "grpc"
    assert cfg.port_mode == "multi"
    assert cfg.timeout == 45.5
    assert cfg.retry.max_retries == 7
    assert cfg.retry.backoff_factor == 3.0
    assert cfg.connection.pool_size == 20
    assert cfg.connection.read_timeout == 55.0
    assert cfg.tls.verify is False
    assert cfg.tls.ca_bundle == "/path/ca.pem"
    assert cfg.tls.cert_file == "/path/cert.pem"
    assert cfg.tls.key_file == "/path/key.pem"
    assert cfg.log_level == "DEBUG"
    assert cfg.enable_debug_logging is True
    assert cfg.default_batch_size == 500
    assert cfg.max_concurrent_requests == 50


def test_from_env_tls_verify_true(clean_env):
    clean_env.setenv("PROXIMADB_URL", "http://h:5678")
    clean_env.setenv("PROXIMADB_TLS_VERIFY", "1")
    cfg = ClientConfig.from_env()
    assert cfg.tls.verify is True


def test_from_env_protocol_string_override_path(clean_env):
    # protocol passed as override string triggers the isinstance(str) conversion branch
    cfg = ClientConfig.from_env(url="http://h:5678", protocol="rest")
    assert cfg.protocol == "rest"


# --------------------------------------------------------------------------- #
# load_config
# --------------------------------------------------------------------------- #
def test_load_config_explicit(clean_env):
    cfg = load_config(url="http://explicit:5678", api_key="explicitkey99")
    assert cfg.url == "http://explicit:5678"
    assert cfg.api_key == "explicitkey99"


def test_load_config_kwargs(clean_env):
    cfg = load_config(url="http://h:5678", timeout=12.0)
    assert cfg.timeout == 12.0


def test_load_config_with_file(clean_env, tmp_path):
    f = tmp_path / "config.json"
    f.write_text(json.dumps({"url": "http://fromfile:5678", "timeout": 99.0}))
    cfg = load_config(config_file=str(f))
    assert cfg.url == "http://fromfile:5678"
    assert cfg.timeout == 99.0


def test_load_config_explicit_overrides_file(clean_env, tmp_path):
    f = tmp_path / "config.json"
    f.write_text(json.dumps({"url": "http://fromfile:5678"}))
    cfg = load_config(url="http://override:5678", config_file=str(f))
    assert cfg.url == "http://override:5678"


# --------------------------------------------------------------------------- #
# load_config_file
# --------------------------------------------------------------------------- #
def test_load_config_file_json(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"url": "http://h:5678", "timeout": 5.0}))
    data = load_config_file(str(f))
    assert data["url"] == "http://h:5678"


def test_load_config_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config_file(str(tmp_path / "nope.json"))


def test_load_config_file_yaml(tmp_path):
    pytest.importorskip("yaml")
    f = tmp_path / "c.yaml"
    f.write_text("url: http://h:5678\ntimeout: 7.0\n")
    data = load_config_file(str(f))
    assert data["url"] == "http://h:5678"
    assert data["timeout"] == 7.0


def test_load_config_file_toml(tmp_path):
    # No importorskip: `tomllib` is stdlib from 3.11, this package's floor. The
    # guard used to be `importorskip("tomli")`, which would now skip forever
    # (the backport is no longer a dependency) and silently retire the only
    # coverage of the TOML branch.
    f = tmp_path / "c.toml"
    f.write_text('url = "http://h:5678"\ntimeout = 8.0\n')
    data = load_config_file(str(f))
    assert data["url"] == "http://h:5678"
    assert data["timeout"] == 8.0
