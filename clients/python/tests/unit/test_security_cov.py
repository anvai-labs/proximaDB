"""Offline unit tests for proximadb_sdk.security.

Fully offline: every OAuth2/audit network call goes through `requests`,
which we monkeypatch with a fake module. No real sockets, no sleeps that
matter (refresh backoff sleep is patched away on the retry paths).
"""

import base64
import sys
from datetime import UTC, datetime, timedelta

import pytest

from proximadb_sdk import security
from proximadb_sdk.security import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    MTLSConfig,
    OAuth2Config,
    OAuth2Error,
    OAuth2GrantType,
    OAuth2Provider,
    OAuth2TokenManager,
    OAuth2TokenResponse,
    RBACManager,
    Role,
    RoleDefinition,
    SecurityContext,
    SecurityManager,
    clear_security_context,
    get_current_security_context,
    security_context,
    set_security_context,
)

# ---------------------------------------------------------------------------
# Fake requests transport
# ---------------------------------------------------------------------------


class FakeResp:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        return self._json


class FakeRequests:
    """Minimal stand-in for the `requests` module."""

    class exceptions:
        class RequestException(Exception):
            pass

    def __init__(self):
        self.calls = []
        self._responses = []
        self._raise = None

    def queue(self, resp):
        self._responses.append(resp)

    def set_raise(self, exc):
        self._raise = exc

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self._raise is not None:
            exc = self._raise
            self._raise = None
            raise exc
        if self._responses:
            return self._responses.pop(0)
        return FakeResp(200, {"access_token": "default-token"})


@pytest.fixture
def fake_requests(monkeypatch):
    fr = FakeRequests()
    # security.py does `import requests` lazily inside methods, so install
    # a fake module into sys.modules.
    monkeypatch.setitem(sys.modules, "requests", fr)
    return fr


# ---------------------------------------------------------------------------
# OAuth2TokenResponse
# ---------------------------------------------------------------------------


def test_token_response_expiry_properties():
    tok = OAuth2TokenResponse(access_token="a", expires_in=3600)
    assert tok.expires_at is not None
    assert not tok.is_expired
    delta = tok.time_until_expiry()
    assert delta is not None and delta.total_seconds() > 0


def test_token_response_no_expiry():
    tok = OAuth2TokenResponse(access_token="a")
    assert tok.expires_at is None
    assert tok.is_expired is False
    assert tok.time_until_expiry() is None


def test_token_response_is_expired_true():
    tok = OAuth2TokenResponse(access_token="a", expires_in=10)
    tok.issued_at = datetime.now(UTC) - timedelta(seconds=100)
    assert tok.is_expired is True


# ---------------------------------------------------------------------------
# OAuth2Config.get_token_url
# ---------------------------------------------------------------------------


def test_config_get_token_url_explicit():
    cfg = OAuth2Config(token_url="https://explicit/token")
    assert cfg.get_token_url() == "https://explicit/token"


def test_config_get_token_url_google():
    cfg = OAuth2Config(provider=OAuth2Provider.GOOGLE)
    assert cfg.get_token_url() == "https://oauth2.googleapis.com/token"


def test_config_get_token_url_okta():
    cfg = OAuth2Config(provider=OAuth2Provider.OKTA, client_id="myorg.app")
    assert "myorg" in cfg.get_token_url()


def test_config_get_token_url_auth0():
    cfg = OAuth2Config(provider=OAuth2Provider.AUTH0, audience="tenant.auth0.com")
    # Assert the full URL, not a substring: a substring check would also pass
    # for a host like "tenant.auth0.com.evil.example" (CodeQL
    # py/incomplete-url-substring-sanitization).
    assert cfg.get_token_url() == "https://tenant.auth0.com/oauth/token"


def test_config_get_token_url_generic_empty():
    cfg = OAuth2Config(provider=OAuth2Provider.GENERIC)
    assert cfg.get_token_url() == ""


def test_grant_type_enum():
    assert OAuth2GrantType.CLIENT_CREDENTIALS.value == "client_credentials"


# ---------------------------------------------------------------------------
# OAuth2TokenManager
# ---------------------------------------------------------------------------


def test_exchange_code_success(fake_requests):
    fake_requests.queue(
        FakeResp(
            200,
            {
                "access_token": "AT",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": "RT",
                "scope": "openid",
                "id_token": "ID",
            },
        )
    )
    cfg = OAuth2Config(token_url="https://idp/token", client_id="c", client_secret="s")
    mgr = OAuth2TokenManager(cfg)
    tok = mgr.exchange_code("the-code", code_verifier="verifier")
    assert tok.access_token == "AT"
    assert tok.refresh_token == "RT"
    _, kw = fake_requests.calls[0]
    assert kw["data"]["code_verifier"] == "verifier"
    assert kw["data"]["client_secret"] == "s"


def test_exchange_code_uses_stored_pkce_verifier(fake_requests):
    fake_requests.queue(FakeResp(200, {"access_token": "AT"}))
    cfg = OAuth2Config(token_url="https://idp/token", client_id="c")
    mgr = OAuth2TokenManager(cfg)
    mgr.generate_pkce()  # sets _pkce_verifier
    mgr.exchange_code("code")
    _, kw = fake_requests.calls[0]
    assert "code_verifier" in kw["data"]


def test_exchange_code_failure(fake_requests):
    fake_requests.queue(FakeResp(401, {}, text="bad"))
    cfg = OAuth2Config(token_url="https://idp/token", client_id="c")
    mgr = OAuth2TokenManager(cfg)
    with pytest.raises(OAuth2Error):
        mgr.exchange_code("code")


def test_client_credentials_success(fake_requests):
    fake_requests.queue(
        FakeResp(200, {"access_token": "AT", "expires_in": 600, "scope": "api"})
    )
    cfg = OAuth2Config(
        token_url="https://idp/token",
        client_id="c",
        client_secret="s",
        scopes=["a", "b"],
        audience="aud",
    )
    mgr = OAuth2TokenManager(cfg)
    tok = mgr.client_credentials()
    assert tok.access_token == "AT"
    _, kw = fake_requests.calls[0]
    assert kw["data"]["scope"] == "a b"
    assert kw["data"]["audience"] == "aud"


def test_client_credentials_requires_secret(fake_requests):
    cfg = OAuth2Config(token_url="https://idp/token", client_id="c")
    mgr = OAuth2TokenManager(cfg)
    with pytest.raises(OAuth2Error):
        mgr.client_credentials()


def test_client_credentials_failure(fake_requests):
    fake_requests.queue(FakeResp(500, {}, text="boom"))
    cfg = OAuth2Config(token_url="https://idp/token", client_id="c", client_secret="s")
    mgr = OAuth2TokenManager(cfg)
    with pytest.raises(OAuth2Error):
        mgr.client_credentials()


def test_refresh_success_and_callback(fake_requests):
    fake_requests.queue(
        FakeResp(200, {"access_token": "AT2", "expires_in": 3600, "scope": "x"})
    )
    cfg = OAuth2Config(token_url="https://idp/token", client_id="c", client_secret="s")
    mgr = OAuth2TokenManager(cfg)
    mgr.token = OAuth2TokenResponse(access_token="AT1", refresh_token="RT1")

    seen = []
    mgr.on_token_refresh(lambda t: seen.append(t.access_token))
    mgr.on_token_refresh(lambda t: (_ for _ in ()).throw(RuntimeError("x")))

    tok = mgr.refresh()
    assert tok.access_token == "AT2"
    assert tok.refresh_token == "RT1"  # preserved when IdP omits it
    assert seen == ["AT2"]


def test_refresh_no_token_raises(fake_requests):
    cfg = OAuth2Config(token_url="https://idp/token", client_id="c")
    mgr = OAuth2TokenManager(cfg)
    with pytest.raises(OAuth2Error):
        mgr.refresh()


def test_refresh_invalid_400_no_retry(fake_requests):
    fake_requests.queue(FakeResp(400, {}, text="invalid_grant"))
    cfg = OAuth2Config(token_url="https://idp/token", client_id="c")
    mgr = OAuth2TokenManager(cfg)
    mgr.token = OAuth2TokenResponse(access_token="AT", refresh_token="RT")
    with pytest.raises(OAuth2Error):
        mgr.refresh()


def test_refresh_request_exception_exhausts_attempts(fake_requests, monkeypatch):
    monkeypatch.setattr(security.time, "sleep", lambda *a, **k: None)
    cfg = OAuth2Config(
        token_url="https://idp/token", client_id="c", max_refresh_attempts=2
    )
    mgr = OAuth2TokenManager(cfg)
    mgr.token = OAuth2TokenResponse(access_token="AT", refresh_token="RT")

    def always_raise(url, **kw):
        raise fake_requests.exceptions.RequestException("net down")

    monkeypatch.setattr(fake_requests, "post", always_raise)
    with pytest.raises(OAuth2Error):
        mgr.refresh()


def test_refresh_non_200_non_400_exhausts(fake_requests, monkeypatch):
    monkeypatch.setattr(security.time, "sleep", lambda *a, **k: None)
    fake_requests.queue(FakeResp(503, {}))
    fake_requests.queue(FakeResp(503, {}))
    cfg = OAuth2Config(
        token_url="https://idp/token", client_id="c", max_refresh_attempts=2
    )
    mgr = OAuth2TokenManager(cfg)
    mgr.token = OAuth2TokenResponse(access_token="AT", refresh_token="RT")
    with pytest.raises(OAuth2Error):
        mgr.refresh()


def test_token_property_auto_refreshes(fake_requests):
    fake_requests.queue(FakeResp(200, {"access_token": "FRESH", "expires_in": 3600}))
    cfg = OAuth2Config(token_url="https://idp/token", client_id="c", auto_refresh=True)
    mgr = OAuth2TokenManager(cfg)
    expired = OAuth2TokenResponse(access_token="OLD", expires_in=10, refresh_token="RT")
    expired.issued_at = datetime.now(UTC) - timedelta(seconds=100)
    mgr.token = expired
    assert mgr.token.access_token == "FRESH"


def test_token_property_none():
    cfg = OAuth2Config(token_url="https://idp/token", client_id="c")
    mgr = OAuth2TokenManager(cfg)
    assert mgr.token is None


def test_generate_pkce_values():
    cfg = OAuth2Config(client_id="c")
    mgr = OAuth2TokenManager(cfg)
    verifier, challenge = mgr.generate_pkce()
    assert isinstance(verifier, str) and isinstance(challenge, str)
    assert "=" not in verifier and "=" not in challenge
    import hashlib

    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    assert challenge == expected


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------


def test_role_definition_post_init_defaults_inherits():
    rd = RoleDefinition(name="x", permissions={"a:b"})
    assert rd.inherits == []


def test_rbac_default_roles_and_enum():
    assert Role.ADMIN.value == "admin"
    rbac = RBACManager()
    perms = rbac.get_effective_permissions(["admin"])
    assert "system:*" in perms


def test_rbac_inheritance():
    rbac = RBACManager()
    perms = rbac.get_effective_permissions(["developer"])
    assert "collection:list" in perms  # from viewer
    assert "vector:search" in perms  # from analyst
    assert "collection:create" in perms  # developer's own


def test_rbac_permission_cache_hit():
    rbac = RBACManager()
    first = rbac.get_effective_permissions(["viewer"])
    second = rbac.get_effective_permissions(["viewer"])
    assert first is second  # cached object returned


def test_rbac_unknown_role():
    rbac = RBACManager()
    assert rbac.get_effective_permissions(["nope"]) == set()


def test_rbac_register_role_invalidates_cache():
    rbac = RBACManager()
    rbac.get_effective_permissions(["viewer"])
    rbac.register_role(RoleDefinition(name="custom", permissions={"thing:do"}))
    assert "thing:do" in rbac.get_effective_permissions(["custom"])


def test_rbac_custom_roles_constructor():
    custom = {"sp": RoleDefinition(name="sp", permissions={"x:y"})}
    rbac = RBACManager(custom_roles=custom)
    assert "x:y" in rbac.get_effective_permissions(["sp"])


def test_rbac_check_permission_wildcard():
    rbac = RBACManager()
    assert rbac.check_permission(["admin"], "vector:search") is True


def test_rbac_check_permission_exact_and_deny():
    rbac = RBACManager()
    assert rbac.check_permission(["viewer"], "collection:list") is True
    assert rbac.check_permission(["viewer"], "vector:delete") is False


def test_rbac_audit_callback_invoked():
    rbac = RBACManager()
    events = []
    rbac.set_audit_callback(events.append)
    rbac.check_permission(["viewer"], "collection:list", resource="col-1")
    assert events and events[0]["allowed"] is True
    assert events[0]["resource"] == "col-1"


def test_require_permission_decorator_allows():
    rbac = RBACManager()

    class Service:
        def __init__(self, ctx):
            self._security_context = ctx

        @rbac.require_permission("collection:list")
        def do(self):
            return "ok"

    ctx = SecurityContext(user_id="u", roles=["viewer"])
    assert Service(ctx).do() == "ok"


def test_require_permission_decorator_denies():
    rbac = RBACManager()

    class Service:
        def __init__(self, ctx):
            self._security_context = ctx

        @rbac.require_permission("vector:delete")
        def do(self):
            return "ok"

    ctx = SecurityContext(user_id="u", roles=["viewer"])
    with pytest.raises(PermissionError):
        Service(ctx).do()


def test_require_permission_decorator_no_context():
    rbac = RBACManager()

    @rbac.require_permission("vector:read")
    def standalone():
        return "ok"

    with pytest.raises(PermissionError):
        standalone()


def test_require_permission_via_kwarg_context():
    rbac = RBACManager()

    @rbac.require_permission("collection:list")
    def fn(security_context=None):
        return "ok"

    ctx = SecurityContext(user_id="u", roles=["viewer"])
    assert fn(security_context=ctx) == "ok"


def test_require_any_permission_allows():
    rbac = RBACManager()

    class Service:
        def __init__(self, ctx):
            self._security_context = ctx

        @rbac.require_any_permission(["vector:delete", "collection:list"])
        def do(self):
            return "ok"

    ctx = SecurityContext(user_id="u", roles=["viewer"])
    assert Service(ctx).do() == "ok"


def test_require_any_permission_denies():
    rbac = RBACManager()

    class Service:
        def __init__(self, ctx):
            self._security_context = ctx

        @rbac.require_any_permission(["vector:delete", "graph:delete"])
        def do(self):
            return "ok"

    ctx = SecurityContext(user_id="u", roles=["viewer"])
    with pytest.raises(PermissionError):
        Service(ctx).do()


def test_require_any_permission_no_context():
    rbac = RBACManager()

    @rbac.require_any_permission(["vector:read"])
    def standalone():
        return "ok"

    with pytest.raises(PermissionError):
        standalone()


# ---------------------------------------------------------------------------
# SecurityContext + thread-local helpers
# ---------------------------------------------------------------------------


def test_security_context_has_permission_and_role():
    ctx = SecurityContext(
        user_id="u", roles=["viewer"], permissions={"collection:list", "vector:*"}
    )
    assert ctx.has_permission("collection:list") is True
    assert ctx.has_permission("vector:search") is True  # wildcard
    assert ctx.has_permission("graph:read") is False
    assert ctx.has_role("viewer") is True
    assert ctx.has_role("admin") is False


def test_thread_local_set_get_clear():
    clear_security_context()
    assert get_current_security_context() is None
    ctx = SecurityContext(user_id="u")
    set_security_context(ctx)
    assert get_current_security_context() is ctx
    clear_security_context()
    assert get_current_security_context() is None
    clear_security_context()  # idempotent when absent


def test_security_context_manager_restores_previous():
    clear_security_context()
    outer = SecurityContext(user_id="outer")
    inner = SecurityContext(user_id="inner")
    set_security_context(outer)
    with security_context(inner) as c:
        assert c is inner
        assert get_current_security_context() is inner
    assert get_current_security_context() is outer
    clear_security_context()


def test_security_context_manager_clears_when_no_previous():
    clear_security_context()
    inner = SecurityContext(user_id="inner")
    with security_context(inner):
        assert get_current_security_context() is inner
    assert get_current_security_context() is None


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


def test_audit_event_to_dict_and_json():
    ev = AuditEvent(
        event_id="e1",
        event_type=AuditEventType.SYSTEM,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        user_id="u",
        tenant_id="t",
        action="act",
        resource_type="rt",
        resource_id="rid",
        outcome="success",
    )
    d = ev.to_dict()
    assert d["event_type"] == "system"
    assert d["event_id"] == "e1"
    import json

    parsed = json.loads(ev.to_json())
    assert parsed["action"] == "act"


def test_audit_logger_log_uses_context_and_callback():
    clear_security_context()
    audit = AuditLogger()
    received = []
    audit.on_event(received.append)
    audit.on_event(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))

    ctx = SecurityContext(
        user_id="alice",
        tenant_id="t1",
        client_ip="1.2.3.4",
        user_agent="agent",
        request_id="rq",
        session_id="ss",
    )
    set_security_context(ctx)
    ev = audit.log(
        AuditEventType.DATA_ACCESS,
        action="read",
        resource_type="vector",
        resource_id="v1",
        details={"k": "v"},
    )
    clear_security_context()
    assert ev.user_id == "alice"
    assert ev.tenant_id == "t1"
    assert ev.client_ip == "1.2.3.4"
    assert received and received[0] is ev


def test_audit_logger_log_without_context():
    clear_security_context()
    audit = AuditLogger()
    ev = audit.log(AuditEventType.SYSTEM, "boot", "system")
    assert ev.user_id == "system"
    assert ev.tenant_id is None


def test_audit_logger_helpers():
    clear_security_context()
    audit = AuditLogger()
    a = audit.log_authentication("u", "password", "success")
    assert a.event_type == AuditEventType.AUTHENTICATION
    assert a.action == "authenticate_password"

    b = audit.log_authorization("vector:search", "v1", allowed=False)
    assert b.outcome == "denied"
    assert b.resource_type == "vector"

    c = audit.log_data_access("query", "graph", "g1", details={"x": 1})
    assert c.event_type == AuditEventType.DATA_ACCESS


def test_audit_logger_file_output(tmp_path):
    clear_security_context()
    log_file = tmp_path / "audit.log"
    audit = AuditLogger(log_file=str(log_file))
    audit.log(AuditEventType.SYSTEM, "boot", "system")
    content = log_file.read_text().strip()
    assert "boot" in content


def test_audit_logger_signing_chain_hash():
    clear_security_context()
    audit = AuditLogger(enable_signing=True, signing_key=b"key")
    ev1 = audit.log(AuditEventType.SYSTEM, "a", "system")
    ev2 = audit.log(AuditEventType.SYSTEM, "b", "system")
    assert "chain_hash" in ev1.metadata
    assert ev1.metadata["chain_hash"] != ev2.metadata["chain_hash"]


def test_audit_logger_signing_without_key():
    clear_security_context()
    audit = AuditLogger(enable_signing=True)  # plain sha256 path
    ev = audit.log(AuditEventType.SYSTEM, "a", "system")
    assert "chain_hash" in ev.metadata


def test_audit_logger_remote_batch_flush(fake_requests):
    clear_security_context()
    fake_requests.queue(FakeResp(200, {}))
    audit = AuditLogger(remote_endpoint="https://collector/ingest")
    audit._batch_size = 2
    audit.log(AuditEventType.SYSTEM, "a", "system")
    assert len(audit._batch) == 1
    audit.log(AuditEventType.SYSTEM, "b", "system")
    assert audit._batch == []  # flushed on success
    assert fake_requests.calls


def test_audit_logger_flush_empty_noop(fake_requests):
    audit = AuditLogger(remote_endpoint="https://collector/ingest")
    audit._flush_batch()  # empty -> early return
    assert fake_requests.calls == []


def test_audit_logger_flush_failure_keeps_batch(fake_requests):
    clear_security_context()
    fake_requests.set_raise(RuntimeError("net"))
    audit = AuditLogger(remote_endpoint="https://collector/ingest")
    audit._batch.append(
        AuditEvent(
            event_id="e",
            event_type=AuditEventType.SYSTEM,
            timestamp=datetime.now(UTC),
            user_id="u",
            tenant_id=None,
            action="a",
            resource_type="system",
            resource_id=None,
            outcome="success",
        )
    )
    audit._flush_batch()
    assert len(audit._batch) == 1  # exception swallowed, batch preserved


def test_audit_logger_write_file_error_swallowed():
    clear_security_context()
    audit = AuditLogger(log_file="/nonexistent_dir/does/not/exist.log")
    ev = audit.log(AuditEventType.SYSTEM, "a", "system")
    assert ev is not None


# ---------------------------------------------------------------------------
# mTLS config
# ---------------------------------------------------------------------------


def test_mtls_validate_disabled_no_issues():
    cfg = MTLSConfig(enabled=False, client_cert_path="/nope.pem")
    assert cfg.validate() == []


def test_mtls_validate_reports_missing_files():
    cfg = MTLSConfig(
        enabled=True,
        client_cert_path="/no/cert.pem",
        client_key_path="/no/key.pem",
        ca_cert_path="/no/ca.pem",
    )
    issues = cfg.validate()
    assert len(issues) == 3


def test_mtls_create_ssl_context_default_certs():
    cfg = MTLSConfig(min_tls_version="TLSv1.3", verify_hostname=False)
    ctx = cfg.create_ssl_context()
    import ssl

    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_3
    assert ctx.check_hostname is False


def test_mtls_create_ssl_context_unknown_version_defaults():
    cfg = MTLSConfig(min_tls_version="bogus")
    ctx = cfg.create_ssl_context()
    import ssl

    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2


def test_mtls_create_ssl_context_with_certs_and_ciphers(monkeypatch, tmp_path):
    cfg = MTLSConfig(
        client_cert_path=str(tmp_path / "c.pem"),
        client_key_path=str(tmp_path / "k.pem"),
        ca_cert_path=str(tmp_path / "ca.pem"),
        allowed_ciphers=["ECDHE-RSA-AES128-GCM-SHA256"],
    )
    import ssl as ssl_mod

    loaded = {}

    def fake_load_cert_chain(self, certfile, keyfile):
        loaded["cert"] = (certfile, keyfile)

    def fake_load_verify_locations(self, cafile):
        loaded["ca"] = cafile

    def fake_set_ciphers(self, c):
        loaded["ciphers"] = c

    monkeypatch.setattr(ssl_mod.SSLContext, "load_cert_chain", fake_load_cert_chain)
    monkeypatch.setattr(
        ssl_mod.SSLContext, "load_verify_locations", fake_load_verify_locations
    )
    monkeypatch.setattr(ssl_mod.SSLContext, "set_ciphers", fake_set_ciphers)

    cfg.create_ssl_context()
    assert "cert" in loaded
    assert "ca" in loaded
    assert "ciphers" in loaded


# ---------------------------------------------------------------------------
# SecurityManager
# ---------------------------------------------------------------------------


def test_security_manager_defaults():
    mgr = SecurityManager()
    assert mgr.oauth2 is None
    assert isinstance(mgr.rbac, RBACManager)
    assert isinstance(mgr.audit, AuditLogger)
    assert mgr.mtls is None


def test_security_manager_create_context_and_check():
    clear_security_context()
    mgr = SecurityManager()
    ctx = mgr.create_context(
        user_id="u",
        tenant_id="t",
        roles=["viewer"],
        client_ip="1.1.1.1",
        user_agent="ua",
    )
    assert "collection:list" in ctx.permissions
    assert ctx.request_id is not None

    assert mgr.check_permission("collection:list") is False  # no current ctx

    with security_context(ctx):
        assert mgr.check_permission("collection:list") is True
        assert mgr.check_permission("vector:delete") is False
    clear_security_context()


def test_security_manager_create_context_no_roles():
    mgr = SecurityManager()
    ctx = mgr.create_context(user_id="u")
    assert ctx.roles == []
    assert ctx.permissions == set()


def test_security_manager_rbac_audit_wiring():
    mgr = SecurityManager()
    seen = []
    mgr.audit.on_event(seen.append)
    mgr.rbac.check_permission(["viewer"], "collection:list", resource="c1")
    assert seen
    assert seen[0].event_type == AuditEventType.AUTHORIZATION


def test_security_manager_require_permission_decorator():
    mgr = SecurityManager()
    dec = mgr.require_permission("collection:list")

    class S:
        def __init__(self, ctx):
            self._security_context = ctx

        @dec
        def do(self):
            return "ok"

    ctx = SecurityContext(user_id="u", roles=["viewer"])
    assert S(ctx).do() == "ok"


def test_security_manager_get_ssl_context_none_when_disabled():
    mgr = SecurityManager(mtls_config=MTLSConfig(enabled=False))
    assert mgr.get_ssl_context() is None
    assert SecurityManager().get_ssl_context() is None


def test_security_manager_get_ssl_context_when_enabled():
    mgr = SecurityManager(mtls_config=MTLSConfig(enabled=True))
    ctx = mgr.get_ssl_context()
    import ssl

    assert isinstance(ctx, ssl.SSLContext)


def test_security_manager_with_oauth2_config():
    cfg = OAuth2Config(token_url="https://idp/token", client_id="c")
    mgr = SecurityManager(oauth2_config=cfg)
    assert isinstance(mgr.oauth2, OAuth2TokenManager)
