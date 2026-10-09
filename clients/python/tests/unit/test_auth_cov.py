"""Offline unit tests for proximadb_sdk.auth — coverage-focused.

Fully offline: a fake requests.Session is injected, so no real HTTP is ever
performed. time/sleep are never invoked by this module (retry lives in urllib3,
which we never reach because the session is faked).
"""

from datetime import UTC, datetime, timedelta

import pytest
import requests

from proximadb_sdk.auth import (
    AuthConfig,
    AuthenticationError,
    AuthMethod,
    AuthorizationError,
    AuthResult,
    Permission,
    ProximaDBAuth,
    TokenExpiredError,
    create_api_key_auth,
    create_cert_auth,
    create_jwt_auth,
    create_oauth2_auth,
)

# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}

    def json(self):
        return self._json


class FakeSession:
    """Drop-in for requests.Session that records calls and returns queued responses."""

    def __init__(self):
        self.cert = None
        self.verify = True
        self.calls = []
        # map of (method, url-suffix) -> FakeResponse or Exception
        self._post_handler = None
        self._get_handler = None

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if self._post_handler is not None:
            return self._post_handler(url, kwargs)
        return FakeResponse(200, {})

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if self._get_handler is not None:
            return self._get_handler(url, kwargs)
        return FakeResponse(200, {})


def make_auth(config, session=None):
    session = session or FakeSession()
    return ProximaDBAuth(config=config, base_url="http://testserver/", session=session)


# ---------------------------------------------------------------------------
# AuthResult
# ---------------------------------------------------------------------------


def test_authresult_post_init_defaults():
    r = AuthResult(user_id="u")
    assert r.roles == []
    assert r.permissions == []
    assert r.auth_method == AuthMethod.API_KEY


def test_authresult_not_expired_when_none():
    r = AuthResult(user_id="u", token_expires_at=None)
    assert r.is_expired() is False


def test_authresult_expired_true():
    past = datetime.now(UTC) - timedelta(hours=1)
    r = AuthResult(user_id="u", token_expires_at=past)
    assert r.is_expired() is True


def test_authresult_expired_false_future():
    future = datetime.now(UTC) + timedelta(hours=1)
    r = AuthResult(user_id="u", token_expires_at=future)
    assert r.is_expired() is False


def test_authresult_has_permission():
    r = AuthResult(user_id="u", permissions=[Permission.SEARCH_VECTORS])
    assert r.has_permission(Permission.SEARCH_VECTORS) is True
    assert r.has_permission(Permission.MANAGE_USERS) is False


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_authorization_error_carries_permission():
    e = AuthorizationError("nope", required_permission=Permission.MANAGE_ROLES)
    assert e.required_permission == Permission.MANAGE_ROLES
    assert "nope" in str(e)


def test_token_expired_is_authentication_error():
    assert issubclass(TokenExpiredError, AuthenticationError)


# ---------------------------------------------------------------------------
# Init / session config
# ---------------------------------------------------------------------------


def test_init_strips_trailing_slash():
    a = make_auth(AuthConfig())
    assert a.base_url == "http://testserver"


def test_init_default_session_created(monkeypatch):
    # _create_session builds a real requests.Session but mounts adapters only.
    cfg = AuthConfig()
    a = ProximaDBAuth(config=cfg, base_url="http://x")
    assert isinstance(a.session, requests.Session)


def test_init_applies_client_cert_and_ca():
    cfg = AuthConfig(
        client_cert_path="/c.pem",
        client_key_path="/k.pem",
        ca_cert_path="/ca.pem",
    )
    sess = FakeSession()
    a = make_auth(cfg, sess)
    assert sess.cert == ("/c.pem", "/k.pem")
    assert sess.verify == "/ca.pem"


# ---------------------------------------------------------------------------
# authenticate() dispatch
# ---------------------------------------------------------------------------


def test_authenticate_disabled_grants_all():
    a = make_auth(AuthConfig(enabled=False))
    r = a.authenticate()
    assert r.user_id == "anonymous"
    assert set(r.permissions) == set(Permission)


def test_authenticate_api_key():
    a = make_auth(AuthConfig(enabled=True, api_key="k1"))
    r = a.authenticate()
    assert r.user_id == "api_key_user"
    assert r.auth_method == AuthMethod.API_KEY
    assert Permission.SEARCH_VECTORS in r.permissions


def test_authenticate_no_method_raises():
    a = make_auth(AuthConfig(enabled=True))
    with pytest.raises(AuthenticationError):
        a.authenticate()


def test_authenticate_dispatch_jwt():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(
        200,
        {
            "user_id": "jwtuser",
            "tenant_id": "t1",
            "roles": ["admin"],
            "permissions": ["SearchVectors"],
            "expires_at": "2030-01-01T00:00:00Z",
        },
    )
    a = make_auth(AuthConfig(enabled=True, jwt_token="tok"), sess)
    r = a.authenticate()
    assert r.auth_method == AuthMethod.JWT_TOKEN
    assert r.user_id == "jwtuser"


def test_authenticate_dispatch_oauth2():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(200, {"user_id": "o1"})
    a = make_auth(AuthConfig(enabled=True, oauth2_token="otok"), sess)
    r = a.authenticate()
    assert r.auth_method == AuthMethod.OAUTH2


def test_authenticate_dispatch_cert():
    sess = FakeSession()
    sess._get_handler = lambda url, kw: FakeResponse(200, {"user_id": "c1"})
    cfg = AuthConfig(enabled=True, client_cert_path="/c.pem", client_key_path="/k.pem")
    a = make_auth(cfg, sess)
    r = a.authenticate()
    assert r.auth_method == AuthMethod.CLIENT_CERTIFICATE


# ---------------------------------------------------------------------------
# _authenticate_api_key error path
# ---------------------------------------------------------------------------


def test_authenticate_api_key_missing_raises():
    a = make_auth(AuthConfig(enabled=True, api_key="x"))
    a.config.api_key = None
    with pytest.raises(AuthenticationError):
        a._authenticate_api_key()


# ---------------------------------------------------------------------------
# JWT validation
# ---------------------------------------------------------------------------


def test_jwt_missing_token_raises():
    a = make_auth(AuthConfig(enabled=True))
    with pytest.raises(AuthenticationError):
        a._authenticate_jwt()


def test_jwt_401_raises():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(401)
    a = make_auth(AuthConfig(enabled=True, jwt_token="t"), sess)
    with pytest.raises(AuthenticationError):
        a._authenticate_jwt()


def test_jwt_403_raises_authorization():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(403)
    a = make_auth(AuthConfig(enabled=True, jwt_token="t"), sess)
    with pytest.raises(AuthorizationError):
        a._authenticate_jwt()


def test_jwt_other_status_raises():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(500)
    a = make_auth(AuthConfig(enabled=True, jwt_token="t"), sess)
    with pytest.raises(AuthenticationError):
        a._authenticate_jwt()


def test_jwt_success_full_parse():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(
        200,
        {
            "user_id": "u",
            "tenant_id": "t",
            "roles": ["r1"],
            "permissions": ["SearchVectors", "ReadVectors"],
            "expires_at": "2030-06-01T12:00:00Z",
        },
    )
    cfg = AuthConfig(enabled=True, jwt_token="acc", jwt_refresh_token="ref")
    a = make_auth(cfg, sess)
    r = a._authenticate_jwt()
    assert r.user_id == "u"
    assert r.tenant_id == "t"
    assert r.roles == ["r1"]
    assert Permission.READ_VECTORS in r.permissions
    assert r.access_token == "acc"
    assert r.refresh_token == "ref"
    assert r.token_expires_at is not None


def test_jwt_network_failure_falls_back_offline():
    sess = FakeSession()

    def boom(url, kw):
        raise requests.exceptions.ConnectionError("down")

    sess._post_handler = boom
    cfg = AuthConfig(enabled=True, jwt_token="acc", jwt_refresh_token="ref")
    a = make_auth(cfg, sess)
    with pytest.warns(UserWarning):
        r = a._authenticate_jwt()
    assert r.user_id == "jwt_user_offline"
    assert r.access_token == "acc"


# ---------------------------------------------------------------------------
# OAuth2
# ---------------------------------------------------------------------------


def test_oauth2_missing_token_raises():
    a = make_auth(AuthConfig(enabled=True))
    with pytest.raises(AuthenticationError):
        a._authenticate_oauth2()


def test_oauth2_non_200_raises():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(400)
    a = make_auth(AuthConfig(enabled=True, oauth2_token="t"), sess)
    with pytest.raises(AuthenticationError):
        a._authenticate_oauth2()


def test_oauth2_success():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(
        200,
        {
            "user_id": "ou",
            "tenant_id": "ot",
            "roles": ["x"],
            "permissions": ["SearchVectors"],
            "expires_at": "2030-01-01T00:00:00+00:00",
        },
    )
    a = make_auth(AuthConfig(enabled=True, oauth2_token="otok"), sess)
    r = a._authenticate_oauth2()
    assert r.user_id == "ou"
    assert r.access_token == "otok"
    assert r.auth_method == AuthMethod.OAUTH2


def test_oauth2_network_error_raises():
    sess = FakeSession()

    def boom(url, kw):
        raise requests.exceptions.Timeout("t")

    sess._post_handler = boom
    a = make_auth(AuthConfig(enabled=True, oauth2_token="t"), sess)
    with pytest.raises(AuthenticationError):
        a._authenticate_oauth2()


# ---------------------------------------------------------------------------
# Client cert
# ---------------------------------------------------------------------------


def test_cert_missing_raises():
    a = make_auth(AuthConfig(enabled=True))
    with pytest.raises(AuthenticationError):
        a._authenticate_client_cert()


def test_cert_non_200_raises():
    sess = FakeSession()
    sess._get_handler = lambda url, kw: FakeResponse(401)
    cfg = AuthConfig(enabled=True, client_cert_path="/c.pem", client_key_path="/k.pem")
    a = make_auth(cfg, sess)
    with pytest.raises(AuthenticationError):
        a._authenticate_client_cert()


def test_cert_success():
    sess = FakeSession()
    sess._get_handler = lambda url, kw: FakeResponse(
        200, {"user_id": "cu", "permissions": ["ReadVectors"]}
    )
    cfg = AuthConfig(enabled=True, client_cert_path="/c.pem", client_key_path="/k.pem")
    a = make_auth(cfg, sess)
    r = a._authenticate_client_cert()
    assert r.user_id == "cu"
    assert r.auth_method == AuthMethod.CLIENT_CERTIFICATE


def test_cert_network_error_raises():
    sess = FakeSession()

    def boom(url, kw):
        raise requests.exceptions.ConnectionError("x")

    sess._get_handler = boom
    cfg = AuthConfig(enabled=True, client_cert_path="/c.pem", client_key_path="/k.pem")
    a = make_auth(cfg, sess)
    with pytest.raises(AuthenticationError):
        a._authenticate_client_cert()


# ---------------------------------------------------------------------------
# _parse_expiration
# ---------------------------------------------------------------------------


def test_parse_expiration_none():
    a = make_auth(AuthConfig())
    assert a._parse_expiration(None) is None
    assert a._parse_expiration("") is None


def test_parse_expiration_iso_with_z():
    a = make_auth(AuthConfig())
    dt = a._parse_expiration("2030-01-01T00:00:00Z")
    assert dt.year == 2030


def test_parse_expiration_unix_seconds():
    a = make_auth(AuthConfig())
    dt = a._parse_expiration("1893456000")  # ~2030
    assert dt.tzinfo is not None
    assert dt.year >= 2030


def test_parse_expiration_unix_milliseconds():
    a = make_auth(AuthConfig())
    dt = a._parse_expiration("1893456000000")
    assert dt.year >= 2030


def test_parse_expiration_garbage_returns_none():
    a = make_auth(AuthConfig())
    assert a._parse_expiration("not-a-date") is None


# ---------------------------------------------------------------------------
# get_auth_headers
# ---------------------------------------------------------------------------


def test_headers_disabled_empty():
    a = make_auth(AuthConfig(enabled=False))
    assert a.get_auth_headers() == {}


def test_headers_api_key():
    a = make_auth(AuthConfig(enabled=True, api_key="kk"))
    h = a.get_auth_headers()
    assert h["Authorization"] == "API-Key kk"


def test_headers_jwt_bearer():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(
        200, {"user_id": "u", "permissions": []}
    )
    cfg = AuthConfig(enabled=True, jwt_token="jtok", auto_refresh_jwt=False)
    a = make_auth(cfg, sess)
    h = a.get_auth_headers()
    assert h["Authorization"] == "Bearer jtok"


def test_headers_triggers_refresh_when_near_expiry():
    sess = FakeSession()
    # validate returns near-expiry token, refresh returns new token
    near = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    calls = {"n": 0}

    def handler(url, kw):
        calls["n"] += 1
        if url.endswith("/auth/validate"):
            return FakeResponse(
                200, {"user_id": "u", "permissions": [], "expires_at": near}
            )
        if url.endswith("/auth/refresh"):
            return FakeResponse(
                200,
                {
                    "access_token": "newtok",
                    "refresh_token": "newref",
                    "expires_at": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
                },
            )
        return FakeResponse(200, {})

    sess._post_handler = handler
    cfg = AuthConfig(
        enabled=True,
        jwt_token="old",
        jwt_refresh_token="oldref",
        auto_refresh_jwt=True,
        refresh_threshold_minutes=5,
    )
    a = make_auth(cfg, sess)
    h = a.get_auth_headers()
    assert h["Authorization"] == "Bearer newtok"
    assert a.config.jwt_token == "newtok"


# ---------------------------------------------------------------------------
# _should_refresh_token
# ---------------------------------------------------------------------------


def test_should_refresh_disabled():
    a = make_auth(AuthConfig(enabled=True, auto_refresh_jwt=False))
    a.auth_result = AuthResult(user_id="u", token_expires_at=datetime.now(UTC))
    assert a._should_refresh_token() is False


def test_should_refresh_no_expiry():
    a = make_auth(AuthConfig(enabled=True, auto_refresh_jwt=True))
    a.auth_result = AuthResult(user_id="u", token_expires_at=None)
    assert a._should_refresh_token() is False


def test_should_refresh_no_auth_result():
    a = make_auth(AuthConfig(enabled=True, auto_refresh_jwt=True))
    a.auth_result = None
    assert a._should_refresh_token() is False


def test_should_refresh_true_near_expiry():
    a = make_auth(
        AuthConfig(enabled=True, auto_refresh_jwt=True, refresh_threshold_minutes=5)
    )
    a.auth_result = AuthResult(
        user_id="u",
        token_expires_at=datetime.now(UTC) + timedelta(minutes=2),
    )
    assert a._should_refresh_token() is True


def test_should_refresh_false_far_expiry():
    a = make_auth(
        AuthConfig(enabled=True, auto_refresh_jwt=True, refresh_threshold_minutes=5)
    )
    a.auth_result = AuthResult(
        user_id="u",
        token_expires_at=datetime.now(UTC) + timedelta(hours=2),
    )
    assert a._should_refresh_token() is False


# ---------------------------------------------------------------------------
# _refresh_token
# ---------------------------------------------------------------------------


def test_refresh_no_refresh_token_noop():
    a = make_auth(AuthConfig(enabled=True))
    a.config.jwt_refresh_token = None
    a._refresh_token()  # should just log + return


def test_refresh_success_updates_state_and_callback():
    captured = {}

    def cb(result):
        captured["result"] = result

    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(
        200,
        {
            "access_token": "a2",
            "refresh_token": "r2",
            "expires_at": "2030-01-01T00:00:00Z",
        },
    )
    cfg = AuthConfig(
        enabled=True,
        jwt_token="a1",
        jwt_refresh_token="r1",
        token_refresh_callback=cb,
    )
    a = make_auth(cfg, sess)
    a.auth_result = AuthResult(user_id="u", access_token="a1", refresh_token="r1")
    a._refresh_token()
    assert a.config.jwt_token == "a2"
    assert a.auth_result.access_token == "a2"
    assert a.auth_result.token_expires_at is not None
    assert captured["result"] is a.auth_result


def test_refresh_non_200_calls_error_callback_and_raises():
    errors = []
    cfg = AuthConfig(
        enabled=True,
        jwt_refresh_token="r1",
        auth_error_callback=lambda e: errors.append(e),
    )
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(500)
    a = make_auth(cfg, sess)
    with pytest.raises(AuthenticationError):
        a._refresh_token()


def test_refresh_network_error_calls_error_callback_and_raises():
    errors = []
    cfg = AuthConfig(
        enabled=True,
        jwt_refresh_token="r1",
        auth_error_callback=lambda e: errors.append(e),
    )
    sess = FakeSession()

    def boom(url, kw):
        raise requests.exceptions.ConnectionError("x")

    sess._post_handler = boom
    a = make_auth(cfg, sess)
    with pytest.raises(AuthenticationError):
        a._refresh_token()
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# check_permission / require_permission
# ---------------------------------------------------------------------------


def test_check_permission_disabled_true():
    a = make_auth(AuthConfig(enabled=False))
    assert a.check_permission(Permission.MANAGE_USERS) is True


def test_check_permission_authenticates_if_needed():
    a = make_auth(AuthConfig(enabled=True, api_key="k"))
    assert a.check_permission(Permission.SEARCH_VECTORS) is True
    assert a.check_permission(Permission.MANAGE_USERS) is False


def test_require_permission_passes():
    a = make_auth(AuthConfig(enabled=True, api_key="k"))
    a.require_permission(Permission.SEARCH_VECTORS)  # no raise


def test_require_permission_raises():
    a = make_auth(AuthConfig(enabled=True, api_key="k"))
    with pytest.raises(AuthorizationError) as ei:
        a.require_permission(Permission.MANAGE_USERS)
    assert ei.value.required_permission == Permission.MANAGE_USERS


# ---------------------------------------------------------------------------
# login / logout
# ---------------------------------------------------------------------------


def test_login_success():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(
        200,
        {
            "user_id": "lu",
            "tenant_id": "lt",
            "roles": ["a"],
            "permissions": ["SearchVectors"],
            "access_token": "at",
            "refresh_token": "rt",
            "expires_at": "2030-01-01T00:00:00Z",
        },
    )
    a = make_auth(AuthConfig(enabled=True), sess)
    r = a.login("user", "pass")
    assert r.user_id == "lu"
    assert a.config.jwt_token == "at"
    assert a.config.auth_method == AuthMethod.JWT_TOKEN
    assert a.auth_result is r


def test_login_non_200_raises():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(401)
    a = make_auth(AuthConfig(enabled=True), sess)
    with pytest.raises(AuthenticationError):
        a.login("u", "p")


def test_login_network_error_raises():
    sess = FakeSession()

    def boom(url, kw):
        raise requests.exceptions.ConnectionError("x")

    sess._post_handler = boom
    a = make_auth(AuthConfig(enabled=True), sess)
    with pytest.raises(AuthenticationError):
        a.login("u", "p")


def test_logout_posts_and_clears():
    sess = FakeSession()
    sess._post_handler = lambda url, kw: FakeResponse(200, {})
    a = make_auth(AuthConfig(enabled=True, jwt_token="t"), sess)
    a.auth_result = AuthResult(user_id="u", access_token="acc")
    a.logout()
    assert a.auth_result is None
    assert a.config.jwt_token is None
    # logout endpoint was hit
    assert any(url.endswith("/auth/logout") for _, url, _ in sess.calls)


def test_logout_ignores_network_error():
    sess = FakeSession()

    def boom(url, kw):
        raise requests.exceptions.ConnectionError("x")

    sess._post_handler = boom
    a = make_auth(AuthConfig(enabled=True), sess)
    a.auth_result = AuthResult(user_id="u", access_token="acc")
    a.logout()  # swallows the error
    assert a.auth_result is None


def test_logout_no_auth_result():
    a = make_auth(AuthConfig(enabled=True))
    a.auth_result = None
    a.logout()  # no post, just clears
    assert a.config.jwt_token is None


# ---------------------------------------------------------------------------
# get_user_info
# ---------------------------------------------------------------------------


def test_get_user_info_none():
    a = make_auth(AuthConfig(enabled=True))
    a.auth_result = None
    assert a.get_user_info() is None


def test_get_user_info_with_expiry():
    a = make_auth(AuthConfig(enabled=True))
    exp = datetime(2030, 1, 1, tzinfo=UTC)
    a.auth_result = AuthResult(
        user_id="u",
        tenant_id="t",
        roles=["r"],
        permissions=[Permission.SEARCH_VECTORS],
        auth_method=AuthMethod.JWT_TOKEN,
        token_expires_at=exp,
    )
    info = a.get_user_info()
    assert info["user_id"] == "u"
    assert info["permissions"] == ["SearchVectors"]
    assert info["auth_method"] == "jwt_token"
    assert info["expires_at"] == exp.isoformat()


def test_get_user_info_no_expiry():
    a = make_auth(AuthConfig(enabled=True))
    a.auth_result = AuthResult(user_id="u")
    info = a.get_user_info()
    assert info["expires_at"] is None


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def test_create_api_key_auth():
    c = create_api_key_auth("k", refresh_threshold_minutes=10)
    assert c.enabled and c.api_key == "k"
    assert c.refresh_threshold_minutes == 10


def test_create_jwt_auth():
    c = create_jwt_auth("acc", refresh_token="ref", auto_refresh=False)
    assert c.jwt_token == "acc"
    assert c.jwt_refresh_token == "ref"
    assert c.auto_refresh_jwt is False


def test_create_oauth2_auth():
    c = create_oauth2_auth("tok", provider="google", client_id="cid")
    assert c.oauth2_token == "tok"
    assert c.oauth2_provider == "google"
    assert c.oauth2_client_id == "cid"


def test_create_cert_auth():
    c = create_cert_auth("/c.pem", "/k.pem", ca_path="/ca.pem")
    assert c.client_cert_path == "/c.pem"
    assert c.client_key_path == "/k.pem"
    assert c.ca_cert_path == "/ca.pem"
