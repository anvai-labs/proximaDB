"""Pins for the ``connect_*`` factory consolidation (2026-11-07).

The six public factories are now thin wrappers over ``proximadb_sdk._connect``.
These tests pin the semantics that must NOT change in the collapse:

1. Each factory still stamps exactly its own default (protocol or port_mode).
2. ``connect`` stamps nothing.
3. Factory defaults override caller-supplied kwargs — the pre-consolidation
   factories stamped their protocol/port_mode last, so
   ``connect_rest(url, protocol=GRPC)`` silently yielded REST. Preserving
   that override is what makes the collapse a true clone.
"""

import pytest

import proximadb_sdk as sdk
from proximadb_sdk import PortMode, Protocol


class _FakeClient:
    """Captures constructor kwargs without touching any transport."""

    last_kwargs = None

    def __init__(self, **kwargs):
        _FakeClient.last_kwargs = kwargs


@pytest.fixture()
def capture(monkeypatch):
    _FakeClient.last_kwargs = None
    monkeypatch.setattr(sdk, "ProximaDBClient", _FakeClient)
    return lambda: _FakeClient.last_kwargs


@pytest.mark.parametrize(
    ("factory", "key", "value"),
    [
        (sdk.connect_rest, "protocol", Protocol.REST),
        (sdk.connect_grpc, "protocol", Protocol.GRPC),
        (sdk.connect_arrow_flight, "protocol", Protocol.ARROW_FLIGHT),
        (sdk.connect_unified, "port_mode", PortMode.UNIFIED),
        (sdk.connect_legacy, "port_mode", PortMode.MULTI),
    ],
)
def test_each_factory_stamps_exactly_its_default(capture, factory, key, value):
    factory("http://localhost:5678")
    kwargs = capture()
    assert kwargs is not None, "factory must construct a client"
    # Full-dict compare: the factory must stamp its default and NOTHING
    # else — guards against a regression that stamps extra keys.
    assert kwargs == {"url": "http://localhost:5678", key: value}


def test_falsy_url_stamps_no_url_key(capture):
    """Pin the truthy-check: url="" must not stamp an url kwarg at all
    (a regression to `if url is not None` would silently stamp url="").
    """
    sdk.connect_rest("")
    kwargs = capture()
    assert "url" not in kwargs
    assert kwargs == {"protocol": Protocol.REST}


def test_connect_stamps_nothing(capture):
    sdk.connect("http://localhost:5678", port_mode=PortMode.MULTI)
    kwargs = capture()
    assert kwargs == {"url": "http://localhost:5678", "port_mode": PortMode.MULTI}


def test_factory_defaults_override_caller_kwargs(capture):
    """Clone pin: the factory's protocol wins over the caller's.

    This documents (deliberately surprising) pre-consolidation behavior:
    the factories stamped their default LAST, silently discarding a
    caller-supplied conflicting value.
    """
    sdk.connect_rest("http://localhost:5678", protocol=Protocol.GRPC)
    kwargs = capture()
    assert kwargs["protocol"] == Protocol.REST, (
        "the consolidation must preserve the original stamp-last semantics — "
        "switching to setdefault would be a behavior change, not a clone"
    )


def test_url_kwarg_form_still_lands(capture):
    sdk.connect_rest(url="http://localhost:9999")
    assert capture()["url"] == "http://localhost:9999"
