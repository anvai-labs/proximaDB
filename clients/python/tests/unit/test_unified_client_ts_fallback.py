"""TD-EMB-1 review F1: the v1 timeseries repository fallback re-enters
the client (timeseries.py wraps the SAME client) — a failed dispatch
falling back to it was an infinite cycle swallowed into an empty
success-shaped dict. The guard must break it with a clear error.
"""

from unittest.mock import MagicMock, patch

import pytest

import proximadb_sdk.unified_client as uc
from proximadb_sdk.adapters.embedded_adapter import EmbeddedProtocolAdapter
from proximadb_sdk.config import Protocol
from proximadb_sdk.exceptions import ProximaDBError
from proximadb_sdk.unified_client import ProximaDBClient


class _NoTimeseriesDB:
    """In-process flavor without any native timeseries surface."""

    def __getattr__(self, name):  # any attribute probe -> absent
        raise AttributeError(name)


def test_v1_explicit_aggregation_raises_clearly_not_recurses():
    adapter = EmbeddedProtocolAdapter.__new__(EmbeddedProtocolAdapter)
    adapter._db = _NoTimeseriesDB()
    adapter._connected = True
    adapter._transport_kind = "in-process"

    with (
        patch.object(uc.ProximaDBClient, "_create_adapter", return_value=adapter),
        patch.object(uc.ProximaDBClient, "_get_rest_adapter", return_value=None),
    ):
        c = ProximaDBClient(protocol=Protocol.EMBEDDED, data_dir="/tmp/x")

    with pytest.raises(ProximaDBError, match="re-entered the client"):
        c.query_timeseries(
            "tscoll001",
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            aggregation="avg",
        )


def test_v1_plain_fallback_still_works(monkeypatch):
    """No explicit aggregation: the repository fallback path still runs
    (the guard must not block first-entry fallbacks)."""
    adapter = MagicMock()
    adapter.protocol_name = "embedded"
    adapter.query_timeseries = None  # dispatch fails -> fallback

    with (
        patch.object(uc.ProximaDBClient, "_create_adapter", return_value=adapter),
        patch.object(uc.ProximaDBClient, "_get_rest_adapter", return_value=None),
    ):
        c = ProximaDBClient(protocol=Protocol.EMBEDDED, data_dir="/tmp/x")

    repo = MagicMock()
    resp = MagicMock()
    resp.to_dict.return_value = {"raw_points": [], "total_points": 0}
    repo.query.return_value = resp
    monkeypatch.setattr(c, "_get_timeseries_repository", lambda: repo)

    out = c.query_timeseries(
        "tscoll001", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"
    )
    assert out == {"raw_points": [], "total_points": 0}
    repo.query.assert_called_once()


def test_v1_ingest_reentry_guard(monkeypatch):
    """Round-2 F2: the repository's ingest calls BACK into the client —
    the identical cycle shape. The guard must break it (previously a
    ~330-deep recursion swallowed into {'success': True, ...})."""
    adapter = MagicMock()
    adapter.protocol_name = "embedded"
    adapter.ingest_timeseries = None  # dispatch fails -> fallback

    with (
        patch.object(uc.ProximaDBClient, "_create_adapter", return_value=adapter),
        patch.object(uc.ProximaDBClient, "_get_rest_adapter", return_value=None),
    ):
        c = ProximaDBClient(protocol=Protocol.EMBEDDED, data_dir="/tmp/x")

    repo = MagicMock()
    # The repository's ingest re-calls the client (the cycle):
    repo.ingest.side_effect = lambda cid, pts: c.ingest_timeseries(cid, pts)
    monkeypatch.setattr(c, "_get_timeseries_repository", lambda: repo)

    with pytest.raises(ProximaDBError, match="re-entered the client"):
        c.ingest_timeseries("tscoll001", [{"timestamp": "t1"}])


def test_v1_create_reentry_guard(monkeypatch):
    """Round-2 F2: create_timeseries_collection has the same cycle."""
    adapter = MagicMock()
    adapter.protocol_name = "embedded"
    adapter.create_timeseries_collection = None

    with (
        patch.object(uc.ProximaDBClient, "_create_adapter", return_value=adapter),
        patch.object(uc.ProximaDBClient, "_get_rest_adapter", return_value=None),
    ):
        c = ProximaDBClient(protocol=Protocol.EMBEDDED, data_dir="/tmp/x")

    repo = MagicMock()
    repo.create_collection.side_effect = lambda cfg: c.create_timeseries_collection(
        cfg.name
    )
    monkeypatch.setattr(c, "_get_timeseries_repository", lambda: repo)

    with pytest.raises(ProximaDBError, match="re-entered the client"):
        c.create_timeseries_collection("tscoll001")
