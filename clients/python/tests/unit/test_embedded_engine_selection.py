"""`EmbeddedConfig.vector_engine` must actually reach the server.

The field existed, defaulted to "SST", and was documented as "Best for real-time
code indexing" — but `create_collection` never sent it, so the server applied
`engine = "auto"` and every embedded collection was created on HELIX regardless
of configuration.

That is not a cosmetic default. Engine choice gates capability server-side:
PAX-based features such as the filter-aware cascade exist only on the SST path —
so a caller asking for SST and silently getting HELIX loses functionality
without any signal.
"""

import pytest

from proximadb_sdk.embedded import EmbeddedConfig, EmbeddedProximaDB


class _CapturingClient:
    """Captures the create-collection POST body instead of issuing it."""

    def __init__(self, sink):
        self._sink = sink

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, timeout=None):
        self._sink.append((url, json))

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"name": json.get("name"), "dimension": json.get("dimension")}

            text = ""

        return _Resp()


def _db(monkeypatch, sink, **cfg):
    db = EmbeddedProximaDB(
        config=EmbeddedConfig(data_dir="/tmp/does-not-matter", **cfg)
    )
    db._started = True
    monkeypatch.setattr(db, "_http_client", lambda: _CapturingClient(sink))
    return db


@pytest.mark.asyncio
async def test_configured_engine_is_sent(monkeypatch):
    sink = []
    db = _db(monkeypatch, sink, vector_engine="SST")
    await db.create_collection("c", dimension=8)
    assert sink, "no create-collection request was issued"
    _, body = sink[0]
    assert body["engine"] == "sst", f"engine not transmitted; body was {body}"


@pytest.mark.asyncio
async def test_explicit_engine_argument_wins(monkeypatch):
    sink = []
    db = _db(monkeypatch, sink, vector_engine="SST")
    await db.create_collection("c", dimension=8, engine="viper")
    _, body = sink[0]
    assert body["engine"] == "viper"


@pytest.mark.asyncio
async def test_engine_defaults_to_config_not_auto(monkeypatch):
    """Regression guard: omitting the field let the server pick, and it picked HELIX."""
    sink = []
    db = _db(monkeypatch, sink)  # EmbeddedConfig default
    await db.create_collection("c", dimension=8)
    _, body = sink[0]
    assert (
        "engine" in body
    ), "engine must always be sent, never left to the server default"
    assert body["engine"] == EmbeddedConfig().vector_engine.lower()
