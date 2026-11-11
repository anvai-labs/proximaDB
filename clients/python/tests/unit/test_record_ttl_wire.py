"""Record TTL reaches the REST wire.

REST v2 `ProximaRecordInput` had no expiry field, so a TTL set through the
SDK was dropped on REST while gRPC carried it. The normalizer now forwards
the explicit `expires_at_ms`; a bare `expires_at` (mixed units across the
codebase) is deliberately not forwarded.
"""

from proximadb_sdk.models_v2 import ProximaRecord
from proximadb_sdk.protocols.rest_sync import ProximaDBClient


def _client():
    return ProximaDBClient(url="http://testserver")


def test_normalizer_forwards_expires_at_ms_from_dict():
    out = _client()._normalize_record_payload(
        {"id": "r1", "vector": [1.0], "expires_at_ms": 1_700_000_000_000}, 0
    )
    assert out["expires_at_ms"] == 1_700_000_000_000


def test_normalizer_forwards_expires_at_ms_from_typed_record():
    rec = ProximaRecord(id="r1", vector=[1.0], expires_at_ms=1_700_000_000_000)
    out = _client()._normalize_record_payload(rec, 0)
    assert out["expires_at_ms"] == 1_700_000_000_000


def test_normalizer_omits_expiry_when_unset_and_ignores_bare_expires_at():
    client = _client()
    assert "expires_at_ms" not in client._normalize_record_payload(
        {"id": "r1", "vector": [1.0]}, 0
    )
    out = client._normalize_record_payload(
        {"id": "r1", "vector": [1.0], "expires_at": 1_700_000_000}, 0
    )
    assert "expires_at_ms" not in out and "expires_at" not in out


def test_generated_rest_model_carries_expires_at_ms():
    from proximadb_sdk._generated.rest.models.proxima_record_input import (
        ProximaRecordInput,
    )

    body = ProximaRecordInput.from_dict(
        {"vector": [1.0], "expires_at_ms": 1_700_000_000_000}
    ).to_dict()
    assert body["expires_at_ms"] == 1_700_000_000_000


def test_async_normalizer_forwards_expires_at_ms():
    """Async twin of the sync normalizer (review M1: it dropped the TTL)."""
    from proximadb_sdk.unified_client_async import ProximaDBAsyncUnified

    client = ProximaDBAsyncUnified.__new__(ProximaDBAsyncUnified)
    out = client._normalize_record(
        {"id": "r1", "vector": [1.0], "expires_at_ms": 1_700_000_000_000}, 0
    )
    assert out["expires_at_ms"] == 1_700_000_000_000
    rec = ProximaRecord(id="r1", vector=[1.0], expires_at_ms=1_700_000_000_000)
    assert client._normalize_record(rec, 0)["expires_at_ms"] == 1_700_000_000_000
    assert "expires_at_ms" not in client._normalize_record(
        {"id": "r1", "vector": [1.0]}, 0
    )
