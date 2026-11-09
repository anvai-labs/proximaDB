"""Regression test for the release-smoke-caught vector_ids drop.

The release smoke's PyPI leg inserts three records and asserts the legacy
alias returns their server-assigned IDs — which the rest_sync chain dropped
twice (in ``_record_write_result`` and again in ``insert_records``'s
aggregation), leaving ``VectorOperationResponse.vector_ids`` permanently
empty on the REST path. See the v0.6.1 release run's PyPI smoke legs.
"""

from proximadb_sdk.adapters.base import RecordOpsMixin
from proximadb_sdk.protocols.rest_sync import ProximaDBClient

_V2_RESPONSE = {
    "inserted_count": 3,
    "failed_count": 0,
    "inserted_ids": ["a", "b", "c"],
    "errors": [],
    "duration_ms": 1.5,
}


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _client_with_response(monkeypatch, body):
    c = ProximaDBClient(url="http://testserver")
    monkeypatch.setattr(
        c, "_make_request", lambda method, endpoint, **kw: _FakeResponse(body)
    )
    return c


def test_insert_records_carries_inserted_ids(monkeypatch):
    c = _client_with_response(monkeypatch, _V2_RESPONSE)
    result = c.insert_records(
        "col",
        [
            {"id": "a", "vector": [1.0, 0.0, 0.0, 0.0]},
            {"id": "b", "vector": [0.0, 1.0, 0.0, 0.0]},
            {"id": "c", "vector": [0.0, 0.0, 1.0, 0.0]},
        ],
    )
    assert result.inserted_ids == ["a", "b", "c"]
    assert result.success == 3
    assert result.failed == 0


def test_insert_accumulates_ids_across_batches(monkeypatch):
    # one batch → the fixture's 3 ids; two batches → 6. The count tracks
    # the mocked responses, proving per-batch accumulation runs.
    c = _client_with_response(monkeypatch, _V2_RESPONSE)
    result = c.insert_records(
        "col",
        [{"id": f"r{i}", "vector": [float(i), 1.0, 0.0, 0.0]} for i in range(6)],
        batch_size=3,
    )
    assert len(result.inserted_ids) == 6


def test_mixin_maps_inserted_ids_to_vector_ids(monkeypatch):
    c = _client_with_response(monkeypatch, _V2_RESPONSE)
    result = c.insert_records(
        "col",
        [
            {"id": "a", "vector": [1.0, 0.0, 0.0, 0.0]},
            {"id": "b", "vector": [0.0, 1.0, 0.0, 0.0]},
            {"id": "c", "vector": [0.0, 0.0, 1.0, 0.0]},
        ],
    )
    resp = RecordOpsMixin._batch_to_vector_response(result, "INSERT")
    assert resp.vector_ids == ["a", "b", "c"]
