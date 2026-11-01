"""client_v1 record write migration tests."""

from __future__ import annotations

from typing import Any

from proximadb_sdk.client_v1 import ProximaDBClientV1
from proximadb_sdk.models import VectorRecord


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_client_v1_insert_vectors_uses_v2_record_batch(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any], float]] = []

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> _Response:
        calls.append((url, json, timeout))
        return _Response({"success": True, "success_count": len(json["records"])})

    monkeypatch.setattr("proximadb_sdk.client_v1.requests.post", fake_post)

    client = ProximaDBClientV1(
        url="http://localhost:5678", protocol="rest", timeout=3.0
    )
    result = client.insert_vectors(
        "items",
        [
            VectorRecord(
                id="r1",
                vector=[1.0, 2.0],
                metadata={"kind": "note"},
                source="hello",
            )
        ],
    )

    assert result == {"success": True, "success_count": 1}
    url, payload, timeout = calls[0]
    assert url == "http://localhost:5678/api/v2/collections/items/records/batch"
    assert timeout == 3.0
    # TD-REC-1: `source` rides in text_fields only — a top-level `source` was
    # always silently dropped by the server and is now a contract 400
    # (deny_unknown_fields on ProximaRecordInput).
    assert payload["records"] == [
        {
            "id": "r1",
            "vector": [1.0, 2.0],
            "props": {"kind": "note"},
            "text_fields": [{"name": "text", "content": "hello"}],
        }
    ]
    assert payload["validate_schema"] is True
