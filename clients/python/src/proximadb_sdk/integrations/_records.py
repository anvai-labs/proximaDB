"""Record-native helpers shared by optional framework integrations."""

from __future__ import annotations

from typing import Any


def record_payload(
    *,
    record_id: str,
    vector: Any,
    text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the SDK's native ProximaRecord-shaped payload."""
    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    payload: dict[str, Any] = {
        "id": record_id,
        "vector": values,
        "props": dict(metadata or {}),
    }
    if text is not None:
        # TD-REC-1: text rides in text_fields (where it persists) — a
        # top-level `source` was always silently dropped server-side and is
        # now a contract 400.
        payload["text_fields"] = [{"name": "text", "content": text}]
    return payload


def insert_records(
    client: Any, collection_name: str, records: list[dict[str, Any]]
) -> Any:
    """Insert records through the native SDK method with compatibility fallback."""
    if hasattr(client, "insert_records"):
        return client.insert_records(collection_name, records)
    return client.insert_vectors(collection_name, records=records)
