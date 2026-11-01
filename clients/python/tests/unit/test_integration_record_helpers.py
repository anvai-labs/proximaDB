"""Optional integration record helper tests."""

from proximadb_sdk.integrations._records import insert_records, record_payload


class RecordingClient:
    def __init__(self, native=True):
        self.native = native
        self.record_calls = []
        self.vector_calls = []

    def insert_records(self, collection_name, records):
        if not self.native:
            raise AttributeError("native disabled")
        self.record_calls.append((collection_name, records))
        return {"success": len(records)}

    def insert_vectors(self, collection_name, records):
        self.vector_calls.append((collection_name, records))
        return {"success": len(records)}


class VectorOnlyClient:
    def __init__(self):
        self.vector_calls = []

    def insert_vectors(self, collection_name, records):
        self.vector_calls.append((collection_name, records))
        return {"success": len(records)}


def test_record_payload_preserves_text_and_props():
    payload = record_payload(
        record_id="r1",
        vector=[1, 2],
        text="hello",
        metadata={"kind": "note"},
    )

    # TD-REC-1: no top-level `source` (never a contract field — silently
    # dropped server-side before, a 400 now); the text rides in text_fields.
    assert payload == {
        "id": "r1",
        "vector": [1, 2],
        "props": {"kind": "note"},
        "text_fields": [{"name": "text", "content": "hello"}],
    }


def test_insert_records_prefers_native_client_method():
    client = RecordingClient()
    records = [{"id": "r1", "vector": [1.0], "props": {}}]

    result = insert_records(client, "items", records)

    assert result == {"success": 1}
    assert client.record_calls == [("items", records)]
    assert client.vector_calls == []


def test_insert_records_falls_back_to_vector_alias():
    client = VectorOnlyClient()
    records = [{"id": "r1", "vector": [1.0], "props": {}}]

    result = insert_records(client, "items", records)

    assert result == {"success": 1}
    assert client.vector_calls == [("items", records)]
