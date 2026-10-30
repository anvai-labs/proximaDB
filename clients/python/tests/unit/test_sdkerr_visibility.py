"""TD-SDKERR-1: failures must be visible — no sticky local fallback, no
swallowed insert errors, no kwarg collisions, plain property values, a
working AuthResult contract.

Each test drives the real client/adapter code with a fake transport, the
same seam a live server sits behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# TD-SDKERR-1 review: evicting a cached SDK from sys.modules split-brains the
# process when other test files run alongside. Instead, PROVE the provenance:
# if an editable install pointing elsewhere won the import, say so and skip —
# never silently test the wrong copy.
import proximadb_sdk

if not proximadb_sdk.__file__.startswith(str(Path(__file__).resolve().parents[2])):
    pytest.skip(
        f"tests the worktree SDK, but {proximadb_sdk.__file__} won the import; "
        "run with PYTHONPATH=<worktree>/clients/python/src",
        allow_module_level=True,
    )

from proximadb_sdk.adapters.grpc_adapter import GrpcProtocolAdapter as GrpcAdapter
from proximadb_sdk.auth import AuthResult
from proximadb_sdk.models import BatchResult
from proximadb_sdk.protocols import rest_sync
from proximadb_sdk.unified_client import ProximaDBClient


class RecordingAdapter:
    """Adapter double: raises what it is told, records every call."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return []


def _client_with_adapter(adapter) -> ProximaDBClient:
    client = ProximaDBClient.__new__(ProximaDBClient)
    client._adapter = adapter
    client._prefer_local_fallback = False
    return client


class TestStickyFallbackGone:
    def test_server_rejected_search_raises_and_does_not_flip_fallback(self):
        adapter = RecordingAdapter(error=RuntimeError("INVALID_ARGUMENT: bad filter"))
        client = _client_with_adapter(adapter)

        with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
            client.search_single("c", [0.1, 0.2], top_k=1)

        assert (
            not client._prefer_local_fallback
        ), "a server-rejected search must not activate the local fallback"
        # The next search still reaches the adapter — no sticky emptiness.
        adapter.error = None
        client.search_single("c", [0.1, 0.2], top_k=1)
        assert len(adapter.calls) == 2

    def test_unreachable_server_still_falls_back(self):
        adapter = RecordingAdapter(error=ConnectionError("refused"))
        client = _client_with_adapter(adapter)

        from proximadb_sdk.exceptions import ProximaDBError

        # The local store has no collection 'c' — the local mode's own
        # not-found is the honest answer once the transport is gone.
        with pytest.raises(ProximaDBError, match="not found"):
            client.search_single("c", [0.1, 0.2], top_k=1)
        assert (
            client._prefer_local_fallback
        ), "an unreachable server is the one legitimate fallback case"

    def test_filter_kwarg_is_an_alias_not_a_collision(self):
        adapter = RecordingAdapter()
        client = _client_with_adapter(adapter)

        client.search_single("c", [0.1, 0.2], top_k=1, filter={"lang": "en"})

        assert adapter.calls[0]["filter"] == {
            "lang": "en"
        }, "`filter=` must reach the adapter instead of raising TypeError"


class TestInsertErrorsVisible:
    def _adapter_returning(self, raw_result) -> GrpcAdapter:
        adapter = GrpcAdapter.__new__(GrpcAdapter)
        adapter._client = _FakeClient(raw_result)
        return adapter

    def test_failed_records_carry_their_errors(self):
        raw = {
            "success": False,
            "successful_count": 0,
            "failed_count": 1,
            "errors": [
                {
                    "record_index": 0,
                    "record_id": "r1",
                    "error_code": "RECORD_INSERT_FAILED",
                    "error_message": "vector dimension mismatch",
                }
            ],
        }
        adapter = self._adapter_returning(raw)
        result = adapter.insert_records("c", [{"id": "r1", "vector": [1.0]}])
        assert isinstance(result, BatchResult)
        assert result.failed == 1
        assert result.errors, "the per-record failure must not vanish"
        assert "vector dimension mismatch" in "; ".join(result.errors)

    def test_failed_without_detail_says_so(self):
        raw = {"success": False, "successful_count": 0, "failed_count": 2}
        adapter = self._adapter_returning(raw)
        result = adapter.insert_records("c", [{"id": "r1", "vector": [1.0]}])
        assert result.errors and "failed" in result.errors[0]


class _FakeClient:
    def __init__(self, result):
        self._result = result

    def insert_records(self, collection_id, records, **kwargs):
        return self._result


class TestRestPlainPropertyValues:
    def test_typed_cells_unwrap_to_plain_values(self):
        props = {
            "lang": {"type": "string", "value": "en"},
            "year": {"type": "integer", "value": 2024},
            "plain": 7,
        }
        assert rest_sync.ProximaDBClient._unwrap_props_cells(props) == {
            "lang": "en",
            "year": 2024,
            "plain": 7,
        }


class TestAuthResultContract:
    def test_success_is_derived_and_error_field_exists(self):
        ok = AuthResult(user_id="u1")
        assert ok.success is True

        failed = AuthResult(user_id="", error="rejected")
        assert failed.success is False
        assert failed.error == "rejected"
