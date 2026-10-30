"""TD-RESTGET-1: a misspelt create setting must fail, never be dropped.

The server now rejects unknown create fields (400). The spec marks those inputs
``additionalProperties: false``, and the generated ``from_dict`` for such a
model silently DROPS keys it does not know — so without a check the SDK would
discard a typo client-side and the server's 400 would never fire. The facade's
``_from_dict`` raises instead, naming the dropped key's path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import proximadb_sdk

if not proximadb_sdk.__file__.startswith(str(Path(__file__).resolve().parents[2])):
    pytest.skip(
        f"tests the worktree SDK, but {proximadb_sdk.__file__} won the import; "
        "run with PYTHONPATH=<worktree>/clients/python/src",
        allow_module_level=True,
    )

from proximadb_sdk.exceptions import ValidationError  # noqa: E402
from proximadb_sdk.protocols import _rest_codegen  # noqa: E402


@pytest.mark.parametrize(
    "body,path",
    [
        ({"name": "x", "dimension": 4, "replicas": 3}, "replicas"),
        (
            {
                "name": "x",
                "dimension": 4,
                "index_configs": [{"algorithm": "hnsw", "hnsw": {"m": 32}}],
            },
            "index_configs[0].hnsw",
        ),
        (
            {
                "name": "x",
                "dimension": 4,
                "quantization": {"enabled": True, "type": "scalar"},
            },
            "quantization.type",
        ),
        (
            {
                "name": "x",
                "dimension": 4,
                "schema": {
                    "columns": [{"name": "c", "data_type": "text", "filterabel": True}]
                },
            },
            "schema.columns[0].filterabel",
        ),
    ],
    ids=["top-level", "nested-index", "nested-quantization", "schema-column"],
)
def test_unknown_create_field_raises_naming_its_path(body, path):
    with pytest.raises(
        ValidationError, match=path.replace("[", r"\[").replace("]", r"\]")
    ):
        _rest_codegen.create_collection(body)


def test_canonical_create_body_passes_unchanged():
    # Exactly the shape rest_sync / the async client build (TD-SDKENG-1).
    body = {
        "name": "x",
        "dimension": 4,
        "engine": "sst",
        "distance_metric": "cosine",
        "enable_proxima_record": True,
        "index_configs": [
            {
                "index_name": "i",
                "algorithm": "hnsw",
                "hnsw_config": {"m": 16, "ef_construction": 64, "ef_search": 32},
                "is_primary": True,
            }
        ],
        "quantization": {"enabled": True, "strategy": "custom_levels"},
        "schema": {"columns": [{"name": "c", "data_type": "text", "filterable": True}]},
        "description": "d",
    }
    call = _rest_codegen.create_collection(body)
    assert call.json["index_configs"][0]["hnsw_config"]["m"] == 16
    assert call.json["quantization"] == {"enabled": True, "strategy": "custom_levels"}
    assert call.json["description"] == "d"


def test_typo_is_rejected_before_any_request_is_sent():
    from proximadb_sdk.protocols.rest_sync import ProximaDBClient as RestClient

    client = RestClient.__new__(RestClient)

    def no_request(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a request was sent for an invalid body")

    client._make_request = no_request
    with pytest.raises(ValidationError, match="filterabel"):
        client.create_collection(
            "x",
            dimension=4,
            schema={
                "columns": [{"name": "c", "data_type": "text", "filterabel": True}]
            },
        )
