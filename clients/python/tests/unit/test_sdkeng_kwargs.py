"""TD-SDKENG-1: create_collection must never silently drop a keyword —
`engine=` asked for HELIX and got the SST default; `index_params=` and
`quantization=` vanished the same way.
"""

from pathlib import Path

import pytest

import proximadb_sdk

if not proximadb_sdk.__file__.startswith(str(Path(__file__).resolve().parents[2])):
    pytest.skip(
        f"tests the worktree SDK, but {proximadb_sdk.__file__} won the import; "
        "run with PYTHONPATH=<worktree>/clients/python/src",
        allow_module_level=True,
    )

from proximadb_sdk.models import (
    CollectionConfig,
    StorageEngine,
    normalize_collection_config_kwargs,
)
from proximadb_sdk.unified_client import ProximaDBClient


def _client_with_recording_adapter():
    class RecordingAdapter:
        def __init__(self):
            self.config: CollectionConfig | None = None

        def create_collection(self, name, config, **kwargs):
            self.config = config
            from proximadb_sdk.models import Collection

            return Collection(
                id=name,
                config=config,
            )

    adapter = RecordingAdapter()
    client = ProximaDBClient.__new__(ProximaDBClient)
    client._adapter = adapter
    client._prefer_local_fallback = False
    # The local store is CLASS-level state — clear it so a prior test's 'x'
    # does not read as "already exists".
    ProximaDBClient._shared_local_collections.clear()
    ProximaDBClient._shared_local_vectors.clear()
    return client, adapter


class TestEngineAlias:
    def test_engine_kwarg_creates_the_asked_engine(self):
        client, adapter = _client_with_recording_adapter()
        client.create_collection("x", dimension=3, engine="helix")
        assert adapter.config.storage_engine == StorageEngine.HELIX

    def test_storage_engine_still_works_verbatim(self):
        client, adapter = _client_with_recording_adapter()
        client.create_collection("x", dimension=3, storage_engine="helix")
        assert adapter.config.storage_engine == StorageEngine.HELIX


class TestPassThrough:
    def test_quantization_alias_populates_quantization_config(self):
        client, adapter = _client_with_recording_adapter()
        client.create_collection(
            "x", dimension=3, quantization={"enabled": True, "type": "scalar"}
        )
        assert adapter.config.quantization_config is not None

    def test_index_params_alias_populates_index_configs(self):
        client, adapter = _client_with_recording_adapter()
        client.create_collection(
            "x",
            dimension=3,
            index_params=[
                {"index_name": "x_hnsw", "algorithm": "hnsw", "hnsw_config": {"m": 16}}
            ],
        )
        assert adapter.config.index_configs, "index settings must not vanish"


class TestUnknownRejected:
    def test_unknown_kwarg_raises_a_clear_type_error(self):
        client, _ = _client_with_recording_adapter()
        with pytest.raises(TypeError, match="enginez"):
            client.create_collection("x", dimension=3, enginez="helix")

    def test_helper_names_the_aliases(self):
        with pytest.raises(TypeError, match="engine=storage_engine"):
            normalize_collection_config_kwargs({"nope": 1})

    def test_conflicting_alias_and_canonical_rejected(self):
        with pytest.raises(TypeError, match="conflicting"):
            normalize_collection_config_kwargs(
                {"engine": "helix", "storage_engine": "sst"}
            )

    def test_quantization_pydantic_alias_accepted(self):
        # Round-1 review: `quantization` is pydantic's own field alias for
        # quantization_config — the normalizer must accept it (it always
        # worked pre-fix via pydantic).
        config_kwargs, _ = normalize_collection_config_kwargs(
            {"quantization": {"enabled": True}}
        )
        assert config_kwargs["quantization"] == {"enabled": True}


class TestPassthroughAndOverrides:
    def test_transport_kwargs_pass_through_and_reach_the_adapter(self):
        # Round-1 review: schema=/initial_capacity= are consumed by the REST
        # transport — rejecting them broke the SDK's only route to
        # CreateCollectionRequest.schema.
        client, adapter = _client_with_recording_adapter()
        captured = {}

        def create_collection(name, config, **kwargs):
            captured["kwargs"] = kwargs
            from proximadb_sdk.models import Collection

            return Collection(id=name, config=config)

        adapter.create_collection = create_collection
        client.create_collection(
            "x", dimension=3, schema={"columns": []}, initial_capacity=10
        )
        assert captured["kwargs"]["schema"] == {"columns": []}
        assert captured["kwargs"]["initial_capacity"] == 10

    def test_config_given_engine_kwarg_overrides(self):
        # Round-1 review: the config-given branch used to silently ignore
        # engine= entirely.
        from proximadb_sdk.models import CollectionConfig, StorageEngine

        client, adapter = _client_with_recording_adapter()
        base = CollectionConfig(name="x", dimension=3)
        assert base.storage_engine != StorageEngine.HELIX
        client.create_collection("x", config=base, engine="helix")
        assert adapter.config.storage_engine == StorageEngine.HELIX
