"""Offline coverage tests for proximadb_sdk.proto_conversion.

Pure module — no transport, no heavy deps. Exercises every conversion
method in both directions across all supported input types and the
model-helper functions.
"""

from enum import Enum

import pytest

from proximadb_sdk.proto_conversion import (
    ProtoConverter,
    distance_metric_to_int,
    distance_metric_to_str,
    index_type_to_int,
    index_type_to_str,
    quantization_type_to_int,
    quantization_type_to_str,
    storage_engine_to_int,
    storage_engine_to_str,
)

# --- helper enums to exercise the Enum branches -----------------------------


class IntMetric(Enum):
    COSINE = 1
    EUCLIDEAN = 2


class StrMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class StrEngine(Enum):
    VIPER = "viper"


class IntEngine(Enum):
    NOVA = 3


class StrIndex(Enum):
    HNSW = "hnsw"


class IntIndex(Enum):
    IVF = 2


class StrQuant(Enum):
    SCALAR = "scalar"


class IntQuant(Enum):
    PQ = 2


# ============================================================================
# Distance metric
# ============================================================================


def test_distance_metric_to_int_all_inputs():
    assert ProtoConverter.distance_metric_to_int(None) == 0
    assert ProtoConverter.distance_metric_to_int(7) == 7  # int passthrough
    assert ProtoConverter.distance_metric_to_int("cosine") == 1
    assert ProtoConverter.distance_metric_to_int("COSINE") == 1  # case-insensitive
    assert ProtoConverter.distance_metric_to_int("euclidean") == 2
    assert ProtoConverter.distance_metric_to_int("minkowski") == 9
    assert ProtoConverter.distance_metric_to_int("custom") == 13
    # invalid string -> 0
    assert ProtoConverter.distance_metric_to_int("nope") == 0
    # enum with int value
    assert ProtoConverter.distance_metric_to_int(IntMetric.EUCLIDEAN) == 2
    # enum with str value
    assert ProtoConverter.distance_metric_to_int(StrMetric.COSINE) == 1
    # unsupported type -> 0
    assert ProtoConverter.distance_metric_to_int(3.14) == 0


def test_distance_metric_to_str_all_inputs():
    assert ProtoConverter.distance_metric_to_str(None) == "cosine"
    assert ProtoConverter.distance_metric_to_str("euclidean") == "euclidean"
    assert ProtoConverter.distance_metric_to_str("EUCLIDEAN") == "euclidean"
    assert (
        ProtoConverter.distance_metric_to_str("bogus") == "cosine"
    )  # unknown str default
    assert ProtoConverter.distance_metric_to_str(1) == "cosine"
    assert ProtoConverter.distance_metric_to_str(2) == "euclidean"
    assert ProtoConverter.distance_metric_to_str(999) == "cosine"  # unknown int default
    assert ProtoConverter.distance_metric_to_str(IntMetric.EUCLIDEAN) == "euclidean"
    assert ProtoConverter.distance_metric_to_str(StrMetric.EUCLIDEAN) == "euclidean"
    assert ProtoConverter.distance_metric_to_str(3.14) == "cosine"  # unsupported type


def test_distance_metric_round_trip():
    for name, i in ProtoConverter._DISTANCE_METRIC_STR_TO_INT.items():
        assert ProtoConverter.distance_metric_to_int(name) == i


# ============================================================================
# Storage engine
# ============================================================================


def test_storage_engine_to_int_all_inputs():
    # TD-SDKENG-1: SST (2) is the default everywhere; VIPER is deprecated
    # (ADR-093) and the server rejects it.
    assert ProtoConverter.storage_engine_to_int(None) == 2  # default SST
    assert ProtoConverter.storage_engine_to_int(5) == 5
    assert ProtoConverter.storage_engine_to_int("nova") == 3
    assert ProtoConverter.storage_engine_to_int("RAPTOR") == 6
    assert ProtoConverter.storage_engine_to_int("mmap") == 1  # legacy alias
    assert ProtoConverter.storage_engine_to_int("hybrid") == 1  # legacy alias
    assert ProtoConverter.storage_engine_to_int("unknown") == 2  # default
    assert ProtoConverter.storage_engine_to_int(IntEngine.NOVA) == 3
    assert ProtoConverter.storage_engine_to_int(StrEngine.VIPER) == 1
    assert ProtoConverter.storage_engine_to_int(3.14) == 2  # unsupported type


def test_storage_engine_to_str_all_inputs():
    # TD-SDKENG-1: unknown/None fall back to SST; the known legacy aliases
    # (mmap/hybrid) still resolve through the int table to viper.
    assert ProtoConverter.storage_engine_to_str(None) == "sst"
    assert ProtoConverter.storage_engine_to_str("nova") == "nova"
    assert ProtoConverter.storage_engine_to_str("NOVA") == "nova"
    assert ProtoConverter.storage_engine_to_str("mmap") == "viper"
    assert ProtoConverter.storage_engine_to_str("hybrid") == "viper"
    assert ProtoConverter.storage_engine_to_str("unknown") == "sst"  # default
    assert ProtoConverter.storage_engine_to_str(2) == "sst"
    assert ProtoConverter.storage_engine_to_str(999) == "sst"  # unknown int default
    assert ProtoConverter.storage_engine_to_str(IntEngine.NOVA) == "nova"
    assert ProtoConverter.storage_engine_to_str(StrEngine.VIPER) == "viper"
    assert ProtoConverter.storage_engine_to_str(3.14) == "sst"  # unsupported type


# ============================================================================
# Index type
# ============================================================================


def test_index_type_to_int_all_inputs():
    assert ProtoConverter.index_type_to_int(None) == 1  # default HNSW
    assert ProtoConverter.index_type_to_int(4) == 4
    assert ProtoConverter.index_type_to_int("ivf") == 2
    assert ProtoConverter.index_type_to_int("LSH") == 6
    assert ProtoConverter.index_type_to_int("unknown") == 1  # default
    assert ProtoConverter.index_type_to_int(IntIndex.IVF) == 2
    assert ProtoConverter.index_type_to_int(StrIndex.HNSW) == 1
    assert ProtoConverter.index_type_to_int(3.14) == 1  # unsupported type


def test_index_type_to_str_all_inputs():
    assert ProtoConverter.index_type_to_str(None) == "hnsw"
    assert ProtoConverter.index_type_to_str("ivf") == "ivf"
    assert ProtoConverter.index_type_to_str("IVF") == "ivf"
    assert ProtoConverter.index_type_to_str("unknown") == "hnsw"  # default
    assert ProtoConverter.index_type_to_str(3) == "pq"
    assert ProtoConverter.index_type_to_str(999) == "hnsw"  # unknown int default
    assert ProtoConverter.index_type_to_str(IntIndex.IVF) == "ivf"
    assert ProtoConverter.index_type_to_str(StrIndex.HNSW) == "hnsw"
    assert ProtoConverter.index_type_to_str(3.14) == "hnsw"  # unsupported type


# ============================================================================
# Quantization type
# ============================================================================


def test_quantization_type_to_int_all_inputs():
    assert ProtoConverter.quantization_type_to_int(None) == 0
    assert ProtoConverter.quantization_type_to_int(4) == 4
    assert ProtoConverter.quantization_type_to_int("scalar") == 3
    assert ProtoConverter.quantization_type_to_int("BINARY") == 4
    assert ProtoConverter.quantization_type_to_int("unknown") == 0  # default
    assert ProtoConverter.quantization_type_to_int(IntQuant.PQ) == 2
    assert ProtoConverter.quantization_type_to_int(StrQuant.SCALAR) == 3
    assert ProtoConverter.quantization_type_to_int(3.14) == 0  # unsupported type


def test_quantization_type_to_str_all_inputs():
    assert ProtoConverter.quantization_type_to_str(None) == "none"
    assert ProtoConverter.quantization_type_to_str("scalar") == "scalar"
    assert ProtoConverter.quantization_type_to_str("SCALAR") == "scalar"
    assert ProtoConverter.quantization_type_to_str("unknown") == "none"  # default
    assert ProtoConverter.quantization_type_to_str(2) == "pq"
    assert ProtoConverter.quantization_type_to_str(999) == "none"  # unknown int default
    assert ProtoConverter.quantization_type_to_str(IntQuant.PQ) == "pq"
    assert ProtoConverter.quantization_type_to_str(StrQuant.SCALAR) == "scalar"
    assert ProtoConverter.quantization_type_to_str(3.14) == "none"  # unsupported type


# ============================================================================
# Model conversion helpers
# ============================================================================


def test_vector_record_to_dict_passthrough_dict():
    d = {"id": "a", "vector": [1.0, 2.0]}
    assert ProtoConverter.vector_record_to_dict(d) is d


def test_vector_record_to_dict_model_dump():
    class WithModelDump:
        def model_dump(self, exclude_none=False):
            return {"id": "md", "exclude_none": exclude_none}

    out = ProtoConverter.vector_record_to_dict(WithModelDump())
    assert out == {"id": "md", "exclude_none": True}


def test_vector_record_to_dict_legacy_dict_method():
    class WithDict:
        def dict(self, exclude_none=False):
            return {"id": "legacy", "exclude_none": exclude_none}

    out = ProtoConverter.vector_record_to_dict(WithDict())
    assert out == {"id": "legacy", "exclude_none": True}


def test_vector_record_to_dict_attr_fallback():
    class Plain:
        id = "p1"
        vector = (1.0, 2.0, 3.0)
        metadata = {"k": "v"}

    out = ProtoConverter.vector_record_to_dict(Plain())
    assert out == {"id": "p1", "vector": [1.0, 2.0, 3.0], "metadata": {"k": "v"}}


def test_vector_record_to_dict_attr_fallback_defaults():
    class Empty:
        pass

    out = ProtoConverter.vector_record_to_dict(Empty())
    assert out == {"id": "", "vector": [], "metadata": None}


def test_dict_to_search_result_primary_keys():
    out = ProtoConverter.dict_to_search_result(
        {"id": "x", "score": 0.9, "vector": [1.0], "metadata": {"a": 1}}
    )
    assert out == {"id": "x", "score": 0.9, "vector": [1.0], "metadata": {"a": 1}}


def test_dict_to_search_result_alias_keys():
    out = ProtoConverter.dict_to_search_result({"vector_id": "y", "distance": 0.5})
    assert out["id"] == "y"
    assert out["score"] == 0.5
    assert out["vector"] == []
    assert out["metadata"] == {}


def test_dict_to_search_result_empty():
    out = ProtoConverter.dict_to_search_result({})
    assert out == {"id": "", "score": 0.0, "vector": [], "metadata": {}}


def test_collection_config_to_dict_defaults():
    cfg = ProtoConverter.collection_config_to_dict("col", 128)
    assert cfg == {
        "name": "col",
        "dimension": 128,
        "distance_metric": "cosine",
        "storage_engine": "sst",
        "index_type": "hnsw",
    }


def test_collection_config_to_dict_with_values_and_kwargs():
    cfg = ProtoConverter.collection_config_to_dict(
        "col2",
        256,
        distance_metric="euclidean",
        storage_engine="nova",
        index_type="ivf",
        replication=2,
        custom_flag=True,
    )
    assert cfg["distance_metric"] == "euclidean"
    assert cfg["storage_engine"] == "nova"
    assert cfg["index_type"] == "ivf"
    assert cfg["replication"] == 2
    assert cfg["custom_flag"] is True


# ============================================================================
# Module-level convenience functions
# ============================================================================


def test_convenience_functions_delegate():
    assert distance_metric_to_int("cosine") == 1
    assert distance_metric_to_str(2) == "euclidean"
    assert storage_engine_to_int("nova") == 3
    assert storage_engine_to_str(2) == "sst"
    assert index_type_to_int("ivf") == 2
    assert index_type_to_str(3) == "pq"
    assert quantization_type_to_int("scalar") == 3
    assert quantization_type_to_str(2) == "pq"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
