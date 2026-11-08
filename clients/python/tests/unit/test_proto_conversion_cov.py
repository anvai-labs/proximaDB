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
    assert ProtoConverter.distance_metric_to_int("minkowski") == 10
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


class TestProtoDriftGuard:
    """The SDK's hand-written enum tables must match the .proto.

    Born from R4 wave-2 PR B: three rest_sync reverse maps, the
    proto_conversion table, and the models IntEnum had ALL silently rotated
    ANGULAR/CHEBYSHEV/CANBERRA/MINKOWSKI by one — a create_collection with
    metric="angular" encoded 7 and the response decoded 7 as "chebyshev".
    The generated types_pb2 module is the single source of truth; this test
    makes any future hand-edit that diverges from it fail CI.
    """

    def test_distance_metric_table_matches_proto(self) -> None:
        from proximadb_sdk.v1 import vector_types_pb2 as types_pb2

        proto = {
            proto.name.lower(): proto.number
            for proto in types_pb2.DistanceMetric.DESCRIPTOR.values
        }
        ours = ProtoConverter._DISTANCE_METRIC_STR_TO_INT
        for name, number in proto.items():
            assert ours.get(name) == number, (
                f"SDK table drifted from proto: {name} should be {number}, "
                f"table says {ours.get(name)}"
            )

    def test_inline_reverse_maps_match_proto(self) -> None:
        """Every hand-written int->str DistanceMetric dict in the SDK must
        decode proto values correctly (review round 1 found a sixth drifted
        site in client_v1 and a 1-5 truncation in rest_sync get_collection
        that this guard's absence let live). Parses the source files for
        int-literal -> "name" dict entries and compares each to the proto.
        """
        import re
        from pathlib import Path

        sdk_src = Path(__file__).parents[2] / "src" / "proximadb_sdk"
        from proximadb_sdk.v1 import vector_types_pb2 as types_pb2

        proto = {
            v.number: v.name.lower() for v in types_pb2.DistanceMetric.DESCRIPTOR.values
        }
        proto[0] = "cosine"  # client_v1 maps 0->cosine as a safe default

        files = [
            sdk_src / "protocols" / "rest_sync.py",
            sdk_src / "client_v1.py",
        ]
        checked = 0
        for f in files:
            text = f.read_text()
            for m in re.finditer(r"(\d+):\s*\"(\w+)\"", text):
                num, name = int(m.group(1)), m.group(2)
                if (
                    num in proto
                    and name
                    in (
                        "cosine euclidean dot_product hamming manhattan jaccard "
                        "angular chebyshev canberra minkowski bray_curtis "
                        "hellinger custom"
                    ).split()
                ):
                    assert proto[num] == name, (
                        f"{f.name}:{text[:m.start()].count(chr(10)) + 1} "
                        f"drifted: {num} should be {proto[num]}, says {name}"
                    )
                    checked += 1
        # the maps must actually be found — a rename that hides them from
        # the regex is itself drift-shaped
        assert checked >= 40, f"only {checked} entries found — maps moved?"

    def test_storage_engine_reverse_covers_proto(self) -> None:
        """Decode-side coverage: every proto StorageEngine number must decode
        truthfully (the forward aliases are the deliberate TD-SDKENG-1
        design; the reverse truncation silently decoded 7-12)."""
        from proximadb_sdk.v1 import vector_types_pb2 as types_pb2

        for v in types_pb2.StorageEngine.DESCRIPTOR.values:
            name = v.name.lower().removeprefix("storage_engine_")
            decoded = ProtoConverter._STORAGE_ENGINE_INT_TO_STR.get(v.number)
            # Every proto number PRESENT in the decode table must decode
            # to its proto name — no exceptions (#2446 review MEDIUM-5:
            # the former ("viper",) exemption was a mutation-proven hole
            # with zero beneficiaries).
            if decoded is not None:
                assert decoded == name, (
                    f"StorageEngine decode drift: {v.number} decoded "
                    f"{decoded!r}, proto member {name!r}"
                )
        for n in range(0, 13):
            assert (
                n in ProtoConverter._STORAGE_ENGINE_INT_TO_STR
            ), f"StorageEngine reverse truncated: {n} missing"

    def test_filterable_datatype_enum_matches_proto(self) -> None:
        """models.FilterableDataType must carry every proto value (the
        8-value truncation made wire-map extensions unreachable and
        crashed decode — #2446 review BLOCKER-1)."""
        from proximadb_sdk.models import FilterableDataType
        from proximadb_sdk.v1 import vector_types_pb2 as types_pb2

        proto_names = set()
        for v in types_pb2.FilterableDataType.DESCRIPTOR.values:
            name = v.name.lower().removeprefix("filterable_")
            if name == "data_type_unspecified":
                # proto's fully-qualified UNSPECIFIED; the enum spells
                # nothing for it (absent = deliberate, same as the wire
                # maps' 0->"string" default)
                proto_names.add(name)
                continue
            proto_names.add(name)
            member = getattr(FilterableDataType, name.upper(), None)
            assert member is not None, f"enum missing proto member {name}"
            assert member.value == name, f"enum value drift: {name}"
        proto_names.discard("data_type_unspecified")
        ours = {m.value for m in FilterableDataType}
        assert ours == proto_names, f"phantom/missing: {ours ^ proto_names}"

    def test_inline_engine_and_filterable_maps_match_proto(self) -> None:
        """rest_sync + client_v1 inline StorageEngine/FilterableDataType
        dicts must decode proto numbers truthfully (the converter-table
        guard above cannot see them — #2446 review MAJOR-4)."""
        import re
        from pathlib import Path

        sdk_src = Path(__file__).parents[2] / "src" / "proximadb_sdk"
        from proximadb_sdk.v1 import vector_types_pb2 as types_pb2

        engine_proto = {
            v.number: v.name.lower().removeprefix("storage_engine_")
            for v in types_pb2.StorageEngine.DESCRIPTOR.values
        }
        filt_proto = {
            v.number: v.name.lower().removeprefix("filterable_")
            for v in types_pb2.FilterableDataType.DESCRIPTOR.values
        }
        filt_names = set(filt_proto.values())

        files = [
            sdk_src / "protocols" / "rest_sync.py",
            sdk_src / "client_v1.py",
        ]
        engine_checked = filt_checked = 0
        for f in files:
            text = f.read_text()
            for m in re.finditer(r"(\d+):\s*\"(\w+)\"", text):
                num, name = int(m.group(1)), m.group(2)
                if num in engine_proto and name in set(engine_proto.values()):
                    if not (num == 0 and name == "viper"):
                        assert engine_proto[num] == name, (
                            f"{f.name} engine drift: {num} should be "
                            f"{engine_proto[num]}, says {name}"
                        )
                    engine_checked += 1
                if num in filt_proto and name in filt_names:
                    if not (num == 0 and name == "string"):
                        assert filt_proto[num] == name, (
                            f"{f.name} filterable drift: {num} should be "
                            f"{filt_proto[num]}, says {name}"
                        )
                    filt_checked += 1
        assert engine_checked >= 10, f"only {engine_checked} engine entries found"
        assert filt_checked >= 20, f"only {filt_checked} filterable entries found"

        # COMPLETENESS: the value checks above only validate entries that
        # exist — a deleted entry would silently shrink the count (#2446
        # review NC: re-truncation initially passed). Every proto number
        # must be present in each file that carries the map family.
        for f_ in files:
            text2 = f_.read_text()
            for num, name in filt_proto.items():
                if num == 0 or f_.name == "client_v1.py":
                    continue  # 0 is the deliberate default; client_v1 has no filterable map
                assert (
                    f'{num}: "{name}"' in text2
                ), f"{f_.name} filterable map missing proto {num} ({name})"
            if f_.name == "rest_sync.py":
                for num, name in engine_proto.items():
                    if num == 0:
                        continue
                    assert (
                        f'{num}: "{name}"' in text2
                    ), f"{f_.name} engine map missing proto {num} ({name})"
            else:
                # client_v1: 0/1 both -> "viper" deliberately
                for num, name in engine_proto.items():
                    if num in (0, 1):
                        continue
                    assert (
                        f'{num}: "{name}"' in text2
                    ), f"{f_.name} engine map missing proto {num} ({name})"

    def test_distance_metric_intenum_matches_proto(self) -> None:
        from proximadb_sdk.models import DistanceMetricType
        from proximadb_sdk.v1 import vector_types_pb2 as types_pb2

        for proto in types_pb2.DistanceMetric.DESCRIPTOR.values:
            ours = getattr(DistanceMetricType, proto.name, None)
            if proto.name == "DISTANCE_METRIC_UNSPECIFIED":
                # the IntEnum spells it UNSPECIFIED (documented alias)
                ours = DistanceMetricType.UNSPECIFIED
            assert ours is not None, f"IntEnum missing proto member {proto.name}"
            assert ours.value == proto.number, (
                f"IntEnum drift: {proto.name} should be {proto.number}, "
                f"enum says {ours.value}"
            )
