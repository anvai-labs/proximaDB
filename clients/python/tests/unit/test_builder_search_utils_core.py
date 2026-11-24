import numpy as np
import pytest

from proximadb_sdk.builders.collection import (
    CollectionBuilder,
    collection,
    high_performance_collection,
    image_collection,
    text_collection,
)
from proximadb_sdk.builders.insert import (
    InsertBuilder,
    batch_insert,
    from_numpy,
    insert,
)
from proximadb_sdk.models import (
    CompressionType,
    DistanceMetric,
    IndexingAlgorithm,
    StorageEngine,
)
from proximadb_sdk.search_utils import (
    _python_value_to_sql_value,
    build_search_hints,
    build_search_optimization_rest,
    build_search_params_grpc,
)


class DumpRecord:
    def model_dump(self, exclude_none=False):
        return {
            "id": "dump",
            "vector": (1.0, 2.0),
            "metadata": {"a": 1},
            "props": {"b": 2},
            "flexible_fields": {"c": 3},
            "typed_fields": {"price": {"value": 9.99}},
            "text_fields": {"body": "text"},
            "timestamp_ms": 10,
            "updated_at_ms": 11,
            "expires_at_ms": 12,
            "version": 2,
            "source": "unit",
            "source_type": "test",
            "schema_id": "schema",
        }


def test_collection_builder_fluent_methods_and_helpers():
    builder = (
        CollectionBuilder("test_collection", 128)
        .euclidean_distance()
        .dot_product()
        .manhattan_distance()
        .hamming_distance()
        .jaccard_similarity()
        .distance_metric(DistanceMetric.COSINE)
        .sst_storage()
        .hybrid_storage()
        .storage_engine(StorageEngine.NOVA)
        .hnsw_index()
        .ivf_index()
        .flat_index()
        .annoy_index()
        .lsh_index()
        .index_type(IndexingAlgorithm.HNSW)
        .description("unit test collection")
        .compression("gzip")
        .no_compression()
        .gzip_compression()
        .zstd_compression()
        .lz4_compression()
        .enable_bloom_filter()
        .disable_bloom_filter()
        .index_param("ef", 100)
        .hnsw_params(m=32, ef_construction=400)
        .ivf_params(n_lists=64, n_probes=8)
        .annoy_params(n_trees=12)
        .lsh_params(n_tables=5, n_bits=7)
    )

    config = builder.build()
    as_dict = builder.to_dict()

    assert config.name == "test_collection"
    assert config.dimension == 128
    assert config.distance_metric == DistanceMetric.COSINE
    assert config.storage_engine == StorageEngine.NOVA
    assert config.description == "unit test collection"
    assert builder._compression == CompressionType.LZ4
    assert builder._enable_bloom_filter is False
    assert builder._index_params["m"] == 32
    assert as_dict["primary_indexing_algorithm"] is None

    assert collection("another_collection", 16).build().dimension == 16
    assert text_collection("text_collection").build().dimension == 768
    assert (
        text_collection("mini_collection", "all-MiniLM-L6-v2").build().dimension == 384
    )
    assert image_collection("image_collection", "resnet").build().dimension == 2048
    assert (
        high_performance_collection("fast_collection", 64).build().storage_engine
        == StorageEngine.SST
    )


def test_insert_builder_full_record_flow_and_validation():
    builder = (
        insert()
        .add_record(DumpRecord())
        .add_record({"id": "dict", "vector": [3.0, 4.0], "metadata": {"kind": "dict"}})
        .add_vector("dict", [9.0, 9.0])
        .filter_duplicates()
        .add_metadata_field("tenant", "acme")
        .transform_metadata(lambda props: {**props, "transformed": True})
        .normalize_vectors()
        .batch_size(10)
        .overwrite_existing()
        .validate_vectors(False)
        .async_mode()
    )

    records, options = builder.build()

    assert builder.count() == 2
    assert builder.is_empty() is False
    assert builder.get_vector_ids() == ["dump", "dict"]
    assert builder.get_dimensions() == [2, 2]
    assert records[0]["props"] == {
        "a": 1,
        "b": 2,
        "c": 3,
        "tenant": "acme",
        "transformed": True,
    }
    assert records[0]["typed_fields"] == {"price": {"value": 9.99}}
    assert records[0]["text_fields"] == {"body": "text"}
    assert records[0]["source"] == "unit"
    assert round(sum(x * x for x in records[1]["vector"]), 6) == 1.0
    assert options == {
        "batch_size": 10,
        "overwrite": True,
        "validate_vectors": False,
        "async_mode": True,
    }
    assert builder.build_options() == options
    assert builder.summary()["duplicate_ids"] == 0
    assert builder.validate_dimensions(2) is builder
    assert builder.build_vectors()[0].id == "dump"
    assert builder.clear().is_empty() is True
    assert builder.summary() == {"count": 0, "dimensions": [], "has_metadata": False}


def test_insert_builder_array_and_error_paths():
    builder = InsertBuilder()
    builder.from_arrays(["a"], np.array([[1.0, 0.0]]), [{"label": "A"}])
    assert builder.build_records() == [
        {"id": "a", "vector": [1.0, 0.0], "props": {"label": "A"}}
    ]

    with pytest.raises(TypeError):
        InsertBuilder().add_record(object())
    with pytest.raises(ValueError, match="same length"):
        InsertBuilder().from_arrays(["a"], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="Metadata list"):
        InsertBuilder().from_arrays(["a"], [[1.0]], [{}, {}])
    with pytest.raises(ValueError, match="positive"):
        InsertBuilder().batch_size(0)
    with pytest.raises(ValueError, match="cannot exceed"):
        InsertBuilder().batch_size(10001)
    with pytest.raises(ValueError, match="expected 3"):
        InsertBuilder().add_vector("a", [1.0]).validate_dimensions(3)

    records, options = batch_insert([{"id": "r", "vector": [1.0]}], batch_size=5)
    assert records == [{"id": "r", "vector": [1.0], "props": {}}]
    assert options["batch_size"] == 5

    np_records, np_options = from_numpy(
        ["n"], np.array([[0.0, 1.0]]), [{"label": "N"}], batch_size=1
    )
    assert np_records[0]["id"] == "n"
    assert np_options["batch_size"] == 1


def test_search_optimization_rest_quantization_and_streaming_hints():
    common = {
        "top_k": 10,
        "filters": {"tenant": "acme"},
        "accuracy_threshold": 0.95,
        "include_expired": False,
        "timeout_ms": 1000,
        "enable_two_stage": True,
        "enable_clustering_hint": True,
        "enable_metadata_filtering_hint": False,
        "custom_hints": {"custom": "hint"},
        "distance_metric": "cosine",
        "requires_ordering": True,
        "candidate_multiplier": 1.5,
        "streaming_buffer_size": 100,
        "streaming_concurrent_search": True,
        "streaming_max_concurrent_tasks": 2,
        "streaming_batch_size": 16,
    }

    for hint, expected in [
        ("none", {"hint_type": "none"}),
        ("binary", {"hint_type": "binary"}),
        ("scalar", {"hint_type": "scalar", "parameters": {"bits": 8}}),
        ("int16", {"hint_type": "scalar", "parameters": {"bits": 16}}),
        (
            "pq4",
            {
                "hint_type": "product",
                "parameters": {"num_subvectors": 8, "bits_per_code": 4},
            },
        ),
        (
            "pqbad",
            {
                "hint_type": "product",
                "parameters": {"num_subvectors": 8, "bits_per_code": 8},
            },
        ),
    ]:
        result = build_search_optimization_rest(quantization_hint=hint, **common)
        assert result["quantization_hint"] == expected
        assert result["custom_hints"]["streaming_buffer_size"] == 100
        assert result["custom_hints"]["streaming_concurrent_search"] is True
        assert result["distance_metric"] == "cosine"

    custom_hint = {"hint_type": "custom"}
    assert (
        build_search_optimization_rest(quantization_hint=custom_hint)[
            "quantization_hint"
        ]
        == custom_hint
    )
    assert build_search_optimization_rest() == {}


def test_search_hints_grpc_python_value_conversion_and_errors():
    from google.protobuf.struct_pb2 import NullValue

    from proximadb_sdk.v1 import types_pb2

    params = build_search_params_grpc(
        top_k=5,
        custom_hints={
            "none": None,
            "bool": True,
            "int": 1,
            "float": 2.5,
            "str": "value",
            "bytes": b"bytes",
            "list": [1, "two"],
            "dict": {"nested": False},
            "other": object(),
        },
        streaming_batch_size=8,
    )

    assert params.top_k == 5
    assert params.custom_hints["none"].null_value == NullValue.NULL_VALUE
    assert params.custom_hints["bool"].bool_value is True
    assert params.custom_hints["int"].int64_value == 1
    assert params.custom_hints["float"].number_value == 2.5
    assert params.custom_hints["str"].string_value == "value"
    assert params.custom_hints["bytes"].bytes_value == b"bytes"
    assert params.custom_hints["list"].array_value.values[0].int64_value == 1
    assert params.custom_hints["dict"].object_value.fields["nested"].bool_value is False
    assert params.custom_hints["streaming_batch_size"].int64_value == 8

    memory_value = _python_value_to_sql_value(memoryview(b"x"), types_pb2)
    assert memory_value.bytes_value == b"x"

    rest_hints = build_search_hints("rest", top_k=3)
    assert rest_hints == {"top_k": 3}
    grpc_hints = build_search_hints("grpc", top_k=3)
    assert grpc_hints.top_k == 3
    with pytest.raises(ValueError):
        build_search_hints("websocket")
