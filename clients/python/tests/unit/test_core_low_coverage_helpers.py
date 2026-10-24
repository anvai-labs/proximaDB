from enum import Enum

import pytest

from proximadb_sdk import enum_packing
from proximadb_sdk.builders.search import SearchBuilder, search, similarity_search
from proximadb_sdk.filters import FilterBuilder
from proximadb_sdk.metadata_utils import (
    dict_to_proto_metadata,
    json_compatible_value,
    proto_metadata_to_dict,
)
from proximadb_sdk.performance.data_models import (
    BenchmarkMetrics,
    BenchmarkResult,
    EnginePerformance,
    LatencyStats,
    MemoryMetrics,
    PerformanceReport,
    PerformanceSummary,
    ValidationStatus,
    create_latency_stats,
    create_throughput_metrics,
    create_validation_result,
)
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


class StringMetric(Enum):
    COSINE = "cosine"
    UNKNOWN = "unknown"


class IntMetric(Enum):
    EUCLIDEAN = 2


class ModelDumpRecord:
    def model_dump(self, exclude_none=False):
        return {"id": "dumped", "vector": [1.0], "metadata": {"source": "model"}}


class DictRecord:
    def dict(self, exclude_none=False):
        return {"id": "dict", "vector": [2.0], "metadata": {"source": "dict"}}


class PlainRecord:
    id = "plain"
    vector = (3.0, 4.0)
    metadata = {"source": "plain"}


def test_metadata_roundtrip_preserves_supported_proto_value_shapes():
    metadata = {
        "name": "paper",
        "score": 4.5,
        "count": 3,
        "active": True,
        "missing": None,
        "tags": ["db", "vector"],
    }

    proto_items = dict_to_proto_metadata(metadata)

    assert [item.key for item in proto_items] == list(metadata)
    assert proto_metadata_to_dict(proto_items) == {
        "name": "paper",
        "score": 4.5,
        "count": 3.0,
        "active": True,
        "missing": "",
        "tags": "['db', 'vector']",
    }


def test_metadata_proto_to_dict_handles_int64_and_empty_values():
    class FakeMetadataValue:
        def __init__(self, key, field_name=None, value=None):
            self.key = key
            self._field_name = field_name
            if field_name is not None:
                setattr(self, field_name, value)

        def HasField(self, field_name):
            return self._field_name == field_name

    item_with_int64 = FakeMetadataValue("rows", "int64_value", 42)
    item_with_int = FakeMetadataValue("attempts", "int_value", 3)
    item_without_value = FakeMetadataValue("unset")

    assert proto_metadata_to_dict(
        [item_with_int64, item_with_int, item_without_value]
    ) == {
        "rows": 42,
        "attempts": 3,
        "unset": None,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (3, 3.0),
        (2.5, 2.5),
        ("ready", "ready"),
        (None, None),
        ({"nested": "object"}, "{'nested': 'object'}"),
    ],
)
def test_json_compatible_value(value, expected):
    assert json_compatible_value(value) == expected


def test_search_builder_builds_fluent_options_and_serializable_dict():
    filter_builder = FilterBuilder().equals("category", "docs")

    built = (
        SearchBuilder([0.1, 0.2])
        .top_k(25)
        .include_vectors()
        .include_metadata(False)
        .filter(filter_builder)
        .filter_by("tenant", "acme")
        .filter_range("score", min_value=0.2, max_value=0.9)
        .filter_in("status", ["active", "pending"])
        .filter_exists("lang")
        .explain()
        .use_index(False)
        .timeout(2500)
        .build()
    )

    assert built["top_k"] == 25
    assert built["include_vectors"] is True
    assert built["include_metadata"] is False
    assert built["explain"] is True
    assert built["use_index"] is False
    assert built["timeout_ms"] == 2500
    assert built["filter"]["operator"] == "and"
    assert built["filter"]["conditions"][0] == {
        "field": "category",
        "operation": "equals",
        "value": "docs",
    }
    assert built["filter"]["conditions"][-1] == {
        "field": "lang",
        "operation": "exists",
    }

    as_dict = search([0.1, 0.2]).top_k(3).timeout(100).to_dict()
    assert as_dict == {
        "vector": [0.1, 0.2],
        "k": 3,
        "include_vectors": False,
        "include_metadata": True,
        "explain": False,
        "use_index": True,
        "timeout_ms": 100,
    }


@pytest.mark.parametrize(
    "call",
    [
        lambda: SearchBuilder([0.1]).top_k(0),
        lambda: SearchBuilder([0.1]).top_k(10001),
        lambda: SearchBuilder([0.1]).filter_range("score"),
        lambda: SearchBuilder([0.1]).filter_in("status", []),
        lambda: SearchBuilder([0.1]).timeout(0),
    ],
)
def test_search_builder_validates_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_similarity_search_uses_canonical_defaults_and_overrides():
    assert similarity_search([1.0], top_k=2, include_metadata=False) == {
        "top_k": 2,
        "include_vectors": False,
        "include_metadata": False,
        "filter": None,
        "explain": False,
        "use_index": True,
        "timeout_ms": None,
    }


def test_performance_data_models_and_helper_factories():
    empty_latency = LatencyStats.from_samples([])
    assert empty_latency.min_ms == 0
    assert empty_latency.max_ms == 0
    assert empty_latency.avg_ms == 0

    latency = LatencyStats.from_samples([4.0, 1.0, 9.0, 2.0])
    assert latency.min_ms == 1.0
    assert latency.max_ms == 9.0
    assert latency.avg_ms == 4.0
    assert latency.p50_ms == 4.0
    assert latency.p95_ms == 9.0
    assert latency.p99_ms == 9.0
    assert latency.std_dev_ms > 0

    factory_latency = create_latency_stats(1.0, 10.0, 3.0, p99_ms=8.0)
    throughput = create_throughput_metrics(120.0, total_ops=12, duration_ms=100.0)
    memory = MemoryMetrics(peak_memory_mb=32.0, avg_memory_mb=16.0)
    metrics = BenchmarkMetrics(
        latency=factory_latency,
        throughput=throughput,
        memory=memory,
        recall=0.9,
        precision=0.8,
    )
    engine = EnginePerformance(
        engine_name="viper",
        insert_metrics=metrics,
        flush_time_ms=3.0,
        storage_size_mb=1.5,
    )
    benchmark = BenchmarkResult(
        benchmark_name="smoke",
        duration_seconds=1.0,
        vector_count=10,
        dimension=8,
        engine_results=[engine],
        metadata={"profile": "unit"},
    )
    summary = PerformanceSummary(
        total_vectors_tested=10,
        total_queries_executed=2,
        avg_insert_latency_ms=1.2,
        avg_search_latency_ms=2.3,
        best_engine_insert="viper",
        best_engine_search="viper",
        recommendations=["keep baseline"],
    )
    report = PerformanceReport(
        report_id="report-1",
        environment={"python": "test"},
        benchmark_results=[benchmark],
        validation_results=[
            create_validation_result("recall", 0.92, 0.9, threshold=0.9),
            create_validation_result(
                "latency", 3.0, 2.0, threshold=2.0, comparator="<="
            ),
            create_validation_result("exact", "viper", "viper"),
        ],
        summary=summary,
        competitor_comparison={"other": {"latency": 5.0}},
    )

    assert report.benchmark_results[0].engine_results[0].engine_name == "viper"
    assert report.validation_results[0].status == ValidationStatus.PASS
    assert report.validation_results[1].status == ValidationStatus.FAIL
    assert report.validation_results[2].status == ValidationStatus.PASS
    assert throughput.operations_per_second == 120.0
    assert memory.gc_collections == 0


@pytest.mark.parametrize(
    ("actual", "threshold", "comparator", "expected_status"),
    [
        (1.0, 2.0, ">=", ValidationStatus.FAIL),
        (1.0, 2.0, "<=", ValidationStatus.PASS),
        (2.0, 2.0, "==", ValidationStatus.PASS),
        (1.0, 2.0, "unsupported", ValidationStatus.PASS),
    ],
)
def test_create_validation_result_comparators(
    actual, threshold, comparator, expected_status
):
    result = create_validation_result(
        "threshold", actual, "ignored", threshold=threshold, comparator=comparator
    )
    assert result.status == expected_status
    assert result.actual_value == actual
    assert result.threshold == threshold


def test_proto_converter_distance_storage_index_and_quantization_mappings():
    assert ProtoConverter.distance_metric_to_int(None) == 0
    assert ProtoConverter.distance_metric_to_int("COSINE") == 1
    assert ProtoConverter.distance_metric_to_int(IntMetric.EUCLIDEAN) == 2
    assert ProtoConverter.distance_metric_to_int(StringMetric.UNKNOWN) == 0
    assert ProtoConverter.distance_metric_to_str(None) == "cosine"
    assert ProtoConverter.distance_metric_to_str(2) == "euclidean"
    assert ProtoConverter.distance_metric_to_str("not-a-metric") == "cosine"

    # TD-SDKENG-1: SST is the fallback everywhere; VIPER is deprecated
    # (ADR-093) and the server rejects it.
    assert ProtoConverter.storage_engine_to_int(None) == 2
    assert ProtoConverter.storage_engine_to_int("unknown") == 2
    # mmap is a KNOWN engine that maps to the viper int slot (pre-existing
    # table); only unknown/None fall back to SST now.
    assert ProtoConverter.storage_engine_to_str("mmap") == "viper"
    assert ProtoConverter.storage_engine_to_str(3) == "nova"
    assert ProtoConverter.storage_engine_to_str("unknown") == "sst"

    assert ProtoConverter.index_type_to_int(None) == 1
    assert ProtoConverter.index_type_to_int("ivf") == 2
    assert ProtoConverter.index_type_to_str(4) == "flat"
    assert ProtoConverter.index_type_to_str("bad") == "hnsw"

    assert ProtoConverter.quantization_type_to_int(None) == 0
    assert ProtoConverter.quantization_type_to_int("scalar") == 3
    assert ProtoConverter.quantization_type_to_str(2) == "pq"
    assert ProtoConverter.quantization_type_to_str("bad") == "none"

    assert distance_metric_to_int("dot_product") == 3
    assert distance_metric_to_str(3) == "dot_product"
    assert storage_engine_to_int("sst") == 2
    assert storage_engine_to_str(2) == "sst"
    assert index_type_to_int("pq") == 3
    assert index_type_to_str(3) == "pq"
    assert quantization_type_to_int("binary") == 4
    assert quantization_type_to_str(4) == "binary"


def test_proto_converter_model_helpers_normalize_records_and_configs():
    assert ProtoConverter.vector_record_to_dict({"id": "dict"}) == {"id": "dict"}
    assert ProtoConverter.vector_record_to_dict(ModelDumpRecord()) == {
        "id": "dumped",
        "vector": [1.0],
        "metadata": {"source": "model"},
    }
    assert ProtoConverter.vector_record_to_dict(DictRecord()) == {
        "id": "dict",
        "vector": [2.0],
        "metadata": {"source": "dict"},
    }
    assert ProtoConverter.vector_record_to_dict(PlainRecord()) == {
        "id": "plain",
        "vector": [3.0, 4.0],
        "metadata": {"source": "plain"},
    }
    assert ProtoConverter.vector_record_to_dict(object()) == {
        "id": "",
        "vector": [],
        "metadata": None,
    }

    assert ProtoConverter.dict_to_search_result(
        {"vector_id": "v1", "distance": 0.25}
    ) == {
        "id": "v1",
        "score": 0.25,
        "vector": [],
        "metadata": {},
    }
    assert ProtoConverter.collection_config_to_dict(
        "docs",
        128,
        distance_metric="euclidean",
        storage_engine="nova",
        index_type="flat",
        replicas=2,
    ) == {
        "name": "docs",
        "dimension": 128,
        "distance_metric": "euclidean",
        "storage_engine": "nova",
        "index_type": "flat",
        "replicas": 2,
    }


def test_enum_packing_roundtrips_and_builds_proto_ready_dicts():
    packed = enum_packing.pack_processing_enums(
        enum_packing.ExtractionMethod.PDF_PARSING,
        enum_packing.ProcessingStatus.PROCESSED,
        enum_packing.QualityLevel.HIGH,
        enum_packing.DataSource.API_INGESTION,
    )
    assert enum_packing.unpack_processing_enums(packed) == (
        enum_packing.ExtractionMethod.PDF_PARSING,
        enum_packing.ProcessingStatus.PROCESSED,
        enum_packing.QualityLevel.HIGH,
        enum_packing.DataSource.API_INGESTION,
    )

    attributes = enum_packing.pack_source_attributes(
        enum_packing.ContentCategory.SCIENTIFIC,
        enum_packing.QualityLevel.MEDIUM,
    )
    assert enum_packing.unpack_source_attributes(attributes) == (
        enum_packing.ContentCategory.SCIENTIFIC,
        enum_packing.QualityLevel.MEDIUM,
    )

    assert (
        enum_packing.unpack_language_code(
            enum_packing.pack_language_code(enum_packing.LanguageCode.JAPANESE)
        )
        == enum_packing.LanguageCode.JAPANESE
    )

    processing = enum_packing.create_processing_info(
        model_id="model-a",
        extraction=enum_packing.ExtractionMethod.OCR,
        status=enum_packing.ProcessingStatus.PROCESSED,
        quality=enum_packing.QualityLevel.HIGH,
        source=enum_packing.DataSource.USER_UPLOAD,
        processing_time_ms=10,
        processor_version=2,
    )
    text = enum_packing.create_text_content(
        "hello",
        language=enum_packing.LanguageCode.ENGLISH,
        custom_language="en-US",
        chunk_context={"chunk_id": "c1"},
    )
    source_content = enum_packing.create_source_content(
        {"text": text},
        category=enum_packing.ContentCategory.DOCUMENT,
        quality=enum_packing.QualityLevel.HIGH,
        mime_type="text/plain",
        size_bytes=5,
        compressed_size=4,
        checksum=1234,
        processing_info=processing,
    )

    assert processing["model_id"] == "model-a"
    assert processing["processing_time_ms"] == 10
    assert text["language_code"] == enum_packing.LanguageCode.ENGLISH
    assert text["chunk"] == {"chunk_id": "c1"}
    assert source_content["processing"] == processing
    assert source_content["compressed_size"] == 4
    assert source_content["checksum"] == 1234


@pytest.mark.parametrize(
    "call",
    [
        lambda: enum_packing.unpack_processing_enums(0xFF),
        lambda: enum_packing.unpack_source_attributes(0xFF),
        lambda: enum_packing.unpack_language_code(254),
    ],
)
def test_enum_packing_rejects_invalid_packed_values(call):
    with pytest.raises(ValueError):
        call()


def test_enum_packing_storage_efficiency_analysis():
    analysis = enum_packing.storage_efficiency_analysis()

    assert analysis["old_total_bytes"] == 28
    assert analysis["new_total_bytes"] == 12
    assert analysis["savings_bytes"] == 16
    assert analysis["efficiency_ratio"] > 2
