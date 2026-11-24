"""Offline coverage tests for models.py, models_v2.py, and builders/*.

Fully offline: pure model/builder construction, validators, round-trips.
"""

import warnings
from datetime import datetime

import numpy as np
import pytest

from proximadb_sdk import models, models_v2
from proximadb_sdk.builders import collection as bcoll
from proximadb_sdk.builders import insert as binsert
from proximadb_sdk.builders import search as bsearch
from proximadb_sdk.builders.collection import CollectionBuilder
from proximadb_sdk.builders.insert import InsertBuilder
from proximadb_sdk.builders.search import SearchBuilder

# ---------------------------------------------------------------------------
# models.py - enums
# ---------------------------------------------------------------------------


def test_enums_have_expected_values():
    assert models.DistanceMetric.COSINE.value == "cosine"
    assert models.DistanceMetricType.COSINE.value == 1
    assert models.StorageEngine.VIPER.value == "viper"
    assert models.IndexingAlgorithm.HNSW.value == "hnsw"
    assert models.IndexType.HNSW.value == 1
    assert models.StorageEngineType.VIPER.value == 1
    assert models.IndexUpdateMode.SYNCHRONOUS.value == "synchronous"
    assert models.FilterableDataType.STRING.value == "string"
    assert models.CompressionAlgorithm.ZSTD.value == "zstd"
    assert models.CompressionLevel.BALANCED.value == 6
    assert models.QuantizationType.PRODUCT.value == "pq"
    assert models.AccessPattern.READ_HEAVY.value == "read_heavy"
    assert models.DataDensity.DENSE.value == "dense"
    assert models.RandomProjectionType.GAUSSIAN.value == "gaussian"
    assert models.CollectionOperationType.CREATE.value == "create"
    assert models.VectorOperationType.INSERT.value == "insert"
    assert models.FilterOperator.AND.value == "and"
    assert models.FilterOperation.EQUALS.value == "equals"
    assert models.CompressionType.GZIP.value == "gzip"


# ---------------------------------------------------------------------------
# EmbeddingPrecision normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("fp32", models.EmbeddingPrecision.FP32),
        ("f32", models.EmbeddingPrecision.FP32),
        ("float32", models.EmbeddingPrecision.FP32),
        ("fp16", models.EmbeddingPrecision.FP16),
        ("half", models.EmbeddingPrecision.FP16),
        ("float16", models.EmbeddingPrecision.FP16),
        ("EMBEDDING_PRECISION_FP16", models.EmbeddingPrecision.FP16),
        ("bf16", models.EmbeddingPrecision.BF16),
        ("bfloat16", models.EmbeddingPrecision.BF16),
        ("int8", models.EmbeddingPrecision.INT8),
        ("i8", models.EmbeddingPrecision.INT8),
        ("uint8", models.EmbeddingPrecision.UINT8),
        ("u8", models.EmbeddingPrecision.UINT8),
    ],
)
def test_embedding_precision_normalize(raw, expected):
    assert models.EmbeddingPrecision._normalize(raw) is expected


def test_embedding_precision_normalize_enum_passthrough():
    p = models.EmbeddingPrecision.FP16
    assert models.EmbeddingPrecision._normalize(p) is p


def test_embedding_precision_normalize_invalid():
    with pytest.raises(ValueError):
        models.EmbeddingPrecision._normalize("nope")
    with pytest.raises(ValueError):
        models.EmbeddingPrecision._normalize(123)


# ---------------------------------------------------------------------------
# ServerCapabilities
# ---------------------------------------------------------------------------


def test_server_capabilities_methods():
    assert models.ServerCapabilities.is_supported("distance_metric", "cosine")
    assert not models.ServerCapabilities.is_supported("storage_engine", "mmap")
    assert models.ServerCapabilities.is_supported("indexing_algorithm", "hnsw")
    assert models.ServerCapabilities.is_supported("quantization_type", "binary")
    assert models.ServerCapabilities.is_supported("unknown_type", "whatever")
    # fallbacks
    assert (
        models.ServerCapabilities.get_fallback_for("storage_engine", "mmap") == "sst"
    )
    assert (
        models.ServerCapabilities.get_fallback_for("distance_metric", "cosine") is None
    )
    assert (
        models.ServerCapabilities.get_fallback_for("indexing_algorithm", "hnsw") is None
    )
    assert models.ServerCapabilities.get_fallback_for("bogus", "x") is None


# ---------------------------------------------------------------------------
# Quantization models
# ---------------------------------------------------------------------------


def test_quantization_models_roundtrip():
    level = models.QuantizationLevel(
        level_type="pq", bits=8, num_subvectors=4, config={"a": "b"}
    )
    sq = models.StorageQuantizationConfig(enabled=True, level=level, codebook_id="cb")
    strat = models.IndexQuantizationStrategy(index_name="primary", level=level)
    iq = models.IndexQuantizationConfig(enabled=True, strategies=[strat])
    srch = models.SearchQuantizationConfig(enabled=True, default_level=level)
    val = models.QuantizationValidation()
    comp = models.ComprehensiveQuantizationConfig(
        enabled=True,
        storage_quantization=sq,
        index_quantization=iq,
        search_quantization=srch,
        validation=val,
    )
    # round-trip
    dumped = comp.model_dump()
    restored = models.ComprehensiveQuantizationConfig(**dumped)
    assert restored.enabled is True
    assert restored.storage_quantization.level.bits == 8

    qc = models.QuantizationConfig(enabled=True, type=models.QuantizationType.SCALAR)
    assert qc.type == models.QuantizationType.SCALAR


# ---------------------------------------------------------------------------
# CompressionConfig validators
# ---------------------------------------------------------------------------


def test_compression_config_valid():
    c = models.CompressionConfig(
        algorithm=models.CompressionAlgorithm.ZSTD,
        level=10,
        min_ratio=0.5,
        quantization_type="int8",
        normalization_method="mean",
        block_size_kb=512,
    )
    assert c.level == 10
    assert c.quantization_type == "int8"


def test_compression_config_invalid_level():
    with pytest.raises(ValueError):
        models.CompressionConfig(level=99)


def test_compression_config_invalid_ratio():
    with pytest.raises(ValueError):
        models.CompressionConfig(min_ratio=2.0)


def test_compression_config_invalid_quant_type():
    with pytest.raises(ValueError):
        models.CompressionConfig(quantization_type="bogus")


def test_compression_config_invalid_normalization():
    with pytest.raises(ValueError):
        models.CompressionConfig(normalization_method="bogus")


def test_compression_config_invalid_block_size():
    with pytest.raises(ValueError):
        models.CompressionConfig(block_size_kb=1)


# ---------------------------------------------------------------------------
# Storage / index settings models
# ---------------------------------------------------------------------------


def test_settings_models():
    models.ParquetWriterSettings(row_group_size=1000, enable_bloom_filters=True)
    models.FooterCacheSettings(enable=True, max_entries=100)
    models.HybridWriterSettings(enable=True, initial_mode="adaptive")
    models.SstEngineSettings(enable_bloom_filters=True, block_size_kb=512)
    models.NovaEngineSettings(enable_real_time_mode=True)
    models.HnswConfig(m=32)
    models.IvfConfig(n_lists=200)
    models.FlatConfig()
    models.PqConfig()
    models.AnnoyConfig()
    lsh = models.LshConfig(projection=models.RandomProjectionType.SPARSE)
    assert lsh.projection == models.RandomProjectionType.SPARSE


def test_index_configuration():
    ic = models.IndexConfiguration(
        index_name="primary",
        algorithm=models.IndexingAlgorithm.HNSW,
        hnsw_config=models.HnswConfig(),
        ivf_config=models.IvfConfig(),
        use_cases=["search"],
    )
    assert ic.index_name == "primary"
    ic2 = models.IndexConfiguration(index_name="grpc", algorithm=models.IndexType.IVF)
    assert ic2.algorithm == models.IndexType.IVF


def test_filterable_column():
    fc = models.FilterableColumn(
        name="cat", data_type=models.FilterableDataType.STRING, supports_range=True
    )
    assert fc.indexed is True


# ---------------------------------------------------------------------------
# CollectionConfig
# ---------------------------------------------------------------------------


def test_collection_config_basic():
    cfg = models.CollectionConfig(name="my_collection", dimension=128)
    assert cfg.distance_metric == models.DistanceMetric.COSINE
    assert cfg.storage_engine == models.StorageEngine.SST
    assert cfg.index_config is None
    assert cfg.quantization is None


def test_collection_config_name_short_accepted():
    # The former 8-char minimum was relaxed for relational-DDL tables (#1113);
    # short names are now accepted (empty/whitespace-only still rejected).
    assert models.CollectionConfig(name="short", dimension=128).name == "short"


def test_collection_config_name_empty():
    with pytest.raises(ValueError):
        models.CollectionConfig(name="        ", dimension=128)


def test_collection_config_dimension_bounds():
    with pytest.raises(ValueError):
        models.CollectionConfig(name="collection1", dimension=0)
    with pytest.raises(ValueError):
        models.CollectionConfig(name="collection1", dimension=100000)


def test_collection_config_precision_normalization():
    cfg = models.CollectionConfig(
        name="collection1", dimension=8, canonical_embedding_precision="half"
    )
    assert cfg.canonical_embedding_precision == models.EmbeddingPrecision.FP16
    cfg2 = models.CollectionConfig(name="collection1", dimension=8)
    assert cfg2.canonical_embedding_precision is None


def test_collection_config_index_config_property():
    ic = models.IndexConfiguration(
        index_name="primary", algorithm=models.IndexingAlgorithm.HNSW
    )
    cfg = models.CollectionConfig(name="collection1", dimension=8, index_configs=[ic])
    assert cfg.index_config is ic


def test_collection_config_quantization_alias():
    qc = models.QuantizationConfig(enabled=True)
    cfg = models.CollectionConfig(name="collection1", dimension=8, quantization=qc)
    assert cfg.quantization is qc
    assert cfg.quantization_config is qc


def test_collection_config_sst_quant_warns_and_default_block():
    comp = models.CompressionConfig(enable_quantization=True, quantization_type="int8")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = models.CollectionConfig(
            name="collection1",
            dimension=8,
            storage_engine=models.StorageEngine.SST,
            compression=comp,
        )
    assert any("ignored by SST" in str(x.message) for x in w)
    assert cfg.compression.block_size_kb == 8192


def test_collection_config_zstd_level_too_high():
    comp = models.CompressionConfig(
        algorithm=models.CompressionAlgorithm.ZSTD, level=20
    )
    # 20 is valid (<=22). Bump to test the post-init >22 branch via direct mutation.
    comp.level = 23
    with pytest.raises(ValueError):
        models.CollectionConfig(
            name="collection1",
            dimension=8,
            storage_engine=models.StorageEngine.SST,
            compression=comp,
        )


def test_collection_config_nonzstd_level_too_high():
    comp = models.CompressionConfig(algorithm=models.CompressionAlgorithm.LZ4, level=5)
    comp.level = 10
    with pytest.raises(ValueError):
        models.CollectionConfig(
            name="collection1",
            dimension=8,
            storage_engine=models.StorageEngine.SST,
            compression=comp,
        )


# ---------------------------------------------------------------------------
# Collection / CollectionInfo / stats
# ---------------------------------------------------------------------------


def test_collection_info_backcompat_props():
    info = models.CollectionInfo(
        id="abc",
        name="collection1",
        dimension=8,
        metric="cosine",
        created_at_ms=2000,
        updated_at_ms=4000,
    )
    assert info.created_at == 2
    assert info.updated_at == 4
    info.created_at = 5
    info.updated_at = 6
    assert info.created_at_ms == 5000
    assert info.updated_at_ms == 6000


def test_collection_model_props():
    cfg = models.CollectionConfig(name="collection1", dimension=8)
    coll = models.Collection(id="abc", config=cfg)
    assert coll.name == "collection1"
    assert coll.dimension == 8
    assert coll.distance_metric == models.DistanceMetric.COSINE
    assert coll.storage_engine == models.StorageEngine.SST
    # No stats were supplied, so the count is unknown rather than zero.
    assert coll.vector_count is None
    assert coll.timestamp == coll.created_at_ms // 1000
    coll.created_at = 7
    coll.updated_at = 8
    assert coll.created_at_ms == 7000
    assert coll.updated_at_ms == 8000
    assert coll.created_at == 7
    assert coll.updated_at == 8


# ---------------------------------------------------------------------------
# VectorRecord
# ---------------------------------------------------------------------------


def test_vector_record_basic_and_dict_access():
    vr = models.VectorRecord(id="v1", vector=[0.1, 0.2], metadata={"k": "val"})
    assert vr["id"] == "v1"
    assert vr.get("id") == "v1"
    assert vr.get("missing", "d") == "d"


def test_vector_record_empty_vector():
    with pytest.raises(ValueError):
        models.VectorRecord(vector=[])


def test_vector_record_nonfinite():
    with pytest.raises(ValueError):
        models.VectorRecord(vector=[float("inf"), 0.1])


def test_vector_record_timestamp_props():
    vr = models.VectorRecord(vector=[0.1])
    vr.timestamp = 10
    assert vr.timestamp == 10
    assert vr.timestamp_ms == 10000
    vr.updated_at = 20
    assert vr.updated_at == 20
    vr.updated_at = None
    assert vr.updated_at is None
    vr.expires_at = 30
    assert vr.expires_at == 30
    vr.expires_at = None
    assert vr.expires_at is None
    assert models.Vector is models.VectorRecord


# ---------------------------------------------------------------------------
# Search models
# ---------------------------------------------------------------------------


def test_search_models():
    cond = models.FilterCondition(
        field_name="cat", operation=models.FilterOperation.EQUALS, value="x"
    )
    mf = models.MetadataFilter(conditions=[cond])
    assert mf.operator == models.FilterOperator.AND
    sq = models.SearchQuery(vector=[0.1, 0.2], metadata_filter=mf)
    assert sq.filters == {}
    models.SearchParameters(ef_search=10, n_probe=5)
    inc = models.IncludeFields()
    assert inc.metadata is True
    hint = models.QuantizationHint(hint_type="binary", parameters={"a": 1})
    opt = models.SearchOptimization(top_k=5, quantization_hint=hint)
    assert opt.use_decompression_cache is True
    res = models.SearchResult(id="r1", score=0.9, timestamp=12345)
    assert res.timestamp_ms == 12345
    prog = models.SearchProgress(stage=1, stages=3, complete=False)
    env = models.SearchEnvelope(items=[res], progress=prog)
    assert env.has_more is False
    models.VectorGetResponse(id="x", collection_id="c", vector=[0.1])
    li = models.ListCollectionsResponse(
        collections=[
            models.CollectionInfo(
                id="a",
                name="collection1",
                dimension=8,
                metric="cosine",
                created_at_ms=1,
                updated_at_ms=1,
            )
        ],
        total_count=1,
    )
    assert li.total_count == 1


def test_request_response_models():
    cfg = models.CollectionConfig(name="collection1", dimension=8)
    req = models.CollectionOperationRequest(
        operation=models.CollectionOperationType.CREATE, config=cfg
    )
    assert req.operation == models.CollectionOperationType.CREATE
    coll = models.Collection(id="x", config=cfg)
    resp = models.CollectionResponse(success=True, operation="create", collection=coll)
    assert resp.affected_count == 0

    vr = models.VectorRecord(id="v1", vector=[0.1])
    batch = models.VectorBatchRequest(collection_id="c", vectors=[vr])
    assert len(batch.vectors) == 1

    vsr = models.VectorSearchRequest(
        collection_id="c", queries=[models.SearchQuery(vector=[0.1])]
    )
    assert vsr.top_k == 10

    metrics = models.OperationMetrics(successful_count=3)
    dr = models.DeleteResult(deleted_count=2, metrics=metrics)
    assert dr.success is True
    br = models.BatchResult(total=5, success=5)
    assert isinstance(br.metrics, models.OperationMetrics)
    vor = models.VectorOperationResponse(
        success=True, operation="insert", metrics=metrics
    )
    assert vor.count == 3
    err = models.ApiError(code="E", message="boom")
    api = models.ApiResponse(success=False, error=err, extra_field="ok")
    assert api.error.code == "E"


# ---------------------------------------------------------------------------
# Health / storage / schema models
# ---------------------------------------------------------------------------


def test_storage_and_health_models():
    sc = models.StorageConfig(
        storage_location="/data",
        compression=models.CompressionConfig(),
        access_pattern=models.AccessPattern.BALANCED,
        data_density=models.DataDensity.MIXED,
        preset="balanced",
        parquet_writer=models.ParquetWriterSettings(),
        footer_cache=models.FooterCacheSettings(),
        hybrid_writer=models.HybridWriterSettings(),
        sst_settings=models.SstEngineSettings(),
        nova_settings=models.NovaEngineSettings(),
    )
    assert sc.persistent is True
    models.FlushConfig(force_flush=True)
    hs = models.HealthStatus(
        status="ok",
        version="1.0",
        uptime_seconds=10,
        services={"db": "up"},
        timestamp_ms=5000,
    )
    assert hs.timestamp == 5
    hs.timestamp = 9
    assert hs.timestamp_ms == 9000
    models.ProbeResponse(status="ready", extra=1)
    col = models.ColumnDefinition(name="c", data_type="text")
    schema = models.SchemaDefinition(columns=[col], enforcement="strict")
    resp = models.SchemaResponse(
        schema_id="s",
        schema_version="1",
        collection_id="c",
        **{"schema": schema},
        created_at="now",
    )
    assert resp.schema_.columns[0].name == "c"
    models.UpdateSchemaResponse(
        schema_id="s",
        schema_version="2",
        previous_schema_id="p",
        changes=[{"a": "b"}],
        warnings=[],
        updated_at="now",
    )


# ---------------------------------------------------------------------------
# models_v2.py
# ---------------------------------------------------------------------------


def test_v2_column_data_type_enum():
    assert models_v2.ColumnDataType.TEXT.value == "text"
    assert models_v2.TextStorageStrategy.CHUNKED.value == "chunked"
    assert models_v2.SchemaEnforcement.STRICT.value == "strict"


def test_text_field_validators():
    tf = models_v2.TextField(name="  body  ", content="hello")
    assert tf.name == "body"
    assert tf.storage_hint == models_v2.TextStorageStrategy.ADAPTIVE
    with pytest.raises(ValueError):
        models_v2.TextField(name=" ", content="x")
    with pytest.raises(ValueError):
        models_v2.TextField(name="x", content="a" * (10 * 1024 * 1024 + 1))


def test_text_column_config_validators_and_factories():
    cfg = models_v2.TextColumnConfig(
        column_name="  col_1  ", chunk_size=512, chunk_overlap=50
    )
    assert cfg.column_name == "col_1"
    assert cfg.to_dict()["column_name"] == "col_1"
    with pytest.raises(ValueError):
        models_v2.TextColumnConfig(column_name=" ")
    with pytest.raises(ValueError):
        models_v2.TextColumnConfig(column_name="1bad")
    with pytest.raises(ValueError):
        models_v2.TextColumnConfig(column_name="bad-name")
    with pytest.raises(ValueError):
        models_v2.TextColumnConfig(column_name="col", chunk_size=100, chunk_overlap=200)

    rag = models_v2.TextColumnConfig.for_rag("content", embedding_model="m")
    assert rag.strategy == models_v2.TextStorageStrategy.CHUNKED
    short = models_v2.TextColumnConfig.for_short_text(
        "title", enable_full_text_search=True
    )
    assert short.strategy == models_v2.TextStorageStrategy.INLINE
    large = models_v2.TextColumnConfig.for_large_documents("body", language="en")
    assert large.strategy == models_v2.TextStorageStrategy.SIDECAR
    hyb = models_v2.TextColumnConfig.for_hybrid_search("art", ngram_size=4)
    assert hyb.enable_ngram_bloom is True


def test_typed_value_factories():
    assert models_v2.TypedValue.text("x").value_type == models_v2.ColumnDataType.TEXT
    assert (
        models_v2.TypedValue.text_large("x").value_type
        == models_v2.ColumnDataType.TEXT_LARGE
    )
    assert models_v2.TypedValue.integer(1).value == 1
    assert models_v2.TypedValue.float_(1.5).value == 1.5
    assert models_v2.TypedValue.decimal("1.5").value == "1.5"
    assert models_v2.TypedValue.boolean(True).value is True
    uid = "550e8400-e29b-41d4-a716-446655440000"
    assert models_v2.TypedValue.uuid(uid).value == uid
    with pytest.raises(ValueError):
        models_v2.TypedValue.uuid("not-a-uuid")
    dt = datetime(2024, 1, 1, 12, 0, 0)
    assert isinstance(models_v2.TypedValue.timestamp(dt).value, int)
    assert models_v2.TypedValue.timestamp(123).value == 123
    tz = models_v2.TypedValue.timestamp_tz(dt, timezone="UTC")
    assert tz.value["timezone"] == "UTC"
    assert models_v2.TypedValue.timestamp_tz(456).value["timestamp"] == 456
    assert models_v2.TypedValue.date(dt).value == "2024-01-01"
    assert models_v2.TypedValue.date("2024-01-01").value == "2024-01-01"
    assert models_v2.TypedValue.time_(dt).value == "12:00:00"
    assert models_v2.TypedValue.time_("12:00:00").value == "12:00:00"
    assert isinstance(models_v2.TypedValue.binary(b"abc").value, str)
    assert models_v2.TypedValue.json_({"a": 1}).value == {"a": 1}
    assert models_v2.TypedValue.array_text(["a"]).value == ["a"]
    assert models_v2.TypedValue.array_integer([1]).value == [1]
    assert models_v2.TypedValue.array_float([1.0]).value == [1.0]
    assert models_v2.TypedValue.map_string_string({"a": "b"}).value == {"a": "b"}
    assert models_v2.TypedValue.map_string_any({"a": 1}).value == {"a": 1}


def test_proxima_record_fluent_and_props():
    rec = models_v2.ProximaRecord(id="r1", vector=[0.1, 0.2])
    rec.set_typed("price", models_v2.TypedValue.float_(9.99))
    rec.set_flexible("note", "free")
    rec.add_text("body", "content text")
    rec.with_version(3)
    rec.with_ttl(3600)
    assert rec.version == 3
    assert rec.expires_at_ms is not None
    assert rec.text_fields[0].name == "body"
    meta = rec.metadata
    assert meta["price"] == 9.99
    assert meta["note"] == "free"
    rec.timestamp = 100
    assert rec.timestamp == 100
    assert rec.timestamp_ms == 100000
    d = rec.to_dict()
    assert "id" in d


def test_proxima_record_empty_vector():
    with pytest.raises(ValueError):
        models_v2.ProximaRecord(vector=[])


def test_column_definition_and_record_schema():
    schema = (
        models_v2.RecordSchema()
        .add_text_column("title", max_length=256, indexed=True)
        .add_integer_column("year", nullable=False)
        .add_float_column("price")
        .add_boolean_column("active")
        .add_timestamp_column("created")
        .add_json_column("meta")
        .add_uuid_column("uid")
        .add_column("status", models_v2.ColumnDataType.TEXT)
    )
    assert schema.get_column("title").name == "title"
    assert schema.get_column("missing") is None

    schema.add_text_column_config(models_v2.TextColumnConfig.for_short_text("desc"))
    schema.add_rag_text_column("content", chunk_size=256)
    schema.add_large_text_column("pdf", language="en")
    assert schema.get_text_column_config("content") is not None
    assert schema.get_text_column_config("nope") is None


def test_record_schema_validate_record():
    schema = models_v2.RecordSchema(enforcement=models_v2.SchemaEnforcement.STRICT)
    schema.add_column("price", models_v2.ColumnDataType.FLOAT, nullable=False)
    # missing required + wrong type + unknown
    rec = models_v2.ProximaRecord(
        vector=[0.1],
        typed_fields={"price": models_v2.TypedValue.text("oops")},
    )
    errors = schema.validate_record(rec)
    assert any("type" in e for e in errors)

    rec_unknown = models_v2.ProximaRecord(
        vector=[0.1],
        typed_fields={"price": models_v2.TypedValue.float_(1.0)},
        flexible_fields={"extra": 1},
    )
    errors2 = schema.validate_record(rec_unknown)
    assert any("Unknown column 'extra'" in e for e in errors2)

    # missing required column
    rec_missing = models_v2.ProximaRecord(vector=[0.1])
    errors3 = schema.validate_record(rec_missing)
    assert any("missing" in e for e in errors3)

    # flexible mode = no errors
    flex_schema = models_v2.RecordSchema(
        enforcement=models_v2.SchemaEnforcement.FLEXIBLE
    )
    flex_schema.add_column("price", models_v2.ColumnDataType.FLOAT, nullable=False)
    assert flex_schema.validate_record(rec_missing) == []


def test_filter_builder_v2():
    fb = (
        models_v2.FilterBuilderV2("price")
        .gte(10.0)
        .lte(100.0)
        .and_("category")
        .eq("electronics")
        .ne("books")
        .and_("name")
        .contains("phone")
        .starts_with("i")
        .ends_with("x")
        .and_("rating")
        .gt(3)
        .lt(5)
        .and_("range")
        .between(1, 2)
        .and_("status")
        .in_(["a", "b"])
        .and_("deleted")
        .is_null()
        .and_("active")
        .is_not_null()
    )
    conds = fb.build()
    assert len(conds) == 13
    d = fb.to_dict()
    assert d[0]["operator"] == "gte"
    assert models_v2.FilterBuilder is models_v2.FilterBuilderV2


def test_typed_filter_condition():
    c = models_v2.TypedFilterCondition(
        field_name="price",
        operator=models_v2.FilterOperator.BETWEEN,
        value=10.0,
        value_upper=100.0,
    )
    assert c.value_upper == 100.0


def test_search_request_v2_fluent():
    req = (
        models_v2.SearchRequestV2.create([0.1, 0.2, 0.3], top_k=5)
        .with_filter(models_v2.FilterBuilderV2("category").eq("electronics"))
        .with_filters(
            [
                models_v2.TypedFilterCondition(
                    field_name="x", operator=models_v2.FilterOperator.EQ, value=1
                )
            ]
        )
        .with_text()
        .with_vectors()
        .with_ef_search(50)
    )
    assert req.top_k == 5
    assert req.include_text is True
    assert req.include_vectors is True
    assert req.ef_search == 50
    assert len(req.filters) == 2


def test_v2_convenience_functions():
    schema = models_v2.create_text_column_schema(
        text_columns=[models_v2.TextColumnConfig.for_rag("content")],
        additional_columns=[
            models_v2.ColumnDefinition(
                name="title", data_type=models_v2.ColumnDataType.TEXT
            )
        ],
    )
    assert schema.get_text_column_config("content") is not None
    assert schema.get_column("title") is not None
    tc = models_v2.text_column(
        "desc", strategy=models_v2.TextStorageStrategy.CHUNKED, chunk_size=256
    )
    assert tc.chunk_size == 256


# ---------------------------------------------------------------------------
# builders/collection.py
# ---------------------------------------------------------------------------


def test_collection_builder_fluent():
    cfg = (
        CollectionBuilder("my_collection", 384)
        .cosine_similarity()
        .euclidean_distance()
        .dot_product()
        .manhattan_distance()
        .hamming_distance()
        .jaccard_similarity()
        .distance_metric(models.DistanceMetric.COSINE)
        .sst_storage()
        .hybrid_storage()
        .storage_engine(models.StorageEngine.NOVA)
        .hnsw_index()
        .ivf_index()
        .flat_index()
        .annoy_index()
        .lsh_index()
        .index_type(models.IndexingAlgorithm.HNSW)
        .description("a description for the collection")
        .compression("zstd")
        .no_compression()
        .gzip_compression()
        .zstd_compression()
        .lz4_compression()
        .enable_bloom_filter()
        .disable_bloom_filter()
        .index_param("ef", 100)
        .hnsw_params(m=32, ef_construction=400)
        .ivf_params(n_lists=200, n_probes=20)
        .annoy_params(n_trees=20)
        .lsh_params(n_tables=20, n_bits=20)
        .build()
    )
    assert cfg.name == "my_collection"
    assert cfg.dimension == 384


def test_collection_builder_to_dict():
    builder = CollectionBuilder("my_collection", 128).euclidean_distance().sst_storage()
    d = builder.to_dict()
    assert d["name"] == "my_collection"
    assert d["distance_metric"] == "euclidean"
    assert d["storage_engine"] == "sst"
    assert d["primary_indexing_algorithm"] is None


def test_collection_builder_convenience():
    assert bcoll.collection("my_collection", 64).build().dimension == 64
    tc = bcoll.text_collection("text_coll", "all-MiniLM-L6-v2").build()
    assert tc.dimension == 384
    tc2 = bcoll.text_collection("text_coll", "unknown-model").build()
    assert tc2.dimension == 768
    ic = bcoll.image_collection("img_coll", "resnet").build()
    assert ic.dimension == 2048
    ic2 = bcoll.image_collection("img_coll", "unknown").build()
    assert ic2.dimension == 512
    hp = bcoll.high_performance_collection("perf_coll", 256).build()
    assert hp.dimension == 256


# ---------------------------------------------------------------------------
# builders/insert.py
# ---------------------------------------------------------------------------


def test_insert_builder_add_methods():
    b = InsertBuilder()
    b.add_vector("v1", [0.1, 0.2], {"cat": "a"})
    b.add_record({"id": "v2", "vector": [0.3, 0.4], "props": {"cat": "b"}})
    b.add_records([{"id": "v3", "vector": [0.5, 0.6]}])
    rec = models_v2.ProximaRecord(id="v4", vector=[0.7, 0.8])
    b.add_record(rec)
    vr = models.VectorRecord(id="v5", vector=[0.9, 1.0], metadata={"m": 1})
    b.add_vectors([vr])
    assert b.count() == 5
    assert not b.is_empty()
    assert "v1" in b.get_vector_ids()
    assert b.get_dimensions() == [2, 2, 2, 2, 2]


def test_insert_builder_from_arrays_numpy():
    b = InsertBuilder()
    b.from_arrays(["a", "b"], np.array([[0.1, 0.2], [0.3, 0.4]]), [{"x": 1}, {"x": 2}])
    assert b.count() == 2
    with pytest.raises(ValueError):
        InsertBuilder().from_arrays(["a"], [[0.1], [0.2]])
    with pytest.raises(ValueError):
        InsertBuilder().from_arrays(["a", "b"], [[0.1], [0.2]], [{"x": 1}])


def test_insert_builder_options_and_validation():
    b = (
        InsertBuilder()
        .batch_size(500)
        .overwrite_existing()
        .validate_vectors(False)
        .async_mode()
    )
    opts = b.build_options()
    assert opts["batch_size"] == 500
    assert opts["overwrite"] is True
    assert opts["async_mode"] is True
    with pytest.raises(ValueError):
        InsertBuilder().batch_size(0)
    with pytest.raises(ValueError):
        InsertBuilder().batch_size(20000)


def test_insert_builder_transforms():
    b = InsertBuilder()
    b.add_vector("v1", [3.0, 4.0])
    b.add_vector("v1", [1.0, 0.0])  # duplicate id
    b.add_vector("v2", [0.0, 5.0])
    b.filter_duplicates()
    assert b.count() == 2
    b.normalize_vectors()
    # 3,4 -> 0.6, 0.8
    assert abs(b.records[0]["vector"][0] - 0.6) < 1e-9
    b.add_metadata_field("env", "test")
    assert b.records[0]["props"]["env"] == "test"
    b.transform_metadata(lambda p: {**p, "t": 1})
    assert b.records[0]["props"]["t"] == 1
    b.validate_dimensions(2)
    with pytest.raises(ValueError):
        b.validate_dimensions(3)


def test_insert_builder_build_and_summary():
    b = (
        InsertBuilder()
        .add_vector("v1", [0.1, 0.2], {"a": 1})
        .add_vector("v2", [0.3, 0.4])
    )
    records, options = b.build()
    assert len(records) == 2
    assert options["batch_size"] == 1000
    plain = b.build_records()
    assert plain[0]["id"] == "v1"
    legacy = b.build_vectors()
    assert isinstance(legacy[0], models.VectorRecord)
    summ = b.summary()
    assert summ["count"] == 2
    assert summ["dimensions"]["min"] == 2
    assert summ["has_metadata"] is True
    assert InsertBuilder().summary()["count"] == 0
    b.clear()
    assert b.is_empty()


def test_insert_builder_normalize_record_typeerror():
    with pytest.raises(TypeError):
        InsertBuilder._normalize_record(["not", "a", "dict"])


def test_insert_builder_from_dataframe():
    import pandas as pd

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "vec": [[0.1, 0.2], np.array([0.3, 0.4])],
            "label": ["a", "b"],
        }
    )
    b = InsertBuilder().from_dataframe(df, "id", "vec", metadata_cols=["label"])
    assert b.count() == 2
    assert b.records[0]["id"] == "1"
    assert b.records[1]["vector"] == [0.3, 0.4]
    assert b.records[0]["props"]["label"] == "a"

    # string-encoded vector path
    df2 = pd.DataFrame({"id": ["x"], "vec": ["[0.5, 0.6]"]})
    b2 = InsertBuilder().from_dataframe(df2, "id", "vec")
    assert b2.records[0]["vector"] == [0.5, 0.6]

    with pytest.raises(ValueError):
        InsertBuilder().from_dataframe([1, 2, 3], "id", "vec")


def test_insert_convenience_functions():
    records, opts = binsert.batch_insert(
        [{"id": "a", "vector": [0.1]}, {"id": "b", "vector": [0.2]}], batch_size=50
    )
    assert opts["batch_size"] == 50
    assert len(records) == 2
    records2, opts2 = binsert.from_numpy(
        ["a", "b"], np.array([[0.1], [0.2]]), batch_size=10
    )
    assert opts2["batch_size"] == 10
    assert isinstance(binsert.insert(), InsertBuilder)


# ---------------------------------------------------------------------------
# builders/search.py
# ---------------------------------------------------------------------------


def test_search_builder_fluent_and_build():
    sb = (
        SearchBuilder([0.1, 0.2, 0.3])
        .top_k(20)
        .include_vectors()
        .include_metadata()
        .filter_by("category", "electronics")
        .filter_range("price", 100, 1000)
        .filter_in("brand", ["a", "b"])
        .filter_exists("color")
        .explain()
        .use_index()
        .timeout(5000)
    )
    out = sb.build()
    assert out["top_k"] == 20
    assert out["timeout_ms"] == 5000
    assert out["filter"] is not None
    d = sb.to_dict()
    assert d["vector"] == [0.1, 0.2, 0.3]
    assert d["k"] == 20
    assert "filter" in d
    assert d["timeout_ms"] == 5000


def test_search_builder_validation():
    with pytest.raises(ValueError):
        SearchBuilder([0.1]).top_k(0)
    with pytest.raises(ValueError):
        SearchBuilder([0.1]).top_k(20000)
    with pytest.raises(ValueError):
        SearchBuilder([0.1]).timeout(0)
    with pytest.raises(ValueError):
        SearchBuilder([0.1]).filter_range("price")
    with pytest.raises(ValueError):
        SearchBuilder([0.1]).filter_in("x", [])


def test_search_builder_filter_only_min():
    sb = SearchBuilder([0.1]).filter_range("price", min_value=10)
    out = sb.build()
    assert out["filter"]["conditions"][0]["operation"] == "gte"


def test_search_builder_with_filterbuilder():
    from proximadb_sdk.filters import FilterBuilder

    fb = FilterBuilder().equals("category", "electronics")
    sb = SearchBuilder([0.1]).filter(fb)
    assert sb.build()["filter"] is not None


def test_search_convenience_functions():
    assert isinstance(bsearch.search([0.1]), SearchBuilder)
    opts = bsearch.similarity_search([0.1, 0.2], top_k=15, include_vectors=True)
    assert opts["top_k"] == 15
    assert opts["include_vectors"] is True
