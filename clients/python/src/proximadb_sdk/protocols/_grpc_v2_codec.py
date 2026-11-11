"""Pure proto build/parse helpers for the v2 ``ProximaRecordService`` gRPC surface.

These are the *transport-agnostic* converters: they build / parse the generated
``proximadb.v2.record_pb2`` messages and have no dependency on a channel, stub,
connection pool, or event loop. The synchronous client
(``protocols/grpc_sync.py``) and the native-async client
(``protocols/grpc_async.py``) share this single mapping so the wire encoding is
defined once, not duplicated per transport. The generated stubs are reused
as-is on whatever channel (``grpc`` or ``grpc.aio``) the caller supplies.

Every function takes the generated ``pb2`` module explicitly (it is imported
lazily at the call site) so this module itself imports nothing protobuf at
import time, preserving the SDK's lazy gRPC boundary.
"""

from __future__ import annotations

import json
from typing import Any

from .._record_ids import new_record_id
from ..exceptions import ProximaDBError
from ..models import BatchResult, OperationMetrics, SearchResult

__all__ = [
    "python_to_v2_typed_value",
    "v2_typed_value_to_python",
    "record_proto_for_grpc",
    "v2_record_batch_result",
]


def python_to_v2_typed_value(pb2: Any, value: Any):
    """Encode Python values into v2 ProximaValue/TypedValue protobufs."""
    if pb2 is None:
        raise ProximaDBError("v2 record protobuf stubs not available")

    type_hint = None
    if isinstance(value, dict) and set(value.keys()) == {"type", "value"}:
        type_hint = str(value["type"]).lower()
        value = value["value"]

    tv = pb2.TypedValue()
    if value is None:
        tv.declared_type = pb2.COLUMN_TYPE_UNSPECIFIED
        tv.is_null = True
    elif isinstance(value, bool):
        tv.declared_type = pb2.BOOLEAN
        tv.boolean_value = value
    elif isinstance(value, int) and not isinstance(value, bool):
        tv.declared_type = pb2.INTEGER
        tv.integer_value = value
    elif isinstance(value, float):
        tv.declared_type = pb2.FLOAT32 if type_hint == "float32" else pb2.FLOAT
        if type_hint == "float32":
            tv.float32_value = value
        else:
            tv.float_value = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        tv.declared_type = pb2.BINARY
        tv.binary_value = bytes(value)
    elif isinstance(value, str):
        tv.declared_type = pb2.SYMBOL if type_hint == "symbol" else pb2.TEXT
        if type_hint == "symbol":
            tv.symbol_value = value
        else:
            tv.text_value = value
    elif isinstance(value, (list, tuple)):
        tv.declared_type = pb2.ARRAY_ANY
        tv.array_value.values.extend(
            python_to_v2_typed_value(pb2, item) for item in value
        )
    elif isinstance(value, dict):
        tv.declared_type = pb2.JSONB
        tv.jsonb_value = json.dumps(value, separators=(",", ":")).encode("utf-8")
    else:
        tv.declared_type = pb2.TEXT
        tv.text_value = str(value)
    return tv


def v2_typed_value_to_python(value: Any) -> Any:
    """Decode v2 TypedValue protobufs into Python values."""
    which = value.WhichOneof("value")
    if which in (None, "is_null"):
        return None
    if which == "text_value":
        return value.text_value
    if which == "integer_value":
        return value.integer_value
    if which == "float_value":
        return value.float_value
    if which == "boolean_value":
        return value.boolean_value
    if which == "timestamp_value":
        return value.timestamp_value
    if which == "date_value":
        return value.date_value
    if which == "time_value":
        return value.time_value
    if which == "duration_value":
        return value.duration_value
    if which == "uuid_value":
        return bytes(value.uuid_value).hex()
    if which == "binary_value":
        return bytes(value.binary_value)
    if which == "json_value":
        try:
            return json.loads(value.json_value)
        except json.JSONDecodeError:
            return value.json_value
    if which == "jsonb_value":
        try:
            return json.loads(bytes(value.jsonb_value).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return bytes(value.jsonb_value)
    if which == "array_value":
        return [v2_typed_value_to_python(item) for item in value.array_value.values]
    if which == "map_value":
        return {
            key: v2_typed_value_to_python(item)
            for key, item in value.map_value.entries.items()
        }
    if which == "struct_value":
        return {
            key: v2_typed_value_to_python(item)
            for key, item in value.struct_value.entries.items()
        }
    if which == "float32_value":
        return value.float32_value
    if which.endswith("_array"):
        return list(getattr(value, which).values)
    return getattr(value, which)


def record_proto_for_grpc(pb2: Any, record: Any, index: int = 0):
    """Build a v2 ``ProximaRecord`` proto from an SDK record-like input."""
    if pb2 is None:
        raise ProximaDBError("v2 record protobuf stubs not available")

    if hasattr(record, "model_dump"):
        record = record.model_dump(exclude_none=True)
    elif hasattr(record, "dict"):
        record = record.dict(exclude_none=True)
    if not isinstance(record, dict):
        raise TypeError(f"Unsupported record input: {type(record)!r}")

    vector = record.get("vector")
    if vector is None and record.get("embeddings"):
        first_embedding = record["embeddings"][0]
        vector = (
            first_embedding.get("values")
            if isinstance(first_embedding, dict)
            else first_embedding
        )
    if vector is None:
        raise ValueError("record is missing vector")

    proto = pb2.ProximaRecord()
    proto.id = str(record.get("id") or record.get("oid") or new_record_id())
    proto.vector.extend(float(v) for v in vector)
    if record.get("vector_dimension") is not None:
        proto.vector_dimension = int(record["vector_dimension"])

    for source in ("props", "metadata", "flexible_fields"):
        values = record.get(source)
        if isinstance(values, dict):
            for key, value in values.items():
                proto.props[str(key)].CopyFrom(python_to_v2_typed_value(pb2, value))

    typed_fields = record.get("typed_fields")
    if isinstance(typed_fields, dict):
        for key, value in typed_fields.items():
            if hasattr(value, "model_dump"):
                value = value.model_dump(exclude_none=True)
            if isinstance(value, dict) and "value" in value:
                value = {
                    "type": value.get("value_type") or value.get("type"),
                    "value": value["value"],
                }
            proto.props[str(key)].CopyFrom(python_to_v2_typed_value(pb2, value))

    for text_field in record.get("text_fields") or []:
        if hasattr(text_field, "model_dump"):
            text_field = text_field.model_dump(exclude_none=True)
        if isinstance(text_field, dict):
            proto.text_fields.add(
                name=str(text_field.get("name") or ""),
                content=str(text_field.get("content") or ""),
                storage_hint=str(text_field.get("storage_hint") or ""),
                chunk_count=int(text_field.get("chunk_count") or 0),
                chunk_reference=str(text_field.get("chunk_reference") or ""),
            )

    if record.get("timestamp_ms") is not None:
        proto.timestamp_ms = int(record["timestamp_ms"])
    for field in (
        "updated_at_ms",
        "expires_at_ms",
        "version",
        "source",
        "source_type",
        "schema_id",
        "partition_key",
        "created_by",
        "updated_by",
    ):
        if record.get(field) is not None:
            setattr(proto, field, record[field])
    if isinstance(record.get("partition_values"), dict):
        proto.partition_values.update(
            {str(k): str(v) for k, v in record["partition_values"].items()}
        )
    if isinstance(record.get("custom_metadata"), dict):
        proto.custom_metadata.update(
            {str(k): str(v) for k, v in record["custom_metadata"].items()}
        )
    return proto


def v2_record_batch_result(response) -> BatchResult:
    """Parse a v2 ``ProximaRecordBatchResult`` proto into a ``BatchResult``."""
    errors = [
        f"{error.record_id or error.record_index}: {error.error_message}"
        for error in response.errors
    ]
    return BatchResult(
        total=int(response.total_processed),
        success=int(response.success_count),
        failed=int(response.failed_count),
        errors=errors,
        # the client requests return_ids=True; the proto's inserted_ids
        # (field 5) carries the server-assigned per-record IDs.
        inserted_ids=list(getattr(response, "inserted_ids", []) or []),
        metrics=OperationMetrics(
            total_processed=int(response.total_processed),
            successful_count=int(response.success_count),
            failed_count=int(response.failed_count),
            processing_time_us=int(response.processing_time_us),
        ),
    )


def search_results_from_proto(
    response,
    *,
    include_vectors: bool,
    include_metadata: bool,
) -> list[SearchResult]:
    """Parse a v2 Search response's results into SDK ``SearchResult`` rows.

    Mirrors the per-result mapping in ``grpc_sync.search_vectors`` so both
    transports return identical ``SearchResult`` shapes.
    """
    results: list[SearchResult] = []
    for rank, result in enumerate(response.results):
        metadata = None
        if include_metadata:
            metadata = {
                key: v2_typed_value_to_python(value)
                for key, value in result.props.items()
            }
        results.append(
            SearchResult(
                id=result.id,
                score=result.score,
                rank=rank,
                vector=(
                    list(result.vector) if include_vectors and result.vector else None
                ),
                metadata=metadata,
                timestamp=(
                    result.timestamp_ms if result.HasField("timestamp_ms") else None
                ),
                version=(result.version if result.HasField("version") else None),
                source=(result.source if result.HasField("source") else None),
            )
        )
    return results
