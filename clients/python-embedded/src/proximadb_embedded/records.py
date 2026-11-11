"""Modern ProximaRecord normalization helpers for embedded Python.

These helpers are the Python-side normalizer for the canonical
ProximaRecord/ProximaValue contract. They intentionally return the same
JSON-friendly shape accepted by the v2 REST record endpoint:

    {"id": "...", "vector": [...], "props": {...}, "text_fields": [...]}

Legacy VectorRecord naming is not used here. Vector-only helpers can call these
functions while they are being retired.
"""

from __future__ import annotations
import uuid

from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


RecordJson = dict[str, Any]


@dataclass(frozen=True)
class ProximaValue:
    """Explicit typed value wrapper for embedded record props."""

    type: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "value": _json_scalar(self.value)}


@dataclass(frozen=True)
class ProximaRecord:
    """Canonical embedded SDK record payload.

    The native extension still accepts the current low-level
    `(ids, vectors, metadata)` batch boundary. This type is the public Python
    shape; module helpers lower it into that boundary until the Rust embedded
    FFI accepts full records directly.
    """

    id: str
    vector: Sequence[float]
    props: Mapping[str, Any] = field(default_factory=dict)
    text_fields: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    source: str | None = None
    schema_id: str | None = None

    def to_dict(self) -> RecordJson:
        record: RecordJson = {
            "id": self.id,
            "vector": list(self.vector),
            "props": dict(self.props),
        }
        if self.text_fields:
            record["text_fields"] = [dict(field) for field in self.text_fields]
        if self.source is not None:
            record["source"] = self.source
        if self.schema_id is not None:
            record["schema_id"] = self.schema_id
        return record


def proxima_value(type_name: str, value: Any) -> ProximaValue:
    """Build an explicit typed ProximaValue for record props."""
    return ProximaValue(type=type_name, value=value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Decimal):
        return {"type": "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {"type": "timestamp_tz", "value": value.isoformat()}
    if isinstance(value, date) and not isinstance(value, datetime):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, time):
        return {"type": "time", "value": value.isoformat()}
    return value


def _proxima_value(value: Any, type_name: str | None = None) -> Any:
    if isinstance(value, ProximaValue):
        return value.to_dict()
    value = _json_scalar(value)
    if type_name is not None:
        return {"type": type_name, "value": value}
    if isinstance(value, Mapping) and set(value.keys()) == {"type", "value"}:
        return dict(value)
    if isinstance(value, Mapping):
        return {"type": "jsonb", "value": {str(k): _json_scalar(v) for k, v in value.items()}}
    if isinstance(value, tuple):
        return {"type": "array", "value": [_json_scalar(v) for v in value]}
    return value


def _props_from_mapping(
    values: Mapping[str, Any] | None,
    *,
    typed_columns: Mapping[str, str] | None = None,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    if not values:
        return {}

    exclude = exclude or set()
    typed_columns = typed_columns or {}
    props: dict[str, Any] = {}
    for key, value in values.items():
        key = str(key)
        if key in exclude:
            continue
        props[key] = _proxima_value(value, typed_columns.get(key))
    return props


def _vector_list(vector: Any) -> list[float]:
    if isinstance(vector, np.ndarray):
        values = vector.astype(np.float32, copy=False).tolist()
        if not values:
            raise ValueError("record vector must not be empty")
        return values
    values = list(vector)
    if not values:
        raise ValueError("record vector must not be empty")
    if all(isinstance(v, float) for v in values):
        return values
    return [float(v) for v in values]


def normalize_record(
    record: Mapping[str, Any] | ProximaRecord,
    *,
    id_field: str = "id",
    vector_field: str = "vector",
    props_field: str = "props",
    text_columns: Sequence[str] | None = None,
    typed_columns: Mapping[str, str] | None = None,
    modality: str | None = None,
) -> RecordJson:
    """Normalize one mapping into v2 ProximaRecord JSON."""

    if isinstance(record, ProximaRecord):
        props = _props_from_mapping(record.props, typed_columns=typed_columns)
        if modality is not None:
            props.setdefault("_modality", modality)
        normalized: RecordJson = {
            "id": str(record.id),
            "vector": _vector_list(record.vector),
            "props": props,
        }
        if record.text_fields:
            normalized["text_fields"] = [dict(field) for field in record.text_fields]
        if record.source is not None:
            normalized["source"] = record.source
        if record.schema_id is not None:
            normalized["schema_id"] = record.schema_id
        return normalized

    record_id = record.get(id_field) or record.get("oid")
    vector = record.get(vector_field)
    if vector is None and "embeddings" in record:
        embeddings = record["embeddings"]
        if embeddings:
            first = embeddings[0]
            vector = first.get("values") if isinstance(first, Mapping) else first
    if vector is None:
        raise ValueError(f"record is missing vector field {vector_field!r}")

    text_columns = tuple(text_columns or ())
    explicit_props = record.get(props_field)
    legacy_metadata = record.get("metadata")
    flexible_fields = record.get("flexible_fields")
    exclude = {
        id_field,
        "oid",
        vector_field,
        "embeddings",
        props_field,
        "metadata",
        "flexible_fields",
        "text_fields",
        "source",
        "schema_id",
    }
    exclude.update(text_columns)

    props = _props_from_mapping(record, typed_columns=typed_columns, exclude=exclude)
    for prop_source in (legacy_metadata, flexible_fields, explicit_props):
        if isinstance(prop_source, Mapping):
            props.update(_props_from_mapping(prop_source, typed_columns=typed_columns))
    if modality is not None:
        props.setdefault("_modality", modality)

    text_fields = list(record.get("text_fields") or [])
    for column in text_columns:
        if column in record and record[column] is not None:
            text_fields.append(
                {
                    "name": column,
                    "content": str(record[column]),
                    "storage_hint": "adaptive",
                }
            )

    normalized: RecordJson = {
        "id": str(record_id) if record_id is not None else None,
        "vector": _vector_list(vector),
        "props": props,
    }
    if text_fields:
        normalized["text_fields"] = text_fields
    if record.get("source") is not None:
        normalized["source"] = str(record["source"])
    if record.get("schema_id") is not None:
        normalized["schema_id"] = str(record["schema_id"])
    return {k: v for k, v in normalized.items() if v is not None}


def normalize_records(
    source: Any = None,
    *,
    ids: Sequence[str] | None = None,
    vectors: Any = None,
    props: Sequence[Mapping[str, Any]] | None = None,
    id_field: str = "id",
    vector_field: str = "vector",
    text_columns: Sequence[str] | None = None,
    typed_columns: Mapping[str, str] | None = None,
    modality: str | None = None,
) -> list[RecordJson]:
    """Normalize Python inputs into v2 ProximaRecord JSON.

    Supported inputs:
    - one mapping or a list of mappings;
    - NumPy/list vector matrices plus optional ids/props;
    - pandas DataFrame;
    - pyarrow Table or RecordBatch.
    """

    if vectors is not None:
        source = vectors
    if source is None:
        raise ValueError("source or vectors is required")

    if hasattr(source, "to_pylist"):
        return [
            normalize_record(
                row,
                id_field=id_field,
                vector_field=vector_field,
                text_columns=text_columns,
                typed_columns=typed_columns,
                modality=modality,
            )
            for row in source.to_pylist()
        ]

    if hasattr(source, "to_dict") and source.__class__.__name__ == "DataFrame":
        rows = source.to_dict(orient="records")
        return [
            normalize_record(
                row,
                id_field=id_field,
                vector_field=vector_field,
                text_columns=text_columns,
                typed_columns=typed_columns,
                modality=modality,
            )
            for row in rows
        ]

    if isinstance(source, ProximaRecord):
        return [
            normalize_record(
                source,
                id_field=id_field,
                vector_field=vector_field,
                text_columns=text_columns,
                typed_columns=typed_columns,
                modality=modality,
            )
        ]

    if isinstance(source, Mapping):
        return [
            normalize_record(
                source,
                id_field=id_field,
                vector_field=vector_field,
                text_columns=text_columns,
                typed_columns=typed_columns,
                modality=modality,
            )
        ]

    if isinstance(source, np.ndarray) or _is_vector_matrix(source):
        rows = np.asarray(source, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        ids = ids or [str(uuid.uuid4()) for _ in range(rows.shape[0])]
        props = props or [{} for _ in range(rows.shape[0])]
        if len(ids) != rows.shape[0] or len(props) != rows.shape[0]:
            raise ValueError("ids and props must match the number of records")
        return [
            normalize_record(
                {"id": record_id, "vector": rows[i], "props": record_props},
                typed_columns=typed_columns,
                modality=modality,
            )
            for i, (record_id, record_props) in enumerate(zip(ids, props))
        ]

    if isinstance(source, Iterable):
        return [
            normalize_record(
                row,
                id_field=id_field,
                vector_field=vector_field,
                text_columns=text_columns,
                typed_columns=typed_columns,
                modality=modality,
            )
            for row in source
        ]

    raise TypeError(f"Unsupported record source: {type(source)!r}")


def _is_vector_matrix(source: Any) -> bool:
    return (
        isinstance(source, Sequence)
        and not isinstance(source, (str, bytes, bytearray))
        and bool(source)
        and isinstance(source[0], Sequence)
    )


def normalize_document(
    document_id: str,
    document: Mapping[str, Any],
    vector: Any,
    *,
    text_columns: Sequence[str] | None = None,
) -> ProximaRecord:
    """Normalize a document facade write into a canonical record shape."""

    text_fields = []
    for column in text_columns or ():
        if column in document and document[column] is not None:
            text_fields.append(
                {
                    "name": column,
                    "content": str(document[column]),
                    "storage_hint": "adaptive",
                }
            )

    return ProximaRecord(
        id=document_id,
        vector=vector,
        props={"_modality": "document", **dict(document)},
        text_fields=text_fields,
    )


def normalize_graph_node(
    node_id: str,
    labels: Sequence[str],
    properties: Mapping[str, Any],
    vector: Any,
) -> ProximaRecord:
    """Normalize a graph node facade write into a canonical record shape."""

    return ProximaRecord(
        id=node_id,
        vector=vector,
        props={"_modality": "graph_node", "labels": list(labels), **dict(properties)},
    )


def normalize_observability_event(
    event_id: str,
    fields: Mapping[str, Any],
    vector: Any,
    *,
    event_type: str,
) -> ProximaRecord:
    """Normalize log/metric/span-like input into a canonical record shape."""

    return ProximaRecord(
        id=event_id,
        vector=vector,
        props={"_modality": "observability", "event_type": event_type, **dict(fields)},
    )
