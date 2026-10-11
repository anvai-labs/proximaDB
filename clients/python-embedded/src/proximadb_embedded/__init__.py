"""
ProximaDB Embedded - In-process vector database for Python

This package provides a zero-overhead embedded mode for ProximaDB,
enabling direct in-process access to the high-performance Rust core
without network communication.

Features:
- Zero network overhead - direct in-process calls
- Multi-disk support with weighted distribution
- SIMD-accelerated vector operations
- Full WAL persistence and crash recovery
- NumPy zero-copy vector transfer

Example:
    >>> from proximadb_embedded import ProximaDB, DiskConfig
    >>> import numpy as np
    >>>
    >>> # Configure multi-disk storage
    >>> disks = [
    ...     DiskConfig("/nvme/data", weight=2),  # Fast SSD
    ...     DiskConfig("/hdd/data", weight=1),   # Slower HDD
    ... ]
    >>>
    >>> # Create embedded database
    >>> db = ProximaDB(data_dirs=disks, cache_size_mb=1024)
    >>>
    >>> # Create collection
    >>> db.create_collection("embeddings", dimension=768)
    >>>
    >>> # Insert vectors (direct NumPy support)
    >>> vectors = np.random.rand(1000, 768).astype(np.float32)
    >>> db.insert("embeddings", ids=[f"v{i}" for i in range(1000)], vectors=vectors)
    >>>
    >>> # Search
    >>> results = db.search("embeddings", query=vectors[0], top_k=10)
"""

from ._proximadb_embedded import (
    CollectionInfo,
    DiskConfig,
    GraphEdge,
    GraphNode,
    GraphStats,
    ProximaDB,
    SearchResult,
    SearchStreamIterator,
    StorageStats,
    StreamingSearchResult,
    init_logging,
    # DataFusion DataFrame API
    PyDataFrame as DataFrame,
    PyDataFusionSession as DataFusionSession,
    PyExpr as Expr,
    py_col as col,
    py_lit as lit,
    py_count as count,
    py_sum as sum,
    py_avg as avg,
    py_min as min,
    py_max as max,
)
from .records import (
    ProximaRecord,
    ProximaValue,
    normalize_document,
    normalize_graph_node,
    normalize_observability_event,
    normalize_record,
    normalize_records,
    proxima_value,
)
from .notebook import (
    Column,
    GroupedProximaFrame,
    Predicate,
    ProximaFrame,
    ProximaSession,
    ProximaSessionBuilder,
)
import time

__version__ = "0.4.0"
__all__ = [
    "ProximaDB",
    "DiskConfig",
    "SearchResult",
    "StreamingSearchResult",
    "SearchStreamIterator",
    "CollectionInfo",
    "StorageStats",
    "GraphNode",
    "GraphEdge",
    "GraphStats",
    "init_logging",
    "insert_arrow",
    "upsert_arrow",
    "insert_pandas",
    "insert_records",
    "insert_records_profiled",
    "insert_proxima_records",
    "upsert_records",
    "upsert_proxima_records",
    "ProximaRecord",
    "ProximaValue",
    "proxima_value",
    "normalize_record",
    "normalize_records",
    "normalize_document",
    "normalize_graph_node",
    "normalize_observability_event",
    "profile_record_batch_parts",
    "Column",
    "GroupedProximaFrame",
    "Predicate",
    "ProximaFrame",
    "ProximaSession",
    "ProximaSessionBuilder",
    "DataFrame",
    "DataFusionSession",
    "Expr",
    "col",
    "lit",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "__version__",
]


def open(
    path: str = "./data",
    cache_size_mb: int = 512,
    default_engine: str = "sst",
) -> ProximaDB:
    """
    Open an embedded ProximaDB database with simple configuration.

    This is a convenience function for quick setup. For multi-disk
    configuration, use the ProximaDB class directly.

    Args:
        path: Path to the data directory
        cache_size_mb: Cache size in megabytes
        default_engine: Default storage engine type

    Returns:
        ProximaDB instance

    Example:
        >>> import proximadb_embedded as pdb
        >>> db = pdb.open("./my_database")
        >>> db.create_collection("vectors", dimension=128)
    """
    return ProximaDB(
        data_dirs=path,
        cache_size_mb=cache_size_mb,
        default_engine=default_engine,
    )


def _arrow_source_to_ipc_bytes(source) -> bytes:
    """Convert pyarrow/pandas/IPC-byte inputs to Arrow IPC stream bytes."""
    if isinstance(source, bytes):
        return source
    if isinstance(source, bytearray):
        return bytes(source)
    if isinstance(source, memoryview):
        return source.tobytes()

    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise ImportError(
            "insert_arrow/insert_pandas require pyarrow. Install with "
            "`pip install proximadb_embedded[arrow]` or pass Arrow IPC bytes."
        ) from exc

    if isinstance(source, pa.RecordBatch):
        table = pa.Table.from_batches([source])
    elif isinstance(source, pa.Table):
        table = source
    else:
        try:
            table = pa.Table.from_pandas(source, preserve_index=False)
        except Exception as exc:
            raise TypeError(
                "Expected pyarrow.Table, pyarrow.RecordBatch, pandas.DataFrame, "
                "or Arrow IPC bytes"
            ) from exc

    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _arrow_source_to_batches(source):
    """Convert a pyarrow Table/RecordBatch or pandas DataFrame to a list of pyarrow
    RecordBatches for the **zero-copy** Arrow C Data Interface hand-off. Returns
    ``None`` for already-serialized inputs (raw IPC bytes), which must take the IPC
    path because their buffers are not live Arrow arrays."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return None

    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ImportError(
            "insert_arrow/insert_pandas require pyarrow. Install with "
            "`pip install proximadb_embedded[arrow]` or pass Arrow IPC bytes."
        ) from exc

    if isinstance(source, pa.RecordBatch):
        return [source]
    if isinstance(source, pa.Table):
        return source.to_batches()
    try:
        return pa.Table.from_pandas(source, preserve_index=False).to_batches()
    except Exception as exc:
        raise TypeError(
            "Expected pyarrow.Table, pyarrow.RecordBatch, pandas.DataFrame, "
            "or Arrow IPC bytes"
        ) from exc


def insert_arrow(
    db: ProximaDB,
    collection: str,
    source,
    *,
    mode: str = "insert",
    tenant_id=None,
) -> int:
    """Insert/upsert a pyarrow Table/RecordBatch, pandas DataFrame, or IPC bytes.

    Live Arrow inputs cross the FFI boundary zero-copy via the Arrow C Data
    Interface (``insert_arrow_batches``); only pre-serialized IPC bytes take the
    legacy IPC path (``insert_arrow_ipc``)."""
    batches = _arrow_source_to_batches(source)
    if batches is None:
        return db.insert_arrow_ipc(
            collection,
            _arrow_source_to_ipc_bytes(source),
            mode,
            tenant_id,
        )
    return db.insert_arrow_batches(collection, batches, mode, tenant_id)


def upsert_arrow(
    db: ProximaDB,
    collection: str,
    source,
    *,
    tenant_id=None,
) -> int:
    """Upsert a pyarrow Table/RecordBatch, pandas DataFrame, or IPC bytes (zero-copy
    Arrow C Data Interface for live Arrow inputs)."""
    batches = _arrow_source_to_batches(source)
    if batches is None:
        return db.upsert_arrow_ipc(
            collection,
            _arrow_source_to_ipc_bytes(source),
            tenant_id,
        )
    return db.insert_arrow_batches(collection, batches, "upsert", tenant_id)


def insert_pandas(
    db: ProximaDB,
    collection: str,
    dataframe,
    *,
    mode: str = "insert",
    tenant_id=None,
) -> int:
    """Insert/upsert a pandas DataFrame through the embedded Arrow batch path."""
    return insert_arrow(
        db,
        collection,
        dataframe,
        mode=mode,
        tenant_id=tenant_id,
    )


def _record_batch_parts(
    records,
    **normalize_kwargs,
) -> tuple[list[str], list[list[float]], list[dict]]:
    fast_parts = _proxima_record_batch_parts(records, normalize_kwargs)
    if fast_parts is not None:
        return fast_parts

    normalized = normalize_records(records, **normalize_kwargs)
    ids = [record.get("id") or f"record_{index}" for index, record in enumerate(normalized)]
    vectors = [record["vector"] for record in normalized]
    props = []
    for record in normalized:
        record_props = dict(record.get("props", {}))
        if record.get("text_fields"):
            record_props.setdefault("_text_fields", record["text_fields"])
        if record.get("source") is not None:
            record_props.setdefault("_source", record["source"])
        if record.get("schema_id") is not None:
            record_props.setdefault("_schema_id", record["schema_id"])
        props.append(record_props)
    return ids, vectors, props


def _default_record_normalization(normalize_kwargs):
    return (
        normalize_kwargs.get("id_field", "id") == "id"
        and normalize_kwargs.get("vector_field", "vector") == "vector"
        and not normalize_kwargs.get("text_columns")
        and not normalize_kwargs.get("typed_columns")
        and normalize_kwargs.get("modality") is None
    )


def _vector_values(vector):
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and isinstance(vector, np.ndarray):
        values = vector.astype(np.float32, copy=False).tolist()
    else:
        values = list(vector)
        if not all(isinstance(value, float) for value in values):
            values = [float(value) for value in values]
    if not values:
        raise ValueError("record vector must not be empty")
    return values


def _proxima_record_numpy_batch_parts(records, normalize_kwargs):
    if not _default_record_normalization(normalize_kwargs):
        return None

    try:
        import numpy as np
    except ImportError:
        return None

    if isinstance(records, ProximaRecord):
        records = [records]
    elif isinstance(records, (str, bytes, bytearray)):
        return None
    else:
        try:
            records = list(records)
        except TypeError:
            return None

    if not records or not all(isinstance(record, ProximaRecord) for record in records):
        return None

    vectors = [record.vector for record in records]
    if not all(isinstance(vector, np.ndarray) and vector.ndim == 1 for vector in vectors):
        return None

    dimensions = {vector.shape[0] for vector in vectors}
    if len(dimensions) != 1 or next(iter(dimensions)) == 0:
        return None

    matrix = np.asarray(vectors, dtype=np.float32)
    if not matrix.flags.c_contiguous:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    ids = [record.id for record in records]
    props = []
    for record in records:
        record_props = dict(record.props)
        if record.text_fields:
            record_props.setdefault("_text_fields", [dict(field) for field in record.text_fields])
        if record.source is not None:
            record_props.setdefault("_source", record.source)
        if record.schema_id is not None:
            record_props.setdefault("_schema_id", record.schema_id)
        props.append(record_props)
    return ids, matrix, props


def _proxima_record_batch_parts(records, normalize_kwargs):
    if not _default_record_normalization(normalize_kwargs):
        return None

    if isinstance(records, ProximaRecord):
        records = [records]
    elif isinstance(records, (str, bytes, bytearray)):
        return None
    else:
        try:
            records = list(records)
        except TypeError:
            return None

    if not records or not all(isinstance(record, ProximaRecord) for record in records):
        return None

    ids = [record.id for record in records]
    vectors = [_vector_values(record.vector) for record in records]
    props = []
    for record in records:
        record_props = dict(record.props)
        if record.text_fields:
            record_props.setdefault("_text_fields", [dict(field) for field in record.text_fields])
        if record.source is not None:
            record_props.setdefault("_source", record.source)
        if record.schema_id is not None:
            record_props.setdefault("_schema_id", record.schema_id)
        props.append(record_props)
    return ids, vectors, props


def _native_record_batch(records, normalize_kwargs):
    if isinstance(records, ProximaRecord):
        return [records], "proxima_record_native"
    if isinstance(records, (str, bytes, bytearray)):
        return None
    try:
        records = list(records)
    except TypeError:
        return None

    if not records:
        return None
    if all(isinstance(record, ProximaRecord) for record in records):
        return records, "proxima_record_native"
    if all(isinstance(record, dict) for record in records):
        return normalize_records(records, **normalize_kwargs), "normalized_native"
    return None


def profile_record_batch_parts(records, **normalize_kwargs):
    """Return batch parts plus timing counters for Python-side record lowering."""
    started = time.perf_counter()
    fast_path = _proxima_record_batch_parts(records, normalize_kwargs)
    if fast_path is not None:
        elapsed = time.perf_counter() - started
        ids, vectors, props = fast_path
        return ids, vectors, props, {
            "path": "proxima_record_fast",
            "records": len(ids),
            "total_seconds": elapsed,
            "normalize_seconds": 0.0,
            "batch_parts_seconds": elapsed,
        }

    normalized_started = time.perf_counter()
    normalized = normalize_records(records, **normalize_kwargs)
    normalized_elapsed = time.perf_counter() - normalized_started

    parts_started = time.perf_counter()
    ids = [record.get("id") or f"record_{index}" for index, record in enumerate(normalized)]
    vectors = [record["vector"] for record in normalized]
    props = []
    for record in normalized:
        record_props = dict(record.get("props", {}))
        if record.get("text_fields"):
            record_props.setdefault("_text_fields", record["text_fields"])
        if record.get("source") is not None:
            record_props.setdefault("_source", record["source"])
        if record.get("schema_id") is not None:
            record_props.setdefault("_schema_id", record["schema_id"])
        props.append(record_props)
    parts_elapsed = time.perf_counter() - parts_started
    total_elapsed = time.perf_counter() - started
    return ids, vectors, props, {
        "path": "normalized",
        "records": len(ids),
        "total_seconds": total_elapsed,
        "normalize_seconds": normalized_elapsed,
        "batch_parts_seconds": parts_elapsed,
    }


def _insert_batch(db, collection, ids, vectors, props, *, prefer_numpy=False):
    if prefer_numpy and hasattr(db, "insert_numpy"):
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None and isinstance(vectors, np.ndarray):
            return db.insert_numpy(collection, ids, vectors, props)
    return db.insert(collection, ids, vectors, props)


def _native_record_batch_parts(records, normalize_kwargs):
    numpy_parts = _proxima_record_numpy_batch_parts(records, normalize_kwargs)
    if numpy_parts is None:
        return None
    ids, vectors, props = numpy_parts
    return ids, vectors, props


def _insert_native_record_batch(db, collection, ids, vectors, props):
    if not hasattr(db, "_insert_proxima_record_batch_native"):
        return None
    return db._insert_proxima_record_batch_native(collection, ids, vectors, props)


def _insert_native_record_batch_profiled(db, collection, ids, vectors, props):
    if not hasattr(db, "_insert_proxima_record_batch_native_profiled"):
        return None
    return db._insert_proxima_record_batch_native_profiled(collection, ids, vectors, props)


def insert_records_profiled(
    db: ProximaDB,
    collection: str,
    records,
    **normalize_kwargs,
) -> tuple[int, dict]:
    """Insert records and return Python lowering/native-call timing counters."""
    started = time.perf_counter()
    if hasattr(db, "_insert_proxima_record_batch_native"):
        native_batch = _native_record_batch_parts(records, normalize_kwargs)
    else:
        native_batch = None
    if native_batch is not None:
        ids, vectors, props = native_batch
        lowering_elapsed = time.perf_counter() - started
        native_started = time.perf_counter()
        profiled_result = _insert_native_record_batch_profiled(db, collection, ids, vectors, props)
        if profiled_result is not None:
            result, native_profile = profiled_result
        else:
            result = _insert_native_record_batch(db, collection, ids, vectors, props)
            native_profile = {}
        native_elapsed = time.perf_counter() - native_started
        profile = {
            "path": "proxima_record_batch_native",
            "records": len(ids),
            "total_seconds": lowering_elapsed,
            "normalize_seconds": 0.0,
            "batch_parts_seconds": lowering_elapsed,
            "native_insert_seconds": native_elapsed,
            "total_insert_seconds": lowering_elapsed + native_elapsed,
        }
        profile.update(native_profile)
        return result, profile

    native_records = None
    if hasattr(db, "_insert_proxima_records_native"):
        native_records = _native_record_batch(records, normalize_kwargs)
    if native_records is not None:
        record_batch, path = native_records
        lowering_elapsed = time.perf_counter() - started
        native_started = time.perf_counter()
        result = db._insert_proxima_records_native(collection, record_batch)
        native_elapsed = time.perf_counter() - native_started
        profile = {
            "path": path,
            "records": len(record_batch),
            "total_seconds": lowering_elapsed,
            "normalize_seconds": lowering_elapsed if path == "normalized_native" else 0.0,
            "batch_parts_seconds": 0.0 if path == "normalized_native" else lowering_elapsed,
            "native_insert_seconds": native_elapsed,
            "total_insert_seconds": lowering_elapsed + native_elapsed,
        }
        return result, profile

    numpy_parts = _proxima_record_numpy_batch_parts(records, normalize_kwargs)
    if numpy_parts is not None and hasattr(db, "insert_numpy"):
        lowering_elapsed = time.perf_counter() - started
        ids, vectors, props = numpy_parts
        profile = {
            "path": "proxima_record_numpy",
            "records": len(ids),
            "total_seconds": lowering_elapsed,
            "normalize_seconds": 0.0,
            "batch_parts_seconds": lowering_elapsed,
        }
    else:
        ids, vectors, props, profile = profile_record_batch_parts(records, **normalize_kwargs)
    native_started = time.perf_counter()
    result = _insert_batch(
        db,
        collection,
        ids,
        vectors,
        props,
        prefer_numpy=profile["path"] == "proxima_record_numpy",
    )
    native_elapsed = time.perf_counter() - native_started
    profile["native_insert_seconds"] = native_elapsed
    profile["total_insert_seconds"] = profile["total_seconds"] + native_elapsed
    return result, profile


def insert_records(
    db: ProximaDB,
    collection: str,
    records,
    **normalize_kwargs,
) -> int:
    """Insert records through the canonical ProximaRecord normalization path."""
    if hasattr(db, "_insert_proxima_record_batch_native"):
        native_batch = _native_record_batch_parts(records, normalize_kwargs)
        if native_batch is not None:
            ids, vectors, props = native_batch
            return _insert_native_record_batch(db, collection, ids, vectors, props)

    if hasattr(db, "_insert_proxima_records_native"):
        native_records = _native_record_batch(records, normalize_kwargs)
        if native_records is not None:
            record_batch, _path = native_records
            return db._insert_proxima_records_native(collection, record_batch)

    numpy_parts = _proxima_record_numpy_batch_parts(records, normalize_kwargs)
    if numpy_parts is not None:
        ids, vectors, props = numpy_parts
        return _insert_batch(db, collection, ids, vectors, props, prefer_numpy=True)

    ids, vectors, props = _record_batch_parts(records, **normalize_kwargs)
    return db.insert(collection, ids, vectors, props)


def insert_proxima_records(
    db: ProximaDB,
    collection: str,
    records,
    **normalize_kwargs,
) -> int:
    """Insert canonical ProximaRecord payloads.

    This is an explicit name for callers avoiding legacy vector terminology.
    """
    return insert_records(db, collection, records, **normalize_kwargs)


def upsert_records(
    db: ProximaDB,
    collection: str,
    records,
    **normalize_kwargs,
) -> tuple[int, int]:
    """Upsert records through the canonical ProximaRecord normalization path."""
    ids, vectors, props = _record_batch_parts(records, **normalize_kwargs)
    return db.upsert(collection, ids, vectors, props)


def upsert_proxima_records(
    db: ProximaDB,
    collection: str,
    records,
    **normalize_kwargs,
) -> tuple[int, int]:
    """Upsert canonical ProximaRecord payloads."""
    return upsert_records(db, collection, records, **normalize_kwargs)


try:
    ProximaDB.insert_arrow = lambda self, collection, source, **kwargs: insert_arrow(
        self,
        collection,
        source,
        **kwargs,
    )
    ProximaDB.upsert_arrow = lambda self, collection, source, **kwargs: upsert_arrow(
        self,
        collection,
        source,
        **kwargs,
    )
    ProximaDB.insert_pandas = lambda self, collection, dataframe, **kwargs: insert_pandas(
        self,
        collection,
        dataframe,
        **kwargs,
    )
    ProximaDB.insert_records = lambda self, collection, records, **kwargs: insert_records(
        self,
        collection,
        records,
        **kwargs,
    )
    ProximaDB.upsert_records = lambda self, collection, records, **kwargs: upsert_records(
        self,
        collection,
        records,
        **kwargs,
    )
    ProximaDB.insert_proxima_records = (
        lambda self, collection, records, **kwargs: insert_proxima_records(
            self,
            collection,
            records,
            **kwargs,
        )
    )
    ProximaDB.upsert_proxima_records = (
        lambda self, collection, records, **kwargs: upsert_proxima_records(
            self,
            collection,
            records,
            **kwargs,
        )
    )
except (AttributeError, TypeError):
    # Some Python extension class builds reject monkey-patching. The module-level
    # helpers above remain available and route to the same native IPC methods.
    pass
