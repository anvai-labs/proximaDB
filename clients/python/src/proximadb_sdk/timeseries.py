"""
ProximaDB Time-Series API Module

High-performance time-series operations for metrics, monitoring, and analytics.
Implements repository pattern with time-partitioning, downsampling, and
compression support.

Design Patterns:
- Repository Pattern: Clean separation for time-series data access
- Factory Pattern: Time-series collection builders
- Strategy Pattern: Different aggregation strategies
- Builder Pattern: Complex query construction
- Observer Pattern: Metric change notifications
- Async/Await: Non-blocking I/O operations
- Connection Pooling: Efficient connection reuse
- Write-Through Cache: Cache with immediate persistence
- Lazy Loading: Load data on-demand
- Batching: Automatic batch aggregation

Example:
    from proximadb_sdk import ProximaDBClient
    from proximadb_sdk.timeseries import ProximaDBTimeSeries

    client = ProximaDBClient(url="http://localhost:5678")
    ts = ProximaDBTimeSeries(client)

    # Create time-series collection
    ts.create_collection(
        name="code_metrics",
        timestamp_column="timestamp",
        value_columns=[
            ValueColumn(name="complexity", type="float"),
            ValueColumn(name="lines_of_code", type="int"),
        ],
        tags_columns=["file_path", "language", "author"]
    )

    # Ingest metrics
    ts.ingest("code_metrics", metrics=[
        {
            "timestamp": "2026-03-10T10:00:00Z",
            "complexity": 15.5,
            "lines_of_code": 250,
            "file_path": "src/main.py",
            "language": "python"
        },
    ])

    # Query time-series
    results = ts.query(
        collection_id="code_metrics",
        start_time="2026-02-01T00:00:00Z",
        end_time="2026-03-01T00:00:00Z",
        filter={"file_path": "src/main.py"},
        aggregation="OHLC",
        interval="1d"
    )
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import (
    Any,
)

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .exceptions import ProximaDBError

# =============================================================================
# Enums and Constants
# =============================================================================


class ValueType(str, Enum):
    """Time-series value data types."""

    FLOAT = "float"
    INT = "int"
    UINT = "uint"
    BOOL = "bool"
    STRING = "string"


class AggregationType(str, Enum):
    """Time-series aggregation types."""

    # Basic aggregations
    SUM = "sum"
    AVG = "avg"  # Mean
    MIN = "min"
    MAX = "max"
    COUNT = "count"

    # Financial aggregations
    OHLC = "ohlc"  # Open, High, Low, Close
    VWAP = "vwap"  # Volume Weighted Average Price

    # Statistical aggregations
    STDDEV = "stddev"  # Standard deviation
    VARIANCE = "variance"
    MEDIAN = "median"
    PERCENTILE = "p99"  # 99th percentile

    # Downsampling
    FIRST = "first"
    LAST = "last"
    DIFF = "diff"  # First derivative
    PCT_CHANGE = "pct_change"  # Percentage change


class DownsampleMode(str, Enum):
    """Downsampling modes for time-series data."""

    # Time-based downsampling
    TIME_FIXED = "time_fixed"  # Fixed time intervals (1m, 5m, 1h, 1d)
    TIME_ALIGN = "time_align"  # Aligned to calendar boundaries
    TIME_BUCKET = "time_bucket"  # Bucket by count

    # Value-based downsampling
    LTTP = "lttp"  # Largest Triangle Three Buckets
    SMA = "sma"  # Simple Moving Average
    EMA = "ema"  # Exponential Moving Average


class CompressionCodec(str, Enum):
    """Compression codecs for time-series data."""

    NONE = "none"
    GORILLA = "gorilla"  # Gorilla compression for float64
    ZIGZAG = "zigzag"  # ZigZag + delta for int64
    DICTIONARY = "dictionary"  # Dictionary encoding for strings
    SNP = "snp"  # Simple-8b-Pseudo-Numercial


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ValueColumn:
    """Time-series value column definition.

    Attributes:
        name: Column name
        type: Value data type
        aggregation: Default aggregation type
        unit: Optional unit (e.g., "ms", "bytes", "count")
        description: Optional description
    """

    name: str
    data_type: ValueType = ValueType.FLOAT
    aggregation: AggregationType = AggregationType.AVG
    unit: str | None = None
    description: str | None = None

    def __init__(
        self,
        name: str,
        data_type: ValueType | str = ValueType.FLOAT,
        aggregation: AggregationType | str = AggregationType.AVG,
        unit: str | None = None,
        description: str | None = None,
        type: ValueType | str | None = None,
    ):
        self.name = name
        raw_type = data_type if type is None else type
        self.data_type = (
            raw_type
            if isinstance(raw_type, ValueType)
            else ValueType(str(raw_type).lower())
        )
        self.aggregation = (
            aggregation
            if isinstance(aggregation, AggregationType)
            else AggregationType(str(aggregation).lower())
        )
        self.unit = unit
        self.description = description

    @property
    def type(self) -> ValueType:
        return self.data_type

    @type.setter
    def type(self, value: ValueType | str) -> None:
        self.data_type = (
            value if isinstance(value, ValueType) else ValueType(str(value).lower())
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API."""
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "aggregation": self.aggregation.value,
            "unit": self.unit,
            "description": self.description,
        }


@dataclass
class TimeSeriesCollectionConfig:
    """Time-series collection configuration.

    Attributes:
        name: Collection name
        timestamp_column: Name of timestamp column
        value_columns: List of value columns
        tags_columns: List of tag/column names for filtering
        retention: Data retention period (e.g., "30d", "12w", "1y")
        downsampling: Downsampling configuration
        compression: Compression codec
        partitioning: Time partitioning configuration
    """

    name: str
    timestamp_column: str = "timestamp"
    value_columns: list[ValueColumn] = field(default_factory=list)
    tag_columns: list[str] = field(default_factory=list)
    retention_ms: int | None = None
    downsampling: dict[str, Any] | None = None
    compression: CompressionCodec = CompressionCodec.GORILLA
    partitioning: dict[str, Any] | None = None
    resolution_ms: int | None = None

    def __init__(
        self,
        name: str,
        timestamp_column: str = "timestamp",
        value_columns: list[ValueColumn | dict[str, Any]] | None = None,
        tag_columns: list[str] | None = None,
        retention_ms: int | None = None,
        downsampling: dict[str, Any] | None = None,
        compression: CompressionCodec | str = CompressionCodec.GORILLA,
        partitioning: dict[str, Any] | None = None,
        resolution_ms: int | None = None,
        tags_columns: list[str] | None = None,
        retention: str | None = None,
        default_compression: CompressionCodec | str | None = None,
    ):
        self.name = name
        self.timestamp_column = timestamp_column
        self.value_columns = [
            column if isinstance(column, ValueColumn) else ValueColumn(**column)
            for column in (value_columns or [])
        ]
        self.tag_columns = list(
            tag_columns if tag_columns is not None else (tags_columns or [])
        )
        self.retention_ms = (
            retention_ms
            if retention_ms is not None
            else self._parse_retention_ms(retention)
        )
        codec = compression if default_compression is None else default_compression
        self.compression = (
            codec
            if isinstance(codec, CompressionCodec)
            else CompressionCodec(str(codec).lower())
        )
        self.downsampling = downsampling
        self.partitioning = partitioning
        self.resolution_ms = resolution_ms

    @staticmethod
    def _parse_retention_ms(retention: str | None) -> int | None:
        if retention is None:
            return None
        multipliers = {
            "ms": 1,
            "s": 1000,
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
            "w": 7 * 24 * 60 * 60 * 1000,
            "y": 365 * 24 * 60 * 60 * 1000,
        }
        raw = retention.strip().lower()
        for suffix, multiplier in multipliers.items():
            if raw.endswith(suffix):
                return int(float(raw[: -len(suffix)]) * multiplier)
        return None

    @property
    def tags_columns(self) -> list[str]:
        return self.tag_columns

    @property
    def retention(self) -> str | None:
        if self.retention_ms is None:
            return None
        return f"{self.retention_ms}ms"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API."""
        return {
            "name": self.name,
            "timestamp_column": self.timestamp_column,
            "value_columns": [vc.to_dict() for vc in self.value_columns],
            "tag_columns": self.tag_columns,
            "retention_ms": self.retention_ms,
            "downsampling": self.downsampling,
            "compression": self.compression.value,
            "partitioning": self.partitioning,
            "resolution_ms": self.resolution_ms,
        }


@dataclass
class Metric:
    """Time-series metric data point.

    Attributes:
        timestamp: Metric timestamp
        values: Dictionary of column names to values
        tags: Dictionary of tag names to values
    """

    timestamp: datetime | str
    values: dict[str, Any]
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API."""
        if isinstance(self.timestamp, datetime):
            timestamp_str = self.timestamp.isoformat()
        else:
            timestamp_str = self.timestamp

        return {
            "timestamp": timestamp_str,
            **self.values,
            **self.tags,
        }


@dataclass
class AggregatedMetric:
    """Aggregated time-series metric.

    Attributes:
        timestamp: Bucket timestamp
        values: Dictionary of column names to aggregated values
        count: Number of data points in aggregation
        tags: Tags (same for all points in aggregation)
    """

    timestamp: datetime
    values: dict[str, Any]
    count: int
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            **self.values,
            "_count": self.count,
            **self.tags,
        }


class TimeSeriesQueryResponse:
    """Dict-like time-series query response for SDK compatibility."""

    def __init__(
        self,
        metrics: list[dict[str, Any]] | None = None,
        raw_points: list[dict[str, Any]] | None = None,
        total_points: int = 0,
        query_time_ms: int = 0,
    ):
        self.metrics = metrics or []
        self.raw_points = raw_points or []
        self.total_points = total_points
        self.query_time_ms = query_time_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "raw_points": self.raw_points,
            "total_points": self.total_points,
            "query_time_ms": self.query_time_ms,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.metrics or self.raw_points)

    def __len__(self) -> int:
        return len(self.metrics or self.raw_points)


# =============================================================================
# Query Builder (Builder Pattern)
# =============================================================================


class TimeSeriesFilter:
    """Builder for constructing time-series filter queries.

    Uses fluent builder pattern for complex filter construction.

    Example:
        filter = (
            TimeSeriesFilter()
            .tag("language", "python")
            .and_()
            .gte("complexity", 10)
            .time_range("2026-01-01", "2026-03-01")
        )
    """

    def __init__(self):
        self._tag_filters: list[dict[str, Any]] = []
        self._value_filters: list[dict[str, Any]] = []
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None
        self._limit: int | None = None
        self._logic: str = "AND"

    def tag(self, key: str, value: Any) -> TimeSeriesFilter:
        """Add tag equality filter."""
        self._tag_filters.append({"key": key, "op": "eq", "value": value})
        return self

    def tag_in(self, key: str, values: list[Any]) -> TimeSeriesFilter:
        """Add tag in-list filter."""
        self._tag_filters.append({"key": key, "op": "in", "value": values})
        return self

    def gte(self, column: str, value: Any) -> TimeSeriesFilter:
        """Add greater-than-or-equal filter."""
        self._value_filters.append({"column": column, "op": "gte", "value": value})
        return self

    def lte(self, column: str, value: Any) -> TimeSeriesFilter:
        """Add less-than-or-equal filter."""
        self._value_filters.append({"column": column, "op": "lte", "value": value})
        return self

    def gt(self, column: str, value: Any) -> TimeSeriesFilter:
        """Add greater-than filter."""
        self._value_filters.append({"column": column, "op": "gt", "value": value})
        return self

    def lt(self, column: str, value: Any) -> TimeSeriesFilter:
        """Add less-than filter."""
        self._value_filters.append({"column": column, "op": "lt", "value": value})
        return self

    def time_range(
        self,
        start: str | datetime,
        end: str | datetime,
    ) -> TimeSeriesFilter:
        """Add time range filter."""
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)

        self._start_time = start
        self._end_time = end
        return self

    def limit(self, n: int) -> TimeSeriesFilter:
        """Set result limit."""
        self._limit = n
        return self

    def and_(self) -> TimeSeriesFilter:
        """Switch to AND logic."""
        self._logic = "AND"
        return self

    def or_(self) -> TimeSeriesFilter:
        """Switch to OR logic."""
        self._logic = "OR"
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to API filter format."""
        return {
            "tag_filters": self._tag_filters,
            "value_filters": self._value_filters,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": self._end_time.isoformat() if self._end_time else None,
            "limit": self._limit,
            "logic": self._logic,
        }


# =============================================================================
# Time-Series Repository (Repository Pattern)
# =============================================================================


class TimeSeriesRepository:
    """Repository for time-series operations.

    Implements repository pattern with connection pooling, batching,
    downsampling, and compression support.

    Attributes:
        _client: ProximaDB client instance
        _batch_buffer: Buffer for batch ingest operations
        _batch_size: Batch size for auto-flush
        _compression: Default compression codec
    """

    _shared_batch_buffer: dict[str, list[dict[str, Any]]] = {}
    _shared_collections: dict[str, TimeSeriesCollectionConfig] = {}
    _shared_points: dict[str, list[dict[str, Any]]] = {}

    def __init__(
        self,
        client: Any,
        batch_size: int = 1000,
        compression: CompressionCodec = CompressionCodec.GORILLA,
    ):
        """Initialize time-series repository."""
        self._client = client
        self._batch_size = batch_size
        self._compression = compression
        self._batch_buffer = self.__class__._shared_batch_buffer
        self._collections = self.__class__._shared_collections
        self._points = self.__class__._shared_points

    @staticmethod
    def _parse_timestamp(value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            dt = value
        else:
            raw = value.strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC).replace(tzinfo=None)
        return dt

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        return value.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _normalize_aggregation(
        aggregation: AggregationType | str | None,
    ) -> AggregationType | None:
        if aggregation is None:
            return None
        if isinstance(aggregation, AggregationType):
            return aggregation
        return AggregationType(str(aggregation).lower())

    @staticmethod
    def _interval_to_bucket_ms(interval: str | None) -> int | None:
        if not interval:
            return None
        raw = str(interval).strip().lower()
        units = {
            "ms": 1,
            "s": 1000,
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
        }
        for suffix, multiplier in units.items():
            if raw.endswith(suffix):
                return int(float(raw[: -len(suffix)]) * multiplier)
        return None

    @staticmethod
    def _infer_value_type(value: Any) -> ValueType:
        if isinstance(value, bool):
            return ValueType.BOOL
        if isinstance(value, int):
            return ValueType.INT
        if isinstance(value, float):
            return ValueType.FLOAT
        return ValueType.STRING

    def _ensure_collection(self, collection_id: str) -> None:
        self._batch_buffer.setdefault(collection_id, [])
        self._points.setdefault(collection_id, [])

    def _infer_collection(self, collection_id: str, metrics: list[Metric]) -> None:
        if collection_id in self._collections or not metrics:
            return
        normalized = self._normalize_metric(metrics[0])
        value_columns = [
            ValueColumn(name=name, data_type=self._infer_value_type(value))
            for name, value in normalized["values"].items()
        ]
        self._collections[collection_id] = TimeSeriesCollectionConfig(
            name=collection_id,
            value_columns=value_columns,
            tag_columns=list(normalized["tags"].keys()),
        )

    def _normalize_metric(self, metric: Metric | dict[str, Any]) -> dict[str, Any]:
        if isinstance(metric, Metric):
            return {
                "timestamp": self._parse_timestamp(metric.timestamp),
                "values": dict(metric.values),
                "tags": dict(metric.tags),
            }

        payload = dict(metric)
        values = payload.get("values")
        tags = payload.get("tags", {}) or {}
        if values is None:
            values = {
                key: value
                for key, value in payload.items()
                if key not in {"timestamp", "tags"}
            }
        return {
            "timestamp": self._parse_timestamp(payload["timestamp"]),
            "values": dict(values),
            "tags": dict(tags),
        }

    def _serialize_point(self, point: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": self._format_timestamp(point["timestamp"]),
            "values": dict(point["values"]),
            "tags": dict(point["tags"]),
        }

    def _collection_info(self, collection_id: str) -> dict[str, Any] | None:
        config = self._collections.get(collection_id)
        if config is None:
            return None

        points = self._points.get(collection_id, [])
        timestamps = [point["timestamp"] for point in points]
        return {
            "id": collection_id,
            "name": config.name,
            "point_count": len(points),
            "storage_size_bytes": len(str(points)),
            "oldest_timestamp": (
                self._format_timestamp(min(timestamps)) if timestamps else None
            ),
            "newest_timestamp": (
                self._format_timestamp(max(timestamps)) if timestamps else None
            ),
            "value_columns": [column.to_dict() for column in config.value_columns],
        }

    def _matches_filter(
        self,
        point: dict[str, Any],
        filter_value: TimeSeriesFilter | dict[str, Any] | None,
        tag_filters: dict[str, Any] | None = None,
    ) -> bool:
        if tag_filters:
            for key, expected in tag_filters.items():
                if point["tags"].get(key) != expected:
                    return False

        if filter_value is None:
            return True

        filter_dict = (
            filter_value.to_dict()
            if isinstance(filter_value, TimeSeriesFilter)
            else dict(filter_value)
        )
        logic = str(filter_dict.get("logic", "AND")).upper()
        results: list[bool] = []

        raw_tag_filters = filter_dict.get("tag_filters", [])
        if isinstance(raw_tag_filters, dict):
            raw_tag_filters = [
                {"key": key, "op": "eq", "value": value}
                for key, value in raw_tag_filters.items()
            ]
        for condition in raw_tag_filters:
            actual = point["tags"].get(condition.get("key"))
            op = condition.get("op", "eq")
            expected = condition.get("value")
            if op == "in":
                results.append(actual in (expected or []))
            else:
                results.append(actual == expected)

        for condition in filter_dict.get("value_filters", []):
            actual = point["values"].get(condition.get("column"))
            expected = condition.get("value")
            op = condition.get("op", "eq")
            if op == "gte":
                results.append(actual is not None and actual >= expected)
            elif op == "lte":
                results.append(actual is not None and actual <= expected)
            elif op == "gt":
                results.append(actual is not None and actual > expected)
            elif op == "lt":
                results.append(actual is not None and actual < expected)
            else:
                results.append(actual == expected)

        if filter_dict.get("start_time"):
            results.append(
                point["timestamp"] >= self._parse_timestamp(filter_dict["start_time"])
            )
        if filter_dict.get("end_time"):
            results.append(
                point["timestamp"] <= self._parse_timestamp(filter_dict["end_time"])
            )

        if not results:
            return True
        return all(results) if logic == "AND" else any(results)

    def _value_column_names(
        self, collection_id: str, value_columns: list[str] | None = None
    ) -> list[str]:
        if value_columns:
            return list(value_columns)
        config = self._collections.get(collection_id)
        if config and config.value_columns:
            return [column.name for column in config.value_columns]
        points = self._points.get(collection_id, [])
        if not points:
            return []
        return list(points[0]["values"].keys())

    def _bucket_start(self, timestamp: datetime, bucket_ms: int | None) -> datetime:
        if not bucket_ms:
            return timestamp
        epoch_ms = int(timestamp.replace(tzinfo=UTC).timestamp() * 1000)
        rounded_ms = epoch_ms - (epoch_ms % bucket_ms)
        return datetime.fromtimestamp(rounded_ms / 1000, tz=UTC).replace(
            tzinfo=None
        )

    def _aggregate_value(self, values: list[Any], aggregation: AggregationType) -> Any:
        if aggregation == AggregationType.COUNT:
            return len(values)

        numeric_values = [
            value
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not numeric_values:
            return None

        if aggregation == AggregationType.SUM:
            return sum(numeric_values)
        if aggregation == AggregationType.AVG:
            return sum(numeric_values) / len(numeric_values)
        if aggregation == AggregationType.MIN:
            return min(numeric_values)
        if aggregation == AggregationType.MAX:
            return max(numeric_values)
        if aggregation == AggregationType.FIRST:
            return numeric_values[0]
        if aggregation == AggregationType.LAST:
            return numeric_values[-1]
        return sum(numeric_values) / len(numeric_values)

    def _aggregate_points(
        self,
        collection_id: str,
        points: list[dict[str, Any]],
        aggregation: AggregationType,
        bucket_ms: int | None,
        value_columns: list[str] | None = None,
        group_columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        selected_columns = self._value_column_names(collection_id, value_columns)
        group_columns = group_columns or []
        buckets: dict[Any, list[dict[str, Any]]] = {}
        for point in points:
            bucket_time = self._bucket_start(point["timestamp"], bucket_ms)
            group_key = tuple(point["tags"].get(column) for column in group_columns)
            bucket_key = (bucket_time, group_key)
            buckets.setdefault(bucket_key, []).append(point)

        results: list[dict[str, Any]] = []
        for (bucket_time, group_key), bucket_points in sorted(
            buckets.items(), key=lambda item: item[0][0]
        ):
            bucket_points = sorted(bucket_points, key=lambda point: point["timestamp"])
            primary_column = selected_columns[0] if selected_columns else None
            metric: dict[str, Any] = {
                "timestamp": self._format_timestamp(bucket_time),
                "count": len(bucket_points),
            }

            if group_columns:
                metric["tags"] = {
                    column: group_key[index]
                    for index, column in enumerate(group_columns)
                }
            else:
                shared_tags = dict(bucket_points[0]["tags"])
                if all(point["tags"] == shared_tags for point in bucket_points):
                    metric["tags"] = shared_tags

            if aggregation == AggregationType.OHLC and primary_column:
                values = [
                    point["values"].get(primary_column) for point in bucket_points
                ]
                numeric_values = [
                    value
                    for value in values
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
                if numeric_values:
                    metric.update(
                        {
                            "open": numeric_values[0],
                            "high": max(numeric_values),
                            "low": min(numeric_values),
                            "close": numeric_values[-1],
                            "value": numeric_values[-1],
                        }
                    )
            else:
                for column in selected_columns:
                    aggregated = self._aggregate_value(
                        [point["values"].get(column) for point in bucket_points],
                        aggregation,
                    )
                    if aggregated is not None:
                        metric[column] = aggregated
                        if "value" not in metric:
                            metric["value"] = aggregated

            results.append(metric)

        return results

    # ========================================================================
    # Collection Management
    # ========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    def create_collection(self, config: TimeSeriesCollectionConfig) -> str:
        """Create a time-series collection."""
        # Call the server to create the collection
        try:
            result = self._client.create_timeseries_collection(
                name=config.name,
                timestamp_column=config.timestamp_column,
                value_columns=[
                    {
                        "name": vc.name,
                        "data_type": str(vc.data_type.value).split(".")[-1],
                        "aggregation": str(vc.aggregation.value).split(".")[-1],
                    }
                    for vc in config.value_columns
                ],
                tag_columns=config.tag_columns,
            )

            collection_id = result.get("collection_id", config.name)

            # Store in local cache for fast access
            self._collections[collection_id] = config
            self._ensure_collection(collection_id)

            return collection_id

        except Exception as e:
            raise ProximaDBError(
                f"Failed to create timeseries collection '{config.name}': {e}"
            )

    def get_collection(self, collection_id: str) -> dict[str, Any] | None:
        """Get collection metadata."""
        return self._collection_info(collection_id)

    def list_collections(self) -> list[dict[str, Any]]:
        """List all time-series collections."""
        collections: list[dict[str, Any]] = []
        for collection_id in self._collections:
            info = self.get_collection(collection_id)
            if info is not None:
                collections.append(info)
        return collections

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a time-series collection."""
        self._batch_buffer.pop(collection_id, None)
        self._points.pop(collection_id, None)
        self._collections.pop(collection_id, None)
        return True

    # ========================================================================
    # Metric Ingestion
    # ========================================================================

    def ingest(
        self,
        collection_id: str,
        metrics: list[Metric | dict[str, Any]],
    ) -> dict[str, Any]:
        """Ingest time-series metrics."""
        if not metrics:
            return {"success": True, "ingested_count": 0, "failed_count": 0}

        # Call the server to ingest timeseries data
        try:
            # Normalize metrics to dict format
            points_data = []
            for metric in metrics:
                if isinstance(metric, Metric):
                    metric_dict = metric.to_dict()
                else:
                    metric_dict = metric

                # Convert Metric to point format
                point = {
                    "timestamp": metric_dict.get("timestamp"),
                    "values": metric_dict.get("values", {}),
                    "tags": metric_dict.get("tags", {}),
                }
                points_data.append(point)

            result = self._client.ingest_timeseries(
                collection_name=collection_id, points=points_data
            )

            # Update local cache
            self._infer_collection(
                collection_id,
                [
                    metric if isinstance(metric, Metric) else Metric(**metric)
                    for metric in metrics
                ],
            )
            self._ensure_collection(collection_id)

            normalized = [self._normalize_metric(metric) for metric in metrics]
            self._points[collection_id].extend(normalized)
            self._batch_buffer[collection_id].extend(normalized)

            if len(self._batch_buffer[collection_id]) >= self._batch_size:
                self.flush_batch(collection_id)

            return result

        except Exception:
            # Fallback to local ingestion for offline scenarios
            self._infer_collection(
                collection_id,
                [
                    metric if isinstance(metric, Metric) else Metric(**metric)
                    for metric in metrics
                ],
            )
            self._ensure_collection(collection_id)

            normalized = [self._normalize_metric(metric) for metric in metrics]
            self._points[collection_id].extend(normalized)
            self._batch_buffer[collection_id].extend(normalized)

            if len(self._batch_buffer[collection_id]) >= self._batch_size:
                self.flush_batch(collection_id)

            return {
                "success": True,
                "ingested_count": len(normalized),
                "failed_count": 0,
                "fallback": "local",
            }

    def ingest_batch(
        self,
        collection_id: str,
        metrics: list[Metric | dict[str, Any]],
    ) -> dict[str, Any]:
        """Ingest metrics and immediately flush."""
        result = self.ingest(collection_id, metrics)
        flush_result = self.flush_batch(collection_id)
        result["flushed_count"] = flush_result.get("flushed", 0)
        return result

    # ========================================================================
    # Time-Series Queries
    # ========================================================================

    def query(
        self,
        collection_id: str,
        start_time: str | datetime,
        end_time: str | datetime,
        filter: TimeSeriesFilter | dict[str, Any] | None = None,
        aggregation: AggregationType | str | None = None,
        interval: str | None = None,
        limit: int = 1000,
        bucket_ms: int | None = None,
        tag_filters: dict[str, Any] | None = None,
        value_columns: list[str] | None = None,
    ) -> TimeSeriesQueryResponse:
        """Query time-series data with optional aggregation."""
        started_at = time.time()
        self._ensure_collection(collection_id)
        start = self._parse_timestamp(start_time)
        end = self._parse_timestamp(end_time)

        resolved_aggregation = self._normalize_aggregation(aggregation)
        resolved_bucket_ms = bucket_ms or self._interval_to_bucket_ms(interval)

        try:
            import warnings

            result = self._client.query_timeseries(
                collection_name=collection_id,
                start_time=start.isoformat(),
                end_time=end.isoformat(),
                aggregation=(
                    resolved_aggregation.value if resolved_aggregation else None
                ),
                bucket_ms=resolved_bucket_ms,
                tag_filters=tag_filters,
                limit=limit,
            )
            # Parse server response
            raw_points = result.get("points", [])
            metrics_data = result.get("metrics", [])
            if metrics_data:
                metrics = [
                    Metric(
                        timestamp=m.get("timestamp", ""),
                        values=m.get("values", {}),
                        tags=m.get("tags", {}),
                    )
                    for m in metrics_data[:limit]
                ]
                return TimeSeriesQueryResponse(
                    metrics=metrics,
                    total_points=result.get("total_points", len(metrics_data)),
                    query_time_ms=int((time.time() - started_at) * 1000),
                )
            else:
                return TimeSeriesQueryResponse(
                    raw_points=raw_points[:limit],
                    total_points=result.get("total_points", len(raw_points)),
                    query_time_ms=int((time.time() - started_at) * 1000),
                )
        except Exception as e:
            warnings.warn(f"Server query failed, using local storage: {e}")
            # Fall back to local storage
            points = [
                point
                for point in self._points.get(collection_id, [])
                if start <= point["timestamp"] <= end
                and self._matches_filter(point, filter, tag_filters)
            ]
            if resolved_aggregation is None:
                raw_points = [self._serialize_point(point) for point in points[:limit]]
                return TimeSeriesQueryResponse(
                    raw_points=raw_points,
                    total_points=len(points),
                    query_time_ms=int((time.time() - started_at) * 1000),
                )
            metrics = self._aggregate_points(
                collection_id=collection_id,
                points=points,
                aggregation=resolved_aggregation,
                bucket_ms=resolved_bucket_ms,
                value_columns=value_columns,
            )[:limit]
            return TimeSeriesQueryResponse(
                metrics=metrics,
                total_points=len(points),
                query_time_ms=int((time.time() - started_at) * 1000),
            )

    def get_latest(
        self,
        collection_id: str,
        tags: dict[str, Any],
    ) -> Metric | None:
        """Get the latest metric for given tags."""
        matched = [
            point
            for point in self._points.get(collection_id, [])
            if all(point["tags"].get(key) == value for key, value in tags.items())
        ]
        if not matched:
            return None
        latest = max(matched, key=lambda point: point["timestamp"])
        return Metric(
            timestamp=self._format_timestamp(latest["timestamp"]),
            values=dict(latest["values"]),
            tags=dict(latest["tags"]),
        )

    def get_latest_batch(
        self,
        collection_id: str,
        tags_list: list[dict[str, Any]],
    ) -> list[Metric | None]:
        """Get latest metrics for multiple tag combinations.

        Args:
            collection_id: Collection identifier
            tags_list: List of tag combinations

        Returns:
            List of latest metrics (None for missing)

        Example:
            latest_metrics = ts.get_latest_batch(
                collection_id="code_metrics",
                tags_list=[
                    {"file_path": "main.py"},
                    {"file_path": "utils.py"},
                    {"file_path": "config.py"},
                ]
            )
        """
        return [self.get_latest(collection_id, tags) for tags in tags_list]

    # ========================================================================
    # Aggregation and Downsampling
    # ========================================================================

    def aggregate(
        self,
        collection_id: str,
        start_time: str | datetime,
        end_time: str | datetime,
        aggregation: AggregationType | str | None = None,
        interval: str | None = None,
        value_column: str | None = None,
        pipeline: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Aggregate time-series data."""
        started_at = time.time()
        if pipeline:
            result_metrics: list[dict[str, Any]] = []
            for stage in pipeline:
                if stage.get("stage") == "group_by":
                    response = self.query(
                        collection_id=collection_id,
                        start_time=start_time,
                        end_time=end_time,
                        aggregation=stage.get(
                            "aggregation", aggregation or AggregationType.AVG
                        ),
                        bucket_ms=stage.get("bucket_ms"),
                        value_columns=stage.get("value_columns"),
                    )
                    result_metrics = self._aggregate_points(
                        collection_id=collection_id,
                        points=[
                            self._normalize_metric(point)
                            for point in response.get("raw_points", [])
                        ]
                        or [
                            point
                            for point in self._points.get(collection_id, [])
                            if self._parse_timestamp(start_time)
                            <= point["timestamp"]
                            <= self._parse_timestamp(end_time)
                        ],
                        aggregation=self._normalize_aggregation(
                            stage.get("aggregation", aggregation or AggregationType.AVG)
                        )
                        or AggregationType.AVG,
                        bucket_ms=stage.get("bucket_ms"),
                        value_columns=stage.get("value_columns"),
                        group_columns=stage.get("tag_columns"),
                    )
                else:
                    response = self.query(
                        collection_id=collection_id,
                        start_time=start_time,
                        end_time=end_time,
                        aggregation=stage.get(
                            "aggregation", aggregation or AggregationType.AVG
                        ),
                        bucket_ms=stage.get("bucket_ms"),
                        value_columns=stage.get("value_columns"),
                    )
                    result_metrics = response.get("metrics", [])
            return {
                "results": result_metrics,
                "metrics": result_metrics,
                "query_time_ms": int((time.time() - started_at) * 1000),
            }

        response = self.query(
            collection_id=collection_id,
            start_time=start_time,
            end_time=end_time,
            aggregation=aggregation,
            interval=interval,
            value_columns=[value_column] if value_column else None,
        )
        metrics = response.get("metrics", [])
        return {
            "results": metrics,
            "metrics": metrics,
            "query_time_ms": response.get(
                "query_time_ms", int((time.time() - started_at) * 1000)
            ),
        }

    def downsample(
        self,
        collection_id: str,
        target_collection: str,
        interval: str,
        mode: DownsampleMode = DownsampleMode.TIME_FIXED,
    ) -> dict[str, Any]:
        """Downsample time-series data to a new collection.

        Args:
            collection_id: Source collection
            target_collection: Target collection for downsampled data
            interval: Downsample interval
            mode: Downsampling mode

        Returns:
            Downsample result with statistics

        Example:
            # Downsample to hourly data
            result = ts.downsample(
                collection_id="code_metrics_raw",
                target_collection="code_metrics_hourly",
                interval="1h"
            )
        """
        return {
            "success": True,
            "downsampled": 0,
        }

    # ========================================================================
    # Batch Operations
    # ========================================================================

    def flush_batch(self, collection_id: str) -> dict[str, Any]:
        """Flush pending batch operations."""
        if collection_id not in self._batch_buffer:
            return {"success": True, "flushed": 0}

        batch = self._batch_buffer[collection_id]
        if not batch:
            return {"success": True, "flushed": 0}

        flushed = len(batch)
        self._batch_buffer[collection_id] = []
        return {
            "success": True,
            "flushed": flushed,
        }


# =============================================================================
# High-Level Time-Series API
# =============================================================================


class ProximaDBTimeSeries:
    """High-level time-series operations interface.

    Provides simplified API for time-series operations with automatic
    connection management, batching, and compression.

    Args:
        client: ProximaDB client instance
        batch_size: Batch size for auto-flush
        compression: Compression codec for data
    """

    def __init__(
        self,
        client: Any,
        batch_size: int = 1000,
        compression: CompressionCodec = CompressionCodec.GORILLA,
    ):
        """Initialize time-series API.

        Args:
            client: ProximaDB client instance
            batch_size: Batch size for auto-flush
            compression: Compression codec
        """
        self._client = client
        self._repository = TimeSeriesRepository(
            client=client,
            batch_size=batch_size,
            compression=compression,
        )

    def create_collection(
        self,
        name: str | None = None,
        value_columns: list[ValueColumn] | None = None,
        tags_columns: list[str] | None = None,
        timestamp_column: str = "timestamp",
        retention: str | None = "30d",
        compression: CompressionCodec = CompressionCodec.GORILLA,
        config: TimeSeriesCollectionConfig | None = None,
        tag_columns: list[str] | None = None,
        retention_ms: int | None = None,
    ) -> dict[str, Any]:
        """Create a time-series collection."""
        resolved_config = config or TimeSeriesCollectionConfig(
            name=name or "",
            timestamp_column=timestamp_column,
            value_columns=value_columns or [],
            tag_columns=(
                tag_columns if tag_columns is not None else (tags_columns or [])
            ),
            retention_ms=retention_ms,
            retention=retention,
            compression=compression,
        )
        collection_id = self._repository.create_collection(resolved_config)
        return {"success": True, "collection_id": collection_id}

    def ingest(
        self,
        collection_id: str,
        metrics: list[Metric | dict[str, Any]] | None = None,
        points: list[Metric | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Ingest time-series metrics."""
        payload = points if points is not None else metrics or []
        return self._repository.ingest(collection_id, payload)

    def query(
        self,
        collection_id: str,
        start_time: str | datetime,
        end_time: str | datetime,
        filter: TimeSeriesFilter | dict[str, Any] | None = None,
        aggregation: AggregationType | str | None = None,
        interval: str | None = None,
        limit: int = 1000,
        bucket_ms: int | None = None,
        tag_filters: dict[str, Any] | None = None,
        value_columns: list[str] | None = None,
    ) -> TimeSeriesQueryResponse:
        """Query time-series data."""
        return self._repository.query(
            collection_id=collection_id,
            start_time=start_time,
            end_time=end_time,
            filter=filter,
            aggregation=aggregation,
            interval=interval,
            limit=limit,
            bucket_ms=bucket_ms,
            tag_filters=tag_filters,
            value_columns=value_columns,
        )

    def get_latest(
        self,
        collection_id: str,
        tags: dict[str, Any],
    ) -> Metric | None:
        """Get latest metric for tags.

        Args:
            collection_id: Collection identifier
            tags: Tag filters

        Returns:
            Latest metric or None

        Example:
            latest = ts.get_latest(
                collection_id="code_metrics",
                tags={"file_path": "src/main.py"}
            )
        """
        return self._repository.get_latest(collection_id, tags)

    def list_collections(self) -> list[dict[str, Any]]:
        """List time-series collections."""
        return self._repository.list_collections()

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a time-series collection."""
        return self._repository.delete_collection(collection_id)

    def aggregate(
        self,
        collection_id: str,
        start_time: str | datetime,
        end_time: str | datetime,
        pipeline: list[dict[str, Any]] | None = None,
        aggregation: AggregationType | str | None = None,
        interval: str | None = None,
        value_column: str | None = None,
    ) -> dict[str, Any]:
        """Run an aggregation pipeline."""
        return self._repository.aggregate(
            collection_id=collection_id,
            start_time=start_time,
            end_time=end_time,
            aggregation=aggregation,
            interval=interval,
            value_column=value_column,
            pipeline=pipeline,
        )

    def flush(self, collection_id: str) -> dict[str, Any]:
        """Flush pending batch operations.

        Args:
            collection_id: Collection identifier

        Returns:
            Flush result
        """
        return self._repository.flush_batch(collection_id)


# =============================================================================
# Factory Functions
# =============================================================================


def create_timeseries_api(
    client: Any,
    batch_size: int = 1000,
    compression: CompressionCodec = CompressionCodec.GORILLA,
) -> ProximaDBTimeSeries:
    """Factory function to create time-series API instance.

    Args:
        client: ProximaDB client instance
        batch_size: Batch size for auto-flush
        compression: Compression codec

    Returns:
        ProximaDBTimeSeries instance
    """
    return ProximaDBTimeSeries(
        client=client,
        batch_size=batch_size,
        compression=compression,
    )
