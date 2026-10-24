"""
ProximaDB Python SDK - Centralized Proto/Type Conversion

This module provides a single location for all type conversions between:
- REST API (string-based enums)
- gRPC API (integer-based enums)
- Pydantic models

Eliminates scattered conversion code across the SDK (~123 lines consolidated here).

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

from enum import Enum
from typing import Any


class ProtoConverter:
    """Centralized converter for REST (string) <-> gRPC (integer) enum mappings.

    Usage:
        # Convert string to int (for gRPC)
        metric_int = ProtoConverter.distance_metric_to_int("cosine")  # Returns 1

        # Convert int to string (from gRPC response)
        metric_str = ProtoConverter.distance_metric_to_str(1)  # Returns "cosine"

        # Handle any input type
        metric = ProtoConverter.distance_metric_to_int(DistanceMetric.COSINE)  # Works with enum
    """

    # =========================================================================
    # Distance Metric Mappings (13 metrics)
    # =========================================================================

    _DISTANCE_METRIC_STR_TO_INT: dict[str, int] = {
        "unspecified": 0,
        "cosine": 1,
        "euclidean": 2,
        "dot_product": 3,
        "hamming": 4,
        "manhattan": 5,
        "jaccard": 6,
        "chebyshev": 7,
        "canberra": 8,
        "minkowski": 9,
        "angular": 10,
        "bray_curtis": 11,
        "hellinger": 12,
        "custom": 13,
    }

    _DISTANCE_METRIC_INT_TO_STR: dict[int, str] = {
        v: k for k, v in _DISTANCE_METRIC_STR_TO_INT.items()
    }

    # =========================================================================
    # Storage Engine Mappings (6 engines + legacy aliases)
    # =========================================================================

    _STORAGE_ENGINE_STR_TO_INT: dict[str, int] = {
        "unspecified": 0,
        "viper": 1,
        "sst": 2,
        "nova": 3,
        "helix": 4,
        "swift": 5,
        "raptor": 6,
        # Legacy fallbacks (map to viper)
        "mmap": 1,
        "hybrid": 1,
    }

    _STORAGE_ENGINE_INT_TO_STR: dict[int, str] = {
        0: "unspecified",
        1: "viper",
        2: "sst",
        3: "nova",
        4: "helix",
        5: "swift",
        6: "raptor",
    }

    # =========================================================================
    # Index Type Mappings (6 algorithms)
    # =========================================================================

    _INDEX_TYPE_STR_TO_INT: dict[str, int] = {
        "unspecified": 0,
        "hnsw": 1,
        "ivf": 2,
        "pq": 3,
        "flat": 4,
        "annoy": 5,
        "lsh": 6,
    }

    _INDEX_TYPE_INT_TO_STR: dict[int, str] = {
        v: k for k, v in _INDEX_TYPE_STR_TO_INT.items()
    }

    # =========================================================================
    # Quantization Type Mappings
    # =========================================================================

    _QUANTIZATION_TYPE_STR_TO_INT: dict[str, int] = {
        "none": 0,
        "uniform": 1,
        "pq": 2,
        "scalar": 3,
        "binary": 4,
        "custom": 5,
    }

    _QUANTIZATION_TYPE_INT_TO_STR: dict[int, str] = {
        v: k for k, v in _QUANTIZATION_TYPE_STR_TO_INT.items()
    }

    # =========================================================================
    # Distance Metric Conversion Methods
    # =========================================================================

    @classmethod
    def distance_metric_to_int(cls, value: str | int | Enum | None) -> int:
        """Convert distance metric to gRPC integer representation.

        Args:
            value: String name, integer value, or enum instance

        Returns:
            Integer representation for gRPC (0 if unspecified/invalid)
        """
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, Enum):
            val = value.value
            if isinstance(val, int):
                return val
            return cls._DISTANCE_METRIC_STR_TO_INT.get(str(val).lower(), 0)
        if isinstance(value, str):
            return cls._DISTANCE_METRIC_STR_TO_INT.get(value.lower(), 0)
        return 0

    @classmethod
    def distance_metric_to_str(cls, value: str | int | Enum | None) -> str:
        """Convert distance metric to REST string representation.

        Args:
            value: Integer value, string name, or enum instance

        Returns:
            String representation for REST API ("cosine" as default)
        """
        if value is None:
            return "cosine"
        if isinstance(value, str):
            lower_val = value.lower()
            if lower_val in cls._DISTANCE_METRIC_STR_TO_INT:
                return lower_val
            return "cosine"
        if isinstance(value, int):
            return cls._DISTANCE_METRIC_INT_TO_STR.get(value, "cosine")
        if isinstance(value, Enum):
            val = value.value
            if isinstance(val, int):
                return cls._DISTANCE_METRIC_INT_TO_STR.get(val, "cosine")
            return str(val).lower()
        return "cosine"

    # =========================================================================
    # Storage Engine Conversion Methods
    # =========================================================================

    @classmethod
    def storage_engine_to_int(cls, value: str | int | Enum | None) -> int:
        """Convert storage engine to gRPC integer representation.

        Args:
            value: String name, integer value, or enum instance

        Returns:
            Integer representation for gRPC (1=viper as default)
        """
        if value is None:
            # TD-SDKENG-1 (round-2 review): SST is the default; VIPER is
            # deprecated (ADR-093) and the server rejects it.
            return cls._STORAGE_ENGINE_STR_TO_INT.get("sst", 2)
        if isinstance(value, int):
            return value
        if isinstance(value, Enum):
            val = value.value
            if isinstance(val, int):
                return val
            return cls._STORAGE_ENGINE_STR_TO_INT.get(
                str(val).lower(), cls._STORAGE_ENGINE_STR_TO_INT.get("sst", 2)
            )
        if isinstance(value, str):
            return cls._STORAGE_ENGINE_STR_TO_INT.get(
                value.lower(), cls._STORAGE_ENGINE_STR_TO_INT.get("sst", 2)
            )
        return cls._STORAGE_ENGINE_STR_TO_INT.get("sst", 2)

    @classmethod
    def storage_engine_to_str(cls, value: str | int | Enum | None) -> str:
        """Convert storage engine to REST string representation.

        Args:
            value: Integer value, string name, or enum instance

        Returns:
            String representation for REST API ("sst" as default)
        """
        if value is None:
            # TD-SDKENG-1: SST default; VIPER deprecated (ADR-093).
            return "sst"
        if isinstance(value, str):
            lower_val = value.lower()
            if lower_val in cls._STORAGE_ENGINE_STR_TO_INT:
                int_val = cls._STORAGE_ENGINE_STR_TO_INT[lower_val]
                return cls._STORAGE_ENGINE_INT_TO_STR.get(int_val, "sst")
            return "sst"
        if isinstance(value, int):
            return cls._STORAGE_ENGINE_INT_TO_STR.get(value, "sst")
        if isinstance(value, Enum):
            val = value.value
            if isinstance(val, int):
                return cls._STORAGE_ENGINE_INT_TO_STR.get(val, "sst")
            return str(val).lower()
        return "sst"

    # =========================================================================
    # Index Type Conversion Methods
    # =========================================================================

    @classmethod
    def index_type_to_int(cls, value: str | int | Enum | None) -> int:
        """Convert index type/algorithm to gRPC integer representation.

        Args:
            value: String name, integer value, or enum instance

        Returns:
            Integer representation for gRPC (1=hnsw as default)
        """
        if value is None:
            return 1  # Default to HNSW
        if isinstance(value, int):
            return value
        if isinstance(value, Enum):
            val = value.value
            if isinstance(val, int):
                return val
            return cls._INDEX_TYPE_STR_TO_INT.get(str(val).lower(), 1)  # hnsw
        if isinstance(value, str):
            return cls._INDEX_TYPE_STR_TO_INT.get(value.lower(), 1)
        return 1

    @classmethod
    def index_type_to_str(cls, value: str | int | Enum | None) -> str:
        """Convert index type/algorithm to REST string representation.

        Args:
            value: Integer value, string name, or enum instance

        Returns:
            String representation for REST API ("hnsw" as default)
        """
        if value is None:
            return "hnsw"
        if isinstance(value, str):
            lower_val = value.lower()
            if lower_val in cls._INDEX_TYPE_STR_TO_INT:
                return lower_val
            return "hnsw"
        if isinstance(value, int):
            return cls._INDEX_TYPE_INT_TO_STR.get(value, "hnsw")
        if isinstance(value, Enum):
            val = value.value
            if isinstance(val, int):
                return cls._INDEX_TYPE_INT_TO_STR.get(val, "hnsw")
            return str(val).lower()
        return "hnsw"

    # =========================================================================
    # Quantization Type Conversion Methods
    # =========================================================================

    @classmethod
    def quantization_type_to_int(cls, value: str | int | Enum | None) -> int:
        """Convert quantization type to gRPC integer representation."""
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, Enum):
            val = value.value
            if isinstance(val, int):
                return val
            return cls._QUANTIZATION_TYPE_STR_TO_INT.get(str(val).lower(), 0)
        if isinstance(value, str):
            return cls._QUANTIZATION_TYPE_STR_TO_INT.get(value.lower(), 0)
        return 0

    @classmethod
    def quantization_type_to_str(cls, value: str | int | Enum | None) -> str:
        """Convert quantization type to REST string representation."""
        if value is None:
            return "none"
        if isinstance(value, str):
            lower_val = value.lower()
            if lower_val in cls._QUANTIZATION_TYPE_STR_TO_INT:
                return lower_val
            return "none"
        if isinstance(value, int):
            return cls._QUANTIZATION_TYPE_INT_TO_STR.get(value, "none")
        if isinstance(value, Enum):
            val = value.value
            if isinstance(val, int):
                return cls._QUANTIZATION_TYPE_INT_TO_STR.get(val, "none")
            return str(val).lower()
        return "none"

    # =========================================================================
    # Model Conversion Helpers
    # =========================================================================

    @classmethod
    def vector_record_to_dict(cls, record: Any) -> dict[str, Any]:
        """Convert a VectorRecord to a dictionary for REST API."""
        if isinstance(record, dict):
            return record
        if hasattr(record, "model_dump"):
            return record.model_dump(exclude_none=True)
        if hasattr(record, "dict"):
            return record.dict(exclude_none=True)
        return {
            "id": getattr(record, "id", ""),
            "vector": list(getattr(record, "vector", [])),
            "metadata": getattr(record, "metadata", None),
        }

    @classmethod
    def dict_to_search_result(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize a search result dictionary."""
        return {
            "id": data.get("id", data.get("vector_id", "")),
            "score": data.get("score", data.get("distance", 0.0)),
            "vector": data.get("vector", []),
            "metadata": data.get("metadata", {}),
        }

    @classmethod
    def collection_config_to_dict(
        cls,
        name: str,
        dimension: int,
        distance_metric: str | int | Enum | None = None,
        storage_engine: str | int | Enum | None = None,
        index_type: str | int | Enum | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Build a collection config dictionary for REST API."""
        config = {
            "name": name,
            "dimension": dimension,
            "distance_metric": cls.distance_metric_to_str(distance_metric),
            "storage_engine": cls.storage_engine_to_str(storage_engine),
            "index_type": cls.index_type_to_str(index_type),
        }
        config.update(kwargs)
        return config


# =========================================================================
# Convenience Functions (for direct import)
# =========================================================================


def distance_metric_to_int(value: str | int | Enum | None) -> int:
    """Convert distance metric to integer."""
    return ProtoConverter.distance_metric_to_int(value)


def distance_metric_to_str(value: str | int | Enum | None) -> str:
    """Convert distance metric to string."""
    return ProtoConverter.distance_metric_to_str(value)


def storage_engine_to_int(value: str | int | Enum | None) -> int:
    """Convert storage engine to integer."""
    return ProtoConverter.storage_engine_to_int(value)


def storage_engine_to_str(value: str | int | Enum | None) -> str:
    """Convert storage engine to string."""
    return ProtoConverter.storage_engine_to_str(value)


def index_type_to_int(value: str | int | Enum | None) -> int:
    """Convert index type to integer."""
    return ProtoConverter.index_type_to_int(value)


def index_type_to_str(value: str | int | Enum | None) -> str:
    """Convert index type to string."""
    return ProtoConverter.index_type_to_str(value)


def quantization_type_to_int(value: str | int | Enum | None) -> int:
    """Convert quantization type to integer."""
    return ProtoConverter.quantization_type_to_int(value)


def quantization_type_to_str(value: str | int | Enum | None) -> str:
    """Convert quantization type to string."""
    return ProtoConverter.quantization_type_to_str(value)
