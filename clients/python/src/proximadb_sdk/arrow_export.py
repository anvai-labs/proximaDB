"""
ProximaDB Arrow Export Client

High-level Python wrapper for accessing ProximaDB data via Arrow Flight.
Enables zero-copy data export to PyArrow, Polars, DuckDB, and pandas.

Supports all storage formats:
- ArrowBlock (.arrow) - Arrow IPC format from SST engine
- ProximaBlocks (.sst) - Native SST format (converted on-the-fly)
- Parquet (.parquet) - From the Nova engine

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import pyarrow as pa
    import pyarrow.flight as flight

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False
    pa = None
    flight = None

try:
    import polars as pl

    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False
    pl = None

try:
    import duckdb

    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False
    duckdb = None


class FileFormat(Enum):
    """Supported file formats for export."""

    ARROW = "arrow"  # Arrow IPC format
    PARQUET = "parquet"  # Parquet columnar format
    SST = "sst"  # ProximaBlocks native format


@dataclass
class FileInfo:
    """Metadata about an exportable file."""

    path: str
    filename: str
    size_bytes: int
    num_batches: int
    total_records: int
    dimension: int
    modified_at: int
    format: FileFormat

    @classmethod
    def from_flight_info(cls, info: "flight.FlightInfo") -> "FileInfo":
        """Parse FileInfo from Arrow Flight FlightInfo."""

        # Parse metadata from FlightInfo
        metadata = {}
        if info.schema and info.schema.metadata:
            for key, value in info.schema.metadata.items():
                try:
                    if isinstance(key, bytes):
                        key = key.decode("utf-8")
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    metadata[key] = value
                except (UnicodeDecodeError, AttributeError):
                    pass

        # Extract path from descriptor
        path = ""
        if info.descriptor and info.descriptor.path:
            path = "/".join(
                p.decode() if isinstance(p, bytes) else p for p in info.descriptor.path
            )

        # Determine format from path
        fmt = FileFormat.ARROW
        if path.endswith(".parquet"):
            fmt = FileFormat.PARQUET
        elif path.endswith(".sst"):
            fmt = FileFormat.SST

        # Extract dimension from schema
        dimension = 0
        if info.schema:
            for field in info.schema:
                if field.name == "vector" and hasattr(field.type, "list_size"):
                    dimension = field.type.list_size
                    break

        return cls(
            path=path,
            filename=path.split("/")[-1] if path else "",
            size_bytes=info.total_bytes if info.total_bytes >= 0 else 0,
            num_batches=int(metadata.get("num_batches", 1)),
            total_records=info.total_records if info.total_records >= 0 else 0,
            dimension=dimension,
            modified_at=int(metadata.get("modified_at", 0)),
            format=fmt,
        )


class ArrowExportClient:
    """
    Arrow Flight client for exporting ProximaDB data.

    Provides high-level access to ProximaDB storage files via Arrow Flight,
    enabling zero-copy data transfer to analytics tools.

    Example:
        >>> client = ArrowExportClient("localhost:5680")
        >>>
        >>> # List files in a collection
        >>> files = client.list_files("my_collection")
        >>> for f in files:
        ...     print(f"{f.filename}: {f.total_records} records")
        >>>
        >>> # Read into PyArrow Table
        >>> table = client.read_file("my_collection/data/block_0.arrow")
        >>> print(table.schema)
        >>>
        >>> # Read into Polars DataFrame
        >>> df = client.to_polars("my_collection/data/block_0.arrow")
        >>> print(df.head())
        >>>
        >>> # Read into DuckDB
        >>> conn = client.to_duckdb("my_collection/data/block_0.arrow")
        >>> result = conn.execute("SELECT id, metadata FROM vectors").fetchall()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5680,
        scheme: str = "grpc",
        tls: bool = False,
        auth_token: str | None = None,
    ):
        """
        Initialize Arrow Flight client.

        Args:
            host: ProximaDB server hostname
            port: Arrow Flight port (default 5680)
            scheme: Connection scheme ("grpc" or "grpc+tls")
            tls: Whether to use TLS (overrides scheme)
            auth_token: Optional authentication token
        """
        if not _PYARROW_AVAILABLE:
            raise ImportError(
                "PyArrow is required for Arrow export. "
                "Install with: pip install pyarrow"
            )

        self._host = host
        self._port = port
        self._tls = tls
        self._auth_token = auth_token

        # Build connection URI
        if tls:
            scheme = "grpc+tls"
        self._uri = f"{scheme}://{host}:{port}"

        # Initialize connection (lazy)
        self._client: flight.FlightClient | None = None

    @property
    def client(self) -> "flight.FlightClient":
        """Get or create Arrow Flight client."""
        if self._client is None:
            options = []
            if self._auth_token:
                options.append(("authorization", f"Bearer {self._auth_token}"))

            self._client = flight.connect(self._uri)
            logger.info(f"Connected to Arrow Flight at {self._uri}")

        return self._client

    def close(self):
        """Close the Arrow Flight connection."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -------------------------------------------------------------------------
    # List and Query Files
    # -------------------------------------------------------------------------

    def list_files(
        self,
        collection_id: str,
        pattern: str | None = None,
        format_filter: FileFormat | None = None,
    ) -> list[FileInfo]:
        """
        List available files in a collection.

        Args:
            collection_id: Collection name or ID
            pattern: Optional glob pattern (e.g., "*.arrow", "block_*.parquet")
            format_filter: Filter by file format

        Returns:
            List of FileInfo objects describing available files

        Example:
            >>> files = client.list_files("embeddings")
            >>> arrow_files = client.list_files("embeddings", format_filter=FileFormat.ARROW)
        """
        # Build criteria for list_flights
        criteria = collection_id.encode("utf-8")

        files = []
        for info in self.client.list_flights(criteria):
            file_info = FileInfo.from_flight_info(info)

            # Apply format filter
            if format_filter and file_info.format != format_filter:
                continue

            # Apply pattern filter (simple glob matching)
            if pattern:
                import fnmatch

                if not fnmatch.fnmatch(file_info.filename, pattern):
                    continue

            files.append(file_info)

        return files

    def get_file_info(self, path: str) -> FileInfo:
        """
        Get detailed information about a specific file.

        Args:
            path: File path (e.g., "my_collection/data/block_0.arrow")

        Returns:
            FileInfo with schema and statistics
        """
        descriptor = flight.FlightDescriptor.for_path(*path.split("/"))
        info = self.client.get_flight_info(descriptor)
        return FileInfo.from_flight_info(info)

    def get_schema(self, path: str) -> "pa.Schema":
        """
        Get Arrow schema for a file.

        Args:
            path: File path

        Returns:
            PyArrow Schema
        """
        descriptor = flight.FlightDescriptor.for_path(*path.split("/"))
        info = self.client.get_flight_info(descriptor)
        return info.schema

    # -------------------------------------------------------------------------
    # Read Data
    # -------------------------------------------------------------------------

    def read_file(self, path: str) -> "pa.Table":
        """
        Read a file into a PyArrow Table.

        This is a zero-copy operation when possible.

        Args:
            path: File path (e.g., "my_collection/data/block_0.arrow")

        Returns:
            PyArrow Table containing all records

        Example:
            >>> table = client.read_file("embeddings/data/block_0.arrow")
            >>> print(f"Read {table.num_rows} vectors")
            >>> vectors = table['vector'].to_numpy()
        """
        # Get flight info to find endpoint
        descriptor = flight.FlightDescriptor.for_path(*path.split("/"))
        info = self.client.get_flight_info(descriptor)

        # Get data from first endpoint
        if not info.endpoints:
            raise ValueError(f"No endpoints available for {path}")

        ticket = info.endpoints[0].ticket
        reader = self.client.do_get(ticket)

        return reader.read_all()

    def read_batches(
        self,
        path: str,
        batch_size: int | None = None,
    ) -> Iterator["pa.RecordBatch"]:
        """
        Stream file as record batches (memory-efficient).

        Args:
            path: File path
            batch_size: Optional batch size limit

        Yields:
            PyArrow RecordBatch objects

        Example:
            >>> for batch in client.read_batches("large_collection/data/block_0.arrow"):
            ...     # Process each batch
            ...     print(f"Batch with {batch.num_rows} rows")
        """
        descriptor = flight.FlightDescriptor.for_path(*path.split("/"))
        info = self.client.get_flight_info(descriptor)

        if not info.endpoints:
            return

        ticket = info.endpoints[0].ticket
        reader = self.client.do_get(ticket)

        for batch in reader:
            yield batch.data

    def read_collection(
        self,
        collection_id: str,
        format_filter: FileFormat | None = None,
    ) -> "pa.Table":
        """
        Read all files from a collection into a single table.

        Args:
            collection_id: Collection name or ID
            format_filter: Only read files of this format

        Returns:
            Combined PyArrow Table

        Example:
            >>> table = client.read_collection("embeddings")
            >>> print(f"Total: {table.num_rows} vectors")
        """
        files = self.list_files(collection_id, format_filter=format_filter)

        if not files:
            return pa.table({})

        tables = []
        for file_info in files:
            table = self.read_file(file_info.path)
            tables.append(table)

        return pa.concat_tables(tables)

    # -------------------------------------------------------------------------
    # Format Conversions
    # -------------------------------------------------------------------------

    def to_polars(
        self,
        path: str,
        rechunk: bool = True,
    ) -> "pl.DataFrame":
        """
        Read file directly into a Polars DataFrame.

        Args:
            path: File path
            rechunk: Whether to rechunk for optimal memory layout

        Returns:
            Polars DataFrame

        Example:
            >>> df = client.to_polars("embeddings/data/block_0.arrow")
            >>> df.filter(pl.col("metadata.category") == "tech").head()
        """
        if not _POLARS_AVAILABLE:
            raise ImportError("Polars is required. Install with: pip install polars")

        table = self.read_file(path)
        df = pl.from_arrow(table, rechunk=rechunk)
        return df

    def to_duckdb(
        self,
        path: str,
        table_name: str = "vectors",
        conn: Optional["duckdb.DuckDBPyConnection"] = None,
    ) -> "duckdb.DuckDBPyConnection":
        """
        Load file into DuckDB for SQL analytics.

        Args:
            path: File path
            table_name: Name for the registered table
            conn: Existing DuckDB connection (creates new if None)

        Returns:
            DuckDB connection with registered table

        Example:
            >>> conn = client.to_duckdb("embeddings/data/block_0.arrow")
            >>> result = conn.execute('''
            ...     SELECT id, metadata->>'category' as category
            ...     FROM vectors
            ...     WHERE array_length(vector) = 768
            ... ''').fetchdf()
        """
        if not _DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is required. Install with: pip install duckdb")

        table = self.read_file(path)

        if conn is None:
            conn = duckdb.connect(":memory:")

        conn.register(table_name, table)
        return conn

    def to_pandas(self, path: str) -> "pa.lib.pandas_api.DataFrame":
        """
        Read file into a pandas DataFrame.

        Note: For large datasets, prefer to_polars() for better performance.

        Args:
            path: File path

        Returns:
            pandas DataFrame
        """
        table = self.read_file(path)
        return table.to_pandas()

    def to_numpy(
        self,
        path: str,
        vector_column: str = "vector",
    ) -> "pa.lib.pandas_api.DataFrame":
        """
        Extract vectors as a NumPy array.

        Args:
            path: File path
            vector_column: Name of the vector column

        Returns:
            NumPy array of shape (num_vectors, dimension)

        Example:
            >>> vectors = client.to_numpy("embeddings/data/block_0.arrow")
            >>> print(vectors.shape)  # (1000, 768)
        """
        import numpy as np

        table = self.read_file(path)
        vector_col = table.column(vector_column)

        # Convert list column to numpy
        vectors = []
        for chunk in vector_col.chunks:
            for i in range(len(chunk)):
                vectors.append(chunk[i].as_py())

        return np.array(vectors, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Collection Statistics
    # -------------------------------------------------------------------------

    def collection_stats(self, collection_id: str) -> dict:
        """
        Get statistics for a collection's exported files.

        Args:
            collection_id: Collection name or ID

        Returns:
            Dictionary with collection statistics

        Example:
            >>> stats = client.collection_stats("embeddings")
            >>> print(f"Total size: {stats['total_size_mb']:.2f} MB")
        """
        files = self.list_files(collection_id)

        if not files:
            return {
                "collection_id": collection_id,
                "num_files": 0,
                "total_records": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0.0,
                "formats": {},
            }

        format_counts = {}
        for f in files:
            fmt_name = f.format.value
            if fmt_name not in format_counts:
                format_counts[fmt_name] = {"count": 0, "records": 0, "bytes": 0}
            format_counts[fmt_name]["count"] += 1
            format_counts[fmt_name]["records"] += f.total_records
            format_counts[fmt_name]["bytes"] += f.size_bytes

        total_bytes = sum(f.size_bytes for f in files)

        return {
            "collection_id": collection_id,
            "num_files": len(files),
            "total_records": sum(f.total_records for f in files),
            "total_size_bytes": total_bytes,
            "total_size_mb": total_bytes / (1024 * 1024),
            "dimension": files[0].dimension if files else 0,
            "formats": format_counts,
        }


# Convenience functions


def connect_arrow(
    host: str = "localhost", port: int = 5680, **kwargs
) -> ArrowExportClient:
    """
    Create an Arrow export client.

    Example:
        >>> from proximadb_sdk.arrow_export import connect_arrow
        >>> with connect_arrow() as client:
        ...     table = client.read_file("my_collection/data/block_0.arrow")
    """
    return ArrowExportClient(host=host, port=port, **kwargs)


def read_proximadb_file(
    path: str,
    host: str = "localhost",
    port: int = 5680,
) -> "pa.Table":
    """
    One-liner to read a ProximaDB file into PyArrow.

    Example:
        >>> table = read_proximadb_file("embeddings/data/block_0.arrow")
    """
    with ArrowExportClient(host=host, port=port) as client:
        return client.read_file(path)


def read_proximadb_collection(
    collection_id: str,
    host: str = "localhost",
    port: int = 5680,
) -> "pa.Table":
    """
    One-liner to read an entire collection into PyArrow.

    Example:
        >>> table = read_proximadb_collection("embeddings")
        >>> print(f"Loaded {table.num_rows} vectors")
    """
    with ArrowExportClient(host=host, port=port) as client:
        return client.read_collection(collection_id)
