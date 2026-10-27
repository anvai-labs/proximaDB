"""
Test suite for Arrow Export SDK

Tests the ArrowExportClient wrapper for accessing ProximaDB data
via Arrow Flight.

Run with: PYTHONPATH=clients/python/src pytest clients/python/tests/test_arrow_export_sdk.py -v
"""

from unittest.mock import MagicMock, patch

import pytest


# Test imports and availability
def test_arrow_export_imports():
    """Test that Arrow export module can be imported."""
    from proximadb_sdk.arrow_export import ArrowExportClient, FileFormat

    assert ArrowExportClient is not None
    assert FileFormat is not None


def test_file_format_enum():
    """Test FileFormat enum values."""
    from proximadb_sdk.arrow_export import FileFormat

    assert FileFormat.ARROW.value == "arrow"
    assert FileFormat.PARQUET.value == "parquet"
    assert FileFormat.SST.value == "sst"


def test_file_info_dataclass():
    """Test FileInfo dataclass creation."""
    from proximadb_sdk.arrow_export import FileFormat, FileInfo

    info = FileInfo(
        path="collection/data/block_0.arrow",
        filename="block_0.arrow",
        size_bytes=1024,
        num_batches=1,
        total_records=100,
        dimension=128,
        modified_at=1234567890,
        format=FileFormat.ARROW,
    )

    assert info.path == "collection/data/block_0.arrow"
    assert info.filename == "block_0.arrow"
    assert info.total_records == 100
    assert info.dimension == 128
    assert info.format == FileFormat.ARROW


class TestArrowExportClientUnit:
    """Unit tests for ArrowExportClient (mocked flight)."""

    def test_client_creation(self):
        """Test client creation with various parameters."""
        from proximadb_sdk.arrow_export import ArrowExportClient

        client = ArrowExportClient(host="localhost", port=5680)
        assert client._host == "localhost"
        assert client._port == 5680
        assert client._uri == "grpc://localhost:5680"

    def test_client_tls(self):
        """Test client creation with TLS."""
        from proximadb_sdk.arrow_export import ArrowExportClient

        client = ArrowExportClient(host="localhost", port=5680, tls=True)
        assert client._tls == True
        assert client._uri == "grpc+tls://localhost:5680"

    def test_connect_arrow_convenience(self):
        """Test connect_arrow convenience function."""
        from proximadb_sdk.arrow_export import connect_arrow

        client = connect_arrow(host="testhost", port=9999)
        assert client._host == "testhost"
        assert client._port == 9999


class TestArrowExportWithMocks:
    """Tests with mocked Arrow Flight client."""

    @pytest.fixture
    def mock_flight_client(self):
        """Create a mocked Flight client."""
        with patch("proximadb_sdk.arrow_export.flight") as mock_flight:
            mock_client = MagicMock()
            mock_flight.connect.return_value = mock_client
            yield mock_client

    def test_list_files_empty(self, mock_flight_client):
        """Test listing files from empty collection."""
        from proximadb_sdk.arrow_export import ArrowExportClient

        mock_flight_client.list_flights.return_value = []

        client = ArrowExportClient()
        client._client = mock_flight_client

        files = client.list_files("empty_collection")
        assert files == []

    def test_collection_stats_empty(self, mock_flight_client):
        """Test collection stats for empty collection."""
        from proximadb_sdk.arrow_export import ArrowExportClient

        mock_flight_client.list_flights.return_value = []

        client = ArrowExportClient()
        client._client = mock_flight_client

        stats = client.collection_stats("empty_collection")
        assert stats["num_files"] == 0
        assert stats["total_records"] == 0


# Conditional tests that require pyarrow
try:
    import pyarrow as pa
    import pyarrow.flight as flight

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")
class TestArrowExportWithPyArrow:
    """Tests that require PyArrow."""

    def test_pyarrow_table_creation(self):
        """Test creating a PyArrow table (basic sanity check)."""
        table = pa.table(
            {
                "id": ["vec_0", "vec_1", "vec_2"],
                "vector": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            }
        )
        assert table.num_rows == 3
        assert "id" in table.column_names
        assert "vector" in table.column_names


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
class TestPolarsIntegration:
    """Tests for Polars integration."""

    def test_polars_from_arrow(self):
        """Test converting PyArrow to Polars."""
        if not PYARROW_AVAILABLE:
            pytest.skip("PyArrow required")

        table = pa.table(
            {
                "id": ["vec_0", "vec_1"],
                "value": [1.0, 2.0],
            }
        )

        df = pl.from_arrow(table)
        assert len(df) == 2
        assert "id" in df.columns


@pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not installed")
class TestDuckDBIntegration:
    """Tests for DuckDB integration."""

    def test_duckdb_register(self):
        """Test registering Arrow table with DuckDB."""
        if not PYARROW_AVAILABLE:
            pytest.skip("PyArrow required")

        table = pa.table(
            {
                "id": ["vec_0", "vec_1"],
                "value": [1.0, 2.0],
            }
        )

        conn = duckdb.connect(":memory:")
        conn.register("vectors", table)

        result = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        assert result[0] == 2


class TestFileFormatDetection:
    """Tests for file format detection."""

    def test_arrow_format_detection(self):
        """Test detection of .arrow files."""
        from proximadb_sdk.arrow_export import FileFormat

        # Simulate format detection based on path
        path = "collection/data/block_0.arrow"
        if path.endswith(".arrow"):
            fmt = FileFormat.ARROW
        elif path.endswith(".parquet"):
            fmt = FileFormat.PARQUET
        elif path.endswith(".sst"):
            fmt = FileFormat.SST
        else:
            fmt = None

        assert fmt == FileFormat.ARROW

    def test_parquet_format_detection(self):
        """Test detection of .parquet files."""
        from proximadb_sdk.arrow_export import FileFormat

        path = "collection/data/vectors.parquet"
        if path.endswith(".arrow"):
            fmt = FileFormat.ARROW
        elif path.endswith(".parquet"):
            fmt = FileFormat.PARQUET
        elif path.endswith(".sst"):
            fmt = FileFormat.SST
        else:
            fmt = None

        assert fmt == FileFormat.PARQUET

    def test_sst_format_detection(self):
        """Test detection of .sst files."""
        from proximadb_sdk.arrow_export import FileFormat

        path = "collection/data/block_0.sst"
        if path.endswith(".arrow"):
            fmt = FileFormat.ARROW
        elif path.endswith(".parquet"):
            fmt = FileFormat.PARQUET
        elif path.endswith(".sst"):
            fmt = FileFormat.SST
        else:
            fmt = None

        assert fmt == FileFormat.SST


class TestSDKExports:
    """Test that Arrow export is properly exported from SDK."""

    def test_sdk_exports_arrow_client(self):
        """Test ArrowExportClient is exported from main SDK."""
        from proximadb_sdk import ArrowExportClient

        assert ArrowExportClient is not None

    def test_sdk_exports_connect_arrow(self):
        """Test connect_arrow is exported from main SDK."""
        from proximadb_sdk import connect_arrow

        assert connect_arrow is not None

    def test_sdk_exports_file_format(self):
        """Test FileFormat is exported from main SDK."""
        from proximadb_sdk import FileFormat

        assert FileFormat is not None

    def test_sdk_exports_file_info(self):
        """Test FileInfo is exported from main SDK."""
        from proximadb_sdk import FileInfo

        assert FileInfo is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
