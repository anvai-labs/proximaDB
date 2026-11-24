"""
Test PyArrow/DuckDB/Polars interoperability with ProximaDB Nova Parquet files.

This test verifies that Parquet files produced by ProximaDB's Nova storage
engines can be read by standard data processing tools:
- PyArrow: pq.read_table(), pq.ParquetFile()
- DuckDB: SELECT * FROM 'file.parquet'
- Polars: pl.read_parquet()

The tests create Parquet files matching ProximaDB's columnar schema and verify
that external tools can correctly parse the data, extract vectors, and run queries.
"""

import json
import os
import tempfile

import pytest

# Check if pyarrow is available
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Check if numpy is available
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# =============================================================================
# ProximaDB Parquet Schema Constants (matching src/storage/engines/core/formats/columnar/constants.rs)
# =============================================================================

# Core columns
FIELD_ID = "id"
FIELD_VECTOR_FP32 = "vector_fp32"
FIELD_TIMESTAMP = "timestamp"
FIELD_UPDATED_AT = "updated_at"
FIELD_EXPIRES_AT = "expires_at"
FIELD_VERSION = "version"
FIELD_SOURCE = "source"
FIELD_EXTRA_META = "extra_meta"
FIELD_ROW_GROUP_OFFSET = "row_group_offset"
FIELD_ROW_INDEX = "row_index"

# Quantization columns
FIELD_Q_BINARY = "q_binary"
FIELD_Q_INT8 = "q_int8"
FIELD_QP_INT8_SCALE = "qp_int8_scale"
FIELD_QP_INT8_MIN = "qp_int8_min"
FIELD_QP_INT8_MAX = "qp_int8_max"
FIELD_Q_PQ8 = "q_pq8"


# =============================================================================
# Helper Functions
# =============================================================================


def create_proximadb_parquet_schema(
    dimension: int, include_quantization: bool = False
) -> pa.Schema:
    """Create an Arrow schema matching ProximaDB's columnar format.

    This matches the schema defined in:
    - src/storage/engines/core/formats/columnar/parquet_write_engine/schema_builder.rs
    - src/storage/engines/core/formats/columnar/constants.rs
    """
    if not PYARROW_AVAILABLE or not NUMPY_AVAILABLE:
        pytest.skip("PyArrow and NumPy required for this test")

    fields = [
        # Core identity column (NOT NULL)
        pa.field(FIELD_ID, pa.utf8(), nullable=False),
        # Row group management
        pa.field(FIELD_ROW_GROUP_OFFSET, pa.uint32(), nullable=False),
        pa.field(FIELD_ROW_INDEX, pa.uint32(), nullable=False),
        # Vector data as fixed-size list of float32
        pa.field(FIELD_VECTOR_FP32, pa.list_(pa.float32(), dimension), nullable=False),
        # Temporal columns
        pa.field(FIELD_TIMESTAMP, pa.int64(), nullable=False),
        pa.field(FIELD_UPDATED_AT, pa.int64(), nullable=True),
        pa.field(FIELD_EXPIRES_AT, pa.int64(), nullable=True),
        pa.field(FIELD_VERSION, pa.uint32(), nullable=True),
        pa.field(FIELD_SOURCE, pa.utf8(), nullable=True),
        # Extra metadata as JSON string (simplified from Map type)
        pa.field(FIELD_EXTRA_META, pa.utf8(), nullable=True),
    ]

    if include_quantization:
        # Binary quantization (1 bit per dimension)
        binary_size = (dimension + 7) // 8  # Round up to bytes
        fields.append(pa.field(FIELD_Q_BINARY, pa.binary(), nullable=True))
        # INT8 quantization with scale parameters
        fields.append(pa.field(FIELD_Q_INT8, pa.binary(), nullable=True))
        fields.append(pa.field(FIELD_QP_INT8_SCALE, pa.float32(), nullable=True))
        fields.append(pa.field(FIELD_QP_INT8_MIN, pa.float32(), nullable=True))
        fields.append(pa.field(FIELD_QP_INT8_MAX, pa.float32(), nullable=True))
        # PQ8 quantization
        fields.append(pa.field(FIELD_Q_PQ8, pa.binary(), nullable=True))

    return pa.schema(fields)


def create_nova_parquet_file(
    path: str,
    num_vectors: int = 100,
    dimension: int = 64,
    include_quantization: bool = False,
    compression: str = "zstd",
) -> tuple:
    """Create a test Parquet file matching NOVA engine format.

    NOVA uses progressive columnar format with optional quantization columns
    for staged search (Binary -> INT8 -> FP32).
    """
    if not PYARROW_AVAILABLE or not NUMPY_AVAILABLE:
        pytest.skip("PyArrow and NumPy required for this test")

    # Generate random vectors (normalized for realistic test data)
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-10, None)  # Normalize

    # Generate test data
    ids = [f"nova_vec_{i:05d}" for i in range(num_vectors)]
    timestamps = [
        1700000000000 + i * 1000 for i in range(num_vectors)
    ]  # Millisecond timestamps
    row_group_offsets = [0] * num_vectors  # All in first row group for simplicity
    row_indices = list(range(num_vectors))
    categories = ["tech", "science", "health", "finance", "education"]
    metadata = [
        json.dumps(
            {
                "category": categories[i % len(categories)],
                "importance": (i % 10) + 1,
                "source": "nova_test",
            }
        )
        for i in range(num_vectors)
    ]
    versions = [1] * num_vectors
    sources = ["nova_flush"] * num_vectors

    # Build arrays
    arrays = [
        pa.array(ids),
        pa.array(row_group_offsets, type=pa.uint32()),
        pa.array(row_indices, type=pa.uint32()),
        pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.flatten(), type=pa.float32()), list_size=dimension
        ),
        pa.array(timestamps, type=pa.int64()),
        pa.array([None] * num_vectors, type=pa.int64()),  # updated_at
        pa.array([None] * num_vectors, type=pa.int64()),  # expires_at
        pa.array(versions, type=pa.uint32()),
        pa.array(sources),
        pa.array(metadata),
    ]

    if include_quantization:
        # Generate binary quantization (1 bit per dimension)
        binary_vectors = (vectors > 0).astype(np.uint8)
        binary_packed = np.packbits(binary_vectors, axis=1)
        arrays.append(pa.array([bytes(bv) for bv in binary_packed], type=pa.binary()))

        # Generate INT8 quantization
        vec_min = vectors.min(axis=1)
        vec_max = vectors.max(axis=1)
        vec_scale = (vec_max - vec_min) / 255.0
        vec_scale = np.where(vec_scale == 0, 1.0, vec_scale)  # Avoid division by zero
        int8_vectors = (
            (vectors - vec_min[:, np.newaxis]) / vec_scale[:, np.newaxis]
        ).astype(np.uint8)
        arrays.append(pa.array([bytes(v) for v in int8_vectors], type=pa.binary()))
        arrays.append(pa.array(vec_scale, type=pa.float32()))
        arrays.append(pa.array(vec_min, type=pa.float32()))
        arrays.append(pa.array(vec_max, type=pa.float32()))

        # Generate PQ8 placeholder (8 subquantizers)
        pq_codes = np.random.randint(0, 256, size=(num_vectors, 8), dtype=np.uint8)
        arrays.append(pa.array([bytes(c) for c in pq_codes], type=pa.binary()))

    schema = create_proximadb_parquet_schema(dimension, include_quantization)
    table = pa.Table.from_arrays(arrays, schema=schema)

    # Write with ZSTD compression (default for NOVA)
    pq.write_table(table, path, compression=compression)

    return num_vectors, dimension, vectors


def create_nova_parquet_file(
    path: str,
    num_vectors: int = 100,
    dimension: int = 64,
    include_quantization: bool = True,  # the columnar engine typically uses quantization
    compression: str = "zstd",
) -> tuple:
    """Create a test Parquet file matching the NOVA engine format.

    NOVA is columnar Parquet optimized for analytics with progressive search
    (Binary -> INT8 -> FP32) and aggressive quantization.
    """
    # Same schema shape the columnar engines share
    return create_nova_parquet_file(
        path,
        num_vectors=num_vectors,
        dimension=dimension,
        include_quantization=include_quantization,
        compression=compression,
    )


# =============================================================================
# PyArrow Tests
# =============================================================================


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
class TestPyArrowInterop:
    """Test suite for PyArrow reading ProximaDB Parquet files."""

    def test_pyarrow_read_nova_parquet(self):
        """Verify PyArrow can read NOVA Parquet files."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, dimension, _ = create_nova_parquet_file(parquet_path)

            # Read with pq.read_table()
            table = pq.read_table(parquet_path)

            assert len(table) == num_vectors
            assert FIELD_ID in table.schema.names
            assert FIELD_VECTOR_FP32 in table.schema.names
            assert FIELD_TIMESTAMP in table.schema.names

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_pyarrow_read_nova_parquet(self):
        """Verify PyArrow can read NOVA Parquet files."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, dimension, _ = create_nova_parquet_file(parquet_path)

            # Read with pq.read_table()
            table = pq.read_table(parquet_path)

            assert len(table) == num_vectors
            assert FIELD_ID in table.schema.names
            assert FIELD_VECTOR_FP32 in table.schema.names
            # the columnar engine includes quantization columns by default
            assert FIELD_Q_BINARY in table.schema.names
            assert FIELD_Q_INT8 in table.schema.names

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_pyarrow_parquet_file_api(self):
        """Verify PyArrow ParquetFile API works correctly."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, dimension, _ = create_nova_parquet_file(
                parquet_path, num_vectors=500
            )

            # Use ParquetFile for metadata inspection
            pf = pq.ParquetFile(parquet_path)

            # Check metadata
            assert pf.metadata.num_rows == num_vectors
            assert pf.metadata.num_columns >= 10  # At least 10 core columns

            # Check schema
            schema = pf.schema_arrow
            assert FIELD_ID in schema.names
            assert FIELD_VECTOR_FP32 in schema.names

            # Read specific columns
            table = pf.read(columns=[FIELD_ID, FIELD_TIMESTAMP])
            assert len(table) == num_vectors
            assert len(table.schema) == 2

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_pyarrow_row_group_iteration(self):
        """Verify row groups can be read individually."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path, num_vectors=1000)

            pf = pq.ParquetFile(parquet_path)

            total_rows = 0
            for i in range(pf.metadata.num_row_groups):
                rg = pf.read_row_group(i)
                total_rows += len(rg)

            assert total_rows == 1000

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_pyarrow_schema_field_types(self):
        """Verify schema field types match expected types."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            _, dimension, _ = create_nova_parquet_file(parquet_path, dimension=128)

            pf = pq.ParquetFile(parquet_path)
            schema = pf.schema_arrow

            # Verify core field types
            id_field = schema.field(FIELD_ID)
            assert id_field.type == pa.utf8()
            assert not id_field.nullable

            # Verify vector field type
            vector_field = schema.field(FIELD_VECTOR_FP32)
            assert pa.types.is_fixed_size_list(vector_field.type)
            assert vector_field.type.list_size == dimension
            assert vector_field.type.value_type == pa.float32()

            # Verify timestamp field type
            timestamp_field = schema.field(FIELD_TIMESTAMP)
            assert timestamp_field.type == pa.int64()

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)


# =============================================================================
# Vector Extraction Tests
# =============================================================================


@pytest.mark.skipif(
    not PYARROW_AVAILABLE or not NUMPY_AVAILABLE, reason="PyArrow and NumPy required"
)
class TestVectorExtraction:
    """Test vector extraction from Parquet files."""

    def test_extract_vectors_to_numpy(self):
        """Verify vectors can be extracted as NumPy arrays."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, dimension, original_vectors = create_nova_parquet_file(
                parquet_path, num_vectors=50, dimension=64
            )

            table = pq.read_table(parquet_path)

            # Extract vectors column
            vectors_column = table.column(FIELD_VECTOR_FP32)

            # Convert to numpy array
            vector_list = [v.as_py() for v in vectors_column]
            vector_matrix = np.array(vector_list, dtype=np.float32)

            assert vector_matrix.shape == (num_vectors, dimension)
            assert vector_matrix.dtype == np.float32

            # Verify vectors match original (normalized vectors)
            np.testing.assert_allclose(vector_matrix, original_vectors, rtol=1e-5)

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_extract_binary_quantized_vectors(self):
        """Verify binary quantized vectors can be extracted."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, dimension, original_vectors = create_nova_parquet_file(
                parquet_path, num_vectors=50, dimension=64, include_quantization=True
            )

            table = pq.read_table(parquet_path)

            # Extract binary vectors
            q_binary_column = table.column(FIELD_Q_BINARY)

            # Convert to numpy
            binary_packed = np.array(
                [np.frombuffer(v.as_py(), dtype=np.uint8) for v in q_binary_column]
            )

            # Expected packed size
            packed_size = (dimension + 7) // 8
            assert binary_packed.shape == (num_vectors, packed_size)

            # Unpack and verify matches sign of original vectors
            unpacked = np.unpackbits(binary_packed, axis=1)[:, :dimension]
            expected = (original_vectors > 0).astype(np.uint8)
            np.testing.assert_array_equal(unpacked, expected)

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_extract_int8_quantized_vectors(self):
        """Verify INT8 quantized vectors can be extracted and dequantized."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, dimension, _ = create_nova_parquet_file(
                parquet_path, num_vectors=50, dimension=64, include_quantization=True
            )

            table = pq.read_table(parquet_path)

            # Extract INT8 vectors and parameters
            q_int8 = table.column(FIELD_Q_INT8)
            scales = table.column(FIELD_QP_INT8_SCALE).to_numpy()
            mins = table.column(FIELD_QP_INT8_MIN).to_numpy()

            # Convert INT8 to numpy
            int8_vectors = np.array(
                [np.frombuffer(v.as_py(), dtype=np.uint8) for v in q_int8]
            )

            assert int8_vectors.shape == (num_vectors, dimension)
            assert int8_vectors.dtype == np.uint8

            # Dequantize
            dequantized = (
                int8_vectors.astype(np.float32) * scales[:, np.newaxis]
                + mins[:, np.newaxis]
            )

            # Should be similar to original FP32 vectors (with quantization error)
            fp32_column = table.column(FIELD_VECTOR_FP32)
            fp32_vectors = np.array([v.as_py() for v in fp32_column], dtype=np.float32)

            # Allow for quantization error
            assert np.allclose(dequantized, fp32_vectors, atol=0.02)

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_vector_distance_computation(self):
        """Verify we can compute distances using extracted vectors."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, dimension, _ = create_nova_parquet_file(
                parquet_path, num_vectors=100, dimension=64
            )

            table = pq.read_table(parquet_path)

            # Extract vectors
            vectors = np.array(
                [v.as_py() for v in table.column(FIELD_VECTOR_FP32)], dtype=np.float32
            )

            # Compute L2 distances from first vector
            query = vectors[0]
            l2_distances = np.linalg.norm(vectors - query, axis=1)

            assert l2_distances[0] == pytest.approx(0.0, abs=1e-6)  # Distance to self
            assert all(d >= 0 for d in l2_distances)

            # Compute cosine similarities
            query_norm = query / np.linalg.norm(query)
            vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            cosine_similarities = np.dot(vectors_norm, query_norm)

            assert cosine_similarities[0] == pytest.approx(
                1.0, abs=1e-5
            )  # Self similarity
            # Allow small floating point tolerance (values might be slightly outside [-1, 1])
            assert all(-1.001 <= s <= 1.001 for s in cosine_similarities)

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)


# =============================================================================
# DuckDB Tests
# =============================================================================


class TestDuckDBInterop:
    """Test DuckDB interoperability with ProximaDB Parquet files."""

    def test_duckdb_read_nova_parquet(self):
        """Verify DuckDB can read NOVA Parquet files."""
        duckdb = pytest.importorskip("duckdb")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, _, _ = create_nova_parquet_file(parquet_path)

            # DuckDB can read Parquet files directly
            conn = duckdb.connect()
            result = conn.execute(f"SELECT * FROM '{parquet_path}'").fetchall()

            assert len(result) == num_vectors

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_duckdb_read_nova_parquet(self):
        """Verify DuckDB can read NOVA Parquet files with quantization columns."""
        duckdb = pytest.importorskip("duckdb")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, _, _ = create_nova_parquet_file(parquet_path)

            conn = duckdb.connect()
            result = conn.execute(f"SELECT * FROM '{parquet_path}'").fetchall()

            assert len(result) == num_vectors

            # Check columns
            columns = conn.execute(
                f"DESCRIBE SELECT * FROM '{parquet_path}'"
            ).fetchall()
            column_names = [col[0] for col in columns]

            assert FIELD_ID in column_names
            assert FIELD_VECTOR_FP32 in column_names
            assert FIELD_Q_BINARY in column_names
            assert FIELD_Q_INT8 in column_names

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_duckdb_where_clause_filtering(self):
        """Verify DuckDB can filter Parquet data with WHERE clause."""
        duckdb = pytest.importorskip("duckdb")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path, num_vectors=100)

            conn = duckdb.connect()

            # Filter by timestamp
            result = conn.execute(f"""
                SELECT id, timestamp
                FROM '{parquet_path}'
                WHERE timestamp >= 1700000050000
            """).fetchall()

            assert len(result) == 50  # Last 50 vectors
            assert all(row[1] >= 1700000050000 for row in result)

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_duckdb_order_by(self):
        """Verify DuckDB can sort Parquet data with ORDER BY."""
        duckdb = pytest.importorskip("duckdb")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path, num_vectors=50)

            conn = duckdb.connect()

            # Sort by timestamp descending
            result = conn.execute(f"""
                SELECT id, timestamp
                FROM '{parquet_path}'
                ORDER BY timestamp DESC
                LIMIT 10
            """).fetchall()

            assert len(result) == 10
            assert result[0][0] == "nova_vec_00049"  # Highest timestamp
            assert result[-1][0] == "nova_vec_00040"

            # Verify ordering
            timestamps = [row[1] for row in result]
            assert timestamps == sorted(timestamps, reverse=True)

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_duckdb_aggregations(self):
        """Verify DuckDB can perform aggregations on Parquet data."""
        duckdb = pytest.importorskip("duckdb")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path, num_vectors=100)

            conn = duckdb.connect()

            # Test aggregation functions
            result = conn.execute(f"""
                SELECT
                    COUNT(*) as count,
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts,
                    AVG(timestamp) as avg_ts
                FROM '{parquet_path}'
            """).fetchone()

            assert result[0] == 100  # COUNT
            assert result[1] == 1700000000000  # MIN timestamp
            assert result[2] == 1700000099000  # MAX timestamp

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_duckdb_json_metadata_extraction(self):
        """Verify DuckDB can extract data from JSON metadata column."""
        duckdb = pytest.importorskip("duckdb")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path, num_vectors=100)

            conn = duckdb.connect()

            # Extract JSON fields
            result = conn.execute(f"""
                SELECT
                    json_extract_string(extra_meta, '$.category') as category,
                    COUNT(*) as count
                FROM '{parquet_path}'
                GROUP BY json_extract_string(extra_meta, '$.category')
                ORDER BY category
            """).fetchall()

            # 5 categories, 100 vectors -> 20 per category
            assert len(result) == 5
            categories = [row[0] for row in result]
            counts = [row[1] for row in result]
            assert set(categories) == {
                "tech",
                "science",
                "health",
                "finance",
                "education",
            }
            assert all(count == 20 for count in counts)

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)


# =============================================================================
# Polars Tests
# =============================================================================


class TestPolarsInterop:
    """Test Polars interoperability with ProximaDB Parquet files."""

    def test_polars_read_nova_parquet(self):
        """Verify Polars can read NOVA Parquet files."""
        pl = pytest.importorskip("polars")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, _, _ = create_nova_parquet_file(parquet_path)

            # Polars can read Parquet files directly
            df = pl.read_parquet(parquet_path)

            assert len(df) == num_vectors
            assert FIELD_ID in df.columns
            assert FIELD_VECTOR_FP32 in df.columns
            assert FIELD_TIMESTAMP in df.columns

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_polars_read_nova_parquet(self):
        """Verify Polars can read NOVA Parquet files."""
        pl = pytest.importorskip("polars")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, _, _ = create_nova_parquet_file(parquet_path)

            df = pl.read_parquet(parquet_path)

            assert len(df) == num_vectors
            assert FIELD_Q_BINARY in df.columns
            assert FIELD_Q_INT8 in df.columns

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_polars_lazy_scan(self):
        """Verify Polars lazy scan works for efficient query planning."""
        pl = pytest.importorskip("polars")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path, num_vectors=500)

            # Use lazy API for efficient query planning
            lazy_df = pl.scan_parquet(parquet_path)

            # Build and execute lazy query
            result = (
                lazy_df.filter(pl.col(FIELD_TIMESTAMP) >= 1700000250000)
                .select([FIELD_ID, FIELD_TIMESTAMP])
                .collect()
            )

            assert len(result) == 250  # Last 250 vectors
            assert all(ts >= 1700000250000 for ts in result[FIELD_TIMESTAMP].to_list())

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_polars_sorting_and_limiting(self):
        """Verify Polars can sort and limit data."""
        pl = pytest.importorskip("polars")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path, num_vectors=100)

            lazy_df = pl.scan_parquet(parquet_path)

            # Sort by timestamp descending and take top 10
            result = lazy_df.sort(FIELD_TIMESTAMP, descending=True).head(10).collect()

            assert len(result) == 10
            ids = result[FIELD_ID].to_list()
            assert ids[0] == "nova_vec_00099"
            assert ids[-1] == "nova_vec_00090"

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_polars_aggregations(self):
        """Verify Polars can perform aggregations."""
        pl = pytest.importorskip("polars")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path, num_vectors=100)

            df = pl.read_parquet(parquet_path)

            # Basic statistics
            stats = df.select(
                [
                    pl.col(FIELD_TIMESTAMP).count().alias("count"),
                    pl.col(FIELD_TIMESTAMP).min().alias("min_ts"),
                    pl.col(FIELD_TIMESTAMP).max().alias("max_ts"),
                    pl.col(FIELD_TIMESTAMP).mean().alias("avg_ts"),
                ]
            )

            assert stats["count"][0] == 100
            assert stats["min_ts"][0] == 1700000000000
            assert stats["max_ts"][0] == 1700000099000

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_polars_vector_extraction(self):
        """Verify vectors can be extracted from Polars DataFrame."""
        pl = pytest.importorskip("polars")
        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy required for this test")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, dimension, _ = create_nova_parquet_file(
                parquet_path, num_vectors=50, dimension=32
            )

            df = pl.read_parquet(parquet_path)

            # Extract vectors column
            vectors_series = df[FIELD_VECTOR_FP32]

            # Convert to list of lists then to numpy
            vector_list = vectors_series.to_list()
            vector_matrix = np.array(vector_list, dtype=np.float32)

            assert vector_matrix.shape == (num_vectors, dimension)
            assert vector_matrix.dtype == np.float32

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_polars_json_metadata_parsing(self):
        """Verify Polars can parse JSON metadata."""
        pl = pytest.importorskip("polars")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path, num_vectors=50)

            df = pl.read_parquet(parquet_path)

            # Parse JSON and group by category
            result = (
                df.with_columns(
                    pl.col(FIELD_EXTRA_META)
                    .str.json_path_match("$.category")
                    .alias("category")
                )
                .group_by("category")
                .agg(pl.len())
                .sort("category")
            )

            assert len(result) == 5  # 5 categories
            categories = result["category"].to_list()
            assert set(categories) == {
                "tech",
                "science",
                "health",
                "finance",
                "education",
            }
            # Each category should have 10 vectors (50 / 5)
            assert all(count == 10 for count in result["len"].to_list())

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)


# =============================================================================
# Cross-Tool Consistency Tests
# =============================================================================


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow required")
class TestCrossToolConsistency:
    """Verify data consistency across PyArrow, DuckDB, and Polars."""

    def test_row_count_consistency(self):
        """Verify all tools report the same row count."""
        duckdb = pytest.importorskip("duckdb")
        pl = pytest.importorskip("polars")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path, num_vectors=100)

            # PyArrow
            arrow_table = pq.read_table(parquet_path)
            arrow_count = len(arrow_table)

            # DuckDB
            conn = duckdb.connect()
            duck_count = conn.execute(
                f"SELECT COUNT(*) FROM '{parquet_path}'"
            ).fetchone()[0]

            # Polars
            polars_df = pl.read_parquet(parquet_path)
            polars_count = len(polars_df)

            assert arrow_count == duck_count == polars_count == 100

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_column_names_consistency(self):
        """Verify all tools report the same column names."""
        duckdb = pytest.importorskip("duckdb")
        pl = pytest.importorskip("polars")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path)

            # PyArrow
            arrow_table = pq.read_table(parquet_path)
            arrow_columns = set(arrow_table.schema.names)

            # DuckDB
            conn = duckdb.connect()
            duck_desc = conn.execute(
                f"DESCRIBE SELECT * FROM '{parquet_path}'"
            ).fetchall()
            duck_columns = set(col[0] for col in duck_desc)

            # Polars
            polars_df = pl.read_parquet(parquet_path)
            polars_columns = set(polars_df.columns)

            assert arrow_columns == duck_columns == polars_columns

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_id_values_consistency(self):
        """Verify ID values are consistent across tools."""
        duckdb = pytest.importorskip("duckdb")
        pl = pytest.importorskip("polars")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            create_nova_parquet_file(parquet_path, num_vectors=50)

            # PyArrow
            arrow_ids = set(pq.read_table(parquet_path).column(FIELD_ID).to_pylist())

            # DuckDB
            conn = duckdb.connect()
            duck_ids = set(
                row[0]
                for row in conn.execute(
                    f"SELECT {FIELD_ID} FROM '{parquet_path}'"
                ).fetchall()
            )

            # Polars
            polars_ids = set(pl.read_parquet(parquet_path)[FIELD_ID].to_list())

            assert arrow_ids == duck_ids == polars_ids
            assert len(arrow_ids) == 50

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)


# =============================================================================
# Compression Tests
# =============================================================================


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow required")
class TestCompressionFormats:
    """Test different compression formats used by NOVA."""

    @pytest.mark.parametrize("compression", ["zstd", "snappy", "gzip", "none"])
    def test_compression_readable(self, compression):
        """Verify Parquet files with different compressions are readable."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            num_vectors, _, _ = create_nova_parquet_file(
                parquet_path,
                num_vectors=50,
                compression=compression if compression != "none" else None,
            )

            # PyArrow should read any compression
            table = pq.read_table(parquet_path)
            assert len(table) == num_vectors

        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

    def test_zstd_compression_efficiency(self):
        """Verify ZSTD compression (NOVA default) is applied and provides some compression.

        Note: Random float32 vectors don't compress as well as real-world data
        with structure, so we use a modest threshold. Real ProximaDB workloads
        with quantization columns and metadata typically achieve 2-5x compression.
        """
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f2:
            uncompressed_path = f2.name

        try:
            # Create compressed and uncompressed versions
            create_nova_parquet_file(parquet_path, num_vectors=1000, compression="zstd")
            create_nova_parquet_file(
                uncompressed_path, num_vectors=1000, compression=None
            )

            compressed_size = os.path.getsize(parquet_path)
            uncompressed_size = os.path.getsize(uncompressed_path)

            # ZSTD should provide some compression even on random data
            # Random float32 data doesn't compress well, so we use a low threshold
            compression_ratio = uncompressed_size / compressed_size
            assert (
                compression_ratio > 1.0
            ), f"Expected compression ratio > 1.0, got {compression_ratio}"

            # Verify both files are readable
            pq.read_table(parquet_path)
            pq.read_table(uncompressed_path)

        finally:
            for path in [parquet_path, uncompressed_path]:
                if os.path.exists(path):
                    os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
