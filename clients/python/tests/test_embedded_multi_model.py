"""
Integration tests for ProximaDB Embedded Multi-Model Provider.

Tests the embedded multi-model code analysis functionality including:
- Code indexing with multi-model storage
- Code analysis utilities (chunking, metrics extraction)
- Graph operations (create_node, create_edge, graph query)
- Document operations via embedded adapter
- Hybrid search capabilities
- Repository-level batch operations
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from proximadb_sdk import EmbeddedMultiModelProvider


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for embedded database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
async def embedded_provider(temp_data_dir):
    """Create an embedded multi-model provider for testing."""
    provider = EmbeddedMultiModelProvider(
        data_dir=temp_data_dir,
        workspace="test_workspace",
    )

    await provider.initialize()
    yield provider

    await provider.shutdown()


class TestEmbeddedMultiModelProvider:
    """Test suite for EmbeddedMultiModelProvider."""

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self, temp_data_dir):
        """Test initializing and shutting down the embedded provider."""
        provider = EmbeddedMultiModelProvider(
            data_dir=temp_data_dir,
            workspace="test_init",
        )

        await provider.initialize()
        assert provider._is_initialized is True
        assert provider._adapter is not None

        await provider.shutdown()
        assert provider._is_initialized is False

    @pytest.mark.asyncio
    async def test_index_code_file(self, embedded_provider):
        """Test indexing a single code file."""
        code_content = '''
def hello_world():
    """Print a greeting."""
    print("Hello, World!")

class Greeter:
    """A class that greets people."""

    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"
'''

        results = await embedded_provider.index_code_file(
            file_path="test.py",
            content=code_content,
            language="python",
        )

        # Verify results
        assert results is not None
        assert "vectors" in results or "vectors_error" in results
        assert "document" in results or "document_error" in results
        assert "graph" in results or "graph_error" in results
        assert "timeseries" in results or "timeseries_error" in results

    @pytest.mark.asyncio
    async def test_code_chunking(self, embedded_provider):
        """Test code chunking functionality."""
        code = """
def function_one():
    pass

def function_two():
    pass

class MyClass:
    pass
"""

        chunks = embedded_provider._chunk_code(code)

        # Verify chunks were created
        assert len(chunks) > 0

        # Verify chunk structure
        for chunk in chunks:
            assert "content" in chunk
            assert "start_line" in chunk
            assert "end_line" in chunk
            assert "line_count" in chunk

    @pytest.mark.asyncio
    async def test_extract_code_metrics(self, embedded_provider):
        """Test code metrics extraction."""
        code = '''
# This is a comment
def example_function():
    """Example function."""
    if True:
        if nested:
            pass
    return 1

class ExampleClass:
    pass
'''

        metrics = embedded_provider._extract_code_metrics(code, "python")

        # Verify metrics
        assert len(metrics) > 0

        # Check for expected metrics
        metric_names = [m["name"] for m in metrics]
        assert "lines_of_code" in metric_names
        assert "function_count" in metric_names
        assert "class_count" in metric_names
        assert "max_nesting_depth" in metric_names

    @pytest.mark.asyncio
    async def test_find_similar_functions(self, embedded_provider):
        """Test finding similar functions."""
        # First index some code
        code1 = '''
def parse_data(data):
    """Parse input data."""
    return json.loads(data)
'''

        code2 = '''
def parse_input(input_str):
    """Parse user input."""
    return json.loads(input_str)
'''

        code3 = '''
def compute_result(x, y):
    """Compute result."""
    return x + y
'''

        await embedded_provider.index_code_file("file1.py", code1, "python")
        await embedded_provider.index_code_file("file2.py", code2, "python")
        await embedded_provider.index_code_file("file3.py", code3, "python")

        # Find similar functions
        results = await embedded_provider.find_similar_functions(
            code="def parse_data(data):",
            language="python",
            top_k=5,
        )

        # Verify results
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search(self, embedded_provider):
        """Test hybrid search across models."""
        # Index some code first
        code = """
def example_function():
    pass
"""
        await embedded_provider.index_code_file("test.py", code, "python")

        # Perform hybrid search
        results = await embedded_provider.hybrid_search(
            query="example function",
            top_k=10,
        )

        # Verify results
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_index_repository(self, embedded_provider, tmp_path):
        """Test repository-level indexing."""
        # Create temporary files
        (tmp_path / "file1.py").write_text("""
def func_one():
    pass
""")

        (tmp_path / "file2.py").write_text("""
class ClassOne:
    pass
""")

        # Index repository
        results = await embedded_provider.index_repository(
            repo_path=str(tmp_path),
            max_files=10,
        )

        # Verify results
        assert results is not None
        assert "files_processed" in results
        assert results["files_processed"] > 0


class TestEmbeddedAdapterMultiModel:
    """Test suite for EmbeddedProtocolAdapter multi-model methods."""

    @pytest.fixture
    async def embedded_adapter(self, temp_data_dir):
        """Create an embedded adapter for testing."""
        from proximadb_sdk.adapters.embedded_adapter import EmbeddedProtocolAdapter

        adapter = EmbeddedProtocolAdapter(data_dir=temp_data_dir)

        # Create collections
        adapter.create_collection("test_vectors", config={"dimension": 384})

        yield adapter

        adapter.close()

    @pytest.mark.asyncio
    async def test_create_document_collection(self, embedded_adapter):
        """Test creating a document collection."""
        result = embedded_adapter.create_document_collection(
            name="test_docs",
            config={"enable_fulltext": True},
        )

        assert result is not None
        assert "success" in result or "collection_id" in result

    @pytest.mark.asyncio
    async def test_insert_and_get_document(self, embedded_adapter):
        """Test inserting and getting documents."""
        # Create document collection
        embedded_adapter.create_document_collection("test_docs")

        # Insert document
        document = {"title": "Test", "content": "Content here"}
        result = embedded_adapter.insert_document(
            collection_name="test_docs",
            document=document,
            id="doc1",
        )

        assert result is not None
        assert result.get("id") == "doc1"

        # Get document
        doc = embedded_adapter.get_document(
            collection_name="test_docs",
            doc_id="doc1",
        )

        assert doc is not None

    @pytest.mark.asyncio
    async def test_query_documents(self, embedded_adapter):
        """Test querying documents."""
        # Create document collection
        embedded_adapter.create_document_collection("test_docs")

        # Insert documents
        for i in range(5):
            embedded_adapter.insert_document(
                collection_name="test_docs",
                document={"category": "test", "value": i},
                id=f"doc{i}",
            )

        # Query documents
        results = embedded_adapter.query_documents(
            collection_name="test_docs",
            filter={"category": "test"},
            limit=10,
        )

        assert results is not None

    @pytest.mark.asyncio
    async def test_update_and_delete_document(self, embedded_adapter):
        """Test updating and deleting documents."""
        # Create document collection
        embedded_adapter.create_document_collection("test_docs")

        # Insert document
        embedded_adapter.insert_document(
            collection_name="test_docs",
            document={"value": 1},
            id="doc1",
        )

        # Update document
        result = embedded_adapter.update_document(
            collection_name="test_docs",
            doc_id="doc1",
            updates=[{"operation": "SET", "path": "$.value", "value": 2}],
        )

        assert result is not None

        # Delete document
        deleted = embedded_adapter.delete_document(
            collection_name="test_docs",
            doc_id="doc1",
        )

        # Verify deletion
        assert deleted is True or deleted.get("deleted") is True

    @pytest.mark.asyncio
    async def test_create_node_and_edge(self, embedded_adapter):
        """Test graph node and edge creation."""
        # Create a node
        node_result = embedded_adapter.create_node(
            graph="test_graph",
            node_id="node1",
            labels=["Function", "python"],
            properties={"name": "example", "file": "test.py"},
        )

        assert node_result is not None
        assert "success" in node_result or "node_id" in node_result

        # Create an edge
        edge_result = embedded_adapter.create_edge(
            graph="test_graph",
            edge_id="edge1",
            from_node="node1",
            to_node="node2",
            edge_type="CALLS",
            properties={"line": 42},
        )

        assert edge_result is not None
        assert "success" in edge_result or "edge_id" in edge_result

    @pytest.mark.asyncio
    async def test_graph_query(self, embedded_adapter):
        """Test graph query execution."""
        # Create a node first
        embedded_adapter.create_node(
            graph="test_graph",
            node_id="node1",
            labels=["Function"],
            properties={"name": "example"},
        )

        # Execute graph query
        result = embedded_adapter.execute_graph_query(
            graph="test_graph",
            query="MATCH (n:Function) WHERE n.name = 'example' RETURN n",
        )

        assert result is not None
        assert "results" in result or "query" in result

    @pytest.mark.asyncio
    async def test_create_timeseries_collection(self, embedded_adapter):
        """Test creating a time-series collection."""
        result = embedded_adapter.create_timeseries_collection(
            name="test_metrics",
            config={
                "timestamp_column": "timestamp",
                "value_columns": [
                    {"name": "cpu", "data_type": "float", "aggregation": "avg"},
                ],
            },
        )

        assert result is not None
        assert "success" in result or "collection_id" in result

    @pytest.mark.asyncio
    async def test_ingest_and_query_timeseries(self, embedded_adapter):
        """Test time-series data ingestion and query."""
        from datetime import datetime, timedelta

        # Create time-series collection
        embedded_adapter.create_timeseries_collection("test_metrics")

        # Ingest data points
        now = datetime.utcnow()
        points = []
        for i in range(10):
            timestamp = now + timedelta(seconds=i)
            points.append(
                {
                    "timestamp": timestamp.isoformat() + "Z",
                    "values": {"cpu": 50.0 + i},
                    "tags": {"host": "server1"},
                }
            )

        result = embedded_adapter.ingest_timeseries(
            collection_name="test_metrics",
            points=points,
        )

        assert result is not None
        assert "ingested_count" in result or result.get("success") is True

        # Query time-series data
        start_time = (now - timedelta(minutes=1)).isoformat() + "Z"
        end_time = (now + timedelta(minutes=1)).isoformat() + "Z"

        # TD-EMB-1: the engine has no timeseries surface, so the vector
        # fallback runs and EXPLICIT aggregation raises the honest
        # capability error (tags-only fallback).
        query_result = embedded_adapter.query_timeseries(
            collection_name="test_metrics",
            start_time=start_time,
            end_time=end_time,
        )

        assert query_result is not None
        with pytest.raises(NotImplementedError, match="tag_filters"):
            embedded_adapter.query_timeseries(
                collection_name="test_metrics",
                start_time=start_time,
                end_time=end_time,
                aggregation="AVG",
            )

    @pytest.mark.asyncio
    async def test_list_collections(self, embedded_adapter):
        """Test listing multi-model collections."""
        # Create collections
        embedded_adapter.create_document_collection("docs")
        embedded_adapter.create_timeseries_collection("metrics")

        # List document collections
        doc_collections = embedded_adapter.list_document_collections()
        assert isinstance(doc_collections, list)

        # List time-series collections
        ts_collections = embedded_adapter.list_timeseries_collections()
        assert isinstance(ts_collections, list)

    @pytest.mark.asyncio
    async def test_hybrid_search(self, embedded_adapter):
        """Test hybrid search functionality."""
        # Create a vector collection and insert some data
        from proximadb_sdk.models import VectorRecord

        embedded_adapter.create_collection("test_vectors", config={"dimension": 384})

        # Insert vectors
        vectors = [
            VectorRecord(
                id="doc1",
                vector=[0.1] * 384,
                metadata={"content": "test content 1"},
            ),
            VectorRecord(
                id="doc2",
                vector=[0.2] * 384,
                metadata={"content": "test content 2"},
            ),
        ]
        embedded_adapter.insert_vectors("test_vectors", vectors)

        # Perform hybrid search
        result = embedded_adapter.hybrid_search(
            collection="test_vectors",
            text_query="test",
            query_vector=[0.1] * 384,
            fusion_strategy="rrf",
            top_k=5,
        )

        assert result is not None
        assert "results" in result or "fusion_strategy" in result


class TestEmbeddedProviderCodeAnalysis:
    """Test suite for code analysis features."""

    @pytest.fixture
    async def provider(self, temp_data_dir):
        """Create a provider instance."""
        provider = EmbeddedMultiModelProvider(
            data_dir=temp_data_dir,
            workspace="code_analysis",
        )
        await provider.initialize()
        yield provider
        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_python_code_indexing(self, provider):
        """Test indexing Python code."""
        python_code = '''
"""Example Python module."""

def process_data(items):
    """Process a list of items."""
    results = []
    for item in items:
        if item.is_valid():
            results.append(item.transform())
    return results


class DataProcessor:
    """Process data in different ways."""

    def __init__(self, config):
        self.config = config

    def run(self, data):
        """Run the processor."""
        return self.config.apply(data)
'''

        results = await provider.index_code_file(
            file_path="example.py",
            content=python_code,
            language="python",
        )

        # Verify results
        assert "vectors" in results
        assert "graph" in results
        assert results["graph"]["functions"] >= 2  # process_data, run

    @pytest.mark.asyncio
    async def test_javascript_code_indexing(self, provider):
        """Test indexing JavaScript code."""
        js_code = """
// Example JavaScript module
function processData(items) {
    return items.map(item => item.value);
}

class DataProcessor {
    constructor(config) {
        this.config = config;
    }

    process(data) {
        return this.config.apply(data);
    }
}
"""

        results = await provider.index_code_file(
            file_path="example.js",
            content=js_code,
            language="javascript",
        )

        # Verify results
        assert "vectors" in results
        assert "graph" in results

    @pytest.mark.asyncio
    async def test_complexity_metrics(self, provider):
        """Test code complexity metrics extraction."""
        complex_code = """
def complex_function(x):
    if x > 0:
        if x > 10:
            for i in range(10):
                if i % 2 == 0:
                    yield i
                else:
                    yield -i
    elif x < 0:
        return -x
    else:
        return 0
"""

        metrics = provider._extract_code_metrics(complex_code, "python")

        # Find max nesting depth metric
        nesting_metric = next(
            (m for m in metrics if m["name"] == "max_nesting_depth"), None
        )
        assert nesting_metric is not None
        assert nesting_metric["value"] > 3  # Should detect deep nesting


# Fixtures
@pytest.fixture
def tmp_path(temp_data_dir):
    """Create a temporary path for test files."""
    path = Path(temp_data_dir) / "test_repo"
    path.mkdir(exist_ok=True)
    return path


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
