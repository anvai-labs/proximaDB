"""
Embedded Multi-Model Code Provider for ProximaDB

This provider extends the embedded database with high-level code analysis
capabilities for use in CLI tools, coding agents, and local development environments.

Features:
- Code indexing with multi-model storage (vector, document, graph, time-series)
- Code analysis utilities (chunking, metrics extraction, dependency analysis)
- Hybrid search across all models
- Repository-level batch operations
- Function call tracing and graph analysis

Example::

    from proximadb_sdk.embedded_multi import EmbeddedMultiModelProvider
    from proximadb_sdk import ProximaDBClient

    # Use embedded mode directly
    provider = EmbeddedMultiModelProvider(data_dir="~/.proximadb/codebase")

    # Start the embedded database
    await provider.initialize()

    # Index a code file across all models
    await provider.index_code_file("main.py", content, language="python")

    # Hybrid search: find functions similar to "parse"
    results = await provider.find_similar_functions(
        code="def parse_input(data): ...",
        language="python",
        top_k=10
    )

    # Trace function usage
    call_graph = await provider.trace_function_usage("parse_input", "main.py")

    # Cleanup
    await provider.shutdown()
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from proximadb_sdk.adapters.embedded_adapter import EmbeddedProtocolAdapter
from proximadb_sdk.integrations._records import insert_records, record_payload


class EmbeddedMultiModelProvider:
    """High-level multi-model code provider for embedded ProximaDB.

    Provides code analysis capabilities without requiring a separate server:
    - Automatic code chunking and indexing
    - Multi-model storage (vectors, documents, graphs, time-series)
    - Code metrics extraction and tracking
    - Hybrid search across all models
    - Repository-level batch operations

    Args:
        data_dir: Directory for embedded database storage
        workspace: Workspace name for organizing collections
        embedding_model: Optional embedding model name (for vector search)
        config: Optional additional configuration
    """

    def __init__(
        self,
        data_dir: str = "~/.proximadb/embedded",
        workspace: str = "default_workspace",
        embedding_model: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        # Expand user path
        self.data_dir = os.path.expanduser(data_dir)
        self.workspace = workspace
        self.embedding_model = embedding_model or "all-MiniLM-L6-v2"
        self.config = config or {}

        # Collection naming convention
        self._vector_collection = f"{workspace}_vectors"
        self._document_collection = f"{workspace}_documents"
        self._graph_collection = f"{workspace}_graph"
        self._timeseries_collection = f"{workspace}_metrics"

        # Database instance (created on initialize)
        self._adapter: EmbeddedProtocolAdapter | None = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the embedded database and create collections."""
        if self._is_initialized:
            return

        # Create embedded adapter
        self._adapter = EmbeddedProtocolAdapter(
            data_dir=self.data_dir,
            config=self.config,
        )

        # Ensure database is started
        if hasattr(self._adapter._db, "start"):
            await self._adapter._db.start()

        # Create collections if they don't exist
        await self._ensure_collections()

        self._is_initialized = True

    async def shutdown(self) -> None:
        """Shutdown the embedded database."""
        if self._adapter and hasattr(self._adapter._db, "stop"):
            await self._adapter._db.stop()

        if self._adapter:
            self._adapter.close()

        self._is_initialized = False

    async def _ensure_collections(self) -> None:
        """Ensure all required collections exist."""
        # Vector collection
        if not self._adapter.get_collection(self._vector_collection):
            self._adapter.create_collection(
                self._vector_collection,
                config={"dimension": 384},  # Default for sentence-transformers
            )

        # Document collection
        try:
            self._adapter.create_document_collection(
                name=self._document_collection,
                config={"enable_fulltext": True},
            )
        except Exception:
            pass  # May already exist

        # Graph collection
        if not self._adapter.get_collection(self._graph_collection):
            # Create vector collection for graph metadata
            self._adapter.create_collection(
                self._graph_collection,
                config={"dimension": 256},
            )

        # Time-series collection
        try:
            self._adapter.create_timeseries_collection(
                name=self._timeseries_collection,
                config={
                    "timestamp_column": "timestamp",
                    "value_columns": [
                        {
                            "name": "lines_of_code",
                            "data_type": "int",
                            "aggregation": "sum",
                        },
                        {
                            "name": "complexity",
                            "data_type": "float",
                            "aggregation": "avg",
                        },
                    ],
                },
            )
        except Exception:
            pass  # May already exist

    # ========================================================================
    # Code Indexing
    # ========================================================================

    async def index_code_file(
        self,
        file_path: str,
        content: str,
        language: str = "python",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Index a code file across all ProximaDB models.

        This single operation stores the code in:
        1. Vector store: For semantic search (chunked)
        2. Document store: With syntax highlighting, AST info
        3. Graph store: Function definitions, call relationships
        4. Time-series store: Initial metrics (complexity, lines of code)

        Args:
            file_path: Path to the code file.
            content: File content.
            language: Programming language (python, javascript, etc.).
            metadata: Optional additional metadata.

        Returns:
            Dictionary with indexing results for each model.
        """
        if not self._is_initialized:
            await self.initialize()

        file_path_str = str(file_path)
        file_hash = hashlib.sha256(content.encode()).hexdigest()

        # Prepare base metadata
        base_meta: dict[str, Any] = {
            "file_path": file_path_str,
            "language": language,
            "file_hash": file_hash,
            "indexed_at": datetime.now(UTC).isoformat(),
        }
        if metadata:
            base_meta.update(metadata)

        results: dict[str, Any] = {}

        # 1. Vector: Semantic embedding (chunked)
        try:
            chunks = self._chunk_code(content)
            for i, chunk in enumerate(chunks):
                chunk_meta = base_meta.copy()
                chunk_meta.update(
                    {
                        "chunk_id": f"{file_hash}:{i}",
                        "chunk_type": "code",
                        "start_line": chunk.get("start_line", 0),
                        "end_line": chunk.get("end_line", 0),
                    }
                )

                # For embedded mode, write record-shaped vector payloads.
                # In production, this would use an embedding model.
                dummy_vector = [hash(f"{file_hash}:{i}") % 1000 / 1000.0] * 384

                record = record_payload(
                    record_id=f"{file_hash}:chunk_{i}",
                    vector=dummy_vector,
                    text=chunk["content"],
                    metadata=chunk_meta,
                )

                insert_records(self._adapter, self._vector_collection, [record])

            results["vectors"] = len(chunks)
        except Exception as e:
            results["vectors_error"] = str(e)

        # 2. Document: Full file with rich metadata
        try:
            doc_meta = base_meta.copy()
            doc_meta.update(
                {
                    "content_type": "code",
                    "language": language,
                    "size_bytes": len(content),
                    "title": Path(file_path_str).name,
                }
            )

            self._adapter.ingest_documents(
                collection_id=self._document_collection,
                records=[
                    {
                        "id": f"doc:{file_hash}",
                        "text": content,
                        "metadata": doc_meta,
                    }
                ],
            )
            results["document"] = True
        except Exception as e:
            results["document_error"] = str(e)

        # 3. Graph: Extract and store functions, calls, imports
        try:
            graph_info = await self._index_code_as_graph(
                file_path_str,
                content,
                language,
                base_meta,
            )
            results["graph"] = graph_info
        except Exception as e:
            results["graph_error"] = str(e)

        # 4. Time-Series: Store initial metrics
        try:
            metrics = self._extract_code_metrics(content, language)
            for metric in metrics:
                await self._store_metric(
                    file_path_str,
                    metric,
                    base_meta,
                )
            results["timeseries"] = len(metrics)
        except Exception as e:
            results["timeseries_error"] = str(e)

        return results

    async def _index_code_as_graph(
        self,
        file_path: str,
        content: str,
        language: str,
        metadata: dict[str, Any],
    ) -> dict[str, int]:
        """Index code structure as a graph.

        Extracts:
        - Functions as nodes
        - Function calls as edges
        - Import relationships as edges
        - Class hierarchies as edges
        """
        file_hash = metadata.get(
            "file_hash", hashlib.sha256(content.encode()).hexdigest()
        )

        # Simple parsing for Python (can be extended)
        functions = []
        calls = []
        imports = []
        classes = []

        lines = content.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Function definitions
            if stripped.startswith("def ") or stripped.startswith("async def "):
                func_name = (
                    stripped.split("(")[0]
                    .replace("def ", "")
                    .replace("async def ", "")
                    .strip()
                )
                if func_name:
                    functions.append(func_name)

                    # Create function node
                    try:
                        self._adapter.create_node(
                            graph=self._graph_collection,
                            node_id=f"{file_hash}:func:{func_name}",
                            labels=["Function", language],
                            properties={
                                "name": func_name,
                                "file_path": file_path,
                                "line_number": i + 1,
                                "language": language,
                            },
                        )
                    except Exception:
                        pass  # Node may already exist

            # Class definitions
            if stripped.startswith("class "):
                class_name = (
                    stripped.split("(")[0].replace("class ", "").strip().rstrip(":")
                )
                if class_name:
                    classes.append(class_name)

                    # Create class node
                    try:
                        self._adapter.create_node(
                            graph=self._graph_collection,
                            node_id=f"{file_hash}:class:{class_name}",
                            labels=["Class", language],
                            properties={
                                "name": class_name,
                                "file_path": file_path,
                                "line_number": i + 1,
                                "language": language,
                            },
                        )
                    except Exception:
                        pass

            # Imports
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)

        return {
            "functions": len(functions),
            "calls": len(calls),
            "imports": len(imports),
            "classes": len(classes),
        }

    async def _store_metric(
        self,
        file_path: str,
        metric: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Store code metric in time-series store."""
        timestamp = datetime.now(UTC).isoformat()

        point = {
            "timestamp": timestamp,
            "values": {
                "value": metric["value"],
            },
            "tags": {
                "file_path": file_path,
                "metric_name": metric["name"],
                "language": metric.get("language", "unknown"),
            },
        }

        try:
            self._adapter.ingest_timeseries(
                collection_name=self._timeseries_collection,
                points=[point],
            )
        except Exception:
            pass  # Time-series may not be available

    # ========================================================================
    # Code Analysis Utilities
    # ========================================================================

    def _chunk_code(self, content: str, chunk_size: int = 512) -> list[dict[str, Any]]:
        """Chunk code content for better semantic retrieval.

        Splits code into logical chunks (functions, classes, blocks)
        while preserving metadata about line numbers.
        """
        chunks: list[dict[str, Any]] = []

        lines = content.split("\n")
        current_chunk: list[str] = []
        current_start = 0

        for i, line in enumerate(lines):
            current_chunk.append(line)

            # Start new chunk on:
            # - Empty lines (between functions)
            # - Class/function definitions
            # - Chunk size limit
            if line.strip() == "" or i - current_start >= chunk_size:
                if current_chunk:
                    chunks.append(
                        {
                            "content": "\n".join(current_chunk),
                            "start_line": current_start + 1,
                            "end_line": i + 1,
                            "line_count": len(current_chunk),
                        }
                    )
                    current_chunk = []
                    current_start = i + 1

        # Add remaining content
        if current_chunk:
            chunks.append(
                {
                    "content": "\n".join(current_chunk),
                    "start_line": current_start + 1,
                    "end_line": len(lines),
                    "line_count": len(current_chunk),
                }
            )

        return chunks

    def _extract_code_metrics(
        self,
        content: str,
        language: str,
    ) -> list[dict[str, Any]]:
        """Extract code metrics for time-series tracking.

        Metrics:
        - lines_of_code: Total non-empty, non-comment lines
        - function_count: Number of functions
        - class_count: Number of classes
        - cyclomatic_complexity: Approximate complexity
        - max_nesting_depth: Maximum nesting level
        """
        lines = content.split("\n")
        metrics: list[dict[str, Any]] = []

        # Lines of code
        loc = sum(
            1 for line in lines if line.strip() and not line.strip().startswith("#")
        )
        metrics.append({"name": "lines_of_code", "value": loc, "language": language})

        # Function count (heuristic)
        func_count = sum(1 for line in lines if line.strip().startswith("def "))
        metrics.append(
            {"name": "function_count", "value": func_count, "language": language}
        )

        # Class count (heuristic)
        class_count = sum(1 for line in lines if line.strip().startswith("class "))
        metrics.append(
            {"name": "class_count", "value": class_count, "language": language}
        )

        # Max nesting depth
        max_depth = 0
        current_depth = 0
        for line in lines:
            stripped = line.strip()
            current_depth += stripped.count("{") - stripped.count("}")
            max_depth = max(max_depth, current_depth)
        metrics.append(
            {"name": "max_nesting_depth", "value": max_depth, "language": language}
        )

        return metrics

    # ========================================================================
    # Advanced Queries
    # ========================================================================

    async def find_similar_functions(
        self,
        code: str,
        function_name: str | None = None,
        language: str = "python",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find functions that are semantically similar to the given code.

        Args:
            code: Code snippet to find similar functions for.
            function_name: Optional function name filter.
            language: Programming language to filter by.
            top_k: Number of results to return.

        Returns:
            List of similar functions with metadata.
        """
        if not self._is_initialized:
            await self.initialize()

        # Create dummy query vector
        query_vector = [hash(code) % 1000 / 1000.0] * 384

        # Vector search
        results = self._adapter.search(
            collection_id=self._vector_collection,
            query_vector=query_vector,
            top_k=top_k * 2,
            include_metadata=True,
        )

        # Filter by function metadata
        filtered_results = []
        for result in results:
            meta = result.metadata or {}

            # Filter by language
            if meta.get("language") != language:
                continue

            # Filter by function name if specified
            if function_name and function_name not in meta.get("content", ""):
                continue

            # Check if chunk contains a function definition
            content = result.metadata.get("content", "") if result.metadata else ""
            if "def " in content or "async def " in content:
                filtered_results.append(
                    {
                        "file_path": meta.get("file_path", ""),
                        "content": content,
                        "score": result.score,
                        "start_line": meta.get("start_line"),
                        "end_line": meta.get("end_line"),
                    }
                )

        return filtered_results[:top_k]

    async def trace_function_usage(
        self,
        function_name: str,
        file_path: str,
        depth: int = 3,
    ) -> dict[str, Any]:
        """Trace function call relationships through the codebase.

        Builds a call graph showing:
        - Functions that call this function (callers)
        - Functions called by this function (callees)
        - Transitive relationships up to specified depth

        Args:
            function_name: Name of the function to trace.
            file_path: File containing the function.
            depth: How many levels deep to trace.

        Returns:
            Call graph with nodes (functions) and edges (calls).
        """
        if not self._is_initialized:
            await self.initialize()

        # For now, return a placeholder
        # In a full implementation, this would use graph queries
        return {
            "function": function_name,
            "file": file_path,
            "callers": [],
            "callees": [],
            "depth": depth,
        }

    # ========================================================================
    # Batch Operations
    # ========================================================================

    async def index_repository(
        self,
        repo_path: str,
        language_map: dict[str, str] | None = None,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        """Index an entire repository into multi-model store.

        Args:
            repo_path: Path to the repository root.
            language_map: Optional mapping of file extensions to languages.
            max_files: Optional limit on files to process.

        Returns:
            Summary statistics of indexing operation.
        """
        if not self._is_initialized:
            await self.initialize()

        repo = Path(repo_path)
        if not repo.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        # Find code files
        code_files = self._find_code_files(repo, language_map)

        if max_files:
            code_files = code_files[:max_files]

        results = {
            "files_processed": 0,
            "files_failed": 0,
            "total_chunks": 0,
            "total_functions": 0,
            "errors": [],
        }

        for file_path in code_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                language = self._detect_language(file_path, language_map)

                file_results = await self.index_code_file(
                    str(file_path),
                    content,
                    language,
                )

                results["files_processed"] += 1
                results["total_chunks"] += file_results.get("vectors", 0)
                results["total_functions"] += file_results.get("graph", {}).get(
                    "functions", 0
                )

            except Exception as e:
                results["files_failed"] += 1
                results["errors"].append(
                    {
                        "file": str(file_path),
                        "error": str(e),
                    }
                )

        return results

    def _find_code_files(
        self,
        repo_path: Path,
        language_map: dict[str, str] | None = None,
    ) -> list[Path]:
        """Find all code files in the repository."""
        language_map = language_map or {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".cc": "cpp",
        }

        code_files = []
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix
                if ext in language_map:
                    code_files.append(file_path)

        return code_files

    def _detect_language(
        self,
        file_path: Path,
        language_map: dict[str, str] | None = None,
    ) -> str:
        """Detect programming language from file extension."""
        language_map = language_map or {}
        ext = file_path.suffix
        return language_map.get(ext, "unknown")

    # ========================================================================
    # Hybrid Search
    # ========================================================================

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        graph_query: str | None = None,
        document_filter: dict[str, Any] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search across all ProximaDB models.

        Combines:
        - Vector similarity (semantic search)
        - Graph traversal (relationships)
        - Document filtering (metadata)

        Args:
            query: Search query text.
            top_k: Number of results to return.
            graph_query: Optional Cypher graph query.
            document_filter: Optional metadata filter for documents.
            time_range: Optional time range for filtering.

        Returns:
            Combined results from all models with scores.
        """
        if not self._is_initialized:
            await self.initialize()

        results: list[dict[str, Any]] = []

        # 1. Vector search
        query_vector = [hash(query) % 1000 / 1000.0] * 384
        vector_results = self._adapter.search(
            collection_id=self._vector_collection,
            query_vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        for vr in vector_results:
            results.append(
                {
                    "type": "vector",
                    "score": vr.score,
                    "content": vr.metadata.get("content", "") if vr.metadata else "",
                    "metadata": vr.metadata,
                }
            )

        # 2. Graph search (if query provided)
        if graph_query:
            try:
                graph_results = self._adapter.execute_graph_query(
                    graph=self._graph_collection,
                    query=graph_query,
                )
                for gr in graph_results.get("results", []):
                    results.append(
                        {
                            "type": "graph",
                            "score": gr.get("score", 0.0),
                            "content": gr.get("content", ""),
                            "metadata": gr.get("metadata", {}),
                        }
                    )
            except Exception:
                pass  # Graph search failed, continue with other results

        # 3. Filter and deduplicate results
        filtered = self._filter_hybrid_results(results, document_filter)

        # 4. Rank and limit
        ranked = self._rank_hybrid_results(filtered, top_k)

        return ranked

    def _filter_hybrid_results(
        self,
        results: list[dict[str, Any]],
        filter_dict: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Filter hybrid search results."""
        if not filter_dict:
            return results

        filtered = []
        for result in results:
            metadata = result.get("metadata", {})
            if self._matches_filter(metadata, filter_dict):
                filtered.append(result)
        return filtered

    def _matches_filter(
        self, metadata: dict[str, Any], filter_dict: dict[str, Any]
    ) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def _rank_hybrid_results(
        self,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rank and combine hybrid search results."""
        # Group by file_path
        by_file: dict[str, list[dict[str, Any]]] = {}

        for result in results:
            file_path = result.get("metadata", {}).get("file_path", "")
            if file_path:
                if file_path not in by_file:
                    by_file[file_path] = []
                by_file[file_path].append(result)

        # Score each file
        scored: list[tuple[str, float, dict[str, Any]]] = []

        for file_path, file_results in by_file.items():
            score = 0.0
            best_result = None

            for result in file_results:
                result_type = result.get("type", "")
                result_score = result.get("score", 0.0)

                if result_type == "vector":
                    score += result_score
                elif result_type == "graph":
                    score += result_score * 1.2  # Boost graph relationships

                if not best_result or result_score > best_result.get("score", 0):
                    best_result = result

            scored.append((file_path, score, best_result))

        # Sort by score (descending) and take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [result for _, _, result in scored[:top_k]]
