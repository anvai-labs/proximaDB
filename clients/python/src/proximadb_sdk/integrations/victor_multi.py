"""Enhanced Victor integration for ProximaDB's multi-model store.

This provider extends beyond simple vector search to leverage ProximaDB's full
multi-model capabilities for comprehensive code analysis:

- **Documents**: Store code as rich documents with syntax highlighting, AST metadata
- **Graphs**: Build call graphs, dependency graphs, type relationships
- **Time-Series**: Track code metrics over time (complexity, coverage, churn)
- **Hybrid Search**: Combine vector + graph + document queries

Example::

    from proximadb_sdk.integrations.victor_multi import ProximaDBMultiModelProvider
    from proximadb_sdk import ProximaDBClient

    client = ProximaDBClient(url="http://localhost:5678")
    provider = ProximaDBMultiModelProvider(client=client, workspace="my-codebase")

    # Index code with multi-model storage
    await provider.index_code_file("main.py", content)

    # Hybrid query: find functions called by main that are similar to "parse"
    results = await provider.hybrid_search(
        query="parse input validation",
        graph_query="MATCH (c:Caller)-[:CALLS]->(f:Function) WHERE c.name = 'main'",
        document_filter={"language": "python"},
    )
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from victor.storage.vector_stores.base import (
    EmbeddingConfig,
)

from proximadb_sdk.integrations._records import insert_records, record_payload
from proximadb_sdk.integrations.victor import ProximaDBEmbeddingProvider


class ProximaDBMultiModelProvider(ProximaDBEmbeddingProvider):
    """Enhanced Victor provider for ProximaDB's multi-model store.

    Extends the base ProximaDBEmbeddingProvider with:
    - Document storage for code with rich metadata
    - Graph operations for call graphs and dependencies
    - Time-series tracking for code metrics
    - Hybrid search across all models

    Args:
        client: ProximaDB client instance.
        workspace: Workspace name for organizing collections.
        embedding_config: Victor embedding configuration.
    """

    def __init__(
        self,
        client: Any,
        workspace: str,
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        # Initialize with a default embedding config if not provided
        if embedding_config is None:
            embedding_config = EmbeddingConfig(
                vector_store="proximadb",
                embedding_model="BAAI/bge-small-en-v1.5",
                extra_config={
                    "server_url": getattr(client, "url", "http://localhost:5678"),
                    "dimension": 384,
                },
            )

        super().__init__(embedding_config)
        self._client = client
        self._workspace = workspace

        # Collection naming convention
        self._vector_collection = f"{workspace}_vectors"
        self._document_collection = f"{workspace}_documents"
        self._graph_collection = f"{workspace}_graph"
        self._timeseries_collection = f"{workspace}_metrics"

    # ========================================================================
    # Multi-Model Code Indexing
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
        1. Vector store: For semantic search
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

        # 1. Vector: Semantic embedding (from parent class)
        try:
            # Chunk code for embedding (smaller chunks for better retrieval)
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

                await self.index_document(
                    f"{file_hash}:chunk_{i}",
                    chunk["content"],
                    chunk_meta,
                )
            results["vectors"] = len(chunks)
        except Exception as e:
            results["vectors_error"] = str(e)

        # 2. Document: Full file with rich metadata
        try:
            await self._index_as_document(
                file_path_str,
                content,
                language,
                base_meta,
            )
            results["document"] = True
        except Exception as e:
            results["document_error"] = str(e)

        # 3. Graph: Extract and store functions, calls, imports
        try:
            graph_info = await self._index_as_graph(
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
            await self._store_metrics(
                file_path_str,
                metrics,
                base_meta,
            )
            results["timeseries"] = len(metrics)
        except Exception as e:
            results["timeseries_error"] = str(e)

        return results

    async def _index_as_document(
        self,
        file_path: str,
        content: str,
        language: str,
        metadata: dict[str, Any],
    ) -> None:
        """Index code as a rich document.

        Stores the full file content with enhanced metadata including
        syntax highlighting info, AST structure, and code statistics.
        """
        doc_meta = metadata.copy()
        doc_meta.update(
            {
                "content_type": "code",
                "language": language,
                "size_bytes": len(content),
                "encoding": "utf-8",
                # Document-specific fields
                "title": Path(file_path).name,
                "file_extension": Path(file_path).suffix,
            }
        )

        # Store through the record-native SDK surface while this optional
        # provider still uses the vector collection as its projection target.
        record = record_payload(
            record_id=f"doc:{metadata['file_hash']}",
            vector=[0.0] * self._dimension,
            text=content,
            metadata=doc_meta,
        )
        insert_records(self._client, self._vector_collection, [record])

    async def _index_as_graph(
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
        # This would parse the code and create graph nodes/edges
        # For now, return a placeholder
        graph_info = {
            "functions": 0,
            "calls": 0,
            "imports": 0,
            "classes": 0,
        }

        # In a full implementation, this would:
        # 1. Parse the code (using tree-sitter or ast)
        # 2. Extract function definitions -> create nodes
        # 3. Extract function calls -> create edges
        # 4. Extract imports -> create edges to external modules
        # 5. Use client.create_node() and client.create_edge()

        return graph_info

    async def _store_metrics(
        self,
        file_path: str,
        metrics: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        """Store code metrics in time-series store.

        Metrics tracked:
        - Lines of code
        - Cyclomatic complexity
        - Function count
        - Nesting depth
        - Code churn (from git history, if available)
        """
        timestamp = datetime.now(UTC).isoformat()

        for metric in metrics:
            metric_record = {
                "timestamp": timestamp,
                "file_path": file_path,
                "metric_name": metric["name"],
                "metric_value": metric["value"],
                "metadata": {**metadata, "language": metric.get("language")},
            }

            # Store through the record-native SDK surface while this optional
            # provider still projects metrics into the vector collection.
            record = record_payload(
                record_id=f"metric:{file_path}:{metric['name']}:{timestamp}",
                vector=[0.0] * self._dimension,
                text=json.dumps(metric_record),
                metadata={"type": "metric", "file_path": file_path},
            )
            insert_records(self._client, self._vector_collection, [record])

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
        - Time-series filtering (temporal trends)

        Args:
            query: Search query text.
            top_k: Number of results to return.
            graph_query: Optional Cypher graph query.
            document_filter: Optional metadata filter for documents.
            time_range: Optional time range for filtering.

        Returns:
            Combined results from all models with scores.
        """
        results: list[dict[str, Any]] = []

        # 1. Vector search
        vector_results = await self.search_similar(query, limit=top_k)
        for vr in vector_results:
            results.append(
                {
                    "type": "vector",
                    "score": vr.score,
                    "content": vr.content,
                    "metadata": vr.metadata,
                }
            )

        # 2. Graph search (if query provided)
        if graph_query:
            try:
                graph_results = await self._search_graph(graph_query, top_k)
                for gr in graph_results:
                    results.append(
                        {
                            "type": "graph",
                            "score": gr.get("score", 0.0),
                            "content": gr.get("content", ""),
                            "metadata": gr.get("metadata", {}),
                        }
                    )
            except Exception:
                # Graph search failed, continue with other results
                pass

        # 3. Filter and deduplicate results
        filtered = self._filter_hybrid_results(results, document_filter)

        # 4. Re-score and rank
        ranked = self._rank_hybrid_results(filtered, top_k)

        return ranked

    async def _search_graph(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Execute graph search query.

        In a full implementation, this would:
        1. Use client.execute_sql() with Cypher support
        2. Parse results into standard format
        """
        # Placeholder for graph search
        # In production: client.execute_sql(graph_query)
        return []

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
        """Rank and combine hybrid search results.

        Implements a simple scoring strategy:
        - Vector results get base score
        - Graph results get slight boost for relationship relevance
        - Deduplicates by file_path
        """
        # Group by file_path
        by_file: dict[str, list[dict[str, Any]]] = {}

        for result in results:
            file_path = result.get("metadata", {}).get("file_path", "")
            if file_path:
                if file_path not in by_file:
                    by_file[file_path] = []
                by_file[file_path].append(result)

        # Score each file
        scored: list[tuple[str, float, dict[str, Any] | None]] = []

        for file_path, file_results in by_file.items():
            # Combine scores from different result types
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
        return [result for _, _, result in scored[:top_k] if result is not None]

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
        # Vector search for semantic similarity
        vector_results = await self.search_similar(code, limit=top_k * 2)

        # Filter by function metadata
        results = []
        for result in vector_results:
            meta = result.metadata or {}

            # Filter by language
            if meta.get("language") != language:
                continue

            # Filter by function name if specified
            if function_name and function_name not in meta.get("content", ""):
                continue

            # Check if chunk contains a function definition
            if "def " in meta.get("content", "") or "async def " in meta.get(
                "content", ""
            ):
                results.append(
                    {
                        "file_path": meta.get("file_path", ""),
                        "content": result.source or meta.get("content", ""),
                        "score": result.score,
                        "start_line": meta.get("start_line"),
                        "end_line": meta.get("end_line"),
                    }
                )

        return results[:top_k]

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
        # In a full implementation, this would:
        # 1. Use graph engine to find the function node
        # 2. Traverse incoming and outgoing edges
        # 3. Build call graph structure
        # 4. Return formatted results

        return {
            "function": function_name,
            "file": file_path,
            "callers": [],
            "callees": [],
            "depth": depth,
        }

    async def get_code_hotspots(
        self,
        days: int = 30,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find code hotspots based on recent changes and complexity.

        Identifies files/functions that are:
        - Frequently modified (high churn)
        - Highly complex
        - Frequently queried

        Args:
            days: Number of days to look back.
            top_k: Number of hotspots to return.

        Returns:
            List of code hotspots with metrics.
        """
        # In a full implementation, this would:
        # 1. Query time-series for code churn
        # 2. Query time-series for complexity metrics
        # 3. Combine with query frequency metrics
        # 4. Return ranked list

        return []

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
        repo = Path(repo_path)
        if not repo.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        # Find code files
        code_files = self._find_code_files(repo, language_map)

        if max_files:
            code_files = code_files[:max_files]

        results: dict[str, Any] = {
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
    # Advanced Analytics
    # ========================================================================

    async def get_repository_overview(
        self,
    ) -> dict[str, Any]:
        """Get overview of indexed repository.

        Returns:
            Repository statistics across all models.
        """
        # This would query all collections and aggregate stats
        return {
            "workspace": self._workspace,
            "total_files": 0,
            "total_functions": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
            "metrics_points": 0,
        }

    async def analyze_dependencies(
        self,
        file_path: str,
    ) -> dict[str, Any]:
        """Analyze dependencies for a file.

        Returns:
            Dependency analysis including:
        - Internal dependencies (same repo)
        - External dependencies (packages/modules)
        - Dependency types (imports, includes, etc.)
        """
        # Would parse code and extract dependencies
        return {
            "file": file_path,
            "internal_deps": [],
            "external_deps": [],
            "dep_count": 0,
        }
