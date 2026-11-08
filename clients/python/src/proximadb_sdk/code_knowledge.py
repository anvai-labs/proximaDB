"""
Code Knowledge Builder - Coordinated Vector + Graph Population for Code Intelligence

This module provides a high-level API for building semantic code knowledge stores
that combine:
- Vector embeddings for semantic code search
- Graph relationships for structural code navigation
- Rich metadata for code assistant RAG capabilities

The CodeKnowledgeBuilder coordinates:
1. AST-aware code parsing and chunking
2. Embedding generation for code symbols
3. Vector store population with proper metadata
4. Graph store population with symbol nodes and relationship edges
5. Incremental indexing with change detection

Usage:
    from proximadb import ProximaDBClient
    from proximadb_sdk.code_knowledge import CodeKnowledgeBuilder

    client = ProximaDBClient(url="http://localhost:5678")
    builder = CodeKnowledgeBuilder(client)

    # Index a codebase
    await builder.index_directory("/path/to/code", recursive=True)

    # Search for relevant code
    results = await builder.search_code(
        query="function that handles authentication",
        top_k=10,
        include_context=True
    )

    # Find call graph
    callers = await builder.find_callers("my_function")
    callees = await builder.find_callees("my_function")

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import fnmatch
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._code_oid import record_symbol_id
from .chunking_strategies.base import TextChunk
from .chunking_strategies.code import (
    STATIC_EXTENSION_TO_LANGUAGE,
    CodeChunkingConfig,
    CodeChunkingStrategy,
    static_supported_extensions,
)


@dataclass
class CodeIndexConfig:
    """Configuration for code indexing"""

    # Vector collection settings
    vector_collection_name: str = "code_symbols"
    vector_dimension: int = 1536  # OpenAI ada-002 default

    # Graph settings
    graph_name: str = "code_graph"

    # Indexing behavior
    include_private: bool = True
    include_tests: bool = True
    include_documentation: bool = True

    # File filtering
    include_patterns: list[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "*.pyc",
            "__pycache__/*",
            ".git/*",
            ".hg/*",
            ".svn/*",
            "node_modules/*",
            "vendor/*",
            "target/*",
            "build/*",
            "dist/*",
            "*.min.js",
            "*.min.css",
            "*.map",
            ".venv/*",
            "venv/*",
            ".env/*",
            "env/*",
        ]
    )

    # Embedding settings
    embedding_batch_size: int = 32
    max_content_length: int = 8000  # Max chars for embedding

    # Change detection
    enable_incremental: bool = True
    hash_algorithm: str = "sha256"


@dataclass
class IndexingResult:
    """Result of indexing operation"""

    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    symbols_indexed: int = 0
    relations_created: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    file_hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class CodeSearchResult:
    """Result from code search"""

    symbol_id: str
    symbol_type: str
    fully_qualified_name: str
    simple_name: str
    source_code: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    score: float
    documentation: str | None = None
    signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Graph-derived context (populated when include_context=True)
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    parent_symbols: list[str] = field(default_factory=list)


class CodeKnowledgeBuilder:
    """
    High-level builder for code knowledge stores.

    Coordinates:
    - Code parsing with tree-sitter
    - Embedding generation
    - Vector store population
    - Graph store population
    - Incremental updates
    """

    def __init__(
        self,
        client: Any,  # ProximaDBClient
        config: CodeIndexConfig | None = None,
        embedding_provider: Any | None = None,
    ):
        """
        Initialize CodeKnowledgeBuilder.

        Args:
            client: ProximaDB client instance
            config: Indexing configuration
            embedding_provider: Optional embedding provider (uses client's default if not provided)
        """
        self.client = client
        self.config = config or CodeIndexConfig()
        self.embedding_provider = embedding_provider

        # Initialize chunker
        self._chunker = CodeChunkingStrategy(
            CodeChunkingConfig(
                include_private=self.config.include_private,
                include_tests=self.config.include_tests,
            )
        )

        # Cache for file hashes (for incremental indexing)
        self._file_hashes: dict[str, str] = {}

        # Track initialized resources
        self._vector_collection_ready = False
        self._graph_ready = False

    async def initialize(self) -> None:
        """Initialize vector collection and graph if they don't exist."""
        await self._ensure_vector_collection()
        await self._ensure_graph()

    async def _ensure_vector_collection(self) -> None:
        """Ensure vector collection exists with proper schema."""
        if self._vector_collection_ready:
            return

        try:
            # Check if collection exists
            collections = await self.client.list_collections()
            collection_names = [
                c.name if hasattr(c, "name") else c for c in collections
            ]

            if self.config.vector_collection_name not in collection_names:
                # Create collection with code-optimized schema
                await self.client.create_collection(
                    name=self.config.vector_collection_name,
                    dimension=self.config.vector_dimension,
                    distance_metric="cosine",
                    metadata={
                        "type": "code_knowledge",
                        "version": "1.0",
                    },
                )

            self._vector_collection_ready = True

        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector collection: {e}")

    async def _ensure_graph(self) -> None:
        """Ensure graph exists for code relationships."""
        if self._graph_ready:
            return

        try:
            # Check if graph exists
            graphs = await self.client.list_graphs()
            graph_names = [g.name if hasattr(g, "name") else g for g in graphs]

            if self.config.graph_name not in graph_names:
                # Create graph for code relationships
                await self.client.create_graph(
                    name=self.config.graph_name,
                    metadata={
                        "type": "code_knowledge",
                        "version": "1.0",
                    },
                )

            self._graph_ready = True

        except Exception as e:
            raise RuntimeError(f"Failed to initialize graph: {e}")

    async def index_file(
        self,
        file_path: str | Path,
        content: str | None = None,
        force: bool = False,
    ) -> IndexingResult:
        """
        Index a single code file.

        Args:
            file_path: Path to the file
            content: Optional file content (read from disk if not provided)
            force: Force re-indexing even if file hasn't changed

        Returns:
            IndexingResult with statistics
        """
        await self.initialize()

        result = IndexingResult()
        file_path = Path(file_path)

        # Read content if not provided
        if content is None:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                result.files_failed = 1
                result.errors.append(
                    {"file": str(file_path), "error": f"Failed to read file: {e}"}
                )
                return result

        # Check for changes (incremental indexing)
        content_hash = self._compute_hash(content)
        if not force and self.config.enable_incremental:
            cached_hash = self._file_hashes.get(str(file_path))
            if cached_hash == content_hash:
                result.files_skipped = 1
                return result

        # Detect language
        ext = file_path.suffix.lower()
        # ATTRIBUTION: what language is this? Naming it is not a claim to
        # parse it, so this is answered statically; whether symbols can
        # actually be extracted is the chunker's call, and it fails loudly.
        language = STATIC_EXTENSION_TO_LANGUAGE.get(ext)

        if not language:
            result.files_skipped = 1
            return result

        try:
            # Parse and chunk the file
            chunks = self._chunker.chunk(
                text=content, source_id=str(file_path), metadata={"language": language}
            )

            if not chunks:
                result.files_skipped = 1
                return result

            # Generate embeddings
            embeddings = await self._generate_embeddings(chunks)

            # Insert vector-bearing code records.
            await self._insert_records(chunks, embeddings, file_path, language)
            result.symbols_indexed = len(chunks)

            # Extract and insert graph relationships
            relations_count = await self._insert_graph_data(chunks, file_path, language)
            result.relations_created = relations_count

            # Update hash cache
            self._file_hashes[str(file_path)] = content_hash
            result.file_hashes[str(file_path)] = content_hash
            result.files_processed = 1

        except Exception as e:
            result.files_failed = 1
            result.errors.append({"file": str(file_path), "error": str(e)})

        return result

    async def index_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        force: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> IndexingResult:
        """
        Index all code files in a directory.

        Args:
            directory: Directory path
            recursive: Whether to recurse into subdirectories
            force: Force re-indexing all files
            progress_callback: Optional callback(file_path, current, total)

        Returns:
            Aggregated IndexingResult
        """
        await self.initialize()

        directory = Path(directory)
        result = IndexingResult()

        # Collect files to index
        files_to_index = self._collect_files(directory, recursive)
        total_files = len(files_to_index)

        for i, file_path in enumerate(files_to_index):
            if progress_callback:
                progress_callback(str(file_path), i + 1, total_files)

            file_result = await self.index_file(file_path, force=force)

            # Aggregate results
            result.files_processed += file_result.files_processed
            result.files_skipped += file_result.files_skipped
            result.files_failed += file_result.files_failed
            result.symbols_indexed += file_result.symbols_indexed
            result.relations_created += file_result.relations_created
            result.errors.extend(file_result.errors)
            result.file_hashes.update(file_result.file_hashes)

        return result

    def _collect_files(self, directory: Path, recursive: bool) -> list[Path]:
        """Collect files to index based on patterns."""
        files = []
        # ROUTING: stay wide. This must not depend on the optional parser
        # extra -- when it did, an absent extra emptied the map and this
        # collected zero files, so indexing a repository silently produced
        # nothing at all.
        supported_extensions = set(static_supported_extensions())

        if recursive:
            iterator = directory.rglob("*")
        else:
            iterator = directory.glob("*")

        for path in iterator:
            if not path.is_file():
                continue

            # Check extension
            if path.suffix.lower() not in supported_extensions:
                continue

            # Check exclude patterns
            rel_path = str(path.relative_to(directory))
            if self._matches_patterns(rel_path, self.config.exclude_patterns):
                continue

            # Check include patterns
            if not self._matches_patterns(rel_path, self.config.include_patterns):
                continue

            files.append(path)

        return sorted(files)

    def _matches_patterns(self, path: str, patterns: list[str]) -> bool:
        """Check if path matches any of the patterns."""
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for change detection."""
        if self.config.hash_algorithm == "sha256":
            return hashlib.sha256(content.encode()).hexdigest()
        elif self.config.hash_algorithm == "md5":
            return hashlib.md5(content.encode()).hexdigest()
        else:
            return hashlib.sha256(content.encode()).hexdigest()

    async def _generate_embeddings(self, chunks: list[TextChunk]) -> list[list[float]]:
        """Generate embeddings for code chunks."""
        if self.embedding_provider:
            # Use provided embedding provider
            texts = [self._prepare_text_for_embedding(chunk) for chunk in chunks]
            return await self.embedding_provider.embed_batch(texts)
        else:
            # Use client's default embedding (if available)
            # This is a placeholder - actual implementation depends on client
            embeddings = []
            for chunk in chunks:
                text = self._prepare_text_for_embedding(chunk)
                # Generate a simple hash-based embedding for testing
                # In production, this would use a real embedding model
                embedding = self._generate_placeholder_embedding(text)
                embeddings.append(embedding)
            return embeddings

    def _prepare_text_for_embedding(self, chunk: TextChunk) -> str:
        """Prepare chunk text for embedding generation."""
        parts = []

        # Add symbol context
        if chunk.metadata.get("fully_qualified_name"):
            parts.append(f"Symbol: {chunk.metadata['fully_qualified_name']}")

        if chunk.metadata.get("documentation"):
            doc = chunk.metadata["documentation"][:500]  # Limit doc length
            parts.append(f"Documentation: {doc}")

        if chunk.metadata.get("signature"):
            parts.append(f"Signature: {chunk.metadata['signature']}")

        # Add the code itself
        code = chunk.text
        if len(code) > self.config.max_content_length:
            code = code[: self.config.max_content_length] + "..."
        parts.append(f"Code:\n{code}")

        return "\n\n".join(parts)

    def _generate_placeholder_embedding(self, text: str) -> list[float]:
        """Generate placeholder embedding for testing (deterministic based on content)."""
        # Use hash to generate deterministic pseudo-random vector
        h = hashlib.sha256(text.encode()).hexdigest()
        embedding = []
        for i in range(0, min(len(h), self.config.vector_dimension * 2), 2):
            byte_val = int(h[i : i + 2], 16)
            embedding.append((byte_val - 128) / 128.0)  # Normalize to [-1, 1]

        # Pad or truncate to correct dimension
        while len(embedding) < self.config.vector_dimension:
            embedding.append(0.0)

        return embedding[: self.config.vector_dimension]

    async def _insert_records(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        file_path: Path,
        language: str,
    ) -> None:
        """Insert code symbols as ProximaRecord-shaped vector-bearing records."""
        records = []

        for chunk, embedding in zip(chunks, embeddings):
            # Build rich metadata for RAG
            metadata = {
                "symbol_id": chunk.metadata.get("symbol_id", chunk.chunk_id),
                "symbol_type": chunk.metadata.get("symbol_type", "UNKNOWN"),
                "fully_qualified_name": chunk.metadata.get("fully_qualified_name", ""),
                "simple_name": chunk.metadata.get("simple_name", ""),
                "file_path": str(file_path),
                "language": language,
                "start_line": chunk.metadata.get("start_line", 0),
                "end_line": chunk.metadata.get("end_line", 0),
                "source_code": chunk.text,  # Store full source for retrieval
            }

            # Add optional metadata
            if chunk.metadata.get("documentation"):
                metadata["documentation"] = chunk.metadata["documentation"]

            if chunk.metadata.get("signature"):
                metadata["signature"] = chunk.metadata["signature"]

            if chunk.metadata.get("modifiers"):
                metadata["modifiers"] = ",".join(chunk.metadata["modifiers"])

            if chunk.metadata.get("parameters"):
                metadata["parameters"] = str(chunk.metadata["parameters"])

            if chunk.metadata.get("return_type"):
                metadata["return_type"] = chunk.metadata["return_type"]

            if chunk.metadata.get("complexity"):
                metadata["complexity"] = str(chunk.metadata["complexity"])

            records.append(
                {
                    # ADR-044: line-independent canonical oid when gated (matches
                    # Victor's adapter + the AnvaiOps connector for the same symbol);
                    # legacy symbol_id otherwise. node_id below uses the same value so
                    # the vector + graph projections correlate.
                    "id": record_symbol_id(metadata, self.config.graph_name),
                    "vector": embedding,
                    "props": metadata,
                    # TD-REC-1: the code text rides in text_fields; a
                    # top-level `source` was never a contract field (silently
                    # dropped before, a 400 now).
                    "text_fields": [{"name": "source_code", "content": chunk.text}],
                }
            )

        # Batch insert
        if records:
            collection = await self.client.get_collection(
                self.config.vector_collection_name
            )
            if hasattr(collection, "insert_records"):
                await collection.insert_records(records)
            else:
                await collection.insert(records)

    async def _insert_graph_data(
        self,
        chunks: list[TextChunk],
        file_path: Path,
        language: str,
    ) -> int:
        """Insert symbol nodes and relationship edges into graph."""
        relations_count = 0

        try:
            graph = await self.client.get_graph(self.config.graph_name)

            # Insert nodes for each symbol
            for chunk in chunks:
                # ADR-044: same id derivation as the vector record (above) so the
                # graph node and the vector correlate on one oid.
                node_id = record_symbol_id(
                    {
                        **chunk.metadata,
                        "symbol_id": chunk.metadata.get("symbol_id", chunk.chunk_id),
                    },
                    self.config.graph_name,
                )

                node_properties = {
                    "symbol_type": chunk.metadata.get("symbol_type", "UNKNOWN"),
                    "fully_qualified_name": chunk.metadata.get(
                        "fully_qualified_name", ""
                    ),
                    "simple_name": chunk.metadata.get("simple_name", ""),
                    "file_path": str(file_path),
                    "language": language,
                    "start_line": chunk.metadata.get("start_line", 0),
                    "end_line": chunk.metadata.get("end_line", 0),
                }

                if chunk.metadata.get("documentation"):
                    node_properties["documentation"] = chunk.metadata["documentation"]

                if chunk.metadata.get("signature"):
                    node_properties["signature"] = chunk.metadata["signature"]

                labels = [
                    chunk.metadata.get("symbol_type", "Symbol"),
                    language.capitalize(),
                ]

                await graph.insert_node(
                    {
                        "id": node_id,
                        "labels": labels,
                        "properties": node_properties,
                    }
                )

            # Insert edges for relationships
            for chunk in chunks:
                relations = chunk.metadata.get("relations", [])
                from_id = chunk.metadata.get("symbol_id", chunk.chunk_id)

                for rel in relations:
                    to_id = rel.get("to")
                    rel_type = rel.get("type", "REFERENCES")
                    confidence = rel.get("confidence", 1.0)

                    if to_id:
                        await graph.insert_edge(
                            {
                                "id": f"{from_id}_{rel_type}_{to_id}",
                                "from_node_id": from_id,
                                "to_node_id": to_id,
                                "edge_type": rel_type,
                                "properties": {
                                    "confidence": confidence,
                                },
                            }
                        )
                        relations_count += 1

            # Insert containment relationships (class contains method, etc.)
            for chunk in chunks:
                scope_chain = chunk.metadata.get("scope_chain", [])
                if scope_chain:
                    child_id = chunk.metadata.get("symbol_id", chunk.chunk_id)
                    parent_name = scope_chain[-1]  # Immediate parent

                    # Find parent symbol
                    for other_chunk in chunks:
                        if other_chunk.metadata.get("simple_name") == parent_name:
                            parent_id = other_chunk.metadata.get(
                                "symbol_id", other_chunk.chunk_id
                            )
                            await graph.insert_edge(
                                {
                                    "id": f"{parent_id}_CONTAINS_{child_id}",
                                    "from_node_id": parent_id,
                                    "to_node_id": child_id,
                                    "edge_type": "CONTAINS",
                                }
                            )
                            relations_count += 1
                            break

        except Exception:
            # Log but don't fail the whole operation
            pass

        return relations_count

    async def search_code(
        self,
        query: str,
        top_k: int = 10,
        filter_language: str | None = None,
        filter_symbol_types: list[str] | None = None,
        include_context: bool = False,
    ) -> list[CodeSearchResult]:
        """
        Search for code using semantic similarity.

        Args:
            query: Natural language query or code snippet
            top_k: Number of results to return
            filter_language: Filter by programming language
            filter_symbol_types: Filter by symbol types (e.g., ["FUNCTION", "CLASS"])
            include_context: Include graph-derived context (callers, callees)

        Returns:
            List of CodeSearchResult
        """
        await self.initialize()

        # Generate query embedding
        query_embedding = await self._generate_query_embedding(query)

        # Build filter
        metadata_filter = {}
        if filter_language:
            metadata_filter["language"] = filter_language
        if filter_symbol_types:
            metadata_filter["symbol_type"] = {"$in": filter_symbol_types}

        # Search vector store
        collection = await self.client.get_collection(
            self.config.vector_collection_name
        )
        search_results = await collection.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter=metadata_filter if metadata_filter else None,
        )

        # Convert to CodeSearchResult
        results = []
        for result in search_results:
            metadata = result.get("metadata", {})

            search_result = CodeSearchResult(
                symbol_id=metadata.get("symbol_id", ""),
                symbol_type=metadata.get("symbol_type", "UNKNOWN"),
                fully_qualified_name=metadata.get("fully_qualified_name", ""),
                simple_name=metadata.get("simple_name", ""),
                source_code=metadata.get("source_code", ""),
                file_path=metadata.get("file_path", ""),
                start_line=metadata.get("start_line", 0),
                end_line=metadata.get("end_line", 0),
                language=metadata.get("language", ""),
                score=result.get("score", 0.0),
                documentation=metadata.get("documentation"),
                signature=metadata.get("signature"),
                metadata=metadata,
            )

            # Optionally fetch graph context
            if include_context:
                await self._enrich_with_graph_context(search_result)

            results.append(search_result)

        return results

    async def _generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for search query."""
        if self.embedding_provider:
            embeddings = await self.embedding_provider.embed_batch([query])
            return embeddings[0]
        else:
            return self._generate_placeholder_embedding(query)

    async def _enrich_with_graph_context(self, result: CodeSearchResult) -> None:
        """Enrich search result with graph-derived context."""
        try:
            graph = await self.client.get_graph(self.config.graph_name)

            # Find callers
            callers = await graph.traverse(
                start_node_id=result.symbol_id,
                edge_type="CALLS",
                direction="incoming",
                max_depth=1,
            )
            result.callers = [c.get("id", "") for c in callers]

            # Find callees
            callees = await graph.traverse(
                start_node_id=result.symbol_id,
                edge_type="CALLS",
                direction="outgoing",
                max_depth=1,
            )
            result.callees = [c.get("id", "") for c in callees]

            # Find parent symbols
            parents = await graph.traverse(
                start_node_id=result.symbol_id,
                edge_type="CONTAINS",
                direction="incoming",
                max_depth=1,
            )
            result.parent_symbols = [p.get("id", "") for p in parents]

        except Exception:
            pass  # Graph context is optional

    async def find_callers(
        self,
        symbol_name: str,
        max_depth: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Find all symbols that call the given symbol.

        Args:
            symbol_name: Name or ID of the symbol
            max_depth: Maximum traversal depth

        Returns:
            List of caller symbols with their metadata
        """
        await self.initialize()

        # First, find the symbol ID
        symbol_id = await self._resolve_symbol_id(symbol_name)
        if not symbol_id:
            return []

        graph = await self.client.get_graph(self.config.graph_name)
        callers = await graph.traverse(
            start_node_id=symbol_id,
            edge_type="CALLS",
            direction="incoming",
            max_depth=max_depth,
        )

        return callers

    async def find_callees(
        self,
        symbol_name: str,
        max_depth: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Find all symbols that are called by the given symbol.

        Args:
            symbol_name: Name or ID of the symbol
            max_depth: Maximum traversal depth

        Returns:
            List of callee symbols with their metadata
        """
        await self.initialize()

        symbol_id = await self._resolve_symbol_id(symbol_name)
        if not symbol_id:
            return []

        graph = await self.client.get_graph(self.config.graph_name)
        callees = await graph.traverse(
            start_node_id=symbol_id,
            edge_type="CALLS",
            direction="outgoing",
            max_depth=max_depth,
        )

        return callees

    async def find_usages(
        self,
        symbol_name: str,
    ) -> list[dict[str, Any]]:
        """
        Find all usages (references) of a symbol.

        Args:
            symbol_name: Name or ID of the symbol

        Returns:
            List of symbols that reference the given symbol
        """
        await self.initialize()

        symbol_id = await self._resolve_symbol_id(symbol_name)
        if not symbol_id:
            return []

        graph = await self.client.get_graph(self.config.graph_name)
        usages = await graph.traverse(
            start_node_id=symbol_id,
            edge_type="REFERENCES",
            direction="incoming",
            max_depth=1,
        )

        return usages

    async def get_impact_analysis(
        self,
        symbol_name: str,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """
        Analyze the impact of changing a symbol.

        Returns symbols that would be affected by changes to the given symbol.

        Args:
            symbol_name: Name or ID of the symbol
            max_depth: Maximum depth for impact analysis

        Returns:
            Impact analysis with affected symbols at each level
        """
        await self.initialize()

        symbol_id = await self._resolve_symbol_id(symbol_name)
        if not symbol_id:
            return {"error": "Symbol not found"}

        graph = await self.client.get_graph(self.config.graph_name)

        # Find all symbols affected through various relationships
        impact = {
            "symbol": symbol_name,
            "direct_callers": [],
            "indirect_callers": [],
            "dependent_files": set(),
            "total_affected": 0,
        }

        # Direct callers
        direct = await graph.traverse(
            start_node_id=symbol_id,
            edge_type="CALLS",
            direction="incoming",
            max_depth=1,
        )
        impact["direct_callers"] = direct

        # Indirect callers (up to max_depth)
        if max_depth > 1:
            indirect = await graph.traverse(
                start_node_id=symbol_id,
                edge_type="CALLS",
                direction="incoming",
                max_depth=max_depth,
            )
            # Remove direct callers from indirect
            direct_ids = {c.get("id") for c in direct}
            impact["indirect_callers"] = [
                c for c in indirect if c.get("id") not in direct_ids
            ]

        # Collect affected files
        for caller in impact["direct_callers"] + impact["indirect_callers"]:
            file_path = caller.get("properties", {}).get("file_path")
            if file_path:
                impact["dependent_files"].add(file_path)

        impact["dependent_files"] = list(impact["dependent_files"])
        impact["total_affected"] = len(impact["direct_callers"]) + len(
            impact["indirect_callers"]
        )

        return impact

    async def _resolve_symbol_id(self, symbol_name: str) -> str | None:
        """Resolve symbol name to ID."""
        # First check if it's already an ID
        if len(symbol_name) == 16 and symbol_name.isalnum():
            return symbol_name

        # Search by name
        results = await self.search_code(
            query=symbol_name,
            top_k=1,
            include_context=False,
        )

        if results:
            return results[0].symbol_id

        return None

    async def delete_file_index(self, file_path: str | Path) -> bool:
        """
        Remove all indexed data for a file.

        Args:
            file_path: Path to the file

        Returns:
            True if successful
        """
        await self.initialize()

        file_path = str(file_path)

        try:
            # Delete from vector store
            collection = await self.client.get_collection(
                self.config.vector_collection_name
            )
            await collection.delete(filter={"file_path": file_path})

            # Delete from graph
            graph = await self.client.get_graph(self.config.graph_name)
            # Note: This would need a more sophisticated deletion in production
            # that also handles edges

            # Remove from hash cache
            self._file_hashes.pop(file_path, None)

            return True

        except Exception as exc:
            # NOT False. False here meant "the file was not indexed", which is a
            # normal outcome a caller acts on by moving along. A delete that
            # FAILED must not be reported as a delete that found nothing --
            # otherwise cleanup silently does not happen and nothing says so.
            raise RuntimeError(
                f"failed to delete the index entry for {file_path!r}: {exc}"
            ) from exc

    def get_indexed_files(self) -> list[str]:
        """Get list of currently indexed files."""
        return list(self._file_hashes.keys())

    def get_file_hash(self, file_path: str | Path) -> str | None:
        """Get the hash of an indexed file."""
        return self._file_hashes.get(str(file_path))


# Convenience functions


async def create_code_knowledge_store(
    client: Any,
    directory: str | Path,
    config: CodeIndexConfig | None = None,
    embedding_provider: Any | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[CodeKnowledgeBuilder, IndexingResult]:
    """
    Create and populate a code knowledge store.

    Args:
        client: ProximaDB client
        directory: Directory to index
        config: Optional configuration
        embedding_provider: Optional embedding provider
        progress_callback: Optional progress callback

    Returns:
        Tuple of (builder, indexing_result)
    """
    builder = CodeKnowledgeBuilder(
        client=client,
        config=config,
        embedding_provider=embedding_provider,
    )

    result = await builder.index_directory(
        directory=directory,
        recursive=True,
        progress_callback=progress_callback,
    )

    return builder, result
