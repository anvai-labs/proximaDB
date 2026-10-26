"""
Unified Document Processing Pipeline for ProximaDB

Provides a complete workflow for:
Document → Detect Type → Chunk → Embed → Create Records → Store

This pipeline integrates:
- Document processors (code, text, binary, OCR)
- Embedding providers (any provider via adapter)
- Vector stores (ProximaDB collections)
- Metrics and progress tracking

Design Patterns:
- Pipeline Pattern: Sequential processing stages
- Strategy Pattern: Pluggable processors and providers
- Observer Pattern: Progress callbacks and metrics
- Factory Pattern: Auto-detection and creation

Usage:
    # Simple usage
    pipeline = DocumentPipeline(embedding_provider=my_provider)
    result = await pipeline.process("code content", "file.py")

    # With vector store
    pipeline = DocumentPipeline(
        embedding_provider=my_provider,
        vector_store=my_collection
    )
    result = await pipeline.process_and_store("content", "doc.md")

    # Batch processing
    results = await pipeline.process_batch([
        {"content": "...", "source_id": "file1.py"},
        {"content": "...", "source_id": "file2.py"},
    ])
"""

import asyncio
import logging
import threading
import time
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .document_processor import (
    DocumentProcessor,
    DocumentType,
    ProcessedChunk,
    ProcessingResult,
    ProcessorConfig,
    VectorRecord,
    create_embedding_adapter,
    create_processor,
    get_processor_registry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Configuration
# =============================================================================


class PipelineMode(Enum):
    """Processing mode for the pipeline"""

    PROCESS_ONLY = "process_only"  # Chunk + prepare (no embedding)
    EMBED = "embed"  # Chunk + embed (no store)
    STORE = "store"  # Chunk + embed + store


class ErrorStrategy(Enum):
    """Error handling strategy"""

    FAIL_FAST = "fail_fast"
    SKIP = "skip"
    COLLECT = "collect"


@dataclass
class PipelineConfig:
    """Configuration for the document pipeline"""

    # Processing mode
    mode: PipelineMode = PipelineMode.EMBED

    # Processor settings
    processor_config: ProcessorConfig | None = None
    auto_detect_type: bool = True
    default_processor: str = "text"

    # Embedding settings
    embedding_batch_size: int = 32
    use_placeholder_embeddings: bool = False
    placeholder_dimension: int = 384

    # Concurrency settings
    max_concurrent: int = 4
    batch_size: int = 10

    # Error handling
    error_strategy: ErrorStrategy = ErrorStrategy.COLLECT
    max_retries: int = 3

    # Progress and metrics
    enable_metrics: bool = True
    progress_callback: Callable[[int, int, str], None] | None = None

    def __post_init__(self):
        if self.processor_config is None:
            self.processor_config = ProcessorConfig()


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution"""

    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    total_vectors: int = 0
    total_processing_time_sec: float = 0.0
    embedding_time_sec: float = 0.0
    storage_time_sec: float = 0.0
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return self.processed_documents / self.total_documents

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "processed_documents": self.processed_documents,
            "failed_documents": self.failed_documents,
            "total_chunks": self.total_chunks,
            "total_vectors": self.total_vectors,
            "success_rate": self.success_rate,
            "total_processing_time_sec": self.total_processing_time_sec,
            "embedding_time_sec": self.embedding_time_sec,
            "storage_time_sec": self.storage_time_sec,
            "error_count": len(self.errors),
        }


@dataclass
class BatchResult:
    """Result of batch processing"""

    results: list[ProcessingResult] = field(default_factory=list)
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    def get_vectors(self) -> list[VectorRecord]:
        """Get all vectors from successful results"""
        vectors = []
        for result in self.results:
            if result.success:
                vectors.extend(result.vectors)
        return vectors


# =============================================================================
# Vector Store Protocol
# =============================================================================


class VectorStoreProtocol(Protocol):
    """Protocol for vector stores"""

    async def insert(self, records: list[dict[str, Any]]) -> None:
        """Insert records into the store"""
        ...


# =============================================================================
# Progress Tracker
# =============================================================================


class ProgressTracker:
    """Tracks and reports pipeline progress"""

    def __init__(self, callback: Callable[[int, int, str], None] | None = None):
        self.callback = callback
        self._current = 0
        self._total = 0
        self._status = "idle"
        self._lock = threading.Lock()
        self._start_time: float | None = None

    def start(self, total: int, status: str = "processing") -> None:
        with self._lock:
            self._current = 0
            self._total = total
            self._status = status
            self._start_time = time.time()
        self._notify()

    def update(self, increment: int = 1, status: str | None = None) -> None:
        with self._lock:
            self._current += increment
            if status:
                self._status = status
        self._notify()

    def complete(self, status: str = "completed") -> None:
        with self._lock:
            self._current = self._total
            self._status = status
        self._notify()

    def _notify(self) -> None:
        if self.callback:
            self.callback(self._current, self._total, self._status)

    @property
    def elapsed_time(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time


# =============================================================================
# Document Pipeline
# =============================================================================


class DocumentPipeline:
    """
    Unified document processing pipeline.

    Coordinates the complete workflow:
    1. Document type detection
    2. Processor selection
    3. Chunking
    4. Embedding generation
    5. Vector record creation
    6. Storage (optional)

    Example:
        # Basic usage
        pipeline = DocumentPipeline(embedding_provider=my_provider)
        result = await pipeline.process(code_content, "main.py")

        # With storage
        pipeline = DocumentPipeline(
            embedding_provider=my_provider,
            vector_store=my_collection,
            config=PipelineConfig(mode=PipelineMode.STORE)
        )
        result = await pipeline.process_and_store(content, "doc.md")

        # Batch processing
        results = await pipeline.process_directory("/path/to/code")
    """

    def __init__(
        self,
        embedding_provider: Any | None = None,
        vector_store: VectorStoreProtocol | None = None,
        config: PipelineConfig | None = None,
    ):
        self.config = config or PipelineConfig()
        self.vector_store = vector_store

        # Initialize embedding adapter
        if embedding_provider:
            self.embedding_adapter = create_embedding_adapter(
                embedding_provider, batch_size=self.config.embedding_batch_size
            )
        elif self.config.use_placeholder_embeddings:
            self.embedding_adapter = create_embedding_adapter(
                None,
                use_placeholder=True,
                placeholder_dimension=self.config.placeholder_dimension,
            )
        else:
            self.embedding_adapter = None

        # Get processor registry
        self._registry = get_processor_registry()

        # Progress tracking
        self._progress = ProgressTracker(self.config.progress_callback)

        # Metrics
        self._metrics = PipelineMetrics()
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Configuration API
    # -------------------------------------------------------------------------

    def with_embedding_provider(self, provider: Any) -> "DocumentPipeline":
        """Set embedding provider"""
        self.embedding_adapter = create_embedding_adapter(
            provider, batch_size=self.config.embedding_batch_size
        )
        return self

    def with_vector_store(self, store: VectorStoreProtocol) -> "DocumentPipeline":
        """Set vector store"""
        self.vector_store = store
        return self

    def with_progress_callback(
        self, callback: Callable[[int, int, str], None]
    ) -> "DocumentPipeline":
        """Set progress callback"""
        self.config.progress_callback = callback
        self._progress = ProgressTracker(callback)
        return self

    # -------------------------------------------------------------------------
    # Core Processing
    # -------------------------------------------------------------------------

    async def process(
        self,
        content: str,
        source_id: str,
        metadata: dict[str, Any] | None = None,
        document_type: DocumentType | None = None,
        processor_name: str | None = None,
    ) -> ProcessingResult:
        """
        Process a single document.

        Args:
            content: Document content
            source_id: Unique identifier (usually file path)
            metadata: Additional metadata
            document_type: Force specific document type
            processor_name: Force specific processor

        Returns:
            ProcessingResult with chunks and optionally vectors
        """
        start_time = time.time()

        try:
            # Step 1: Get processor
            processor = self._get_processor(
                content, source_id, document_type, processor_name
            )

            # Step 2: Process with processor
            result = await processor.process(
                content=content,
                source_id=source_id,
                embedding_adapter=(
                    self.embedding_adapter
                    if self.config.mode != PipelineMode.PROCESS_ONLY
                    else None
                ),
                metadata=metadata,
            )

            # Update metrics
            with self._lock:
                self._metrics.total_documents += 1
                if result.success:
                    self._metrics.processed_documents += 1
                else:
                    self._metrics.failed_documents += 1
                self._metrics.total_chunks += result.chunk_count
                self._metrics.total_vectors += result.vector_count
                self._metrics.total_processing_time_sec += time.time() - start_time

            return result

        except Exception as e:
            logger.error(f"Pipeline processing failed for {source_id}: {e}")

            with self._lock:
                self._metrics.total_documents += 1
                self._metrics.failed_documents += 1
                self._metrics.errors.append({"source_id": source_id, "error": str(e)})

            return ProcessingResult(
                success=False,
                source_id=source_id,
                document_type=document_type or DocumentType.UNKNOWN,
                errors=[{"stage": "pipeline", "error": str(e)}],
                metrics={"processing_time_sec": time.time() - start_time},
            )

    async def process_and_store(
        self,
        content: str,
        source_id: str,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Process document and store vectors in vector store.

        Args:
            content: Document content
            source_id: Unique identifier
            metadata: Additional metadata
            **kwargs: Additional arguments for process()

        Returns:
            ProcessingResult with storage status in metrics
        """
        if not self.vector_store:
            raise ValueError("No vector store configured")

        # Process the document
        result = await self.process(content, source_id, metadata, **kwargs)

        if not result.success or not result.vectors:
            return result

        # Store vectors
        try:
            storage_start = time.time()
            records = [v.to_dict() for v in result.vectors]
            await self.vector_store.insert(records)

            storage_time = time.time() - storage_start
            result.metrics["storage_time_sec"] = storage_time
            result.metrics["records_stored"] = len(records)

            with self._lock:
                self._metrics.storage_time_sec += storage_time

        except Exception as e:
            result.success = False
            result.errors.append({"stage": "storage", "error": str(e)})
            logger.error(f"Storage failed for {source_id}: {e}")

        return result

    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------

    async def process_batch(
        self, documents: list[dict[str, Any]], concurrent_limit: int | None = None
    ) -> BatchResult:
        """
        Process multiple documents concurrently.

        Args:
            documents: List of dicts with 'content' and 'source_id' keys
            concurrent_limit: Max concurrent tasks (default from config)

        Returns:
            BatchResult with all results and aggregated metrics

        Example:
            results = await pipeline.process_batch([
                {"content": "...", "source_id": "file1.py"},
                {"content": "...", "source_id": "file2.py", "metadata": {...}},
            ])
        """
        concurrent_limit = concurrent_limit or self.config.max_concurrent
        semaphore = asyncio.Semaphore(concurrent_limit)

        self._progress.start(len(documents), "batch_processing")

        async def process_one(doc: dict[str, Any]) -> ProcessingResult:
            async with semaphore:
                result = await self.process(
                    content=doc.get("content", ""),
                    source_id=doc.get("source_id", "unknown"),
                    metadata=doc.get("metadata"),
                    document_type=doc.get("document_type"),
                    processor_name=doc.get("processor"),
                )
                self._progress.update(1)
                return result

        # Process all documents
        tasks = [process_one(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        self._progress.complete()

        # Collect results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ProcessingResult(
                        success=False,
                        source_id=documents[i].get("source_id", "unknown"),
                        document_type=DocumentType.UNKNOWN,
                        errors=[{"error": str(result)}],
                    )
                )
            else:
                processed_results.append(result)

        return BatchResult(results=processed_results, metrics=self._metrics)

    async def process_batch_and_store(
        self, documents: list[dict[str, Any]], concurrent_limit: int | None = None
    ) -> BatchResult:
        """
        Process and store multiple documents.

        Same as process_batch but stores all vectors.
        """
        if not self.vector_store:
            raise ValueError("No vector store configured")

        # Process all documents
        batch_result = await self.process_batch(documents, concurrent_limit)

        # Collect all vectors and store
        all_vectors = batch_result.get_vectors()
        if all_vectors:
            try:
                storage_start = time.time()
                records = [v.to_dict() for v in all_vectors]
                await self.vector_store.insert(records)

                with self._lock:
                    self._metrics.storage_time_sec += time.time() - storage_start

            except Exception as e:
                logger.error(f"Batch storage failed: {e}")
                self._metrics.errors.append({"stage": "batch_storage", "error": str(e)})

        return batch_result

    # -------------------------------------------------------------------------
    # File/Directory Processing
    # -------------------------------------------------------------------------

    async def process_file(
        self, file_path: str | Path, encoding: str = "utf-8"
    ) -> ProcessingResult:
        """
        Process a single file.

        Args:
            file_path: Path to the file
            encoding: File encoding

        Returns:
            ProcessingResult
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return ProcessingResult(
                success=False,
                source_id=str(file_path),
                document_type=DocumentType.UNKNOWN,
                errors=[{"error": f"File not found: {file_path}"}],
            )

        try:
            content = file_path.read_text(encoding=encoding)

            return await self.process(
                content=content,
                source_id=str(file_path),
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_extension": file_path.suffix,
                },
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                source_id=str(file_path),
                document_type=DocumentType.UNKNOWN,
                errors=[{"error": f"Failed to read file: {e}"}],
            )

    async def process_directory(
        self,
        directory: str | Path,
        pattern: str = "**/*",
        recursive: bool = True,
        extensions: list[str] | None = None,
    ) -> BatchResult:
        """
        Process all matching files in a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern
            recursive: Whether to search recursively
            extensions: Filter by extensions (e.g., [".py", ".rs"])

        Returns:
            BatchResult with all results
        """
        directory = Path(directory)

        if not directory.is_dir():
            return BatchResult(
                metrics=PipelineMetrics(
                    errors=[{"error": f"Not a directory: {directory}"}]
                )
            )

        # Collect files
        if recursive:
            files = list(directory.glob(pattern))
        else:
            files = list(directory.glob(pattern.replace("**", "*", 1)))

        # Filter to files only
        files = [f for f in files if f.is_file()]

        # Filter by extension if specified
        if extensions:
            files = [f for f in files if f.suffix.lower() in extensions]

        # Process as batch
        self._progress.start(len(files), "directory_processing")

        results = []
        for file_path in files:
            result = await self.process_file(file_path)
            results.append(result)
            self._progress.update(1)

        self._progress.complete()

        # Aggregate metrics
        return BatchResult(results=results, metrics=self._metrics)

    # -------------------------------------------------------------------------
    # Streaming Processing
    # -------------------------------------------------------------------------

    async def process_stream(
        self, content: str, source_id: str, metadata: dict[str, Any] | None = None
    ) -> AsyncGenerator[ProcessedChunk, None]:
        """
        Process document as a stream, yielding chunks as they're ready.

        Memory-efficient for large documents.

        Args:
            content: Document content
            source_id: Source identifier
            metadata: Additional metadata

        Yields:
            ProcessedChunk objects
        """
        processor = self._get_processor(content, source_id)

        # Get chunks
        chunks = processor.chunk(content, source_id, metadata)

        for chunk in chunks:
            chunk.metadata = processor.enrich_metadata(chunk, metadata)
            yield chunk

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_processor(
        self,
        content: str,
        source_id: str,
        document_type: DocumentType | None = None,
        processor_name: str | None = None,
    ) -> DocumentProcessor:
        """Get appropriate processor for content"""
        if processor_name:
            processor = self._registry.get(processor_name)
            if processor:
                return processor

        if document_type:
            processor = self._registry.get_for_type(document_type)
            if processor:
                return processor

        if self.config.auto_detect_type:
            return self._registry.detect_and_get(content, source_id)

        return self._registry.get(self.config.default_processor) or create_processor(
            "text", self.config.processor_config
        )

    # -------------------------------------------------------------------------
    # Metrics and Monitoring
    # -------------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Get current pipeline metrics"""
        return self._metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self._metrics = PipelineMetrics()


# =============================================================================
# Factory Functions
# =============================================================================


def create_document_pipeline(
    embedding_provider: Any | None = None,
    vector_store: VectorStoreProtocol | None = None,
    mode: PipelineMode = PipelineMode.EMBED,
    **kwargs,
) -> DocumentPipeline:
    """
    Create a document processing pipeline.

    Args:
        embedding_provider: Provider for generating embeddings
        vector_store: Store for vectors (required for STORE mode)
        mode: Processing mode
        **kwargs: Additional PipelineConfig options

    Returns:
        Configured DocumentPipeline

    Example:
        pipeline = create_document_pipeline(
            embedding_provider=my_provider,
            mode=PipelineMode.EMBED,
            max_concurrent=8
        )
    """
    config = PipelineConfig(mode=mode, **kwargs)

    return DocumentPipeline(
        embedding_provider=embedding_provider, vector_store=vector_store, config=config
    )


def create_code_pipeline(
    embedding_provider: Any | None = None, **kwargs
) -> DocumentPipeline:
    """
    Create a pipeline optimized for code processing.

    Uses code-specific defaults:
    - Smaller batch sizes (code embeddings are more expensive)
    - Larger max text length (code files can be large)
    - Code-specific chunking
    """
    processor_config = ProcessorConfig(
        chunk_size=1024,
        chunk_overlap=100,
        extract_symbols=True,
        include_docstrings=True,
        include_type_hints=True,
        max_text_length=16384,
    )

    config = PipelineConfig(
        processor_config=processor_config,
        embedding_batch_size=16,
        default_processor="code",
        **kwargs,
    )

    return DocumentPipeline(embedding_provider=embedding_provider, config=config)


# =============================================================================
# Context Managers
# =============================================================================


@contextmanager
def pipeline_context(
    embedding_provider: Any | None = None, **kwargs
) -> Generator[DocumentPipeline, None, None]:
    """
    Sync context manager for pipeline usage.

    Example:
        with pipeline_context(embedding_provider=provider) as pipeline:
            # Use pipeline synchronously via asyncio.run()
            result = asyncio.run(pipeline.process(content, source_id))
    """
    pipeline = create_document_pipeline(embedding_provider, **kwargs)
    try:
        yield pipeline
    finally:
        pipeline.reset_metrics()


@asynccontextmanager
async def async_pipeline_context(
    embedding_provider: Any | None = None, **kwargs
) -> AsyncGenerator[DocumentPipeline, None]:
    """
    Async context manager for pipeline usage.

    Example:
        async with async_pipeline_context(embedding_provider=provider) as pipeline:
            result = await pipeline.process(content, source_id)
    """
    pipeline = create_document_pipeline(embedding_provider, **kwargs)
    try:
        yield pipeline
    finally:
        pipeline.reset_metrics()
