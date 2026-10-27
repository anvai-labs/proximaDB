"""
Unified Chunking Pipeline for ProximaDB

Production-ready pipeline that orchestrates:
- Multiple chunking strategies
- Batch embedding operations
- Async/streaming processing
- Error handling and validation
- Metrics collection and monitoring

Design Patterns:
- Pipeline Pattern: Sequential processing stages
- Strategy Pattern: Pluggable chunking strategies
- Observer Pattern: Progress callbacks and metrics
- Builder Pattern: Fluent configuration API
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from .base import ChunkingConfig, ChunkingStrategy, TextChunk, config_kwargs
from .code import CodeChunkingConfig
from .factory import ChunkingStrategyFactory
from .parser_utils import ConfigValidator, ParserMetrics, get_metrics_collector

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")
ChunkType = TypeVar("ChunkType", bound=TextChunk)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts"""
        ...

    async def embed_texts_async(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_texts"""
        ...

    @property
    def dimension(self) -> int:
        """Return embedding dimension"""
        ...


class VectorStore(Protocol):
    """Protocol for vector stores"""

    async def insert(self, records: list[dict[str, Any]]) -> None:
        """Insert records into the store"""
        ...

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors"""
        ...


# =============================================================================
# Configuration
# =============================================================================


class ProcessingMode(Enum):
    """Processing mode for the pipeline"""

    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"
    BATCH = "batch"


class ErrorHandling(Enum):
    """Error handling strategy"""

    FAIL_FAST = "fail_fast"
    SKIP_ERRORS = "skip_errors"
    COLLECT_ERRORS = "collect_errors"
    RETRY = "retry"


@dataclass
class PipelineConfig:
    """Configuration for the chunking pipeline"""

    # Chunking settings
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    chunking_config: ChunkingConfig | None = None

    # Embedding settings
    embedding_batch_size: int = 32
    embedding_timeout: float = 30.0
    max_text_length: int = 8192
    truncate_long_texts: bool = True

    # Processing settings
    processing_mode: ProcessingMode = ProcessingMode.ASYNC
    max_concurrent_tasks: int = 4
    buffer_size: int = 100

    # Error handling
    error_handling: ErrorHandling = ErrorHandling.COLLECT_ERRORS
    max_retries: int = 3
    retry_delay: float = 1.0

    # Progress and metrics
    enable_metrics: bool = True
    progress_callback: Callable[[int, int, str], None] | None = None

    # Memory management
    max_memory_mb: int = 512
    gc_threshold: int = 1000

    # Validation
    validate_chunks: bool = True
    min_chunk_quality: float = 0.5

    def __post_init__(self):
        if self.chunking_config is None:
            self.chunking_config = ChunkingConfig(strategy=self.chunking_strategy)


@dataclass
class PipelineResult:
    """Result of a pipeline operation"""

    success: bool
    chunks: list[TextChunk] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "chunk_count": self.chunk_count,
            "error_count": self.error_count,
            "metrics": self.metrics,
            "errors": self.errors[:10] if self.errors else [],
        }


@dataclass
class BatchResult:
    """Result of a batch processing operation"""

    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    results: list[PipelineResult] = field(default_factory=list)
    total_chunks: int = 0
    total_embeddings: int = 0
    processing_time_sec: float = 0.0
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.processed_items / self.total_items


# =============================================================================
# Processing Stages
# =============================================================================


class PipelineStage(ABC, Generic[T]):
    """Abstract base class for pipeline stages"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name"""
        pass

    @abstractmethod
    def process(self, input_data: T) -> T:
        """Process data synchronously"""
        pass

    async def process_async(self, input_data: T) -> T:
        """Process data asynchronously (default: wrap sync)"""
        return self.process(input_data)


class ValidationStage(PipelineStage[TextChunk]):
    """Validates chunks before processing"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validator = ConfigValidator()

    @property
    def name(self) -> str:
        return "validation"

    def process(self, chunk: TextChunk) -> TextChunk:
        if not self.config.validate_chunks:
            return chunk

        # Validate chunk text
        if not chunk.text or not chunk.text.strip():
            raise ValueError(f"Empty chunk: {chunk.chunk_id}")

        # Check text length
        if len(chunk.text) > self.config.max_text_length:
            if self.config.truncate_long_texts:
                chunk.text = chunk.text[: self.config.max_text_length]
                chunk.metadata["truncated"] = True
            else:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} exceeds max length: "
                    f"{len(chunk.text)} > {self.config.max_text_length}"
                )

        return chunk


class EnrichmentStage(PipelineStage[TextChunk]):
    """Enriches chunks with additional metadata"""

    def __init__(
        self, enrichment_funcs: list[Callable[[TextChunk], TextChunk]] | None = None
    ):
        self.enrichment_funcs = enrichment_funcs or []

    @property
    def name(self) -> str:
        return "enrichment"

    def process(self, chunk: TextChunk) -> TextChunk:
        for func in self.enrichment_funcs:
            chunk = func(chunk)
        return chunk

    def add_enrichment(self, func: Callable[[TextChunk], TextChunk]) -> None:
        """Add an enrichment function"""
        self.enrichment_funcs.append(func)


class FilterStage(PipelineStage[list[TextChunk]]):
    """Filters chunks based on criteria"""

    def __init__(self, predicates: list[Callable[[TextChunk], bool]] | None = None):
        self.predicates = predicates or []

    @property
    def name(self) -> str:
        return "filter"

    def process(self, chunks: list[TextChunk]) -> list[TextChunk]:
        if not self.predicates:
            return chunks

        return [
            chunk for chunk in chunks if all(pred(chunk) for pred in self.predicates)
        ]

    def add_predicate(self, predicate: Callable[[TextChunk], bool]) -> None:
        """Add a filter predicate"""
        self.predicates.append(predicate)


# =============================================================================
# Batch Processor
# =============================================================================


class BatchEmbedder:
    """Handles batch embedding operations with rate limiting and retries"""

    def __init__(
        self,
        provider: EmbeddingProvider,
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
    ):
        self.provider = provider
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._lock = threading.Lock()
        self._request_count = 0
        self._total_tokens = 0

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches with retries"""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Async version of batch embedding"""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = await self._embed_with_retry_async(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embed with retries on failure"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                with self._lock:
                    self._request_count += 1

                return self.provider.embed_texts(texts)

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embedding attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"Embedding failed after {self.max_retries} attempts: {last_error}"
        )

    async def _embed_with_retry_async(self, texts: list[str]) -> list[list[float]]:
        """Async embed with retries"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                with self._lock:
                    self._request_count += 1

                if hasattr(self.provider, "embed_texts_async"):
                    return await self.provider.embed_texts_async(texts)
                else:
                    # Fallback to sync in thread pool
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, self.provider.embed_texts, texts
                    )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Async embedding attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"Async embedding failed after {self.max_retries} attempts: {last_error}"
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "batch_size": self.batch_size,
        }


# =============================================================================
# Progress Tracking
# =============================================================================


class ProgressTracker:
    """Tracks progress of pipeline operations"""

    def __init__(self, callback: Callable[[int, int, str], None] | None = None):
        self.callback = callback
        self._current = 0
        self._total = 0
        self._status = "idle"
        self._lock = threading.Lock()
        self._start_time: float | None = None

    def start(self, total: int, status: str = "processing") -> None:
        """Start tracking progress"""
        with self._lock:
            self._current = 0
            self._total = total
            self._status = status
            self._start_time = time.time()
        self._notify()

    def update(self, increment: int = 1, status: str | None = None) -> None:
        """Update progress"""
        with self._lock:
            self._current += increment
            if status:
                self._status = status
        self._notify()

    def complete(self, status: str = "completed") -> None:
        """Mark as complete"""
        with self._lock:
            self._current = self._total
            self._status = status
        self._notify()

    def _notify(self) -> None:
        """Notify callback of progress"""
        if self.callback:
            self.callback(self._current, self._total, self._status)

    @property
    def elapsed_time(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def items_per_second(self) -> float:
        elapsed = self.elapsed_time
        if elapsed == 0:
            return 0.0
        return self._current / elapsed


# =============================================================================
# Main Pipeline
# =============================================================================


class ChunkingPipeline:
    """
    Unified chunking pipeline that orchestrates all processing stages.

    Features:
    - Multiple processing modes (sync, async, streaming, batch)
    - Batch embedding with retries
    - Progress tracking
    - Error collection
    - Metrics integration

    Example:
        pipeline = ChunkingPipeline(config)

        # Simple usage
        result = await pipeline.process_text("Your text here", "doc_1")

        # Batch processing
        results = await pipeline.process_batch([
            {"text": "Text 1", "source_id": "doc_1"},
            {"text": "Text 2", "source_id": "doc_2"},
        ])

        # Streaming
        async for chunk in pipeline.process_stream(large_text, "doc_1"):
            process(chunk)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config or PipelineConfig()
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

        # Initialize components
        self._init_chunker()
        self._init_stages()
        self._init_embedder()

        # State
        self._metrics_collector = get_metrics_collector()
        self._progress = ProgressTracker(self.config.progress_callback)
        self._errors: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _init_chunker(self) -> None:
        """Initialize the chunking strategy"""
        # For CODE strategy, ensure we use CodeChunkingConfig
        if self.config.chunking_strategy == ChunkingStrategy.CODE:
            if isinstance(self.config.chunking_config, CodeChunkingConfig):
                config = self.config.chunking_config
            else:
                # Widen a ChunkingConfig into a CodeChunkingConfig (a subclass,
                # so every base field is valid here). Derived rather than
                # hand-listed: the list this replaces carried five of the base
                # fields and dropped the rest, so min_chunk_size, max_chunk_size
                # and the semantic knobs were reset to defaults on the way in --
                # silently, because a dropped field is indistinguishable from an
                # unset one.
                source = self.config.chunking_config
                config = CodeChunkingConfig(**(config_kwargs(source) if source else {}))
            self.chunker = ChunkingStrategyFactory.create_strategy(
                self.config.chunking_strategy, config
            )
        else:
            self.chunker = ChunkingStrategyFactory.create_strategy(
                self.config.chunking_strategy, self.config.chunking_config
            )

    def _init_stages(self) -> None:
        """Initialize processing stages"""
        self.validation_stage = ValidationStage(self.config)
        self.enrichment_stage = EnrichmentStage()
        self.filter_stage = FilterStage()

    def _init_embedder(self) -> None:
        """Initialize batch embedder if provider available"""
        if self.embedding_provider:
            self.embedder = BatchEmbedder(
                self.embedding_provider,
                batch_size=self.config.embedding_batch_size,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
                timeout=self.config.embedding_timeout,
            )
        else:
            self.embedder = None

    # -------------------------------------------------------------------------
    # Configuration API (Builder Pattern)
    # -------------------------------------------------------------------------

    def with_strategy(self, strategy: ChunkingStrategy) -> "ChunkingPipeline":
        """Set chunking strategy"""
        self.config.chunking_strategy = strategy
        self._init_chunker()
        return self

    def with_embedding_provider(
        self, provider: EmbeddingProvider
    ) -> "ChunkingPipeline":
        """Set embedding provider"""
        self.embedding_provider = provider
        self._init_embedder()
        return self

    def with_vector_store(self, store: VectorStore) -> "ChunkingPipeline":
        """Set vector store"""
        self.vector_store = store
        return self

    def with_enrichment(
        self, func: Callable[[TextChunk], TextChunk]
    ) -> "ChunkingPipeline":
        """Add chunk enrichment function"""
        self.enrichment_stage.add_enrichment(func)
        return self

    def with_filter(self, predicate: Callable[[TextChunk], bool]) -> "ChunkingPipeline":
        """Add chunk filter predicate"""
        self.filter_stage.add_predicate(predicate)
        return self

    def with_progress_callback(
        self, callback: Callable[[int, int, str], None]
    ) -> "ChunkingPipeline":
        """Set progress callback"""
        self.config.progress_callback = callback
        self._progress = ProgressTracker(callback)
        return self

    # -------------------------------------------------------------------------
    # Core Processing
    # -------------------------------------------------------------------------

    def process_text(
        self, text: str, source_id: str, metadata: dict[str, Any] | None = None
    ) -> PipelineResult:
        """
        Process text synchronously.

        Args:
            text: Text to process
            source_id: Unique identifier for the source
            metadata: Optional metadata to include

        Returns:
            PipelineResult with chunks and optionally embeddings
        """
        start_time = time.time()
        errors = []

        try:
            # Stage 1: Chunk
            chunks = self.chunker.chunk(text, source_id, metadata)

            # Stage 2: Validate and enrich
            processed_chunks = []
            for chunk in chunks:
                try:
                    chunk = self.validation_stage.process(chunk)
                    chunk = self.enrichment_stage.process(chunk)
                    processed_chunks.append(chunk)
                except Exception as e:
                    errors.append(
                        {
                            "stage": "validation",
                            "chunk_id": chunk.chunk_id,
                            "error": str(e),
                        }
                    )
                    if self.config.error_handling == ErrorHandling.FAIL_FAST:
                        raise

            # Stage 3: Filter
            filtered_chunks = self.filter_stage.process(processed_chunks)

            # Stage 4: Embed (if provider available)
            embeddings = []
            if self.embedder and filtered_chunks:
                try:
                    texts = [c.text for c in filtered_chunks]
                    embeddings = self.embedder.embed_batch(texts)
                except Exception as e:
                    errors.append({"stage": "embedding", "error": str(e)})
                    if self.config.error_handling == ErrorHandling.FAIL_FAST:
                        raise

            # Collect metrics
            processing_time = time.time() - start_time
            metrics = {
                "processing_time_sec": processing_time,
                "input_length": len(text),
                "chunk_count": len(filtered_chunks),
                "embedding_count": len(embeddings),
                "chunks_per_second": (
                    len(filtered_chunks) / processing_time if processing_time > 0 else 0
                ),
            }

            if self.config.enable_metrics:
                self._record_metrics(metrics)

            return PipelineResult(
                success=len(errors) == 0,
                chunks=filtered_chunks,
                embeddings=embeddings,
                errors=errors,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return PipelineResult(
                success=False,
                errors=[{"stage": "pipeline", "error": str(e)}],
                metrics={"processing_time_sec": time.time() - start_time},
            )

    async def process_text_async(
        self, text: str, source_id: str, metadata: dict[str, Any] | None = None
    ) -> PipelineResult:
        """
        Process text asynchronously.

        Provides better performance for I/O-bound operations.
        """
        start_time = time.time()
        errors = []

        try:
            # Stage 1: Chunk (run in thread pool for CPU-bound work)
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None, lambda: self.chunker.chunk(text, source_id, metadata)
            )

            # Stage 2: Validate and enrich
            processed_chunks = []
            for chunk in chunks:
                try:
                    chunk = await self.validation_stage.process_async(chunk)
                    chunk = await self.enrichment_stage.process_async(chunk)
                    processed_chunks.append(chunk)
                except Exception as e:
                    errors.append(
                        {
                            "stage": "validation",
                            "chunk_id": chunk.chunk_id,
                            "error": str(e),
                        }
                    )
                    if self.config.error_handling == ErrorHandling.FAIL_FAST:
                        raise

            # Stage 3: Filter
            filtered_chunks = await self.filter_stage.process_async(processed_chunks)

            # Stage 4: Embed asynchronously
            embeddings = []
            if self.embedder and filtered_chunks:
                try:
                    texts = [c.text for c in filtered_chunks]
                    embeddings = await self.embedder.embed_batch_async(texts)
                except Exception as e:
                    errors.append({"stage": "embedding", "error": str(e)})
                    if self.config.error_handling == ErrorHandling.FAIL_FAST:
                        raise

            # Collect metrics
            processing_time = time.time() - start_time
            metrics = {
                "processing_time_sec": processing_time,
                "input_length": len(text),
                "chunk_count": len(filtered_chunks),
                "embedding_count": len(embeddings),
                "chunks_per_second": (
                    len(filtered_chunks) / processing_time if processing_time > 0 else 0
                ),
                "mode": "async",
            }

            if self.config.enable_metrics:
                self._record_metrics(metrics)

            return PipelineResult(
                success=len(errors) == 0,
                chunks=filtered_chunks,
                embeddings=embeddings,
                errors=errors,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Async pipeline processing failed: {e}")
            return PipelineResult(
                success=False,
                errors=[{"stage": "pipeline", "error": str(e)}],
                metrics={"processing_time_sec": time.time() - start_time},
            )

    # -------------------------------------------------------------------------
    # Streaming Processing
    # -------------------------------------------------------------------------

    def process_stream(
        self,
        text_source: "str | Iterable[str]",
        source_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[TextChunk, None, None]:
        """
        Yield validated/enriched chunks one at a time.

        ``text_source`` may be a single ``str`` (backward compatible) or an
        iterable of text pieces (e.g. successive file reads).

        Memory profile depends on the strategy:

        * **Streamable strategy** (``chunker.supports_streaming is True`` —
          sliding-window / fixed-size / sentence / paragraph) **and** an
          iterable ``text_source``: a *genuinely incremental, bounded-memory*
          stream. Chunking maintains only a small local buffer and chunks are
          produced before the whole input is consumed; the full input and the
          full chunk list are never materialized.
        * **Non-streamable strategy** (code / semantic-embedding / recursive /
          structural-semantic): the *honest fallback* — ``chunk_stream``
          materializes the input and runs the batch chunker (these need the
          whole document), but chunks are still yielded one at a time so the
          downstream validation/enrichment stages run per-chunk.

        A plain ``str`` ``text_source`` is always materialized (it already is),
        so the bounded-memory benefit applies specifically to a streamable
        strategy fed an iterable source.
        """
        try:
            for chunk in self.chunker.chunk_stream(text_source, source_id, metadata):
                try:
                    chunk = self.validation_stage.process(chunk)
                    chunk = self.enrichment_stage.process(chunk)
                    yield chunk
                except Exception as e:
                    if self.config.error_handling == ErrorHandling.FAIL_FAST:
                        raise
                    logger.warning(f"Skipping chunk {chunk.chunk_id}: {e}")

        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            if self.config.error_handling == ErrorHandling.FAIL_FAST:
                raise

    async def process_stream_async(
        self,
        text_source: "str | Iterable[str]",
        source_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[TextChunk, None]:
        """
        Async version of :meth:`process_stream`.

        Same memory profile applies (genuinely incremental for a streamable
        strategy + iterable source; honest materialize-then-chunk fallback
        otherwise). The chunk iterator is advanced in a thread-pool executor so
        CPU-bound chunking does not block the event loop, while the
        validation/enrichment stages run per-chunk on the loop.
        """
        try:
            loop = asyncio.get_event_loop()
            chunk_iter = iter(
                self.chunker.chunk_stream(text_source, source_id, metadata)
            )
            _SENTINEL = object()

            while True:
                chunk = await loop.run_in_executor(
                    None, lambda: next(chunk_iter, _SENTINEL)
                )
                if chunk is _SENTINEL:
                    break
                try:
                    chunk = await self.validation_stage.process_async(chunk)
                    chunk = await self.enrichment_stage.process_async(chunk)
                    yield chunk
                except Exception as e:
                    if self.config.error_handling == ErrorHandling.FAIL_FAST:
                        raise
                    logger.warning(f"Skipping chunk {chunk.chunk_id}: {e}")

        except Exception as e:
            logger.error(f"Async stream processing failed: {e}")
            if self.config.error_handling == ErrorHandling.FAIL_FAST:
                raise

    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------

    def process_batch(self, items: list[dict[str, Any]]) -> BatchResult:
        """
        Process multiple items in batch.

        Args:
            items: List of dicts with 'text' and 'source_id' keys

        Returns:
            BatchResult with aggregated statistics
        """
        start_time = time.time()
        results = []
        errors = []
        total_chunks = 0
        total_embeddings = 0

        self._progress.start(len(items), "batch_processing")

        for i, item in enumerate(items):
            try:
                text = item.get("text", "")
                source_id = item.get("source_id", f"batch_{i}")
                metadata = item.get("metadata")

                result = self.process_text(text, source_id, metadata)
                results.append(result)
                total_chunks += result.chunk_count
                total_embeddings += len(result.embeddings)

                if not result.success:
                    errors.extend(result.errors)

            except Exception as e:
                errors.append(
                    {
                        "item_index": i,
                        "source_id": item.get("source_id"),
                        "error": str(e),
                    }
                )
                if self.config.error_handling == ErrorHandling.FAIL_FAST:
                    break

            self._progress.update(1)

        self._progress.complete()

        processed = sum(1 for r in results if r.success)

        return BatchResult(
            total_items=len(items),
            processed_items=processed,
            failed_items=len(items) - processed,
            results=results,
            total_chunks=total_chunks,
            total_embeddings=total_embeddings,
            processing_time_sec=time.time() - start_time,
            errors=errors,
        )

    async def process_batch_async(
        self, items: list[dict[str, Any]], concurrent_limit: int | None = None
    ) -> BatchResult:
        """
        Process multiple items concurrently.

        Args:
            items: List of dicts with 'text' and 'source_id' keys
            concurrent_limit: Max concurrent tasks (default from config)

        Returns:
            BatchResult with aggregated statistics
        """
        start_time = time.time()
        concurrent_limit = concurrent_limit or self.config.max_concurrent_tasks

        self._progress.start(len(items), "async_batch_processing")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent_limit)

        async def process_item(
            item: dict[str, Any], index: int
        ) -> tuple[int, PipelineResult]:
            async with semaphore:
                text = item.get("text", "")
                source_id = item.get("source_id", f"batch_{index}")
                metadata = item.get("metadata")

                result = await self.process_text_async(text, source_id, metadata)
                self._progress.update(1)
                return index, result

        # Process all items concurrently
        tasks = [process_item(item, i) for i, item in enumerate(items)]

        completed = await asyncio.gather(*tasks, return_exceptions=True)

        self._progress.complete()

        # Collect results
        results = [None] * len(items)
        errors = []
        total_chunks = 0
        total_embeddings = 0

        for item in completed:
            if isinstance(item, Exception):
                errors.append({"error": str(item)})
            else:
                index, result = item
                results[index] = result
                total_chunks += result.chunk_count
                total_embeddings += len(result.embeddings)
                if not result.success:
                    errors.extend(result.errors)

        processed = sum(1 for r in results if r and r.success)

        return BatchResult(
            total_items=len(items),
            processed_items=processed,
            failed_items=len(items) - processed,
            results=[r for r in results if r is not None],
            total_chunks=total_chunks,
            total_embeddings=total_embeddings,
            processing_time_sec=time.time() - start_time,
            errors=errors,
        )

    # -------------------------------------------------------------------------
    # File Processing
    # -------------------------------------------------------------------------

    def process_file(
        self, file_path: str | Path, encoding: str = "utf-8"
    ) -> PipelineResult:
        """
        Process a file.

        Args:
            file_path: Path to the file
            encoding: File encoding

        Returns:
            PipelineResult
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return PipelineResult(
                success=False, errors=[{"error": f"File not found: {file_path}"}]
            )

        try:
            text = file_path.read_text(encoding=encoding)
            return self.process_text(
                text,
                source_id=str(file_path),
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_extension": file_path.suffix,
                },
            )
        except Exception as e:
            return PipelineResult(
                success=False, errors=[{"error": f"Failed to read file: {e}"}]
            )

    async def process_file_async(
        self, file_path: str | Path, encoding: str = "utf-8"
    ) -> PipelineResult:
        """Async version of process_file"""
        file_path = Path(file_path)

        if not file_path.exists():
            return PipelineResult(
                success=False, errors=[{"error": f"File not found: {file_path}"}]
            )

        try:
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, lambda: file_path.read_text(encoding=encoding)
            )

            return await self.process_text_async(
                text,
                source_id=str(file_path),
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_extension": file_path.suffix,
                },
            )
        except Exception as e:
            return PipelineResult(
                success=False, errors=[{"error": f"Failed to read file: {e}"}]
            )

    def process_directory(
        self, directory: str | Path, pattern: str = "**/*", recursive: bool = True
    ) -> BatchResult:
        """
        Process all matching files in a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern for files
            recursive: Whether to search recursively

        Returns:
            BatchResult
        """
        directory = Path(directory)

        if not directory.is_dir():
            return BatchResult(errors=[{"error": f"Not a directory: {directory}"}])

        # Collect files
        if recursive:
            files = list(directory.glob(pattern))
        else:
            files = list(directory.glob(pattern.replace("**", "*", 1)))

        # Filter to actual files (not directories)
        files = [f for f in files if f.is_file()]

        # Process as batch
        items = [
            {"text": None, "file_path": str(f), "source_id": str(f)} for f in files
        ]

        start_time = time.time()
        results = []
        errors = []

        self._progress.start(len(files), "directory_processing")

        for file_path in files:
            result = self.process_file(file_path)
            results.append(result)
            if not result.success:
                errors.extend(result.errors)
            self._progress.update(1)

        self._progress.complete()

        processed = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunk_count for r in results)
        total_embeddings = sum(len(r.embeddings) for r in results)

        return BatchResult(
            total_items=len(files),
            processed_items=processed,
            failed_items=len(files) - processed,
            results=results,
            total_chunks=total_chunks,
            total_embeddings=total_embeddings,
            processing_time_sec=time.time() - start_time,
            errors=errors,
        )

    async def process_directory_async(
        self,
        directory: str | Path,
        pattern: str = "**/*",
        recursive: bool = True,
        concurrent_limit: int | None = None,
    ) -> BatchResult:
        """Async version of process_directory"""
        directory = Path(directory)

        if not directory.is_dir():
            return BatchResult(errors=[{"error": f"Not a directory: {directory}"}])

        # Collect files
        if recursive:
            files = list(directory.glob(pattern))
        else:
            files = list(directory.glob(pattern.replace("**", "*", 1)))

        files = [f for f in files if f.is_file()]
        concurrent_limit = concurrent_limit or self.config.max_concurrent_tasks

        start_time = time.time()
        self._progress.start(len(files), "async_directory_processing")

        semaphore = asyncio.Semaphore(concurrent_limit)

        async def process_file_with_semaphore(f: Path) -> PipelineResult:
            async with semaphore:
                result = await self.process_file_async(f)
                self._progress.update(1)
                return result

        tasks = [process_file_with_semaphore(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        self._progress.complete()

        # Collect results
        processed_results = []
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({"file": str(files[i]), "error": str(result)})
            else:
                processed_results.append(result)
                if not result.success:
                    errors.extend(result.errors)

        processed = sum(1 for r in processed_results if r.success)
        total_chunks = sum(r.chunk_count for r in processed_results)
        total_embeddings = sum(len(r.embeddings) for r in processed_results)

        return BatchResult(
            total_items=len(files),
            processed_items=processed,
            failed_items=len(files) - processed,
            results=processed_results,
            total_chunks=total_chunks,
            total_embeddings=total_embeddings,
            processing_time_sec=time.time() - start_time,
            errors=errors,
        )

    # -------------------------------------------------------------------------
    # Vector Store Integration
    # -------------------------------------------------------------------------

    async def process_and_store(
        self, text: str, source_id: str, metadata: dict[str, Any] | None = None
    ) -> PipelineResult:
        """
        Process text and store results in vector store.

        Combines chunking, embedding, and storage in one operation.
        """
        if not self.vector_store:
            raise ValueError("No vector store configured")

        if not self.embedding_provider:
            raise ValueError("No embedding provider configured")

        # Process text
        result = await self.process_text_async(text, source_id, metadata)

        if not result.success or not result.chunks:
            return result

        # Create vector records
        records = []
        for chunk, embedding in zip(result.chunks, result.embeddings):
            records.append(
                {
                    "id": chunk.chunk_id,
                    "vector": embedding,
                    "metadata": {
                        **chunk.metadata,
                        "text": chunk.text,
                        "source_id": source_id,
                    },
                }
            )

        # Store in vector store
        try:
            await self.vector_store.insert(records)
            result.metrics["records_stored"] = len(records)
        except Exception as e:
            result.errors.append({"stage": "storage", "error": str(e)})
            result.success = False

        return result

    # -------------------------------------------------------------------------
    # Metrics and Monitoring
    # -------------------------------------------------------------------------

    def _record_metrics(self, metrics: dict[str, Any]) -> None:
        """Record metrics to the collector"""
        if not self.config.enable_metrics:
            return

        collector = get_metrics_collector()
        parser_metrics = ParserMetrics(
            language="pipeline",
            file_path="<pipeline>",
            parse_time_ms=metrics.get("processing_time_sec", 0) * 1000,
            symbol_count=metrics.get("chunk_count", 0),
        )
        collector.record(parser_metrics)

    def get_metrics(self) -> dict[str, Any]:
        """Get current pipeline metrics"""
        collector = get_metrics_collector()
        summary = collector.get_summary()

        return {
            **summary,
            "embedder_stats": self.embedder.stats if self.embedder else {},
            "progress": {
                "elapsed_time": self._progress.elapsed_time,
                "items_per_second": self._progress.items_per_second,
            },
        }

    def reset_metrics(self) -> None:
        """Reset all metrics"""
        collector = get_metrics_collector()
        collector.clear()
        self._errors.clear()


# =============================================================================
# Factory Functions
# =============================================================================


def create_pipeline(
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    **kwargs,
) -> ChunkingPipeline:
    """
    Create a configured pipeline instance.

    Args:
        strategy: Chunking strategy to use
        embedding_provider: Optional embedding provider
        vector_store: Optional vector store
        **kwargs: Additional PipelineConfig options

    Returns:
        Configured ChunkingPipeline

    Example:
        pipeline = create_pipeline(
            strategy=ChunkingStrategy.CODE,
            embedding_batch_size=64,
            max_concurrent_tasks=8
        )
    """
    config = PipelineConfig(chunking_strategy=strategy, **kwargs)

    return ChunkingPipeline(
        config=config, embedding_provider=embedding_provider, vector_store=vector_store
    )


def create_code_pipeline(
    embedding_provider: EmbeddingProvider | None = None, **kwargs
) -> ChunkingPipeline:
    """Create a pipeline optimized for code processing"""
    return create_pipeline(
        strategy=ChunkingStrategy.CODE,
        embedding_provider=embedding_provider,
        embedding_batch_size=16,  # Smaller batches for code
        max_text_length=16384,  # Longer for code files
        **kwargs,
    )


def create_document_pipeline(
    embedding_provider: EmbeddingProvider | None = None, **kwargs
) -> ChunkingPipeline:
    """Create a pipeline optimized for document processing"""
    return create_pipeline(
        strategy=ChunkingStrategy.SEMANTIC,
        embedding_provider=embedding_provider,
        embedding_batch_size=32,
        max_text_length=8192,
        **kwargs,
    )


# =============================================================================
# Context Managers
# =============================================================================


@contextmanager
def pipeline_context(
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC, **kwargs
) -> Generator[ChunkingPipeline, None, None]:
    """
    Context manager for pipeline usage.

    Example:
        with pipeline_context(strategy=ChunkingStrategy.CODE) as pipeline:
            result = pipeline.process_text(code, "file.py")
    """
    pipeline = create_pipeline(strategy=strategy, **kwargs)
    try:
        yield pipeline
    finally:
        pipeline.reset_metrics()


@asynccontextmanager
async def async_pipeline_context(
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC, **kwargs
) -> AsyncGenerator[ChunkingPipeline, None]:
    """
    Async context manager for pipeline usage.

    Example:
        async with async_pipeline_context() as pipeline:
            result = await pipeline.process_text_async(text, "doc")
    """
    pipeline = create_pipeline(strategy=strategy, **kwargs)
    try:
        yield pipeline
    finally:
        pipeline.reset_metrics()
