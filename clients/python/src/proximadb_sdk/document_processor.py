"""
Document Processor Abstraction Layer for ProximaDB

Provides a unified interface for processing different document types:
- Code (Python, Rust, Go, etc.)
- Documents (PDF, images with OCR)
- Binaries (DLL, EXE analysis)
- Generic text

Design Patterns:
- Strategy Pattern: Different processors for different document types
- Template Method: Common processing workflow with customizable steps
- Adapter Pattern: Normalizes different embedding providers
- Factory Pattern: Creates appropriate processors based on content

Integration Points:
- Chunking strategies (chunking_strategies/)
- Embedding providers (embedding_providers/)
- CodeKnowledgeBuilder (code_knowledge.py)
- ChunkingPipeline (pipeline.py)
"""

import asyncio
import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================


class DocumentType(Enum):
    """Types of documents that can be processed"""

    CODE = auto()  # Source code (Python, Rust, etc.)
    MARKDOWN = auto()  # Markdown documents
    TEXT = auto()  # Plain text
    PDF = auto()  # PDF documents (with OCR)
    IMAGE = auto()  # Images (with OCR)
    BINARY = auto()  # Binary files (DLL, EXE)
    HTML = auto()  # HTML documents
    JSON = auto()  # JSON data
    XML = auto()  # XML documents
    UNKNOWN = auto()  # Unknown type


class ProcessingStrategy(Enum):
    """Processing strategy hints"""

    FAST = "fast"  # Prioritize speed
    ACCURATE = "accurate"  # Prioritize quality
    BALANCED = "balanced"  # Balance speed/quality
    MINIMAL = "minimal"  # Minimal processing


# =============================================================================
# Protocols (Interface Definitions)
# =============================================================================


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding"""
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension"""
        ...


class AsyncEmbeddingProvider(Protocol):
    """Protocol for async embedding providers"""

    async def embed_texts_async(self, texts: list[str]) -> list[list[float]]:
        """Asynchronous embedding"""
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension"""
        ...


class VectorStore(Protocol):
    """Protocol for vector stores"""

    async def insert(self, records: list[dict[str, Any]]) -> None:
        """Insert records"""
        ...


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ProcessedChunk:
    """A processed chunk ready for embedding"""

    chunk_id: str
    text: str
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_text: str | None = (
        None  # Text prepared for embedding (may differ from text)
    )

    def __post_init__(self):
        if self.embedding_text is None:
            self.embedding_text = self.text


@dataclass
class VectorRecord:
    """A vector record ready for storage"""

    id: str
    vector: list[float]
    metadata: dict[str, Any]
    text: str
    source_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vector": self.vector,
            "metadata": {
                **self.metadata,
                "text": self.text,
                "source_id": self.source_id,
            },
        }


@dataclass
class ProcessingResult:
    """Result of document processing"""

    success: bool
    source_id: str
    document_type: DocumentType
    chunks: list[ProcessedChunk] = field(default_factory=list)
    vectors: list[VectorRecord] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def vector_count(self) -> int:
        return len(self.vectors)


@dataclass
class ProcessorConfig:
    """Configuration for document processors"""

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 4096

    # Embedding settings
    embedding_batch_size: int = 32
    max_text_length: int = 8192
    truncate_long_texts: bool = True

    # Processing settings
    strategy: ProcessingStrategy = ProcessingStrategy.BALANCED
    include_metadata: bool = True
    #: Label each text chunk with the heading path it lives under
    #: (`heading_path`, `heading_title`, `heading_level`).
    #:
    #: This flag existed but was READ BY NOTHING, so it is being given the
    #: meaning its name already implied rather than adding a third
    #: structure-ish flag beside a dead one. Nothing depended on the old
    #: (absent) behaviour, so no caller can break: `False` simply skips
    #: annotation, which is what happens today.
    preserve_structure: bool = True

    #: Drop near-duplicate chunks before they are embedded and stored.
    #:
    #: **Default OFF**, unlike heading annotation, and the difference is the
    #: point: annotation only ADDS metadata, while this REMOVES chunks, so
    #: enabling it changes what a tenant has stored. Measured safe (24-31%
    #: fewer chunks with no eval answer lost) and still opt-in, per the
    #: repo's ship-default-OFF-until-baked rule. Turning it on by default is a
    #: product decision about stored data, not a code one.
    #:
    #: Removals are recorded on the chunk that absorbed them
    #: (`duplicates_absorbed`, `duplicate_spans`), never silent.
    deduplicate_chunks: bool = False

    #: Jaccard similarity at or above which two chunks are the same. High by
    #: design: keeping a duplicate costs a little money, while dropping a
    #: distinct chunk removes content nothing downstream can detect is
    #: missing. Measured: almost all of the available saving is at 1.0 (exact
    #: duplicates), so a conservative threshold gives up very little.
    deduplicate_threshold: float = 0.9

    #: Prefix each chunk's EMBEDDED text with where it sits in the document
    #: (title > heading path), leaving the stored span and text untouched.
    #:
    #: **Default OFF, and this is the only capability that costs money.**
    #: Enrichment lengthens what is sent to the model, so KEU rises for every
    #: chunk in the collection. TD-CHUNK-3 requires it to be gated on
    #: recall-per-KEU rather than recall alone; `_last_pass_stats`
    #: ["enrichment_tax"] reports the multiplier actually applied per document,
    #: which is the denominator of that ratio.
    enrich_context: bool = False

    #: Small-to-big retrieval: group consecutive chunks into larger parent
    #: spans, so a precise child matches and the roomier parent is returned.
    #:
    #: **Default OFF.** It adds parent chunks, so KSU rises; what it buys is
    #: read cost and answer quality, and that trade has to be measured on the
    #: caller's corpus rather than assumed. Parents are emitted ONCE and
    #: referenced by `parent_id`, never copied onto each child.
    link_parent_chunks: bool = False

    #: Maximum characters a parent may span. ``None`` derives it as **twice
    #: `chunk_size`**, which is the measured optimum rather than a round
    #: number: on the eval corpus the containment gain (0.75 -> 0.88) is
    #: already complete at 2x, while KSU is flat at ~1.74x for every window
    #: tried (the parent layer is a second copy of the covered text either way)
    #: and density degrades as the window grows (0.247 -> 0.229 at 8x). So a
    #: bigger window costs the same and buys less. Derived rather than
    #: hardcoded so it cannot drift when `chunk_size` changes.
    #:
    #: A parent also never crosses a heading-path change, because a parent
    #: spanning two sections answers as neither.
    parent_window: int | None = None

    # Code-specific settings
    include_docstrings: bool = True
    include_comments: bool = True
    include_type_hints: bool = True
    extract_symbols: bool = True

    # Document-specific settings
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    pdf_dpi: int = 300


# =============================================================================
# Embedding Provider Adapter
# =============================================================================


class EmbeddingProviderAdapter:
    """
    Adapts different embedding provider interfaces to a unified API.

    Handles:
    - Sync to async conversion
    - Batch processing
    - Error handling and retries
    - Cost tracking
    """

    def __init__(
        self,
        provider: EmbeddingProvider | AsyncEmbeddingProvider,
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.provider = provider
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._is_async = hasattr(provider, "embed_texts_async")
        self._lock = threading.Lock()
        self._request_count = 0
        self._total_texts = 0

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if hasattr(self.provider, "dimension"):
            return self.provider.dimension
        return 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding with batching"""
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._embed_batch_sync(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_texts_async(self, texts: list[str]) -> list[list[float]]:
        """Asynchronous embedding with batching"""
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = await self._embed_batch_async(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch synchronously with retries"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                with self._lock:
                    self._request_count += 1
                    self._total_texts += len(texts)

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

    async def _embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch asynchronously with retries"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                with self._lock:
                    self._request_count += 1
                    self._total_texts += len(texts)

                if self._is_async:
                    return await self.provider.embed_texts_async(texts)
                else:
                    # Run sync provider in thread pool
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
        """Get adapter statistics"""
        return {
            "request_count": self._request_count,
            "total_texts": self._total_texts,
            "batch_size": self.batch_size,
            "is_async": self._is_async,
        }


class PlaceholderEmbeddingProvider:
    """
    Generates deterministic placeholder embeddings for testing.

    NOT for production use - creates hash-based embeddings that
    are not semantically meaningful.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate placeholder embeddings"""
        return [self._generate_embedding(text) for text in texts]

    async def embed_texts_async(self, texts: list[str]) -> list[list[float]]:
        """Async version (same as sync for placeholders)"""
        return self.embed_texts(texts)

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text hash"""
        hash_bytes = hashlib.sha256(text.encode()).digest()

        embedding = []
        for i in range(self._dimension):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # Normalize to [-1, 1]
            embedding.append(value)

        return embedding


# =============================================================================
# Document Processor Base Class
# =============================================================================


class DocumentProcessor(ABC):
    """
    Abstract base class for document processors.

    Each processor handles a specific document type and knows how to:
    1. Detect if it can handle a document
    2. Chunk the document appropriately
    3. Prepare text for embedding (may transform the text)
    4. Generate metadata
    """

    def __init__(self, config: ProcessorConfig | None = None):
        self.config = config or ProcessorConfig()

    @property
    @abstractmethod
    def supported_types(self) -> list[DocumentType]:
        """Document types this processor can handle"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Processor name"""
        pass

    @abstractmethod
    def can_process(self, content: str, file_path: str | None = None) -> bool:
        """Check if this processor can handle the content"""
        pass

    @abstractmethod
    def chunk(
        self, content: str, source_id: str, metadata: dict[str, Any] | None = None
    ) -> list[ProcessedChunk]:
        """Chunk the content into processable pieces"""
        pass

    def prepare_for_embedding(self, chunk: ProcessedChunk) -> str:
        """
        Prepare chunk text for embedding.

        Override this to customize how text is prepared for different
        embedding models or document types.
        """
        return chunk.embedding_text or chunk.text

    def enrich_metadata(
        self, chunk: ProcessedChunk, source_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Enrich chunk metadata with additional information.

        Override this to add processor-specific metadata.
        """
        metadata = {
            **chunk.metadata,
            "processor": self.name,
            "chunk_id": chunk.chunk_id,
            "start_pos": chunk.start_pos,
            "end_pos": chunk.end_pos,
            "text_length": len(chunk.text),
        }
        # start_pos/end_pos are persisted from here, so their MEANING must be
        # persisted alongside them. Strategies stamp this via add_chunk_metadata;
        # a chunk that bypassed it (the deprecated byte-offset code path) is
        # recorded as legacy rather than left ambiguous — an unmarked offset is
        # indistinguishable from an exact one to a later reader.
        from .chunking_strategies.base import OFFSET_BASIS_CHAR, OFFSET_CONTRACT_LEGACY

        metadata.setdefault("offset_basis", OFFSET_BASIS_CHAR)
        metadata.setdefault("offset_contract", OFFSET_CONTRACT_LEGACY)

        if source_metadata:
            metadata["source"] = source_metadata

        return metadata

    async def process(
        self,
        content: str,
        source_id: str,
        embedding_adapter: EmbeddingProviderAdapter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProcessingResult:
        """
        Process a document end-to-end.

        Args:
            content: Document content
            source_id: Unique identifier for the source
            embedding_adapter: Adapter for generating embeddings
            metadata: Additional metadata to include

        Returns:
            ProcessingResult with chunks and optionally vectors
        """
        start_time = time.time()
        errors = []

        try:
            # Step 1: Chunk the content
            chunks = self.chunk(content, source_id, metadata)

            # Step 2: Enrich metadata
            for chunk in chunks:
                chunk.metadata = self.enrich_metadata(chunk, metadata)

            # Step 3: Generate embeddings if adapter provided
            vectors = []
            if embedding_adapter and chunks:
                try:
                    # Prepare texts for embedding
                    texts = [self.prepare_for_embedding(c) for c in chunks]

                    # Generate embeddings
                    embeddings = await embedding_adapter.embed_texts_async(texts)

                    # Create vector records
                    for chunk, embedding in zip(chunks, embeddings):
                        vectors.append(
                            VectorRecord(
                                id=chunk.chunk_id,
                                vector=embedding,
                                metadata=chunk.metadata,
                                text=chunk.text,
                                source_id=source_id,
                            )
                        )

                except Exception as e:
                    errors.append({"stage": "embedding", "error": str(e)})
                    logger.error(f"Embedding failed: {e}")

            processing_time = time.time() - start_time

            return ProcessingResult(
                success=len(errors) == 0,
                source_id=source_id,
                document_type=(
                    self.supported_types[0]
                    if self.supported_types
                    else DocumentType.UNKNOWN
                ),
                chunks=chunks,
                vectors=vectors,
                errors=errors,
                metrics={
                    "processing_time_sec": processing_time,
                    "chunk_count": len(chunks),
                    "vector_count": len(vectors),
                    "content_length": len(content),
                    "processor": self.name,
                },
            )

        except Exception as e:
            logger.error(f"Processing failed for {source_id}: {e}")
            return ProcessingResult(
                success=False,
                source_id=source_id,
                document_type=DocumentType.UNKNOWN,
                errors=[{"stage": "processing", "error": str(e)}],
                metrics={"processing_time_sec": time.time() - start_time},
            )


# =============================================================================
# Code Document Processor
# =============================================================================


class CodeDocumentProcessor(DocumentProcessor):
    """
    Processor for source code documents.

    Uses AST-based parsing via tree-sitter for accurate symbol extraction.
    Prepares code-specific embeddings with documentation context.
    """

    #: Extensions this processor CLAIMS, for routing only.
    #:
    #: Derived from the shared STATIC table rather than restated here. The
    #: reasoning that used to live in this comment now lives with the table:
    #: routing must stay wide so the capability check can fail loudly, which
    #: means it must NOT depend on the optional `codegraph` extra. Keeping a
    #: second literal copy is what let that decision be applied here and nowhere
    #: else, so the same class of bug reappeared in code_knowledge.
    @staticmethod
    def _language_patterns() -> dict[str, list[str]]:
        from .chunking_strategies.code import STATIC_EXTENSION_TO_LANGUAGE

        out: dict[str, list[str]] = {}
        for ext, lang in STATIC_EXTENSION_TO_LANGUAGE.items():
            out.setdefault(lang, []).append(ext)
        return out

    def __init__(self, config: ProcessorConfig | None = None):
        super().__init__(config)
        self._chunker = None

    @property
    def supported_types(self) -> list[DocumentType]:
        return [DocumentType.CODE]

    @property
    def name(self) -> str:
        return "code"

    def can_process(self, content: str, file_path: str | None = None) -> bool:
        """Check if content looks like source code"""
        if file_path:
            ext = Path(file_path).suffix.lower()
            for exts in self._language_patterns().values():
                if ext in exts:
                    return True

        # Content-based detection
        code_indicators = [
            "def ",
            "class ",
            "import ",
            "from ",  # Python
            "fn ",
            "struct ",
            "impl ",
            "use ",  # Rust
            "func ",
            "package ",
            "import ",  # Go
            "public ",
            "private ",
            "void ",  # Java/C++
            "const ",
            "let ",
            "var ",
            "function ",  # JavaScript
        ]

        return any(indicator in content for indicator in code_indicators)

    def _get_chunker(self):
        """Lazy initialization of code chunker"""
        if self._chunker is None:
            from .chunking_strategies.code import (
                CodeChunkingConfig,
                CodeChunkingStrategy,
            )

            # Map ProcessorConfig settings to CodeChunkingConfig parameters
            code_config = CodeChunkingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                # Map extract_symbols to extract_relations (symbol relationship extraction)
                extract_relations=self.config.extract_symbols,
                include_private=True,  # Include private symbols for comprehensive analysis
                include_code_context=True,  # Include surrounding code context
            )
            self._chunker = CodeChunkingStrategy(code_config)

        return self._chunker

    def chunk(
        self, content: str, source_id: str, metadata: dict[str, Any] | None = None
    ) -> list[ProcessedChunk]:
        """Chunk code using AST-aware chunking"""
        chunker = self._get_chunker()

        # Use code chunking strategy
        text_chunks = chunker.chunk(content, source_id, metadata)

        # Convert to ProcessedChunks with code-specific preparation
        processed = []
        for chunk in text_chunks:
            processed_chunk = ProcessedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                metadata=chunk.metadata,
                embedding_text=self._prepare_code_for_embedding(chunk),
            )
            processed.append(processed_chunk)

        return processed

    def _prepare_code_for_embedding(self, chunk) -> str:
        """
        Prepare code for embedding with context.

        Combines:
        - Symbol name and type
        - Documentation/docstrings
        - Signature/parameters
        - Actual code
        """
        parts = []

        # Add symbol context
        if chunk.metadata.get("fully_qualified_name"):
            parts.append(f"Symbol: {chunk.metadata['fully_qualified_name']}")

        if chunk.metadata.get("symbol_type"):
            parts.append(f"Type: {chunk.metadata['symbol_type']}")

        # Add documentation
        if self.config.include_docstrings and chunk.metadata.get("documentation"):
            doc = chunk.metadata["documentation"]
            if len(doc) > 500:
                doc = doc[:500] + "..."
            parts.append(f"Documentation: {doc}")

        # Add signature
        if chunk.metadata.get("signature"):
            parts.append(f"Signature: {chunk.metadata['signature']}")

        # Add the actual code
        parts.append(f"Code:\n{chunk.text}")

        return "\n".join(parts)

    def prepare_for_embedding(self, chunk: ProcessedChunk) -> str:
        """Use pre-prepared embedding text"""
        return chunk.embedding_text or chunk.text

    def enrich_metadata(
        self, chunk: ProcessedChunk, source_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Add code-specific metadata"""
        metadata = super().enrich_metadata(chunk, source_metadata)

        # Ensure code-specific fields are present
        metadata.setdefault("language", chunk.metadata.get("language", "unknown"))
        metadata.setdefault("symbol_type", chunk.metadata.get("symbol_type", "unknown"))
        metadata.setdefault("is_code", True)

        return metadata


# =============================================================================
# Generic Text Document Processor
# =============================================================================


class TextDocumentProcessor(DocumentProcessor):
    """
    Processor for generic text documents.

    Uses semantic chunking for prose content.
    """

    def __init__(self, config: ProcessorConfig | None = None):
        super().__init__(config)
        self._chunker = None
        #: Per-document cost accounting from the last `chunk()` call, including
        #: the KEU multiplier enrichment applied. Kept so enabling a pass is a
        #: decision made against a measured number rather than an intuition.
        self._last_pass_stats: dict[str, Any] = {}
        self._last_chunk_edges: tuple[Any, ...] = ()

    @property
    def supported_types(self) -> list[DocumentType]:
        return [DocumentType.TEXT, DocumentType.MARKDOWN]

    @property
    def name(self) -> str:
        return "text"

    def can_process(self, content: str, file_path: str | None = None) -> bool:
        """Text processor can handle most content"""
        if file_path:
            ext = Path(file_path).suffix.lower()
            return ext in [".txt", ".md", ".rst", ".adoc", ""]
        return True  # Default fallback

    def _get_chunker(self):
        """Lazy initialization of text chunker.

        Uses RECURSIVE rather than SEMANTIC. SEMANTIC was measured losing
        between 4% and 100% of a document depending on its shape — on ordinary
        header-dense Markdown it returned ZERO chunks — and this is the only
        production caller in the SDK, so that loss reached every `.txt`, `.md`,
        `.rst` and `.adoc` ingest, including anvaiops connector ingest, which
        delegates here. RECURSIVE is clean across the whole conformance corpus
        and is the only strategy with a cap-enforcing terminal fallback.

        Revisit for `.md` specifically once SEMANTIC is rewritten and has
        soaked — as a separate, measured decision rather than a default nobody
        chose.
        """
        if self._chunker is None:
            from .chunking_strategies.base import ChunkingConfig, ChunkingStrategy
            from .chunking_strategies.recursive import RecursiveStrategy

            config = ChunkingConfig(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                # Previously dropped on the floor, so the chunker silently ran
                # on ChunkingConfig's defaults (100/2048) instead of the
                # processor's configured 50/4096.
                min_chunk_size=self.config.min_chunk_size,
                max_chunk_size=self.config.max_chunk_size,
            )
            self._chunker = RecursiveStrategy(config)

        return self._chunker

    def chunk(
        self, content: str, source_id: str, metadata: dict[str, Any] | None = None
    ) -> list[ProcessedChunk]:
        """Chunk text using semantic chunking"""
        chunker = self._get_chunker()

        text_chunks = chunker.chunk(content, source_id, metadata)

        # Imported here, alongside the chunker it complements, to keep
        # `import proximadb_sdk` free of the chunking modules.
        from .chunking_strategies.capabilities import PassPipeline
        from .chunking_strategies.passes import embedded_text_of

        # Everything that happens AFTER the partition runs through one seam.
        #
        # This used to be two `if` flags inline, each with a paragraph
        # explaining why it ran where it did. Two more capabilities were coming
        # (context enrichment, small-to-big), and four flags have twenty-four
        # orderings and no rule. `run_passes` derives the order from the FACE
        # each pass declares -- metadata, then selection, then embedded, then
        # retrieved -- and verifies afterwards that each stayed inside it.
        #
        # Wired here because this is the SDK's only production text-ingest
        # caller: every .txt/.md/.rst/.adoc arrives through it, and anvaiops's
        # connector SDK delegates here too. A capability with no product caller
        # is the exact defect the ADR-091 census documented.
        title = (metadata or {}).get("title")
        pipeline = PassPipeline.from_processor_config(self.config, source_title=title)
        result = pipeline.run(content, list(text_chunks))
        self._last_pass_stats = {
            **result.stats,
            "enrichment_tax": result.enrichment_tax,
            "span_chars": result.span_chars,
            "embedded_chars": result.embedded_chars,
        }
        # Parent/child links for a caller to materialise (ORION edges in the
        # server). Held rather than written: chunking stays free of any client
        # or transport, as its package docstring promises.
        self._last_chunk_edges = result.edges

        processed = []
        for chunk in result.chunks:
            processed_chunk = ProcessedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                metadata=chunk.metadata,
                # The one line that makes enrichment reach the model. Without
                # it the pass would write `embedded_text` and the pipeline
                # would embed `text` anyway -- built-but-never-called, which is
                # the finding this whole program keeps re-deriving.
                embedding_text=embedded_text_of(chunk),
            )
            processed.append(processed_chunk)

        return processed


# =============================================================================
# Document Processor Registry
# =============================================================================


class DocumentProcessorRegistry:
    """
    Registry for document processors.

    Manages processor registration and selection based on document type.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._processors: dict[str, DocumentProcessor] = {}
                    cls._instance._type_mapping: dict[DocumentType, str] = {}
                    cls._instance._initialized = False
        return cls._instance

    def _ensure_initialized(self):
        """Ensure default processors are registered"""
        if not self._initialized:
            self.register(CodeDocumentProcessor())
            self.register(TextDocumentProcessor())
            self._initialized = True

    def register(self, processor: DocumentProcessor) -> None:
        """Register a processor"""
        self._processors[processor.name] = processor
        for doc_type in processor.supported_types:
            self._type_mapping[doc_type] = processor.name

    def get(self, name: str) -> DocumentProcessor | None:
        """Get processor by name"""
        self._ensure_initialized()
        return self._processors.get(name)

    def get_for_type(self, doc_type: DocumentType) -> DocumentProcessor | None:
        """Get processor for document type"""
        self._ensure_initialized()
        name = self._type_mapping.get(doc_type)
        if name:
            return self._processors.get(name)
        return None

    def detect_and_get(
        self, content: str, file_path: str | None = None
    ) -> DocumentProcessor:
        """Detect document type and return appropriate processor"""
        self._ensure_initialized()

        # Try each processor
        for processor in self._processors.values():
            if processor.can_process(content, file_path):
                return processor

        # Default to text processor
        return self._processors.get("text")

    def list_processors(self) -> list[str]:
        """List registered processor names"""
        self._ensure_initialized()
        return list(self._processors.keys())


def get_processor_registry() -> DocumentProcessorRegistry:
    """Get the global processor registry"""
    return DocumentProcessorRegistry()


# =============================================================================
# Document Type Detection
# =============================================================================


def detect_document_type(content: str, file_path: str | None = None) -> DocumentType:
    """
    Detect the type of document from content and/or file path.

    Args:
        content: Document content
        file_path: Optional file path for extension-based detection

    Returns:
        Detected DocumentType
    """
    if file_path:
        ext = Path(file_path).suffix.lower()

        # Code extensions
        code_exts = {
            ".py",
            ".rs",
            ".go",
            ".java",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".cpp",
            ".cc",
            ".c",
            ".h",
            ".hpp",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".cs",
            ".fs",
            ".ex",
            ".exs",
            ".erl",
            ".hs",
            ".lua",
            ".pl",
            ".pm",
            ".r",
            ".m",
            ".mm",
        }
        if ext in code_exts:
            return DocumentType.CODE

        # Document extensions
        if ext == ".pdf":
            return DocumentType.PDF
        if ext in {".md", ".markdown"}:
            return DocumentType.MARKDOWN
        if ext in {".txt", ".text"}:
            return DocumentType.TEXT
        if ext in {".html", ".htm"}:
            return DocumentType.HTML
        if ext == ".json":
            return DocumentType.JSON
        if ext in {".xml", ".xhtml"}:
            return DocumentType.XML
        if ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
            return DocumentType.IMAGE
        if ext in {".exe", ".dll", ".so", ".dylib", ".o", ".obj"}:
            return DocumentType.BINARY

    # Content-based detection
    content_lower = content[:1000].lower() if content else ""

    # Code detection
    code_indicators = [
        "def ",
        "class ",
        "import ",
        "from ",
        "fn ",
        "struct ",
        "impl ",
        "func ",
        "package ",
        "public ",
        "private ",
        "void ",
        "const ",
        "let ",
        "var ",
        "function ",
    ]
    if any(ind in content for ind in code_indicators):
        return DocumentType.CODE

    # Markdown detection
    if content.startswith("#") or "```" in content or content.startswith("---"):
        return DocumentType.MARKDOWN

    # HTML detection
    if "<html" in content_lower or "<!doctype html" in content_lower:
        return DocumentType.HTML

    # JSON detection
    content_stripped = content.strip()
    if content_stripped.startswith("{") or content_stripped.startswith("["):
        return DocumentType.JSON

    # XML detection
    if content_stripped.startswith("<?xml") or content_stripped.startswith("<"):
        return DocumentType.XML

    return DocumentType.TEXT


# =============================================================================
# Factory Functions
# =============================================================================


def create_processor(
    processor_type: str = "auto", config: ProcessorConfig | None = None
) -> DocumentProcessor:
    """
    Create a document processor.

    Args:
        processor_type: "code", "text", or "auto" for auto-detection
        config: Optional processor configuration

    Returns:
        DocumentProcessor instance
    """
    registry = get_processor_registry()

    if processor_type == "auto":
        # Return text processor as default
        return registry.get("text") or TextDocumentProcessor(config)

    processor = registry.get(processor_type)
    if processor:
        if config:
            processor.config = config
        return processor

    raise ValueError(f"Unknown processor type: {processor_type}")


def create_embedding_adapter(
    provider: EmbeddingProvider | AsyncEmbeddingProvider | None = None,
    batch_size: int = 32,
    use_placeholder: bool = False,
    placeholder_dimension: int = 384,
) -> EmbeddingProviderAdapter:
    """
    Create an embedding provider adapter.

    Args:
        provider: Embedding provider instance
        batch_size: Batch size for embedding
        use_placeholder: Use placeholder embeddings if no provider
        placeholder_dimension: Dimension for placeholder embeddings

    Returns:
        EmbeddingProviderAdapter instance
    """
    if provider is None:
        if use_placeholder:
            provider = PlaceholderEmbeddingProvider(placeholder_dimension)
        else:
            raise ValueError("No embedding provider provided and use_placeholder=False")

    return EmbeddingProviderAdapter(provider, batch_size=batch_size)
