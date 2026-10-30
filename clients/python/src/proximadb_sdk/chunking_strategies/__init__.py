"""
Pluggable chunking strategies for ProximaDB Python SDK

This package provides a clean interface for text chunking with proper separation of concerns:
- Each chunking strategy in its own module
- No embedding logic mixed with chunking
- Metadata generation is purely about chunk properties
- Extensible interface for custom strategies

For code-aware chunking, use the CodeChunkingStrategy which provides:
- AST-based symbol and relation extraction, delegated to the shared
  `victor-codegraph` package (ADR-029). Language support is not asserted here
  but measured, and the three states are named separately so none can hide in
  another: `code.SYMBOL_EXTRACTING_LANGUAGES` (16) genuinely yields symbols;
  `code.COVERED_WITHOUT_SYMBOLS` (2) is detected and chunked but symbol-less;
  `code.WITHDRAWN_LANGUAGES` (6) is no longer advertised at all, yet still
  chunks through the text fallback rather than being dropped.
- Byte-basis offsets, declared via `offset_basis` rather than inferred.
- `max_chunk_size` honoured, so an oversized symbol splits instead of being
  truncated or rejected by the embedding provider.
- Requires the `codegraph` extra; without it, code chunking fails loudly naming
  it rather than degrading to unlabelled text windows.
"""

from .base import (
    ChunkingConfig,
    ChunkingStrategy,
    ChunkingStrategyInterface,
    TextChunk,
    config_kwargs,
)
from .boundaries import (
    Boundary,
    BoundaryKind,
    BoundarySource,
    CompositeBoundarySource,
    HeadingBoundarySource,
    StrategyBoundarySource,
    annotate_heading_paths,
    merge_boundaries,
)

# Code-aware chunking
from .capabilities import (
    ContextEnrichmentPass,
    DedupPass,
    HeadingPathPass,
    ParentLinkagePass,
    PassPipeline,
    structural_context,
)
from .code import (
    COVERED_WITHOUT_SYMBOLS,
    EXTENSION_TO_LANGUAGE,
    SYMBOL_EXTRACTING_LANGUAGES,
    WITHDRAWN_LANGUAGES,
    CodeChunkingConfig,
    CodeChunkingStrategy,
    CodeRelation,
    CodeRelationType,
    CodeSymbol,
    CodeSymbolType,
    ParsedCode,
    SourceLocation,
    create_code_chunker,
    get_supported_extensions,
    get_supported_languages,
)
from .contracts import (
    ChunkContextRenderer,
    CompositeInputContract,
    InputRenderer,
    InputRole,
    OverflowPolicy,
    ResolvedInputContract,
    ShortChunkPolicy,
    TokenBudget,
    TokenCounter,
)
from .dedup import DedupResult, deduplicate, jaccard, shingles

# Document and binary parsers (OCR, reverse engineering)
from .document_parsers import (  # Enums; Data structures; Tool detection; Parsers; Factory functions
    BinaryAnalysis,
    BinaryParser,
    BinaryParserConfig,
    BinarySymbol,
    BinaryType,
    DocumentParser,
    DocumentType,
    OCRConfig,
    OCRResult,
    ToolDetector,
    create_binary_parser,
    create_document_parser,
    get_available_tools,
)
from .factory import ChunkingStrategyFactory, get_chunking_strategy
from .native_boundaries import NativeSentenceBoundarySource, native_sentences_available
from .paragraph import ParagraphStrategy

# Parser utilities (enhanced design patterns)
from .parser_utils import (  # Errors; Metrics; Parser base class; Validation
    BaseLanguageParser,
    ConfigValidator,
    MetricsCollector,
    ParseError,
    ParserError,
    ParserMetrics,
    ValidationResult,
    get_metrics_collector,
    with_metrics,
)
from .passes import (
    FACE_ORDER,
    ChunkEdge,
    ChunkPass,
    Face,
    PassPipelineResult,
    PassResult,
    embedded_text_of,
    run_passes,
)

# Unified pipeline (orchestration, batch processing, streaming)
from .pipeline import (  # Configuration; Pipeline stages; Core components; Factory functions; Context managers
    BatchEmbedder,
    BatchResult,
    ChunkingPipeline,
    EnrichmentStage,
    ErrorHandling,
    FilterStage,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    ProcessingMode,
    ProgressTracker,
    ValidationStage,
    async_pipeline_context,
    create_code_pipeline,
    create_document_pipeline,
    create_pipeline,
    pipeline_context,
)
from .recursive import RecursiveStrategy
from .semantic import SemanticStrategy
from .semantic_embedding import SemanticEmbeddingStrategy
from .sentence import SentenceStrategy
from .sizing import Absolute, Fraction, Of, ResolvedSizing, SizingPolicy
from .sliding_window import SlidingWindowStrategy
from .structure import Heading, HeadingOutline, find_headings, protected_spans
from .token_budget import TokenBudgetStrategy
from .tokenizers import HuggingFaceTokenCounter

__all__ = [
    # Base classes
    "ChunkingStrategy",
    "ChunkingStrategyInterface",
    "TextChunk",
    "ChunkingConfig",
    "TokenBudget",
    "TokenCounter",
    "InputRole",
    "InputRenderer",
    "OverflowPolicy",
    "ShortChunkPolicy",
    "ResolvedInputContract",
    "CompositeInputContract",
    "TokenBudgetStrategy",
    "HuggingFaceTokenCounter",
    # Text chunking strategies
    "SlidingWindowStrategy",
    "SentenceStrategy",
    # Deduplication (TD-CHUNK-3 item 2)
    "deduplicate",
    "DedupResult",
    "shingles",
    "jaccard",
    # Boundary sources (ADR-091 D2)
    "Boundary",
    "BoundaryKind",
    "BoundarySource",
    "StrategyBoundarySource",
    "CompositeBoundarySource",
    "HeadingBoundarySource",
    "annotate_heading_paths",
    "merge_boundaries",
    # Document structure (shared detection)
    "Heading",
    "HeadingOutline",
    "find_headings",
    "protected_spans",
    # Sizing (declarative budget)
    "SizingPolicy",
    "ResolvedSizing",
    "Absolute",
    "Fraction",
    "Of",
    "ParagraphStrategy",
    "SemanticStrategy",
    "SemanticEmbeddingStrategy",
    "RecursiveStrategy",
    # Factory
    "ChunkingStrategyFactory",
    "get_chunking_strategy",
    # Code-aware chunking
    "CodeChunkingStrategy",
    "CodeChunkingConfig",
    "CodeSymbol",
    "CodeSymbolType",
    "CodeRelation",
    "CodeRelationType",
    "ParsedCode",
    "SourceLocation",
    # Language parsers - Primary
    # Language parsers - Additional
    # Plugin functions
    "get_supported_languages",
    "get_supported_extensions",
    "create_code_chunker",
    # Registry constants
    "EXTENSION_TO_LANGUAGE",
    "SYMBOL_EXTRACTING_LANGUAGES",
    "COVERED_WITHOUT_SYMBOLS",
    "WITHDRAWN_LANGUAGES",
    # Post-partition capability seam (TD-CHUNK-3)
    "Face",
    "FACE_ORDER",
    "ChunkPass",
    "ChunkEdge",
    "PassResult",
    "PassPipelineResult",
    "run_passes",
    "embedded_text_of",
    "PassPipeline",
    "HeadingPathPass",
    "DedupPass",
    "ContextEnrichmentPass",
    "ParentLinkagePass",
    "structural_context",
    "NativeSentenceBoundarySource",
    "native_sentences_available",
    # Parser utilities - Errors
    "ParserError",
    "ParseError",
    # Parser utilities - Metrics
    "ParserMetrics",
    "MetricsCollector",
    "get_metrics_collector",
    "with_metrics",
    # Parser utilities - Base class
    "BaseLanguageParser",
    # Parser utilities - Validation
    "ValidationResult",
    "ConfigValidator",
    # Document/Binary parsers
    "BinaryType",
    "DocumentType",
    "BinarySymbol",
    "BinaryAnalysis",
    "OCRResult",
    "BinaryParserConfig",
    "OCRConfig",
    "ToolDetector",
    "BinaryParser",
    "DocumentParser",
    "create_binary_parser",
    "create_document_parser",
    "get_available_tools",
    # Pipeline - Configuration
    "ProcessingMode",
    "ErrorHandling",
    "PipelineConfig",
    "PipelineResult",
    "BatchResult",
    # Pipeline - Stages
    "PipelineStage",
    "ValidationStage",
    "EnrichmentStage",
    "FilterStage",
    # Pipeline - Core
    "BatchEmbedder",
    "ProgressTracker",
    "ChunkingPipeline",
    # Pipeline - Factory functions
    "create_pipeline",
    "create_code_pipeline",
    "create_document_pipeline",
    # Pipeline - Context managers
    "pipeline_context",
    "async_pipeline_context",
]
