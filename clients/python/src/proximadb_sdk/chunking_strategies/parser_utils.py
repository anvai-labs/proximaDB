"""
Parser utilities for code chunking.

This module provides the pieces that the live chunking path actually uses:
- Parser error hierarchy (`ParserError`, `ParseError`).
- A base class for language parsers (`BaseLanguageParser`), used by the
  document parsers as a tree-sitter-with-regex-fallback skeleton.
- Lightweight parse metrics (`ParserMetrics`, `MetricsCollector`) consumed by
  the pipeline executor.
- Configuration validation (`ValidationResult`, `ConfigValidator`).

NOTE (2026-06): A second, parallel parser stack used to live here
(`CFamilyParser`/`ParserCache`/`ParserPlugin`/family base classes/decorators).
None of it was referenced by the live code-chunking path (`code.py`,
`factory.py`, `CodeChunkingStrategy`) -- `code.py` carries its own concrete
per-language parsers and lazy cache -- so the dead stack was removed.
"""

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Error Handling
# =============================================================================


class ParserError(Exception):
    """Base exception for parser errors"""

    def __init__(self, message: str, language: str = None, file_path: str = None):
        self.language = language
        self.file_path = file_path
        super().__init__(message)


class ParseError(ParserError):
    """Raised when parsing fails"""

    def __init__(self, message: str, line: int = None, column: int = None, **kwargs):
        self.line = line
        self.column = column
        super().__init__(message, **kwargs)


# =============================================================================
# Performance Metrics
# =============================================================================


@dataclass
class ParserMetrics:
    """Metrics collected during parsing"""

    language: str
    file_path: str
    parse_time_ms: float = 0.0
    symbol_count: int = 0
    relation_count: int = 0
    error_count: int = 0
    fallback_used: bool = False
    cache_hit: bool = False
    tree_sitter_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "file_path": self.file_path,
            "parse_time_ms": round(self.parse_time_ms, 2),
            "symbol_count": self.symbol_count,
            "relation_count": self.relation_count,
            "error_count": self.error_count,
            "fallback_used": self.fallback_used,
            "cache_hit": self.cache_hit,
            "tree_sitter_available": self.tree_sitter_available,
        }


class MetricsCollector:
    """Collects and aggregates parser metrics"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._metrics: list[ParserMetrics] = []
                    cls._instance._enabled = True
        return cls._instance

    def record(self, metrics: ParserMetrics):
        """Record metrics for a parse operation"""
        if self._enabled:
            self._metrics.append(metrics)

    def get_summary(self) -> dict[str, Any]:
        """Get aggregated metrics summary"""
        if not self._metrics:
            return {}

        by_language: dict[str, list[ParserMetrics]] = {}
        for m in self._metrics:
            if m.language not in by_language:
                by_language[m.language] = []
            by_language[m.language].append(m)

        summary = {}
        for lang, metrics in by_language.items():
            summary[lang] = {
                "total_parses": len(metrics),
                "avg_parse_time_ms": sum(m.parse_time_ms for m in metrics)
                / len(metrics),
                "total_symbols": sum(m.symbol_count for m in metrics),
                "total_relations": sum(m.relation_count for m in metrics),
                "error_rate": sum(1 for m in metrics if m.error_count > 0)
                / len(metrics),
                "fallback_rate": sum(1 for m in metrics if m.fallback_used)
                / len(metrics),
                "cache_hit_rate": sum(1 for m in metrics if m.cache_hit) / len(metrics),
            }
        return summary

    def clear(self):
        """Clear collected metrics"""
        self._metrics.clear()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False


def get_metrics_collector() -> MetricsCollector:
    """Get the singleton metrics collector"""
    return MetricsCollector()


# =============================================================================
# Decorators for Metrics
# =============================================================================


def with_metrics(func: Callable) -> Callable:
    """Decorator to collect parsing metrics"""

    @wraps(func)
    def wrapper(self, content: str, file_path: str, *args, **kwargs):
        collector = get_metrics_collector()
        start_time = time.perf_counter()

        metrics = ParserMetrics(
            language=getattr(self, "language", "unknown"),
            file_path=file_path,
            tree_sitter_available=getattr(self, "_parser", None) is not None,
        )

        try:
            result = func(self, content, file_path, *args, **kwargs)

            # Extract metrics from result
            if hasattr(result, "symbols"):
                metrics.symbol_count = len(result.symbols)
            if hasattr(result, "relations"):
                metrics.relation_count = len(result.relations)

            return result
        except Exception:
            metrics.error_count += 1
            raise
        finally:
            metrics.parse_time_ms = (time.perf_counter() - start_time) * 1000
            collector.record(metrics)

    return wrapper


# =============================================================================
# Parser Base Class
# =============================================================================


class BaseLanguageParser(ABC):
    """
    Base class for language parsers with common functionality.

    Provides:
    - Tree-sitter initialization with fallback
    - Metrics-friendly attributes
    - Error handling helpers
    """

    def __init__(self):
        self._parser = None
        self._language_binding = None
        self._init_tree_sitter()

    @property
    @abstractmethod
    def language(self) -> str:
        """Language identifier"""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """Supported file extensions"""
        pass

    @property
    def tree_sitter_language_name(self) -> str:
        """Tree-sitter language name (may differ from language property)"""
        return self.language

    def _init_tree_sitter(self):
        """Initialize tree-sitter parser"""
        try:
            from tree_sitter_language_pack import get_language, get_parser

            self._parser = get_parser(self.tree_sitter_language_name)
            self._language_binding = get_language(self.tree_sitter_language_name)
            logger.debug(f"Tree-sitter initialized for {self.language}")
        except (ImportError, OSError):
            logger.info(
                f"Tree-sitter not available for {self.language}, using regex fallback"
            )
            self._parser = None
            self._language_binding = None
        except Exception as e:
            logger.warning(f"Tree-sitter init failed for {self.language}: {e}")
            self._parser = None
            self._language_binding = None

    @property
    def has_tree_sitter(self) -> bool:
        """Check if tree-sitter is available"""
        return self._parser is not None

    @abstractmethod
    def parse(self, content: str, file_path: str) -> "ParsedCode":
        """Parse content and extract symbols/relations"""
        pass

    @abstractmethod
    def _fallback_regex_parse(self, content: str, file_path: str) -> "ParsedCode":
        """Fallback parsing using regex patterns"""
        pass

    def _fallback_semantic_parse(self, content: str, file_path: str) -> "ParsedCode":
        """Fallback to semantic text chunking"""
        # Import here to avoid circular dependency
        from .code import ParsedCode

        return ParsedCode(
            file_path=file_path,
            language=self.language,
            symbols=[],
            relations=[],
            imports=[],
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
        )

    def _create_empty_result(self, file_path: str) -> "ParsedCode":
        """Create empty parse result"""
        from .code import ParsedCode

        return ParsedCode(
            file_path=file_path,
            language=self.language,
            symbols=[],
            relations=[],
            imports=[],
            content_hash="",
        )

    def _compute_content_hash(self, content: str) -> str:
        """Compute hash of content for change detection"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


# =============================================================================
# Configuration Validation
# =============================================================================


@dataclass
class ValidationResult:
    """Result of configuration validation"""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ConfigValidator:
    """Validates chunking configuration"""

    @staticmethod
    def validate_chunk_size(
        chunk_size: int,
        min_chunk_size: int = 0,
        max_chunk_size: int = 100000,
        measure_name: str = "char",
    ) -> ValidationResult:
        """Validate chunk size configuration.

        The ordering errors hold in any measure. The magnitude WARNINGS below do
        not: 100 and 10 000 are character intuitions, and a 100-TOKEN chunk is
        perfectly ordinary while a 10 000-token one exceeds most model contexts.
        Emitting character advice about a token budget is worse than emitting
        none -- it trains readers to ignore the warnings -- so they are scoped
        to the measure they were calibrated for and named in the text.
        """
        result = ValidationResult(valid=True)

        if chunk_size < min_chunk_size:
            result.valid = False
            result.errors.append(
                f"chunk_size ({chunk_size}) must be >= min_chunk_size ({min_chunk_size})"
            )

        if chunk_size > max_chunk_size:
            result.valid = False
            result.errors.append(
                f"chunk_size ({chunk_size}) must be <= max_chunk_size ({max_chunk_size})"
            )

        if measure_name == "char":
            if chunk_size < 100:
                result.warnings.append(
                    f"chunk_size ({chunk_size} chars) is very small, may result "
                    "in too many chunks"
                )
            if chunk_size > 10000:
                result.warnings.append(
                    f"chunk_size ({chunk_size} chars) is large, may affect "
                    "embedding quality"
                )

        return result

    @staticmethod
    def validate_overlap(chunk_overlap: int, chunk_size: int) -> ValidationResult:
        """Validate chunk overlap configuration"""
        result = ValidationResult(valid=True)

        if chunk_overlap < 0:
            result.valid = False
            result.errors.append(f"chunk_overlap ({chunk_overlap}) must be >= 0")

        if chunk_overlap >= chunk_size:
            result.valid = False
            result.errors.append(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )

        if chunk_overlap > chunk_size * 0.5:
            result.warnings.append(
                f"chunk_overlap ({chunk_overlap}) is >50% of chunk_size, "
                "may cause high redundancy"
            )

        return result

    @staticmethod
    def validate_languages(languages: list[str]) -> ValidationResult:
        """Validate language configuration against the live parser registry"""
        result = ValidationResult(valid=True)
        # Static language knowledge, not parser capability. `python` is a real
        # language whether or not the optional `codegraph` extra is installed,
        # so validating a config against the DERIVED map made every language
        # warn on a machine without the extra.
        from .code import STATIC_EXTENSION_TO_LANGUAGE

        supported = {lang.lower() for lang in STATIC_EXTENSION_TO_LANGUAGE.values()}

        for lang in languages:
            if lang.lower() not in supported:
                result.warnings.append(f"Language '{lang}' may not be fully supported")

        return result

    @classmethod
    def validate_config(cls, config: Any) -> ValidationResult:
        """Validate complete configuration"""
        result = ValidationResult(valid=True)

        # Validate chunk size if present
        if hasattr(config, "chunk_size"):
            min_size = getattr(config, "min_chunk_size", 0)
            max_size = getattr(config, "max_chunk_size", 100000)
            # Forward the measure, or the scoping above never takes effect
            # where it matters -- this is the only in-tree caller.
            measure = getattr(config, "measure", None)
            size_result = cls.validate_chunk_size(
                config.chunk_size,
                min_size,
                max_size,
                measure_name=(
                    "char" if measure is None else str(getattr(measure, "name", "char"))
                ),
            )
            if not size_result.valid:
                result.valid = False
            result.errors.extend(size_result.errors)
            result.warnings.extend(size_result.warnings)

        # Validate overlap if present
        if hasattr(config, "chunk_overlap") and hasattr(config, "chunk_size"):
            overlap_result = cls.validate_overlap(
                config.chunk_overlap, config.chunk_size
            )
            if not overlap_result.valid:
                result.valid = False
            result.errors.extend(overlap_result.errors)
            result.warnings.extend(overlap_result.warnings)

        # Validate languages if present
        if hasattr(config, "languages") and config.languages:
            lang_result = cls.validate_languages(config.languages)
            result.warnings.extend(lang_result.warnings)

        return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Errors
    "ParserError",
    "ParseError",
    # Metrics
    "ParserMetrics",
    "MetricsCollector",
    "get_metrics_collector",
    "with_metrics",
    # Parser base class
    "BaseLanguageParser",
    # Validation
    "ValidationResult",
    "ConfigValidator",
]
