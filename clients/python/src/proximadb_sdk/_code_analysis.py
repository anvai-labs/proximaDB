"""Shared code-analysis helpers for the multi-model code providers.

Extracted from `proximadb_sdk.embedded_multi` and
`proximadb_sdk.integrations.victor_multi`, which carried AST-identical
copies of these helpers (dedup wave 4). The one collaborator
(`_matches_filter`) is inlined as :func:`matches_filter`, so every
helper here is a pure function; the provider classes keep thin
delegating methods for API compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def chunk_code(content: str, chunk_size: int = 512) -> list[dict[str, Any]]:
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


def extract_code_metrics(content: str, language: str) -> list[dict[str, Any]]:
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
    loc = sum(
        1 for line in lines if line.strip() and (not line.strip().startswith("#"))
    )
    metrics.append({"name": "lines_of_code", "value": loc, "language": language})
    func_count = sum(1 for line in lines if line.strip().startswith("def "))
    metrics.append(
        {"name": "function_count", "value": func_count, "language": language}
    )
    class_count = sum(1 for line in lines if line.strip().startswith("class "))
    metrics.append({"name": "class_count", "value": class_count, "language": language})
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


def find_code_files(
    repo_path: Path, language_map: dict[str, str] | None = None
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


def filter_hybrid_results(
    results: list[dict[str, Any]], filter_dict: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Filter hybrid search results."""
    if not filter_dict:
        return results
    filtered = []
    for result in results:
        metadata = result.get("metadata", {})
        if matches_filter(metadata, filter_dict):
            filtered.append(result)
    return filtered


def matches_filter(metadata: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
    """Check if metadata matches filter criteria."""
    for key, value in filter_dict.items():
        if key not in metadata:
            return False
        if metadata[key] != value:
            return False
    return True
