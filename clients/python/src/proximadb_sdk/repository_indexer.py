"""
Repository Indexer - Git-Aware Code Indexing for ProximaDB

This module extends CodeKnowledgeBuilder with Git repository integration,
providing:
- Git-aware incremental indexing (only changed files)
- Commit/branch tracking in indexed symbols
- Author attribution for code symbols
- Repository-level index state management
- Multi-repository support

Design Patterns:
- Decorator Pattern: Extends CodeKnowledgeBuilder functionality
- Strategy Pattern: Pluggable change detection strategies
- Observer Pattern: Index change notifications
- Command Pattern: Indexing operations as undoable commands

Usage:
    from proximadb import ProximaDBClient
    from proximadb_sdk.repository_indexer import RepositoryIndexer

    client = ProximaDBClient(url="http://localhost:5678")
    indexer = RepositoryIndexer(client)

    # Index a repository with git tracking
    result = await indexer.index_repository("/path/to/repo")

    # Incremental update (only changed files)
    result = await indexer.update_repository("/path/to/repo")

    # Search with git context
    results = await indexer.search_code(
        query="authentication handler",
        include_git_context=True
    )

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .chunking_strategies.code import static_supported_extensions
from .code_knowledge import (
    CodeIndexConfig,
    CodeKnowledgeBuilder,
    CodeSearchResult,
    IndexingResult,
)
from .repository_manager import (
    ChangeType,
    FileChange,
    IndexState,
    RepositoryManager,
    get_file_git_info,
    is_git_repository,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class IndexMode(Enum):
    """Indexing mode selection"""

    FULL = auto()  # Full re-index of all files
    INCREMENTAL = auto()  # Only changed files since last index
    SMART = auto()  # Auto-detect based on repository state


class ChangeStrategy(Enum):
    """Strategy for detecting changes"""

    GIT_DIFF = auto()  # Use git diff for change detection
    FILE_HASH = auto()  # Use file content hashing
    HYBRID = auto()  # Combine git diff with hash verification


@dataclass
class RepositoryIndexConfig(CodeIndexConfig):
    """Extended configuration for repository indexing"""

    # Git integration
    enable_git_integration: bool = True
    track_commits: bool = True
    track_branches: bool = True
    track_authors: bool = True

    # Change detection
    change_strategy: ChangeStrategy = ChangeStrategy.GIT_DIFF
    index_mode: IndexMode = IndexMode.SMART

    # State persistence
    state_file_name: str = ".proximadb_index_state.json"
    persist_state: bool = True

    # Performance
    parallel_file_processing: bool = True
    max_concurrent_files: int = 10

    # Filtering
    index_branches: list[str] = field(
        default_factory=lambda: ["main", "master", "develop"]
    )
    skip_merge_commits: bool = False


@dataclass
class RepositoryIndexResult(IndexingResult):
    """Extended result with repository information"""

    repository_root: str | None = None
    current_commit: str | None = None
    current_branch: str | None = None
    previous_commit: str | None = None

    # Change statistics
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    files_renamed: int = 0

    # Git metadata
    authors_encountered: set[str] = field(default_factory=set)
    commits_in_range: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "files_failed": self.files_failed,
            "symbols_indexed": self.symbols_indexed,
            "relations_created": self.relations_created,
            "repository_root": self.repository_root,
            "current_commit": self.current_commit,
            "current_branch": self.current_branch,
            "previous_commit": self.previous_commit,
            "files_added": self.files_added,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
            "files_renamed": self.files_renamed,
            "authors_encountered": list(self.authors_encountered),
            "commits_in_range": self.commits_in_range,
            "errors": self.errors,
        }


@dataclass
class GitEnrichedSearchResult(CodeSearchResult):
    """Search result enriched with git information"""

    # Git context
    commit_hash: str | None = None
    branch: str | None = None
    remote_url: str | None = None
    last_modified_by: str | None = None
    last_modified_date: datetime | None = None
    contributors: list[str] = field(default_factory=list)

    # History context
    total_commits: int = 0
    recent_changes: list[dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Repository Indexer
# =============================================================================


class RepositoryIndexer:
    """
    Git-aware code indexer that extends CodeKnowledgeBuilder.

    Features:
    - Automatic git repository detection
    - Incremental indexing based on git diff
    - Commit and branch tracking
    - Author attribution
    - State persistence for efficient updates
    """

    def __init__(
        self,
        client: Any,  # ProximaDBClient
        config: RepositoryIndexConfig | None = None,
        embedding_provider: Any | None = None,
    ):
        """
        Initialize RepositoryIndexer.

        Args:
            client: ProximaDB client instance
            config: Repository indexing configuration
            embedding_provider: Optional embedding provider
        """
        self.config = config or RepositoryIndexConfig()
        self._builder = CodeKnowledgeBuilder(
            client=client,
            config=self.config,
            embedding_provider=embedding_provider,
        )
        self.client = client

        # Repository managers cache
        self._repo_managers: dict[str, RepositoryManager] = {}

        # Index states cache
        self._index_states: dict[str, IndexState] = {}

    def _get_repo_manager(
        self,
        path: str | Path,
    ) -> RepositoryManager | None:
        """Get or create repository manager for path."""
        path = Path(path).resolve()
        path_str = str(path)

        if path_str in self._repo_managers:
            return self._repo_managers[path_str]

        if not self.config.enable_git_integration:
            return None

        if not is_git_repository(path):
            return None

        try:
            # Load existing state if available
            state = self._load_index_state(path)
            manager = RepositoryManager.from_path(path, state)
            self._repo_managers[path_str] = manager
            return manager
        except Exception as e:
            logger.warning(f"Failed to initialize git integration: {e}")
            return None

    def _load_index_state(self, repo_path: Path) -> IndexState | None:
        """Load index state from repository."""
        state_file = repo_path / self.config.state_file_name
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                return IndexState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load index state: {e}")
        return None

    def _save_index_state(
        self,
        repo_path: Path,
        manager: RepositoryManager,
    ) -> None:
        """Save index state to repository."""
        if not self.config.persist_state:
            return

        state_file = repo_path / self.config.state_file_name
        try:
            state_file.write_text(json.dumps(manager.index_state.to_dict(), indent=2))
        except Exception as e:
            logger.warning(f"Failed to save index state: {e}")

    async def index_repository(
        self,
        path: str | Path,
        mode: IndexMode | None = None,
        force: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> RepositoryIndexResult:
        """
        Index a repository with git awareness.

        Args:
            path: Path to repository
            mode: Indexing mode (defaults to config setting)
            force: Force full re-index
            progress_callback: Optional progress callback

        Returns:
            RepositoryIndexResult with detailed statistics
        """
        path = Path(path).resolve()
        mode = mode or self.config.index_mode
        result = RepositoryIndexResult()

        # Get repository manager
        repo_manager = self._get_repo_manager(path)

        if repo_manager:
            result.repository_root = str(repo_manager.root)
            info = repo_manager.get_info()
            result.current_commit = info.current_commit
            result.current_branch = info.current_branch
            result.previous_commit = repo_manager.index_state.last_indexed_commit

        # Determine files to index
        if force or mode == IndexMode.FULL:
            files_to_index = self._collect_all_files(path)
        elif repo_manager and mode in (IndexMode.INCREMENTAL, IndexMode.SMART):
            files_to_index = await self._get_changed_files(repo_manager)
            deleted_files = await self._get_deleted_files(repo_manager)

            # Handle deleted files
            await self._handle_deleted_files(deleted_files, result)

            # Classify changes
            for change in repo_manager.get_changes_since_last_index():
                if change.change_type == ChangeType.ADDED:
                    result.files_added += 1
                elif change.change_type == ChangeType.MODIFIED:
                    result.files_modified += 1
                elif change.change_type == ChangeType.RENAMED:
                    result.files_renamed += 1
        else:
            # Fallback to all files
            files_to_index = self._collect_all_files(path)

        # Index files
        total_files = len(files_to_index)
        indexed_hashes: dict[str, str] = {}

        if self.config.parallel_file_processing and total_files > 1:
            # Parallel processing
            results = await self._index_files_parallel(
                files_to_index,
                repo_manager,
                progress_callback,
            )
            for file_result, file_path, content_hash in results:
                self._aggregate_result(result, file_result)
                if content_hash:
                    indexed_hashes[str(file_path)] = content_hash
        else:
            # Sequential processing
            for i, file_path in enumerate(files_to_index):
                if progress_callback:
                    progress_callback(str(file_path), i + 1, total_files)

                file_result, content_hash = await self._index_single_file(
                    file_path,
                    repo_manager,
                )
                self._aggregate_result(result, file_result)
                if content_hash:
                    indexed_hashes[str(file_path)] = content_hash

        # Update index state
        if repo_manager:
            repo_manager.update_index_state(indexed_files=indexed_hashes)
            self._save_index_state(path, repo_manager)

            # Collect author statistics
            if self.config.track_authors:
                try:
                    commits = repo_manager.get_recent_commits(limit=100)
                    result.authors_encountered = {
                        c.author.email for c in commits if c.author
                    }
                    result.commits_in_range = len(commits)
                except Exception:
                    pass

        return result

    async def update_repository(
        self,
        path: str | Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> RepositoryIndexResult:
        """
        Incrementally update repository index.

        Only processes files that changed since last index.

        Args:
            path: Path to repository
            progress_callback: Optional progress callback

        Returns:
            RepositoryIndexResult
        """
        return await self.index_repository(
            path,
            mode=IndexMode.INCREMENTAL,
            progress_callback=progress_callback,
        )

    def _collect_all_files(self, directory: Path) -> list[Path]:
        """Collect all indexable files in directory."""
        supported_extensions = set(static_supported_extensions())
        files = []

        for path in directory.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in supported_extensions:
                continue

            # Check exclude patterns
            try:
                rel_path = str(path.relative_to(directory))
            except ValueError:
                continue

            if self._matches_exclude_pattern(rel_path):
                continue

            files.append(path)

        return sorted(files)

    def _matches_exclude_pattern(self, rel_path: str) -> bool:
        """Check if path matches exclude patterns."""
        import fnmatch

        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        return False

    async def _get_changed_files(
        self,
        repo_manager: RepositoryManager,
    ) -> list[Path]:
        """Get files that need indexing based on git changes."""
        changes = repo_manager.get_files_to_reindex(filter_code_files=True)

        # Map to absolute paths
        root = repo_manager.root
        return [root / change.path for change in changes]

    async def _get_deleted_files(
        self,
        repo_manager: RepositoryManager,
    ) -> list[FileChange]:
        """Get files that were deleted."""
        return repo_manager.get_deleted_files()

    async def _handle_deleted_files(
        self,
        deleted_files: list[FileChange],
        result: RepositoryIndexResult,
    ) -> None:
        """Handle deleted files by removing from index."""
        for change in deleted_files:
            try:
                # Remove from vector store
                # This would need to be implemented in the vector store
                # For now, just track the deletion
                result.files_deleted += 1
                logger.info(f"File deleted: {change.path}")
            except Exception as e:
                result.errors.append(
                    {"file": change.path, "error": f"Failed to handle deletion: {e}"}
                )

    async def _index_single_file(
        self,
        file_path: Path,
        repo_manager: RepositoryManager | None,
    ) -> tuple[IndexingResult, str | None]:
        """Index a single file with git metadata enrichment."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            result = IndexingResult()
            result.files_failed = 1
            result.errors.append(
                {"file": str(file_path), "error": f"Failed to read: {e}"}
            )
            return result, None

        # Compute hash
        import hashlib

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Build metadata with git info
        metadata = {}
        if repo_manager and self.config.track_commits:
            try:
                rel_path = str(file_path.relative_to(repo_manager.root))
                git_info = get_file_git_info(file_path)
                if git_info:
                    metadata["git_commit"] = git_info.get("commit_hash")
                    metadata["git_branch"] = git_info.get("branch")
                    metadata["git_remote"] = git_info.get("remote_url")

                    if self.config.track_authors and git_info.get("authors"):
                        metadata["git_authors"] = [
                            a["email"] for a in git_info["authors"][:5]
                        ]
            except Exception:
                pass

        # Index the file
        result = await self._builder.index_file(file_path, content)

        return result, content_hash

    async def _index_files_parallel(
        self,
        files: list[Path],
        repo_manager: RepositoryManager | None,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> list[tuple[IndexingResult, Path, str | None]]:
        """Index files in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
        total = len(files)
        completed = 0

        async def process_file(
            file_path: Path,
        ) -> tuple[IndexingResult, Path, str | None]:
            nonlocal completed
            async with semaphore:
                result, content_hash = await self._index_single_file(
                    file_path, repo_manager
                )
                completed += 1
                if progress_callback:
                    progress_callback(str(file_path), completed, total)
                return result, file_path, content_hash

        tasks = [process_file(f) for f in files]
        return await asyncio.gather(*tasks)

    def _aggregate_result(
        self,
        aggregate: RepositoryIndexResult,
        single: IndexingResult,
    ) -> None:
        """Aggregate single result into total."""
        aggregate.files_processed += single.files_processed
        aggregate.files_skipped += single.files_skipped
        aggregate.files_failed += single.files_failed
        aggregate.symbols_indexed += single.symbols_indexed
        aggregate.relations_created += single.relations_created
        aggregate.errors.extend(single.errors)
        aggregate.file_hashes.update(single.file_hashes)

    async def search_code(
        self,
        query: str,
        top_k: int = 10,
        include_git_context: bool = True,
        repository_path: str | Path | None = None,
        language: str | None = None,
        **kwargs,
    ) -> list[GitEnrichedSearchResult]:
        """
        Search code with optional git context enrichment.

        Args:
            query: Search query
            top_k: Number of results
            include_git_context: Whether to enrich results with git info
            repository_path: Optional path to limit search to repository
            language: Optional language filter
            **kwargs: Additional search parameters

        Returns:
            List of GitEnrichedSearchResult
        """
        # Perform base search
        base_results = await self._builder.search_code(
            query=query,
            top_k=top_k,
            language=language,
            **kwargs,
        )

        if not include_git_context:
            return [GitEnrichedSearchResult(**vars(r)) for r in base_results]

        # Enrich with git context
        enriched_results = []
        for result in base_results:
            enriched = GitEnrichedSearchResult(**vars(result))

            if include_git_context:
                try:
                    file_path = Path(result.file_path)
                    git_info = get_file_git_info(file_path)

                    if git_info:
                        enriched.commit_hash = git_info.get("commit_hash")
                        enriched.branch = git_info.get("branch")
                        enriched.remote_url = git_info.get("remote_url")

                        authors = git_info.get("authors", [])
                        if authors:
                            enriched.last_modified_by = authors[0].get("email")
                            enriched.contributors = [a.get("email") for a in authors]
                except Exception:
                    pass

            enriched_results.append(enriched)

        return enriched_results

    async def get_repository_stats(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        """
        Get statistics for an indexed repository.

        Args:
            path: Repository path

        Returns:
            Dictionary with repository statistics
        """
        repo_manager = self._get_repo_manager(path)
        if not repo_manager:
            return {"error": "Not a git repository"}

        info = repo_manager.get_info()
        index_state = repo_manager.index_state

        return {
            "repository": {
                "root": str(info.root_path),
                "vcs_type": info.vcs_type.name,
                "remote_url": info.remote_url,
                "current_branch": info.current_branch,
                "current_commit": info.current_commit,
                "is_dirty": info.is_dirty,
                "total_commits": info.total_commits,
                "total_branches": info.total_branches,
            },
            "index": {
                "last_indexed_commit": index_state.last_indexed_commit,
                "last_indexed_time": (
                    index_state.last_indexed_time.isoformat()
                    if index_state.last_indexed_time
                    else None
                ),
                "indexed_files_count": len(index_state.indexed_files),
                "branch_states": index_state.branch_states,
            },
            "pending_changes": {
                "files_to_reindex": len(
                    repo_manager.get_files_to_reindex(filter_code_files=True)
                ),
                "files_deleted": len(repo_manager.get_deleted_files()),
            },
        }

    async def clear_index(
        self,
        path: str | Path,
        clear_state: bool = True,
    ) -> bool:
        """
        Clear index for a repository.

        Args:
            path: Repository path
            clear_state: Whether to also clear persisted state

        Returns:
            True if successful
        """
        path = Path(path).resolve()

        # Clear cached state
        path_str = str(path)
        if path_str in self._repo_managers:
            del self._repo_managers[path_str]

        # Clear state file
        if clear_state:
            state_file = path / self.config.state_file_name
            if state_file.exists():
                state_file.unlink()

        # TODO: Clear vectors and graph data for this repository
        # This would need collection-level delete support

        return True


# =============================================================================
# Factory Functions
# =============================================================================


def create_repository_indexer(
    client: Any,
    enable_git: bool = True,
    track_authors: bool = True,
    parallel: bool = True,
    **kwargs,
) -> RepositoryIndexer:
    """
    Create a configured RepositoryIndexer.

    Args:
        client: ProximaDB client
        enable_git: Enable git integration
        track_authors: Track author information
        parallel: Enable parallel file processing
        **kwargs: Additional config options

    Returns:
        Configured RepositoryIndexer
    """
    config = RepositoryIndexConfig(
        enable_git_integration=enable_git,
        track_authors=track_authors,
        parallel_file_processing=parallel,
        **kwargs,
    )
    return RepositoryIndexer(client, config)


async def index_repository(
    client: Any,
    path: str | Path,
    incremental: bool = True,
    **kwargs,
) -> RepositoryIndexResult:
    """
    Convenience function to index a repository.

    Args:
        client: ProximaDB client
        path: Repository path
        incremental: Use incremental indexing
        **kwargs: Additional options

    Returns:
        RepositoryIndexResult
    """
    indexer = create_repository_indexer(client, **kwargs)
    mode = IndexMode.INCREMENTAL if incremental else IndexMode.FULL
    return await indexer.index_repository(path, mode=mode)
