"""
Repository Manager - Git Integration for Code Knowledge Store

This module provides comprehensive Git repository integration for:
- Repository detection and initialization
- Commit/branch/tag tracking
- Incremental change detection via git diff
- Author and blame information
- Multi-repository support

Design Patterns Used:
- Strategy Pattern: VCS backends (Git, future: Mercurial, SVN)
- Repository Pattern: Encapsulates VCS operations
- Factory Pattern: Creates appropriate VCS handlers
- Observer Pattern: Change notifications

Usage:
    from proximadb_sdk.repository_manager import RepositoryManager, GitRepository

    # Auto-detect repository
    repo = RepositoryManager.from_path("/path/to/repo")

    # Get changed files since last index
    changes = repo.get_changes_since(last_commit_hash)

    # Get file history
    history = repo.get_file_history("src/main.py")

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import hashlib
import json
import logging
import re
import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================


class VCSType(Enum):
    """Supported version control systems"""

    GIT = auto()
    MERCURIAL = auto()
    SVN = auto()
    NONE = auto()  # Not a VCS repository


class ChangeType(Enum):
    """Types of file changes"""

    ADDED = "A"
    MODIFIED = "M"
    DELETED = "D"
    RENAMED = "R"
    COPIED = "C"
    UNTRACKED = "?"
    TYPE_CHANGED = "T"


class BranchType(Enum):
    """Branch classification"""

    MAIN = auto()  # main, master
    DEVELOP = auto()  # develop, dev
    FEATURE = auto()  # feature/*
    RELEASE = auto()  # release/*
    HOTFIX = auto()  # hotfix/*
    OTHER = auto()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Author:
    """Git author information"""

    name: str
    email: str

    def __hash__(self):
        return hash((self.name, self.email))

    def __eq__(self, other):
        if not isinstance(other, Author):
            return False
        return self.name == other.name and self.email == other.email


@dataclass
class Commit:
    """Git commit information"""

    hash: str
    short_hash: str
    author: Author
    committer: Author | None
    timestamp: datetime
    message: str
    parent_hashes: list[str] = field(default_factory=list)

    # Optional detailed info (populated on demand)
    files_changed: int | None = None
    insertions: int | None = None
    deletions: int | None = None

    @property
    def is_merge(self) -> bool:
        """Check if this is a merge commit"""
        return len(self.parent_hashes) > 1


@dataclass
class Branch:
    """Git branch information"""

    name: str
    commit_hash: str
    is_current: bool = False
    is_remote: bool = False
    upstream: str | None = None
    branch_type: BranchType = BranchType.OTHER

    @classmethod
    def classify(cls, name: str) -> BranchType:
        """Classify branch by name"""
        name_lower = name.lower()
        if name_lower in ("main", "master"):
            return BranchType.MAIN
        elif name_lower in ("develop", "dev", "development"):
            return BranchType.DEVELOP
        elif name_lower.startswith("feature/") or name_lower.startswith("feat/"):
            return BranchType.FEATURE
        elif name_lower.startswith("release/") or name_lower.startswith("rel/"):
            return BranchType.RELEASE
        elif name_lower.startswith("hotfix/") or name_lower.startswith("fix/"):
            return BranchType.HOTFIX
        return BranchType.OTHER


@dataclass
class Tag:
    """Git tag information"""

    name: str
    commit_hash: str
    is_annotated: bool = False
    tagger: Author | None = None
    message: str | None = None
    timestamp: datetime | None = None


@dataclass
class FileChange:
    """Represents a file change between commits"""

    path: str
    change_type: ChangeType
    old_path: str | None = None  # For renames/copies
    additions: int = 0
    deletions: int = 0

    # Blame information (populated on demand)
    authors: list[Author] = field(default_factory=list)

    @property
    def is_code_file(self) -> bool:
        """Check if this is a code file based on extension"""
        code_extensions = {
            ".py",
            ".rs",
            ".go",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".kt",
            ".scala",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".rb",
            ".php",
            ".swift",
            ".m",
            ".mm",
            ".sh",
            ".bash",
            ".zsh",
            ".ps1",
            ".sql",
            ".lua",
            ".pl",
            ".pm",
            ".r",
            ".hs",
            ".ex",
            ".exs",
            ".erl",
            ".clj",
            ".lisp",
        }
        return Path(self.path).suffix.lower() in code_extensions


@dataclass
class DiffHunk:
    """A hunk from a unified diff"""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    content: str
    header: str = ""


@dataclass
class FileDiff:
    """Detailed diff for a single file"""

    path: str
    old_path: str | None
    change_type: ChangeType
    hunks: list[DiffHunk] = field(default_factory=list)
    is_binary: bool = False

    @property
    def total_additions(self) -> int:
        return sum(
            line.startswith("+") and not line.startswith("+++")
            for hunk in self.hunks
            for line in hunk.content.split("\n")
        )

    @property
    def total_deletions(self) -> int:
        return sum(
            line.startswith("-") and not line.startswith("---")
            for hunk in self.hunks
            for line in hunk.content.split("\n")
        )


@dataclass
class BlameEntry:
    """Blame information for a line range"""

    commit_hash: str
    author: Author
    timestamp: datetime
    line_start: int
    line_end: int
    content: str


@dataclass
class RepositoryInfo:
    """Repository metadata"""

    root_path: Path
    vcs_type: VCSType
    remote_url: str | None = None
    current_branch: str | None = None
    current_commit: str | None = None
    is_dirty: bool = False

    # Repository stats
    total_commits: int | None = None
    total_branches: int | None = None
    total_tags: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "root_path": str(self.root_path),
            "vcs_type": self.vcs_type.name,
            "remote_url": self.remote_url,
            "current_branch": self.current_branch,
            "current_commit": self.current_commit,
            "is_dirty": self.is_dirty,
            "total_commits": self.total_commits,
            "total_branches": self.total_branches,
            "total_tags": self.total_tags,
        }


@dataclass
class IndexState:
    """Tracks the state of the index for a repository"""

    repository_id: str
    last_indexed_commit: str | None = None
    last_indexed_time: datetime | None = None
    indexed_files: dict[str, str] = field(default_factory=dict)  # path -> hash
    branch_states: dict[str, str] = field(default_factory=dict)  # branch -> commit

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "last_indexed_commit": self.last_indexed_commit,
            "last_indexed_time": (
                self.last_indexed_time.isoformat() if self.last_indexed_time else None
            ),
            "indexed_files": self.indexed_files,
            "branch_states": self.branch_states,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexState":
        return cls(
            repository_id=data["repository_id"],
            last_indexed_commit=data.get("last_indexed_commit"),
            last_indexed_time=(
                datetime.fromisoformat(data["last_indexed_time"])
                if data.get("last_indexed_time")
                else None
            ),
            indexed_files=data.get("indexed_files", {}),
            branch_states=data.get("branch_states", {}),
        )


# =============================================================================
# Abstract VCS Interface (Strategy Pattern)
# =============================================================================


class VCSBackend(ABC):
    """Abstract interface for version control systems"""

    @property
    @abstractmethod
    def vcs_type(self) -> VCSType:
        """Get VCS type"""
        pass

    @abstractmethod
    def get_root(self) -> Path:
        """Get repository root path"""
        pass

    @abstractmethod
    def get_current_commit(self) -> Commit | None:
        """Get current HEAD commit"""
        pass

    @abstractmethod
    def get_current_branch(self) -> str | None:
        """Get current branch name"""
        pass

    @abstractmethod
    def get_branches(self, include_remote: bool = False) -> list[Branch]:
        """Get list of branches"""
        pass

    @abstractmethod
    def get_tags(self) -> list[Tag]:
        """Get list of tags"""
        pass

    @abstractmethod
    def get_commit(self, ref: str) -> Commit | None:
        """Get commit by reference (hash, branch, tag)"""
        pass

    @abstractmethod
    def get_commits(
        self,
        since: str | None = None,
        until: str | None = None,
        path: str | None = None,
        limit: int | None = None,
    ) -> list[Commit]:
        """Get list of commits with optional filtering"""
        pass

    @abstractmethod
    def get_changed_files(
        self,
        from_ref: str | None = None,
        to_ref: str = "HEAD",
        include_untracked: bool = True,
    ) -> list[FileChange]:
        """Get list of changed files between refs"""
        pass

    @abstractmethod
    def get_file_diff(
        self,
        path: str,
        from_ref: str | None = None,
        to_ref: str = "HEAD",
    ) -> FileDiff | None:
        """Get detailed diff for a file"""
        pass

    @abstractmethod
    def get_file_content(
        self,
        path: str,
        ref: str = "HEAD",
    ) -> str | None:
        """Get file content at specific ref"""
        pass

    @abstractmethod
    def get_blame(
        self,
        path: str,
        ref: str = "HEAD",
    ) -> list[BlameEntry]:
        """Get blame information for a file"""
        pass

    @abstractmethod
    def is_dirty(self) -> bool:
        """Check if working directory has uncommitted changes"""
        pass

    @abstractmethod
    def get_remote_url(self) -> str | None:
        """Get primary remote URL"""
        pass


# =============================================================================
# Git Implementation
# =============================================================================


class GitRepository(VCSBackend):
    """Git repository implementation"""

    def __init__(self, path: str | Path):
        self._path = Path(path).resolve()
        self._root = self._find_root()
        if not self._root:
            raise ValueError(f"Not a git repository: {path}")

        # Cache for expensive operations
        self._commit_cache: dict[str, Commit] = {}

    def _find_root(self) -> Path | None:
        """Find git repository root"""
        current = self._path
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
        return None

    def _run_git(
        self,
        *args: str,
        capture_output: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run git command"""
        cmd = ["git", "-C", str(self._root)] + list(args)
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=check,
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(cmd)}\n{e.stderr}")
            raise

    def _parse_commit(self, output: str) -> Commit:
        """Parse git log --format output into Commit"""
        lines = output.strip().split("\n")
        if len(lines) < 6:
            raise ValueError(f"Invalid commit output: {output}")

        commit_hash = lines[0]
        parent_hashes = lines[1].split() if lines[1] else []
        author_name, author_email = self._parse_author(lines[2])
        committer_name, committer_email = self._parse_author(lines[3])
        timestamp = datetime.fromtimestamp(int(lines[4]))
        message = "\n".join(lines[5:])

        return Commit(
            hash=commit_hash,
            short_hash=commit_hash[:7],
            author=Author(author_name, author_email),
            committer=(
                Author(committer_name, committer_email) if committer_name else None
            ),
            timestamp=timestamp,
            message=message,
            parent_hashes=parent_hashes,
        )

    def _parse_author(self, line: str) -> tuple[str, str]:
        """Parse author line 'Name <email>'"""
        match = re.match(r"^(.+?)\s*<(.+?)>$", line.strip())
        if match:
            return match.group(1), match.group(2)
        return line.strip(), ""

    @property
    def vcs_type(self) -> VCSType:
        return VCSType.GIT

    def get_root(self) -> Path:
        return self._root

    def get_current_commit(self) -> Commit | None:
        try:
            result = self._run_git(
                "log",
                "-1",
                "--format=%H%n%P%n%an <%ae>%n%cn <%ce>%n%ct%n%B",
            )
            return self._parse_commit(result.stdout)
        except subprocess.CalledProcessError:
            return None

    def get_current_branch(self) -> str | None:
        try:
            result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
            branch = result.stdout.strip()
            return branch if branch != "HEAD" else None
        except subprocess.CalledProcessError:
            return None

    def get_branches(self, include_remote: bool = False) -> list[Branch]:
        branches = []

        # Get local branches
        try:
            result = self._run_git(
                "branch",
                "-v",
                "--format=%(refname:short)|%(objectname)|%(HEAD)|%(upstream:short)",
            )
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    name = parts[0]
                    commit_hash = parts[1]
                    is_current = parts[2] == "*"
                    upstream = parts[3] if len(parts) > 3 and parts[3] else None

                    branches.append(
                        Branch(
                            name=name,
                            commit_hash=commit_hash,
                            is_current=is_current,
                            is_remote=False,
                            upstream=upstream,
                            branch_type=Branch.classify(name),
                        )
                    )
        except subprocess.CalledProcessError:
            pass

        # Get remote branches if requested
        if include_remote:
            try:
                result = self._run_git(
                    "branch", "-r", "-v", "--format=%(refname:short)|%(objectname)"
                )
                for line in result.stdout.strip().split("\n"):
                    if not line or "->" in line:  # Skip symbolic refs
                        continue
                    parts = line.split("|")
                    if len(parts) >= 2:
                        name = parts[0]
                        commit_hash = parts[1]

                        branches.append(
                            Branch(
                                name=name,
                                commit_hash=commit_hash,
                                is_current=False,
                                is_remote=True,
                                branch_type=Branch.classify(name.split("/")[-1]),
                            )
                        )
            except subprocess.CalledProcessError:
                pass

        return branches

    def get_tags(self) -> list[Tag]:
        tags = []
        try:
            result = self._run_git(
                "tag",
                "-l",
                "--format=%(refname:short)|%(objectname)|%(objecttype)|%(taggername) <%(taggeremail)>|%(taggerdate:unix)|%(subject)",
            )
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 2:
                    name = parts[0]
                    commit_hash = parts[1]
                    obj_type = parts[2] if len(parts) > 2 else "commit"
                    is_annotated = obj_type == "tag"

                    tag = Tag(
                        name=name,
                        commit_hash=commit_hash,
                        is_annotated=is_annotated,
                    )

                    if is_annotated and len(parts) > 4:
                        tagger_name, tagger_email = self._parse_author(parts[3])
                        tag.tagger = Author(tagger_name, tagger_email)
                        if parts[4]:
                            tag.timestamp = datetime.fromtimestamp(int(parts[4]))
                        if len(parts) > 5:
                            tag.message = parts[5]

                    tags.append(tag)
        except subprocess.CalledProcessError:
            pass

        return tags

    def get_commit(self, ref: str) -> Commit | None:
        if ref in self._commit_cache:
            return self._commit_cache[ref]

        try:
            result = self._run_git(
                "log",
                "-1",
                ref,
                "--format=%H%n%P%n%an <%ae>%n%cn <%ce>%n%ct%n%B",
            )
            commit = self._parse_commit(result.stdout)
            self._commit_cache[ref] = commit
            return commit
        except subprocess.CalledProcessError:
            return None

    def get_commits(
        self,
        since: str | None = None,
        until: str | None = None,
        path: str | None = None,
        limit: int | None = None,
    ) -> list[Commit]:
        commits = []
        args = ["log", "--format=%H%n%P%n%an <%ae>%n%cn <%ce>%n%ct%n%B%x00"]

        if limit:
            args.append(f"-{limit}")

        if since:
            args.append(f"{since}..{until or 'HEAD'}")
        elif until:
            args.append(until)

        if path:
            args.extend(["--", path])

        try:
            result = self._run_git(*args)
            entries = result.stdout.split("\x00")

            for entry in entries:
                entry = entry.strip()
                if entry:
                    try:
                        commit = self._parse_commit(entry)
                        commits.append(commit)
                    except ValueError:
                        continue
        except subprocess.CalledProcessError:
            pass

        return commits

    def get_changed_files(
        self,
        from_ref: str | None = None,
        to_ref: str = "HEAD",
        include_untracked: bool = True,
    ) -> list[FileChange]:
        changes = []

        if from_ref:
            # Get changes between two refs
            try:
                result = self._run_git(
                    "diff", "--name-status", "--numstat", f"{from_ref}..{to_ref}"
                )
                changes.extend(self._parse_diff_name_status(result.stdout))
            except subprocess.CalledProcessError:
                pass
        else:
            # Get uncommitted changes
            try:
                # Staged changes
                result = self._run_git("diff", "--cached", "--name-status")
                changes.extend(self._parse_diff_name_status(result.stdout))

                # Unstaged changes
                result = self._run_git("diff", "--name-status")
                staged_paths = {c.path for c in changes}
                for change in self._parse_diff_name_status(result.stdout):
                    if change.path not in staged_paths:
                        changes.append(change)
            except subprocess.CalledProcessError:
                pass

            if include_untracked:
                try:
                    result = self._run_git("ls-files", "--others", "--exclude-standard")
                    for line in result.stdout.strip().split("\n"):
                        if line:
                            changes.append(
                                FileChange(
                                    path=line,
                                    change_type=ChangeType.UNTRACKED,
                                )
                            )
                except subprocess.CalledProcessError:
                    pass

        return changes

    def _parse_diff_name_status(self, output: str) -> list[FileChange]:
        """Parse git diff --name-status output"""
        changes = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                status = parts[0][0]  # First char is the status
                path = parts[-1]
                old_path = parts[1] if len(parts) > 2 else None

                change_type = {
                    "A": ChangeType.ADDED,
                    "M": ChangeType.MODIFIED,
                    "D": ChangeType.DELETED,
                    "R": ChangeType.RENAMED,
                    "C": ChangeType.COPIED,
                    "T": ChangeType.TYPE_CHANGED,
                }.get(status, ChangeType.MODIFIED)

                changes.append(
                    FileChange(
                        path=path,
                        change_type=change_type,
                        old_path=old_path,
                    )
                )
        return changes

    def get_file_diff(
        self,
        path: str,
        from_ref: str | None = None,
        to_ref: str = "HEAD",
    ) -> FileDiff | None:
        try:
            if from_ref:
                result = self._run_git("diff", f"{from_ref}..{to_ref}", "--", path)
            else:
                result = self._run_git("diff", to_ref, "--", path)

            return self._parse_unified_diff(result.stdout, path)
        except subprocess.CalledProcessError:
            return None

    def _parse_unified_diff(self, output: str, path: str) -> FileDiff:
        """Parse unified diff format"""
        hunks = []
        lines = output.split("\n")

        i = 0
        old_path = None
        is_binary = False

        # Parse header
        while i < len(lines):
            line = lines[i]
            if line.startswith("Binary files"):
                is_binary = True
                break
            if line.startswith("--- a/"):
                old_path = line[6:]
            if line.startswith("@@"):
                break
            i += 1

        # Parse hunks
        current_hunk = None
        hunk_content = []

        while i < len(lines):
            line = lines[i]

            if line.startswith("@@"):
                # Save previous hunk
                if current_hunk:
                    current_hunk.content = "\n".join(hunk_content)
                    hunks.append(current_hunk)

                # Parse hunk header: @@ -start,count +start,count @@
                match = re.match(
                    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$", line
                )
                if match:
                    current_hunk = DiffHunk(
                        old_start=int(match.group(1)),
                        old_count=int(match.group(2) or 1),
                        new_start=int(match.group(3)),
                        new_count=int(match.group(4) or 1),
                        content="",
                        header=match.group(5).strip(),
                    )
                    hunk_content = []
            elif current_hunk:
                hunk_content.append(line)

            i += 1

        # Save last hunk
        if current_hunk:
            current_hunk.content = "\n".join(hunk_content)
            hunks.append(current_hunk)

        # Determine change type
        change_type = ChangeType.MODIFIED
        if old_path is None or old_path == "/dev/null":
            change_type = ChangeType.ADDED

        return FileDiff(
            path=path,
            old_path=old_path if old_path != path else None,
            change_type=change_type,
            hunks=hunks,
            is_binary=is_binary,
        )

    def get_file_content(
        self,
        path: str,
        ref: str = "HEAD",
    ) -> str | None:
        try:
            result = self._run_git("show", f"{ref}:{path}")
            return result.stdout
        except subprocess.CalledProcessError:
            return None

    def get_blame(
        self,
        path: str,
        ref: str = "HEAD",
    ) -> list[BlameEntry]:
        entries = []
        try:
            result = self._run_git("blame", "-p", ref, "--", path)
            entries = self._parse_blame_porcelain(result.stdout)
        except subprocess.CalledProcessError:
            pass
        return entries

    def _parse_blame_porcelain(self, output: str) -> list[BlameEntry]:
        """Parse git blame -p (porcelain) output"""
        entries = []
        lines = output.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Header line: hash orig-line final-line [num-lines]
            match = re.match(r"^([0-9a-f]{40}) (\d+) (\d+)(?: (\d+))?$", line)
            if match:
                commit_hash = match.group(1)
                line_num = int(match.group(3))
                num_lines = int(match.group(4)) if match.group(4) else 1

                # Parse metadata
                author_name = ""
                author_email = ""
                timestamp = None
                content_lines = []

                i += 1
                while i < len(lines) and not lines[i].startswith("\t"):
                    meta_line = lines[i]
                    if meta_line.startswith("author "):
                        author_name = meta_line[7:]
                    elif meta_line.startswith("author-mail "):
                        author_email = meta_line[12:].strip("<>")
                    elif meta_line.startswith("author-time "):
                        timestamp = datetime.fromtimestamp(int(meta_line[12:]))
                    i += 1

                # Get content line
                if i < len(lines) and lines[i].startswith("\t"):
                    content_lines.append(lines[i][1:])
                    i += 1

                if author_name and timestamp:
                    entries.append(
                        BlameEntry(
                            commit_hash=commit_hash,
                            author=Author(author_name, author_email),
                            timestamp=timestamp,
                            line_start=line_num,
                            line_end=line_num + num_lines - 1,
                            content="\n".join(content_lines),
                        )
                    )
            else:
                i += 1

        return entries

    def is_dirty(self) -> bool:
        try:
            result = self._run_git("status", "--porcelain")
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

    def get_remote_url(self) -> str | None:
        try:
            result = self._run_git("remote", "get-url", "origin")
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    # Additional utility methods

    def get_file_history(
        self,
        path: str,
        limit: int | None = None,
    ) -> list[Commit]:
        """Get commit history for a specific file"""
        return self.get_commits(path=path, limit=limit)

    def get_contributors(self) -> list[Author]:
        """Get list of contributors"""
        authors = set()
        try:
            result = self._run_git("log", "--format=%an|%ae")
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("|")
                    if len(parts) == 2:
                        authors.add(Author(parts[0], parts[1]))
        except subprocess.CalledProcessError:
            pass
        return list(authors)

    def get_stats(self) -> dict[str, Any]:
        """Get repository statistics"""
        stats = {
            "total_commits": 0,
            "total_branches": 0,
            "total_tags": 0,
            "total_contributors": 0,
        }

        try:
            result = self._run_git("rev-list", "--count", "HEAD")
            stats["total_commits"] = int(result.stdout.strip())
        except subprocess.CalledProcessError:
            pass

        stats["total_branches"] = len(self.get_branches())
        stats["total_tags"] = len(self.get_tags())
        stats["total_contributors"] = len(self.get_contributors())

        return stats


# =============================================================================
# Repository Manager (Facade)
# =============================================================================


class RepositoryManager:
    """
    High-level repository manager providing unified access to VCS operations.

    Features:
    - Auto-detection of VCS type
    - Index state tracking for incremental updates
    - Multi-repository support
    - Change detection and filtering
    """

    def __init__(
        self,
        backend: VCSBackend,
        index_state: IndexState | None = None,
    ):
        self._backend = backend
        self._index_state = index_state or IndexState(
            repository_id=self._compute_repo_id()
        )

    def _compute_repo_id(self) -> str:
        """Compute unique repository ID"""
        root = str(self._backend.get_root())
        remote = self._backend.get_remote_url() or ""
        return hashlib.sha256(f"{root}:{remote}".encode()).hexdigest()[:16]

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        index_state: IndexState | None = None,
    ) -> "RepositoryManager":
        """
        Create RepositoryManager from path with auto-detection.

        Args:
            path: Path to repository or file within repository
            index_state: Optional previous index state

        Returns:
            RepositoryManager instance

        Raises:
            ValueError: If no supported VCS is detected
        """
        path = Path(path).resolve()

        # Try Git first (most common)
        try:
            backend = GitRepository(path)
            return cls(backend, index_state)
        except ValueError:
            pass

        # Add support for other VCS here in the future
        # e.g., MercurialRepository, SVNRepository

        raise ValueError(f"No supported VCS detected at: {path}")

    @classmethod
    def detect_vcs(cls, path: str | Path) -> VCSType:
        """Detect VCS type at path without creating manager"""
        path = Path(path).resolve()

        current = path
        while current != current.parent:
            if (current / ".git").exists():
                return VCSType.GIT
            if (current / ".hg").exists():
                return VCSType.MERCURIAL
            if (current / ".svn").exists():
                return VCSType.SVN
            current = current.parent

        return VCSType.NONE

    @property
    def vcs_type(self) -> VCSType:
        """Get VCS type"""
        return self._backend.vcs_type

    @property
    def root(self) -> Path:
        """Get repository root"""
        return self._backend.get_root()

    @property
    def index_state(self) -> IndexState:
        """Get current index state"""
        return self._index_state

    def get_info(self) -> RepositoryInfo:
        """Get repository information"""
        stats = self._backend.get_stats() if hasattr(self._backend, "get_stats") else {}
        current_commit = self._backend.get_current_commit()

        return RepositoryInfo(
            root_path=self._backend.get_root(),
            vcs_type=self._backend.vcs_type,
            remote_url=self._backend.get_remote_url(),
            current_branch=self._backend.get_current_branch(),
            current_commit=current_commit.hash if current_commit else None,
            is_dirty=self._backend.is_dirty(),
            total_commits=stats.get("total_commits"),
            total_branches=stats.get("total_branches"),
            total_tags=stats.get("total_tags"),
        )

    def get_changes_since_last_index(
        self,
        include_untracked: bool = True,
    ) -> list[FileChange]:
        """
        Get files that changed since last indexing.

        Returns:
            List of FileChange objects
        """
        if self._index_state.last_indexed_commit:
            return self._backend.get_changed_files(
                from_ref=self._index_state.last_indexed_commit,
                to_ref="HEAD",
                include_untracked=include_untracked,
            )
        else:
            # First index - return all tracked files
            return self._get_all_tracked_files()

    def _get_all_tracked_files(self) -> list[FileChange]:
        """Get all tracked files as 'added' changes"""
        changes = []
        if isinstance(self._backend, GitRepository):
            try:
                result = self._backend._run_git("ls-files")
                for line in result.stdout.strip().split("\n"):
                    if line:
                        changes.append(
                            FileChange(
                                path=line,
                                change_type=ChangeType.ADDED,
                            )
                        )
            except subprocess.CalledProcessError:
                pass
        return changes

    def get_files_to_reindex(
        self,
        filter_code_files: bool = True,
        extensions: set[str] | None = None,
    ) -> list[FileChange]:
        """
        Get files that need reindexing.

        Args:
            filter_code_files: Only return code files
            extensions: Optional set of extensions to filter (e.g., {'.py', '.rs'})

        Returns:
            Filtered list of FileChange objects
        """
        changes = self.get_changes_since_last_index()

        if filter_code_files:
            changes = [c for c in changes if c.is_code_file]

        if extensions:
            changes = [c for c in changes if Path(c.path).suffix.lower() in extensions]

        # Filter out deleted files (they need different handling)
        return [c for c in changes if c.change_type != ChangeType.DELETED]

    def get_deleted_files(self) -> list[FileChange]:
        """Get files that were deleted since last index"""
        changes = self.get_changes_since_last_index(include_untracked=False)
        return [c for c in changes if c.change_type == ChangeType.DELETED]

    def update_index_state(
        self,
        commit_hash: str | None = None,
        indexed_files: dict[str, str] | None = None,
    ) -> None:
        """
        Update index state after successful indexing.

        Args:
            commit_hash: Commit hash to mark as indexed (defaults to HEAD)
            indexed_files: Dict of path -> content_hash for indexed files
        """
        if commit_hash is None:
            current = self._backend.get_current_commit()
            commit_hash = current.hash if current else None

        self._index_state.last_indexed_commit = commit_hash
        self._index_state.last_indexed_time = datetime.now()

        if indexed_files:
            self._index_state.indexed_files.update(indexed_files)

        # Update branch state
        branch = self._backend.get_current_branch()
        if branch and commit_hash:
            self._index_state.branch_states[branch] = commit_hash

    def get_file_content(
        self,
        path: str,
        ref: str = "HEAD",
    ) -> str | None:
        """Get file content at specific ref"""
        return self._backend.get_file_content(path, ref)

    def get_file_blame(
        self,
        path: str,
        ref: str = "HEAD",
    ) -> list[BlameEntry]:
        """Get blame information for a file"""
        return self._backend.get_blame(path, ref)

    def get_file_authors(
        self,
        path: str,
        ref: str = "HEAD",
    ) -> list[Author]:
        """Get unique authors for a file"""
        blame = self.get_file_blame(path, ref)
        return list({entry.author for entry in blame})

    def get_commit_info(self, ref: str = "HEAD") -> Commit | None:
        """Get commit information"""
        return self._backend.get_commit(ref)

    def get_recent_commits(self, limit: int = 10) -> list[Commit]:
        """Get recent commits"""
        return self._backend.get_commits(limit=limit)

    def save_state(self, path: str | Path) -> None:
        """Save index state to file"""
        path = Path(path)
        path.write_text(json.dumps(self._index_state.to_dict(), indent=2))

    @classmethod
    def load_state(cls, state_path: str | Path) -> IndexState | None:
        """Load index state from file"""
        path = Path(state_path)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return IndexState.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                return None
        return None


# =============================================================================
# Utility Functions
# =============================================================================


def is_git_repository(path: str | Path) -> bool:
    """Check if path is inside a git repository"""
    return RepositoryManager.detect_vcs(path) == VCSType.GIT


def get_repository_root(path: str | Path) -> Path | None:
    """Get repository root for a path"""
    try:
        repo = RepositoryManager.from_path(path)
        return repo.root
    except ValueError:
        return None


def get_current_commit_hash(path: str | Path) -> str | None:
    """Get current commit hash for a repository"""
    try:
        repo = RepositoryManager.from_path(path)
        commit = repo.get_commit_info()
        return commit.hash if commit else None
    except ValueError:
        return None


def get_file_git_info(
    file_path: str | Path,
) -> dict[str, Any] | None:
    """
    Get git information for a file.

    Returns dict with:
    - repo_root: Repository root path
    - relative_path: File path relative to repo root
    - commit_hash: Current commit hash
    - branch: Current branch
    - remote_url: Remote URL
    - authors: List of authors who modified the file
    """
    try:
        repo = RepositoryManager.from_path(file_path)
        file_path = Path(file_path).resolve()
        relative_path = file_path.relative_to(repo.root)

        info = repo.get_info()
        authors = repo.get_file_authors(str(relative_path))

        return {
            "repo_root": str(repo.root),
            "relative_path": str(relative_path),
            "commit_hash": info.current_commit,
            "branch": info.current_branch,
            "remote_url": info.remote_url,
            "authors": [{"name": a.name, "email": a.email} for a in authors],
        }
    except (ValueError, Exception):
        return None


# =============================================================================
# Context Manager for Repository Operations
# =============================================================================


@contextmanager
def repository_context(
    path: str | Path,
    state_file: str | Path | None = None,
):
    """
    Context manager for repository operations with automatic state management.

    Usage:
        with repository_context("/path/to/repo", ".index_state.json") as repo:
            changes = repo.get_files_to_reindex()
            # ... do indexing ...
            repo.update_index_state()
    """
    # Load previous state if available
    index_state = None
    if state_file:
        index_state = RepositoryManager.load_state(state_file)

    repo = RepositoryManager.from_path(path, index_state)

    try:
        yield repo
    finally:
        # Save state on exit
        if state_file:
            repo.save_state(state_file)
