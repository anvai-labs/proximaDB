#!/usr/bin/env python3
"""Export deterministic source documents for token-budget corpus builders.

This adapter owns source discovery only. It emits whole source documents as
JSONL and deliberately contains no chunking or tokenizer logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# corpus_io is a sibling module in scripts/bench/. Inject that directory so the
# import works both when run as a script and when loaded via importlib in tests.
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from corpus_io import sha256_file as _sha256  # noqa: E402  — path-injected sibling bench module

DEFAULT_EXTENSIONS = (
    ".adoc",
    ".c",
    ".cpp",
    ".go",
    ".h",
    ".java",
    ".md",
    ".py",
    ".rs",
    ".sql",
    ".ts",
)
DEFAULT_SKIP_DIRS = (
    ".cache",
    ".git",
    ".venv",
    ".worktrees",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "target",
    "vendor",
)


def repository_provenance(root: Path) -> dict[str, Any]:
    """Return reproducible VCS context without making Git the content authority."""
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        check=False,
        text=False,
    )
    if revision.returncode != 0:
        return {"vcs": "unversioned"}
    status = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ],
        capture_output=True,
        check=False,
        text=False,
    )
    if status.returncode != 0:
        raise RuntimeError(f"git status failed for source repository {root}")
    status_bytes = status.stdout
    return {
        "vcs": "git",
        "head_revision": revision.stdout.decode("ascii").strip(),
        "dirty": bool(status_bytes.strip()),
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
    }


def iter_source_files(
    code_root: Path,
    repositories: Iterable[str],
    extensions: set[str],
    skip_dirs: set[str],
):
    for repository in repositories:
        root = code_root / repository
        if not root.is_dir() or root.is_symlink():
            continue
        for current, directories, files in os.walk(root, followlinks=False):
            directories[:] = sorted(
                name
                for name in directories
                if name not in skip_dirs and not (Path(current) / name).is_symlink()
            )
            for name in sorted(files):
                path = Path(current) / name
                if (
                    path.suffix.lower() in extensions
                    and path.is_file()
                    and not path.is_symlink()
                ):
                    yield repository, root, path


def export(args: argparse.Namespace) -> dict[str, Any]:
    repositories = tuple(dict.fromkeys(args.repositories))
    if not repositories:
        raise ValueError("at least one repository is required")
    missing_repositories = [
        repository
        for repository in repositories
        if not (args.code_root / repository).is_dir()
        or (args.code_root / repository).is_symlink()
    ]
    if missing_repositories:
        raise ValueError(
            "missing source repositories: " + ", ".join(missing_repositories)
        )
    extensions = {
        value if value.startswith(".") else f".{value}" for value in args.extensions
    }
    skip_dirs = set(args.skip_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(fd)
    temporary = Path(temp_name)
    inventory = hashlib.sha256()
    repository_counts = dict.fromkeys(repositories, 0)
    provenance = {
        repository: repository_provenance(args.code_root / repository)
        for repository in repositories
        if (args.code_root / repository).is_dir()
    }
    unreadable: list[str] = []
    document_count = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for repository, root, path in iter_source_files(
                args.code_root, repositories, extensions, skip_dirs
            ):
                relative = path.relative_to(root).as_posix()
                source_id = f"{repository}/{relative}"
                try:
                    text = path.read_text(encoding="utf-8", errors="strict")
                except (OSError, UnicodeDecodeError):
                    unreadable.append(source_id)
                    continue
                content_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                inventory.update(source_id.encode("utf-8"))
                inventory.update(b"\0")
                inventory.update(content_digest.encode("ascii"))
                inventory.update(b"\n")
                output.write(
                    json.dumps(
                        {
                            "id": source_id,
                            "text": text,
                            "repository": repository,
                            "relative_path": relative,
                            "content_sha256": content_digest,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                repository_counts[repository] += 1
                document_count += 1
            if unreadable and not getattr(args, "allow_unreadable_sources", False):
                raise ValueError(
                    "unreadable source files require --allow-unreadable-sources: "
                    + ", ".join(unreadable[:5])
                )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "exporter_sha256": _sha256(Path(__file__).resolve()),
        "code_root": str(args.code_root.resolve()),
        "repositories": list(repositories),
        "extensions": sorted(extensions),
        "skip_dirs": sorted(skip_dirs),
        "document_count": document_count,
        "repository_document_counts": repository_counts,
        "repository_provenance": provenance,
        "unreadable_sources": unreadable,
        "source_inventory_sha256": inventory.hexdigest(),
        "jsonl_sha256": _sha256(args.output),
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, manifest_temp_name = tempfile.mkstemp(
        prefix=f"{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
    )
    os.close(fd)
    manifest_temporary = Path(manifest_temp_name)
    try:
        with manifest_temporary.open("w", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(manifest_temporary, manifest_path)
    finally:
        manifest_temporary.unlink(missing_ok=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument(
        "--repository", dest="repositories", action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--extension", dest="extensions", action="append")
    parser.add_argument("--skip-dir", dest="skip_dirs", action="append")
    parser.add_argument(
        "--allow-unreadable-sources",
        action="store_true",
        help="record and drop files that cannot be decoded as UTF-8",
    )
    args = parser.parse_args()
    args.extensions = args.extensions or list(DEFAULT_EXTENSIONS)
    args.skip_dirs = args.skip_dirs or list(DEFAULT_SKIP_DIRS)
    return args


def main() -> None:
    print(json.dumps(export(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
