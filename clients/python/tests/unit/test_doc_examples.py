"""TD-SDKDOC-1: doc examples are EXECUTED, not just reviewed.

The SDK guide and quick starts drifted from the SDK repeatedly
(`FilterBuilder().add`, `client.create_nodes`, a duplicated keyword argument
that is a SyntaxError) and fixing them by review alone took several passes.
This module extracts every Python block from the doc families the TD names
and holds them to the SDK as it exists:

* **compile tier** — every non-skipped block must at least be valid Python
  (the duplicated-keyword SyntaxError class).
* **symbol tier** — every `proximadb_sdk` name a block imports or touches
  must resolve against the real package (the `create_nodes` /
  `knowledge_graph_search` class).
* **execution tier** — with a live server (``PROXIMADB_DOC_SMOKE_URL``),
  each document's blocks run IN ORDER in one namespace, exactly as a reader
  would type them: a guide block may use the ``client`` an earlier block
  built. The server URL is substituted for the docs' ``localhost:5678``.

Blocks that cannot run against the smoke server carry ``# doc-smoke: skip`` as
their first line (the rest of the line is the reason). That exempts them from
EXECUTION only: they still have to compile and name real SDK symbols. Every
python fence in a source file must be extracted, so a fence variant cannot
drop a block out silently. A block written for a fresh server (a "complete example" that
re-creates what the walkthrough already made) carries
``# doc-smoke: fresh-server``: the runner drops the collections the document
created so far and runs the block in a fresh namespace.

A block also fails when it leaves a coroutine un-awaited, as a bare call or
held in a name (an async SDK call written as if it were sync does nothing);
when the client falls back to its offline in-memory store (on a transport
error it returns fake success and logs only at DEBUG, so a server that died
mid-run would pass every later block); when it runs past a per-block timeout;
or when the SDK logs a WARNING or worse while it runs: the
SDK reduces some failures to a log line (the multi-modal executor turned
every failed leg into a warning and returned empty results), and an example
that "passes" that way is broken.

The external embedding model the quick starts use (``sentence-transformers``)
is replaced, only when it is not installed, by a deterministic stub of the
same shape — the examples' ProximaDB calls still run for real.
"""

from __future__ import annotations

import ast
import gc
import inspect
import logging
import os
import re
import signal
import sys
import textwrap
import types
import urllib.request
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import proximadb_sdk

DOC_SMOKE_URL = os.environ.get("PROXIMADB_DOC_SMOKE_URL", "").rstrip("/")

if not proximadb_sdk.__file__.startswith(str(Path(__file__).resolve().parents[2])):
    _wrong_copy = (
        f"tests the worktree SDK, but {proximadb_sdk.__file__} won the import; "
        "run with PYTHONPATH=<worktree>/clients/python/src"
    )
    if DOC_SMOKE_URL:
        # The CI doc-smoke step: a skip here would turn the gate green.
        raise RuntimeError(_wrong_copy)
    pytest.skip(_wrong_copy, allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[4]
# Per-block wall-clock limit: a hung request must fail the block, not burn the
# CI job's timeout.
BLOCK_TIMEOUT_S = 120

SKIP_MARKER = re.compile(r"^\s*#\s*doc-smoke:\s*skip\b[:\s]*(?P<reason>.*)$")
FRESH_MARKER = re.compile(r"^\s*#\s*doc-smoke:\s*fresh-server\b")
DOC_HOSTPORT = "localhost:5678"

DOC_SOURCES = sorted(
    [REPO_ROOT / "docs" / "02-guides" / "sdk-python-guide.adoc"]
    + list((REPO_ROOT / "docs" / "01-quick-start").glob("*.md"))
    + list((REPO_ROOT / "docs" / "05-concepts").glob("*.adoc"))
)

# asciidoc: [source,python(,attrs)] then a ---- fence; markdown: a ```python
# fence (info-string attributes allowed), possibly indented inside a list.
ADOC_BLOCK = re.compile(
    r"^\[source,\s*python\b[^\]\n]*\][^\n]*\n----\n(?P<body>.*?)\n----[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
MD_BLOCK = re.compile(
    r"^(?P<indent>[ \t]*)```[ \t]*python\b[^\n]*\n(?P<body>.*?)\n(?P=indent)```",
    re.MULTILINE | re.DOTALL,
)
# Loose detectors for anything that LOOKS like a python block opener: every
# one must be extracted, so a fence variant can never drop out silently.
ADOC_OPENER = re.compile(
    r"^\[(?:source)?,\s*(?:python|py)\w*\b", re.MULTILINE | re.IGNORECASE
)
MD_OPENER = re.compile(
    r"^[ \t]*(?:```|~~~)[ \t]*\{?\.?(?:python|py)\w*\b",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class DocBlock:
    file: Path
    index: int
    body: str
    skip_reason: str | None = None

    @property
    def id(self) -> str:
        return f"{self.file.relative_to(REPO_ROOT)}#{self.index}"


def _extract_blocks() -> list[DocBlock]:
    blocks: list[DocBlock] = []
    for path in DOC_SOURCES:
        text = path.read_text(encoding="utf-8")
        pattern = MD_BLOCK if path.suffix == ".md" else ADOC_BLOCK
        for index, match in enumerate(pattern.finditer(text)):
            body = textwrap.dedent(match.group("body"))
            marker = SKIP_MARKER.search(body.splitlines()[0] if body else "")
            blocks.append(
                DocBlock(
                    file=path,
                    index=index,
                    body=body,
                    skip_reason=marker.group("reason").strip() if marker else None,
                )
            )
    return blocks


ALL_BLOCKS = _extract_blocks()
# `# doc-smoke: skip` exempts a block from EXECUTION only: it must still
# compile and name real SDK symbols.
EXECUTABLE = [b for b in ALL_BLOCKS if b.skip_reason is None]


def test_doc_sources_found_and_blocks_extracted():
    """The extraction itself is pinned: losing a doc file or a fence variant
    the regexes do not match would silently empty part of the suite (the
    exact failure mode this module exists to prevent)."""
    assert DOC_SOURCES, "no doc sources found — the extractor is broken"
    assert ALL_BLOCKS, "no python blocks extracted — the fence regex drifted"
    for path in DOC_SOURCES:
        text = path.read_text(encoding="utf-8")
        opener = MD_OPENER if path.suffix == ".md" else ADOC_OPENER
        found = len(opener.findall(text))
        extracted = sum(1 for b in ALL_BLOCKS if b.file == path)
        assert extracted == found, (
            f"{path.relative_to(REPO_ROOT)}: {found} python block openers but "
            f"{extracted} extracted — a fence variant the extractor misses"
        )


@pytest.mark.parametrize("block", ALL_BLOCKS, ids=[b.id for b in ALL_BLOCKS])
def test_block_compiles(block: DocBlock):
    try:
        compile(block.body, f"<{block.id}>", "exec")
    except SyntaxError as e:
        pytest.fail(
            f"{block.id} does not compile (line {e.lineno}): {e.msg}\n"
            f"--- block ---\n{block.body}"
        )


def _resolve_dotted(root: Any, parts: list[str]) -> bool:
    obj = root
    for part in parts:
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    return True


def _sdk_names_used(block: DocBlock) -> list[str]:
    """Dotted SDK names: `from proximadb_sdk import X` and
    `proximadb_sdk.a.b` attribute chains."""
    try:
        tree = ast.parse(block.body)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
            node.module == "proximadb_sdk"
            or (node.module or "").startswith("proximadb_sdk.")
        ):
            names.extend((node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Attribute):
            parts: list[str] = []
            cur: ast.AST = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id == "proximadb_sdk":
                names.append(("proximadb_sdk", ".".join(reversed(parts))))
    return names


@pytest.mark.parametrize("block", ALL_BLOCKS, ids=[b.id for b in ALL_BLOCKS])
def test_block_sdk_symbols_resolve(block: DocBlock):
    problems = []
    for module, name in _sdk_names_used(block):
        if module == "proximadb_sdk":
            if not _resolve_dotted(proximadb_sdk, name.split(".")):
                problems.append(name)
        else:
            try:
                __import__(module)
            except ImportError:
                problems.append(module)
                continue
            if not _resolve_dotted(__import__(module, fromlist=["_"]), name.split(".")):
                problems.append(f"{module}.{name}")
    if problems:
        pytest.fail(
            f"{block.id} references SDK names that do not exist: "
            f"{', '.join(sorted(set(problems)))}\n--- block ---\n{block.body}"
        )


def test_silent_failure_detectors():
    """The execution tier's detectors for failures the SDK does not raise.

    Pinned without a server: a client that fell back to its offline store
    (a dead port makes create_collection return a fake Collection and set
    this flag) and a coroutine held in a name must both be reported."""
    from proximadb_sdk import ProximaDBClient

    client = ProximaDBClient.__new__(ProximaDBClient)
    client._prefer_local_fallback = True

    async def later() -> None:
        return None

    namespace: dict[str, Any] = {"client": client, "coll": later()}
    problems = _silent_failures(namespace)
    assert "`client` fell back to the offline store" in problems
    assert "`coll` holds an un-awaited coroutine" in problems
    assert namespace["coll"] is None  # closed, so no late warning either

    healthy = ProximaDBClient.__new__(ProximaDBClient)
    healthy._prefer_local_fallback = False
    assert _silent_failures({"client": healthy}) == []


def test_fallback_recorder_sees_clients_outside_the_namespace():
    """A client that never reaches the namespace (built inside a function,
    or held by a wrapper as `.client`) is still caught when it falls back,
    and the original method is restored afterwards."""
    from proximadb_sdk import ProximaDBClient

    original = ProximaDBClient._activate_local_fallback

    def hidden_client_falls_back() -> None:
        client = ProximaDBClient.__new__(ProximaDBClient)
        client._prefer_local_fallback = False
        client._activate_local_fallback(ConnectionError("refused"))
        client._activate_local_fallback(ConnectionError("again"))  # already on

    with _FallbackRecorder() as recorder:
        hidden_client_falls_back()
    assert [str(e) for e in recorder.activations] == ["refused"]
    assert ProximaDBClient._activate_local_fallback is original


def test_execution_tier_runs_when_server_available(tmp_path, monkeypatch):
    """One namespace per document, blocks in order, against a live server.

    Skipped entirely (not failed) when no server is configured — the
    compile/symbol tiers always run; the round-trip CI lane sets
    PROXIMADB_DOC_SMOKE_URL and gets this tier.
    """
    if not DOC_SMOKE_URL:
        pytest.skip("execution tier needs a live server: set PROXIMADB_DOC_SMOKE_URL")
    # Relative paths in the examples (./data, audit.jsonl) land in a temp
    # dir, never in the repo.
    monkeypatch.chdir(tmp_path)
    smoke_hostport = re.sub(r"^[a-z]+://", "", DOC_SMOKE_URL)
    by_file: dict[Path, list[DocBlock]] = {}
    for block in EXECUTABLE:
        by_file.setdefault(block.file, []).append(block)

    failures: list[str] = []
    with _stub_sentence_transformers():
        for path, blocks in by_file.items():
            # Each document is a walkthrough on a fresh server: what its
            # blocks create is dropped afterwards, so the NEXT document does
            # not fail COLLECTION_EXISTS on a name it shares.
            namespace: dict[str, Any] = {"__name__": "doc_smoke"}
            created: set[str] = set()
            graphs: set[str] = set()
            for block in blocks:
                # Fail fast on a dead server: every request against it waits
                # out its retries, so carrying on would only burn the block
                # timeout once per remaining block.
                if not _server_alive():
                    failures.append(
                        f"{block.id}: smoke server {DOC_SMOKE_URL} is not healthy "
                        "before this block (it died during an earlier one)"
                    )
                    pytest.fail("\n".join(failures))
                if FRESH_MARKER.search(block.body.splitlines()[0]):
                    _drop_collections(namespace, created, graphs)
                    namespace = {"__name__": "doc_smoke"}
                before = _collection_names(namespace)
                graphs_before = _graph_ids(namespace)
                source = block.body.replace(DOC_HOSTPORT, smoke_hostport)
                with (
                    _SdkLogCapture() as sdk_logs,
                    warnings.catch_warnings(record=True) as caught,
                    _FallbackRecorder() as fallbacks,
                ):
                    warnings.simplefilter("always")
                    try:
                        # The alarm covers the block alone: the checks and
                        # gc.collect() below must not be cut short by it.
                        with _block_timeout(block.id):
                            exec(compile(source, f"<{block.id}>", "exec"), namespace)
                    except (
                        Exception,
                        _BlockTimeout,
                    ) as e:  # noqa: BLE001 — report, keep going
                        failures.append(f"{block.id}: {type(e).__name__}: {e}")
                    failures.extend(
                        f"{block.id}: {problem}"
                        for problem in _silent_failures(namespace)
                    )
                    failures.extend(
                        f"{block.id}: a client fell back to the offline store "
                        f"({error!r})"
                        for error in fallbacks.activations
                    )
                    gc.collect()  # "never awaited" is reported at collection
                failures.extend(
                    f"{block.id}: SDK logged {r.levelname}: {r.getMessage()}"
                    for r in sdk_logs.records
                )
                failures.extend(
                    f"{block.id}: {w.message}"
                    for w in caught
                    if "was never awaited" in str(w.message)
                )
                created |= _collection_names(namespace) - before
                graphs |= _graph_ids(namespace) - graphs_before
            if _server_alive():
                _drop_collections(namespace, created, graphs)
    assert not failures, "\n".join(failures)


def _server_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{DOC_SMOKE_URL}/health", timeout=5) as resp:
            return resp.status == 200
    except OSError:
        return False


class _BlockTimeout(BaseException):
    """BaseException, so the SDK's broad `except Exception` handlers (e.g. a
    gRPC pool's connection validation) cannot swallow it."""


class _FallbackRecorder:
    """Record every offline-fallback activation during a block, whatever
    holds the client: a function-local client, a rebound name and a wrapper
    storing it as `.client` are all invisible to a namespace scan."""

    def __enter__(self) -> _FallbackRecorder:
        from proximadb_sdk import ProximaDBClient

        self.activations: list[Exception] = []
        self.original = ProximaDBClient._activate_local_fallback
        original = self.original
        activations = self.activations

        def recording(client: Any, error: Exception) -> None:
            if not getattr(client, "_prefer_local_fallback", False):
                activations.append(error)
            original(client, error)

        ProximaDBClient._activate_local_fallback = recording  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        from proximadb_sdk import ProximaDBClient

        ProximaDBClient._activate_local_fallback = self.original  # type: ignore[method-assign]


class _block_timeout:
    """SIGALRM wall-clock limit for one block (main thread, POSIX only)."""

    def __init__(self, block_id: str) -> None:
        self.block_id = block_id
        self.armed = hasattr(signal, "SIGALRM")

    def _fire(self, *_: Any) -> None:
        raise _BlockTimeout(f"{self.block_id} exceeded {BLOCK_TIMEOUT_S}s")

    def __enter__(self) -> None:
        if self.armed:
            self.previous = signal.signal(signal.SIGALRM, self._fire)
            signal.alarm(BLOCK_TIMEOUT_S)

    def __exit__(self, *exc: Any) -> None:
        if self.armed:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.previous)


def _silent_failures(namespace: dict[str, Any]) -> list[str]:
    """Failures the SDK does not raise.

    * Offline fallback: on a transport error ProximaDBClient switches to an
      in-memory store and returns fake success, logging only at DEBUG — a
      server that died mid-run would otherwise pass every later block.
    * A coroutine kept in a name: an async call written as if it were sync
      does nothing, and while the name holds it no "never awaited" warning
      fires.
    """
    from proximadb_sdk import ProximaDBClient

    # The flag, not the class-level store: the client also caches every
    # collection it creates successfully in that store.
    problems: list[str] = []
    for name, value in list(namespace.items()):
        candidates = [value, getattr(value, "_client", None)]
        if any(
            isinstance(c, ProximaDBClient)
            and getattr(c, "_prefer_local_fallback", False)
            for c in candidates
        ):
            problems.append(f"`{name}` fell back to the offline store")
        if inspect.iscoroutine(value):
            problems.append(f"`{name}` holds an un-awaited coroutine")
            value.close()
            namespace[name] = None
    return sorted(set(problems))


class _SdkLogCapture(logging.Handler):
    """Collect WARNING+ records from the SDK's loggers for one block."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("proximadb"):
            self.records.append(record)

    def __enter__(self) -> _SdkLogCapture:
        logging.getLogger().addHandler(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        logging.getLogger().removeHandler(self)


def _drop_collections(
    namespace: dict[str, Any], names: set[str], graphs: set[str]
) -> None:
    """Drop what a document created (collections and graphs), best-effort."""
    client = namespace.get("client")
    if client is not None:
        for name in sorted(names):
            try:
                client.delete_collection(name)
            except Exception:  # noqa: BLE001 — cleanup is best-effort
                pass
        for graph_id in sorted(graphs):
            try:
                client.delete_graph(graph_id)
            except Exception:  # noqa: BLE001 — cleanup is best-effort
                pass
    names.clear()
    graphs.clear()


def _graph_ids(namespace: dict[str, Any]) -> set[str]:
    """Graph ids visible to the namespace's client (REST only; else empty)."""
    client = namespace.get("client")
    if client is None or not hasattr(client, "list_graphs"):
        return set()
    try:
        listing = client.list_graphs()
    except Exception:  # noqa: BLE001 — gRPC client / listing unsupported
        return set()
    rows = listing.get("data", []) if isinstance(listing, dict) else listing
    return {row["graph_id"] for row in rows or [] if "graph_id" in row}


class _StubSentenceTransformer:
    """Deterministic stand-in for the quick starts' embedding model: same
    call shape and dimension (all-MiniLM-L6-v2 is 384-d), no download."""

    dimension = 384

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        self.model_name = model_name

    def encode(self, texts: Any, *args: Any, **kwargs: Any) -> Any:
        import numpy as np

        def embed(text: str) -> Any:
            # crc32, not hash(): str hashing is salted per process.
            rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
            return rng.random(self.dimension, dtype=np.float32)

        if isinstance(texts, str):
            return embed(texts)
        return np.stack([embed(t) for t in texts])


class _stub_sentence_transformers:
    """Install the stub only when the real package is absent; restore after."""

    def __enter__(self) -> None:
        self._installed = False
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            module = types.ModuleType("sentence_transformers")
            module.SentenceTransformer = _StubSentenceTransformer  # type: ignore[attr-defined]
            sys.modules["sentence_transformers"] = module
            self._installed = True

    def __exit__(self, *exc: Any) -> None:
        if self._installed:
            sys.modules.pop("sentence_transformers", None)


def _collection_names(namespace: dict[str, Any]) -> set[str]:
    """Collection names visible to the namespace's client (empty without one)."""
    client = namespace.get("client")
    if client is None:
        return set()
    try:
        return {c.config.name for c in client.list_collections() if c.config}
    except Exception:  # noqa: BLE001 — no server yet / listing unsupported
        return set()
