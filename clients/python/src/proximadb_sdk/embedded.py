"""
ProximaDB Embedded Mode - Lightweight embedded vector + graph database.

This module provides an embedded database mode for ProximaDB that can be used
without running a separate server process. Perfect for:
- CLI tools (like Victor/codingagent)
- Single-user applications
- Development and testing
- Semantic code knowledge stores

Features:
- In-process database (no separate server needed)
- Vector similarity search with HNSW index
- Graph storage with ORION engine
- Persistent storage to disk
- Low memory footprint
- Thread-safe operations
- Pluggable embedding models (sentence-transformers, Ollama, OpenAI, etc.)

Usage:
    from proximadb import EmbeddedProximaDB

    # Initialize embedded database with auto-embedding
    db = EmbeddedProximaDB(data_dir="~/.myapp/data")
    await db.start()

    # Create collection with embedding model
    collection = await db.create_collection(
        "code_symbols",
        dimension=384,
        embedding_model="BAAI/bge-small-en-v1.5"  # or custom embedding function
    )

    # Insert with auto-embedding (pass text, get embeddings automatically)
    await collection.insert_with_embedding([
        {"id": "func_1", "text": "def hello(): print('world')", "metadata": {"name": "hello"}}
    ])

    # Or insert pre-computed vectors
    await collection.insert([
        {"id": "func_1", "vector": [...], "metadata": {"name": "my_func"}}
    ])

    # Search with text (auto-embedded)
    results = await collection.search_text("hello function", top_k=10)

    # Or search with vector
    results = await collection.search(query_vector, top_k=10)

    # Cleanup
    await db.stop()
"""

import asyncio
import atexit
import base64
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

# Socket file names MUST match the Rust constants in
# crates/platform/proximadb-runtime/src/bootstrap_config.rs.
UDS_REST_SOCKET_NAME = "rest.sock"
UDS_GRPC_SOCKET_NAME = "grpc.sock"
UDS_FLIGHT_SOCKET_NAME = "flight.sock"
UDS_SOCKET_PATH_LIMIT = 100

# =============================================================================
# Embedding Model Abstraction
# =============================================================================


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for embedding functions.

    Any callable that takes a string and returns a list of floats can be used.
    This allows integration with any embedding provider.
    """

    def __call__(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...


@runtime_checkable
class AsyncEmbeddingFunction(Protocol):
    """Protocol for async embedding functions."""

    async def __call__(self, text: str) -> list[float]:
        """Generate embedding for text asynchronously."""
        ...


@runtime_checkable
class BatchEmbeddingFunction(Protocol):
    """Protocol for batch embedding functions."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models.

    Implement this for custom embedding providers.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        pass

    async def embed_async(self, text: str) -> list[float]:
        """Async version of embed (default: runs sync in executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, text)

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_batch (default: runs sync in executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts)


class SentenceTransformerModel(BaseEmbeddingModel):
    """Sentence-transformers embedding model.

    This is the default embedding model for ProximaDB.
    Supports all models from https://huggingface.co/sentence-transformers

    Popular models:
    - BAAI/bge-small-en-v1.5: Default benchmark model, 384-dim (130MB)
    - all-MiniLM-L6-v2: Fast legacy option, 384-dim (80MB)
    - all-MiniLM-L12-v2: Balanced legacy option, 384-dim (120MB)
    - all-mpnet-base-v2: High quality, 768-dim (420MB)
    """

    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None
    ):
        """Initialize sentence-transformer model.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cpu, cuda, mps). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded (lazy loading)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        self._model = SentenceTransformer(
                            self.model_name, device=self.device
                        )
                        self._dimension = self._model.get_sentence_embedding_dimension()
                    except ImportError:
                        raise ImportError(
                            "sentence-transformers not installed. "
                            "Install with: pip install sentence-transformers"
                        )

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        self._ensure_loaded()
        embedding = self._model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        self._ensure_loaded()
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        self._ensure_loaded()
        return self._dimension or 384


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """Ollama embedding model.

    Use local Ollama for embeddings. Supports high-quality models like:
    - nomic-embed-text: 768-dim, fast
    - mxbai-embed-large: 1024-dim, high quality
    - qwen3-embedding:8b: 4096-dim, best quality
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: int = 768,
    ):
        """Initialize Ollama embedding model.

        Args:
            model_name: Ollama model name (must be pulled first)
            base_url: Ollama server URL
            dimension: Embedding dimension (model-specific)
        """
        self.model_name = model_name
        self.base_url = base_url
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Generate embedding using Ollama."""
        import httpx

        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": text},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # Ollama doesn't have native batch API, so we parallelize
        import httpx

        embeddings = []
        with httpx.Client(timeout=60.0) as client:
            for text in texts:
                response = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
        return embeddings

    async def embed_async(self, text: str) -> list[float]:
        """Generate embedding asynchronously."""
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings asynchronously with concurrency."""
        tasks = [self.embed_async(text) for text in texts]
        return await asyncio.gather(*tasks)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI embedding model.

    Supports:
    - text-embedding-3-small: 1536-dim, fast, cheap
    - text-embedding-3-large: 3072-dim, high quality
    - text-embedding-ada-002: 1536-dim, legacy
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        """Initialize OpenAI embedding model.

        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def embed(self, text: str) -> list[float]:
        """Generate embedding using OpenAI."""
        import httpx

        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model_name, "input": text},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        import httpx

        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model_name, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()["data"]
        # Sort by index to maintain order
        data.sort(key=lambda x: x["index"])
        return [d["embedding"] for d in data]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimensions.get(self.model_name, 1536)


class FunctionEmbeddingModel(BaseEmbeddingModel):
    """Wrapper for custom embedding functions.

    Use this to integrate any embedding provider by providing a function.

    Example:
        # Use Victor's EmbeddingService
        from victor.embeddings.service import EmbeddingService
        service = EmbeddingService.get_instance()

        model = FunctionEmbeddingModel(
            embed_fn=lambda text: service.embed_text_sync(text).tolist(),
            batch_fn=lambda texts: service.embed_batch_sync(texts).tolist(),
            dimension=384,
        )
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        dimension: int,
        batch_fn: Callable[[list[str]], list[list[float]]] | None = None,
        async_embed_fn: Callable[[str], Any] | None = None,
    ):
        """Initialize with custom functions.

        Args:
            embed_fn: Function to embed a single text
            dimension: Embedding dimension
            batch_fn: Optional function to embed batch of texts
            async_embed_fn: Optional async function for embedding
        """
        self._embed_fn = embed_fn
        self._batch_fn = batch_fn
        self._async_embed_fn = async_embed_fn
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Generate embedding using custom function."""
        return self._embed_fn(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if self._batch_fn:
            return self._batch_fn(texts)
        # Fallback to sequential
        return [self._embed_fn(text) for text in texts]

    async def embed_async(self, text: str) -> list[float]:
        """Generate embedding asynchronously."""
        if self._async_embed_fn:
            return await self._async_embed_fn(text)
        return await super().embed_async(text)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


def create_embedding_model(
    model_type: str = "sentence-transformers",
    model_name: str | None = None,
    **kwargs,
) -> BaseEmbeddingModel:
    """Factory function to create embedding models.

    Args:
        model_type: Type of model (sentence-transformers, ollama, openai)
        model_name: Model name (type-specific)
        **kwargs: Additional model-specific arguments

    Returns:
        Embedding model instance

    Example:
        # Sentence-transformers (default, local)
        model = create_embedding_model("sentence-transformers", "BAAI/bge-small-en-v1.5")

        # Ollama (local, high quality)
        model = create_embedding_model("ollama", "nomic-embed-text")

        # OpenAI (cloud)
        model = create_embedding_model("openai", "text-embedding-3-small", api_key="...")
    """
    if model_type == "sentence-transformers":
        return SentenceTransformerModel(
            model_name=model_name or "BAAI/bge-small-en-v1.5",
            device=kwargs.get("device"),
        )
    elif model_type == "ollama":
        return OllamaEmbeddingModel(
            model_name=model_name or "nomic-embed-text",
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            dimension=kwargs.get("dimension", 768),
        )
    elif model_type == "openai":
        return OpenAIEmbeddingModel(
            model_name=model_name or "text-embedding-3-small",
            api_key=kwargs.get("api_key"),
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: sentence-transformers, ollama, openai"
        )


def _collection_field(entry: Any, field: str) -> Any:
    """Read a collection descriptor field across both payload shapes.

    The v2 REST API returns collection descriptors **flat**::

        {"collection_id": "1", "name": "docs", "dimension": 384, ...}

    Older payloads nested the same fields under ``config``::

        {"config": {"name": "docs", "dimension": 384}}

    This SDK previously assumed the nested form unconditionally, so
    ``list_collections()`` raised ``KeyError('config')`` and ``get_collection()``
    returned ``None`` even on a 200 against a live v2 server. Reading either
    shape keeps both server versions working.
    """
    if not isinstance(entry, dict):
        return None
    if field in entry:
        return entry[field]
    config = entry.get("config")
    if isinstance(config, dict):
        return config.get(field)
    return None


logger = logging.getLogger(__name__)

# Servers spawned by THIS process, weakly held so a garbage-collected handle
# does not keep an instance alive. Drained at interpreter exit.
_OWNED_SERVERS: "weakref.WeakSet[Any]" = weakref.WeakSet()
_ATEXIT_REGISTERED = False


def _register_owned_server(db: Any) -> None:
    """Track a spawned server so interpreter exit tears it down."""
    global _ATEXIT_REGISTERED
    _OWNED_SERVERS.add(db)
    if not _ATEXIT_REGISTERED:
        atexit.register(_stop_owned_servers)
        _ATEXIT_REGISTERED = True


def _stop_owned_servers() -> None:
    """Stop every server this process started. Best-effort and idempotent:
    exit-time teardown must never raise, and stopping twice is harmless."""
    for db in list(_OWNED_SERVERS):
        try:
            if getattr(db, "_process", None) is not None:
                db._kill_process()
        except Exception:  # pragma: no cover - exit path must not raise
            pass


RUNTIME_STATE_FILE = ".proximadb-runtime.json"
# Mirrors the server's contract constants (apps/proximadb-server/src/runtime_state.rs).
_STALE_AFTER_INTERVALS = 15


def read_runtime_state(data_dir: "str | Path") -> dict[str, Any] | None:
    """Read the server lifecycle record for a data dir, or None if absent.

    A malformed record reads as None ("no owner") — a corrupt file must never
    wedge a client, exactly as it must never block the server.
    """
    try:
        raw = (Path(data_dir) / RUNTIME_STATE_FILE).read_text()
        state = json.loads(raw)
        return state if isinstance(state, dict) else None
    except Exception as exc:
        # A probe: an unreadable/absent state file legitimately means "no state".
        # Logged so an unexpected cause (permissions, corruption) is visible
        # rather than reading as a clean absence.
        logger.debug("could not read runtime state from %s: %s", data_dir, exc)
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        # Exists but owned by another user.
        return True


def runtime_state_is_stale(state: dict[str, Any], now_ms: float) -> bool:
    """True when the recorded owner is gone or its heartbeat stopped advancing.

    Pure function of the record + a caller-supplied clock, so the decision is
    testable without spawning anything.
    """
    pid = state.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return True
    interval = float(state.get("heartbeat_interval_ms") or 2000) or 2000.0
    updated = float(state.get("updated_at_ms") or 0)
    return (now_ms - updated) > interval * _STALE_AFTER_INTERVALS


# =============================================================================
# Configuration
# =============================================================================


class _PooledClientLease:
    """Async context manager yielding a shared ``httpx.AsyncClient`` without
    closing it on exit — so ``async with db._http_client() as client:`` keeps
    working at every call site while the client (and its connection pool)
    lives for the server instance. Closing happens in ``stop()``."""

    __slots__ = ("_client",)

    def __init__(self, client) -> None:
        self._client = client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


@dataclass
class EmbeddedConfig:
    """Configuration for embedded ProximaDB."""

    data_dir: str = field(default_factory=lambda: str(Path.home() / ".proximadb"))
    rest_port: int = 15678  # Use non-standard port to avoid conflicts (TCP mode only)
    grpc_port: int = 15679  # TCP mode only
    arrow_flight_port: int = 15680  # TCP mode only
    log_level: str = "warn"
    memory_flush_size_mb: int = 16
    cache_size_mb: int = 128

    # Transport for the embedded server's REST/gRPC/Arrow Flight surfaces:
    #   "uds" (default) — PORTLESS: bind Unix-domain sockets, no TCP ports at all.
    #   "tcp"           — legacy: bind the *_port fields on loopback.
    # Portless is the embedded default precisely because the *_port values are a
    # collision footgun for concurrent embedded instances.
    transport: str = "uds"
    socket_dir: str | None = None

    # Engine selection (opinionated defaults for code knowledge store)
    vector_engine: str = "SST"  # Best for real-time code indexing
    graph_engine: str = "ORION"  # In-memory with WAL for code relationships

    # Absolute ceiling on startup. This is a SAFETY STOP, not a schedule:
    # while the server's runtime-state record shows an advancing heartbeat the
    # SDK keeps waiting, because recovery time scales with data (a 1.3GB dir
    # measured >180s) and any fixed constant eventually kills a healthy
    # server and silently falls back to another backend (proximaDB#1667).
    startup_timeout_s: float = 900.0
    # Refuse to spawn when another LIVE server already owns the data dir.
    # Set False to spawn anyway (tests/diagnostics) — never silently, since a
    # second writer on one data dir is how "cleared" state resurrects
    # (anvai-labs/victor#911).
    fail_on_foreign_owner: bool = True

    # SIGTERM -> SIGKILL grace window on stop(). The server's final
    # flush-on-stop is a real materialize (tens of seconds for large
    # sessions); killing it strands the whole session's WAL on disk. The
    # wait polls, so small sessions still exit in ~1-3s — this is a
    # ceiling, not a sleep.
    shutdown_timeout_s: float = 60.0


class EmbeddedCollection:
    """A collection in the embedded database.

    Supports record-native operations and text-based operations
    with automatic embedding generation.
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        db: "EmbeddedProximaDB",
        embedding_model: BaseEmbeddingModel | None = None,
    ):
        self.name = name
        self.dimension = dimension
        self._db = db
        self._embedding_model = embedding_model

    def set_embedding_model(self, model: BaseEmbeddingModel) -> None:
        """Set the embedding model for this collection.

        Args:
            model: Embedding model instance
        """
        self._embedding_model = model

    async def insert_records(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Insert ProximaRecord-shaped records into collection.

        Args:
            records: List of dicts with 'id', 'vector', and optional 'props'

        Returns:
            Insert result with success status
        """
        return await self._db._insert_records(self.name, records)

    async def insert(
        self,
        vectors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compatibility alias for record-native insert."""
        return await self.insert_records(vectors)

    async def insert_with_embedding(
        self,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Insert documents with automatic embedding generation.

        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'
                       The 'text' field will be embedded automatically.

        Returns:
            Insert result with success status

        Example:
            await collection.insert_with_embedding([
                {"id": "doc1", "text": "Hello world", "metadata": {"type": "greeting"}},
                {"id": "doc2", "text": "Goodbye world", "metadata": {"type": "farewell"}},
            ])
        """
        if not self._embedding_model:
            raise RuntimeError(
                "No embedding model configured. Either:\n"
                "1. Create collection with embedding_model parameter\n"
                "2. Call set_embedding_model() on the collection\n"
                "3. Use insert() with pre-computed vectors"
            )

        # Extract texts and generate embeddings
        texts = [doc["text"] for doc in documents]
        embeddings = await self._embedding_model.embed_batch_async(texts)

        # Build record-native payloads.
        records = []
        for doc, embedding in zip(documents, embeddings):
            record = {
                "id": doc["id"],
                "vector": embedding,
            }
            props = doc.get("props") or doc.get("metadata", {}).copy()
            props["text"] = doc["text"]
            record["props"] = props
            records.append(record)

        return await self._db._insert_records(self.name, records)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results with id, score, and metadata
        """
        return await self._db._search_vectors(self.name, query_vector, top_k, filters)

    async def search_text(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using text query with automatic embedding.

        Args:
            query: Text query (will be embedded automatically)
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results with id, score, text, and metadata
        """
        if not self._embedding_model:
            raise RuntimeError(
                "No embedding model configured. Either:\n"
                "1. Create collection with embedding_model parameter\n"
                "2. Call set_embedding_model() on the collection\n"
                "3. Use search() with pre-computed query vector"
            )

        query_vector = await self._embedding_model.embed_async(query)
        return await self._db._search_vectors(self.name, query_vector, top_k, filters)

    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID.

        Args:
            ids: List of vector IDs to delete

        Returns:
            Number of vectors deleted
        """
        return await self._db._delete_vectors(self.name, ids)

    async def count(self) -> int:
        """Get number of vectors in collection."""
        stats = await self._db._get_collection_stats(self.name)
        return stats.get("vector_count", 0)

    @property
    def has_embedding_model(self) -> bool:
        """Check if collection has an embedding model configured."""
        return self._embedding_model is not None


class EmbeddedProximaDB:
    """Embedded ProximaDB instance.

    Runs ProximaDB as a subprocess, providing an in-process-like experience
    with full persistence and durability.

    Example:
        db = EmbeddedProximaDB(data_dir="~/.victor/proximadb")
        await db.start()

        collection = await db.create_collection("codebase", dimension=384)
        await collection.insert([...])
        results = await collection.search(query_vector)

        await db.stop()
    """

    def __init__(
        self,
        data_dir: str | None = None,
        config: EmbeddedConfig | None = None,
        binary_path: str | None = None,
    ):
        """Initialize embedded database.

        Args:
            data_dir: Directory for persistent storage (default: ~/.proximadb)
            config: Full configuration (overrides data_dir if provided)
            binary_path: Path to proximadb-server binary (auto-detected if None)
        """
        self.config = config or EmbeddedConfig(
            data_dir=data_dir or str(Path.home() / ".proximadb")
        )
        self._binary_path = binary_path
        self._process: subprocess.Popen | None = None
        # One pooled HTTP client per server instance (see _http_client).
        self._shared_http_client = None
        self._started = False
        self._collections: dict[str, EmbeddedCollection] = {}
        self._lock = threading.Lock()
        self._instance_id = uuid.uuid4().hex

        # Resolve data directory
        self._data_dir = Path(self.config.data_dir).expanduser()
        self._config_path = self._data_dir / f"embedded-{self._instance_id}.toml"
        self._server_log_path = (
            self._data_dir / f"embedded-server-{self._instance_id}.log"
        )
        self._socket_dir = self._resolve_socket_dir()

    @property
    def rest_url(self) -> str:
        """REST API URL."""
        if self._is_uds_transport:
            return "http://localhost"
        return f"http://localhost:{self.config.rest_port}"

    @property
    def grpc_url(self) -> str:
        """gRPC URL."""
        if self._is_uds_transport:
            return f"unix://{self.grpc_socket_path}"
        return f"localhost:{self.config.grpc_port}"

    @property
    def arrow_flight_url(self) -> str:
        """Arrow Flight URL. Uses grpc+unix:// for UDS (pyarrow), grpc:// for TCP."""
        if self._is_uds_transport:
            # pyarrow flight.Location.for_grpc_unix() parses grpc+unix://PATH
            return f"grpc+unix://{self.arrow_flight_socket_path}"
        return f"grpc://localhost:{self.config.arrow_flight_port}"

    @property
    def _is_uds_transport(self) -> bool:
        return self.config.transport.lower() == "uds"

    @property
    def socket_dir(self) -> Path | None:
        return self._socket_dir

    @property
    def rest_socket_path(self) -> Path:
        if self._socket_dir is None:
            raise RuntimeError(
                "REST socket path is only available in UDS transport mode"
            )
        return self._socket_dir / UDS_REST_SOCKET_NAME

    @property
    def grpc_socket_path(self) -> Path:
        if self._socket_dir is None:
            raise RuntimeError(
                "gRPC socket path is only available in UDS transport mode"
            )
        return self._socket_dir / UDS_GRPC_SOCKET_NAME

    @property
    def arrow_flight_socket_path(self) -> Path:
        if self._socket_dir is None:
            raise RuntimeError(
                "Arrow Flight socket path is only available in UDS transport mode"
            )
        return self._socket_dir / UDS_FLIGHT_SOCKET_NAME

    @staticmethod
    def _toml_string(value: str | Path) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _uds_paths_fit(socket_dir: Path) -> bool:
        return all(
            len(str(socket_dir / name).encode("utf-8")) < UDS_SOCKET_PATH_LIMIT
            for name in (
                UDS_REST_SOCKET_NAME,
                UDS_GRPC_SOCKET_NAME,
                UDS_FLIGHT_SOCKET_NAME,
            )
        )

    def _fallback_socket_dir(self) -> Path:
        for root in (Path(tempfile.gettempdir()), Path("/tmp")):
            candidate = root / f"pxdb-{uuid.uuid4().hex[:12]}"
            if self._uds_paths_fit(candidate):
                return candidate
        raise RuntimeError(
            "Unable to derive a Unix-domain socket path under the OS path limit"
        )

    def _resolve_socket_dir(self) -> Path | None:
        if not self._is_uds_transport:
            return None
        if os.name != "posix":
            raise RuntimeError("Embedded UDS transport requires a POSIX platform")

        if self.config.socket_dir:
            socket_dir = Path(self.config.socket_dir).expanduser()
        else:
            socket_dir = self._data_dir / "sockets"

        if self._uds_paths_fit(socket_dir):
            return socket_dir
        return self._fallback_socket_dir()

    def _http_client(self, **kwargs):
        """Hand out the shared pooled client wrapped in a non-closing lease.

        Every call site does ``async with self._http_client() as client:``,
        and this used to construct a fresh ``httpx.AsyncClient`` — transport,
        connection pool, and all — per operation, then tear it down. Client
        construction plus a cold UDS connection measured **~13 ms per call**,
        which was the single largest component of the 35 ms embedded vector
        search (the server answered the same query in 2-13 ms). A search that
        LanceDB serves in 4 ms spent three of those milliseconds' worth of
        budget three times over just building HTTP clients.

        The lease keeps every call site unchanged: ``__aenter__`` returns the
        shared client, ``__aexit__`` does NOT close it. The shared client is
        created lazily and closed in :meth:`stop`. Explicit ``kwargs`` still
        get a dedicated, caller-owned client (old semantics) — the only such
        caller is the startup health poll.
        """
        import httpx

        if kwargs:
            if self._is_uds_transport:
                return httpx.AsyncClient(
                    transport=httpx.AsyncHTTPTransport(uds=str(self.rest_socket_path)),
                    **kwargs,
                )
            return httpx.AsyncClient(**kwargs)

        if self._shared_http_client is None or self._shared_http_client.is_closed:
            if self._is_uds_transport:
                self._shared_http_client = httpx.AsyncClient(
                    transport=httpx.AsyncHTTPTransport(uds=str(self.rest_socket_path)),
                )
            else:
                self._shared_http_client = httpx.AsyncClient()
        return _PooledClientLease(self._shared_http_client)

    def _sync_http_client(self, **kwargs):
        import httpx

        if self._is_uds_transport:
            return httpx.Client(
                transport=httpx.HTTPTransport(uds=str(self.rest_socket_path)),
                **kwargs,
            )
        return httpx.Client(**kwargs)

    def _find_binary(self) -> str:
        """Find proximadb-server binary."""
        if self._binary_path:
            return self._binary_path

        # Check common locations
        search_paths = [
            # Installed via pip (future)
            Path(sys.prefix) / "bin" / "proximadb-server",
            # Development build
            Path(__file__).parent.parent.parent.parent.parent
            / "target"
            / "release"
            / "proximadb-server",
            Path(__file__).parent.parent.parent.parent.parent
            / "target"
            / "debug"
            / "proximadb-server",
            # System PATH
            "proximadb-server",
        ]

        for path in search_paths:
            if isinstance(path, Path):
                if path.exists():
                    return str(path)
            else:
                # Check PATH
                import shutil

                if shutil.which(path):
                    return path

        # Debug logging

        print(f"[DEBUG] __file__ = {__file__}")
        print("[DEBUG] Search paths checked:")
        for i, path in enumerate(search_paths):
            if isinstance(path, Path):
                print(f"  {i}. {path} (exists: {path.exists()})")
            else:
                print(f"  {i}. {path} (in PATH: {shutil.which(path) is not None})")

        raise RuntimeError(
            "proximadb-server binary not found. "
            "Please build with `cargo build --release` or install via pip."
        )

    def _generate_config(self) -> str:
        """Generate TOML config for embedded mode."""
        self._data_dir.mkdir(parents=True, exist_ok=True)

        config_content = f"""# ProximaDB Embedded Mode Configuration
# Auto-generated - do not edit manually

[server]
node_id = "embedded-{self._instance_id}"
bind_address = "127.0.0.1"
port = {self.config.rest_port}
data_dir = "{self._toml_string(self._data_dir)}"

[storage]
metadata_url = "file://{self._toml_string(self._data_dir)}/metadata"
mmap_enabled = true

[[storage.storage_locations]]
url = "file://{self._toml_string(self._data_dir)}/data"
weight = 1
tags = ["embedded"]

[storage.wal_config]
enable_wal = true
memory_flush_size_bytes = {self.config.memory_flush_size_mb * 1024 * 1024}
# Embedded flush profile: the server's default 128MB predicted-segment floor
# (TD-FLUSH-3) targets object-store GET economics; embedded is local mmap,
# where the config docs themselves endorse a lower floor. 8MB keeps the
# anti-tiny-segment floor while letting small sessions actually flush (and
# with it reclaim WAL); 60s bounds the final at-shutdown flush to at most
# one minute of ingest.
flush_floor_predicted_mb = 8
flush_interval_secs = 60

[storage.sst_config]
cache_size_mb = {self.config.cache_size_mb}
compression = "lz4"
compression_level = 3

[api]
grpc_port = {self.config.grpc_port}
rest_port = {self.config.rest_port}
arrow_flight_port = {self.config.arrow_flight_port}
transport = "{self.config.transport.lower()}"
max_request_size_mb = 32
timeout_seconds = 30
"""
        if self._socket_dir is not None:
            config_content += f'socket_dir = "{self._toml_string(self._socket_dir)}"\n'

        config_content += f"""

[monitoring]
metrics_enabled = false
log_level = "{self.config.log_level}"

[graph]
engine = "{self.config.graph_engine}"
enable_prefetch = true
prefetch_budget = 4
"""
        self._config_path.write_text(config_content)
        return str(self._config_path)

    async def start(self, timeout: float | None = None) -> None:
        """Start the embedded database.

        Args:
            timeout: Maximum time to wait for startup in seconds
        """
        if self._started:
            return

        with self._lock:
            if self._started:
                return

            # Ownership check BEFORE spawning. A server spawned with setsid
            # outlives its parent and keeps writing its data dir, so a second
            # spawn on the same dir yields two writers and silently
            # resurrects state a caller believed it had cleared
            # (anvai-labs/victor#911). A STALE record (owner dead or frozen)
            # is reclaimed automatically; a LIVE one is reported, never
            # ignored.
            owner = read_runtime_state(self._data_dir)
            if owner is not None:
                if runtime_state_is_stale(owner, time.time() * 1000):
                    logger.info(
                        "Reclaiming data dir from a dead ProximaDB owner "
                        "(pid=%s, phase=%s)",
                        owner.get("pid"),
                        owner.get("phase"),
                    )
                    try:
                        (self._data_dir / RUNTIME_STATE_FILE).unlink()
                    except OSError:
                        pass
                elif self.config.fail_on_foreign_owner:
                    raise RuntimeError(
                        "Another ProximaDB server is already serving "
                        f"{self._data_dir} (pid={owner.get('pid')}, "
                        f"phase={owner.get('phase')}). Stop it first, or set "
                        "EmbeddedConfig.fail_on_foreign_owner=False to run a "
                        "second writer deliberately."
                    )
                else:
                    logger.warning(
                        "Spawning a SECOND writer on %s (existing pid=%s) — "
                        "concurrent writers can corrupt or resurrect state",
                        self._data_dir,
                        owner.get("pid"),
                    )

            # Find binary
            binary = self._find_binary()

            # Generate config
            config_path = self._generate_config()

            # Start process. Graph WAL retention opt-ins: the canonical
            # replay scope turns on snapshot+truncate in flush_wal, and the
            # 8MB segment size lets truncation actually reclaim (whole
            # segments below the checkpoint marker are deleted; an embedded
            # graph WAL rarely reaches the 64MB server default, which would
            # reclaim nothing). An explicit value in the caller's environment
            # wins.
            child_env = dict(os.environ)
            child_env.setdefault("PROXIMADB_GRAPH_CANONICAL_REPLAY_SCOPE", "1")
            child_env.setdefault("PROXIMADB_GRAPH_WAL_SEGMENT_MB", "8")
            child_env.setdefault(
                "PROXIMADB_CORPUS_VERSION_PATH",
                str(self._data_dir / "corpus-versions.json"),
            )
            # Own the child's lifetime. It is spawned setsid (its own process
            # group) so it survives our exit — which is how orphaned servers
            # accumulated and kept rewriting data dirs callers believed they
            # had cleared (anvai-labs/victor#911). Registering teardown at
            # spawn time makes ownership symmetric with the runtime-state
            # record the server writes.
            _register_owned_server(self)
            self._data_dir.mkdir(parents=True, exist_ok=True)
            # PIPEs without a reader eventually fill and stop the server. Keep
            # launch-specific diagnostics in a file instead; this also prevents
            # concurrent launch attempts from mixing output.
            with self._server_log_path.open("ab") as server_log:
                self._process = subprocess.Popen(
                    [binary, "--config", config_path],
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    # The server's tracing appender uses a relative `logs/`
                    # directory. Embedded mode owns the child environment, so
                    # never let that path depend on the caller's cwd (which may
                    # be read-only, deleted, or shared with another instance).
                    cwd=str(self._data_dir),
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                    env=child_env,
                )

            # Wait for readiness. The server publishes a lifecycle record
            # (pid + phase + heartbeat) BEFORE recovery, precisely because
            # listeners bind only after recovery finishes: polling HTTP alone
            # cannot tell "replaying a 1.3GB WAL" from "crashed", so a fixed
            # deadline kills healthy servers (proximaDB#1667). We therefore
            # wait as long as the server proves progress, bounded by an
            # absolute safety ceiling.
            # An EXPLICIT caller timeout wins — callers that pass one mean it
            # (tests, probes, callers with their own deadline). Only the
            # implicit case gets the generous config ceiling, because that is
            # the case that used to hard-code 30s and kill healthy servers.
            ceiling = (
                float(timeout)
                if timeout is not None
                else float(self.config.startup_timeout_s)
            )
            start_time = time.time()
            last_phase: str | None = None
            last_heartbeat: float = -1.0
            last_progress_at = start_time

            while time.time() - start_time < ceiling:
                exit_code = self._process.poll() if self._process else None
                if exit_code is not None:
                    log_tail = ""
                    try:
                        log_tail = self._server_log_path.read_text(
                            encoding="utf-8", errors="replace"
                        )[-2000:]
                    except Exception:
                        pass
                    self._process = None
                    self._remove_generated_config()
                    raise RuntimeError(
                        "ProximaDB server exited during startup with status "
                        f"{exit_code}. Server log: {self._server_log_path}. {log_tail}"
                    )

                state = read_runtime_state(self._data_dir)
                try:
                    with self._sync_http_client(timeout=1.0) as client:
                        response = client.get(f"{self.rest_url}/health")
                    if response.status_code == 200 and self._state_belongs_to_child(
                        state
                    ):
                        self._started = True
                        self._remove_generated_config()
                        return
                except Exception:
                    pass

                # No socket yet — consult the lifecycle record.
                if state is not None:
                    phase = state.get("phase")
                    heartbeat = float(state.get("updated_at_ms") or 0)
                    if phase != last_phase or heartbeat > last_heartbeat:
                        # Progress: the server is alive and working.
                        last_phase, last_heartbeat = phase, heartbeat
                        last_progress_at = time.time()
                        logger.debug(
                            "ProximaDB starting: phase=%s (waiting; ceiling %.0fs)",
                            phase,
                            ceiling,
                        )
                    elif runtime_state_is_stale(state, time.time() * 1000):
                        self._kill_process()
                        raise RuntimeError(
                            "ProximaDB stopped making progress during startup "
                            f"(phase={phase}, heartbeat frozen for more than "
                            f"{_STALE_AFTER_INTERVALS} beats)"
                        )
                await asyncio.sleep(0.2)

            # Ceiling reached — report the phase we last saw, so the failure
            # says what the server was doing rather than just "didn't start".
            self._kill_process()
            raise TimeoutError(
                f"ProximaDB failed to start within {ceiling:.0f}s "
                f"(last phase: {last_phase or 'unknown'}, "
                f"stalled {time.time() - last_progress_at:.0f}s). "
                f"Server log: {self._server_log_path}"
            )

    def _state_belongs_to_child(self, state: dict[str, Any] | None) -> bool:
        """Prove the healthy endpoint is the subprocess this object owns."""
        if self._process is None or state is None:
            return False
        try:
            return int(state.get("pid")) == self._process.pid
        except (TypeError, ValueError):
            return False

    def _remove_generated_config(self) -> None:
        """Remove launch-only configuration without masking the real result."""
        try:
            self._config_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("could not remove generated config", exc_info=True)

    def _kill_process(self) -> None:
        """Kill the server process."""
        if self._process:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                else:
                    self._process.terminate()
                self._process.wait(timeout=self.config.shutdown_timeout_s)
            except Exception:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                else:
                    self._process.kill()
            self._process = None

        self._remove_generated_config()

        # Cleanup: remove the fallback tmpdir (pxdb-*) when we created one.
        # The data_dir/sockets/ path is left behind for reuse/restart.
        if self._socket_dir is not None and self._socket_dir.parent.name.startswith(
            "pxdb-"
        ):
            import shutil

            try:
                shutil.rmtree(self._socket_dir.parent)
            except Exception:
                pass  # Best-effort

    async def stop(self) -> None:
        """Stop the embedded database gracefully."""
        if not self._started:
            return

        with self._lock:
            if not self._started:
                return

            self._kill_process()
            self._started = False
            self._collections.clear()

        # Close the pooled HTTP client outside the (sync) lock.
        client, self._shared_http_client = self._shared_http_client, None
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except Exception:  # pragma: no cover - teardown best-effort
                logger.debug("pooled http client close failed", exc_info=True)

    async def create_collection(
        self,
        name: str,
        dimension: int | None = None,
        distance_metric: str = "cosine",
        embedding_model: BaseEmbeddingModel | str | None = None,
        engine: str | None = None,
    ) -> EmbeddedCollection:
        """Create a new collection.

        Args:
            name: Collection name
            dimension: Vector dimension (auto-detected from embedding model if not provided)
            distance_metric: Distance metric (cosine, euclidean, dot)
            engine: Storage engine ("sst", "helix", "viper", "auto"). Defaults to
                ``EmbeddedConfig.vector_engine``. Previously never sent, so the
                server silently applied "auto".
            embedding_model: Embedding model for auto-embedding. Can be:
                - BaseEmbeddingModel instance
                - String model name (uses sentence-transformers)
                - None (no auto-embedding, use raw vectors)

        Returns:
            Collection object for further operations

        Example:
            # With embedding model (recommended)
            collection = await db.create_collection(
                "code_symbols",
                embedding_model="BAAI/bge-small-en-v1.5"
            )

            # With custom embedding model
            model = OllamaEmbeddingModel("nomic-embed-text")
            collection = await db.create_collection("docs", embedding_model=model)

            # Raw vectors only (no auto-embedding)
            collection = await db.create_collection("vectors", dimension=384)
        """
        if not self._started:
            await self.start()

        # Handle embedding model
        model_instance: BaseEmbeddingModel | None = None
        if isinstance(embedding_model, str):
            # Create sentence-transformers model from name
            model_instance = SentenceTransformerModel(model_name=embedding_model)
        elif isinstance(embedding_model, BaseEmbeddingModel):
            model_instance = embedding_model

        # Auto-detect dimension from embedding model
        if dimension is None:
            if model_instance:
                dimension = model_instance.get_dimension()
            else:
                raise ValueError(
                    "dimension is required when not using an embedding model. "
                    "Either provide dimension or embedding_model."
                )

        # The canonical v2 endpoint takes a flat body with a STRING distance
        # metric and returns the created collection record (no {success} envelope).
        # An already-existing collection responds 409/CONFLICT, which is benign.
        # `EmbeddedConfig.vector_engine` existed but was never transmitted, so the
        # server fell back to `engine = "auto"` and every embedded collection was
        # created on HELIX regardless of what the caller configured. That is not a
        # cosmetic default: engine choice gates real capability (for example
        # `object_economy_eligible` requires "sst"), and PAX-based features such as
        # the filter-aware cascade only exist on the SST path. Send it.
        selected_engine = engine or self.config.vector_engine
        async with self._http_client() as client:
            response = await client.post(
                f"{self.rest_url}/api/v2/collections",
                json={
                    "name": name,
                    "dimension": dimension,
                    "engine": selected_engine.lower(),
                    "distance_metric": distance_metric.lower(),
                },
                timeout=30.0,
            )

            if response.status_code not in (200, 201, 409):
                raise RuntimeError(
                    f"Failed to create collection: HTTP {response.status_code} "
                    f"{response.text[:300]}"
                )

        collection = EmbeddedCollection(
            name, dimension, self, embedding_model=model_instance
        )
        self._collections[name] = collection
        return collection

    async def get_collection(self, name: str) -> EmbeddedCollection | None:
        """Get an existing collection.

        Args:
            name: Collection name

        Returns:
            Collection object or None if not found
        """
        if not self._started:
            await self.start()

        if name in self._collections:
            return self._collections[name]

        # Resolved from the LIST endpoint rather than GET /collections/{name},
        # for two reasons:
        #
        # 1. GET returns 200 for a collection that does not exist, with
        #    dimension 0 and a created_at equal to the request instant. Trusting
        #    it hands back a phantom handle that silently writes nowhere.
        #    LIST reports only collections that actually exist.
        # 2. LIST already carries the dimension, so this needs one request, not
        #    two (GET to fetch + LIST to verify).
        for entry in await self._collection_entries():
            if _collection_field(entry, "name") == name:
                dim = _collection_field(entry, "dimension") or 0
                collection = EmbeddedCollection(name, int(dim), self)
                self._collections[name] = collection
                return collection

        return None

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if deleted successfully
        """
        if not self._started:
            await self.start()

        async with self._http_client() as client:
            response = await client.delete(
                f"{self.rest_url}/api/v2/collections/{name}",
                timeout=30.0,
            )

            if name in self._collections:
                del self._collections[name]

            return response.status_code in (200, 204, 404)

    async def _collection_entries(self) -> list[dict[str, Any]]:
        """Return the raw collection descriptors from the LIST endpoint."""
        if not self._started:
            await self.start()

        async with self._http_client() as client:
            response = await client.get(
                f"{self.rest_url}/api/v2/collections",
                timeout=10.0,
            )
            if response.status_code != 200:
                return []
            data = response.json()
            entries = data.get("collections", [])
            return [e for e in entries if isinstance(e, dict)]

    async def list_collections(self) -> list[str]:
        """List all collection names."""
        names = [_collection_field(e, "name") for e in await self._collection_entries()]
        return [n for n in names if n]

    def _to_sql_value(self, value: Any) -> dict[str, Any]:
        """Convert Python value to SqlValue format."""
        if value is None:
            return {"null_value": None}
        if isinstance(value, bool):
            return {"bool_value": value}
        if isinstance(value, int) and not isinstance(value, bool):
            return {"int64_value": value}
        if isinstance(value, float):
            return {"number_value": value}
        if isinstance(value, str):
            return {"string_value": value}
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {"bytes_value": base64.b64encode(bytes(value)).decode("ascii")}
        if isinstance(value, (list, tuple)):
            return {"array_value": {"values": [self._to_sql_value(v) for v in value]}}
        if isinstance(value, dict):
            return {
                "object_value": {
                    "fields": {str(k): self._to_sql_value(v) for k, v in value.items()}
                }
            }
        return {"string_value": str(value)}

    def _convert_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert metadata dict to SqlValue format."""
        return {str(k): self._to_sql_value(v) for k, v in metadata.items()}

    def _to_proxima_value(self, value: Any) -> Any:
        """Convert Python values to v2 REST ProximaValue JSON."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {
                "type": "binary",
                "value": base64.b64encode(bytes(value)).decode("ascii"),
            }
        if isinstance(value, (list, tuple)):
            return {
                "type": "array",
                "value": [self._to_proxima_value(v) for v in value],
            }
        if isinstance(value, dict):
            if set(value.keys()) == {"type", "value"}:
                return value
            return {
                "type": "jsonb",
                "value": {str(k): self._to_proxima_value(v) for k, v in value.items()},
            }
        return str(value)

    def _normalize_record_payload(
        self, record: dict[str, Any], index: int = 0
    ) -> dict[str, Any]:
        """Normalize embedded inputs to the v2 ProximaRecord REST shape."""
        vector = record.get("vector")
        if vector is None:
            raise ValueError("record is missing vector")

        props = {}
        for source in ("props", "metadata", "flexible_fields"):
            values = record.get(source)
            if isinstance(values, dict):
                props.update(
                    {str(k): self._to_proxima_value(v) for k, v in values.items()}
                )

        typed_fields = record.get("typed_fields")
        if isinstance(typed_fields, dict):
            for key, typed_value in typed_fields.items():
                if hasattr(typed_value, "model_dump"):
                    typed_value = typed_value.model_dump(exclude_none=True)
                if isinstance(typed_value, dict) and "value" in typed_value:
                    props[str(key)] = {
                        "type": typed_value.get("value_type")
                        or typed_value.get("type"),
                        "value": typed_value["value"],
                    }
                else:
                    props[str(key)] = self._to_proxima_value(typed_value)

        payload = {
            "id": record.get("id") or record.get("oid") or f"record_{index}",
            "vector": [float(v) for v in vector],
            "props": props,
        }
        if record.get("text_fields"):
            payload["text_fields"] = record["text_fields"]
        return payload

    async def _insert_records(
        self,
        collection_name: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Insert ProximaRecord-shaped records into a collection."""
        formatted = [
            self._normalize_record_payload(record, index)
            for index, record in enumerate(records)
        ]

        async with self._http_client() as client:
            response = await client.post(
                f"{self.rest_url}/api/v2/collections/{collection_name}/records/batch",
                json={
                    "records": formatted,
                    "validate_schema": True,
                },
                timeout=60.0,
            )
            # Surface insert failures (404 unknown collection, 422 bad payload)
            # rather than returning an error body the caller would ignore — a
            # silently-dropped vector would later look like a recall miss.
            response.raise_for_status()
            return response.json()

    async def _insert_vectors(
        self,
        collection_name: str,
        vectors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compatibility alias for record-native insert."""
        return await self._insert_records(collection_name, vectors)

    async def _search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Posts the canonical v2 ``TypedSearchRequest`` (a flat
        ``{vector, top_k, filters}`` body — NOT a ``{queries:[...]}`` wrapper)
        to ``/api/v2/collections/{name}/search`` and returns the flat
        ``results`` list (each item a dict with ``id``/``score``/``props``).
        """
        body: dict[str, Any] = {
            "vector": [float(v) for v in query_vector],
            "top_k": top_k,
        }
        if filters:
            body["filters"] = self._to_typed_filters(filters)

        async with self._http_client() as client:
            response = await client.post(
                f"{self.rest_url}/api/v2/collections/{collection_name}/search",
                json=body,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        # v2 TypedSearchResponse: {"results": [...], "total_matches", "latency_ms"}.
        return data.get("results") or []

    @staticmethod
    def _to_typed_filters(
        filters: dict[str, Any] | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Map a simple ``{field: value}`` dict to v2 typed filters.

        The v2 search endpoint takes ``[{field, op, value}]`` (TypedFilter), not
        a flat metadata map. Equality is assumed for scalar values; callers that
        need range/in operators should pass already-typed filters (a list), which
        are forwarded unchanged.
        """
        if isinstance(filters, list):
            return filters
        return [
            {"field": str(field), "op": "eq", "value": value}
            for field, value in filters.items()
        ]

    async def _delete_vectors(
        self,
        collection_name: str,
        ids: list[str],
    ) -> int:
        """Delete records by id via the canonical v2 per-record endpoint.

        Issues ``DELETE /api/v2/collections/{name}/records/{id}`` for each id and
        returns the count actually deleted (a 404 for an absent id is treated as
        "nothing to delete", not an error — delete is idempotent).
        """
        if not ids:
            return 0

        deleted = 0
        from urllib.parse import quote

        async with self._http_client() as client:
            for record_id in ids:
                # Record ids are caller-defined and may contain characters
                # illegal or ambiguous in a URL path (newlines from
                # signature-derived ids, '/', '%', spaces). Encode the path
                # component; axum percent-decodes it back to the exact id.
                response = await client.delete(
                    f"{self.rest_url}/api/v2/collections/{collection_name}"
                    f"/records/{quote(record_id, safe='')}",
                    timeout=30.0,
                )
                if response.status_code in (200, 204):
                    deleted += 1
                elif response.status_code == 404:
                    continue
                else:
                    response.raise_for_status()
        return deleted

    async def _get_collection_stats(
        self,
        collection_name: str,
    ) -> dict[str, Any]:
        """Get collection statistics."""
        async with self._http_client() as client:
            response = await client.get(
                f"{self.rest_url}/api/v2/collections/{collection_name}",
                timeout=10.0,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("collection"):
                    return data["collection"].get("stats", {})

            return {}

    async def health_check(self) -> bool:
        """Check if database is healthy.

        Returns:
            True if healthy
        """
        if not self._started:
            return False

        try:
            async with self._http_client() as client:
                response = await client.get(
                    f"{self.rest_url}/health",
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception as exc:
            # Unhealthy is the correct answer here. Logging is what separates
            # "the server said no" from "the check itself could not run".
            logger.debug("health check failed: %s", exc)
            return False

    # =============================================================================
    # Multi-Model API: Document API
    # =============================================================================

    async def create_document_collection(
        self,
        name: str,
        indexes: list[dict[str, Any]] | None = None,
        enable_fulltext: bool = False,
        fulltext_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a document collection.

        Args:
            name: Collection name
            indexes: List of index definitions (path, type)
            enable_fulltext: Enable full-text search
            fulltext_paths: JSON paths for full-text indexing

        Returns:
            Creation result with collection_id
        """
        if not self._started:
            await self.start()

        payload = {
            "name": name,
            "indexes": indexes or [],
            "enable_fulltext": enable_fulltext,
        }
        if fulltext_paths:
            payload["fulltext_paths"] = fulltext_paths

        async with self._http_client() as client:
            response = await client.post(
                f"{self.rest_url}/api/v2/document-collections",
                json=payload,
                timeout=30.0,
            )
            return response.json()

    async def get_document(
        self,
        collection_name: str,
        doc_id: str,
    ) -> dict[str, Any] | None:
        """Get a document by ID.

        Args:
            collection_name: Collection name
            doc_id: Document ID

        Returns:
            Document data or None if not found
        """
        if not self._started:
            await self.start()

        async with self._http_client() as client:
            response = await client.get(
                f"{self.rest_url}/api/v2/document-collections/{collection_name}/documents/{doc_id}",
                timeout=10.0,
            )

            if response.status_code == 200:
                return response.json()
            return None

    async def query_documents(
        self,
        collection_name: str,
        filter: dict[str, Any] | None = None,
        projection: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Query documents in a collection.

        Args:
            collection_name: Collection name
            filter: Filter criteria (dict)
            projection: Fields to return
            limit: Max results
            offset: Result offset

        Returns:
            Query results with documents list
        """
        if not self._started:
            await self.start()

        params: dict[str, Any] = {"limit": limit}
        if offset:
            params["skip"] = offset
        if filter:
            params["filter"] = filter if isinstance(filter, str) else str(filter)
        if projection:
            params["projection"] = ",".join(projection)

        async with self._http_client() as client:
            response = await client.get(
                f"{self.rest_url}/api/v2/document-collections/{collection_name}/documents",
                params=params,
                timeout=30.0,
            )
            return response.json()

    async def update_document(
        self,
        collection_name: str,
        doc_id: str,
        updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Update a document.

        Args:
            collection_name: Collection name
            doc_id: Document ID
            updates: List of update operations (SET, PUSH, PULL, etc.)

        Returns:
            Update result with new version
        """
        if not self._started:
            await self.start()

        payload = {"updates": updates}

        async with self._http_client() as client:
            response = await client.patch(
                f"{self.rest_url}/api/v2/document-collections/{collection_name}/documents/{doc_id}",
                json=payload,
                timeout=30.0,
            )
            return response.json()

    async def delete_document(
        self,
        collection_name: str,
        doc_id: str,
    ) -> bool:
        """Delete a document.

        Args:
            collection_name: Collection name
            doc_id: Document ID

        Returns:
            True if deleted successfully
        """
        if not self._started:
            await self.start()

        async with self._http_client() as client:
            response = await client.delete(
                f"{self.rest_url}/api/v2/document-collections/{collection_name}/documents/{doc_id}",
                timeout=30.0,
            )
            return response.status_code in (200, 204)

    async def delete_document_collection(
        self,
        collection_name: str,
    ) -> bool:
        """Delete a document collection.

        Args:
            collection_name: Collection name

        Returns:
            True if deleted successfully
        """
        if not self._started:
            await self.start()

        async with self._http_client() as client:
            response = await client.delete(
                f"{self.rest_url}/api/v2/document-collections/{collection_name}",
                timeout=30.0,
            )
            return response.status_code in (200, 204)

    # =============================================================================
    # Multi-Model API: Time Series API
    # =============================================================================

    async def create_timeseries_collection(
        self,
        name: str,
        timestamp_column: str = "timestamp",
        value_columns: list[dict[str, Any]] | None = None,
        tag_columns: list[str] | None = None,
        retention_ms: int | None = None,
    ) -> dict[str, Any]:
        """Create a time-series collection.

        Args:
            name: Collection name
            timestamp_column: Name of timestamp column
            value_columns: List of value column definitions (name, data_type, aggregation)
            tag_columns: List of tag column names
            retention_ms: Data retention period in milliseconds

        Returns:
            Creation result with collection_id
        """
        if not self._started:
            await self.start()

        payload = {
            "name": name,
            "timestamp_column": timestamp_column,
            "value_columns": value_columns or [],
            "tag_columns": tag_columns or [],
        }
        if retention_ms:
            payload["retention_ms"] = retention_ms

        async with self._http_client() as client:
            response = await client.post(
                f"{self.rest_url}/api/v2/timeseries/collections",
                json=payload,
                timeout=30.0,
            )
            return response.json()

    async def ingest_timeseries(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ingest time-series data points.

        Args:
            collection_name: Collection name
            points: List of data points with timestamp, values, and tags

        Returns:
            Ingest result with counts
        """
        if not self._started:
            await self.start()

        payload = {
            "collection_id": collection_name,
            "points": points,
        }

        async with self._http_client() as client:
            response = await client.post(
                f"{self.rest_url}/api/v2/timeseries/{collection_name}/ingest",
                json=payload,
                timeout=60.0,
            )
            return response.json()

    async def query_timeseries(
        self,
        collection_name: str,
        start_time: str,
        end_time: str,
        aggregation: str | None = None,
        bucket_ms: int | None = None,
        tag_filters: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Query time-series data.

        Args:
            collection_name: Collection name
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            aggregation: Aggregation type (AVG, SUM, MIN, MAX, COUNT, OHLC)
            bucket_ms: Bucket size in milliseconds for aggregation
            tag_filters: Tag filters
            limit: Max results

        Returns:
            Query results with metrics/raw_points
        """
        if not self._started:
            await self.start()

        payload = {
            "collection_id": collection_name,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit,
        }
        if aggregation:
            payload["aggregation"] = aggregation
        if bucket_ms:
            payload["bucket_ms"] = bucket_ms
        if tag_filters:
            payload["tag_filters"] = tag_filters

        async with self._http_client() as client:
            response = await client.post(
                f"{self.rest_url}/api/v2/timeseries/{collection_name}/query",
                json=payload,
                timeout=30.0,
            )
            return response.json()

    async def aggregate_timeseries(
        self,
        collection_name: str,
        start_time: str,
        end_time: str,
        pipeline: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate time-series data with a pipeline.

        Args:
            collection_name: Collection name
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            pipeline: Aggregation pipeline stages

        Returns:
            Aggregation results
        """
        if not self._started:
            await self.start()

        payload = {
            "collection_id": collection_name,
            "start_time": start_time,
            "end_time": end_time,
            "pipeline": pipeline,
        }

        async with self._http_client() as client:
            response = await client.post(
                f"{self.rest_url}/api/v2/timeseries/{collection_name}/aggregate",
                json=payload,
                timeout=60.0,
            )
            return response.json()

    async def delete_timeseries_collection(
        self,
        collection_name: str,
    ) -> bool:
        """Delete a time-series collection.

        Args:
            collection_name: Collection name

        Returns:
            True if deleted successfully
        """
        if not self._started:
            await self.start()

        async with self._http_client() as client:
            response = await client.delete(
                f"{self.rest_url}/api/v2/timeseries/collections/{collection_name}",
                timeout=30.0,
            )
            return response.status_code in (200, 204)

    # =============================================================================
    # Multi-Model API: Hybrid Search API
    # =============================================================================

    async def hybrid_search(
        self,
        vector_collection: str,
        query_vector: list[float],
        text_query: str | None = None,
        fusion_strategy: str = "rrf",
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        fusion_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform hybrid search combining vector and text search.

        Args:
            vector_collection: Vector collection name
            query_vector: Query embedding vector
            text_query: Optional text query for BM25
            fusion_strategy: Fusion strategy (rrf, weighted_linear, cascade, etc.)
            top_k: Number of results
            filters: Optional metadata filters
            fusion_params: Optional fusion strategy parameters

        Returns:
            Hybrid search results with fused scores
        """
        if not self._started:
            await self.start()

        payload = {
            "vector_collection": vector_collection,
            "query_vector": query_vector,
            "fusion_strategy": fusion_strategy,
            "top_k": top_k,
        }
        if text_query:
            payload["text_query"] = text_query
        if filters:
            payload["filters"] = self._convert_metadata(filters)
        if fusion_params:
            payload["fusion_params"] = fusion_params

        async with self._http_client() as client:
            response = await client.post(
                f"{self.rest_url}/api/v2/hybrid/search",
                json=payload,
                timeout=30.0,
            )
            return response.json()

    async def list_fusion_strategies(self) -> list[dict[str, Any]]:
        """List available fusion strategies for hybrid search.

        Returns:
            List of fusion strategy definitions
        """
        if not self._started:
            await self.start()

        async with self._http_client() as client:
            response = await client.get(
                f"{self.rest_url}/api/v2/hybrid/strategies",
                timeout=10.0,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("strategies", [])

        return []

    async def execute_multi_modal_query(
        self,
        query: "MultiModalQuery",
        rerank_config: Optional["RerankConfig"] = None,
        context: Optional["QueryContext"] = None,
    ) -> "MultiModalQueryResult":
        """Execute a multi-modal query combining vector, graph, and document searches.

        This method allows combining different query types (vector similarity,
        graph traversal, document filtering) into a single unified query with
        result fusion and optional reranking.

        Args:
            query: A MultiModalQuery object built with MultiModalQueryBuilder
            rerank_config: Optional configuration for cross-modal reranking
            context: Optional query context for context-aware scoring

        Returns:
            MultiModalQueryResult with fused results from all query components

        Example:
            from proximadb_sdk.multimodal_query import (
                MultiModalQueryBuilder, FusionStrategy, RerankConfig, QueryContext
            )

            # Build a multi-modal query
            query = (MultiModalQueryBuilder()
                .vector("products", query_embedding, top_k=20)
                .graph("knowledge", start_label="Category", edge_types=["CONTAINS"])
                .fuse(FusionStrategy.RRF)
                .limit(10)
                .build())

            # Execute with optional reranking
            results = await db.execute_multi_modal_query(
                query,
                rerank_config=RerankConfig(diversity_optimization=True)
            )

            for record in results:
                print(f"{record['id']}: {record.get('_rrf_score', 0)}")
        """
        if not self._started:
            await self.start()

        # Import here to avoid circular imports
        from .multimodal_query import CrossModalReranker, MultiModalQueryResult

        # Create embedded executor
        executor = EmbeddedMultiModalQueryExecutor(self)
        result = await executor.execute(query)

        # Apply reranking if configured
        if rerank_config is not None:
            reranker = CrossModalReranker(config=rerank_config)
            reranked = reranker.rerank(result.records, context)
            result = MultiModalQueryResult(
                records=reranked.records,
                total_count=len(reranked.records),
                query_time_ms=result.query_time_ms,
                component_times=result.component_times,
                fusion_strategy=result.fusion_strategy,
                metadata={
                    **result.metadata,
                    "reranked": True,
                    "quality_score": reranked.quality_score,
                    "diversity_score": reranked.diversity_score,
                },
            )

        return result

    def __enter__(self):
        """Context manager entry."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.start())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.stop())

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Convenience function
async def connect_embedded(data_dir: str | None = None, **kwargs) -> EmbeddedProximaDB:
    """Create and start an embedded ProximaDB instance.

    Args:
        data_dir: Directory for persistent storage
        **kwargs: Additional EmbeddedConfig options

    Returns:
        Started EmbeddedProximaDB instance
    """
    config = EmbeddedConfig(
        data_dir=data_dir or str(Path.home() / ".proximadb"), **kwargs
    )
    db = EmbeddedProximaDB(config=config)
    await db.start()
    return db


# =============================================================================
# Embedded Multi-Modal Query Executor
# =============================================================================


class EmbeddedMultiModalQueryExecutor:
    """
    Multi-modal query executor for embedded ProximaDB mode.

    Executes vector, graph, and document queries against an embedded database
    and fuses results using various strategies (RRF, intersection, union, weighted).

    This executor is designed to work with the subprocess-based embedded mode
    that communicates via REST API.
    """

    def __init__(self, db: EmbeddedProximaDB):
        """Initialize with an embedded database instance.

        Args:
            db: EmbeddedProximaDB instance (must be started)
        """
        self._db = db

    async def execute(self, query: "MultiModalQuery") -> "MultiModalQueryResult":
        """Execute a multi-modal query.

        Args:
            query: Compiled MultiModalQuery object

        Returns:
            MultiModalQueryResult with fused results
        """

        from .multimodal_query import MultiModalQueryResult

        start_time = time.time()
        component_times: dict[str, float] = {}
        component_errors: dict[str, str] = {}
        component_results: list[list[dict[str, Any]]] = []

        # Execute each component
        for i, component in enumerate(query.components):
            comp_start = time.time()

            comp_type = component.get("type")
            comp_key = f"{comp_type}_{i}"
            try:
                results = await self._run_component(
                    comp_type, component, component_results
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not discarded
                # Best-effort fusion is intended: one failing leg must not lose
                # the others. What is not acceptable is the caller being unable
                # to tell, so the failure is named on the result.
                logger.warning(
                    "embedded multi-modal component %s failed: %s",
                    comp_key,
                    exc,
                    exc_info=True,
                )
                component_errors[comp_key] = f"{type(exc).__name__}: {exc}"
                results = []

            component_results.append(results)
            component_times[comp_key] = (time.time() - comp_start) * 1000

        return await self._fuse_and_package(
            query, component_results, component_times, component_errors, start_time
        )

    async def _run_component(
        self,
        comp_type: str | None,
        component: dict[str, Any],
        component_results: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Run one leg. Raises; `execute` decides what a failure means."""
        if comp_type == "vector":
            return await self._execute_vector(component)
        elif comp_type == "graph":
            # Check if this depends on previous results
            if component.get("_from_previous") and component_results:
                prev_results = component_results[-1]
                id_field = component.get("_id_field", "id")
                start_nodes = [r.get(id_field) for r in prev_results if r.get(id_field)]
                component["start_nodes"] = start_nodes
            return await self._execute_graph(component)
        elif comp_type == "document":
            return await self._execute_document(component)
        elif comp_type == "logs":
            return await self._execute_logs(component)
        elif comp_type == "metrics":
            return await self._execute_metrics(component)
        else:
            return []

    async def _fuse_and_package(
        self,
        query: "MultiModalQuery",
        component_results: list[list[dict[str, Any]]],
        component_times: dict[str, float],
        component_errors: dict[str, str],
        start_time: float,
    ) -> "MultiModalQueryResult":
        """Join, fuse and package, so `execute` reads as its own policy."""
        from .multimodal_query import MultiModalQueryResult

        # Apply joins if specified
        if query.joins:
            component_results = self._apply_joins(component_results, query.joins)

        # Fuse results
        fused = self._fuse_results(
            component_results,
            query.fusion_strategy,
            query.fusion_weights,
        )

        # Apply time decay if specified
        if query.time_decay:
            fused = self._apply_time_decay(fused, query.time_decay)

        # Apply custom scorer if specified
        if query.custom_scorer:
            for record in fused:
                record["_custom_score"] = query.custom_scorer(record)
            fused.sort(key=lambda x: x.get("_custom_score", 0), reverse=True)

        # Apply limit and offset
        fused = fused[query.offset : query.offset + query.limit]

        total_time = (time.time() - start_time) * 1000

        return MultiModalQueryResult(
            records=fused,
            total_count=len(fused),
            query_time_ms=total_time,
            component_times=component_times,
            fusion_strategy=query.fusion_strategy,
            component_errors=component_errors,
            metadata={
                "component_count": len(query.components),
                "join_count": len(query.joins),
                "executor": "embedded",
            },
        )

    async def _execute_vector(self, component: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute vector search component against embedded database."""
        try:
            collection = component.get("collection", "")
            query_vector = component.get("query_vector", [])
            top_k = component.get("top_k", 10)
            filters = component.get("filter")

            # Use the embedded DB's search method via REST API
            results = await self._db._search_vectors(
                collection,
                query_vector,
                top_k,
                filters,
            )

            # Transform results to standard format
            return [
                {
                    "id": r.get("id", r.get("vector_id", "")),
                    "score": r.get("score", r.get("similarity", 0.0)),
                    "metadata": r.get("metadata", {}),
                    "_source_type": "vector",
                }
                for r in results
            ]
        except Exception as exc:
            # Re-raised, not swallowed. This comment used to say "log error but
            # return empty results" -- and it did not log, so a failed leg was
            # indistinguishable from one that matched nothing. `execute` owns
            # the best-effort policy and records which component failed.
            raise exc

    async def _execute_graph(self, component: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute graph traversal component against embedded database.

        Performs graph traversal by:
        1. Getting start nodes (by label or explicit IDs)
        2. Traversing outgoing edges up to max_depth
        3. Filtering by edge types if specified
        """
        try:
            graph_id = component.get("graph_id", "")
            start_nodes = component.get("start_nodes")
            start_label = component.get("start_label")
            edge_types = component.get("edge_types")
            max_depth = component.get("max_depth", 2)
            limit = component.get("limit", 100)

            results: list[dict[str, Any]] = []

            # Get start nodes
            node_ids: list[str] = []
            if start_nodes:
                node_ids = start_nodes
            elif start_label:
                # Query nodes by label via REST API
                async with self._db._http_client() as client:
                    response = await client.post(
                        f"{self._db.rest_url}/api/v2/graphs/{graph_id}/nodes/query",
                        json={"labels": [start_label]},
                        timeout=30.0,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        nodes = data.get("nodes", [])
                        node_ids = [n.get("id") for n in nodes if n.get("id")]

            if not node_ids:
                return []

            # BFS traversal
            visited = set()
            queue = [(nid, 0) for nid in node_ids[:limit]]  # (node_id, depth)

            async with self._db._http_client() as client:
                while queue and len(results) < limit:
                    node_id, depth = queue.pop(0)

                    if node_id in visited:
                        continue
                    visited.add(node_id)

                    # Get node info
                    response = await client.get(
                        f"{self._db.rest_url}/api/v2/graphs/{graph_id}/nodes/{node_id}",
                        timeout=10.0,
                    )

                    if response.status_code == 200:
                        node_data = response.json().get("node", {})
                        results.append(
                            {
                                "id": node_id,
                                "node": node_data,
                                "depth": depth,
                                "labels": node_data.get("labels", []),
                                "properties": node_data.get("properties", {}),
                                "_source_type": "graph",
                            }
                        )

                        # Get outgoing edges for further traversal
                        if depth < max_depth:
                            edge_response = await client.get(
                                f"{self._db.rest_url}/api/v2/graphs/{graph_id}/nodes/{node_id}/edges/outgoing",
                                timeout=10.0,
                            )

                            if edge_response.status_code == 200:
                                edges = edge_response.json().get("edges", [])
                                for edge in edges:
                                    edge_type = edge.get("edge_type", "")
                                    if edge_types is None or edge_type in edge_types:
                                        target = edge.get("to_node_id")
                                        if target and target not in visited:
                                            queue.append((target, depth + 1))

            return results[:limit]

        except Exception as exc:
            # Re-raised; see _execute_vector. A truncated traversal that returns
            # [] is indistinguishable from a node with no edges.
            raise exc

    async def _execute_document(
        self, component: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Execute document query component against embedded database."""
        try:
            collection = component.get("collection", "")
            filter_expr = component.get("filter")
            text_query = component.get("text_query")
            limit = component.get("limit", 100)

            # Build filter expression
            filter_str = None
            if filter_expr:
                # Convert dict filter to string expression
                if isinstance(filter_expr, dict):
                    parts = []
                    for k, v in filter_expr.items():
                        if isinstance(v, str):
                            parts.append(f"$.{k} = '{v}'")
                        else:
                            parts.append(f"$.{k} = {v}")
                    filter_str = " AND ".join(parts)
                else:
                    filter_str = str(filter_expr)

            # Query documents via REST API
            async with self._db._http_client() as client:
                response = await client.post(
                    f"{self._db.rest_url}/api/v2/document-collections/{collection}/query",
                    json={
                        "filter": filter_str,
                        "text_query": text_query,
                        "limit": limit,
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    documents = data.get("documents", [])
                    return [
                        {
                            "id": doc.get("id", doc.get("_id", "")),
                            "document": doc,
                            "_source_type": "document",
                        }
                        for doc in documents
                    ]

            return []

        except Exception as exc:
            # A read leg returning [] on failure is indistinguishable from a
            # query that matched nothing. Logged at warning because, unlike the
            # probes above, an empty answer here will be acted on as data.
            logger.warning("embedded document query failed: %s", exc, exc_info=True)
            return []

    async def _execute_logs(self, component: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute log query component.

        Note: Log queries are not yet fully implemented in embedded mode.
        Returns empty results for now.
        """
        # Log queries would require observability backend integration
        return []

    async def _execute_metrics(self, component: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute metric aggregation component.

        Note: Metric queries are not yet fully implemented in embedded mode.
        Returns empty results for now.
        """
        # Metric queries would require observability backend integration
        return []

    def _apply_joins(
        self,
        component_results: list[list[dict[str, Any]]],
        joins: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Apply joins between component results."""
        if len(component_results) < 2:
            return component_results

        for join in joins:
            join_type = join.get("join_type", "inner")
            left_field = join.get("left_field", "id")
            right_field = join.get("right_field", "id")

            if join_type in ("inner", "semantic"):
                left_results = component_results[0]
                right_results = component_results[1]

                # Build index of right results
                right_index: dict[str, dict[str, Any]] = {}
                for r in right_results:
                    key = self._extract_field(r, right_field)
                    if key:
                        right_index[key] = r

                # Join
                joined: list[dict[str, Any]] = []
                for left in left_results:
                    left_key = self._extract_field(left, left_field)
                    if left_key and left_key in right_index:
                        merged = {**left, **right_index[left_key]}
                        merged["_join_type"] = join_type
                        joined.append(merged)

                component_results = [joined] + component_results[2:]

        return component_results

    def _extract_field(self, record: dict[str, Any], field_path: str) -> str | None:
        """Extract a field value from a nested record."""
        parts = field_path.split(".")
        current: Any = record
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return str(current) if current is not None else None

    def _fuse_results(
        self,
        component_results: list[list[dict[str, Any]]],
        strategy: str,
        weights: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Fuse results from multiple components."""
        if not component_results:
            return []

        if len(component_results) == 1:
            return component_results[0]

        if strategy == "intersection":
            return self._fuse_intersection(component_results)
        elif strategy == "union":
            return self._fuse_union(component_results)
        elif strategy == "rrf":
            return self._fuse_rrf(component_results, weights)
        elif strategy == "weighted":
            return self._fuse_weighted(component_results, weights)
        else:
            # Default to RRF
            return self._fuse_rrf(component_results, weights)

    def _fuse_intersection(
        self,
        component_results: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Return only records present in all components."""
        if not component_results:
            return []

        # Get IDs from first component
        common_ids = set(r.get("id") for r in component_results[0] if r.get("id"))

        # Intersect with other components
        for results in component_results[1:]:
            ids = set(r.get("id") for r in results if r.get("id"))
            common_ids &= ids

        # Return records with common IDs, merging data from all components
        merged_records: dict[str, dict[str, Any]] = {}
        for results in component_results:
            for r in results:
                record_id = r.get("id")
                if record_id in common_ids:
                    if record_id not in merged_records:
                        merged_records[record_id] = r.copy()
                    else:
                        # Merge additional fields
                        for k, v in r.items():
                            if k not in merged_records[record_id]:
                                merged_records[record_id][k] = v

        return list(merged_records.values())

    def _fuse_union(
        self,
        component_results: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Return all records from any component (deduplicated)."""
        seen_ids: set = set()
        result: list[dict[str, Any]] = []

        for results in component_results:
            for r in results:
                record_id = r.get("id")
                if record_id and record_id not in seen_ids:
                    seen_ids.add(record_id)
                    result.append(r)
                elif not record_id:
                    result.append(r)

        return result

    def _fuse_rrf(
        self,
        component_results: list[list[dict[str, Any]]],
        weights: dict[str, float],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion.

        RRF score = sum(weight_i / (k + rank_i)) for each component

        This is a robust fusion method that works well when different
        components have different score scales.
        """
        scores: dict[str, float] = {}
        records: dict[str, dict[str, Any]] = {}

        for comp_idx, results in enumerate(component_results):
            weight = weights.get(f"component_{comp_idx}", 1.0)
            # Also check for type-based weights
            if results and results[0].get("_source_type"):
                source_type = results[0]["_source_type"]
                weight = weights.get(f"{source_type}_{comp_idx + 1}", weight)

            for rank, record in enumerate(results):
                record_id = record.get("id", f"_anon_{comp_idx}_{rank}")
                rrf_score = weight / (k + rank + 1)

                if record_id in scores:
                    scores[record_id] += rrf_score
                else:
                    scores[record_id] = rrf_score
                    records[record_id] = record

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        result: list[dict[str, Any]] = []
        for record_id in sorted_ids:
            record = records[record_id].copy()
            record["_rrf_score"] = scores[record_id]
            result.append(record)

        return result

    def _fuse_weighted(
        self,
        component_results: list[list[dict[str, Any]]],
        weights: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Weighted score combination."""
        scores: dict[str, float] = {}
        records: dict[str, dict[str, Any]] = {}

        for comp_idx, results in enumerate(component_results):
            weight = weights.get(f"component_{comp_idx}", 1.0)

            for record in results:
                record_id = record.get("id", id(record))
                component_score = record.get("score", 1.0) * weight

                if record_id in scores:
                    scores[record_id] += component_score
                else:
                    scores[record_id] = component_score
                    records[record_id] = record

        # Sort by weighted score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        result: list[dict[str, Any]] = []
        for record_id in sorted_ids:
            record = records[record_id].copy()
            record["_weighted_score"] = scores[record_id]
            result.append(record)

        return result

    def _apply_time_decay(
        self,
        records: list[dict[str, Any]],
        time_decay: tuple,
    ) -> list[dict[str, Any]]:
        """Apply time decay to record scores."""
        import math

        func, params = time_decay
        reference_time = params.get("reference_time", int(time.time() * 1e9))
        halflife_ns = params.get("halflife_hours", 24) * 3600 * 1e9
        time_field = params.get("time_field", "timestamp")

        for record in records:
            timestamp = record.get(time_field)
            if timestamp is None:
                continue

            age_ns = reference_time - timestamp
            if age_ns < 0:
                age_ns = 0

            # Get function type (handle both enum and string)
            func_name = func.value if hasattr(func, "value") else str(func)

            if func_name == "linear":
                decay = max(0, 1 - (age_ns / (halflife_ns * 2)))
            elif func_name == "exponential":
                decay = math.exp(-0.693 * age_ns / halflife_ns)  # ln(2) = 0.693
            elif func_name == "gaussian":
                decay = math.exp(-0.5 * (age_ns / halflife_ns) ** 2)
            else:
                decay = 1.0

            # Apply decay to existing score
            current_score = record.get("score", record.get("_rrf_score", 1.0))
            record["_decayed_score"] = current_score * decay
            record["_time_decay"] = decay

        # Re-sort by decayed score
        records.sort(key=lambda x: x.get("_decayed_score", 0), reverse=True)

        return records
