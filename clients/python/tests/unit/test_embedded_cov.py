"""Offline unit tests for proximadb_sdk.embedded.

All HTTP transport is mocked. The embedded module creates its HTTP clients
lazily, so we patch ``httpx.AsyncClient`` with a fake async-context-manager
whose verb methods return a ``FakeResp``. We never start a real subprocess
server or touch the network.
"""

import asyncio
import base64

import httpx
import pytest

from tests.unit._id_helpers import is_uuid as _is_uuid


@pytest.fixture(autouse=True)
def _fresh_event_loop():
    """Per-test fresh event loop — embedded sync wrappers use
    asyncio.get_event_loop(), which returns a closed loop after a sibling test
    closes the default one (suite-order fragility). See test_hybrid_cov.py."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        loop.close()
        asyncio.set_event_loop(None)


from proximadb_sdk import embedded as E
from proximadb_sdk.embedded import (
    EmbeddedCollection,
    EmbeddedConfig,
    EmbeddedMultiModalQueryExecutor,
    EmbeddedProximaDB,
    FunctionEmbeddingModel,
    OllamaEmbeddingModel,
    OpenAIEmbeddingModel,
    SentenceTransformerModel,
    connect_embedded,
    create_embedding_model,
)


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeResp:
    def __init__(self, json_body=None, status_code=200, text="", content=b""):
        self._json = json_body if json_body is not None else {}
        self.status_code = status_code
        self.headers = {}
        self.text = text
        self.content = content

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://test.local")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError(
                self.text or "HTTP error",
                request=request,
                response=response,
            )


class FakeAsyncClient:
    """Drop-in for ``httpx.AsyncClient`` used as an async context manager.

    ``responder`` is a callable ``(verb, url, **kwargs) -> FakeResp``. Each call
    is recorded for assertions.
    """

    calls = []
    init_kwargs = []
    responder = staticmethod(lambda verb, url, **kw: FakeResp({"success": True}))

    #: `_http_client` pools one client per instance and re-creates it when the
    #: pooled one is closed, so it reads `.is_closed` on the way out. A fake
    #: standing in for `httpx.AsyncClient` has to carry the attributes the code
    #: under test reads, or it stops being a drop-in and starts being a source
    #: of AttributeErrors that look like product bugs.
    is_closed = False

    def __init__(self, *a, **kw):
        FakeAsyncClient.init_kwargs.append(kw)

    async def aclose(self):
        type(self).is_closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def _do(self, verb, url, **kw):
        FakeAsyncClient.calls.append((verb, url, kw))
        return FakeAsyncClient.responder(verb, url, **kw)

    async def get(self, url, **kw):
        return await self._do("GET", url, **kw)

    async def post(self, url, **kw):
        return await self._do("POST", url, **kw)

    async def put(self, url, **kw):
        return await self._do("PUT", url, **kw)

    async def patch(self, url, **kw):
        return await self._do("PATCH", url, **kw)

    async def delete(self, url, **kw):
        return await self._do("DELETE", url, **kw)


@pytest.fixture
def patched_http(monkeypatch):
    """Patch httpx.AsyncClient and return the fake so tests set the responder."""
    FakeAsyncClient.calls = []
    FakeAsyncClient.init_kwargs = []
    FakeAsyncClient.responder = staticmethod(
        lambda verb, url, **kw: FakeResp({"success": True})
    )
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    return FakeAsyncClient


def make_started_db():
    """An EmbeddedProximaDB flagged as started (so methods skip start())."""
    db = EmbeddedProximaDB(data_dir="/tmp/proximadb-test-embedded")
    db._started = True
    return db


# ---------------------------------------------------------------------------
# Embedding models (pure, no network)
# ---------------------------------------------------------------------------


def test_function_embedding_model_with_batch():
    m = FunctionEmbeddingModel(
        embed_fn=lambda t: [float(len(t))],
        dimension=1,
        batch_fn=lambda ts: [[float(len(t))] for t in ts],
    )
    assert m.embed("abc") == [3.0]
    assert m.embed_batch(["a", "bb"]) == [[1.0], [2.0]]
    assert m.get_dimension() == 1


def test_function_embedding_model_fallback_batch_and_async():
    m = FunctionEmbeddingModel(embed_fn=lambda t: [1.0, 2.0], dimension=2)
    # No batch_fn -> sequential fallback
    assert m.embed_batch(["x", "y"]) == [[1.0, 2.0], [1.0, 2.0]]
    # No async_embed_fn -> base class executor path
    assert run(m.embed_async("z")) == [1.0, 2.0]
    # batch async default path
    assert run(m.embed_batch_async(["a"])) == [[1.0, 2.0]]


def test_function_embedding_model_async_fn():
    async def aembed(text):
        return [9.0]

    m = FunctionEmbeddingModel(
        embed_fn=lambda t: [0.0], dimension=1, async_embed_fn=aembed
    )
    assert run(m.embed_async("hi")) == [9.0]


def test_sentence_transformer_lazy_import_error(monkeypatch):
    model = SentenceTransformerModel(model_name="some/model")
    # Force the import to fail inside _ensure_loaded.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "sentence_transformers":
            raise ImportError("missing")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        model.embed("hello")


def test_ollama_model_dimension_and_init():
    m = OllamaEmbeddingModel(model_name="nomic-embed-text", dimension=768)
    assert m.get_dimension() == 768
    assert m.base_url == "http://localhost:11434"


def test_ollama_embed_mocked(monkeypatch):
    m = OllamaEmbeddingModel()
    monkeypatch.setattr(
        httpx, "post", lambda *a, **kw: FakeResp({"embedding": [0.1, 0.2]})
    )
    assert m.embed("text") == [0.1, 0.2]


def test_ollama_embed_batch_mocked(monkeypatch):
    m = OllamaEmbeddingModel()

    class C:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            return FakeResp({"embedding": [0.5]})

    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: C())
    assert m.embed_batch(["a", "b"]) == [[0.5], [0.5]]


def test_ollama_embed_async_mocked(monkeypatch):
    m = OllamaEmbeddingModel()

    class AC:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **kw):
            return FakeResp({"embedding": [3.0]})

    monkeypatch.setattr(httpx, "AsyncClient", AC)
    assert run(m.embed_async("x")) == [3.0]
    assert run(m.embed_batch_async(["x", "y"])) == [[3.0], [3.0]]


def test_openai_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIEmbeddingModel()


def test_openai_dimension_and_embed(monkeypatch):
    m = OpenAIEmbeddingModel(api_key="sk-test")
    assert m.get_dimension() == 1536
    assert (
        OpenAIEmbeddingModel(
            model_name="text-embedding-3-large", api_key="k"
        ).get_dimension()
        == 3072
    )
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **kw: FakeResp({"data": [{"index": 0, "embedding": [1.0]}]}),
    )
    assert m.embed("hi") == [1.0]


def test_openai_embed_batch_orders_by_index(monkeypatch):
    m = OpenAIEmbeddingModel(api_key="sk-test")
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **kw: FakeResp(
            {
                "data": [
                    {"index": 1, "embedding": [2.0]},
                    {"index": 0, "embedding": [1.0]},
                ]
            }
        ),
    )
    assert m.embed_batch(["a", "b"]) == [[1.0], [2.0]]


def test_create_embedding_model_factory():
    assert isinstance(
        create_embedding_model("sentence-transformers", "m"), SentenceTransformerModel
    )
    assert isinstance(create_embedding_model("ollama", "nomic"), OllamaEmbeddingModel)
    assert isinstance(
        create_embedding_model("openai", "x", api_key="k"), OpenAIEmbeddingModel
    )
    with pytest.raises(ValueError):
        create_embedding_model("bogus")


# ---------------------------------------------------------------------------
# Config / URL / construction
# ---------------------------------------------------------------------------


def test_config_defaults_and_urls():
    db = EmbeddedProximaDB(data_dir="/tmp/x")
    assert db.config.transport == "uds"
    assert db.rest_url == "http://localhost"
    assert db.grpc_url.startswith("unix://")
    assert db.socket_dir == E.Path("/tmp/x/sockets")
    assert db.rest_socket_path.name == E.UDS_REST_SOCKET_NAME
    assert isinstance(db.config, EmbeddedConfig)


def test_config_override():
    cfg = EmbeddedConfig(
        data_dir="/tmp/y", rest_port=20001, grpc_port=20002, transport="tcp"
    )
    db = EmbeddedProximaDB(config=cfg)
    assert db.config.rest_port == 20001
    assert db.rest_url == "http://localhost:20001"
    assert "20002" in db.grpc_url


def test_find_binary_explicit_path():
    db = EmbeddedProximaDB(binary_path="/usr/local/bin/proximadb-server")
    assert db._find_binary() == "/usr/local/bin/proximadb-server"


def test_find_binary_not_found(monkeypatch):
    db = EmbeddedProximaDB()
    # Force every Path.exists() to False and shutil.which to None.
    monkeypatch.setattr(E.Path, "exists", lambda self: False)
    import shutil

    monkeypatch.setattr(shutil, "which", lambda p: None)
    with pytest.raises(RuntimeError):
        db._find_binary()


def test_generate_config_writes_toml(tmp_path):
    db = EmbeddedProximaDB(data_dir=str(tmp_path / "embcfg"))
    path = db._generate_config()
    text = E.Path(path).read_text()
    assert "[server]" in text
    assert f"embedded-{db._instance_id}" in text
    assert E.Path(path).name == f"embedded-{db._instance_id}.toml"
    assert 'transport = "uds"' in text
    assert "socket_dir" in text
    assert "arrow_flight_port" in text


def test_uds_socket_dir_falls_back_when_path_is_too_long(tmp_path):
    db = EmbeddedProximaDB(data_dir=str(tmp_path / ("x" * 120)))
    assert db.socket_dir is not None
    assert db.socket_dir != db._data_dir / "sockets"
    assert db._uds_paths_fit(db.socket_dir)


def test_find_binary_on_path(monkeypatch):
    db = EmbeddedProximaDB()
    # No Path exists, but shutil.which resolves the PATH entry.
    monkeypatch.setattr(E.Path, "exists", lambda self: False)
    import shutil

    monkeypatch.setattr(shutil, "which", lambda p: "/opt/bin/proximadb-server")
    assert db._find_binary() == "proximadb-server"


class _FakeProcess:
    def __init__(self, returncode=None):
        self.pid = 4321
        self.waited = False
        self.terminated = False
        self.killed = False
        self.stderr = None
        # Mirrors subprocess.Popen: None while the child is running, and the
        # exit status once it has exited. Startup polls this to fail fast on a
        # child that died instead of waiting out the ceiling.
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waited = True

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_kill_process_graceful(monkeypatch):
    db = EmbeddedProximaDB()
    proc = _FakeProcess()
    db._process = proc
    monkeypatch.setattr(E.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(E.os, "getpgid", lambda pid: pid)
    db._kill_process()
    assert db._process is None
    assert proc.waited


def test_kill_process_force_on_exception(monkeypatch):
    db = EmbeddedProximaDB()
    proc = _FakeProcess()
    db._process = proc

    calls = {"n": 0}

    def killpg(pid, sig):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("graceful failed")

    monkeypatch.setattr(E.os, "killpg", killpg)
    monkeypatch.setattr(E.os, "getpgid", lambda pid: pid)
    db._kill_process()
    assert db._process is None
    assert calls["n"] == 2  # SIGTERM then SIGKILL


def test_kill_process_no_process():
    db = EmbeddedProximaDB()
    db._process = None
    db._kill_process()  # no-op, no error


def test_sync_context_manager(monkeypatch):
    seen = {"start": 0, "stop": 0}

    async def fake_start(self, timeout=30.0):
        seen["start"] += 1
        self._started = True

    async def fake_stop(self):
        seen["stop"] += 1

    monkeypatch.setattr(EmbeddedProximaDB, "start", fake_start)
    monkeypatch.setattr(EmbeddedProximaDB, "stop", fake_stop)

    db = EmbeddedProximaDB(data_dir="/tmp/sync-cm")
    with db as ctx:
        assert ctx is db
    assert seen["start"] == 1
    assert seen["stop"] == 1


def test_start_full_flow_then_timeout(monkeypatch, tmp_path):
    """Exercise start(): find binary, gen config, spawn (mocked), health poll
    succeeds on first probe."""
    db = EmbeddedProximaDB(data_dir=str(tmp_path / "startflow"))
    monkeypatch.setattr(db, "_find_binary", lambda: "/bin/true")
    monkeypatch.setattr(db, "_generate_config", lambda: "/tmp/cfg.toml")

    spawned = {}

    def fake_popen(args, **kw):
        spawned["args"] = args
        spawned["kwargs"] = kw
        return _FakeProcess()

    monkeypatch.setattr(E.subprocess, "Popen", fake_popen)

    class HealthClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **kw):
            return FakeResp({}, 200)

    monkeypatch.setattr(db, "_sync_http_client", lambda **kw: HealthClient())
    monkeypatch.setattr(
        E,
        "read_runtime_state",
        lambda _data_dir: {"pid": 4321, "phase": "serving"},
    )

    run(db.start(timeout=5.0))
    assert db._started is True
    assert spawned["args"][0] == "/bin/true"
    assert spawned["kwargs"]["stderr"] is E.subprocess.STDOUT
    assert spawned["kwargs"]["stdout"] is not E.subprocess.PIPE
    assert spawned["kwargs"]["cwd"] == str(db._data_dir)
    assert spawned["kwargs"]["env"]["PROXIMADB_CORPUS_VERSION_PATH"] == str(
        db._data_dir / "corpus-versions.json"
    )
    assert db._server_log_path.exists()


def test_start_rejects_health_from_a_different_process(monkeypatch, tmp_path):
    db = EmbeddedProximaDB(data_dir=str(tmp_path / "owner-mismatch"))
    monkeypatch.setattr(db, "_find_binary", lambda: "/bin/true")
    monkeypatch.setattr(E.subprocess, "Popen", lambda args, **kw: _FakeProcess())

    class ExistingServerHealthClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **kw):
            return FakeResp({}, 200)

    monkeypatch.setattr(
        db, "_sync_http_client", lambda **kw: ExistingServerHealthClient()
    )
    monkeypatch.setattr(
        E,
        "read_runtime_state",
        lambda _data_dir: {"pid": 9999, "phase": "serving"},
    )
    killed = {"v": False}
    monkeypatch.setattr(db, "_kill_process", lambda: killed.__setitem__("v", True))
    times = iter([0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(E.time, "time", lambda: next(times, 100.0))

    async def no_sleep(*a, **kw):
        return None

    monkeypatch.setattr(E.asyncio, "sleep", no_sleep)

    with pytest.raises(TimeoutError):
        run(db.start(timeout=1.0))
    assert db._started is False
    assert killed["v"] is True


def test_start_timeout_kills(monkeypatch, tmp_path):
    """Health probe never returns 200 -> start() times out and kills process."""
    db = EmbeddedProximaDB(data_dir=str(tmp_path / "startto"))
    monkeypatch.setattr(db, "_find_binary", lambda: "/bin/true")
    monkeypatch.setattr(db, "_generate_config", lambda: "/tmp/cfg2.toml")
    monkeypatch.setattr(E.subprocess, "Popen", lambda args, **kw: _FakeProcess())

    class BadHealthClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **kw):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(db, "_sync_http_client", lambda **kw: BadHealthClient())

    killed = {"v": False}
    monkeypatch.setattr(db, "_kill_process", lambda: killed.__setitem__("v", True))

    # Make time advance instantly past the timeout and sleep a no-op.
    times = iter([0.0, 0.05, 100.0, 200.0])
    monkeypatch.setattr(E.time, "time", lambda: next(times))

    async def no_sleep(*a, **kw):
        return None

    monkeypatch.setattr(E.asyncio, "sleep", no_sleep)

    with pytest.raises(TimeoutError):
        run(db.start(timeout=1.0))
    assert killed["v"] is True


def test_insert_vectors_alias(patched_http):
    db = make_started_db()
    res = run(db._insert_vectors("c", [{"id": "z", "vector": [1.0]}]))
    assert res["success"] is True


# ---------------------------------------------------------------------------
# Value conversion helpers (pure)
# ---------------------------------------------------------------------------


def test_to_sql_value_variants():
    db = EmbeddedProximaDB()
    assert db._to_sql_value(None) == {"null_value": None}
    assert db._to_sql_value(True) == {"bool_value": True}
    assert db._to_sql_value(5) == {"int64_value": 5}
    assert db._to_sql_value(1.5) == {"number_value": 1.5}
    assert db._to_sql_value("s") == {"string_value": "s"}
    b = db._to_sql_value(b"ab")
    assert b["bytes_value"] == base64.b64encode(b"ab").decode("ascii")
    arr = db._to_sql_value([1, "x"])
    assert arr["array_value"]["values"][0] == {"int64_value": 1}
    obj = db._to_sql_value({"k": 2})
    assert obj["object_value"]["fields"]["k"] == {"int64_value": 2}

    class Weird:
        def __str__(self):
            return "weird"

    assert db._to_sql_value(Weird()) == {"string_value": "weird"}


def test_convert_metadata():
    db = EmbeddedProximaDB()
    out = db._convert_metadata({"a": 1, "b": "x"})
    assert out == {"a": {"int64_value": 1}, "b": {"string_value": "x"}}


def test_to_proxima_value_variants():
    db = EmbeddedProximaDB()
    assert db._to_proxima_value(None) is None
    assert db._to_proxima_value(3) == 3
    assert db._to_proxima_value("s") == "s"
    binv = db._to_proxima_value(b"ab")
    assert binv["type"] == "binary"
    arr = db._to_proxima_value([1, 2])
    assert arr == {"type": "array", "value": [1, 2]}
    # Already typed dict passes through.
    typed = {"type": "binary", "value": "AAA"}
    assert db._to_proxima_value(typed) == typed
    jsonb = db._to_proxima_value({"a": 1})
    assert jsonb["type"] == "jsonb"

    class Weird:
        def __str__(self):
            return "w"

    assert db._to_proxima_value(Weird()) == "w"


def test_normalize_record_payload_paths():
    db = EmbeddedProximaDB()
    rec = {
        "id": "r1",
        "vector": [1, 2, 3],
        "metadata": {"m": 1},
        "props": {"p": "x"},
        "typed_fields": {"tf": {"value": 5, "value_type": "int64"}},
        "text_fields": {"body": "hi"},
    }
    out = db._normalize_record_payload(rec, 0)
    assert out["id"] == "r1"
    assert out["vector"] == [1.0, 2.0, 3.0]
    assert out["props"]["m"] == 1
    assert out["props"]["p"] == "x"
    assert out["props"]["tf"] == {"type": "int64", "value": 5}
    assert out["text_fields"] == {"body": "hi"}


def test_normalize_record_payload_typed_model_dump():
    db = EmbeddedProximaDB()

    class TF:
        def model_dump(self, exclude_none=True):
            return {"value": 7}

    out = db._normalize_record_payload(
        {"vector": [0.0], "typed_fields": {"x": TF()}}, 3
    )
    # model_dump result has a "value" key but no value_type/type -> type is None
    assert out["props"]["x"] == {"type": None, "value": 7}
    assert _is_uuid(out["id"])


def test_normalize_record_payload_missing_vector():
    db = EmbeddedProximaDB()
    with pytest.raises(ValueError):
        db._normalize_record_payload({"id": "x"}, 0)


# ---------------------------------------------------------------------------
# DB collection lifecycle (mocked HTTP)
# ---------------------------------------------------------------------------


def test_create_collection_success(patched_http):
    captured = {}

    def responder(v, u, **kw):
        captured["verb"] = v
        captured["url"] = u
        captured["json"] = kw.get("json")
        return FakeResp({"collection_id": "uuid-c1", "name": "c1", "dimension": 4})

    patched_http.responder = staticmethod(responder)
    db = make_started_db()
    col = run(db.create_collection("c1", dimension=4, distance_metric="euclidean"))
    assert isinstance(col, EmbeddedCollection)
    assert col.name == "c1"
    assert db._collections["c1"] is col
    assert captured["verb"] == "POST"
    assert captured["url"].endswith("/api/v2/collections")
    # Assert the required fields are sent; tolerate additive server-shape fields
    # (e.g. the default `engine`) so the contract check doesn't break on additions.
    assert {
        "name": "c1",
        "dimension": 4,
        "distance_metric": "euclidean",
    }.items() <= captured["json"].items()
    assert "transport" in patched_http.init_kwargs[-1]


def test_create_collection_with_model_autodim(patched_http):
    db = make_started_db()
    model = FunctionEmbeddingModel(embed_fn=lambda t: [0.0] * 8, dimension=8)
    col = run(db.create_collection("c2", embedding_model=model))
    assert col.dimension == 8
    assert col.has_embedding_model


def test_create_collection_string_model(patched_http, monkeypatch):
    # Avoid real sentence-transformers load by patching get_dimension.
    monkeypatch.setattr(SentenceTransformerModel, "get_dimension", lambda self: 384)
    db = make_started_db()
    col = run(db.create_collection("c3", embedding_model="some/model"))
    assert col.dimension == 384


def test_create_collection_requires_dimension(patched_http):
    db = make_started_db()
    with pytest.raises(ValueError):
        run(db.create_collection("c4"))


def test_create_collection_failure_raises(patched_http):
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({"error": "boom"}, status_code=500, text="boom")
    )
    db = make_started_db()
    with pytest.raises(RuntimeError):
        run(db.create_collection("c5", dimension=2))


def test_create_collection_already_exists_ok(patched_http):
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({"error": "already exists"}, status_code=409)
    )
    db = make_started_db()
    col = run(db.create_collection("c6", dimension=2))
    assert col.name == "c6"


def test_get_collection_cached(patched_http):
    db = make_started_db()
    db._collections["cached"] = EmbeddedCollection("cached", 3, db)
    got = run(db.get_collection("cached"))
    assert got is db._collections["cached"]


def test_get_collection_from_server(patched_http):
    # get_collection resolves from the LIST endpoint (GET /collections/{name} returns
    # 200 even for non-existent collections), so mock the flat v2 LIST payload.
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp(
            {"collections": [{"name": "remote", "dimension": 12}]}
        )
    )
    db = make_started_db()
    got = run(db.get_collection("remote"))
    assert got.dimension == 12


def test_get_collection_not_found(patched_http):
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({}, status_code=404)
    )
    db = make_started_db()
    assert run(db.get_collection("nope")) is None


def test_delete_collection(patched_http):
    patched_http.responder = staticmethod(lambda v, u, **kw: FakeResp({}, 204))
    db = make_started_db()
    db._collections["d"] = EmbeddedCollection("d", 2, db)
    assert run(db.delete_collection("d")) is True
    assert "d" not in db._collections


def test_list_collections(patched_http):
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp(
            {"collections": [{"config": {"name": "a"}}, {"config": {"name": "b"}}]}
        )
    )
    db = make_started_db()
    assert run(db.list_collections()) == ["a", "b"]


def test_list_collections_error(patched_http):
    patched_http.responder = staticmethod(lambda v, u, **kw: FakeResp({}, 500))
    db = make_started_db()
    assert run(db.list_collections()) == []


# ---------------------------------------------------------------------------
# Records / search / stats (mocked HTTP) + EmbeddedCollection wrappers
# ---------------------------------------------------------------------------


def test_insert_records_and_collection_wrapper(patched_http):
    captured = {}

    def responder(v, u, **kw):
        captured["json"] = kw.get("json")
        return FakeResp({"success": True, "inserted": 1})

    patched_http.responder = staticmethod(responder)
    db = make_started_db()
    col = EmbeddedCollection("c", 3, db)
    res = run(col.insert_records([{"id": "r", "vector": [1, 2, 3]}]))
    assert res["inserted"] == 1
    assert captured["json"]["records"][0]["id"] == "r"
    # insert() alias
    run(col.insert([{"id": "r2", "vector": [0, 0, 0]}]))


def test_insert_with_embedding(patched_http):
    db = make_started_db()
    model = FunctionEmbeddingModel(
        embed_fn=lambda t: [1.0, 2.0],
        dimension=2,
        batch_fn=lambda ts: [[1.0, 2.0] for _ in ts],
    )
    col = EmbeddedCollection("c", 2, db, embedding_model=model)
    res = run(
        col.insert_with_embedding(
            [{"id": "d1", "text": "hello", "metadata": {"k": "v"}}]
        )
    )
    assert res["success"] is True


def test_insert_with_embedding_no_model(patched_http):
    db = make_started_db()
    col = EmbeddedCollection("c", 2, db)
    with pytest.raises(RuntimeError):
        run(col.insert_with_embedding([{"id": "d", "text": "t"}]))


def test_search_vector_posts_v2_typed_search(patched_http):
    captured = {}

    def responder(v, u, **kw):
        captured["verb"] = v
        captured["url"] = u
        captured["json"] = kw.get("json")
        return FakeResp({"results": [{"id": "x", "score": 0.9}]})

    patched_http.responder = staticmethod(responder)
    db = make_started_db()
    col = EmbeddedCollection("c", 2, db)
    out = run(col.search([0.1, 0.2], top_k=5, filters={"f": 1}))
    assert out[0]["id"] == "x"
    assert captured["verb"] == "POST"
    assert captured["url"].endswith("/api/v2/collections/c/search")
    assert {
        "vector": [0.1, 0.2],
        "top_k": 5,
        "filters": [{"field": "f", "op": "eq", "value": 1}],
    }.items() <= captured["json"].items()


def test_search_vector_list_results(patched_http):
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({"results": [{"id": "y"}]})
    )
    db = make_started_db()
    assert run(db._search_vectors("c", [0.0], 3))[0]["id"] == "y"


def test_search_vector_empty(patched_http):
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({"success": False})
    )
    db = make_started_db()
    assert run(db._search_vectors("c", [0.0])) == []


def test_search_text(patched_http):
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({"results": [{"id": "t"}]})
    )
    db = make_started_db()
    model = FunctionEmbeddingModel(embed_fn=lambda t: [0.1, 0.2], dimension=2)
    col = EmbeddedCollection("c", 2, db, embedding_model=model)
    out = run(col.search_text("query", top_k=4))
    assert out[0]["id"] == "t"


def test_search_text_no_model():
    db = make_started_db()
    col = EmbeddedCollection("c", 2, db)
    with pytest.raises(RuntimeError):
        run(col.search_text("q"))


def test_delete_vectors_and_count(patched_http):
    calls = []

    def responder(v, u, **kw):
        calls.append((v, u))
        if v == "DELETE":
            return FakeResp({}, 204)
        return FakeResp({"collection": {"stats": {"vector_count": 7}}})

    patched_http.responder = staticmethod(responder)
    db = make_started_db()
    col = EmbeddedCollection("c", 2, db)
    assert run(col.delete(["a", "b"])) == 2
    assert calls[0] == ("DELETE", f"{db.rest_url}/api/v2/collections/c/records/a")
    assert calls[1] == ("DELETE", f"{db.rest_url}/api/v2/collections/c/records/b")
    assert run(col.count()) == 7


def test_delete_vectors_skips_404(patched_http):
    def responder(v, u, **kw):
        if u.endswith("/missing"):
            return FakeResp({}, 404)
        return FakeResp({}, 204)

    patched_http.responder = staticmethod(responder)
    db = make_started_db()
    assert run(db._delete_vectors("c", ["ok", "missing"])) == 1


def test_collection_stats_error_returns_empty(patched_http):
    patched_http.responder = staticmethod(lambda v, u, **kw: FakeResp({}, 500))
    db = make_started_db()
    assert run(db._get_collection_stats("c")) == {}


def test_set_embedding_model():
    db = make_started_db()
    col = EmbeddedCollection("c", 2, db)
    assert not col.has_embedding_model
    col.set_embedding_model(FunctionEmbeddingModel(lambda t: [0.0], 1))
    assert col.has_embedding_model


# ---------------------------------------------------------------------------
# health_check / start guard / stop
# ---------------------------------------------------------------------------


def test_health_check_not_started():
    db = EmbeddedProximaDB()
    assert run(db.health_check()) is False


def test_health_check_ok(patched_http):
    patched_http.responder = staticmethod(lambda v, u, **kw: FakeResp({}, 200))
    db = make_started_db()
    assert run(db.health_check()) is True


def test_health_check_exception(patched_http):
    def boom(v, u, **kw):
        raise RuntimeError("net")

    patched_http.responder = staticmethod(boom)
    db = make_started_db()
    assert run(db.health_check()) is False


def test_start_returns_when_already_started():
    db = make_started_db()
    run(db.start())  # no-op, must not block
    assert db._started


def test_stop_when_not_started():
    db = EmbeddedProximaDB()
    run(db.stop())  # no-op
    assert not db._started


def test_stop_when_started_kills(monkeypatch):
    db = make_started_db()
    killed = {"v": False}
    monkeypatch.setattr(db, "_kill_process", lambda: killed.__setitem__("v", True))
    run(db.stop())
    assert killed["v"] is True
    assert not db._started


# ---------------------------------------------------------------------------
# Document API (mocked HTTP)
# ---------------------------------------------------------------------------


def test_document_api(patched_http):
    db = make_started_db()
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({"collection_id": "dc", "id": "doc1"})
    )
    assert (
        run(
            db.create_document_collection(
                "dc",
                indexes=[{"path": "$.a"}],
                enable_fulltext=True,
                fulltext_paths=["$.a"],
            )
        )["collection_id"]
        == "dc"
    )
    assert run(
        db.query_documents("dc", filter={"a": 1}, projection=["a"], limit=5, offset=2)
    )


def test_get_document_found_and_missing(patched_http):
    db = make_started_db()
    patched_http.responder = staticmethod(lambda v, u, **kw: FakeResp({"id": "d"}, 200))
    assert run(db.get_document("dc", "d"))["id"] == "d"
    patched_http.responder = staticmethod(lambda v, u, **kw: FakeResp({}, 404))
    assert run(db.get_document("dc", "d")) is None


def test_update_and_delete_document(patched_http):
    db = make_started_db()
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({"version": 2}, 200)
    )
    assert run(db.update_document("dc", "d", [{"op": "SET"}]))["version"] == 2
    assert run(db.delete_document("dc", "d")) is True
    assert run(db.delete_document_collection("dc")) is True


def test_query_documents_str_filter(patched_http):
    db = make_started_db()
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({"documents": []})
    )
    run(db.query_documents("dc", filter="$.x = 1"))


# ---------------------------------------------------------------------------
# Time series API (mocked HTTP)
# ---------------------------------------------------------------------------


def test_timeseries_api(patched_http):
    db = make_started_db()
    patched_http.responder = staticmethod(lambda v, u, **kw: FakeResp({"ok": True}))
    assert run(
        db.create_timeseries_collection(
            "ts",
            value_columns=[{"name": "v", "data_type": "float"}],
            tag_columns=["host"],
            retention_ms=1000,
        )
    )["ok"]
    assert run(db.ingest_timeseries("ts", [{"timestamp": 1, "values": {"v": 1.0}}]))[
        "ok"
    ]
    assert run(
        db.query_timeseries(
            "ts", "t0", "t1", aggregation="AVG", bucket_ms=60, tag_filters={"h": "a"}
        )
    )["ok"]
    assert run(db.aggregate_timeseries("ts", "t0", "t1", [{"stage": 1}]))["ok"]
    patched_http.responder = staticmethod(lambda v, u, **kw: FakeResp({}, 204))
    assert run(db.delete_timeseries_collection("ts")) is True


# ---------------------------------------------------------------------------
# Hybrid search API (mocked HTTP)
# ---------------------------------------------------------------------------


def test_hybrid_search(patched_http):
    db = make_started_db()
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({"results": [{"id": "h"}]})
    )
    out = run(
        db.hybrid_search(
            "vc",
            [0.1, 0.2],
            text_query="hello",
            filters={"f": 1},
            fusion_params={"alpha": 0.5},
        )
    )
    assert out["results"][0]["id"] == "h"


def test_list_fusion_strategies(patched_http):
    db = make_started_db()
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({"strategies": [{"name": "rrf"}]})
    )
    assert run(db.list_fusion_strategies())[0]["name"] == "rrf"
    patched_http.responder = staticmethod(lambda v, u, **kw: FakeResp({}, 500))
    assert run(db.list_fusion_strategies()) == []


# ---------------------------------------------------------------------------
# Multi-modal executor
# ---------------------------------------------------------------------------


def _mmq(components, joins=None, fusion="rrf", weights=None, **kw):
    from proximadb_sdk.multimodal_query import MultiModalQuery

    return MultiModalQuery(
        components=components,
        joins=joins or [],
        fusion_strategy=fusion,
        fusion_weights=weights or {},
        time_decay=kw.get("time_decay"),
        limit=kw.get("limit", 10),
        offset=kw.get("offset", 0),
        timeout_ms=1000,
        include_scores=True,
        include_metadata=True,
        custom_scorer=kw.get("custom_scorer"),
    )


def test_executor_vector_component(patched_http):
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp(
            {"success": True, "results": [{"id": "v1", "score": 0.5}]}
        )
    )
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    q = _mmq([{"type": "vector", "collection": "c", "query_vector": [0.1], "top_k": 3}])
    res = run(ex.execute(q))
    assert res.total_count == 1
    assert res.records[0]["_source_type"] == "vector"


def test_executor_document_component(patched_http):
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp({"documents": [{"id": "d1"}]})
    )
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    q = _mmq([{"type": "document", "collection": "dc", "filter": {"a": "x", "b": 2}}])
    res = run(ex.execute(q))
    assert res.records[0]["id"] == "d1"


def test_executor_graph_component(patched_http):
    def responder(v, u, **kw):
        if u.endswith("/nodes/query"):
            return FakeResp({"nodes": [{"id": "n1"}]})
        if u.endswith("/edges/outgoing"):
            return FakeResp({"edges": [{"edge_type": "REL", "to_node_id": "n2"}]})
        return FakeResp({"node": {"labels": ["L"], "properties": {"p": 1}}})

    patched_http.responder = staticmethod(responder)
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    q = _mmq(
        [
            {
                "type": "graph",
                "graph_id": "g",
                "start_label": "L",
                "edge_types": ["REL"],
                "max_depth": 1,
            }
        ]
    )
    res = run(ex.execute(q))
    assert any(r["_source_type"] == "graph" for r in res.records)


def test_executor_graph_no_start_nodes(patched_http):
    # start_label query returns no nodes -> empty.
    patched_http.responder = staticmethod(lambda v, u, **kw: FakeResp({"nodes": []}))
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    out = run(ex._execute_graph({"graph_id": "g", "start_label": "L"}))
    assert out == []


def test_executor_logs_and_metrics_empty():
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    assert run(ex._execute_logs({})) == []
    assert run(ex._execute_metrics({})) == []


def test_executor_vector_then_graph_from_previous(patched_http):
    def responder(v, u, **kw):
        if u.endswith("/search"):
            return FakeResp({"success": True, "results": [{"id": "n1", "score": 0.5}]})
        if u.endswith("/nodes/n1"):
            return FakeResp({"node": {"labels": ["L"], "properties": {}}})
        if u.endswith("/edges/outgoing"):
            return FakeResp({"edges": []})
        return FakeResp({})

    patched_http.responder = staticmethod(responder)
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    q = _mmq(
        [
            {"type": "vector", "collection": "c", "query_vector": [0.1]},
            {
                "type": "graph",
                "graph_id": "g",
                "_from_previous": True,
                "_id_field": "id",
                "max_depth": 1,
            },
        ],
        fusion="union",
    )
    res = run(ex.execute(q))
    assert res.total_count >= 1


def test_executor_full_with_joins_and_time_decay(patched_http):
    from proximadb_sdk.multimodal_query import TimeDecayFunction

    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp(
            {"success": True, "results": [{"id": "1", "score": 0.9, "timestamp": 5}]}
        )
    )
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    q = _mmq(
        [
            {"type": "vector", "collection": "c1", "query_vector": [0.1]},
            {"type": "vector", "collection": "c2", "query_vector": [0.2]},
        ],
        joins=[{"join_type": "inner", "left_field": "id", "right_field": "id"}],
        fusion="rrf",
        time_decay=(
            TimeDecayFunction.EXPONENTIAL,
            {"reference_time": 1000, "halflife_hours": 1, "time_field": "timestamp"},
        ),
    )
    res = run(ex.execute(q))
    assert res.total_count >= 0


def test_executor_logs_metrics_in_execute(patched_http):
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    q = _mmq([{"type": "logs"}, {"type": "metrics"}])
    res = run(ex.execute(q))
    assert res.total_count == 0


def test_executor_unknown_component_type():
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    q = _mmq([{"type": "mystery"}])
    res = run(ex.execute(q))
    assert res.total_count == 0


def test_executor_custom_scorer(patched_http):
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp(
            {"success": True, "results": [{"id": "a"}, {"id": "b"}]}
        )
    )
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    q = _mmq(
        [{"type": "vector", "collection": "c", "query_vector": [0.1]}],
        custom_scorer=lambda r: 1.0 if r["id"] == "b" else 0.0,
    )
    res = run(ex.execute(q))
    assert res.records[0]["id"] == "b"


def test_fuse_intersection_union_weighted():
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    a = [{"id": "1", "score": 0.5}, {"id": "2", "score": 0.3}]
    b = [{"id": "1", "score": 0.7}, {"id": "3"}]
    inter = ex._fuse_results([a, b], "intersection", {})
    assert {r["id"] for r in inter} == {"1"}
    uni = ex._fuse_results([a, b], "union", {})
    assert {r["id"] for r in uni} == {"1", "2", "3"}
    w = ex._fuse_results([a, b], "weighted", {"component_0": 2.0})
    assert any("_weighted_score" in r for r in w)
    default = ex._fuse_results([a, b], "unknownstrat", {})
    assert any("_rrf_score" in r for r in default)
    # single + empty
    assert ex._fuse_results([a], "rrf", {}) == a
    assert ex._fuse_results([], "rrf", {}) == []


def test_fuse_union_anon_record():
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    out = ex._fuse_union([[{"noid": 1}], [{"noid": 2}]])
    assert len(out) == 2


def test_fuse_intersection_empty():
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    assert ex._fuse_intersection([]) == []


def test_fuse_rrf_with_source_type_weight():
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    a = [{"id": "1", "_source_type": "vector"}]
    b = [{"id": "1", "_source_type": "graph"}]
    out = ex._fuse_rrf([a, b], {"vector_1": 2.0, "graph_2": 1.0})
    assert out[0]["_rrf_score"] > 0


def test_apply_joins_and_extract_field():
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    left = [{"id": "1", "x": "a"}]
    right = [{"id": "1", "y": "b"}, {"id": "2"}]
    joined = ex._apply_joins(
        [left, right], [{"join_type": "inner", "left_field": "id", "right_field": "id"}]
    )
    assert joined[0][0]["y"] == "b"
    # < 2 components -> unchanged
    assert ex._apply_joins([left], []) == [left]
    # nested field extraction
    assert ex._extract_field({"a": {"b": 5}}, "a.b") == "5"
    assert ex._extract_field({"a": 1}, "a.missing") is None


def test_apply_time_decay():
    import time as _t

    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)
    now = int(_t.time() * 1e9)
    recs = [
        {"id": "1", "score": 1.0, "timestamp": now - int(3600 * 1e9)},
        {"id": "2", "score": 1.0, "timestamp": now},
        {"id": "3", "score": 1.0},  # no timestamp -> skipped
    ]
    for fn in ("linear", "exponential", "gaussian", "other"):
        out = ex._apply_time_decay(
            [dict(r) for r in recs], (fn, {"reference_time": now, "halflife_hours": 1})
        )
        assert out[0]["id"] in {"1", "2"}


def test_apply_time_decay_enum_value_and_future_ts():
    db = make_started_db()
    ex = EmbeddedMultiModalQueryExecutor(db)

    class F:
        value = "exponential"

    now = 1000
    # future timestamp -> age clamped to 0
    out = ex._apply_time_decay(
        [{"id": "1", "score": 1.0, "timestamp": 2000}],
        (F(), {"reference_time": now, "halflife_hours": 1}),
    )
    assert "_decayed_score" in out[0]


# ---------------------------------------------------------------------------
# execute_multi_modal_query top-level + reranking
# ---------------------------------------------------------------------------


def test_execute_multi_modal_query_no_rerank(patched_http):
    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp(
            {"success": True, "results": [{"id": "z", "score": 0.9}]}
        )
    )
    db = make_started_db()
    q = _mmq([{"type": "vector", "collection": "c", "query_vector": [0.1]}])
    res = run(db.execute_multi_modal_query(q))
    assert res.records[0]["id"] == "z"


def test_execute_multi_modal_query_with_rerank(patched_http):
    from proximadb_sdk.multimodal_query import RerankConfig

    patched_http.responder = staticmethod(
        lambda v, u, **kw: FakeResp(
            {"success": True, "results": [{"id": "z", "score": 0.9}]}
        )
    )
    db = make_started_db()
    q = _mmq([{"type": "vector", "collection": "c", "query_vector": [0.1]}])
    res = run(db.execute_multi_modal_query(q, rerank_config=RerankConfig()))
    assert res.metadata.get("reranked") is True


# ---------------------------------------------------------------------------
# connect_embedded convenience (start patched out)
# ---------------------------------------------------------------------------


def test_connect_embedded(monkeypatch):
    async def fake_start(self, timeout=30.0):
        self._started = True

    monkeypatch.setattr(EmbeddedProximaDB, "start", fake_start)
    db = run(connect_embedded(data_dir="/tmp/ce"))
    assert db._started is True


# ---------------------------------------------------------------------------
# async context manager
# ---------------------------------------------------------------------------


def test_async_context_manager(monkeypatch):
    started = {"v": 0}

    async def fake_start(self, timeout=30.0):
        started["v"] += 1
        self._started = True

    async def fake_stop(self):
        started["v"] += 10

    monkeypatch.setattr(EmbeddedProximaDB, "start", fake_start)
    monkeypatch.setattr(EmbeddedProximaDB, "stop", fake_stop)

    async def go():
        async with EmbeddedProximaDB(data_dir="/tmp/acm") as db:
            assert db._started
        return started["v"]

    assert run(go()) == 11
