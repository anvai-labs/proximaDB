"""Offline unit tests for proximadb_sdk.cli.

The CLI is driven via click's CliRunner with the network-facing client mocked
out. (The historical import-time shims — stand-in ProximaDBConfig/SearchFilter
classes — were removed 2026-11-08 when cli.py was repaired against the real
SDK surface; a bare `import proximadb_sdk.cli` is now pinned by the regression
tests at the bottom of this file.) Render-path contract tests near the bottom
drive the CLI with REAL SDK model instances, pinning that the renderers handle
the pydantic types the clients actually return.

ABSOLUTELY no network / no server / no real client construction happens.

NOTE on coverage measurement: the repo coverage config sets
``source_pkgs = ["proximadb_sdk"]``, so at session end coverage imports EVERY
not-yet-imported module in the package to attribute 0%. In this heavy-ML env
that walk transitively imports chromadb -> OpenTelemetry SDK logs, whose
module-body ``Resource.create()`` spawns a ThreadPoolExecutor that deadlocks
against coverage's tracing thread on the CPython import lock (confirmed via
faulthandler). This deadlock reproduces even with a trivial one-line test file,
so it is environmental and independent of these tests. To measure cli coverage
cleanly, scope coverage to the single file, e.g.::

    coverage run --include="*/proximadb_sdk/cli.py" -m pytest <thisfile>
    coverage report --include="*/proximadb_sdk/cli.py"

which yields 99% (only the ``if __name__ == "__main__"`` guard is unhit).
"""

import os

# ---------------------------------------------------------------------------
# OpenTelemetry neutralization — MUST run before any heavy import.
#
# The repo coverage config measures the whole ``proximadb_sdk`` package, so at
# session end coverage imports every not-yet-imported module to attribute 0%.
# Some transitively-imported package kicks off OpenTelemetry resource detection
# in a background ThreadPoolExecutor; that worker thread and coverage's tracing
# thread deadlock on the CPython import lock (confirmed via faulthandler:
# main thread joins the detector worker forever while a coverage worker blocks
# in importlib `_lock_unlock_module`). Disabling the SDK and the experimental
# resource detectors removes the worker thread, so no deadlock can form.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OTEL_EXPERIMENTAL_RESOURCE_DETECTORS", "")
os.environ.setdefault("OTEL_RESOURCE_DETECTORS", "")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb telemetry off

# Eagerly import the chromadb -> OpenTelemetry SDK logs chain ONCE, here, in the
# import thread, before pytest-cov's concurrent data machinery is active. That
# chain runs `Resource.create()` at module-body time (spawning a ThreadPool that
# otherwise deadlocks against coverage's tracing thread on the import lock). By
# importing it now it is cached in sys.modules, so coverage's end-of-session
# "import every uncovered module" walk never re-executes the offending bodies.
for _mod in (
    "opentelemetry.sdk.resources",
    "opentelemetry.sdk._logs",
    "opentelemetry.sdk._logs._internal",
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    "chromadb",
):
    try:
        __import__(_mod)
    except BaseException:
        pass

import json
import sys
import types

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Direct import — the shims that used to inject ProximaDBConfig/SearchFilter
# stand-ins were removed when cli.py was repaired against the real SDK surface
# (2026-11-08): a published console script (`proximadb-cli`) must import
# WITHOUT test-side rescue, and the regression tests below pin exactly that.
# ---------------------------------------------------------------------------
import proximadb_sdk.cli as cli_mod

cli = cli_mod.cli


@pytest.fixture
def runner():
    return CliRunner()


class FakeClient:
    """Hand fake recording calls and returning canned responses."""

    def __init__(self):
        self.calls = []

    def _record(self, _op, *a, **kw):
        self.calls.append((_op, a, kw))

    def list_collections(self):
        self._record("list_collections")
        return [
            {
                "name": "c1",
                "dimension": 128,
                "vector_count": 10,
                "storage_engine": "sst",
            },
            {"name": "c2"},
        ]

    def create_collection(self, **kw):
        self._record("create_collection", **kw)
        return {"name": kw.get("name"), "status": "created"}

    def delete_collection(self, name):
        self._record("delete_collection", name)
        return None

    def get_collection(self, name):
        self._record("get_collection", name)
        return {"name": name, "dimension": 64}

    def insert_records(self, collection, records):
        self._record("insert_records", collection, records)
        return {"inserted": len(records)}

    def get_vector(self, collection, vector_id):
        self._record("get_vector", collection, vector_id)
        return {"id": vector_id, "vector": [1.0, 2.0]}

    def delete_vectors(self, collection, ids):
        self._record("delete_vectors", collection, ids)
        return {"deleted": len(ids)}

    def search(self, *a, **kw):
        self._record("search", *a, **kw)
        return [
            {"id": "v1", "score": 0.99, "metadata": {"a": 1}},
            {"id": "v2", "score": 0.5, "metadata": {"x": "y" * 100}},
        ]


@pytest.fixture
def fake_client(monkeypatch):
    fc = FakeClient()
    # get_client builds UnifiedProximaDBClient; patch get_client directly so we
    # never touch the real (stale) constructor.
    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **kw: fc)
    return fc


# ---------------------------------------------------------------------------
# Top-level / help
# ---------------------------------------------------------------------------
def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ProximaDB CLI" in result.output


def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0


def test_main_entrypoint(monkeypatch):
    # main() calls cli(obj={}); make it a no-op invocation of --help.
    called = {}

    def fake_cli(**kw):
        called["kw"] = kw

    monkeypatch.setattr(cli_mod, "cli", fake_cli)
    cli_mod.main()
    assert called["kw"] == {"obj": {}}


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------
def test_collections_list_table(runner, fake_client):
    result = runner.invoke(cli, ["collections", "list"])
    assert result.exit_code == 0
    assert "c1" in result.output


def test_collections_list_json(runner, fake_client):
    result = runner.invoke(cli, ["--json-output", "collections", "list"])
    assert result.exit_code == 0
    assert json.loads(result.output)[0]["name"] == "c1"


def test_collections_list_empty(runner, monkeypatch):
    fc = FakeClient()
    fc.list_collections = lambda: []
    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **kw: fc)
    result = runner.invoke(cli, ["collections", "list"])
    assert result.exit_code == 0
    assert "No collections found" in result.output


def test_collections_list_error(runner, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("net down")

    monkeypatch.setattr(cli_mod, "get_client", boom)
    result = runner.invoke(cli, ["collections", "list"])
    assert result.exit_code == 1
    assert "Error" in result.output


def test_collections_create_table(runner, fake_client):
    result = runner.invoke(
        cli,
        ["collections", "create", "mycoll", "-d", "128", "--description", "desc"],
    )
    assert result.exit_code == 0
    assert "created successfully" in result.output
    assert (
        "create_collection",
        (),
        {
            "name": "mycoll",
            "dimension": 128,
            "storage_engine": "sst",
            "description": "desc",
        },
    ) in fake_client.calls


def test_collections_create_json(runner, fake_client):
    result = runner.invoke(
        cli,
        [
            "--json-output",
            "collections",
            "create",
            "mycoll",
            "-d",
            "32",
            "--engine",
            "nova",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["name"] == "mycoll"


def test_collections_create_error(runner, monkeypatch):
    fc = FakeClient()

    def boom(**kw):
        raise RuntimeError("dup")

    fc.create_collection = boom
    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **kw: fc)
    result = runner.invoke(cli, ["collections", "create", "x", "-d", "8"])
    assert result.exit_code == 1
    assert "Error creating collection" in result.output


def test_collections_delete_confirm_yes(runner, fake_client):
    result = runner.invoke(cli, ["collections", "delete", "mycoll"], input="y\n")
    assert result.exit_code == 0
    assert "deleted successfully" in result.output


def test_collections_delete_confirm_no(runner, fake_client):
    result = runner.invoke(cli, ["collections", "delete", "mycoll"], input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert fake_client.calls == []  # never built/called client


def test_collections_delete_force_json(runner, fake_client):
    result = runner.invoke(
        cli, ["--json-output", "collections", "delete", "mycoll", "--force"]
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["status"] == "deleted"


def test_collections_delete_error(runner, monkeypatch):
    fc = FakeClient()

    def boom(name):
        raise RuntimeError("missing")

    fc.delete_collection = boom
    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **kw: fc)
    result = runner.invoke(cli, ["collections", "delete", "x", "-f"])
    assert result.exit_code == 1
    assert "Error deleting collection" in result.output


def test_collections_info_table(runner, fake_client):
    result = runner.invoke(cli, ["collections", "info", "mycoll"])
    assert result.exit_code == 0
    assert "mycoll" in result.output


def test_collections_info_json(runner, fake_client):
    result = runner.invoke(cli, ["--json-output", "collections", "info", "mycoll"])
    assert result.exit_code == 0
    assert json.loads(result.output)["name"] == "mycoll"


def test_collections_info_error(runner, monkeypatch):
    fc = FakeClient()
    fc.get_collection = lambda name: (_ for _ in ()).throw(RuntimeError("x"))
    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **kw: fc)
    result = runner.invoke(cli, ["collections", "info", "x"])
    assert result.exit_code == 1
    assert "Error" in result.output


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------
def test_vectors_insert_single(runner, fake_client):
    result = runner.invoke(
        cli,
        [
            "vectors",
            "insert",
            "-c",
            "coll",
            "-v",
            "[1.0, 2.0, 3.0]",
            "--id",
            "v1",
            "-m",
            '{"k": "v"}',
        ],
    )
    assert result.exit_code == 0
    assert "Successfully inserted 1" in result.output


def test_vectors_insert_single_json(runner, fake_client):
    result = runner.invoke(
        cli,
        ["--json-output", "vectors", "insert", "-c", "coll", "-v", "[1.0]"],
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["inserted"] == 1


def test_vectors_insert_file(runner, fake_client, tmp_path):
    f = tmp_path / "vecs.json"
    f.write_text(
        json.dumps(
            [
                {"id": "a", "vector": [1.0], "metadata": {"m": 1}},
                {"id": "b", "vector": [2.0]},
            ]
        )
    )
    result = runner.invoke(cli, ["vectors", "insert", "-c", "coll", "-f", str(f)])
    assert result.exit_code == 0
    assert "Successfully inserted 2" in result.output
    # metadata key normalized to props
    name, args, kw = next(c for c in fake_client.calls if c[0] == "insert_records")
    assert args[1][0]["props"] == {"m": 1}


def test_vectors_insert_missing_input(runner, fake_client):
    result = runner.invoke(cli, ["vectors", "insert", "-c", "coll"])
    assert result.exit_code == 1
    assert "Either --file or --vector" in result.output


def test_vectors_insert_error(runner, fake_client):
    # invalid JSON for --vector triggers exception path
    result = runner.invoke(cli, ["vectors", "insert", "-c", "coll", "-v", "not-json"])
    assert result.exit_code == 1
    assert "Error inserting vectors" in result.output


def test_vectors_get_table(runner, fake_client):
    result = runner.invoke(cli, ["vectors", "get", "-c", "coll", "v1"])
    assert result.exit_code == 0
    assert "v1" in result.output


def test_vectors_get_json(runner, fake_client):
    result = runner.invoke(cli, ["--json-output", "vectors", "get", "-c", "coll", "v1"])
    assert result.exit_code == 0
    assert json.loads(result.output)["id"] == "v1"


def test_vectors_get_error(runner, monkeypatch):
    fc = FakeClient()
    fc.get_vector = lambda c, i: (_ for _ in ()).throw(RuntimeError("nope"))
    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **kw: fc)
    result = runner.invoke(cli, ["vectors", "get", "-c", "coll", "v1"])
    assert result.exit_code == 1
    assert "Error" in result.output


def test_vectors_delete_no_ids(runner, fake_client):
    result = runner.invoke(cli, ["vectors", "delete", "-c", "coll"])
    assert result.exit_code == 1
    assert "At least one vector ID" in result.output


def test_vectors_delete_confirm_no(runner, fake_client):
    result = runner.invoke(
        cli, ["vectors", "delete", "-c", "coll", "v1", "v2"], input="n\n"
    )
    assert result.exit_code == 0
    assert "Aborted" in result.output


def test_vectors_delete_confirm_yes(runner, fake_client):
    # user confirms at the prompt -> proceeds to delete (closes the
    # confirm-returns-True branch in delete_vectors).
    result = runner.invoke(
        cli, ["vectors", "delete", "-c", "coll", "v1", "v2"], input="y\n"
    )
    assert result.exit_code == 0
    assert "Deleted 2" in result.output


def test_vectors_delete_force(runner, fake_client):
    result = runner.invoke(cli, ["vectors", "delete", "-c", "coll", "v1", "v2", "-f"])
    assert result.exit_code == 0
    assert "Deleted 2" in result.output


def test_vectors_delete_force_json(runner, fake_client):
    result = runner.invoke(
        cli, ["--json-output", "vectors", "delete", "-c", "coll", "v1", "-f"]
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["deleted"] == 1


def test_vectors_delete_error(runner, monkeypatch):
    fc = FakeClient()
    fc.delete_vectors = lambda c, ids: (_ for _ in ()).throw(RuntimeError("x"))
    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **kw: fc)
    result = runner.invoke(cli, ["vectors", "delete", "-c", "coll", "v1", "-f"])
    assert result.exit_code == 1
    assert "Error" in result.output


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def test_search_table(runner, fake_client):
    result = runner.invoke(cli, ["search", "-c", "coll", "-q", "[1.0, 2.0]", "-k", "5"])
    assert result.exit_code == 0
    assert "Search Results" in result.output


def test_search_with_filter(runner, fake_client):
    result = runner.invoke(
        cli,
        [
            "search",
            "-c",
            "coll",
            "-q",
            "[1.0]",
            "-f",
            '{"field": "x"}',
        ],
    )
    assert result.exit_code == 0
    name, args, kw = next(c for c in fake_client.calls if c[0] == "search")
    # The repaired call matches the client's real signature: positional
    # (collection, vector) + metadata_filter dict.
    assert args[0] == "coll"
    assert args[1] == [1.0]
    assert kw["metadata_filter"] == {"field": "x"}


def test_search_json(runner, fake_client):
    result = runner.invoke(
        cli, ["--json-output", "search", "-c", "coll", "-q", "[1.0]"]
    )
    assert result.exit_code == 0
    assert json.loads(result.output)[0]["id"] == "v1"


def test_search_error(runner, fake_client):
    result = runner.invoke(cli, ["search", "-c", "coll", "-q", "bad-json"])
    assert result.exit_code == 1
    assert "Error" in result.output


# ---------------------------------------------------------------------------
# Server commands (httpx mocked)
# ---------------------------------------------------------------------------
class FakeResp:
    def __init__(self, status_code=200, data=None, text=None):
        self.status_code = status_code
        self._data = data or {"status": "ok"}
        self.text = text if text is not None else json.dumps(self._data)

    def json(self):
        return self._data


def _patch_httpx(monkeypatch, resp):
    fake_httpx = types.SimpleNamespace(get=lambda *a, **kw: resp)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)


def test_server_health_healthy_table(runner, monkeypatch):
    _patch_httpx(monkeypatch, FakeResp(200, {"version": "0.2"}))
    result = runner.invoke(cli, ["server", "health"])
    assert result.exit_code == 0
    assert "healthy" in result.output


def test_server_health_healthy_json(runner, monkeypatch):
    _patch_httpx(monkeypatch, FakeResp(200, {"version": "0.2"}))
    result = runner.invoke(cli, ["--json-output", "server", "health"])
    assert result.exit_code == 0
    assert json.loads(result.output)["status"] == "healthy"


def test_server_health_bad_status(runner, monkeypatch):
    _patch_httpx(monkeypatch, FakeResp(503, {"err": "down"}))
    result = runner.invoke(cli, ["server", "health"])
    assert result.exit_code == 0
    assert "503" in result.output


def test_server_health_unreachable(runner, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("conn refused")

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(get=boom))
    result = runner.invoke(cli, ["server", "health"])
    assert result.exit_code == 1
    assert "unreachable" in result.output


def test_server_info_table(runner, monkeypatch):
    _patch_httpx(monkeypatch, FakeResp(200, {"name": "proximadb", "version": "0.2"}))
    result = runner.invoke(cli, ["server", "info"])
    assert result.exit_code == 0
    assert "Server Information" in result.output


def test_server_info_json(runner, monkeypatch):
    _patch_httpx(monkeypatch, FakeResp(200, text='{"raw": true}'))
    result = runner.invoke(cli, ["--json-output", "server", "info"])
    assert result.exit_code == 0
    assert '"raw"' in result.output


def test_server_info_error(runner, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("x")

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(get=boom))
    result = runner.invoke(cli, ["server", "info"])
    assert result.exit_code == 1
    assert "Error" in result.output


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def test_benchmark_table(runner, fake_client):
    result = runner.invoke(
        cli, ["benchmark", "-c", "bench", "-d", "4", "-n", "3", "-q", "2"]
    )
    assert result.exit_code == 0
    assert "Benchmark Results" in result.output


def test_benchmark_json_existing_collection(runner, monkeypatch):
    fc = FakeClient()
    # create_collection raises -> "using existing collection" branch
    fc.create_collection = lambda **kw: (_ for _ in ()).throw(RuntimeError("exists"))
    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **kw: fc)
    result = runner.invoke(
        cli,
        ["--json-output", "benchmark", "-c", "bench", "-d", "4", "-n", "2", "-q", "2"],
    )
    assert result.exit_code == 0
    assert "insert_total_time" in result.output


def test_benchmark_error(runner, monkeypatch):
    fc = FakeClient()
    fc.create_collection = lambda **kw: None
    fc.insert_records = lambda c, r: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **kw: fc)
    result = runner.invoke(
        cli, ["benchmark", "-c", "bench", "-d", "4", "-n", "2", "-q", "2"]
    )
    assert result.exit_code == 1
    assert "Benchmark failed" in result.output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_record_payload_normalizes_metadata():
    out = cli_mod._record_payload({"id": "x", "metadata": {"a": 1}})
    assert out["props"] == {"a": 1}
    assert "metadata" not in out


def test_record_payload_keeps_props():
    out = cli_mod._record_payload({"id": "x", "props": {"a": 1}})
    assert out["props"] == {"a": 1}


def test_get_client_builds_unified(monkeypatch):
    captured = {}

    class FakeUnified:
        def __init__(self, url=None, protocol=None, port_mode=None, **kw):
            captured["url"] = url
            captured["proto"] = protocol
            captured["mode"] = port_mode
            captured["rest"] = kw

    monkeypatch.setattr(cli_mod, "UnifiedProximaDBClient", FakeUnified)
    client = cli_mod.get_client("h", 5678, 5678, "rest", 5.0)
    assert isinstance(client, FakeUnified)
    assert captured["proto"] == "rest"
    assert captured["url"] == "http://h:5678"
    assert captured["rest"]["timeout"] == 5.0


def test_get_client_multi_port_mode(monkeypatch):
    captured = {}

    class FakeUnified:
        def __init__(self, url=None, protocol=None, port_mode=None, **kw):
            captured["mode"] = port_mode

    monkeypatch.setattr(cli_mod, "UnifiedProximaDBClient", FakeUnified)
    cli_mod.get_client("h", 5678, 5679, "grpc", 5.0)
    assert captured["mode"] is not None
    assert "MULTI" in str(captured["mode"])


# ---------------------------------------------------------------------------
# Regression: the CLI is a published console script (proximadb-cli / pdb-cli in
# pyproject) — it must import and render --help with NO test-side shims.
# These tests fail at import-collection time if cli.py drifts from the SDK
# surface again (the previous failure mode: ImportError, invisible to CI
# because this file's old shims masked it).
# ---------------------------------------------------------------------------
def test_cli_imports_without_shims():
    import subprocess
    import sys as _sys
    from pathlib import Path

    # Point the child interpreter at THIS source tree (not any installed
    # wheel — the previously published wheel carries the same breakage).
    src = Path(__file__).resolve().parents[2] / "src"
    assert (src / "proximadb_sdk" / "cli.py").is_file()

    proc = subprocess.run(
        [_sys.executable, "-c", "import proximadb_sdk.cli"],
        capture_output=True,
        text=True,
        env={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("OTEL_", "ANONYMIZED_", "PROXIMADB_"))
        }
        | {
            "OTEL_SDK_DISABLED": "true",
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(src),
        },
        timeout=60,
    )
    assert proc.returncode == 0, f"bare import failed:\n{proc.stderr}"


def test_cli_help_smoke(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


# ---------------------------------------------------------------------------
# Render-path CONTRACT tests: the real clients return pydantic models, not
# dicts — these pin that the CLI's renderers handle the real types (the
# 2026-11-08 review proved every read command crashed on .get()/json.dumps
# against healthy servers before the _jsonable normalization).
# ---------------------------------------------------------------------------
class _ModelClient:
    def __init__(self, *args, **kwargs):
        pass

    def list_collections(self):
        from proximadb_sdk.models import Collection, CollectionConfig

        return [Collection(id="c1", config=CollectionConfig(name="c1", dimension=4))]

    def search(self, collection_id, vector, top_k=10, metadata_filter=None, **kw):
        from proximadb_sdk.models import SearchResult

        return [SearchResult(id="v1", score=0.9, rank=1)]


def test_render_handles_real_models_table(runner, monkeypatch):
    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **k: _ModelClient())
    result = runner.invoke(cli, ["collections", "list"])
    assert result.exit_code == 0, result.output
    assert "c1" in result.output


def test_render_handles_real_models_json(runner, monkeypatch):
    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **k: _ModelClient())
    result = runner.invoke(
        cli, ["--json-output", "search", "-c", "coll", "-q", "[1.0]"]
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed[0]["id"] == "v1"
    assert parsed[0]["score"] == 0.9


def test_render_handles_real_models_create_json(runner, monkeypatch):
    """Round-2 pin: create_collection returns a Collection model — the JSON
    path must render it (previously exit-1 AFTER a successful create)."""
    from proximadb_sdk.models import Collection, CollectionConfig

    class _CreateClient:
        def create_collection(self, **kwargs):
            return Collection(
                id=kwargs["name"],
                config=CollectionConfig(name=kwargs["name"], dimension=4),
            )

    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **k: _CreateClient())
    result = runner.invoke(
        cli, ["--json-output", "collections", "create", "-d", "4", "c9"]
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["id"] == "c9"
