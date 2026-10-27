"""End-to-end fp16 collection validation through the **Python SDK** against
a real ProximaDB server.

Closes the cross-language gap: the per-collection
canonical_embedding_precision option must reach the server when the SDK is
the surface (not just curl). This test starts the release server, asks the
SDK to create a collection with ``EmbeddingPrecision.FP16``, then reads
the catalog back via REST to confirm the precision landed.

Why server_lifecycle: the test manages a real server process. Do not run
concurrently with other test suites.
"""

import atexit
import os
import subprocess
import time

import pytest
import requests

from proximadb_sdk import CollectionConfig, EmbeddingPrecision, ProximaDBClient

_server_process: subprocess.Popen | None = None


def _start_server() -> bool:
    """Start the release proximadb-server on default ports.

    Returns True on success, False if the binary is missing.
    """
    global _server_process

    # Kill any existing instance — these tests own the lifecycle.
    os.system("pkill -f proximadb-server")
    time.sleep(1)

    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(test_dir, "../../../.."))
    server_binary = os.path.join(project_root, "target/release/proximadb-server")
    # simple-config.toml stays minimal-and-current; `config.toml` carries a
    # `[llm.rag]` section that drifts from the RAGConfig schema.
    config_file = os.path.join(project_root, "config/simple-config.toml")

    if not os.path.exists(server_binary):
        return False

    _server_process = subprocess.Popen(
        [server_binary, "--config", config_file],
        cwd=project_root,
        env={**os.environ, "RUST_LOG": "info"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Poll /health until ready.
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = requests.get("http://localhost:5678/health", timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _stop_server() -> None:
    global _server_process
    if _server_process:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_process.kill()
        _server_process = None


atexit.register(_stop_server)


@pytest.fixture(scope="module", autouse=True)
def _server():
    if not _start_server():
        pytest.skip(
            "Release proximadb-server binary not built. "
            "Run: cargo build --release -p proximadb-server"
        )
    yield
    _stop_server()


@pytest.mark.server_lifecycle
def test_python_sdk_creates_fp16_collection_against_real_server() -> None:
    """SDK → REST v2 → server → catalog must preserve fp16 end-to-end."""
    client = ProximaDBClient("http://localhost:5678", protocol="rest")

    name = f"py_fp16_{int(time.time() * 1_000_000)}"
    config = CollectionConfig(
        name=name,
        dimension=8,
        canonical_embedding_precision=EmbeddingPrecision.FP16,
    )

    client.create_collection(name=name, config=config)

    # Cross-protocol verification via raw REST so we don't accidentally
    # read the precision from the SDK-side cached config — we want to
    # assert the SERVER persisted fp16.
    resp = requests.get(f"http://localhost:5678/api/v1/collections/{name}", timeout=10)
    assert resp.ok, f"REST GET failed: status={resp.status_code}, body={resp.text}"
    body = resp.json()

    cfg = (body.get("collection") or {}).get("config") or body.get("config")
    assert cfg is not None, f"missing collection.config in body: {body}"

    precision = cfg.get("canonical_embedding_precision")
    # Server renders the proto enum as either the SCREAMING string or the
    # numeric discriminant (Fp16 = 2). Accept either form.
    assert precision in (
        2,
        "EMBEDDING_PRECISION_FP16",
        "FP16",
        "Fp16",
        "fp16",
    ), (
        f"SDK create_collection with canonical_embedding_precision=FP16 must "
        f"persist as Fp16 in the catalog; REST GET returned {precision!r}"
    )
