"""End-to-end fp16 collection validation through the **Python SDK over
gRPC** against a real ProximaDB server.

Closes the cross-language gRPC gap: today the Python gRPC adapter
(`adapters/grpc_adapter.py::create_collection`) and the sync protocol
layer (`protocols/grpc_sync.py::create_collection`) build the proto
`CollectionConfig` by hand and silently drop
`canonical_embedding_precision`. This is a real bug — the field reaches
the wire from REST but not from gRPC.

TDD: this test asserts the catalog row reports Fp16 after a Python SDK
gRPC `create_collection` call with `EmbeddingPrecision.FP16`. Should fail
until both layers forward the field.
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
    global _server_process

    os.system("pkill -f proximadb-server")
    time.sleep(1)

    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(test_dir, "../../../.."))
    server_binary = os.path.join(project_root, "target/release/proximadb-server")
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
def test_python_sdk_creates_fp16_collection_via_grpc() -> None:
    """SDK → gRPC → server → catalog must preserve fp16 end-to-end."""
    client = ProximaDBClient("grpc://localhost:5679", protocol="grpc")

    name = f"py_grpc_fp16_{int(time.time() * 1_000_000)}"
    config = CollectionConfig(
        name=name,
        dimension=8,
        canonical_embedding_precision=EmbeddingPrecision.FP16,
    )

    client.create_collection(name=name, config=config)

    # Cross-protocol verification via REST — we want the SERVER's record.
    resp = requests.get(f"http://localhost:5678/api/v1/collections/{name}", timeout=10)
    assert resp.ok, f"REST GET failed: status={resp.status_code}, body={resp.text}"
    body = resp.json()

    cfg = (body.get("collection") or {}).get("config") or body.get("config")
    assert cfg is not None, f"missing collection.config in body: {body}"

    precision = cfg.get("canonical_embedding_precision")
    assert precision in (
        2,
        "EMBEDDING_PRECISION_FP16",
        "FP16",
        "Fp16",
        "fp16",
    ), (
        f"SDK gRPC create_collection with EmbeddingPrecision.FP16 must persist "
        f"as Fp16 in the catalog; REST GET returned {precision!r}"
    )
