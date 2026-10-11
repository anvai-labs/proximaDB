"""Offline contract tests for the tracked benchmark corpus transport."""

from __future__ import annotations

import importlib.util
import json
import re
import tomllib
from argparse import Namespace
from pathlib import Path

import pytest

from proximadb_sdk.chunking_strategies import (
    CompositeInputContract,
    InputRenderer,
    ResolvedInputContract,
)


def _load_builder_module():
    repo = Path(__file__).resolve().parents[4]
    path = repo / "scripts" / "bench" / "build_token_budget_corpus.py"
    spec = importlib.util.spec_from_file_location("build_token_budget_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_exporter_module():
    repo = Path(__file__).resolve().parents[4]
    path = repo / "scripts" / "bench" / "export_source_tree_jsonl.py"
    spec = importlib.util.spec_from_file_location("export_source_tree_jsonl", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_embedder_module():
    repo = Path(__file__).resolve().parents[4]
    path = repo / "scripts" / "bench" / "embed_open_corpus.py"
    spec = importlib.util.spec_from_file_location("embed_open_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_scorer_module():
    repo = Path(__file__).resolve().parents[4]
    path = repo / "scripts" / "bench" / "score_embedding_shards.py"
    spec = importlib.util.spec_from_file_location("score_embedding_shards", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embeddings_extra_includes_trusted_open_model_runtime_dependencies():
    repo = Path(__file__).resolve().parents[4]
    pyproject = tomllib.loads(
        (repo / "clients" / "python" / "pyproject.toml").read_text(encoding="utf-8")
    )

    requirements = pyproject["project"]["optional-dependencies"]["embeddings"]

    assert any(requirement.startswith("einops") for requirement in requirements)


def _load_tokenizer_validator_module():
    repo = Path(__file__).resolve().parents[4]
    path = repo / "scripts" / "bench" / "validate_open_embedding_tokenizers.py"
    spec = importlib.util.spec_from_file_location(
        "validate_open_embedding_tokenizers", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WordCounter:
    name = "test/word-counter"
    fingerprint = "word-counter-v1"
    advertised_limit = 8

    @staticmethod
    def count(text: str) -> int:
        return len(list(re.finditer(r"\S+", text))) + 2

    @staticmethod
    def content_offsets(text: str):
        return tuple(match.span() for match in re.finditer(r"\S+", text))


def _args(tmp_path: Path, *, max_chunks: int | None) -> Namespace:
    source = tmp_path / "documents.jsonl"
    source.write_text(
        json.dumps(
            {"id": "doc-1", "text": "one two three four five six seven eight nine"}
        )
        + "\n"
        + json.dumps({"id": "doc-2", "text": "ten eleven twelve"})
        + "\n",
        encoding="utf-8",
    )
    contracts = tmp_path / "unused.toml"
    contracts.write_text(
        f"""\
[[models]]
model_id = "test/model"
revision = "{'a' * 40}"
effective_context_limit = 8
native_dimension = 8
output_dimension = 8
""",
        encoding="utf-8",
    )
    return Namespace(
        input=source,
        output_dir=tmp_path / "corpus",
        contracts=contracts,
        text_field="text",
        context_field=None,
        context_template="{context}\n\n{text}",
        id_field="id",
        strategy="fixed_size",
        target_tokens=7,
        overlap_tokens=1,
        min_content_tokens=3,
        boundary_char_size=10_000,
        overflow_policy="split",
        short_chunk_policy="keep",
        max_chunks=max_chunks,
        allow_partial_corpus=False,
        deduplicate="exact",
    )


def test_builder_propagates_context_inside_the_exact_model_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    builder = _load_builder_module()
    args = _args(tmp_path, max_chunks=None)
    args.input.write_text(
        json.dumps(
            {
                "id": "doc-1",
                "body": "one two three four five six seven",
                "title": "Useful heading",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args.text_field = "body"
    args.context_field = "title"
    args.context_template = "Title: {context}\n\n{text}"
    args.target_tokens = 8
    args.overlap_tokens = 1
    args.min_content_tokens = 1
    args.deduplicate = "none"
    monkeypatch.setattr(builder, "_load_contracts", lambda _path: _contract())

    manifest = builder.build(args)

    texts = json.loads((args.output_dir / "texts.json").read_text(encoding="utf-8"))
    chunks = [
        json.loads(line)
        for line in (args.output_dir / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(texts) > 1
    assert all(text.startswith("Title: Useful heading\n\n") for text in texts)
    assert all(chunk["context_propagated"] for chunk in chunks)
    assert all(chunk["token_counts"]["test/model"] <= 8 for chunk in chunks)
    assert manifest["context_propagation"] == {
        "metadata_key": "title",
        "template": "Title: {context}\n\n{text}",
    }
    assert manifest["source_fields"] == {
        "context": "title",
        "id": "id",
        "text": "body",
    }


def _contract() -> CompositeInputContract:
    return CompositeInputContract(
        (
            ResolvedInputContract(
                model_id="test/model",
                model_revision="immutable-revision",
                counter=WordCounter(),
                effective_context_limit=8,
                renderer=InputRenderer(document_template="passage: {text}"),
                native_dimension=8,
            ),
        )
    )


def test_builder_resolves_sentence_transformer_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    builder = _load_builder_module()
    revision = "a" * 40
    contracts_path = tmp_path / "contracts.toml"
    contracts_path.write_text(
        f"""\
[[models]]
model_id = "test/model"
revision = "{revision}"
effective_context_limit = 8
document_template = "passage: {{text}}"
query_template = "query: {{text}}"
native_dimension = 8
output_dimension = 8
runtime_provider = "open-weights"
local_files_only = true
""",
        encoding="utf-8",
    )
    runtime_contract = ResolvedInputContract(
        model_id="test/model",
        model_revision=revision,
        counter=WordCounter(),
        effective_context_limit=8,
        renderer=InputRenderer(
            document_template="passage: {text}", query_template="query: {text}"
        ),
        native_dimension=8,
        output_dimension=8,
    )
    calls = []

    class RuntimeProvider:
        @staticmethod
        def get_input_contract():
            return runtime_contract

    def create_provider(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return RuntimeProvider()

    monkeypatch.setattr(builder, "create_open_model_provider", create_provider)

    loaded = builder._load_contracts(contracts_path)

    assert loaded.contracts == (runtime_contract,)
    assert calls == [
        (
            "test/model",
            {
                "revision": revision,
                "dimension": 8,
                "document_template": "passage: {text}",
                "query_template": "query: {text}",
                "device": "cpu",
                "extra": {"local_files_only": True},
            },
        )
    ]


def test_builder_rejects_runtime_contract_declaration_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    builder = _load_builder_module()
    revision = "b" * 40
    contracts_path = tmp_path / "contracts.toml"
    contracts_path.write_text(
        f"""\
[[models]]
model_id = "test/model"
revision = "{revision}"
effective_context_limit = 512
native_dimension = 8
output_dimension = 8
runtime_provider = "open-weights"
""",
        encoding="utf-8",
    )

    class RuntimeProvider:
        @staticmethod
        def get_input_contract():
            return ResolvedInputContract(
                model_id="test/model",
                model_revision=revision,
                counter=WordCounter(),
                effective_context_limit=8,
                native_dimension=8,
                output_dimension=8,
            )

    monkeypatch.setattr(
        builder,
        "create_open_model_provider",
        lambda *_args, **_kwargs: RuntimeProvider(),
    )

    with pytest.raises(ValueError, match="effective_context_limit"):
        builder._load_contracts(contracts_path)


def test_builder_reuses_a_sealed_runtime_contract_without_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    builder = _load_builder_module()
    revision = "c" * 40
    expected = ResolvedInputContract(
        model_id="test/model",
        model_revision=revision,
        counter=WordCounter(),
        effective_context_limit=8,
        renderer=InputRenderer(
            document_template="passage: {text}", query_template="query: {text}"
        ),
        native_dimension=8,
        output_dimension=8,
    )
    contracts_path = tmp_path / "contracts.toml"
    contracts_path.write_text(
        f"""\
[[models]]
model_id = "test/model"
revision = "{revision}"
tokenizer_revision = "{revision}"
effective_context_limit = 8
document_template = "passage: {{text}}"
query_template = "query: {{text}}"
native_dimension = 8
output_dimension = 8
runtime_provider = "open-weights"
sealed_runtime_contract_fingerprint = "{expected.fingerprint}"
local_files_only = true
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        builder.HuggingFaceTokenCounter,
        "from_pretrained",
        lambda *_args, **_kwargs: WordCounter(),
    )
    monkeypatch.setattr(
        builder,
        "create_open_model_provider",
        lambda *_args, **_kwargs: pytest.fail("sealed contracts must not load weights"),
    )

    loaded = builder._load_contracts(contracts_path)

    assert loaded.contracts == (expected,)
    assert builder._contract_resolution_manifest(contracts_path) == {
        "contracts_sha256": builder._sha256(contracts_path),
        "models": {
            "test/model": {
                "mode": "sealed_runtime_fingerprint",
                "sealed_runtime_contract_fingerprint": expected.fingerprint,
            }
        },
        "path": str(contracts_path.resolve()),
    }


def test_builder_rejects_an_accidentally_partial_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "_load_contracts", lambda _path: _contract())

    with pytest.raises(ValueError, match="partial corpus"):
        builder.build(_args(tmp_path, max_chunks=2))

    assert not (tmp_path / "corpus" / "texts.json").exists()
    assert not (tmp_path / "corpus" / "chunks.jsonl").exists()


def test_builder_caps_deterministically_only_when_partial_scope_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "_load_contracts", lambda _path: _contract())
    args = _args(tmp_path, max_chunks=2)
    args.allow_partial_corpus = True

    manifest = builder.build(args)

    texts = json.loads((tmp_path / "corpus" / "texts.json").read_text())
    assert len(texts) == 2
    assert manifest["chunk_count"] == 2
    assert manifest["max_chunks"] == 2
    assert manifest["stopped_at_max_chunks"] is True
    assert manifest["corpus_scope"] == "partial_prefix"
    assert manifest["zero_truncation_asserted"] is True
    assert manifest["builder_sha256"] == builder._sha256(Path(builder.__file__))
    assert manifest["sdk_chunking_package_sha256"] == builder._python_tree_sha256(
        Path(builder.chunking_package.__file__).resolve().parent
    )
    assert manifest["token_histogram"]["test/model"]["max"] <= 7
    assert manifest["token_histogram"]["test/model"]["over_target"] == 0
    assert manifest["packing_histogram"]["new_content_tokens"]["min"] > 0
    assert manifest["packing_histogram"]["actual_overlap_tokens"]["max"] <= 1
    assert manifest["token_budget"]["overflow_policy"] == "split"
    assert manifest["deduplication_policy"] == "exact"


def test_builder_records_and_removes_exact_duplicate_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "_load_contracts", lambda _path: _contract())
    args = _args(tmp_path, max_chunks=None)
    args.input.write_text(
        json.dumps({"id": "a", "text": "one two three"})
        + "\n"
        + json.dumps({"id": "b", "text": "one two three"})
        + "\n",
        encoding="utf-8",
    )

    manifest = builder.build(args)

    assert json.loads((tmp_path / "corpus" / "texts.json").read_text()) == [
        "one two three"
    ]
    assert manifest["duplicate_chunk_count"] == 1


def test_builder_rejects_a_non_positive_corpus_cap(tmp_path: Path):
    builder = _load_builder_module()
    with pytest.raises(ValueError, match="max_chunks must be positive"):
        builder.build(_args(tmp_path, max_chunks=0))


def test_builder_rejects_an_empty_emitted_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "_load_contracts", lambda _path: _contract())
    args = _args(tmp_path, max_chunks=None)
    args.input.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="emitted no chunks"):
        builder.build(args)


def test_source_export_is_stable_sorted_and_excludes_skipped_or_symlinked_inputs(
    tmp_path: Path,
):
    exporter = _load_exporter_module()
    code_root = tmp_path / "code"
    repo = code_root / "repo-a"
    (repo / "src").mkdir(parents=True)
    (repo / "target").mkdir()
    (repo / "src" / "z.rs").write_text("fn z() {}\n", encoding="utf-8")
    (repo / "src" / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    (repo / "src" / "ignored.bin").write_bytes(b"binary")
    (repo / "target" / "generated.rs").write_text("generated", encoding="utf-8")
    (repo / "src" / "link.rs").symlink_to(repo / "src" / "z.rs")
    output = tmp_path / "sources.jsonl"
    args = Namespace(
        code_root=code_root,
        repositories=["repo-a"],
        output=output,
        manifest=None,
        extensions=[".rs", ".py"],
        skip_dirs=["target"],
    )

    first = exporter.export(args)
    first_bytes = output.read_bytes()
    second = exporter.export(args)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["id"] for record in records] == [
        "repo-a/src/a.py",
        "repo-a/src/z.rs",
    ]
    assert first_bytes == output.read_bytes()
    assert first["source_inventory_sha256"] == second["source_inventory_sha256"]
    assert first["jsonl_sha256"] == second["jsonl_sha256"]
    assert first["document_count"] == 2
    assert first["exporter_sha256"] == exporter._sha256(Path(exporter.__file__))
    assert first["repository_provenance"]["repo-a"]["vcs"] == "unversioned"


def test_source_export_records_git_revision_and_dirty_state(tmp_path: Path):
    exporter = _load_exporter_module()
    code_root = tmp_path / "code"
    repo = code_root / "repo-a"
    repo.mkdir(parents=True)
    (repo / "tracked.py").write_text("print('clean')\n", encoding="utf-8")
    output = tmp_path / "sources.jsonl"
    args = Namespace(
        code_root=code_root,
        repositories=["repo-a"],
        output=output,
        manifest=None,
        extensions=[".py"],
        skip_dirs=[],
    )

    import subprocess

    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "tracked.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    clean = exporter.export(args)
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert clean["repository_provenance"]["repo-a"] == {
        "dirty": False,
        "head_revision": revision,
        "status_sha256": exporter.hashlib.sha256(b"").hexdigest(),
        "vcs": "git",
    }

    (repo / "tracked.py").write_text("print('dirty')\n", encoding="utf-8")
    dirty = exporter.export(args)
    assert dirty["repository_provenance"]["repo-a"]["dirty"] is True
    assert dirty["repository_provenance"]["repo-a"]["head_revision"] == revision


def test_source_export_fails_closed_on_missing_or_unreadable_sources(tmp_path: Path):
    exporter = _load_exporter_module()
    code_root = tmp_path / "code"
    code_root.mkdir()
    output = tmp_path / "sources.jsonl"
    args = Namespace(
        code_root=code_root,
        repositories=["missing"],
        output=output,
        manifest=None,
        extensions=[".py"],
        skip_dirs=[],
        allow_unreadable_sources=False,
    )

    with pytest.raises(ValueError, match="missing source repositories"):
        exporter.export(args)

    repo = code_root / "repo-a"
    repo.mkdir()
    (repo / "invalid.py").write_bytes(b"valid-prefix\xffinvalid")
    args.repositories = ["repo-a"]
    with pytest.raises(ValueError, match="unreadable source files"):
        exporter.export(args)
    assert not output.exists()


def test_embedding_shards_are_content_and_contract_addressed(tmp_path: Path):
    embedder = _load_embedder_module()
    runtime = {
        "backend": "torch",
        "compute_dtype": "float32",
        "device": "mps",
        "requested_device": "mps",
    }
    paths = embedder.prepare_shards(
        output_dir=tmp_path,
        corpus_sha256="a" * 64,
        model_id="BAAI/bge-base-en-v1.5",
        revision="b" * 40,
        contract_fingerprint="c" * 64,
        dimension=3,
        shard_size=2,
        rows=5,
        runtime=runtime,
    )
    assert [path.name for path in paths] == [
        "00000000.npy",
        "00000002.npy",
        "00000004.npy",
    ]
    assert "bge-base-en-v1.5" in str(paths[0].parent)

    import numpy as np

    np.save(paths[0], np.zeros((2, 3), dtype=np.float32))
    pending = embedder.pending_shards(paths, rows=5, dimension=3, shard_size=2)
    assert pending == [(2, paths[1]), (4, paths[2])]

    query_paths = embedder.prepare_shards(
        output_dir=tmp_path,
        corpus_sha256="a" * 64,
        model_id="BAAI/bge-base-en-v1.5",
        revision="b" * 40,
        contract_fingerprint="c" * 64,
        dimension=3,
        shard_size=2,
        rows=5,
        input_role="query",
        runtime=runtime,
    )
    assert query_paths[0].parent != paths[0].parent
    identity = json.loads((paths[0].parent / "manifest.json").read_text())
    assert identity["runtime"] == runtime
    assert "mps" in str(paths[0].parent)
    cpu_paths = embedder.prepare_shards(
        output_dir=tmp_path,
        corpus_sha256="a" * 64,
        model_id="BAAI/bge-base-en-v1.5",
        revision="b" * 40,
        contract_fingerprint="c" * 64,
        dimension=3,
        shard_size=2,
        rows=5,
        runtime={**runtime, "device": "cpu", "requested_device": "cpu"},
    )
    assert cpu_paths[0].parent != paths[0].parent


def test_tokenizer_validator_isolates_unknown_models(monkeypatch: pytest.MonkeyPatch):
    validator = _load_tokenizer_validator_module()

    class Metadata:
        tokenizer_name = None
        name = "known/model"
        revision = "revision"
        max_length = 8
        document_template = "{text}"
        query_template = None

    class Spec:
        metadata = Metadata()
        trust_remote_code = False

    class Counter:
        advertised_limit = 8
        fingerprint = "fingerprint"

        @staticmethod
        def count(_text):
            return 3

        @staticmethod
        def content_offsets(_text):
            return ((0, 5),)

    def get_spec(model_id):
        if model_id == "unknown/model":
            raise ValueError("unknown open embedding model")
        return Spec()

    monkeypatch.setattr(validator, "get_open_model_spec", get_spec)
    monkeypatch.setattr(
        validator.HuggingFaceTokenCounter,
        "from_pretrained",
        lambda *args, **kwargs: Counter(),
    )

    results = validator.validate_models(
        ["unknown/model", "known/model"], local_files_only=True
    )

    assert [row["status"] for row in results] == ["error", "ok"]
    assert results[0]["model_id"] == "unknown/model"
    assert results[1]["tokenizer_fingerprint"] == "fingerprint"


def test_embedding_finalizer_emits_query_base_headers_and_pca_spectrum(tmp_path: Path):
    embedder = _load_embedder_module()
    import numpy as np

    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    np.save(second, np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32))

    result = embedder.finalize_embeddings(
        [first, second],
        output_dir=tmp_path / "out",
        prefix="tiny",
        dimension=2,
        query_rows=1,
    )

    query = np.fromfile(tmp_path / "out" / "tiny_query.u8bin", dtype=np.uint8)
    base = np.fromfile(tmp_path / "out" / "tiny_base.u8bin", dtype=np.uint8)
    assert np.frombuffer(query[:8].tobytes(), dtype="<i4").tolist() == [1, 2]
    assert np.frombuffer(base[:8].tobytes(), dtype="<i4").tolist() == [3, 2]
    assert len(query) == 8 + 2
    assert len(base) == 8 + 6
    assert result["spectrum_full"]["0.7"] == 2
    assert result["evaluation_mode"] == "geometry_probe"


def test_embedding_finalizer_keeps_qrels_queries_out_of_the_passage_base(
    tmp_path: Path,
):
    embedder = _load_embedder_module()
    import numpy as np

    base_shard = tmp_path / "base.npy"
    query_shard = tmp_path / "query.npy"
    np.save(
        base_shard,
        np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
    )
    np.save(query_shard, np.asarray([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32))

    result = embedder.finalize_embeddings(
        [base_shard],
        output_dir=tmp_path / "out",
        prefix="qrels",
        dimension=2,
        query_rows=0,
        query_shard_paths=[query_shard],
    )

    query = np.fromfile(tmp_path / "out" / "qrels_query.u8bin", dtype=np.uint8)
    base = np.fromfile(tmp_path / "out" / "qrels_base.u8bin", dtype=np.uint8)
    assert np.frombuffer(query[:8].tobytes(), dtype="<i4").tolist() == [2, 2]
    assert np.frombuffer(base[:8].tobytes(), dtype="<i4").tolist() == [4, 2]
    assert result["base_rows"] == 4
    assert result["query_rows"] == 2
    assert result["evaluation_mode"] == "qrels"
    assert result["quantization_min"] == 0.0
    assert result["quantization_max"] == 1.0
    assert result["query_clip_low_count"] == 0
    assert result["query_clip_high_count"] == 0


def test_qrels_queries_cannot_change_passage_quantization_bounds(tmp_path: Path):
    embedder = _load_embedder_module()
    import numpy as np

    base_shard = tmp_path / "base.npy"
    query_shard = tmp_path / "query.npy"
    np.save(base_shard, np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    np.save(query_shard, np.asarray([[-2.0, 3.0]], dtype=np.float32))

    result = embedder.finalize_embeddings(
        [base_shard],
        output_dir=tmp_path / "out",
        prefix="stable-base",
        dimension=2,
        query_rows=0,
        query_shard_paths=[query_shard],
    )

    assert result["quantization_min"] == 0.0
    assert result["quantization_max"] == 1.0
    assert result["query_clip_low_count"] == 1
    assert result["query_clip_high_count"] == 1


def test_qrels_validation_requires_unique_known_query_ids(tmp_path: Path):
    embedder = _load_embedder_module()
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        json.dumps({"query_id": "q1", "corpus_id": "doc-1", "relevance": 1}) + "\n",
        encoding="utf-8",
    )

    summary = embedder.validate_qrels(qrels, ("q1",))
    assert summary["row_count"] == 1
    assert summary["query_count"] == 1

    with pytest.raises(ValueError, match="unknown query_id"):
        embedder.validate_qrels(qrels, ("different",))
    with pytest.raises(ValueError, match="unknown corpus_id"):
        embedder.validate_qrels(qrels, ("q1",), {"different-document"})


def test_qrels_mode_routes_passages_and_queries_through_distinct_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    embedder = _load_embedder_module()
    import numpy as np

    revision = "a" * 40
    contract = ResolvedInputContract(
        model_id="test/model",
        model_revision=revision,
        counter=WordCounter(),
        effective_context_limit=8,
        renderer=InputRenderer(
            document_template="passage: {text}", query_template="query: {text}"
        ),
        native_dimension=2,
        output_dimension=2,
    )
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps(["a", "b", "c", "d"]), encoding="utf-8")
    corpus_manifest = tmp_path / "corpus.manifest.json"
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps({"chunk_id": "chunk-1", "chunk_index": 0}) + "\n",
        encoding="utf-8",
    )
    corpus_manifest.write_text(
        json.dumps(
            {
                "texts_sha256": embedder._sha256(texts),
                "chunks_sha256": embedder._sha256(chunks),
                "input_contract": {"contracts": [contract.to_manifest()]},
            }
        ),
        encoding="utf-8",
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"id": "q1", "text": "find a"}) + "\n", encoding="utf-8"
    )
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        json.dumps({"query_id": "q1", "corpus_id": "chunk-1", "relevance": 1}) + "\n",
        encoding="utf-8",
    )

    class FakeProvider:
        passage_calls: list[list[str]] = []
        query_calls: list[list[str]] = []

        @staticmethod
        def get_input_contract():
            return contract

        @staticmethod
        def get_dimension():
            return 2

        @staticmethod
        def get_runtime_info():
            return {
                "backend": "torch",
                "compute_dtype": "float32",
                "device": "mps",
                "requested_device": "mps",
            }

        @classmethod
        def embed_passages(cls, values):
            cls.passage_calls.append(list(values))
            return np.asarray(
                [[float(index % 2), float(index // 2)] for index in range(len(values))],
                dtype=np.float32,
            )

        @classmethod
        def embed_queries(cls, values):
            cls.query_calls.append(list(values))
            return np.asarray([[0.5, 0.5] for _ in values], dtype=np.float32)

    monkeypatch.setattr(
        embedder, "create_open_model_provider", lambda *args, **kwargs: FakeProvider()
    )
    result = embedder.run(
        Namespace(
            texts=texts,
            corpus_manifest=corpus_manifest,
            model="test/model",
            revision=revision,
            dimension=2,
            output_dir=tmp_path / "output",
            prefix="tiny",
            shard_size=4,
            query_rows=1,
            queries_jsonl=queries,
            qrels=qrels,
            chunks_jsonl=chunks,
            query_id_field="id",
            query_text_field="text",
            batch_size=4,
            device="mps",
            backend="torch",
        )
    )

    assert FakeProvider.passage_calls == [["a", "b", "c", "d"]]
    assert FakeProvider.query_calls == [["find a"]]
    assert result["evaluation_mode"] == "qrels"
    assert result["base_rows"] == 4
    assert result["query_rows"] == 1
    assert result["qrels"]["query_count"] == 1
    assert result["transport_sha256"] == embedder._sha256(Path(embedder.__file__))
    assert result["runtime"]["device"] == "mps"
    assert result["artifact_dtype"] == "float32"
    assert json.loads(Path(result["query_ids"]["path"]).read_text()) == ["q1"]
    assert result["shards"]["passage"]["manifest_sha256"]
    assert result["shards"]["query"]["manifest_sha256"]


def test_exact_embedding_scorer_preserves_row_lineage_and_float_scores(
    tmp_path: Path,
):
    embedder = _load_embedder_module()
    scorer = _load_scorer_module()
    import numpy as np

    runtime = {
        "backend": "torch",
        "compute_dtype": "float32",
        "device": "mps",
        "requested_device": "mps",
    }
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "\n".join(json.dumps({"chunk_id": chunk_id}) for chunk_id in ("c1", "c2", "c3"))
        + "\n",
        encoding="utf-8",
    )
    corpus_manifest = tmp_path / "corpus.manifest.json"
    corpus_manifest.write_text(
        json.dumps(
            {
                "texts_sha256": embedder._sha256(texts),
                "chunks_sha256": embedder._sha256(chunks),
            }
        ),
        encoding="utf-8",
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text("queries", encoding="utf-8")
    query_ids = tmp_path / "query_ids.json"
    query_ids.write_text(json.dumps(["q1", "q2"]), encoding="utf-8")

    passage_paths = embedder.prepare_shards(
        output_dir=tmp_path / "embeddings",
        corpus_sha256=embedder._sha256(texts),
        model_id="test/model",
        revision="a" * 40,
        contract_fingerprint="b" * 64,
        dimension=2,
        shard_size=3,
        rows=3,
        runtime=runtime,
    )
    query_paths = embedder.prepare_shards(
        output_dir=tmp_path / "embeddings",
        corpus_sha256=embedder._sha256(queries),
        model_id="test/model",
        revision="a" * 40,
        contract_fingerprint="b" * 64,
        dimension=2,
        shard_size=2,
        rows=2,
        runtime=runtime,
        input_role="query",
    )
    np.save(
        passage_paths[0],
        np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32),
    )
    np.save(query_paths[0], np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    passage_shard_manifest = passage_paths[0].parent / "manifest.json"
    query_shard_manifest = query_paths[0].parent / "manifest.json"
    embedding_manifest = tmp_path / "embedding.manifest.json"
    embedding_manifest.write_text(
        json.dumps(
            {
                "evaluation_mode": "qrels",
                "model_id": "test/model",
                "model_revision": "a" * 40,
                "contract_fingerprint": "b" * 64,
                "dimension": 2,
                "runtime": runtime,
                "corpus_sha256": embedder._sha256(texts),
                "qrels": {
                    "chunks_sha256": embedder._sha256(chunks),
                    "queries_sha256": embedder._sha256(queries),
                },
                "query_ids": {
                    "path": str(query_ids),
                    "sha256": embedder._sha256(query_ids),
                    "rows": 2,
                },
                "shards": {
                    "passage": {
                        "manifest_path": str(passage_shard_manifest),
                        "manifest_sha256": embedder._sha256(passage_shard_manifest),
                    },
                    "query": {
                        "manifest_path": str(query_shard_manifest),
                        "manifest_sha256": embedder._sha256(query_shard_manifest),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run.jsonl"

    result = scorer.score_embedding_shards(
        embedding_manifest, corpus_manifest, chunks, output, top_k=2
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["query_id"], row["corpus_id"]) for row in rows] == [
        ("q1", "c1"),
        ("q1", "c2"),
        ("q2", "c3"),
        ("q2", "c2"),
    ]
    assert [row["score"] for row in rows] == pytest.approx([1.0, 0.8, 1.0, 0.6])
    assert result["scoring"] == "exact normalized float32 inner product"
    assert result["run_sha256"] == scorer._sha256(output)


def test_exact_embedding_scorer_can_rank_complete_distinct_sources(tmp_path: Path):
    embedder = _load_embedder_module()
    scorer = _load_scorer_module()
    import numpy as np

    runtime = {
        "backend": "torch",
        "compute_dtype": "float32",
        "device": "cpu",
        "requested_device": "cpu",
    }
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps(["a-1", "a-2", "b", "c"]), encoding="utf-8")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "\n".join(
            json.dumps({"chunk_id": chunk_id}) for chunk_id in ("c1", "c2", "c3", "c4")
        )
        + "\n",
        encoding="utf-8",
    )
    occurrences = tmp_path / "occurrences.jsonl"
    occurrences.write_text(
        "\n".join(
            json.dumps(
                {
                    "corpus_id": chunk_id,
                    "source_id": source_id,
                    "deduplicated_alias": False,
                }
            )
            for chunk_id, source_id in (
                ("c1", "doc-a"),
                ("c2", "doc-a"),
                ("c3", "doc-b"),
                ("c4", "doc-c"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_manifest = tmp_path / "corpus.manifest.json"
    corpus_manifest.write_text(
        json.dumps(
            {
                "texts_sha256": embedder._sha256(texts),
                "chunks_sha256": embedder._sha256(chunks),
                "chunk_occurrences_sha256": embedder._sha256(occurrences),
                "source_count": 3,
            }
        ),
        encoding="utf-8",
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text("queries", encoding="utf-8")
    query_ids = tmp_path / "query_ids.json"
    query_ids.write_text(json.dumps(["q1"]), encoding="utf-8")
    passage_paths = embedder.prepare_shards(
        output_dir=tmp_path / "embeddings",
        corpus_sha256=embedder._sha256(texts),
        model_id="test/model",
        revision="a" * 40,
        contract_fingerprint="b" * 64,
        dimension=2,
        shard_size=4,
        rows=4,
        runtime=runtime,
    )
    query_paths = embedder.prepare_shards(
        output_dir=tmp_path / "embeddings",
        corpus_sha256=embedder._sha256(queries),
        model_id="test/model",
        revision="a" * 40,
        contract_fingerprint="b" * 64,
        dimension=2,
        shard_size=1,
        rows=1,
        runtime=runtime,
        input_role="query",
    )
    np.save(
        passage_paths[0],
        np.asarray([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0], [0.7, 0.0]], dtype=np.float32),
    )
    np.save(query_paths[0], np.asarray([[1.0, 0.0]], dtype=np.float32))
    passage_manifest = passage_paths[0].parent / "manifest.json"
    query_manifest = query_paths[0].parent / "manifest.json"
    embedding_manifest = tmp_path / "embedding.manifest.json"
    embedding_manifest.write_text(
        json.dumps(
            {
                "evaluation_mode": "qrels",
                "model_id": "test/model",
                "model_revision": "a" * 40,
                "contract_fingerprint": "b" * 64,
                "dimension": 2,
                "runtime": runtime,
                "corpus_sha256": embedder._sha256(texts),
                "qrels": {
                    "chunks_sha256": embedder._sha256(chunks),
                    "queries_sha256": embedder._sha256(queries),
                },
                "query_ids": {
                    "path": str(query_ids),
                    "sha256": embedder._sha256(query_ids),
                    "rows": 1,
                },
                "shards": {
                    "passage": {
                        "manifest_path": str(passage_manifest),
                        "manifest_sha256": embedder._sha256(passage_manifest),
                    },
                    "query": {
                        "manifest_path": str(query_manifest),
                        "manifest_sha256": embedder._sha256(query_manifest),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "source-run.jsonl"

    result = scorer.score_embedding_shards(
        embedding_manifest,
        corpus_manifest,
        chunks,
        output,
        top_k=3,
        candidate_granularity="source",
        occurrences_path=occurrences,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["source_id"] for row in rows] == ["doc-a", "doc-b", "doc-c"]
    assert [row["score"] for row in rows] == pytest.approx([1.0, 0.8, 0.7])
    assert result["candidate_granularity"] == "source"
    assert result["source_count"] == 3

    corpus_manifest.write_text(
        json.dumps(
            {
                "texts_sha256": embedder._sha256(texts),
                "chunks_sha256": embedder._sha256(chunks),
                "chunk_occurrences_sha256": embedder._sha256(occurrences),
                "source_count": 2,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source count disagrees"):
        scorer.score_embedding_shards(
            embedding_manifest,
            corpus_manifest,
            chunks,
            output,
            top_k=2,
            candidate_granularity="source",
            occurrences_path=occurrences,
        )
