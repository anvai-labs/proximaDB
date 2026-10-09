#!/usr/bin/env python3
"""Build an auditable token-budgeted embedding corpus from JSONL documents.

This is the benchmark transport around the authoritative SDK chunking contracts.
It deliberately contains no splitting algorithm of its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import tomllib  # stdlib from 3.11, this project's floor
from collections import defaultdict
from pathlib import Path
from typing import Any

import proximadb_sdk.chunking_strategies as chunking_package
from proximadb_sdk.chunking_strategies import (
    ChunkContextRenderer,
    ChunkingConfig,
    ChunkingStrategy,
    CompositeInputContract,
    HuggingFaceTokenCounter,
    InputRenderer,
    InputRole,
    OverflowPolicy,
    ResolvedInputContract,
    ShortChunkPolicy,
    TokenBudget,
    get_chunking_strategy,
)
from proximadb_sdk.embedding_providers.providers.local.open_weights import (
    create_open_model_provider,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _runtime_contract(model: dict[str, Any], index: int) -> ResolvedInputContract:
    provider_name = model.get("runtime_provider")
    if provider_name != "open-weights":
        raise ValueError(
            f"models[{index}].runtime_provider must be 'open-weights', "
            f"got {provider_name!r}"
        )
    model_id = str(model["model_id"])
    revision = str(model["revision"])
    tokenizer_id = str(model.get("tokenizer_id", model_id))
    tokenizer_revision = str(model.get("tokenizer_revision", revision))
    if tokenizer_id != model_id or tokenizer_revision != revision:
        raise ValueError(
            f"models[{index}] runtime resolution requires tokenizer_id/revision "
            "to match model_id/revision"
        )
    output_dimension = model.get("output_dimension")
    provider = create_open_model_provider(
        model_id,
        revision=revision,
        dimension=int(output_dimension) if output_dimension is not None else None,
        document_template=(
            str(model["document_template"]) if "document_template" in model else None
        ),
        query_template=(
            str(model["query_template"]) if "query_template" in model else None
        ),
        device=str(model.get("runtime_device", "cpu")),
        extra={"local_files_only": bool(model.get("local_files_only", False))},
    )
    contract = provider.get_input_contract()
    expected = {
        "model_id": model_id,
        "model_revision": revision,
        "effective_context_limit": int(model["effective_context_limit"]),
        "native_dimension": (
            int(model["native_dimension"]) if "native_dimension" in model else None
        ),
        "output_dimension": (
            int(output_dimension) if output_dimension is not None else None
        ),
    }
    actual = {
        "model_id": contract.model_id,
        "model_revision": contract.model_revision,
        "effective_context_limit": contract.effective_context_limit,
        "native_dimension": contract.native_dimension,
        "output_dimension": contract.output_dimension,
    }
    drift = {
        key: {"declared": value, "runtime": actual[key]}
        for key, value in expected.items()
        if value is not None and actual[key] != value
    }
    dimensions = tuple(int(value) for value in model.get("output_dimensions", ()))
    if dimensions and contract.supported_output_dimensions != dimensions:
        drift["supported_output_dimensions"] = {
            "declared": dimensions,
            "runtime": contract.supported_output_dimensions,
        }
    if drift:
        details = ", ".join(
            f"{key}={values['declared']!r} (runtime {values['runtime']!r})"
            for key, values in sorted(drift.items())
        )
        raise ValueError(f"models[{index}] declaration differs from runtime: {details}")
    return contract


def _declared_contract(model: dict[str, Any], index: int) -> ResolvedInputContract:
    tokenizer_id = str(model.get("tokenizer_id", model["model_id"]))
    tokenizer_revision = str(model.get("tokenizer_revision", model["revision"]))
    counter = HuggingFaceTokenCounter.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        local_files_only=bool(model.get("local_files_only", False)),
    )
    resolved_revision = getattr(counter, "resolved_revision", None)
    if resolved_revision is not None and resolved_revision != tokenizer_revision:
        raise ValueError(
            f"models[{index}] tokenizer resolved revision {resolved_revision!r} "
            f"does not match {tokenizer_revision!r}"
        )
    dimensions = tuple(int(value) for value in model.get("output_dimensions", ()))
    document_parameters = tuple(
        sorted(
            (str(key), str(value))
            for key, value in model.get("document_encode_parameters", {}).items()
        )
    )
    query_parameters = tuple(
        sorted(
            (str(key), str(value))
            for key, value in model.get("query_encode_parameters", {}).items()
        )
    )
    return ResolvedInputContract(
        model_id=str(model["model_id"]),
        model_revision=str(model["revision"]),
        counter=counter,
        effective_context_limit=int(model["effective_context_limit"]),
        renderer=InputRenderer(
            document_template=str(model.get("document_template", "{text}")),
            query_template=str(model.get("query_template", "{text}")),
        ),
        native_dimension=(
            int(model["native_dimension"]) if "native_dimension" in model else None
        ),
        output_dimension=(
            int(model["output_dimension"]) if "output_dimension" in model else None
        ),
        supported_output_dimensions=dimensions,
        minimum_output_dimension=(
            int(model["minimum_output_dimension"])
            if "minimum_output_dimension" in model
            else None
        ),
        document_encode_parameters=document_parameters,
        query_encode_parameters=query_parameters,
    )


def _read_models(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("contract TOML must contain at least one [[models]] table")
    if not all(isinstance(model, dict) for model in models):
        raise ValueError("every [[models]] entry must be a table")
    return models


def _contract_resolution_manifest(path: Path) -> dict[str, Any]:
    models = _read_models(path)
    resolved: dict[str, dict[str, str]] = {}
    for model in models:
        model_id = str(model.get("model_id", ""))
        sealed = model.get("sealed_runtime_contract_fingerprint")
        if sealed is not None:
            detail = {
                "mode": "sealed_runtime_fingerprint",
                "sealed_runtime_contract_fingerprint": str(sealed),
            }
        elif "runtime_provider" in model:
            detail = {"mode": "full_runtime"}
        else:
            detail = {"mode": "tokenizer_only"}
        resolved[model_id] = detail
    return {
        "path": str(path.resolve()),
        "contracts_sha256": _sha256(path),
        "models": resolved,
    }


def _load_contracts(path: Path) -> CompositeInputContract:
    models = _read_models(path)

    contracts = []
    for index, model in enumerate(models):
        for required in ("model_id", "revision", "effective_context_limit"):
            if required not in model:
                raise ValueError(f"models[{index}] is missing {required!r}")
        tokenizer_id = model.get("tokenizer_id", model["model_id"])
        tokenizer_revision = model.get("tokenizer_revision", model["revision"])
        if re.fullmatch(r"[0-9a-f]{40}", str(model["revision"])) is None:
            raise ValueError(
                f"models[{index}].revision must be an immutable 40-character "
                "Hugging Face commit SHA"
            )
        if re.fullmatch(r"[0-9a-f]{40}", str(tokenizer_revision)) is None:
            raise ValueError(
                f"models[{index}].tokenizer_revision must be an immutable "
                "40-character Hugging Face commit SHA"
            )
        sealed = model.get("sealed_runtime_contract_fingerprint")
        if sealed is not None:
            if model.get("runtime_provider") != "open-weights":
                raise ValueError(
                    f"models[{index}].sealed_runtime_contract_fingerprint requires "
                    "runtime_provider='open-weights'"
                )
            if re.fullmatch(r"[0-9a-f]{64}", str(sealed)) is None:
                raise ValueError(
                    f"models[{index}].sealed_runtime_contract_fingerprint must be "
                    "a 64-character SHA-256"
                )
            contract = _declared_contract(model, index)
            if contract.fingerprint != sealed:
                raise ValueError(
                    f"models[{index}] sealed runtime contract fingerprint drift: "
                    f"expected {sealed}, resolved {contract.fingerprint}"
                )
            contracts.append(contract)
            continue
        if "runtime_provider" in model:
            contracts.append(_runtime_contract(model, index))
            continue
        contracts.append(_declared_contract(model, index))
    return CompositeInputContract(tuple(contracts))


def _percentile(sorted_values: list[int], fraction: float) -> int | None:
    if not sorted_values:
        return None
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def _histogram(values: list[int]) -> dict[str, int | None]:
    ordered = sorted(values)
    return {
        "min": ordered[0] if ordered else None,
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1] if ordered else None,
    }


def _atomic_paths(
    output_dir: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts_fd, texts_name = tempfile.mkstemp(
        prefix="texts.", suffix=".tmp", dir=output_dir
    )
    chunks_fd, chunks_name = tempfile.mkstemp(
        prefix="chunks.", suffix=".tmp", dir=output_dir
    )
    occurrences_fd, occurrences_name = tempfile.mkstemp(
        prefix="chunk_occurrences.", suffix=".tmp", dir=output_dir
    )
    os.close(texts_fd)
    os.close(chunks_fd)
    os.close(occurrences_fd)
    texts_tmp = Path(texts_name)
    chunks_tmp = Path(chunks_name)
    occurrences_tmp = Path(occurrences_name)
    return (
        texts_tmp,
        chunks_tmp,
        occurrences_tmp,
        output_dir / "texts.json",
        output_dir / "chunks.jsonl",
        output_dir / "chunk_occurrences.jsonl",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("max_chunks must be positive when set")
    input_sha256 = _sha256(args.input)
    contracts = _load_contracts(args.contracts)
    context_field = getattr(args, "context_field", None)
    context_template = getattr(args, "context_template", "{context}\n\n{text}")
    context_renderer = (
        ChunkContextRenderer(
            metadata_key=context_field,
            template=context_template,
        )
        if context_field is not None
        else None
    )
    budget = TokenBudget(
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        min_content_tokens=args.min_content_tokens,
        overflow_policy=OverflowPolicy(args.overflow_policy),
        short_chunk_policy=ShortChunkPolicy(args.short_chunk_policy),
    )
    config = ChunkingConfig(
        strategy=ChunkingStrategy(args.strategy),
        chunk_size=args.boundary_char_size,
        chunk_overlap=0,
        min_chunk_size=1,
        max_chunk_size=args.boundary_char_size,
        token_budget=budget,
        input_contract=contracts,
        input_role=InputRole.DOCUMENT,
        context_renderer=context_renderer,
    )
    strategy = get_chunking_strategy(
        config.strategy,
        **{key: value for key, value in vars(config).items() if key != "strategy"},
    )

    (
        texts_tmp,
        chunks_tmp,
        occurrences_tmp,
        texts_path,
        chunks_path,
        occurrences_path,
    ) = _atomic_paths(args.output_dir)
    token_lengths: dict[str, list[int]] = defaultdict(list)
    new_content_token_lengths: list[int] = []
    actual_overlap_token_lengths: list[int] = []
    source_count = 0
    chunk_count = 0
    split_sources = 0
    dropped_spans: list[dict[str, Any]] = []
    stopped_at_max_chunks = False
    allow_partial_corpus = getattr(args, "allow_partial_corpus", False)
    canonical_chunk_ids: dict[bytes, str] = {}
    seen_source_ids: set[str] = set()
    duplicate_chunk_count = 0
    chunk_occurrence_count = 0
    try:
        with (
            args.input.open("r", encoding="utf-8") as source,
            texts_tmp.open("w", encoding="utf-8") as texts,
            chunks_tmp.open("w", encoding="utf-8") as chunks_out,
            occurrences_tmp.open("w", encoding="utf-8") as occurrences_out,
        ):
            texts.write("[")
            first_text = True
            for line_number, line in enumerate(source, 1):
                if args.max_chunks is not None and chunk_count >= args.max_chunks:
                    stopped_at_max_chunks = True
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get(args.text_field)
                if not isinstance(text, str):
                    raise ValueError(
                        f"line {line_number}: {args.text_field!r} must be a string"
                    )
                context = ""
                if context_field is not None:
                    context = record.get(context_field)
                    if not isinstance(context, str):
                        raise ValueError(
                            f"line {line_number}: {context_field!r} must be a string"
                        )
                source_id = str(record.get(args.id_field, line_number))
                if source_id in seen_source_ids:
                    raise ValueError(
                        f"line {line_number}: duplicate source id {source_id!r}"
                    )
                seen_source_ids.add(source_id)
                source_count += 1
                base_metadata = {"source_line": line_number}
                if context_field is not None:
                    base_metadata[context_field] = context
                whole_input = (
                    context_renderer.render(text, base_metadata)
                    if context_renderer is not None
                    else text
                )
                was_oversized = not contracts.fits(
                    whole_input, InputRole.DOCUMENT, budget.target_tokens
                )
                emitted = strategy.chunk(text, source_id, base_metadata)
                if was_oversized and emitted:
                    split_sources += 1
                if not emitted and text.strip():
                    dropped_spans.append(
                        {
                            "source_id": source_id,
                            "start_pos": 0,
                            "end_pos": len(text),
                            "reason": "oversized_source_or_short_chunk_policy",
                        }
                    )
                elif emitted:
                    offsets = contracts.primary.counter.content_offsets(text) or ()
                    content_end = offsets[-1][1] if offsets else 0
                    if emitted[-1].end_pos < content_end:
                        dropped_spans.append(
                            {
                                "source_id": source_id,
                                "start_pos": emitted[-1].end_pos,
                                "end_pos": content_end,
                                "reason": "short_chunk_policy",
                            }
                        )

                for chunk in emitted:
                    if args.max_chunks is not None and chunk_count >= args.max_chunks:
                        stopped_at_max_chunks = True
                        break
                    new_content_tokens = chunk.metadata.get("new_content_tokens")
                    actual_overlap_tokens = chunk.metadata.get("overlap_tokens")
                    if (
                        not isinstance(new_content_tokens, int)
                        or new_content_tokens <= 0
                    ):
                        raise AssertionError(
                            f"chunk {chunk_count} adds no new source-token coverage"
                        )
                    if (
                        not isinstance(actual_overlap_tokens, int)
                        or actual_overlap_tokens < 0
                    ):
                        raise AssertionError(
                            f"chunk {chunk_count} has invalid actual overlap metadata"
                        )
                    model_input_text = chunk.input_text
                    canonical_id = chunk.chunk_id
                    deduplicated_alias = False
                    if args.deduplicate == "exact":
                        digest = hashlib.sha256(
                            model_input_text.encode("utf-8")
                        ).digest()
                        existing_id = canonical_chunk_ids.get(digest)
                        if existing_id is not None:
                            canonical_id = existing_id
                            deduplicated_alias = True
                            duplicate_chunk_count += 1
                        else:
                            canonical_chunk_ids[digest] = canonical_id
                    occurrences_out.write(
                        json.dumps(
                            {
                                "corpus_id": canonical_id,
                                "source_id": source_id,
                                "start_pos": chunk.start_pos,
                                "end_pos": chunk.end_pos,
                                "deduplicated_alias": deduplicated_alias,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    chunk_occurrence_count += 1
                    if deduplicated_alias:
                        continue
                    new_content_token_lengths.append(new_content_tokens)
                    actual_overlap_token_lengths.append(actual_overlap_tokens)
                    if not first_text:
                        texts.write(",")
                    json.dump(model_input_text, texts, ensure_ascii=False)
                    first_text = False
                    counts = contracts.validate(model_input_text, InputRole.DOCUMENT)
                    for model_id, count in counts.items():
                        if count > budget.target_tokens:
                            raise AssertionError(
                                f"chunk {chunk_count} exceeds target for {model_id}"
                            )
                        token_lengths[model_id].append(count)
                    chunks_out.write(
                        json.dumps(
                            {
                                "chunk_index": chunk_count,
                                "chunk_id": chunk.chunk_id,
                                "source_id": source_id,
                                "start_pos": chunk.start_pos,
                                "end_pos": chunk.end_pos,
                                "context_propagated": (
                                    chunk.model_input_text is not None
                                ),
                                "token_counts": counts,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    chunk_count += 1
            texts.write("]\n")
        if chunk_count == 0:
            raise ValueError("corpus builder emitted no chunks")
        if stopped_at_max_chunks and not allow_partial_corpus:
            raise ValueError(
                "max_chunks produced a partial corpus; remove --max-chunks for a "
                "full build or pass --allow-partial-corpus for an explicitly "
                "prefix-sampled diagnostic corpus"
            )
        if _sha256(args.input) != input_sha256:
            raise RuntimeError("input JSONL changed while the corpus was being built")
        os.replace(texts_tmp, texts_path)
        os.replace(chunks_tmp, chunks_path)
        os.replace(occurrences_tmp, occurrences_path)
    finally:
        texts_tmp.unlink(missing_ok=True)
        chunks_tmp.unlink(missing_ok=True)
        occurrences_tmp.unlink(missing_ok=True)

    histogram = {
        model_id: {
            **_histogram(values),
            "over_effective_context": sum(
                value
                > next(
                    contract.effective_context_limit
                    for contract in contracts.contracts
                    if contract.model_id == model_id
                )
                for value in values
            ),
            "over_target": sum(value > budget.target_tokens for value in values),
        }
        for model_id, values in token_lengths.items()
    }
    manifest = {
        "schema_version": 2,
        "builder_sha256": _sha256(Path(__file__).resolve()),
        "sdk_chunking_package_sha256": _python_tree_sha256(
            Path(chunking_package.__file__).resolve().parent
        ),
        "input": str(args.input.resolve()),
        "input_sha256": input_sha256,
        "texts_sha256": _sha256(texts_path),
        "chunks_sha256": _sha256(chunks_path),
        "chunk_occurrences_sha256": _sha256(occurrences_path),
        "source_count": source_count,
        "chunk_count": chunk_count,
        "chunk_occurrence_count": chunk_occurrence_count,
        "max_chunks": args.max_chunks,
        "stopped_at_max_chunks": stopped_at_max_chunks,
        "corpus_scope": (
            "partial_prefix" if stopped_at_max_chunks else "complete_source"
        ),
        "deduplication_policy": args.deduplicate,
        "duplicate_chunk_count": duplicate_chunk_count,
        "split_oversized_source_count": split_sources,
        "dropped_spans": dropped_spans,
        "boundary_strategy": args.strategy,
        "boundary_char_size": args.boundary_char_size,
        "source_fields": {
            "id": args.id_field,
            "text": args.text_field,
            **({"context": context_field} if context_field is not None else {}),
        },
        "context_propagation": (
            context_renderer.to_manifest() if context_renderer is not None else None
        ),
        "token_budget": budget.to_manifest(),
        "input_contract": contracts.to_manifest(),
        "contract_resolution": _contract_resolution_manifest(args.contracts),
        "token_histogram": histogram,
        "packing_histogram": {
            "new_content_tokens": _histogram(new_content_token_lengths),
            "actual_overlap_tokens": _histogram(actual_overlap_token_lengths),
        },
        "zero_truncation_asserted": all(
            values["over_effective_context"] == 0 and values["over_target"] == 0
            for values in histogram.values()
        ),
    }
    manifest_path = args.output_dir / "corpus.manifest.json"
    manifest_fd, manifest_name = tempfile.mkstemp(
        prefix="corpus.manifest.", suffix=".tmp", dir=args.output_dir
    )
    os.close(manifest_fd)
    manifest_tmp = Path(manifest_name)
    try:
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(manifest_tmp, manifest_path)
    finally:
        manifest_tmp.unlink(missing_ok=True)
    if not manifest["zero_truncation_asserted"]:
        raise AssertionError("emitted corpus contains inputs that would truncate")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL source")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contracts", type=Path, required=True, help="TOML model contracts"
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--context-field",
        help="optional source field propagated into every exact model input",
    )
    parser.add_argument(
        "--context-template",
        default="{context}\n\n{text}",
        help="sealed template with exactly one {context} and one {text}",
    )
    parser.add_argument("--id-field", default="id")
    parser.add_argument(
        "--strategy",
        choices=[value.value for value in ChunkingStrategy],
        default="recursive",
    )
    parser.add_argument("--target-tokens", type=int, default=480)
    parser.add_argument("--overlap-tokens", type=int, default=72)
    parser.add_argument("--min-content-tokens", type=int, default=32)
    parser.add_argument(
        "--max-chunks",
        type=int,
        help=(
            "stop after this many chunks, preserving deterministic input order; "
            "requires --allow-partial-corpus when the cap truncates the source"
        ),
    )
    parser.add_argument(
        "--allow-partial-corpus",
        action="store_true",
        help="explicitly permit a prefix-sampled diagnostic corpus",
    )
    parser.add_argument(
        "--deduplicate",
        choices=("none", "exact"),
        default="none",
        help="exact removes byte-identical emitted texts and records the count",
    )
    parser.add_argument("--boundary-char-size", type=int, default=4096)
    parser.add_argument(
        "--overflow-policy",
        choices=[value.value for value in OverflowPolicy],
        default="split",
    )
    parser.add_argument(
        "--short-chunk-policy",
        choices=[value.value for value in ShortChunkPolicy],
        default="keep",
    )
    return parser.parse_args()


def main() -> None:
    manifest = build(parse_args())
    print(
        json.dumps(
            {
                "chunk_count": manifest["chunk_count"],
                "corpus_scope": manifest["corpus_scope"],
                "texts_sha256": manifest["texts_sha256"],
                "token_histogram": manifest["token_histogram"],
                "zero_truncation_asserted": manifest["zero_truncation_asserted"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
