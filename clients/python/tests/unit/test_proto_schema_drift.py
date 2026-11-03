"""Schema-level proto/stub drift gate (TD-PROTO-1).

`test_grpc_proto_drift.py` computes missing-stub / missing-servicer / missing-rpc.
All three are about **services and rpcs**, so a `.proto` change that adds a
message, a field or an enum and touches no service moves nothing in that drift
set and the gate is green by construction. `collection_types.proto`, where this
surfaced, declares no service at all — which is why a field sat unreflected in
the checked-in stub indefinitely.

This gate closes that hole by comparing the **serialized `FileDescriptorProto`**
embedded in each committed `*_pb2.py` against one generated from the `.proto` at
HEAD. Byte equality is the whole test:

* it needs no proto parsing, so it cannot drift from the schema language;
* it catches messages, fields, enums, rpcs **and options** together — the
  `deprecated: true` markers a hand-rolled comparison would miss;
* it is immune to protoc's *formatting*, which differs from the committed files
  and would otherwise dominate a textual diff. Of 26 files with a text diff when
  TD-PROTO-1 was measured, **12 had byte-identical descriptors**.

Skipped, not failed, when `grpcio-tools` is unavailable: the generator is a test
dependency and its absence is a missing tool, not drift.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PROTO_DIR = REPO_ROOT / "proto"
PYTHON_SRC = REPO_ROOT / "clients" / "python" / "src"

#: Both packages ship generated stubs and both are live: `unified_client_async`,
#: `fusion_v2` and `document_v2` import from `proximadb.v2`, while the rest of
#: the SDK uses `proximadb_sdk.v*`. They are generated independently, which is
#: how they came to disagree — see `test_the_two_stub_trees_agree`.
STUB_PACKAGES = ("proximadb_sdk", "proximadb")

_SERIALIZED = re.compile(
    r"AddSerializedFile\(\s*(b'(?:[^'\\]|\\.)*'|b\"(?:[^\"\\]|\\.)*\")\s*\)", re.S
)

pytest.importorskip(
    "grpc_tools",
    reason="grpcio-tools generates the comparison stub; absence is not drift",
)


#: The docstring marker emitted by ``generate_protos.write_legacy_shims``.
#: A file is treated as a shim ONLY if it carries this exact marker AND
#: re-exports the module with the SAME stem — so an incidental
#: ``from proximadb_sdk...`` import inside a real (non-shim) generated file
#: can never divert content checks, and a wrong-stem shim fails loudly here.
_SHIM_MARKER = "Compatibility shim: the canonical generated module lives in"


def _resolve_shim(path: Path) -> Path:
    """Follow a legacy compatibility shim to its canonical generated file.

    Files under ``src/proximadb/`` are generated-form re-export shims of
    ``proximadb_sdk`` (the single generated source of truth). Gates inspect
    the descriptor of the module they resolve to, so a shim is validated
    through the canonical content it re-exports.
    """
    if path.parent.parent.name != "proximadb":
        return path  # only the legacy package is shimmed
    text = path.read_text()
    if _SHIM_MARKER not in text:
        return path  # real protoc output carries no marker
    for line in text.splitlines():
        if line.startswith("from proximadb_sdk.") and " import" in line:
            target = line.split(" import")[0].removeprefix("from ").strip()
            resolved = PYTHON_SRC / Path(*target.split(".")).with_suffix(".py")
            assert (
                resolved.stem == path.stem
            ), f"legacy shim {path.name} re-exports wrong module {target}"
            return resolved
    return path


def _descriptor(path: Path) -> bytes | None:
    """The serialized FileDescriptorProto embedded in a generated stub."""
    match = _SERIALIZED.search(_resolve_shim(path).read_text())
    return ast.literal_eval(match.group(1)) if match else None


def _proto_files() -> list[Path]:
    return sorted(PROTO_DIR.rglob("*.proto"))


def _stub_for(proto: Path, package: str) -> Path:
    """Where the committed stub for ``proto`` lives, by convention."""
    rel = proto.relative_to(PROTO_DIR)  # proximadb/v1/foo.proto
    return PYTHON_SRC / package / Path(*rel.parts[1:]).with_name(rel.stem + "_pb2.py")


def _generate(proto: Path, out: Path) -> Path | None:
    """Generate a stub for ``proto`` into ``out``; None if protoc declines."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={out}",
            str(proto),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    rel = proto.relative_to(PROTO_DIR)
    generated = out / rel.with_name(rel.stem + "_pb2.py")
    return generated if generated.exists() else None


def _pairs() -> list[tuple[Path, str, Path]]:
    out = []
    for proto in _proto_files():
        for package in STUB_PACKAGES:
            stub = _stub_for(proto, package)
            if stub.exists():
                out.append((proto, package, stub))
    return out


PAIRS = _pairs()


#: ProximaDB protos deliberately not generated for Python. They are server-side
#: contracts the client never speaks, so a missing stub is a decision, not drift.
#: Listed explicitly because the alternative -- skipping anything unmatched -- is
#: how `timeseries_pb2` sat orphaned under a nested path for a release cycle
#: while its .proto was deleted and nothing noticed.
NO_PYTHON_STUB_BY_DESIGN = frozenset(
    {
        "proximadb/explain.proto",
        "proximadb/v1/catalog.proto",
        "proximadb/v1/cluster.proto",
        "proximadb/v1/observability.proto",
        "proximadb/v1/ranking.proto",
        "proximadb/v1/security.proto",
        "proximadb/v1/streaming.proto",
        "proximadb/v1/unified.proto",
        "proximadb/v2/ledger.proto",
        # model_registry: the canonical tree (proximadb_sdk) dropped its stub
        # when the registry moved off the client surface; the legacy tree kept
        # serving a stale copy until the shim conversion (2026-10-31) removed
        # it — server-side contract the client never speaks.
        "proximadb/v2/model_registry.proto",
    }
)


def test_every_proximadb_proto_is_either_generated_or_declared_exempt():
    """No ProximaDB proto may be silently unmatched.

    The comparison above runs only for protos that already have a committed
    stub, so a proto with no stub anywhere is skipped rather than reported --
    the same shape of blindness this gate exists to remove, one level up.
    Anything without a stub must be named as deliberate.

    Vendored `google/protobuf/**` is excluded: well-known types, not our
    contracts.
    """
    unmatched = set()
    for proto in _proto_files():
        rel = proto.relative_to(PROTO_DIR).as_posix()
        if rel.startswith("google/"):
            continue
        if any(_stub_for(proto, pkg).exists() for pkg in STUB_PACKAGES):
            continue
        unmatched.add(rel)

    surprises = unmatched - NO_PYTHON_STUB_BY_DESIGN
    assert not surprises, (
        "These protos have no Python stub and are not declared exempt:\n  - "
        + "\n  - ".join(sorted(surprises))
        + "\n\nEither generate the stub, or add it to NO_PYTHON_STUB_BY_DESIGN "
        "with the reason it is server-side only."
    )

    stale = NO_PYTHON_STUB_BY_DESIGN - unmatched
    assert not stale, (
        "Listed as having no Python stub, but one now exists:\n  - "
        + "\n  - ".join(sorted(stale))
        + "\n\nRemove them from NO_PYTHON_STUB_BY_DESIGN so the gate compares them."
    )


def test_no_stub_lives_outside_its_expected_path():
    """A generated stub must sit where its import path says it does.

    protoc emits into a nested `proximadb/vN/` directory that has to be moved
    up; when that move is missed the file is committed at
    `proximadb_sdk/v1/proximadb/v1/foo_pb2.py`, where
    `from proximadb_sdk.v1 import foo_pb2` cannot reach it. Not hypothetical --
    that is exactly how `timeseries_pb2` shipped, and it survived a release
    because every check either skipped it or was never run.
    """
    strays = []
    for package in STUB_PACKAGES:
        root = PYTHON_SRC / package
        if not root.exists():
            continue
        strays += [
            str(p.relative_to(PYTHON_SRC))
            for p in root.rglob("*_pb2*.py")
            if "proximadb" in p.relative_to(root).parts[:-1]
        ]
    assert not strays, (
        "Generated stubs are committed under a nested proximadb/ directory, so "
        "their documented import path cannot reach them:\n  - "
        + "\n  - ".join(sorted(strays))
        + "\n\nprotoc emits that nesting; move the file up and rewrite its "
        "`from proximadb.vN import` lines to the owning package."
    )


def test_there_is_something_to_compare():
    """Guard the guard: a path convention change must not silently empty this."""
    assert PAIRS, (
        "No committed stub matched any .proto by the expected path convention. "
        "Either the layout moved or _stub_for is wrong — this gate is inert "
        "until that is fixed."
    )
    assert len({p for p, _, _ in PAIRS}) >= 10, (
        f"Only {len({p for p, _, _ in PAIRS})} protos matched a stub; the tree "
        "has far more. The convention is probably wrong for most of them."
    )


@pytest.mark.parametrize(
    "proto,package,stub",
    PAIRS,
    ids=[f"{pkg}/{p.relative_to(PROTO_DIR)}" for p, pkg, _ in PAIRS],
)
def test_committed_stub_matches_its_proto(proto: Path, package: str, stub: Path):
    """The committed stub must encode exactly the schema its .proto declares.

    This is the assertion the service/rpc gate cannot make. A message-only
    change — a new field, a new enum member, a `deprecated` option — leaves that
    gate green while the SDK silently loses the ability to see it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        generated = _generate(proto, Path(tmp))
        if generated is None:
            pytest.skip(f"protoc could not generate {proto.name}")
        fresh = _descriptor(generated)
        committed = _descriptor(stub)

    assert committed is not None, f"{stub} has no serialized descriptor"
    assert fresh is not None, f"generated stub for {proto.name} has no descriptor"
    assert committed == fresh, (
        f"{stub.relative_to(REPO_ROOT)} no longer encodes "
        f"{proto.relative_to(REPO_ROOT)}.\n\n"
        "A .proto changed without its Python stub being regenerated. Fix with:\n"
        f"  python -m grpc_tools.protoc -Iproto --python_out=clients/python/src/{package} "
        f"{proto.relative_to(REPO_ROOT)}\n"
        "then move the file out of the nested proximadb/ directory protoc "
        f"creates, rewrite its `from proximadb.vN import` lines to `{package}`, "
        "and run BOTH isort and black over it -- the committed stubs carry that "
        "pass, and `isort --check-only src tests` is a CI gate, so protoc output "
        "alone is rejected.\n\n"
        "Descriptors are compared as bytes, so this is a real schema difference "
        "— formatting cannot trigger it."
    )


def test_the_two_stub_trees_agree():
    """`proximadb` and `proximadb_sdk` must encode the same contract.

    Both are live and both are generated independently, which is exactly how
    they drifted apart: when TD-PROTO-1 measured this, 24 of 26 descriptors
    matched and two did not. Collapsing the duplication is the real fix; this
    assertion is the cheap one that stops it recurring meanwhile.
    """
    mismatched = []
    for proto in _proto_files():
        a, b = (_stub_for(proto, pkg) for pkg in STUB_PACKAGES)
        if not (a.exists() and b.exists()):
            continue
        if _descriptor(a) != _descriptor(b):
            mismatched.append(str(proto.relative_to(PROTO_DIR)))
    assert not mismatched, (
        "The two generated stub trees encode different schemas for:\n  - "
        + "\n  - ".join(mismatched)
        + "\n\nThey are two copies of one contract. Regenerate both from the "
        "same .proto in the same change."
    )
