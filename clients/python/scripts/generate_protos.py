#!/usr/bin/env python3
"""
Generate Python protobuf files from .proto definitions.

This script automates the generation of Python protobuf bindings from
ProximaDB's proto definitions. It handles both the protobuf messages
(_pb2.py) and gRPC service stubs (_pb2_grpc.py).

Usage:
    python scripts/generate_protos.py

The script will:
1. Generate protobuf files for all defined proto files
2. Fix import paths to work with the Python SDK structure
3. Place generated files in src/proximadb_sdk/v1/
"""

import subprocess
import sys
from pathlib import Path

# Root directories
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROTO_DIR = PROJECT_ROOT / "proto"
PYTHON_OUT = SCRIPT_DIR.parent / "src" / "proximadb_sdk" / "v1"
# Compatibility shims: legacy import path kept alive against the single
# generated source of truth (see src/proximadb/v1/).
LEGACY_SHIM_DIR = SCRIPT_DIR.parent / "src" / "proximadb" / "v1"


def write_legacy_shims(version: str = "v1", out_dir=None, shim_dir=None) -> None:
    """Mirror every generated module as a re-export shim under proximadb.<version>.

    The emitted file form is the one ``tests/unit/test_proto_schema_drift.py``'s
    ``_resolve_shim`` recognizes (docstring marker + exact import lines); keep
    the two in sync when editing either.
    """
    out_dir = out_dir if out_dir is not None else PYTHON_OUT
    shim_dir = shim_dir if shim_dir is not None else LEGACY_SHIM_DIR
    for mod in sorted(out_dir.glob("*_pb2*.py")):
        name = mod.stem
        import_line = f"from proximadb_sdk.{version}.{name}"
        if name.endswith("_grpc"):
            body = f"{import_line} import *  # noqa: F401,F403\n"
        else:
            body = (
                f"{import_line} import *  # noqa: F401,F403\n"
                f"{import_line} import DESCRIPTOR as DESCRIPTOR  # noqa: F401\n"
            )
        header = (
            '"""Compatibility shim: the canonical generated module lives in\n'
            f"``proximadb_sdk.{version}.{name}`` (single source of truth, emitted by\n"
            "``scripts/generate_protos.py``). This path is kept so legacy\n"
            f"``import proximadb.{version}.{name}`` keeps working against fresh generated code.\n"
            '"""\n\n'
        )
        (shim_dir / mod.name).write_text(header + body)
    print(f"Wrote legacy shims to {shim_dir}")


# Proto files to generate (relative to proto directory)
PROTOS = [
    "proximadb/v1/document.proto",
    "proximadb/v1/hybrid.proto",
    "proximadb/v1/graph.proto",
    "proximadb/v1/vector.proto",
    "proximadb/v1/collection.proto",
    "proximadb/v1/entity.proto",
    "proximadb/v1/types.proto",
]


def generate_proto(proto_path: str, proto_dir: Path, python_out: Path) -> bool:
    """Generate Python protobuf files for a single proto file.

    Args:
        proto_path: Path to the proto file (relative to proto_dir)
        proto_dir: Directory containing proto files
        python_out: Directory to write generated Python files

    Returns:
        True if generation succeeded, False otherwise
    """
    proto_file = proto_dir / proto_path

    if not proto_file.exists():
        print(f"Warning: Proto file not found: {proto_file}")
        return False

    print(f"Generating: {proto_path}")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"--proto_path={proto_dir}",
                f"--python_out={python_out}",
                f"--grpc_python_out={python_out}",
                str(proto_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating {proto_path}:")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def fix_imports(python_out: Path) -> None:
    """Fix import paths in generated proto files.

    The protoc compiler generates imports with 'from proximadb.v1 import ...'
    but we need 'from . import ...' for the Python SDK structure.

    Args:
        python_out: Directory containing generated Python files
    """
    print("Fixing import paths...")

    # Find all generated _pb2.py and _pb2_grpc.py files
    proto_files = list(python_out.glob("*_pb2.py")) + list(
        python_out.glob("*_pb2_grpc.py")
    )

    for proto_file in proto_files:
        # Read the file
        content = proto_file.read_text()

        # Fix import paths
        original_content = content
        content = content.replace("from proximadb.v1 import", "from . import")

        # Write back if changed
        if content != original_content:
            proto_file.write_text(content)
            print(f"  Fixed: {proto_file.name}")


def move_nested_files(python_out: Path) -> None:
    """Move proto files from nested proximadb/v1/ directory to v1/ directory.

    Args:
        python_out: Directory containing generated Python files
    """
    nested_dir = python_out / "proximadb" / "v1"

    if not nested_dir.exists():
        return

    print("Moving nested files...")

    # Find all files in the nested directory
    nested_files = list(nested_dir.glob("*_pb2.py")) + list(
        nested_dir.glob("*_pb2_grpc.py")
    )

    for nested_file in nested_files:
        # Move to parent directory
        target = python_out / nested_file.name
        nested_file.rename(target)
        print(f"  Moved: {nested_file.name}")

    # Remove empty directories
    if nested_dir.exists():
        try:
            (nested_dir / "proximadb").rmdir()
            nested_dir.rmdir()
            (python_out / "proximadb").rmdir()
        except OSError:
            pass  # Directory not empty, leave it


def verify_generation(python_out: Path) -> bool:
    """Verify that proto files were generated correctly.

    Args:
        python_out: Directory containing generated Python files

    Returns:
        True if verification succeeded, False otherwise
    """
    print("Verifying generated files...")

    # Try to import key proto files
    try:
        sys.path.insert(0, str(python_out.parent))

        from proximadb_sdk.v1 import (
            document_pb2,
            document_pb2_grpc,
            hybrid_pb2,
            hybrid_pb2_grpc,
        )

        print("  All proto files imported successfully!")

        # Check for key services
        assert hasattr(document_pb2_grpc, "DocumentServiceStub")
        assert hasattr(hybrid_pb2_grpc, "HybridSearchServiceStub")

        print("  All service stubs found!")

        return True

    except ImportError as e:
        print(f"  Import failed: {e}")
        return False
    except AssertionError as e:
        print(f"  Service stub not found: {e}")
        return False
    finally:
        # Clean up sys.path
        if str(python_out.parent) in sys.path:
            sys.path.remove(str(python_out.parent))


def main() -> int:
    """Main entry point for proto generation.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("ProximaDB Python SDK Proto Generator")
    print("=" * 50)

    # Ensure output directory exists
    PYTHON_OUT.mkdir(parents=True, exist_ok=True)

    # Generate all proto files
    success_count = 0
    for proto in PROTOS:
        if generate_proto(proto, PROTO_DIR, PYTHON_OUT):
            success_count += 1

    print(f"\nGenerated {success_count}/{len(PROTOS)} proto files")

    # Move files from nested directory if needed
    move_nested_files(PYTHON_OUT)

    # Fix import paths
    fix_imports(PYTHON_OUT)

    # Mirror the generated modules as legacy re-export shims (proximadb.v1)
    LEGACY_SHIM_DIR.mkdir(parents=True, exist_ok=True)
    write_legacy_shims()

    # Verify generation
    print()
    if verify_generation(PYTHON_OUT):
        print("=" * 50)
        print("Proto generation completed successfully!")
        return 0
    else:
        print("=" * 50)
        print("Proto generation had errors. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
