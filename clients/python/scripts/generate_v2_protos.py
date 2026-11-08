#!/usr/bin/env python3
"""
Generate Python v2 protobuf files from .proto definitions.

This script automates the generation of Python protobuf bindings for v2 APIs.
It handles both protobuf messages (_pb2.py) and gRPC service stubs (_pb2_grpc.py).

Usage:
    python scripts/generate_v2_protos.py [proximadb/v2/record.proto ...]

The script will:
1. Generate protobuf files for v2 proto files
2. Place generated files in the canonical src/proximadb_sdk/v2/ runtime
   package (the single generated source of truth)
3. Mirror each generated module as a legacy re-export shim under
   src/proximadb/v2/ (see generate_protos.write_legacy_shims)
"""

import subprocess
import sys
from pathlib import Path

# Root directories - use absolute paths from script location
SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # Go up to repo root
PROTO_DIR = PROJECT_ROOT / "proto"
PYTHON_ROOT = SCRIPT_DIR.parent / "src"
PYTHON_PACKAGE = PYTHON_ROOT / "proximadb_sdk" / "v2"
LEGACY_SHIM_DIR = PYTHON_ROOT / "proximadb" / "v2"

# v2 Proto files to generate (relative to proto directory)
PROTOS = [
    "proximadb/v2/record.proto",
    "proximadb/v2/graph.proto",
    "proximadb/v2/entity.proto",
    "proximadb/v2/document.proto",
    "proximadb/v2/fusion.proto",
]


def generate_proto(proto_path: str, proto_dir: Path, python_root: Path) -> bool:
    """Generate Python protobuf files for a single proto file.

    Args:
        proto_path: Path to the proto file (relative to proto_dir)
        proto_dir: Directory containing proto files
        python_root: Python source root. Protoc preserves the proto package path.

    Returns:
        True if generation succeeded, False otherwise
    """
    proto_file = proto_dir / proto_path

    if not proto_file.exists():
        print(f"Warning: Proto file not found: {proto_file}")
        return False

    print(f"Generating: {proto_path}")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"--proto_path={proto_dir}",
                f"--python_out={python_root}",
                f"--grpc_python_out={python_root}",
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


def main() -> int:
    """Main generation function."""
    print("🔄 Generating v2 Python protobuf files...")

    # Ensure output directory exists
    PYTHON_PACKAGE.mkdir(parents=True, exist_ok=True)

    selected_protos = sys.argv[1:] or PROTOS
    unknown_protos = sorted(set(selected_protos) - set(PROTOS))
    if unknown_protos:
        print(f"Unknown v2 proto(s): {', '.join(unknown_protos)}")
        print(f"Supported protos: {', '.join(PROTOS)}")
        return 2

    # Generate all proto files
    success_count = 0
    for proto in selected_protos:
        if generate_proto(proto, PROTO_DIR, PYTHON_ROOT):
            success_count += 1

    print(f"\n✅ Generated {success_count}/{len(selected_protos)} proto files")

    if success_count != len(selected_protos):
        print("Generation failed.")
        return 1

    print("\n📁 Generated files in src/proximadb/v2/:")
    for proto in selected_protos:
        proto_name = proto.replace("proximadb/v2/", "").replace(".proto", "")
        print(f"   - {proto_name}_pb2.py")
        print(f"   - {proto_name}_pb2_grpc.py")

    print("\n💡 Next steps:")
    print(
        "   1. Verify the generated code compiles: python -m py_compile src/proximadb/v2/*"
    )
    print("   2. Run tests to ensure nothing broke: pytest tests/")
    # Mirror generated modules as legacy re-export shims (proximadb.v2)
    LEGACY_SHIM_DIR.mkdir(parents=True, exist_ok=True)
    write_legacy_shims("v2", out_dir=PYTHON_PACKAGE, shim_dir=LEGACY_SHIM_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
