#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write a retained Delta checkpoint fixture with an independent producer."""

import argparse
from pathlib import Path

import pyarrow as pa
from deltalake import DeltaTable, write_deltalake


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    root = parser.parse_args().output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error("output directory must be empty")

    for values, mode in (
        ([1, 2], "error"),
        ([3], "append"),
        ([4, 5], "overwrite"),
    ):
        write_deltalake(
            str(root), pa.table({"id": pa.array(values, type=pa.int64())}), mode=mode
        )
    DeltaTable(str(root), version=2).create_checkpoint()
    write_deltalake(
        str(root), pa.table({"id": pa.array([6], type=pa.int64())}), mode="append"
    )

    log = root / "_delta_log"
    for version in range(3):
        (log / f"{version:020d}.json").unlink()


if __name__ == "__main__":
    main()
