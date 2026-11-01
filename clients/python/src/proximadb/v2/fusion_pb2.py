"""Compatibility shim: the canonical generated module lives in
``proximadb_sdk.v2.fusion_pb2`` (single source of truth, emitted by
``scripts/generate_protos.py``). This path is kept so legacy
``import proximadb.v2.fusion_pb2`` keeps working against fresh generated code.
"""

from proximadb_sdk.v2.fusion_pb2 import *  # noqa: F401,F403
from proximadb_sdk.v2.fusion_pb2 import DESCRIPTOR as DESCRIPTOR  # noqa: F401
