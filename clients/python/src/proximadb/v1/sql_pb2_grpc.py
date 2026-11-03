"""Compatibility shim: the canonical generated module lives in
``proximadb_sdk.v1.sql_pb2_grpc`` (single source of truth, emitted by
``scripts/generate_protos.py``). This path is kept so legacy
``import proximadb.v1.sql_pb2_grpc`` keeps working against fresh generated code.
"""

from proximadb_sdk.v1.sql_pb2_grpc import *  # noqa: F401,F403
