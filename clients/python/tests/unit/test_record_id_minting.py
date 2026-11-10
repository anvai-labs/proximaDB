"""Client-minted record ids at the wire chokepoints.

Every write path that receives a record without an id must mint a fresh
UUID per call. The old defaults were position-derived (``record_0``,
``vec_0``, ``edge_0``), so a second ID-less write reused the first one's id
and silently overwrote it. Each test calls the chokepoint TWICE: a
position-derived default passes a single-call "is it set?" check but fails
here.
"""

from proximadb.v2 import record_pb2
from proximadb_sdk.graph import ProximaDBGraph
from proximadb_sdk.protocols._grpc_v2_codec import record_proto_for_grpc
from proximadb_sdk.protocols.rest_sync import ProximaDBClient
from tests.unit._id_helpers import is_uuid


def _two_distinct(first, second):
    assert is_uuid(first) and is_uuid(second), (first, second)
    assert first != second


def test_rest_normalize_record_payload_mints_per_call():
    client = ProximaDBClient(url="http://testserver")
    a = client._normalize_record_payload({"vector": [1.0]}, 0)
    b = client._normalize_record_payload({"vector": [2.0]}, 0)
    _two_distinct(a["id"], b["id"])
    kept = client._normalize_record_payload({"id": "mine", "vector": [1.0]}, 0)
    assert kept["id"] == "mine"


def test_grpc_codec_mints_per_call():
    a = record_proto_for_grpc(record_pb2, {"vector": [1.0]}, 0)
    b = record_proto_for_grpc(record_pb2, {"vector": [2.0]}, 0)
    _two_distinct(a.id, b.id)
    kept = record_proto_for_grpc(record_pb2, {"id": "mine", "vector": [1.0]}, 0)
    assert kept.id == "mine"


def test_present_but_falsy_id_is_minted():
    """{"id": None} / {"id": ""} must be minted too; setdefault kept them."""
    from tests.unit.test_unified_client_cov import make_client

    client = make_client()
    for empty in (None, ""):
        out = client._record_payload_from_legacy_input(
            {"id": empty, "vector": [1.0]}, 0
        )
        assert is_uuid(out["id"]), (empty, out["id"])


def test_graph_batch_edges_mint_per_call():
    sent = []

    class _FakeClient:
        def batch_create_edges(self, graph_id, edges, **kwargs):
            sent.append([e["id"] for e in edges])
            return {"created_count": len(edges)}

    graph = ProximaDBGraph(_FakeClient(), "g")
    edge = {"from_node_id": "a", "to_node_id": "b", "edge_type": "calls"}
    graph.batch_create_edges([dict(edge)])
    graph.batch_create_edges([dict(edge)])
    assert len(sent) == 2, sent
    _two_distinct(sent[0][0], sent[1][0])
