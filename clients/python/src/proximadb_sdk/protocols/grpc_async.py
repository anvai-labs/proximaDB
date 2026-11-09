"""ProximaDB native-async gRPC client (TD-126).

A genuine ``grpc.aio`` client for the v2 ``ProximaRecordService``: it builds an
``grpc.aio`` channel, instantiates the **existing generated**
``proximadb.v2.record_pb2_grpc.ProximaRecordServiceStub`` on it (the generated
stubs are channel-agnostic, so their RPC methods are awaitable on an aio
channel — no stub/proto regeneration), and ``await``\\s each RPC. The proto
build/parse logic is the transport-agnostic codec shared with the synchronous
client (``protocols/_grpc_v2_codec``); it is not re-hand-rolled here.

Covered v2 ``ProximaRecordService`` ops (all awaited over the aio channel):
``insert_records``, ``upsert_records``, ``delete_vector``/``delete_vectors``,
``search``, ``get_vector``. Collection ops are served by the same v2 service,
but the async facade routes them over the generated async-REST transport
(matching how the sync unified client keeps collections on its codegen path),
so this client focuses on the record core ops.

The ``grpc`` import is performed lazily inside ``connect``/methods so that
``import proximadb_sdk`` stays grpc-free (lazy boundary, as today).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..exceptions import ProximaDBError
from ..models import SearchResult, VectorOperationResponse, resolve_vector_ids
from . import _grpc_v2_codec as _codec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..models import BatchResult
    from ..models_v2 import ProximaRecord

logger = logging.getLogger(__name__)


def _strip_scheme(endpoint: str) -> str:
    """grpc.aio.insecure_channel expects a bare ``host:port`` target."""
    return (
        endpoint.replace("https://", "")
        .replace("http://", "")
        .replace("grpcs://", "")
        .replace("grpc://", "")
        .rstrip("/")
    )


class ProximaDBAsyncGrpcClient:
    """Native-async gRPC client over a ``grpc.aio`` channel.

    Usage::

        client = ProximaDBAsyncGrpcClient("localhost:5679")
        await client.connect()
        try:
            await client.insert_records("coll", records)
        finally:
            await client.close()
    """

    def __init__(
        self,
        endpoint: str = "localhost:5679",
        timeout: float = 60.0,
        secure: bool = False,
        max_message_size: int = 64 * 1024 * 1024,
    ):
        self.endpoint = endpoint
        self.target = _strip_scheme(endpoint)
        self.timeout = timeout
        self.secure = secure
        self.max_message_size = max_message_size

        self._channel = None  # grpc.aio.Channel
        self._stub = None  # ProximaRecordServiceStub
        self._pb2 = None  # proximadb.v2.record_pb2 module

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self) -> "ProximaDBAsyncGrpcClient":
        """Open the aio channel and bind the generated v2 record stub.

        grpc imported lazily here to preserve the SDK's grpc-free import.
        """
        if self._channel is not None:
            return self

        import grpc  # noqa: PLC0415 - lazy boundary

        from proximadb.v2 import record_pb2 as v2_record_pb2  # noqa: PLC0415
        from proximadb.v2 import record_pb2_grpc as v2_record_pb2_grpc  # noqa: PLC0415

        options = [
            ("grpc.max_send_message_length", self.max_message_size),
            ("grpc.max_receive_message_length", self.max_message_size),
        ]
        if self.secure:
            self._channel = grpc.aio.secure_channel(
                self.target, grpc.ssl_channel_credentials(), options=options
            )
        else:
            self._channel = grpc.aio.insecure_channel(self.target, options=options)

        # The generated stub is channel-agnostic: on a grpc.aio channel its RPC
        # methods return awaitables. No regeneration required.
        self._stub = v2_record_pb2_grpc.ProximaRecordServiceStub(self._channel)
        self._pb2 = v2_record_pb2
        logger.info("Opened async gRPC channel to %s", self.target)
        return self

    async def close(self) -> None:
        """Close the aio channel (awaitable, unlike the sync client)."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
            self._pb2 = None

    async def __aenter__(self) -> "ProximaDBAsyncGrpcClient":
        return await self.connect()

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def _require(self):
        if self._stub is None or self._pb2 is None:
            raise ProximaDBError("Async gRPC client not connected; call connect()")
        return self._stub, self._pb2

    async def _await_rpc(self, op: str, coro):
        """Await an aio RPC, mapping grpc.aio errors to ProximaDBError."""
        import grpc  # noqa: PLC0415 - lazy boundary

        try:
            return await coro
        except grpc.aio.AioRpcError as e:  # pragma: no cover - exercised via mocks
            details = e.details() or str(e)
            logger.error("async gRPC %s failed: %s - %s", op, e.code(), details)
            if e.code() == grpc.StatusCode.UNAVAILABLE or "connect" in details.lower():
                raise ProximaDBError(f"{op} connection failed: {details}")
            raise ProximaDBError(f"{op} RPC failed: {details}")

    # ------------------------------------------------------------------
    # Record core ops (awaited over the aio channel)
    # ------------------------------------------------------------------
    async def insert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord | dict[str, Any]],
        **kwargs,
    ) -> "BatchResult":
        if bool(kwargs.pop("upsert", False)):
            return await self.upsert_records(collection_id, records, **kwargs)

        stub, pb2 = self._require()
        request = pb2.ProximaRecordBatch(
            collection_id=collection_id,
            write_mode=pb2.INSERT,
            validate_schema=bool(kwargs.get("validate_schema", True)),
            return_ids=bool(kwargs.get("return_ids", True)),
            return_errors=bool(kwargs.get("return_errors", True)),
        )
        request.records.extend(
            _codec.record_proto_for_grpc(pb2, record, index)
            for index, record in enumerate(records)
        )
        if kwargs.get("schema_id"):
            request.schema_id = str(kwargs["schema_id"])

        response = await self._await_rpc(
            "insert_records", stub.InsertRecords(request, timeout=self.timeout)
        )
        return _codec.v2_record_batch_result(response)

    async def upsert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord | dict[str, Any]],
        **kwargs,
    ) -> "BatchResult":
        kwargs.pop("upsert", None)
        stub, pb2 = self._require()
        request = pb2.ProximaRecordBatch(
            collection_id=collection_id,
            write_mode=pb2.UPSERT,
            validate_schema=bool(kwargs.get("validate_schema", True)),
            return_ids=bool(kwargs.get("return_ids", True)),
            return_errors=bool(kwargs.get("return_errors", True)),
        )
        request.records.extend(
            _codec.record_proto_for_grpc(pb2, record, index)
            for index, record in enumerate(records)
        )
        if kwargs.get("schema_id"):
            request.schema_id = str(kwargs["schema_id"])

        response = await self._await_rpc(
            "upsert_records", stub.UpsertRecords(request, timeout=self.timeout)
        )
        return _codec.v2_record_batch_result(response)

    async def delete_vector(self, collection_id: str, vector_id: str) -> dict[str, Any]:
        stub, pb2 = self._require()
        request = pb2.ProximaRecordBatch(
            collection_id=collection_id,
            write_mode=pb2.DELETE,
            return_ids=True,
            return_errors=True,
        )
        request.records.append(pb2.ProximaRecord(id=vector_id))
        response = await self._await_rpc(
            "delete_vector", stub.DeleteRecords(request, timeout=self.timeout)
        )
        ok = response.failed_count == 0
        return {
            "status": "deleted" if ok else "failed",
            "vector_id": vector_id,
            "success": ok,
        }

    async def delete_vectors(
        self, collection_id: str, vector_ids: list[str]
    ) -> dict[str, Any]:
        stub, pb2 = self._require()
        request = pb2.ProximaRecordBatch(
            collection_id=collection_id,
            write_mode=pb2.DELETE,
            return_ids=True,
            return_errors=True,
        )
        for vector_id in vector_ids:
            request.records.append(pb2.ProximaRecord(id=vector_id))
        response = await self._await_rpc(
            "delete_vectors", stub.DeleteRecords(request, timeout=self.timeout)
        )
        return {
            "status": "completed",
            "deleted_count": int(response.success_count),
            "failed_count": int(response.failed_count),
            "total_requested": len(vector_ids),
        }

    async def search(
        self,
        collection_id: str,
        query_vector: list[float] | None = None,
        query_vectors: list[list[float]] | None = None,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        search_hints: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search via the v2 ``Search`` RPC, awaited over the aio channel."""
        if query_vectors is None and query_vector is not None:
            query_vectors = [query_vector]
        elif query_vectors is None:
            raise ValueError("Either query_vector or query_vectors must be provided")

        stub, pb2 = self._require()
        all_results: list[SearchResult] = []
        for qv in query_vectors:
            request = pb2.TypedSearchRequest(
                collection_id=collection_id,
                top_k=top_k,
                include_vector=include_vectors,
                include_text_fields=False,
            )
            request.query_vector.extend(float(value) for value in qv)
            request.filter_logic = pb2.AND
            if metadata_filters:
                for key, value in metadata_filters.items():
                    fc = request.filters.add()
                    fc.field_name = str(key)
                    fc.operator = pb2.EQ
                    fc.value.CopyFrom(_codec.python_to_v2_typed_value(pb2, value))
            if search_hints:
                request.search_hints.update(
                    {str(k): str(v) for k, v in search_hints.items()}
                )

            response = await self._await_rpc(
                "search", stub.Search(request, timeout=self.timeout)
            )
            all_results.extend(
                _codec.search_results_from_proto(
                    response,
                    include_vectors=include_vectors,
                    include_metadata=include_metadata,
                )
            )
        return all_results

    async def get_vector(
        self,
        collection_id: str,
        vector_id: str,
        include_vector: bool = True,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Get a single record via the v2 ``GetRecord`` RPC (aio)."""
        stub, pb2 = self._require()
        request = pb2.GetRecordRequest(
            collection_id=collection_id,
            id=vector_id,
            include_vector=include_vector,
        )
        response = await self._await_rpc(
            "get_vector", stub.GetRecord(request, timeout=self.timeout)
        )
        if not response.found or not response.HasField("record"):
            raise ProximaDBError(f"Vector {vector_id} not found")
        rec = response.record
        result: dict[str, Any] = {"id": rec.id}
        if include_vector and rec.vector:
            result["vector"] = list(rec.vector)
        if include_metadata and rec.props:
            result["metadata"] = {
                k: _codec.v2_typed_value_to_python(v) for k, v in rec.props.items()
            }
        return result

    async def insert_vectors(
        self,
        collection_id: str,
        vectors: list[dict[str, Any]],
        upsert: bool = False,
    ) -> VectorOperationResponse:
        """Vector-alias insert/upsert over the v2 record surface (aio)."""
        records = [
            {
                "id": v.get("id") or v.get("oid") or f"record_{i}",
                "vector": v.get("vector"),
                "props": v.get("props") or v.get("metadata") or {},
            }
            for i, v in enumerate(vectors)
        ]
        batch = (
            await self.upsert_records(collection_id, records)
            if upsert
            else await self.insert_records(collection_id, records)
        )
        return VectorOperationResponse(
            success=batch.failed == 0,
            operation="UPSERT" if upsert else "INSERT",
            metrics=batch.metrics,
            # the ONE policy (models.resolve_vector_ids — shared with the
            # sync grpc client and the REST unified client)
            vector_ids=resolve_vector_ids(batch, records),
            error_message="; ".join(batch.errors) if batch.errors else None,
        )


# Backward-compatible aliases for the prior deprecated symbols. These now point
# at the genuine async client (no longer a sync subclass).
ProximaDBClient = ProximaDBAsyncGrpcClient
AsyncGrpcClient = ProximaDBAsyncGrpcClient
