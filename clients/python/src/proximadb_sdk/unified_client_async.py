"""
ProximaDB Unified Async Python Client

Native-async core operations (collections CRUD, record insert/upsert/delete,
search, get_vector) wire the **generated** ``asyncio``/``asyncio_detailed``
endpoint functions over a single shared ``httpx.AsyncClient`` — the spec-driven
transport mandated by Core Directive #15 (CLAUDE.md) / #29 (GEMINI.md), TD-126.
The async ops mirror the sync facade (``unified_client.py`` /
``protocols/rest_sync.py``) signatures + return types; they delegate to
``protocols/_rest_codegen_async`` (the async analog of the sync
``protocols/_rest_codegen``), which calls the generated ``asyncio_detailed``
functions exactly as the sync path uses the generated ``sync``/``_get_kwargs``.

When a gRPC endpoint is configured (``protocol=grpc`` or ``auto`` with grpc
available), the record core ops (insert/upsert/delete/search/get_vector) are
routed to the **native-async** ``grpc.aio`` client
(``protocols/grpc_async.ProximaDBAsyncGrpcClient``) — mirroring how the sync
``unified_client.py`` prefers gRPC over REST for these ops — otherwise they use
the generated async-REST transport. Collection CRUD always uses the
generated async-REST transport.

Graph operations are NOT in the generated OpenAPI client; they stay on the
hand-written ``rest_async.py`` httpx path (and async gRPC when available).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .config import ClientConfig, Protocol, load_config
from .exceptions import CollectionNotFoundError, ProximaDBError
from .models import (
    BatchResult,
    Collection,
    CollectionConfig,
    CollectionStats,
    DeleteResult,
    OperationMetrics,
    SearchResult,
    normalize_collection_config_kwargs,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from .models_v2 import ProximaRecord

logger = logging.getLogger(__name__)

from .protocols.rest_async import ProximaDBAsyncClient as RestAsyncClient


def _grpc_available() -> bool:
    """Lazily probe whether grpc.aio + generated v2 stubs are importable.

    Kept out of module scope so ``import proximadb_sdk`` stays grpc-free (the
    lazy gRPC boundary): the import only happens when the facade actually
    considers the gRPC transport in ``astart``.
    """
    try:
        import grpc.aio  # noqa: F401, PLC0415

        from proximadb.v2 import record_pb2_grpc  # noqa: F401, PLC0415

        return True
    except Exception:
        return False


def _coalesce_name(raw_name: Any, collection_id: str) -> str:
    """Return a collection name satisfying CollectionConfig's 8-char minimum.

    Mirrors the sync facade's get_collection name-padding so async returns the
    same Collection shape for short ids/names.
    """
    name = raw_name if isinstance(raw_name, str) and raw_name else collection_id
    if len(name) < 8:
        name = (
            collection_id if len(collection_id) >= 8 else f"collection_{collection_id}"
        )
    return name


class ProximaDBAsyncUnified:
    """Async ProximaDB client.

    Core REST ops route through the generated async transport; graph ops use the
    hand-written async REST/gRPC paths. Use as an async context manager::

        async with ProximaDBAsyncUnified(url="http://localhost:5678") as client:
            await client.create_collection("my_vectors", CollectionConfig(...))
    """

    def __init__(
        self,
        url: str | None = None,
        protocol: Protocol | str = Protocol.AUTO,
        config: ClientConfig | None = None,
        grpc_endpoint: str | None = None,
        rest_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.config = config or load_config(url=url)
        self.protocol = Protocol(protocol) if isinstance(protocol, str) else protocol
        self.grpc_endpoint = grpc_endpoint or "localhost:5679"
        self.rest_url = rest_url or (url or self.config.url)
        self.timeout = timeout

        self._grpc = None
        self._rest = None
        # True once the native-async gRPC channel is connected and core ops
        # should route to it (set in astart).
        self._use_grpc = False
        # Shared async transport for the generated REST client.
        self._async_http = None
        self._gen_client = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def astart(self):
        # Always stand up the generated-async REST transport for core ops: a
        # single pooled httpx.AsyncClient injected into the generated Client.
        import httpx

        from ._generated.rest.client import Client as GenClient

        base_url = (self.rest_url or "").rstrip("/")
        self._async_http = httpx.AsyncClient(base_url=base_url, timeout=self.timeout)
        self._gen_client = GenClient(base_url=base_url).set_async_httpx_client(
            self._async_http
        )

        # Prefer the native-async gRPC channel for core record ops when grpc is
        # selected (or auto + available) — mirrors the sync unified client's
        # gRPC-over-REST preference. Falls back to async REST on failure.
        want_grpc = self.protocol == Protocol.GRPC or (
            self.protocol == Protocol.AUTO and _grpc_available()
        )
        if want_grpc:
            try:
                from .protocols.grpc_async import ProximaDBAsyncGrpcClient

                self._grpc = ProximaDBAsyncGrpcClient(
                    endpoint=self.grpc_endpoint, timeout=self.timeout
                )
                await self._grpc.connect()
                self._use_grpc = True
                logger.info("Using native-async gRPC client for core record ops")
            except Exception as e:
                logger.warning(f"gRPC async init failed: {e}; falling back to REST")
                self._grpc = None
                self._use_grpc = False

        # Always stand up an async REST client too: it backs graph ops and is
        # the fallback for core ops when gRPC is unavailable.
        self._rest = RestAsyncClient(url=self.rest_url, timeout=self.timeout)
        if not self._use_grpc:
            logger.info("Using async REST client for core record ops")
        return self

    async def aclose(self):
        if self._grpc is not None:
            await self._grpc.close()
            self._grpc = None
            self._use_grpc = False
        if self._rest:
            await self._rest.aclose()
            self._rest = None
        if self._async_http is not None:
            await self._async_http.aclose()
            self._async_http = None
            self._gen_client = None

    async def __aenter__(self):
        await self.astart()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def _require_gen_client(self):
        if self._gen_client is None:
            raise RuntimeError("Client not started; call astart() first")
        return self._gen_client

    @staticmethod
    def _raise_for_error(parsed: Any, *, context: str) -> None:
        """Raise if the generated parsed model is an ErrorResponse-shaped object."""
        if parsed is None:
            raise ProximaDBError(f"{context}: empty response from server")
        err = getattr(parsed, "error", None) or getattr(parsed, "error_message", None)
        if err:
            raise ProximaDBError(f"{context}: {err}")

    # ------------------------------------------------------------------
    # Collections (generated-async transport)
    # ------------------------------------------------------------------
    async def create_collection(
        self,
        name: str,
        config: CollectionConfig | None = None,
        **kwargs,
    ) -> Collection:
        """Create a collection via the generated ``create_collection`` async op."""
        from .protocols import _rest_codegen_async as _gen_async

        # TD-SDKENG-1: never silently drop an unknown keyword; only the
        # transport-consumed pass-through travels past the config.
        config_kwargs, passthrough = normalize_collection_config_kwargs(kwargs)
        if config is None:
            config = CollectionConfig(name=name, **config_kwargs)
        elif config_kwargs:
            # Round-2 review: re-validate (model_copy skips validation and
            # does not translate field aliases like quantization=).
            merged = config.model_dump(exclude={"name"})
            merged.update(config_kwargs)
            config = CollectionConfig(name=name, **merged)
        body: dict[str, Any] = {
            "name": name,
            "dimension": config.dimension,
            "engine": getattr(config.storage_engine, "value", config.storage_engine),
            "distance_metric": getattr(
                config.distance_metric, "value", config.distance_metric
            ),
            "enable_proxima_record": True,
        }
        # TD-SDKENG-1 (round-2 review): index and quantization settings must
        # reach the server under the REQUEST schema's names — index_configs
        # (not the response's index_specs) with hnsw_config, and quantization
        # with strategy (not type).
        if config.index_configs:
            body["index_configs"] = [
                {
                    "index_name": index.index_name,
                    "algorithm": getattr(index.algorithm, "value", index.algorithm),
                    **(
                        {
                            "hnsw_config": {
                                "m": index.hnsw_config.m,
                                "ef_construction": index.hnsw_config.ef_construction,
                                "ef_search": index.hnsw_config.ef_search,
                            }
                        }
                        if index.hnsw_config
                        else {}
                    ),
                    "is_primary": bool(index.is_primary),
                }
                for index in config.index_configs
            ]
        quant = config.quantization_config
        if quant and quant.enabled:
            v2_quant: dict[str, Any] = {"enabled": True}
            # Round-3 review: the server's strategy enum is
            # SMART_DEFAULTS|CUSTOM_LEVELS|MINIMAL|AGGRESSIVE — the SDK's
            # type (scalar/pq/...) is a method, not a strategy.
            if getattr(quant.type, "value", quant.type) not in (None, "none"):
                v2_quant["strategy"] = "custom_levels"
            body["quantization"] = v2_quant
        body.update({k: v for k, v in passthrough.items() if k != "indexing_algorithm"})
        resp = await _gen_async.create_collection(self._require_gen_client(), body)
        parsed = resp.parsed
        self._raise_for_error(parsed, context="create_collection failed")
        data = parsed.to_dict()
        return Collection(
            id=data.get("collection_id", name),
            config=CollectionConfig(
                name=_coalesce_name(data.get("name"), name),
                dimension=data.get("dimension", config.dimension),
                distance_metric=body["distance_metric"] or "cosine",
                storage_engine=data.get("engine") or "sst",
            ),
        )

    async def get_collection(self, collection_id: str) -> Collection:
        """Get collection metadata via the generated ``get_collection`` async op."""
        from .protocols import _rest_codegen_async as _gen_async

        resp = await _gen_async.get_collection(
            self._require_gen_client(), collection_id
        )
        parsed = resp.parsed
        if parsed is None or getattr(parsed, "error", None):
            raise CollectionNotFoundError(f"Collection '{collection_id}' not found")
        data = parsed.to_dict()
        if data.get("error") or data.get("error_message"):
            raise CollectionNotFoundError(f"Collection '{collection_id}' not found")
        stats_src = data.get("stats") or {}
        # The generated CollectionStatsV2 uses record_count / storage_size_bytes.
        #
        # `record_count` is nullable: null means the server does not know, which
        # is NOT the same as zero. Carry the None through rather than coalescing
        # it — reporting a fabricated 0 here is the exact bug this field was made
        # nullable to fix. Note `.get(key, default)` cannot express this on its
        # own: a present-but-null key returns None, not the default, so the older
        # `vector_count` fallback has to be selected on key presence.
        if "record_count" in stats_src:
            vector_count = stats_src["record_count"]
        else:
            vector_count = stats_src.get("vector_count")
        return Collection(
            id=data.get("collection_id", collection_id),
            config=CollectionConfig(
                name=_coalesce_name(data.get("name"), collection_id),
                dimension=data.get("dimension", 128),
                distance_metric=data.get("distance_metric") or "cosine",
                storage_engine=data.get("engine") or "sst",
            ),
            stats=CollectionStats(
                vector_count=vector_count or 0,
                index_size_bytes=stats_src.get("index_size_bytes", 0),
                data_size_bytes=stats_src.get("storage_size_bytes", 0),
            ),
        )

    async def list_collections(
        self,
        limit: int | None = None,
        offset: int | None = None,
        include_stats: bool | None = None,
    ) -> list[Collection]:
        """List collections via the generated ``list_collections`` async op."""
        from .protocols import _rest_codegen_async as _gen_async

        resp = await _gen_async.list_collections(
            self._require_gen_client(),
            limit=limit,
            offset=offset,
            include_stats=include_stats,
        )
        parsed = resp.parsed
        self._raise_for_error(parsed, context="list_collections failed")
        data = parsed.to_dict()
        out: list[Collection] = []
        for coll in data.get("collections", []) or []:
            cid = coll.get("collection_id", coll.get("id", ""))
            out.append(
                Collection(
                    id=cid,
                    config=CollectionConfig(
                        name=_coalesce_name(coll.get("name"), cid),
                        dimension=coll.get("dimension", 0),
                        distance_metric=coll.get("distance_metric") or "cosine",
                        storage_engine=coll.get("engine") or "sst",
                    ),
                    stats=CollectionStats(
                        # `or 0` would turn both a null (unknown) and an absent
                        # field into a confident zero. Keep unknown as None.
                        vector_count=coll.get("record_count"),
                    ),
                )
            )
        return out

    async def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection via the generated ``delete_collection`` async op."""
        from .protocols import _rest_codegen_async as _gen_async

        resp = await _gen_async.delete_collection(
            self._require_gen_client(), collection_id
        )
        parsed = resp.parsed
        if parsed is None:
            return False
        data = parsed.to_dict()
        return bool(data.get("success", False))

    # ------------------------------------------------------------------
    # Records / vectors (generated-async transport)
    # ------------------------------------------------------------------
    def _normalize_record(self, record: Any, index: int = 0) -> dict[str, Any]:
        """Normalize SDK record-like inputs to the v2 REST ProximaRecord shape.

        Shapes the SDK's own input model only; the wire request body + method +
        URL still come from the generated InsertRecordsRequest / asyncio_detailed.
        """
        if hasattr(record, "model_dump"):
            record = record.model_dump(exclude_none=True)
        elif hasattr(record, "dict"):
            record = record.dict(exclude_none=True)
        if not isinstance(record, dict):
            if hasattr(record, "vector"):
                record = {
                    "id": getattr(record, "id", None),
                    "vector": record.vector,
                    "props": getattr(record, "metadata", {}) or {},
                }
            else:
                raise TypeError(f"Unsupported record input: {type(record)!r}")

        vector = record.get("vector")
        if vector is None:
            raise ValueError("record is missing vector")
        try:
            import numpy as _np

            if isinstance(vector, _np.ndarray):
                vector = vector.astype(_np.float32, copy=False).tolist()
        except Exception:
            pass
        vector = [float(v) for v in vector]
        if not vector:
            raise ValueError("record vector cannot be empty")

        props: dict[str, Any] = {}
        for src in ("props", "metadata", "flexible_fields"):
            values = record.get(src)
            if isinstance(values, dict):
                props.update({str(k): v for k, v in values.items()})

        normalized: dict[str, Any] = {
            "id": record.get("id") or record.get("oid") or f"record_{index}",
            "vector": vector,
            "props": props,
        }
        text_fields = record.get("text_fields")
        if text_fields:
            normalized["text_fields"] = text_fields
        return normalized

    async def insert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord | dict[str, Any]],
        *,
        upsert: bool = False,
        validate_schema: bool = True,
    ) -> BatchResult:
        """Insert records via gRPC-async when configured, else REST-async."""
        if self._use_grpc and self._grpc is not None:
            return await self._grpc.insert_records(
                collection_id,
                records,
                upsert=upsert,
                validate_schema=validate_schema,
            )

        from .protocols import _rest_codegen_async as _gen_async

        body = {
            "records": [self._normalize_record(r, i) for i, r in enumerate(records)],
            "validate_schema": validate_schema,
            "upsert": upsert,
        }
        resp = await _gen_async.insert_records(
            self._require_gen_client(), collection_id, body
        )
        parsed = resp.parsed
        self._raise_for_error(parsed, context="insert_records failed")
        data = parsed.to_dict()
        inserted = int(data.get("inserted_count", 0) or 0)
        failed = int(data.get("failed_count", 0) or 0)
        total = inserted + failed
        return BatchResult(
            total=total,
            success=inserted,
            failed=failed,
            errors=data.get("errors", []) or [],
            duration_ms=0.0,
            metrics=OperationMetrics(
                total_processed=total,
                successful_count=inserted,
                failed_count=failed,
            ),
        )

    async def upsert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord | dict[str, Any]],
        *,
        validate_schema: bool = True,
    ) -> BatchResult:
        """Upsert records (insert_records with upsert=True)."""
        return await self.insert_records(
            collection_id,
            records,
            upsert=True,
            validate_schema=validate_schema,
        )

    async def get_vector(
        self,
        collection_id: str,
        vector_id: str,
        include_vector: bool = True,
        include_metadata: bool = True,
    ) -> dict[str, Any] | None:
        """Get a single record via gRPC-async when configured, else REST-async."""
        if self._use_grpc and self._grpc is not None:
            return await self._grpc.get_vector(
                collection_id,
                vector_id,
                include_vector=include_vector,
                include_metadata=include_metadata,
            )

        from .protocols import _rest_codegen_async as _gen_async

        resp = await _gen_async.get_record(
            self._require_gen_client(),
            collection_id,
            vector_id,
            include_vector=include_vector,
            include_text=False,
        )
        parsed = resp.parsed
        if parsed is None:
            raise ProximaDBError(f"Vector not found: {vector_id}")
        data = parsed.to_dict()
        if data.get("error") or data.get("error_message") or data.get("error_code"):
            raise ProximaDBError(f"Vector not found: {vector_id}")
        return data

    async def delete_vector(self, collection_id: str, vector_id: str) -> DeleteResult:
        """Delete a single record via gRPC-async when configured, else REST."""
        if self._use_grpc and self._grpc is not None:
            res = await self._grpc.delete_vector(collection_id, vector_id)
            success = bool(res.get("success", False))
            return DeleteResult(
                success=success,
                deleted_count=1 if success else 0,
                errors=[],
            )

        from .protocols import _rest_codegen_async as _gen_async

        resp = await _gen_async.delete_record(
            self._require_gen_client(), collection_id, vector_id
        )
        parsed = resp.parsed
        success = bool(parsed.to_dict().get("success", False)) if parsed else False
        return DeleteResult(
            success=success,
            deleted_count=1 if success else 0,
            errors=[],
        )

    async def delete_vectors(
        self, collection_id: str, vector_ids: list[str]
    ) -> DeleteResult:
        """Delete multiple records: batched gRPC-async when configured, else REST."""
        if self._use_grpc and self._grpc is not None:
            res = await self._grpc.delete_vectors(collection_id, vector_ids)
            deleted_count = int(res.get("deleted_count", 0))
            failed_count = int(res.get("failed_count", 0))
            return DeleteResult(
                success=failed_count == 0,
                deleted_count=deleted_count,
                errors=([f"{failed_count} deletes failed"] if failed_count else []),
            )

        deleted = 0
        errors: list[str] = []
        for vid in vector_ids:
            try:
                result = await self.delete_vector(collection_id, vid)
                if result.success:
                    deleted += result.deleted_count
                else:
                    errors.append(f"Delete failed for {vid}")
            except Exception as e:  # noqa: BLE001 - surface per-id failures
                errors.append(f"Delete failed for {vid}: {e}")
        return DeleteResult(
            success=not errors,
            deleted_count=deleted,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Search (generated-async transport)
    # ------------------------------------------------------------------
    async def search(
        self,
        collection_id: str,
        vector: list[float] | np.ndarray,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
    ) -> list[SearchResult]:
        """Search via gRPC-async when configured, else the REST-async op."""
        try:
            import numpy as _np

            if isinstance(vector, _np.ndarray):
                vector = vector.astype(_np.float32, copy=False).tolist()
        except Exception:
            pass

        if self._use_grpc and self._grpc is not None:
            return await self._grpc.search(
                collection_id,
                query_vector=list(vector),
                top_k=top_k,
                metadata_filters=metadata_filter,
                include_vectors=include_vectors,
                include_metadata=include_metadata,
            )

        from .protocols import _rest_codegen_async as _gen_async

        body: dict[str, Any] = {
            "vector": list(vector),
            "top_k": top_k,
            "include_vector": include_vectors,
            "include_text": False,
        }
        if metadata_filter:
            body["filters"] = [
                {"field": k, "op": "eq", "value": v} for k, v in metadata_filter.items()
            ]

        resp = await _gen_async.search_records(
            self._require_gen_client(), collection_id, body
        )
        parsed = resp.parsed
        if parsed is None:
            return []
        data = parsed.to_dict()
        if data.get("error") or data.get("error_message"):
            err = data.get("error") or data.get("error_message")
            if isinstance(err, str) and "not found" in err.lower():
                return []
            raise ProximaDBError(f"Search failed: {err}")

        results: list[SearchResult] = []
        for rank, item in enumerate(data.get("results", []) or [], start=1):
            if not isinstance(item, dict):
                continue
            props = item.get("props") if isinstance(item.get("props"), dict) else None
            results.append(
                SearchResult(
                    id=str(item.get("id", "")),
                    score=float(item.get("score", 0.0)),
                    vector=item.get("vector") if include_vectors else None,
                    metadata=props if include_metadata else None,
                    rank=rank,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Metadata scan (generated-async transport, REST-only)
    # ------------------------------------------------------------------
    #: Server-enforced upper bound on a single scan page (SCAN_RECORDS_MAX_PAGE).
    SCAN_MAX_PAGE = 10_000
    #: Default safety bound on the total rows ``scan`` accumulates across pages.
    SCAN_DEFAULT_MAX_ROWS = 1_000_000

    async def scan_page(
        self,
        collection_id: str,
        *,
        filter: list[dict[str, Any]] | dict[str, Any] | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        include_vectors: bool = False,
        include_text: bool = False,
    ) -> tuple[list[SearchResult], str | None]:
        """Scan a single page of records via the vector-free metadata-scan op.

        Wires the generated async ``scan_records`` op. Returns
        ``(records, next_cursor)`` (``next_cursor`` is ``None`` at end-of-scan).
        ``filter`` is the scan endpoint's metadata-filter shape (typed list
        ``[{field,op,value}]`` or equality map ``{field: value}``), pushed into
        the scan predicate server-side. This is a true metadata scan with no ANN
        ordering. The scan endpoint is REST-only (no gRPC scan surface).
        """
        from .protocols import _rest_codegen_async as _gen_async

        body: dict[str, Any] = {
            "include_vector": include_vectors,
            "include_text": include_text,
        }
        if filter:
            body["filter"] = filter
        if limit is not None:
            body["limit"] = limit
        if cursor:
            body["cursor"] = cursor

        resp = await _gen_async.scan_records(
            self._require_gen_client(), collection_id, body
        )
        parsed = resp.parsed
        if parsed is None:
            return [], None
        data = parsed.to_dict()
        if data.get("error") or data.get("error_message"):
            err = data.get("error") or data.get("error_message")
            if isinstance(err, str) and "not found" in err.lower():
                return [], None
            raise ProximaDBError(f"Scan failed: {err}")

        results: list[SearchResult] = []
        for rank, record in enumerate(data.get("records", []) or [], start=1):
            if not isinstance(record, dict):
                continue
            props = record.get("props") if isinstance(record.get("props"), dict) else {}
            results.append(
                SearchResult(
                    id=str(record.get("id", "")),
                    score=0.0,
                    rank=rank,
                    vector=record.get("vector") if include_vectors else None,
                    metadata=props,
                    version=record.get("version"),
                    timestamp=record.get("timestamp"),
                    source=record.get("source"),
                )
            )
        next_cursor = data.get("next_cursor") or None
        return results, next_cursor

    async def scan(
        self,
        collection_id: str,
        *,
        filter: list[dict[str, Any]] | dict[str, Any] | None = None,
        page_size: int | None = None,
        max_rows: int | None = None,
        include_vectors: bool = False,
        include_text: bool = False,
    ) -> list[SearchResult]:
        """Scan ALL records matching ``filter`` via cursor pagination (async).

        Follows ``next_cursor`` page-to-page until end-of-scan, returning the
        full matching set (not a bounded ANN window). ``max_rows`` (default
        :attr:`SCAN_DEFAULT_MAX_ROWS`) bounds total accumulated rows.
        """
        cap = self.SCAN_DEFAULT_MAX_ROWS if max_rows is None else max_rows
        per_page = self.SCAN_MAX_PAGE if page_size is None else page_size
        per_page = max(1, min(per_page, self.SCAN_MAX_PAGE))

        out: list[SearchResult] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while len(out) < cap:
            page_limit = min(per_page, cap - len(out))
            records, cursor = await self.scan_page(
                collection_id,
                filter=filter,
                limit=page_limit,
                cursor=cursor,
                include_vectors=include_vectors,
                include_text=include_text,
            )
            out.extend(records)
            if not cursor:
                break
            if cursor in seen_cursors:
                logger.warning(
                    "scan: repeated cursor %s for %s; stopping to avoid a loop",
                    cursor,
                    collection_id,
                )
                break
            seen_cursors.add(cursor)
            if not records:
                break
        return out[:cap]

    # ------------------------------------------------------------------
    # Graph operations (hand-written async REST/gRPC — not in OpenAPI spec)
    # ------------------------------------------------------------------
    async def graph_shortest_path(
        self,
        start_node_id: str,
        target_node_id: str,
        max_depth: int | None = None,
        edge_types: list[str] | None = None,
        algorithm: str = "DIJKSTRA",
        k: int | None = None,
        enable_prefetch: bool | None = None,
        prefetch_budget: int | None = None,
    ):
        if self._grpc and hasattr(self._grpc, "shortest_path"):
            return self._grpc.shortest_path(
                start_node_id,
                target_node_id,
                max_depth,
                edge_types,
                algorithm,
                k,
                enable_prefetch,
                prefetch_budget,
            )
        if self._rest:
            return await self._rest.graph_shortest_path(
                start_node_id,
                target_node_id,
                max_depth,
                edge_types,
                algorithm,
                k,
                enable_prefetch,
                prefetch_budget,
            )
        raise RuntimeError("Client not started; call astart() first")

    async def graph_traverse(
        self,
        start_node_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
        algorithm: str = "BFS",
        limit: int | None = None,
        timeout_ms: int | None = None,
        max_frontier: int | None = None,
        enable_prefetch: bool | None = None,
        prefetch_budget: int | None = None,
    ):
        # REST path for traversal (gRPC streaming traversal not exposed here)
        if self._rest:
            return await self._rest.graph_traverse(
                start_node_id,
                max_depth,
                edge_types,
                algorithm,
                limit,
                timeout_ms,
                max_frontier,
                enable_prefetch,
                prefetch_budget,
            )
        raise RuntimeError("Client not started; call astart() first")
