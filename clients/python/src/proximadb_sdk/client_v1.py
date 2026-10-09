"""
ProximaDB v1 Protocol Client

This client uses the v1 proto messages that align with the server's unified handlers.
"""

import logging
from typing import Any
from urllib.parse import urljoin, urlparse

import grpc
import requests

from .exceptions import NetworkError, ProximaDBError
from .models import (
    Collection,
    CollectionConfig,
    DistanceMetric,
    SearchResult,
    StorageEngine,
    VectorRecord,
)
from .v1 import (
    collection_pb2_grpc,
    collection_types_pb2,
    graph_pb2,
    graph_pb2_grpc,
    sql_pb2_grpc,
    types_pb2,
    vector_pb2_grpc,
    vector_types_pb2,
)

try:
    from .v2 import record_pb2, record_pb2_grpc
except Exception:  # pragma: no cover - generated v2 stubs may be absent in old installs
    record_pb2 = None
    record_pb2_grpc = None

logger = logging.getLogger(__name__)


class ProximaDBClientV1:
    """ProximaDB client using v1 protocol messages"""

    def __init__(
        self,
        url: str = "http://localhost:5678",
        protocol: str = "auto",  # "grpc", "rest", or "auto"
        timeout: float = 30.0,
        **kwargs,
    ):
        self.base_url = url
        self.timeout = timeout
        self.protocol = protocol

        # Parse URL to determine protocol if auto
        parsed = urlparse(url)
        if protocol == "auto":
            if parsed.port == 5679 or "grpc" in parsed.scheme:
                self.protocol = "grpc"
            else:
                self.protocol = "rest"

        # Setup gRPC if needed
        if self.protocol == "grpc":
            grpc_url = url.replace("http://", "").replace("https://", "")
            self.channel = grpc.insecure_channel(grpc_url)
            self.vector_stub = vector_pb2_grpc.VectorServiceStub(self.channel)
            if record_pb2_grpc is not None:
                self.record_stub = record_pb2_grpc.ProximaRecordServiceStub(
                    self.channel
                )
            self.collection_stub = collection_pb2_grpc.CollectionServiceStub(
                self.channel
            )
            self.sql_stub = sql_pb2_grpc.QueryServiceStub(self.channel)
            self.graph_stub = graph_pb2_grpc.GraphServiceStub(self.channel)

        logger.info(
            f"ProximaDB client initialized with protocol: {self.protocol}, url: {url}"
        )

    def close(self):
        """Close the client connection"""
        if hasattr(self, "channel"):
            self.channel.close()

    # Collection operations
    def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str | DistanceMetric = DistanceMetric.COSINE,
        storage_engine: str | StorageEngine = StorageEngine.SST,
        **kwargs,
    ) -> Collection:
        """Create a new collection"""

        # Convert enums to proto values
        if isinstance(distance_metric, DistanceMetric):
            distance_metric = distance_metric.value
        if isinstance(storage_engine, StorageEngine):
            storage_engine = storage_engine.value

        if self.protocol == "grpc":
            return self._create_collection_grpc(
                name, dimension, distance_metric, storage_engine, **kwargs
            )
        else:
            return self._create_collection_rest(
                name, dimension, distance_metric, storage_engine, **kwargs
            )

    def _create_collection_grpc(
        self,
        name: str,
        dimension: int,
        distance_metric: str,
        storage_engine: str,
        **kwargs,
    ):
        """Create collection via gRPC"""
        request = collection_types_pb2.CollectionConfig(
            name=name,
            dimension=dimension,
            distance_metric=distance_metric.upper(),
            storage_engine=storage_engine.upper(),
        )

        try:
            response = self.collection_stub.CreateCollection(
                request, timeout=self.timeout
            )
            # Collection response has: id, config (CollectionConfig), stats, created_at, updated_at
            # Proto enums come back as integers, need to map them to strings
            DISTANCE_METRIC_MAP = {
                0: "cosine",
                1: "cosine",
                2: "euclidean",
                3: "dot_product",
                4: "manhattan",
                5: "hamming",
                6: "jaccard",
                7: "chebyshev",
                8: "canberra",
                9: "minkowski",
                10: "angular",
                11: "bray_curtis",
                12: "hellinger",
                13: "custom",
            }
            STORAGE_ENGINE_MAP = {
                0: "viper",
                1: "viper",
                2: "sst",
                3: "nova",
                4: "helix",
                5: "swift",
                6: "raptor",
                7: "mmap",
                8: "hybrid",
            }

            dm_val = response.config.distance_metric if response.config else 0
            se_val = response.config.storage_engine if response.config else 0

            from .models import CollectionConfig, CollectionStats

            config = CollectionConfig(
                name=response.config.name if response.config else "",
                dimension=response.config.dimension if response.config else 0,
                distance_metric=DistanceMetric(
                    DISTANCE_METRIC_MAP.get(dm_val, "cosine")
                ),
                storage_engine=StorageEngine(STORAGE_ENGINE_MAP.get(se_val, "sst")),
            )

            stats = CollectionStats(
                vector_count=response.stats.vector_count if response.stats else 0,
                index_size_bytes=(
                    response.stats.index_size_bytes if response.stats else 0
                ),
                data_size_bytes=response.stats.data_size_bytes if response.stats else 0,
            )

            return Collection(
                id=response.id,
                config=config,
                stats=stats,
                created_at_ms=(
                    response.created_at // 1000 if response.created_at else 0
                ),  # Convert micros to millis
                updated_at_ms=response.updated_at // 1000 if response.updated_at else 0,
            )
        except grpc.RpcError as e:
            raise ProximaDBError(f"gRPC error: {e.details()}")

    def _create_collection_rest(
        self,
        name: str,
        dimension: int,
        distance_metric: str,
        storage_engine: str,
        **kwargs,
    ):
        """Create collection via REST"""
        payload = {
            "name": name,
            "dimension": dimension,
            "distance_metric": distance_metric.lower(),
            "engine": storage_engine.lower(),
            "enable_proxima_record": True,
        }

        url = urljoin(self.base_url, "/api/v2/collections")
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return Collection(
                id=data.get("collection_id", data.get("id", name)),
                config=CollectionConfig(
                    name=data.get("name", name),
                    dimension=data.get("dimension", dimension),
                    distance_metric=DistanceMetric(distance_metric.lower()),
                    storage_engine=StorageEngine(
                        data.get("engine", storage_engine).lower()
                    ),
                ),
            )
        except requests.RequestException as e:
            raise NetworkError(f"REST request failed: {e}")

    def get_collection(self, name: str) -> Collection | None:
        """Get collection by name"""
        if self.protocol == "grpc":
            return self._get_collection_grpc(name)
        else:
            return self._get_collection_rest(name)

    def _get_collection_grpc(self, name: str) -> Collection | None:
        """Get collection via gRPC"""
        request = collection_types_pb2.GetCollectionRequest(collection_id=name)

        try:
            response = self.collection_stub.GetCollection(request, timeout=self.timeout)
            return Collection(
                id=response.id,
                name=response.name,
                dimension=response.dimension,
                distance_metric=DistanceMetric(response.distance_metric.lower()),
                storage_engine=StorageEngine(response.storage_engine.lower()),
            )
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise ProximaDBError(f"gRPC error: {e.details()}")

    def _get_collection_rest(self, name: str) -> Collection | None:
        """Get collection via REST"""
        url = urljoin(self.base_url, f"/api/v2/collections/{name}")
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return Collection(
                id=data.get("collection_id", data.get("id", name)),
                config=CollectionConfig(
                    name=data.get("name", name),
                    dimension=data["dimension"],
                    distance_metric=DistanceMetric(data["distance_metric"].lower()),
                    storage_engine=StorageEngine(data["engine"].lower()),
                ),
            )
        except requests.RequestException as e:
            raise NetworkError(f"REST request failed: {e}")

    def list_collections(self) -> list[Collection]:
        """List all collections"""
        if self.protocol == "grpc":
            return self._list_collections_grpc()
        else:
            return self._list_collections_rest()

    def _list_collections_grpc(self) -> list[Collection]:
        """List collections via gRPC"""
        request = collection_types_pb2.ListCollectionsRequest()

        try:
            response = self.collection_stub.ListCollections(
                request, timeout=self.timeout
            )
            return [
                Collection(
                    id=col.id,
                    name=col.name,
                    dimension=col.dimension,
                    distance_metric=DistanceMetric(col.distance_metric.lower()),
                    storage_engine=StorageEngine(col.storage_engine.lower()),
                )
                for col in response.collections
            ]
        except grpc.RpcError as e:
            raise ProximaDBError(f"gRPC error: {e.details()}")

    def _list_collections_rest(self) -> list[Collection]:
        """List collections via REST"""
        url = urljoin(self.base_url, "/api/v2/collections")
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return [
                Collection(
                    id=col.get("collection_id", col.get("id", col["name"])),
                    config=CollectionConfig(
                        name=col["name"],
                        dimension=col["dimension"],
                        distance_metric=DistanceMetric(
                            col.get("distance_metric", "cosine").lower()
                        ),
                        storage_engine=StorageEngine(col["engine"].lower()),
                    ),
                )
                for col in data.get("collections", [])
            ]
        except requests.RequestException as e:
            raise NetworkError(f"REST request failed: {e}")

    # Vector operations
    def insert_records(
        self, collection_id: str, records: list[VectorRecord | dict[str, Any]]
    ) -> dict[str, Any]:
        """Insert ProximaRecord-shaped payloads through the record API."""
        if self.protocol == "grpc":
            return self._insert_records_grpc(collection_id, records)
        return self._insert_records_rest(collection_id, records)

    def insert_vectors(
        self, collection_id: str, vectors: list[VectorRecord]
    ) -> dict[str, Any]:
        """Compatibility alias for record-native inserts."""
        return self.insert_records(collection_id, vectors)

    def _record_payload(
        self, record: VectorRecord | dict[str, Any], index: int
    ) -> dict[str, Any]:
        if isinstance(record, dict):
            payload = dict(record)
            if "props" not in payload and "metadata" in payload:
                payload["props"] = payload.pop("metadata")
            payload.setdefault("id", payload.get("oid") or f"record_{index}")
            return payload

        payload = {
            "id": record.id or f"record_{index}",
            "vector": record.vector,
            "props": record.metadata or {},
        }
        source = getattr(record, "source", None)
        if source:
            payload["source"] = source
            payload["text_fields"] = [{"name": "text", "content": source}]
        return payload

    def _insert_records_rest(
        self, collection_id: str, records: list[VectorRecord | dict[str, Any]]
    ) -> dict[str, Any]:
        payload = {
            "records": [
                self._record_payload(record, index)
                for index, record in enumerate(records)
            ],
            "validate_schema": True,
        }

        url = urljoin(
            self.base_url, f"/api/v2/collections/{collection_id}/records/batch"
        )
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise NetworkError(f"REST request failed: {e}")

    def _typed_value(self, value: Any):
        if record_pb2 is None:
            raise ProximaDBError("v2 record protobuf stubs are unavailable")
        if value is None:
            return record_pb2.TypedValue(
                declared_type=record_pb2.COLUMN_TYPE_UNSPECIFIED,
                is_null=True,
            )
        if isinstance(value, bool):
            return record_pb2.TypedValue(
                declared_type=record_pb2.BOOLEAN,
                boolean_value=value,
            )
        if isinstance(value, int) and not isinstance(value, bool):
            return record_pb2.TypedValue(
                declared_type=record_pb2.INTEGER,
                integer_value=value,
            )
        if isinstance(value, float):
            return record_pb2.TypedValue(
                declared_type=record_pb2.FLOAT,
                float_value=value,
            )
        return record_pb2.TypedValue(
            declared_type=record_pb2.TEXT,
            text_value=str(value),
        )

    def _insert_records_grpc(
        self, collection_id: str, records: list[VectorRecord | dict[str, Any]]
    ) -> dict[str, Any]:
        if record_pb2 is None or not hasattr(self, "record_stub"):
            raise ProximaDBError("v2 ProximaRecord gRPC stubs are unavailable")

        proto_records = []
        for index, record in enumerate(records):
            payload = self._record_payload(record, index)
            props = {
                str(key): self._typed_value(value)
                for key, value in (payload.get("props") or {}).items()
            }
            text_fields = [
                record_pb2.TextField(
                    name=str(field.get("name", "text")),
                    content=str(field.get("content", "")),
                )
                for field in payload.get("text_fields", [])
            ]
            proto_records.append(
                record_pb2.ProximaRecord(
                    id=payload["id"],
                    vector=payload.get("vector") or [],
                    props=props,
                    source=payload.get("source"),
                    text_fields=text_fields,
                )
            )

        request = record_pb2.ProximaRecordBatch(
            collection_id=collection_id,
            records=proto_records,
            write_mode=record_pb2.INSERT,
            validate_schema=True,
            return_ids=True,
            return_errors=True,
        )
        try:
            response = self.record_stub.InsertRecords(request, timeout=self.timeout)
            return {
                "success": response.success,
                "total_processed": response.total_processed,
                "success_count": response.success_count,
                "failed_count": response.failed_count,
                "inserted_ids": list(response.inserted_ids),
                "errors": [
                    {
                        "record_index": error.record_index,
                        "record_id": error.record_id,
                        "error_code": error.error_code,
                        "error_message": error.error_message,
                    }
                    for error in response.errors
                ],
            }
        except grpc.RpcError as e:
            raise ProximaDBError(f"gRPC error: {e.details()}")

    def _convert_metadata_to_sql_value(
        self, metadata_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert Python dict metadata to gRPC SqlValue format"""
        sql_metadata = {}
        for key, value in (metadata_dict or {}).items():
            sql_metadata[key] = self._convert_to_sql_value(value)
        return sql_metadata

    def _insert_vectors_grpc(
        self, collection_id: str, vectors: list[VectorRecord]
    ) -> dict[str, Any]:
        """Compatibility alias for record-native gRPC inserts."""
        return self._insert_records_grpc(collection_id, vectors)

    def _insert_vectors_rest(
        self, collection_id: str, vectors: list[VectorRecord]
    ) -> dict[str, Any]:
        """Compatibility alias for record-native REST inserts."""
        return self._insert_records_rest(collection_id, vectors)

    def search_vectors(
        self,
        collection_id: str,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Search for similar vectors"""
        if self.protocol == "grpc":
            return self._search_vectors_grpc(collection_id, vector, top_k, filters)
        else:
            return self._search_vectors_rest(collection_id, vector, top_k, filters)

    def _search_vectors_grpc(
        self,
        collection_id: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Search vectors via gRPC"""
        # Create SearchQuery with vector and filters
        search_query = vector_types_pb2.SearchQuery(
            vector=vector, filters=filters or {}
        )

        request = vector_types_pb2.VectorSearchRequest(
            collection_id=collection_id, queries=[search_query], top_k=top_k
        )

        try:
            response = self.vector_stub.VectorSearch(request, timeout=self.timeout)

            results = []
            if response.results and response.results.results:
                for result in response.results.results:
                    results.append(
                        SearchResult(
                            id=result.id,
                            score=result.score,
                            vector=list(result.vector) if result.vector else None,
                            metadata=dict(result.metadata) if result.metadata else {},
                        )
                    )

            return results
        except grpc.RpcError as e:
            raise ProximaDBError(f"gRPC error: {e.details()}")

    def _search_vectors_rest(
        self,
        collection_id: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> SearchResult:
        """Search vectors via REST"""
        payload = {
            "vector": vector,
            "top_k": top_k,
        }
        if filters:
            payload["filters"] = [
                {"field": key, "op": "eq", "value": value}
                for key, value in filters.items()
            ]

        url = urljoin(self.base_url, f"/api/v2/collections/{collection_id}/search")
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            return SearchResult(
                results=data.get("results", []),
                total_found=data.get("total_found", 0),
                collection_id=collection_id,
            )
        except requests.RequestException as e:
            raise NetworkError(f"REST request failed: {e}")

    def get_vector(self, collection_id: str, vector_id: str) -> VectorRecord | None:
        """Get a vector by ID"""
        if self.protocol == "grpc":
            return self._get_vector_grpc(collection_id, vector_id)
        else:
            return self._get_vector_rest(collection_id, vector_id)

    def _get_vector_grpc(
        self, collection_id: str, vector_id: str
    ) -> VectorRecord | None:
        """Get vector via gRPC"""
        request = vector_types_pb2.VectorGetRequest(
            collection_id=collection_id, vector_id=vector_id
        )

        try:
            response = self.vector_stub.VectorGet(request, timeout=self.timeout)
            if response.success and response.results and response.results.results:
                result = response.results.results[0]
                return VectorRecord(
                    id=result.id,
                    vector=list(result.vector) if result.vector else [],
                    metadata=dict(result.metadata) if result.metadata else {},
                )
            return None
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise ProximaDBError(f"gRPC error: {e.details()}")

    def _get_vector_rest(
        self, collection_id: str, vector_id: str
    ) -> VectorRecord | None:
        """Get vector via REST"""
        url = urljoin(
            self.base_url,
            f"/api/v2/collections/{collection_id}/records/{vector_id}?include_vector=true&include_text=false",
        )
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()

            if data.get("id"):
                return VectorRecord(
                    id=data["id"],
                    vector=data.get("vector", []),
                    metadata=data.get("props", {}),
                )
            return None
        except requests.RequestException as e:
            raise NetworkError(f"REST request failed: {e}")

    # SQL operations
    def execute_sql(
        self, query: str, parameters: list[Any] | None = None
    ) -> dict[str, Any]:
        """Execute SQL query"""
        if self.protocol == "grpc":
            return self._execute_sql_grpc(query, parameters)
        else:
            return self._execute_sql_rest(query, parameters)

    def _execute_sql_grpc(
        self, query: str, parameters: list[Any] | None = None
    ) -> dict[str, Any]:
        """Execute SQL via gRPC"""
        # Convert parameters to proto SqlValue format if provided
        proto_parameters = []
        if parameters:
            for param in parameters:
                proto_param = self._convert_to_sql_value(param)
                proto_parameters.append(proto_param)

        request = types_pb2.ExecuteQueryRequest(
            query=query, parameters=proto_parameters
        )

        try:
            response = self.sql_stub.ExecuteQuery(request, timeout=self.timeout)

            # Convert response to dictionary format
            rows = []
            for row in response.rows:
                row_dict = {}
                # SqlRow has repeated SqlRowField with key/value pairs
                for field in row.fields:
                    row_dict[field.key] = self._convert_from_sql_value(field.value)
                rows.append(row_dict)

            return {
                "rows": rows,
                "rows_scanned": response.rows_scanned,
                "rows_returned": response.rows_returned,
                "execution_time_ms": getattr(response, "execution_time_ms", 0),
            }
        except grpc.RpcError as e:
            raise ProximaDBError(f"SQL gRPC error: {e.details()}")

    def _execute_sql_rest(
        self, query: str, parameters: list[Any] | None = None
    ) -> dict[str, Any]:
        """Execute SQL via REST (canonical v2 query facade, language="uql")."""
        payload: dict[str, Any] = {
            "language": "uql",
            "query": query,
        }
        if parameters:
            payload["parameters"] = parameters

        url = urljoin(self.base_url, "/api/v2/query")
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise NetworkError(f"REST request failed: {e}")

    def health_check(self) -> dict[str, Any]:
        """Check server health"""
        url = urljoin(self.base_url, "/health")
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise NetworkError(f"Health check failed: {e}")

    def _convert_to_sql_value(self, value: Any) -> types_pb2.SqlValue:
        """Convert Python value to SQL proto value without flattening nested data."""
        from google.protobuf.struct_pb2 import NullValue

        if value is None:
            return types_pb2.SqlValue(null_value=NullValue.NULL_VALUE)
        if isinstance(value, bool):
            return types_pb2.SqlValue(bool_value=value)
        if isinstance(value, int) and not isinstance(value, bool):
            return types_pb2.SqlValue(int64_value=value)
        if isinstance(value, float):
            return types_pb2.SqlValue(number_value=value)
        if isinstance(value, str):
            return types_pb2.SqlValue(string_value=value)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return types_pb2.SqlValue(bytes_value=bytes(value))
        if isinstance(value, (list, tuple)):
            array_values = [self._convert_to_sql_value(item) for item in value]
            return types_pb2.SqlValue(
                array_value=types_pb2.SqlArray(values=array_values)
            )
        if isinstance(value, dict):
            object_fields = {
                str(key): self._convert_to_sql_value(val) for key, val in value.items()
            }
            return types_pb2.SqlValue(
                object_value=types_pb2.SqlObject(fields=object_fields)
            )
        return types_pb2.SqlValue(string_value=str(value))

    def _convert_from_sql_value(self, sql_value: types_pb2.SqlValue) -> Any:
        """Convert SQL proto value to Python value"""
        # Check which field is set using HasField
        if sql_value.HasField("string_value"):
            return sql_value.string_value
        elif sql_value.HasField("number_value"):
            return sql_value.number_value
        elif sql_value.HasField("int64_value"):
            return sql_value.int64_value
        elif sql_value.HasField("bool_value"):
            return sql_value.bool_value
        elif sql_value.HasField("bytes_value"):
            return bytes(sql_value.bytes_value)
        elif sql_value.HasField("jsonb_value"):
            # types.proto tag 9: JSONB bytes — surface the raw bytes like
            # bytes_value; decoding to a JSON document is the caller's choice.
            return bytes(sql_value.jsonb_value)
        elif sql_value.HasField("null_value"):
            return None
        elif sql_value.HasField("array_value"):
            return [
                self._convert_from_sql_value(item)
                for item in sql_value.array_value.values
            ]
        elif sql_value.HasField("object_value"):
            return {
                key: self._convert_from_sql_value(value)
                for key, value in sql_value.object_value.fields.items()
            }
        else:
            # Unknown field, return None
            return None

    # === GRAPH SEARCH OPERATIONS ===

    def create_node(
        self,
        node_id: str,
        labels: list[str],
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        """Create a graph node"""
        if self.protocol == "grpc":
            return self._create_node_grpc(node_id, labels, properties, embedding)
        else:
            return self._create_node_rest(node_id, labels, properties, embedding)

    def _create_node_grpc(
        self,
        node_id: str,
        labels: list[str],
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        """Create node via gRPC"""
        node_properties = {}
        if properties:
            for key, value in properties.items():
                node_properties[key] = self._convert_to_property_value(value)

        node = graph_pb2.Node(id=node_id, labels=labels, properties=node_properties)

        if embedding:
            # Add embedding if provided (assuming EmbeddingVersion is available)
            pass  # TODO: Implement embedding handling when proto is clarified

        request = graph_pb2.CreateNodeRequest(node=node)

        try:
            response = self.graph_stub.CreateNode(request, timeout=self.timeout)
            return self._convert_node_from_proto(response)
        except grpc.RpcError as e:
            raise ProximaDBError(f"Graph gRPC error: {e.details()}")

    def _create_node_rest(
        self,
        node_id: str,
        labels: list[str],
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        """Create node via REST"""
        payload = {
            "id": node_id,
            "labels": labels,
            "properties": properties or {},
        }
        if embedding:
            payload["embedding"] = embedding

        try:
            response = requests.post(
                urljoin(self.base_url, "/api/v2/graphs/default/nodes"),
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise NetworkError(f"Graph REST error: {e}")

    def create_edge(
        self,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
        weight: float | None = None,
    ) -> dict[str, Any]:
        """Create a graph edge"""
        if self.protocol == "grpc":
            return self._create_edge_grpc(
                edge_id, from_node_id, to_node_id, edge_type, properties, weight
            )
        else:
            return self._create_edge_rest(
                edge_id, from_node_id, to_node_id, edge_type, properties, weight
            )

    def _create_edge_grpc(
        self,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
        weight: float | None = None,
    ) -> dict[str, Any]:
        """Create edge via gRPC"""
        edge_properties = {}
        if properties:
            for key, value in properties.items():
                edge_properties[key] = self._convert_to_property_value(value)

        edge = graph_pb2.Edge(
            id=edge_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            edge_type=edge_type,
            properties=edge_properties,
        )

        if weight is not None:
            edge.weight = weight

        request = graph_pb2.CreateEdgeRequest(edge=edge)

        try:
            response = self.graph_stub.CreateEdge(request, timeout=self.timeout)
            return self._convert_edge_from_proto(response)
        except grpc.RpcError as e:
            raise ProximaDBError(f"Graph gRPC error: {e.details()}")

    def _create_edge_rest(
        self,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
        weight: float | None = None,
    ) -> dict[str, Any]:
        """Create edge via REST"""
        payload = {
            "id": edge_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "edge_type": edge_type,
            "properties": properties or {},
        }
        if weight is not None:
            payload["weight"] = weight

        try:
            response = requests.post(
                urljoin(self.base_url, "/api/v2/graphs/default/edges"),
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise NetworkError(f"Graph REST error: {e}")

    def traverse_graph(
        self,
        start_node_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
        node_labels: list[str] | None = None,
        algorithm: str = "BFS",
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Traverse graph from a starting node"""
        if self.protocol == "grpc":
            return self._traverse_graph_grpc(
                start_node_id, max_depth, edge_types, node_labels, algorithm, limit
            )
        else:
            return self._traverse_graph_rest(
                start_node_id, max_depth, edge_types, node_labels, algorithm, limit
            )

    def _traverse_graph_grpc(
        self,
        start_node_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
        node_labels: list[str] | None = None,
        algorithm: str = "BFS",
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Traverse graph via gRPC"""
        # Map algorithm string to enum
        algorithm_enum = graph_pb2.TRAVERSAL_ALGORITHM_BFS
        if algorithm.upper() == "DFS":
            algorithm_enum = graph_pb2.TRAVERSAL_ALGORITHM_DFS
        elif algorithm.upper() == "PARALLEL_BFS":
            algorithm_enum = graph_pb2.TRAVERSAL_ALGORITHM_PARALLEL_BFS

        request = graph_pb2.TraversalRequest(
            start_node_id=start_node_id,
            max_depth=max_depth,
            edge_types=edge_types or [],
            node_labels=node_labels or [],
            algorithm=algorithm_enum,
        )

        if limit is not None:
            request.limit = limit

        try:
            response = self.graph_stub.TraverseGraph(request, timeout=self.timeout)

            return {
                "nodes": [
                    self._convert_node_from_proto(node) for node in response.nodes
                ],
                "edges": [
                    self._convert_edge_from_proto(edge) for edge in response.edges
                ],
                "paths": [
                    self._convert_path_from_proto(path) for path in response.paths
                ],
                "stats": {
                    "nodes_visited": response.stats.nodes_visited,
                    "edges_traversed": response.stats.edges_traversed,
                    "max_depth_reached": response.stats.max_depth_reached,
                    "execution_time_microseconds": response.stats.execution_time_microseconds,
                },
            }
        except grpc.RpcError as e:
            raise ProximaDBError(f"Graph traversal gRPC error: {e.details()}")

    def _traverse_graph_rest(
        self,
        start_node_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
        node_labels: list[str] | None = None,
        algorithm: str = "BFS",
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Traverse graph via REST"""
        payload = {
            "start_node_id": start_node_id,
            "max_depth": max_depth,
            "edge_types": edge_types or [],
            "node_labels": node_labels or [],
            "algorithm": algorithm.upper(),
        }
        if limit is not None:
            payload["limit"] = limit

        try:
            response = requests.post(
                urljoin(self.base_url, "/api/v2/graphs/default/traverse"),
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise NetworkError(f"Graph traversal REST error: {e}")

    def query_nodes(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Query nodes by labels and properties"""
        if self.protocol == "grpc":
            return self._query_nodes_grpc(labels, properties, limit, offset)
        else:
            return self._query_nodes_rest(labels, properties, limit, offset)

    def _query_nodes_grpc(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Query nodes via gRPC"""
        filters = []
        if properties:
            for key, value in properties.items():
                filters.append(
                    graph_pb2.PropertyFilter(
                        key=key,
                        operator=graph_pb2.PROPERTY_FILTER_OPERATOR_EQUALS,
                        value=self._convert_to_property_value(value),
                    )
                )

        request = graph_pb2.NodeQuery(labels=labels or [], filters=filters)

        if limit is not None:
            request.limit = limit
        if offset is not None:
            request.offset = offset

        try:
            response = self.graph_stub.QueryNodes(request, timeout=self.timeout)
            return {
                "success": response.success,
                "nodes": [
                    self._convert_node_from_proto(node) for node in response.nodes
                ],
                "total_count": len(response.nodes),
            }
        except grpc.RpcError as e:
            raise ProximaDBError(f"Node query gRPC error: {e.details()}")

    def _query_nodes_rest(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Query nodes via REST"""
        payload = {"labels": labels or [], "properties": properties or {}}
        if limit is not None:
            payload["limit"] = limit
        if offset is not None:
            payload["offset"] = offset

        try:
            response = requests.post(
                urljoin(self.base_url, "/api/v2/graphs/default/query/nodes"),
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise NetworkError(f"Node query REST error: {e}")

    # === HYBRID SEARCH OPERATIONS ===

    def hybrid_search(
        self,
        collection_id: str,
        vector: list[float],
        top_k: int = 10,
        start_node_id: str | None = None,
        max_depth: int = 2,
        combination_strategy: str = "VECTOR_THEN_GRAPH",
        edge_types: list[str] | None = None,
        vector_filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search combining vector similarity and graph traversal"""
        if self.protocol == "grpc":
            return self._hybrid_search_grpc(
                collection_id,
                vector,
                top_k,
                start_node_id,
                max_depth,
                combination_strategy,
                edge_types,
                vector_filters,
                limit,
            )
        else:
            return self._hybrid_search_rest(
                collection_id,
                vector,
                top_k,
                start_node_id,
                max_depth,
                combination_strategy,
                edge_types,
                vector_filters,
                limit,
            )

    def _hybrid_search_grpc(
        self,
        collection_id: str,
        vector: list[float],
        top_k: int = 10,
        start_node_id: str | None = None,
        max_depth: int = 2,
        combination_strategy: str = "VECTOR_THEN_GRAPH",
        edge_types: list[str] | None = None,
        vector_filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search via gRPC"""
        # Create vector search request
        search_query = vector_types_pb2.SearchQuery(vector=vector)
        if vector_filters:
            # Convert filters to SqlValue format
            for key, value in vector_filters.items():
                search_query.filters[key] = self._convert_to_sql_value(value)

        vector_search_request = vector_types_pb2.VectorSearchRequest(
            collection_id=collection_id, queries=[search_query], top_k=top_k
        )

        # Create graph traversal request (if start_node_id provided)
        graph_traversal_request = None
        if start_node_id:
            graph_traversal_request = graph_pb2.TraversalRequest(
                start_node_id=start_node_id,
                max_depth=max_depth,
                edge_types=edge_types or [],
            )

        # Map combination strategy
        strategy_enum = graph_pb2.COMBINATION_STRATEGY_VECTOR_THEN_GRAPH
        if combination_strategy.upper() == "GRAPH_THEN_VECTOR":
            strategy_enum = graph_pb2.COMBINATION_STRATEGY_GRAPH_THEN_VECTOR
        elif combination_strategy.upper() == "BALANCED":
            strategy_enum = graph_pb2.COMBINATION_STRATEGY_BALANCED

        request = graph_pb2.HybridSearchRequest(
            vector_search_request=vector_search_request,
            combination_strategy=strategy_enum,
        )

        if graph_traversal_request:
            request.graph_traversal_request = graph_traversal_request
        if limit is not None:
            request.limit = limit

        try:
            response = self.graph_stub.ExecuteHybridQuery(request, timeout=self.timeout)

            return {
                "nodes": [
                    self._convert_node_from_proto(node) for node in response.nodes
                ],
                "edges": [
                    self._convert_edge_from_proto(edge) for edge in response.edges
                ],
                "paths": [
                    self._convert_path_from_proto(path) for path in response.paths
                ],
                "vector_results": [
                    self._convert_search_result_from_proto(result)
                    for result in response.vector_results
                ],
                "stats": {
                    "vector_results_count": response.stats.vector_results_count,
                    "graph_traversal_count": response.stats.graph_traversal_count,
                    "execution_time_microseconds": response.stats.execution_time_microseconds,
                },
            }
        except grpc.RpcError as e:
            raise ProximaDBError(f"Hybrid search gRPC error: {e.details()}")

    def _hybrid_search_rest(
        self,
        collection_id: str,
        vector: list[float],
        top_k: int = 10,
        start_node_id: str | None = None,
        max_depth: int = 2,
        combination_strategy: str = "VECTOR_THEN_GRAPH",
        edge_types: list[str] | None = None,
        vector_filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search via REST"""
        payload = {
            "vector_search": {
                "collection_id": collection_id,
                "vector": vector,
                "top_k": top_k,
                "filters": vector_filters or {},
            },
            "combination_strategy": combination_strategy.upper(),
        }

        if start_node_id:
            payload["graph_traversal"] = {
                "start_node_id": start_node_id,
                "max_depth": max_depth,
                "edge_types": edge_types or [],
            }

        if limit is not None:
            payload["limit"] = limit

        try:
            response = requests.post(
                urljoin(self.base_url, "/api/v2/hybrid/search"),
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise NetworkError(f"Hybrid search REST error: {e}")

    # === ENHANCED VECTOR SEARCH ===

    def advanced_vector_search(
        self,
        collection_id: str,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_vector: bool = False,
        include_metadata: bool = True,
        accuracy_threshold: float | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Enhanced vector search with advanced parameters"""
        if self.protocol == "grpc":
            return self._advanced_vector_search_grpc(
                collection_id,
                vector,
                top_k,
                filters,
                include_vector,
                include_metadata,
                accuracy_threshold,
                search_params,
            )
        else:
            return self._advanced_vector_search_rest(
                collection_id,
                vector,
                top_k,
                filters,
                include_vector,
                include_metadata,
                accuracy_threshold,
                search_params,
            )

    def _advanced_vector_search_grpc(
        self,
        collection_id: str,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_vector: bool = False,
        include_metadata: bool = True,
        accuracy_threshold: float | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Advanced vector search via gRPC"""
        # Create search query with filters
        search_query = vector_types_pb2.SearchQuery(vector=vector)
        if filters:
            for key, value in filters.items():
                search_query.filters[key] = self._convert_to_sql_value(value)

        # Setup include fields
        include_fields = vector_types_pb2.IncludeFields(
            vector=include_vector, metadata=include_metadata, score=True
        )

        # Setup search parameters
        search_params_proto = vector_types_pb2.SearchParams()
        if accuracy_threshold is not None:
            search_params_proto.accuracy_threshold = accuracy_threshold

        if search_params:
            for key, value in search_params.items():
                if key == "timeout_ms":
                    search_params_proto.timeout_ms = value
                elif key == "enable_two_stage":
                    search_params_proto.enable_two_stage = value
                elif key == "enable_clustering_hint":
                    search_params_proto.enable_clustering_hint = value
                elif key == "enable_metadata_filtering_hint":
                    search_params_proto.enable_metadata_filtering_hint = value

        request = vector_types_pb2.VectorSearchRequest(
            collection_id=collection_id,
            queries=[search_query],
            top_k=top_k,
            include_fields=include_fields,
            search_params=search_params_proto,
        )

        try:
            response = self.vector_stub.SearchVectors(request, timeout=self.timeout)

            results = []
            for result_list in response.results:
                for result in result_list.results:
                    results.append(self._convert_search_result_from_proto(result))

            return {
                "results": results,
                "total_count": len(results),
                "execution_time_ms": getattr(response, "execution_time_ms", 0),
            }
        except grpc.RpcError as e:
            raise ProximaDBError(f"Advanced vector search gRPC error: {e.details()}")

    def _advanced_vector_search_rest(
        self,
        collection_id: str,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_vector: bool = False,
        include_metadata: bool = True,
        accuracy_threshold: float | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Advanced vector search via REST"""
        payload = {
            "vector": vector,
            "top_k": top_k,
            "include_vector": include_vector,
            "include_text": False,
            "filters": [
                {"field": key, "op": "eq", "value": value}
                for key, value in (filters or {}).items()
            ],
        }

        if accuracy_threshold is not None:
            payload["accuracy_threshold"] = accuracy_threshold

        if search_params:
            payload["search_params"] = search_params

        try:
            response = requests.post(
                urljoin(self.base_url, f"/api/v2/collections/{collection_id}/search"),
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise NetworkError(f"Advanced vector search REST error: {e}")

    # === HELPER METHODS FOR PROTO CONVERSIONS ===

    def _convert_to_property_value(self, value: Any) -> graph_pb2.PropertyValue:
        """Convert Python value to PropertyValue proto"""
        if isinstance(value, str):
            return graph_pb2.PropertyValue(string_value=value)
        elif isinstance(value, bool):
            return graph_pb2.PropertyValue(bool_value=value)
        elif isinstance(value, int):
            return graph_pb2.PropertyValue(int_value=value)
        elif isinstance(value, float):
            return graph_pb2.PropertyValue(double_value=value)
        elif isinstance(value, bytes):
            return graph_pb2.PropertyValue(bytes_value=value)
        elif isinstance(value, list):
            array_values = [self._convert_to_property_value(item) for item in value]
            return graph_pb2.PropertyValue(
                array_value=graph_pb2.PropertyArray(values=array_values)
            )
        elif isinstance(value, dict):
            object_fields = {
                k: self._convert_to_property_value(v) for k, v in value.items()
            }
            return graph_pb2.PropertyValue(
                object_value=graph_pb2.PropertyObject(fields=object_fields)
            )
        else:
            # Default to string representation
            return graph_pb2.PropertyValue(string_value=str(value))

    def _convert_from_property_value(self, prop_value: graph_pb2.PropertyValue) -> Any:
        """Convert PropertyValue proto to Python value"""
        if prop_value.HasField("string_value"):
            return prop_value.string_value
        elif prop_value.HasField("int_value"):
            return prop_value.int_value
        elif prop_value.HasField("double_value"):
            return prop_value.double_value
        elif prop_value.HasField("bool_value"):
            return prop_value.bool_value
        elif prop_value.HasField("bytes_value"):
            return prop_value.bytes_value
        elif prop_value.HasField("array_value"):
            return [
                self._convert_from_property_value(item)
                for item in prop_value.array_value.values
            ]
        elif prop_value.HasField("object_value"):
            return {
                k: self._convert_from_property_value(v)
                for k, v in prop_value.object_value.fields.items()
            }
        else:
            return None

    def _convert_node_from_proto(self, node: graph_pb2.Node) -> dict[str, Any]:
        """Convert Node proto to dictionary"""
        return {
            "id": node.id,
            "labels": list(node.labels),
            "properties": {
                k: self._convert_from_property_value(v)
                for k, v in node.properties.items()
            },
            "created_at": (
                node.created_at.ToDatetime().isoformat()
                if node.HasField("created_at")
                else None
            ),
            "updated_at": (
                node.updated_at.ToDatetime().isoformat()
                if node.HasField("updated_at")
                else None
            ),
        }

    def _convert_edge_from_proto(self, edge: graph_pb2.Edge) -> dict[str, Any]:
        """Convert Edge proto to dictionary"""
        return {
            "id": edge.id,
            "from_node_id": edge.from_node_id,
            "to_node_id": edge.to_node_id,
            "edge_type": edge.edge_type,
            "properties": {
                k: self._convert_from_property_value(v)
                for k, v in edge.properties.items()
            },
            "weight": edge.weight if edge.HasField("weight") else None,
            "created_at": (
                edge.created_at.ToDatetime().isoformat()
                if edge.HasField("created_at")
                else None
            ),
            "updated_at": (
                edge.updated_at.ToDatetime().isoformat()
                if edge.HasField("updated_at")
                else None
            ),
        }

    def _convert_path_from_proto(self, path) -> list[str]:
        """Convert GraphPath proto to list of node IDs"""
        # Assuming GraphPath has node_ids field - adjust based on actual proto definition
        if hasattr(path, "node_ids"):
            return list(path.node_ids)
        else:
            return []

    def _convert_search_result_from_proto(self, result) -> dict[str, Any]:
        """Convert SearchVectorRecord proto to dictionary"""
        return {
            "id": result.id,
            "score": result.score,
            "vector": list(result.vector) if result.vector else None,
            "metadata": {
                k: self._convert_from_sql_value(v) for k, v in result.metadata.items()
            },
            "similarity": result.similarity if hasattr(result, "similarity") else None,
            "timestamp": result.timestamp if hasattr(result, "timestamp") else None,
            "source": result.source if hasattr(result, "source") else None,
        }


# Convenience function for backwards compatibility
def create_client_v1(**kwargs) -> ProximaDBClientV1:
    """Create a v1 protocol client"""
    return ProximaDBClientV1(**kwargs)
