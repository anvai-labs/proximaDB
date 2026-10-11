"""OpenAPI contract gate for the Python REST SDK.

For every covered SDK method, this test:
  1. Monkey-patches ``_make_request`` to capture the outbound request and
     return a programmed minimal-valid response.
  2. Calls the SDK method with a sample input.
  3. Validates the captured request body / query parameters against the
     corresponding OpenAPI operation defined in
     ``docs/openapi/proximadb-openapi.yaml``.

It runs entirely offline — no server is started. Drift between the SDK and
the OpenAPI contract therefore fails this test deterministically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from jsonschema import Draft202012Validator, RefResolver

from proximadb_sdk.models import CollectionConfig, ColumnDefinition, SchemaDefinition
from proximadb_sdk.protocols.rest_sync import ProximaDBClient

# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = REPO_ROOT / "docs" / "openapi" / "proximadb-openapi.yaml"


class _Yaml12CoreBoolLoader(yaml.SafeLoader):
    """SafeLoader restricted to YAML 1.2 core-schema booleans.

    The published spec is emitted by serde_yaml (YAML 1.2), where bare
    ``Off``/``On``/``Yes``/``No`` are plain strings. PyYAML resolves them per
    YAML 1.1 as *booleans*, silently corrupting enum values (``Off`` -> False)
    in whatever this gate validates. Same narrowing as the Go/Rust
    down-converters (clients/{go,rust}/codegen/openapi_31_to_30.py).

    Mirrors the loader patch from TD-SPECRAT-1 wave 3; keep the copies in sync.
    """

    pass


_Yaml12CoreBoolLoader.yaml_implicit_resolvers = {
    first_char: [
        (
            tag,
            (
                re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
                if tag == "tag:yaml.org,2002:bool"
                else regexp
            ),
        )
        for tag, regexp in resolvers
    ]
    for first_char, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


@pytest.fixture(scope="module")
def openapi_spec() -> dict[str, Any]:
    assert SPEC_PATH.exists(), f"OpenAPI spec missing at {SPEC_PATH}"
    with SPEC_PATH.open() as f:
        return yaml.load(f, Loader=_Yaml12CoreBoolLoader)


@pytest.fixture(scope="module")
def ref_resolver(openapi_spec):
    return RefResolver.from_schema(openapi_spec)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _operation_for(
    spec: dict[str, Any], path_template: str, method: str
) -> dict[str, Any]:
    paths = spec.get("paths", {})
    assert path_template in paths, f"Path {path_template} not in OpenAPI spec"
    op = paths[path_template].get(method.lower())
    assert op is not None, f"{method} {path_template} not defined in spec"
    return op


def _path_parameters(spec: dict[str, Any], path_template: str) -> list[dict[str, Any]]:
    """Return parameter objects declared at the path level (shared params)."""
    raw = spec["paths"][path_template].get("parameters", [])
    resolved: list[dict[str, Any]] = []
    for p in raw:
        if "$ref" in p:
            resolved.append(_resolve_ref(spec, p["$ref"]))
        else:
            resolved.append(p)
    return resolved


def _resolve_ref(spec: dict[str, Any], ref: str) -> dict[str, Any]:
    assert ref.startswith("#/"), f"unsupported $ref form: {ref}"
    node: Any = spec
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _request_body_schema(
    spec: dict[str, Any], operation: dict[str, Any]
) -> dict[str, Any] | None:
    body = operation.get("requestBody")
    if not body:
        return None
    schema = body["content"]["application/json"]["schema"]
    if "$ref" in schema:
        schema = _resolve_ref(spec, schema["$ref"])
    return schema


def _validate(schema: dict[str, Any], instance: Any, resolver: RefResolver) -> None:
    Draft202012Validator(schema, resolver=resolver).validate(instance)


def test_abac_tenant_posture_response_requires_enforcement_mode(openapi_spec):
    """The handler always serializes this non-optional Rust field."""
    schema = openapi_spec["components"]["schemas"]["AbacTenantSecurityPosture"]
    assert "grant_enforcement" in schema["required"]


def test_yaml_12_loader_preserves_abac_enforcement_enum_scalars():
    """YAML 1.1's bool aliases must not corrupt the published enum values."""
    parsed = yaml.load(
        "values: [Off, On, Yes, No, y, n, true, false]",
        Loader=_Yaml12CoreBoolLoader,
    )
    assert parsed == {"values": ["Off", "On", "Yes", "No", "y", "n", True, False]}


def test_abac_surface_has_the_complete_operation_set(openapi_spec):
    expected_operations = {
        "putPolicyBinding",
        "deletePolicyBinding",
        "listPolicyBindings",
        "getTenantPosture",
        "putTenantPosture",
        "postGrant",
        "listGrants",
        "deleteGrant",
        "listAttributeBindings",
        "postAttributeBinding",
        "listPredicateObjects",
        "getPredicateObject",
        "putPredicateObject",
        "deletePredicateObject",
    }
    abac_paths = {
        path: item
        for path, item in openapi_spec["paths"].items()
        if path.startswith("/api/v2/abac/")
    }
    actual_operations = {
        operation["operationId"]
        for item in abac_paths.values()
        for method, operation in item.items()
        if method in {"get", "put", "post", "delete", "patch"}
    }

    assert len(abac_paths) == 9
    assert actual_operations == expected_operations


def test_collections_admin_surface_has_the_complete_operation_set(openapi_spec):
    """TD-SPECRAT-1 wave 4: pinning + affinity + primary-pod (6 paths / 10 ops)."""
    expected_operations = {
        "patchCollectionPin",
        "getCollectionPin",
        "listCollectionPins",
        "getCollectionAffinity",
        "deleteCollectionAffinity",
        "listCollectionAffinities",
        "getPrimaryPod",
        "putPrimaryPod",
        "deletePrimaryPod",
        "listPrimaryPods",
    }
    admin_paths = {
        path: item
        for path, item in openapi_spec["paths"].items()
        if path.startswith("/api/v2/primary-pod")
        or (
            path.startswith("/api/v2/collections/")
            and ("/pin" in path or "affinity" in path)
        )
    }
    actual_operations = {
        operation["operationId"]
        for item in admin_paths.values()
        for method, operation in item.items()
        if method in {"get", "put", "post", "delete", "patch"}
    }

    assert len(admin_paths) == 6
    assert actual_operations == expected_operations

    # The enum vocabularies must match the Rust serde wire format exactly.
    schemas = openapi_spec["components"]["schemas"]
    assert schemas["PinTarget"]["enum"] == ["memory", "nvme_ssd", "cloud"]
    assert schemas["AssignmentReason"]["enum"] == [
        "create",
        "operator",
        "failover",
        "rebalance",
        "catalog_replay",
    ]
    assert schemas["PinResponse"]["properties"]["status"]["enum"] == [
        "pinned",
        "unpinned",
    ]
    assert schemas["PrimaryPodLookupResponse"]["properties"]["status"]["enum"] == [
        "bound",
        "unbound",
    ]


def test_catalog_routing_surface_has_the_complete_operation_set(openapi_spec):
    """TD-SPECRAT-1 wave 5: catalog table-routing + table-write explain."""
    catalog_paths = {
        path: item
        for path, item in openapi_spec["paths"].items()
        if path.startswith("/api/v2/catalog/")
    }
    actual_operations = {
        operation["operationId"]
        for item in catalog_paths.values()
        for method, operation in item.items()
        if method in {"get", "put", "post", "delete", "patch"}
    }

    assert len(catalog_paths) == 2
    assert actual_operations == {"getCatalogTableRouting", "explainTableWriteRoute"}

    # The explain request carries the source XOR rule and the alias
    # vocabularies as strings (not enums) — the required/optional split
    # must match the Rust handler exactly.
    schemas = openapi_spec["components"]["schemas"]
    explain = schemas["TableWriteExplainRequest"]
    assert explain["required"] == ["target_table"]
    for optional_field in (
        "source_table",
        "source_sql",
        "write_mode",
        "distribution",
        "target_columns",
        "tenant_id",
        "actor",
        "idempotency_key",
        "row_count_hint",
        "estimated_bytes",
        "requires_row_level_semantics",
        "batch_local_constraints_sufficient",
    ):
        assert optional_field in explain["properties"], optional_field
    # The introspection result is table-shaped, all strings.
    intro = schemas["CatalogIntrospectionResult"]
    assert intro["required"] == ["columns", "column_types", "rows"]

    # Response property sets are pinned EXACTLY — the round-1 review
    # caught 7 dropped fields that every gate was blind to (SDK models
    # silently discarded response fields the server always emits).
    expected_properties = {
        "TableWriteRouteExplanation": {
            "target_table",
            "source",
            "write_mode",
            "distribution",
            "write_intent",
            "write_lane",
            "write_lane_reason",
            "write_lane_required_guards",
            "rejected_write_lanes",
            "selected_backend",
            "selected_access_method",
            "estimated_cost",
            "data_movement",
            "required_guards",
            "route_metadata",
            "candidate_paths",
            "rejected_paths",
            "execution_elapsed_us",
            "execution_rows_written",
        },
        "TableWriteIntentExplanation": {
            "target_table",
            "operation_kind",
            "durability",
            "isolation",
            "projection_freshness",
            "tenant_id",
            "actor",
            "idempotency_key",
            "catalog_schema_version",
            "row_count_hint",
            "estimated_bytes",
            "requires_row_level_semantics",
            "batch_local_constraints_sufficient",
        },
        "TableWriteDataMovementExplanation": {
            "source_rows",
            "source_bytes",
            "target_rows_before_write",
            "target_bytes_before_write",
            "estimated_read_bytes",
            "estimated_write_bytes",
            "estimated_rewrite_bytes",
            "estimate_source",
            "source_last_analyzed_ms",
            "target_last_analyzed_ms",
            "source_stats_age_ms",
            "target_stats_age_ms",
            "freshness_sla_ms",
            "stats_freshness",
        },
        "TableWriteRouteMetadataExplanation": {
            "authority_mode",
            "workload_profile",
            "storage_specialization",
            "primary_format",
            "preferred_compute_route",
            "partitioning",
            "isolation_profile",
            "freshness_sla",
            "projection_freshness_state",
            "projection_metadata",
            "policy_boundary",
            "constraint_enforcement",
            "constraint_gaps",
        },
        "ProjectionRouteMetadataExplanation": {
            "name",
            "kind",
            "physical_format",
            "rebuild_source",
            "freshness",
            "freshness_state",
            "max_lag_ms",
            "source_range",
            "last_included_position",
            "rebuildable",
            "invalidation_policy",
            "policy_boundary",
            "lossy",
            "support_status",
            "benchmark_gate",
        },
    }
    for schema_name, expected in expected_properties.items():
        actual = set(schemas[schema_name]["properties"])
        assert actual == expected, (
            f"{schema_name} property set drifted: "
            f"missing={expected - actual} extra={actual - expected}"
        )


def test_graph_surface_has_the_complete_operation_set(openapi_spec):
    """TD-SPECRAT-1 wave 6: graph surface — envelope-corrected + gap ops."""
    expected_operations = {
        "createGraph",
        "listGraphs",
        "getGraph",
        "deleteGraph",
        "updateGraphSchema",
        "createNode",
        "getNode",
        "updateNode",
        "deleteNode",
        "getNodeNeighbors",
        "createEdge",
        "getEdge",
        "updateEdge",
        "deleteEdge",
        "batchCreateNodes",
        "batchCreateEdges",
        "traverseGraph",
        "walkGraph",
        "stepGraph",
        "shortestPath",
        "executeGraphQuery",
        "queryNodes",
        "queryEdges",
        "getConnectedComponents",
        "checkCycles",
        "addUniqueConstraint",
        "removeUniqueConstraint",
        "getGraphStats",
        # The two utoipa-annotated ops (root crate, drift-gated core —
        # NOT the supplement): counted in the 23 paths.
        "fusion_search_v2",
        "impact_analysis_v2",
    }
    graph_paths = {
        path: item
        for path, item in openapi_spec["paths"].items()
        if path.startswith("/api/v2/graphs")
    }
    actual_operations = {
        operation["operationId"]
        for item in graph_paths.values()
        for method, operation in item.items()
        if method in {"get", "put", "post", "delete", "patch"}
    }

    assert len(graph_paths) == 23
    assert actual_operations == expected_operations

    # Wave 6 corrected the published response shape: every graph op
    # returns the {success, data?, error?, metadata?} envelope. The
    # previously-published FLAT models could not parse the real wire
    # format — pin the envelope + payload property sets so neither can
    # silently regress.
    schemas = openapi_spec["components"]["schemas"]
    envelope = {"success", "data", "error", "metadata"}
    for envelope_schema in (
        "GraphNodeResponse",
        "GraphEdgeResponse",
        "GraphNodeListResponse",
        "GraphCollectionResponse",
        "GraphCollectionListResponse",
        "GraphTraversalResponse",
        "GraphStatsResponse",
        "GraphBatchNodesResponse",
        "GraphBatchEdgesResponse",
        "GraphNodeQueryResponse",
        "GraphEdgeQueryResponse",
        "GraphShortestPathResponse",
        "GraphQueryResponse",
        "GraphComponentsResponse",
        "GraphCyclesResponse",
        "GraphDdlResponse",
    ):
        actual = set(schemas[envelope_schema]["properties"])
        assert (
            actual == envelope
        ), f"{envelope_schema} drifted: missing={envelope - actual} extra={actual - envelope}"
        assert schemas[envelope_schema]["required"] == ["success"]

    expected_payloads = {
        "CanonicalNode": {
            "id",
            "labels",
            "properties",
            "embedding",
            "created_at",
            "updated_at",
        },
        "CanonicalEdge": {
            "id",
            "from_node_id",
            "to_node_id",
            "edge_type",
            "properties",
            "weight",
            "created_at",
            "updated_at",
        },
        "GraphStats": {
            "total_nodes",
            "total_edges",
            "label_stats",
            "edge_type_stats",
            "total_properties",
            "memory_usage_bytes",
            "average_degree",
            "max_degree",
            "connected_components",
        },
        "GraphTraversalData": {"nodes", "edges", "paths", "stats"},
        "GraphShortestPathData": {"path", "total_weight", "found"},
        "GraphNodeBatchResults": {
            "created_count",
            "updated_count",
            "failed_count",
            "results",
            "errors",
        },
        "GraphNodeQueryResults": {"items", "total_count", "has_more", "next_token"},
        "GraphErrorBody": {"code", "message", "details"},
    }
    for schema_name, expected in expected_payloads.items():
        actual = set(schemas[schema_name]["properties"])
        assert (
            actual == expected
        ), f"{schema_name} drifted: missing={expected - actual} extra={actual - expected}"
    assert schemas["GraphErrorBody"]["properties"]["code"]["enum"] == [
        "NOT_FOUND",
        "ALREADY_EXISTS",
        "INVALID_ARGUMENT",
        "CONSTRAINT_VIOLATION",
        "INTERNAL_ERROR",
        "TIMEOUT",
        "PERMISSION_DENIED",
    ]

    # Error statuses carry the FULL envelope (success=false + error),
    # not the bare error body — pin the error-response component shape
    # (round-1 review: the bare-body form made generated SDKs KeyError
    # on real 404/500s).
    for error_component in ("GraphBadRequest", "GraphNotFound", "GraphInternal"):
        ref = openapi_spec["components"]["responses"][error_component]["content"][
            "application/json"
        ]["schema"]["$ref"]
        assert ref == "#/components/schemas/GraphErrorResponse", error_component
    err = schemas["GraphErrorResponse"]
    assert set(err["properties"]) == {"success", "error", "metadata"}
    assert err["required"] == ["success", "error"]

    # Edge-side payloads + the remaining data payloads (round-1 gap).
    expected_payloads.update(
        {
            "GraphEdgeBatchResults": {
                "created_count",
                "updated_count",
                "failed_count",
                "results",
                "errors",
            },
            "GraphEdgeQueryResults": {"items", "total_count", "has_more", "next_token"},
            "GraphDdlData": {"success"},
            "GraphCyclesData": {"has_cycle"},
            "GraphComponentsData": {"components"},
            "GraphQueryData": {"rows", "row_count"},
            "GraphEmbedding": {"model_id", "model_version", "vector", "dimension"},
            "GraphTraversalStats": {
                "nodes_visited",
                "edges_traversed",
                "max_depth_reached",
                "execution_time_ms",
            },
        }
    )
    for schema_name, expected in expected_payloads.items():
        actual = set(schemas[schema_name]["properties"])
        assert (
            actual == expected
        ), f"{schema_name} drifted: missing={expected - actual} extra={actual - expected}"

    # Required lists pinned for the always-serialized payloads.
    for schema_name, required in {
        "CanonicalNode": ["id", "labels", "properties", "created_at", "updated_at"],
        "CanonicalEdge": [
            "id",
            "from_node_id",
            "to_node_id",
            "edge_type",
            "properties",
            "created_at",
            "updated_at",
        ],
        "GraphStats": [
            "total_nodes",
            "total_edges",
            "label_stats",
            "edge_type_stats",
            "total_properties",
            "memory_usage_bytes",
            "average_degree",
            "max_degree",
            "connected_components",
        ],
    }.items():
        assert schemas[schema_name]["required"] == required, schema_name


def test_timeseries_surface_has_the_complete_operation_set(openapi_spec):
    """TD-SPECRAT-1 wave 7: time-series (TD-TS-1) — 5 paths / 6 ops."""
    expected_operations = {
        "createTimeseriesCollection",
        "listTimeseriesCollections",
        "deleteTimeseriesCollection",
        "ingestTimeseries",
        "queryTimeseries",
        "aggregateTimeseries",
    }
    ts_paths = {
        path: item
        for path, item in openapi_spec["paths"].items()
        if path.startswith("/api/v2/timeseries/")
    }
    actual_operations = {
        operation["operationId"]
        for item in ts_paths.values()
        for method, operation in item.items()
        if method in {"get", "put", "post", "delete", "patch"}
    }

    assert len(ts_paths) == 5
    assert actual_operations == expected_operations

    schemas = openapi_spec["components"]["schemas"]
    expected_properties = {
        "TsPoint": {"timestamp", "values", "tags"},
        "TsCollectionConfig": {
            "name",
            "timestamp_column",
            "value_columns",
            "tag_columns",
            "retention_ms",
        },
        "TsValueColumn": {"name", "unit", "aggregation"},
        "TsIngestRequest": {"points"},
        "TsIngestResponse": {"ingested"},
        "TsQueryRequest": {"start_time", "end_time", "limit"},
        "TsQueryResponse": {"points"},
        "TsAggregateRequest": {"start_time", "end_time", "aggregation", "bucket_ms"},
        "TsAggregateResponse": {"buckets"},
        "TsCreateResponse": {"name"},
        "TsListResponse": {"collections"},
        "TsDeleteResponse": {"success"},
    }
    for schema_name, expected in expected_properties.items():
        actual = set(schemas[schema_name]["properties"])
        assert (
            actual == expected
        ), f"{schema_name} drifted: missing={expected - actual} extra={actual - expected}"
    assert schemas["TsPoint"]["required"] == ["timestamp"]
    assert schemas["TsCollectionConfig"]["required"] == ["name"]


def test_rank_surface_and_program_census(openapi_spec):
    """TD-SPECRAT-1 wave 8: rank search exposure + census resolutions."""
    op = openapi_spec["paths"]["/api/v2/rank/search"]["post"]
    assert op["operationId"] == "rankSearch"
    assert sorted(op["responses"]) == [
        "200",
        "400",
        "404",
        "415",
        "422",
        "500",
        "501",
    ]
    assert set(op["responses"]["400"]["content"]) == {"application/json"}
    assert set(op["responses"]["415"]["content"]) == {"application/json"}
    assert set(op["responses"]["422"]["content"]) == {"application/json"}

    schemas = openapi_spec["components"]["schemas"]
    expected = {
        "RankSearchRequest": {
            "collection",
            "query_vector",
            "query_text",
            "k",
            "rank_profile",
            "rank_overrides",
        },
        "RankOverrides": {"second_phase", "global_phase"},
        "RankPhaseOverride": {"rerank_count", "batch_size"},
        "RankScoredHit": {
            "id",
            "score",
            "score_vector",
            "match_features",
            "summary_features",
        },
        "RankScoreVector": {"primary", "phase", "components"},
        "ScoreComponent": {"name", "value", "weight", "contribution"},
        "RankSearchResponse": {
            "hits",
            "phase_truncated",
            "rank_profile",
            "rank_profile_version",
        },
    }
    for schema_name, exp in expected.items():
        actual = set(schemas[schema_name]["properties"])
        assert (
            actual == exp
        ), f"{schema_name} drifted: missing={exp - actual} extra={actual - exp}"
    assert schemas["RankSearchRequest"]["required"] == ["collection"]
    assert schemas["RankSearchResponse"]["required"] == ["hits", "phase_truncated"]
    assert schemas["RankScoredHit"]["required"] == ["id", "score"]
    assert schemas["RankScoreVector"]["required"] == ["primary", "phase"]
    assert schemas["ScoreComponent"]["required"] == [
        "name",
        "value",
        "weight",
        "contribution",
    ]

    # Census dispositions (d)/(e) pinned: model-registries present in the
    # generated core, hybrid search/index present in the supplement.
    all_paths = openapi_spec["paths"]
    assert sum(1 for p in all_paths if p.startswith("/api/v2/model-registries")) == 4
    assert "/api/v2/hybrid/search" in all_paths and "/api/v2/hybrid/index" in all_paths


def test_document_surface_has_the_complete_operation_set(openapi_spec):
    """TD-SPECRAT-1 wave 9: document collections — 7 paths / 13 ops."""
    expected_operations = {
        "createDocumentCollection",
        "listDocumentCollections",
        "getDocumentCollection",
        "deleteDocumentCollection",
        "insertDocument",
        "queryDocuments",
        "getDocument",
        "updateDocument",
        "deleteDocument",
        "batchInsertDocuments",
        "aggregateDocuments",
        "createDocumentIndex",
        "listDocumentIndexes",
    }
    doc_paths = {
        path: item
        for path, item in openapi_spec["paths"].items()
        if path.startswith("/api/v2/document-collections")
    }
    actual_operations = {
        operation["operationId"]
        for item in doc_paths.values()
        for method, operation in item.items()
        if method in {"get", "put", "post", "delete", "patch"}
    }

    assert len(doc_paths) == 7
    assert actual_operations == expected_operations

    # createDocumentIndex is the HONEST always-400 (unsupported op).
    idx = doc_paths["/api/v2/document-collections/{collection}/indexes"]["post"]
    assert sorted(idx["responses"]) == ["400"]

    schemas = openapi_spec["components"]["schemas"]
    expected_properties = {
        "DocResponse": {"id", "document", "version"},
        "DocQueryResponse": {"documents", "total_count", "has_more"},
        "DocUpdateRequest": {"updates", "expected_version"},
        "DocUpdateResponse": {"success", "id", "new_version"},
        "DocBatchInsertRequest": {"documents"},
        "DocBatchInsertResponse": {"inserted", "failed"},
        "DocAggregateRequest": {"pipeline"},
        "DocAggregateResponse": {"results", "query_time_ms"},
        "DocIndexDefinition": {"name", "path", "index_type", "unique", "sparse"},
        "CreateDocumentCollectionRequest": {"name", "indexes"},
        "DocInsertRequest": {"id", "document"},
        "DocIndexListResponse": {"indexes"},
    }
    for schema_name, exp in expected_properties.items():
        actual = set(schemas[schema_name]["properties"])
        assert (
            actual == exp
        ), f"{schema_name} drifted: missing={exp - actual} extra={actual - exp}"
    assert schemas["DocResponse"]["required"] == ["id", "document", "version"]
    assert schemas["DocUpdateResponse"]["required"] == ["success", "id", "new_version"]

    # Round-1 review: pin the list 200 top-level TYPE (an object carrying
    # `collections` — the array-typed predecessor broke every SDK).
    list_200 = doc_paths["/api/v2/document-collections"]["get"]["responses"]["200"]
    assert list_200["content"]["application/json"]["schema"]["$ref"].endswith(
        "DocCollectionListResponse"
    )
    assert schemas["DocCollectionListResponse"]["required"] == ["collections"]
    assert (
        schemas["DocCollectionListResponse"]["properties"]["collections"]["type"]
        == "array"
    )
    assert schemas["DocCreateCollectionResponse"]["required"] == [
        "success",
        "collection",
    ]

    # Census resolutions pinned (wave 8; unified updated by wave 10):
    # WS streaming is not REST-spec'd and no REST events surface exists.
    # (The unified surface WAS all-501 at wave 8; wave 10 exposed it as
    # LIVE — see test_unified_surface_has_the_complete_operation_set.)
    paths = openapi_spec["paths"]
    assert not any(
        p.startswith("/ws/") for p in paths
    ), "WS streaming must not be REST-spec'd"
    assert not any(
        p.startswith("/api/v2/events") for p in paths
    ), "no REST events surface exists"


def test_unified_surface_has_the_complete_operation_set(openapi_spec):
    """TD-SPECRAT-1 wave 10: unified query (LIVE) — 9 paths / 9 ops."""
    expected_operations = {
        "executeUnifiedQuery",
        "executeMultiModelQuery",
        "executeFederatedQuery",
        "executeDistributedQuery",
        "explainUnifiedQuery",
        "prepareStatement",
        "executePreparedStatement",
        "deletePreparedStatement",
        "getPreparedStats",
    }
    unified_paths = {
        path: item
        for path, item in openapi_spec["paths"].items()
        if path.startswith("/api/v2/unified")
    }
    actual_operations = {
        operation["operationId"]
        for item in unified_paths.values()
        for method, operation in item.items()
        if method in {"get", "put", "post", "delete", "patch"}
    }

    assert len(unified_paths) == 9
    assert actual_operations == expected_operations

    # Distinctive contracts: prepare = 201 with a statement_id; prepared
    # DELETE = 204; every JSON extractor rejection uses the same BARE
    # {error: string} shape as the handler errors.
    prep = unified_paths["/api/v2/unified/prepare"]["post"]["responses"]
    assert sorted(prep) == ["201", "400", "415", "422", "500", "501"]
    dele = unified_paths["/api/v2/unified/prepared/{statement_id}"]["delete"][
        "responses"
    ]
    assert sorted(dele) == ["204", "500", "501"]
    schemas = openapi_spec["components"]["schemas"]
    assert schemas["UnifiedBareError"]["required"] == ["error"]
    assert schemas["UnifiedPrepareResponse"]["required"] == ["statement_id"]

    json_operations = [
        ("/api/v2/unified/execute", "post"),
        ("/api/v2/unified/multi-model", "post"),
        ("/api/v2/unified/federated", "post"),
        ("/api/v2/unified/distributed", "post"),
        ("/api/v2/unified/explain", "post"),
        ("/api/v2/unified/prepare", "post"),
        ("/api/v2/unified/execute/{statement_id}", "post"),
        ("/api/v2/unified/prepared/stats", "post"),
    ]
    for path, method in json_operations:
        responses = unified_paths[path][method]["responses"]
        assert {"400", "415", "422"} <= set(responses)
        for status in ("400", "415", "422"):
            response = responses[status]
            if "$ref" in response:
                response = _resolve_ref(openapi_spec, response["$ref"])
            schema = response["content"]["application/json"]["schema"]
            assert schema["$ref"].endswith("UnifiedBareError")

    distributed = schemas["UnifiedDistributedRequest"]
    assert set(distributed["properties"]) == {"query", "limit"}


# ---------------------------------------------------------------------------
# Capturing transport
# ---------------------------------------------------------------------------


class _CapturedRequest:
    def __init__(self) -> None:
        self.method: str | None = None
        self.endpoint: str | None = None
        self.json_body: Any = None
        self.params: dict[str, Any] | None = None


def _patched_client(
    monkeypatch, response_body: Any, status_code: int = 200
) -> tuple[ProximaDBClient, _CapturedRequest]:
    client = ProximaDBClient(url="http://contract.test")
    captured = _CapturedRequest()

    def fake_make_request(self, method: str, endpoint: str, **kwargs):  # noqa: ARG001
        captured.method = method
        captured.endpoint = endpoint
        captured.json_body = kwargs.get("json")
        captured.params = kwargs.get("params")
        return httpx.Response(
            status_code=status_code,
            request=httpx.Request(method, f"http://contract.test{endpoint}"),
            content=json.dumps(response_body).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr(ProximaDBClient, "_make_request", fake_make_request)
    return client, captured


# ---------------------------------------------------------------------------
# Per-operation contract checks
# ---------------------------------------------------------------------------


def test_get_health_live_matches_spec(monkeypatch, openapi_spec, ref_resolver):
    response_body = {"status": "ok"}
    client, captured = _patched_client(monkeypatch, response_body)

    probe = client.live()

    # Operation exists; path matches.
    op = _operation_for(openapi_spec, "/health/live", "get")
    assert op["operationId"] == "getLiveness"
    assert captured.method == "GET"
    assert captured.endpoint == "/health/live"

    # Response shape conforms.
    resp_schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    if "$ref" in resp_schema:
        resp_schema = _resolve_ref(openapi_spec, resp_schema["$ref"])
    _validate(resp_schema, probe.model_dump(), ref_resolver)


def test_get_health_ready_matches_spec(monkeypatch, openapi_spec, ref_resolver):
    response_body = {"status": "ready"}
    client, captured = _patched_client(monkeypatch, response_body)

    probe = client.ready()

    op = _operation_for(openapi_spec, "/health/ready", "get")
    assert op["operationId"] == "getReadiness"
    assert captured.method == "GET"
    assert captured.endpoint == "/health/ready"

    resp_schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    if "$ref" in resp_schema:
        resp_schema = _resolve_ref(openapi_spec, resp_schema["$ref"])
    _validate(resp_schema, probe.model_dump(), ref_resolver)


def test_create_collection_request_matches_spec(
    monkeypatch, openapi_spec, ref_resolver
):
    response_body = {
        "collection_id": "col_abc",
        "name": "my_collection",
        "dimension": 128,
        "engine": "viper",
        "proxima_record_enabled": True,
        "created_at": "2026-05-23T00:00:00Z",
    }
    client, captured = _patched_client(monkeypatch, response_body)

    schema = SchemaDefinition(
        columns=[
            ColumnDefinition(name="title", data_type="text"),
            ColumnDefinition(name="price", data_type="float", filterable=True),
        ]
    )
    client.create_collection(
        "my_collection",
        config=CollectionConfig(name="my_collection", dimension=128),
        schema=schema,
        initial_capacity=10_000,
    )

    op = _operation_for(openapi_spec, "/api/v2/collections", "post")
    assert op["operationId"] == "createCollection"
    assert captured.method == "POST"
    assert captured.endpoint == "/api/v2/collections"

    req_schema = _request_body_schema(openapi_spec, op)
    assert req_schema is not None
    _validate(req_schema, captured.json_body, ref_resolver)

    # Ensure the new pass-throughs actually landed in the body.
    assert captured.json_body["initial_capacity"] == 10_000
    assert "columns" in captured.json_body["schema"]


def test_list_collections_query_params_match_spec(monkeypatch, openapi_spec):
    response_body = {"collections": [], "total_count": 0}
    client, captured = _patched_client(monkeypatch, response_body)

    client.list_collections(limit=25, offset=50, include_stats=True)

    op = _operation_for(openapi_spec, "/api/v2/collections", "get")
    assert op["operationId"] == "listCollections"
    assert captured.method == "GET"

    # Resolve declared parameters and check the SDK uses only spec-defined keys
    # with values that satisfy the parameter schemas.
    declared: dict[str, dict[str, Any]] = {}
    for p in op.get("parameters", []):
        param = _resolve_ref(openapi_spec, p["$ref"]) if "$ref" in p else p
        if param["in"] == "query":
            declared[param["name"]] = param["schema"]

    assert set(captured.params or {}).issubset(
        declared.keys()
    ), f"Unknown query params: {set(captured.params) - declared.keys()}"
    for name, schema in declared.items():
        if name in (captured.params or {}):
            value = captured.params[name]
            # Coerce SDK-supplied booleans serialized as 'true'/'false' strings
            if schema.get("type") == "boolean" and isinstance(value, str):
                value = value.lower() == "true"
            Draft202012Validator(schema).validate(value)


def test_get_schema_matches_spec(monkeypatch, openapi_spec, ref_resolver):
    response_body = {
        "schema_id": "sch_1",
        "schema_version": "v1",
        "collection_id": "col_abc",
        "schema": {
            "columns": [
                {"name": "title", "data_type": "text"},
            ],
        },
        "created_at": "2026-05-23T00:00:00Z",
    }
    client, captured = _patched_client(monkeypatch, response_body)

    schema_resp = client.get_schema("col_abc")

    op = _operation_for(
        openapi_spec, "/api/v2/collections/{collection_id}/schema", "get"
    )
    assert op["operationId"] == "getCollectionSchema"
    assert captured.method == "GET"
    assert captured.endpoint == "/api/v2/collections/col_abc/schema"

    resp_schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    if "$ref" in resp_schema:
        resp_schema = _resolve_ref(openapi_spec, resp_schema["$ref"])

    # by_alias=True so the `schema_` field re-emits as `schema` per OpenAPI;
    # exclude_none avoids emitting Python None for unset optionals (OpenAPI
    # spec does not mark them nullable).
    _validate(
        resp_schema,
        schema_resp.model_dump(by_alias=True, exclude_none=True),
        ref_resolver,
    )


def test_update_schema_request_matches_spec(monkeypatch, openapi_spec, ref_resolver):
    response_body = {
        "schema_id": "sch_2",
        "schema_version": "v2",
        "previous_schema_id": "sch_1",
        "changes": [],
        "warnings": [],
        "updated_at": "2026-05-23T00:00:00Z",
    }
    client, captured = _patched_client(monkeypatch, response_body)

    schema = SchemaDefinition(
        columns=[ColumnDefinition(name="title", data_type="text")],
        enforcement="strict",
    )
    client.update_schema("col_abc", schema, force=True)

    op = _operation_for(
        openapi_spec, "/api/v2/collections/{collection_id}/schema", "put"
    )
    assert op["operationId"] == "updateCollectionSchema"
    assert captured.method == "PUT"
    assert captured.endpoint == "/api/v2/collections/col_abc/schema"

    req_schema = _request_body_schema(openapi_spec, op)
    assert req_schema is not None
    _validate(req_schema, captured.json_body, ref_resolver)
    assert captured.json_body["force"] is True


def test_create_graph_request_matches_spec(monkeypatch, openapi_spec, ref_resolver):
    # Proves the rebased graph create_graph op (full-model round-trip through
    # CreateGraphRequest) emits a spec-conformant method/URL/body via the
    # generated REST client.
    response_body = {"graph_id": "social", "name": "Social", "node_count": 0}
    client, captured = _patched_client(monkeypatch, response_body)

    client.create_graph("social", name="Social", description="people")

    op = _operation_for(openapi_spec, "/api/v2/graphs", "post")
    assert op["operationId"] == "createGraph"
    assert captured.method == "POST"
    assert captured.endpoint == "/api/v2/graphs"

    req_schema = _request_body_schema(openapi_spec, op)
    assert req_schema is not None
    _validate(req_schema, captured.json_body, ref_resolver)
    assert captured.json_body["graph_id"] == "social"


def test_create_node_request_matches_spec(monkeypatch, openapi_spec, ref_resolver):
    # Proves the rebased graph create_node op (sourced via _path_only because the
    # public API's bare-list embedding does not conform to EmbeddingInput) still
    # emits the spec-governed method/URL and a spec-conformant body when no
    # embedding is supplied.
    response_body = {"id": "n1", "labels": ["Person"]}
    client, captured = _patched_client(monkeypatch, response_body)

    client.create_node("n1", ["Person"], properties={"age": 30}, graph_id="social")

    op = _operation_for(openapi_spec, "/api/v2/graphs/{graph_id}/nodes", "post")
    assert op["operationId"] == "createNode"
    assert captured.method == "POST"
    assert captured.endpoint == "/api/v2/graphs/social/nodes"

    req_schema = _request_body_schema(openapi_spec, op)
    assert req_schema is not None
    _validate(req_schema, captured.json_body, ref_resolver)
    assert captured.json_body["node"]["id"] == "n1"
