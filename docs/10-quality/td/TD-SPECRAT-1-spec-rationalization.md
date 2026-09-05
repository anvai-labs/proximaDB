// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
= TD-SPECRAT-1: spec rationalization — expose live-but-unspec'd REST surfaces in controlled per-area waves

[cols="1,3", options="header"]
|===
| Field | Value
| Status | Waves 1-5 landed (observability+nl, stub-rule fixes, ABAC, collections admin, catalog routing/explain); wave 6 (graph surface) landing
| Severity | Medium — ~100 route literals across ~15 areas are live on the wire but invisible to every SDK
| Component | `src/network/rest/openapi_supplement.yaml`; `docs/openapi/proximadb-openapi.yaml`; `clients/*` (regenerated)
| Relates to | [[TD-SSO-3]] (the census that surfaced the gap); TD-126 (spec-from-code); ADR-041
|===

== The gap

`make verify-openapi-spec` only diffs *annotated* handlers against the committed
spec. A mounted-but-unannotated route is therefore silently born unexposed: it
works over raw REST but no generated SDK (Python/Go/TS/Rust — all spec-driven,
TD-126) can see it. A 2026-09-03 census counted ~100 such route literals across
~15 areas (ABAC, catalog, collections admin, observability, events,
model-registries, external-collections, timeseries, graph analytics, streaming,
prepared-SQL, Iceberg-REST, legacy v1, primary-pod, progressive/rank search).

Notably `/api/v2/nl/translate` was LIVE (mounted, real LLM calls) while absent
from the spec — found by the TD-SSO-3 adversarial fact-check.

== The program (maintainer direction, 2026-09-03)

Expose the unexposed in **controlled, reasoned, rationalized iterations — one
area at a time** — then rebuild the SDKs from the specs. Per area:

1. **Rationalize** each endpoint: does it stay as-is, and is it REAL?
   Fabricated-success stubs are NOT exposed (below).
2. **Expose** via the supplement yaml (the established staging pattern for
   handlers that live outside the root crate's annotated set) or utoipa
   annotations when the handler is in the root crate.
3. **Regenerate** the spec + all four SDKs in the same PR (drift gates enforce).
4. **Adversarial review** per PR; LLM/ranked surfaces additionally need eval
   suites (repo directive 13).

Later waves migrate supplement entries → in-code annotations mechanically
(requires utoipa in `proximadb-api` or moving DTOs; deliberately deferred).

== Wave 1: observability + nl/translate

**Exposed (6 paths; spec 39 → 45):**

* `POST /api/v2/observability/namespaces` — createObservabilityNamespace
* `POST /api/v2/observability/namespaces/{ns}/logs/bulk` — ingestLogs
* `POST /api/v2/observability/namespaces/{ns}/metrics` — ingestMetric
* `POST /api/v2/observability/namespaces/{ns}/metrics/bulk` — ingestMetrics
* `POST /api/v2/observability/namespaces/{ns}/metrics/aggregate` — aggregateMetrics
* `POST /api/v2/nl/translate` — translateNaturalLanguage (AV-SQL 3-agent flow;
  real LLM calls; output is model-generated — the schema says so)

All carry typed request/response schemas (not loose `additionalProperties`),
default-value documentation, and 400/401 (500 for the LLM path) error refs.

**Deliberately NOT exposed — stubs (rationalization rule: never expose
fabricated success):**

* `POST …/metrics/promql` — returns an empty vector; "Full PromQL wiring comes
  with the CHRONO engine" (documented stub).
* `POST …/traces/bulk`, `POST …/traces/search` — return success/empty WITHOUT
  touching the observability port (silent-Ok anti-pattern). Exposure deferred
  until the trace path is actually wired.

**Already exposed via supplement (unchanged):** logs + logs/search.

== Follow-ups filed by this wave

* Eval suite for `/api/v2/nl/translate` (directive 13 — model-generated
  surface; prompts + rubric versioned when behavior changes).
* Trace ingest/query: wire the port or return 501 — then expose.
* Waves 2+: area order per product priority (observability diagnostics,
  ABAC, collections admin, ...).

== Wave 3: ABAC control-plane (landed)

**Exposed (9 paths / 14 operations; spec 45 → 54):** the complete
`/api/v2/abac/*` operator surface from
`src/network/rest/canonical/abac_admin.rs` — policy-bindings
(PUT/DELETE/GET), attribute-bindings (POST/GET), predicate-objects
(PUT/GET/DELETE/list), grants (POST/list/DELETE, ADR-090 entitlement
layer), and tenant-posture (GET/PUT, TD-SEC-2).

Rationalization notes specific to this surface:

* **Real, not stubs:** every handler writes through the same durable
  `FileSystem*` stores the live enforcer reads (hot-reload, no restart);
  the handlers carry their own operator-gate + fail-closed test suite.
* **Deployment gating documented, not hidden:** these routes are
  registered only under `--features abac-policy` (default OFF); the spec
  notes that a default build 404s the path rather than returning the
  documented error shapes.
* **Error envelope carried verbatim:** the surface's flat
  `{error, message, code}` `OperatorErrorResponse` is documented as its own
  schema (`AbacOperatorErrorResponse`) rather than being normalized into
  the canonical nested `ErrorResponse` — what the handler emits is what
  the spec says.
* **`FilterExpression` modeled as free-form JSON** (`AbacFilterExpression`,
  additionalProperties): the recursive externally-tagged union cannot be
  expressed as a self-referential `oneOf` that openapi-python-client
  processes (it drops the schema and every referencing endpoint); the
  description documents the exact construction shape. This is the one
  untyped model on the surface — recorded as a deliberate trade.

**Tooling defect found and fixed (root cause, both generators):**
`clients/{go,rust}/codegen/openapi_31_to_30.py` down-converts the
serde_yaml-emitted (YAML 1.2) spec with PyYAML, which resolves scalars per
YAML 1.1 — bare `Off` (and `On`/`Yes`/`No`/`y`/`n`) parsed as *booleans*.
Effect: progenitor hard-failed ("type error unexpected value type") while
oapi-codegen **silently emitted `False AbacGrantEnforcement = "false"`**
into the committed Go SDK shape. Fix: a `SafeLoader` subclass restricted
to the YAML 1.2 core-schema boolean set (`true|false` + case variants), so
`Off` stays the string `"Off"` while real booleans parse unchanged. A
full-spec scan confirmed only the two new ABAC enums (`Off`,
`Null` — `Null` is quoted by serde_yaml itself) were ever exposed to the
hazard, so no previously-committed generated code changes.

**Adversarial-review ratchets:** the contract gate now asserts the exact
9-path / 14-operation ABAC surface, verifies YAML 1.2 boolean handling, and
requires `grant_enforcement` in `AbacTenantSecurityPosture`. The latter caught
and corrected a response-schema mismatch: the Rust handler's non-optional
field is always serialized, while the initial supplement allowed generated
SDKs to treat it as absent.

**Waves 4+ (candidate order, product-priority pending):** collections
admin (affinity/pinning/primary-pod/branch-merge), catalog routing/explain,
graph analytics (fusion-search, impact-analysis), timeseries, streaming
ingest, prepared-SQL, external-collections, model-registries, events,
progressive/rank search. Legacy `/api/v1` is excluded (retiring per
SUPPORTED_SURFACE); Iceberg-REST is excluded (an external standard
protocol consumed directly by Spark/Trino — not our OpenAPI→SDK loop).

== Wave 4: collections-admin operator surfaces (landed)

**Exposed (6 paths / 10 operations; spec 54 → 60):**

* `/api/v2/collections/{collection_id}/pin` — PATCH set/clear + GET read
  (`pinning.rs`; the pin registry, hot for the access-pattern engine).
* `/api/v2/collections/pinning` — GET list.
* `/api/v2/collections/{collection_id}/affinity` — GET inspect + DELETE
  invalidate (`affinity.rs`; per-node cache-affinity registry).
* `/api/v2/collections/affinity` — GET list.
* `/api/v2/primary-pod/{tenant_id}/{collection_id}` — GET lookup + PUT
  assign + DELETE unassign (`primary_pod.rs`; WAL write routing).
* `/api/v2/primary-pod` — GET list.

Rationalization notes specific to this surface:

* **Real, not stubs:** every handler reads/writes the live registry the
  router/policy engine consults; pinning matches the operator UX contract
  (immediate ack, movement out of band); primary-pod PUT mirrors to the
  catalog with mirror-failure-is-not-fatal semantics (documented in the
  spec).
* **Permission asymmetry documented, not hidden:** pinning + affinity
  carry NO per-route permission gate — and the auth MIDDLEWARE itself
  only attaches when REST auth is enabled + a security coordinator
  exists (`src/network/rest/server.rs`; the shipped default config
  disables it). In an auth-disabled deployment the six pinning/affinity
  operations (cluster-wide, cross-tenant, state-mutating) are reachable
  UNAUTHENTICATED, and every primary-pod call 401s (`missing_auth_context`)
  because the operator gate has no context to read. Primary-pod is
  operator-gated (`SystemAdmin` ∪ `ConfigureSystem`) inside each
  handler. The spec says which is which per operation, and every
  primary-pod op carries the "requires REST auth enabled" availability
  note.
* **Tenant scoping documented:** pinning/affinity registries are
  collection-keyed and cross-tenant by construction; primary-pod is
  `(tenant_id, collection_id)`-scoped via the path. Every op carries the
  "X-Tenant-ID header not consulted" note (wave-3 convention).
* **Internally-tagged responses** (`{"status": "pinned", ...}`) modeled
  as a single object with a `status` enum + variant fields optional —
  wire-exact and codegen-robust (unlike a `oneOf`, which openapi-python-
  client cannot discriminate on an internal tag).
* **Plain-text error body on the pin 400** (the axum `(StatusCode,
  String)` path) documented as `text/plain` — not silently dressed up as
  a JSON envelope.
* **Stale doc comments corrected in-wave:** the handler doc comments in
  `pinning.rs` / `affinity.rs` / `primary_pod.rs` said `/api/v1/...`
  while the mount is `/api/v2/...` — corrected here (comment-only .rs
  edits) since this PR's purpose is discoverability of exactly these
  files.

**Deliberately NOT exposed — deferred out of this wave:**

* `POST /api/v2/collections/{collection_id}/branches/{branch}/merge`
  (`merge_graph_branch`) — the original census lumped it into
  "collections admin", but it is the GRAPH branch-merge (reads/filters
  the canonical WAL, `merge_branches`, write-back) with a free-form
  `serde_json::Value` response; it belongs to the graph-analytics wave
  and needs its response schema rationalized first. It stays
  unexposed until then — NOT silently done with this wave.

**No behavioral code changes** — all handlers are live and mounted
unconditionally (no feature gate, unlike ABAC wave 3); the only .rs
diff is the doc-comment v1→v2 correction above.

**Adversarial-review ratchets:** the contract gate now also asserts the
exact 6-path / 10-operation collections-admin surface plus its enum
vocabularies (`PinTarget`, `AssignmentReason`, the internally-tagged
`status` enums) — `test_collections_admin_surface_has_the_complete_operation_set`,
matching the wave-3 precedent.

== Wave 5: catalog routing/explain (landed)

**Exposed (2 paths / 2 operations; spec 60 → 62):**

* `GET /api/v2/catalog/table-routing` — `getCatalogTableRouting`: the
  `information_schema.table_routing` projection as a table-shaped
  result (all cells strings — the same shape pgwire/SQL clients see),
  optional `table_name` exact-match query filter.
* `POST /api/v2/catalog/table-write/explain` — `explainTableWriteRoute`:
  the DML write planner in explain-only mode (nothing is written) —
  selected backend + access method, write lane (+ rejected lanes with
  reasons), candidate/rejected paths, estimated cost + data movement,
  required guards, write intent. 11 new schemas (the 9-schema
  `TableWriteRouteExplanation` tree + the request + the introspection
  result; plus the shared `InternalError` response entry) model the
  **complete** tree — round-1 adversarial review caught 7 dropped
  fields (5 on every response: `stats_freshness`, `constraint_gaps`,
  `lossy`, `support_status`, `batch_local_constraints_sufficient`, plus
  optional `benchmark_gate`/`freshness_sla_ms`) and the ratchet now
  pins every response property set.

Rationalization notes:

* **Alias-accepting request fields are strings, not enums**:
  `write_mode`/`distribution` accept many case-insensitive spellings
  (`insert_only`, `insert-only`, …) — modeled as plain strings with the
  accepted vocabulary documented; the RESPONSE echoes the resolved
  canonical PascalCase enum name (e.g. `PseudoDistributed`) — also
  documented.
* **No per-route permission gate** on either op (global auth middleware
  applies when enabled — same posture note as wave 4); X-Tenant-ID
  not-consulted notes carried (tenant context rides the explain body's
  `tenant_id` when present).
* **Reused `InternalError` response** added to the shared responses
  (canonical error envelope) — these handlers go through `ApiResult`
  mapping, unlike wave-4's plain-text/flat-error paths.

**Deliberately NOT exposed — deferred out of this wave:**

* `/api/v2/catalogs/*` (enterprise external-catalog surface, the
  `enterprise-catalogs` feature gate) — a large separate area (Iceberg/
  Delta/UC federation); needs its own census + rationalization wave.
* `POST /api/v2/collections/{collection_id}/branches/{branch}/merge` —
  still deferred to the graph-analytics wave (wave-4 decision).

== Wave 6: graph surface truth + gap exposure (this PR)

**Pre-existing defect FIXED — the published graph surface modeled the
wrong response shape.** Every `/api/v2/graphs/*` handler (api-crate
`graph.rs`, mounted under `/api/v2`) returns the envelope
`{success, data?, error?, metadata?}` with a SCREAMING_SNAKE_CASE error
code — but the previously-published supplement schemas (NodeResponse,
EdgeResponse, TraverseResponse, …) were FLAT, and NodeResponse also
dropped `embedding`/`created_at`/`updated_at`. Every generated SDK's
graph return model could not parse real server responses. Wave 6
remodels the whole surface to the envelope (`Graph*Response` schemas
with pinned property sets in the contract gate) with the complete
`CanonicalNode`/`CanonicalEdge` payloads. Note: `fusion-search` and
`impact-analysis` are utoipa-annotated in the root crate (drift-gated
half) and were already correct.

**Exposed (21 paths / 28 operations; spec 62 → 74):** the 9 existing
supplement graph paths envelope-corrected + 12 new paths — schema (PUT),
nodes/{id} PUT (updateNode was missing entirely), neighbors GET,
edges/{id} GET/PUT/DELETE, walk, step, shortest-path, query +
query/nodes + query/edges, components, cycles, constraints/unique
POST+DELETE.

Honesty notes: create-node/create-edge take a WRAPPED body
(`{node: …}` / `{edge: …}`); update ops take the bare input with the
path id overriding `id`; 201s on creates; delete-graph returns 204
(the handler's success body is not transmitted on 204); per-item
batch rejections ride `failed_count`/`errors[]` at HTTP 200;
`continuation_token` is the `offset:<n>` protocol; graph-collection
payloads are open objects (serialized port records); `language`/
`timeout_ms` on query are accepted but unused server-side (dead
params documented as such); error statuses carry the GRAPH envelope
(`GraphErrorBody`), not the canonical error envelope; extractor
rejections are plain text.

**Deliberately NOT exposed — deferred:**

* `/api/v2/graphs/{graph_id}/rag` — `not_implemented_handler` (honest
  501); expose when the port lands.
* graph branch-merge (wave-4 deferral; free-form response).
* Legacy 308-redirect shims (`/api/v2/nodes|edges|stats`) — clients
  must use the canonical graph-scoped paths.

**Adversarial-review correction:** the initial wave-6 spec and generated-model
updates fixed the declared envelopes, but the hand-written TypeScript and Rust
facades still used flat response mocks. Graph lists exposed serialized
`GraphCollection` records as if they were SDK-facing `GraphInfo` values; the
live records use `graph_id`, an optional/possibly-empty display `name`, and
counts nested under `stats.total_nodes` / `stats.total_edges`. Node lookup and
traversal likewise consumed the envelope itself as the payload instead of
lowering `CanonicalNode` / `CanonicalEdge`. Finally, batch helpers read the
nonexistent `data.count` and could report every requested item as successful
despite canonical `created_count` / `failed_count` partial results. Both
facades now explicitly unwrap and lower every public graph result seam, fall
back from an empty display name to `graph_id`, preserve nested counts and
embeddings, and return `created_count` for batches. Contract tests pin the real
list, get, node, traversal, and partial-batch envelopes so flat fake responses
cannot mask these boundaries again.
