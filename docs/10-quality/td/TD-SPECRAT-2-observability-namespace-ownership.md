// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
= TD-SPECRAT-2: observability namespace ownership — the multi-tenant binding decision

[cols="1,3", options="header"]
|===
| Field | Value
| Status | Filed (product decision required before SaaS multi-tenant GA)
| Severity | High for multi-tenant SaaS deployments; inert for single-tenant-per-server
| Component | `crates/modalities/proximadb-observability`; `crates/platform/proximadb-api/src/rest/canonical/observability.rs`
| Relates to | [[TD-SPECRAT-1]] (wave-1 exposure surfaced this)
|===

== The gap (wave-1 adversarial-review finding)

The observability surface scopes by the **path namespace**, and that namespace
maps 1:1 onto the storage tenant key — `base_observability_record()` sets
`record.tenant_id = namespace.to_string()`
(`crates/modalities/proximadb-observability/src/lib.rs:207-209`). Storage-level
separation between namespaces therefore exists. What does NOT exist is any
binding between the **requesting tenant** and the namespace:

* the handlers extract no `TenantContext` (zero tenant references in the
  handler file);
* the unified auth middleware has no per-path RBAC entries for
  observability paths (`determine_required_permission` has none);
* consequently ANY authenticated caller can create/ingest/query ANY
  observability namespace.

All seven exposed observability ops (the five wave-1 additions AND the two
pre-existing loose entries — ingestLog/queryLogs, upgraded in wave 2) now
carry per-path scope notes ("path namespace IS the isolation boundary;
X-Tenant-ID not consulted") plus the accurate availability gating: mounted
unconditionally on the unified server (the default, port 5678); legacy
multi-port mode mounts it only with gRPC enabled; cluster-mode REST does not
mount it.

== Why this is a product decision, not a mechanical fix

Two defensible designs, and the choice changes the API contract:

1. **Namespace-as-tenant (current, implicit):** namespaces are global
   admin-created scopes; the deployment itself is single-tenant-per-server
   (the current deployment model — ds2). The honest fix is exactly the spec
   carve wave 2 ships, plus optionally restricting the surface to admin
   roles.
2. **Namespace-owned-by-tenant (SaaS):** a namespace belongs to the creating
   tenant; every read/write checks ownership. This requires an ownership
   record on `ObservabilityNamespaceConfig`, an ownership check at each
   handler, per-path RBAC entries in the middleware, and a decision on what
   the `tenant_id` storage key becomes (composite `tenant/namespace`?).

Plumbing option 2 speculatively — without the product call on cross-tenant
sharing, admin visibility, and billing attribution (observability storage is
metered KSU per mandate) — would bake an API contract we may have to break.

== The decision needed

For SaaS multi-tenant GA: who may create a namespace, who may ingest into an
existing one, and can tenants read each other's namespaces (ops/support
flows)? Until answered, the surface is documented as
deployment-scoped (option 1) and this TD blocks on product input.

== Wave-2 companion changes (this PR)

* Spec per-path scope notes (the honest carve above) on all five exposed
  observability ops.
* promql + traces endpoints now return **501 Not Implemented** instead of
  fabricated success/empty shapes (the silent-Ok anti-pattern).
