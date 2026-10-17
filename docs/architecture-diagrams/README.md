# ProximaDB Architecture Diagrams

Structured suite of Mermaid diagrams covering every major ProximaDB subsystem, organized into
**16 layer directories** of increasing detail. **47 diagrams** total (all Mermaid — renders
natively on GitHub).

> **Start here for the single-page system view:** [`atlas.md`](./atlas.md) (all-Mermaid,
> GitHub-native end-to-end tour). This file is the **catalog/index** of the per-topic drill-down
> diagrams below. Format rules: [`RENDERING_STRATEGY.md`](./RENDERING_STRATEGY.md);
> inline zero-tooling diagrams: [`ASCII_ART.md`](./ASCII_ART.md).

## How to view
- `.mermaid` files render in any Mermaid-aware viewer (VS Code Mermaid extension, GitHub when
  embedded in `.md`). `atlas.md` embeds the key diagrams inline so the whole system renders on
  GitHub with zero install.
- Export to PNG/SVG: `bash scripts/diagrams/render_atlas.sh` (Mermaid CLI / Kroki fallback).

## Type legend
**Component** · **Sequence** · **State** · **Use Case** · **Deployment** · **Class** · **Flow**

---

### `00-system-overview/` — Top-level system views
| File | Type | Description |
|---|---|---|
| `00-top-level-architecture.mermaid` | Component | Whole-system component map (clients → server → storage → object store) |
| `01-deployment-diagram.mermaid` | Deployment | C4 level-1 deployment topology |
| `02-usecase-comprehensive.mermaid` | Use Case | Actors × 21 use cases across all modalities |
| `03-usecase-entry-points.mermaid` | Use Case | Use cases keyed by API entry points / endpoints |
| `04-usecase-operators.mermaid` | Use Case | Operator / analyst / SRE-facing use cases |

### `01-network-layer/` — Protocol surfaces & request flow (5678 multiplex, 5433 pgwire, MCP)
| File | Type | Description |
|---|---|---|
| `00-network-overview-flow.mermaid` | Flow | High-level protocol-multiplex flow (clients → :5678 / :5433 / MCP) |
| `01-network-component.mermaid` | Component | Network subsystem detail (MultiServer, multiplex, rest/grpc/arrow/pgwire, middleware) |
| `02-request-flow-sequence.mermaid` | Sequence | REST & gRPC request-processing sequence |

### `02-query-pipeline/` — Query routing & execution (ComputeScheduler, DataFusion/Volcano)
| File | Type | Description |
|---|---|---|
| `01-query-component.mermaid` | Component | Query-pipeline component diagram |
| `02-compute-scheduler-detail.mermaid` | Component | ComputeScheduler engine-selection detail (read-side routing) |
| `03-read-path-sequence.mermaid` | Sequence | Vector-search read-path sequence |
| `04-read-path-detail.mermaid` | Component | Vector-search read-path component detail |
| `05-compute-routing.mermaid` | Flow | `ComputeScheduler::route_select` routing-decision flow |

### `03-storage-layer/` — Engines, WAL, PAX, Iceberg, 2PC, cache
| File | Type | Description |
|---|---|---|
| `01-storage-component.mermaid` | Component | Storage-layer component diagram |
| `02-storage-engine-internals.mermaid` | Component | Storage-engine internal architecture (SST / HELIX / NOVA; VIPER deprecated) |
| `03-write-path-sequence.mermaid` | Sequence | WriteLane routing: WAL → memtable → PAX flush |
| `04-wal-internals.mermaid` | Component | WAL internal component diagram (god-file) |
| `05-pax-block-format.mermaid` | Class | PAX block-format class diagram (custom columnar storage) |
| `06-iceberg-manifest-lifecycle.mermaid` | Sequence | Iceberg manifest lifecycle (warehouse / OLAP tables) |
| `07-transaction-2pc.mermaid` | Component | Transaction coordinator & 2-phase commit |
| `08-cache-subsystem.mermaid` | Component | Cache subsystem (cross-cache orchestrator) |
| `09-write-path-e2e-sequence.mermaid` | Sequence | Write-path E2E with 2PC + background flush/compaction |

### `04-index-layer/` — HNSW, Compact, IVF indices
| File | Type | Description |
|---|---|---|
| `01-index-subsystem.mermaid` | Component | Index subsystem (HNSW + Compact + IVF) |
| `02-hnsw-internals.mermaid` | Component | HNSW index internal architecture |
| `03-index-component.mermaid` | Component | Index-layer component diagram |

### `04-services-catalog/` — Services, RecordStore, catalog, CDC, connectors
| File | Type | Description |
|---|---|---|
| `01-services-component.mermaid` | Component | Services & catalog layer component diagram |
| `02-services-recordstore-internals.mermaid` | Component | Services & record-store internal architecture |
| `03-catalog-metadata.mermaid` | Component | Catalog & metadata subsystem |
| `04-streaming-cdc.mermaid` | Component | Streaming / CDC subsystem |
| `05-connectors-integrations.mermaid` | Component | Connectors & integrations subsystem |

### `05-cluster-distributed/` — Cluster, partition lease, Raft coordination
| File | Type | Description |
|---|---|---|
| `01-cluster-component.mermaid` | Component | Cluster & distributed-systems component diagram |
| `02-partition-lease-state-machine.mermaid` | State | Partition-lease state machine (per-range leaseholder) |

### `07-datafusion-integration/` — DataFusion OLAP (ADR-052)
| File | Type | Description |
|---|---|---|
| `01-datafusion-olap-adr052.mermaid` | Component | DataFusion OLAP integration (ADR-052) |

### `07-graph-subsystem/` — Graph engine (ORION)
| File | Type | Description |
|---|---|---|
| `01-graph-component.mermaid` | Component | Graph subsystem (ORION engine) |
| `02-graph-orion-internals.mermaid` | Component | ORION internal architecture |

### `08-ai-llm/` — AI/LLM subsystem, embedding pipeline
| File | Type | Description |
|---|---|---|
| `01-ai-component.mermaid` | Component | AI & LLM subsystem (component view) |
| `02-ai-llm-subsystem.mermaid` | Component | AI & LLM subsystem (integration & data-flow view) |
| `03-embedding-pipeline.mermaid` | Component | Embedding pipeline (modalities) |

### `09-clients-sdk/` — Client SDKs
| File | Type | Description |
|---|---|---|
| `01-python-sdk-architecture.mermaid` | Component | Python SDK internal architecture (generated + handwritten) |

### `09-coupling-analysis/` — God-file coupling hotspots (see also `README.md`)
| File | Type | Description |
|---|---|---|
| `01-coupling-god-files.mermaid` | Component | Coupling analysis & TD god-file map |

### `09-security-governance/` — Security, tenant isolation, governance
| File | Type | Description |
|---|---|---|
| `01-security-component.mermaid` | Component | Security, auth & governance component diagram |
| `02-security-governance-detail.mermaid` | Component | Security & governance subsystem detail |

### `10-cross-cutting/` — Observability, bootstrap
| File | Type | Description |
|---|---|---|
| `01-observability-subsystem.mermaid` | Component | Observability subsystem (metrics + tracing + auditing) |
| `02-infrastructure-bootstrap.mermaid` | Component | Infrastructure & bootstrap subsystem |

### `10-deployment/` — Deployment topology
| File | Type | Description |
|---|---|---|
| `01-system-deployment.mermaid` | Deployment | System deployment diagram |

### `11-crate-dependency-map/` — Workspace crate layering (CI-enforced)
| File | Type | Description |
|---|---|---|
| `01-crate-workspace-layers.mermaid` | Component | Workspace crate dependency map (layered) |
| `02-crate-layering-ci-enforced.mermaid` | Component | Workspace crate layering (CI enforced) |

---

**Totals:** 47 diagrams across 16 layer directories. All Mermaid; no `.puml`/`.mmd` sidecar formats.
