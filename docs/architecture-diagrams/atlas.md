# ProximaDB Architecture Atlas (GitHub-Native, Mermaid-First)
> Single-page, **natively rendered** view of ProximaDB. Every diagram below is Mermaid, so it
> renders on GitHub with zero install. Each section is a **condensed summary**; the canonical,
> full-detail diagram for every subsystem lives in its layer directory (cataloged in `README.md`).
**Evidence provenance:** every diagram cites real file paths + symbols (see headers). Graph
stats: 1,049,585 nodes / 3,225,692 edges / 6,573 IMPLEMENTS (`.victor/init.md`).
```
docs/architecture-diagrams/
├── atlas.md                  ← you are here (all-Mermaid, GitHub-native)
├── README.md                 ← catalog: every diagram, type, one-line description
├── RENDERING_STRATEGY.md     ← format rules (Mermaid-first; PNG/SVG export)
├── ASCII_ART.md              ← inline, zero-tooling diagrams
├── 00-system-overview/       ← top-level, deployment, 3 use-case views
├── 01-network-layer/         ← protocol multiplex (5678), pgwire (5433), MCP, request flow
├── 02-query-pipeline/        ← ComputeScheduler routing, DataFusion/Volcano, read path
├── 03-storage-layer/         ← engines, WAL, PAX, Iceberg, 2PC, cache, write path
├── 04-index-layer/           ← HNSW internals, index subsystem
├── 04-services-catalog/      ← DML services, RecordStore, catalog, CDC, connectors
├── 05-cluster-distributed/   ← cluster coordination, partition-lease state machine
├── 07-datafusion-integration/← DataFusion OLAP (ADR-052)
├── 07-graph-subsystem/       ← graph component, ORION internals
├── 08-ai-llm/                ← AI/LLM component, embedding pipeline
├── 09-clients-sdk/           ← Python SDK architecture
├── 09-coupling-analysis/     ← god-file coupling hotspots (README + diagram)
├── 09-security-governance/   ← security component, governance detail
├── 10-cross-cutting/         ← observability, infrastructure bootstrap
├── 10-deployment/            ← system deployment
└── 11-crate-dependency-map/  ← workspace crate layering (CI-enforced)
```
Export Mermaid to PNG/SVG: `bash scripts/diagrams/render_atlas.sh`
---
_`src/network/multi_server.rs`, `src/network/postgres/` (5433), `src/network/rest|grpc|arrow_ipc|multiplex` (5678), `src/embedded/`_
```mermaid
flowchart TD
    subgraph Clients["Client SDKs (spec-driven)"]
        PY[Python SDK]
        RS[Rust SDK]
        GO[Go SDK]
        TS[Node/TS SDK]
        JVM[JVM SDK]
    end
    EMB["Embedded (in-process)<br/>src/embedded/<br/>PyO3 / JNI / NAPI / C-FFI"]
    subgraph Srv["ProximaDB Server (apps/proximadb-server)"]
        MULTI["MultiServer<br/>network/multi_server.rs"]
        P5678{{"5678: REST+gRPC+Arrow Flight (multiplex)"}}
        P5433{{"5433: pgwire (postgres/)"}}
        PMCP{{"MCP (network/mcp/)"}}
    end
    subgraph Core["Core Library (proximadb crate)"]
        DB["ProximaDB facade<br/>src/database.rs"]
        Q["Query (ComputeScheduler)"]
        S["Services (DML, Collection)"]
        ST["Storage (engines, WAL)"]
        CL["Cluster (partition_lease)"]
        CAT["Catalog (syscat, federation)"]
    end
    OBJ[("Object Storage<br/>S3 / Azure Blob / GCS<br/>DrPathBuilder: data/tenant/ns/")]
    PY --> P5678
    RS --> P5678
    GO --> P5678
    TS --> EMB
    JVM --> EMB
    EMB --> DB
    P5678 --> MULTI
    P5433 --> MULTI
    PMCP --> MULTI
    MULTI --> DB
    DB --> Q & S & ST & CL & CAT
    S --> ST
    Q --> ST
    S --> CAT
    CL --> CAT
    ST --> OBJ
```

### 0b. Top-Level Component Decomposition
`src/lib.rs` modules + `crates/{foundation,horizontal,storage,query,control,modalities,platform}`_
```mermaid
flowchart TD
    subgraph Found["foundation (leaf types)"]
        proto[proto]
        records[records]
        dist[distance-types/kernel]
        reltypes[relational-types]
        tenant[tenant]
        idxtypes[index-types/traits]
    end
    subgraph Horiz["horizontal (cross-cutting)"]
        codec[codec]
        compression[compression]
        security[security/tls]
        telemetry[telemetry]
        serial[serialization]
    end
    subgraph Stor["storage crates"]
        objstore[object-store]
        blockfmt[block-format PAX]
        ports[storage-ports]
        iceberg[iceberg-engine]
    end
    subgraph Qry["query crates"]
        relfe[relational-frontend/planner]
        relex[relational-executor/algebra]
        query[query / query-capability]
        graphq[graph-query/arrow]
    end
    CATC[control: catalog]
    subgraph Root["proximadb root (src/) - monolith"]
        net[network/ 5678+5433]
        q2[query/ ComputeScheduler]
        sv[services/ DML Collection]
        st2[storage/ engines persistence]
        cl2[cluster/]
        cat2[catalog/]
        df[datafusion/]
    end
    net --> q2 & sv
    q2 --> st2
    sv --> st2 & cat2
    df --> q2
    cl2 --> cat2
    st2 --> ports & blockfmt & objstore
    q2 --> relex
    cat2 --> records & proto
```
### 0c. Use Cases & Entry Points (Ports)
`rest/v2/{collections,records,query,documents,entities,graphs,schema,timeseries}.rs`, `rest/v1/`, `postgres/` (SQL), `mcp/` (agents)_
```mermaid
flowchart LR
    Dev((App Developer))
    ML((ML/AI Engineer<br/>RAG))
    AN((Data Analyst<br/>BI/psql))
    AG((AI Agent/LLM))
    OP((Operator/Admin))
    subgraph VS["Vector Search (REST/gRPC v2)"]
        UC1([Create Collection])
        UC2([Upsert/Embed Records])
        UC3([Similarity/Hybrid Search])
    end
    subgraph RL["Relational (pgwire 5433)"]
        UC5([DDL: CREATE/ALTER TABLE])
        UC6([ANSI SQL SELECT<br/>OLAP/OLTP])
        UC7([MATERIALIZE to Parquet])
    end
    subgraph OPz["Operate"]
        UC12([Admin/Discovery via MCP])
        UC13([Iceberg REST Catalog])
    end
    Dev --> UC1
    ML --> UC2 & UC3
    AN --> UC6 & UC5
    AG --> UC12
    OP --> UC14_([Force Flush/Compaction]) & UC13
    UC2 -.->|include| UC1
    UC3 -.->|extend| UC2
    UC7 -.-> UC5
```
---
## 1. Network / API Layer
`network/multi_server.rs`, `network/multiplex/`, `network/middleware/tenant.rs`, `network/postgres/relational_pipeline.rs`_
```mermaid
flowchart LR
    APP([App/SDK]) -->|HTTP/gRPC/Flight| MUX
    BI([BI/psql]) -->|Postgres wire| PGWIRE
    AG2([Agent/LLM]) -->|MCP| MCP
    subgraph S5678["Unified Server :5678 (MultiServer)"]
        MUX["Multiplexer<br/>network/multiplex/"]
        MID["Middleware<br/>middleware/tenant.rs<br/>(injects TenantContext)"]
        REST["REST v1+v2"]
        GRPC["gRPC proto v1+v2"]
        ARROW["Arrow Flight (zero-copy bulk)"]
    end
    subgraph S5433["pgwire Server :5433"]
        PGWIRE["postgres/<br/>protocol, session, translator"]
        PIPE["relational_pipeline.rs (SQL lowering)"]
    end
    MCP["network/mcp/<br/>(catalog introspection)"]
    MUX --> MID --> REST & GRPC & ARROW
    PGWIRE --> PIPE
    REST --> SVC[Services]
    GRPC --> SVC
    ARROW --> SVC
    PIPE --> ROUTER["ComputeScheduler"]
    REST --> ROUTER
```
**Key invariant:** Tenant identity enters at `middleware/tenant.rs` and flows as `TenantContext` (`proximadb-tenant`) to **every I/O boundary** (isolation + billing).
---
## 2. Query Routing & ComputeBackend Seam
`query/compute_scheduler.rs:329` `route_select`, `QueryShape(130)`, `SelectRouteDecision(189)`; `datafusion/proxima_table_provider.rs`, `datafusion/proxima_scan_exec.rs`; ADR-052_
```mermaid
flowchart TD
    SQL["SQL arrives (pgwire:5433 or REST)"] --> LOWER["relational_pipeline.rs<br/>SQL -> logical plan"]
    LOWER --> SCHED["ComputeScheduler::route_select<br/>query/compute_scheduler.rs:329"]
    SCHED --> SHAPE{"Inspect QueryShape"}
    SHAPE -->|"parquet-backed (MATERIALIZE)"| DF["DataFusion OLAP backend<br/>datafusion/ proxima_table_provider + proxima_scan_exec<br/>(PAX-native scan TD-OLAP-1)"]
    SHAPE -->|"vector/point-lookup"| NATIVE["Native Volcano<br/>query/execution/engine.rs (OLTP floor)"]
    SHAPE -->|"vector/ANN"| ANN["Vector engines SST/HELIX/NOVA<br/>(VIPER deprecated: existing collections only)"]
    DF & NATIVE & ANN --> RRP["RoutedReadPlan (SelectRouteDecision)"]
    RRP --> EXEC["ExecNode tree<br/>relational-executor/lib.rs:427"]
    EXEC --> DIST["Distributed shuffle ShuffleKey<br/>crates/query/.../distributed/shuffle.rs:145"]
    EXEC --> RESULT([Result set])
    classDef seam fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    class DF,NATIVE,ANN seam
```
**ComputeBackend seam (ADR-052):** The engine owns selection. DataFusion's vectorized kernels do OLAP; the native investment is the PAX-native **SCAN** (TD-OLAP-1). Do **not** build native HashAgg/HashJoin/Sort. The seam keeps the backend swappable.

---
## 3. Storage Engine Strategy
`storage/traits/mod.rs:532` `UnifiedStorageFormat` (strategy, do_flush, do_compact, search_vectors_unified, create_scan, read_all_records, health_check, rls_record_filter); `storage/engines/factory.rs:312` `create_from_strategy`, `recommend_engine(687)`, `create_for_workload(655)`_
```mermaid
flowchart TD
    FACTORY["EngineFactory<br/>create_from_strategy / recommend_engine / create_for_workload"]
    TRAIT(("UnifiedStorageFormat (trait, port)<br/>~30 methods: strategy/do_flush/<br/>search_vectors_unified/create_scan/<br/>read_all_records/health_check/rls_record_filter"))
    FACTORY -.->|creates Arc<dyn>| TRAIT
    SST["SSTEngine (columnar, write-opt)"] --> TRAIT
    VIPER["ViperEngine (deprecated, ADR-093)"] --> TRAIT
    NOVA["NovaEngine (next-gen mixed)"] --> TRAIT
    SWIFT["SwiftEngine (experimental)"] --> TRAIT
    HELIX[HelixEngine] --> TRAIT
    RAPTOR["RaptorEngine (experimental)"] --> TRAIT
    CEDAR[CedarEngine] --> TRAIT
    TST[TstEngine] --> TRAIT
    FACTORY --> CAPREG["CapabilityRegistry factory.rs:126"]
    classDef portt fill:#fff8e1,stroke:#f57f17
    class TRAIT portt
```
**Primary seam for adding a storage backend:** implement `UnifiedStorageFormat` - no service-layer changes (Strategy Pattern). Default trait methods absorb shared logic across engines.
### 3b. Storage Internal Layers
`src/storage/` subdirs + `crates/storage/*`_
```mermaid
flowchart LR
    subgraph Ports["Ports & Contracts"]
        TRAITS[traits/ UnifiedStorageFormat]
        PATHRES[trait_components/ path_resolver.rs DrPathBuilder]
        CAP[engine_capabilities.rs]
    end
    subgraph ENG["Engines"]
        ENGINES[engines/ factory + sst/viper/nova/swift/helix/raptor]
    end
    subgraph PERSIST["Persistence (write path)"]
        WAL[write_ahead_log/ WriteAheadLogManager]
        DM[disk_manager]
        FS[filesystem]
        SER[serialization avro/bincode]
        FLUSH[flush_materializer.rs]
        MEM[memtable/]
    end
    subgraph FMT["Formats & Scan"]
        FMT2[formats/ PAX]
        SCAN[scan_strategy.rs]
    end
    subgraph MM2["Multi-model Stores"]
        ENT[entity_store/]
        DOC[document/]
        KG[knowledge_graph/]
    end
    ENGINES --> TRAITS
    PATHRES --> OBJC[object-store crate]
    WAL --> MEM --> FLUSH --> ENGINES
    FMT2 --> BLK[block-format crate]
    ENGINES --> ICE[iceberg-engine crate]
    SCAN --> TRAITS
```
---
## 4. Service Layer
`services/dml/mod.rs:742` `DmlService`, `execute(946)`, `execute_scoped(954)`, `scan_table_records(1111)`; `services/collection/{manager,engine_selector,recall_target,security}.rs`_
```mermaid
flowchart TD
    subgraph DMLz["Data Manipulation"]
        DML["DmlService dml/mod.rs:742 execute(946)"]
        TW["table_write_executor"]
        WI["write_intent.rs"]
    end
    subgraph COLL["Collection Mgmt"]
        CM[CollectionManager]
        SEL["engine_selector"]
    end
    subgraph STZ["Stores"]
        REC["record_store.rs TableRecordStore"]
        EO["entity_orchestrator.rs"]
    end
    subgraph BR["Persistence Bridge"]
        CW["canonical_wal.rs CanonicalWal"]
    end
    subgraph FUS["Fusion / Search / Tx"]
        FUS2[fusion_service.rs]
        SRCH["search/"]
        DDL["ddl/"]
    end
    SYS["system_catalog.rs"]
    DML --> REC & CW
    CM --> SEL --> ENGF["EngineFactory (storage/engines/factory.rs)"]
    REC --> CW --> WLM["WriteAheadLogManager (storage/persistence/)"]
    FUS2 --> SRCH
    EO --> REC
    DDL --> SYS
```
**Invariant 16b:** after `DmlService::execute` commits, it **MUST invalidate** the query/plan cache for (tenant, collection) and route via `TenantContext`.
---
## 5. Write Path (Sequence)
`storage/persistence/write_ahead_log/mod.rs` (`WriteAheadLogManager`, `WALOperation`), `storage/flush_materializer.rs`, `storage/trait_components/path_resolver.rs` (DrPathBuilder), `services/canonical_wal.rs`, `services/dml/mod.rs`_
```mermaid
sequenceDiagram
    participant C as Client
    participant H as REST/gRPC handler
    participant DML as DmlService
    participant WAL as CanonicalWal
    participant WLM as WriteAheadLogManager
    participant MT as memtable
    participant FM as FlushMaterializer
    participant ENG as Engine (UnifiedStorageFormat)
    participant DRP as DrPathBuilder
    participant OBJ as Object Store
    participant CIC as CacheInvalidation
    C->>H: UPSERT records (TenantContext)
    H->>DML: execute(DmlStatement)
    DML->>WAL: write op (TenantContext)
    WAL->>WLM: append WALOperation
    WLM-->>WAL: ok (durable)
    WAL-->>DML: committed
    DML->>MT: insert into memtable
    DML->>CIC: invalidate query/plan cache (tenant, collection)
    DML-->>H: DmlResult (rows_affected, ids)
    H-->>C: 200 OK
    Note over CIC: INVARIANT 16b: every write path invalidates read-serving caches post-commit
    Note over WLM,FM: ... threshold reached ...
    WLM->>FM: flush trigger
    FM->>MT: drain memtable
    FM->>ENG: do_flush(FlushParameters) [trait]
    ENG->>DRP: build key data/{tenant}/{ns}/...
    DRP->>OBJ: PUT (flat key + ObjectAccessTier)
    OBJ-->>ENG: ok
    ENG-->>FM: FlushResult
```

---
## 6. Cluster Coordination
`cluster/partition_lease.rs` (255 symbols - distributed ownership), `cluster/consensus.rs` (Raft, feature: cluster), `cluster/{replication,routing,shard,node_registry,primary_pod_registry}.rs`, `cluster/rpc/grpc_server`_
```mermaid
flowchart TD
    LEASE["PartitionLease partition_lease.rs (255 symbols)"]
    CONS["Consensus (Raft) consensus.rs [feature: cluster]"]
    REPL["Replication replication.rs"]
    ROUTE["Routing routing.rs"]
    SHARD["Shard shard.rs"]
    NODE["NodeRegistry node_registry.rs"]
    PRIMARY["PrimaryPodRegistry primary_pod_registry.rs"]
    META["MetadataService metadata_service.rs"]
    DIST["DistributedOps distributed_ops.rs"]
    AFFIN["CacheAffinity cache_affinity.rs"]
    RPC["RPC cluster::rpc::grpc_server (Consensus/Health/Replication svcs)"]
    LEASE --> CONS & PRIMARY & SHARD
    CONS --> REPL
    ROUTE --> LEASE & NODE
    META --> CONS
    DIST --> ROUTE
    AFFIN --> ROUTE
    CONS -.-> RPC
```
`PartitionLease` is the hub of distributed ownership. Lease fencing prevents split-brain writes; reads route to the partition's owning primary pod.
---
## 7. Catalog: Object Model, Federation & DR
`crates/control/proximadb-catalog/src/` (native, oltp, iceberg, hive, unity, polaris, glue, delta, schema, id_allocator, collection_dr_policy, dr_policy_store, dr_reconciler, dr_restore, corpus_version); `src/catalog/` (syscat_cache, federation, iceberg_rest_service, partition_pruning, segment_registry, index_location_resolver, tenant_tier, budget_guard, recall_probe)_
```mermaid
flowchart LR
    subgraph Core["Catalog Core"]
        LIB["lib.rs (OID model)"]
        SCH["schema.rs"]
        IDA["id_allocator.rs"]
    end
    subgraph BACK["Storage Backends (polymorphic)"]
        NATIVE[native.rs]
        OLTP[oltp.rs]
        ICE[iceberg.rs]
        DELTA[delta.rs]
    end
    subgraph FED["Federation Adapters"]
        HIVE[hive.rs]
        UNITY[unity.rs]
        POLARIS[polaris.rs]
        GLUE[glue.rs]
    end
    subgraph DR["DR / Replication"]
        DRSTORE[dr_policy_store.rs]
        DRRECON[dr_reconciler.rs]
        DRRESTORE[dr_restore.rs]
        CORPUS[corpus_version.rs]
    end
    subgraph SVRCAT["src/catalog/ (server glue)"]
        SYSCACHE[syscat_cache / syscat_warm]
        FED2[federation/]
        ICEREST[iceberg_rest_service.rs]
        PARTPRUNE[partition_pruning.rs]
        SEGREG[segment_registry.rs]
        IDXLOC[index_location_resolver.rs]
        TIER[tenant_tier / tier_transition]
        BUDGET[budget_guard.rs]
        RECALL[recall_probe.rs]
    end
    NATIVE & OLTP & ICE & DELTA --> Core
    HIVE & UNITY & POLARIS & GLUE --> Core
    Core --> SYSCACHE & FED2
    ICEREST --> ICE
    PARTPRUNE --> Core
    SEGREG --> Core
    IDXLOC --> SEGREG
    TIER --> Core
    BUDGET --> TIER
    RECALL --> Core
    DRRECON --> DRSTORE
    DRRESTORE --> CORPUS
```
Catalog uses an OID/object model with polymorphic backends (native/iceberg/delta) selected per table; federation adapters expose external catalogs (Hive/Unity/Polaris/Glue).
---
## 8. Embedded Mode & SDK Ecosystem
`src/embedded/` (python.rs PyO3 203 symbols, java.rs, nodejs.rs, c_ffi.rs, python_dataframe.rs); Python SDK: `embedding_providers/core/base.py` BaseEmbeddingProvider (9 impls), `chunking_strategies` ChunkingStrategyInterface (8 impls); SDK REST surfaces spec-driven from `docs/openapi/proximadb-openapi.yaml`_
```mermaid
flowchart TD
    CORE2["ProximaDB core src/lib.rs + database.rs"]
    subgraph EMB["Embedded (in-process, no socket)"]
        PYO3["PyO3 embedded/python.rs (203 symbols)"]
        JNI["JNI/Java embedded/java.rs"]
        NAPI["Node.js NAPI embedded/nodejs.rs"]
        CFFI["C-FFI embedded/c_ffi.rs"]
    end
    PYO3 & JNI & NAPI & CFFI --> CORE2
    subgraph PYSDK["Python SDK (clients/python)"]
        PYPROTO["protocols/ _generated/rest (REST codegen sync+async)"]
        EMB2["embedding_providers/ BaseEmbeddingProvider (9 impls)"]
        CHUNK["chunking_strategies/ ChunkingStrategyInterface (8 impls)"]
    end
    subgraph SDKS["Other SDKs (spec-driven REST)"]
        RSC["Rust SDK (genrest.rs auto-gen)"]
        GOC[Go SDK]
        JVMC[JVM SDK]
        TSC[Node/TS SDK]
    end
    OPENAPI["OpenAPI spec docs/openapi/proximadb-openapi.yaml (utoipa-emitted)"]
    PYPROTO --> OPENAPI
    RSC & GOC & JVMC & TSC --> OPENAPI
```
**SDK REST transport is spec-driven (CLAUDE.md §15):** every SDK's REST surface is generated from `docs/openapi/proximadb-openapi.yaml` (utoipa-emitted); the `<lang>-sdk-codegen-drift` gate rejects hand-rolled REST. Full detail: [`09-clients-sdk/01-python-sdk-architecture.mermaid`](./09-clients-sdk/01-python-sdk-architecture.mermaid).
---
## 9. Coupling Analysis (Hub Types, Hotspots, Duplication)
Full analysis: `09-coupling-analysis/README.md` (3 Mermaid diagrams + recommendation table)._
**Hub types (highest blast radius - treat as stable seams):** `now()` (1,466 refs), `Duration` (98 conn), `DistanceMetric` (61 conn), `TenantContext` (all I/O boundaries).
```mermaid
flowchart LR
    subgraph GEN["AUTO-GENERATED (do NOT hand-edit)"]
        GR["genrest.rs 19,244 lines / 1,749 symbols"]
        PV1["proto v1 17,299 lines"]
        PV2["proto v2 8,469 lines"]
    end
    subgraph GOD["GOD MODULES (decomposition candidates)"]
        DML2["dml/mod.rs 11,256 lines / 226 symbols"]
        PLE["partition_lease.rs 4,578 lines / 255 symbols"]
        PY3["embedded/python.rs 4,841 lines / 203 symbols"]
        TR["traits/mod.rs 1,815 lines / ~30-fn trait"]
    end
    classDef gen fill:#eceff1,stroke:#607d8b
    classDef god fill:#ffebee,stroke:#c62828
    class GR,PV1,PV2 gen
    class DML2,PLE,PY3,TR god
```
**Duplication targets:** proto v1/v2 overlap (25,768 lines parallel defs); trait-stub boilerplate (~30 fns x 8 engines - absorb in trait default methods); catalog backend glue (native/iceberg/delta - promote shared logic to `glue.rs`). **Graph scale:** 1,049,585 nodes / 3,225,692 edges / 6,573 IMPLEMENTS / 163 INHERITS.

## 10. AI / LLM Subsystem & Embedding Pipeline
`src/ai/` (`llm.rs`, `natural_language_api.rs`, `nlp.rs`, `insights.rs`, `llm_integration/`), `src/automl/`, `src/prompts/`. Full detail: [`08-ai-llm/01-ai-component.mermaid`](./08-ai-llm/01-ai-component.mermaid), [`02-ai-llm-subsystem.mermaid`](./08-ai-llm/02-ai-llm-subsystem.mermaid), [`03-embedding-pipeline.mermaid`](./08-ai-llm/03-embedding-pipeline.mermaid)._
```mermaid
flowchart TD
    NL["Natural Language API<br/>ai/natural_language_api.rs<br/>Text-to-SQL / Text-to-AQL"]
    LLM["LLM Manager ai/llm.rs<br/>multi-provider: OpenAI/Anthropic/Llama/local"]
    NLP["NLP ai/nlp.rs<br/>tokenize + intent + entities"]
    INS["AI Insights ai/insights.rs<br/>anomaly / trend analysis"]
    AUTOML["AutoML src/automl/<br/>index tuning, quant strategy"]
    PROMPT["Prompt Store src/prompts/"]
    NL --> NLP & LLM
    LLM --> LLM_INT["llm_integration/<br/>provider adapters + fallback chain"]
    INS --> NL
    AUTOML --> OPT["optimization.rs / prediction.rs"]
    PROMPT --> LLM
```
---
## 11. Security, Auth & Governance
`src/security/`, `src/network/middleware/{auth,tenant,rate_limit}.rs`, `crates/foundation/proximadb-tenant`. Full detail: [`09-security-governance/01-security-component.mermaid`](./09-security-governance/01-security-component.mermaid), [`02-security-governance-detail.mermaid`](./09-security-governance/02-security-governance-detail.mermaid)._
```mermaid
flowchart LR
    TLS["TLS termination<br/>middleware/tls.rs"]
    AUTHN["Auth (JWT / API key)<br/>network/auth + middleware/auth.rs"]
    RBAC["RBAC<br/>security/rbac_service.rs"]
    TENANT["TenantContext<br/>proximadb-tenant -> every I/O boundary"]
    RATE["Rate limit + backpressure<br/>middleware/"]
    AUDIT["Audit / compliance<br/>telemetry"]
    TLS --> AUTHN
    AUTHN --> RBAC --> TENANT
    AUTHN --> RATE
    TENANT --> AUDIT
```
**Key invariant:** `TenantContext` (isolation + billing) flows **fail-closed** to every I/O boundary; all object-storage writes go under `DrPathBuilder` (`data/{tenant}/{ns}/…`).
---
## 12. Observability & Telemetry
`src/telemetry/`, Prometheus billing metrics, distributed tracing. Full detail: [`10-cross-cutting/01-observability-subsystem.mermaid`](./10-cross-cutting/01-observability-subsystem.mermaid)._
```mermaid
flowchart LR
    IO["Every I/O boundary<br/>(TenantContext-stamped)"]
    MET["Metrics: Prometheus<br/>KSU / KRU / KOU / KIU / KEU"]
    TRACE["Distributed tracing"]
    AUDIT["Audit trail"]
    LOG["Structured logs"]
    HEALTH["Health / readiness"]
    IO --> MET & TRACE & AUDIT & LOG
    MET --> HEALTH
```
---
## 13. Index Layer (HNSW / Compact / IVF)
`src/storage/engines/.../index/` (AXIS). Full detail: [`04-index-layer/01-index-subsystem.mermaid`](./04-index-layer/01-index-subsystem.mermaid), [`02-hnsw-internals.mermaid`](./04-index-layer/02-hnsw-internals.mermaid)._
```mermaid
flowchart TD
    SUB["Index Subsystem<br/>HNSW + Compact + IVF"]
    HNSW["HNSW<br/>graph ANN (M, ef_construction)"]
    IVF["IVF<br/>clustered inverted-file"]
    COMPACT["Compact<br/>segmented / quantized"]
    QUANT["Quantization<br/>SQ8 / RaBitQ + f32 rerank"]
    SUB --> HNSW & IVF & COMPACT
    HNSW --> QUANT
    IVF --> QUANT
```
---
- **This file (`atlas.md`)** renders fully on GitHub (Mermaid-native).
- **Per-topic drill-down:** each layer directory holds the authoritative full-detail diagrams — see [`README.md`](./README.md) for the complete catalog (47 diagrams, 16 directories).
- **ASCII art** ([`ASCII_ART.md`](./ASCII_ART.md)) for the simplest one-box/one-liner ideas.
- Rules: [`RENDERING_STRATEGY.md`](./RENDERING_STRATEGY.md).
