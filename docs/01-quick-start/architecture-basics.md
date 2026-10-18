# Architecture Basics

**How ProximaDB works under the hood**

```mermaid
%%{init: {"theme": "neutral"}}%%
flowchart TB
  subgraph Clients["Client Layer"]
    REST[REST API<br/>:5678]
    gRPC[gRPC API<br/>:5678]
    PGSQL[PostgreSQL Wire<br/>:5433]
  end

  subgraph Services["Service Layer"]
    VS[Vector Service]
    DS[Document Service]
    GS[Graph Service]
    OS[Observability Service]
    UQ[Unified Query Engine]
  end

  subgraph DataPlane["Data Plane"]
    WAL[Unified WAL]
    MEM[Memtables]
    CACHE[Block Cache]

    subgraph Engines["Vector engines<br/>(SST Supported; HELIX, NOVA Beta)"]
      SST[SST]
      HELIX[HELIX]
      NOVA[NOVA]
    end

    subgraph GraphEngines["Graph Engine (ORION)"]
      ORION[ORION]
    end
  end

  Clients --> Services
  Services --> WAL
  WAL --> MEM
  MEM --> CACHE
  CACHE --> Engines
  CACHE --> GraphEngines

  style UQ fill:#e74c3c,stroke:#c0392b,color:#fff
  style WAL fill:#3498db,stroke:#2980b9,color:#fff
  style CACHE fill:#9b59b6,stroke:#8e44ad,color:#fff
```

---

## Core Concepts

### 1. Unified API Layer

Single port (`5678`) for multiple protocols:

| Protocol | Use Case | Example |
|----------|----------|---------|
| **REST** | Web apps, curl | `POST /api/v2/collections` |
| **gRPC** | High-performance services | `proximadb.v2.ProximaRecordService` |
| **Arrow Flight** _(experimental)_ | Data analytics, BI tools | `do_put()` streaming |

Plus PostgreSQL wire protocol on port `5433` for SQL clients — bound to `127.0.0.1` unless
the listener authenticates (pgwire SCRAM), so it is host-local and not reachable through a
port published from a container (`TD-PGWIRE-CONTAINER-1`). (Arrow Flight as a
customer-facing transport is **experimental**; REST/gRPC are the supported surfaces.)

**Why it matters:** One connection, any protocol. No need to run multiple services.

---

### 2. Unified WAL (Write-Ahead Log)

```mermaid
%%{init: {"theme": "neutral"}}%%
flowchart LR
  A[Write Request] --> B[WAL Append]
  B --> C[Memtable Write]
  C --> D[Ack to Client]

  D --> E[Background Flush]
  E --> F[SSTable]

  style B fill:#3498db,color:#fff
  style F fill:#27ae60,color:#fff
```

All writes (vectors, documents, graphs, logs) go through a single WAL:

- **Durability**: Writes are acknowledged only after WAL flush
- **Consistency**: Single ordering for all data types
- **Recovery**: Replay WAL on restart

**Why it matters:** No partial failures. All your data is consistent.

---

### 3. Storage Engines

```mermaid
%%{init: {"theme": "neutral"}}%%
flowchart TB
  subgraph Default["Supported (default)"]
    SST[SST]
  end

  subgraph Beta["Beta options"]
    HELIX[HELIX]
    NOVA[NOVA]
  end

  subgraph SQLA["Analytical SQL"]
    DF["Relational table + MATERIALIZE<br/>DataFusion over Parquet"]
  end

  style SST fill:#27ae60,color:#fff
  style HELIX fill:#e67e22,color:#fff
  style NOVA fill:#e67e22,color:#fff
  style DF fill:#9b59b6,color:#fff
```

SST is the Supported storage engine. Today a vector collection created without `engine` (and without an index or quantization) is auto-routed to HELIX, which is Beta; pass `engine="sst"` for the Supported tier. SST becomes the automatic default in v0.5 (TD-ENGSEL-1). HELIX and NOVA are Beta
options. VIPER is deprecated (new VIPER collections are rejected). SWIFT and RAPTOR are
**experimental**, off by default behind the `experimental-engines` cargo feature, and are deleted
in roadmap release v0.6. Analytical SQL is not an engine choice: it runs on DataFusion over
Parquet once a relational table is materialized.

| Engine | Tier | Best for |
|--------|------|----------|
| **SST** | Supported | Real-time ingestion and filtered vector search |
| **HELIX** | Beta | Locality-ordered high-dimensional embeddings (measure recall first) |
| **NOVA** | Beta | Mixed read/write workloads |

**Why it matters:** start on SST; switch only after measuring a need on your own data
([storage engine guide](../05-concepts/storage-engines.adoc)).

---

### 4. Graph Engines

```mermaid
%%{init: {"theme": "neutral"}}%%
flowchart LR
  subgraph GraphEngines["Graph Engine"]
    ORION[ORION<br/>In-Memory CSR]
  end

  subgraph Cap["Capabilities"]
    C1[1M+ edges/sec]
    C2[Vector + Graph]
  end

  ORION --> C1
  ORION --> C2

  style ORION fill:#e74c3c,color:#fff
```

ORION is the only graph engine (PULSAR and QUASAR names are **retired** — former
distributed/tiered behavior now lives in the relational distributed substrate
and xCatalog/storage projection policy).

| Engine | Scale | Features |
|--------|-------|----------|
| **ORION** | Single node | Fastest, in-memory CSR |

**Why it matters:** Graph relationships without a separate graph database.

---

### 5. Multi-Model Query Engine

```mermaid
%%{init: {"theme": "neutral"}}%%
flowchart TB
  A[SQL Query] --> B[Parser]
  B --> C[Decomposer]

  C --> D1[Vector Search]
  C --> D2[Document Query]
  C --> D3[Graph Traversal]
  C --> D4[Log Search]

  D1 --> E[Fusion]
  D2 --> E
  D3 --> E
  D4 --> E

  E --> F[Result]

  style C fill:#3498db,color:#fff
  style E fill:#e74c3c,color:#fff
```

SQL extensions for multi-model queries:

These table functions run through the federated REST endpoint, not over pgwire (TD-FEDPATH-1);
see the README's SQL Extensions for what works today.

```sql
-- Vector search
SELECT * FROM VECTOR_SEARCH('collection', vector, 10)

-- Graph traversal
SELECT * FROM GRAPH_QUERY('graph', 'MATCH (a)-[:KNOWS]->(b)')

-- Document query
SELECT * FROM DOCUMENT_QUERY('docs', 'category = "tech"')

-- Cross-model join
SELECT v.id, d.content
FROM VECTOR_SEARCH('vectors', query, 10) v
JOIN DOCUMENT_QUERY('docs', 'id = "' || v.id || '"') d
```

**Why it matters:** Query across data models without ETL or joins between systems.

---

## Data Flow

### Write Path

```mermaid
%%{init: {"theme": "neutral"}}%%
sequenceDiagram
    participant C as Client
    participant A as API Layer
    participant S as Service
    participant W as WAL
    participant M as Memtable
    participant E as Engine

    C->>A: POST /vectors
    A->>S: Validate
    S->>W: Append
    W->>M: Write
    M->>E: Index (async)
    E-->>S: Done
    S-->>A: Ack
    A-->>C: 200 OK
```

1. Client writes to API layer
2. Service validates request
3. WAL appends write (durable)
4. Memtable updates (in-memory)
5. Engine indexes asynchronously

### Read Path

```mermaid
%%{init: {"theme": "neutral"}}%%
sequenceDiagram
    participant C as Client
    participant A as API Layer
    participant S as Service
    participant Q as Query Planner
    participant E as Engine
    participant C as Cache

    C->>A: POST /search
    A->>S: Parse query
    S->>Q: Plan execution
    Q->>C: Check cache
    alt Cache Hit
        C-->>S: Cached result
    else Cache Miss
        Q->>E: Engine search
        E-->>Q: Results
        Q->>C: Populate cache
    end
    S-->>A: Results
    A-->>C: 200 OK
```

1. Client sends search request
2. Query planner optimizes
3. Check cache (block cache)
4. On miss: query storage engine
5. Populate cache for next request

---

## Storage Hierarchy

```mermaid
%%{init: {"theme": "neutral"}}%%
flowchart TB
  subgraph Memory["Memory (Hot)"]
    WAL[WAL Buffer]
    MEM[Memtables]
    CACHE[Block Cache]
  end

  subgraph Disk["SSD (Warm)"]
    SST[SSTables]
    IDX[Indexes]
  end

  subgraph Cold["Cold Storage"]
    S3[S3/GCS]
    AZ[Azure]
  end

  Memory --> Disk
  Disk --> Cold

  style MEM fill:#e74c3c,color:#fff
  style CACHE fill:#e74c3c,color:#fff
  style SST fill:#f39c12
  style S3 fill:#95a5a6
```

| Tier | Latency | Purpose |
|------|----------|---------|
| **Memory** | <1ms | Active data, hot cache |
| **SSD** | 1-10ms | Persistent storage |
| **Cloud** | 100ms+ | Archive, backup |

---

## Key Design Decisions

### Unified Port vs Multi-Port

**Unified (default, port 5678):**
- ✅ Simpler deployment
- ✅ Single firewall rule
- ✅ HTTP/2 multiplexing

**Multi-Port (legacy):**
- ❌ Multiple ports to manage
- ❌ Separate connections

### WAL Per Model vs Unified WAL

**Unified WAL (current):**
- ✅ Single ordering across models
- ✅ Cross-model transactions
- ✅ Simpler recovery

**Per-Model WAL (alternative):**
- ❌ Complex coordination
- ❌ No cross-model ACID

### Engine Selection

**Manual (current):**
- ✅ Full control
- �:: Must know workload

**Auto (future):**
- ✅ Engine auto-selection
- �:: Less predictable

---

## Next Steps

- [Storage Engines Guide](../05-concepts/storage-engines.adoc) - Deep dive on each engine
- [Graph Engines Guide](../05-concepts/graph-engines.adoc) - Graph internals
- [Query Planner](../05-concepts/query-planner.md) - How queries are optimized
- [Configuration](../03-api-reference/configuration.adoc) - Tuning parameters

---

*Need help?* [GitHub Issues](https://github.com/anvai-labs/proximaDB/issues)
