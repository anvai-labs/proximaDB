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

    subgraph Engines["Storage Engines<br/>(SST, VIPER, HELIX, NOVA)"]
      SST[SST]
      HELIX[HELIX]
      VIPER[VIPER]
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
  subgraph Write["Write-Optimized"]
    SST[SST<br/>~5ms]
  end

  subgraph Read["Read-Optimized"]
    HELIX[HELIX<br/>~13ms]
  end

  subgraph Analytics["Analytics"]
    VIPER[VIPER<br/>~89ms]
    NOVA[NOVA<br/>~101ms]
  end

  style SST fill:#27ae60,color:#fff
  style HELIX fill:#e67e22,color:#fff
  style VIPER fill:#9b59b6,color:#fff
  style NOVA fill:#9b59b6,color:#fff
```

SST is the Supported storage engine. VIPER, HELIX, and NOVA are Beta options for
different workloads. SWIFT and RAPTOR are **experimental**, off by default behind
the `experimental-engines` cargo feature and not part of the supported surface:

| Engine | Best For | Performance |
|--------|----------|-------------|
| **SST** | Real-time, write-heavy | Fastest writes |
| **HELIX** | Locality-optimized reads | Spatial queries |
| **VIPER** | Columnar analytics | Aggregations |
| **NOVA** | Mixed workloads | Balanced |

**Why it matters:** Pick the right engine for your workload. Don't use analytics engine for real-time writes.

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
