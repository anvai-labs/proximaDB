# ProximaDB as a Lakehouse-Native, Multi-Engine Postgres-Wire Server

**Status:** Design proposal / architecture review

> **Superseded in part (2026-10-16).** §2's "ProximaDB today" table is a 2026-06 snapshot. The
> router no longer always picks Native: Parquet-backed tables route to DataFusion (ADR-052/064),
> and ADR-094 has since unified the storage substrate. The OLTP and transaction half of this study
> continues in [ADR-095](adr/ADR-095-postgres-compatible-oltp-on-object-storage.adoc), which
> refreshes §3's market survey (Aurora, DSQL, HorizonDB, AlloyDB, PlanetScale, Supabase 2026,
> Snowflake Postgres, SlateDB). §3–§10 remain the reference for branching and HTAP dispatch.
**Date:** 2026-06-17
**Author:** Architecture review (Claude)
**Scope:** Refine ProximaDB's architecture toward a single server that (a) speaks the Postgres wire protocol, (b) dispatches each query shape to a purpose-built engine, (c) uses decoupled lakehouse object storage (Iceberg/Delta/Parquet) as the analytical base, (d) keeps a hot LSM/WAL tier for fast OLTP read-after-write, and (e) gives agents cheap, instant, object-store-metadata-driven branches.

> This doc is grounded in three inputs: a read of the cloned PostgreSQL source (`~/code/postgres`), a code map of ProximaDB at HEAD (`py-sdk-cov-followups`), and a comparative study of Neon, Databricks Lakebase, Supabase, DuckLake/pg_lake, OrioleDB, and HTAP engines (TiDB, SingleStore, Umbra/CedarDB). Citations inline.

---

## 0. TL;DR — the seven decisions

1. **Stay a custom Rust engine that speaks pgwire — do not become a Postgres extension.** All three of ProximaDB's differentiators (decoupled lakehouse storage, vector search, agent branching) are structurally blocked by Postgres's Table-AM TID constraint. OrioleDB had to *patch Postgres core* just to get row-oriented decoupled storage. ProximaDB already made the right call; this validates it.
2. **You cannot serve OLTP read-after-write *directly* from columnar lakehouse files.** Every system that does both runs a **row/append hot tier + columnar immutable base + background compactor**. Make this the explicit spine of the storage layer (ProximaDB already half-has it: memtable + PAX/Parquet flush).
3. **The WAL is the interface, and the LSN is the branch point.** Adopt Neon's model: WAL is durability *and* the substrate for branching/time-travel. Tier the WAL itself (hot local → archived object store).
4. **Branching = one small object-store metadata object** that pins `(catalog snapshot, per-collection WAL fork LSN, per-table manifest version)`. ProximaDB already has the two halves — `SnapshotPin` (LSN range + manifest checkpoint) and `ManifestCommitter` (versioned `v{N}.manifest` + CAS). Unify them into a branch ref. Branch cost is O(metadata), independent of data size.
5. **Multi-engine dispatch should be cost-based with a selectivity crossover**, not a hardcoded route. `ComputeScheduler` exists (P0 always→Native); the P1 flip infra (`QueryShape.parquet_backed`, `routed_read_plan`) is already wired. Allow **hybrid plans** (row fragment → OLTP engine, scan fragment → DataFusion) and keep **manual hints** as an escape hatch — even TiDB admits its cost model is "still rough."
6. **The two storage seams answer different questions — keep them separate, but split out a third.** The `FileSystem` trait is *mechanism* (which backend, how to do I/O). The tier-policy engine is *policy* (where bytes should live by SLA/cost). These map cleanly onto Postgres's own `smgr` (mechanism) vs buffer-eviction (policy) split — keep them apart. But the **cache/paging** concern (currently fused into `UnifiedCachingFilesystem`) is a distinct third layer, and its eviction policy is *the same idea* as tier demotion. **Consolidate the policy brain, not the mechanism.**
7. **Pay down the `pg_catalog` tax deliberately.** The #1 way "Postgres-compatible" engines break real ORMs/tools is incomplete catalog introspection. Treat `pg_catalog` completeness as a first-class roadmap item (reference: `datafusion-postgres`'s catalog-over-DataFusion layer).

---

## 1. PostgreSQL architecture — the seams that actually matter here

I cloned `postgres` and read the storage/access layers. For *this* design, four seams are load-bearing. PostgreSQL's enduring lesson is **clean separation of mechanism from policy from addressing**.

### 1.1 The storage manager switch (`smgr`) — *addressing/mechanism*
`src/backend/storage/smgr/README`: a dispatch layer (`smgr.c`) over storage managers; today only `md.c` ("magnetic disk", really just the kernel filesystem) survives, but **the switch is deliberately retained** "in case anyone ever wants to reintroduce other kinds of storage managers." `md.c` relies on `fd.c`. Relations are addressed by `(relfilelocator, fork number, block number)`. This is the seam an object-store backend would slot into.

### 1.2 The buffer manager (`bufmgr`) — *caching + eviction policy*
`src/backend/storage/buffer/README`: shared buffer pool with **pin counts** (refcounts — you may not touch a page without a pin), **content locks**, and a **clock-sweep** victim selection (`nextVictimBuffer` moves circularly, decrementing usage counts). Crucially, **`BufferAccessStrategy` ring buffers**: large seq scans / VACUUM / COPY use a small ring of buffers and recycle them with the normal clock-sweep, *so a big scan does not evict the whole working set*. This is the OS page-cache-pollution problem, solved.

### 1.3 The virtual file descriptor pool (`fd.c`) — *scarce-resource pooling*
`src/backend/storage/file/fd.c`: VFDs are "managed as an LRU pool, with actual OS file descriptors being opened and closed as needed," because a process easily exceeds the ~1024 fd limit. Temporary files are subtransaction-scoped and auto-deleted; external fds must be *registered* (`AcquireExternalFD`) so the pool accounts for them. This is a textbook resource-virtualization layer.

### 1.4 The table access method (`access/table`) — *the pluggable-engine seam, and its limit*
`TableAmRoutine` of C callbacks; heap is the reference impl. **The constraint that decides ProximaDB's whole strategy:** to support modifications/indexes, "each tuple must have a tuple identifier (TID) of a block number and item number," and the table-AM and index-AM APIs are tightly coupled (every index assumes a physical TID). This is why a columnar or vector engine *cannot* be a clean Postgres extension — OrioleDB had to patch core.

**Why this matters for ProximaDB:** Postgres factors storage into *addressing* (`smgr`), *cache+eviction policy* (`bufmgr`), and *resource pooling* (`fd.c`) — three layers, not one. ProximaDB currently has two seams (FileSystem + tiering) that straddle these concerns. §7 uses this factoring to answer the "consolidate or separate?" question.

### 1.5 The OS-paging analogy the user raised, made precise
| OS / Postgres construct | Mechanism | ProximaDB analog |
|---|---|---|
| Page cache + **CLOCK** replacement | `bufmgr` clock-sweep | RAM/NVMe byte cache + eviction policy |
| Wired/locked pages | buffer **pin counts** | "do not evict footer/zone-map/hot SQ8 stripes" |
| `madvise(SEQUENTIAL/DONTNEED)` / `O_DIRECT` | **`BufferAccessStrategy` rings** | a big S3 Parquet scan must **not** flush the OLTP working set |
| Swap to disk | — | **tier demotion** to S3 Standard/Glacier (already `Tier1..Tier5`) |
| fd table virtualization | **`fd.c` VFD LRU pool** | S3 connection/handle pooling, ranged-reader handles |
| Demand paging via `mmap` | `mmap()` | ProximaDB `mmap()` (local only) + cloud guard |

The punchline (developed in §7): **"page bytes into RAM" and "page bytes down to cold object storage" are the same paging idea at two rungs of one hierarchy.** Unify the *policy*; keep the *mechanism* (`FileSystem`) separate.

---

## 2. ProximaDB today — what exists, what's a stub

From the HEAD code map. Status flags matter more than completeness.

| Subsystem | Status | Key types / files | One-line reality |
|---|---|---|---|
| **pgwire** | LIVE | `network/postgres/{mod,protocol,session}.rs`; `PostgresServer`, `Portal`, `PortalExecutionState` | Extended-query w/ portal paging; tenant bound via `session.database`. |
| **Multi-engine router** | **P0 stub-ish** | `query/compute_scheduler.rs`; `ComputeScheduler::route()`, `ComputeBackend{Native,DataFusionLocal,…}`, `QueryShape` | **Always routes to Native (Volcano) today**; OLAP classification + P1-flip infra present, not flipped. |
| **WAL / OLTP RAW** | LIVE | `storage/persistence/write_ahead_log/`, `services/record_store.rs`, `storage/memtable/`; `DirectWalTableRecordStore`, `scan_index: BTreeMap`, `handle_flushed_vectors` | Memtable (unordered `HashMap`) + WAL + lazy `BTreeMap` scan index; synchronous flush; post-flush AXIS rebuild shipped (TD-112). |
| **Lakehouse / Iceberg** | LIVE (v1) | `proximadb-iceberg-engine/manifest.rs` (`ManifestCommitter`, CAS), `object_store_bridge.rs`, `iceberg_rest_service.rs`, `materialize_table_to_parquet` | Versioned `v{N}.manifest` via optimistic CAS; **v1 key-list manifest, not full Iceberg** (no snapshot layer / metadata.json yet); Iceberg REST server live for external readers. |
| **Tiering — SEAM A** | LIVE | `storage/persistence/filesystem/mod.rs`; `FileSystem` trait, `LocalFileSystem`, `AwsS3FileSystem`, `Azure…`, `Gcs…`, `UnifiedCachingFilesystem` | *Mechanism*: scheme-based backend (`file://`, `s3://`, …), atomic-write strategies, `read_range`, `mmap`, **caching wrapper fused in**. |
| **Tiering — SEAM B** | LIVE | `infrastructure/tier_policy_engine.rs`; `FileStorageTier{Memory,NVMe,SSD,HDD,S3Express,S3Standard,Glacier,…}`, `Tier1..Tier5` | *Policy*: assigns collections to tiers by SLA/cost; emits a `base_url` consumed by Seam A. |
| **Snapshots / branching** | PARTIAL | `services/snapshot/coordinator.rs` (`SnapshotPin`, `SnapshotPublishCoordinator`); manifest versioning | `SnapshotPin` captures `(wal from..to LSN, manifest checkpoint)` — but only for discovery pinning. **No CoW, no time-travel SQL, no branch refs yet.** |
| **Multi-tenancy** | LIVE | `TenantContext`, `DrPathBuilder` (`data/{tenant}/{ns}/{coll}/`), `check_tenant_path_guard.py` | Hard path isolation + catalog/DML scoping + write gates. |
| **PAX block format** | LIVE (SQ8 on branch) | `proximadb-block-format`, `pax_block.rs`; `PaxSegmentWriter`, zone-maps, SQ8 codec, footer-first ranged reads | Block+row-group min/max pruning; SQ8 4× quant + predicate pushdown complete on `feat/pax-v2-sq8-pushdown` (not merged); `BlockStats`→Iceberg `DataFile`. |

**Reading of the gap:** ProximaDB is ~70% of the way to the target architecture but doesn't *name* it as such. It already has: a pgwire front, a router with the right enum, an LSM-ish hot tier (memtable+WAL+scan index), a columnar base (PAX/Parquet) with pruning, a versioned manifest committer with CAS, and an LSN-capturing snapshot pin. What's missing is **(a) the explicit hot→cold compaction spine, (b) branch refs over the existing manifest/LSN primitives, (c) a cost-based router flip, and (d) a unified paging/tier policy.** The rest of this doc is mostly *connecting parts you already have.*

---

## 3. Comparative study (Neon, Lakebase, Supabase, + the field)

### 3.1 Neon — the transferable mechanism
Stateless Postgres compute + **Safekeepers (Paxos WAL quorum)** + **Pageservers** + S3. *The WAL is the interface.* Pageserver is an **LSM-like store of page versions keyed by LSN**: immutable **image layers** (snapshot of a key range at an LSN) + **delta layers** (WAL over an LSN range); `GetPage@LSN` finds the newest image ≤ LSN and replays WAL on top; layers offload to S3 ("bottomless"); GC drops layers outside the restore window. **Branching = record parent LSN; branch starts empty; reads of unmodified data fall through to the parent; only post-branch deltas are stored** → O(1), size-independent. PITR is the same machinery (restore-to-LSN). *Neon does not use Parquet/Iceberg — custom format.* (Sources: neon.com/blog/get-page-at-lsn, neon.com/docs/introduction/branching, github.com/neondatabase/neon pageserver-storage.md.)

> **Adopt for ProximaDB's hot tier:** LSN-pointer CoW branching is exactly what agent branches need, and ProximaDB's WAL+`SnapshotPin` already capture LSN ranges.

### 3.2 Databricks Lakebase (2025) — and the honest caveat
Serverless Postgres OLTP built on Neon (Databricks acquired Neon ~May 2025). <10 ms / >10k QPS / scale-to-zero / 35-day PITR; git-style branching inherited from Neon. **Critical nuance:** as launched, Lakebase does **not** store Postgres data in Delta — it keeps its own copy and bridges to the lakehouse via **sync pipelines**: lakehouse→Postgres "synced tables" (Snapshot/Triggered/Continuous, CDF-based) and Postgres→lakehouse "Lakehouse Sync" (CDC → Delta as **SCD Type 2**). True single-copy convergence is the *later* **LTAP** announcement. (Sources: databricks.com/blog/announcing-lakebase-public-preview, docs.databricks.com/.../sync-tables, .../lakehouse-sync, thebuild.com pg_lake-vs-lakebase.)

> **Lesson:** even the best-funded attempt runs **two copies bridged by CDC**, not one magic format. Don't over-promise single-copy HTAP; design the compaction/sync bridge explicitly.

### 3.3 Supabase — branching is *not* what agents need
Stock Postgres + sidecars (PostgREST, GoTrue, Realtime via WAL logical replication, Storage, pgvector). **Supabase "branching" = separate ephemeral Postgres instances seeded by re-running migrations — NOT copy-on-write, and production data is deliberately never cloned.** CoW-from-snapshot is an aspiration, not shipped. 2025: **Analytics Buckets** (Iceberg+Parquet on S3, queried via an Iceberg **FDW**) in private alpha. (Sources: supabase.com/docs/guides/deployment/branching, supabase.com/blog/analytics-buckets.)

> **Lesson:** Supabase's branching is the *weak* model (instances + migrations) — fine for preview envs, useless for "fork 10 GB of agent state instantly." ProximaDB should target the **Neon/Iceberg-ref CoW model**, which is a competitive advantage Supabase lacks.

### 3.4 The rest of the field (one line each)
- **OrioleDB:** index-organized tables, undo log (no bloat/VACUUM), CoW checkpoints, S3 "bottomless" tiering — *but requires Postgres core patches* (TID constraint). Proves decoupled storage in PG is possible only by forking core.
- **pg_lake (Crunchy→Snowflake):** Postgres acts as the Iceberg **catalog**; DuckDB runs out-of-process (`pgduck_server`) over pgwire; **cross-boundary heap↔Iceberg txns are not fully ACID.**
- **DuckLake:** puts *all* table metadata in a transactional SQL DB (not file-based manifests); data stays Parquet on object store; small writes inlined into metadata. The "BigQuery model: Colossus (data) + Spanner (metadata)." Directly relevant to ProximaDB's manifest design (§6).
- **Iceberg branching:** branches/tags are entries in the `refs` map of `metadata.json`; a branch is just a named pointer to a snapshot; data files shared by reference; WAP for write-audit-publish. **This is the columnar-base branch model to copy.**
- **Delta shallow clone:** metadata-only clone referencing source files — but no shared ref-counting safety (source VACUUM breaks the clone). Iceberg `refs` is the safer model.
- **HTAP dispatch:** *TiDB/TiFlash* = two engines + a **cost-based router** (row→TiKV, column→TiFlash, can mix in one query; cost model admittedly "rough," manual hints as escape hatch). *SingleStore* = one converging "Universal Storage" (columnstore made seekable for OLTP). *Umbra/CedarDB* = one engine + one hybrid store (Colibri), data *migrates* row↔column, no routing. **ProximaDB is closest to the TiDB model** (it has distinct engines + a router) — so adopt TiDB's cost-based, hint-augmented dispatch.

---

## 4. The central tension (and the only honest resolution)

**You cannot serve sub-millisecond point read-after-write *and* efficient columnar scans from one physical layout.** Parquet is immutable with large row groups; a single-row update means rewriting a column chunk; a row's fields are scattered across column sections. Table formats (Iceberg/Delta/Hudi) add ACID metadata but **don't change Parquet's physics.** (Delta Lake VLDB 2020.)

Every system resolves this the same way — **delta store (row, mutable/append, hot) + base store (columnar, immutable, compacted) + background merge** — differing only in *when* they merge (eager CoW = write-amp; lazy MoR = read-amp):

| System | Hot/delta | Cold/base | Merge |
|---|---|---|---|
| Hudi MoR | row log files | Parquet base | async compaction |
| Delta Lake | deletion vectors + small Parquet | compacted Parquet | `OPTIMIZE` (deferred) |
| SingleStore | in-mem rowstore segment | on-disk column segments | background flusher |
| ClickHouse | small parts | merged parts | background merge |
| **ProximaDB (target)** | **memtable + WAL (LSM)** | **PAX/Parquet + Iceberg manifest** | **background compactor** |

ProximaDB already has both ends (memtable+WAL; PAX/Parquet). **What's missing is the named, tunable compactor and the rule for read-time merge** (read sees memtable ∪ flushed-but-uncompacted PAX ∪ compacted base). Make this the explicit spine.

---

## 5. Refined target architecture

```
                    ┌─────────────────────────────────────────────┐
   Postgres wire ──▶ │  pgwire front (session, auth, portals)       │
   (psql/ORMs)       │  + pg_catalog completeness layer  [§0.7]     │
                    └───────────────────┬─────────────────────────┘
                                        │ parsed/bound + TenantContext + BranchRef
                    ┌───────────────────▼─────────────────────────┐
                    │  ComputeScheduler  (COST-BASED, hint-aware)  │
                    │  selectivity crossover; may split one query  │
                    └───┬──────────────┬───────────────┬──────────┘
                point/  │       scan/  │        vector │
                short   │   agg/join   │           ANN │
              ┌─────────▼───┐  ┌───────▼────────┐  ┌───▼──────────────┐
              │ OLTP engine │  │ DataFusion     │  │ AXIS/HNSW/IVF    │
              │ Volcano over│  │ over Parquet/  │  │ over PAX-SQ8     │
              │ memtable+WAL│  │ Iceberg base   │  │ (+ rerank)       │
              └──────┬──────┘  └───────┬────────┘  └───┬──────────────┘
                     │  read = memtable ∪ uncompacted PAX ∪ compacted base
              ┌──────▼───────────────────────────────────────────────┐
              │ STORAGE SPINE: delta(hot LSM) → compactor → base(cold) │
              │  WAL (LSN, tiered)  •  memtable  •  PAX/Parquet+manifest│
              └──────┬───────────────────────────────────────────────┘
        ┌────────────▼───────────┐   ┌──────────────────────────────────┐
        │ PAGING / CACHE policy   │   │ TIER / LIFECYCLE policy           │   ← unified
        │ (RAM/NVMe admission+evict)│ │ (NVMe→SSD→S3→Glacier demotion)    │     POLICY brain [§7]
        └────────────┬───────────┘   └───────────────┬──────────────────┘
                     │      one cost/temperature model │
              ┌──────▼─────────────────────────────────▼──────┐
              │ FileSystem trait — MECHANISM only               │  ← keep separate [§7]
              │ (s3/adls/gcs/local, atomic write, ranged read)  │
              └─────────────────────────────────────────────────┘

   BranchRef = small object-store metadata object:
     { parent, catalog_snapshot_id, per_collection_fork_LSN, per_table_manifest_version }
```

### 5.1 The storage spine (delta → compactor → base)
- **Delta/hot (OLTP read-after-write):** keep the memtable + WAL. Treat it as an LSM: memtable (mutable) → on flush, an immutable **L0 PAX segment** (row-group-friendly but small); the `scan_index` BTreeMap stays the in-memory seek structure. Read-after-write is served from memtable (already works).
- **Base/cold (analytics):** background **compactor** merges L0 PAX segments into large columnar Parquet files registered in the Iceberg manifest. This is where Delta's `OPTIMIZE`/Hudi async-compaction lives. Use `dataChange=false`-style semantics so compaction is invisible to snapshot isolation.
- **Read-time merge:** every read = `memtable ∪ uncompacted-L0 ∪ compacted-base`, minus a **deletion-vector** overlay (adopt Delta's merge-on-read deletes so you don't rewrite Parquet on every delete). The router decides whether the OLTP engine (point) or DataFusion (scan) drives the merge.
- **Tunable knob:** compaction eagerness (write-amp vs read-amp vs small-file count) becomes a per-collection policy, co-located with the tier policy (§7).

### 5.2 WAL as interface + branch substrate (Neon-style)
- WAL is durability *and* the time axis. Make it **tiered**: recent WAL on local NVMe ("temp"), sealed segments offloaded to object store ("permanent"), addressed by LSN — exactly Neon's safekeeper→pageserver→S3 progression. ProximaDB's `SnapshotPin` already records `wal:from..to` + `checkpoint:N`.
- This gives **PITR and time-travel for free** (read-as-of-LSN), and is the hot-tier half of branching.
- **Disaggregated read-after-write protocol (if/when storage is decoupled from query nodes):** copy Neon's two-LSN `GetPage@LSN` — a read carries `request_lsn` (version wanted) + `not_modified_since` (a conservative last-written-LSN hint), and the storage side **waits-for-LSN** before answering only when the page actually changed. This delivers read-your-writes without forcing every read to block on the latest WAL. ProximaDB's synchronous flush makes this a non-issue today, but it's the protocol to adopt before splitting compute from storage.

### 5.3 Multi-engine dispatch (cost-based, TiDB lesson)
- Flip `ComputeScheduler` from "always Native" to cost-based. The infra is present (`QueryShape.parquet_backed`, `routed_read_plan`). Decision rule: **point/high-selectivity → OLTP Volcano; large-scan/low-selectivity/agg/join → DataFusion over base; ANN → AXIS.** Model a **selectivity crossover** where row-scan and column-scan costs equalize.
- **Hybrid plans:** allow one query to push a row fragment to the OLTP engine and a scan fragment to DataFusion (TiDB/PolarDB-IMCI do this).
- **Manual hints + `tidb_isolation_read_engines`-style session knobs** as escape hatch — even TiDB's CBO is "still rough."

### 5.4 Agent branching — see §6 (the headline feature).

---

## 6. Branching for agents — object-store-metadata snapshots (the headline)

**Goal:** an agent forks the database state instantly, gets an isolated read-write branch, and the fork cost is independent of data size.

**Mechanism — one branch ref, two CoW substrates you already have:**

A **`BranchRef`** is a small immutable object written under `data/{tenant}/{ns}/_branches/{branch_id}.json`:
```jsonc
{
  "branch_id": "agent-42-exp-7",
  "parent": "main",
  "created_at_ns": 1718600000000000000,
  "catalog_snapshot_id": "cat:v128",
  "collections": {                         // hot OLTP tier (Neon-style LSN CoW)
    "docs":   { "fork_lsn": 920183 },
    "vectors":{ "fork_lsn": 920183 }
  },
  "tables": {                              // columnar base (Iceberg-ref-style)
    "events": { "manifest_version": 57 },
    "users":  { "manifest_version": 41 }
  }
}
```

- **Hot tier (memtable+WAL):** the branch records each collection's **fork LSN**. Reads of unmodified records fall through to the parent's WAL/segments at ≤ fork_lsn (Neon's "fetch from the parent" rule); the branch writes only its own post-fork WAL/memtable. ProximaDB's WAL + `SnapshotPin` already capture the LSN range; this is wiring, not new infra.
- **Cold base (Parquet/Iceberg):** the branch records each table's **manifest version**. ProximaDB's `ManifestCommitter` already writes immutable, monotonically versioned `v{N}.manifest` with optimistic CAS — that *is* the Iceberg `refs` substrate. A branch = a ref pointing at a version; data files are shared by reference (CoW); divergent writes create new manifest versions under the branch's own ref. WAP (write-audit-publish) and `merge`(fast-forward) follow directly.
- **Catalog:** pin a catalog snapshot id so schema is consistent with the data versions.

**Why this is the right design:**
- Branch create = write *one* JSON object → **O(1), size-independent** (Neon/Lakebase property; the thing Supabase's migration-replay branching cannot do).
- It is **object-store-metadata-driven** exactly as requested — no data copy; immutable data shared by reference until divergence.
- It composes with the DuckLake insight: if manifest-version lookup over S3 becomes a bottleneck at branch scale, **move the branch/manifest registry into a transactional catalog DB** (DuckLake's "metadata in a DB, data in Parquet" model) while keeping data files in object store. Start file-based (matches current `ManifestCommitter`), graduate to catalog-DB if metadata ops dominate.

**Two mechanisms to copy verbatim from Neon's `index_part.json` design** (the part that makes this robust at scale):
1. **The manifest is the authoritative file list.** Neon's per-timeline `index_part.json` is the source of truth: *"if a file is not referenced from IndexPart, it's not part of the remote storage state."* Adopt this for the BranchRef/manifest: a file exists for a branch iff its manifest references it. Enforce **upload ordering** (write data files before the manifest that references them) so a crash never leaves a manifest pointing at absent data.
2. **Generation/epoch numbers for single-writer fencing.** Neon stamps each timeline's remote writes with a monotonic *generation* so the control plane guarantees no two pageservers write the same tenant. ProximaDB has `DrPathBuilder`+`TenantContext` path isolation but **no writer fence** — add a per-`(tenant, branch)` generation guard to the `ManifestCommitter` CAS so a stale/forked writer can't corrupt a branch. This is the missing piece for safe concurrent agent branches.

**Read fall-through (the CoW read path):** a read on a branch resolves a record/page by walking *up the ancestor chain* — if the branch's own WAL/manifest has no entry at ≤ the requested version, recurse to the parent (Neon's "returns data from the ancestor timeline if it's not found on the current timeline"). Within a single branch the history is linear, so the requested LSN/version unambiguously selects the layer.

**Lifecycle / GC:** branch GC = drop the ref + reference-count data files/WAL layers (Iceberg `expire_snapshots` semantics — *safer than Delta shallow clone*, which can't ref-count and breaks on source VACUUM). Use Neon's precise removal rule: a layer/file is collectable when it is **older than the retention horizon AND superseded by a newer file for the same key range, UNLESS pinned by a child branch or the PITR window**. Time-travel and PITR reuse the same `(LSN, manifest_version)` addressing — *PITR is just "branch at a historic version,"* one mechanism for both.

---

## 7. The two seams — consolidate or keep separate?

**The crisp answer: there are really *three* concerns, currently packed into two seams. Keep the mechanism seam separate; consolidate the policy brain; split the cache out of the mechanism.**

Map ProximaDB's seams onto Postgres's proven factoring (§1):

| Concern | Postgres | ProximaDB today | Recommendation |
|---|---|---|---|
| **Addressing + I/O mechanism** ("which backend, how to read bytes") | `smgr`/`md.c` + `fd.c` | `FileSystem` trait (backends, atomic write, `read_range`, mmap) | **Keep as-is, single seam.** This is mechanism. It should *not* know about SLAs, cost, or temperature. |
| **Cache / paging** ("which bytes live in RAM/NVMe right now; admit/evict") | `bufmgr` (clock-sweep, pins, `BufferAccessStrategy` rings) | **fused into `UnifiedCachingFilesystem`** | **Split it out** of the FileSystem trait into its own layer. It is a *policy*, not I/O. Add `BufferAccessStrategy`-style ring buffers so a big S3 scan doesn't evict the OLTP working set. |
| **Tier / lifecycle** ("where bytes should live long-term by SLA/cost") | (no real analog; tablespaces) | `tier_policy_engine` (`Tier1..Tier5`, `base_url`) | **Keep as a policy layer**, but unify its brain with the cache policy. |

**Why keep FileSystem (mechanism) separate from tiering (policy):** Postgres deliberately keeps `smgr` (which storage) independent of `bufmgr` eviction (cache policy). The `FileSystem` trait answering "is this `s3://` or `file://` and how do I do an atomic write" must not be entangled with "this collection is cold, demote it to Glacier." They change for different reasons and at different rates. **Do not merge them.**

**Why consolidate the *policy brain*:** "page a hot block into RAM" (cache admission) and "demote a cold segment to S3 Glacier" (tier demotion) are **the same OS-paging decision at two rungs of one memory hierarchy** (RAM → NVMe → SSD → S3 Standard → Glacier). Today they're decided by two unrelated components (the caching filesystem's LRU vs the tier policy engine). Unify them under **one temperature/cost model** — a "hierarchy manager" that sees the full continuum and makes admission/eviction/demotion/promotion decisions coherently — while it actuates through the *single* `FileSystem` mechanism. This is precisely how an OS unifies page cache + swap under one VM policy while the block layer stays a dumb mechanism.

**Concrete refactor:**
1. Extract caching out of `UnifiedCachingFilesystem` → a `PagingCache` layer (byte-addressable, pin-aware, ring-buffer-aware). ProximaDB already has `TenantCache`/footer-cache primitives to build on.
2. Define one `HierarchyPolicy` trait consumed by *both* the paging cache (admit/evict RAM↔NVMe) and the tier engine (promote/demote NVMe↔S3↔Glacier), parameterized by the same `(temperature, tenant SLA, cost)` signals.
3. `FileSystem` stays the pure mechanism both layers call. `DrPathBuilder` path isolation stays in front of all of it (unchanged).

Net: **2 seams → 1 mechanism + 1 unified policy brain (with 2 actuation rungs).** That is the consolidation that pays off; merging mechanism into policy would not.

---

## 8. Strategic decision: extend Postgres vs custom pgwire engine

ProximaDB is already a **custom Rust engine speaking pgwire** — the research strongly says *stay the course*:
- The Table-AM **TID constraint** structurally blocks hosting a columnar/vector engine as a Postgres extension. OrioleDB needed *core patches* for even row-oriented decoupled storage.
- Object-storage-first + Arrow/DataFusion is the native idiom for the lakehouse+vector half (GreptimeDB, DataFusion ecosystem).
- **CoW branching over immutable object-store manifests is achievable in a custom engine but is an open feature request in CockroachDB and absent from core Postgres.** This is ProximaDB's moat.

**The price (budget for it explicitly):**
- **`pg_catalog` completeness** is the #1 reason "compatible" engines break ORMs (Materialize/RisingWave/QuestDB all hit this). Make it a roadmap line item; reference `datafusion-postgres`'s catalog-over-DataFusion implementation and the `pgwire` (sunng87) crate (which ProximaDB-style servers already use).
- **OLTP/MVCC correctness** must be earned, not inherited. Define the isolation level you actually offer and test it (don't accidentally ship CockroachDB's "SERIALIZABLE-only + mandatory client retries" surprise).

---

## 9. Phased roadmap (small steps over parts you already have)

- **P1 — Name the spine.** Document memtable+WAL = delta, PAX/Parquet+manifest = base. Add the **background compactor** (L0 PAX → big Parquet) + **deletion vectors** for merge-on-read deletes. *(Mostly wiring existing flush + manifest paths.)*
- **P2 — Flip the router.** Make `ComputeScheduler` cost-based with a selectivity crossover; set `parquet_backed=true` for compacted tables; add session hints. *(Infra present.)*
- **P3 — WAL tiering + time-travel.** Seal+offload WAL segments to object store; expose read-as-of-LSN. *(Builds on `SnapshotPin`.)*
- **P4 — BranchRef.** Implement the `_branches/{id}.json` ref over existing `(fork_lsn, manifest_version)` primitives; read fall-through to parent; branch GC via ref-counting (Iceberg `expire_snapshots` semantics). *(The headline agent feature.)*
- **P5 — Unify paging+tier policy.** Extract `PagingCache` from `UnifiedCachingFilesystem`; one `HierarchyPolicy`; add `BufferAccessStrategy`-style scan rings.
- **P6 — Full Iceberg + catalog-DB option.** Promote v1 key-list manifest to real Iceberg snapshots/metadata.json; optionally move branch/manifest registry into a transactional catalog DB (DuckLake model) if metadata ops dominate at branch scale.
- **P7 — pg_catalog hardening.** Close ORM-introspection gaps; CI test against psql/sqlx/Diesel/SQLAlchemy.

---

## 10. Risks & honest caveats
- **Don't promise single-copy HTAP.** Even Lakebase ships two copies + CDC; ProximaDB's delta+base+compactor is the same shape — say so.
- **Cost model will be rough** (TiDB's own admission). Ship hints from day one.
- **Branch ref-counting GC is the subtle part.** Delta shallow clone's source-VACUUM breakage is the cautionary tale; use Iceberg-style snapshot expiry with reference counts, not naive deletes.
- **`pg_catalog` is a perpetual tax**, not a one-time task.
- **mmap is local-only** (existing cloud guard); the paging layer must degrade to ranged reads on object store.

---

### Appendix — primary sources
Neon get-page-at-lsn & branching (neon.com/blog/get-page-at-lsn, neon.com/docs/introduction/branching); Lakebase (databricks.com/blog/announcing-lakebase-public-preview, docs.databricks.com sync-tables / lakehouse-sync); Supabase branching & analytics buckets (supabase.com/docs/guides/deployment/branching, supabase.com/blog/analytics-buckets); Delta Lake VLDB 2020 (vldb.org/pvldb/vol13/p3411-armbrust.pdf); Iceberg branching & spec (iceberg.apache.org/docs/latest/branching, /spec); DuckLake manifesto (ducklake.select/manifesto); pg_lake (github.com/Snowflake-Labs/pg_lake); OrioleDB (orioledb.com/docs/architecture/overview, /blog/better-table-access-methods); Postgres internals (postgresql.org/docs/current/tableam.html, custom-scan-path.html; `~/code/postgres` smgr/buffer/fd READMEs); TiDB HTAP (docs.pingcap.com tiflash-overview, vldb.org/pvldb/vol13/p3072-huang.pdf); CedarDB on PG compatibility (cedardb.com/blog/postgres_compatibility); pgwire crate (github.com/sunng87/pgwire), datafusion-postgres (github.com/datafusion-contrib/datafusion-postgres).
