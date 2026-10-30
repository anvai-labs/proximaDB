# Changelog

All notable changes to ProximaDB will be documented in this file.

## [0.5.0] - 2026-10-29

Lakehouse-publication release: **211** merged PRs (203 merge commits plus 8
squash or custom-titled merges on the first-parent line; 645 commits) since
`v0.4.1`, counted from origin's `v0.4.1` tag (commit `c4043fb3f`) to `develop`
at `62213e4ca`.

### Upgrade notes (read before deploying)
- **Back up first: rollback to v0.4.x is unsupported.** Back up the
  `data_dir` AND every object-store prefix the deployment writes (the Iceberg
  warehouse, PAX segments, spill): deletion vectors and new PAX footers can live
  outside `data_dir`. Data format 2 is the
  default: canonical-WAL segment rotation, WAL truncation and Layer A spill are
  on (ADR-097 P2). Once the truncator runs, spill segments are the only copy of
  some rows, and v0.4.x binaries do not read sealed WAL segments. The
  `{data_dir}/pgwire/DATA_FORMAT` marker (format 2) makes this and later
  binaries refuse a newer-format directory, but v0.4.x predates the marker and
  cannot check it. To go back, restore the pre-upgrade backup.
- **Incremental MATERIALIZE is ON by default** (`PROXIMADB_MATERIALIZE_INCREMENTAL_DV`).
  After an UPDATE or DELETE, `ALTER TABLE … MATERIALIZE` carries the published
  files, appends the changed rows and hides old row versions with Iceberg v3
  deletion vectors (`dv-mat-*.puffin`) instead of rewriting the table. It falls
  back to a compacting rewrite past 20% of rows or 500k positions deleted, at
  64 or more parts, or above 1M changed rows. Insert-only republication
  likewise appends new multi-part, generation-named files. Both incremental
  paths share `PROXIMADB_MATERIALIZE_APPEND_MAX_ROWS` (default 1M; `0`
  disables both); `PROXIMADB_MATERIALIZE_INCREMENTAL_DV=0` restores full
  rewrites after updates and deletes only.
- **Iceberg format versions.** v0.4.1 published every table as format-version 3
  without row lineage. 0.5.0 adopts those tables: their first 0.5.0
  MATERIALIZE starts row lineage at 0, which is safe because they never handed
  out a row id. New 0.5.0 tables publish **v2** by default
  (`PROXIMADB_ICEBERG_V3_DEFAULT` is OFF) **unless they have a TIMESTAMP or
  TIMESTAMPTZ column**: nanosecond timestamps need v3, so those tables are v3
  from their first MATERIALIZE. Other tables become v3 at their first
  republication that writes deletion vectors. External read-back is verified
  for pyiceberg 0.12 (REST and static) and DuckDB 1.5, not Trino; v2-only
  consumers cannot read a table once it is v3.
- **External readers of deletion-vector tables.** A v0.4.x binary, or any reader
  that ignores Iceberg deletes, silently resurrects deleted rows and shows
  updated rows twice. To publish without vectors, set
  `PROXIMADB_MATERIALIZE_INCREMENTAL_DV=0` and re-MATERIALIZE.
- **Do not disable `PROXIMADB_OLAP_ORDERED_DV_BASE` on its own** while incremental
  MATERIALIZE is on: OLAP reads of tables with deletion vectors would refuse
  after every write. Disarm both and re-MATERIALIZE (see `ENV_GATE_REGISTRY`).
- **PAX residual tail is ON** (`PROXIMADB_PAX_RESIDUAL_TAIL`). New vector/PAX
  segments with typed filterable columns carry a v2 field-map section that
  v0.4.x readers reject at the footer parse. During a rolling or mixed-version
  upgrade, set `PROXIMADB_PAX_RESIDUAL_TAIL=0` until every node runs 0.5.0.
- **Spill cannot be switched off after use.** Setting
  `PROXIMADB_RELATIONAL_SPILL_MAX_RESIDENT=0` over a spill root that already
  holds segments refuses boot, because those rows may be in no retained WAL
  entry.
- **SST is the default engine** for vector collections created with no index,
  quantization, recall target or policy (previously the Beta HELIX engine).

### Added
- **Iceberg deletes and incremental publication (TD-USUB-10):** v3 deletion
  vectors, Puffin files and delete manifests; delete-aware readers on every
  path (DataFusion, DuckDB, external read-back); an ordered-DV-base OLAP merge so
  later writes merge over a deletion-vector base; insert-only republication
  appends; Puffin GC; a seeded incremental soak with external read-back in CI.
  An opt-in (default OFF) per-DELETE live publisher behind the visibility fence
  contract.
- **Bounded-memory MATERIALIZE (P2.4):** multi-part, generation-named Parquet,
  spill-merged chunked scans, a hidden `__proxima_oid` row key, fenced snapshot
  advance; the TPC-DS SF1 harness loads in bounded memory.
- **Layer A streaming SQL (ADR-097 P3.1, P3.2a), opt-in:** behind
  `PROXIMADB_NATIVE_TABLE_PROVIDER=1` (default OFF), ungoverned sessions only —
  a streaming table provider for native SQL with no row cap, oid-range partitions, row-group pruning with a key
  spine, residual-aware pruning, DataFusion dynamic filters; queries mixing
  Parquet and native tables (P3.1), or over table functions alone (P3.2a), run
  on DataFusion.
- **Canonical WAL:** segmented WAL with a per-segment LSN index, a scheduled
  truncator, durable branch merge-base images (TD-SEAM-12), per-partition
  truncation floors, lagging-partition flush so small tables stop pinning
  truncation.
- **PAX vector search:** a collection-global centroid directory (TD-PAXDIR-1,
  defaults ON), parallel object-store waves, residual-tail store-once and an
  instrumented filtered path (TD-FPRUNE-1).
- **Storage classes (TD-TIER-1):** storage-class policy, reservation grants that
  protect cached segment data, pin-safe eviction and live residency status.
- **OLTP (ADR-095):** atomic pgwire COMMIT and writer-epoch fencing on a
  reattachable WAL volume (default OFF).
- A one-command agent-memory demo; release evidence (provenance plus registry
  install smoke) for every artifact.

### Fixed
- Upgrade (found by a v0.4.1 → 0.5.0 drill on real data): tables MATERIALIZEd
  by v0.4.1 read correctly before their first 0.5.0 MATERIALIZE (v0.4.1
  recorded paths relative to the warehouse root), and republish instead of
  failing closed (Iceberg v3 without row lineage is adopted at row id 0). A
  golden fixture of real v0.4.1 output guards both.
- Read-after-write: deleted rows no longer resurrected and flushed updates no
  longer lost on vector collections; filtered-path search is WAL-authoritative
  with TTL on PAX hits.
- DML: `INSERT … ON CONFLICT` honours its target and action (TD-DML-3); UNIQUE
  and FOREIGN KEY use the declared key (TD-RELIDEQ-2).
- pgwire: extended-protocol Describe reports the columns Execute streams
  (TD-PGEXT-1); window functions without a DataFusion route decline (0A000)
  instead of answering wrong.
- Catalog: atomic table replace and ordered schema updates (TD-CATREPLACE-1).
- Tenancy: every collection surface resolves within the request tenant; an
  absent scope denies.
- REST v2 rejects unknown and invalid create settings (TD-RESTGET-1); gRPC/REST
  serve the persisted collection schema (TD-GRPCSCHEMA-1/2).
- SDKs: the Python SDK surfaces failures and honours `engine=`; the TypeScript
  SDK validates `index()`.
- Packaging: shipped RPM/DEB artifacts persist data and the service starts
  (TD-PKG-1/2/3). Object stores report a missing object as NotFound on S3,
  Azure and GCS. The server installs SIGINT/SIGTERM handlers before boot.

### Security
- RUSTSEC-2026-0367/0368/0369 (zip via duckdb), 0381/0382/0383 (webpki trust
  anchors), 0385 (ctutils), 0414 (pulldown-cmark, build-time via prost-build)
  and 0418 (libz-sys) cleared; the panic policy now
  covers the whole workspace (TD-PANICSCOPE-1).
- **Known, not fixed:** 13 zstd advisories published 2026-10-26
  (RUSTSEC-2026-0391 for zstd 0.13.3; 0395–0406 for zstd-safe 7.2.4) ship
  ignored in `.cargo/audit.toml`. Three are vulnerability-class (0391, 0398,
  0403); the rest are unsound warnings on streaming, seekable and buffer APIs.
  No lock-level fix exists: the patched releases need zstd >= 0.14.3 and
  zstd-safe >= 8.2.0, while arrow-ipc and parquet 59 pin zstd ^0.13 and
  compression-codecs (async-compression, via tower-http) co-pins zstd-safe
  7.x. Retiring them needs both an arrow major and a tower-http /
  async-compression move (TD-AUDITGATE-1).

### Internal
- Duplication ratchet (may only go down); more than 18k lines of verified-dead
  code and 52 orphaned source files removed; a shared pgwire e2e harness; dependency upgrades (avro, tantivy,
  OpenTelemetry, sha2, toml, metrics, ONNX Runtime with one shared environment).

## [0.4.1] - 2026-10-16

### Fixed
- WAL replay honours a config-file `wal_local_dir`, so acknowledged writes survive a crash (TD-WAL-5, critical).
- `fp32` point reads are byte-exact after flush (TD-READFID-1).
- UPSERT enforces UNIQUE constraints, the constraint key store (CKS) and foreign keys through the INSERT sequence (TD-DML-1, TD-DML-2).
- pgwire COPY and the DML primary-pod gate resolve the write tenant (TD-PGWIRE-COPY-1).

### Security
- One shared `top_k` cap on every search surface; wrong-dimension query vectors are client errors, not a kernel abort; IMDSv2-only AWS probes (TD-SECSCAN-1).
- rustls 0.23.46 for RUSTSEC-2026-0346..0349, with the CryptoProvider installed at boot; GCS client upgrade closes a jsonwebtoken advisory (TD-RUSTDEP-1).
- Admin UI moved to Vite and self-hosts its editor runtime (TD-UICDN-1).

### Added
- Offline cold backup and restore server subcommands.
- One tag `vX.Y.Z` releases every registry at one version (TD-RELTAG-1).

## [0.4.0] - 2026-10-11

Storage-substrate release: **172** commits since the `v0.3.0` tag, counted at
this release branch's merge-base; 174 on `develop` at the time of writing, and
what finally ships is whatever `develop` holds at promotion — the two extra
commits include the spill-tombstone-persistence work credited below, so treat
172 as the anchored figure rather than the shipped one.

The anchor is `refs/tags/v0.3.0` **as it exists on origin** — a lightweight tag
on `9e19fc2a3`, dated 2026-09-04 — because that is what a clone resolves and
therefore the only anchor under which these numbers reproduce. A reviewer caught
an earlier revision of this entry quoting 415 commits, 35 OpenAPI paths and three
new ADRs: all of those were computed against a *stale local* `v0.3.0`, an
annotated tag at `b1ed7817` (2026-08-15, an ancestor of origin's). `git fetch
--tags` does not correct it — the update is rejected as "would clobber existing
tag" — so anyone recomputing these figures should verify
`git rev-parse v0.3.0` against `git ls-remote --tags origin v0.3.0` first.

The 0.3.0 entry below is dated 2026-08-05, when `chore(release): prepare v0.3.0`
landed; origin's tag was cut at the 2026-09-04 promotion. Both dates are real and
mean different things — the prepare commit and the promotion that shipped it.

Headline: ADR-094 unified the three
divergent storage substrates behind one compute seam, and the I/O-cost work that followed made
round-trip count — not bytes, not CPU — the term the engine measures and the operator can tune.
Full notes follow in their own change: a release-notes page must ship together
with its nav entry, or `mkdocs build --strict` fails on an unpublished page.

### Storage substrate (ADR-094)
- Durable relational spill store with read-merge, bounded memtable, segment discovery with
  order-recoverable names, purge-on-DROP, and spill tombstone persistence (TD-USUB-1).
- Native relational tables register with DataFusion, closing the capability cliff with no format
  change (TD-USUB-2); publication is additive rather than clobbering (TD-USUB-4).
- One `RecordReader` seam + Parquet `PAR1` detect arm (TD-USUB-5).
- Typed shredded columns with a residual PROPS tail — measured 41.2% smaller relational blocks
  (TD-USUB-6).
- Faithful canonical-record round trip for Parquet Layer A; Parquet had been dropping
  `valid_to_ns`, which resurrected tombstoned records (TD-USUB-11).
- Index-acquisition round trips 3 -> 0 on a **warm** read (`range_gets` 203 -> 200),
  measured and ratcheted at `cold - warm == 3` (TD-USUB-8 slices 0-2). The
  catalog-resident Layer B that would move the **cold** path is still open, and
  TD-USUB-8 is explicit that it must now be argued on cold start, cross-node
  sharing and planner-time pruning rather than on a depth number a read-side
  cache already delivers.

### I/O cost and read geometry
- Operator-configurable per-location `io_budget` ranged-GET geometry (TD-IOBUDGET-1).
- Bounded-concurrent `read_ranges` with `fetch_rounds`/`max_inflight` metrics (TD-RDSTRAT-12).
- Footer-resident pruning: self-describing footer field map, footer block stats, tag-aware
  two-level compaction layout default-ON per collection with a kill-switch (TD-FPRUNE-1).

### SQL / pgwire
- Real transaction control — `BEGIN`/`COMMIT`/`ROLLBACK` per ADR-018 P2.D (TD-076).
- SCRAM-SHA-256 authentication (TD-PGWIRE-AUTH-1).
- Identifier case-folding: unquoted folds, quoted stays case-exact (TD-OLAP-18).
- TPC-H/TPC-DS anchored accuracy ratchets (TD-182 P1).

### MLOps
- MLflow-compatible tracking, registry and artifacts over the existing substrate, with a
  tracked-S3 artifact backend behind an `ArtifactBackend` seam and a vendored UI at `/mlflow-ui`.
  Default OFF (TD-MLOPS-1..4).

### Security & governance
- Generic OIDC provider with multi-IdP portability; legacy SSO removed (TD-SSO-1).
- API-key gateway roles (TD-TENANT-1 follow-up); subject-parameterized row predicates
  (ADR-090 L2.1); trust-gated tier entitlement (TD-TENANT-3); REST request limiter,
  default-off (TD-RATE-1).
- v1 residue removed — sunset middleware deleted, Flight alias honoring removed
  (TD-V1SUNSET-1).

### SDK & spec surface
- TD-SPECRAT-1 took the generated OpenAPI surface from 45 paths at the v0.3.0 tag
  to 94, exposing the ABAC control
  plane, collections-admin, graph, time-series, rank search, CRUD and unified query to every
  generated SDK. Node SDK published to npm as `@anvailabs/proximadb-client`.

### Breaking
- **Minimum supported Python is now 3.11** (`requires-python = ">=3.11"`); the published
  wheels' abi3 tag moves `cp310` -> `cp311`, so 3.10 installs are refused by metadata rather
  than failing at import. 3.10 reached upstream end-of-life on 2026-10-01. This also fixes a
  real defect: `load_config_file()` for `*.toml` raised `ImportError: tomli is required` in
  every normal install, because `tomli` was declared only in the `dev`/`test` extras behind
  `python_version < '3.11'` markers and never in `[project].dependencies`. It now uses the
  stdlib `tomllib`.

### Deprecations
- VIPER engine deprecated (ADR-093); Hive proto arm retired (TD-CAT-8); `OltpCatalog` gated
  behind the `oltp-catalog` feature (TD-CAT-7.4).

### Known gaps
- The vector I/O-cost harness is a repo integration test, not a shipped tool (TD-VECEVAL-1).
- 30 user-facing AsciiDoc pages do not render on the docs site (TD-DOCSITE-1).
- Four breaking dependency upgrades deferred with named gates, incl. pgwire 0.21 -> 0.41
  (TD-DEPS-2).
- The published OpenAPI contract still reports `info.version: 0.2.0`; it is hardcoded rather than
  derived from the crate version.
- 75 e2e test harnesses pick their server port with a bind-read-drop TOCTOU, which turns CI red
  at random on PRs that did not cause it (TD-TESTPORT-1).
- A duplicate Helm chart under `deploy/infrastructure/helm/proximadb` cannot render (its template
  `include`s a helper defined nowhere), so the dev-deploy script and two deployment guides point
  at a broken path. Its stale version pins (chart 0.1.0, appVersion and image tags 0.2.0) were
  deliberately NOT bumped for this release
  (TD-HELMDUP-1).

## [0.3.0] - 2026-08-05

Major feature release (802 commits since 0.2.2). Headlines: a Postgres-wire query surface for
vector search, DuckDB-backed OLAP routing, relational ABAC enforcement, and delete-vector-aware
SST compaction across the storage engine.

### SQL / pgwire
- `vector_search` UDTF and the `<->` distance operator resolve over the Postgres wire protocol,
  returning `id + score + payload`, unified behind a single `unified_search_native` v2 kernel
  (TD-XMODAL-4).

### OLAP & analytics
- DuckDB-Local production routing for join/aggregate Parquet OLAP (ADR-059, default-OFF).
- Per-column HLL NDV populated at materialize for cost-based planning (TD-OLAP-2).

### Storage engine (SST / PAX / WAL / delete-vectors)
- Delete-vector-aware compaction: drops DV-deleted rows and retires input `.dv` files; merge-on-read
  on the RaBitQ cascade path (TD-DELVEC-1).
- Continued always-PAX flush/compaction and WAL hardening.

### Search & routing
- `CompiledGlobalFilter` unifies the AXIS filtered-ANN predicate path (F7).

### Security & multi-tenancy (ABAC)
- Live relational policy enforcement with canonical identity carried through relational reads.

### Observability
- Persistent-L2 cache probes plumbed into the io-trace envelope + warehouse; real operational
  counters and score-scale contract tests (TD-METRICS / TD-IOTRACE).

### Dependencies
- Bump `sandhi-core` to 0.1.6 (usage metering; one-way ProximaDB → Sandhi, ADR-060).

_See the git history (`v0.2.2..v0.3.0`) for the full 802-commit detail across the sst, storage,
abac, decomp, olap, search, catalog, observability, wal, tenant, and pgwire areas._

## [0.2.2] - 2026-07-04

### Storage
- Always-PAX flush/compaction for the SST engine (ADR-049 M1-3); PAX segment compaction; AXIS rebuild from SST on index-store loss.

### Statistics & routing
- Canonical segment-statistics contract (ADR-042) + neutral envelope (ADR-037) + freshness floor (ADR-038); ADR-050 cost-based routing recorded.

### API
- gRPC v2 GetRecord Python stubs regenerated to match the proto; two-surface SQL model (pgwire + JWT gRPC ExecuteQuery) documented — neither deprecated.

### OSS adoption
- First-run funnel fixed (docker refs -> published vjsingh1984/proximadb, :latest live multi-arch); README status reconciled to SUPPORTED_SURFACE ("the contract wins"); OPEN_CORE.md + COMPETITIVE_LANDSCAPE.adoc added.

### Hygiene
- Clippy real-bug tail + mechanical sweep; attribution/doc-authority CI gates; deterministic-commit contract.

## [0.2.0] - 2026-02-22

### 🎉 Major Release: Platform Packages

This release introduces **native platform packages** for Linux and Windows, making installation easier than ever.

### Platform Packages

#### Linux
- **RPM Packages** (Red Hat/CentOS/Fedora 8+)
  - `proximadb-0.2.0-1.el8.x86_64.rpm`
  - Systemd service integration
  - Automatic user and directory creation
  - Installation: `sudo rpm -ivh proximadb-0.2.0-1.el8.x86_64.rpm`

- **DEB Packages** (Debian/Ubuntu)
  - `proximadb_0.2.0_amd64.deb`
  - Systemd service integration
  - Configuration management
  - Installation: `sudo dpkg -i proximadb_0.2.0_amd64.deb`

#### Windows
- **MSI Installer** (Windows 10+)
  - `proximadb-0.2.0-x64.msi`
  - Installation to `C:\Program Files\ProximaDB\`
  - Start menu shortcuts
  - PATH environment variable setup
  - Installation: `msiexec /i proximadb-0.2.0-x64.msi`

### Release Infrastructure

- ✅ Automated platform package builds (RPM via fpm, DEB via fpm, MSI via WiX v4)
- ✅ Multi-platform binary support (Linux x86_64, Windows x86_64)
- ✅ Pre-release CI validation workflow
- ✅ Automated release workflow with GitHub Releases integration
- ✅ Version consistency automation across all packages
- ✅ PyPI and crates.io publishing automation

### Installation Improvements

- **Native package manager integration** for Linux distributions
- **Systemd service support** with automatic start/stop
- **Configuration file management** (/etc/proximadb/config.toml)
- **Data directory creation** (/var/lib/proximadb)
- **Log directory management** (/var/log/proximadb)

### Documentation

- Installation guides for all platforms
- Platform package documentation
- Release preparation and validation procedures
- Automated changelog generation

### Known Limitations

- **macOS packages (DMG)**: Not available in v0.2.0 due to ring crate CPU feature detection issues on CI. Planned for v0.2.1.
- **Python embedded wheels**: Disabled for v0.2.0 (clients/python-embedded is pure Python, not Rust)
- **ARM64 packages**: Planned for future releases

### Testing

- Platform package builds validated (RPM: 41s, DEB: 46s, MSI: 54s)
- Installation testing on Linux (RHEL/Ubuntu) and Windows
- Pre-release CI validation passed
- Dry-run release validation completed successfully

### Migration from v0.1.x

No breaking changes from v0.1.x. Existing configurations and data are compatible.

### Contributors

Thanks to all contributors who made this release possible!

---

## [0.1.5] - Previous Release

### Major Features

#### Unified Multi-Model Storage Architecture
Complete implementation of the 14-phase unified storage architecture plan.

#### Document Storage (Phase 1A)
- JSON document storage with WAL-backed durability
- JSON path indexing and queries (`$.path.to.field`)
- Full-text search integration with Tantivy
- Array indexing for nested document queries

#### Observability Pipeline (Phase 1B)
- High-throughput log ingestion (1M+ logs/sec target)
- 6 SIEM adapter formats: OTLP, Syslog, Fluent, CEF/LEEF, OCSF, HTTP JSON
- Time-partitioned storage with hot/warm/cold tiering
- Metric aggregation with downsampling
- Trace assembly and span relationships

#### PostgreSQL Wire Protocol (Phase 2)
- Full v3.0 protocol compatibility
- DDL support: CREATE/DROP/ALTER TABLE, INDEX, COLLECTION
- DML support: INSERT, UPDATE, DELETE with prepared statements
- Extended query protocol with Bind/Execute
- COPY protocol for bulk imports (Text, CSV, Binary, Arrow IPC)

#### Unified Query Layer (Phase 3)
- Cross-model query decomposition and execution
- Parallel execution with configurable concurrency
- 5 fusion strategies: Intersection, Union, RRF, Weighted, First-With-Filter
- Vector + Graph + Document + Observability joins

#### Multi-Tenant Isolation (Phase 6.1)
- Tenant-aware storage paths
- X-Tenant-ID header and JWT claim extraction
- Per-tenant resource isolation

#### Distributed Query Coordination (Phase 6.2)
- Shard-aware query routing
- Parallel remote execution with retry logic
- Result aggregation strategies

#### Auto-Tiering Policy Engine (Phase 6.3)
- Hot/Warm/Cold/Archive performance tiers
- Access pattern tracking with hotness scoring
- Policy DSL for age, access, and size-based rules
- Migration coordination with priority queues

#### Multi-Model Transaction Coordinator (Phase 7)
- ACID transactions across Vector, Document, Graph, Observability stores
- 5 isolation levels: ReadUncommitted to Serializable
- 2PC protocol with participant coordination
- Savepoints and nested transaction support

#### Cross-Model Joins (Phase 10)
- Hash-based join execution
- Inner, Left Outer, Semi, Anti join types
- StartNodeSpec resolution for graph integration
- Query optimization with selectivity estimation

#### SQL Parser Upgrade (Phase 10.4)
- EXISTS/NOT EXISTS subqueries
- LIKE/ILIKE operators
- BETWEEN expressions
- IS NULL/IS NOT NULL
- IN list expressions
- CROSS JOIN support

### New Components

#### Unified Port Architecture (Phase 14)
- Single port (5678) for REST, gRPC, and Arrow Flight
- Protocol multiplexing with automatic detection
- HTTP/2 support with ALPN negotiation
- Backward-compatible multi-port mode

#### Web UI Dashboard (Phase 12)
- SQL Query Editor with Monaco Editor
  - ProximaDB SQL syntax highlighting
  - Query history and sample queries
  - Results table with execution metrics
- Graph Explorer with Cytoscape.js
  - 6 layout algorithms (Force-directed, Circle, Grid, etc.)
  - Node/edge filtering and traversal control
  - PNG and JSON export
- Dark/Light theme support
- 10-tab dashboard: Overview, Collections, Query, Graph, Performance, Cache, Security, Alerts, Metrics, Diagnostics

#### Python SDK Enhancements (Phase 13)
- **Graph Analytics**: PageRank, centrality, community detection, pattern matching
- **AutoML Integration**: Engine selection, workload prediction, hyperparameter optimization
- **Observability**: Prometheus metrics, OpenTelemetry tracing, structured logging
- **Multi-Modal Queries**: Unified query builder, semantic joins, graph-vector fusion
- **Security**: OAuth2 token management, RBAC, audit logging, mTLS

### Documentation
- Storage Engine Selection Guide
- Graph Engine Selection Guide
- Unified Port Migration Guide
- Python SDK Guide

### Testing
- 3,560 unit tests passing
- Integration tests for all engines
- Python SDK tests with all 6 storage engines

### Breaking Changes
- Default port changed to unified mode (5678 for all protocols)
- PostgreSQL wire protocol moved to port 5433

## [0.1.5] - Previous Release
- Initial multi-engine vector storage
- ORION graph engine
- Basic REST and gRPC APIs
