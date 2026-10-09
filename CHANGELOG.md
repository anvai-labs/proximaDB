# Changelog

All notable changes to ProximaDB will be documented in this file.

## [0.4.0] - unreleased

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
Full notes land with the docs-site PR (they must ship together with their nav
entry, or the strict build fails on an unpublished page).

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
