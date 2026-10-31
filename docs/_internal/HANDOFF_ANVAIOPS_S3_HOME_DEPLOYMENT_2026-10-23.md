# HANDOFF: anvaiops → proximaDB — S3-backed home deployment requirements (2026-10-23)

> Filed from the **anvaiops** repo after its ADR-0056 realignment (#507,
> merged 2026-10-23). This document is the requirement source of truth for
> the gaps below; anvaiops will cross-link your TDs by the R-numbers.
> Audience: the engine session. Author context: vijay (owner, both repos).

## Context (what anvaiops decided, and why you're receiving this)

anvaiops filed **ADR-0056**: Cloudflare and Entra are **out**. The whole
product self-hosts on a home server (ds3) behind an OVH VPS acting as a
dumb TLS edge (SSH reverse tunnels); Kanidm is the only IdP.
**Decision D2a (owner-directed, same day): ProximaDB is the end-state
system of record — one engine on ds3 serving the data plane AND the
account directory, with OVH S3 (S3-compatible object storage) as the
object-store backing.** SQLite is only the transitional D1-import target.

We verified the current `origin/develop` before deciding. Facts your
session can take as given (please don't re-derive):

- `s3://` **is wired end-to-end**: `AwsS3FileSystem` (feature `aws`,
  canonical `object_store` crate) behind `FilesystemFactory`
  (`src/storage/persistence/filesystem/aws_s3.rs`,
  `mod.rs:440-447,495+`); S3-compatible endpoints via
  `PROXIMADB_S3_ENDPOINT` + `PROXIMADB_S3_FORCE_PATH_STYLE` + static
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` (MinIO/LocalStack/R2
  explicitly named).
- The shipped image builds `--features cloud-full`
  (`deploy/docker/Dockerfile:30,43,79`) → `aws` included.
- A real **three-crash-cycle object-store durability proof** exists:
  `apps/proximadb-server/tests/object_store_restart_recovery.rs` +
  `scripts/prove_object_store_durability.sh`, driven by
  `PROXIMADB_OBJECT_STORE_URL=s3://…` — but it is credentialed-test-only
  (ignored by default), and **has never run against a non-AWS endpoint**.
- Iceberg **MATERIALIZE** (P2.4, merged): bounded-memory Parquet
  publication + create-only Avro manifest commits to object storage
  (`src/services/dml/materialize_stream.rs`, slices #2149).
- The **hot WAL stays on a local reattachable volume** (ADR-095
  writer-epoch fencing); sealed-segment offload is open debt.

## Gaps → requirements (anvaiops priority order)

### R1 [P0 — blocks the cutover drill] S3-compatible endpoint qualification (OVH S3)

The code path exists but nothing has ever exercised a non-AWS endpoint,
and there is **no MinIO/s3-compatible CI coverage** (`aws_s3.rs` header:
"not built by default CI").

1. Add a CI job running `object_store_restart_recovery.rs` against
   **MinIO** (service container; the env matrix is the three vars in the
   box below).
2. Fix `with_allow_http(true)` coupling if a **HTTPS** custom endpoint is
   rejected (`aws_s3.rs:118` — it assumes MinIO-style http; OVH S3 is
   https).
3. Confirm the SigV4 **region string** OVH accepts (OVH regions are e.g.
   `us-east-va` / `gra`) and document it.
4. One recorded manual drill of `prove_object_store_durability.sh`
   against **OVH S3** (`s3://<bucket>`, endpoint
   `https://s3.<region>.io.cloud.ovh.net`, path-style).

**Acceptance:** 3-cycle proof green on MinIO in CI + the OVH drill
recorded.

### R2 [P1] WAL sealed-segment offload to object storage

Your `TECHNICAL_DEBT.adoc:190` (TD_116-family;
`WalEntryStatus::Archived` never set). Why it matters here: the WAL is
the **last non-S3 durable state** in the home topology — a ds3 disk loss
currently loses WAL-bounded writes even with everything else on S3.
Note TD-066 already made segment deletion object-store-safe, so the
archive writer has a precedent.

**Requirements:** copy `Flushed` segments to the object store, set
`Archived`, read-back, `scan_records_at_lsn`, `SELECT … AS OF`.
**Acceptance:** variant of the 3-cycle proof where restart happens on an
**empty local disk with only the object store** and recovers to the last
archived LSN. (Unblocks your TD-117.)

### R3 [P1] pgwire/OLTP record store → object-store durability

TD-168 residual #3: "`DirectWalTableRecordStore` is WAL + RAM-memtable
only — it never flushes to object storage"; "Open: catalog prefix
registration (object-store placement)". Why it matters: anvaiops
deliberately put the account directory on the **v2 record-KV surface
instead of pgwire SQL tables** *because of this gap* (ADR-0056 D2a).
When closed, the directory and GST relational data move to real SQL
tables.

**Requirements:** Layer A segment flush to object store + catalog
registration, crash-safe. **Acceptance:** a pgwire table
(CREATE/INSERT/DELETE) survives the 3-cycle proof with empty local
data_dirs.

### R4 [P1] asyncpg compatibility = officially supported pgwire subset

anvaiops notebook kernels and the SQL editor connect via **asyncpg**
(extended query protocol, prepared statements, `executemany`); TD-PGEXT-1
Describe just landed (#2165).

**Requirements:** an asyncpg-driven e2e test in your suite (Python,
asyncpg current stable) covering connect/auth (`user`/`password`/
`database` labels), prepared statements, `fetch`/`execute`, and error
propagation; or an explicit SUPPORTED_SURFACE.adoc statement of the
asyncpg subset that is covered.
**Acceptance:** `asyncpg.connect(); conn.fetch("SELECT 1")` + prepared +
`executemany` green in CI.

### R5 [P2] Home-server footprint profile

The engine will run on ds3 beside Kanidm + FastAPI + a Node edge
service; the RAM envelope is unknown (anvaiops tracks this as TD-N1 and
gates engine activation on it).

**Requirements:** a documented low-memory profile (bounded
memtable/caches/spill knobs), measured steady-state RSS at a small
workload (~1-2 GB target), and the knob list published.
**Acceptance:** a `docs/` deployment profile with measured numbers.

### R6 [P2] MATERIALIZE reachable over REST (not only DataFusion SQL)

anvaiops GST/analytics and the governed-MCP surface want to trigger
Iceberg publication programmatically.

**Requirements:** REST trigger for `ALTER TABLE … MATERIALIZE`-equivalent,
progress/status, manifest listing; output table readable by **DuckDB
1.x** via the Iceberg extension (verify in CI).
**Acceptance:** a REST-triggered materialized table queried successfully
from DuckDB.

### R7 [P2, verify-may-already-exist] Runbook-grade cold backup

`apps/proximadb-server/src/cold_backup/mod.rs` references S3. If it
already snapshots catalog+WAL+segments restorably: write the runbook +
one recorded restore drill, close. If not: requirement =
`proximadb cold-backup`/restore covering the object-store topology.

### R8 [P2, docs-only] Supported trust-auth posture for single-box deployment

anvaiops runs the engine with `[security] enabled=false`, loopback-bound,
behind an SSH-tunnel edge with auth at the boundary (anvaiops validates
JWTs before any engine call). **Requirement:** a SUPPORTED_SURFACE.adoc
statement naming this posture as supported (engine = internal-only), so
it's a decision, not an accident.

## Non-asks (explicitly fine as-is)

- **Iceberg as the live OLTP format** — not requested; MATERIALIZE/
  lakehouse is the right layering as built.
- **v2 REST record-KV durability** — works today; it's our transitional
  directory path.
- **Embedded mode** — not needed (server mode on ds3; anvaiops ADR-0019
  D2 unchanged).
- **Multi-region/replication** — single-box home deployment; not now.

## The exact env anvaiops will run (for your test matrix)

```
PROXIMADB_OBJECT_STORE_URL=s3://<bucket>
PROXIMADB_S3_ENDPOINT=https://s3.<region>.io.cloud.ovh.net
PROXIMADB_S3_FORCE_PATH_STYLE=true
AWS_ACCESS_KEY_ID=<ovh s3 key>   AWS_SECRET_ACCESS_KEY=<ovh s3 secret>
AWS_REGION=<ovh region — confirm the SigV4 value; see R1.3>
```

WAL stays on local `/data` (reattachable volume, ADR-095) until R2
lands.

## Status on the anvaiops side

ADR-0056 + topology doc + TD-C2/TD-N1 are merged (#507); P1 (D1 export)
and P2 (edge re-platform) are next and are **not blocked** by any of the
above — R1 gates the *cutover drill* only. Reference the same numbering
(R1-R8) when you file TDs so both repos can cross-link.
