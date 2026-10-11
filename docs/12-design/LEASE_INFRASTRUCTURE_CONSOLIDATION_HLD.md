# Lease infrastructure consolidation: design and TDD execution plan

Owner: Storage / Runtime / Connectivity. Status: implementation plan; P0 counterexample evidence complete,
P1 conditional writes, explicit head migration, and fail-closed legacy policy implemented;
local acquisition and the queue portion of P2 hardened; full backend qualification and shared
ownership-core integration still pending.
No production safety graduation is implied.
Tracker: [TD-LEASE-INFRA-1](../10-quality/td/TD-LEASE-INFRA-1-ownership-publication-consolidation.adoc).
This is a lease-specific feeder to the existing architecture index, not another system authority.

## 1. Objective and evidence boundary

Consolidate ownership, fenced publication, and ownership-conditioned progress into the existing
ProximaDB substrate. Do not create a second Python/cloud-specific lease authority, a generic
transaction service for connector scheduling, or a parallel public locking API.

Inspection baseline: ProximaDB origin/develop `b2436f234eb1e3b0598c7502dddeff90b942f3d8`;
consumer baseline AnvaiOps develop `b1956fa5f63253d7c1cef901045eaf9ae1cdd3e3`.
Refresh baselines before each implementation checkpoint. Work only in isolated task worktrees.
P1 implementation baseline: `3067592f5` (the intervening artifact-backend change
does not overlap the publication/queue files).

Measured existing evidence: AnvaiOps's 22 lease tests pass; six added deterministic negative
tests fail. P0 now reproduces queue double acquisition, read-outage overwrite, and stale manifest publication
including PUTs paused across pruning and generation takeover. Additional negative tests expose
codec compatibility and target fail-closed/input-validation gaps. See the evidence table in the TD;
not every failing test represents a separately reachable production race. Finding a method named `atomic` is not certification.

Authority: `docs/12-design/UNIFIED_RESOURCE_LEASE_LOCKING_REVIEW_2026_06_24.adoc`, architecture
index, dimensional co-design document, workspace layering rules, shared-WAL mandate, and env
registry. That approved review already requires canonical ResourceKey, exclusive-MVP locking,
storage fencing, path compatibility, and guard-owned lifecycle. This plan consolidates that work.

## 2. Existing implementation index and disposition

Paths in this table are relative to ProximaDB unless prefixed AnvaiOps.

| ID | Existing implementation | Current meaning / consumers | Proposed disposition |
|---|---|---|---|
| I01 | `proximadb-object-store::ProximaObjectStore::{put,put_if_absent,put_opts,get_with_meta}` | Existing write helpers now share native option forwarding; body+metadata GET preserves update validators | P1 head uses existing native validators; cloud/local provider qualification remains separate |
| I02 | `proximadb-iceberg-engine::manifest::ManifestCommitter::{commit,commit_fenced,read_fenced,prune_retention}` | Version-slot publication and generation header; warehouse, catalog snapshots, partition leases | Harden in place first; then move generic publication code down to the object-store layer and re-export old paths |
| I03 | `proximadb-iceberg-engine::metadata::MetadataCommitter` | Iceberg metadata publication | Retain format-specific assembly; delegate shared commit mechanics rather than retain a second algorithm |
| I04 | `src/cluster/partition_lease.rs::{ResourceKey,ResourceIdentifier,ResourceType}` | Typed resource identity and legacy-compatible paths | Keep one identity representation; relocate/re-export, do not make a ConnectorKey twin |
| I05 | `PartitionLeaseStore::{read_key,acquire_with_key,begin_writer_incarnation_with_key,release_with_key}` | Durable resource ownership; renewal/acquisition conflated; release best-effort | Canonical ownership implementation to extract and harden; add exact-token/revision conditions here |
| I06 | `PartitionLeaseManager` and `PrimaryPodRegistry` | Renewal, routing, cached ownership, reconciliation | Keep manager as lifecycle owner; registry is a derived routing cache, never commit authority |
| I07 | `DmlLockService`, `DmlLockGuard`, resource strategies | DML exclusivity, hierarchy, lifecycle, TTL policy | Compose ownership; retain DML semantics outside generic storage; validate cross-process hierarchy |
| I08 | `src/storage/write_fence.rs::StorageWriteFence`, `LeaseStorageWriteFence` | Pod-level pre-flush rejection, default-off; not exact run-token conditional commit | Evolve the existing seam into write-boundary enforcement; retire boolean-only ownership proof for strict writes |
| I09 | `proximadb-write-intent` and canonical record/WAL/flush paths | Mutation intent, generation propagation and durable effects | Carry the same authority token; sink validation and fence installation share the write serialization boundary |
| I10 | `proximadb-queue::{leases,offset_store,Consumer}` | Unique consumer incarnation; guarded inclusive contiguous ACK; NACK retry; terminal loss/cancellation; awaited release | Queue P2 slice implemented using existing local CAS; shared ownership-core extraction and sink fencing remain pending |
| I11 | `proximadb-storage-ports::ConditionalKeyStore`, local/object-store implementations | Unique identity claims and tombstones, not general ownership+checkpoint transactions | Preserve uniqueness contract; share publication/generation types where semantics match; do not use tombstone as owner-checked release |
| I12 | `proximadb-ledger::{Ledger,LedgerService,DurableLedger}` | Capacity reserve/settle, TTL reclaim, integer-valued CAS; experimental transport | Preserve accounting state machine; do not encode leases/checkpoints into integer CAS or clone its private WAL |
| I13 | `FileSystem::write_atomic`, `AtomicWriteExecutor` | Replacement/staging, not expected-owner CAS | Preserve file-writing role; never use rename+reread as mutual exclusion |
| I14 | `src/storage/transaction_coordinator.rs` | Staging plus examined simulated prepare/commit participant paths | Not a lease backend; label actual guarantees; do not extend scaffolding to hide missing atomicity |
| I15 | `proximadb-storage-transaction`, `src/transaction`, `src/storage/multimodel/transaction`, `src/services/transaction` | Several transaction families with different wiring/participants | Inventory live consumers separately; no wholesale transaction rewrite in lease work; reuse only proven commit capabilities |
| I16 | `ObjectStoreSnapshotStore`, branch refs, warehouse manifests | Durable published references versus domain-specific branch/version semantics | Reuse generic publication; branch lineage generation is not automatically an ownership epoch |
| I17 | AnvaiOps `apps/worker/state/proximadb_store.py` | REST upsert-based pseudo-CAS, permissive read fallback | Retire unsafe mutation path after cutover; thin generated-client adapter to shared ownership contract |
| I18 | AnvaiOps connector SDK `delta.py`, code-graph `_advance_watermark` | Local checkpoint helper and direct bypass writer; timestamp and commit-hash cursors | Classify local-only helper honestly; migrate live writers to one progress authority or explicitly exclude them from guarantees |
| I19 | `proximadb-runtime-common::file_lock::{FileLockManager,FileLockSet}` | Existing advisory exclusion now also implements byte-conditioned same-directory publication through `compare_exchange_file` | Unix only; cooperating writers/trusted stable lock required; exact bytes are not an ABA-proof token; crash/cancellation qualification remains open |
| I20 | `FileSystem::{compare_exchange,supports_conditional_replace}`, `QueueFs` adapter | Existing conditional replacement also checks a sibling read-set under the same directory lock before one target write | Local nonencrypted Unix implementation and transparent decorator forwarding; no multi-file write transaction; transformations unsupported until qualified |

Index maintenance rule: every touched interface must record owner, callers, scope, durability,
linearization point, error/retry semantics, format, implementation status, and its conformance tests.
Search by both symbols and behavior; this index is the bounded lease-related inventory, not a claim
that every transaction implementation in the repository has been fully audited.

## 3. Compact layering (logical layers, not five new frameworks)

```
Consumers: DML / queue / connector / catalog / warehouse
    | domain policy: lock hierarchy, contiguous offset, typed cursor, metadata schema
Existing manager + guards + authenticated transport adapters
    | exact scope + ownership token + revision + operation identity
Shared ownership state machine and conditional progress transition
    | one atomic publication boundary; no provider-specific policy
Existing storage commit infrastructure: conditional head / local serialized commit
    | existing provider clients, shared WAL where applicable, path and I/O accounting
Filesystem / object store

Data mutations -> existing WriteIntent -> canonical write admission/WAL -> publication
                                 ^ same authority token, sink-enforced barrier
```

Physical placement:

- Move provider-neutral manifest publication out of the Iceberg engine into the existing
  `proximadb-object-store` crate. Keep Iceberg payloads and table policies in Iceberg. Use re-exports
  so callers are migrated without copied implementations.
  The follow-up dependency audit found `CommitOutcome` already lives in foundation
  `proximadb-catalog-schema/src/object_store_bridge.rs`; storage-common is only its
  re-export shim. Reuse that canonical type during extraction—do not create a new
  outcome enum or introduce an object-store -> storage-common dependency. Check the
  catalog-schema dependency footprint before moving the implementation.
- Extract the ownership-only portion of `partition_lease.rs`, its identity/token types, and the
  minimal conditional-state dependency into ONE horizontal coordination package if the dependency
  audit confirms no existing cohesive home. The queue is horizontal and cannot import the storage
  or root crate. This would be an extraction replacing code, not an additional service or duplicate.
- Do not turn `proximadb-concurrent` (process-local maps) into a durable coordination service merely
  to avoid a package name. Do not place behavior in the trait-only storage-ports crate.
- Existing `proximadb-runtime-common::file_lock` already supplies local advisory exclusion.
  Its Unix implementation uses nonblocking `flock`; non-Unix acquisition currently warns and
  succeeds without locking, which is not an admissible strict backend. The inode must remain stable;
  never delete or replace the coordination file. `FileLockSet` provides ordered acquisition with
  rollback, not a distributed multi-key transaction or a globally atomic multi-root observation.
- Storage adapters implement the extracted narrow dependency; composition roots inject them.
  Standalone queue/local mode must have a certified local adapter without an upward dependency.
- Move types once, re-export, then migrate callers. No dependency-cycle exceptions.
- Existing public transport/service ownership is retained. A missing connector operation may need
  a schema addition, but not a parallel REST/SDK stack or an arbitrary remote storage/CAS endpoint.

## 4. Semantic vocabulary: small, non-interchangeable concepts

| Concept | Meaning | Must never be confused with |
|---|---|---|
| ResourceKey | Authenticated tenant/namespace + canonical domain/resource identity | Raw SQL, caller-selected cloud path, pod ID, collection display name |
| Authority incarnation | Identifies a particular authority history; changes on controlled restore/reset | Random run ID or numeric ordering across independent databases |
| Ownership generation | Monotonic succession within one resource authority | Checkpoint revision, WAL LSN, lease expiration, branch lineage |
| Owner incarnation | Distinguishes process/run instances, including reused pod names | Stable pod/host name alone |
| Revision | Opaque expected-version basis for one atomic state transition | Caller-chosen next version or wall time |
| Ownership token | Scope + authority incarnation + generation + owner incarnation | Bearer authentication; possessing a token grants no tenant permission |
| Operation identity | Idempotency key bound to scope, token, operation kind and payload digest | An unconditional instruction to repeat side effects |
| Progress | Domain-typed checkpoint plus its revision | A universally comparable string or a timestamp that proves scan completeness |
| Lease deadline | Time-based eligibility/liveness rule | Proof that a paused process has stopped or that its data writes are rejected |
| Commit receipt | Durable result and committed revision for a specific operation | HTTP success alone, an in-memory update, or an async enqueue without a documented durability contract |

Prevent cross-scope equality and accidental token substitution through typed fields and checked
construction. New counters fail closed on overflow; no saturating generation reuse. Time units are
explicit at boundaries. Do not introduce several synonymous epoch/version wrapper families.

## 5. Minimal operation semantics

These are semantic refinements of existing operations, not a requirement for a new trait per row.

| Operation | Atomic condition | State change | Failure / retry |
|---|---|---|---|
| Inspect | Read authoritative committed state | None | Distinguish absent, unavailable and corrupt; never synthesize empty ownership on errors |
| Acquire | Expected revision plus free/released/eligible predecessor | Fresh owner incarnation and strictly new generation; checkpoint preserved | Contention is not success; duplicate operation returns original receipt, not another generation |
| Renew | Exact current token and expected revision, valid renewal policy | Deadline/revision only | Never reacquire implicitly; expired/lost ownership requires explicit new acquisition |
| Checkpoint | Exact token and expected revision plus domain progress validation | Progress and receipt atomically | No stale overwrite; same operation+payload is replayable; changed payload with reused ID rejected |
| Complete | Same ownership/progress conditions as checkpoint | Final progress plus released state in one state-record commit where supported | Crash before commit changes neither; committed response loss reconciles from receipt |
| Release | Exact token and expected revision | Durable released marker preserving authority history and progress | Stale handle cannot release successor, including same pod name; cleanup may be best-effort but must not report a commit receipt |
| Resolve operation | Matching identity/digest in retained receipt state | None | Unknown remains unknown; expired receipt retention cannot become implicit success |
| Install write fence | Authenticated authority succession at the destination serialization boundary | Persist accepted authority generation before successor writes begin | Missing/foreign authority and untrusted future generation rejected |
| Commit data batch | Authorized current sink authority plus idempotency/sequence rule | WAL acceptance or publication and receipt | Older token rejected at the commit boundary, not only at a prior read |

Internal transition loop: inspect -> pure validated transition -> conditional commit. On conflict,
re-evaluate the original operation against the new state; never blindly replay a stale write.
Do not expose arbitrary caller closures or JSON predicates as a remote transaction language.

TTL contract must be stated precisely: a deadline does not physically stop an old worker. An
operation admitted while valid can complete after wall-clock expiry only according to the explicit
backend admission contract, and never bypass a newer installed fence. If strict expiration-at-commit
is promised, the authority must enforce time and ownership at that same boundary; client timestamps
plus a delayed object PUT do not provide it. Use monotonic clocks for local renewal budgets, bounded
I/O, and authority-stamped policy time. Clock jumps and process suspension are required tests.

## 6. Hardening publication and retention before reuse

Reproduced counterexample: advance manifest log, prune an old successor slot, then submit a
delayed commit based on its old parent. Publishing a new transition into a pruned slot below the
authoritative head must not succeed, even for the same generation. A legitimately committed operation
may return its response after newer commits; response-time tip equality is not required.
The paused-PUT tests also reproduce the race after validation while pruning proceeds, both
within one generation and across a newer generation's takeover.

Required invariant: there is exactly one authoritative committed head; a successful transition
updates that head from its exact expected revision. GC cannot make an old transition valid again.

Preferred bounded-history design if the current slot protocol cannot meet that invariant:

1. Preserve immutable payloads, but publish them through a non-reused, version-conditional durable
   head in the existing object-store wrapper. Upload payload before head change; failed head CAS
   leaves an unreferenced candidate, not a committed result.
2. The head includes format identity, authority incarnation, revision, generation, payload reference
   and digest. Provider ETag/version is an opaque CAS validator, not a domain generation.
3. Extend existing storage methods only where conditional replacement/versioned reads are missing;
   never emulate them using overwrite/reread or copy/delete rename.
4. GC cannot delete a referenced or in-flight candidate; protect reader pins and publication
   candidates with a protocol, not merely an assumed maximum pause. Crash-abandoned candidates need
   explicit reclamation rules. A simple grace age alone is not proof of safety.
5. Local adapter must serialize across independent processes and recover durably. A process mutex
   or atomic rename alone is insufficient. Reuse certified existing local commit/locking machinery;
   if unavailable, declare the local capability unsupported until implemented and tested.

Pinned dependency audit (`object_store` 0.13.2): in-memory storage implements version-conditional
update; its local filesystem explicitly returns NotImplemented for `PutMode::Update`. The S3
adapter's conditional update depends on configuration. P1 extends the existing wrapper with
`put_opts` returning upstream `PutResult` and errors, and `get_with_meta` returning bytes plus
metadata from one GET. Existing create/overwrite/tier helpers delegate to the same write path.
No new storage trait, revision struct or error family is introduced. Provider-native
implementations and emulator behavior require certification; package
feature names do not prove deployment capability.

The wrapper adds no retries, but native clients can retry internally. A conditional error can
follow a committed attempt with a lost response; higher layers must reconcile a durable operation
identity/digest even after Precondition/AlreadyExists. Content-derived ETags can repeat when bytes
return to an earlier value, so the head needs a non-reused logical revision/incarnation in its body.
The memory-backend ABA test does not certify content-derived ETag behavior on cloud storage.

I01 interface semantics: storage plumbing owns the wrapper, existing warehouse/catalog/lease
helpers are its consumers, and paths remain caller-relative to the store's base. The backend's
conditional PUT is the primitive's linearization point; durability is only that backend's contract.
Reads return the metadata from the same successful body GET and retain the existing read observer.
Update without a usable validator fails as invalid input; unsupported local updates fail explicitly.
The interface writes caller bytes unchanged and introduces no persisted format. Conformance lives
in `proximadb-object-store/tests/conditional_publication.rs`; ownership and GC are not certified by it.

Algorithm selection is gated by the red tests and backend capability audit. Do not silently mix
slot and head authority. Existing formats stay on their existing authority until a controlled
cutover; legacy readers must be routed to the correct format. Old writers must be drained/fenced
or lose write permission before the authority changes. Format marker alone cannot restrain old code.

### P1 experimental format refinement: self-contained current snapshot

Format creation stays behind explicit `ManifestCommitter::create_versioned` (new,
exclusively provisioned empty prefix) or `migrate_versioned(expected_tip)` (existing
source log with declared encoding). `open_versioned` only opens existing authority.
Updated serving handles detect and pin a persisted marker/head for reads, writes and
pruning; constructing a handle never provisions or migrates a log. Partition lease and
catalog callers explicitly declare the historical GenerationPrefixed codec. Plain
warehouse callers retain opaque bytes with generation zero; there is no codec guess.

The human-approved fail-closed policy disables ALL legacy retention and rejects
ownership writes to presently gapped history, even with the current parent. Legacy
parent, continuity and generation are validated against the same observed tip.
Continuity proves current occupancy only, NOT that pruning never happened: refilled
holes cannot be detected. Every old writer AND pruner, including suspended operations,
must be externally excluded before relying on this policy or migrating. Readable source
objects are preserved. Metadata can grow and affected ownership renewals can pause.

The selected refinement avoids the in-flight candidate reclamation problem:

1. A create-only `_publication.format` marker records the authority incarnation;
   `_publication.head` contains the entire current snapshot, including its generation,
   logical version and per-invocation operation ID. Neither key is pruned or replaced
   with a different authority. A missing head under a marker is corruption, not absence.
2. The version-1 envelope has magic, a SHA-256 checksum over metadata plus payload,
   and bounded bincode encoding (64 MiB body limit). Generation and opaque payload
   are separate fields, so arbitrary plain bytes cannot masquerade as a fence header.
   Unknown versions, truncation, checksum failure and trailing bytes fail closed.
3. A writer validates exact parent and nondecreasing generation from one head GET,
   archives that already-committed predecessor with create-only PUT, then conditionally
   replaces the head using the revision from that GET. The candidate's payload is
   inside the head PUT, never referenced through an independently reclaimable object.
4. History uses `_history/vN.snapshot`, not legacy slot filenames. A GC pass deletes
   only archives strictly below its observed head, retaining at least its predecessor
   and honoring age. It never writes the head. A delayed archive write may resurrect
   old history until another pass; this is not a commit and cannot move authority.
   History becomes bounded after in-flight operations quiesce, not during arbitrary
   numbers of suspended writers. No new durable reader-pin guarantee is advertised:
   historical reads racing expiration may return NotFound; an obtained head/body
   snapshot remains self-contained and valid after later publication or pruning.
5. On a conditional PUT error, reconcile the operation ID against the current or
   archived committed record. Matching receipt => committed, different receipt at the
   same version => conflict. If retention removed the evidence, return indeterminate;
   never invent success or a definite conflict. This is invocation reconciliation,
   not an external idempotency API or durable receipt retention guarantee for P2.

Cost tradeoff: current lookup is one GET in explicit versioned mode, independent of
archive count. Sequential publication adds a predecessor archival PUT and a head
conditional PUT; current payload bytes are duplicated into history on the next commit.
Legacy auto-detection adds two absent-object GET probes per public operation **now,
including default serving callers**. Private legacy helpers avoid recursive detection:
the counted-backend test observes 3 GET + 1 LIST + 1 PUT for an existing-head fenced
commit (previously 1 GET + 1 LIST + 1 PUT), and 3 GET for a fenced read (previously 1).
These negative GETs are not counted by the existing successful-body-read observer.
Serving latency and attempted-I/O accounting must be measured/addressed before
graduation. This is a correctness prototype, not a performance claim.
Large warehouse metadata and generic-layer extraction still need integration auditing.

The migration API records source prefix, encoding, expected tip, per-source checksums
and authority identity. It validates source records before writing the marker, archives
retained snapshots before publishing the head, then rechecks source inventory and tip.
Completed migration retries open the advanced head; they never reset it. Source objects
are not deleted. Partial migration intentionally prevents serving through that authority
until operator reconciliation; preserved raw source bytes are not a permission to bypass
the marker for ownership writes. No live cutover or repair CLI has been executed/built.

Old binaries cannot recognize these keys. Provisioning/migration requires exclusive
control and old-writer/pruner exclusion, not a racy LIST-as-lock promise.
The marker intentionally leaves failed provisioning closed; a marker without a head
requires explicit repair, not deletion/reinitialization. Native local object-store Update
is still unsupported; the separate existing FileSystem seam now supports byte-conditioned
Unix local replacement, not object-store validator emulation. Permissions must prohibit deletion/rollback of authority objects;
restoring an old authority requires a new incarnation and coordinated sink barrier.

## 7. Left-to-right consumer policies

- DML: exclusive mutation/schema locks first. Ancestor/descendant exclusion must be atomic within
  a shared conflict scope, not independent per-key reads across pods. Local hot locks compose under
  coarse durable ownership; global object-store CAS is not the per-row hot path. Shared reads stay
  on MVCC unless a genuine multi-holder lock contract is implemented.
- Queue: current ownership scope is consumer group, topic and partition. Tenant is message routing
  metadata, NOT an authenticated consumer or lease scope; partition consumers see all tenants
  routed there. A future authenticated tenant-bound API is required before claiming tenant isolation.
  ACK advances only the highest
  contiguous completed offset. Out-of-order ACK must not skip pending work; NACK must not silently
  checkpoint success. Ownership and offset publication use the same conditional transition.
- Connector: scope includes workspace/tenant, source and instance. Opaque source cursors remain
  opaque; timestamp cursors normalize to aware UTC and need a source-specific completed-scan rule.
  Replays must preserve source version semantics; deterministic IDs alone do not prevent stale
  overwrites. Full-sync bootstrap must be distinguished from a failed state lookup.
- Catalog/warehouse: shared fenced publication, domain-specific payloads. Preserve independent
  resource scope and tenant paths; do not serialize every collection behind one global manifest.
- Unique keys: uniqueness and MVCC tombstones remain their own state machine. Reuse commit mechanics
  without pretending `tombstone(key,generation)` means owner-conditional lease release.
- Ledger: reserve/settle are capacity accounting, not single ownership. Reuse compatible types,
  receipts and infrastructure, but preserve budget/window invariants and do not build a new WAL.

## 8. Top-to-bottom write safety

Auth/tenant resolution -> canonical resource resolution -> ownership admission -> typed write
intent -> WAL admission -> materialization/publication -> receipt -> progress commit.

Network prechecks improve errors/routing but are not the safety boundary. Install a successor fence
and accept data under it through the same destination serialization mechanism. Checking the remote
lease and then issuing an unconditional write is expressly forbidden as a proof of fencing.

Separate writer takeover from flushing previously accepted durable writes: an acknowledged batch
accepted under an older generation must remain recoverable and be materialized by the current
authorized publisher. Do not discard acknowledged WAL records merely because ownership changed.
Conversely, a delayed, never-accepted stale request must not enter the WAL after the new barrier.

ProximaDB and an external side effect do not share an atomic transaction. Prefer replayable batches,
durable receipts and destination idempotency; name uncertain outcomes. Do not claim exactly-once
delivery. Keep automatic takeover disabled for paths whose sink does not enforce the token.

## 9. Errors, lifecycle, security, restoration

- Semantic outcomes: committed/replayed, conflict, not-owner/fenced, expired, invalid scope/progress,
  unsupported capability, unavailable, corrupt, indeterminate. Translate through existing transport
  errors; never classify an outage as harmless contention or feature absence.
- Queue audit also found unconditional `lease.meta` deletion in shutdown and polling state that
  survives renewal failure. These require lifecycle/consumer tests; acquisition fixes do not certify
  exclusive processing. Persist released history instead of deleting authority.
- One owner for renewal tasks and shutdown. Explicit async completion is awaitable; Drop performs
  only best-effort cleanup and cannot claim durable release. Bound concurrency, retry budget,
  renewal jitter and cancellation. A retry uses the original idempotency identity.
- Protected resources reject missing tokens. A forged numerically higher generation is not a valid
  fence installation. Keep tenant authentication and authorization canonical; no raw path control.
- Force-release/reset/restore are privileged audited actions. Reusing a logical key does not reset
  its authority history. Restore requires a new authority incarnation and a sink barrier that
  invalidates old tokens; numeric rollback must not resurrect a previous owner.
- No fallback from a configured strict durable backend to local memory or legacy pseudo-CAS.
- Receipts and tombstones have bounded, explicit retention. After deduplication history expires,
  return a defined unknown/reconciliation outcome rather than silently re-executing old effects.

## 10. Co-design and measurable costs

Dominant costs: remote round trips for authority transitions; metadata growth/LIST pagination;
WAL/fsync latency at write admission; cross-cloud hops and failure coupling. Preserve the current
regional storage topology and existing clients. Do not add D1 or a standalone lease service.

Coarse durable ownership plus local hot-path conflict checks; per-resource heads; batch domain
progress where valid; bounded renewal work; no per-record remote authority read. Payload upload and
metadata publication remain distinct costs, explicitly traced.

Measure requests/bytes per acquisition, renewal and checkpoint, p50/p95/p99 latency under contention,
CAS retries, renewal deadline margin, outstanding/unknown operations, metadata-object growth,
recovery time and WAL acknowledgment survival. Use bounded metric labels; per-tenant attribution
goes through existing accounting/traces without leaking run IDs, tokens or raw source cursors.
Report no performance improvement before a representative end-to-end trace exists.

## 11. Robust TDD matrix

Every implementation runs the same behavioral suite; a mock-only pass does not certify a backend.

| Gate | Required adversarial cases |
|---|---|
| Atomic commit | Two independent clients from same revision; duplicate request; conflicting digest; response lost after commit; payload stored but head update lost; corrupt/missing head/payload |
| Retention | Delayed old parent after prune; pause between check and publish; GC versus active reader/candidate; authority deletion/recreation; restoration to older snapshot; generation overflow |
| Ownership | Acquire/acquire; renew/takeover; release/takeover; same pod/new process; reused run ID; expired renewal; clock skew/jumps; paused process beyond TTL; unknown release outcome |
| Progress | Stale checkpoint; checkpoint/release race; crash after accepted data before progress; timestamp regression; timezone normalization; opaque cursors; queue ACK gaps/NACK; tenant/group collisions |
| Fencing | Old request queued before takeover but admitted after barrier; missing/forged/cross-resource token; sink restart after barrier; acknowledged old WAL replay; mixed-generation batches; direct write bypass |
| Hierarchy | Parent/child conflict across separate processes; unrelated siblings; deterministic acquisition order; cancellation; no advertised shared locks without durable reader state |
| Lifecycle | Renewal failure; cancellation during I/O; clean shutdown; no unjoined tasks; no unsafe automatic takeover after uncertain effects |
| Compatibility | Legacy/new readers; old writer excluded at cutover; default-off behavior; checkpoint migration; codec unknown-version rejection; generated SDK drift and enabled protocol parity |

Test scheduling must not presume the unsafe algorithm remains: a future local lock may exclude
A while B is paused in a read. Adapt the queue schedule to the selected serialization seam and
assert ownership/effects; do not treat a harness deadlock as proof of unsafe acquisition. Preserve
positive tests for first acquisition, expired takeover, renewal and separate groups. Fail-closed
read handling must distinguish typed absence from I/O errors; QueueFs currently erases that distinction.
Unmarked historical eight-byte generation headers cannot be universally distinguished from
arbitrary plain bytes; do not implement compatibility with a JSON-prefix heuristic.

Use deterministic barriers and injected clocks for algorithm races, stateful property tests against
a small reference model, and real separate processes for local durability/exclusion. Run provider
adapters against local emulators as required CI lanes without automatic skips; separately qualify
provider-native behavior in a bounded authorized cloud campaign. No always-on test infrastructure.

## 12. Delivery sequence and definition of done

### P0 — index, semantics, and red tests

Create a task worktree off current develop. File one ADR/HLD and a bounded TD following the existing
filing convention (claim IDs only after refreshing develop). Link the index from the architecture
index. Add red Rust tests for manifest retention and queue double acquisition; retain the six
AnvaiOps negative cases as required behavior. Record all callers and capability gaps.

Exit: reproduced counterexamples and agreed invariants; no new production authority.

### P1 — harden and consolidate the existing commit primitive

Fix atomic publication/retention with the selected backend protocol, durability tests and existing
warehouse/catalog consumer regression tests. Move generic mechanics with re-exports, not copying.
Do not combine this with unrelated multimodel transaction rewrites.

Exit: commit acknowledgment, stale revision and GC invariants pass on each claimed backend; no new
claim of connector safety yet.

### P2 — consolidate ownership, queue progress and lifecycle

Extract existing ownership core, migrate PartitionLeaseStore and queue leases, exact-token mutation
semantics, checkpoint receipts, lifecycle ownership and explicit capability admission. Retire queue
rename/reread and same-owner-name authority shortcuts. Preserve legacy identities/paths.

Exit: shared core has real consumers, removed duplicate algorithms, queue restart/ACK tests pass,
and no upward layering edges. Claims remain restricted by destination fencing.

#### P2 queue implementation checkpoint (partial phase, not shared-core completion)

Reuse search rechecked the filesystem atomic writer, runtime file locks, and transaction families.
None provided ownership-conditioned sibling-file progress publication. The existing conditional
replacement method now accepts a read-only sibling guard slice; it is not a new transaction manager
or another public lease authority. All guards and the target are compared under the existing
directory lock, followed by exactly one file-sync/rename/directory-sync target replacement.
Acquisition, renewal, release and ACK cooperate with this same primitive. Queue `lease.meta` and
`offset.meta` JSON layouts are unchanged; old noncooperating writers must still be excluded.

Each Consumer has a UUID incarnation (clones share it). Renewal loss, cancellation during an
operation, or an indeterminate ACK closes that Consumer permanently. Graceful consumer shutdown
signals all renewers and awaits conditional expired-tombstone release; cancelling shutdown retains
its join handles for a subsequent awaited shutdown. Client shutdown closes consumer admission and
awaits live registered consumers. Drop is only best-effort; expiry is still necessary.

ACK publishes the inclusive contiguous completed prefix, guarded by the exact admitted lease bytes
and prior offset bytes. Same-holder renewal conflicts reread and revalidate, with a bounded retry
budget. Prepublication progress-read failure retains pending deliveries for retry; a conditional
publication error is explicitly indeterminate, not permission to NACK a possibly committed message.
New consumers recover from durable progress; a post-publication I/O error does not certify durability.
NACK schedules in-memory retry without advancing progress or claiming a durable DLQ write.
Lease-only groups participate with unknown progress, blocking reaping and recovery skipping.
Corrupt/mismatched group offsets and failed enumeration propagate instead of disappearing.

Compatibility constraints are explicit: existing MessageId lacks topic, so one Consumer may own
only one topic for its lifetime (multiple partitions remain supported). Separate Consumers avoid
late duplicate ACK ambiguity without changing the wire ID. Groups preserve exact ASCII
letters/digits/dash/dot/underscore, at most 255 bytes, with no edge dots/underscores; invalid historical
names require an operator-controlled migration, never silent sanitization or case aliasing.
Mixed-case historical names such as `podA` remain valid; directory enumeration rejects a
conflicting `poda` spelling before acquisition. Topic names must also be exact path components,
validated before any producer/subscriber path creation. No absolute paths or traversal are allowed.

The production executable treats unexpected drainer termination as a critical failure:
it observes the existing task handle at one-second intervals, awaits normal database shutdown,
then exits nonzero for the deployment supervisor to restart. It does not retry uncertain
consumers in place or add a second supervisor service. The predicate is independent of HTTP
enablement, so gRPC-only deployments are not mistaken for failed drainers. Embedded owners
can inspect the same `drainer_has_stopped()` predicate and must still await shutdown themselves.
This is bounded process-level detection, not an immediate per-request admission barrier; work
accepted during detection/shutdown remains subject to the existing durable queue contract.

Database shutdown retains its bounded, retryable drainer-stop contract. The executable
must not interpret that incomplete result as process shutdown: it retries while the queue
remains attached, retaining the same drainer/storage and the `Stopping` runtime record.
The queue is removed before subsequent shutdown failures, so completed errors propagate
to a failing process exit without retry. This classifier depends on that database ordering
and is documented at the call site. The private server sequencing function reuses the
existing queue accessor and runtime-state writer; no public interface or test env gate is
added. A permanently stuck effect keeps graceful shutdown pending; externally forced
termination remains a crash/recovery operation, not proof of completed shutdown.

Upgrade prerequisite: historical unversioned `offset.meta` may contain the old maximum-ACK
watermark despite earlier gaps. Its JSON cannot prove contiguous completion or reconstruct lost
work. Operators must audit/replay from an independently trusted checkpoint (including archive or
upstream source when local history was reaped) before relying on new contiguous-ACK guarantees.
Do not reset or reinterpret legacy progress automatically. This patch prevents new ACK gaps;
it does not repair historical loss or certify uninterrupted mixed-writer upgrades.

Factory-adapter upgrade admission also checks the historical duplicated-root location before
creating root-qualified directories, including when a queue is a descendant of its adapter root.
For `file:///var/lib/q`, the old adapter wrote
under `/var/lib/q/var/lib/q`; the corrected mapping uses `/var/lib/q`. A nonempty legacy location
blocks startup even when canonical data also exists. An empty or missing legacy directory is safe
to admit; inspection errors other than absence fail closed. Roots whose old and new mappings are
identical (filesystem root or explicitly scheme-qualified dot-only roots) need no layout cutover.
This is a private check at the existing directory-creation seam, not another filesystem capability
or format. The probe is scoped to the requested directory: another descendant's legacy history
does not prevent a clean descendant queue from opening. Explicit root-relative operations retain
their old mapping and need no probe.
Bare relative adapter roots are refused before filesystem access: the old backend may have
anchored them under its configured `root_dir`, while adding `file://` bypasses that anchor.
Locate and reconcile that history using the original backend configuration before selecting an
explicit URL. Merely adding a scheme is not a migration. Bare absolute paths and explicitly
scheme-qualified relative roots (including `file://./`) remain supported, as do relative operation
paths under a configured root.
Local LIST results are confined by path components, not literal URL spelling: equivalent `./`
components are normalized, while parent traversal, foreign roots and non-child entries remain
errors. Explicit relative file URLs retain working-directory coordinates even with a custom
backend root directory; bare relative roots are not silently converted into that interpretation.
After validating confinement and direct-child membership, LIST returns each backend child name
under the caller's original parent spelling, preserving `./` and exact filename case for identity checks.

Operators must stop all queue writers/readers/reapers, preserve backups of both layouts, and
explicitly reconcile legacy topics, segments, leases and progress before retrying. Never delete
legacy state or merge colliding segment names blindly; when both histories exist, use an
independently trusted checkpoint/source to resolve them. Even a legitimate canonical subtree at
the possible legacy location is ambiguous and requires explicit operator resolution. No automatic
move, progress reset, concurrent-old-writer fencing, or live migration is provided. The check adds
an inspection at each root-qualified directory creation whose mapping changed (queue/partition
creation and new topic/group directories); steady-state ACK, renewal, and append gain no probe.
No performance uplift is claimed.

Scope limits: expiry is eligibility at authoritative admission, not a clock predicate checked at
rename; this is not fencing for external effects. New-group admission versus retention, dropping the
last consumer without awaited shutdown, network filesystems, restore/ABA across authority histories,
power loss, and provider-native durability remain separate qualification/design concerns. A full
shared ownership core, typed progress receipts and connector/sink cutover have not been delivered.
Additional authority reads/guard comparisons trade I/O for correctness; no throughput/IOPS uplift
is claimed without an end-to-end trace. Cancellation during subscription between durable acquisition
and renewer registration can retain an unusable lease until its configured TTL; shutdown cannot
await an unregistered task. This availability bound is explicit, not prompt-release certification.

### P3 — enforce the existing write-context/fence seam

Persist authority barriers and validate exact token at canonical write admission/publication.
Preserve accepted-WAL replay. Exercise actual server binary and every enabled write protocol plus
internal writers; no protocol-only locks, no missing-token bypass on protected resources.

Exit: paused old writer cannot commit after successor barrier, including across restart.

### P4 — AnvaiOps cutover and sibling contract alignment

Use the existing authenticated v2/spec-generated transport with the minimal necessary operation
extension. Migrate state once with old writers drained, remove unsafe pseudo-CAS mutation methods,
wire typed progress, validate durable batch acknowledgments, remove live direct-checkpoint bypasses.
Then certify one real consumer journey, including retry, crash, edit/delete and recovery. Broader
Victor/AgentBrowser/Invest changes require their own consumer evidence, not opportunistic rewrites.

Exit: complete connector negative suite green, no all-skipped conformance, explicit deployed
backend/version contract. Only then consider automatic takeover enablement.

Each checkpoint is a coherent develop-targeted PR, locally validated before push. Rebase onto
current develop, audit affected feature lanes and server binary, obtain a fresh zero-context
adversarial review, address findings, and require exact-head green CI. Admin review bypass still
requires explicit human authorization for that PR. No CI/merge claim is made by this planning work.

## 13. Anti-proliferation acceptance criteria

1. One canonical ResourceKey and ownership-token family; old imports re-export it.
2. One ownership transition implementation for resource and queue/connector ownership.
3. One conditional publication algorithm per certified backend; Iceberg and catalog are consumers.
4. Existing uniqueness, ledger and transaction contracts retain genuinely different semantics.
5. No new handwritten Python lease algorithm, independent cloud lease service, duplicated WAL,
   parallel generated-client stack, or universal transaction manager.
6. Every extraction lists deleted implementations and migrated callers in the PR. Temporary aliases
   carry no behavior. Tests fail if strict mode composes an uncertified fallback.
7. One architecture index owns the interface map and one conformance suite owns each invariant.

## 14. Decisions deliberately gated by evidence

- Exact physical extraction location follows the dependency graph; one cohesive horizontal package
  is permitted only if necessary to remove the current root/storage coupling, not merely for naming.
- The versioned-head mechanism is the preferred fix, but its implementation cannot be finalized
  without the red retention/GC cases and actual backend conditional-operation capabilities.
- Public operation additions require tracing the existing service schema and auth seam; absence of
  a current connector endpoint does not justify inventing another protocol or reusing an unrelated
  ledger integer field.
- No distributed cross-resource transaction or strict multi-reader lease is inferred from single-key
  atomicity. Those remain separately designed capabilities, not hidden future promises.
