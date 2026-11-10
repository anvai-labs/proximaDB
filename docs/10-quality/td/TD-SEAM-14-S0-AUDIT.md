# TD-SEAM-14 S0 Audit — Baseline and Counterexample Plan

**Status:** In progress (2026-11-09)
**Worktree:** `impl/td-seam-14-s0-baseline`

---

## 1. Caller/Capability Audit

### 1.1 spill_placement.rs — Ownership and Claim

**Current model:**
- `OwnerMarker { instance, epoch, nonce, written_by }` at `{url}/_proximadb_spill_owner.json`
- `PlacementMarker { instance, epoch, pending, placement }` at `{data_dir}/pgwire/SPILL_PLACEMENT`
- `write_owner()` uses `write_if_absent` for create-only, unconditional overwrite otherwise
- `claim()` increments epoch: `epoch = max(state.epoch, owner.epoch) + 1`
- Checks: instance match, epoch ordering, nested prefixes, WAL truncation

**Gaps identified:**
1. **No runtime fence** — ownership checked at boot only, not before destructive ops
2. **Unconditional overwrite** — after first create-only, subsequent boots overwrite owner marker
3. **Instance sharing** — copies share instance ID, racing boots can choose same next epoch
4. **No CAS on owner marker** — two copies booting simultaneously can both pass the instance check
5. **Epoch is monotonic but not unique** — two copies at same epoch can both increment to epoch+1

### 1.2 record_store.rs — Factory and Admission

**Current model:**
- `open_for_server_at()` resolves spill target (local or object-store)
- Checks if spill is configured and if WAL was truncated with existing segments
- Refuses if spill is off but segments exist and WAL was truncated (silent subset risk)

**Gaps identified:**
1. **No incarnation tracking** — DROP/recreate doesn't distinguish generations
2. **No manifest publication** — segments are discovered by listing, not by manifest
3. **No durable witness** — no record of what was published before destructive ops

### 1.3 record_spill.rs — Reads, Publication, Purge

**Current model:**
- Reads segments by listing `{url}/{tenant}/{collection}/`
- Writes segments with `write_if_absent` (idempotent)
- Purge via `wal_truncator` after checkpoint

**Gaps identified:**
1. **Purge is not fenced** — no ownership check before deleting segments
2. **No manifest** — readers discover segments by listing, can see partial writes
3. **No rollback detection** — restored bucket passes boot check

### 1.4 wal_truncator.rs — Truncation Floor

**Current model:**
- Truncates WAL after checkpoint
- Purges segments below truncation floor

**Gaps identified:**
1. **Purge is not fenced** — no ownership check before deleting segments
2. **No coordination with manifest** — can purge segments readers still need

### 1.5 Shared Primitives

**FileSystem trait:**
- `write_if_absent()` — CAS for create-only
- `write()` — unconditional overwrite
- `read()` — read bytes
- `list()` — list prefix
- `delete()` — delete path

**Gaps identified:**
1. **No CAS on overwrite** — can't atomically check-and-update owner marker
2. **No fencing on delete** — can't check ownership before delete
3. **No durable witness** — no way to record "I'm about to delete X"

---

## 2. Counterexample Test Scenarios

### 2.1 Second-Writer Purge Race (TD-SEAM-14 §1.1)

**Scenario:**
1. Server A boots, claims prefix at epoch 1
2. Server A writes segments S1, S2, S3
3. Server B boots with copied data directory (same instance, same epoch)
4. Server B increments to epoch 2, overwrites owner marker
5. Server A truncates WAL, purges S1
6. Server B reads S1 — **GAP: no fence, purge succeeds**

**Expected (after fix):**
- Server B's claim should be refused (instance match but epoch collision)
- OR Server A's purge should be fenced (check owner marker before delete)

**Test location:** `src/services/spill_placement.rs` test module

### 2.2 Rollback Undetected (TD-SEAM-14 §1.2)

**Scenario:**
1. Server boots at epoch 5, writes segments S1-S5
2. Bucket is restored to mid-run state (epoch 3, segments S1-S3)
3. Server reboots, sees owner marker at epoch 3
4. Server increments to epoch 4, boots successfully
5. Server serves S1-S3 — **GAP: rollback undetected, missing S4-S5**

**Expected (after fix):**
- Boot should detect rollback (epoch went backwards)
- OR segments should be epoch-stamped (S4, S5 have epoch 5, not found after restore)

**Test location:** `src/services/spill_placement.rs` test module

### 2.3 Racing Boot Epoch Collision (TD-SEAM-14 §1.3)

**Scenario:**
1. Two copies of data directory exist (A and B)
2. Both boot simultaneously at epoch 5
3. Both read owner marker at epoch 5
4. Both increment to epoch 6
5. Both write owner marker with epoch 6 — **GAP: no CAS, both succeed**
6. Both claim ownership, both can purge

**Expected (after fix):**
- Owner marker write should be CAS (check-and-set)
- Second writer's CAS should fail, boot refused

**Test location:** `src/services/spill_placement.rs` test module

### 2.4 Manifest Publication Gap (TD-SEAM-14 §2.2)

**Scenario:**
1. Server writes segments S1, S2, S3
2. Server publishes manifest M1 = {S1, S2, S3}
3. Server truncates WAL, purges S1
4. Server writes S4, publishes manifest M2 = {S2, S3, S4}
5. Crash before M2 is durable
6. Reboot reads M1, serves S1 — **GAP: S1 was purged, serves missing segment**

**Expected (after fix):**
- Manifest publication should be owner-conditioned (CAS on owner token)
- Purge should wait for manifest update to be durable
- OR manifest should be a durable witness (write-ahead log of publications)

**Test location:** `src/services/record_spill.rs` test module

### 2.5 DROP/Recreate Incarnation Gap (TD-SEAM-14 §3.2)

**Scenario:**
1. Collection C1 at incarnation 1, segments S1-S3
2. DROP C1, segments remain in object store
3. CREATE C1 (new incarnation 2)
4. Replay sees old segments S1-S3 — **GAP: no incarnation tracking, serves stale data**

**Expected (after fix):**
- DROP should record incarnation boundary
- Replay should skip segments from prior incarnations
- OR segments should be incarnation-stamped

**Test location:** `src/services/record_store.rs` test module

---

## 3. Implementation Plan

### 3.1 S0 Deliverables

1. **Audit document** (this file) — caller/capability audit, gap analysis
2. **Counterexample tests** — deterministic tests for each scenario above
3. **Shared primitive inventory** — list of FileSystem methods, ownership primitives, etc.

### 3.2 S1-S5 Overview (deferred)

- **S1:** Exclusive claim with CAS, exact tokens, release/fenced states
- **S2:** Immutable candidates, owner-conditioned manifest, durable witness
- **S3:** Manifest reads, logical DROP/incarnation, replay idempotence
- **S4:** Format migration, recovery/maintenance operations
- **S5:** Server integration, provider qualification, adversarial review

---

## 4. Open Questions

1. **CAS primitive:** Does the FileSystem trait support CAS on overwrite? If not, can we add it?
2. **Manifest format:** What should the manifest contain? Segment list + incarnation + owner token?
3. **Incarnation tracking:** Where should incarnation be stored? In manifest? In segment names? In separate marker?
4. **Durable witness:** Is a write-ahead log of publications the right approach? Or a separate witness marker?
5. **Provider qualification:** Which object-store providers need testing? S3, GCS, Azure, local filesystem?

---

## 5. Next Steps

1. **Write counterexample tests** for scenarios 2.1-2.5
2. **Inventory shared primitives** — FileSystem trait methods, ownership primitives
3. **Document current behavior** — what works, what doesn't, what's undefined
4. **Plan S1 implementation** — exclusive claim with CAS

---

**End of S0 audit document.**
