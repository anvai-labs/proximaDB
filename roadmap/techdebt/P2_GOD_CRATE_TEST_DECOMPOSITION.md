# P2 — Dissolve the god-crate test binary (decomposition roadmap)

Part of the build/workspace/isolation plan (`golden-stirring-meadow`). This is
the actionable scoping for P2; it feeds the broader
`docs/history/planning/WORKSPACE_REFACTOR_PLAN_2026_05_07.adoc`.

## The finding (why P2 is not a quick "move the tests")

The "8400-test binary" is the **root `proximadb` crate's lib test binary** —
**9,538 inline `#[cfg(test)]` tests across 1,217 `src/` files**, all compiled
into *one* binary. That single binary is what OOMs the 16 GB CI runner and
forces `CARGO_BUILD_JOBS=1` + `--test-threads=2`.

Two facts make this a **code-decomposition** problem, not a test-move:

1. The tests are **inline `#[cfg(test)]`** next to the code they test (they use
   private internals), so they can only leave the root crate when the *code*
   does.
2. The 127 `tests/*.rs` integration files are already separate binaries, and
   **113 of 127 `use proximadb::`** (the monolith) — only ~3 target a single
   sub-crate's public API. Moving those 3 wouldn't shrink the lib binary.

The build profiles are **already optimized** (`debug = "line-tables-only"`,
`split-debuginfo = "unpacked"`, dev/test matched) — there is no remaining
build-flag mitigation. The OOM is inherent to 9,538 tests in one crate.

**Conclusion:** P2 == continue decomposing the root monolith into the existing
crate groups. The test binary shrinks in proportion to the `src/` modules
moved out. This is a deliberate, multi-PR initiative requiring the full
build/test loop — **not** something to batch blindly.

## Extraction units (root `src/` module → tests, ranked by payoff)

| `src/` module | files | inline tests | likely target crate(s) |
|---|---:|---:|---|
| `storage` | 666 | 3,024 | `crates/storage/*` (block-format, object-store, …) |
| `query` | 155 | 1,201 | `crates/query/*` (relational-*, multimodel-*) |
| `network` | 103 | 881 | `crates/platform/proximadb-api` |
| `core` | 106 | 543 | `crates/foundation/*` |
| `index` | 78 | 440 | `crates/modalities/proximadb-vector` |
| `observability` | 47 | 397 | `crates/modalities/proximadb-observability` |
| `cdc` | 38 | 305 | new `crates/integration/proximadb-cdc` |
| `cluster` | 18 | 222 | `crates/platform/proximadb-runtime` (`cluster` feat) |
| `catalog` | 18 | 201 | `crates/control/proximadb-catalog` |
| `auth`/`security` | 29 | 226 | `crates/horizontal/proximadb-security` |

Target sub-crates **already exist** for most of these (modalities/*, query/*,
storage/*, control/*) — the work is moving modules into them, not greenfield.

## Sequencing (tractability-first, not size-first)

Pick **low-coupling, high-test-count, existing-target** modules first so each
PR is verifiable and meaningfully shrinks the binary:

1. **`src/observability` → `proximadb-observability`** (397 tests; cohesive,
   modality-owned; low coupling to storage/query). Good pilot.
2. **`src/index` → `proximadb-vector`** (440 tests; ANN/index already partly in
   the vector crate).
3. **`src/cluster` → `proximadb-runtime`** (222 tests; behind the `cluster` feature).
4. **`src/catalog` → `proximadb-catalog`** (201 tests).
5. Then the large ones (`storage`, `query`, `network`) — split per submodule,
   not as one PR.

Also, as a trivial warm-up that establishes the per-crate **integration** test
pattern (separate from the lib-binary work): move the ~3 `tests/*.rs` that
already target a single sub-crate (e.g. `test_record_store_route.rs` →
`proximadb-catalog/tests/`).

## Per-extraction recipe

1. Move the module dir `src/<mod>/` → the target crate's `src/` (carry the
   inline tests with it).
2. Resolve the layering: the target crate may only depend per
   `scripts/check_workspace_boundaries.py` — push shared types down to
   `foundation/` rather than creating an upward edge.
3. Re-export from the root crate (`pub use <crate>::<mod>;`) so downstream paths
   keep working during the transition.
4. **Verify** (the loop that makes this safe, and why it can't be blind):
   - `cargo check -p <target-crate>` and `-p proximadb`
   - `cargo nextest run -p <target-crate>` (tests now run here, in their own binary)
   - `python scripts/check_workspace_boundaries.py` (layering still green)
   - confirm the root lib test count dropped by ~the module's test count.
5. Wire CI: the component already has a `paths-filter` output + integration job;
   point it at the new crate.

## Expected outcome

- Root lib test binary shrinks proportionally → no OOM → drop the
  `CARGO_BUILD_JOBS=1` / `--test-threads=2` crutch → faster CI.
- `cargo nextest -p <crate>` (and `worktree.sh test`, P1-D) runs a modality's
  tests in seconds, in its own process — agents test only what they touch.
- CI change-detection maps 1:1 to crate test targets.

## Recommendation

Run P2 as a **dedicated, sequenced initiative** (one module per PR, each
verified by the recipe above), not a single sweep. Start with the
`observability` pilot to prove the recipe end-to-end, measure the binary/OOM
delta, then proceed down the list. Each PR uses the worktree discipline
(P0–P1) and the affected-only `worktree.sh test` to keep the loop fast.
