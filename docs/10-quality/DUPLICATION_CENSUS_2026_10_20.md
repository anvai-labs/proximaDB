// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
# ProximaDB Duplication Census & Consolidation Program — Pass 3 (jscpd + LLM-grounded review)

**Date:** 2026-10-20 · **Base:** develop @ 7db68ea4d · **Method:** jscpd (Rust rewrite, npx) exact+Type-3 similarity over `rust,python`, then per-family LLM code review for verification and fix design.

## Relation to prior passes

- **Pass 1 → PR #2087 (group1):** 4 mechanical consolidations (canonical SqlValue decode, `StorageError::Serialization` merge, axis `SerializationError` enum merge, shared `now_unix_ms`). DONE.
- **Pass 2 → PRs #2091/#2093 (group2):** `TestPgServer` shared pgwire e2e harness; 26+ files converted. DONE.
- **This pass (pass 3):** full-codebase jscpd census (both languages, both clone kinds) + LLM verification of the top ~50 families, producing grouped, meaty-PR-sized follow-ups (per PR-sizing mandate).

## Scan configuration

The census flags live inline in the Makefile `dup-report` target (the
single source of truth; an earlier `.jscpd.json` approach was never
committed — the blanket `*.json` gitignore dropped it — so b0b0b7696
moved the flags inline). `make dup-report` runs:

```
npx --yes jscpd . --format rust,python --min-tokens 50 --similarity \
  --reporters silent,json,markdown --output target/jscpd-report \
  --ignore **/target/**,**/node_modules/**,**/_generated/**,**/*_pb2*.py,third_party/**,docs/**,
            crates/foundation/proximadb-proto/src/proto/**,data/**,bench-data/**,
            artifacts/**,metadata/**,log/**,**/.git/**,ui/**,roadmap/**
```

(The original pass-3 scan filtered the `*_pb2*.py` protobuf stubs at the
clustering step rather than in the scan; the flags now exclude them at
scan time — same effective methodology, one place.)

Generated code excluded: prost proto (`*.v*.rs`), OpenAPI-generated SDK (`**/_generated/**`),
protobuf python stubs (`*_pb2*.py`, filtered at clustering). License-header clone pairs filtered (411).

## Global numbers

| Metric | Value |
|---|---|
| Files scanned (rust+python) | 3,292 (2,538 rs + 754 py) |
| Total lines | 1,847,203 |
| Exact clone pairs (≥50 tokens) | 10,276 |
| Type-3 "similar" pairs (AST-normalized, ≥0.8) | 2,866 |
| Reported duplicated lines (jscpd total) | 163,450 (8.85%) |
| Non-boilerplate clone pairs clustered | 12,731 |
| **Duplication families (clustered)** | **5,352** |

Duplicated lines by code kind (family-weighted, `(k−1)·L`):
`prod 122,383` · `integration-test 46,438` · `generated 14,327` (excluded) · `test-code 13,828` ·
`client (handwritten SDK) 8,841` · `bench 7,502` · `example 2,590` · `demo 2,529` · `script 2,088`

Top hot files (most duplicated lines involved): src/services/dml/tests.rs (1750L),
sst/segment_format.rs (1672L), query/federated/mod.rs (1595L), sst/readers/tests.rs (1271L),
binding/proximadb-catalog-introspection/src/lib.rs (1227L), orion traversal.rs (1212L),
services/ddl/mod.rs (1058L), core/search/hybrid/fusion.rs (1034L), services/record_store.rs (1020L).

## Method for verification (LLM-grounded pass)

Each of the top ~50 families (selected by `dup_lines` and thematic coherence) was reviewed by a
fresh-context agent that read the actual code at every site (not just the jscpd fragment) and
returned: verdict (REAL / REAL-DRIFTED / PARTIAL / FALSE-POSITIVE / ALREADY-ADDRESSED),
drift evidence, root cause, recommended fix **as an engineering seam** (shared helper / trait +
generics / macro / enum dispatch / harness — not blind extraction to a common module), placement
honoring CI-enforced layering, size, risk, and supplemental jscpd-missed twins.

Verdict tables and per-family detail: see `reviews/batch-*-review.md` (sections below).

## Adjudicated verdicts (51 families, 9 review passes)

**Method correction:** jscpd's kind labels are file-level; reviewers found several "prod" families
entirely inside `#[cfg(test)]` modules (notably #3 — the catalog-introspection "fork" hypothesis
was DISPROVEN: the binding crate is the single canonical implementation and root is a 2-line
re-export; all 212 clone sites are test-preamble). Treat the `prod 122kL` figure as an upper bound;
block-level truth: roughly half of reviewed prod-labeled volume is inline test scaffolding.

Outcome mix: **4 FALSE-POSITIVE** (#6 license headers, #47 & #50 idiomatic `impl Default`/`Display`,
#9 dead file), **1 ALREADY-ADDRESSED** (#54 — canonical helper exists, ~20 call sites never
migrated), **~13 REAL-DRIFTED** (copies already diverged semantically), rest REAL.

| Family | Area | Verdict | Fix (seam) | Size/Risk |
|---|---|---|---|---|
| #19 | hybrid fusion.rs | **REAL-DRIFTED — live prod bug** | min-fold from `0.0`→`NEG_INFINITY` in comb_* /dempster_shafer, then `fuse_by_score(score, Combine)` extraction (~700→250L) | fix S / extract M |
| #13 | codec simd pfor | **REAL-DRIFTED — wire hazard** | SIMD PFor patches 8B vs baseline 12B, SIMD registered first; verify call sites, then mixed-read-safe alignment | M / med |
| #35s | audit storage | **prod hazard (supplemental)** | `store_audit_event` is an `Ok(())` placeholder silently dropping audit events → fail-fast | S |
| #46 | schema twins (4 builders + 2 enums) | **REAL-DRIFTED — worst structural drift** | staged: delete dead twins → converge on storage-common `ColumnarSchemaBuilder` + foundation proto enums, fail-loud lossy conversion, round-trip test | L / staged |
| #31 | engine filter→index converter | REAL-DRIFTED (live) | SST's 5 operator mappings (Contains/StartsWith/EndsWith/Like/Between) missing in 5 engines → shared fn in engines/core | S-M / low |
| #18 | engine extraction.rs | REAL-DRIFTED (live) | TD-DSEFF-1 perf fix in only 3/7 engines → template-method + `RecordSource` trait in trait_components | M |
| #62 | 2PC participants | REAL (prod) | `StagedOpLog<Op>` + `commit(txn, label, apply_fn)` | M / med-low |
| #127 | mlflow run store | REAL-DRIFTED (prod) | generic `ordered_payloads<T>`; error-suffix drift | S-M |
| #101 | embedded python.rs | REAL (prod) | 3 conversion helpers; `upsert_numpy` missing ids-count validation | S |
| #29 | metrics registration | REAL-DRIFTED | `register_counter_vec_safe` ×13 in 3 semantic forks → pub helpers in proximadb-metrics + `try_` variant (decide helix Option policy) | M / low |
| #41 | REST v2 twins | REAL-DRIFTED | delete 521-line unrouted v1 vestige in api crate; `collection_request()`/`map_collection_err()` in root v2 (string-match 404→500 hazard) | M / low-med |
| #11+#28 | progressive stages | REAL-DRIFTED but **prod-dead** | delete ~2,000-LOC chain (RL integration uncalled); RAPTOR SIMD batching + dead threshold documented if revived | M / low |
| #60/#9/#25 | orphaned test files | dead code | delete raptor/tests.rs, sst/readers/tests.rs (port its 4 z-order tests — only coverage of those fns), sst_flush_recovery_test.rs (985L dead fork) | S |
| #2/#23/#43/#26 | tests/ e2e pool | REAL | extend TestPgServer (`start_with`, randomize ports incl. fixed 5680, grpc/flight, `rest_client()`, `unique_name()`) → 5 conversion waves; 31/45 files lack TD-CONV-1 anchor (live flake class); api_v2 file vacuous (21 tests silently skip) | L / low-med |
| #38 | dml/tests.rs | REAL | module harness `setup_manager()/create_table()/make_dml()` (~1,200L) | M-L / low |
| #39 | engine query tests | REAL-DRIFTED | `SearchCtxBuilder` + generic `search_ids<E>` via port trait; keep drifted knobs explicit (2 qa-gate files) | M |
| #20/#12/#25/#36/#56 | unit-test scaffolding | REAL(-DRIFTED) | adopt helpers that already exist next door (create_test_coordinator, TestCollectionBuilder, flush_params, `run_format_benchmark<E>`, sst `test_support` mod) | M / low |
| #3/#122/#133/#33/#34/#58/#27+#45/#30 | inline cfg(test) preamble | REAL (test-only) | fixture fns in `src/testing.rs` / sibling tests.rs moves (federated mod.rs is 88% tests; segment_format tests start at 5084) | M / low |
| #57 | FS test doubles | REAL (mild drift) | `DelegatingFileSystem`/macro in proximadb-storage-filesystem-types (filesystem_type delegation drift) | S-M |
| #16/#37 | config/wire structs without Default | REAL (test-side tax ~1,200L) | additive `Default` impls/derives + fixtures | S-M / low |
| #22+#51 | bitpack/zigzag | REAL | `PackPrim` generic; delete zigzag private copies; byte-identical | S-M |
| #5 | codec width cores | REAL | `WirePrim`/macro width stamping (seam `functions/helpers.rs` exists, width never genericized); nova/parquet metadata cursor helper | M |
| #4 | GPU kernels | REAL | rocm = literal fork of opencl; shared `cpu_fallback.rs`; latent bits>16 truncation (feature-gated) | S-M |
| #54 | quantization wiring | ALREADY-ADDRESSED | migrate ~20 sites to `new_with_config` (benches silently use default metric) | S sweep |
| #10 | EmbeddingCell literals | REAL | `new_fp32` sweep (~150 sites, 151 files); guards documented invariant | M / low |
| #44/#71/#123/#70 | Python SDK | REAL(-DRIFTED) | client_v1 graph section = mandate #15 violation (raw `requests`) → delegate/delete; `_graph_shapes.py`; add `shortest_path` to codegen helpers + fold rest_async.py; one pooled executor | M |
| #17/#42 | Python scripts/benches | REAL | extend `corpus_io.py` (2/32 scripts use it today!) + adopt; `benchmarks/_harness.py` | M / low |
| #6/#47/#50 | — | FALSE-POSITIVE | license headers; idiomatic Default/Display impls | — |

**Dead/zombie inventory beyond families:** engine-core `ColumnarSchema` held only for a Debug impl;
viper `test_data_generator.rs` (700+L) compiled into prod (`pub mod`, not cfg(test));
tests/api_consistency_* POST to removed /api/v1 endpoints and pass because nothing listens;
16 `explain_err` fns byte-equal to harness `db_err`.

## Follow-on program (meaty, cohesive PRs per PR-sizing mandate)

0. **Repeat the census cheaply**: `make dup-report` (flags inline in the Makefile target; this pass's
   exact config); re-run per release train. Two guard ideas from findings: (a) orphaned-test-file
   reachability guard (three dead files found), (b) optional `scripts/check_license_headers.py`
   (headers do drift — one has a typo).
1. **PR-A — correctness quartet (fix-first, red tests)**: fusion min-fold (+ discriminating test —
   current test asserts a value that coincides under the bug); audit no-op fail-fast;
   `upsert_numpy` ids-count validation; mlflow error-suffix (rides #127 generic). Verify PFor
   SIMD/baseline reachability; if live → own mixed-read-safe PR + TD.
2. **PR-B — the big deletion**: progressive-stages chain (~2,000L), 3 orphaned test files (port
   z-order tests first), api-crate 521L vestige, zombie v1 tests, viper pub-mod → cfg(test),
   engine-core dead schema twin. Expect large −LOC, low risk.
3. **PR-C — engine drift harmonization**: shared filter→index converter (#31, behavior improvement
   for 5 engines), extraction template-method (#18), bulk-flush builders (#53),
   `select_blocks_by_centroid` triplication (jscpd-missed).
4. **PR-D — mechanical sweeps**: quantization `new_with_config` (#54), EmbeddingCell `new_fp32`
   (#10 prod sites), Default derives (#16/#37), metrics registration helpers (#29),
   bitpack/zigzag + codec width cores (#22/#51/#5, golden-byte tests prove byte-identical).
5. **PR-E — TestPgServer extension + wave 1** (pgwire stragglers + explain_err deletion), then
   waves: REST (#43) → Flight (randomize 5680 first) → gRPC → security-parameterized; separate PRs
   for dml/tests.rs harness (#38), engine_query builder (#39), api_v2 port (#26).
6. **PR-F — Python SDK + tooling**: client_v1 delegation, `_graph_shapes.py`, `shortest_path`
   codegen helper + rest_async fold, pooled executor; corpus_io adoption; `_harness.py`.
7. **PR-G — schema convergence (staged L)**: file TD first (drift already on the wire: Binary vs
   Utf8 codebook ids, ms vs µs timestamps, Map/List metadata under 3 names); stage 1 delete dead
   twins, stage 2 converge + round-trip ratchet.

Ordering rationale: correctness before deletion before consolidation; every PR group is
independently mergeable and touches disjoint file sets (no >2-PR overlap per SOLID mandate 17d).


## Repeatability

`make dup-report` replays this census (flags inline in the Makefile target;
reports land in `target/jscpd-report/`, gitignored). Re-run per release train and compare the
`total.percentage` figure — the hygiene ratchet. Method note: jscpd kind labels are file-level;
verify `#[cfg(test)]` span placement before counting a family as production duplication (half
the "prod"-labeled volume in this census was inline test scaffolding).

## Status of the findings (updated by PR-B)

- The dead-file classes this PR removes: the orphaned progressive-stages chain (11 files),
  sst/readers/tests.rs (z-order tests ported into block_pruning.rs first),
  raptor/tests.rs, the entire tests/integration/storage/ directory, the api-crate
  canonical/catalog.rs vestige, the viper readers/ test-data generator + its disabled
  consumers, and the engine-core ColumnarSchema twin.
- PR-A (#2100) landed the four drift-born correctness fixes (fusion folds, audit fail-closed,
  embedded ids parity, TD-CODEC-1 PFor pins).

## Planning-path migration (2026-10-26)

The owner-directed current/history split retains frozen historical planning snapshots in
`docs/history/planning/`, including seven files previously excluded under `roadmap/**`.
Those snapshots deliberately preserve original prose and may overlap the active contracts
extracted from them. The census excludes that historical planning directory; `docs/current/`,
historical evidence, all other documents and production code remain checked. The existing
percentage and duplicated-line thresholds are unchanged. Do not place active plans or source
code in history to bypass this guard; the current index and tracker own all open work.

The first migration census failed the duplicated-line cap because of this intentional overlap;
that is a scope change for frozen documentation, not a relaxed implementation threshold.
