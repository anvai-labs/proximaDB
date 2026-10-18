# ProximaDB Documentation

**The context database** — vectors, documents, graphs and relational SQL over object storage,
behind standard interfaces (Postgres wire, REST, gRPC, Arrow Flight).

For what ProximaDB is, how to install it and a first query, start at the
[project README](../README.adoc). This page is the map of the documentation.

> **Support contract:** Code presence is broader than the supported product surface. A feature
> existing in the code does not make it supported; the support contract below decides.

## One authority per question

| Question | Authority |
|---|---|
| What is supported, beta or experimental? | [`SUPPORTED_SURFACE.adoc`](./SUPPORTED_SURFACE.adoc) — wins over every other doc |
| Where is the product going? | [ADR-096](./12-design/adr/ADR-096-product-direction-sql-first-context-engine.adoc) (the decision) · [Vision](./00-product/VISION.adoc) · [PRD](./00-product/PRD.adoc) |
| What is being built next? | [Strategic Roadmap](./_internal/roadmap/STRATEGIC_ROADMAP.adoc) |
| How is the system put together? | [System Map](./12-design/SYSTEM_MAP_2026_05_30.adoc) · [diagrams](./architecture-diagrams/README.md) |
| Why was it built this way? | [ADR index](./12-design/adr/README.adoc) · [design index](./12-design/README.adoc) |
| What is done, open or broken? | [Work Ledger](./12-design/WORK_LEDGER_2026_10_15.adoc) · [Technical Debt Register](./10-quality/TECHNICAL_DEBT.adoc) · per-TD files in [`10-quality/td/`](./10-quality/td/) |
| Is a performance number real? | [Benchmark evidence ledger](./_internal/roadmap/BENCHMARK_EVIDENCE.toml) — a number not in the ledger is not a claim |
| How do I file a TD or ADR? | [How to file a TD and ADR](./12-design/HOW_TO_FILE_TD_AND_ADR.adoc) |

## Sections

| Section | For |
|---|---|
| [01 Quick Start](./01-quick-start/) | Install, first query, the moving parts |
| [02 Guides](./02-guides/) | Task-oriented how-tos |
| [03 API Reference](./03-api-reference/) | REST, gRPC, Arrow Flight and pgwire surfaces |
| [04 Operations](./04-operations/) | Deploying, monitoring, releasing |
| [05 Concepts](./05-concepts/) | Storage engines, quantization, the planner, the WAL |
| [06 Internals](./06-internals/) | Contributor guides |
| [10 Quality](./10-quality/) | Technical debt, benchmarks, quality ratchets |
| [11 Use cases](./11-usecases/) | Worked application designs |
| [12 Design](./12-design/README.adoc) | Architecture blueprints and ADRs |
| [Release notes](./release-notes/) · [RFCs](./rfcs/) | Release contracts; proposal template |

The published site is built from the Markdown pages by `mkdocs.yml` (home page:
[`index.md`](./index.md)).

## Rules for this tree

* Put a doc in the numbered section it belongs to. No one-off reports at the repository root or
  directly under `docs/`.
* A new design decision is an ADR; a new piece of work is a TD file. Follow
  [the filing convention](./12-design/HOW_TO_FILE_TD_AND_ADR.adoc).
* Point-in-time measurement records go in `_internal/status/`; cite them from the evidence ledger
  when they back a claim. Do not update them later; write a new dated record instead.
* When a doc is superseded, delete it and repoint its links to the successor. Git history keeps
  the old text, so there is no archive tree.
