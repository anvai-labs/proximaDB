# ProximaDB Documentation

**The context database** — vectors, documents, graphs and relational SQL over object storage,
behind standard interfaces (Postgres wire, REST, gRPC and experimental general-purpose
Arrow Flight transport; canonical Flight vector_search has its own supported contract).

For what ProximaDB is, how to install it and a first query, start at the
[project README](../README.adoc). This page is the map of the documentation.

> **Support contract:** Code presence is broader than the supported product surface. A feature
> existing in the code does not make it supported; the support contract below decides.

## Start with current work

Read [current/README.adoc](./current/README.adoc), then the
[central tracker](./current/TRACKER.adoc) and the plan relevant to your task.
Seven documents divide direction, sequencing, coordination and three implementation domains.
Completed narratives and dated evidence live in [history](./history/README.adoc).

| Question | Authority |
|---|---|
| What is supported? | [Support contract](./SUPPORTED_SURFACE.adoc) and [MVP corridor](./00-product/MVP_TRUST_CORRIDOR.toml) |
| Where are we going? | [Vision](./current/VISION.adoc); ADR-096/097 own decisions; PRD owns requirements |
| What release and exit criteria? | [Roadmap](./current/ROADMAP.adoc) |
| What is pending, active or blocked; which branch? | [Central tracker](./current/TRACKER.adoc) |
| What must implementation guarantee? | [Lakehouse](./current/LAKEHOUSE_PLAN.adoc), [storage/economics](./current/STORAGE_PLAN.adoc), [platform](./current/PLATFORM_PLAN.adoc) plans |
| How is the system built today? | [System map](./12-design/SYSTEM_MAP_2026_05_30.adoc) and [design references](./12-design/README.adoc) |
| Which decision or detailed defect? | [ADR index](./12-design/adr/README.adoc) and [TD registry](./10-quality/TECHNICAL_DEBT.adoc); stable IDs |
| What completed, and what proves a number? | [History index](./history/README.adoc) and [evidence ledger](./_internal/roadmap/BENCHMARK_EVIDENCE.toml) |
| How do I file a TD or ADR? | [Filing convention](./12-design/HOW_TO_FILE_TD_AND_ADR.adoc) |

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

* Keep active plans and coordination in `current/`; use its existing seven documents before
  creating another. Keep completed detail and dated evidence in `history/`, with short links
  from current. Pending work stays detailed in its current plan.
* Archive a superseded plan once, preserve unique ideas, and repoint links to the appropriate
  current successor or historical evidence. Historical pending labels are not live status.
* ADRs, TDs, support contracts and specialist specifications retain stable reference paths.
  Guides/API/operations documentation stays in its numbered section; this is a planning
  consolidation, not a relocation of the whole documentation tree.
* New measurements go in `history/evidence/`; cite the evidence ledger when they back a claim.
  Existing unrelated `_internal/status/` artifacts remain valid historical references.
* On completion update the TD and central tracker together, archive the detailed evidence,
  and remove the finished task narrative from current. Follow the [maintenance protocol](./current/README.adoc).
