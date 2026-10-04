# HANDOFF: arXive search API — use this instead of ssh+grep for related-work research

> For the ProximaDB session(s) that ran related-work searches (iceberg REST
> catalog, lakehouse table formats, predicate pushdown, Arrow Flight SQL,
> OLTP/OLAP hybrid routing) by ssh-ing into dataserver3 and using the arxive
> CLI. **You no longer need ssh** — arxive now serves a JSON API over the LAN,
> reachable from any machine, no arxive checkout required.
>
> Feedback from an earlier agent run (date filters, multisearch, fault
> behavior, context expansion) has already been implemented — see §4.

## 1. What arxive is

A private arXiv corpus + semantic search service running on dataserver3
(`192.168.1.89`): ~93.5k papers (~4.2M indexed chunks), GPU-embedded
(bge-small via inferflux on aiserver1), IVF-PQ indexed — search latency is
tens of ms server-side. Read-only for consumers; the crawler/indexer is a
separate pipeline.

## 2. Quickstart

```bash
# one-time: fetch the API key from dataserver3 (LAN secret; don't commit it)
KEY=$(ssh vsingh@dataserver3 'grep ARXIVE_API_KEY /data/arxive/deploy/web.env | cut -d= -f2')
echo "export ARXIVE_API_KEY=$KEY" >> ~/.zshrc   # or your secrets manager

curl -H "Authorization: Bearer $ARXIVE_API_KEY" \
  "http://192.168.1.89:8200/api/search?q=iceberg+rest+catalog&k=5&cat=cs.DB"
```

```python
import httpx
r = httpx.get("http://192.168.1.89:8200/api/search",
              params={"q": "predicate pushdown vector search", "k": 8},
              headers={"Authorization": f"Bearer {key}"}, timeout=30)
for hit in r.json()["results"]:
    print(hit["score"], hit["arxiv_id"], hit["title"])
```

## 3. API surface (OpenAPI: `http://192.168.1.89:8200/openapi.json`, UI docs: `/docs`)

| Endpoint | Purpose |
|---|---|
| `GET /api/search?q=&cat=&k=&after=&before=` | Semantic search. `cat` = arXiv category filter, `after`/`before` = published-date bounds (ISO dates). Returns `{took_ms, count, results:[{arxiv_id, title, authors, abstract, score, categories, chunk_index, section, chunk_text}]}` |
| `POST /api/multi` | Multisearch with an explicit query list: `{"queries": ["...", "..."], "k": 10}` → merged, deduplicated ranking |
| `GET /api/similar/{arxiv_id}?k=` | Papers similar to one you already have |
| `GET /api/paper/{arxiv_id}?fulltext=true` | Full metadata + abstract (+ whole article text on demand); `pdf_endpoint` for the PDF |
| `GET /health` | Liveness + embed backend + peak RSS |

Notes that matter:
- Queries are automatically prefixed with the bge instruction server-side — send plain natural-language queries, no prompt engineering.
- Scores are **cosine** (0–1; ≥0.6 is usually a real match on this corpus).
- `chunk_text` is the matched passage with `chunk_index`/`section`; call
  `/api/paper/{id}?fulltext=true` for the whole article (§ context).
- `k` is clamped to 1–100. Invalid dates → 422. All errors are JSON.
- Auth: `Authorization: Bearer <key>` **or** a browser session (the web UI
  at `http://192.168.1.89:8200` uses the same engine).

## 4. Feedback loop closed

The earlier agent feedback was implemented (2026-09-20, commit `fd47166`):
published-date filters, a proper `POST /api/multi`, clamped/validated params,
JSON error responses (500s are never HTML anymore — a double-vs-single quote
DataFusion SQL bug in category filtering was found and fixed while doing it),
`chunk_index`/`section` on results, and `/health` now reports peak RSS +
embed backend for memory-overhead measurement.

## 5. Known limits

- **LAN-only** (no TLS, no internet exposure — by design).
- Search embeds run on **local CPU** while a one-time corpus re-embed finishes
  on the GPU host (~Sep 21 evening); queries are ~1 s now, ~50–80 ms after.
  Nothing to change client-side when it flips.
- The UI (`http://192.168.1.89:8200`) and this API share one engine — feature
  requests belong in the arxive repo (`/Users/vijaysingh/code/arxive`,
  `arxive/web.py`).
