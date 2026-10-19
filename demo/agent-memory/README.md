# Agent-memory demo — production-shaped, one command

The smallest honest demonstration of the memory loop on a **real stack**: Docker
engine with **server-side BGE embeddings** (no client-side embedding, no model
files in your app), native v2 REST API, and the write→read-back loop an agent
actually needs:

1. **Search with provenance** — incidents return RCA, fix, team, source — operational facts, not a document dump.
2. **Agents write memory** — text in, embedded server-side, attributed via metadata (`recorded_by`).
3. **Memory compounds** — a *later*, stateless query surfaces the agent-written memory as the top answer.

## Run it (one command)

```bash
./run.sh
```

What it does — each step is independently re-runnable:

| Step | What | Script |
|---|---|---|
| 1 | Download the embedding model (BAAI/bge-small-en-v1.5 ONNX, ~133MB, once) | `fetch-models.sh` |
| 2 | Build + start the engine (`proximadb-server --features onnx`, volumes for data/models) | `docker-compose.yml`, `Dockerfile.onnx` |
| 3 | Seed 8 realistic incidents → search → agent write → later search finds it | `seed_and_query.py` |

Expected tail:

```
PASS  search 'spark executor OOM after autoscalin' → INC-9001
PASS  search 'kinesis throttling on writes' → INC-9005
PASS  search 'flink checkpoints failing' → INC-9007
PASS  write → later search finds agent-written memory
ALL PASS
```

Teardown: `./run.sh down` (keeps `./data`) · full clean slate: `./run.sh reset` · the engine listens on **loopback only** (it runs unauthenticated).

## The API flow (what production does)

```
POST /api/v2/collections                      {name, dimension: 384}
POST /api/v2/collections/{c}/documents        {records: [{id, text, metadata}]}   ← embeds server-side
POST /api/v2/hybrid/index                     {collection, documents}             ← BM25 leg
POST /api/v2/hybrid/search                    {collection, text_query, top_k}     ← fused ranking
```

This is the same two-step (native embed + BM25 feed) the AnvaiOps governed API
performs on its ingest path.

## Honest notes

- Records persist on the `./data` volume (`metadata`, `collections`, `manifest` — all under
  `/data` via the container-tuned config). The BM25 leg of hybrid search is **in-process by
  engine design**: it starts empty each boot and is re-fed by `run.sh`'s idempotent seed —
  re-running the demo after a restart restores text search in seconds without re-ingest.
- The published lean image (`docker.io/vjsingh1984/proximadb`) builds **without**
  the `onnx` feature — hence the local build step here (`Dockerfile.onnx` is the
  root Dockerfile plus `--features onnx`). First build takes a while (Rust);
  subsequent runs hit the cache.
- Ranking above is carried by the BM25 leg of hybrid search; natively embedded
  vectors are stored (dim 384) and join fusion as the ANN index settles —
  engine-build dependent. Verified full-fidelity numbers (31–37 ms hybrid search
  behind the governed API + MCP server) are in [`transcript.md`](transcript.md).
- Corpus is fictional but faithful to real data-platform failure modes; edit it
  at the top of `seed_and_query.py`.

## The rest of the stack

| Piece | Where |
|---|---|
| ProximaDB engine (this repo) | https://github.com/anvai-labs/proximaDB |
| ProximaDB lean Docker image | https://hub.docker.com/r/vjsingh1984/proximadb |
| MCP server for agents (Claude, any MCP client) | https://github.com/anvai-labs/anvaiops-mcp |
| Governed multi-tenant API + beta | https://anvaiops.com |
