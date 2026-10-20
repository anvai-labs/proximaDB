#!/usr/bin/env python3
"""Agent-memory loop over the ProximaDB v2 API — seed, search, write, read-back.

Production-shaped flow, identical to what the AnvaiOps ingest/search layer does:
  POST /api/v2/collections/{c}/documents   — text in, embeddings computed
                                             SERVER-SIDE (native BGE via onnx)
  POST /api/v2/hybrid/index                — feed the BM25 leg
  POST /api/v2/hybrid/search               — text_query, fused ranking

No client-side embedding, no model files here — the engine owns embedding.
Requires an engine built with `--features onnx` and the BGE model present
(see run.sh / docker-compose.yml — they handle both).

Usage: seed_and_query.py [base_url]     (default http://localhost:5678)
"""
import datetime
import json
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5678").rstrip("/")
COLL = "agent_memory_demo"
EMBED_DIM = 384  # bge-small-en-v1.5

INCIDENTS = [
    dict(id="INC-9001", title="Spark OOM: executor heap exhaustion on EMR dynamic allocation",
         source="jira", severity="high", team="Platform Engineering",
         components=["spark", "emr"], tags=["oom", "autoscaling"],
         rca="spark.executor.memoryOverhead too low after scale-out",
         fix="Set memoryOverheadFactor=0.2 and executorIdleTimeout=120s; restart executors",
         text="Spark OOM executor heap exhaustion on EMR dynamic allocation. "
              "RCA: memoryOverhead too low after scale-out. Fix: memoryOverheadFactor 0.2, restart executors. Team: Platform Engineering."),
    dict(id="INC-9002", title="EMR autoscale triggers OOM on Spark shuffle",
         source="jira", severity="medium", team="Platform Engineering",
         components=["spark", "emr", "yarn"], tags=["oom", "shuffle"],
         rca="YARN container memory limits not updated on scale-out",
         fix="Increase yarn.nodemanager.resource.memory-mb; cap shuffle service nodes",
         text="EMR autoscale triggers OOM on Spark shuffle. RCA: YARN container memory limits stale on scale-out. Fix: raise nodemanager memory-mb. Team: Platform Engineering."),
    dict(id="INC-9003", title="Kafka consumer lag spike after partition rebalance",
         source="slack", severity="high", team="Streaming",
         components=["kafka", "flink"], tags=["lag", "rebalance"],
         rca="Rebalance storm paused processing ~9 min",
         fix="session.timeout.ms=30000, heartbeat 3000, static membership",
         text="Kafka consumer lag spike after partition rebalance. RCA: rebalance storm paused processing. Fix: session timeout 30s, static membership. Team: Streaming."),
    dict(id="INC-9004", title="Iceberg compaction falling behind on 40TB fact table",
         source="confluence", severity="medium", team="Data Platform",
         components=["iceberg", "spark"], tags=["compaction", "small-files"],
         rca="Serialized compaction jobs; 84k small files slowed scan planning",
         fix="Parallelize by partition bucket; hourly rewrite_data_files",
         text="Iceberg compaction falling behind on 40TB fact table. RCA: serialized compaction, 84k small files. Fix: parallelize by bucket, hourly rewrite. Team: Data Platform."),
    dict(id="INC-9005", title="Kinesis throttling on write-heavy ingestion shard",
         source="pagerduty", severity="critical", team="Streaming",
         components=["kinesis", "ingest"], tags=["throttling", "throughput"],
         rca="Producer burst exceeded shard 1MB/s after clickstream firehose onboarding",
         fix="KPL aggregation+retry; split shard 3 to 6; dedicated stream",
         text="Kinesis throttling on write-heavy ingestion shard. RCA: producer burst over shard write limit. Fix: KPL aggregation, split shards, dedicated stream. Team: Streaming."),
    dict(id="INC-9006", title="Unity Catalog permission drift blocked production queries",
         source="jira", severity="high", team="Governance",
         components=["unity-catalog", "iam"], tags=["permissions", "drift"],
         rca="Terraform GRANT drift after workspace migration",
         fix="Re-apply terraform plan; add permission-diff alerting",
         text="Unity Catalog permission drift blocked production queries. RCA: terraform GRANT drift after migration. Fix: re-apply plan, add drift alerts. Team: Governance."),
    dict(id="INC-9007", title="Flink checkpoint failure cascade under backpressure",
         source="slack", severity="high", team="Streaming",
         components=["flink", "kafka"], tags=["checkpoint", "backpressure"],
         rca="Checkpoint alignment timeout from slow sink; unaligned checkpoints off",
         fix="Unaligned checkpoints; 10min timeout; sink batch 500 to 2000",
         text="Flink checkpoint failure cascade under backpressure. RCA: alignment timeout, slow sink. Fix: unaligned checkpoints, larger sink batches. Team: Streaming."),
    dict(id="INC-9008", title="AI/BI Genie NL query returning wrong metric definition",
         source="jira", severity="medium", team="Analytics",
         components=["genie", "semantic-layer"], tags=["genie", "nl-analytics"],
         rca="Two certified metrics shared a display name",
         fix="Rename duplicate; certification review gate in CI",
         text="Genie NL query returning wrong metric definition. RCA: duplicate display names on certified metrics. Fix: rename, add review gate. Team: Analytics."),
]

# Provenance registry (the engine returns ranked ids; presentation joins here —
# exactly how the governed API composes record props over fused rankings).
CORPUS = {i["id"]: i for i in INCIDENTS}


def api(method: str, path: str, body: dict | None = None, headers: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")


def ingest(records: list[dict]) -> dict:
    """Native-embedded write + BM25 feed — the production two-step."""
    r1 = api("POST", f"/api/v2/collections/{COLL}/documents",
             {"records": records}, {"X-Embed-Source": "native", "X-Ingest-Mode": "sync"})
    r2 = api("POST", "/api/v2/hybrid/index",
             {"collection": COLL, "documents": [{"id": x["id"], "text": x["text"]} for x in records]})
    return {"embedded": r1, "bm25": r2}


def to_record(item: dict, recorded_by: str | None = None) -> dict:
    meta = {k: v for k, v in item.items() if k != "text"}
    meta["created_at"] = meta.get("created_at") or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if recorded_by:
        meta["recorded_by"] = recorded_by
    return {"id": item["id"], "text": item["text"], "metadata": meta}


def search(query: str, top_k: int = 3) -> list[dict]:
    res = api("POST", "/api/v2/hybrid/search", {"collection": COLL, "text_query": query, "top_k": top_k})
    return res.get("results", [])


def show(query: str, hits: list[dict]):
    print(f"\n╭─ agent: memory_search — '{query}'")
    print(f"│  {len(hits)} match(es)")
    for h in hits[:3]:
        meta = CORPUS.get(h["id"], {})
        print(f"│  {h['id']}  {meta.get('title', '')[:62]}")
        print(f"│     RCA: {str(meta.get('rca', '—'))[:72]}")
        print(f"│     Fix: {str(meta.get('fix', '—'))[:72]}")
        print(f"│     Team: {meta.get('team', '—')} · Source: {meta.get('source', '—')} "
              f"· bm25 {h.get('bm25_score', 0):.3f} · fused {h.get('fused_score', 0):.4f}")


def main() -> int:
    t0 = time.monotonic()
    print(f"ProximaDB agent-memory demo → {BASE}")

    # Idempotent: a persisted collection (./data volume) returns
    # 500 COLLECTION_EXISTS on re-create — that is the re-run path working.
    try:
        api("POST", "/api/v2/collections", {"name": COLL, "dimension": EMBED_DIM, "distance_metric": "cosine"})
    except urllib.error.HTTPError as exc:
        if exc.code != 500:
            raise
        print(f"collection '{COLL}' already exists (persisted) — re-seeding into it")
    r = ingest([to_record(i) for i in INCIDENTS])
    n = len((r.get("embedded") or {}).get("records", [])) or len(INCIDENTS)
    print(f"seeded {n} incidents (server-side BGE embeddings, dim {EMBED_DIM}) into '{COLL}'")

    checks = []
    for q, expect in [
        ("spark executor OOM after autoscaling", "INC-9001"),
        ("kinesis throttling on writes", "INC-9005"),
        ("flink checkpoints failing", "INC-9007"),
    ]:
        hits = search(q)
        show(q, hits)
        top = hits[0]["id"] if hits else None
        checks.append((f"search '{q[:30]}' → {expect}", top == expect))

    # Deterministic id: re-runs UPSERT the same memory (both in the document
    # store and the id-idempotent BM25 feed) instead of accumulating
    # timestamped duplicates that make the read-back race its predecessors.
    mem_id = "MEM-DEMO-DELTA-VACUUM"
    memory = dict(
        id=mem_id,
        title="Delta writes stalled by concurrent vacuum + auto-compact",
        source="agent", severity="medium", team="Data Platform",
        components=["delta-lake"], tags=["vacuum", "compaction"],
        rca="autoCompact triggered concurrent vacuum; file-lock contention",
        fix="disable duplicate vacuum schedule; auto-compact post-commit",
        text="Delta writes stalled by concurrent vacuum and auto-compact on the orders table. "
             "RCA: autoCompact triggered a concurrent vacuum, file-lock contention stalled commits. "
             "Fix: disable the duplicate vacuum schedule, move auto-compact post-commit. Team: Data Platform.",
    )
    CORPUS[mem_id] = memory
    ingest([to_record(memory, recorded_by="agent-memory-demo")])
    print(f"\n╭─ agent: memory_write — recorded '{memory['title']}' ({mem_id}, attributed recorded_by=agent-memory-demo)")

    hits = search("delta writes stalled vacuum")
    show("delta writes stalled vacuum", hits)
    top = hits[0]["id"] if hits else None
    checks.append(("write → later search finds agent-written memory", top == mem_id))

    print("\n" + "=" * 62)
    ok = True
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'} · {time.monotonic()-t0:.1f}s total")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
