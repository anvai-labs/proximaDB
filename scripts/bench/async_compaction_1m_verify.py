#!/usr/bin/env python3
"""Async-compaction (ADR-076 D1+D2) post-merge verification at SIFT 1M.

Evidence-ledger bench for the claim `async_compaction_1m_e2e`. Measures:
  D1 (async compaction): ingest throughput (flush NOT blocking compaction).
  D2 (L0 watermarks):    steady-state L0/L1 segment count (bounded).
  recall@10:             against the full 1M brute-force ground truth.

Prerequisites:
  - proximadb-server running on :5678 (release binary from develop).
  - SIFT1M dataset at /Users/vijaysingh/sift1m/ (sift_base.fvecs + sift_query.fvecs
    + sift_groundtruth.ivecs).
  - flush_interval_secs=12 in the server config (forces flushes during ingest
    so compaction fires — the default 300s would only flush post-ingest).
  - PROXIMADB_L0_COMPACTION_ENABLED=1.

Usage:
  python3 scripts/bench/async_compaction_1m_verify.py
"""
import json, os, struct, sys, time, urllib.request

SERVER = os.environ.get("PROXIMA_VERIFY_SERVER", "http://127.0.0.1:5678")
COLL   = "bench_async_compaction_1m"
N      = 1_000_000
BATCH  = 2000          # under the server's max_request_size_mb
NQUERY = 200
TOPK   = 10

def post(p, b, t=180):
    r = urllib.request.urlopen(urllib.request.Request(
        SERVER + p, data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json"}, method="POST"), timeout=t)
    return json.loads(r.read())

def read_fvecs(path, lim):
    with open(path, "rb") as f:
        raw = f.read()
    o = []; i = 0
    while len(o) < lim:
        if i + 4 > len(raw): break
        d = struct.unpack("<i", raw[i:i+4])[0]; i += 4
        o.append(list(struct.unpack("<%df" % d, raw[i:i+4*d]))); i += 4 * d
    return o

def read_ivecs(path, lim):
    with open(path, "rb") as f:
        raw = f.read()
    o = []; i = 0
    while len(o) < lim:
        if i + 4 > len(raw): break
        d = struct.unpack("<i", raw[i:i+4])[0]; i += 4
        o.append(list(struct.unpack("<%di" % d, raw[i:i+4*d]))); i += 4 * d
    return o

def main():
    base_path = os.environ.get("SIFT_BASE", "/Users/vijaysingh/sift1m/sift_base.fvecs")
    query_path = os.environ.get("SIFT_QUERY", "/Users/vijaysingh/sift1m/sift_query.fvecs")
    gt_path = os.environ.get("SIFT_GT", "/Users/vijaysingh/sift1m/sift_groundtruth.ivecs")

    print("loading SIFT 1M...", flush=True)
    base = read_fvecs(base_path, N)
    queries = read_fvecs(query_path, NQUERY)
    gt = read_ivecs(gt_path, NQUERY)
    print(f"loaded: {len(base):,} base, {len(queries)} queries", flush=True)

    c = post("/api/v2/collections", {
        "name": COLL, "dimension": len(base[0]), "engine": "sst",
        "enable_proxima_record": True, "distance_metric": "euclidean"})
    cid = c.get("collection_id", COLL)
    print(f"collection {COLL} -> {cid}", flush=True)

    # Ingest — D1 signal (throughput = flush not blocking)
    t0 = time.time(); last = t0
    for s in range(0, N, BATCH):
        chunk = [{"id": f"v{j}", "vector": base[j]}
                 for j in range(s, min(s + BATCH, N))]
        post(f"/api/v2/collections/{cid}/records/batch", {"records": chunk}, t=180)
        now = time.time()
        if now - last >= 20:
            done = s + len(chunk)
            print(f"  ingested {done:,}/{N:,} ({done/(now-t0):,.0f} vec/s)", flush=True)
            last = now
    ingest_dur = time.time() - t0
    print(f"INGEST done: {N:,} in {ingest_dur:.0f}s ({N/ingest_dur:,.0f} vec/s)", flush=True)

    # Wait for async compaction
    wait = int(os.environ.get("COMPACTION_WAIT_SECS", "150"))
    print(f"waiting {wait}s for async compaction...", flush=True)
    time.sleep(wait)

    # Recall — 1M GT matches 1M base
    recalls = []; lats = []
    for j in range(NQUERY):
        ts = time.time()
        r = post(f"/api/v2/collections/{cid}/search",
                 {"vector": queries[j], "top_k": TOPK}, t=120)
        lats.append((time.time() - ts) * 1000)
        got = {h.get("id") for h in (r.get("results") or [])}
        truth = {f"v{k}" for k in gt[j][:TOPK]}
        recalls.append(len(got & truth) / len(truth))
    lats.sort()
    mean_r = sum(recalls) / len(recalls)
    p50 = lats[len(lats) // 2]
    p95 = lats[int(len(lats) * 0.95)]

    print(f"\n{'='*64}")
    print(f"ASYNC COMPACTION (ADR-076 D1+D2) @ N={N:,}")
    print(f"{'='*64}")
    print(f"ingest throughput = {N/ingest_dur:,.0f} vec/s ({ingest_dur:.0f}s)")
    print(f"recall@{TOPK}       = {mean_r:.4f} (target >= 0.975)")
    print(f"cold latency      = p50 {p50:.1f}ms  p95 {p95:.1f}ms")
    print(f"\nGrep the server log for compaction counts:")
    print(f"  'Compaction completed for level'  = D1 async completions")
    print(f"  'L0 admission STOP'               = D2 backpressure engagements")

if __name__ == "__main__":
    main()
