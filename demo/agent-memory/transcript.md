# Verified full-stack run — 2026-10-18

Engine: proximadb-server built from this repo with `cargo build --release -p proximadb-server --features onnx` + BGE-small model files (see demo/agent-memory/fetch-models.sh).
API: AnvaiOps API (governed multi-tenant surface) over the same engine.
Agent surface: anvaiops-mcp driven exactly like an MCP host (stdio JSON-RPC).
Reproduce: anvaiops-mcp repo, demo/ directory — e2e.js asserts every step.

```
╭─ agent: memory_search — 'Kinesis throttling on writes'
2 matches for "Kinesis throttling on writes" (31.5 ms · trace 2a8ab7c3dd424876a08fd24709a6b126):

1. Kinesis throttling on write-heavy ingestion shard [critical] (resolved)
   Source: pagerduty — https://demo.local/pagerduty/INC-9005 · Team: Streaming
   When: created 2026-08-11, resolved 2026-08-11 (38 min)
   RCA: Producer burst exceeded shard 1MB/s write limit after onboarding clickstream firehose; WriteThroughputExceeded for 22 min
   Fix: Enable producer aggregation + retry with KPL; split shard 3→6; move firehose to dedicated stream
   Components: kinesis, ingest
   Tags: throttling, throughput
   Score: 0.016 · id INC-9005

2. Delta writes stalled by concurrent vacuum + auto-compact [medium] (resolved)
   Source: agent · Team: Data Platform
   When: created 2026-10-18
   Components: delta-lake
   Tags: vacuum, compaction
   Score: 0.016 · id mcp_mvdaw7tw_0_5zo7z9gb 

╭─ agent: memory_write — records what it just fixed
Committed 1 memory (mode=async; ids: mcp_mvdaxlt3_0_xoeyw3ef). Searchable once indexed — typically seconds. 

╭─ agent (later, fresh session): memory_search — 'delta checkpoints stalled'
2 matches for "delta checkpoints stalled" (33.0 ms · trace d1f64ff86c9e40459d366e933ca4ca40):

1. S3 request-rate throttling stalled Delta checkpoints [high] (resolved)
   Source: agent · Team: Data Platform
   When: created 2026-10-18
   Components: delta-lake, s3
   Tags: throttling, checkpoints
   Score: 0.016 · id mcp_mvdaxlt3_0_xoeyw3ef

2. Delta writes stalled by concurrent vacuum + auto-compact [medium] (resolved)
   Source: agent · Team: Data Platform
   When: created 2026-10-18
   Components: delta-lake
   Tags: vacuum, compaction
   Score: 0.016 · id mcp_mvdaw7tw_0_5zo7z9gb 

```
