#!/bin/bash
# E2E ingest benchmark for value-codec refactor evidence
# Measures REST document insert throughput with realistic payloads
# Usage: ./scripts/bench_e2e_ingest.sh [doc_count] [server_binary]

set -euo pipefail

DOC_COUNT=${1:-10000}
SERVER_BIN=${2:-./target/release/proximadb-server}
CONFIG=${3:-./config/bench.toml}
RESULTS_DIR="bench_results_$(date +%Y%m%d_%H%M%S)"
COLLECTION="bench_ingest_$$"
PORT=15680

mkdir -p "$RESULTS_DIR"

echo "=== E2E Ingest Benchmark ==="
echo "Documents: $DOC_COUNT"
echo "Server: $SERVER_BIN"
echo "Results: $RESULTS_DIR"
echo ""

# Check server binary
if [ ! -f "$SERVER_BIN" ]; then
    echo "❌ Server binary not found: $SERVER_BIN"
    exit 1
fi

# Start server
echo "🖥️ Starting server..."
DATA_DIR=$(mktemp -d)
"$SERVER_BIN" -c "$CONFIG" > "$RESULTS_DIR/server.log" 2>&1 &
SERVER_PID=$!

# Wait for server
echo "⏳ Waiting for server startup..."
for i in {1..30}; do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✅ Server ready (PID: $SERVER_PID)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Server failed to start"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    kill $SERVER_PID 2>/dev/null || true
    rm -rf "$DATA_DIR"
}
trap cleanup EXIT

# Create collection
echo "📦 Creating collection: $COLLECTION"
curl -sf -X POST "http://localhost:$PORT/api/v2/collections" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$COLLECTION\", \"dimension\": 128, \"enable_proxima_record\": true}" > /dev/null || {
    echo "❌ Failed to create collection"
    exit 1
}

# Generate realistic documents (mix of string-heavy, nested, numeric arrays)
# Using records/batch format with pre-computed vectors to isolate codec cost
echo "📝 Generating $DOC_COUNT documents..."
python3 - "$DOC_COUNT" > "$RESULTS_DIR/documents.jsonl" <<'PYTHON'
import json, random, sys, math

count = int(sys.argv[1])
random.seed(42)

for i in range(count):
    # Pre-computed vector (128-dim) to isolate codec cost from embedding
    vec = [math.sin(i * 0.01 + j * 0.1) * 0.5 for j in range(128)]
    
    # Mix of metadata shapes to exercise the value codec
    shape = i % 4
    if shape == 0:
        # String-heavy
        meta = {"name": f"user_{i}", "email": f"user{i}@test.com", "city": "Chicago", "tags": ["a", "b", "c"]}
    elif shape == 1:
        # Nested
        meta = {"user": {"id": i, "profile": {"name": f"user_{i}", "age": 25 + (i % 50)}}, "active": True}
    elif shape == 2:
        # Numeric arrays
        meta = {"scores": [random.random() for _ in range(10)], "counts": [i, i*2, i*3]}
    else:
        # Mixed
        meta = {"name": f"doc_{i}", "source": "bench", "version": 1, "values": [1.5, 2.5, 3.5]}
    
    record = {"id": f"rec_{i}", "vector": vec, "props": meta}
    print(json.dumps(record))
PYTHON

# Run benchmark
echo "🚀 Running ingest benchmark..."
START_TIME=$(python3 -c "import time; print(time.time())")

# Batch insert (100 docs per request) via records/batch endpoint
BATCH_SIZE=100
TOTAL_BATCHES=$((DOC_COUNT / BATCH_SIZE))
SUCCESS=0
FAIL=0

for batch in $(seq 1 $TOTAL_BATCHES); do
    START_LINE=$(( (batch - 1) * BATCH_SIZE + 1 ))
    END_LINE=$(( batch * BATCH_SIZE ))
    
    # Extract batch and wrap in records array
    BATCH_DOCS=$(sed -n "${START_LINE},${END_LINE}p" "$RESULTS_DIR/documents.jsonl" | jq -s '{records: .}')
    
    RESPONSE=$(curl -sf -w "\n%{http_code}" -X POST "http://localhost:$PORT/api/v2/collections/$COLLECTION/records/batch" \
        -H "Content-Type: application/json" \
        -d "$BATCH_DOCS" 2>&1)
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
        SUCCESS=$((SUCCESS + BATCH_SIZE))
    else
        FAIL=$((FAIL + BATCH_SIZE))
        if [ $FAIL -le 100 ]; then
            echo "  ⚠️ Batch $batch failed (HTTP $HTTP_CODE): $(echo "$RESPONSE" | head -1 | head -c 200)"
        fi
    fi
    
    # Progress indicator
    if [ $((batch % 10)) -eq 0 ]; then
        echo "  Progress: $batch/$TOTAL_BATCHES batches ($SUCCESS docs inserted)"
    fi
done

END_TIME=$(python3 -c "import time; print(time.time())")

# Calculate metrics
DURATION=$(python3 -c "print($END_TIME - $START_TIME)")
THROUGHPUT=$(python3 -c "print($SUCCESS / $DURATION if $DURATION > 0 else 0)")

echo ""
echo "=== Results ==="
echo "Duration: ${DURATION}s"
echo "Documents inserted: $SUCCESS"
echo "Failed: $FAIL"
echo "Throughput: ${THROUGHPUT} docs/sec"

# Save results
cat > "$RESULTS_DIR/summary.json" <<EOF
{
    "doc_count": $DOC_COUNT,
    "duration_sec": $DURATION,
    "inserted": $SUCCESS,
    "failed": $FAIL,
    "throughput_docs_per_sec": $THROUGHPUT,
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "server_binary": "$SERVER_BIN",
    "git_commit": "$(git rev-parse HEAD)"
}
EOF

echo ""
echo "📊 Results saved to $RESULTS_DIR/summary.json"
cat "$RESULTS_DIR/summary.json"
