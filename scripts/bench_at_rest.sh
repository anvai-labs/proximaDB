#!/bin/bash
# At-rest storage benchmark: compare gate OFF vs gate ON
# Measures actual disk usage after ingesting the same document corpus
# Usage: ./scripts/bench_at_rest.sh [doc_count] [server_binary]

set -euo pipefail

DOC_COUNT=${1:-10000}
SERVER_BIN=${2:-./target/release/proximadb-server}
PORT=15680
COLLECTION="at_rest_bench"

echo "=== At-Rest Storage Benchmark ==="
echo "Documents: $DOC_COUNT"
echo "Server: $SERVER_BIN"
echo ""

# Check server binary
if [ ! -f "$SERVER_BIN" ]; then
    echo "❌ Server binary not found: $SERVER_BIN"
    exit 1
fi

# Generate document corpus (same for both runs)
echo "📝 Generating $DOC_COUNT documents..."
DOCS_FILE=$(mktemp)
python3 - "$DOC_COUNT" > "$DOCS_FILE" <<'PYTHON'
import json, random, sys

count = int(sys.argv[1])
random.seed(42)

for i in range(count):
    shape = i % 4
    if shape == 0:
        meta = {"name": f"user_{i}", "email": f"user{i}@test.com", "city": "Chicago", "tags": ["a", "b", "c"]}
    elif shape == 1:
        meta = {"user": {"id": i, "profile": {"name": f"user_{i}", "age": 25 + (i % 50)}}, "active": True, "city": "NYC"}
    elif shape == 2:
        meta = {"id": i, "scores": [random.random() for _ in range(10)], "counts": [i, i*2, i*3], "name": f"item_{i}"}
    else:
        meta = {"name": f"doc_{i}", "source": "bench", "version": 1, "values": [1.5, 2.5, 3.5], "city": "LA"}

    # Documents API format: {id, text, metadata}
    doc = {"id": f"rec_{i}", "text": f"Document {i} content", "metadata": meta}
    print(json.dumps(doc))
PYTHON

# Function to run benchmark with a given gate setting
run_bench() {
    local gate_value=$1
    local label=$2
    local data_dir=$(mktemp -d)
    
    echo ""
    echo "─── $label (PROXIMADB_DOC_BINARY_INGEST=$gate_value) ───"
    
    # Start server
    PROXIMADB_DOC_BINARY_INGEST=$gate_value "$SERVER_BIN" -c config/bench.toml \
        --data-dir "$data_dir" > /dev/null 2>&1 &
    local server_pid=$!
    
    # Wait for server
    for i in {1..30}; do
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Server failed to start"
            kill $server_pid 2>/dev/null || true
            rm -rf "$data_dir"
            return 1
        fi
        sleep 1
    done
    
    # Create collection with schema defining filterable columns for lazy materialization
    # Schema marks city/name/source as filterable → they go to typed overlay only
    # Other fields (email, tags, user, scores, etc.) go to blob only → no duplication
    curl -sf -X POST "http://localhost:$PORT/api/v2/collections" \
        -H "Content-Type: application/json" \
        -d '{"name": "'"$COLLECTION"'", "dimension": 128, "enable_proxima_record": true, "schema": {"columns": [{"name": "city", "data_type": "text", "filterable": true}, {"name": "name", "data_type": "text", "filterable": true}, {"name": "source", "data_type": "text", "filterable": true}], "enforcement": "flexible"}}' > /dev/null
    
    # Ingest documents
    echo "  Ingesting $DOC_COUNT documents..."
    local start_time=$(python3 -c "import time; print(time.time())")
    
    local batch_size=100
    local total_batches=$((DOC_COUNT / batch_size))
    local success=0
    
    for batch in $(seq 1 $total_batches); do
        local start_line=$(( (batch - 1) * batch_size + 1 ))
        local end_line=$(( batch * batch_size ))
        
        local batch_docs=$(sed -n "${start_line},${end_line}p" "$DOCS_FILE" | jq -s '{records: .}')
        
        local http_code=$(curl -sf -o /dev/null -w "%{http_code}" -X POST \
            "http://localhost:$PORT/api/v2/collections/$COLLECTION/documents" \
            -H "Content-Type: application/json" \
            -d "$batch_docs" 2>/dev/null || echo "000")
        
        if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
            success=$((success + batch_size))
        fi
        
        if [ $((batch % 10)) -eq 0 ]; then
            echo "  Progress: $batch/$total_batches batches ($success docs)"
        fi
    done
    
    local end_time=$(python3 -c "import time; print(time.time())")
    local duration=$(python3 -c "print($end_time - $start_time)")
    local throughput=$(python3 -c "print($success / $duration if $duration > 0 else 0)")
    
    # Stop server
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    sleep 2
    
    # Measure storage size (macOS compatible)
    local storage_kb=$(du -sk "$data_dir" 2>/dev/null | cut -f1)
    local storage_bytes=$((storage_kb * 1024))
    local storage_mb=$(python3 -c "print($storage_bytes / (1024 * 1024))")
    local bytes_per_doc=$(python3 -c "print($storage_bytes / $success if $success > 0 else 0)")
    
    echo ""
    echo "  Results:"
    echo "    Documents ingested: $success"
    echo "    Duration: ${duration}s"
    echo "    Throughput: ${throughput} docs/sec"
    echo "    Storage size: ${storage_mb} MB"
    echo "    Bytes/doc: ${bytes_per_doc}"
    
    # Cleanup
    rm -rf "$data_dir"
    
    # Return values via temp files
    echo "$storage_bytes" > "/tmp/at_rest_${label}_bytes"
    echo "$bytes_per_doc" > "/tmp/at_rest_${label}_bpd"
    echo "$throughput" > "/tmp/at_rest_${label}_tps"
}

# Run with gate OFF (baseline)
run_bench "0" "gate_off"

# Run with gate ON (binary path)
run_bench "1" "gate_on"

# Compare results
echo ""
echo "=== Comparison ==="
off_bytes=$(cat /tmp/at_rest_gate_off_bytes)
on_bytes=$(cat /tmp/at_rest_gate_on_bytes)
off_bpd=$(cat /tmp/at_rest_gate_off_bpd)
on_bpd=$(cat /tmp/at_rest_gate_on_bpd)
off_tps=$(cat /tmp/at_rest_gate_off_tps)
on_tps=$(cat /tmp/at_rest_gate_on_tps)

ratio=$(python3 -c "print($on_bytes / $off_bytes * 100 if $off_bytes > 0 else 0)")
savings=$(python3 -c "print((1 - $on_bytes / $off_bytes) * 100 if $off_bytes > 0 else 0)")

echo ""
echo "| Metric         | Gate OFF     | Gate ON      | Delta        |"
echo "|----------------|--------------|--------------|--------------|"
echo "| Storage (MB)   | $(python3 -c "print(f'{$off_bytes / (1024*1024):.1f}')") | $(python3 -c "print(f'{$on_bytes / (1024*1024):.1f}')") | ${savings}% savings |"
echo "| Bytes/doc      | ${off_bpd} | ${on_bpd} | ${ratio}% of baseline |"
echo "| Throughput     | ${off_tps} docs/s | ${on_tps} docs/s | — |"

# Cleanup
rm -f "$DOCS_FILE" /tmp/at_rest_gate_*

echo ""
echo "✅ At-rest benchmark complete"
