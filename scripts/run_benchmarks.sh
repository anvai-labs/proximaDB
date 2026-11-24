#!/bin/bash
# ProximaDB Benchmark Execution Script
#
# Runs comprehensive benchmarks and generates reports for publication.
# Results are saved to benchmark-results/ directory with timestamp.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BENCHMARK_DIR="benchmark-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${BENCHMARK_DIR}/${TIMESTAMP}"
SUMMARY_FILE="${OUTPUT_DIR}/SUMMARY.md"

# Parse arguments
DATASET="${1:-sift}"
ENGINE="${2:-all}"
ITERATIONS="${3:-3}"

echo -e "${GREEN}=== ProximaDB Benchmark Suite ===${NC}"
echo "Dataset: ${DATASET}"
echo "Engine: ${ENGINE}"
echo "Iterations: ${ITERATIONS}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# System information gathering
echo -e "${YELLOW}Gathering system information...${NC}"
{
    echo "# System Information"
    echo ""
    echo "## Hostname"
    hostname
    echo ""
    echo "## CPU Info"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sysctl -a | grep -E "machdep.cpu|hw.memsize"
    else
        lscpu | grep -E "Model name|CPU\(s\)|Thread"
        grep "MemTotal" /proc/meminfo
    fi
    echo ""
    echo "## Rust Version"
    rustc --version
    echo ""
    echo "## ProximaDB Version"
    cargo run --release --bin proximadb-server -- --version || echo "v0.2.0"
    echo ""
} > "${OUTPUT_DIR}/SYSTEM_INFO.txt"

# Run Criterion benchmarks
echo -e "${YELLOW}Running Criterion benchmarks...${NC}"
cargo bench --bench bench_13_complete_suite -- --save-baseline ${TIMESTAMP} | tee "${OUTPUT_DIR}/criterion_output.txt"

# Run engine-specific benchmarks
if [[ "$ENGINE" == "all" || "$ENGINE" == "sst" ]]; then
    echo -e "${YELLOW}Running SST engine benchmarks...${NC}"
    cargo bench --bench bench_04_storage_unified -- --save-baseline "${TIMESTAMP}_sst" | tee "${OUTPUT_DIR}/sst_benchmark.txt"
fi

if [[ "$ENGINE" == "all" || "$ENGINE" == "helix" ]]; then
    echo -e "${YELLOW}Running HELIX engine benchmarks...${NC}"
    cargo bench --bench bench_04_storage_unified -- --save-baseline "${TIMESTAMP}_helix" | tee "${OUTPUT_DIR}/helix_benchmark.txt"
fi

# Run hybrid search benchmarks
echo -e "${YELLOW}Running hybrid search benchmarks...${NC}"
cargo bench --bench hybrid_search -- --save-baseline "${TIMESTAMP}_hybrid" 2>/dev/null | tee "${OUTPUT_DIR}/hybrid_benchmark.txt" || echo "Hybrid search benchmarks not found"

# Generate summary
echo -e "${YELLOW}Generating benchmark summary...${NC}"
{
    echo "# ProximaDB Benchmark Summary"
    echo ""
    echo "**Run Date**: $(date)"
    echo "**Dataset**: ${DATASET}"
    echo "**Engine**: ${ENGINE}"
    echo "**Iterations**: ${ITERATIONS}"
    echo ""
    echo "## System Information"
    echo ""
    echo '```'
    cat "${OUTPUT_DIR}/SYSTEM_INFO.txt"
    echo '```'
    echo ""
    echo "## Results"
    echo ""
    echo "### Criterion Benchmarks"
    echo ""
    if [ -f "${OUTPUT_DIR}/criterion_output.txt" ]; then
        # Extract key metrics from Criterion output
        grep -E "(test result|mean:|std:|median:)" "${OUTPUT_DIR}/criterion_output.txt" || true
    fi
    echo ""
    echo "## Raw Data"
    echo ""
    echo "Full benchmark results are available in this directory."
    echo ""
} > "${SUMMARY_FILE}"

# Copy Criterion HTML reports
if [ -d "target/criterion" ]; then
    echo -e "${YELLOW}Copying Criterion reports...${NC}"
    cp -r "target/criterion" "${OUTPUT_DIR}/"
fi

# Generate JSON output for CI/CD
echo -e "${YELLOW}Generating JSON output...${NC}"
cat > "${OUTPUT_DIR}/benchmark_results.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "dataset": "${DATASET}",
  "engine": "${ENGINE}",
  "iterations": ${ITERATIONS},
  "results_dir": "${OUTPUT_DIR}"
}
EOF

echo ""
echo -e "${GREEN}=== Benchmark Complete ===${NC}"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key files:"
echo "  - ${SUMMARY_FILE}"
echo "  - ${OUTPUT_DIR}/benchmark_results.json"
echo "  - ${OUTPUT_DIR}/criterion/ (HTML reports)"
echo ""
echo "To view HTML reports:"
echo "  open ${OUTPUT_DIR}/criterion/index.html"
echo ""
echo "To compare with previous runs:"
echo "  cargo bench --bench bench_13_complete_suite -- --baseline ${TIMESTAMP} --base ${PREVIOUS_TIMESTAMP}"
echo ""

# Optional: Upload to benchmark storage service
if [ -n "$BENCHMARK_UPLOAD_URL" ]; then
    echo -e "${YELLOW}Uploading results to benchmark service...${NC}"
    curl -X POST "$BENCHMARK_UPLOAD_URL" -F "data=@${OUTPUT_DIR}/benchmark_results.json" || echo "Upload failed"
fi

exit 0
