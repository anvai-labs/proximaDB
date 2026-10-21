#!/bin/bash
#
# ProximaDB Test Runner
# 
# This script runs all tests in the properly organized test packages
#

set -e

echo "🧪 ProximaDB Test Suite Runner"
echo "==============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
RUN_UNIT_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_DOC_TESTS=true
VERBOSE=false
PARALLEL=true
TEST_PATTERN=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit-only)
            RUN_INTEGRATION_TESTS=false
            RUN_DOC_TESTS=false
            shift
            ;;
        --integration-only)
            RUN_UNIT_TESTS=false
            RUN_DOC_TESTS=false
            shift
            ;;
        --doc-only)
            RUN_UNIT_TESTS=false
            RUN_INTEGRATION_TESTS=false
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --sequential)
            PARALLEL=false
            shift
            ;;
        --pattern)
            TEST_PATTERN="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit-only           Run only unit tests"
            echo "  --integration-only    Run only integration tests"
            echo "  --doc-only           Run only documentation tests"
            echo "  --verbose, -v        Enable verbose output"
            echo "  --sequential         Run tests sequentially (not in parallel)"
            echo "  --pattern PATTERN    Run only tests matching pattern"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build the project first
print_status "Building ProximaDB..."
if [[ "$VERBOSE" == "true" ]]; then
    cargo build
else
    cargo build > /dev/null 2>&1
fi

if [[ $? -eq 0 ]]; then
    print_success "Build completed successfully"
else
    print_error "Build failed"
    exit 1
fi

# Test configuration
CARGO_TEST_ARGS=""
if [[ "$VERBOSE" == "true" ]]; then
    CARGO_TEST_ARGS="$CARGO_TEST_ARGS -- --nocapture"
fi

if [[ "$PARALLEL" == "false" ]]; then
    CARGO_TEST_ARGS="$CARGO_TEST_ARGS --test-threads=1"
fi

if [[ -n "$TEST_PATTERN" ]]; then
    CARGO_TEST_ARGS="$CARGO_TEST_ARGS $TEST_PATTERN"
fi

# Track test results
UNIT_TESTS_PASSED=false
INTEGRATION_TESTS_PASSED=false
DOC_TESTS_PASSED=false

# Run unit tests
if [[ "$RUN_UNIT_TESTS" == "true" ]]; then
    print_status "Running unit tests from tests/unit/..."
    
    if cargo test --test unit $CARGO_TEST_ARGS; then
        print_success "Unit tests passed"
        UNIT_TESTS_PASSED=true
    else
        print_error "Unit tests failed"
        UNIT_TESTS_PASSED=false
    fi
    echo ""
fi

# Run integration tests  
if [[ "$RUN_INTEGRATION_TESTS" == "true" ]]; then
    print_status "Running integration tests from tests/integration/..."
    
    # Run each integration test file individually for better error reporting
    INTEGRATION_TEST_FILES=(
        "tests/integration/grpc/test_grpc_vector_operations.rs"
        "tests/integration/rest/test_rest_api_operations.rs"
        "tests/integration/rest/test_rest_vector_search_proto.rs"
        "tests/integration/vector/test_direct_vector_operations.rs"
    )
    
    INTEGRATION_FAILURES=0
    
    for test_file in "${INTEGRATION_TEST_FILES[@]}"; do
        if [[ -f "$test_file" ]]; then
            test_name=$(basename "$test_file" .rs)
            print_status "Running $test_name..."
            
            if cargo test --test "$test_name" $CARGO_TEST_ARGS; then
                print_success "$test_name passed"
            else
                print_error "$test_name failed"
                ((INTEGRATION_FAILURES++))
            fi
        fi
    done
    
    if [[ $INTEGRATION_FAILURES -eq 0 ]]; then
        print_success "All integration tests passed"
        INTEGRATION_TESTS_PASSED=true
    else
        print_error "$INTEGRATION_FAILURES integration test(s) failed"
        INTEGRATION_TESTS_PASSED=false
    fi
    echo ""
fi

# Run documentation tests
if [[ "$RUN_DOC_TESTS" == "true" ]]; then
    print_status "Running documentation tests..."
    
    if cargo test --doc; then
        print_success "Documentation tests passed"
        DOC_TESTS_PASSED=true
    else
        print_error "Documentation tests failed"
        DOC_TESTS_PASSED=false
    fi
    echo ""
fi

# Summary
print_status "Test Summary:"
echo "=============="

if [[ "$RUN_UNIT_TESTS" == "true" ]]; then
    if [[ "$UNIT_TESTS_PASSED" == "true" ]]; then
        print_success "✓ Unit tests"
    else
        print_error "✗ Unit tests"
    fi
fi

if [[ "$RUN_INTEGRATION_TESTS" == "true" ]]; then
    if [[ "$INTEGRATION_TESTS_PASSED" == "true" ]]; then
        print_success "✓ Integration tests"
    else
        print_error "✗ Integration tests"
    fi
fi

if [[ "$RUN_DOC_TESTS" == "true" ]]; then
    if [[ "$DOC_TESTS_PASSED" == "true" ]]; then
        print_success "✓ Documentation tests"
    else
        print_error "✗ Documentation tests"
    fi
fi

# Exit with appropriate code
if [[ ("$RUN_UNIT_TESTS" == "false" || "$UNIT_TESTS_PASSED" == "true") && \
      ("$RUN_INTEGRATION_TESTS" == "false" || "$INTEGRATION_TESTS_PASSED" == "true") && \
      ("$RUN_DOC_TESTS" == "false" || "$DOC_TESTS_PASSED" == "true") ]]; then
    echo ""
    print_success "🎉 All tests passed!"
    exit 0
else
    echo ""
    print_error "❌ Some tests failed!"
    exit 1
fi