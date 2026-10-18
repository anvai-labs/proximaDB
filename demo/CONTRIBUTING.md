# Contributing to ProximaDB Demos

Thank you for your interest in contributing demos to ProximaDB! This guide will help you create high-quality, maintainable demos that showcase ProximaDB features effectively.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Demo Standards](#demo-standards)
- [File Organization](#file-organization)
- [Code Patterns](#code-patterns)
- [Testing Your Demo](#testing-your-demo)
- [Documentation Requirements](#documentation-requirements)
- [Submission Process](#submission-process)

---

## Quick Start

### 1. Setup Development Environment

```bash
# Clone and setup repository
git clone https://github.com/anvai-labs/proximaDB
cd proximaDB

# Install Python SDK in development mode
cd clients/python
pip install -e .
cd ../..

# Set environment variables
export PYTHONPATH=$(pwd)/clients/python/src
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Start ProximaDB server
cargo run --bin proximadb-server
```

### 2. Validate Environment

```bash
# Run health check
python3 demo/check_demo_health.py --verbose

# Should see all checks pass:
# ✅ Python Version: PASS
# ✅ ProximaDB SDK Import: PASS
# ✅ REST Server (port 5678): PASS
# ✅ gRPC Server (port 5679): PASS
```

### 3. Choose Demo Category

Determine where your demo fits:

- **`demo/quickstart/`**: Getting started demos (< 10 lines of code, < 5s runtime)
- **`demo/showcases/features/`**: Feature-specific demos (< 100 lines, < 20s runtime)
- **`demo/showcases/industry/`**: Industry use cases (< 300 lines, < 60s runtime)
- **`demo/showcases/advanced/`**: Advanced topics (no line/time limits)
- **`demo/benchmarks/`**: Performance testing and comparisons

---

## Demo Standards

### Naming Convention

```python
# File naming pattern
{feature_name}_demo.py         # Good: chunking_demo.py
{use_case}_example.py          # Good: ecommerce_example.py
test_{feature}.py              # Bad: This is for tests, not demos

# Collection naming pattern
demo_{feature}_{timestamp}     # Good: demo_chunking_1234567890
test_collection                # Bad: Confusing with test code
```

### Runtime Requirements

| Category | Max Runtime | Max Lines | Complexity |
|----------|-------------|-----------|------------|
| Quickstart | 5s | 50 | Simple |
| Feature Showcase | 20s | 150 | Moderate |
| Industry Use Case | 60s | 300 | Complex |
| Advanced | None | None | Very Complex |

### Code Quality

Your demo must:
- ✅ Pass `python3 -m py_compile your_demo.py` (syntax check)
- ✅ Run successfully with `timeout 60 python3 your_demo.py`
- ✅ Clean up all resources (collections, temporary files)
- ✅ Include proper error handling
- ✅ Use current SDK methods (not deprecated APIs)
- ✅ Follow PEP 8 style guidelines

---

## File Organization

### Demo File Structure

```python
#!/usr/bin/env python3
"""
{Feature Name} Demo

This demo shows how to use {feature} in ProximaDB.

Prerequisites:
    - ProximaDB server running on localhost:5678 (REST) or localhost:5679 (gRPC)
    - Python 3.8+
    - ProximaDB Python SDK installed

Usage:
    export PYTHONPATH=./clients/python/src
    python3 demo/{category}/{your_demo}.py

Expected Output:
    - Collection created with {config}
    - {N} vectors inserted
    - Search results showing {metric}
    - Cleanup confirmation

Duration: ~{X} seconds
"""

import sys
from pathlib import Path

# Add SDK to path if needed
sdk_path = Path(__file__).resolve().parents[2] / "clients" / "python" / "src"
if sdk_path.exists():
    sys.path.insert(0, str(sdk_path))

from proximadb import ProximaDBClient, CollectionConfig, DistanceMetric, StorageEngine
from proximadb.models import VectorRecord

# Configuration constants
SERVER_URL = "http://localhost:5678"
COLLECTION_NAME = "demo_your_feature"
DIMENSION = 128


def setup():
    """Setup demo environment and resources"""
    print("=" * 70)
    print("  ProximaDB {Feature} Demo")
    print("=" * 70)
    print()

    # Create client
    client = ProximaDBClient(url=SERVER_URL, protocol="rest")

    # Clean up existing collection
    print("1. Cleaning up existing collections...")
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   ✓ Deleted existing collection: {COLLECTION_NAME}")
    except:
        print(f"   ✓ No existing collection to clean up")

    return client


def create_collection(client):
    """Create collection with appropriate configuration"""
    print("\n2. Creating collection...")

    config = CollectionConfig(
        name=COLLECTION_NAME,
        dimension=DIMENSION,
        distance_metric=DistanceMetric.COSINE,
        storage_engine=StorageEngine.VIPER
    )

    collection = client.create_collection(COLLECTION_NAME, config)
    print(f"   ✓ Collection created: {collection.id}")
    print(f"   ✓ Dimension: {DIMENSION}")
    print(f"   ✓ Distance metric: {config.distance_metric}")

    return collection


def demonstrate_feature(client):
    """
    Main demo logic showing the feature

    This should be the core of your demo - clear, concise examples
    of the feature in action.
    """
    print("\n3. Demonstrating {feature}...")

    # Your demo code here
    # - Keep it focused on one feature
    # - Show practical usage
    # - Include output/results

    print("   ✓ Feature demonstrated successfully")


def cleanup(client):
    """Clean up resources"""
    print("\n4. Cleaning up...")
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   ✓ Deleted collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")


def main():
    """Main entry point"""
    try:
        client = setup()
        collection = create_collection(client)
        demonstrate_feature(client)
        cleanup(client)

        print("\n" + "=" * 70)
        print("✅ Demo completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## Code Patterns

### 1. Collection Cleanup (Required)

Always clean up collections before and after your demo:

```python
def setup():
    """Setup with cleanup"""
    client = ProximaDBClient(url=SERVER_URL)

    # Clean up existing collection from previous runs
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass  # Collection doesn't exist - OK

    return client

def cleanup(client):
    """Final cleanup"""
    try:
        client.delete_collection(COLLECTION_NAME)
        print("✓ Cleanup complete")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
```

### 2. Error Handling (Required)

Wrap operations in try/except blocks:

```python
try:
    results = client.search(
        collection_id=COLLECTION_NAME,
        vector=query_vector,
        top_k=10
    )
    print(f"✓ Found {len(results)} results")
except Exception as e:
    print(f"❌ Search failed: {e}")
    # Don't crash - show error and continue or exit gracefully
```

### 3. Resource Management (Required)

Use try/finally for guaranteed cleanup:

```python
def main():
    client = None
    try:
        client = ProximaDBClient(url=SERVER_URL)
        # ... demo code ...
    finally:
        if client:
            cleanup(client)
```

### 4. Progressive Output (Recommended)

Show progress to the user:

```python
print("\n1. Creating collection...")
# ... code ...
print("   ✓ Collection created")

print("\n2. Inserting vectors...")
# ... code ...
print(f"   ✓ Inserted {count} vectors")

print("\n3. Running search...")
# ... code ...
print(f"   ✓ Search completed in {elapsed:.2f}s")
```

### 5. Configuration Constants (Required)

Define all configuration at the top:

```python
# Configuration
SERVER_URL = "http://localhost:5678"  # or "grpc://localhost:5679"
COLLECTION_NAME = "demo_your_feature"
DIMENSION = 128
NUM_VECTORS = 1000
BATCH_SIZE = 100
```

### 6. Protocol Selection

Specify which protocol your demo uses:

```python
# For REST (default, recommended)
client = ProximaDBClient(url="http://localhost:5678", protocol="rest")

# For gRPC (specify when needed)
client = ProximaDBClient(url="grpc://localhost:5679", protocol="grpc")
```

### 7. Vector Generation

Use simple, clear patterns:

```python
import numpy as np

# Random vectors
vectors = []
for i in range(NUM_VECTORS):
    vector = VectorRecord(
        id=f"vec_{i}",
        vector=np.random.randn(DIMENSION).tolist(),
        metadata={"index": i, "category": "demo"}
    )
    vectors.append(vector)

# Or use numpy directly
vector_data = np.random.randn(NUM_VECTORS, DIMENSION).astype(np.float32)
```

---

## Testing Your Demo

### Local Testing

```bash
# 1. Syntax check
python3 -m py_compile demo/{category}/{your_demo}.py

# 2. Run demo
export PYTHONPATH=./clients/python/src
python3 demo/{category}/{your_demo}.py

# 3. Verify cleanup (no collections left behind)
# ... manually check or use client.list_collections()

# 4. Run with timeout
timeout 60 python3 demo/{category}/{your_demo}.py
```

### Using Test Infrastructure

```bash
# Run health check
python3 demo/check_demo_health.py

# Run all demos including yours
./demo/run_all_demos.sh --verbose
```

### Validation Checklist

Before submitting, verify:

- [ ] Demo runs successfully without errors
- [ ] Completes within expected time limit for category
- [ ] Cleans up all resources (collections, files)
- [ ] Works with both fresh and existing environments
- [ ] Outputs clear, informative messages
- [ ] Handles errors gracefully (no crashes)
- [ ] Uses current SDK APIs (no deprecated methods)
- [ ] Follows code patterns from this guide
- [ ] Includes proper docstring and comments

---

## Documentation Requirements

### File Header (Required)

Every demo must have a comprehensive docstring:

```python
#!/usr/bin/env python3
"""
{Feature Name} Demo

Brief description of what this demo shows (1-2 sentences).

This demo demonstrates:
    - Feature 1: Brief description
    - Feature 2: Brief description
    - Feature 3: Brief description

Prerequisites:
    - ProximaDB server running (REST: localhost:5678, gRPC: localhost:5679)
    - Python 3.8 or higher
    - ProximaDB Python SDK installed: pip install -e clients/python

Environment Setup:
    export PYTHONPATH=./clients/python/src
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

Usage:
    python3 demo/{category}/{your_demo}.py

Expected Output:
    - Collection "{collection_name}" created with {dimension} dimensions
    - {N} vectors inserted
    - Search results showing {metric} similarity
    - Cleanup confirmation message

Duration: ~{X} seconds
Exit Code: 0 on success, 1 on failure

Related Demos:
    - demo/{category}/{related_demo}.py - Related feature demonstration

Documentation:
    - See demo/README.md for environment setup
    - See docs/{relevant_doc}.adoc for feature details
"""
```

### Inline Comments (Recommended)

Add comments for complex logic:

```python
# Calculate cosine similarity between query and results
# This shows the quality of the search results
similarities = []
for result in results:
    # Dot product of normalized vectors
    similarity = np.dot(query_vector, result.vector)
    similarities.append(similarity)
```

### README Updates (Required)

Add your demo to `demo/README.md`:

```markdown
## Running Demos

### Feature Demos

**Your Feature** (`your_feature_demo.py`):
```bash
export PYTHONPATH=./clients/python/src
python3 demo/showcases/features/your_feature_demo.py
```
- Duration: ~X seconds
- Coverage: Brief description of what it demonstrates
- Prerequisites: Any special requirements
```

---

## Submission Process

### 1. Prepare Your Branch

```bash
# Create feature branch
git checkout -b demo/your-feature-name

# Add your demo file
git add demo/{category}/{your_demo}.py

# Update README
git add demo/README.md

# Commit with descriptive message
git commit -m "Add {feature} demo showcasing {functionality}

- Demonstrates {key feature 1}
- Shows {key feature 2}
- Runtime: ~{X} seconds
- Tested on: REST/gRPC protocol
"
```

### 2. Pre-Submission Checklist

- [ ] Demo runs successfully (`python3 demo/{category}/{your_demo}.py`)
- [ ] Passes syntax check (`python3 -m py_compile ...`)
- [ ] Completes within time limit
- [ ] Cleans up all resources
- [ ] File header docstring complete
- [ ] README.md updated with demo entry
- [ ] Follows code patterns from this guide
- [ ] No hardcoded paths (use Path or relative paths)
- [ ] No credentials or secrets in code

### 3. Testing Commands

Run these commands before submitting:

```bash
# Validate environment
python3 demo/check_demo_health.py

# Test your demo
export PYTHONPATH=./clients/python/src
timeout 60 python3 demo/{category}/{your_demo}.py

# Test with all demos
./demo/run_all_demos.sh --verbose
```

### 4. Create Pull Request

1. Push your branch: `git push origin demo/your-feature-name`
2. Create PR with description:
   - What feature does this demo showcase?
   - What makes it useful?
   - Runtime and resource requirements
   - Screenshots or example output (optional)

### 5. PR Review Criteria

Reviewers will check:
- Code quality and style
- Follows demo patterns
- Proper cleanup and error handling
- Clear documentation
- Runs successfully in CI/CD
- Adds value to demo collection

---

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError

```python
# Problem: Cannot import proximadb
# Solution: Set PYTHONPATH correctly
export PYTHONPATH=./clients/python/src

# Or add to demo file:
import sys
from pathlib import Path
sdk_path = Path(__file__).resolve().parents[2] / "clients" / "python" / "src"
sys.path.insert(0, str(sdk_path))
```

### Issue 2: Connection Refused

```python
# Problem: Cannot connect to server
# Solution: Check server is running
curl http://localhost:5678/health

# If not running:
cargo run --bin proximadb-server
```

### Issue 3: Collection Already Exists

```python
# Problem: COLLECTION_EXISTS error on repeated runs
# Solution: Add cleanup before creating
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass  # Collection doesn't exist - OK
```

### Issue 4: Demo Timeout

```python
# Problem: Demo takes too long, times out in CI/CD
# Solutions:
# 1. Reduce dataset size for quickstart/feature demos
# 2. Move to advanced/ category if inherently slow
# 3. Add progress indicators
# 4. Optimize queries (use indexes, quantization)
```

### Issue 5: Inconsistent Results

```python
# Problem: Demo output varies between runs
# Solution: Use seeded random for reproducibility
import numpy as np
np.random.seed(42)  # Fixed seed for consistent results
```

---

## Demo Quality Guidelines

### Excellent Demo Characteristics

1. **Focused**: Demonstrates one feature clearly
2. **Practical**: Shows real-world usage patterns
3. **Clear**: Easy to understand what's happening
4. **Complete**: Includes setup, demo, cleanup
5. **Resilient**: Handles errors gracefully
6. **Documented**: Clear docstring and comments
7. **Efficient**: Runs quickly within category limits

### Example: High-Quality Demo Structure

```python
"""
Clear, comprehensive docstring explaining:
- What the demo does
- Prerequisites
- Expected output
- Runtime
"""

# Configuration section (all constants at top)
SERVER_URL = "http://localhost:5678"
DIMENSION = 128

def setup():
    """Setup with cleanup"""
    # ... setup code ...

def demonstrate_core_feature():
    """Focused demonstration of one feature"""
    # ... clear, well-commented demo code ...

def cleanup():
    """Guaranteed cleanup"""
    # ... cleanup code ...

def main():
    """Main entry with error handling"""
    try:
        setup()
        demonstrate_core_feature()
        cleanup()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Getting Help

### Resources

- **Demo README**: `demo/README.md` - Environment setup and troubleshooting
- **API Documentation**: `docs/03-api-reference/index.md`
- **Example Demos**: See `demo/quickstart/basic_demo.py` for simple example

### Questions?

- Open an issue: https://github.com/anvai-labs/proximaDB/issues
- Check existing demos for patterns
- Run health check: `python3 demo/check_demo_health.py --verbose`

---

## Thank You!

Thank you for contributing to ProximaDB demos! Quality demos help users understand and adopt ProximaDB features effectively.

**Happy coding!**
