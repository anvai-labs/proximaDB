#!/bin/bash
# Build base image with all dependencies (run once or when dependencies change)

set -e

echo "🏗️ Building ProximaDB base image with all dependencies..."

# Create temporary Dockerfile for base image
cat > Dockerfile.base << 'EOF'
# Base image with all dependencies pre-installed  
FROM rust:1.100.0-slim as rust-base

# Install Rust build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev protobuf-compiler cmake build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock build.rs ./
COPY proto/ ./proto/

# Create dummy source and build dependencies only
RUN mkdir -p src/bin src/proto && \
    echo 'fn main() {}' > src/bin/server.rs && \
    echo 'fn main() {}' > src/lib.rs
COPY src/proto/proximadb_descriptor.bin ./src/proto/
RUN cargo build --release --bin proximadb-server && \
    rm -rf src target/release/deps/proximadb*

# Python base with all dependencies
FROM python:3.11-slim as python-base

RUN apt-get update && apt-get install -y \
    ca-certificates curl gcc g++ git supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install all Python dependencies
RUN pip install --no-cache-dir \
    numpy requests asyncio-mqtt nltk spacy sentence-transformers \
    pandas scikit-learn matplotlib seaborn tqdm click \
    aiohttp aiohttp-cors httpx orjson structlog \
    pytest pytest-asyncio black isort mypy \
    torch transformers tokenizers supervisor

# Pre-download BERT models
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    SentenceTransformer('all-mpnet-base-v2')"

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True)" || true

# Create application structure
RUN mkdir -p /opt/proximadb/{bin,config} /data /app && \
    useradd -m -u 1001 proximadb && \
    chown -R proximadb:proximadb /opt/proximadb /data /app

ENV RUST_LOG=info PYTHONUNBUFFERED=1
WORKDIR /app
EOF

# Build the base images
echo "🔨 Building Rust dependencies base..."
docker build -f Dockerfile.base --target rust-base -t proximadb-rust-base:latest .

echo "🐍 Building Python dependencies base..."  
docker build -f Dockerfile.base --target python-base -t proximadb-python-base:latest .

# Create combined base
cat > Dockerfile.combined-base << 'EOF'
FROM proximadb-python-base:latest
COPY --from=proximadb-rust-base:latest /build /build
EOF

echo "🎯 Building combined base image..."
docker build -f Dockerfile.combined-base -t proximadb-base:latest .

# Cleanup
rm -f Dockerfile.base Dockerfile.combined-base

echo "✅ Base images built successfully!"
echo "💡 Now use: docker build . for builds (will reuse base layers)"
