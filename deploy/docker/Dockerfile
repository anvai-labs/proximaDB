# ProximaDB Production Dockerfile
# Multi-stage build for minimal image size (~150MB target)
# Supports AWS EKS, Azure AKS, and Google GKE deployments

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM rust:1.95-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create dummy source for dependency caching
RUN mkdir -p src/bin && \
    echo "fn main() {}" > src/bin/server.rs && \
    echo "pub fn lib() {}" > src/lib.rs

# Build dependencies only (cached layer)
RUN RUSTFLAGS="-A unsafe_op_in_unsafe_fn" cargo build -p proximadb-server --release --features cloud-full --bin proximadb-server 2>/dev/null || true

# Copy actual source code. Workspace members (apps/, clients/, crates/)
# are required for cargo to resolve the workspace — Cargo.toml at /build
# declares them and refuses to build any single member otherwise.
COPY src/ src/
COPY proto/ proto/
COPY config/ config/
COPY apps/ apps/
COPY clients/ clients/
COPY crates/ crates/

# Build the actual binary (with cloud object_store backends: S3/ADLS/GCS).
RUN RUSTFLAGS="-A unsafe_op_in_unsafe_fn" cargo build -p proximadb-server --release --features cloud-full --bin proximadb-server

# Strip debug symbols for smaller binary
RUN strip /build/target/release/proximadb-server

# ============================================================================
# Stage 1b: Full builder — real ONNX inference (--features onnx)
# ----------------------------------------------------------------------------
# Used only for the `:*-full` test image (build with --target runtime-full).
# `ort` downloads a prebuilt ONNX Runtime at build time (needs network) and
# static-links it into the binary.
#
# Base is Debian TRIXIE, not bookworm: the pyke `ort` prebuilt onnxruntime is
# compiled against a newer glibc/GCC and references symbols absent on bookworm
# (e.g. __isoc23_strtoull → glibc 2.38+, __cxa_call_terminate → newer
# libstdc++), so it cannot link on bookworm's glibc 2.36. trixie ships glibc
# 2.41 / GCC 14. This is independent of the minimal builder — the default image
# stays on bookworm. g++ provides libstdc++ for the C++ onnxruntime link.
# ============================================================================
FROM rust:1.95-slim-trixie AS builder-full
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    cmake \
    protobuf-compiler \
    libprotobuf-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY proto/ proto/
COPY config/ config/
COPY apps/ apps/
COPY clients/ clients/
COPY crates/ crates/
RUN RUSTFLAGS="-A unsafe_op_in_unsafe_fn" cargo build -p proximadb-server --release --features cloud-full,onnx --bin proximadb-server \
 && strip /build/target/release/proximadb-server

# ============================================================================
# Stage 1c: Model fetch — bake bge-small-en-v1.5 (MIT) for the full image
# ----------------------------------------------------------------------------
# bge-small only (384-dim, ~127MB). bge-large / bge-m3 are intentionally NOT
# baked (too heavy) — make them opt-in via the entrypoint model-fetch env vars.
# Pinned by sha256 (ADR-0016): a content change at the source fails the build.
# ============================================================================
FROM debian:bookworm-slim AS models
ARG BGE_ONNX_URL="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model.onnx"
ARG BGE_ONNX_SHA256="828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
ARG BGE_TOKENIZER_URL="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/tokenizer.json"
ARG BGE_TOKENIZER_SHA256="d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /models \
 && curl -fsSL "$BGE_ONNX_URL" -o /models/bge-small-en-v1.5.onnx \
 && echo "${BGE_ONNX_SHA256}  /models/bge-small-en-v1.5.onnx" | sha256sum -c - \
 && curl -fsSL "$BGE_TOKENIZER_URL" -o /models/tokenizer.json \
 && echo "${BGE_TOKENIZER_SHA256}  /models/tokenizer.json" | sha256sum -c -

# ============================================================================
# Stage 2b: Full runtime — onnx server binary + baked bge-small models
# ----------------------------------------------------------------------------
# Separate tag (e.g. :<sha>-full). Same hardened debian-slim runtime as the
# minimal image; adds the ONNX-enabled binary and the baked models.
# ============================================================================
# Trixie runtime to match the trixie-built onnx binary (glibc 2.41). libssl3t64
# is trixie's openssl 3 runtime (post-t64 rename); libstdc++6 for onnxruntime.
FROM debian:trixie-slim AS runtime-full
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3t64 \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
RUN groupadd -r proximadb && useradd -r -g proximadb proximadb
RUN mkdir -p /data/proximadb /config /log /var/lib/proximadb/models && \
    chown -R proximadb:proximadb /data /config /log /var/lib/proximadb
COPY --from=builder-full /build/target/release/proximadb-server /usr/local/bin/proximadb-server
# Config from builder-full (not the bookworm `builder`) so this target doesn't
# pull the minimal builder's full compile into the graph — only builder-full builds.
COPY --from=builder-full /build/config/config.toml /config/config.toml
COPY --from=models --chown=proximadb:proximadb /models/ /var/lib/proximadb/models/
RUN chown proximadb:proximadb /usr/local/bin/proximadb-server
USER proximadb
ENV RUST_LOG=info \
    PROXIMADB_DATA_DIR=/data/proximadb \
    PROXIMADB_BIND_ADDRESS=0.0.0.0 \
    PROXIMADB_REST_PORT=5678 \
    PROXIMADB_GRPC_PORT=5679 \
    PROXIMADB_ARROW_IPC_PORT=5680 \
    PROXIMADB_METRICS_PORT=9090 \
    PROXIMADB_EMBED_MODEL_DIR=/var/lib/proximadb/models
EXPOSE 5678 5679 5680 5433
# NOTE: a HEALTHCHECK runs INSIDE the container, so it passes even when the
# server binds loopback only and the published port is unreachable from the host.
# It is structurally blind to that class of failure -- which is why a loopback
# bind survived here unnoticed; CI's host-side `docker run -p 5678:5678` + curl
# is what catches it. The entrypoint normalises bind_address to 0.0.0.0 for
# exactly this reason.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5678/health || exit 1
VOLUME ["/data/proximadb"]
WORKDIR /data/proximadb
COPY --chmod=0755 deploy/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--config", "/config/config.toml"]

# ============================================================================
# Stage 2: Runtime (minimal — DEFAULT target)
# ============================================================================
FROM debian:bookworm-slim AS runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r proximadb && useradd -r -g proximadb proximadb

# Create data + writable runtime directories. /log and /var/lib/proximadb are
# default paths the server writes (log appender + model/tokenizer dirs); they
# must exist and be owned by the non-root user or the server panics creating
# them on first boot.
RUN mkdir -p /data/proximadb /config /log /var/lib/proximadb && \
    chown -R proximadb:proximadb /data /config /log /var/lib/proximadb

# Copy binary from builder
COPY --from=builder /build/target/release/proximadb-server /usr/local/bin/proximadb-server

# Copy default configuration
COPY --from=builder /build/config/config.toml /config/config.toml

# Set ownership
RUN chown proximadb:proximadb /usr/local/bin/proximadb-server

# Switch to non-root user
USER proximadb

# Environment variables
ENV RUST_LOG=info \
    PROXIMADB_DATA_DIR=/data/proximadb \
    PROXIMADB_BIND_ADDRESS=0.0.0.0 \
    PROXIMADB_REST_PORT=5678 \
    PROXIMADB_GRPC_PORT=5679 \
    PROXIMADB_ARROW_IPC_PORT=5680 \
    PROXIMADB_METRICS_PORT=9090

# Expose ports
# 5678: REST API (Prometheus metrics also served here at /metrics)
# 5679: gRPC API
# 5680: Arrow Flight (bulk ingest, columnar zero-copy)
# 5433: PostgreSQL wire protocol (pgwire) — pgvector / psql / pgAdmin
#       compatibility; enabled by default per PostgresServerConfig
EXPOSE 5678 5679 5680 5433

# Health check
# NOTE: a HEALTHCHECK runs INSIDE the container, so it passes even when the
# server binds loopback only and the published port is unreachable from the host.
# It is structurally blind to that class of failure -- which is why a loopback
# bind survived here unnoticed; CI's host-side `docker run -p 5678:5678` + curl
# is what catches it. The entrypoint normalises bind_address to 0.0.0.0 for
# exactly this reason.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5678/health || exit 1

# Volume for persistent data
VOLUME ["/data/proximadb"]

# Working directory
WORKDIR /data/proximadb

# Entrypoint: optionally fetches a runtime tier-config overlay from
# PROXIMADB_TIER_CONFIG_URL at boot, then execs the server. Fallback:
# if the fetch fails (offline, network error, no URL configured), the
# baked-in /config/tier-config.json wins — pod never crashes on
# network blips. See deploy/docker/entrypoint.sh for the full fetch +
# fallback logic; engine reads /config/tier-config.json at startup
# (Phase B-5; src/catalog/tenant_tier.rs::resolve_tier_config_source).
# COPY with the exec bit set at copy time. A separate `RUN chmod` here would
# run as the non-root `proximadb` user (set above) against a root-owned file
# and fail — so set the mode during COPY instead.
COPY --chmod=0755 deploy/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--config", "/config/config.toml"]
