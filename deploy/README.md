# ProximaDB Deployment Guide

This directory contains deployment configurations for ProximaDB across multiple platforms and cloud providers.

## Directory Structure

```
deploy/
├── docker/              # Docker and Docker Compose configurations
│   ├── Dockerfile       # Multi-stage production build (~150MB)
│   ├── Dockerfile.alpine # Ultra-minimal Alpine build (~80MB)
│   ├── docker-compose.yml # Local development stack
│   └── prometheus.yml   # Prometheus configuration
├── helm/
│   ├── proximadb/       # Canonical Helm chart for Kubernetes
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/
│   ├── proximadb-operator/ # Operator chart
│   └── legacy-proximadb/   # Retained legacy chart values
├── infrastructure/      # Full cloud infrastructure examples
│   ├── terraform/       # Environment-specific Terraform stacks
│   ├── helm/            # Infrastructure Helm chart variant
│   ├── monitoring/      # Prometheus/Grafana examples
│   └── scripts/         # Infrastructure deployment helpers
├── k8s/                 # Versioned enterprise Kubernetes release manifests
├── packaging/           # Package scripts, systemd units, and WiX assets
├── kubernetes/          # Plain Kubernetes manifests (no Helm)
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kustomization.yaml
├── systemd/             # Legacy root-level systemd units retained for compatibility
└── terraform/
    ├── modules/
    │   └── proximadb/   # Reusable Terraform module
    ├── aws/             # AWS EKS deployment
    ├── azure/           # Azure AKS deployment
    └── gcp/             # GCP GKE deployment
```

Release and deployment assets should stay under `deploy/`. Avoid adding new root-level
`helm/`, `k8s/`, `deployment/`, `packaging/`, or `infrastructure/` directories.

## Quick Start

### Docker (Fastest)

```bash
# Pull and run
# The built-in config keeps data in /tmp inside the container (TD-PKG-3); mount a config whose
# roots point at the volume:
docker run --rm --entrypoint cat vjsingh1984/proximadb:latest /config/config.toml \
  | sed 's#/tmp/proximadb#/data/proximadb#g' > proximadb.toml
docker run -d --name proximadb \
  -p 5678:5678 \
  -v proximadb-data:/data/proximadb \
  -v "$PWD/proximadb.toml:/config/config.toml:ro" \
  vjsingh1984/proximadb:latest

# Or build locally (from the repository root, where proximadb.toml was created). The repository
# has no .dockerignore, so the build context includes target/ and .git; in a working checkout,
# run `cargo clean` first or expect a very large context.
docker build -t proximadb:local -f deploy/docker/Dockerfile .
docker run -d --name proximadb -p 5678:5678 -v proximadb-data:/data/proximadb \
  -v "$PWD/proximadb.toml:/config/config.toml:ro" proximadb:local
```

> **pgwire (5433) is not published, and publishing it would not help.** The entrypoint
> rewrites `[server] bind_address` to `0.0.0.0` so published ports are reachable, but
> pgwire is deliberately exempt: the shipped config leaves `[api] pg_port` unset, so the
> listener stays on `127.0.0.1:5433` *inside* the container and `-p 5433:5433` publishes a
> port that accepts no connections. See
> [Reaching pgwire](#postgresql-wire-protocol-pgwire-is-loopback-only-in-the-container).

### Docker Compose (Development)

`deploy/docker/docker-compose.yml` mounts `config/config.toml`, whose storage roots are under
`/tmp/proximadb`, so this stack keeps no data across `down`/`up` (TD-PKG-3). Replace that mount
with a `proximadb.toml` created as above for data that must survive.

```bash
cd deploy/docker
docker-compose up -d

# With monitoring stack
docker-compose --profile monitoring up -d

# Access:
# - Unified REST + gRPC + Arrow Flight: localhost:5678
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
#
# NOT localhost:5433 — pgwire listens on 127.0.0.1:5433 inside the container, so
# Compose deliberately does not publish it. See "PostgreSQL wire protocol" below.
```

### Kubernetes with Helm

```bash
# Add repository (when published)
helm repo add proximadb https://charts.proximadb.io
helm repo update

# Or install from local chart
cd deploy/helm
helm install proximadb ./proximadb \
  --namespace proximadb \
  --create-namespace

# With custom values
helm install proximadb ./proximadb \
  --namespace proximadb \
  --create-namespace \
  -f custom-values.yaml

# Upgrade
helm upgrade proximadb ./proximadb --namespace proximadb

# Uninstall
helm uninstall proximadb --namespace proximadb
```

### Kubernetes with Plain Manifests (no Helm)

```bash
# Using kubectl
cd deploy/kubernetes
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Or using kustomize
kubectl apply -k deploy/kubernetes/

# Verify deployment
kubectl get pods -n proximadb
kubectl get svc -n proximadb

# Port forward for testing
kubectl port-forward -n proximadb svc/proximadb 5678:5678
```

### Terraform (Cloud Deployments)

#### AWS EKS

```bash
cd deploy/terraform/aws

# Initialize
terraform init

# Plan
terraform plan \
  -var="region=us-west-2" \
  -var="cluster_name=proximadb-prod"

# Apply
terraform apply \
  -var="region=us-west-2" \
  -var="cluster_name=proximadb-prod"

# Get kubeconfig
aws eks update-kubeconfig \
  --region us-west-2 \
  --name proximadb-prod
```

#### Azure AKS

```bash
cd deploy/terraform/azure

terraform init
terraform plan \
  -var="location=eastus" \
  -var="cluster_name=proximadb-prod"

terraform apply \
  -var="location=eastus" \
  -var="cluster_name=proximadb-prod"

# Get kubeconfig
az aks get-credentials \
  --resource-group proximadb-rg \
  --name proximadb-prod
```

#### GCP GKE

```bash
cd deploy/terraform/gcp

terraform init
terraform plan \
  -var="project_id=my-gcp-project" \
  -var="region=us-central1"

terraform apply \
  -var="project_id=my-gcp-project" \
  -var="region=us-central1"

# Get kubeconfig
gcloud container clusters get-credentials proximadb-gke \
  --zone us-central1-a \
  --project my-gcp-project
```

## Configuration

### Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 5678 | HTTP/gRPC/Arrow | Unified multiplexed surface (REST + gRPC + Arrow Flight; `unified_mode = true` in the shipped config.toml/docker image — the code default is false, and shipped k8s/helm artifacts run multi-port) |
| 5433 | PostgreSQL wire | SQL clients (pgvector-compatible). **In-container: loopback only** — reachable from a locally-run binary or from inside the container's network namespace, not through a published port. See below. |
| 5679 / 5680 | gRPC / Arrow IPC | Legacy multi-port mode only (`unified_mode = false`) |
| 9090 | HTTP | Metrics (Prometheus) |

### PostgreSQL wire protocol (pgwire) is loopback-only in the container

pgwire runs in the published image — it is not disabled — but it listens on
`127.0.0.1:5433` inside the container, so there is nothing for a published host port to
reach. This is deliberate containment, not a defect.

**Why.** `deploy/docker/entrypoint.sh` rewrites `[server] bind_address` to `0.0.0.0`
because `docker run -p` cannot reach a loopback listener. `src/database.rs` re-derives the
pgwire bind from `bind_address` *only* when `[api] pg_port` is set, and the shipped
`config/config.toml` deliberately leaves it unset. Setting it moves pgwire to
`0.0.0.0:5433`, which `validate_trust_only_listener`
(`crates/platform/proximadb-runtime/src/bootstrap_config.rs`) refuses before binding any
socket: with `[security]` off the listener is trust-mode, and an unauthenticated
single-tenant listener is admitted on **loopback only**. The server then exits 1 and the
whole container dies rather than losing one port.

**Where pgwire does work today**

```bash
# 1. A locally-run binary — loopback is the host's loopback.
cargo run --release --bin proximadb-server
psql -h 127.0.0.1 -p 5433 -U postgres -d proximadb

# 2. From inside the container's network namespace. The image ships no psql client,
#    so attach one that shares the namespace (and therefore its loopback):
docker run --rm -it --network container:proximadb postgres:16 \
  psql -h 127.0.0.1 -p 5433 -U postgres -d proximadb
```

**Serving it from the host** requires per-connection authentication on the listener
(pgwire SCRAM, `TD-PGWIRE-AUTH-1`), after which the loopback containment no longer
applies:

1. enable `[security]` with `[security.authentication] enabled = true` (or
   `[security.pgwire] auth = "password"`) and provision
   `[security.authentication.scram_users]`;
2. set `[api] pg_port = 5433` so the pgwire bind follows `[server] bind_address`;
3. publish the port (`-p 5433:5433`, or a Compose/Service entry).

Step 2 without step 1 is **fail-closed, not degraded**: the boot-time listener preflight
refuses the listener and the server exits 1. Tracked in
`docs/10-quality/td/TD-PGWIRE-CONTAINER-1-pgwire-unreachable-in-a-container-without-auth.adoc`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXIMADB_BIND_ADDRESS` | `0.0.0.0` | Listen address |
| `PROXIMADB_REST_PORT` | `5678` | Not read by the server; set the port in `config.toml` |
| `PROXIMADB_GRPC_PORT` | `5679` | Not read by the server |
| `PROXIMADB_ARROW_IPC_PORT` | `5680` | Not read by the server |
| `PROXIMADB_METRICS_PORT` | `9090` | Not read by the server |
| `PROXIMADB_DATA_DIR` | `/data/proximadb` | Not read by the config loader; set the storage roots in `config.toml` (TD-PKG-3) |
| `PROXIMADB_DEFAULT_ENGINE` | — | Not read by the server; choose the engine per collection (`"engine": "sst"`, TD-ENGSEL-1) |
| `PROXIMADB_WAL_ENABLED` | `true` | Not read by the server |
| `RUST_LOG` | `info` | Log level |

### Storage Engines

| Engine | Best For | Performance |
|--------|----------|-------------|
| `sst` | Write-optimized, real-time | ~5ms (10K vectors) |
| `helix` | Locality-optimized | ~13ms |
| `viper` | Columnar analytics | ~90ms |
| `nova` | Progressive columnar | ~100ms |

`swift` and `raptor` remain behind `experimental-engines` as deprecated research engines. Do not use them for new production deployments.

## Production Recommendations

### Resource Sizing

| Workload | CPU | Memory | Storage |
|----------|-----|--------|---------|
| Small (<100K vectors) | 2 cores | 4GB | 50GB SSD |
| Medium (<1M vectors) | 4 cores | 16GB | 200GB SSD |
| Large (<10M vectors) | 8 cores | 32GB | 500GB NVMe |
| XLarge (>10M vectors) | 16+ cores | 64GB+ | 1TB+ NVMe |

### High Availability

For production deployments:

```yaml
# Helm values for HA
replicaCount: 3

podDisruptionBudget:
  enabled: true
  minAvailable: 2

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPU: 70
  targetMemory: 80

persistence:
  storageClass: "gp3"  # AWS
  size: 200Gi

resources:
  requests:
    cpu: 2000m
    memory: 8Gi
  limits:
    cpu: 4000m
    memory: 16Gi
```

### Security

1. **Network Policies**: Enable in Helm values
2. **Pod Security**: Runs as non-root by default
3. **Secrets**: Use Kubernetes secrets for sensitive config
4. **TLS**: Configure ingress with TLS certificates
5. **RBAC**: Service account with minimal permissions

### Monitoring

Enable ServiceMonitor for Prometheus:

```yaml
serviceMonitor:
  enabled: true
  interval: 15s
  labels:
    release: prometheus
```

## Marketplace Deployments

### AWS Marketplace

Coming soon - EKS add-on and AMI offerings.

### Azure Marketplace

Coming soon - AKS extension and VM offerings.

### GCP Marketplace

Coming soon - GKE application and VM offerings.

## Troubleshooting

### Common Issues

1. **Pod not starting**: Check resource limits and PVC status
2. **Connection refused**: Verify service ports and network policies
3. **Slow performance**: Check storage class (use SSD/NVMe)
4. **Out of memory**: Increase memory limits or use smaller batches

### Debug Commands

```bash
# Check pod status
kubectl get pods -n proximadb -l app.kubernetes.io/name=proximadb

# View logs
kubectl logs -n proximadb -l app.kubernetes.io/name=proximadb --tail=100

# Check persistence
kubectl get pvc -n proximadb

# Test connectivity
kubectl port-forward -n proximadb svc/proximadb 5678:5678
curl http://localhost:5678/health
```

## Support

- Documentation: https://docs.proximadb.io
- Issues: https://github.com/anvai-labs/proximaDB/issues
- Discussions: https://github.com/anvai-labs/proximaDB/discussions
