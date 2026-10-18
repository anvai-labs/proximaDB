# Quick Start

**Get ProximaDB running in 5 minutes**

```mermaid
%%{init: {"theme": "neutral"}}%%
flowchart LR
  A[1. Install] --> B[2. Start]
  B --> C[3. Create Collection]
  C --> D[4. Insert Data]
  D --> E[5. Query]

  style A fill:#e8f5e9
  style B fill:#e8f5e9
  style C fill:#e8f5e9
  style D fill:#e8f5e9
  style E fill:#e8f5e9
```

---

## Choose Your Installation Method

| Method | OS | Time | Best For |
|--------|-------|------|----------|
| [Platform Package](./install.md#platform-packages) | Linux, Windows | 2 min | Servers (see the known issues in the guide) |
| [Docker](./install.md#docker) | Any | 1 min | Testing, dev |
| [From Source](./install.md#from-source) | Any | 10 min | Development |

---

## Quick Test (Docker)

```bash
# 1. Run ProximaDB (a throwaway trial: this container keeps its data inside itself, TD-PKG-3;
#    for data that survives the container, follow the Docker section of the installation guide)
docker run -d -p 5678:5678 --name proximadb-trial vjsingh1984/proximadb:latest

# 2. Wait for startup (5 seconds)
sleep 5

# 3. Verify it's running
curl http://localhost:5678/health
# {"status":"healthy","version":"<installed release>"}

# 4. Create a canonical record collection
curl -X POST http://localhost:5678/api/v2/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "demo",
    "dimension": 3,
    "distance_metric": "cosine",
    "engine": "sst",
    "enable_proxima_record": true
  }'

# 5. Insert ProximaRecords with embeddings
curl -X POST http://localhost:5678/api/v2/collections/demo/records/batch \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"id": "1", "vector": [0.1, 0.2, 0.3], "props": {"category": "A"}},
      {"id": "2", "vector": [0.3, 0.4, 0.5], "props": {"category": "B"}}
    ]
  }'

# 6. Search
curl -X POST http://localhost:5678/api/v2/collections/demo/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "top_k": 5
  }'
```

---

## Python SDK

```python
from proximadb_sdk import ProximaDBClient, ProximaRecord, Protocol

# Connect
client = ProximaDBClient(url="http://localhost:5678", protocol=Protocol.REST)

# Create collection
client.create_collection(
    name="demo",
    dimension=3,
    distance_metric="cosine"
)

# Insert records with embeddings
client.insert_records("demo", [
    ProximaRecord(id="1", vector=[0.1, 0.2, 0.3]).set_flexible("category", "A"),
    ProximaRecord(id="2", vector=[0.3, 0.4, 0.5]).set_flexible("category", "B"),
])

# Search
results = client.search("demo", vector=[0.1, 0.2, 0.3], top_k=5)

for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
```

---

## SQL Interface

> **Connecting on 5433?** pgwire binds `127.0.0.1:5433` unless the listener enforces its
> own per-connection authentication (pgwire SCRAM, `TD-PGWIRE-AUTH-1`). So `localhost:5433`
> works against a locally-run `proximadb-server`, but **not** through a port published from
> a container — there, reach it from inside the container's network namespace, or enable
> SCRAM and then set `[api] pg_port`. See `TD-PGWIRE-CONTAINER-1`.

```sql
-- Connect (from a locally run server, or inside the container's network namespace)
-- psql -h 127.0.0.1 -p 5433 -U postgres -d proximadb

CREATE TABLE catalog_items (id INT PRIMARY KEY, name TEXT, price NUMERIC);

INSERT INTO catalog_items (id, name, price)
VALUES (1, 'Wireless Headphones', 79.99), (2, 'Coffee Maker', 49.99);

SELECT name, price FROM catalog_items WHERE price < 60 ORDER BY price;
```

`ORDER BY embedding <-> '<literal>'::vector LIMIT k` over collections created through REST or the
SDK is Beta. `VECTOR(n)` columns created over pgwire work today only in tables that also have a JSON or JSONB column
(TD-SQLCOLL-1).

---

## What's Next?

- **Learn installation options**: [Installation Guide](./install.md)
- **Build your first app**: [First Query Tutorial](./first-query.md)
- **Understand the architecture**: [Architecture Basics](./architecture-basics.md)
- **Explore features**: [Guides](../02-guides/index.md)

---

## Troubleshooting

**Port already in use?**
```bash
# Find what's using port 5678
lsof -i :5678

# Kill the process
kill -9 <PID>
```

**Server won't start?**
```bash
# Check logs
sudo journalctl -u proximadb -f  # Linux
docker logs proximadb-trial        # Docker (quick test)
```

**Can't connect?**
```bash
# Verify server is running
curl http://localhost:5678/health

# Check firewall
sudo firewall-cmd --list-ports  # Linux
```

---

*Need help?* [GitHub Issues](https://github.com/anvai-labs/proximaDB/issues)
