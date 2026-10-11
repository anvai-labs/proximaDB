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
| [Platform Package](./install.md#platform-packages) | Linux, Windows | 2 min | Production |
| [Docker](./install.md#docker) | Any | 1 min | Testing, dev |
| [From Source](./install.md#from-source) | Any | 10 min | Development |

---

## Quick Test (Docker)

```bash
# 1. Run ProximaDB
docker run -d -p 5678:5678 --name proximadb vjsingh1984/proximadb:latest

# 2. Wait for startup (5 seconds)
sleep 5

# 3. Verify it's running
curl http://localhost:5678/health
# {"status":"healthy","version":"0.2.0"}

# 4. Create a canonical record collection
curl -X POST http://localhost:5678/api/v2/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "demo",
    "dimension": 3,
    "distance_metric": "cosine",
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
from proximadb_sdk import ProximaDBClient, ProximaRecord

# Connect
client = ProximaDBClient(url="http://localhost:5678")

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

```bash
# Connect via psql (PostgreSQL wire protocol)
psql -h localhost -p 5433 -U postgres

# Create table with vector column
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT,
    embedding VECTOR(128)
);

# Insert
INSERT INTO products (name, embedding)
VALUES ('Product A', '[0.1, 0.2, ...]');

# Search with <-> operator
SELECT name, embedding <-> '[0.1, 0.2, ...]' AS distance
FROM products
ORDER BY distance
LIMIT 5;
```

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
docker logs proximadb              # Docker
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
