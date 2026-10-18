# proximadb

Python client SDK for **ProximaDB** — a cloud-native vector + graph database engineered
for AI-first applications (vector search, OLTP/OLAP, graph, document, multimodal) over
REST, gRPC, Arrow Flight, and the PostgreSQL wire protocol.

- **Distribution:** `proximadb` (PyPI) · **import:** `proximadb_sdk`
- **In-process engine:** `proximadb_embedded` (native PyO3 wheels) — `pip install 'proximadb[embedded]'`

## Install

```bash
pip install proximadb                 # client SDK (REST/gRPC/Flight)
pip install 'proximadb[embedded]'     # + in-process native engine (no server)
pip install 'proximadb[embeddings]'   # + local embedding providers
pip install 'proximadb[codegraph]'    # + shared code->CPG chunker (victor-codegraph)
```

Extras compose, e.g. `pip install 'proximadb[embeddings,langchain,llama_index]'`.

## Quickstart

```python
from proximadb_sdk import ProximaDBClient, Protocol

client = ProximaDBClient(url="http://localhost:5678", protocol=Protocol.REST)
client.create_collection("docs", dimension=384)
client.insert("docs", vectors=[...], ids=[...])
hits = client.search("docs", vector=[...], top_k=5)
```

Embedded (in-process, no server):

```python
from proximadb_embedded import ProximaDB  # requires the [embedded] extra

db = ProximaDB(data_dirs="./data")
```

## Transports

The SDK speaks REST (ergonomics), gRPC (typed RPC), and Arrow Flight (zero-copy bulk) on
ProximaDB's multiplexed port, plus standard PostgreSQL drivers via pgwire. The REST surface
is generated from the canonical OpenAPI spec behind an ergonomic facade.

## License

Apache-2.0.
