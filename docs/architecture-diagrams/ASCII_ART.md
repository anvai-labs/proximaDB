Use these for the simplest articulated ideas where a full diagram is overkill. They render
natively everywhere (GitHub, plain text, terminals, PR comments).
```
Client -> [5678 REST/gRPC/Flight | 5433 pgwire | MCP] -> MultiServer -> Database facade
      -> Query(ComputeScheduler) / Services(DML) -> Storage(Engine trait) -> Object Store
```
```
                        +-----------------------+
   SQL / query  ------> |  ComputeScheduler     | --parquet--> DataFusion (OLAP)
                        |  route_select()       |
                        +-----------------------+ --native---> Volcano (OLTP floor)
                                                   --vector---> SST/HELIX/NOVA (+ VIPER, deprecated: existing collections only)
```
```
UPSERT -> DmlService -> CanonicalWal -> [durable ACK]
                     -> memtable (+ invalidate cache post-commit)
          ...threshold... -> FlushMaterializer -> Engine.do_flush -> DrPathBuilder
                          -> Object Store (data/{tenant}/{ns}/...)
```
```
foundation  (proto, records, distance, relational, tenant, index types)
     ^
horizontal (codec, compression, security, telemetry, serialization)
     ^
storage / query / modalities  (vector, graph, document, embedding, rank)
     ^
control (catalog) / platform (runtime, api, mcp, admin-ui)
     ^
proximadb root crate (src/)  <-- the monolith being decomposed
```
```
 1 idea, inline?            -> ASCII art
 shows on GitHub for free?  -> Mermaid in .md
 class/sequence/deployment? -> Mermaid .mermaid (export PNG via render_atlas.sh)
```
