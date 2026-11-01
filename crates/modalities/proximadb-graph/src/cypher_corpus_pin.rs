//! TD-GRAPH-4 §3 — the Cypher corpus pin (contract snapshot before convergence).
//!
//! Two live Cypher grammars exist (the dedup program's wave-3 finding): this
//! crate's `cypher_parser` (the graph-API path, the richer live grammar) and
//! `proximadb-graph-query`'s write seam (the declared convergence target).
//! Before any clause is ported from one to the other, THIS module freezes
//! both grammars' clause-level behavior over a shared corpus, so every
//! ported clause is differential-checked against the behavior the graph API
//! path shipped: a port that silently changes parse semantics fails here,
//! and the pin is only updated as a deliberate, reviewed edit.
//!
//! Each corpus entry parses on BOTH grammars; the snapshot records, per
//! entry, the modality AST JSON (`M:`) and the seam AST JSON (`S:`). Both
//! are canonical: object keys are sorted at every level (the modality AST's
//! `Serialize` impls buffer through maps with process-varying iteration
//! order, so raw `to_string` is not a stable pin), and the modality side's
//! runtime-minted identifiers (node `id`, edge `from`/`to`) are elided —
//! fresh UUIDs per parse are not grammar behavior.
//!
//! Coverage note: entries are the BOTH-grammar intersection (full input
//! consumed on both sides — the seam parser does NOT require eof, so an
//! `Ok` with stranded remainder is a REJECTION here). Aggregations
//! SUM/AVG/MIN/MAX/COUNT(*) and ORDER BY over a bare variable (with DESC)
//! are both-accepted and pinned. Constructs only the modality grammar
//! accepts (float/array/null literals, `IN`, multi-label `:A:B`,
//! `WITH n AS x` aliasing, `COUNT(DISTINCT n)`) are structurally OUTSIDE
//! this pin — the seam must gain them before any replacement of the
//! modality grammar (a convergence-design requirement, recorded in
//! TD-GRAPH-4 §3). Constructs BOTH reject (ORDER BY over a property like
//! `n.name`, UNION, UNWIND, FOREACH, CALL, `COUNT(n)` — the seam's AST has
//! no bare-variable count) are likewise absent.
//!
//! To regenerate after a DELIBERATE behavior change, print the rebuilt
//! snapshot from the corpus (same canonicalization), diff it against
//! `PINNED_SNAPSHOT`, review the diff, and paste. The expected shape when
//! ports land: the `M:`/`S:` lines for a ported clause converge while the
//! others stay byte-identical.

/// Shared corpus: (id, query). Every entry parses cleanly on BOTH grammars
/// (full input consumed, no error) — pinned by the tests below.
#[cfg(test)]
const CORPUS: &[(&str, &str)] = &[
    ("read_where", "MATCH (n:Person) WHERE n.age > 30 RETURN n"),
    (
        "read_edge",
        "MATCH (n:Person)-[r:KNOWS]->(m:Person) RETURN n, r, m",
    ),
    (
        "read_optional",
        "MATCH (n:Person) OPTIONAL MATCH (n)-[r:KNOWS]->(m:Person) RETURN n, m",
    ),
    (
        "read_props",
        "MATCH (n:Person {name: \"Alice\", age: 30}) RETURN n",
    ),
    ("read_limit", "MATCH (n:Person) RETURN n SKIP 1 LIMIT 5"),
    (
        "create_node",
        "CREATE (n:Person {name: \"Alice\", age: 30})",
    ),
    (
        "create_return",
        "CREATE (n:Person {name: \"Alice\"}) RETURN n",
    ),
    (
        "create_edge",
        "MATCH (a:Person), (b:Person) CREATE (a)-[r:KNOWS {since: 2020}]->(b)",
    ),
    (
        "merge_on_create",
        "MERGE (n:Person {name: \"Bob\"}) ON CREATE SET n.created = true",
    ),
    (
        "set_multi",
        "MATCH (n:Person) SET n.age = 31, n.city = \"NYC\"",
    ),
    (
        "set_return",
        "MATCH (n:Person {name: \"Alice\"}) SET n.age = 31 RETURN n",
    ),
    ("remove_prop", "MATCH (n:Person) REMOVE n.age"),
    ("remove_label", "MATCH (n:Person) REMOVE n:Person"),
    (
        "delete_where",
        "MATCH (n:Person) WHERE n.age > 100 DELETE n",
    ),
    (
        "with_projection",
        "MATCH (n:Person) WITH n.name AS name RETURN name",
    ),
    (
        "read_var_length_range",
        "MATCH (n:Person)-[r:KNOWS*1..3]->(m:Person) RETURN n, m",
    ),
    (
        "read_var_length_any",
        "MATCH (n:Person)-[r:KNOWS*]->(m:Person) RETURN n, m",
    ),
    ("read_distinct", "MATCH (n:Person) RETURN DISTINCT n"),
    (
        "read_where_and",
        "MATCH (n:Person) WHERE n.age > 21 AND n.age < 65 RETURN n",
    ),
    (
        "read_where_or",
        "MATCH (n:Person) WHERE n.age < 18 OR n.age > 65 RETURN n",
    ),
    (
        "read_incoming",
        "MATCH (n:Person)<-[r:KNOWS]-(m:Person) RETURN n, m",
    ),
    (
        "read_bidirectional",
        "MATCH (n:Person)-[r:KNOWS]-(m:Person) RETURN n, m",
    ),
    (
        "read_single_quotes",
        "MATCH (n:Person {name: 'Alice'}) RETURN n",
    ),
    ("read_bool_prop", "MATCH (n:Person {active: true}) RETURN n"),
    ("read_limit_only", "MATCH (n:Person) RETURN n LIMIT 5"),
    ("sum_agg", "MATCH (n:Person) RETURN SUM(n.age)"),
    ("detach_delete", "MATCH (n:Person) DETACH DELETE n"),
    (
        "delete_edge",
        "MATCH (n:Person)-[r:KNOWS]->(m:Person) DELETE r",
    ),
    (
        "merge_on_match",
        "MERGE (n:Person {name: \"Bob\"}) ON MATCH SET n.seen = true",
    ),
    (
        "merge_on_both",
        "MERGE (n:Person {name: \"Bob\"}) ON CREATE SET n.created = true ON MATCH SET n.updated = true",
    ),
    (
        "set_map",
        "MATCH (n:Person) SET n += {age: 31, city: \"NYC\"}",
    ),
    ("set_label", "MATCH (n:Person) SET n:Adult"),
    (
        "with_where",
        "MATCH (n:Person) WITH n WHERE n.age > 21 RETURN n",
    ),
    (
        "minted_id_named_props",
        "CREATE (n:Person {id: 1, from: \"a\", to: \"b\", keep: 2})",
    ),
    ("avg_agg", "MATCH (n:Person) RETURN AVG(n.age)"),
    ("min_agg", "MATCH (n:Person) RETURN MIN(n.age)"),
    ("max_agg", "MATCH (n:Person) RETURN MAX(n.age)"),
    ("count_star", "MATCH (n:Person) RETURN COUNT(*)"),
    ("order_by_var", "MATCH (n:Person) RETURN n ORDER BY n"),
    (
        "order_by_var_desc",
        "MATCH (n:Person) RETURN n ORDER BY n DESC",
    ),
    ("count_star_alias", "MATCH (n:Person) RETURN COUNT(*) AS c"),
];

/// The frozen clause-level behavior of both grammars over {CORPUS}.
/// Format per entry: `=== id` / `Q:` query / `M:` modality JSON / `S:` seam JSON.
#[cfg(test)]
const PINNED_SNAPSHOT: &str = r#"=== read_where
Q: MATCH (n:Person) WHERE n.age > 30 RETURN n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":{"Property":{"constraint":{"GreaterThan":30},"property":"age","variable":"n"}},"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":{"Property":{"constraint":{"GreaterThan":30},"property":"age","variable":"n"}}}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_edge
Q: MATCH (n:Person)-[r:KNOWS]->(m:Person) RETURN n, r, m
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Outgoing","edge_types":["KNOWS"],"from_variable":"n","label":"KNOWS","optional":false,"properties":{},"to_variable":"m","variable":"r"}],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n","r","m"],"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"r"},{"Variable":"m"}],"skip":null,"variables":["n","r","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Outgoing","edge_types":["KNOWS"],"from_variable":"n","optional":false,"properties":{},"to_variable":"m","var_length":null,"variable":"r"}],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"r"},{"Variable":"m"}],"skip":null,"variables":["n","r","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_optional
Q: MATCH (n:Person) OPTIONAL MATCH (n)-[r:KNOWS]->(m:Person) RETURN n, m
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}},{"Match":{"optional":true,"pattern":{"edges":[{"direction":"Outgoing","edge_types":["KNOWS"],"from_variable":"n","label":"KNOWS","optional":true,"properties":{},"to_variable":"m","variable":"r"}],"nodes":[{"label":null,"labels":[],"optional":true,"properties":{},"variable":"n"},{"label":"Person","labels":["Person"],"optional":true,"properties":{},"variable":"m"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n","m"],"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"m"}],"skip":null,"variables":["n","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}},{"Match":{"optional":true,"pattern":{"edges":[{"direction":"Outgoing","edge_types":["KNOWS"],"from_variable":"n","optional":true,"properties":{},"to_variable":"m","var_length":null,"variable":"r"}],"nodes":[{"labels":[],"optional":true,"properties":{},"variable":"n"},{"labels":["Person"],"optional":true,"properties":{},"variable":"m"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"m"}],"skip":null,"variables":["n","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_props
Q: MATCH (n:Person {name: "Alice", age: 30}) RETURN n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{"age":30,"name":"Alice"},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{"age":{"Equals":30},"name":{"Equals":"Alice"}},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_limit
Q: MATCH (n:Person) RETURN n SKIP 1 LIMIT 5
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":5,"order_by":[],"projections":[{"Variable":"n"}],"skip":1,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":5,"order_by":[],"projections":[{"Variable":"n"}],"skip":1,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== create_node
Q: CREATE (n:Person {name: "Alice", age: 30})
M: {"clauses":[],"reading_clauses":[],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Create":{"edges":[],"nodes":[{"labels":["Person"],"properties":{"age":30,"name":"Alice"},"variable":"n"}]}}],"with_clauses":[]}
S: {"reading_clauses":[],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Create":{"edges":[],"nodes":[{"labels":["Person"],"properties":{"age":30,"name":"Alice"},"variable":"n"}]}}],"with_clauses":[]}
=== create_return
Q: CREATE (n:Person {name: "Alice"}) RETURN n
M: {"clauses":[],"reading_clauses":[],"return_spec":{"distinct":false,"items":["n"],"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[{"Create":{"edges":[],"nodes":[{"labels":["Person"],"properties":{"name":"Alice"},"variable":"n"}]}}],"with_clauses":[]}
S: {"reading_clauses":[],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[{"Create":{"edges":[],"nodes":[{"labels":["Person"],"properties":{"name":"Alice"},"variable":"n"}]}}],"with_clauses":[]}
=== create_edge
Q: MATCH (a:Person), (b:Person) CREATE (a)-[r:KNOWS {since: 2020}]->(b)
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"a"},{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"b"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Create":{"edges":[{"direction":"Outgoing","edge_type":"KNOWS","from_variable":"a","properties":{"since":2020},"to_variable":"b","variable":"r"}],"nodes":[{"labels":[],"properties":{},"variable":"a"},{"labels":[],"properties":{},"variable":"b"}]}}],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"a"},{"labels":["Person"],"optional":false,"properties":{},"variable":"b"}],"paths":[],"where_clause":null}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Create":{"edges":[{"edge_type":"KNOWS","from_variable":"a","properties":{"since":2020},"to_variable":"b","variable":"r"}],"nodes":[{"labels":[],"properties":{},"variable":"a"},{"labels":[],"properties":{},"variable":"b"}]}}],"with_clauses":[]}
=== merge_on_create
Q: MERGE (n:Person {name: "Bob"}) ON CREATE SET n.created = true
M: {"clauses":[],"reading_clauses":[],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Merge":{"on_create":{"items":[{"Property":{"property":"created","value":true,"variable":"n"}}]},"on_match":null,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{"name":"Bob"},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"with_clauses":[]}
S: {"reading_clauses":[],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Merge":{"on_create":{"items":[{"Property":{"property":"created","value":true,"variable":"n"}}]},"on_match":null,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{"name":{"Equals":"Bob"}},"variable":"n"}],"paths":[],"where_clause":null}}}],"with_clauses":[]}
=== set_multi
Q: MATCH (n:Person) SET n.age = 31, n.city = "NYC"
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Set":{"items":[{"Property":{"property":"age","value":31,"variable":"n"}},{"Property":{"property":"city","value":"NYC","variable":"n"}}]}}],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Set":{"items":[{"Property":{"property":"age","value":31,"variable":"n"}},{"Property":{"property":"city","value":"NYC","variable":"n"}}]}}],"with_clauses":[]}
=== set_return
Q: MATCH (n:Person {name: "Alice"}) SET n.age = 31 RETURN n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{"name":"Alice"},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[{"Set":{"items":[{"Property":{"property":"age","value":31,"variable":"n"}}]}}],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{"name":{"Equals":"Alice"}},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[{"Set":{"items":[{"Property":{"property":"age","value":31,"variable":"n"}}]}}],"with_clauses":[]}
=== remove_prop
Q: MATCH (n:Person) REMOVE n.age
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Remove":{"items":[{"Property":{"property":"age","variable":"n"}}]}}],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Remove":{"items":[{"Property":{"property":"age","variable":"n"}}]}}],"with_clauses":[]}
=== remove_label
Q: MATCH (n:Person) REMOVE n:Person
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Remove":{"items":[{"Label":{"label":"Person","variable":"n"}}]}}],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Remove":{"items":[{"Label":{"label":"Person","variable":"n"}}]}}],"with_clauses":[]}
=== delete_where
Q: MATCH (n:Person) WHERE n.age > 100 DELETE n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":{"Property":{"constraint":{"GreaterThan":100},"property":"age","variable":"n"}},"where_clauses":[],"with_clauses":[]}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Delete":{"detach":false,"variables":["n"]}}],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":{"Property":{"constraint":{"GreaterThan":100},"property":"age","variable":"n"}}}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Delete":{"detach":false,"variables":["n"]}}],"with_clauses":[]}
=== with_projection
Q: MATCH (n:Person) WITH n.name AS name RETURN name
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["name"],"limit":null,"order_by":[],"projections":[{"Variable":"name"}],"skip":null,"variables":["name"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[{"distinct":false,"limit":null,"order_by":[],"projections":[{"Property":{"property":"name","variable":"n"}}],"skip":null,"where_clause":null}]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"name"}],"skip":null,"variables":["name"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[{"distinct":false,"limit":null,"order_by":[],"projections":[["name",{"Property":{"property":"name","variable":"n"}}]],"skip":null,"where_clause":null}]}
=== read_var_length_range
Q: MATCH (n:Person)-[r:KNOWS*1..3]->(m:Person) RETURN n, m
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Outgoing","edge_types":["KNOWS"],"from_variable":"n","label":"KNOWS","optional":false,"properties":{},"to_variable":"m","variable":"r"}],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n","m"],"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"m"}],"skip":null,"variables":["n","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Outgoing","edge_types":["KNOWS"],"from_variable":"n","optional":false,"properties":{},"to_variable":"m","var_length":[1,3],"variable":"r"}],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"m"}],"skip":null,"variables":["n","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_var_length_any
Q: MATCH (n:Person)-[r:KNOWS*]->(m:Person) RETURN n, m
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Outgoing","edge_types":["KNOWS"],"from_variable":"n","label":"KNOWS","optional":false,"properties":{},"to_variable":"m","variable":"r"}],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n","m"],"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"m"}],"skip":null,"variables":["n","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Outgoing","edge_types":["KNOWS"],"from_variable":"n","optional":false,"properties":{},"to_variable":"m","var_length":[1,5],"variable":"r"}],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"m"}],"skip":null,"variables":["n","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_distinct
Q: MATCH (n:Person) RETURN DISTINCT n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":true,"items":["n"],"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":true,"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_where_and
Q: MATCH (n:Person) WHERE n.age > 21 AND n.age < 65 RETURN n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":{"And":[{"Property":{"constraint":{"GreaterThan":21},"property":"age","variable":"n"}},{"Property":{"constraint":{"LessThan":65},"property":"age","variable":"n"}}]},"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":{"And":[{"Property":{"constraint":{"GreaterThan":21},"property":"age","variable":"n"}},{"Property":{"constraint":{"LessThan":65},"property":"age","variable":"n"}}]}}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_where_or
Q: MATCH (n:Person) WHERE n.age < 18 OR n.age > 65 RETURN n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":{"Or":[{"Property":{"constraint":{"LessThan":18},"property":"age","variable":"n"}},{"Property":{"constraint":{"GreaterThan":65},"property":"age","variable":"n"}}]},"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":{"Or":[{"Property":{"constraint":{"LessThan":18},"property":"age","variable":"n"}},{"Property":{"constraint":{"GreaterThan":65},"property":"age","variable":"n"}}]}}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_incoming
Q: MATCH (n:Person)<-[r:KNOWS]-(m:Person) RETURN n, m
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Incoming","edge_types":["KNOWS"],"from_variable":"n","label":"KNOWS","optional":false,"properties":{},"to_variable":"m","variable":"r"}],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n","m"],"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"m"}],"skip":null,"variables":["n","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Incoming","edge_types":["KNOWS"],"from_variable":"n","optional":false,"properties":{},"to_variable":"m","var_length":null,"variable":"r"}],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"m"}],"skip":null,"variables":["n","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_bidirectional
Q: MATCH (n:Person)-[r:KNOWS]-(m:Person) RETURN n, m
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Bidirectional","edge_types":["KNOWS"],"from_variable":"n","label":"KNOWS","optional":false,"properties":{},"to_variable":"m","variable":"r"}],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n","m"],"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"m"}],"skip":null,"variables":["n","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Bidirectional","edge_types":["KNOWS"],"from_variable":"n","optional":false,"properties":{},"to_variable":"m","var_length":null,"variable":"r"}],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"},{"Variable":"m"}],"skip":null,"variables":["n","m"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_single_quotes
Q: MATCH (n:Person {name: 'Alice'}) RETURN n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{"name":"Alice"},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{"name":{"Equals":"Alice"}},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_bool_prop
Q: MATCH (n:Person {active: true}) RETURN n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{"active":true},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{"active":{"Equals":true}},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== read_limit_only
Q: MATCH (n:Person) RETURN n LIMIT 5
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":5,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":5,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== sum_agg
Q: MATCH (n:Person) RETURN SUM(n.age)
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["sum_age"],"limit":null,"order_by":[],"projections":[{"Sum":{"property":"age","variable":"n"}}],"skip":null,"variables":["sum_age"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Sum":{"property":"age","variable":"n"}}],"skip":null,"variables":["sum_age"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== detach_delete
Q: MATCH (n:Person) DETACH DELETE n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Delete":{"detach":true,"variables":["n"]}}],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Delete":{"detach":true,"variables":["n"]}}],"with_clauses":[]}
=== delete_edge
Q: MATCH (n:Person)-[r:KNOWS]->(m:Person) DELETE r
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Outgoing","edge_types":["KNOWS"],"from_variable":"n","label":"KNOWS","optional":false,"properties":{},"to_variable":"m","variable":"r"}],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Delete":{"detach":false,"variables":["r"]}}],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[{"direction":"Outgoing","edge_types":["KNOWS"],"from_variable":"n","optional":false,"properties":{},"to_variable":"m","var_length":null,"variable":"r"}],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"},{"labels":["Person"],"optional":false,"properties":{},"variable":"m"}],"paths":[],"where_clause":null}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Delete":{"detach":false,"variables":["r"]}}],"with_clauses":[]}
=== merge_on_match
Q: MERGE (n:Person {name: "Bob"}) ON MATCH SET n.seen = true
M: {"clauses":[],"reading_clauses":[],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Merge":{"on_create":null,"on_match":{"items":[{"Property":{"property":"seen","value":true,"variable":"n"}}]},"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{"name":"Bob"},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"with_clauses":[]}
S: {"reading_clauses":[],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Merge":{"on_create":null,"on_match":{"items":[{"Property":{"property":"seen","value":true,"variable":"n"}}]},"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{"name":{"Equals":"Bob"}},"variable":"n"}],"paths":[],"where_clause":null}}}],"with_clauses":[]}
=== merge_on_both
Q: MERGE (n:Person {name: "Bob"}) ON CREATE SET n.created = true ON MATCH SET n.updated = true
M: {"clauses":[],"reading_clauses":[],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Merge":{"on_create":{"items":[{"Property":{"property":"created","value":true,"variable":"n"}}]},"on_match":{"items":[{"Property":{"property":"updated","value":true,"variable":"n"}}]},"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{"name":"Bob"},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"with_clauses":[]}
S: {"reading_clauses":[],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Merge":{"on_create":{"items":[{"Property":{"property":"created","value":true,"variable":"n"}}]},"on_match":{"items":[{"Property":{"property":"updated","value":true,"variable":"n"}}]},"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{"name":{"Equals":"Bob"}},"variable":"n"}],"paths":[],"where_clause":null}}}],"with_clauses":[]}
=== set_map
Q: MATCH (n:Person) SET n += {age: 31, city: "NYC"}
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Set":{"items":[{"MergeProperties":{"properties":{"age":31,"city":"NYC"},"variable":"n"}}]}}],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Set":{"items":[{"MergeProperties":{"properties":{"age":31,"city":"NYC"},"variable":"n"}}]}}],"with_clauses":[]}
=== set_label
Q: MATCH (n:Person) SET n:Adult
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Set":{"items":[{"AddLabel":{"label":"Adult","variable":"n"}}]}}],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Set":{"items":[{"AddLabel":{"label":"Adult","variable":"n"}}]}}],"with_clauses":[]}
=== with_where
Q: MATCH (n:Person) WITH n WHERE n.age > 21 RETURN n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"where_clause":{"Property":{"constraint":{"GreaterThan":21},"property":"age","variable":"n"}}}]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[{"distinct":false,"limit":null,"order_by":[],"projections":[["n",{"Variable":"n"}]],"skip":null,"where_clause":{"Property":{"constraint":{"GreaterThan":21},"property":"age","variable":"n"}}}]}
=== minted_id_named_props
Q: CREATE (n:Person {id: 1, from: "a", to: "b", keep: 2})
M: {"clauses":[],"reading_clauses":[],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Create":{"edges":[],"nodes":[{"labels":["Person"],"properties":{"from":"a","id":1,"keep":2,"to":"b"},"variable":"n"}]}}],"with_clauses":[]}
S: {"reading_clauses":[],"return_spec":null,"union_clauses":[],"updating_clauses":[{"Create":{"edges":[],"nodes":[{"labels":["Person"],"properties":{"from":"a","id":1,"keep":2,"to":"b"},"variable":"n"}]}}],"with_clauses":[]}
=== avg_agg
Q: MATCH (n:Person) RETURN AVG(n.age)
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["avg_age"],"limit":null,"order_by":[],"projections":[{"Avg":{"property":"age","variable":"n"}}],"skip":null,"variables":["avg_age"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Avg":{"property":"age","variable":"n"}}],"skip":null,"variables":["avg_age"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== min_agg
Q: MATCH (n:Person) RETURN MIN(n.age)
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["min_age"],"limit":null,"order_by":[],"projections":[{"Min":{"property":"age","variable":"n"}}],"skip":null,"variables":["min_age"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Min":{"property":"age","variable":"n"}}],"skip":null,"variables":["min_age"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== max_agg
Q: MATCH (n:Person) RETURN MAX(n.age)
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["max_age"],"limit":null,"order_by":[],"projections":[{"Max":{"property":"age","variable":"n"}}],"skip":null,"variables":["max_age"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":[{"Max":{"property":"age","variable":"n"}}],"skip":null,"variables":["max_age"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== count_star
Q: MATCH (n:Person) RETURN COUNT(*)
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["count"],"limit":null,"order_by":[],"projections":["Count"],"skip":null,"variables":["count"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":["Count"],"skip":null,"variables":["count"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== order_by_var
Q: MATCH (n:Person) RETURN n ORDER BY n
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":null,"order_by":["n ASC"],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[["n",true]],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== order_by_var_desc
Q: MATCH (n:Person) RETURN n ORDER BY n DESC
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["n"],"limit":null,"order_by":["n DESC"],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[["n",false]],"projections":[{"Variable":"n"}],"skip":null,"variables":["n"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
=== count_star_alias
Q: MATCH (n:Person) RETURN COUNT(*) AS c
M: {"clauses":[],"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"label":"Person","labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null,"where_clauses":[],"with_clauses":[]}}}],"return_spec":{"distinct":false,"items":["c"],"limit":null,"order_by":[],"projections":["Count"],"skip":null,"variables":["c"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
S: {"reading_clauses":[{"Match":{"optional":false,"pattern":{"edges":[],"nodes":[{"labels":["Person"],"optional":false,"properties":{},"variable":"n"}],"paths":[],"where_clause":null}}}],"return_spec":{"distinct":false,"limit":null,"order_by":[],"projections":["Count"],"skip":null,"variables":["c"]},"union_clauses":[],"updating_clauses":[],"with_clauses":[]}
"#;

/// Deterministic JSON rendering: object keys sorted at every level.
#[cfg(test)]
fn canonical_json(v: &serde_json::Value) -> String {
    use std::fmt::Write as _;
    match v {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = String::from("{");
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let _ = write!(out, "{:?}:", k);
                out.push_str(&canonical_json(&map[*k]));
            }
            out.push('}');
            out
        }
        serde_json::Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", parts.join(","))
        }
        other => other.to_string(),
    }
}

/// Elide runtime-minted identifiers from the modality AST JSON — scoped by
/// SHAPE, not by key name: the parser mints `id` on pattern nodes
/// (`"<var>_<uuid>"`) and `from`/`to` on pattern edges (mirroring
/// `from_variable`/`to_variable`). Removing those keys anywhere would also
/// strip USER property maps keyed `id`/`from`/`to` — the most common
/// property names in graph data — so the elision fires only on objects
/// recognized as pattern nodes (variable + labels) or pattern edges
/// (from_variable + to_variable). The scoped behavior is pinned by the
/// `minted_id_named_props` corpus entry, whose property map keeps all four
/// keys on the modality side.
#[cfg(test)]
fn elide_minted_ids(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Object(map) => {
            if map.contains_key("variable") && map.contains_key("labels") {
                map.remove("id");
            }
            if map.contains_key("from_variable") && map.contains_key("to_variable") {
                map.remove("from");
                map.remove("to");
            }
            for child in map.values_mut() {
                elide_minted_ids(child);
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                elide_minted_ids(child);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod pin_tests {
    use super::*;

    fn modality_json(query: &str) -> String {
        let ast = crate::cypher_parser::CypherParser::new()
            .parse(query)
            .expect("modality grammar accepts every corpus entry");
        let mut v = serde_json::to_value(&ast).expect("modality AST serializes");
        elide_minted_ids(&mut v);
        canonical_json(&v)
    }

    fn seam_json(query: &str) -> String {
        let (rest, ast) = proximadb_graph_query::parser::write_seam::parse_cypher_query(query)
            .expect("seam grammar accepts every corpus entry");
        assert!(
            rest.trim().is_empty(),
            "seam grammar left input unconsumed: {rest:?}"
        );
        let v = serde_json::to_value(&ast).expect("seam AST serializes");
        canonical_json(&v)
    }

    fn rebuilt_snapshot() -> String {
        let mut out = String::new();
        for (id, query) in CORPUS {
            out.push_str(&format!(
                "=== {id}\nQ: {query}\nM: {}\nS: {}\n",
                modality_json(query),
                seam_json(query),
            ));
        }
        out
    }

    #[test]
    fn corpus_snapshot_is_pinned() {
        assert_eq!(
            rebuilt_snapshot(),
            PINNED_SNAPSHOT,
            "the Cypher corpus behavior drifted from the TD-GRAPH-4 pin. \
             If the grammar change was deliberate, regenerate + review the \
             snapshot (see the module docs); never edit the pin blind."
        );
    }

    #[test]
    fn every_corpus_entry_parses_cleanly_on_both_grammars() {
        // Negative control for the corpus itself: an entry that either
        // grammar rejects would silently disappear from the pin's coverage.
        for (id, query) in CORPUS {
            assert!(
                crate::cypher_parser::CypherParser::new()
                    .parse(query)
                    .is_ok(),
                "corpus entry {id} no longer parses on the modality grammar"
            );
            let parsed = proximadb_graph_query::parser::write_seam::parse_cypher_query(query);
            assert!(
                matches!(&parsed, Ok((rest, _)) if rest.trim().is_empty()),
                "corpus entry {id} no longer parses on the seam grammar"
            );
        }
    }
}
