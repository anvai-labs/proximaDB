//! Example: Fluent API patterns in ProximaDB Rust SDK
//!
//! This example showcases the fluent builder pattern APIs for
//! creating collections, inserting vectors, searching, filtering,
//! graph operations, and more.
//!
//! Run with: cargo run --example fluent_api --features client

use proximadb_sdk::{
    ClientBuilder, DistanceMetric, FilterBuilder, GraphEdge, GraphNode, IndexType, ProximaClient,
    StorageEngine,
};

#[tokio::main]
async fn main() -> proximadb_sdk::Result<()> {
    println!("ProximaDB Rust SDK - Fluent API Examples");
    println!("========================================\n");

    // Example 1: Client Builder Pattern
    println!("1. Client Builder Pattern");
    println!("   -----------------------");

    // Simple connection
    let _client = ProximaClient::connect("http://localhost:5678")?;
    println!("   Simple: ProximaClient::connect(url)");

    // Builder with full configuration
    let client = ClientBuilder::new()
        .url("http://localhost:5678")
        .timeout_ms(5000)
        .max_retries(3)
        .pool_connections(true)
        .max_idle_connections(10)
        // .api_key("your-api-key")  // Optional authentication
        .build()?;
    println!("   Full builder: ClientBuilder::new().url().timeout_ms()...build()");

    // Example 2: Collection Builder Pattern
    println!("\n2. Collection Builder Pattern");
    println!("   ---------------------------");

    let collection_name = "fluent_api_demo";

    // Clean up from previous run
    let _ = client.delete_collection(collection_name).await;

    // Create collection with fluent API
    println!("   Creating collection with fluent API...");

    client
        .create_collection(collection_name)
        .dimension(512) // Vector dimension
        .engine(StorageEngine::Sst) // Storage engine (SST, HELIX, NOVA, etc.)
        .index(IndexType::Hnsw) // Index type (HNSW, IVF, LSH, Flat)
        .metric(DistanceMetric::Cosine) // Distance metric
        .execute()
        .await?;

    println!("   client.create_collection(\"{}\")", collection_name);
    println!("       .dimension(512)");
    println!("       .engine(StorageEngine::Sst)");
    println!("       .index(IndexType::Hnsw)");
    println!("       .metric(DistanceMetric::Cosine)");
    println!("       .execute().await?;");

    // Example 3: Insert Builder Pattern
    println!("\n3. Insert Builder Pattern");
    println!("   -----------------------");

    // Single vector insert with fluent metadata
    let embedding = vec![0.1f32; 512];

    client
        .collection(collection_name)
        .insert()
        .id("doc_001")
        .vector(&embedding)
        .meta("title", "Introduction to Rust")
        .meta("author", "Jane Doe")
        .meta("category", "programming")
        .meta("rating", "5")
        .execute()
        .await?;

    println!("   client.collection(\"{}\").insert()", collection_name);
    println!("       .id(\"doc_001\")");
    println!("       .vector(&embedding)");
    println!("       .meta(\"title\", \"Introduction to Rust\")");
    println!("       .meta(\"author\", \"Jane Doe\")");
    println!("       .execute().await?;");

    // Batch insert with fluent API
    let ids = vec!["doc_002".to_string(), "doc_003".to_string()];
    let vectors = vec![vec![0.2f32; 512], vec![0.3f32; 512]];

    client
        .collection(collection_name)
        .insert()
        .batch(ids, vectors)?
        .execute()
        .await?;

    println!("\n   Batch insert:");
    println!("   client.collection(\"{}\").insert()", collection_name);
    println!("       .batch(ids, vectors)?");
    println!("       .execute().await?;");

    // Example 4: Search Builder Pattern
    println!("\n4. Search Builder Pattern");
    println!("   ----------------------");

    let query = vec![0.15f32; 512];

    // Basic search
    let results = client
        .collection(collection_name)
        .search()
        .vector(&query)
        .top_k(10)
        .execute()
        .await?;

    println!("   Basic search found {} results", results.len());
    println!("   client.collection(\"{}\").search()", collection_name);
    println!("       .vector(&query)");
    println!("       .top_k(10)");
    println!("       .execute().await?;");

    // Search with filter
    let results = client
        .collection(collection_name)
        .search()
        .vector(&query)
        .top_k(5)
        .filter("category = 'programming'")
        .execute()
        .await?;

    println!("\n   Filtered search found {} results", results.len());
    println!("   client.collection(\"{}\").search()", collection_name);
    println!("       .vector(&query)");
    println!("       .top_k(5)");
    println!("       .filter(\"category = 'programming'\")");
    println!("       .execute().await?;");

    // Search with search mode
    let results = client
        .collection(collection_name)
        .search()
        .vector(&query)
        .top_k(5)
        .approximate_with_nprobe(5)
        .include_metadata(true)
        .min_score(0.5)
        .execute()
        .await?;

    println!("\n   Advanced search found {} results", results.len());
    println!("   client.collection(\"{}\").search()", collection_name);
    println!("       .vector(&query)");
    println!("       .top_k(5)");
    println!("       .approximate_with_nprobe(5)");
    println!("       .include_metadata(true)");
    println!("       .min_score(0.5)");
    println!("       .execute().await?;");

    // Example 5: Search Mode Variations
    println!("\n5. Search Mode Variations");
    println!("   ----------------------");

    // Exact search (100% recall)
    let _ = client
        .collection(collection_name)
        .search()
        .vector(&query)
        .exact()
        .execute()
        .await?;
    println!("   .exact() - 100% recall, searches all partitions");

    // Approximate search (faster, ~95% recall)
    let _ = client
        .collection(collection_name)
        .search()
        .vector(&query)
        .approximate()
        .execute()
        .await?;
    println!("   .approximate() - Faster with ~95% recall");

    // Approximate with custom nprobe
    let _ = client
        .collection(collection_name)
        .search()
        .vector(&query)
        .approximate_with_nprobe(10)
        .execute()
        .await?;
    println!("   .approximate_with_nprobe(10) - Custom partition count");

    // Adaptive search
    let _ = client
        .collection(collection_name)
        .search()
        .vector(&query)
        .adaptive(10000)
        .execute()
        .await?;
    println!("   .adaptive(10000) - Auto-selects based on dataset size");

    // Example 6: Collection Handle Operations
    println!("\n6. Collection Handle Operations");
    println!("   ----------------------------");

    let collection = client.collection(collection_name);
    println!(
        "   let collection = client.collection(\"{}\");",
        collection_name
    );

    let count = collection.count().await?;
    println!("   collection.count() -> {}", count);

    let info = collection.info().await?;
    println!("   collection.info() -> dimension: {}", info.dimension);

    // Example 7: Filter Builder Pattern
    println!("\n7. Filter Builder Pattern");
    println!("   -----------------------");

    // First, recreate collection for filter examples
    client
        .create_collection(collection_name)
        .dimension(512)
        .engine(StorageEngine::Sst)
        .execute()
        .await?;

    // Insert test data
    client
        .collection(collection_name)
        .insert()
        .id("doc_001")
        .vector(&vec![0.1f32; 512])
        .meta("category", "electronics")
        .meta("price", "150")
        .execute()
        .await?;

    // Build a complex filter
    let filter = FilterBuilder::new()
        .eq("category", "electronics")
        .gte("price", 100)
        .lte("price", 500)
        .build();

    println!("   Complex filter:");
    println!("   let filter = FilterBuilder::new()");
    println!("       .eq(\"category\", \"electronics\")");
    println!("       .gte(\"price\", 100)");
    println!("       .lte(\"price\", 500)");
    println!("       .build();");
    println!("   Filter expression: {}", filter);

    // Search with filter builder
    let results = client
        .collection(collection_name)
        .search()
        .vector(&query)
        .top_k(10)
        .with_filter(filter)
        .execute()
        .await?;

    println!("   Search with filter found {} results", results.len());

    // Inline filter methods
    println!("\n   Inline filter methods:");
    let results = client
        .collection(collection_name)
        .search()
        .vector(&query)
        .top_k(10)
        .filter_eq("category", "electronics")
        .filter_gte("price", 100)
        .execute()
        .await?;

    println!("   .filter_eq(\"category\", \"electronics\")");
    println!("   .filter_gte(\"price\", 100)");
    println!("   Found {} results", results.len());

    // Example 8: Update Operations
    println!("\n8. Update Operations");
    println!("   ------------------");

    // Update vector metadata
    client
        .collection(collection_name)
        .update("doc_001")
        .meta("updated_at", "2025-01-04")
        .meta("views", "100")
        .execute()
        .await?;

    println!("   collection.update(\"doc_001\")");
    println!("       .meta(\"updated_at\", \"2025-01-04\")");
    println!("       .meta(\"views\", \"100\")");
    println!("       .execute().await?;");

    // Update with new vector
    let new_embedding = vec![0.2f32; 512];
    client
        .collection(collection_name)
        .update("doc_001")
        .vector(&new_embedding)
        .execute()
        .await?;

    println!("\n   Update with new vector:");
    println!("   collection.update(\"doc_001\")");
    println!("       .vector(&new_embedding)");
    println!("       .execute().await?;");

    // Example 9: Delete Operations
    println!("\n9. Delete Operations");
    println!("   ------------------");

    // Check if vector exists
    let exists = client.collection(collection_name).exists("doc_001").await?;
    println!("   collection.exists(\"doc_001\") -> {}", exists);

    // Delete single vector
    client
        .collection(collection_name)
        .delete_vector("doc_001")
        .await?;
    println!("   collection.delete_vector(\"doc_001\").await?;");

    let exists_after = client.collection(collection_name).exists("doc_001").await?;
    println!("   collection.exists(\"doc_001\") -> {}", exists_after);

    // Example 10: Graph Operations
    println!("\n10. Graph Operations");
    println!("    -----------------");

    let graph_name = "knowledge_graph_demo";

    // Clean up from previous run
    let _ = client.delete_graph(graph_name).await;

    // Create a graph
    client
        .create_graph(graph_name)
        .description("Demo knowledge graph")
        .execute()
        .await?;

    println!(
        "   client.create_graph(\"{}\").execute().await?;",
        graph_name
    );

    // Add nodes with fluent API
    client
        .graph(graph_name)
        .add_node()
        .id("person_1")
        .label("Person")
        .property("name", "Alice")
        .property("age", 30)
        .execute()
        .await?;

    client
        .graph(graph_name)
        .add_node()
        .id("person_2")
        .label("Person")
        .property("name", "Bob")
        .property("age", 28)
        .execute()
        .await?;

    println!("\n   graph.add_node()");
    println!("       .id(\"person_1\")");
    println!("       .label(\"Person\")");
    println!("       .property(\"name\", \"Alice\")");
    println!("       .execute().await?;");

    // Add edges with fluent API
    client
        .graph(graph_name)
        .add_edge()
        .from("person_1")
        .to("person_2")
        .relationship("KNOWS")
        .property("since", "2020")
        .weight(0.9)
        .execute()
        .await?;

    println!("\n   graph.add_edge()");
    println!("       .from(\"person_1\")");
    println!("       .to(\"person_2\")");
    println!("       .relationship(\"KNOWS\")");
    println!("       .weight(0.9)");
    println!("       .execute().await?;");

    // Batch node addition
    let nodes = vec![
        GraphNode::new("company_1")
            .with_label("Company")
            .with_property("name", "TechCorp"),
        GraphNode::new("company_2")
            .with_label("Company")
            .with_property("name", "DataInc"),
    ];
    let added = client.graph(graph_name).add_nodes(nodes).await?;
    println!("\n   Batch add {} nodes", added);

    // Batch edge addition
    let edges = vec![
        GraphEdge::new("person_1", "company_1", "WORKS_AT"),
        GraphEdge::new("person_2", "company_2", "WORKS_AT"),
    ];
    let added = client.graph(graph_name).add_edges(edges).await?;
    println!("   Batch add {} edges", added);

    // Traverse the graph
    let results = client
        .graph(graph_name)
        .traverse()
        .start("person_1")
        .relationship("KNOWS")
        .max_depth(2)
        .limit(10)
        .execute()
        .await?;

    println!("\n   Traversal from person_1:");
    println!(
        "       Found {} nodes, {} edges",
        results.nodes.len(),
        results.edges.len()
    );

    // Get a specific node
    let node = client.graph(graph_name).get_node("person_1").await?;
    if let Some(n) = node {
        println!("   get_node(\"person_1\") -> label: {:?}", n.label);
    }

    // List all graphs
    let graphs = client.list_graphs().await?;
    println!("   list_graphs() -> {} graphs", graphs.len());

    // Cleanup
    println!("\n11. Cleanup");
    println!("    -------");
    client.delete_collection(collection_name).await?;
    println!("    Collection deleted");
    client.delete_graph(graph_name).await?;
    println!("    Graph deleted");

    println!("\nFluent API examples completed!");
    Ok(())
}
