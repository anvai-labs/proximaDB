// Copyright 2025 Vijaykumar Singh
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
Package proximadb provides the official Go client for ProximaDB, a high-performance
vector database with support for multiple storage engines and protocol adapters.

# Quick Start

Create a client and perform basic operations:

	client, err := proximadb.NewClient(
		proximadb.WithURL("http://localhost:5678"),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	ctx := context.Background()

	// Create a collection
	collection, err := client.CreateCollection(ctx, &proximadb.CreateCollectionRequest{
		Name:      "my_vectors",
		Dimension: 128,
		Metric:    proximadb.Cosine,
	})

	// Insert canonical records with embeddings
	err = client.InsertRecords(ctx, "my_vectors", []*proximadb.ProximaRecord{
		{ID: "vec1", Vector: embedding1, Props: map[string]interface{}{"label": "a"}},
		{ID: "vec2", Vector: embedding2, Props: map[string]interface{}{"label": "b"}},
	})

	// Search for similar vectors
	resp, err := client.Search(ctx, "my_vectors", &proximadb.SearchQuery{
		Vector:          queryVector,
		TopK:            10,
		IncludeMetadata: true,
	})

# Configuration Options

The client supports various configuration options through functional options:

	client, err := proximadb.NewClient(
		proximadb.WithURL("http://localhost:5678"),
		proximadb.WithAPIKey("your-api-key"),
		proximadb.WithProtocol(proximadb.ProtocolREST),
		proximadb.WithTimeout(30 * time.Second),
		proximadb.WithMaxRetries(3),
		proximadb.WithMaxRetryDelay(10 * time.Second),
		proximadb.WithPoolSize(20),
		proximadb.WithUserAgent("my-app/1.0"),
	)

# Batch Operations

For high-throughput scenarios, use batch operations with progress tracking:

	// Batch insert with progress callback
	result, err := client.BatchInsert(ctx, "my_vectors", records, &proximadb.BatchOptions{
		BatchSize:   1000,
		Concurrency: 4,
		OnProgress: func(processed, total int) {
			fmt.Printf("Progress: %d/%d\n", processed, total)
		},
		ContinueOnError: true,
	})

	// Batch search for parallel queries
	results, err := client.BatchSearch(ctx, "my_vectors", queries, &proximadb.BatchOptions{
		Concurrency: 8,
	})

# Streaming Operations

For continuous data pipelines, use streaming interfaces:

	// Stream inserter for continuous ingestion
	inserter, err := client.StreamInsert(ctx, "my_vectors", &proximadb.StreamOptions{
		BufferSize:    100,
		FlushInterval: 100 * time.Millisecond,
	})

	for record := range dataSource {
		inserter.Send(record)
	}
	result, _ := inserter.Close()

	// Stream searcher for continuous queries
	searcher, err := client.StreamSearch(ctx, "my_vectors", nil)
	go func() {
		for result := range searcher.Results() {
			processResult(result)
		}
	}()
	searcher.Send(query)
	searcher.Close()

# Middleware

Add middleware for logging, metrics, circuit breaking, and more:

	import "log"

	logger := log.New(os.Stdout, "[ProximaDB] ", log.LstdFlags)

	client, err := proximadb.NewClient(
		proximadb.WithURL("http://localhost:5678"),
		proximadb.WithMiddleware(proximadb.LoggingMiddleware(logger)),
		proximadb.WithMiddleware(proximadb.CircuitBreakerMiddleware(nil)),
		proximadb.WithMiddleware(proximadb.RateLimiterMiddleware(nil)),
	)

Available middlewares:
  - LoggingMiddleware: Log all operations
  - TimeoutMiddleware: Enforce operation timeouts
  - CircuitBreakerMiddleware: Prevent cascade failures
  - RateLimiterMiddleware: Limit request rate
  - RetryMiddleware: Automatic retry with backoff
  - CacheMiddleware: Cache operation results
  - MetricsMiddleware: Record operation metrics
  - TracingMiddleware: Distributed tracing support

# Client Metrics

Monitor client performance with built-in metrics:

	metrics := client.Metrics()
	fmt.Printf("Requests: %d\n", metrics.RequestCount)
	fmt.Printf("Success Rate: %.2f%%\n", metrics.SuccessRate())
	fmt.Printf("Avg Latency: %v\n", metrics.AverageLatency())

# Protocol Support

The client supports both REST and gRPC protocols:

	// REST (default)
	client, _ := proximadb.NewClient(
		proximadb.WithProtocol(proximadb.ProtocolREST),
	)

	// gRPC (requires proto generation)
	client, _ := proximadb.NewClient(
		proximadb.WithProtocol(proximadb.ProtocolGRPC),
	)

# Filtering

Use the filter builder helpers for metadata filtering:

	filter := proximadb.And(
		proximadb.Eq("category", "electronics"),
		proximadb.Gte("price", 100),
		proximadb.In("status", "active", "pending"),
	)

	resp, err := client.Search(ctx, "products", &proximadb.SearchQuery{
		Vector: queryVector,
		TopK:   10,
		Filter: &filter,
	})

# Error Handling

Use the error helper functions to check specific error types:

	if proximadb.IsNotFound(err) {
		// Handle not found
	} else if proximadb.IsRateLimited(err) {
		// Handle rate limiting with backoff
	} else if proximadb.IsRetryable(err) {
		// Retry the operation
	}

# Storage Engines

ProximaDB supports multiple storage engines optimized for different use cases:

  - SST: Write-optimized, real-time ingestion
  - HELIX: Locality-optimized with Hilbert curves
  - NOVA: Progressive columnar for analytics
  - SWIFT: Ultra-low latency for small collections
  - NOVA: Progressive columnar for mixed workloads
  - RAPTOR: Adaptive row-group for dynamic workloads

Specify the engine when creating a collection:

	collection, err := client.CreateCollection(ctx, &proximadb.CreateCollectionRequest{
		Name:      "fast_vectors",
		Dimension: 768,
		Engine:    proximadb.EngineSwift,
	})

# Examples

See the examples directory for complete working examples:
  - examples/basic: Basic client usage
  - examples/advanced: Batch operations and middleware
  - examples/streaming: Real-time data pipelines
*/
package proximadb
