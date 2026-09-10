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

package proximadb_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/proximadb/proximadb-go/proximadb"
)

// TestNewClient tests client creation with various options.
func TestNewClient(t *testing.T) {
	tests := []struct {
		name    string
		opts    []proximadb.Option
		wantErr bool
	}{
		{
			name: "default options",
			opts: nil,
		},
		{
			name: "with URL",
			opts: []proximadb.Option{
				proximadb.WithURL("http://localhost:5678"),
			},
		},
		{
			name: "with all options",
			opts: []proximadb.Option{
				proximadb.WithURL("http://localhost:5678"),
				proximadb.WithAPIKey("test-key"),
				proximadb.WithProtocol(proximadb.ProtocolREST),
				proximadb.WithTimeout(10 * time.Second),
				proximadb.WithMaxRetries(5),
				proximadb.WithPoolSize(20),
			},
		},
		{
			name: "with empty URL",
			opts: []proximadb.Option{
				proximadb.WithURL(""),
			},
			wantErr: true,
		},
		{
			name: "with invalid timeout",
			opts: []proximadb.Option{
				proximadb.WithTimeout(0),
			},
			wantErr: true,
		},
		{
			name: "with negative pool size",
			opts: []proximadb.Option{
				proximadb.WithPoolSize(-1),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := proximadb.NewClient(tt.opts...)
			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if client == nil {
				t.Fatal("expected client, got nil")
			}
			defer client.Close()
		})
	}
}

// TestClientClose tests client close behavior.
func TestClientClose(t *testing.T) {
	client, err := proximadb.NewClient()
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	// First close should succeed
	if err := client.Close(); err != nil {
		t.Errorf("first close failed: %v", err)
	}

	// Second close should be idempotent
	if err := client.Close(); err != nil {
		t.Errorf("second close failed: %v", err)
	}
}

// TestClientClosedError tests that operations fail on a closed client.
func TestClientClosedError(t *testing.T) {
	client, err := proximadb.NewClient()
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	client.Close()

	ctx := context.Background()

	// All operations should fail
	if _, err := client.ListCollections(ctx); err == nil {
		t.Error("expected error for ListCollections on closed client")
	}

	if _, err := client.GetCollection(ctx, "test"); err == nil {
		t.Error("expected error for GetCollection on closed client")
	}

	if err := client.DeleteCollection(ctx, "test"); err == nil {
		t.Error("expected error for DeleteCollection on closed client")
	}

	if err := client.Insert(ctx, "test", nil); err == nil {
		t.Error("expected error for Insert on closed client")
	}

	if _, err := client.Search(ctx, "test", nil); err == nil {
		t.Error("expected error for Search on closed client")
	}
}

// TestMockServer tests the client against a mock HTTP server.
func TestMockServer(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"status":         "healthy",
				"version":        "1.0.0",
				"uptime_seconds": 123.45,
			})
		case "/api/v2/collections":
			if r.Method == http.MethodGet {
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"collections": []map[string]interface{}{
						{
							"collection_id":          "test_collection",
							"name":                   "test_collection",
							"dimension":              128,
							"engine":                 "sst",
							"proxima_record_enabled": true,
							"record_count":           100,
						},
					},
				})
			} else if r.Method == http.MethodPost {
				w.WriteHeader(http.StatusCreated)
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"collection_id":          "new_collection",
					"name":                   "new_collection",
					"dimension":              256,
					"engine":                 "sst",
					"proxima_record_enabled": true,
					"created_at":             time.Now().Format(time.RFC3339),
				})
			}
		case "/api/v2/collections/test_collection":
			if r.Method == http.MethodGet {
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"collection_id":          "test_collection",
					"name":                   "test_collection",
					"dimension":              128,
					"distance_metric":        "cosine",
					"engine":                 "sst",
					"proxima_record_enabled": true,
					"stats": map[string]interface{}{
						"record_count":       100,
						"storage_size_bytes": 4096,
					},
					"created_at": time.Now().Format(time.RFC3339),
				})
			} else if r.Method == http.MethodDelete {
				w.WriteHeader(http.StatusOK)
			}
		case "/api/v2/collections/nonexistent":
			w.WriteHeader(http.StatusNotFound)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"error": "collection not found",
			})
		case "/api/v2/collections/test_collection/records/batch":
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"success_count": 1,
			})
		case "/api/v2/collections/test_collection/search":
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"results": []map[string]interface{}{
					{
						"id":    "vec1",
						"score": 0.95,
					},
					{
						"id":    "vec2",
						"score": 0.85,
					},
				},
				"took_ms":     1.5,
				"total_count": 2,
			})
		case "/api/v2/collections/test_collection/records/vec1":
			if r.Method == http.MethodGet {
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"id":     "vec1",
					"vector": []float32{0.1, 0.2, 0.3},
					"props":  map[string]interface{}{},
				})
			} else if r.Method == http.MethodDelete {
				w.WriteHeader(http.StatusOK)
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Create client pointing to mock server
	client, err := proximadb.NewClient(
		proximadb.WithURL(server.URL),
		proximadb.WithTimeout(5*time.Second),
		proximadb.WithMaxRetries(0), // Disable retries for testing
	)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	// Test Health
	t.Run("Health", func(t *testing.T) {
		health, err := client.Health(ctx)
		if err != nil {
			t.Fatalf("Health failed: %v", err)
		}
		if health.Status != "healthy" {
			t.Errorf("expected status 'healthy', got '%s'", health.Status)
		}
		if health.Version != "1.0.0" {
			t.Errorf("expected version '1.0.0', got '%s'", health.Version)
		}
	})

	// Test ListCollections
	t.Run("ListCollections", func(t *testing.T) {
		collections, err := client.ListCollections(ctx)
		if err != nil {
			t.Fatalf("ListCollections failed: %v", err)
		}
		if len(collections) != 1 {
			t.Fatalf("expected 1 collection, got %d", len(collections))
		}
		if collections[0].Name != "test_collection" {
			t.Errorf("expected collection name 'test_collection', got '%s'", collections[0].Name)
		}
	})

	// Test GetCollection
	t.Run("GetCollection", func(t *testing.T) {
		collection, err := client.GetCollection(ctx, "test_collection")
		if err != nil {
			t.Fatalf("GetCollection failed: %v", err)
		}
		if collection.Dimension != 128 {
			t.Errorf("expected dimension 128, got %d", collection.Dimension)
		}
	})

	// Test GetCollection not found
	t.Run("GetCollectionNotFound", func(t *testing.T) {
		_, err := client.GetCollection(ctx, "nonexistent")
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !proximadb.IsNotFound(err) {
			t.Errorf("expected not found error, got %v", err)
		}
	})

	// Test CreateCollection
	t.Run("CreateCollection", func(t *testing.T) {
		collection, err := client.CreateCollection(ctx, &proximadb.CreateCollectionRequest{
			Name:      "new_collection",
			Dimension: 256,
			Metric:    proximadb.Cosine,
		})
		if err != nil {
			t.Fatalf("CreateCollection failed: %v", err)
		}
		if collection.Name != "new_collection" {
			t.Errorf("expected collection name 'new_collection', got '%s'", collection.Name)
		}
	})

	// Test Insert
	t.Run("Insert", func(t *testing.T) {
		records := []*proximadb.VectorRecord{
			{
				ID:     "vec1",
				Vector: make([]float32, 128),
			},
		}
		err := client.Insert(ctx, "test_collection", records)
		if err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	})

	// Test InsertRecords
	t.Run("InsertRecords", func(t *testing.T) {
		records := []*proximadb.ProximaRecord{
			{
				ID:     "record1",
				Vector: make([]float32, 128),
				Props:  map[string]interface{}{"category": "canonical"},
			},
		}
		err := client.InsertRecords(ctx, "test_collection", records)
		if err != nil {
			t.Fatalf("InsertRecords failed: %v", err)
		}
	})

	// Test Search
	t.Run("Search", func(t *testing.T) {
		resp, err := client.Search(ctx, "test_collection", &proximadb.SearchQuery{
			Vector: make([]float32, 128),
			TopK:   10,
		})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(resp.Results) != 2 {
			t.Fatalf("expected 2 results, got %d", len(resp.Results))
		}
		if resp.Results[0].ID != "vec1" {
			t.Errorf("expected first result ID 'vec1', got '%s'", resp.Results[0].ID)
		}
	})

	// Test Get
	t.Run("Get", func(t *testing.T) {
		vectors, err := client.Get(ctx, "test_collection", []string{"vec1"})
		if err != nil {
			t.Fatalf("Get failed: %v", err)
		}
		if len(vectors) != 1 {
			t.Fatalf("expected 1 vector, got %d", len(vectors))
		}
	})

	// Test Delete
	t.Run("Delete", func(t *testing.T) {
		err := client.Delete(ctx, "test_collection", []string{"vec1"})
		if err != nil {
			t.Fatalf("Delete failed: %v", err)
		}
	})

	// Test DeleteCollection
	t.Run("DeleteCollection", func(t *testing.T) {
		err := client.DeleteCollection(ctx, "test_collection")
		if err != nil {
			t.Fatalf("DeleteCollection failed: %v", err)
		}
	})
}

// TestContextCancellation tests that operations respect context cancellation.
func TestContextCancellation(t *testing.T) {
	// Create a slow mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client, err := proximadb.NewClient(
		proximadb.WithURL(server.URL),
		proximadb.WithTimeout(10*time.Second),
		proximadb.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	defer client.Close()

	// Create a context that times out quickly
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err = client.Health(ctx)
	if err == nil {
		t.Error("expected error due to context timeout")
	}
}

// TestFilterBuilder tests the filter builder helpers.
func TestFilterBuilder(t *testing.T) {
	tests := []struct {
		name   string
		filter proximadb.Filter
		check  func(f proximadb.Filter) bool
	}{
		{
			name:   "Eq filter",
			filter: proximadb.Eq("category", "electronics"),
			check: func(f proximadb.Filter) bool {
				return f.Field == "category" && f.Operator == proximadb.OpEquals && f.Value == "electronics"
			},
		},
		{
			name:   "Gt filter",
			filter: proximadb.Gt("price", 100),
			check: func(f proximadb.Filter) bool {
				return f.Field == "price" && f.Operator == proximadb.OpGreaterThan
			},
		},
		{
			name:   "And filter",
			filter: proximadb.And(proximadb.Eq("a", 1), proximadb.Eq("b", 2)),
			check: func(f proximadb.Filter) bool {
				return len(f.And) == 2
			},
		},
		{
			name:   "Or filter",
			filter: proximadb.Or(proximadb.Eq("a", 1), proximadb.Eq("b", 2)),
			check: func(f proximadb.Filter) bool {
				return len(f.Or) == 2
			},
		},
		{
			name:   "In filter",
			filter: proximadb.In("status", "active", "pending"),
			check: func(f proximadb.Filter) bool {
				return f.Field == "status" && f.Operator == proximadb.OpIn
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !tt.check(tt.filter) {
				t.Errorf("filter check failed for %s", tt.name)
			}
		})
	}
}

// TestErrorHelpers tests the error helper functions.
func TestErrorHelpers(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		isFunc   func(error) bool
		expected bool
	}{
		{
			name:     "IsNotFound true",
			err:      proximadb.NewError(proximadb.ErrCodeNotFound, "not found"),
			isFunc:   proximadb.IsNotFound,
			expected: true,
		},
		{
			name:     "IsNotFound false",
			err:      proximadb.NewError(proximadb.ErrCodeInternal, "internal"),
			isFunc:   proximadb.IsNotFound,
			expected: false,
		},
		{
			name:     "IsAlreadyExists true",
			err:      proximadb.NewError(proximadb.ErrCodeAlreadyExists, "exists"),
			isFunc:   proximadb.IsAlreadyExists,
			expected: true,
		},
		{
			name:     "IsTimeout true",
			err:      proximadb.NewError(proximadb.ErrCodeTimeout, "timeout"),
			isFunc:   proximadb.IsTimeout,
			expected: true,
		},
		{
			name:     "IsRateLimited true",
			err:      proximadb.NewError(proximadb.ErrCodeRateLimited, "rate limited"),
			isFunc:   proximadb.IsRateLimited,
			expected: true,
		},
		{
			name:     "IsConnectionError true",
			err:      proximadb.NewError(proximadb.ErrCodeConnection, "connection error"),
			isFunc:   proximadb.IsConnectionError,
			expected: true,
		},
		{
			name:     "IsRetryable timeout",
			err:      proximadb.NewError(proximadb.ErrCodeTimeout, "timeout"),
			isFunc:   proximadb.IsRetryable,
			expected: true,
		},
		{
			name:     "IsRetryable rate limited",
			err:      proximadb.NewError(proximadb.ErrCodeRateLimited, "rate limited"),
			isFunc:   proximadb.IsRetryable,
			expected: true,
		},
		{
			name:     "IsRetryable unavailable",
			err:      proximadb.NewError(proximadb.ErrCodeUnavailable, "unavailable"),
			isFunc:   proximadb.IsRetryable,
			expected: true,
		},
		{
			name:     "IsRetryable false for not found",
			err:      proximadb.NewError(proximadb.ErrCodeNotFound, "not found"),
			isFunc:   proximadb.IsRetryable,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.isFunc(tt.err)
			if result != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// TestConfigValidation tests configuration validation.
func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name    string
		opts    []proximadb.Option
		wantErr bool
	}{
		{
			name: "valid config",
			opts: []proximadb.Option{
				proximadb.WithURL("http://localhost:5678"),
				proximadb.WithTimeout(30 * time.Second),
			},
		},
		{
			name: "empty URL",
			opts: []proximadb.Option{
				proximadb.WithURL(""),
			},
			wantErr: true,
		},
		{
			name: "zero timeout",
			opts: []proximadb.Option{
				proximadb.WithTimeout(0),
			},
			wantErr: true,
		},
		{
			name: "negative max retries",
			opts: []proximadb.Option{
				proximadb.WithMaxRetries(-1),
			},
			wantErr: true,
		},
		{
			name: "zero pool size",
			opts: []proximadb.Option{
				proximadb.WithPoolSize(0),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := proximadb.NewClient(tt.opts...)
			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}
}

// TestVectorRecordJSON tests JSON marshaling of VectorRecord.
func TestVectorRecordJSON(t *testing.T) {
	record := &proximadb.VectorRecord{
		ID:     "test-id",
		Vector: []float32{0.1, 0.2, 0.3},
		Metadata: map[string]interface{}{
			"key": "value",
		},
	}

	data, err := json.Marshal(record)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var decoded proximadb.VectorRecord
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if decoded.ID != record.ID {
		t.Errorf("ID mismatch: expected %s, got %s", record.ID, decoded.ID)
	}

	if len(decoded.Vector) != len(record.Vector) {
		t.Errorf("Vector length mismatch: expected %d, got %d", len(record.Vector), len(decoded.Vector))
	}
}

// TestBatchInsert tests batch insert operations.
func TestBatchInsert(t *testing.T) {
	// Create mock server that tracks insert counts
	var insertCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v2/collections/test_collection/records/batch" {
			insertCount.Add(1)
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"success_count": 100,
			})
		}
	}))
	defer server.Close()

	client, err := proximadb.NewClient(
		proximadb.WithURL(server.URL),
		proximadb.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	// Create 250 records (should result in 3 batches with batch size 100)
	records := make([]*proximadb.VectorRecord, 250)
	for i := 0; i < 250; i++ {
		records[i] = &proximadb.VectorRecord{
			ID:     string(rune('a' + i%26)),
			Vector: make([]float32, 128),
		}
	}

	opts := &proximadb.BatchOptions{
		BatchSize:   100,
		Concurrency: 2,
	}

	result, err := client.BatchInsert(ctx, "test_collection", records, opts)
	if err != nil {
		t.Fatalf("BatchInsert failed: %v", err)
	}

	if result.TotalProcessed != 250 {
		t.Errorf("expected 250 processed, got %d", result.TotalProcessed)
	}

	if got := insertCount.Load(); got != 3 {
		t.Errorf("expected 3 batch requests, got %d", got)
	}
}

// TestBatchSearch tests batch search operations.
func TestBatchSearch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v2/collections/test_collection/search" {
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"results": []map[string]interface{}{
					{"id": "vec1", "score": 0.95},
				},
				"took_ms":     1.0,
				"total_count": 1,
			})
		}
	}))
	defer server.Close()

	client, err := proximadb.NewClient(
		proximadb.WithURL(server.URL),
		proximadb.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	// Create 5 queries
	queries := make([]*proximadb.SearchQuery, 5)
	for i := 0; i < 5; i++ {
		queries[i] = &proximadb.SearchQuery{
			Vector: make([]float32, 128),
			TopK:   10,
		}
	}

	opts := &proximadb.BatchOptions{
		Concurrency: 3,
	}

	results, err := client.BatchSearch(ctx, "test_collection", queries, opts)
	if err != nil {
		t.Fatalf("BatchSearch failed: %v", err)
	}

	if len(results) != 5 {
		t.Errorf("expected 5 results, got %d", len(results))
	}

	for i, r := range results {
		if r == nil {
			t.Errorf("result %d is nil", i)
		}
	}
}

// TestStreamInsert tests streaming insert operations.
func TestStreamInsert(t *testing.T) {
	insertCalls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v2/collections/test_collection/records/batch" {
			insertCalls++
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"success_count": 1,
			})
		}
	}))
	defer server.Close()

	client, err := proximadb.NewClient(
		proximadb.WithURL(server.URL),
		proximadb.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	opts := &proximadb.StreamOptions{
		BufferSize:    10,
		FlushInterval: 50 * time.Millisecond,
	}

	inserter, err := client.StreamInsert(ctx, "test_collection", opts)
	if err != nil {
		t.Fatalf("StreamInsert failed: %v", err)
	}

	// Send some records
	for i := 0; i < 25; i++ {
		err := inserter.Send(&proximadb.VectorRecord{
			ID:     string(rune('a' + i%26)),
			Vector: make([]float32, 128),
		})
		if err != nil {
			t.Fatalf("Send failed: %v", err)
		}
	}

	// Wait a bit for flush
	time.Sleep(100 * time.Millisecond)

	result, err := inserter.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	if result.TotalProcessed != 25 {
		t.Errorf("expected 25 processed, got %d", result.TotalProcessed)
	}
}

// TestClientMetrics tests client metrics tracking.
func TestClientMetrics(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"status":         "healthy",
				"version":        "1.0.0",
				"uptime_seconds": 123.45,
			})
		}
	}))
	defer server.Close()

	client, err := proximadb.NewClient(
		proximadb.WithURL(server.URL),
		proximadb.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	// Make some requests
	for i := 0; i < 5; i++ {
		_, _ = client.Health(ctx)
	}

	metrics := client.Metrics()
	if metrics == nil {
		t.Fatal("expected metrics, got nil")
	}

	// Metrics should show some activity (at least client was created)
	if metrics.RequestCount < 0 {
		t.Error("request count should be non-negative")
	}
}

// TestBatchOptionsDefaults tests default batch options.
func TestBatchOptionsDefaults(t *testing.T) {
	opts := proximadb.DefaultBatchOptions()

	if opts.BatchSize != 1000 {
		t.Errorf("expected batch size 1000, got %d", opts.BatchSize)
	}

	if opts.Concurrency != 4 {
		t.Errorf("expected concurrency 4, got %d", opts.Concurrency)
	}

	if opts.ContinueOnError != false {
		t.Error("expected continue on error to be false")
	}
}

// TestStreamOptionsDefaults tests default stream options.
func TestStreamOptionsDefaults(t *testing.T) {
	opts := proximadb.DefaultStreamOptions()

	if opts.BufferSize != 100 {
		t.Errorf("expected buffer size 100, got %d", opts.BufferSize)
	}

	if opts.FlushInterval != 100*time.Millisecond {
		t.Errorf("expected flush interval 100ms, got %v", opts.FlushInterval)
	}

	if opts.MaxPending != 1000 {
		t.Errorf("expected max pending 1000, got %d", opts.MaxPending)
	}
}

// TestClientMetricsSuccessRate tests the success rate calculation.
func TestClientMetricsSuccessRate(t *testing.T) {
	tests := []struct {
		name         string
		requestCount int64
		successCount int64
		expected     float64
	}{
		{
			name:         "no requests",
			requestCount: 0,
			successCount: 0,
			expected:     0,
		},
		{
			name:         "all success",
			requestCount: 100,
			successCount: 100,
			expected:     100,
		},
		{
			name:         "half success",
			requestCount: 100,
			successCount: 50,
			expected:     50,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &proximadb.ClientMetrics{
				RequestCount: tt.requestCount,
				SuccessCount: tt.successCount,
			}
			rate := m.SuccessRate()
			if rate != tt.expected {
				t.Errorf("expected success rate %v, got %v", tt.expected, rate)
			}
		})
	}
}

// TestClientMetricsAverageLatency tests the average latency calculation.
func TestClientMetricsAverageLatency(t *testing.T) {
	tests := []struct {
		name           string
		requestCount   int64
		totalLatencyNs int64
		expected       time.Duration
	}{
		{
			name:           "no requests",
			requestCount:   0,
			totalLatencyNs: 0,
			expected:       0,
		},
		{
			name:           "single request",
			requestCount:   1,
			totalLatencyNs: 1000000, // 1ms
			expected:       time.Millisecond,
		},
		{
			name:           "multiple requests",
			requestCount:   10,
			totalLatencyNs: 10000000,         // 10ms total
			expected:       time.Millisecond, // 1ms average
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &proximadb.ClientMetrics{
				RequestCount:   tt.requestCount,
				TotalLatencyNs: tt.totalLatencyNs,
			}
			latency := m.AverageLatency()
			if latency != tt.expected {
				t.Errorf("expected average latency %v, got %v", tt.expected, latency)
			}
		})
	}
}

// TestNewConfigOptions tests all configuration options.
func TestNewConfigOptions(t *testing.T) {
	client, err := proximadb.NewClient(
		proximadb.WithURL("http://localhost:5678"),
		proximadb.WithAPIKey("test-key"),
		proximadb.WithProtocol(proximadb.ProtocolREST),
		proximadb.WithTimeout(60*time.Second),
		proximadb.WithMaxRetries(5),
		proximadb.WithRetryDelay(200*time.Millisecond),
		proximadb.WithMaxRetryDelay(30*time.Second),
		proximadb.WithPoolSize(20),
		proximadb.WithUserAgent("test-agent/1.0"),
		proximadb.WithCompression(true),
		proximadb.WithMaxIdleConns(50),
		proximadb.WithIdleConnTimeout(60*time.Second),
	)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	defer client.Close()

	if client == nil {
		t.Fatal("expected client, got nil")
	}
}

func TestRetryUsesMaxRetryDelay(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"message": "temporary outage",
			})
			return
		}

		_ = json.NewEncoder(w).Encode(map[string]string{
			"status": "ok",
		})
	}))
	defer server.Close()

	client, err := proximadb.NewClient(
		proximadb.WithURL(server.URL),
		proximadb.WithMaxRetries(2),
		proximadb.WithRetryDelay(50*time.Millisecond),
		proximadb.WithMaxRetryDelay(time.Millisecond),
	)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	probe, err := client.HealthLive(ctx)
	if err != nil {
		t.Fatalf("HealthLive failed: %v", err)
	}
	if probe.Status != "ok" {
		t.Fatalf("expected ok status, got %q", probe.Status)
	}
	if attempts != 3 {
		t.Fatalf("expected 3 attempts, got %d", attempts)
	}
}

// Benchmark tests

func BenchmarkVectorRecordMarshal(b *testing.B) {
	record := &proximadb.VectorRecord{
		ID:     "benchmark-id",
		Vector: make([]float32, 1536), // Common embedding size
		Metadata: map[string]interface{}{
			"key": "value",
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = json.Marshal(record)
	}
}

func BenchmarkSearchQueryMarshal(b *testing.B) {
	query := &proximadb.SearchQuery{
		Vector: make([]float32, 1536),
		TopK:   10,
		Filter: &proximadb.Filter{
			Field:    "category",
			Operator: proximadb.OpEquals,
			Value:    "electronics",
		},
		IncludeVectors:  true,
		IncludeMetadata: true,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = json.Marshal(query)
	}
}

func BenchmarkBatchInsert(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"inserted_count": 100})
	}))
	defer server.Close()

	client, _ := proximadb.NewClient(
		proximadb.WithURL(server.URL),
		proximadb.WithMaxRetries(0),
	)
	defer client.Close()

	records := make([]*proximadb.VectorRecord, 1000)
	for i := 0; i < 1000; i++ {
		records[i] = &proximadb.VectorRecord{
			ID:     "vec",
			Vector: make([]float32, 128),
		}
	}

	ctx := context.Background()
	opts := proximadb.DefaultBatchOptions()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.BatchInsert(ctx, "test", records, opts)
	}
}
