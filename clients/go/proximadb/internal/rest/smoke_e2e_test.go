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

package rest

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"
)

// TestLiveSmokeE2E exercises the re-based (spec-generated transport) REST
// adapter against a LIVE ProximaDB server: collection create/list/get/delete
// plus a record insert/get round-trip. It is gated on PROXIMADB_SMOKE_URL so it
// never runs in the normal unit/contract `go test ./...` (which has no server).
//
//	PROXIMADB_SMOKE_URL=http://127.0.0.1:5678 go test -run TestLiveSmokeE2E \
//	  -count=1 ./proximadb/internal/rest/...
func TestLiveSmokeE2E(t *testing.T) {
	baseURL := os.Getenv("PROXIMADB_SMOKE_URL")
	if baseURL == "" {
		t.Skip("PROXIMADB_SMOKE_URL not set; skipping live e2e smoke")
	}

	a, err := NewAdapter(&Config{BaseURL: baseURL, Timeout: 15 * time.Second})
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	defer func() { _ = a.Close() }()

	ctx := context.Background()

	if _, err := a.Health(ctx); err != nil {
		t.Fatalf("Health: %v", err)
	}

	name := fmt.Sprintf("go_smoke_%d", time.Now().UnixNano())
	const dim = 4

	created, err := a.CreateCollection(ctx, &CreateCollectionRequest{
		Name:      name,
		Dimension: dim,
		Metric:    "cosine",
		Engine:    "sst",
	})
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	t.Logf("created collection %q (engine=%s)", created.Name, created.Engine)
	defer func() { _ = a.DeleteCollection(ctx, name) }()

	// ListCollections must succeed and round-trip the wire shape. Membership of
	// our freshly-created collection is asserted via GetCollection by name below
	// (the list endpoint paginates and may return server-assigned identifiers for
	// pre-existing collections, so a name scan over the page is not a reliable
	// membership check against a shared/dirty data dir).
	if _, err := a.ListCollections(ctx); err != nil {
		t.Fatalf("ListCollections: %v", err)
	}

	got, err := a.GetCollection(ctx, name)
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	if got.Name != name {
		t.Fatalf("GetCollection name: got %q, want %q", got.Name, name)
	}

	recID := "smoke_rec_1"
	if err := a.InsertRecords(ctx, name, []*ProximaRecord{
		{ID: recID, Vector: []float32{0.1, 0.2, 0.3, 0.4}, Props: map[string]interface{}{"k": "v"}},
	}); err != nil {
		t.Fatalf("InsertRecords: %v", err)
	}

	// Record visibility may lag the write path; retry the read briefly.
	var recs []*VectorRecord
	for i := 0; i < 10; i++ {
		recs, err = a.Get(ctx, name, []string{recID})
		if err == nil && len(recs) == 1 && recs[0].ID == recID {
			break
		}
		time.Sleep(300 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(recs) != 1 || recs[0].ID != recID {
		t.Fatalf("Get round-trip mismatch: got %+v", recs)
	}
	t.Logf("record round-trip OK: id=%s vector_len=%d", recs[0].ID, len(recs[0].Vector))

	if err := a.DeleteCollection(ctx, name); err != nil {
		t.Fatalf("DeleteCollection: %v", err)
	}
}
