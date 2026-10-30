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

// OpenAPI contract gate for the Go REST SDK.
//
// For each of the 15 v2 SDK operations, this test:
//   1. Starts an httptest.Server that captures the inbound request and
//      returns a programmed minimal-valid JSON response.
//   2. Calls the SDK method with sample input.
//   3. Validates the captured request against the corresponding OpenAPI
//      operation defined in docs/openapi/proximadb-openapi.yaml:
//        - HTTP method matches the operation
//        - Path matches the spec's path template (with parameters substituted)
//        - Content-Type is application/json for write operations
//        - Request body parses as JSON and contains the spec-required
//          top-level keys (lightweight schema check)
//
// This is a lightweight check — not full JSON-schema validation — to keep the
// Go toolchain footprint small. The intent is regression detection: any new
// SDK method that calls a path not declared in the spec, or omits required
// body fields, must fail here.
//
// Mirrors clients/python/tests/unit/test_openapi_contract.py.

package rest

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

// ---------------------------------------------------------------------------
// Spec loading
// ---------------------------------------------------------------------------

// openAPISpec is a structurally-typed view of the OpenAPI spec. Paths and
// schemas are kept as untyped maps so we can tolerate path-level `parameters`
// (which is a sibling of HTTP methods, not an operation) without fighting the
// YAML decoder.
type openAPISpec struct {
	Paths      map[string]map[string]interface{} `yaml:"paths"`
	Components openAPIComponents                 `yaml:"components"`
}

type openAPIComponents struct {
	Schemas    map[string]map[string]interface{} `yaml:"schemas"`
	Parameters map[string]map[string]interface{} `yaml:"parameters"`
	Responses  map[string]map[string]interface{} `yaml:"responses"`
}

// operation is a typed view of a single OpenAPI operation (e.g. paths.X.get).
type operation struct {
	OperationID string
	RequestBody map[string]interface{}
}

// lookupOperation returns the operation at paths[pathTemplate][method] (case-
// insensitive method). Returns ok=false if the path or method is missing or
// the value isn't an operation map.
func lookupOperation(spec *openAPISpec, pathTemplate, method string) (operation, bool) {
	pathItem, ok := spec.Paths[pathTemplate]
	if !ok {
		return operation{}, false
	}
	rawOp, ok := pathItem[strings.ToLower(method)]
	if !ok {
		return operation{}, false
	}
	opMap, ok := rawOp.(map[string]interface{})
	if !ok {
		return operation{}, false
	}
	op := operation{}
	if id, ok := opMap["operationId"].(string); ok {
		op.OperationID = id
	}
	if rb, ok := opMap["requestBody"].(map[string]interface{}); ok {
		op.RequestBody = rb
	}
	return op, true
}

var (
	specOnce   sync.Once
	loadedSpec *openAPISpec
	loadErr    error
)

func loadSpec(t *testing.T) *openAPISpec {
	t.Helper()
	specOnce.Do(func() {
		// internal/rest is 5 dirs below the repo root:
		//   clients/go/proximadb/internal/rest -> repo root.
		_, thisFile, _, ok := runtime.Caller(0)
		if !ok {
			loadErr = nil
			t.Fatalf("could not determine test file location")
		}
		repoRoot := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", "..", "..", ".."))
		specPath := filepath.Join(repoRoot, "docs", "openapi", "proximadb-openapi.yaml")

		raw, err := os.ReadFile(specPath)
		if err != nil {
			loadErr = err
			return
		}
		var s openAPISpec
		if err := yaml.Unmarshal(raw, &s); err != nil {
			loadErr = err
			return
		}
		loadedSpec = &s
	})
	if loadErr != nil {
		t.Fatalf("failed to load OpenAPI spec: %v", loadErr)
	}
	if loadedSpec == nil {
		t.Fatal("OpenAPI spec did not load")
	}
	return loadedSpec
}

// ---------------------------------------------------------------------------
// Spec helpers
// ---------------------------------------------------------------------------

func resolveRef(spec *openAPISpec, ref string) map[string]interface{} {
	// only supports #/components/schemas/... and #/components/responses/...
	if !strings.HasPrefix(ref, "#/") {
		return nil
	}
	parts := strings.Split(ref[2:], "/")
	if len(parts) < 3 || parts[0] != "components" {
		return nil
	}
	switch parts[1] {
	case "schemas":
		return spec.Components.Schemas[parts[2]]
	case "responses":
		return spec.Components.Responses[parts[2]]
	case "parameters":
		return spec.Components.Parameters[parts[2]]
	}
	return nil
}

// requiredKeysForRequestBody returns the union of `required` properties for
// the requestBody schema of the given operation, resolving any top-level
// $ref and walking allOf compositions one level deep.
func requiredKeysForRequestBody(spec *openAPISpec, op operation) []string {
	if op.RequestBody == nil {
		return nil
	}
	content, ok := op.RequestBody["content"].(map[string]interface{})
	if !ok {
		return nil
	}
	media, ok := content["application/json"].(map[string]interface{})
	if !ok {
		return nil
	}
	schema, ok := media["schema"].(map[string]interface{})
	if !ok {
		return nil
	}
	return collectRequired(spec, schema)
}

func collectRequired(spec *openAPISpec, schema map[string]interface{}) []string {
	if schema == nil {
		return nil
	}
	if ref, ok := schema["$ref"].(string); ok {
		return collectRequired(spec, resolveRef(spec, ref))
	}
	keys := map[string]struct{}{}
	if req, ok := schema["required"].([]interface{}); ok {
		for _, k := range req {
			if s, ok := k.(string); ok {
				keys[s] = struct{}{}
			}
		}
	}
	// allOf composition (used by UpdateSchemaRequest).
	if all, ok := schema["allOf"].([]interface{}); ok {
		for _, raw := range all {
			if sub, ok := raw.(map[string]interface{}); ok {
				for _, k := range collectRequired(spec, sub) {
					keys[k] = struct{}{}
				}
			}
		}
	}
	out := make([]string, 0, len(keys))
	for k := range keys {
		out = append(out, k)
	}
	return out
}

// pathMatches checks that an actual request path matches a spec path template
// such as "/api/v2/collections/{collection_id}/schema". Path parameters match
// any non-slash segment.
func pathMatches(template, actual string) bool {
	pattern := "^"
	for _, segment := range strings.Split(template, "/") {
		if segment == "" {
			continue
		}
		pattern += "/"
		if strings.HasPrefix(segment, "{") && strings.HasSuffix(segment, "}") {
			pattern += `[^/]+`
		} else {
			pattern += regexp.QuoteMeta(segment)
		}
	}
	pattern += "$"
	return regexp.MustCompile(pattern).MatchString(actual)
}

// ---------------------------------------------------------------------------
// Capture harness
// ---------------------------------------------------------------------------

type captured struct {
	Method      string
	Path        string
	Query       string
	ContentType string
	Body        []byte
}

// newCapturingServer returns an httptest.Server that records the inbound
// request and replies with the given JSON body.
func newCapturingServer(t *testing.T, replyJSON string) (*httptest.Server, *captured) {
	t.Helper()
	cap := &captured{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.Method = r.Method
		cap.Path = r.URL.Path
		cap.Query = r.URL.RawQuery
		cap.ContentType = r.Header.Get("Content-Type")
		if r.Body != nil {
			data, _ := io.ReadAll(r.Body)
			cap.Body = data
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(replyJSON))
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

func newTestAdapter(t *testing.T, baseURL string) *Adapter {
	t.Helper()
	a, err := NewAdapter(&Config{BaseURL: baseURL, Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	return a
}

// assertOperation looks up the operation in the spec and asserts it has the
// expected operationId.
func assertOperation(t *testing.T, spec *openAPISpec, pathTemplate, method, wantOpID string) operation {
	t.Helper()
	op, ok := lookupOperation(spec, pathTemplate, method)
	if !ok {
		t.Fatalf("%s %s not found in OpenAPI spec", method, pathTemplate)
	}
	if op.OperationID != wantOpID {
		t.Fatalf("operationId mismatch: spec has %q, expected %q", op.OperationID, wantOpID)
	}
	return op
}

// assertBodyHasRequiredKeys parses the JSON body and asserts each required key
// from the OpenAPI spec is present at the top level.
func assertBodyHasRequiredKeys(t *testing.T, body []byte, required []string) map[string]interface{} {
	t.Helper()
	if len(body) == 0 && len(required) > 0 {
		t.Fatalf("body is empty but required keys are %v", required)
	}
	var parsed map[string]interface{}
	if len(body) > 0 {
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("body is not valid JSON: %v (body=%s)", err, string(body))
		}
	}
	for _, k := range required {
		if _, ok := parsed[k]; !ok {
			t.Fatalf("required key %q missing from body (body=%s)", k, string(body))
		}
	}
	return parsed
}

// ---------------------------------------------------------------------------
// Per-operation tests (15 total)
// ---------------------------------------------------------------------------

func TestGetHealthRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t,
		`{"status":"healthy","version":"0.2.0","uptime_seconds":1.0}`)
	a := newTestAdapter(t, srv.URL)

	if _, err := a.Health(context.Background()); err != nil {
		t.Fatalf("Health: %v", err)
	}

	assertOperation(t, spec, "/health", "GET", "getHealth")
	if cap.Method != http.MethodGet {
		t.Fatalf("method: got %s, want GET", cap.Method)
	}
	if !pathMatches("/health", cap.Path) {
		t.Fatalf("path: got %s, want /health", cap.Path)
	}
}

func TestGetLivenessRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t, `{"status":"ok"}`)
	a := newTestAdapter(t, srv.URL)

	if _, err := a.HealthLive(context.Background()); err != nil {
		t.Fatalf("HealthLive: %v", err)
	}

	assertOperation(t, spec, "/health/live", "GET", "getLiveness")
	if cap.Method != http.MethodGet {
		t.Fatalf("method: got %s, want GET", cap.Method)
	}
	if !pathMatches("/health/live", cap.Path) {
		t.Fatalf("path: got %s, want /health/live", cap.Path)
	}
}

func TestGetReadinessRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t, `{"status":"ready"}`)
	a := newTestAdapter(t, srv.URL)

	if _, err := a.HealthReady(context.Background()); err != nil {
		t.Fatalf("HealthReady: %v", err)
	}

	assertOperation(t, spec, "/health/ready", "GET", "getReadiness")
	if cap.Method != http.MethodGet {
		t.Fatalf("method: got %s, want GET", cap.Method)
	}
	if !pathMatches("/health/ready", cap.Path) {
		t.Fatalf("path: got %s, want /health/ready", cap.Path)
	}
}

func TestCreateCollectionRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t,
		`{"collection_id":"c1","name":"my_collection","dimension":128,"engine":"viper","proxima_record_enabled":true,"created_at":"2026-05-23T00:00:00Z"}`)
	a := newTestAdapter(t, srv.URL)

	_, err := a.CreateCollection(context.Background(), &CreateCollectionRequest{
		Name:        "my_collection",
		Dimension:   128,
		Metric:      "cosine",
		Engine:      "viper",
		Description: "a description",
	})
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	op := assertOperation(t, spec, "/api/v2/collections", "POST", "createCollection")
	if cap.Method != http.MethodPost {
		t.Fatalf("method: got %s, want POST", cap.Method)
	}
	if !pathMatches("/api/v2/collections", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/collections", cap.Path)
	}
	if !strings.HasPrefix(cap.ContentType, "application/json") {
		t.Fatalf("content-type: got %q, want application/json", cap.ContentType)
	}
	parsed := assertBodyHasRequiredKeys(t, cap.Body, requiredKeysForRequestBody(spec, op))
	if name, _ := parsed["name"].(string); name != "my_collection" {
		t.Fatalf("body.name: got %v, want my_collection", parsed["name"])
	}
	// TD-RESTGET-1: the server persists it, so the adapter must send it.
	if desc, _ := parsed["description"].(string); desc != "a description" {
		t.Fatalf("body.description: got %v, want \"a description\"", parsed["description"])
	}
}

func TestListCollectionsRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t, `{"collections":[],"total":0,"limit":100,"offset":0,"has_more":false}`)
	a := newTestAdapter(t, srv.URL)

	if _, err := a.ListCollections(context.Background()); err != nil {
		t.Fatalf("ListCollections: %v", err)
	}

	assertOperation(t, spec, "/api/v2/collections", "GET", "listCollections")
	if cap.Method != http.MethodGet {
		t.Fatalf("method: got %s, want GET", cap.Method)
	}
	if !pathMatches("/api/v2/collections", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/collections", cap.Path)
	}
}

func TestGetCollectionRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t,
		`{"collection_id":"c1","name":"col_abc","dimension":128,"engine":"viper","proxima_record_enabled":true,"distance_metric":"cosine","stats":{"record_count":0,"storage_size_bytes":0,"indexed_fields":0,"text_field_count":0},"created_at":"2026-05-23T00:00:00Z"}`)
	a := newTestAdapter(t, srv.URL)

	if _, err := a.GetCollection(context.Background(), "col_abc"); err != nil {
		t.Fatalf("GetCollection: %v", err)
	}

	assertOperation(t, spec, "/api/v2/collections/{collection_id}", "GET", "getCollection")
	if cap.Method != http.MethodGet {
		t.Fatalf("method: got %s, want GET", cap.Method)
	}
	if !pathMatches("/api/v2/collections/{collection_id}", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/collections/{collection_id}", cap.Path)
	}
}

func TestDeleteCollectionRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t, `{"success":true,"collection_id":"col_abc"}`)
	a := newTestAdapter(t, srv.URL)

	if err := a.DeleteCollection(context.Background(), "col_abc"); err != nil {
		t.Fatalf("DeleteCollection: %v", err)
	}

	assertOperation(t, spec, "/api/v2/collections/{collection_id}", "DELETE", "deleteCollection")
	if cap.Method != http.MethodDelete {
		t.Fatalf("method: got %s, want DELETE", cap.Method)
	}
	if !pathMatches("/api/v2/collections/{collection_id}", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/collections/{collection_id}", cap.Path)
	}
}

func TestGetCollectionSchemaRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t,
		`{"schema_id":"sch_1","schema_version":"v1","collection_id":"col_abc","schema":{"columns":[{"name":"title","data_type":"text"}]},"created_at":"2026-05-23T00:00:00Z"}`)
	a := newTestAdapter(t, srv.URL)

	if _, err := a.GetCollectionSchema(context.Background(), "col_abc"); err != nil {
		t.Fatalf("GetCollectionSchema: %v", err)
	}

	assertOperation(t, spec, "/api/v2/collections/{collection_id}/schema", "GET", "getCollectionSchema")
	if cap.Method != http.MethodGet {
		t.Fatalf("method: got %s, want GET", cap.Method)
	}
	if !pathMatches("/api/v2/collections/{collection_id}/schema", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/collections/{collection_id}/schema", cap.Path)
	}
}

func TestUpdateCollectionSchemaRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t,
		`{"schema_id":"sch_2","schema_version":"v2","previous_schema_id":"sch_1","changes":[],"warnings":[],"updated_at":"2026-05-23T00:00:00Z"}`)
	a := newTestAdapter(t, srv.URL)

	req := &UpdateSchemaRequest{
		Columns:     []ColumnDefinition{{Name: "title", DataType: "text"}},
		Enforcement: "strict",
		Force:       true,
	}
	if _, err := a.UpdateCollectionSchema(context.Background(), "col_abc", req); err != nil {
		t.Fatalf("UpdateCollectionSchema: %v", err)
	}

	op := assertOperation(t, spec, "/api/v2/collections/{collection_id}/schema", "PUT", "updateCollectionSchema")
	if cap.Method != http.MethodPut {
		t.Fatalf("method: got %s, want PUT", cap.Method)
	}
	if !pathMatches("/api/v2/collections/{collection_id}/schema", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/collections/{collection_id}/schema", cap.Path)
	}
	if !strings.HasPrefix(cap.ContentType, "application/json") {
		t.Fatalf("content-type: got %q, want application/json", cap.ContentType)
	}
	parsed := assertBodyHasRequiredKeys(t, cap.Body, requiredKeysForRequestBody(spec, op))
	if force, _ := parsed["force"].(bool); !force {
		t.Fatalf("body.force: got %v, want true", parsed["force"])
	}
}

func TestInsertRecordsRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t,
		`{"inserted_count":1,"failed_count":0,"errors":[],"inserted_ids":["r1"]}`)
	a := newTestAdapter(t, srv.URL)

	records := []*ProximaRecord{{ID: "r1", Vector: []float32{0.1, 0.2}}}
	if err := a.InsertRecords(context.Background(), "col_abc", records); err != nil {
		t.Fatalf("InsertRecords: %v", err)
	}

	op := assertOperation(t, spec, "/api/v2/collections/{collection_id}/records/batch", "POST", "insertRecords")
	if cap.Method != http.MethodPost {
		t.Fatalf("method: got %s, want POST", cap.Method)
	}
	if !pathMatches("/api/v2/collections/{collection_id}/records/batch", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/collections/{collection_id}/records/batch", cap.Path)
	}
	assertBodyHasRequiredKeys(t, cap.Body, requiredKeysForRequestBody(spec, op))
}

func TestGetRecordRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t, `{"id":"r1","props":{}}`)
	a := newTestAdapter(t, srv.URL)

	if _, err := a.Get(context.Background(), "col_abc", []string{"r1"}); err != nil {
		t.Fatalf("Get: %v", err)
	}

	assertOperation(t, spec, "/api/v2/collections/{collection_id}/records/{record_id}", "GET", "getRecord")
	if cap.Method != http.MethodGet {
		t.Fatalf("method: got %s, want GET", cap.Method)
	}
	if !pathMatches("/api/v2/collections/{collection_id}/records/{record_id}", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/collections/{collection_id}/records/{record_id}", cap.Path)
	}
}

func TestDeleteRecordRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t, `{"success":true,"id":"r1","processing_time_us":1}`)
	a := newTestAdapter(t, srv.URL)

	if err := a.Delete(context.Background(), "col_abc", []string{"r1"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	assertOperation(t, spec, "/api/v2/collections/{collection_id}/records/{record_id}", "DELETE", "deleteRecord")
	if cap.Method != http.MethodDelete {
		t.Fatalf("method: got %s, want DELETE", cap.Method)
	}
	if !pathMatches("/api/v2/collections/{collection_id}/records/{record_id}", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/collections/{collection_id}/records/{record_id}", cap.Path)
	}
}

func TestSearchRecordsRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t,
		`{"results":[],"latency_ms":1,"request_id":"rq"}`)
	a := newTestAdapter(t, srv.URL)

	q := &SearchQuery{Vector: []float32{0.1, 0.2}, TopK: 5}
	if _, err := a.Search(context.Background(), "col_abc", q); err != nil {
		t.Fatalf("Search: %v", err)
	}

	op := assertOperation(t, spec, "/api/v2/collections/{collection_id}/search", "POST", "searchRecords")
	if cap.Method != http.MethodPost {
		t.Fatalf("method: got %s, want POST", cap.Method)
	}
	if !pathMatches("/api/v2/collections/{collection_id}/search", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/collections/{collection_id}/search", cap.Path)
	}
	assertBodyHasRequiredKeys(t, cap.Body, requiredKeysForRequestBody(spec, op))
}

func TestExecuteQueryRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t, `{"records":[]}`)
	a := newTestAdapter(t, srv.URL)

	req := &QueryRequest{Language: "aql", Query: "FOR r IN col_abc RETURN r"}
	if _, err := a.ExecuteQuery(context.Background(), req); err != nil {
		t.Fatalf("ExecuteQuery: %v", err)
	}

	op := assertOperation(t, spec, "/api/v2/query", "POST", "executeQuery")
	if cap.Method != http.MethodPost {
		t.Fatalf("method: got %s, want POST", cap.Method)
	}
	if !pathMatches("/api/v2/query", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/query", cap.Path)
	}
	if !strings.HasPrefix(cap.ContentType, "application/json") {
		t.Fatalf("content-type: got %q, want application/json", cap.ContentType)
	}
	parsed := assertBodyHasRequiredKeys(t, cap.Body, requiredKeysForRequestBody(spec, op))
	if lang, _ := parsed["language"].(string); lang != "aql" {
		t.Fatalf("body.language: got %v, want aql", parsed["language"])
	}
}

func TestExplainQueryRequestMatchesSpec(t *testing.T) {
	spec := loadSpec(t)
	srv, cap := newCapturingServer(t, `{"plan":{}}`)
	a := newTestAdapter(t, srv.URL)

	req := &ExplainQueryRequest{Language: "uql", Query: "SELECT 1"}
	if _, err := a.ExplainQuery(context.Background(), req); err != nil {
		t.Fatalf("ExplainQuery: %v", err)
	}

	op := assertOperation(t, spec, "/api/v2/query/explain", "POST", "explainQuery")
	if cap.Method != http.MethodPost {
		t.Fatalf("method: got %s, want POST", cap.Method)
	}
	if !pathMatches("/api/v2/query/explain", cap.Path) {
		t.Fatalf("path: got %s, want /api/v2/query/explain", cap.Path)
	}
	if !strings.HasPrefix(cap.ContentType, "application/json") {
		t.Fatalf("content-type: got %q, want application/json", cap.ContentType)
	}
	parsed := assertBodyHasRequiredKeys(t, cap.Body, requiredKeysForRequestBody(spec, op))
	if lang, _ := parsed["language"].(string); lang != "uql" {
		t.Fatalf("body.language: got %v, want uql", parsed["language"])
	}
}

// TestAllV2OperationsCovered is a guard test: it ensures the table of
// (path, method, operationId) tuples we exercise above matches what the
// OpenAPI spec actually publishes for v2 SDK surface (health + collections +
// schema + records + search + query). Graph operations are intentionally
// excluded — they have a dedicated SDK surface in a separate scope.
func TestAllV2OperationsCovered(t *testing.T) {
	spec := loadSpec(t)

	type want struct {
		path, method, opID string
	}
	expected := []want{
		{"/health", "get", "getHealth"},
		{"/health/live", "get", "getLiveness"},
		{"/health/ready", "get", "getReadiness"},
		{"/api/v2/collections", "post", "createCollection"},
		{"/api/v2/collections", "get", "listCollections"},
		{"/api/v2/collections/{collection_id}", "get", "getCollection"},
		{"/api/v2/collections/{collection_id}", "delete", "deleteCollection"},
		{"/api/v2/collections/{collection_id}/schema", "get", "getCollectionSchema"},
		{"/api/v2/collections/{collection_id}/schema", "put", "updateCollectionSchema"},
		{"/api/v2/collections/{collection_id}/records/batch", "post", "insertRecords"},
		{"/api/v2/collections/{collection_id}/records/{record_id}", "get", "getRecord"},
		{"/api/v2/collections/{collection_id}/records/{record_id}", "delete", "deleteRecord"},
		{"/api/v2/collections/{collection_id}/search", "post", "searchRecords"},
		{"/api/v2/query", "post", "executeQuery"},
		{"/api/v2/query/explain", "post", "explainQuery"},
	}
	if len(expected) != 15 {
		t.Fatalf("expected 15 v2 SDK operations, table has %d", len(expected))
	}
	for _, w := range expected {
		op, ok := lookupOperation(spec, w.path, w.method)
		if !ok {
			t.Errorf("%s %s not in spec", strings.ToUpper(w.method), w.path)
			continue
		}
		if op.OperationID != w.opID {
			t.Errorf("%s %s: operationId %q, want %q", strings.ToUpper(w.method), w.path, op.OperationID, w.opID)
		}
	}
}
