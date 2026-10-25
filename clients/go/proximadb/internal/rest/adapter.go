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

// Package rest provides the REST protocol adapter for ProximaDB.
//
// TD-126 Phase 2 (spec-driven SDK pilot): the wire plumbing — URL/path
// construction, query-parameter encoding, JSON request marshaling, and the
// Authorization header — is GENERATED from docs/openapi/proximadb-openapi.yaml
// into ./internal/genrest (oapi-codegen, pinned in clients/go/codegen/tools.go;
// regenerate with `make gen-go-sdk`). This Adapter is the thin, hand-written
// ergonomic facade over that generated client: it owns connection setup
// (pooling, TLS, timeouts), bearer auth, error mapping to AdapterError, the
// idiomatic per-ID Get/Delete fan-out, and the stable public model structs that
// the SDK facade (proximadb/rest_adapter.go) and the OpenAPI contract test
// depend on. Generators don't do ergonomics; this layer is the value-add.
package rest

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/proximadb/proximadb-go/proximadb/internal/genrest"
)

// ErrorCode represents a ProximaDB error code.
type ErrorCode string

const (
	ErrCodeConnection        ErrorCode = "CONNECTION_ERROR"
	ErrCodeTimeout           ErrorCode = "TIMEOUT"
	ErrCodeNotFound          ErrorCode = "NOT_FOUND"
	ErrCodeAlreadyExists     ErrorCode = "ALREADY_EXISTS"
	ErrCodeInvalidArgument   ErrorCode = "INVALID_ARGUMENT"
	ErrCodeDimensionMismatch ErrorCode = "DIMENSION_MISMATCH"
	ErrCodeRateLimited       ErrorCode = "RATE_LIMITED"
	ErrCodeInternal          ErrorCode = "INTERNAL_ERROR"
	ErrCodeUnavailable       ErrorCode = "UNAVAILABLE"
)

// AdapterError represents an error from the REST adapter.
type AdapterError struct {
	Code    ErrorCode
	Message string
	Cause   error
}

func (e *AdapterError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("[%s] %s: %v", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

func (e *AdapterError) Unwrap() error {
	return e.Cause
}

// TLSConfig holds TLS configuration options.
type TLSConfig struct {
	CertFile   string
	KeyFile    string
	CAFile     string
	SkipVerify bool
}

// Config holds the REST adapter configuration.
type Config struct {
	BaseURL string
	APIKey  string
	Timeout time.Duration
	TLS     *TLSConfig
}

// Adapter implements the REST protocol for ProximaDB.
//
// The transport is the generated client (genrest.Client); this struct adds the
// ergonomic facade described in the package doc.
type Adapter struct {
	gen      *genrest.Client
	httpDoer *http.Client
}

// NewAdapter creates a new REST adapter backed by the generated REST client.
func NewAdapter(config *Config) (*Adapter, error) {
	// Parse and validate base URL.
	parsedURL, err := url.Parse(config.BaseURL)
	if err != nil {
		return nil, &AdapterError{
			Code:    ErrCodeInvalidArgument,
			Message: "invalid base URL",
			Cause:   err,
		}
	}

	// Normalize the base URL (remove trailing slash).
	baseURL := strings.TrimSuffix(parsedURL.String(), "/")

	// Build HTTP client (pooling + optional TLS) — connection ergonomics the
	// generator does not provide.
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
	}
	if config.TLS != nil {
		tlsConfig, err := buildTLSConfig(config.TLS)
		if err != nil {
			return nil, err
		}
		transport.TLSClientConfig = tlsConfig
	}
	httpClient := &http.Client{
		Transport: transport,
		Timeout:   config.Timeout,
	}

	// Request editors apply the default headers + bearer auth on every request
	// the generated client issues (the generated client only sets Content-Type
	// for JSON bodies; headers/auth are facade concerns).
	editors := []genrest.RequestEditorFn{
		func(_ context.Context, req *http.Request) error {
			req.Header.Set("Accept", "application/json")
			req.Header.Set("User-Agent", "proximadb-go/1.0.0")
			return nil
		},
	}
	if config.APIKey != "" {
		apiKey := config.APIKey
		editors = append(editors, func(_ context.Context, req *http.Request) error {
			req.Header.Set("Authorization", "Bearer "+apiKey)
			return nil
		})
	}

	opts := []genrest.ClientOption{
		genrest.WithHTTPClient(httpClient),
	}
	for _, e := range editors {
		opts = append(opts, genrest.WithRequestEditorFn(e))
	}

	gen, err := genrest.NewClient(baseURL, opts...)
	if err != nil {
		return nil, &AdapterError{
			Code:    ErrCodeInvalidArgument,
			Message: "failed to construct REST client",
			Cause:   err,
		}
	}

	return &Adapter{gen: gen, httpDoer: httpClient}, nil
}

// buildTLSConfig creates a TLS configuration from the given options.
func buildTLSConfig(config *TLSConfig) (*tls.Config, error) {
	tlsConfig := &tls.Config{
		InsecureSkipVerify: config.SkipVerify,
	}

	// Load client certificate if specified.
	if config.CertFile != "" && config.KeyFile != "" {
		cert, err := tls.LoadX509KeyPair(config.CertFile, config.KeyFile)
		if err != nil {
			return nil, &AdapterError{
				Code:    ErrCodeInvalidArgument,
				Message: "failed to load client certificate",
				Cause:   err,
			}
		}
		tlsConfig.Certificates = []tls.Certificate{cert}
	}

	// Load CA certificate if specified.
	if config.CAFile != "" {
		caCert, err := os.ReadFile(config.CAFile)
		if err != nil {
			return nil, &AdapterError{
				Code:    ErrCodeInvalidArgument,
				Message: "failed to load CA certificate",
				Cause:   err,
			}
		}
		caCertPool := x509.NewCertPool()
		if !caCertPool.AppendCertsFromPEM(caCert) {
			return nil, &AdapterError{
				Code:    ErrCodeInvalidArgument,
				Message: "failed to parse CA certificate",
			}
		}
		tlsConfig.RootCAs = caCertPool
	}

	return tlsConfig, nil
}

// ---------------------------------------------------------------------------
// Public model structs.
//
// These are the stable Go shapes the facade (proximadb/rest_adapter.go) and the
// OpenAPI contract test depend on. They are deliberately hand-kept (idiomatic Go
// field names + JSON tags) and mapped to/from the generated wire types below.
// ---------------------------------------------------------------------------

// CreateCollectionRequest represents a request to create a collection.
type CreateCollectionRequest struct {
	Name        string `json:"name"`
	Dimension   int    `json:"dimension"`
	Metric      string `json:"metric,omitempty"`
	Engine      string `json:"engine,omitempty"`
	Description string `json:"description,omitempty"`
}

// CollectionInfo represents collection information.
type CollectionInfo struct {
	Name        string    `json:"name"`
	Dimension   int       `json:"dimension"`
	Metric      string    `json:"metric"`
	Engine      string    `json:"engine"`
	VectorCount *int64    `json:"vector_count,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
}

// ProximaRecord represents the canonical record payload with optional vector data.
type ProximaRecord struct {
	ID     string                 `json:"id"`
	Vector []float32              `json:"vector,omitempty"`
	Props  map[string]interface{} `json:"props,omitempty"`
	Source string                 `json:"source,omitempty"`
}

// VectorRecord represents a legacy vector-shaped compatibility payload.
type VectorRecord struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchQuery represents a vector search query.
type SearchQuery struct {
	Vector          []float32 `json:"vector"`
	TopK            int       `json:"top_k,omitempty"`
	Filter          *Filter   `json:"filter,omitempty"`
	IncludeVectors  bool      `json:"include_vectors,omitempty"`
	IncludeMetadata bool      `json:"include_metadata,omitempty"`
}

// Filter represents a metadata filter.
type Filter struct {
	Field    string      `json:"field,omitempty"`
	Operator string      `json:"operator,omitempty"`
	Value    interface{} `json:"value,omitempty"`
	And      []Filter    `json:"and,omitempty"`
	Or       []Filter    `json:"or,omitempty"`
}

// SearchResult represents a single search result.
type SearchResult struct {
	ID       string                 `json:"id"`
	Score    float32                `json:"score"`
	Vector   []float32              `json:"vector,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchResponse represents the response from a search query.
type SearchResponse struct {
	Results    []SearchResult `json:"results"`
	TookMs     float64        `json:"took_ms"`
	TotalCount int64          `json:"total_count,omitempty"`
}

// HealthStatus represents the server health status.
type HealthStatus struct {
	Status  string  `json:"status"`
	Version string  `json:"version"`
	Uptime  float64 `json:"uptime_seconds"`
}

// ProbeResponse represents a Kubernetes liveness/readiness probe response.
type ProbeResponse struct {
	Status string `json:"status"`
}

// ColumnDefinition represents a typed column in a SchemaDefinition.
type ColumnDefinition struct {
	Name            string `json:"name"`
	DataType        string `json:"data_type"`
	Nullable        *bool  `json:"nullable,omitempty"`
	Indexed         *bool  `json:"indexed,omitempty"`
	Filterable      *bool  `json:"filterable,omitempty"`
	MaxLength       *int   `json:"max_length,omitempty"`
	Precision       *int   `json:"precision,omitempty"`
	Scale           *int   `json:"scale,omitempty"`
	VectorDimension *int   `json:"vector_dimension,omitempty"`
}

// SchemaDefinition represents a typed collection schema.
type SchemaDefinition struct {
	Columns               []ColumnDefinition `json:"columns"`
	Enforcement           string             `json:"enforcement,omitempty"`
	AllowAdditionalFields *bool              `json:"allow_additional_fields,omitempty"`
}

// SchemaResponse represents the response from a get-schema call.
type SchemaResponse struct {
	SchemaID       string           `json:"schema_id"`
	SchemaVersion  string           `json:"schema_version"`
	CollectionID   string           `json:"collection_id"`
	Schema         SchemaDefinition `json:"schema"`
	CreatedAt      string           `json:"created_at"`
	UpdatedAt      *string          `json:"updated_at,omitempty"`
	ParentSchemaID *string          `json:"parent_schema_id,omitempty"`
}

// UpdateSchemaRequest represents the body for an update-schema call.
type UpdateSchemaRequest struct {
	Columns               []ColumnDefinition `json:"columns"`
	Enforcement           string             `json:"enforcement,omitempty"`
	AllowAdditionalFields *bool              `json:"allow_additional_fields,omitempty"`
	Force                 bool               `json:"force,omitempty"`
}

// UpdateSchemaResponse represents the response from an update-schema call.
type UpdateSchemaResponse struct {
	SchemaID         string                   `json:"schema_id"`
	SchemaVersion    string                   `json:"schema_version"`
	PreviousSchemaID string                   `json:"previous_schema_id"`
	Changes          []map[string]interface{} `json:"changes"`
	Warnings         []string                 `json:"warnings"`
	UpdatedAt        string                   `json:"updated_at"`
}

// QueryRequest is the request body for the executeQuery operation.
type QueryRequest struct {
	Language   string        `json:"language"`
	Query      string        `json:"query"`
	Parameters []interface{} `json:"parameters,omitempty"`
	Collection *string       `json:"collection,omitempty"`
	Limit      *int          `json:"limit,omitempty"`
}

// ExplainQueryRequest is the request body for the explainQuery operation.
type ExplainQueryRequest struct {
	Language   string  `json:"language"`
	Query      string  `json:"query"`
	Collection *string `json:"collection,omitempty"`
}

// QueryResponse is the (open-shape) response from executeQuery / explainQuery.
type QueryResponse map[string]interface{}

// ---------------------------------------------------------------------------
// Operations: each maps the public model -> generated request, invokes the
// generated client method (which builds the URL/path/query + marshals the
// body + applies auth), then decodes the response into the public model.
// ---------------------------------------------------------------------------

// CreateCollection creates a new vector collection.
func (a *Adapter) CreateCollection(ctx context.Context, req *CreateCollectionRequest) (*CollectionInfo, error) {
	body := genrest.CreateCollectionJSONRequestBody{
		Name:      req.Name,
		Dimension: int32(req.Dimension),
	}
	if req.Metric != "" {
		body.DistanceMetric = ptr(req.Metric)
	}
	if req.Engine != "" {
		body.Engine = ptr(req.Engine)
	}
	// TD-RESTGET-1: the server persists a create description; this adapter
	// used to drop the caller's Description without sending it.
	if req.Description != "" {
		body.Description = ptr(req.Description)
	}

	resp, err := a.gen.CreateCollection(ctx, nil, body)
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}

	var out struct {
		CollectionID         string `json:"collection_id"`
		Name                 string `json:"name"`
		Dimension            int    `json:"dimension"`
		Engine               string `json:"engine"`
		ProximaRecordEnabled bool   `json:"proxima_record_enabled"`
		CreatedAt            string `json:"created_at"`
	}
	if err := a.decode(resp, &out); err != nil {
		return nil, err
	}

	return &CollectionInfo{
		Name:      out.Name,
		Dimension: out.Dimension,
		Metric:    req.Metric,
		Engine:    out.Engine,
		CreatedAt: parseTimeOrZero(out.CreatedAt),
	}, nil
}

// ListCollections returns all collections.
func (a *Adapter) ListCollections(ctx context.Context) ([]*CollectionInfo, error) {
	resp, err := a.gen.ListCollections(ctx, &genrest.ListCollectionsParams{})
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}

	var out struct {
		Collections []struct {
			CollectionID         string `json:"collection_id"`
			Name                 string `json:"name"`
			Dimension            int    `json:"dimension"`
			Engine               string `json:"engine"`
			ProximaRecordEnabled bool   `json:"proxima_record_enabled"`
			RecordCount          *int64 `json:"record_count,omitempty"`
		} `json:"collections"`
	}
	if err := a.decode(resp, &out); err != nil {
		return nil, err
	}

	collections := make([]*CollectionInfo, 0, len(out.Collections))
	for _, item := range out.Collections {
		collections = append(collections, &CollectionInfo{
			Name:      firstNonEmpty(item.Name, item.CollectionID),
			Dimension: item.Dimension,
			Engine:    item.Engine,
			// Pass the pointer through. This used to coalesce nil to 0,
			// throwing away a distinction the wire already carried.
			VectorCount: item.RecordCount,
		})
	}
	return collections, nil
}

// GetCollection returns information about a specific collection.
func (a *Adapter) GetCollection(ctx context.Context, name string) (*CollectionInfo, error) {
	resp, err := a.gen.GetCollection(ctx, name, nil)
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}

	var out struct {
		CollectionID   string `json:"collection_id"`
		Name           string `json:"name"`
		Dimension      int    `json:"dimension"`
		Engine         string `json:"engine"`
		DistanceMetric string `json:"distance_metric"`
		CreatedAt      string `json:"created_at"`
		Stats          struct {
			// Pointer, not int64: encoding/json decodes a null into an int64
			// as a silent zero, which is exactly the wrong answer here.
			RecordCount      *int64 `json:"record_count"`
			StorageSizeBytes int64  `json:"storage_size_bytes"`
		} `json:"stats"`
	}
	if err := a.decode(resp, &out); err != nil {
		return nil, err
	}

	return &CollectionInfo{
		Name:        firstNonEmpty(out.Name, out.CollectionID),
		Dimension:   out.Dimension,
		Metric:      out.DistanceMetric,
		Engine:      out.Engine,
		VectorCount: out.Stats.RecordCount,
		CreatedAt:   parseTimeOrZero(out.CreatedAt),
	}, nil
}

// DeleteCollection deletes a collection.
func (a *Adapter) DeleteCollection(ctx context.Context, name string) error {
	resp, err := a.gen.DeleteCollection(ctx, name, nil)
	if err != nil {
		return mapTransportError(ctx, err)
	}
	return a.decode(resp, nil)
}

// InsertRecords inserts canonical records into a collection.
func (a *Adapter) InsertRecords(ctx context.Context, collection string, records []*ProximaRecord) error {
	return a.writeRecords(ctx, collection, records, false)
}

// UpsertRecords inserts or updates canonical records in a collection.
func (a *Adapter) UpsertRecords(ctx context.Context, collection string, records []*ProximaRecord) error {
	return a.writeRecords(ctx, collection, records, true)
}

func (a *Adapter) writeRecords(ctx context.Context, collection string, records []*ProximaRecord, upsert bool) error {
	body := genrest.InsertRecordsJSONRequestBody{
		Records:        toGenRecords(records),
		ValidateSchema: ptr(true),
	}
	if upsert {
		body.Upsert = ptr(true)
	}

	resp, err := a.gen.InsertRecords(ctx, collection, nil, body)
	if err != nil {
		return mapTransportError(ctx, err)
	}
	return a.decode(resp, nil)
}

// Insert inserts records into a collection.
//
// Deprecated: use InsertRecords with ProximaRecord.
func (a *Adapter) Insert(ctx context.Context, collection string, records []*VectorRecord) error {
	return a.InsertRecords(ctx, collection, recordsToProximaRecords(records))
}

// Upsert inserts or updates records in a collection.
//
// Deprecated: use UpsertRecords with ProximaRecord.
func (a *Adapter) Upsert(ctx context.Context, collection string, records []*VectorRecord) error {
	return a.UpsertRecords(ctx, collection, recordsToProximaRecords(records))
}

// Search performs a vector similarity search.
func (a *Adapter) Search(ctx context.Context, collection string, query *SearchQuery) (*SearchResponse, error) {
	body := genrest.SearchRecordsJSONRequestBody{
		Vector:        query.Vector,
		TopK:          query.TopK,
		IncludeVector: ptr(query.IncludeVectors),
	}

	resp, err := a.gen.SearchRecords(ctx, collection, nil, body)
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}

	var result SearchResponse
	if err := a.decode(resp, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Get retrieves vectors by their IDs.
//
// The OpenAPI surface exposes single-record reads; the SDK fans out per ID and
// assembles the slice (an ergonomic the spec does not provide).
func (a *Adapter) Get(ctx context.Context, collection string, ids []string) ([]*VectorRecord, error) {
	vectors := make([]*VectorRecord, 0, len(ids))
	params := &genrest.GetRecordParams{IncludeVector: ptr(true), IncludeText: ptr(false)}
	for _, id := range ids {
		resp, err := a.gen.GetRecord(ctx, collection, id, params)
		if err != nil {
			return nil, mapTransportError(ctx, err)
		}
		var out struct {
			ID     string                 `json:"id"`
			Vector []float32              `json:"vector,omitempty"`
			Props  map[string]interface{} `json:"props,omitempty"`
		}
		if err := a.decode(resp, &out); err != nil {
			return nil, err
		}
		vectors = append(vectors, &VectorRecord{
			ID:       out.ID,
			Vector:   out.Vector,
			Metadata: out.Props,
		})
	}
	return vectors, nil
}

// Delete removes vectors by their IDs.
func (a *Adapter) Delete(ctx context.Context, collection string, ids []string) error {
	for _, id := range ids {
		resp, err := a.gen.DeleteRecord(ctx, collection, id, nil)
		if err != nil {
			return mapTransportError(ctx, err)
		}
		if err := a.decode(resp, nil); err != nil {
			return err
		}
	}
	return nil
}

// Health checks the server health.
func (a *Adapter) Health(ctx context.Context) (*HealthStatus, error) {
	resp, err := a.gen.GetHealth(ctx, nil)
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}
	var result HealthStatus
	if err := a.decode(resp, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// HealthLive issues a Kubernetes liveness probe (GET /health/live).
func (a *Adapter) HealthLive(ctx context.Context) (*ProbeResponse, error) {
	resp, err := a.gen.GetLiveness(ctx, nil)
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}
	var result ProbeResponse
	if err := a.decode(resp, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// HealthReady issues a Kubernetes readiness probe (GET /health/ready).
func (a *Adapter) HealthReady(ctx context.Context) (*ProbeResponse, error) {
	resp, err := a.gen.GetReadiness(ctx, nil)
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}
	var result ProbeResponse
	if err := a.decode(resp, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// GetCollectionSchema fetches the schema for a collection.
func (a *Adapter) GetCollectionSchema(ctx context.Context, collectionID string) (*SchemaResponse, error) {
	resp, err := a.gen.GetCollectionSchema(ctx, collectionID, nil)
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}
	var result SchemaResponse
	if err := a.decode(resp, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// UpdateCollectionSchema updates the schema for a collection.
func (a *Adapter) UpdateCollectionSchema(ctx context.Context, collectionID string, req *UpdateSchemaRequest) (*UpdateSchemaResponse, error) {
	body := genrest.UpdateCollectionSchemaJSONRequestBody{
		Columns:               toGenColumns(req.Columns),
		AllowAdditionalFields: req.AllowAdditionalFields,
		Force:                 ptr(req.Force),
	}
	if req.Enforcement != "" {
		body.Enforcement = ptr(req.Enforcement)
	}

	resp, err := a.gen.UpdateCollectionSchema(ctx, collectionID, nil, body)
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}
	var result UpdateSchemaResponse
	if err := a.decode(resp, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// ExecuteQuery executes an AQL/UQL/federated query via the shared query facade.
func (a *Adapter) ExecuteQuery(ctx context.Context, req *QueryRequest) (QueryResponse, error) {
	body := genrest.ExecuteQueryJSONRequestBody{
		Language:   genrest.QueryLanguage(req.Language),
		Query:      req.Query,
		Collection: req.Collection,
	}
	if req.Limit != nil {
		body.Limit = ptr(int32(*req.Limit))
	}
	if len(req.Parameters) > 0 {
		body.Parameters = toGenQueryParams(req.Parameters)
	}

	resp, err := a.gen.ExecuteQuery(ctx, nil, body)
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}
	var result QueryResponse
	if err := a.decode(resp, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// ExplainQuery returns the lowered plan / diagnostics for an AQL/UQL query.
func (a *Adapter) ExplainQuery(ctx context.Context, req *ExplainQueryRequest) (QueryResponse, error) {
	body := genrest.ExplainQueryJSONRequestBody{
		Language:   genrest.QueryLanguage(req.Language),
		Query:      req.Query,
		Collection: req.Collection,
	}

	resp, err := a.gen.ExplainQuery(ctx, nil, body)
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}
	var result QueryResponse
	if err := a.decode(resp, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// Close closes the adapter and releases resources.
func (a *Adapter) Close() error {
	if a.httpDoer != nil {
		if transport, ok := a.httpDoer.Transport.(*http.Transport); ok {
			transport.CloseIdleConnections()
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Facade plumbing: response decoding + error mapping (the value-add the
// generator doesn't provide).
// ---------------------------------------------------------------------------

// decode reads/closes the generated client's *http.Response, maps non-2xx
// statuses to AdapterError, and unmarshals a 2xx JSON body into result (when
// non-nil).
func (a *Adapter) decode(resp *http.Response, result interface{}) error {
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return &AdapterError{
			Code:    ErrCodeInternal,
			Message: "failed to read response",
			Cause:   err,
		}
	}

	if resp.StatusCode >= 400 {
		return parseErrorResponse(resp.StatusCode, respBody)
	}

	if result != nil && len(respBody) > 0 {
		if err := json.Unmarshal(respBody, result); err != nil {
			return &AdapterError{
				Code:    ErrCodeInternal,
				Message: "failed to parse response",
				Cause:   err,
			}
		}
	}
	return nil
}

// mapTransportError converts a transport-level error (connection / timeout)
// from the generated client into an AdapterError.
func mapTransportError(ctx context.Context, err error) error {
	if ctx.Err() == context.DeadlineExceeded {
		return &AdapterError{
			Code:    ErrCodeTimeout,
			Message: "request timed out",
			Cause:   err,
		}
	}
	return &AdapterError{
		Code:    ErrCodeConnection,
		Message: "connection error",
		Cause:   err,
	}
}

// parseErrorResponse parses an error response from the server.
func parseErrorResponse(statusCode int, body []byte) error {
	var errorResp struct {
		Error   string `json:"error"`
		Message string `json:"message"`
		Code    string `json:"code"`
	}
	if err := json.Unmarshal(body, &errorResp); err != nil {
		errorResp.Message = string(body)
	}

	var code ErrorCode
	switch statusCode {
	case http.StatusNotFound:
		code = ErrCodeNotFound
	case http.StatusConflict:
		code = ErrCodeAlreadyExists
	case http.StatusBadRequest:
		code = ErrCodeInvalidArgument
	case http.StatusUnprocessableEntity:
		code = ErrCodeDimensionMismatch
	case http.StatusTooManyRequests:
		code = ErrCodeRateLimited
	case http.StatusServiceUnavailable:
		code = ErrCodeUnavailable
	default:
		code = ErrCodeInternal
	}

	if errorResp.Code != "" {
		code = ErrorCode(errorResp.Code)
	}

	message := errorResp.Message
	if message == "" {
		message = errorResp.Error
	}
	if message == "" {
		message = fmt.Sprintf("HTTP %d", statusCode)
	}

	return &AdapterError{Code: code, Message: message}
}

// ---------------------------------------------------------------------------
// Mapping helpers (public model <-> generated wire types) + small utilities.
// ---------------------------------------------------------------------------

func ptr[T any](v T) *T { return &v }

func toGenRecords(records []*ProximaRecord) []genrest.ProximaRecordInput {
	out := make([]genrest.ProximaRecordInput, len(records))
	for i, r := range records {
		gr := genrest.ProximaRecordInput{Vector: r.Vector}
		if r.ID != "" {
			gr.Id = ptr(r.ID)
		}
		if r.Props != nil {
			props := r.Props
			gr.Props = &props
		}
		out[i] = gr
	}
	return out
}

func toGenColumns(cols []ColumnDefinition) []genrest.RestColumnDefinition {
	out := make([]genrest.RestColumnDefinition, len(cols))
	for i, c := range cols {
		out[i] = genrest.RestColumnDefinition{
			Name:            c.Name,
			DataType:        c.DataType,
			Nullable:        c.Nullable,
			Indexed:         c.Indexed,
			Filterable:      c.Filterable,
			MaxLength:       intPtrToInt32Ptr(c.MaxLength),
			Precision:       intPtrToInt32Ptr(c.Precision),
			Scale:           intPtrToInt32Ptr(c.Scale),
			VectorDimension: intPtrToInt32Ptr(c.VectorDimension),
		}
	}
	return out
}

func toGenQueryParams(params []interface{}) *[]map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(params))
	for _, p := range params {
		if m, ok := p.(map[string]interface{}); ok {
			out = append(out, m)
		} else {
			out = append(out, map[string]interface{}{"value": p})
		}
	}
	return &out
}

func intPtrToInt32Ptr(v *int) *int32 {
	if v == nil {
		return nil
	}
	out := int32(*v)
	return &out
}

func recordsToProximaRecords(records []*VectorRecord) []*ProximaRecord {
	result := make([]*ProximaRecord, len(records))
	for i, record := range records {
		result[i] = &ProximaRecord{
			ID:     record.ID,
			Vector: record.Vector,
			Props:  record.Metadata,
		}
	}
	return result
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func parseTimeOrZero(value string) time.Time {
	if value == "" {
		return time.Time{}
	}
	parsed, err := time.Parse(time.RFC3339, value)
	if err != nil {
		return time.Time{}
	}
	return parsed
}
