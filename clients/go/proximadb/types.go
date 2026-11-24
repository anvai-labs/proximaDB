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

// Package proximadb provides the official Go client for ProximaDB.
package proximadb

import (
	"encoding/json"
	"time"
)

// Protocol represents the communication protocol.
type Protocol string

const (
	// ProtocolREST uses HTTP/REST protocol.
	ProtocolREST Protocol = "rest"
	// ProtocolGRPC uses gRPC protocol.
	ProtocolGRPC Protocol = "grpc"
)

// DistanceMetric represents the distance metric for vector similarity.
type DistanceMetric string

const (
	// Cosine distance metric.
	Cosine DistanceMetric = "cosine"
	// Euclidean (L2) distance metric.
	Euclidean DistanceMetric = "euclidean"
	// DotProduct distance metric.
	DotProduct DistanceMetric = "dot_product"
)

// StorageEngine represents the storage engine type.
type StorageEngine string

const (
	// EngineSst is the write-optimized SST engine.
	EngineSst StorageEngine = "sst"
	// EngineHelix is the locality-optimized Hilbert curve engine.
	EngineHelix StorageEngine = "helix"
	// EngineViper is a REMOVED engine (ADR-093 stage 3): retained as a wire
	// value only — the server rejects new selections; use EngineNova for
	// columnar analytics.
	EngineViper StorageEngine = "viper"
	// EngineSwift is the ultra-low latency engine for small collections.
	EngineSwift StorageEngine = "swift"
	// EngineNova is the progressive columnar engine.
	EngineNova StorageEngine = "nova"
	// EngineRaptor is the adaptive row-group engine.
	EngineRaptor StorageEngine = "raptor"
)

// ProximaRecord represents the canonical record payload with optional vector data.
type ProximaRecord struct {
	// ID is the unique identifier for the record.
	ID string `json:"id"`
	// Vector is the embedding data.
	Vector []float32 `json:"vector,omitempty"`
	// Props contains rich record properties.
	Props map[string]interface{} `json:"props,omitempty"`
	// Source is the original content reference.
	Source string `json:"source,omitempty"`
}

// VectorRecord represents a legacy vector-shaped compatibility payload.
//
// Deprecated: use ProximaRecord with Client.InsertRecords or Client.UpsertRecords.
type VectorRecord struct {
	// ID is the unique identifier for the vector.
	ID string `json:"id"`
	// Vector is the embedding data.
	Vector []float32 `json:"vector"`
	// Metadata contains additional key-value pairs.
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchResult represents a single search result.
type SearchResult struct {
	// ID is the vector ID.
	ID string `json:"id"`
	// Score is the similarity score.
	Score float32 `json:"score"`
	// Vector is the vector data (if requested).
	Vector []float32 `json:"vector,omitempty"`
	// Metadata contains the vector metadata (if requested).
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchResponse represents the response from a search query.
type SearchResponse struct {
	// Results contains the matching vectors.
	Results []SearchResult `json:"results"`
	// TookMs is the time taken in milliseconds.
	TookMs float64 `json:"took_ms"`
	// TotalCount is the total number of matching vectors.
	TotalCount int64 `json:"total_count,omitempty"`
}

// CollectionInfo contains information about a collection.
type CollectionInfo struct {
	// Name is the collection name.
	Name string `json:"name"`
	// Dimension is the vector dimension.
	Dimension int `json:"dimension"`
	// Metric is the distance metric used.
	Metric DistanceMetric `json:"metric"`
	// Engine is the storage engine.
	Engine StorageEngine `json:"engine"`
	// VectorCount is the number of vectors, or nil when the server does not
	// know. nil and 0 are different answers: 0 means the server knows the
	// collection is empty, nil means the counter was never maintained.
	// They used to be indistinguishable, and reading the fabricated 0 as
	// "empty" produced a false data-loss report against the server.
	VectorCount *int64 `json:"vector_count,omitempty"`
	// CreatedAt is the creation timestamp.
	CreatedAt time.Time `json:"created_at"`
}

// CreateCollectionRequest contains parameters for creating a collection.
type CreateCollectionRequest struct {
	// Name is the collection name.
	Name string `json:"name"`
	// Dimension is the vector dimension.
	Dimension int `json:"dimension"`
	// Metric is the distance metric (default: cosine).
	Metric DistanceMetric `json:"metric,omitempty"`
	// Engine is the storage engine (default: sst).
	Engine StorageEngine `json:"engine,omitempty"`
	// Description is an optional description.
	Description string `json:"description,omitempty"`
}

// SearchQuery represents a vector search query.
type SearchQuery struct {
	// Vector is the query vector.
	Vector []float32 `json:"vector"`
	// TopK is the number of results to return (default: 10).
	TopK int `json:"top_k,omitempty"`
	// Filter is an optional metadata filter.
	Filter *Filter `json:"filter,omitempty"`
	// IncludeVectors indicates whether to include vectors in results.
	IncludeVectors bool `json:"include_vectors,omitempty"`
	// IncludeMetadata indicates whether to include metadata in results.
	IncludeMetadata bool `json:"include_metadata,omitempty"`
}

// Filter represents a metadata filter for queries.
type Filter struct {
	// Field is the metadata field name.
	Field string `json:"field,omitempty"`
	// Operator is the comparison operator.
	Operator FilterOperator `json:"operator,omitempty"`
	// Value is the comparison value.
	Value interface{} `json:"value,omitempty"`
	// And combines multiple filters with AND.
	And []Filter `json:"and,omitempty"`
	// Or combines multiple filters with OR.
	Or []Filter `json:"or,omitempty"`
}

// FilterOperator represents a comparison operator.
type FilterOperator string

const (
	// OpEquals checks for equality.
	OpEquals FilterOperator = "eq"
	// OpNotEquals checks for inequality.
	OpNotEquals FilterOperator = "ne"
	// OpGreaterThan checks if greater than.
	OpGreaterThan FilterOperator = "gt"
	// OpGreaterThanOrEqual checks if greater than or equal.
	OpGreaterThanOrEqual FilterOperator = "gte"
	// OpLessThan checks if less than.
	OpLessThan FilterOperator = "lt"
	// OpLessThanOrEqual checks if less than or equal.
	OpLessThanOrEqual FilterOperator = "lte"
	// OpIn checks if value is in a list.
	OpIn FilterOperator = "in"
	// OpNotIn checks if value is not in a list.
	OpNotIn FilterOperator = "not_in"
	// OpContains checks if string contains substring.
	OpContains FilterOperator = "contains"
)

// HealthStatus represents the health status of the server.
type HealthStatus struct {
	// Status is the overall status (healthy, degraded, unhealthy).
	Status string `json:"status"`
	// Version is the server version.
	Version string `json:"version"`
	// Uptime is the server uptime in seconds.
	Uptime float64 `json:"uptime_seconds"`
}

// ProbeResponse represents a Kubernetes-style liveness/readiness probe.
type ProbeResponse struct {
	// Status is the probe status (e.g., "ok", "ready").
	Status string `json:"status"`
}

// ColumnDefinition declares a typed column in a SchemaDefinition.
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

// SchemaDefinition is a typed collection schema.
type SchemaDefinition struct {
	Columns               []ColumnDefinition `json:"columns"`
	Enforcement           string             `json:"enforcement,omitempty"`
	AllowAdditionalFields *bool              `json:"allow_additional_fields,omitempty"`
}

// SchemaResponse is the response of GetCollectionSchema.
type SchemaResponse struct {
	SchemaID       string           `json:"schema_id"`
	SchemaVersion  string           `json:"schema_version"`
	CollectionID   string           `json:"collection_id"`
	Schema         SchemaDefinition `json:"schema"`
	CreatedAt      string           `json:"created_at"`
	UpdatedAt      *string          `json:"updated_at,omitempty"`
	ParentSchemaID *string          `json:"parent_schema_id,omitempty"`
}

// UpdateSchemaRequest is the body sent to UpdateCollectionSchema.
// It mirrors the OpenAPI `UpdateSchemaRequest` shape (SchemaDefinition + force).
type UpdateSchemaRequest struct {
	Columns               []ColumnDefinition `json:"columns"`
	Enforcement           string             `json:"enforcement,omitempty"`
	AllowAdditionalFields *bool              `json:"allow_additional_fields,omitempty"`
	Force                 bool               `json:"force,omitempty"`
}

// UpdateSchemaResponse is the response from UpdateCollectionSchema.
type UpdateSchemaResponse struct {
	SchemaID         string                   `json:"schema_id"`
	SchemaVersion    string                   `json:"schema_version"`
	PreviousSchemaID string                   `json:"previous_schema_id"`
	Changes          []map[string]interface{} `json:"changes"`
	Warnings         []string                 `json:"warnings"`
	UpdatedAt        string                   `json:"updated_at"`
}

// QueryLanguage indicates the dialect used for a query facade call.
type QueryLanguage string

const (
	// QueryLanguageUQL is the unified query language.
	QueryLanguageUQL QueryLanguage = "uql"
	// QueryLanguageAQL is the ProximaDB AQL dialect.
	QueryLanguageAQL QueryLanguage = "aql"
	// QueryLanguageFederated routes through the federated planner.
	QueryLanguageFederated QueryLanguage = "federated"
)

// QueryRequest is the request body for ExecuteQuery.
type QueryRequest struct {
	Language   QueryLanguage `json:"language"`
	Query      string        `json:"query"`
	Parameters []interface{} `json:"parameters,omitempty"`
	Collection *string       `json:"collection,omitempty"`
	Limit      *int          `json:"limit,omitempty"`
}

// ExplainQueryRequest is the request body for ExplainQuery.
type ExplainQueryRequest struct {
	Language   QueryLanguage `json:"language"`
	Query      string        `json:"query"`
	Collection *string       `json:"collection,omitempty"`
}

// QueryResponse is the open-shape response from the query facade endpoints.
// The server is documented as returning records, total_count, metrics, plan, or
// diagnostics depending on language/endpoint; the SDK passes it through.
type QueryResponse map[string]interface{}

// BatchInsertResult contains the result of a batch insert operation.
type BatchInsertResult struct {
	// InsertedCount is the number of successfully inserted vectors.
	InsertedCount int `json:"inserted_count"`
	// FailedCount is the number of failed insertions.
	FailedCount int `json:"failed_count"`
	// Errors contains error details for failed insertions.
	Errors []BatchError `json:"errors,omitempty"`
}

// BatchError represents an error for a specific vector in a batch operation.
type BatchError struct {
	// ID is the vector ID that failed.
	ID string `json:"id"`
	// Error is the error message.
	Error string `json:"error"`
}

// MarshalJSON implements custom JSON marshaling for Filter.
func (f Filter) MarshalJSON() ([]byte, error) {
	type filterAlias Filter
	return json.Marshal(filterAlias(f))
}

// Eq creates an equality filter.
func Eq(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpEquals, Value: value}
}

// Ne creates an inequality filter.
func Ne(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpNotEquals, Value: value}
}

// Gt creates a greater-than filter.
func Gt(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpGreaterThan, Value: value}
}

// Gte creates a greater-than-or-equal filter.
func Gte(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpGreaterThanOrEqual, Value: value}
}

// Lt creates a less-than filter.
func Lt(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpLessThan, Value: value}
}

// Lte creates a less-than-or-equal filter.
func Lte(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpLessThanOrEqual, Value: value}
}

// In creates an "in" filter.
func In(field string, values ...interface{}) Filter {
	return Filter{Field: field, Operator: OpIn, Value: values}
}

// And combines multiple filters with AND logic.
func And(filters ...Filter) Filter {
	return Filter{And: filters}
}

// Or combines multiple filters with OR logic.
func Or(filters ...Filter) Filter {
	return Filter{Or: filters}
}
