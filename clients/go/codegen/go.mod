// Isolated module for the OpenAPI -> Go code generator (TD-126 Phase 2).
//
// Keeping the generator in its own module means the SDK module
// (github.com/proximadb/proximadb-go) only depends on the small generated-code
// runtime (github.com/oapi-codegen/runtime), not the entire oapi-codegen
// toolchain (kin-openapi, speakeasy, etc.). `make gen-go-sdk` runs the generator
// from here; the version below is the single pin.
module github.com/proximadb/proximadb-go/codegen

go 1.25.0

require github.com/oapi-codegen/oapi-codegen/v2 v2.8.0

require (
	github.com/dprotaso/go-yit v0.0.0-20220510233725-9ba8df137936 // indirect
	github.com/getkin/kin-openapi v0.149.0 // indirect
	github.com/go-openapi/jsonpointer v0.23.1 // indirect
	github.com/go-openapi/swag/jsonname v0.26.0 // indirect
	github.com/oasdiff/yaml v0.1.1 // indirect
	github.com/oasdiff/yaml3 v0.0.14 // indirect
	github.com/onsi/gomega v1.27.6 // indirect
	github.com/santhosh-tekuri/jsonschema/v6 v6.0.3 // indirect
	github.com/speakeasy-api/jsonpath v0.6.3 // indirect
	github.com/speakeasy-api/openapi v1.24.0 // indirect
	github.com/vmware-labs/yaml-jsonpath v0.3.2 // indirect
	go.yaml.in/yaml/v3 v3.0.4 // indirect
	golang.org/x/mod v0.38.0 // indirect
	golang.org/x/sync v0.22.0 // indirect
	golang.org/x/text v0.40.0 // indirect
	golang.org/x/tools v0.48.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
