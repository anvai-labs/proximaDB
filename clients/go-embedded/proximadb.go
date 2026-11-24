// Package proximadb provides Go bindings for ProximaDB embedded mode.
//
// ProximaDB is a high-performance vector database with zero network overhead
// when used in embedded mode. This package provides CGO bindings to the Rust core.
//
// Example usage:
//
//	db, err := proximadb.Open("./my_database", nil)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer db.Close()
//
//	// Create collection
//	err = db.CreateCollection("embeddings", 768, "")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Insert vectors
//	ids := []string{"vec_0", "vec_1"}
//	vectors := [][]float32{{0.1, 0.2, ...}, {0.3, 0.4, ...}}
//	err = db.Insert("embeddings", ids, vectors)
//
//	// Search
//	results, err := db.Search("embeddings", query, 10)
package proximadb

/*
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -lproximadb -ldl -lpthread -lm
#include <stdlib.h>
#include <stdint.h>

// Opaque handle
typedef void* ProximaDBHandle;

// Search result
typedef struct {
    char* id;
    float score;
} CSearchResult;

// Search results array
typedef struct {
    CSearchResult* results;
    int count;
} CSearchResults;

// Result with error info
typedef struct {
    int error_code;
    char* error_message;
} CResult;

// FFI functions
extern ProximaDBHandle proximadb_open(const char* data_dir, const char* metadata_dir, int cache_size_mb, const char* engine);
extern ProximaDBHandle proximadb_open_multi_disk(const char** disk_paths, const int* disk_weights, int disk_count, const char* metadata_dir, int cache_size_mb, const char* engine);
extern void proximadb_close(ProximaDBHandle handle);
extern CResult proximadb_create_collection(ProximaDBHandle handle, const char* name, int dimension, const char* engine);
extern CResult proximadb_delete_collection(ProximaDBHandle handle, const char* name);
extern CResult proximadb_insert(ProximaDBHandle handle, const char* collection, const char** ids, const float* vectors, int count, int dimension);
extern CSearchResults proximadb_search(ProximaDBHandle handle, const char* collection, const float* query, int dimension, int top_k);
extern void proximadb_free_search_results(CSearchResults results);
extern CResult proximadb_flush(ProximaDBHandle handle);
extern void proximadb_free_string(char* s);
extern void proximadb_free_result(CResult result);
extern const char* proximadb_version();
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// DB represents an embedded ProximaDB instance.
type DB struct {
	handle C.ProximaDBHandle
}

// Config holds configuration options for opening a database.
type Config struct {
	// MetadataDir is the path to store metadata (defaults to DataDir/metadata)
	MetadataDir string
	// CacheSizeMB is the cache size in megabytes (default: 512)
	CacheSizeMB int
	// DefaultEngine is the default storage engine ("sst", "nova", "helix", etc.)
	DefaultEngine string
}

// DiskConfig represents a storage disk configuration for multi-disk setups.
type DiskConfig struct {
	Path   string
	Weight int
}

// SearchResult represents a single search result.
type SearchResult struct {
	ID    string
	Score float32
}

// Open creates a new embedded ProximaDB instance with a single data directory.
//
// Example:
//
//	db, err := proximadb.Open("./data", nil)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer db.Close()
func Open(dataDir string, config *Config) (*DB, error) {
	if config == nil {
		config = &Config{}
	}

	cDataDir := C.CString(dataDir)
	defer C.free(unsafe.Pointer(cDataDir))

	var cMetadataDir *C.char
	if config.MetadataDir != "" {
		cMetadataDir = C.CString(config.MetadataDir)
		defer C.free(unsafe.Pointer(cMetadataDir))
	}

	cacheSizeMB := config.CacheSizeMB
	if cacheSizeMB <= 0 {
		cacheSizeMB = 512
	}

	var cEngine *C.char
	if config.DefaultEngine != "" {
		cEngine = C.CString(config.DefaultEngine)
		defer C.free(unsafe.Pointer(cEngine))
	}

	handle := C.proximadb_open(cDataDir, cMetadataDir, C.int(cacheSizeMB), cEngine)
	if handle == nil {
		return nil, errors.New("failed to open database")
	}

	return &DB{handle: handle}, nil
}

// OpenMultiDisk creates a new embedded ProximaDB instance with multiple data directories.
//
// Example:
//
//	disks := []proximadb.DiskConfig{
//		{Path: "/nvme/data", Weight: 2},
//		{Path: "/hdd/data", Weight: 1},
//	}
//	db, err := proximadb.OpenMultiDisk(disks, nil)
func OpenMultiDisk(disks []DiskConfig, config *Config) (*DB, error) {
	if len(disks) == 0 {
		return nil, errors.New("at least one disk config required")
	}

	if config == nil {
		config = &Config{}
	}

	// Prepare C arrays
	diskCount := len(disks)
	cPaths := make([]*C.char, diskCount)
	cWeights := make([]C.int, diskCount)

	for i, disk := range disks {
		cPaths[i] = C.CString(disk.Path)
		defer C.free(unsafe.Pointer(cPaths[i]))

		weight := disk.Weight
		if weight <= 0 {
			weight = 1
		}
		cWeights[i] = C.int(weight)
	}

	var cMetadataDir *C.char
	if config.MetadataDir != "" {
		cMetadataDir = C.CString(config.MetadataDir)
		defer C.free(unsafe.Pointer(cMetadataDir))
	}

	cacheSizeMB := config.CacheSizeMB
	if cacheSizeMB <= 0 {
		cacheSizeMB = 512
	}

	var cEngine *C.char
	if config.DefaultEngine != "" {
		cEngine = C.CString(config.DefaultEngine)
		defer C.free(unsafe.Pointer(cEngine))
	}

	handle := C.proximadb_open_multi_disk(
		(**C.char)(unsafe.Pointer(&cPaths[0])),
		(*C.int)(unsafe.Pointer(&cWeights[0])),
		C.int(diskCount),
		cMetadataDir,
		C.int(cacheSizeMB),
		cEngine,
	)

	if handle == nil {
		return nil, errors.New("failed to open database")
	}

	return &DB{handle: handle}, nil
}

// Close closes the database and releases resources.
func (db *DB) Close() {
	if db.handle != nil {
		C.proximadb_close(db.handle)
		db.handle = nil
	}
}

// CreateCollection creates a new vector collection.
//
// Parameters:
//   - name: Collection name
//   - dimension: Vector dimension
//   - engine: Storage engine type (empty string for default)
func (db *DB) CreateCollection(name string, dimension int, engine string) error {
	if db.handle == nil {
		return errors.New("database not open")
	}

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var cEngine *C.char
	if engine != "" {
		cEngine = C.CString(engine)
		defer C.free(unsafe.Pointer(cEngine))
	}

	result := C.proximadb_create_collection(db.handle, cName, C.int(dimension), cEngine)
	return checkResult(result)
}

// DeleteCollection deletes a collection.
func (db *DB) DeleteCollection(name string) error {
	if db.handle == nil {
		return errors.New("database not open")
	}

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	result := C.proximadb_delete_collection(db.handle, cName)
	return checkResult(result)
}

// Insert adds vectors to a collection.
//
// Parameters:
//   - collection: Collection name
//   - ids: Vector IDs
//   - vectors: 2D slice of vectors (each inner slice has same dimension)
func (db *DB) Insert(collection string, ids []string, vectors [][]float32) error {
	if db.handle == nil {
		return errors.New("database not open")
	}

	if len(ids) != len(vectors) {
		return errors.New("ids and vectors must have same length")
	}

	if len(ids) == 0 {
		return nil
	}

	cCollection := C.CString(collection)
	defer C.free(unsafe.Pointer(cCollection))

	// Convert IDs to C strings
	count := len(ids)
	cIDs := make([]*C.char, count)
	for i, id := range ids {
		cIDs[i] = C.CString(id)
		defer C.free(unsafe.Pointer(cIDs[i]))
	}

	// Flatten vectors
	dimension := len(vectors[0])
	flatVectors := make([]C.float, count*dimension)
	for i, vec := range vectors {
		if len(vec) != dimension {
			return errors.New("all vectors must have same dimension")
		}
		for j, v := range vec {
			flatVectors[i*dimension+j] = C.float(v)
		}
	}

	result := C.proximadb_insert(
		db.handle,
		cCollection,
		(**C.char)(unsafe.Pointer(&cIDs[0])),
		(*C.float)(unsafe.Pointer(&flatVectors[0])),
		C.int(count),
		C.int(dimension),
	)

	return checkResult(result)
}

// Search finds similar vectors in a collection.
//
// Parameters:
//   - collection: Collection name
//   - query: Query vector
//   - topK: Number of results to return
func (db *DB) Search(collection string, query []float32, topK int) ([]SearchResult, error) {
	if db.handle == nil {
		return nil, errors.New("database not open")
	}

	cCollection := C.CString(collection)
	defer C.free(unsafe.Pointer(cCollection))

	cQuery := make([]C.float, len(query))
	for i, v := range query {
		cQuery[i] = C.float(v)
	}

	cResults := C.proximadb_search(
		db.handle,
		cCollection,
		(*C.float)(unsafe.Pointer(&cQuery[0])),
		C.int(len(query)),
		C.int(topK),
	)
	defer C.proximadb_free_search_results(cResults)

	if cResults.count <= 0 || cResults.results == nil {
		return nil, nil
	}

	// Convert C results to Go
	results := make([]SearchResult, cResults.count)
	cResultSlice := unsafe.Slice(cResults.results, cResults.count)

	for i, cr := range cResultSlice {
		results[i] = SearchResult{
			ID:    C.GoString(cr.id),
			Score: float32(cr.score),
		}
	}

	return results, nil
}

// Flush writes all pending data to disk.
func (db *DB) Flush() error {
	if db.handle == nil {
		return errors.New("database not open")
	}

	result := C.proximadb_flush(db.handle)
	return checkResult(result)
}

// Version returns the ProximaDB library version.
func Version() string {
	return C.GoString(C.proximadb_version())
}

// Helper to check CResult for errors
func checkResult(result C.CResult) error {
	if result.error_code != 0 {
		msg := C.GoString(result.error_message)
		C.proximadb_free_result(result)
		return fmt.Errorf("proximadb error: %s", msg)
	}
	return nil
}
