# ProximaDB Embedded for Node.js

Zero-overhead embedded vector database for Node.js using NAPI-RS bindings to ProximaDB's Rust core.

## Features

- **Zero Network Overhead** - Direct in-process access to Rust core
- **Multi-Disk Support** - Configure multiple storage locations with weighted distribution
- **SIMD Acceleration** - Automatic AVX2/NEON optimization
- **Full Persistence** - WAL with configurable sync modes
- **TypeScript Support** - Full type definitions included

## Installation

```bash
npm install proximadb-embedded
```

## Quick Start

```javascript
const { ProximaDB } = require('proximadb-embedded');

// Create database with simple configuration
const db = new ProximaDB({ dataDir: './my_database' });

// Or with multi-disk support
const dbMulti = new ProximaDB({
  dataDirs: [
    { path: '/nvme/data', weight: 2 },  // Fast SSD - gets more data
    { path: '/hdd/data', weight: 1 },   // Slower HDD
  ],
  metadataDir: '/nvme/metadata',
  cacheSizeMb: 2048,
});

// Create collection
db.createCollection('embeddings', 768, 'sst');

// Insert vectors
const ids = ['doc_0', 'doc_1', 'doc_2'];
const vectors = [
  [0.1, 0.2, /* ... 768 dimensions */],
  [0.3, 0.4, /* ... */],
  [0.5, 0.6, /* ... */],
];
const metadata = [
  { category: 'tech' },
  { category: 'science' },
  { category: 'tech' },
];

db.insert('embeddings', ids, vectors, metadata);

// Search for similar vectors
const query = [0.1, 0.2, /* ... */];
const results = db.search('embeddings', query, 10);

for (const r of results) {
  console.log(`${r.id}: score=${r.score.toFixed(4)}`);
}

// Flush to ensure durability
db.flush();
```

## TypeScript

```typescript
import { ProximaDB, SearchResult, CollectionInfo } from 'proximadb-embedded';

const db = new ProximaDB({ dataDir: './data' });

db.createCollection('vectors', 128);

const results: SearchResult[] = db.search('vectors', query, 10);
const info: CollectionInfo | null = db.getCollection('vectors');
```

## API Reference

### Constructor

```javascript
new ProximaDB(config?: ProximaDBConfig)
```

Configuration options:
- `dataDir` - Single data directory path
- `dataDirs` - Array of disk configurations for multi-disk
- `metadataDir` - Metadata storage path
- `cacheSizeMb` - Cache size in MB (default: 512)
- `defaultEngine` - Storage engine: "sst" (recommended), "nova", "helix", etc. (`viper` was REMOVED — see ADR-093)
- `enableWal` - Enable write-ahead logging (default: true)
- `walSyncMode` - WAL sync: "immediate", "batch", "async"

### Methods

#### createCollection(name, dimension, engine?)
Create a new vector collection.

#### deleteCollection(name)
Delete a collection.

#### getCollection(name)
Get collection information or null.

#### listCollections()
List all collections.

#### insert(collection, ids, vectors, metadata?)
Insert vectors. Returns count inserted.

#### search(collection, query, topK?, filter?)
Search for similar vectors. Returns array of results.

#### flush()
Flush pending writes to disk.

#### stats()
Get storage statistics.

### Storage Engines

| Engine | Best For |
|--------|----------|
| `sst` | OLTP, real-time queries (default) |
| `nova` | Mixed workloads, columnar analytics |
| `swift` | Hot data, low latency |
| `raptor` | Adaptive, multi-tenant |
| `helix` | High-dimensional vectors |

## Building from Source

### Prerequisites

- Node.js 16+
- Rust 1.88+
- @napi-rs/cli

### Build Steps

```bash
# Install NAPI CLI
npm install -g @napi-rs/cli

# Build the package
cd clients/nodejs-embedded
npm install
npm run build
```

## License

Apache License 2.0

## Links

- [ProximaDB Repository](https://github.com/anvai-labs/proximaDB)
- [Documentation](https://github.com/anvai-labs/proximaDB#readme)
