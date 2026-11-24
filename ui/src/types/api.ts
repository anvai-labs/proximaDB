export interface Collection {
  id: string;
  config: CollectionConfig;
  stats: CollectionStats;
  created_at: string; // Assuming string representation of int64
  updated_at: string; // Assuming string representation of int64
  storage_assignment?: StorageAssignment;
}

export interface CollectionConfig {
  name: string;
  dimension: number;
  distance_metric: DistanceMetric;
  storage_engine: StorageEngine;
  description?: string;
  tags: string[];
  owner?: string;
}

export interface CollectionStats {
  vector_count: string; // Assuming string representation of int64
  index_size_bytes: string; // Assuming string representation of int64
  data_size_bytes: string; // Assuming string representation of int64
}

export interface StorageAssignment {
  base_location: string;
  assigned_at: string; // Assuming string representation of int64
}

export enum DistanceMetric {
  DISTANCE_METRIC_UNSPECIFIED = 0,
  COSINE = 1,
  EUCLIDEAN = 2,
  DOT_PRODUCT = 3,
  HAMMING = 4,
  MANHATTAN = 5,
  JACCARD = 6,
  CUSTOM = 7,
  CHEBYSHEV = 8,
  CANBERRA = 9,
  MINKOWSKI = 10,
  ANGULAR = 11,
  BRAY_CURTIS = 12,
  HELLINGER = 13,
}

export enum StorageEngine {
  STORAGE_ENGINE_UNSPECIFIED = 0,
  VIPER = 1, // REMOVED (ADR-093) — wire value only
  SST = 2,
  MMAP = 3,
  HYBRID = 4,
  SWIFT = 5,
  NOVA = 6,
}

export enum CollectionOperation {
  COLLECTION_OPERATION_UNSPECIFIED = 0,
  COLLECTION_CREATE = 1,
  COLLECTION_UPDATE = 2,
  COLLECTION_GET = 3,
  COLLECTION_LIST = 4,
  COLLECTION_DELETE = 5,
  COLLECTION_MIGRATE = 6,
  COLLECTION_GET_ID_BY_NAME = 7,
}

export enum VectorOperation {
  VECTOR_OPERATION_UNSPECIFIED = 0,
  VECTOR_BATCH = 1,
  VECTOR_SEARCH = 2,
  VECTOR_GET = 3,
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: string; // Assuming string representation of int64
  active_connections: number;
  memory_usage_bytes: string; // Assuming string representation of int64
  storage_usage_bytes: string; // Assuming string representation of int64
}

export interface MetricsResponse {
  metrics: { [key: string]: number };
  timestamp: string; // Assuming string representation of int64
}
