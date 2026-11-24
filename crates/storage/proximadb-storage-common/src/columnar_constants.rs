//! Column name constants for columnar storage formats (Parquet, ProximaBlocks).

pub const FIELD_ID: &str = "id";
pub const FIELD_COLLECTION_ID: &str = "collection_id";

pub const FIELD_VECTOR_FP32: &str = "vector_fp32";
pub const FIELD_Q_BINARY: &str = "q_binary";
pub const FIELD_Q_INT8: &str = "q_int8";
pub const FIELD_Q_PQ4: &str = "q_pq4";
pub const FIELD_Q_PQ8: &str = "q_pq8";
pub const FIELD_Q_PQ16: &str = "q_pq16";
pub const FIELD_Q_PQ32: &str = "q_pq32";

pub const FIELD_QP_BINARY_THRESHOLD: &str = "qp_binary_threshold";
pub const FIELD_QP_INT8_MIN: &str = "qp_int8_min";
pub const FIELD_QP_INT8_MAX: &str = "qp_int8_max";
pub const FIELD_QP_INT8_SCALE: &str = "qp_int8_scale";
pub const FIELD_QP_PQ_SUBQUANTIZERS: &str = "qp_pq_subquantizers";
pub const FIELD_QP_PQ_CENTROIDS: &str = "qp_pq_centroids";

pub const FIELD_ROW_GROUP_OFFSET: &str = "row_group_offset";
pub const FIELD_ROW_INDEX: &str = "row_index";

pub const FIELD_TIMESTAMP: &str = "timestamp";
pub const FIELD_UPDATED_AT: &str = "updated_at";
pub const FIELD_EXPIRES_AT: &str = "expires_at";
pub const FIELD_VERSION: &str = "version";

pub const FIELD_EXTRA_META: &str = "extra_meta";
pub const FIELD_SOURCE: &str = "source";
pub const FIELD_IS_DELETED: &str = "is_deleted";
pub const FIELD_SCHEMA_VERSION: &str = "schema_version";

pub const PARQUET_EXTENSION: &str = ".parquet";

pub const DEFAULT_ROW_GROUP_SIZE: usize = 2048;
pub const DEFAULT_PAGE_SIZE: usize = 256 * 1024;
pub const DEFAULT_WRITE_BATCH_SIZE: usize = 1000;

pub const QUANTIZED_VECTOR_COLUMNS: &[&str] = &[
    FIELD_Q_BINARY,
    FIELD_Q_INT8,
    FIELD_Q_PQ4,
    FIELD_Q_PQ8,
    FIELD_Q_PQ16,
    FIELD_Q_PQ32,
];

pub const QUANTIZATION_PARAMETER_COLUMNS: &[&str] = &[
    FIELD_QP_BINARY_THRESHOLD,
    FIELD_QP_INT8_MIN,
    FIELD_QP_INT8_MAX,
    FIELD_QP_INT8_SCALE,
    FIELD_QP_PQ_SUBQUANTIZERS,
    FIELD_QP_PQ_CENTROIDS,
];

pub const QUANTIZATION_COLUMNS: &[&str] = &[
    FIELD_Q_BINARY,
    FIELD_Q_INT8,
    FIELD_Q_PQ4,
    FIELD_Q_PQ8,
    FIELD_Q_PQ16,
    FIELD_Q_PQ32,
    FIELD_QP_BINARY_THRESHOLD,
    FIELD_QP_INT8_MIN,
    FIELD_QP_INT8_MAX,
    FIELD_QP_INT8_SCALE,
    FIELD_QP_PQ_SUBQUANTIZERS,
    FIELD_QP_PQ_CENTROIDS,
];

pub const TEMPORAL_COLUMNS: &[&str] = &[FIELD_TIMESTAMP, FIELD_UPDATED_AT, FIELD_EXPIRES_AT];

pub const REQUIRED_COLUMNS: &[&str] = &[FIELD_ID, FIELD_VECTOR_FP32, FIELD_TIMESTAMP];
