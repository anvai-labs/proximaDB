//! # Internal Schema Registry - CORE
//!
//! The Internal Schema Registry is ProximaDB's core catalog providing:
//! - Multi-model object registration (Vector, Document, Graph, RDBMS, Observable)
//! - Schema enforcement (Strict, Flexible, Hybrid modes)
//! - PostgreSQL-compatible INFORMATION_SCHEMA views
//! - Constraint management (PK, FK, UNIQUE, CHECK, NOT NULL)
//!
//! This is always available and does not require feature flags.
//!
//! ## Supported Data Models
//!
//! - **Vector Collections**: Embeddings with dimension and distance metrics
//! - **Document Stores**: JSON/BSON with optional JSON Schema validation
//! - **Graph Databases**: Nodes, edges, and relationships
//! - **RDBMS Tables**: Traditional relational tables
//! - **Observability**: Logs, Metrics, Traces streams
//!
//! ## Key Features
//!
//! - **Unified Object Model**: Single abstraction (`CatalogObject`) for all types
//! - **Schema Enforcement Modes**:
//!   - `Strict`: All constraints validated at write time (RDBMS)
//!   - `Flexible`: Schema-on-read (Document)
//!   - `Hybrid`: Core schema enforced, extensions flexible
//! - **Cross-Model References**: Foreign keys spanning data models
//!   - Table-to-Table (traditional FK)
//!   - Table-to-Graph (reference graph nodes)
//!   - Table-to-Document (reference document IDs)
//!   - Table-to-Vector (reference vector collection IDs)
//! - **Schema Versioning**: Full history with evolution tracking
//! - **INFORMATION_SCHEMA**: PostgreSQL-compatible introspection views

pub mod enforcement;
pub mod information_schema;
pub mod registry;

use crate::{
    CatalogColumn, CatalogIndex, CatalogProjection, CatalogStorageLayout,
    CatalogStorageSpecialization, CatalogTableSchema, CatalogWorkloadProfile,
    RelationalCapabilities,
};
use anyhow::Result;
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-exports
pub use enforcement::{ConstraintEnforcer, ConstraintViolation, EnforcementResult};
pub use information_schema::{InformationSchema, InformationSchemaView};
pub use registry::InternalSchemaRegistry;

/// Object types in the multi-model database
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectType {
    /// Traditional RDBMS table
    RdbmsTable,
    /// Vector collection with embeddings
    VectorCollection,
    /// Document store (JSON/BSON)
    DocumentCollection,
    /// Graph database
    Graph,
    /// Graph node label/type
    GraphNodeLabel,
    /// Graph edge type
    GraphEdgeType,
    /// View (materialized or virtual)
    View,
    /// Materialized view
    MaterializedView,
    /// Observability: Log stream
    LogStream,
    /// Observability: Metric stream
    MetricStream,
    /// Observability: Trace stream
    TraceStream,
    /// Index (standalone)
    Index,
    /// Sequence/Auto-increment generator
    Sequence,
    /// Function/Procedure
    Function,
}

impl ObjectType {
    /// Get display name for the object type
    pub fn display_name(&self) -> &'static str {
        match self {
            ObjectType::RdbmsTable => "TABLE",
            ObjectType::VectorCollection => "VECTOR COLLECTION",
            ObjectType::DocumentCollection => "DOCUMENT COLLECTION",
            ObjectType::Graph => "GRAPH",
            ObjectType::GraphNodeLabel => "NODE LABEL",
            ObjectType::GraphEdgeType => "EDGE TYPE",
            ObjectType::View => "VIEW",
            ObjectType::MaterializedView => "MATERIALIZED VIEW",
            ObjectType::LogStream => "LOG STREAM",
            ObjectType::MetricStream => "METRIC STREAM",
            ObjectType::TraceStream => "TRACE STREAM",
            ObjectType::Index => "INDEX",
            ObjectType::Sequence => "SEQUENCE",
            ObjectType::Function => "FUNCTION",
        }
    }

    /// Check if this object type supports schema enforcement
    pub fn supports_schema_enforcement(&self) -> bool {
        matches!(
            self,
            ObjectType::RdbmsTable
                | ObjectType::VectorCollection
                | ObjectType::DocumentCollection
                | ObjectType::GraphNodeLabel
                | ObjectType::GraphEdgeType
        )
    }
}

impl std::fmt::Display for ObjectType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Schema enforcement mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SchemaEnforcementMode {
    /// Strict enforcement: All constraints validated at write time (RDBMS)
    /// - All columns must match schema
    /// - Type checking enforced
    /// - NOT NULL enforced
    /// - FK constraints enforced
    #[default]
    Strict,

    /// Flexible enforcement: Schema-on-read (Document)
    /// - Extra fields allowed
    /// - Missing fields allowed (treated as NULL)
    /// - Type coercion attempted
    Flexible,

    /// Hybrid enforcement: Core schema enforced, extensions flexible
    /// - Required fields enforced
    /// - Type checking on core fields
    /// - Extra fields allowed but typed
    Hybrid,
}

impl SchemaEnforcementMode {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            SchemaEnforcementMode::Strict => "STRICT",
            SchemaEnforcementMode::Flexible => "FLEXIBLE",
            SchemaEnforcementMode::Hybrid => "HYBRID",
        }
    }
}

impl std::fmt::Display for SchemaEnforcementMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Schema snapshot for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaSnapshot {
    /// Schema version
    pub version: i32,
    /// Snapshot timestamp (millis since epoch)
    pub timestamp_ms: i64,
    /// Schema at this version
    pub schema: ObjectSchema,
    /// Change description
    pub change_description: Option<String>,
    /// User/principal who made the change
    pub changed_by: Option<String>,
}

/// Object schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectSchema {
    /// Column definitions
    pub columns: Vec<CatalogColumn>,
    /// Primary key columns (by name)
    pub primary_key: Vec<String>,
    /// Table-level constraints
    pub constraints: Vec<TableConstraint>,
    /// Indexes
    pub indexes: Vec<CatalogIndex>,
    /// Cataloged storage layouts and authority modes.
    #[serde(default)]
    pub storage_layouts: Vec<CatalogStorageLayout>,
    /// Cataloged rebuildable projections/access methods.
    #[serde(default)]
    pub projections: Vec<CatalogProjection>,
    /// Optional relational integrity and transaction capabilities.
    #[serde(default)]
    pub relational_capabilities: RelationalCapabilities,
    /// Intended workload profile for routing/planning.
    #[serde(default)]
    pub workload_profile: CatalogWorkloadProfile,
    /// Primary physical/access-method specialization for routing/planning.
    #[serde(default)]
    pub storage_specialization: CatalogStorageSpecialization,
    /// Table-level catalog properties, including route knobs.
    #[serde(default)]
    pub properties: HashMap<String, String>,
    /// Model-specific properties
    pub model_properties: ModelProperties,
}

impl Default for ObjectSchema {
    fn default() -> Self {
        Self {
            columns: Vec::new(),
            primary_key: Vec::new(),
            constraints: Vec::new(),
            indexes: Vec::new(),
            storage_layouts: Vec::new(),
            projections: Vec::new(),
            relational_capabilities: RelationalCapabilities::default(),
            workload_profile: CatalogWorkloadProfile::default(),
            storage_specialization: CatalogStorageSpecialization::default(),
            properties: HashMap::new(),
            model_properties: ModelProperties::None,
        }
    }
}

impl ObjectSchema {
    /// Create from a CatalogTableSchema
    pub fn from_table_schema(schema: &CatalogTableSchema) -> Self {
        Self {
            columns: schema.columns.clone(),
            primary_key: schema.primary_key.clone(),
            constraints: Vec::new(),
            indexes: schema.indexes.clone(),
            storage_layouts: schema.storage_layouts.clone(),
            projections: schema.projections.clone(),
            relational_capabilities: schema.relational_capabilities.clone(),
            workload_profile: schema.workload_profile,
            storage_specialization: schema.storage_specialization,
            properties: schema.properties.clone(),
            model_properties: ModelProperties::Rdbms(RdbmsProperties::default()),
        }
    }

    /// Get column by name
    pub fn get_column(&self, name: &str) -> Option<&CatalogColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Check if column exists
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.iter().any(|c| c.name == name)
    }

    /// Convert to Arrow Schema
    pub fn to_arrow_schema(&self) -> Result<ArrowSchema> {
        let fields: Vec<ArrowField> = self
            .columns
            .iter()
            .map(|col| {
                ArrowField::new(
                    &col.name,
                    crate::catalog_arrow_type(&col.data_type),
                    col.nullable,
                )
            })
            .collect();
        Ok(ArrowSchema::new(fields))
    }
}

/// Table-level constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableConstraint {
    /// Constraint name
    pub name: String,
    /// Constraint type and details
    pub constraint_type: ConstraintType,
    /// Is the constraint deferred (checked at transaction commit)
    pub is_deferrable: bool,
    /// Is currently deferred
    pub is_deferred: bool,
}

impl TableConstraint {
    /// Create a new primary key constraint
    pub fn primary_key(name: impl Into<String>, columns: Vec<String>) -> Self {
        Self {
            name: name.into(),
            constraint_type: ConstraintType::PrimaryKey { columns },
            is_deferrable: false,
            is_deferred: false,
        }
    }

    /// Create a new foreign key constraint
    pub fn foreign_key(
        name: impl Into<String>,
        columns: Vec<String>,
        reference: ForeignKeyReference,
        on_delete: ReferentialAction,
        on_update: ReferentialAction,
    ) -> Self {
        Self {
            name: name.into(),
            constraint_type: ConstraintType::ForeignKey {
                columns,
                reference,
                on_delete,
                on_update,
            },
            is_deferrable: true,
            is_deferred: false,
        }
    }

    /// Create a unique constraint
    pub fn unique(name: impl Into<String>, columns: Vec<String>) -> Self {
        Self {
            name: name.into(),
            constraint_type: ConstraintType::Unique { columns },
            is_deferrable: false,
            is_deferred: false,
        }
    }

    /// Create a check constraint
    pub fn check(name: impl Into<String>, expression: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            constraint_type: ConstraintType::Check {
                expression: expression.into(),
            },
            is_deferrable: false,
            is_deferred: false,
        }
    }

    /// Create a not null constraint
    pub fn not_null(name: impl Into<String>, column: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            constraint_type: ConstraintType::NotNull {
                column: column.into(),
            },
            is_deferrable: false,
            is_deferred: false,
        }
    }
}

/// Constraint type details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Primary key constraint
    PrimaryKey {
        /// Column names that form the primary key
        columns: Vec<String>,
    },

    /// Foreign key constraint with cross-model reference support
    ForeignKey {
        /// Local column names that form the foreign key
        columns: Vec<String>,
        /// Target object reference (table, graph node, document, or vector collection)
        reference: ForeignKeyReference,
        /// Action to take when the referenced row is deleted
        on_delete: ReferentialAction,
        /// Action to take when the referenced row is updated
        on_update: ReferentialAction,
    },

    /// Unique constraint
    Unique {
        /// Column names included in the uniqueness constraint
        columns: Vec<String>,
    },

    /// Check constraint (SQL expression)
    Check {
        /// SQL boolean expression that rows must satisfy
        expression: String,
    },

    /// Not null constraint
    NotNull {
        /// Name of the column that must not be NULL
        column: String,
    },

    /// Exclusion constraint (PostgreSQL-style)
    Exclusion {
        /// Column names participating in the exclusion constraint
        columns: Vec<String>,
        /// Exclusion operator (e.g., `&&` for range overlap)
        operator: String,
    },
}

/// Foreign key reference (supports cross-model references)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForeignKeyReference {
    /// Reference to an RDBMS table
    Table {
        /// Optional catalog name of the referenced table
        catalog: Option<String>,
        /// Optional schema name of the referenced table
        schema: Option<String>,
        /// Name of the referenced table
        table: String,
        /// Column names in the referenced table
        columns: Vec<String>,
    },

    /// Reference to a graph node by label
    GraphNode {
        /// Identifier of the graph that owns the node
        graph_id: String,
        /// Node label to reference
        node_label: String,
        /// Node property to use as the key (default: node ID)
        property: Option<String>,
    },

    /// Reference to a document by ID path
    Document {
        /// Name of the document collection
        collection: String,
        /// JSONPath expression identifying the document key (default: `$._id`)
        id_path: String,
    },

    /// Reference to a vector collection
    Vector {
        /// Name of the vector collection
        collection: String,
        /// Name of the ID field in the vector collection (default: `id`)
        id_field: Option<String>,
    },
}

impl ForeignKeyReference {
    /// Create a table reference
    pub fn table(table: impl Into<String>, columns: Vec<String>) -> Self {
        ForeignKeyReference::Table {
            catalog: None,
            schema: None,
            table: table.into(),
            columns,
        }
    }

    /// Create a graph node reference
    pub fn graph_node(graph_id: impl Into<String>, node_label: impl Into<String>) -> Self {
        ForeignKeyReference::GraphNode {
            graph_id: graph_id.into(),
            node_label: node_label.into(),
            property: None,
        }
    }

    /// Create a document reference
    pub fn document(collection: impl Into<String>) -> Self {
        ForeignKeyReference::Document {
            collection: collection.into(),
            id_path: "$._id".to_string(),
        }
    }

    /// Create a vector collection reference
    pub fn vector(collection: impl Into<String>) -> Self {
        ForeignKeyReference::Vector {
            collection: collection.into(),
            id_field: None,
        }
    }
}

/// Referential action for foreign key constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReferentialAction {
    /// No action (error if referencing rows exist)
    #[default]
    NoAction,
    /// Restrict (prevent if referencing rows exist)
    Restrict,
    /// Cascade (propagate to referencing rows)
    Cascade,
    /// Set null (set referencing columns to NULL)
    SetNull,
    /// Set default (set referencing columns to default)
    SetDefault,
}

impl ReferentialAction {
    /// Convert to SQL string
    pub fn as_sql(&self) -> &'static str {
        match self {
            ReferentialAction::NoAction => "NO ACTION",
            ReferentialAction::Restrict => "RESTRICT",
            ReferentialAction::Cascade => "CASCADE",
            ReferentialAction::SetNull => "SET NULL",
            ReferentialAction::SetDefault => "SET DEFAULT",
        }
    }
}

/// Model-specific properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelProperties {
    /// No model-specific properties
    None,

    /// RDBMS table properties
    Rdbms(RdbmsProperties),

    /// Vector collection properties
    Vector(VectorProperties),

    /// Document collection properties
    Document(DocumentProperties),

    /// Graph properties
    Graph(GraphProperties),

    /// Observability properties
    Observability(ObservabilityProperties),
}

/// RDBMS-specific properties
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RdbmsProperties {
    /// Storage engine (SST + columnar engine for HTAP)
    pub storage_engine: Option<String>,
    /// Row format (COMPACT, DYNAMIC, COMPRESSED)
    pub row_format: Option<String>,
    /// Tablespace
    pub tablespace: Option<String>,
    /// Partition scheme
    pub partition_scheme: Option<String>,
    /// Clustering columns
    pub cluster_by: Vec<String>,
}

/// Vector collection properties
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorProperties {
    /// Vector dimension
    pub dimension: u32,
    /// Distance metric (cosine, euclidean, dot_product)
    pub distance_metric: String,
    /// Vector quantization (none, scalar, binary, product)
    pub quantization: Option<String>,
    /// Index type (hnsw, ivf, flat)
    pub index_type: Option<String>,
    /// HNSW M parameter
    pub hnsw_m: Option<u32>,
    /// HNSW ef_construction
    pub hnsw_ef_construction: Option<u32>,
}

/// Document collection properties
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentProperties {
    /// JSON Schema for validation
    pub json_schema: Option<String>,
    /// ID generation strategy (auto, uuid, nanoid, user)
    pub id_generation: String,
    /// Full-text search enabled
    pub enable_full_text: bool,
    /// Nested field indexing
    pub indexed_paths: Vec<String>,
}

/// Graph properties
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphProperties {
    /// Graph type (directed, undirected, mixed)
    pub graph_type: String,
    /// Allow self-loops
    pub allow_self_loops: bool,
    /// Allow multi-edges (multiple edges between same nodes)
    pub allow_multi_edges: bool,
    /// Node labels
    pub node_labels: Vec<String>,
    /// Edge types
    pub edge_types: Vec<String>,
}

/// Observability properties
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObservabilityProperties {
    /// Stream type (logs, metrics, traces)
    pub stream_type: String,
    /// Retention period (seconds)
    pub retention_seconds: u64,
    /// Rollup configuration
    pub rollup_intervals: Vec<String>,
    /// High-cardinality label limits
    pub cardinality_limits: HashMap<String, u64>,
}

/// Catalog object - unified representation of all database objects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogObject {
    /// Unique object identifier (UUID)
    pub object_id: String,

    /// Catalog name (e.g., "default", "production")
    pub catalog: String,

    /// Namespace hierarchy (e.g., ["database", "schema"])
    pub namespace: Vec<String>,

    /// Object name
    pub name: String,

    /// Object type
    pub object_type: ObjectType,

    /// Schema enforcement mode
    pub enforcement_mode: SchemaEnforcementMode,

    /// Current schema
    pub schema: ObjectSchema,

    /// Current schema version
    pub schema_version: i32,

    /// Schema history (for evolution tracking)
    pub schema_history: Vec<SchemaSnapshot>,

    /// Owner principal
    pub owner: Option<String>,

    /// Object properties
    pub properties: HashMap<String, String>,

    /// Creation timestamp (millis since epoch)
    pub created_at_ms: i64,

    /// Last update timestamp (millis since epoch)
    pub updated_at_ms: i64,

    /// Comment/description
    pub comment: Option<String>,
}

impl CatalogObject {
    /// Create a new catalog object
    pub fn new(
        catalog: impl Into<String>,
        namespace: Vec<String>,
        name: impl Into<String>,
        object_type: ObjectType,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        Self {
            object_id: uuid::Uuid::new_v4().to_string(),
            catalog: catalog.into(),
            namespace,
            name: name.into(),
            object_type,
            enforcement_mode: SchemaEnforcementMode::default(),
            schema: ObjectSchema::default(),
            schema_version: 1,
            schema_history: Vec::new(),
            owner: None,
            properties: HashMap::new(),
            created_at_ms: now,
            updated_at_ms: now,
            comment: None,
        }
    }

    /// Get fully qualified name
    pub fn fqn(&self) -> String {
        let mut parts = vec![self.catalog.clone()];
        parts.extend(self.namespace.clone());
        parts.push(self.name.clone());
        parts.join(".")
    }

    /// Set schema with enforcement mode
    pub fn with_schema(mut self, schema: ObjectSchema, mode: SchemaEnforcementMode) -> Self {
        self.schema = schema;
        self.enforcement_mode = mode;
        self
    }

    /// Set owner
    pub fn with_owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    /// Set comment
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Add a property
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Update schema and create history snapshot
    pub fn update_schema(
        &mut self,
        new_schema: ObjectSchema,
        change_description: Option<String>,
        changed_by: Option<String>,
    ) {
        // Create snapshot of current schema
        let snapshot = SchemaSnapshot {
            version: self.schema_version,
            timestamp_ms: self.updated_at_ms,
            schema: self.schema.clone(),
            change_description,
            changed_by,
        };
        self.schema_history.push(snapshot);

        // Update to new schema
        self.schema = new_schema;
        self.schema_version += 1;
        self.updated_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
    }

    /// Get schema at specific version
    pub fn get_schema_at_version(&self, version: i32) -> Option<&ObjectSchema> {
        if version == self.schema_version {
            Some(&self.schema)
        } else {
            self.schema_history
                .iter()
                .find(|s| s.version == version)
                .map(|s| &s.schema)
        }
    }

    /// Check if object is RDBMS table
    pub fn is_rdbms_table(&self) -> bool {
        matches!(self.object_type, ObjectType::RdbmsTable)
    }

    /// Check if object is vector collection
    pub fn is_vector_collection(&self) -> bool {
        matches!(self.object_type, ObjectType::VectorCollection)
    }

    /// Check if object is graph
    pub fn is_graph(&self) -> bool {
        matches!(self.object_type, ObjectType::Graph)
    }

    /// Check if object is document collection
    pub fn is_document_collection(&self) -> bool {
        matches!(self.object_type, ObjectType::DocumentCollection)
    }

    /// Check if object is an observability stream
    pub fn is_observability(&self) -> bool {
        matches!(
            self.object_type,
            ObjectType::LogStream | ObjectType::MetricStream | ObjectType::TraceStream
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_data_model::ProximaType;

    #[test]
    fn test_object_type_display() {
        assert_eq!(ObjectType::RdbmsTable.to_string(), "TABLE");
        assert_eq!(
            ObjectType::VectorCollection.to_string(),
            "VECTOR COLLECTION"
        );
        assert_eq!(ObjectType::Graph.to_string(), "GRAPH");
    }

    #[test]
    fn test_schema_enforcement_mode() {
        assert_eq!(SchemaEnforcementMode::Strict.as_str(), "STRICT");
        assert_eq!(SchemaEnforcementMode::Flexible.as_str(), "FLEXIBLE");
        assert_eq!(SchemaEnforcementMode::Hybrid.as_str(), "HYBRID");
    }

    #[test]
    fn test_catalog_object_creation() {
        let obj = CatalogObject::new(
            "default",
            vec!["public".to_string()],
            "users",
            ObjectType::RdbmsTable,
        );

        assert_eq!(obj.fqn(), "default.public.users");
        assert_eq!(obj.object_type, ObjectType::RdbmsTable);
        assert_eq!(obj.schema_version, 1);
        assert!(obj.schema_history.is_empty());
    }

    #[test]
    fn test_catalog_object_with_schema() {
        let schema = ObjectSchema {
            columns: vec![CatalogColumn::new(1, "id", ProximaType::Int64)],
            primary_key: vec!["id".to_string()],
            ..Default::default()
        };

        let obj = CatalogObject::new(
            "default",
            vec!["public".to_string()],
            "users",
            ObjectType::RdbmsTable,
        )
        .with_schema(schema, SchemaEnforcementMode::Strict);

        assert_eq!(obj.enforcement_mode, SchemaEnforcementMode::Strict);
        assert_eq!(obj.schema.columns.len(), 1);
    }

    #[test]
    fn test_schema_update_with_history() {
        let schema1 = ObjectSchema {
            columns: vec![CatalogColumn::new(1, "id", ProximaType::Int64)],
            ..Default::default()
        };

        let mut obj = CatalogObject::new(
            "default",
            vec!["public".to_string()],
            "users",
            ObjectType::RdbmsTable,
        )
        .with_schema(schema1, SchemaEnforcementMode::Strict);

        let schema2 = ObjectSchema {
            columns: vec![
                CatalogColumn::new(1, "id", ProximaType::Int64),
                CatalogColumn::new(2, "name", ProximaType::String),
            ],
            ..Default::default()
        };

        obj.update_schema(
            schema2,
            Some("Added name column".to_string()),
            Some("admin".to_string()),
        );

        assert_eq!(obj.schema_version, 2);
        assert_eq!(obj.schema.columns.len(), 2);
        assert_eq!(obj.schema_history.len(), 1);
        assert_eq!(obj.schema_history[0].version, 1);
    }

    #[test]
    fn test_foreign_key_reference_creation() {
        let table_ref = ForeignKeyReference::table("orders", vec!["user_id".to_string()]);
        assert!(matches!(table_ref, ForeignKeyReference::Table { .. }));

        let graph_ref = ForeignKeyReference::graph_node("social", "User");
        assert!(matches!(graph_ref, ForeignKeyReference::GraphNode { .. }));

        let doc_ref = ForeignKeyReference::document("products");
        assert!(matches!(doc_ref, ForeignKeyReference::Document { .. }));

        let vec_ref = ForeignKeyReference::vector("embeddings");
        assert!(matches!(vec_ref, ForeignKeyReference::Vector { .. }));
    }

    #[test]
    fn test_table_constraint_creation() {
        let pk = TableConstraint::primary_key("pk_users", vec!["id".to_string()]);
        assert!(matches!(
            pk.constraint_type,
            ConstraintType::PrimaryKey { .. }
        ));

        let unique = TableConstraint::unique("uq_email", vec!["email".to_string()]);
        assert!(matches!(
            unique.constraint_type,
            ConstraintType::Unique { .. }
        ));

        let check = TableConstraint::check("ck_age", "age >= 0 AND age <= 150");
        assert!(matches!(
            check.constraint_type,
            ConstraintType::Check { .. }
        ));
    }

    #[test]
    fn test_referential_action() {
        assert_eq!(ReferentialAction::Cascade.as_sql(), "CASCADE");
        assert_eq!(ReferentialAction::SetNull.as_sql(), "SET NULL");
        assert_eq!(ReferentialAction::NoAction.as_sql(), "NO ACTION");
    }
}
