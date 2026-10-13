//! # DDL/DML Execution Module
//!
//! Provides unified execution of Data Definition Language (DDL) and
//! Data Manipulation Language (DML) statements across all SQL interfaces
//! (SQL Frontend, PostgreSQL Wire Protocol, embedded mode).
//!
//! ## Architecture
//!
//! Uses a protocol-based (trait-based) design for reuse:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    DDL/DML Executor                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │ TableOps    │  │ IndexOps    │  │ TransactionOps      │  │
//! │  │ (Trait)     │  │ (Trait)     │  │ (Trait)             │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//!                           ↓
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Schema Manager  │  Document Service  │  Collection Service │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Supported DDL Statements
//!
//! - CREATE TABLE
//! - DROP TABLE
//! - ALTER TABLE (ADD/DROP COLUMN, ADD/DROP CONSTRAINT)
//! - CREATE INDEX
//! - DROP INDEX
//! - CREATE COLLECTION (vector)
//! - DROP COLLECTION
//!
//! ## Supported DML Statements
//!
//! - INSERT INTO
//! - UPDATE
//! - DELETE FROM

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use crate::schema::relational::{
    ColumnDefinition, IndexDefinition, IndexType, RelationalSchema, SqlDataType, TableDefinition,
};

// ============================================================================
// Protocol/Trait Definitions (for reuse across components)
// ============================================================================

/// Table operations protocol - implemented by storage backends
#[async_trait]
pub trait TableOperations: Send + Sync {
    /// Create a new table
    async fn create_table(&self, table: &TableDefinition) -> Result<()>;

    /// Drop a table
    async fn drop_table(&self, table_name: &str, if_exists: bool) -> Result<()>;

    /// Alter table structure
    async fn alter_table(&self, table_name: &str, alterations: Vec<TableAlteration>) -> Result<()>;

    /// Check if table exists
    async fn table_exists(&self, table_name: &str) -> Result<bool>;

    /// Get table schema
    async fn get_table(&self, table_name: &str) -> Result<Option<TableDefinition>>;

    /// List all tables
    async fn list_tables(&self) -> Result<Vec<String>>;
}

/// Index operations protocol
#[async_trait]
pub trait IndexOperations: Send + Sync {
    /// Create an index
    async fn create_index(&self, index: &IndexDefinition) -> Result<()>;

    /// Drop an index
    async fn drop_index(&self, index_name: &str, if_exists: bool) -> Result<()>;

    /// Check if index exists
    async fn index_exists(&self, index_name: &str) -> Result<bool>;

    /// List indexes for a table
    async fn list_indexes(&self, table_name: &str) -> Result<Vec<IndexDefinition>>;
}

/// DML operations protocol
#[async_trait]
pub trait DmlOperations: Send + Sync {
    /// Insert rows into a table
    async fn insert(
        &self,
        table: &str,
        columns: &[String],
        rows: Vec<Vec<SqlValue>>,
    ) -> Result<u64>;

    /// Update rows in a table
    async fn update(
        &self,
        table: &str,
        updates: Vec<(String, SqlValue)>,
        where_clause: Option<String>,
    ) -> Result<u64>;

    /// Delete rows from a table
    async fn delete(&self, table: &str, where_clause: Option<String>) -> Result<u64>;
}

// ============================================================================
// Data Types
// ============================================================================

/// SQL value for DML operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SqlValue {
    Null,
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
    Bytes(Vec<u8>),
    Vector(Vec<f32>),
    Json(serde_json::Value),
    Timestamp(chrono::DateTime<chrono::Utc>),
}

/// Table alteration types
#[derive(Debug, Clone)]
pub enum TableAlteration {
    AddColumn(ColumnDefinition),
    DropColumn(String),
    RenameColumn {
        old_name: String,
        new_name: String,
    },
    AlterColumnType {
        column: String,
        new_type: SqlDataType,
    },
    AddConstraint(ConstraintDefinition),
    DropConstraint(String),
}

/// Constraint definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDefinition {
    pub name: String,
    pub constraint_type: ConstraintType,
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    PrimaryKey(Vec<String>),
    Unique(Vec<String>),
    ForeignKey {
        columns: Vec<String>,
        references_table: String,
        references_columns: Vec<String>,
    },
    Check(String),
    NotNull(String),
}

// ============================================================================
// DDL/DML Execution Result
// ============================================================================

/// Backwards-compat alias for [`DdlDmlExecutionResult`].
pub type ExecutionResult = DdlDmlExecutionResult;

/// Result of DDL/DML execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DdlDmlExecutionResult {
    /// Number of rows affected (for DML)
    pub rows_affected: u64,
    /// Status message
    pub message: String,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Any warnings generated
    pub warnings: Vec<String>,
}

impl DdlDmlExecutionResult {
    pub fn ok(message: impl Into<String>) -> Self {
        Self {
            rows_affected: 0,
            message: message.into(),
            execution_time_ms: 0,
            warnings: Vec::new(),
        }
    }

    pub fn rows(affected: u64, message: impl Into<String>) -> Self {
        Self {
            rows_affected: affected,
            message: message.into(),
            execution_time_ms: 0,
            warnings: Vec::new(),
        }
    }
}

// ============================================================================
// DDL/DML Executor
// ============================================================================

/// Unified DDL/DML executor
pub struct DdlDmlExecutor {
    /// Table operations backend
    table_ops: Arc<dyn TableOperations>,
    /// Index operations backend
    index_ops: Arc<dyn IndexOperations>,
    /// DML operations backend
    dml_ops: Arc<dyn DmlOperations>,
    /// Schema registry for validation
    #[allow(dead_code)]
    schemas: Arc<tokio::sync::RwLock<HashMap<String, RelationalSchema>>>,
}

impl DdlDmlExecutor {
    /// Create a new executor with the specified backends
    pub fn new(
        table_ops: Arc<dyn TableOperations>,
        index_ops: Arc<dyn IndexOperations>,
        dml_ops: Arc<dyn DmlOperations>,
    ) -> Self {
        Self {
            table_ops,
            index_ops,
            dml_ops,
            schemas: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Execute a DDL statement
    pub async fn execute_ddl(&self, statement: DdlStatement) -> Result<DdlDmlExecutionResult> {
        let start = std::time::Instant::now();

        let result = match statement {
            DdlStatement::CreateTable {
                table,
                if_not_exists,
            } => {
                if if_not_exists && self.table_ops.table_exists(&table.name).await? {
                    DdlDmlExecutionResult::ok(format!("Table '{}' already exists", table.name))
                } else {
                    self.table_ops.create_table(&table).await?;
                    info!("Created table: {}", table.name);
                    DdlDmlExecutionResult::ok(format!("CREATE TABLE {}", table.name))
                }
            }

            DdlStatement::DropTable { name, if_exists } => {
                self.table_ops.drop_table(&name, if_exists).await?;
                info!("Dropped table: {}", name);
                DdlDmlExecutionResult::ok(format!("DROP TABLE {}", name))
            }

            DdlStatement::AlterTable { name, alterations } => {
                self.table_ops
                    .alter_table(&name, alterations.clone())
                    .await?;
                info!("Altered table: {} ({} changes)", name, alterations.len());
                DdlDmlExecutionResult::ok(format!("ALTER TABLE {}", name))
            }

            DdlStatement::CreateIndex {
                index,
                if_not_exists,
            } => {
                if if_not_exists && self.index_ops.index_exists(&index.name).await? {
                    DdlDmlExecutionResult::ok(format!("Index '{}' already exists", index.name))
                } else {
                    self.index_ops.create_index(&index).await?;
                    info!("Created index: {} on {}", index.name, index.table);
                    DdlDmlExecutionResult::ok(format!("CREATE INDEX {}", index.name))
                }
            }

            DdlStatement::DropIndex { name, if_exists } => {
                self.index_ops.drop_index(&name, if_exists).await?;
                info!("Dropped index: {}", name);
                DdlDmlExecutionResult::ok(format!("DROP INDEX {}", name))
            }
        };

        let mut result = result;
        result.execution_time_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }

    /// Execute a DML statement
    pub async fn execute_dml(&self, statement: DmlStatement) -> Result<DdlDmlExecutionResult> {
        let start = std::time::Instant::now();

        let result = match statement {
            DmlStatement::Insert {
                table,
                columns,
                values,
            } => {
                let rows = self.dml_ops.insert(&table, &columns, values).await?;
                debug!("Inserted {} rows into {}", rows, table);
                DdlDmlExecutionResult::rows(rows, format!("INSERT {} rows", rows))
            }

            DmlStatement::Update {
                table,
                updates,
                where_clause,
            } => {
                let rows = self.dml_ops.update(&table, updates, where_clause).await?;
                debug!("Updated {} rows in {}", rows, table);
                DdlDmlExecutionResult::rows(rows, format!("UPDATE {} rows", rows))
            }

            DmlStatement::Delete {
                table,
                where_clause,
            } => {
                let rows = self.dml_ops.delete(&table, where_clause).await?;
                debug!("Deleted {} rows from {}", rows, table);
                DdlDmlExecutionResult::rows(rows, format!("DELETE {} rows", rows))
            }
        };

        let mut result = result;
        result.execution_time_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }
}

/// DDL statement types
#[derive(Debug, Clone)]
pub enum DdlStatement {
    CreateTable {
        table: TableDefinition,
        if_not_exists: bool,
    },
    DropTable {
        name: String,
        if_exists: bool,
    },
    AlterTable {
        name: String,
        alterations: Vec<TableAlteration>,
    },
    CreateIndex {
        index: IndexDefinition,
        if_not_exists: bool,
    },
    DropIndex {
        name: String,
        if_exists: bool,
    },
}

/// DML statement types
#[derive(Debug, Clone)]
pub enum DmlStatement {
    Insert {
        table: String,
        columns: Vec<String>,
        values: Vec<Vec<SqlValue>>,
    },
    Update {
        table: String,
        updates: Vec<(String, SqlValue)>,
        where_clause: Option<String>,
    },
    Delete {
        table: String,
        where_clause: Option<String>,
    },
}

// ============================================================================
// SQL Parser Integration
// ============================================================================

/// Parse DDL/DML from SQL string
pub fn parse_ddl(sql: &str) -> Result<DdlStatement> {
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    let normalized = sql.trim().trim_end_matches(';').trim();
    if normalized.eq_ignore_ascii_case("DROP TABLE") {
        return Err(anyhow!("DROP TABLE requires a table name"));
    }

    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)?;

    if statements.is_empty() {
        return Err(anyhow!("No SQL statements found"));
    }

    let stmt = &statements[0];

    match stmt {
        sqlparser::ast::Statement::CreateTable(create) => {
            if create.query.is_some() || create.like.is_some() || create.clone.is_some() {
                return Err(anyhow!(
                    "CREATE TABLE with query/LIKE/CLONE clauses is not supported"
                ));
            }

            let mut columns = HashMap::new();
            let mut primary_key = Vec::new();

            for col_def in &create.columns {
                let col_name = col_def.name.value.clone();
                let data_type = convert_sql_type(&col_def.data_type);
                let nullable = !col_def
                    .options
                    .iter()
                    .any(|opt| matches!(opt.option, sqlparser::ast::ColumnOption::NotNull));

                // Check for primary key in column options
                // sqlparser 0.62 removed `ColumnOption::Unique.is_primary` and
                // gave PRIMARY KEY its OWN variant. Matching `Unique(..)` here
                // instead would compile and silently treat every column-level
                // PRIMARY KEY as a mere UNIQUE constraint.
                if col_def
                    .options
                    .iter()
                    .any(|opt| matches!(opt.option, sqlparser::ast::ColumnOption::PrimaryKey(_)))
                {
                    primary_key.push(col_name.clone());
                }

                columns.insert(
                    col_name.clone(),
                    ColumnDefinition {
                        name: col_name,
                        data_type,
                        nullable,
                        default_value: None,
                        constraints: Vec::new(),
                        description: None,
                    },
                );
            }

            // Extract primary key from table constraints
            for constraint in &create.constraints {
                // 0.62 wraps the fields in a named struct; `columns` is
                // `Vec<IndexColumn>` in BOTH 0.59 and 0.62, so only the pattern
                // shape changes here.
                if let sqlparser::ast::TableConstraint::PrimaryKey(pk) = constraint {
                    for col in &pk.columns {
                        // NOTE (pre-existing, not introduced here): `col.column`
                        // is an `OrderByExpr`, so `PRIMARY KEY (c DESC)` yields
                        // the column name "c DESC". Tracked in TD-SQLPARSE-1.
                        primary_key.push(col.column.to_string());
                    }
                }
            }

            let table = TableDefinition {
                name: create.name.to_string(),
                columns,
                primary_key,
                unique_constraints: Vec::new(),
                check_constraints: Vec::new(),
                indexes: Vec::new(),
                description: None,
            };

            Ok(DdlStatement::CreateTable {
                table,
                if_not_exists: create.if_not_exists,
            })
        }

        sqlparser::ast::Statement::Drop {
            object_type,
            names,
            if_exists,
            table,
            ..
        } => match object_type {
            sqlparser::ast::ObjectType::Table => Ok(DdlStatement::DropTable {
                name: names
                    .first()
                    .ok_or_else(|| anyhow!("DROP TABLE requires a table name"))?
                    .to_string(),
                if_exists: *if_exists,
            }),
            sqlparser::ast::ObjectType::Index => {
                if table.is_none() {
                    return Err(anyhow!("DROP INDEX requires a table name"));
                }
                Ok(DdlStatement::DropIndex {
                    name: names
                        .first()
                        .ok_or_else(|| anyhow!("DROP INDEX requires an index name"))?
                        .to_string(),
                    if_exists: *if_exists,
                })
            }
            _ => Err(anyhow!("Unsupported DROP object type: {:?}", object_type)),
        },

        sqlparser::ast::Statement::CreateIndex(create_index) => {
            use sqlparser::ast::{IndexOption, IndexType as SqlIndexType};

            if let Some(index_type) = create_index.using.as_ref().cloned().or_else(|| {
                create_index.index_options.iter().find_map(|opt| match opt {
                    IndexOption::Using(index_type) => Some(index_type.clone()),
                    _ => None,
                })
            }) {
                match index_type {
                    SqlIndexType::BTree => {}
                    other => return Err(anyhow!("Unsupported CREATE INDEX USING {}", other)),
                }
            }

            // CreateIndex.name (not index_name) is the index name
            let index_name = create_index
                .name
                .as_ref()
                .map_or_else(|| "unnamed_index".to_string(), |n| n.to_string());

            // table_name is the table the index is on
            let table_name = create_index.table_name.to_string();

            // IndexColumn has a column field (OrderByExpr)
            let columns: Vec<String> = create_index
                .columns
                .iter()
                .map(|c| c.column.to_string())
                .collect();

            let index = IndexDefinition {
                name: index_name,
                table: table_name,
                columns,
                index_type: IndexType::BTree,
                unique: create_index.unique,
                where_clause: None,
            };

            Ok(DdlStatement::CreateIndex {
                index,
                if_not_exists: create_index.if_not_exists,
            })
        }

        _ => Err(anyhow!("Unsupported DDL statement: {:?}", stmt)),
    }
}

/// Parse DML from SQL string
pub fn parse_dml(sql: &str) -> Result<DmlStatement> {
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)?;

    if statements.is_empty() {
        return Err(anyhow!("No SQL statements found"));
    }

    let stmt = &statements[0];

    match stmt {
        sqlparser::ast::Statement::Insert(insert) => {
            // Insert.table is a TableObject in 0.59.0
            let table = insert.table.to_string();

            // 0.62 changed `Insert.columns` from `Vec<Ident>` to
            // `Vec<ObjectName>`. Going through the parts keeps the RAW
            // identifier; `ObjectName::to_string()` would re-apply the
            // dialect's quoting, turning `col` into `"col"`. A part can also be
            // a function call, which has no identifier -- those are skipped
            // rather than stringified into a bogus column name.
            let columns: Vec<String> = insert
                .columns
                .iter()
                .filter_map(|c| {
                    c.0.last()
                        .and_then(|part| part.as_ident())
                        .map(|ident| ident.value.clone())
                })
                .collect();

            // Extract values from the source
            let values = extract_insert_values(&insert.source)?;

            Ok(DmlStatement::Insert {
                table,
                columns,
                values,
            })
        }

        sqlparser::ast::Statement::Update(update) => {
            let (table, assignments, selection) =
                (&update.table, &update.assignments, &update.selection);
            // TableWithJoins.relation is a TableFactor
            let table_name = extract_table_name(&table.relation)?;

            let updates: Vec<(String, SqlValue)> = assignments
                .iter()
                .filter_map(|a| {
                    // AssignmentTarget is an enum, convert to string
                    let col = a.target.to_string();
                    extract_value(&a.value).ok().map(|v| (col, v))
                })
                .collect();

            let where_clause = selection.as_ref().map(|s| s.to_string());

            Ok(DmlStatement::Update {
                table: table_name,
                updates,
                where_clause,
            })
        }

        sqlparser::ast::Statement::Delete(delete) => {
            // FromTable is an enum that contains a TableWithJoins
            let table = extract_delete_table(&delete.from)?;
            let where_clause = delete.selection.as_ref().map(|s| s.to_string());

            Ok(DmlStatement::Delete {
                table,
                where_clause,
            })
        }

        _ => Err(anyhow!("Unsupported DML statement: {:?}", stmt)),
    }
}

/// Extract table name from TableFactor
fn extract_table_name(factor: &sqlparser::ast::TableFactor) -> Result<String> {
    match factor {
        sqlparser::ast::TableFactor::Table { name, .. } => Ok(name.to_string()),
        _ => Err(anyhow!("Expected simple table reference")),
    }
}

/// Extract table name from DELETE's FromTable
fn extract_delete_table(from: &sqlparser::ast::FromTable) -> Result<String> {
    match from {
        sqlparser::ast::FromTable::WithFromKeyword(tables) => {
            let first = tables
                .first()
                .ok_or_else(|| anyhow!("DELETE requires FROM clause"))?;
            extract_table_name(&first.relation)
        }
        sqlparser::ast::FromTable::WithoutKeyword(tables) => {
            let first = tables
                .first()
                .ok_or_else(|| anyhow!("DELETE requires FROM clause"))?;
            extract_table_name(&first.relation)
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn convert_sql_type(data_type: &sqlparser::ast::DataType) -> SqlDataType {
    match data_type {
        sqlparser::ast::DataType::Int(_) | sqlparser::ast::DataType::Integer(_) => {
            SqlDataType::Integer
        }
        sqlparser::ast::DataType::BigInt(_) => SqlDataType::BigInt,
        sqlparser::ast::DataType::Real | sqlparser::ast::DataType::Float(_) => SqlDataType::Real,
        sqlparser::ast::DataType::Double(_) | sqlparser::ast::DataType::DoublePrecision => {
            SqlDataType::Double
        }
        sqlparser::ast::DataType::Decimal(info) | sqlparser::ast::DataType::Numeric(info) => {
            let (p, s) = match info {
                sqlparser::ast::ExactNumberInfo::PrecisionAndScale(p, s) => (*p as u32, *s as u32),
                sqlparser::ast::ExactNumberInfo::Precision(p) => (*p as u32, 0),
                sqlparser::ast::ExactNumberInfo::None => (18, 0),
            };
            SqlDataType::Decimal(p, s)
        }
        sqlparser::ast::DataType::Varchar(len) => {
            let max_len = len
                .as_ref()
                .and_then(|l| {
                    if let sqlparser::ast::CharacterLength::IntegerLength { length, .. } = l {
                        Some(*length as u32)
                    } else {
                        None
                    }
                })
                .unwrap_or(255);
            SqlDataType::VarChar(max_len)
        }
        sqlparser::ast::DataType::Char(len) => {
            let max_len = len
                .as_ref()
                .and_then(|l| {
                    if let sqlparser::ast::CharacterLength::IntegerLength { length, .. } = l {
                        Some(*length as u32)
                    } else {
                        None
                    }
                })
                .unwrap_or(1);
            SqlDataType::Char(max_len)
        }
        sqlparser::ast::DataType::Text => SqlDataType::Text,
        sqlparser::ast::DataType::Boolean => SqlDataType::Boolean,
        sqlparser::ast::DataType::Date => SqlDataType::Date,
        sqlparser::ast::DataType::Time(..) => SqlDataType::Time,
        sqlparser::ast::DataType::Timestamp(..) => SqlDataType::Timestamp,
        sqlparser::ast::DataType::Uuid => SqlDataType::Uuid,
        sqlparser::ast::DataType::Bytea => SqlDataType::Bytea,
        sqlparser::ast::DataType::JSON => SqlDataType::Json,
        sqlparser::ast::DataType::JSONB => SqlDataType::Jsonb,
        sqlparser::ast::DataType::Custom(name, tokens) => {
            // Handle vector type - use to_string() for ObjectName
            let type_name = name.to_string().to_uppercase();
            if type_name == "VECTOR" {
                let dim = tokens
                    .first()
                    .and_then(|t| t.to_string().parse::<u32>().ok())
                    .unwrap_or(1536);
                SqlDataType::Vector(dim)
            } else {
                SqlDataType::Text // Fallback
            }
        }
        _ => SqlDataType::Text, // Fallback for unsupported types
    }
}

fn extract_insert_values(
    source: &Option<Box<sqlparser::ast::Query>>,
) -> Result<Vec<Vec<SqlValue>>> {
    let query = source
        .as_ref()
        .ok_or_else(|| anyhow!("INSERT requires VALUES"))?;

    let mut result = Vec::new();

    if let sqlparser::ast::SetExpr::Values(values) = query.body.as_ref() {
        for row in &values.rows {
            let mut row_values = Vec::new();
            // 0.62 wraps each VALUES row in `Parens { opening_token, content,
            // closing_token }` -- a NAMED-field struct, so the payload is
            // `.content`, not `.0`.
            for expr in &row.content {
                row_values.push(extract_value(expr)?);
            }
            result.push(row_values);
        }
    }

    Ok(result)
}

fn extract_value(expr: &sqlparser::ast::Expr) -> Result<SqlValue> {
    match expr {
        // In sqlparser 0.59.0, Value is wrapped in ValueWithSpan
        sqlparser::ast::Expr::Value(value_with_span) => match &value_with_span.value {
            sqlparser::ast::Value::Null => Ok(SqlValue::Null),
            sqlparser::ast::Value::Number(n, _) => {
                if n.contains('.') {
                    Ok(SqlValue::Float(n.parse()?))
                } else {
                    Ok(SqlValue::Integer(n.parse()?))
                }
            }
            sqlparser::ast::Value::SingleQuotedString(s)
            | sqlparser::ast::Value::DoubleQuotedString(s) => Ok(SqlValue::Text(s.clone())),
            sqlparser::ast::Value::Boolean(b) => Ok(SqlValue::Boolean(*b)),
            _ => Ok(SqlValue::Null),
        },
        sqlparser::ast::Expr::UnaryOp { op, expr } => {
            if matches!(op, sqlparser::ast::UnaryOperator::Minus) {
                let inner = extract_value(expr)?;
                match inner {
                    SqlValue::Integer(i) => Ok(SqlValue::Integer(-i)),
                    SqlValue::Float(f) => Ok(SqlValue::Float(-f)),
                    _ => Err(anyhow!("Cannot negate non-numeric value")),
                }
            } else {
                extract_value(expr)
            }
        }
        _ => Ok(SqlValue::Null),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn test_parse_create_table() {
        let sql =
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL, email TEXT)";
        let result = parse_ddl(sql).unwrap();

        if let DdlStatement::CreateTable { table, .. } = result {
            assert_eq!(table.name, "users");
            assert_eq!(table.columns.len(), 3);
            assert!(table.columns.contains_key("id"));
            assert!(table.columns.contains_key("name"));
            assert!(table.columns.contains_key("email"));
        } else {
            panic!("Expected CreateTable");
        }
    }

    #[test]
    fn test_parse_create_table_supports_jsonb() {
        let sql = "CREATE TABLE users (id INTEGER, payload JSONB)";
        let result = parse_ddl(sql).unwrap();

        if let DdlStatement::CreateTable { table, .. } = result {
            assert_eq!(table.name, "users");
            let payload = table
                .columns
                .get("payload")
                .expect("expected payload column");
            assert!(matches!(payload.data_type, SqlDataType::Jsonb));
        } else {
            panic!("Expected CreateTable");
        }
    }

    #[test]
    fn test_parse_agentic_relational_jsonb_vector_table() {
        let sql = "CREATE TABLE IF NOT EXISTS agent_store (
            \"record_id\" TEXT NOT NULL,
            \"tenant_id\" TEXT NOT NULL,
            \"thread_id\" TEXT NOT NULL,
            \"namespace\" TEXT NOT NULL,
            \"key\" TEXT NOT NULL,
            \"payload\" JSONB NOT NULL DEFAULT '{}'::jsonb,
            \"metadata\" JSONB NOT NULL DEFAULT '{}'::jsonb,
            \"embedding\" VECTOR(384),
            PRIMARY KEY (\"record_id\")
        )";
        let result = parse_ddl(sql).unwrap();

        if let DdlStatement::CreateTable {
            table,
            if_not_exists,
        } = result
        {
            assert!(if_not_exists);
            assert_eq!(table.name, "agent_store");
            assert!(matches!(
                table.columns.get("payload").unwrap().data_type,
                SqlDataType::Jsonb
            ));
            assert!(matches!(
                table.columns.get("metadata").unwrap().data_type,
                SqlDataType::Jsonb
            ));
            assert!(matches!(
                table.columns.get("embedding").unwrap().data_type,
                SqlDataType::Vector(384)
            ));
            assert_eq!(table.primary_key, vec!["\"record_id\""]);
        } else {
            panic!("Expected CreateTable");
        }
    }

    #[tokio::test]
    async fn test_execute_agentic_relational_jsonb_vector_table() {
        #[derive(Default)]
        struct MockTableOps {
            created: Mutex<Vec<TableDefinition>>,
        }

        #[async_trait]
        impl TableOperations for MockTableOps {
            async fn create_table(&self, table: &TableDefinition) -> Result<()> {
                self.created.lock().unwrap().push(table.clone());
                Ok(())
            }

            async fn drop_table(&self, _table_name: &str, _if_exists: bool) -> Result<()> {
                Ok(())
            }

            async fn alter_table(
                &self,
                _table_name: &str,
                _alterations: Vec<TableAlteration>,
            ) -> Result<()> {
                Ok(())
            }

            async fn table_exists(&self, _table_name: &str) -> Result<bool> {
                Ok(false)
            }

            async fn get_table(&self, _table_name: &str) -> Result<Option<TableDefinition>> {
                Ok(None)
            }

            async fn list_tables(&self) -> Result<Vec<String>> {
                Ok(vec![])
            }
        }

        struct MockIndexOps;

        #[async_trait]
        impl IndexOperations for MockIndexOps {
            async fn create_index(&self, _index: &IndexDefinition) -> Result<()> {
                Ok(())
            }

            async fn drop_index(&self, _index_name: &str, _if_exists: bool) -> Result<()> {
                Ok(())
            }

            async fn index_exists(&self, _index_name: &str) -> Result<bool> {
                Ok(false)
            }

            async fn list_indexes(&self, _table_name: &str) -> Result<Vec<IndexDefinition>> {
                Ok(vec![])
            }
        }

        struct MockDmlOps;

        #[async_trait]
        impl DmlOperations for MockDmlOps {
            async fn insert(
                &self,
                _table: &str,
                _columns: &[String],
                _rows: Vec<Vec<SqlValue>>,
            ) -> Result<u64> {
                Ok(0)
            }

            async fn update(
                &self,
                _table: &str,
                _updates: Vec<(String, SqlValue)>,
                _where_clause: Option<String>,
            ) -> Result<u64> {
                Ok(0)
            }

            async fn delete(&self, _table: &str, _where_clause: Option<String>) -> Result<u64> {
                Ok(0)
            }
        }

        let table_ops = Arc::new(MockTableOps::default());
        let executor = DdlDmlExecutor::new(
            table_ops.clone(),
            Arc::new(MockIndexOps),
            Arc::new(MockDmlOps),
        );
        let statement = parse_ddl(
            "CREATE TABLE IF NOT EXISTS agent_store (
                record_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                embedding VECTOR(384),
                PRIMARY KEY (record_id)
            )",
        )
        .unwrap();

        let result = executor.execute_ddl(statement).await.unwrap();

        assert_eq!(result.message, "CREATE TABLE agent_store");
        let created = table_ops.created.lock().unwrap();
        assert_eq!(created.len(), 1);
        assert!(matches!(
            created[0].columns.get("payload").unwrap().data_type,
            SqlDataType::Jsonb
        ));
        assert!(matches!(
            created[0].columns.get("embedding").unwrap().data_type,
            SqlDataType::Vector(384)
        ));
    }

    #[test]
    fn test_parse_drop_table() {
        let sql = "DROP TABLE IF EXISTS users";
        let result = parse_ddl(sql).unwrap();

        if let DdlStatement::DropTable { name, if_exists } = result {
            assert_eq!(name, "users");
            assert!(if_exists);
        } else {
            panic!("Expected DropTable");
        }
    }

    #[test]
    fn test_parse_insert() {
        let sql = "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')";
        let result = parse_dml(sql).unwrap();

        if let DmlStatement::Insert {
            table,
            columns,
            values,
        } = result
        {
            assert_eq!(table, "users");
            assert_eq!(columns, vec!["id", "name"]);
            assert_eq!(values.len(), 2);
        } else {
            panic!("Expected Insert");
        }
    }

    #[test]
    fn test_parse_create_index() {
        let sql = "CREATE INDEX users_name_idx ON users (name)";
        let result = parse_ddl(sql).unwrap();

        if let DdlStatement::CreateIndex {
            index,
            if_not_exists,
        } = result
        {
            assert_eq!(index.name, "users_name_idx");
            assert_eq!(index.table, "users");
            assert_eq!(index.columns, vec!["name".to_string()]);
            assert!(matches!(index.index_type, IndexType::BTree));
            assert!(!if_not_exists);
        } else {
            panic!("Expected CreateIndex");
        }
    }

    #[test]
    fn test_parse_drop_index() {
        let sql = "DROP INDEX users_name_idx ON users";
        let result = parse_ddl(sql).unwrap();

        if let DdlStatement::DropIndex { name, if_exists } = result {
            assert_eq!(name, "users_name_idx");
            assert!(!if_exists);
        } else {
            panic!("Expected DropIndex");
        }
    }

    #[test]
    fn test_parse_update() {
        let sql = "UPDATE users SET name = 'Charlie' WHERE id = 1";
        let result = parse_dml(sql).unwrap();

        if let DmlStatement::Update {
            table,
            updates,
            where_clause,
        } = result
        {
            assert_eq!(table, "users");
            assert_eq!(updates.len(), 1);
            assert!(where_clause.is_some());
        } else {
            panic!("Expected Update");
        }
    }

    #[test]
    fn test_parse_delete() {
        let sql = "DELETE FROM users WHERE id = 1";
        let result = parse_dml(sql).unwrap();

        if let DmlStatement::Delete {
            table,
            where_clause,
        } = result
        {
            assert_eq!(table, "users");
            assert!(where_clause.is_some());
        } else {
            panic!("Expected Delete");
        }
    }

    #[test]
    fn parse_ddl_unsupported_statements_still_fail_fast() {
        let unsupported = [
            (
                "CREATE TABLE demo AS SELECT * FROM events;",
                "CREATE TABLE with query/LIKE/CLONE clauses is not supported",
                true,
            ),
            (
                "CREATE TABLE demo LIKE users;",
                "CREATE TABLE with query/LIKE/CLONE clauses is not supported",
                true,
            ),
            (
                "CREATE TABLE demo CLONE users;",
                "CREATE TABLE with query/LIKE/CLONE clauses is not supported",
                true,
            ),
            ("DROP TABLE;", "DROP TABLE requires a table name", true),
            (
                "CREATE INDEX demo_payload_idx ON demo USING GIN (payload);",
                "Unsupported CREATE INDEX USING",
                false,
            ),
            ("DROP VIEW users;", "Unsupported DROP object type", false),
            (
                "DROP INDEX demo_payload_idx;",
                "DROP INDEX requires a table name",
                true,
            ),
            (
                "INSERT INTO users (id) VALUES (1);",
                "Unsupported DDL statement",
                false,
            ),
            ("SELECT * FROM users;", "Unsupported DDL statement", false),
        ];

        for (sql, expected, exact) in unsupported {
            let err = parse_ddl(sql).expect_err("expected unsupported ddl to be rejected");
            let err_msg = err.to_string();
            if exact {
                assert_eq!(err_msg, expected, "unexpected parse error for `{sql}`");
            } else {
                assert!(
                    err_msg.contains(expected),
                    "unexpected parse error for `{sql}`: {err_msg}"
                );
            }
        }
    }
}
