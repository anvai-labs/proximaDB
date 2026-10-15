//! Canonical relational row contract for xCatalog tables.
//!
//! This module is intentionally storage-neutral. SQL, Arrow Flight, REST/gRPC,
//! and embedded loaders can validate/project rows here before they become
//! canonical `ProximaRecord` writes in the storage layer.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow};
use proximadb_data_model::{ProximaType, ProximaValue};
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};
use serde::{Deserialize, Serialize};

use crate::{CatalogColumn, CatalogTableSchema, ColumnConstraint};

/// Which relational integrity rule a write violated (TD-DML-1).
///
/// Carried by [`RelationalConstraintViolation`] so protocol surfaces map a
/// rejected write to the SQLSTATE a PostgreSQL driver keys on, instead of a
/// generic error class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationalConstraintKind {
    /// PRIMARY KEY or secondary UNIQUE duplicate — SQLSTATE `23505`.
    Unique,
    /// FOREIGN KEY reference to a missing parent row — SQLSTATE `23503`.
    ForeignKey,
    /// CHECK constraint evaluated to FALSE — SQLSTATE `23514`.
    Check,
    /// One `INSERT … ON CONFLICT DO UPDATE` statement proposes the same
    /// conflict key twice — SQLSTATE `21000` (cardinality_violation).
    UpsertAffectsRowTwice,
}

impl RelationalConstraintKind {
    /// PostgreSQL SQLSTATE for this violation class.
    pub fn sqlstate(self) -> &'static str {
        match self {
            Self::Unique => "23505",
            Self::ForeignKey => "23503",
            Self::Check => "23514",
            Self::UpsertAffectsRowTwice => "21000",
        }
    }
}

/// A typed relational integrity violation (TD-DML-1).
///
/// Raised by the shared row-write enforcement (INSERT, UPSERT, UPDATE) and by
/// [`CatalogRow::validate`] for CHECK constraints. It travels inside
/// `anyhow::Error`; protocol surfaces recover it with
/// [`find_relational_constraint_violation`]. `Display` is the specific,
/// human-readable message only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalConstraintViolation {
    /// Violation class (selects the SQLSTATE).
    pub kind: RelationalConstraintKind,
    /// Specific message naming the table, constraint and offending value.
    pub message: String,
}

impl RelationalConstraintViolation {
    /// Build a violation of `kind` with a specific `message`.
    pub fn new(kind: RelationalConstraintKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// PostgreSQL SQLSTATE for this violation.
    pub fn sqlstate(&self) -> &'static str {
        self.kind.sqlstate()
    }
}

impl std::fmt::Display for RelationalConstraintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RelationalConstraintViolation {}

/// Find a [`RelationalConstraintViolation`] anywhere in `err`'s chain (it may
/// be wrapped by `.context(..)` on its way to the protocol surface).
pub fn find_relational_constraint_violation(
    err: &anyhow::Error,
) -> Option<&RelationalConstraintViolation> {
    err.chain()
        .find_map(|source| source.downcast_ref::<RelationalConstraintViolation>())
}

/// Schema enforcement mode for table writes and external/schema-on-read scans.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationalSchemaMode {
    /// Enforce catalog schema before accepting a row into canonical storage.
    #[default]
    SchemaOnWrite,
    /// Validate/project known columns but allow source-shaped rows to carry extras.
    SchemaOnRead,
}

/// Validation profile shared by OLTP, OLAP, Arrow, SQL, and protocol surfaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationalWriteProfile {
    /// Schema-on-write for canonical durable tables; schema-on-read for external files/tables.
    pub mode: RelationalSchemaMode,
    /// Reject values for columns not cataloged in `CatalogTableSchema`.
    pub reject_unknown_columns: bool,
    /// Require all catalog columns to be present, even nullable/defaulted columns.
    pub require_all_columns: bool,
}

impl Default for RelationalWriteProfile {
    fn default() -> Self {
        Self {
            mode: RelationalSchemaMode::SchemaOnWrite,
            reject_unknown_columns: true,
            require_all_columns: false,
        }
    }
}

impl RelationalWriteProfile {
    /// A strict OLTP-style profile for schema-on-write table inserts.
    pub fn oltp() -> Self {
        Self::default()
    }

    /// A permissive profile for external/schema-on-read files and table scans.
    pub fn schema_on_read() -> Self {
        Self {
            mode: RelationalSchemaMode::SchemaOnRead,
            reject_unknown_columns: false,
            require_all_columns: false,
        }
    }

    /// Profile for fast-lane REST/gRPC/Arrow Flight record batch writes: validates only the
    /// columns that are present in the incoming record (null constraints, type compatibility),
    /// without rejecting extra fields or requiring full column coverage.
    pub fn fast_lane() -> Self {
        Self {
            mode: RelationalSchemaMode::SchemaOnWrite,
            reject_unknown_columns: false,
            require_all_columns: false,
        }
    }
}

/// Catalog-validated row values keyed by logical column name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogRow {
    /// Catalog table name this row was validated against.
    pub table: String,
    /// Values for cataloged columns.
    pub values: HashMap<String, ProximaValue>,
}

/// Options used when projecting a validated catalog row into a canonical record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationalRecordOptions {
    /// Ingestion method to store in `ProximaRecord::method`.
    pub method: Option<String>,
    /// Tenant to store in `ProximaRecord::tenant_id`.
    pub tenant_id: Option<String>,
    /// Source/origin system.
    pub origin: Option<String>,
    /// Principal or actor that produced the row.
    pub actor: Option<String>,
    /// Override created timestamp in nanoseconds.
    pub created_at_ns: Option<i64>,
    /// Override updated timestamp in nanoseconds.
    pub updated_at_ns: Option<i64>,
    /// Override valid-from timestamp for MVCC/bi-temporal visibility.
    pub valid_from_ns: Option<i64>,
    /// Override valid-to timestamp for MVCC/bi-temporal visibility.
    pub valid_to_ns: Option<i64>,
    /// Explicit record version for OCC/CDC ordering.
    pub record_version: Option<u64>,
    /// Also project vector columns into `EmbeddingCell`s.
    pub include_vector_embeddings: bool,
}

impl Default for RelationalRecordOptions {
    fn default() -> Self {
        Self {
            method: Some("relational_row".to_string()),
            tenant_id: None,
            origin: None,
            actor: None,
            created_at_ns: None,
            updated_at_ns: None,
            valid_from_ns: None,
            valid_to_ns: None,
            record_version: None,
            include_vector_embeddings: true,
        }
    }
}

/// Relational mutation kind lowered below SQL/pgwire into canonical records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationalMutationKind {
    Insert,
    Upsert,
    Update,
    Delete,
}

impl CatalogRow {
    /// Validate raw values against a catalog schema and return a projected row.
    pub fn validate(
        schema: &CatalogTableSchema,
        values: HashMap<String, ProximaValue>,
        profile: &RelationalWriteProfile,
    ) -> Result<Self> {
        let column_names: HashSet<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();

        if profile.reject_unknown_columns {
            for key in values.keys() {
                if !column_names.contains(key.as_str()) {
                    return Err(anyhow!(
                        "Unknown column '{}' for table '{}'",
                        key,
                        schema.name
                    ));
                }
            }
        }

        let mut projected = HashMap::new();
        for column in &schema.columns {
            match values.get(&column.name) {
                Some(value) => {
                    validate_column_value(schema, column, value)?;
                    projected.insert(column.name.clone(), value.clone());
                }
                None if !column.nullable && column.default_value.is_none() => {
                    return Err(anyhow!(
                        "Missing required column '{}' for table '{}'",
                        column.name,
                        schema.name
                    ));
                }
                None if profile.require_all_columns => {
                    return Err(anyhow!(
                        "Missing column '{}' for table '{}'",
                        column.name,
                        schema.name
                    ));
                }
                None => {}
            }
        }

        let row = Self {
            table: schema.name.clone(),
            values: projected,
        };
        row.primary_key_values(schema)?;
        row.validate_check_constraints(schema)?;
        Ok(row)
    }

    /// Return primary-key values in catalog order.
    pub fn primary_key_values(&self, schema: &CatalogTableSchema) -> Result<Vec<ProximaValue>> {
        let primary_key = effective_primary_key(schema);
        let mut values = Vec::with_capacity(primary_key.len());
        for column in primary_key {
            match self.values.get(column) {
                Some(ProximaValue::Null) | None => {
                    return Err(anyhow!(
                        "Primary key column '{}' is missing or null for table '{}'",
                        column,
                        schema.name
                    ));
                }
                Some(value) => values.push(value.clone()),
            }
        }
        Ok(values)
    }

    fn validate_check_constraints(&self, schema: &CatalogTableSchema) -> Result<()> {
        for constraint in &schema.relational_capabilities.constraints {
            if let ColumnConstraint::Check { expression } = constraint {
                validate_check_expression(schema, self, expression)?;
            }
        }
        Ok(())
    }

    /// Validate primary-key-only values for point UPDATE/DELETE mutation paths.
    ///
    /// Unlike full row validation, this allows non-key required columns to be
    /// absent because the mutation targets an existing row version.
    pub fn validate_primary_key(
        schema: &CatalogTableSchema,
        values: HashMap<String, ProximaValue>,
    ) -> Result<Self> {
        let primary_key = effective_primary_key(schema);
        if primary_key.is_empty() {
            return Err(anyhow!(
                "Table '{}' has no primary key for key-only mutation",
                schema.name
            ));
        }

        let mut projected = HashMap::new();
        for key_column in primary_key {
            let column = schema
                .columns
                .iter()
                .find(|column| column.name == key_column)
                .ok_or_else(|| {
                    anyhow!(
                        "Primary key column '{}' not found in table '{}'",
                        key_column,
                        schema.name
                    )
                })?;
            let value = values.get(key_column).ok_or_else(|| {
                anyhow!(
                    "Missing primary key column '{}' for table '{}'",
                    key_column,
                    schema.name
                )
            })?;
            validate_column_value(schema, column, value)?;
            projected.insert(key_column.to_string(), value.clone());
        }

        Ok(Self {
            table: schema.name.clone(),
            values: projected,
        })
    }

    /// Stable logical row key derived from primary-key values.
    pub fn primary_key_string(&self, schema: &CatalogTableSchema) -> Result<Option<String>> {
        let pk_values = self.primary_key_values(schema)?;
        if pk_values.is_empty() {
            return Ok(None);
        }
        Ok(Some(encode_primary_key_tuple(&pk_values)?))
    }

    /// Project this validated row into the canonical durable record envelope.
    ///
    /// The record keeps all relational column values in `props`. Vector columns
    /// are additionally projected into `embeddings` when requested, making ANN
    /// structures rebuildable sidecars rather than durable authority.
    pub fn to_proxima_record(
        &self,
        schema: &CatalogTableSchema,
        options: &RelationalRecordOptions,
    ) -> Result<ProximaRecord> {
        let record_id = self
            .primary_key_string(schema)?
            .unwrap_or_else(|| new_local_record_id(schema));

        let props = self
            .values
            .iter()
            .map(|(key, value)| (key.clone(), ProximaTreeNode::Value(value.clone())))
            .collect();

        let embeddings = if options.include_vector_embeddings {
            self.vector_embeddings(schema)?
        } else {
            Vec::new()
        };

        let mut record = ProximaRecord {
            oid: record_id.clone(),
            local_id: Some(record_id),
            variation_id: Some(schema.name.clone()),
            tenant_id: options.tenant_id.clone().unwrap_or_default(),
            origin: options.origin.clone(),
            actor: options.actor.clone(),
            method: options.method.clone(),
            props,
            embeddings,
            ..ProximaRecord::default()
        };

        if let Some(created_at_ns) = options.created_at_ns {
            record.created_at_ns = created_at_ns;
        }
        if let Some(updated_at_ns) = options.updated_at_ns.or(options.created_at_ns) {
            record.updated_at_ns = updated_at_ns;
        }
        record.valid_from_ns = options.valid_from_ns;
        record.valid_to_ns = options.valid_to_ns;
        if let Some(record_version) = options.record_version {
            record.record_version = record_version;
        }

        Ok(record)
    }

    /// Project this row into a canonical record for a relational mutation.
    ///
    /// DELETE creates a tombstone-shaped record: same identity, current
    /// timestamp, `valid_to_ns = Some(0)` for legacy-compatible tombstone
    /// detection, and no embeddings. Future storage layers may additionally
    /// close the previous visible version at the snapshot timestamp.
    pub fn to_mutation_record(
        &self,
        schema: &CatalogTableSchema,
        kind: RelationalMutationKind,
        mut options: RelationalRecordOptions,
    ) -> Result<ProximaRecord> {
        let (default_method, default_origin) = match kind {
            RelationalMutationKind::Insert => ("sql_insert", "insert"),
            RelationalMutationKind::Upsert => ("sql_upsert", "upsert"),
            RelationalMutationKind::Update => ("sql_update", "update"),
            RelationalMutationKind::Delete => ("sql_delete", "delete"),
        };
        if options.method.as_deref() == Some("relational_row") || options.method.is_none() {
            options.method = Some(default_method.to_string());
        }
        options
            .origin
            .get_or_insert_with(|| default_origin.to_string());

        if matches!(kind, RelationalMutationKind::Delete) {
            options.include_vector_embeddings = false;
            options.valid_to_ns.get_or_insert(0);
        }

        let mut record = self.to_proxima_record(schema, &options)?;
        if matches!(kind, RelationalMutationKind::Delete) {
            record.embeddings.clear();
        }
        Ok(record)
    }

    fn vector_embeddings(&self, schema: &CatalogTableSchema) -> Result<Vec<EmbeddingCell>> {
        let mut embeddings = Vec::new();
        for column in schema
            .columns
            .iter()
            .filter(|column| matches!(column.data_type, ProximaType::DenseVector { .. }))
        {
            let Some(value) = self.values.get(&column.name) else {
                continue;
            };
            let ProximaValue::DenseVector(vector) = value else {
                continue;
            };
            if vector.is_empty() {
                continue;
            }

            if let Some(expected) = column
                .properties
                .get("dimension")
                .and_then(|dimension| dimension.parse::<usize>().ok())
                && vector.len() != expected
            {
                return Err(anyhow!(
                    "Vector column '{}' expects dimension {}, got {}",
                    column.name,
                    expected,
                    vector.len()
                ));
            }

            embeddings.push(EmbeddingCell {
                model_id: column
                    .properties
                    .get("model_id")
                    .cloned()
                    .unwrap_or_else(|| "default".to_string()),
                modality: column
                    .properties
                    .get("modality")
                    .cloned()
                    .unwrap_or_else(|| "dense_vector".to_string()),
                dim: vector.len() as u32,
                // INT-2.5b: vector is Vec<f32>, wrap in the typed variant.
                values: proximadb_records::EmbeddingValues::Fp32(vector.clone()),
                ..Default::default()
            });
        }

        Ok(embeddings)
    }
}

fn effective_primary_key(schema: &CatalogTableSchema) -> Vec<&str> {
    if !schema.relational_capabilities.primary_key.is_empty() {
        schema
            .relational_capabilities
            .primary_key
            .iter()
            .map(String::as_str)
            .collect()
    } else {
        schema.primary_key.iter().map(String::as_str).collect()
    }
}

fn validate_column_value(
    schema: &CatalogTableSchema,
    column: &CatalogColumn,
    value: &ProximaValue,
) -> Result<()> {
    if matches!(value, ProximaValue::Null) {
        if column.nullable {
            return Ok(());
        }
        return Err(anyhow!(
            "Column '{}' in table '{}' is not nullable",
            column.name,
            schema.name
        ));
    }

    if !matches_catalog_type(value, &column.data_type) {
        return Err(anyhow!(
            "Column '{}' in table '{}' expects {:?}, got {}",
            column.name,
            schema.name,
            column.data_type,
            proxima_value_type_name(value)
        ));
    }

    Ok(())
}

fn validate_check_expression(
    schema: &CatalogTableSchema,
    row: &CatalogRow,
    expression: &str,
) -> Result<()> {
    match evaluate_check_expression(schema, row, expression)? {
        CheckOutcome::Pass | CheckOutcome::Unknown => Ok(()),
        CheckOutcome::Fail => Err(RelationalConstraintViolation::new(
            RelationalConstraintKind::Check,
            format!(
                "CHECK constraint '{}' failed for table '{}'",
                expression, schema.name
            ),
        )
        .into()),
    }
}

/// Evaluate a CHECK expression that may combine simple comparisons with `AND` / `OR`
/// (e.g. `CHECK (a > 0 AND b < 100)`, `CHECK (status = 'ok' OR status = 'pending')`).
/// SQL precedence: `OR` binds looser than `AND`, so we split on the top-level `OR` first, then
/// `AND`; parentheses group sub-expressions. Leaves are single comparisons handled by
/// [`evaluate_simple_check_expression`]. Combination uses SQL three-valued logic.
fn evaluate_check_expression(
    schema: &CatalogTableSchema,
    row: &CatalogRow,
    expression: &str,
) -> Result<CheckOutcome> {
    let expr = strip_outer_parens(expression.trim());
    if let Some((left, right)) = split_top_level_once(expr, "OR") {
        let l = evaluate_check_expression(schema, row, &left)?;
        let r = evaluate_check_expression(schema, row, &right)?;
        return Ok(combine_or(l, r));
    }
    if let Some((left, right)) = split_top_level_once(expr, "AND") {
        let l = evaluate_check_expression(schema, row, &left)?;
        let r = evaluate_check_expression(schema, row, &right)?;
        return Ok(combine_and(l, r));
    }
    evaluate_simple_check_expression(schema, row, expr)
}

/// Strip one balanced pair of outer parentheses wrapping the whole expression
/// (`(a AND b)` → `a AND b`), but leave `(a) AND (b)` untouched.
fn strip_outer_parens(s: &str) -> &str {
    let t = s.trim();
    if let Some(inner) = t.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
        let mut depth = 0i32;
        for c in inner.chars() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        return t; // the opening paren closes before the end → not a wrapper
                    }
                }
                _ => {}
            }
        }
        if depth == 0 {
            return inner.trim();
        }
    }
    t
}

/// Split `s` on the FIRST top-level (paren-depth 0) ` <keyword> ` connective (case-insensitive).
/// Recursion handles chains; the rest of the chain stays in `right`.
fn split_top_level_once(s: &str, keyword: &str) -> Option<(String, String)> {
    let kw = format!(" {keyword} ");
    // `to_ascii_uppercase` keeps byte length (non-ASCII passes through), so byte indices align.
    let upper = s.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    let mut depth = 0i32;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ => {
                if depth == 0 && upper[i..].starts_with(&kw) {
                    return Some((
                        s[..i].trim().to_string(),
                        s[i + kw.len()..].trim().to_string(),
                    ));
                }
            }
        }
        i += 1;
    }
    None
}

fn combine_or(left: CheckOutcome, right: CheckOutcome) -> CheckOutcome {
    use CheckOutcome::*;
    if left == Pass || right == Pass {
        Pass
    } else if left == Unknown || right == Unknown {
        Unknown
    } else {
        Fail
    }
}

fn combine_and(left: CheckOutcome, right: CheckOutcome) -> CheckOutcome {
    use CheckOutcome::*;
    if left == Fail || right == Fail {
        Fail
    } else if left == Unknown || right == Unknown {
        Unknown
    } else {
        Pass
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckOutcome {
    Pass,
    Fail,
    Unknown,
}

fn evaluate_simple_check_expression(
    schema: &CatalogTableSchema,
    row: &CatalogRow,
    expression: &str,
) -> Result<CheckOutcome> {
    let tokens = tokenize_simple_check_expression(expression);
    if tokens.len() != 3 {
        return Err(anyhow!(
            "Unsupported CHECK constraint expression '{}'; only simple column-literal comparisons are enforced",
            expression
        ));
    }

    let left = tokens[0].as_str();
    let op = tokens[1].as_str();
    let right = tokens[2].as_str();

    // Column-to-column comparison (e.g. `CHECK (start_date < end_date)`): both operands are
    // catalog columns. NULL on either side yields SQL UNKNOWN (the row is allowed).
    let left_is_column = schema.columns.iter().any(|column| column.name == left);
    let right_is_column = schema.columns.iter().any(|column| column.name == right);
    if left_is_column && right_is_column {
        let (Some(left_value), Some(right_value)) = (row.values.get(left), row.values.get(right))
        else {
            return Ok(CheckOutcome::Unknown);
        };
        if matches!(left_value, ProximaValue::Null) || matches!(right_value, ProximaValue::Null) {
            return Ok(CheckOutcome::Unknown);
        }
        let passed = compare_value_to_value(left_value, op, right_value).ok_or_else(|| {
            anyhow!(
                "Unsupported CHECK constraint expression '{}'; incompatible column comparison",
                expression
            )
        })?;
        return Ok(if passed {
            CheckOutcome::Pass
        } else {
            CheckOutcome::Fail
        });
    }

    let Some((column_name, literal, column_on_left)) = resolve_check_operands(schema, left, right)
    else {
        return Err(anyhow!(
            "Unsupported CHECK constraint expression '{}'; one side must be a catalog column and the other a literal",
            expression
        ));
    };

    let Some(value) = row.values.get(column_name) else {
        return Ok(CheckOutcome::Unknown);
    };
    if matches!(value, ProximaValue::Null) {
        return Ok(CheckOutcome::Unknown);
    }

    let passed = compare_value_to_literal(value, op, literal, column_on_left).ok_or_else(|| {
        anyhow!(
            "Unsupported CHECK constraint expression '{}'; incompatible value/literal comparison",
            expression
        )
    })?;
    Ok(if passed {
        CheckOutcome::Pass
    } else {
        CheckOutcome::Fail
    })
}

fn tokenize_simple_check_expression(expression: &str) -> Vec<String> {
    expression
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .map(|token| token.trim_matches('"').to_string())
        .collect()
}

fn resolve_check_operands<'a>(
    schema: &'a CatalogTableSchema,
    left: &'a str,
    right: &'a str,
) -> Option<(&'a str, &'a str, bool)> {
    if schema.columns.iter().any(|column| column.name == left) {
        Some((left, right, true))
    } else if schema.columns.iter().any(|column| column.name == right) {
        Some((right, left, false))
    } else {
        None
    }
}

fn compare_value_to_literal(
    value: &ProximaValue,
    op: &str,
    literal: &str,
    column_on_left: bool,
) -> Option<bool> {
    if let Some(value) = proxima_value_as_f64(value)
        && let Ok(literal) = literal.parse::<f64>()
    {
        return compare_ordered(value, op, literal, column_on_left);
    }

    if let Some(value) = proxima_value_as_str(value) {
        let literal = literal.trim_matches('\'');
        return compare_ordered(value, op, literal, column_on_left);
    }

    if let ProximaValue::Boolean(value) = value
        && let Some(literal) = parse_bool_literal(literal)
    {
        return compare_equality(*value, op, literal, column_on_left);
    }

    None
}

/// Compare two row values (both catalog columns) under `op`. Numeric vs numeric and string vs
/// string use ordering; boolean vs boolean uses equality only. Mismatched/unsupported kinds
/// return `None` (the caller surfaces an "incompatible comparison" error).
fn compare_value_to_value(left: &ProximaValue, op: &str, right: &ProximaValue) -> Option<bool> {
    if let (Some(left), Some(right)) = (proxima_value_as_f64(left), proxima_value_as_f64(right)) {
        return compare_ordered(left, op, right, true);
    }
    if let (Some(left), Some(right)) = (proxima_value_as_str(left), proxima_value_as_str(right)) {
        return compare_ordered(left, op, right, true);
    }
    if let (ProximaValue::Boolean(left), ProximaValue::Boolean(right)) = (left, right) {
        return compare_equality(*left, op, *right, true);
    }
    None
}

fn proxima_value_as_f64(value: &ProximaValue) -> Option<f64> {
    match value {
        ProximaValue::Int8(value) => Some(*value as f64),
        ProximaValue::Int16(value) => Some(*value as f64),
        ProximaValue::Int32(value) => Some(*value as f64),
        ProximaValue::Int64(value) => Some(*value as f64),
        ProximaValue::UInt8(value) => Some(*value as f64),
        ProximaValue::UInt16(value) => Some(*value as f64),
        ProximaValue::UInt32(value) => Some(*value as f64),
        ProximaValue::UInt64(value) => Some(*value as f64),
        ProximaValue::Float32(value) => Some(*value as f64),
        ProximaValue::Float64(value) => Some(*value),
        _ => None,
    }
}

fn proxima_value_as_str(value: &ProximaValue) -> Option<&str> {
    match value {
        ProximaValue::String(value) | ProximaValue::Symbol(value) => Some(value.as_str()),
        _ => None,
    }
}

fn parse_bool_literal(literal: &str) -> Option<bool> {
    match literal.trim_matches('\'').to_ascii_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn compare_ordered<T: PartialOrd + PartialEq>(
    value: T,
    op: &str,
    literal: T,
    column_on_left: bool,
) -> Option<bool> {
    let result = match op {
        ">" if column_on_left => value > literal,
        ">" => literal > value,
        ">=" if column_on_left => value >= literal,
        ">=" => literal >= value,
        "<" if column_on_left => value < literal,
        "<" => literal < value,
        "<=" if column_on_left => value <= literal,
        "<=" => literal <= value,
        "=" | "==" => value == literal,
        "!=" | "<>" => value != literal,
        _ => return None,
    };
    Some(result)
}

fn compare_equality<T: PartialEq>(
    value: T,
    op: &str,
    literal: T,
    _column_on_left: bool,
) -> Option<bool> {
    match op {
        "=" | "==" => Some(value == literal),
        "!=" | "<>" => Some(value != literal),
        _ => None,
    }
}

fn matches_catalog_type(value: &ProximaValue, data_type: &ProximaType) -> bool {
    match data_type {
        ProximaType::Boolean => matches!(value, ProximaValue::Boolean(_)),
        ProximaType::Int8 => matches!(value, ProximaValue::Int8(_)),
        ProximaType::Int16 => matches!(value, ProximaValue::Int8(_) | ProximaValue::Int16(_)),
        ProximaType::Int32 => matches!(
            value,
            ProximaValue::Int8(_) | ProximaValue::Int16(_) | ProximaValue::Int32(_)
        ),
        ProximaType::Int64 => matches!(
            value,
            ProximaValue::Int8(_)
                | ProximaValue::Int16(_)
                | ProximaValue::Int32(_)
                | ProximaValue::Int64(_)
        ),
        ProximaType::Float32 => {
            matches!(value, ProximaValue::Float16(_) | ProximaValue::Float32(_))
        }
        ProximaType::Float64 => matches!(
            value,
            ProximaValue::Float16(_) | ProximaValue::Float32(_) | ProximaValue::Float64(_)
        ),
        ProximaType::String => {
            matches!(value, ProximaValue::String(_) | ProximaValue::Symbol(_))
        }
        ProximaType::Binary => matches!(value, ProximaValue::Binary(_)),
        ProximaType::Date => matches!(value, ProximaValue::Date(_)),
        ProximaType::Time(_) => matches!(value, ProximaValue::Time(_, _)),
        ProximaType::Timestamp(_) => matches!(value, ProximaValue::Timestamp(_, _)),
        ProximaType::TimestampTz(_) => matches!(value, ProximaValue::TimestampTz(_, _)),
        ProximaType::Decimal { .. } => matches!(value, ProximaValue::Decimal(_)),
        ProximaType::Uuid => matches!(value, ProximaValue::Uuid(_) | ProximaValue::String(_)),
        ProximaType::Json => matches!(
            value,
            ProximaValue::Json(_)
                | ProximaValue::Jsonb(_)
                | ProximaValue::Map(_)
                | ProximaValue::Struct(_)
        ),
        ProximaType::DenseVector { .. } => matches!(value, ProximaValue::DenseVector(_)),
        ProximaType::SparseVector { .. } => matches!(value, ProximaValue::SparseVector { .. }),
        ProximaType::BinaryVector { .. } => {
            matches!(
                value,
                ProximaValue::BinaryVector(_) | ProximaValue::Binary(_)
            )
        }
        // Richer ProximaType variants without a catalog-column type-check rule
        // (unsigned ints, Float16, Symbol, Jsonb, Array, Map, Struct, Interval,
        // Duration, ULID, geo, Null) are accepted permissively here; relational
        // tables only construct the subset above.
        _ => true,
    }
}

fn stable_value_string(value: &ProximaValue) -> Result<String> {
    let key = match value {
        ProximaValue::String(value)
        | ProximaValue::Symbol(value)
        | ProximaValue::Decimal(value) => value.clone(),
        ProximaValue::Boolean(value) => value.to_string(),
        ProximaValue::Int8(value) => value.to_string(),
        ProximaValue::Int16(value) => value.to_string(),
        ProximaValue::Int32(value) => value.to_string(),
        ProximaValue::Int64(value) => value.to_string(),
        ProximaValue::UInt8(value) => value.to_string(),
        ProximaValue::UInt16(value) => value.to_string(),
        ProximaValue::UInt32(value) => value.to_string(),
        ProximaValue::UInt64(value) => value.to_string(),
        ProximaValue::Uuid(value) | ProximaValue::ULID(value) => {
            value.iter().map(|byte| format!("{byte:02x}")).collect()
        }
        _ => serde_json::to_string(value)
            .map_err(|e| anyhow!("Failed to encode primary key value: {}", e))?,
    };
    Ok(key)
}

/// Encode a primary-key tuple (in catalog PK-column order) into the stable
/// `\u{1f}`-joined form used for the record `oid` and key-only lookups
/// (`get_by_key` compares `record.oid == key` exactly). Single source of truth:
/// both [`CatalogRow::primary_key_string`] and the DML FK-existence / ON DELETE
/// cascade paths route through it, so a probe always matches the encoded `oid`.
pub fn encode_primary_key_tuple(values: &[ProximaValue]) -> Result<String> {
    let mut parts = Vec::with_capacity(values.len());
    for value in values {
        parts.push(stable_value_string(value)?);
    }
    Ok(parts.join("\u{1f}"))
}

/// Inverse of [`encode_primary_key_tuple`]: split an encoded primary-key string
/// into its per-column components (in PK order). A single-column key (no
/// separator) yields a one-element slice; an empty string yields `[""]` (one
/// empty column value), matching `str::split`. `stable_value_string` does not
/// escape `\u{1f}` within String values, so a value containing the separator is
/// ambiguous (a pre-existing limitation for composite-PK insert; not introduced
/// here).
pub fn split_primary_key_tuple(encoded: &str) -> Vec<&str> {
    encoded.split('\u{1f}').collect()
}

fn new_local_record_id(schema: &CatalogTableSchema) -> String {
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("{}:row:{}", schema.name, now_ns)
}

fn proxima_value_type_name(value: &ProximaValue) -> &'static str {
    match value {
        ProximaValue::Boolean(_) => "boolean",
        ProximaValue::Int8(_) => "int8",
        ProximaValue::Int16(_) => "int16",
        ProximaValue::Int32(_) => "int32",
        ProximaValue::Int64(_) => "int64",
        ProximaValue::UInt8(_) => "uint8",
        ProximaValue::UInt16(_) => "uint16",
        ProximaValue::UInt32(_) => "uint32",
        ProximaValue::UInt64(_) => "uint64",
        ProximaValue::Float16(_) => "float16",
        ProximaValue::Float32(_) => "float32",
        ProximaValue::Float64(_) => "float64",
        ProximaValue::Decimal(_) => "decimal",
        ProximaValue::String(_) => "string",
        ProximaValue::Symbol(_) => "symbol",
        ProximaValue::Binary(_) => "binary",
        ProximaValue::Date(_) => "date",
        ProximaValue::Time(_, _) => "time",
        ProximaValue::Timestamp(_, _) => "timestamp",
        ProximaValue::TimestampTz(_, _) => "timestamptz",
        ProximaValue::Uuid(_) => "uuid",
        ProximaValue::ULID(_) => "ulid",
        ProximaValue::Json(_) => "json",
        ProximaValue::Jsonb(_) => "jsonb",
        ProximaValue::Array(_) => "array",
        ProximaValue::Map(_) => "map",
        ProximaValue::Struct(_) => "struct",
        ProximaValue::DenseVector(_) => "dense_vector",
        ProximaValue::SparseVector { .. } => "sparse_vector",
        ProximaValue::BinaryVector(_) => "binary_vector",
        ProximaValue::Null => "null",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CatalogColumn, CatalogIndex, CatalogIndexType, ColumnConstraint, RelationalCapabilities,
    };

    fn users_schema() -> CatalogTableSchema {
        CatalogTableSchema::new("users")
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
            .with_column(CatalogColumn::new(2, "email", ProximaType::String).nullable(false))
            .with_column(CatalogColumn::new(3, "profile", ProximaType::Json))
            .with_column(CatalogColumn::new(
                4,
                "embedding",
                ProximaType::DenseVector {
                    element: proximadb_data_model::VectorElement::Float32,
                    dim: 0,
                },
            ))
            .with_relational_capabilities(RelationalCapabilities {
                primary_key: vec!["id".to_string()],
                ..Default::default()
            })
    }

    #[test]
    fn primary_key_tuple_encode_decode_round_trip() {
        // Composite tuple in PK order; note Boolean encodes as "true" (the `oid`
        // encoder used by `get_by_key`), NOT "t" — the divergence S2 unifies on.
        let values = vec![
            ProximaValue::String("us-east".to_string()),
            ProximaValue::Int64(42),
            ProximaValue::Boolean(true),
        ];
        let encoded = encode_primary_key_tuple(&values).expect("encode");
        assert_eq!(encoded, "us-east\u{1f}42\u{1f}true");
        assert_eq!(
            split_primary_key_tuple(&encoded),
            vec!["us-east", "42", "true"]
        );

        // Single-column tuple has no separator.
        let one = encode_primary_key_tuple(&[ProximaValue::Int64(7)]).expect("encode one");
        assert_eq!(one, "7");
        assert_eq!(split_primary_key_tuple(&one), vec!["7"]);

        // An empty String PK-column value is preserved (leading separator).
        let with_empty = encode_primary_key_tuple(&[
            ProximaValue::String(String::new()),
            ProximaValue::Int64(1),
        ])
        .expect("encode empty-first");
        assert_eq!(with_empty, "\u{1f}1");
        assert_eq!(split_primary_key_tuple(&with_empty), vec!["", "1"]);
    }

    #[test]
    fn validates_schema_on_write_row() {
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert(
            "email".to_string(),
            ProximaValue::String("a@example.com".to_string()),
        );
        values.insert(
            "embedding".to_string(),
            ProximaValue::DenseVector(vec![0.1, 0.2]),
        );

        let row = CatalogRow::validate(&users_schema(), values, &RelationalWriteProfile::oltp())
            .expect("row should validate");
        assert_eq!(row.table, "users");
        assert_eq!(
            row.primary_key_string(&users_schema()).unwrap(),
            Some("42".to_string())
        );
    }

    #[test]
    fn projects_validated_row_to_canonical_record() {
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert(
            "email".to_string(),
            ProximaValue::String("a@example.com".to_string()),
        );
        values.insert(
            "embedding".to_string(),
            ProximaValue::DenseVector(vec![0.1, 0.2]),
        );

        let row = CatalogRow::validate(&users_schema(), values, &RelationalWriteProfile::oltp())
            .expect("row should validate");
        let record = row
            .to_proxima_record(
                &users_schema(),
                &RelationalRecordOptions {
                    method: Some("sql_dml".to_string()),
                    created_at_ns: Some(123),
                    ..Default::default()
                },
            )
            .expect("row should project to record");

        assert_eq!(record.oid, "42");
        assert_eq!(record.variation_id.as_deref(), Some("users"));
        assert_eq!(record.method.as_deref(), Some("sql_dml"));
        assert_eq!(record.created_at_ns, 123);
        assert_eq!(record.updated_at_ns, 123);
        assert_eq!(record.valid_from_ns, None);
        assert_eq!(record.embeddings.len(), 1);
        assert_eq!(record.embeddings[0].dim, 2);
        assert!(record.props.contains_key("email"));
    }

    #[test]
    fn delete_mutation_projects_to_tombstone_record() {
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert(
            "email".to_string(),
            ProximaValue::String("a@example.com".to_string()),
        );

        let row = CatalogRow::validate(&users_schema(), values, &RelationalWriteProfile::oltp())
            .expect("row should validate");
        let record = row
            .to_mutation_record(
                &users_schema(),
                RelationalMutationKind::Delete,
                RelationalRecordOptions {
                    updated_at_ns: Some(456),
                    ..Default::default()
                },
            )
            .expect("delete should project");

        assert_eq!(record.oid, "42");
        assert_eq!(record.valid_to_ns, Some(0));
        assert_eq!(record.origin.as_deref(), Some("delete"));
        assert!(record.embeddings.is_empty());
    }

    #[test]
    fn mutation_projection_marks_insert_and_upsert_distinctly() {
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert(
            "email".to_string(),
            ProximaValue::String("a@example.com".to_string()),
        );

        let row = CatalogRow::validate(&users_schema(), values, &RelationalWriteProfile::oltp())
            .expect("row should validate");
        let insert = row
            .to_mutation_record(
                &users_schema(),
                RelationalMutationKind::Insert,
                RelationalRecordOptions::default(),
            )
            .expect("insert should project");
        let upsert = row
            .to_mutation_record(
                &users_schema(),
                RelationalMutationKind::Upsert,
                RelationalRecordOptions::default(),
            )
            .expect("upsert should project");

        assert_eq!(insert.method.as_deref(), Some("sql_insert"));
        assert_eq!(insert.origin.as_deref(), Some("insert"));
        assert_eq!(upsert.method.as_deref(), Some("sql_upsert"));
        assert_eq!(upsert.origin.as_deref(), Some("upsert"));
    }

    #[test]
    fn rejects_unknown_columns_for_schema_on_write() {
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert(
            "email".to_string(),
            ProximaValue::String("a@example.com".to_string()),
        );
        values.insert("extra".to_string(), ProximaValue::String("x".to_string()));

        let err = CatalogRow::validate(&users_schema(), values, &RelationalWriteProfile::oltp())
            .expect_err("unknown column should fail");
        assert!(err.to_string().contains("Unknown column"));
    }

    #[test]
    fn rejects_missing_required_column() {
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));

        let err = CatalogRow::validate(&users_schema(), values, &RelationalWriteProfile::oltp())
            .expect_err("missing required column should fail");
        assert!(err.to_string().contains("Missing required column 'email'"));
    }

    #[test]
    fn validates_primary_key_only_row_for_delete() {
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));

        let row = CatalogRow::validate_primary_key(&users_schema(), values)
            .expect("primary key should validate");
        assert_eq!(
            row.primary_key_string(&users_schema()).unwrap(),
            Some("42".to_string())
        );
    }

    #[test]
    fn allows_unknown_columns_for_schema_on_read() {
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert(
            "email".to_string(),
            ProximaValue::String("a@example.com".to_string()),
        );
        values.insert("extra".to_string(), ProximaValue::String("x".to_string()));

        let row = CatalogRow::validate(
            &users_schema(),
            values,
            &RelationalWriteProfile::schema_on_read(),
        )
        .expect("schema-on-read row should validate projected columns");
        assert!(!row.values.contains_key("extra"));
    }

    #[test]
    fn rejects_type_mismatch() {
        let mut values = HashMap::new();
        values.insert(
            "id".to_string(),
            ProximaValue::String("not-an-int".to_string()),
        );
        values.insert(
            "email".to_string(),
            ProximaValue::String("a@example.com".to_string()),
        );

        let err = CatalogRow::validate(&users_schema(), values, &RelationalWriteProfile::oltp())
            .expect_err("type mismatch should fail");
        assert!(err.to_string().contains("expects Int64"));
    }

    fn orders_schema_with_check(expression: &str) -> CatalogTableSchema {
        CatalogTableSchema::new("orders")
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
            .with_column(CatalogColumn::new(2, "amount", ProximaType::Float64))
            .with_relational_capabilities(RelationalCapabilities {
                primary_key: vec!["id".to_string()],
                constraints: vec![ColumnConstraint::Check {
                    expression: expression.to_string(),
                }],
                ..Default::default()
            })
    }

    #[test]
    fn enforces_simple_check_constraint_on_write() {
        let schema = orders_schema_with_check("amount > 0");
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert("amount".to_string(), ProximaValue::Float64(10.5));

        CatalogRow::validate(&schema, values, &RelationalWriteProfile::oltp())
            .expect("positive amount should pass CHECK");
    }

    #[test]
    fn rejects_simple_check_constraint_violation_on_write() {
        let schema = orders_schema_with_check("amount > 0");
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert("amount".to_string(), ProximaValue::Float64(-1.0));

        let err = CatalogRow::validate(&schema, values, &RelationalWriteProfile::oltp())
            .expect_err("negative amount should fail CHECK");
        assert!(
            err.to_string()
                .contains("CHECK constraint 'amount > 0' failed")
        );
    }

    fn range_schema_with_check(expression: &str) -> CatalogTableSchema {
        CatalogTableSchema::new("ranges")
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
            .with_column(CatalogColumn::new(2, "lo", ProximaType::Float64))
            .with_column(CatalogColumn::new(3, "hi", ProximaType::Float64))
            .with_relational_capabilities(RelationalCapabilities {
                primary_key: vec!["id".to_string()],
                constraints: vec![ColumnConstraint::Check {
                    expression: expression.to_string(),
                }],
                ..Default::default()
            })
    }

    #[test]
    fn enforces_column_to_column_check_on_write() {
        // TD-110 Slice B: CHECK comparing two columns (`lo < hi`).
        let schema = range_schema_with_check("lo < hi");
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(1));
        values.insert("lo".to_string(), ProximaValue::Float64(1.0));
        values.insert("hi".to_string(), ProximaValue::Float64(2.0));
        CatalogRow::validate(&schema, values, &RelationalWriteProfile::oltp())
            .expect("lo < hi should pass");
    }

    #[test]
    fn rejects_column_to_column_check_violation_on_write() {
        let schema = range_schema_with_check("lo < hi");
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(1));
        values.insert("lo".to_string(), ProximaValue::Float64(5.0));
        values.insert("hi".to_string(), ProximaValue::Float64(2.0));
        let err = CatalogRow::validate(&schema, values, &RelationalWriteProfile::oltp())
            .expect_err("lo >= hi should fail");
        assert!(
            err.to_string()
                .contains("CHECK constraint 'lo < hi' failed")
        );
    }

    #[test]
    fn column_to_column_check_with_null_operand_is_unknown() {
        // SQL UNKNOWN (NULL operand) allows the row — matches column-literal NULL semantics.
        let schema = range_schema_with_check("lo < hi");
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(1));
        values.insert("lo".to_string(), ProximaValue::Float64(5.0));
        values.insert("hi".to_string(), ProximaValue::Null);
        CatalogRow::validate(&schema, values, &RelationalWriteProfile::oltp())
            .expect("NULL operand → UNKNOWN → row allowed");
    }

    fn ab_schema_with_check(expression: &str) -> CatalogTableSchema {
        CatalogTableSchema::new("ab")
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
            .with_column(CatalogColumn::new(2, "a", ProximaType::Float64))
            .with_column(CatalogColumn::new(3, "b", ProximaType::Float64))
            .with_relational_capabilities(RelationalCapabilities {
                primary_key: vec!["id".to_string()],
                constraints: vec![ColumnConstraint::Check {
                    expression: expression.to_string(),
                }],
                ..Default::default()
            })
    }

    fn ab_row(a: f64, b: f64) -> HashMap<String, ProximaValue> {
        HashMap::from([
            ("id".to_string(), ProximaValue::Int64(1)),
            ("a".to_string(), ProximaValue::Float64(a)),
            ("b".to_string(), ProximaValue::Float64(b)),
        ])
    }

    #[test]
    fn enforces_compound_and_check() {
        // TD-110 Slice B: CHECK with AND.
        let schema = ab_schema_with_check("a > 0 AND b < 100");
        CatalogRow::validate(&schema, ab_row(5.0, 10.0), &RelationalWriteProfile::oltp())
            .expect("a>0 AND b<100 holds");
        let err =
            CatalogRow::validate(&schema, ab_row(5.0, 200.0), &RelationalWriteProfile::oltp())
                .expect_err("b>=100 fails the AND");
        assert!(
            err.to_string()
                .contains("CHECK constraint 'a > 0 AND b < 100' failed")
        );
    }

    #[test]
    fn enforces_compound_or_check() {
        let schema = ab_schema_with_check("a > 100 OR b > 100");
        CatalogRow::validate(&schema, ab_row(1.0, 200.0), &RelationalWriteProfile::oltp())
            .expect("b>100 satisfies the OR");
        CatalogRow::validate(&schema, ab_row(1.0, 1.0), &RelationalWriteProfile::oltp())
            .expect_err("neither side holds → OR fails");
    }

    #[test]
    fn compound_check_respects_and_over_or_precedence() {
        // `a > 0 OR b > 0 AND b < 0` == `a > 0 OR (b > 0 AND b < 0)`; the AND is unsatisfiable,
        // so the row passes iff a > 0.
        let schema = ab_schema_with_check("a > 0 OR b > 0 AND b < 0");
        CatalogRow::validate(&schema, ab_row(5.0, 5.0), &RelationalWriteProfile::oltp())
            .expect("a>0 satisfies the OR");
        CatalogRow::validate(&schema, ab_row(-1.0, 5.0), &RelationalWriteProfile::oltp())
            .expect_err("a<=0 and (b>0 AND b<0) is false");
    }

    #[test]
    fn compound_check_with_parens_groups_correctly() {
        // `(a > 0 OR a < -10) AND b > 0` — the parenthesized OR binds before the AND.
        let schema = ab_schema_with_check("(a > 0 OR a < -10) AND b > 0");
        CatalogRow::validate(&schema, ab_row(5.0, 5.0), &RelationalWriteProfile::oltp())
            .expect("a>0 and b>0");
        CatalogRow::validate(&schema, ab_row(-5.0, 5.0), &RelationalWriteProfile::oltp())
            .expect_err("a in [-10,0] fails the parenthesized OR");
    }

    #[test]
    fn check_constraint_unknown_value_passes() {
        let schema = orders_schema_with_check("amount > 0");
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));

        CatalogRow::validate(&schema, values, &RelationalWriteProfile::oltp())
            .expect("missing nullable CHECK input should evaluate to SQL unknown");
    }

    #[test]
    fn unsupported_check_constraint_fails_closed() {
        let schema = orders_schema_with_check("amount > discount + 1");
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert("amount".to_string(), ProximaValue::Float64(10.5));

        let err = CatalogRow::validate(&schema, values, &RelationalWriteProfile::oltp())
            .expect_err("unsupported CHECK expression should fail closed");
        assert!(
            err.to_string()
                .contains("Unsupported CHECK constraint expression")
        );
    }

    fn users_schema_with_unique_email() -> CatalogTableSchema {
        CatalogTableSchema::new("users")
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
            .with_column(CatalogColumn::new(2, "email", ProximaType::String))
            .with_relational_capabilities(RelationalCapabilities {
                primary_key: vec!["id".to_string()],
                unique_indexes: vec![
                    CatalogIndex::new(
                        "unique_email",
                        vec!["email".to_string()],
                        CatalogIndexType::BTree,
                    )
                    .unique(),
                ],
                ..Default::default()
            })
    }

    #[test]
    fn unique_constraint_no_longer_fails_closed_in_row_validation() {
        // TD-110: UNIQUE enforcement moved to the DML insert path (within-batch
        // dedup + cross-existing scan in DmlService::execute_insert). Row-local
        // validation can't see other rows, so it no longer rejects a non-null
        // UNIQUE tuple — it accepts the row and lets the DML path enforce it.
        let schema = users_schema_with_unique_email();
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert(
            "email".to_string(),
            ProximaValue::String("a@example.com".to_string()),
        );

        CatalogRow::validate(&schema, values, &RelationalWriteProfile::oltp())
            .expect("non-null UNIQUE tuple is now accepted by row validation (DML enforces it)");
    }

    #[test]
    fn nullable_unique_constraint_allows_null_tuple_without_index_probe() {
        let schema = users_schema_with_unique_email();
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert("email".to_string(), ProximaValue::Null);

        CatalogRow::validate(&schema, values, &RelationalWriteProfile::oltp())
            .expect("nullable UNIQUE tuple with NULL does not require a uniqueness probe");
    }

    fn orders_schema_with_foreign_key() -> CatalogTableSchema {
        CatalogTableSchema::new("orders")
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
            .with_column(CatalogColumn::new(2, "customer_id", ProximaType::Int64))
            .with_relational_capabilities(RelationalCapabilities {
                primary_key: vec!["id".to_string()],
                constraints: vec![ColumnConstraint::ForeignKey {
                    columns: vec!["customer_id".to_string()],
                    references_table: "customers".to_string(),
                    references_columns: vec!["id".to_string()],
                    on_delete: None,
                    on_update: None,
                }],
                ..Default::default()
            })
    }

    #[test]
    fn foreign_key_no_longer_fails_closed_in_row_validation() {
        // TD-110: FK reference enforcement moved to the DML path
        // (DmlService::enforce_foreign_keys does the cross-table parent lookup,
        // which row-local validation cannot). Validation therefore accepts a
        // non-null FK tuple and lets the DML path verify the reference.
        let schema = orders_schema_with_foreign_key();
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert("customer_id".to_string(), ProximaValue::Int64(7));

        CatalogRow::validate(&schema, values, &RelationalWriteProfile::oltp()).expect(
            "non-null FK tuple is now accepted by row validation (DML enforces references)",
        );
    }

    #[test]
    fn nullable_foreign_key_allows_null_tuple_without_reference_probe() {
        let schema = orders_schema_with_foreign_key();
        let mut values = HashMap::new();
        values.insert("id".to_string(), ProximaValue::Int64(42));
        values.insert("customer_id".to_string(), ProximaValue::Null);

        CatalogRow::validate(&schema, values, &RelationalWriteProfile::oltp())
            .expect("nullable FK tuple with NULL does not require a reference probe");
    }
}
