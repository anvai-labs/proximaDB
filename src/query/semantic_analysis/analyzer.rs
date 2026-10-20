//! Contains the main logic for the semantic analysis pass.

use crate::query::ast::{
    BinaryOp, Expr, Literal, ProjectionItem, Query, Select, TableRef, UnaryOp,
};
use crate::query::semantic_analysis::scope::{Column, DataType, Scope, Symbol};
use anyhow::{Result, anyhow};
use proximadb_runtime::CollectionPort;
use std::collections::HashMap;
use std::sync::Arc;

/// The semantic analyzer.
pub struct Analyzer {
    collection_port: Arc<dyn CollectionPort>,
}

impl Analyzer {
    /// Creates a new analyzer.
    pub fn new(collection_port: Arc<dyn CollectionPort>) -> Self {
        Self { collection_port }
    }

    /// Analyzes a query.
    pub async fn analyze(&self, query: &Query) -> Result<Scope> {
        let mut scope = Scope::new();
        self.analyze_query(query, &mut scope).await?;
        Ok(scope)
    }

    fn analyze_query<'a>(
        &'a self,
        query: &'a Query,
        scope: &'a mut Scope,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            match query {
                Query::Select(select) => self.analyze_select(select, scope).await,
                Query::With { ctes, query } => {
                    // Analyze CTEs and add them to the scope
                    for cte in ctes {
                        let mut cte_scope = Scope::new_with_parent(scope.clone());
                        self.analyze_query(&cte.query, &mut cte_scope).await?;
                        // For now, just add the CTE name as a table symbol without columns
                        // Deferred: Extract columns from CTE query for proper type checking
                        scope.insert(
                            &cte.name,
                            Symbol::Table {
                                name: cte.name.clone(),
                                columns: HashMap::new(),
                            },
                        );
                    }
                    self.analyze_query(query, scope).await
                }
                Query::Set { .. } => Err(anyhow!("SET operations are not yet supported")),
            }
        })
    }

    async fn analyze_select(&self, select: &Select, scope: &mut Scope) -> Result<()> {
        // Analyze the FROM clause and populate the scope with tables and columns.
        for table_ref in &select.from {
            self.analyze_table_ref(table_ref, scope).await?;
        }

        // Analyze JOINs
        for join in &select.joins {
            self.analyze_table_ref(&join.right_table, scope).await?;
            if let Some(on_condition) = &join.on_condition {
                self.analyze_expr(on_condition, scope).await?;
            }
        }

        // Analyze the WHERE clause.
        if let Some(selection) = &select.selection {
            self.analyze_expr(selection, scope).await?;
        }

        // Analyze the GROUP BY clause.
        for expr in &select.group_by {
            self.analyze_expr(expr, scope).await?;
        }

        // Analyze the HAVING clause.
        if let Some(having) = &select.having {
            self.analyze_expr(having, scope).await?;
        }

        // Analyze the projection.
        for item in &select.projection {
            self.analyze_projection_item(item, scope).await?;
        }

        // Analyze the ORDER BY clause.
        for order_expr in &select.order_by {
            self.analyze_expr(&order_expr.expr, scope).await?;
        }

        Ok(())
    }

    async fn analyze_table_ref(&self, table_ref: &TableRef, scope: &mut Scope) -> Result<()> {
        if let Some(table_name) = &table_ref.name {
            let collection = self
                .collection_port
                .get_collection_internal(table_name)
                .await?;
            if let Some(collection) = collection {
                let mut columns = HashMap::new();

                // Add standard VectorRecord fields that always exist
                columns.insert(
                    "id".to_string(),
                    Column {
                        name: "id".to_string(),
                        data_type: DataType::String,
                    },
                );
                columns.insert(
                    "timestamp".to_string(),
                    Column {
                        name: "timestamp".to_string(),
                        data_type: DataType::Int64,
                    },
                );

                if let Some(config) = collection.config {
                    // Add vector field (supports both "vector" and "embedding" names)
                    if config.dimension > 0 {
                        columns.insert(
                            "vector".to_string(),
                            Column {
                                name: "vector".to_string(),
                                data_type: DataType::Vector(config.dimension as usize),
                            },
                        );
                        columns.insert(
                            "embedding".to_string(),
                            Column {
                                name: "embedding".to_string(),
                                data_type: DataType::Vector(config.dimension as usize),
                            },
                        );
                    }

                    // Add filterable_columns to schema for type checking
                    // These are metadata fields optimized for filtering (stored as separate columns internally)
                    // Users access them as either "field_name" or "metadata.field_name"
                    for field in config.filterable_columns {
                        // Skip filterable columns that conflict with standard fields
                        // Standard fields (id, timestamp, vector, embedding) take precedence
                        if field.name == "id"
                            || field.name == "timestamp"
                            || field.name == "vector"
                            || field.name == "embedding"
                        {
                            continue;
                        }

                        let data_type = match crate::proto::proximadb_v1::FilterableDataType::try_from(field.data_type) {
                            Ok(crate::proto::proximadb_v1::FilterableDataType::FilterableString) => DataType::String,
                            Ok(crate::proto::proximadb_v1::FilterableDataType::FilterableInteger) => DataType::Int64,
                            Ok(crate::proto::proximadb_v1::FilterableDataType::FilterableFloat) => DataType::Float64,
                            Ok(crate::proto::proximadb_v1::FilterableDataType::FilterableBoolean) => DataType::Boolean,
                            _ => DataType::Unknown,
                        };
                        // Register with unqualified name for direct access (e.g., "name")
                        columns.insert(
                            field.name.clone(),
                            Column {
                                name: field.name.clone(),
                                data_type,
                            },
                        );
                    }

                    // Note: Non-filterable metadata fields are NOT pre-registered
                    // They are handled dynamically via "metadata.{key}" syntax with Unknown type
                    // This matches the VectorRecord schema where all metadata is in the metadata map
                }
                // Use alias if provided, otherwise use table name
                let table_symbol = Symbol::Table {
                    name: table_name.clone(),
                    columns,
                };
                if let Some(alias) = &table_ref.alias {
                    scope.insert(alias, table_symbol);
                } else {
                    scope.insert(table_name, table_symbol);
                }
            } else {
                return Err(anyhow!("Table not found: {}", table_name));
            }
        } else if let Some(subquery) = &table_ref.subquery {
            // Analyze subquery and add its projected columns to the scope
            let mut subquery_scope = Scope::new_with_parent(scope.clone());
            self.analyze_query(subquery, &mut subquery_scope).await?;
            // Deferred: Extract projected columns from subquery for proper type checking
            // For now, just add a placeholder table symbol
            if let Some(alias) = &table_ref.alias {
                scope.insert(
                    alias,
                    Symbol::Table {
                        name: alias.clone(),
                        columns: HashMap::new(),
                    },
                );
            }
        }
        Ok(())
    }

    async fn analyze_projection_item(
        &self,
        item: &ProjectionItem,
        scope: &mut Scope,
    ) -> Result<()> {
        self.analyze_expr(&item.expr, scope).await?;
        // If there's an alias, add it to the current scope
        if let Some(alias) = &item.alias {
            scope.insert(alias, Symbol::Alias(item.expr.clone()));
        }
        Ok(())
    }

    fn analyze_expr<'a>(
        &'a self,
        expr: &'a Expr,
        scope: &'a mut Scope,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<DataType>> + Send + 'a>> {
        Box::pin(async move {
            match expr {
                Expr::Identifier(ident) => {
                    // Handle wildcard SELECT *
                    if ident == "*" {
                        // Wildcard is valid in projection context, return Unknown type
                        // The actual expansion happens in the lowering phase
                        return Ok(DataType::Unknown);
                    }

                    // Handle qualified identifiers (e.g., table.column or metadata.field)
                    // Use fallback strategy: try table.column first, then compound identifier
                    let parts: Vec<&str> = ident.split('.').collect();
                    if parts.len() == 2 {
                        let table_name = parts[0];
                        let column_name = parts[1];

                        // Special handling for wildcard table.* (e.g., metadata.*)
                        if column_name == "*" {
                            // table.* is valid, return Unknown type
                            // The actual expansion happens in the lowering phase
                            return Ok(DataType::Unknown);
                        }

                        // Strategy: Try multiple interpretations before failing
                        // 1. Try as table.column (standard SQL)
                        if let Some(Symbol::Table { columns, .. }) = scope.lookup(table_name) {
                            // First try full qualified name (for compound identifiers like metadata.field)
                            if let Some(column) = columns.get(ident) {
                                let column_clone = column.clone();
                                scope.insert(ident, Symbol::Column(column_clone.clone()));
                                return Ok(column_clone.data_type);
                            }
                            // Then try unqualified column name (standard table.column)
                            if let Some(column) = columns.get(column_name) {
                                let column_clone = column.clone();
                                scope.insert(ident, Symbol::Column(column_clone.clone()));
                                return Ok(column_clone.data_type);
                            }
                        }

                        // 2. Try as compound identifier (JSON field access like metadata.field)
                        // Look for the full qualified name in ANY table's columns
                        let (found_column, _, _) = scope.find_column_in_tables(ident);
                        if let Some(column) = found_column {
                            let column_clone = column.clone();
                            scope.insert(ident, Symbol::Column(column_clone.clone()));
                            return Ok(column_clone.data_type);
                        }

                        // 3. If table_name is "metadata", try looking up the unqualified column name
                        // This handles "metadata.name" references where "name" is in filterable_columns
                        if table_name == "metadata" {
                            let (found_column, _, _) = scope.find_column_in_tables(column_name);
                            if let Some(column) = found_column {
                                let column_clone = column.clone();
                                scope.insert(ident, Symbol::Column(column_clone.clone()));
                                return Ok(column_clone.data_type);
                            }
                            // If not in filterable_columns, allow dynamic field access with Unknown type
                            let inferred_column = Column {
                                name: ident.to_string(),
                                data_type: DataType::Unknown,
                            };
                            scope.insert(ident, Symbol::Column(inferred_column.clone()));
                            return Ok(DataType::Unknown);
                        }

                        // 4. All strategies failed - report error
                        Err(anyhow!(
                            "Identifier '{}' not found. Tried as table.column and as compound identifier.",
                            ident
                        ))
                    } else if parts.len() == 1 {
                        // Unqualified identifier (e.g., column)
                        // Special handling for "metadata" - treat as reserved identifier for the entire metadata object
                        if ident == "metadata" {
                            return Ok(DataType::Unknown); // JSON object type
                        }

                        if let Some(symbol) = scope.lookup(ident) {
                            match symbol {
                                Symbol::Column(column) => Ok(column.data_type.clone()),
                                Symbol::Alias(aliased_expr) => {
                                    let expr_clone = aliased_expr.clone();
                                    self.analyze_expr(&expr_clone, scope).await
                                }
                                _ => {
                                    Err(anyhow!("Identifier '{}' is not a column or alias", ident))
                                }
                            }
                        } else {
                            // Try to find in any table - check for ambiguity
                            let (found_column, found_count, table_names) =
                                scope.find_column_in_tables(ident);

                            if found_count > 1 {
                                Err(anyhow!(
                                    "Ambiguous column reference: '{}' found in tables: {}. Please qualify with table name.",
                                    ident,
                                    table_names.join(", ")
                                ))
                            } else if let Some(column) = found_column {
                                scope.insert(ident, Symbol::Column(column.clone()));
                                Ok(column.data_type.clone())
                            } else {
                                Err(anyhow!("Identifier not found: {}", ident))
                            }
                        }
                    } else {
                        Err(anyhow!("Invalid identifier format: {}", ident))
                    }
                }
                Expr::Literal(literal) => self.analyze_literal(literal),
                Expr::Binary { left, op, right } => {
                    let left_type = self.analyze_expr(left, scope).await?;
                    let right_type = self.analyze_expr(right, scope).await?;
                    self.analyze_binary_op(op, left_type, right_type)
                }
                Expr::Unary { op, expr } => {
                    let expr_type = self.analyze_expr(expr, scope).await?;
                    self.analyze_unary_op(op, expr_type)
                }
                Expr::FuncCall { name, args } => {
                    let mut arg_types = Vec::new();
                    for arg in args {
                        arg_types.push(self.analyze_expr(arg, scope).await?);
                    }
                    self.analyze_function_call(name, arg_types)
                }
                Expr::Case {
                    operand,
                    conditions,
                    else_expr,
                } => {
                    let mut result_type = DataType::Unknown;
                    if let Some(op) = operand {
                        self.analyze_expr(op, scope).await?;
                    }
                    for (when_expr, then_expr) in conditions {
                        let when_type = self.analyze_expr(when_expr, scope).await?;
                        if when_type != DataType::Boolean {
                            return Err(anyhow!(
                                "CASE WHEN clause must be boolean, found {:?}",
                                when_type
                            ));
                        }
                        let then_type = self.analyze_expr(then_expr, scope).await?;
                        if result_type == DataType::Unknown {
                            result_type = then_type;
                        } else if result_type != then_type {
                            // Deferred: Implement type coercion rules
                            return Err(anyhow!("CASE THEN clauses must have compatible types"));
                        }
                    }
                    if let Some(el) = else_expr {
                        let else_type = self.analyze_expr(el, scope).await?;
                        if result_type == DataType::Unknown {
                            result_type = else_type;
                        } else if result_type != else_type {
                            // Deferred: Implement type coercion rules
                            return Err(anyhow!("CASE ELSE clause must have compatible type"));
                        }
                    }
                    Ok(result_type)
                }
                Expr::Subquery(subquery) => {
                    let mut subquery_scope = Scope::new_with_parent(scope.clone());
                    self.analyze_query(subquery, &mut subquery_scope).await?;
                    // Deferred: Determine the return type of the subquery (e.g., single column, single row)
                    // For now, assume it returns a single value of unknown type.
                    // A more robust implementation would analyze the subquery's projection.
                    Ok(DataType::Unknown) // Placeholder
                }
                Expr::SksSimilar {
                    field,
                    query,
                    metric: _,
                    threshold: _,
                } => {
                    // Semantic analysis for SKS_SIMILAR
                    // field is a String (field name), query is an Expr
                    let query_type = self.analyze_expr(query, scope).await?;

                    // Look up field type from scope
                    let field_ident = Expr::Identifier(field.clone());
                    let field_type = self.analyze_expr(&field_ident, scope).await?;

                    // Validate field is a vector type
                    if !matches!(field_type, DataType::Vector(_)) {
                        return Err(anyhow!(
                            "SIMILAR field must be a vector type, found {:?}",
                            field_type
                        ));
                    }

                    if !matches!(query_type, DataType::Vector(_))
                        && !matches!(query_type, DataType::String)
                    {
                        return Err(anyhow!(
                            "SIMILAR query must be a vector or string literal, found {:?}",
                            query_type
                        ));
                    }
                    // Deferred: Validate metric and threshold types
                    Ok(DataType::Float64) // Returns a similarity score
                }
                Expr::SksFollow {
                    start,
                    edge: _,
                    max_depth: _,
                } => {
                    // Semantic analysis for SKS_FOLLOW
                    let start_type = self.analyze_expr(start, scope).await?;
                    if !matches!(start_type, DataType::String)
                        && !matches!(start_type, DataType::Int64)
                    {
                        return Err(anyhow!(
                            "FOLLOW start node must be a string or integer ID, found {:?}",
                            start_type
                        ));
                    }
                    // Deferred: Validate edge type and max_depth
                    Ok(DataType::Unknown) // Returns graph traversal results
                }
                Expr::SksAssemble {
                    context_items,
                    strategy: _,
                    max_size: _,
                } => {
                    // Semantic analysis for SKS_ASSEMBLE
                    for item in context_items {
                        self.analyze_expr(item, scope).await?;
                    }
                    // Deferred: Validate strategy and max_size
                    Ok(DataType::String) // Returns assembled text
                }
                Expr::Array { elem, .. } => {
                    // Array literal (e.g., [0.1, 0.2, 0.3])
                    if elem.is_empty() {
                        return Ok(DataType::Vector(0));
                    }

                    let mut element_type = None;
                    for element in elem {
                        let elem_type = self.analyze_expr(element, scope).await?;
                        if element_type.is_none() {
                            element_type = Some(elem_type.clone());
                        } else if element_type.as_ref() != Some(&elem_type) {
                            return Err(anyhow!("Array elements must have uniform type"));
                        }
                    }

                    match element_type {
                        Some(DataType::Float64) | Some(DataType::Int64) => {
                            Ok(DataType::Vector(elem.len()))
                        }
                        Some(other) => Err(anyhow!("Unsupported array element type: {:?}", other)),
                        None => Ok(DataType::Vector(0)),
                    }
                }
                Expr::AggCall { name, args } => {
                    // Aggregate function calls (COUNT, SUM, AVG, MIN, MAX, etc.)
                    match name.to_uppercase().as_str() {
                        "COUNT" => {
                            // COUNT(*) or COUNT(column)
                            if args.len() == 1 {
                                // Validate argument - for COUNT(*), arg will be Identifier("*")
                                if let Expr::Identifier(ident) = &args[0]
                                    && ident == "*"
                                {
                                    // COUNT(*) - valid
                                    return Ok(DataType::Int64);
                                }
                                // COUNT(column) - analyze the column
                                let _arg_type = self.analyze_expr(&args[0], scope).await?;
                                Ok(DataType::Int64)
                            } else {
                                Err(anyhow!("COUNT expects exactly one argument"))
                            }
                        }
                        "SUM" | "AVG" => {
                            if args.len() != 1 {
                                return Err(anyhow!("{} expects exactly one argument", name));
                            }
                            let arg_type = self.analyze_expr(&args[0], scope).await?;
                            match arg_type {
                                DataType::Int64 | DataType::Float64 => Ok(DataType::Float64),
                                _ => Err(anyhow!("{} requires numeric argument", name)),
                            }
                        }
                        "MIN" | "MAX" => {
                            if args.len() != 1 {
                                return Err(anyhow!("{} expects exactly one argument", name));
                            }
                            let arg_type = self.analyze_expr(&args[0], scope).await?;
                            // MIN/MAX return the same type as their argument
                            Ok(arg_type)
                        }
                        _ => Err(anyhow!("Unsupported aggregate function: {}", name)),
                    }
                }
                _ => Err(anyhow!("Unsupported expression: {:?}", expr)),
            }
        })
    }

    fn analyze_literal(&self, literal: &Literal) -> Result<DataType> {
        match literal {
            Literal::String(_) => Ok(DataType::String),
            Literal::Number(_) => Ok(DataType::Float64), // Treat all numbers as float64 for now
            Literal::Bool(_) => Ok(DataType::Boolean),
            Literal::Null => Ok(DataType::Unknown),
        }
    }

    fn analyze_binary_op(
        &self,
        op: &BinaryOp,
        left: DataType,
        right: DataType,
    ) -> Result<DataType> {
        match op {
            BinaryOp::Eq | BinaryOp::Ne => {
                // Equality can compare compatible types
                if left == right || left == DataType::Unknown || right == DataType::Unknown {
                    Ok(DataType::Boolean)
                } else {
                    Err(anyhow!(
                        "Type mismatch in equality operation: {:?} vs {:?}",
                        left,
                        right
                    ))
                }
            }
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                // Comparison operators require numeric types, but allow Unknown/String for metadata fields
                // that might not have explicit type information at query planning time
                let left_numeric = left == DataType::Int64
                    || left == DataType::Float64
                    || left == DataType::Unknown
                    || left == DataType::String;
                let right_numeric = right == DataType::Int64
                    || right == DataType::Float64
                    || right == DataType::Unknown
                    || right == DataType::String;

                if left_numeric && right_numeric {
                    Ok(DataType::Boolean)
                } else {
                    Err(anyhow!(
                        "Comparison operations require numeric operands: {:?} vs {:?}",
                        left,
                        right
                    ))
                }
            }
            BinaryOp::And | BinaryOp::Or => {
                // Logical operators require boolean types
                if left == DataType::Boolean && right == DataType::Boolean {
                    Ok(DataType::Boolean)
                } else {
                    Err(anyhow!("Logical operations require boolean operands"))
                }
            }
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                // Arithmetic operators require numeric types and return numeric
                if (left == DataType::Int64 || left == DataType::Float64)
                    && (right == DataType::Int64 || right == DataType::Float64)
                {
                    // Promote to Float64 if either is Float64
                    if left == DataType::Float64 || right == DataType::Float64 {
                        Ok(DataType::Float64)
                    } else {
                        Ok(DataType::Int64)
                    }
                } else {
                    Err(anyhow!(
                        "Arithmetic operations require numeric operands: {:?} vs {:?}",
                        left,
                        right
                    ))
                }
            }
            BinaryOp::In | BinaryOp::NotIn => {
                // IN operator: left is a value, right is typically a subquery or list
                // Returns boolean
                // For now, just validate that we have valid types
                // Right side is typically Unknown (subquery) or compatible with left
                Ok(DataType::Boolean)
            }
            BinaryOp::Like => {
                // LIKE operator for string pattern matching
                if left == DataType::String || left == DataType::Unknown {
                    Ok(DataType::Boolean)
                } else {
                    Err(anyhow!("LIKE operator requires string operand"))
                }
            }
            _ => Err(anyhow!("Unsupported binary operator: {:?}", op)),
        }
    }

    fn analyze_unary_op(&self, op: &UnaryOp, expr_type: DataType) -> Result<DataType> {
        match op {
            UnaryOp::Not => {
                if expr_type == DataType::Boolean {
                    Ok(DataType::Boolean)
                } else {
                    Err(anyhow!("NOT operator requires a boolean operand"))
                }
            }
            UnaryOp::Neg => {
                if expr_type == DataType::Float64 || expr_type == DataType::Int64 {
                    Ok(expr_type)
                } else {
                    Err(anyhow!("Negation operator requires a numeric operand"))
                }
            }
        }
    }

    fn analyze_function_call(&self, name: &str, args: Vec<DataType>) -> Result<DataType> {
        // This is a placeholder for a proper function registry.
        match name.to_lowercase().as_str() {
            "cosine_distance" | "vector_similarity" | "similar" => {
                // Accept 2 args (vector, query) or 3 args (vector, query, metric)
                if args.len() == 2 || args.len() == 3 {
                    // For SIMILAR function, first arg should be a column (can be vector), second should be vector
                    // Third arg (if present) is distance metric (string)
                    match (&args[0], &args[1]) {
                        (DataType::Vector(_), DataType::Vector(_)) => Ok(DataType::Float64),
                        (DataType::Float64, DataType::Vector(_)) => Ok(DataType::Float64), // embedding column + vector
                        (DataType::String, DataType::Vector(_)) => {
                            // String field name referring to vector column
                            Ok(DataType::Float64)
                        }
                        (DataType::Unknown, DataType::Vector(_)) => {
                            // Unknown type (e.g., identifier not yet resolved) + vector
                            Ok(DataType::Float64)
                        }
                        (DataType::Vector(_), DataType::Unknown) => {
                            // Vector + unknown query (e.g., subquery or complex expr)
                            Ok(DataType::Float64)
                        }
                        _ => Err(anyhow!(
                            "Invalid arguments for vector similarity function. Expected (Vector/Field, Vector)"
                        )),
                    }
                } else {
                    Err(anyhow!(
                        "Invalid arguments for vector similarity function. Expected 2 or 3 arguments"
                    ))
                }
            }
            "follow" => {
                if args.len() == 3 {
                    // FOLLOW(start_node, edge_type, max_depth)
                    Ok(DataType::Unknown) // Returns graph traversal results
                } else {
                    Err(anyhow!(
                        "Invalid arguments for FOLLOW function. Expected 3 arguments"
                    ))
                }
            }
            "assemble" => {
                // ASSEMBLE can take variable number of arguments
                Ok(DataType::String) // Returns assembled text
            }
            "count" => Ok(DataType::Int64),
            "sum" | "avg" | "min" | "max" => {
                if args.len() == 1
                    && (matches!(args[0], DataType::Int64) || matches!(args[0], DataType::Float64))
                {
                    Ok(DataType::Float64) // Aggregates often return float
                } else {
                    Err(anyhow!(
                        "Invalid arguments for aggregate function. Expected (Numeric)"
                    ))
                }
            }
            _ => Err(anyhow!("Unknown function: {}", name)),
        }
    }
}
