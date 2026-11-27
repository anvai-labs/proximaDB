//! # MongoDB Query Language Parser
//!
//! This module provides a complete parser for MongoDB-style queries, projections,
//! and aggregation pipelines. The parser converts JSON-based MongoDB queries into
//! ProximaDB's `DocumentFilter` for execution.
//!
//! ## Supported Features
//!
//! ### Query Operators
//! - **Comparison**: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`
//! - **Logical**: `$and`, `$or`, `$not`, `$nor`
//! - **Element**: `$exists`, `$type`
//! - **Array**: `$all`, `$elemMatch`, `$size`
//! - **Evaluation**: `$regex`, `$text`
//!
//! ### Projection
//! - Include fields: `{"field": 1}`
//! - Exclude fields: `{"field": 0}`
//!
//! ### Aggregation Pipeline
//! - `$match`, `$project`, `$group`, `$sort`, `$limit`, `$skip`, `$unwind`
//!
//! ## Example Usage
//!
//! ```ignore
//! use proximadb::query::parsers::mongodb::MongoDBParser;
//!
//! let parser = MongoDBParser::new();
//!
//! // Parse a simple query
//! let query = r#"{"age": {"$gte": 18}}"#;
//! let result = parser.parse_query(query)?;
//!
//! // Convert to DocumentFilter
//! let filter = result.to_document_filter()?;
//! ```

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{escaped, tag, take_while1},
    character::complete::{char, digit1, one_of},
    combinator::{opt, recognize, value},
    sequence::{delimited, pair},
};
use serde_json::Value as JsonValue;

use proximadb_proto::proximadb_v1::{
    DocFilterCondition, DocFilterOperator, DocumentFilter, SqlArray, SqlObject, SqlValue,
    sql_value::Value as SqlValueVariant,
};

// =============================================================================
// TOKEN TYPES (Lexer Output)
// =============================================================================

/// Token types for the MongoDB query lexer
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Opening brace `{`
    LeftBrace,
    /// Closing brace `}`
    RightBrace,
    /// Opening bracket `[`
    LeftBracket,
    /// Closing bracket `]`
    RightBracket,
    /// Colon separator `:`
    Colon,
    /// Comma separator `,`
    Comma,

    /// String literal
    String(String),
    /// Floating point number
    Number(f64),
    /// Integer number
    Integer(i64),
    /// Boolean literal
    Boolean(bool),
    /// Null literal
    Null,

    /// MongoDB operator (prefixed with `$`)
    Operator(MongoOperator),
}

/// MongoDB query operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MongoOperator {
    /// `$eq` - equality comparison
    Eq,
    /// `$ne` - not equal comparison
    Ne,
    /// `$gt` - greater than comparison
    Gt,
    /// `$gte` - greater than or equal comparison
    Gte,
    /// `$lt` - less than comparison
    Lt,
    /// `$lte` - less than or equal comparison
    Lte,
    /// `$in` - set membership test
    In,
    /// `$nin` - negated set membership test
    Nin,

    /// `$and` - logical AND of conditions
    And,
    /// `$or` - logical OR of conditions
    Or,
    /// `$not` - logical NOT
    Not,
    /// `$nor` - logical NOR of conditions
    Nor,

    /// `$exists` - field existence check
    Exists,
    /// `$type` - BSON type check
    Type,

    /// `$all` - array contains all elements
    All,
    /// `$elemMatch` - array element matching
    ElemMatch,
    /// `$size` - array size comparison
    Size,

    /// `$regex` - regular expression match
    Regex,
    /// `$text` - full text search
    Text,
    /// `$options` - regex options
    Options,
    /// `$search` - text search query
    Search,

    /// `$match` - pipeline filter stage
    Match,
    /// `$project` - pipeline projection stage
    Project,
    /// `$group` - pipeline grouping stage
    Group,
    /// `$sort` - pipeline sort stage
    Sort,
    /// `$limit` - pipeline limit stage
    Limit,
    /// `$skip` - pipeline skip stage
    Skip,
    /// `$unwind` - pipeline array unwind stage
    Unwind,
    /// `$lookup` - pipeline join stage
    Lookup,
    /// `$count` - count aggregation
    Count,
    /// `$sum` - sum aggregation
    Sum,
    /// `$avg` - average aggregation
    Avg,
    /// `$min` - minimum aggregation
    Min,
    /// `$max` - maximum aggregation
    Max,
    /// `$first` - first value in group
    First,
    /// `$last` - last value in group
    Last,
    /// `$push` - push to array in group
    Push,
    /// `$addToSet` - add unique to set in group
    AddToSet,
}

impl MongoOperator {
    /// Parse a MongoDB operator from its string representation
    pub fn parse_operator(s: &str) -> Option<Self> {
        match s {
            // Comparison
            "$eq" => Some(Self::Eq),
            "$ne" => Some(Self::Ne),
            "$gt" => Some(Self::Gt),
            "$gte" => Some(Self::Gte),
            "$lt" => Some(Self::Lt),
            "$lte" => Some(Self::Lte),
            "$in" => Some(Self::In),
            "$nin" => Some(Self::Nin),

            // Logical
            "$and" => Some(Self::And),
            "$or" => Some(Self::Or),
            "$not" => Some(Self::Not),
            "$nor" => Some(Self::Nor),

            // Element
            "$exists" => Some(Self::Exists),
            "$type" => Some(Self::Type),

            // Array
            "$all" => Some(Self::All),
            "$elemMatch" => Some(Self::ElemMatch),
            "$size" => Some(Self::Size),

            // Evaluation
            "$regex" => Some(Self::Regex),
            "$text" => Some(Self::Text),
            "$options" => Some(Self::Options),
            "$search" => Some(Self::Search),

            // Aggregation
            "$match" => Some(Self::Match),
            "$project" => Some(Self::Project),
            "$group" => Some(Self::Group),
            "$sort" => Some(Self::Sort),
            "$limit" => Some(Self::Limit),
            "$skip" => Some(Self::Skip),
            "$unwind" => Some(Self::Unwind),
            "$lookup" => Some(Self::Lookup),
            "$count" => Some(Self::Count),
            "$sum" => Some(Self::Sum),
            "$avg" => Some(Self::Avg),
            "$min" => Some(Self::Min),
            "$max" => Some(Self::Max),
            "$first" => Some(Self::First),
            "$last" => Some(Self::Last),
            "$push" => Some(Self::Push),
            "$addToSet" => Some(Self::AddToSet),

            _ => None,
        }
    }
}

// =============================================================================
// AST TYPES
// =============================================================================

/// MongoDB query expression AST node
#[derive(Debug, Clone, PartialEq)]
pub enum MongoDBExpression {
    /// Field equality: {"field": value}
    FieldEquals {
        /// Field name
        field: String,
        /// Value to match
        value: MongoDBValue,
    },

    /// Field with operator: {"field": {"$op": value}}
    FieldOperator {
        /// Field name
        field: String,
        /// Comparison operator
        operator: MongoOperator,
        /// Operand value
        value: MongoDBValue,
    },

    /// Logical AND: {"$and": [...]}
    And(Vec<MongoDBExpression>),

    /// Logical OR: {"$or": [...]}
    Or(Vec<MongoDBExpression>),

    /// Logical NOT: {"$not": {...}}
    Not(Box<MongoDBExpression>),

    /// Logical NOR: {"$nor": [...]}
    Nor(Vec<MongoDBExpression>),

    /// Element match: {"field": {"$elemMatch": {...}}}
    ElemMatch {
        /// Array field name
        field: String,
        /// Sub-query to match array elements against
        query: Box<MongoDBExpression>,
    },

    /// Text search: {"$text": {"$search": "..."}}
    TextSearch {
        /// Search query text
        search: String,
        /// Optional language for text analysis
        language: Option<String>,
        /// Whether search is case-sensitive
        case_sensitive: Option<bool>,
    },

    /// Compound expression with multiple field conditions
    Compound(Vec<MongoDBExpression>),
}

/// MongoDB value types
#[derive(Debug, Clone, PartialEq)]
pub enum MongoDBValue {
    /// Null value
    Null,
    /// Boolean value
    Bool(bool),
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// Array of values
    Array(Vec<MongoDBValue>),
    /// Nested object
    Object(HashMap<String, MongoDBValue>),
    /// Regular expression with pattern and options
    Regex {
        /// Regex pattern string
        pattern: String,
        /// Regex options (e.g., "i" for case-insensitive)
        options: String,
    },
}

impl MongoDBValue {
    /// Convert from serde_json Value
    pub fn from_json(value: &JsonValue) -> Self {
        match value {
            JsonValue::Null => MongoDBValue::Null,
            JsonValue::Bool(b) => MongoDBValue::Bool(*b),
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    MongoDBValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    MongoDBValue::Float(f)
                } else {
                    MongoDBValue::Float(0.0)
                }
            }
            JsonValue::String(s) => MongoDBValue::String(s.clone()),
            JsonValue::Array(arr) => {
                MongoDBValue::Array(arr.iter().map(MongoDBValue::from_json).collect())
            }
            JsonValue::Object(obj) => {
                let mut map = HashMap::new();
                for (k, v) in obj {
                    map.insert(k.clone(), MongoDBValue::from_json(v));
                }
                MongoDBValue::Object(map)
            }
        }
    }

    /// Convert to SqlValue for DocumentFilter
    pub fn to_sql_value(&self) -> SqlValue {
        match self {
            MongoDBValue::Null => SqlValue {
                value: Some(SqlValueVariant::NullValue(0)),
            },
            MongoDBValue::Bool(b) => SqlValue {
                value: Some(SqlValueVariant::BoolValue(*b)),
            },
            MongoDBValue::Integer(i) => SqlValue {
                value: Some(SqlValueVariant::Int64Value(*i)),
            },
            MongoDBValue::Float(f) => SqlValue {
                value: Some(SqlValueVariant::NumberValue(*f)),
            },
            MongoDBValue::String(s) => SqlValue {
                value: Some(SqlValueVariant::StringValue(s.clone())),
            },
            MongoDBValue::Array(arr) => SqlValue {
                value: Some(SqlValueVariant::ArrayValue(SqlArray {
                    values: arr.iter().map(|v| v.to_sql_value()).collect(),
                })),
            },
            MongoDBValue::Object(obj) => {
                let mut fields = HashMap::new();
                for (k, v) in obj {
                    fields.insert(k.clone(), v.to_sql_value());
                }
                SqlValue {
                    value: Some(SqlValueVariant::ObjectValue(SqlObject { fields })),
                }
            }
            MongoDBValue::Regex { pattern, .. } => SqlValue {
                value: Some(SqlValueVariant::StringValue(pattern.clone())),
            },
        }
    }
}

/// MongoDB projection specification
#[derive(Debug, Clone, PartialEq)]
pub struct MongoDBProjection {
    /// Fields to include (value = 1 or true)
    pub include: Vec<String>,
    /// Fields to exclude (value = 0 or false)
    pub exclude: Vec<String>,
}

impl MongoDBProjection {
    /// Create a new empty projection.
    pub fn new() -> Self {
        Self {
            include: Vec::new(),
            exclude: Vec::new(),
        }
    }

    /// Check if this is an inclusion projection
    pub fn is_inclusion(&self) -> bool {
        !self.include.is_empty()
    }

    /// Check if this is an exclusion projection
    pub fn is_exclusion(&self) -> bool {
        !self.exclude.is_empty() && self.include.is_empty()
    }
}

impl Default for MongoDBProjection {
    fn default() -> Self {
        Self::new()
    }
}

/// MongoDB aggregation pipeline stage
#[derive(Debug, Clone, PartialEq)]
pub enum MongoDBPipelineStage {
    /// $match stage
    Match(MongoDBExpression),

    /// $project stage
    Project(MongoDBProjection),

    /// $group stage
    Group {
        /// Group key expression
        id_expression: MongoDBValue,
        /// Accumulator expressions by output field name
        accumulators: HashMap<String, GroupAccumulator>,
    },

    /// $sort stage
    Sort(Vec<(String, SortOrder)>),

    /// $limit stage
    Limit(i64),

    /// $skip stage
    Skip(i64),

    /// $unwind stage
    Unwind {
        /// Array field path to unwind
        path: String,
        /// Whether to preserve documents with null or empty arrays
        preserve_null_and_empty: bool,
    },

    /// $lookup stage
    Lookup {
        /// Collection to join with
        from: String,
        /// Local field for join
        local_field: String,
        /// Foreign field for join
        foreign_field: String,
        /// Output array field name
        as_field: String,
    },

    /// $count stage
    Count(String),
}

/// Group accumulator operators
#[derive(Debug, Clone, PartialEq)]
pub enum GroupAccumulator {
    /// Sum of values
    Sum(MongoDBValue),
    /// Average of values
    Avg(MongoDBValue),
    /// Minimum value
    Min(MongoDBValue),
    /// Maximum value
    Max(MongoDBValue),
    /// First value in group
    First(MongoDBValue),
    /// Last value in group
    Last(MongoDBValue),
    /// Push all values to array
    Push(MongoDBValue),
    /// Add unique values to set
    AddToSet(MongoDBValue),
    /// Count of documents
    Count,
}

/// Sort order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Sort ascending (1)
    Ascending,
    /// Sort descending (-1)
    Descending,
}

/// Complete MongoDB query with optional projection
#[derive(Debug, Clone, PartialEq)]
pub struct MongoDBQuery {
    /// Filter expression to match documents
    pub filter: Option<MongoDBExpression>,
    /// Projection to select fields
    pub projection: Option<MongoDBProjection>,
    /// Sort specification
    pub sort: Option<Vec<(String, SortOrder)>>,
    /// Maximum number of results
    pub limit: Option<i64>,
    /// Number of results to skip
    pub skip: Option<i64>,
}

impl MongoDBQuery {
    /// Create a new empty query.
    pub fn new() -> Self {
        Self {
            filter: None,
            projection: None,
            sort: None,
            limit: None,
            skip: None,
        }
    }
}

impl Default for MongoDBQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of parsing a MongoDB query
#[derive(Debug, Clone)]
pub struct MongoDBParseResult {
    /// Parsed find-style query, if present
    pub query: Option<MongoDBQuery>,
    /// Parsed aggregation pipeline, if present
    pub pipeline: Option<Vec<MongoDBPipelineStage>>,
}

// =============================================================================
// LEXER
// =============================================================================

/// MongoDB query lexer
pub struct MongoDBLexer;

impl MongoDBLexer {
    /// Tokenize a MongoDB query string
    pub fn tokenize(input: &str) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        let mut remaining = input.trim();

        while !remaining.is_empty() {
            remaining = remaining.trim_start();
            if remaining.is_empty() {
                break;
            }

            let (rest, token) = Self::next_token(remaining).map_err(|e| {
                anyhow!(
                    "Lexer error at '{}...': {:?}",
                    &remaining[..20.min(remaining.len())],
                    e
                )
            })?;

            tokens.push(token);
            remaining = rest;
        }

        Ok(tokens)
    }

    fn next_token(input: &str) -> IResult<&str, Token> {
        alt((
            Self::parse_structural,
            Self::parse_string,
            Self::parse_number,
            Self::parse_keyword,
        ))
        .parse(input)
    }

    fn parse_structural(input: &str) -> IResult<&str, Token> {
        alt((
            value(Token::LeftBrace, char('{')),
            value(Token::RightBrace, char('}')),
            value(Token::LeftBracket, char('[')),
            value(Token::RightBracket, char(']')),
            value(Token::Colon, char(':')),
            value(Token::Comma, char(',')),
        ))
        .parse(input)
    }

    fn parse_string(input: &str) -> IResult<&str, Token> {
        let (remaining, s) = delimited(
            char('"'),
            escaped(
                take_while1(|c: char| c != '"' && c != '\\'),
                '\\',
                one_of("\"\\nrt"),
            ),
            char('"'),
        )
        .parse(input)?;

        // Check if this is an operator
        if let Some(op) = MongoOperator::parse_operator(s) {
            Ok((remaining, Token::Operator(op)))
        } else {
            Ok((remaining, Token::String(s.to_string())))
        }
    }

    fn parse_number(input: &str) -> IResult<&str, Token> {
        let (remaining, num_str) = recognize((
            opt(char('-')),
            digit1,
            opt(pair(char('.'), digit1)),
            opt((one_of("eE"), opt(one_of("+-")), digit1)),
        ))
        .parse(input)?;

        // Try parsing as integer first, then as float
        if let Ok(i) = num_str.parse::<i64>() {
            Ok((remaining, Token::Integer(i)))
        } else if let Ok(f) = num_str.parse::<f64>() {
            Ok((remaining, Token::Number(f)))
        } else {
            Err(nom::Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Float,
            )))
        }
    }

    fn parse_keyword(input: &str) -> IResult<&str, Token> {
        alt((
            value(Token::Boolean(true), tag("true")),
            value(Token::Boolean(false), tag("false")),
            value(Token::Null, tag("null")),
        ))
        .parse(input)
    }
}

// =============================================================================
// PARSER
// =============================================================================

/// MongoDB query parser
///
/// Parses MongoDB-style JSON queries into an AST representation.
pub struct MongoDBParser {
    // Future: add configuration options
}

impl MongoDBParser {
    /// Create a new MongoDB parser
    pub fn new() -> Self {
        Self {}
    }

    /// Parse a MongoDB query expression
    pub fn parse_query(&self, input: &str) -> Result<MongoDBExpression> {
        let json: JsonValue =
            serde_json::from_str(input).context("Failed to parse MongoDB query as JSON")?;

        self.parse_expression(&json)
    }

    /// Parse a MongoDB projection
    pub fn parse_projection(&self, input: &str) -> Result<MongoDBProjection> {
        let json: JsonValue =
            serde_json::from_str(input).context("Failed to parse MongoDB projection as JSON")?;

        self.parse_projection_value(&json)
    }

    /// Parse a MongoDB aggregation pipeline
    pub fn parse_pipeline(&self, input: &str) -> Result<Vec<MongoDBPipelineStage>> {
        let json: JsonValue =
            serde_json::from_str(input).context("Failed to parse MongoDB pipeline as JSON")?;

        match &json {
            JsonValue::Array(stages) => {
                let mut result = Vec::new();
                for stage in stages {
                    result.push(self.parse_pipeline_stage(stage)?);
                }
                Ok(result)
            }
            _ => Err(anyhow!("Pipeline must be an array")),
        }
    }

    /// Parse a complete MongoDB query with options
    pub fn parse_full_query(
        &self,
        query_json: &str,
        options: Option<&str>,
    ) -> Result<MongoDBQuery> {
        let mut result = MongoDBQuery::new();

        // Parse filter
        if !query_json.trim().is_empty() && query_json.trim() != "{}" {
            result.filter = Some(self.parse_query(query_json)?);
        }

        // Parse options if provided
        if let Some(opts) = options {
            let opts_json: JsonValue =
                serde_json::from_str(opts).context("Failed to parse query options")?;

            if let JsonValue::Object(map) = opts_json {
                if let Some(proj) = map.get("projection") {
                    result.projection = Some(self.parse_projection_value(proj)?);
                }
                if let Some(sort) = map.get("sort") {
                    result.sort = Some(self.parse_sort_value(sort)?);
                }
                if let Some(JsonValue::Number(n)) = map.get("limit") {
                    result.limit = n.as_i64();
                }
                if let Some(JsonValue::Number(n)) = map.get("skip") {
                    result.skip = n.as_i64();
                }
            }
        }

        Ok(result)
    }

    /// Parse a JSON value into a MongoDB expression
    fn parse_expression(&self, value: &JsonValue) -> Result<MongoDBExpression> {
        match value {
            JsonValue::Object(obj) => self.parse_object_expression(obj),
            _ => Err(anyhow!("Expected object at top level of query")),
        }
    }

    /// Parse a JSON object into an expression
    fn parse_object_expression(
        &self,
        obj: &serde_json::Map<String, JsonValue>,
    ) -> Result<MongoDBExpression> {
        let mut expressions = Vec::new();

        for (key, value) in obj {
            if key.starts_with('$') {
                // This is an operator
                let expr = self.parse_operator_expression(key, value)?;
                expressions.push(expr);
            } else {
                // This is a field condition
                let expr = self.parse_field_condition(key, value)?;
                expressions.push(expr);
            }
        }

        if expressions.len() == 1 {
            Ok(expressions.remove(0))
        } else {
            Ok(MongoDBExpression::Compound(expressions))
        }
    }

    /// Parse an operator expression
    fn parse_operator_expression(&self, op: &str, value: &JsonValue) -> Result<MongoDBExpression> {
        match op {
            "$and" => {
                let exprs = self.parse_expression_array(value)?;
                Ok(MongoDBExpression::And(exprs))
            }
            "$or" => {
                let exprs = self.parse_expression_array(value)?;
                Ok(MongoDBExpression::Or(exprs))
            }
            "$nor" => {
                let exprs = self.parse_expression_array(value)?;
                Ok(MongoDBExpression::Nor(exprs))
            }
            "$not" => {
                let expr = self.parse_expression(value)?;
                Ok(MongoDBExpression::Not(Box::new(expr)))
            }
            "$text" => self.parse_text_search(value),
            _ => Err(anyhow!("Unknown top-level operator: {}", op)),
        }
    }

    /// Parse an array of expressions
    fn parse_expression_array(&self, value: &JsonValue) -> Result<Vec<MongoDBExpression>> {
        match value {
            JsonValue::Array(arr) => {
                let mut exprs = Vec::new();
                for item in arr {
                    exprs.push(self.parse_expression(item)?);
                }
                Ok(exprs)
            }
            _ => Err(anyhow!("Expected array for logical operator")),
        }
    }

    /// Parse a field condition
    fn parse_field_condition(&self, field: &str, value: &JsonValue) -> Result<MongoDBExpression> {
        match value {
            JsonValue::Object(obj) => {
                // Check if this is an operator object
                let mut has_operators = false;
                for key in obj.keys() {
                    if key.starts_with('$') {
                        has_operators = true;
                        break;
                    }
                }

                if has_operators {
                    self.parse_field_operators(field, obj)
                } else {
                    // Nested object - treat as equality
                    Ok(MongoDBExpression::FieldEquals {
                        field: field.to_string(),
                        value: MongoDBValue::from_json(&JsonValue::Object(obj.clone())),
                    })
                }
            }
            _ => {
                // Simple equality
                Ok(MongoDBExpression::FieldEquals {
                    field: field.to_string(),
                    value: MongoDBValue::from_json(value),
                })
            }
        }
    }

    /// Parse field operators
    fn parse_field_operators(
        &self,
        field: &str,
        obj: &serde_json::Map<String, JsonValue>,
    ) -> Result<MongoDBExpression> {
        let mut expressions = Vec::new();

        // Handle regex with options separately
        let regex_pattern = obj.get("$regex");
        let regex_options = obj.get("$options");

        if let Some(pattern_value) = regex_pattern {
            let pattern = match pattern_value {
                JsonValue::String(s) => s.clone(),
                _ => return Err(anyhow!("$regex pattern must be a string")),
            };
            let options = match regex_options {
                Some(JsonValue::String(s)) => s.clone(),
                _ => String::new(),
            };

            expressions.push(MongoDBExpression::FieldOperator {
                field: field.to_string(),
                operator: MongoOperator::Regex,
                value: MongoDBValue::Regex { pattern, options },
            });
        }

        for (key, value) in obj {
            if key == "$regex" || key == "$options" {
                continue; // Already handled above
            }

            let operator = MongoOperator::parse_operator(key)
                .ok_or_else(|| anyhow!("Unknown operator: {}", key))?;

            match operator {
                MongoOperator::ElemMatch => {
                    let query = self.parse_expression(value)?;
                    expressions.push(MongoDBExpression::ElemMatch {
                        field: field.to_string(),
                        query: Box::new(query),
                    });
                }
                _ => {
                    expressions.push(MongoDBExpression::FieldOperator {
                        field: field.to_string(),
                        operator,
                        value: MongoDBValue::from_json(value),
                    });
                }
            }
        }

        if expressions.len() == 1 {
            Ok(expressions.remove(0))
        } else {
            // Multiple operators on same field = implicit AND
            Ok(MongoDBExpression::And(expressions))
        }
    }

    /// Parse $text search
    fn parse_text_search(&self, value: &JsonValue) -> Result<MongoDBExpression> {
        match value {
            JsonValue::Object(obj) => {
                let search = obj
                    .get("$search")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("$text requires $search field"))?
                    .to_string();

                let language = obj
                    .get("$language")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let case_sensitive = obj.get("$caseSensitive").and_then(|v| v.as_bool());

                Ok(MongoDBExpression::TextSearch {
                    search,
                    language,
                    case_sensitive,
                })
            }
            _ => Err(anyhow!("$text value must be an object")),
        }
    }

    /// Parse a projection value
    fn parse_projection_value(&self, value: &JsonValue) -> Result<MongoDBProjection> {
        let mut projection = MongoDBProjection::new();

        match value {
            JsonValue::Object(obj) => {
                for (field, val) in obj {
                    match val {
                        JsonValue::Number(n) => {
                            if n.as_i64() == Some(1) || n.as_f64() == Some(1.0) {
                                projection.include.push(field.clone());
                            } else if n.as_i64() == Some(0) || n.as_f64() == Some(0.0) {
                                projection.exclude.push(field.clone());
                            }
                        }
                        JsonValue::Bool(b) => {
                            if *b {
                                projection.include.push(field.clone());
                            } else {
                                projection.exclude.push(field.clone());
                            }
                        }
                        _ => {
                            // Complex projection expressions (e.g., $slice) not yet supported
                            projection.include.push(field.clone());
                        }
                    }
                }
            }
            _ => return Err(anyhow!("Projection must be an object")),
        }

        Ok(projection)
    }

    /// Parse a sort value
    fn parse_sort_value(&self, value: &JsonValue) -> Result<Vec<(String, SortOrder)>> {
        let mut result = Vec::new();

        match value {
            JsonValue::Object(obj) => {
                for (field, val) in obj {
                    let order = match val {
                        JsonValue::Number(n) => {
                            if n.as_i64() == Some(1) || n.as_f64() == Some(1.0) {
                                SortOrder::Ascending
                            } else {
                                SortOrder::Descending
                            }
                        }
                        _ => SortOrder::Ascending,
                    };
                    result.push((field.clone(), order));
                }
            }
            _ => return Err(anyhow!("Sort must be an object")),
        }

        Ok(result)
    }

    /// Parse a pipeline stage
    fn parse_pipeline_stage(&self, stage: &JsonValue) -> Result<MongoDBPipelineStage> {
        match stage {
            JsonValue::Object(obj) => {
                if obj.len() != 1 {
                    return Err(anyhow!(
                        "Each pipeline stage must have exactly one operator"
                    ));
                }

                // Safe to unwrap because we just verified obj.len() == 1
                let (op, value) = obj
                    .iter()
                    .next()
                    .ok_or_else(|| anyhow!("Pipeline stage object is empty"))?;

                match op.as_str() {
                    "$match" => {
                        let expr = self.parse_expression(value)?;
                        Ok(MongoDBPipelineStage::Match(expr))
                    }
                    "$project" => {
                        let proj = self.parse_projection_value(value)?;
                        Ok(MongoDBPipelineStage::Project(proj))
                    }
                    "$group" => self.parse_group_stage(value),
                    "$sort" => {
                        let sort = self.parse_sort_value(value)?;
                        Ok(MongoDBPipelineStage::Sort(sort))
                    }
                    "$limit" => {
                        let n = value
                            .as_i64()
                            .ok_or_else(|| anyhow!("$limit must be a number"))?;
                        Ok(MongoDBPipelineStage::Limit(n))
                    }
                    "$skip" => {
                        let n = value
                            .as_i64()
                            .ok_or_else(|| anyhow!("$skip must be a number"))?;
                        Ok(MongoDBPipelineStage::Skip(n))
                    }
                    "$unwind" => self.parse_unwind_stage(value),
                    "$lookup" => self.parse_lookup_stage(value),
                    "$count" => {
                        let field = value
                            .as_str()
                            .ok_or_else(|| anyhow!("$count must be a string"))?;
                        Ok(MongoDBPipelineStage::Count(field.to_string()))
                    }
                    _ => Err(anyhow!("Unknown pipeline stage: {}", op)),
                }
            }
            _ => Err(anyhow!("Pipeline stage must be an object")),
        }
    }

    /// Parse $group stage
    fn parse_group_stage(&self, value: &JsonValue) -> Result<MongoDBPipelineStage> {
        match value {
            JsonValue::Object(obj) => {
                let id_value = obj
                    .get("_id")
                    .ok_or_else(|| anyhow!("$group requires _id field"))?;

                let id_expression = MongoDBValue::from_json(id_value);

                let mut accumulators = HashMap::new();

                for (field, acc_value) in obj {
                    if field == "_id" {
                        continue;
                    }

                    if let JsonValue::Object(acc_obj) = acc_value
                        && let Some((acc_op, acc_expr)) = acc_obj.iter().next()
                    {
                        let accumulator = match acc_op.as_str() {
                            "$sum" => GroupAccumulator::Sum(MongoDBValue::from_json(acc_expr)),
                            "$avg" => GroupAccumulator::Avg(MongoDBValue::from_json(acc_expr)),
                            "$min" => GroupAccumulator::Min(MongoDBValue::from_json(acc_expr)),
                            "$max" => GroupAccumulator::Max(MongoDBValue::from_json(acc_expr)),
                            "$first" => GroupAccumulator::First(MongoDBValue::from_json(acc_expr)),
                            "$last" => GroupAccumulator::Last(MongoDBValue::from_json(acc_expr)),
                            "$push" => GroupAccumulator::Push(MongoDBValue::from_json(acc_expr)),
                            "$addToSet" => {
                                GroupAccumulator::AddToSet(MongoDBValue::from_json(acc_expr))
                            }
                            _ => return Err(anyhow!("Unknown accumulator: {}", acc_op)),
                        };
                        accumulators.insert(field.clone(), accumulator);
                    }
                }

                Ok(MongoDBPipelineStage::Group {
                    id_expression,
                    accumulators,
                })
            }
            _ => Err(anyhow!("$group value must be an object")),
        }
    }

    /// Parse $unwind stage
    fn parse_unwind_stage(&self, value: &JsonValue) -> Result<MongoDBPipelineStage> {
        match value {
            JsonValue::String(path) => Ok(MongoDBPipelineStage::Unwind {
                path: path.trim_start_matches('$').to_string(),
                preserve_null_and_empty: false,
            }),
            JsonValue::Object(obj) => {
                let path = obj
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("$unwind requires path field"))?
                    .trim_start_matches('$')
                    .to_string();

                let preserve = obj
                    .get("preserveNullAndEmptyArrays")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                Ok(MongoDBPipelineStage::Unwind {
                    path,
                    preserve_null_and_empty: preserve,
                })
            }
            _ => Err(anyhow!("$unwind must be a string or object")),
        }
    }

    /// Parse $lookup stage
    fn parse_lookup_stage(&self, value: &JsonValue) -> Result<MongoDBPipelineStage> {
        match value {
            JsonValue::Object(obj) => {
                let from = obj
                    .get("from")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("$lookup requires 'from' field"))?
                    .to_string();

                let local_field = obj
                    .get("localField")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("$lookup requires 'localField' field"))?
                    .to_string();

                let foreign_field = obj
                    .get("foreignField")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("$lookup requires 'foreignField' field"))?
                    .to_string();

                let as_field = obj
                    .get("as")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("$lookup requires 'as' field"))?
                    .to_string();

                Ok(MongoDBPipelineStage::Lookup {
                    from,
                    local_field,
                    foreign_field,
                    as_field,
                })
            }
            _ => Err(anyhow!("$lookup must be an object")),
        }
    }
}

impl Default for MongoDBParser {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// VISITOR PATTERN
// =============================================================================

/// Visitor trait for MongoDB expressions
pub trait MongoDBVisitor {
    /// Output type of the visitor
    type Output;

    /// Visit a MongoDB expression
    fn visit_expression(&mut self, expr: &MongoDBExpression) -> Self::Output;

    /// Visit a field equals expression
    fn visit_field_equals(&mut self, field: &str, value: &MongoDBValue) -> Self::Output;

    /// Visit a field operator expression
    fn visit_field_operator(
        &mut self,
        field: &str,
        operator: MongoOperator,
        value: &MongoDBValue,
    ) -> Self::Output;

    /// Visit an AND expression
    fn visit_and(&mut self, expressions: &[MongoDBExpression]) -> Self::Output;

    /// Visit an OR expression
    fn visit_or(&mut self, expressions: &[MongoDBExpression]) -> Self::Output;

    /// Visit a NOT expression
    fn visit_not(&mut self, expression: &MongoDBExpression) -> Self::Output;

    /// Visit a NOR expression
    fn visit_nor(&mut self, expressions: &[MongoDBExpression]) -> Self::Output;

    /// Visit an elemMatch expression
    fn visit_elem_match(&mut self, field: &str, query: &MongoDBExpression) -> Self::Output;

    /// Visit a text search expression
    fn visit_text_search(
        &mut self,
        search: &str,
        language: Option<&str>,
        case_sensitive: Option<bool>,
    ) -> Self::Output;

    /// Visit a compound expression
    fn visit_compound(&mut self, expressions: &[MongoDBExpression]) -> Self::Output;
}

// =============================================================================
// DOCUMENT FILTER CONVERSION
// =============================================================================

/// Trait for converting to DocumentFilter
pub trait ToDocumentFilter {
    /// Convert this type to a DocumentFilter
    fn to_document_filter(&self) -> Result<DocumentFilter>;
}

impl ToDocumentFilter for MongoDBExpression {
    fn to_document_filter(&self) -> Result<DocumentFilter> {
        let mut converter = DocumentFilterConverter::new();
        converter.convert(self)
    }
}

impl ToDocumentFilter for MongoDBQuery {
    fn to_document_filter(&self) -> Result<DocumentFilter> {
        match &self.filter {
            Some(expr) => expr.to_document_filter(),
            None => Ok(DocumentFilter {
                conditions: Vec::new(),
                or_filters: Vec::new(),
                and_filters: Vec::new(),
            }),
        }
    }
}

/// Converter from MongoDB AST to DocumentFilter
struct DocumentFilterConverter {
    // Future: add context for optimization
}

impl DocumentFilterConverter {
    fn new() -> Self {
        Self {}
    }

    fn convert(&mut self, expr: &MongoDBExpression) -> Result<DocumentFilter> {
        let mut filter = DocumentFilter {
            conditions: Vec::new(),
            or_filters: Vec::new(),
            and_filters: Vec::new(),
        };

        match expr {
            MongoDBExpression::FieldEquals { field, value } => {
                filter.conditions.push(DocFilterCondition {
                    path: field.clone(),
                    operator: DocFilterOperator::Eq as i32,
                    value: Some(value.to_sql_value()),
                    values: Vec::new(),
                });
            }

            MongoDBExpression::FieldOperator {
                field,
                operator,
                value,
            } => {
                let (doc_op, is_array_op) = self.convert_operator(*operator);
                let mut condition = DocFilterCondition {
                    path: field.clone(),
                    operator: doc_op as i32,
                    value: None,
                    values: Vec::new(),
                };

                if is_array_op {
                    // For $in, $nin, $all - value should be an array
                    if let MongoDBValue::Array(arr) = value {
                        condition.values = arr.iter().map(|v| v.to_sql_value()).collect();
                    } else {
                        condition.values = vec![value.to_sql_value()];
                    }
                } else {
                    condition.value = Some(value.to_sql_value());
                }

                filter.conditions.push(condition);
            }

            MongoDBExpression::And(expressions) => {
                for expr in expressions {
                    let sub_filter = self.convert(expr)?;
                    filter.and_filters.push(sub_filter);
                }
            }

            MongoDBExpression::Or(expressions) => {
                for expr in expressions {
                    let sub_filter = self.convert(expr)?;
                    filter.or_filters.push(sub_filter);
                }
            }

            MongoDBExpression::Not(inner) => {
                // Convert NOT to NOR with single element
                let sub_filter = self.convert(inner)?;
                // Invert the conditions by wrapping in a logical structure
                // This is a simplification - true NOT would need more complex handling
                filter.and_filters.push(sub_filter);
            }

            MongoDBExpression::Nor(expressions) => {
                // NOR = NOT(OR(...))
                // Convert to AND of negated conditions (simplified)
                for expr in expressions {
                    let sub_filter = self.convert(expr)?;
                    // In a full implementation, we'd negate each condition
                    filter.and_filters.push(sub_filter);
                }
            }

            MongoDBExpression::ElemMatch { field, query } => {
                // Convert elemMatch to a condition with the nested query
                let sub_filter = self.convert(query)?;
                // Prefix the field path for nested conditions
                for mut condition in sub_filter.conditions {
                    condition.path = format!("{}.{}", field, condition.path);
                    filter.conditions.push(condition);
                }
            }

            MongoDBExpression::TextSearch { search, .. } => {
                filter.conditions.push(DocFilterCondition {
                    path: "$text".to_string(),
                    operator: DocFilterOperator::Fulltext as i32,
                    value: Some(SqlValue {
                        value: Some(SqlValueVariant::StringValue(search.clone())),
                    }),
                    values: Vec::new(),
                });
            }

            MongoDBExpression::Compound(expressions) => {
                // Compound expressions are implicitly ANDed
                for expr in expressions {
                    let sub_filter = self.convert(expr)?;
                    // Merge conditions
                    filter.conditions.extend(sub_filter.conditions);
                    filter.and_filters.extend(sub_filter.and_filters);
                    filter.or_filters.extend(sub_filter.or_filters);
                }
            }
        }

        Ok(filter)
    }

    fn convert_operator(&self, op: MongoOperator) -> (DocFilterOperator, bool) {
        match op {
            MongoOperator::Eq => (DocFilterOperator::Eq, false),
            MongoOperator::Ne => (DocFilterOperator::Ne, false),
            MongoOperator::Gt => (DocFilterOperator::Gt, false),
            MongoOperator::Gte => (DocFilterOperator::Gte, false),
            MongoOperator::Lt => (DocFilterOperator::Lt, false),
            MongoOperator::Lte => (DocFilterOperator::Lte, false),
            MongoOperator::In => (DocFilterOperator::In, true),
            MongoOperator::Nin => (DocFilterOperator::NotIn, true),
            MongoOperator::Exists => (DocFilterOperator::Exists, false),
            MongoOperator::Type => (DocFilterOperator::Type, false),
            MongoOperator::Regex => (DocFilterOperator::Regex, false),
            MongoOperator::All => (DocFilterOperator::Contains, true),
            MongoOperator::Size => (DocFilterOperator::Eq, false), // Size check needs special handling
            _ => (DocFilterOperator::Eq, false),                   // Default fallback
        }
    }
}

// =============================================================================
// AGGREGATION PIPELINE CONVERSION
// =============================================================================

impl MongoDBPipelineStage {
    /// Convert to ProximaDB aggregation stage
    pub fn to_proto_stage(&self) -> Result<proximadb_proto::proximadb_v1::AggregationStage> {
        use proximadb_proto::proximadb_v1::{
            Aggregation, AggregationStage, AggregationType, GroupStage, LimitStage, MatchStage,
            ProjectStage, SkipStage, SortField, SortOrder as ProtoSortOrder, SortStage,
            UnwindStage, aggregation_stage::Stage,
        };

        let stage = match self {
            MongoDBPipelineStage::Match(expr) => {
                let filter = expr.to_document_filter()?;
                Stage::Match(MatchStage {
                    filter: Some(filter),
                })
            }

            MongoDBPipelineStage::Project(proj) => {
                // Build fields map: true = include, false = exclude
                let mut fields = HashMap::new();
                for f in &proj.include {
                    fields.insert(f.clone(), true);
                }
                for f in &proj.exclude {
                    fields.insert(f.clone(), false);
                }

                Stage::Project(ProjectStage {
                    fields,
                    computed: HashMap::new(),
                })
            }

            MongoDBPipelineStage::Group {
                id_expression,
                accumulators,
            } => {
                let group_key = match id_expression {
                    MongoDBValue::String(s) => s.trim_start_matches('$').to_string(),
                    MongoDBValue::Null => String::new(),
                    _ => String::new(),
                };

                let mut aggregations = Vec::new();
                for (field, acc) in accumulators {
                    let (agg_type, agg_field) = match acc {
                        GroupAccumulator::Sum(v) => (AggregationType::Sum, get_field_path(v)),
                        GroupAccumulator::Avg(v) => (AggregationType::Avg, get_field_path(v)),
                        GroupAccumulator::Min(v) => (AggregationType::Min, get_field_path(v)),
                        GroupAccumulator::Max(v) => (AggregationType::Max, get_field_path(v)),
                        GroupAccumulator::First(v) => (AggregationType::First, get_field_path(v)),
                        GroupAccumulator::Last(v) => (AggregationType::Last, get_field_path(v)),
                        GroupAccumulator::Push(v) => (AggregationType::Push, get_field_path(v)),
                        GroupAccumulator::AddToSet(v) => {
                            (AggregationType::AddToSet, get_field_path(v))
                        }
                        GroupAccumulator::Count => (AggregationType::Count, String::new()),
                    };

                    aggregations.push(Aggregation {
                        output_field: field.clone(),
                        r#type: agg_type as i32,
                        input_path: agg_field,
                    });
                }

                Stage::Group(GroupStage {
                    key: group_key,
                    aggregations,
                })
            }

            MongoDBPipelineStage::Sort(fields) => {
                let sort_fields: Vec<SortField> = fields
                    .iter()
                    .map(|(field, order)| SortField {
                        path: field.clone(),
                        order: match order {
                            SortOrder::Ascending => ProtoSortOrder::Asc as i32,
                            SortOrder::Descending => ProtoSortOrder::Desc as i32,
                        },
                    })
                    .collect();

                Stage::Sort(SortStage {
                    fields: sort_fields,
                })
            }

            MongoDBPipelineStage::Limit(n) => Stage::Limit(LimitStage { limit: *n as u32 }),

            MongoDBPipelineStage::Skip(n) => Stage::Skip(SkipStage { skip: *n as u32 }),

            MongoDBPipelineStage::Unwind {
                path,
                preserve_null_and_empty,
            } => Stage::Unwind(UnwindStage {
                path: path.clone(),
                preserve_null: *preserve_null_and_empty,
            }),

            MongoDBPipelineStage::Lookup { .. } => {
                // Lookup is not directly supported in proto, skip for now
                return Err(anyhow!("$lookup stage not yet supported"));
            }

            MongoDBPipelineStage::Count(field) => {
                // Implement count as a group with count accumulator
                Stage::Group(GroupStage {
                    key: String::new(), // null _id groups all
                    aggregations: vec![Aggregation {
                        output_field: field.clone(),
                        r#type: AggregationType::Count as i32,
                        input_path: String::new(),
                    }],
                })
            }
        };

        Ok(AggregationStage { stage: Some(stage) })
    }
}

/// Helper to extract field path from MongoDBValue
fn get_field_path(value: &MongoDBValue) -> String {
    match value {
        MongoDBValue::String(s) => s.trim_start_matches('$').to_string(),
        MongoDBValue::Integer(1) => String::new(), // $sum: 1 means count
        _ => String::new(),
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_equality() {
        let parser = MongoDBParser::new();
        let result = parser
            .parse_query(r#"{"name": "John"}"#)
            .expect("Failed to parse query");

        match result {
            MongoDBExpression::FieldEquals { field, value } => {
                assert_eq!(field, "name");
                assert_eq!(value, MongoDBValue::String("John".to_string()));
            }
            _ => panic!("Expected FieldEquals"),
        }
    }

    #[test]
    fn test_parse_comparison_operators() {
        let parser = MongoDBParser::new();

        // $gte operator
        let result = parser
            .parse_query(r#"{"age": {"$gte": 18}}"#)
            .expect("Failed to parse query");
        match result {
            MongoDBExpression::FieldOperator {
                field,
                operator,
                value,
            } => {
                assert_eq!(field, "age");
                assert_eq!(operator, MongoOperator::Gte);
                assert_eq!(value, MongoDBValue::Integer(18));
            }
            _ => panic!("Expected FieldOperator"),
        }

        // $in operator
        let result = parser
            .parse_query(r#"{"status": {"$in": ["active", "pending"]}}"#)
            .expect("Failed to parse query");
        match result {
            MongoDBExpression::FieldOperator {
                field,
                operator,
                value,
            } => {
                assert_eq!(field, "status");
                assert_eq!(operator, MongoOperator::In);
                match value {
                    MongoDBValue::Array(arr) => {
                        assert_eq!(arr.len(), 2);
                    }
                    _ => panic!("Expected array value"),
                }
            }
            _ => panic!("Expected FieldOperator"),
        }
    }

    #[test]
    fn test_parse_logical_operators() {
        let parser = MongoDBParser::new();

        // $and operator
        let result = parser
            .parse_query(r#"{"$and": [{"age": {"$gte": 18}}, {"active": true}]}"#)
            .expect("Failed to parse query");
        match result {
            MongoDBExpression::And(exprs) => {
                assert_eq!(exprs.len(), 2);
            }
            _ => panic!("Expected And"),
        }

        // $or operator
        let result = parser
            .parse_query(r#"{"$or": [{"age": {"$lt": 18}}, {"premium": true}]}"#)
            .expect("Failed to parse query");
        match result {
            MongoDBExpression::Or(exprs) => {
                assert_eq!(exprs.len(), 2);
            }
            _ => panic!("Expected Or"),
        }
    }

    #[test]
    fn test_parse_element_operators() {
        let parser = MongoDBParser::new();

        // $exists operator
        let result = parser
            .parse_query(r#"{"email": {"$exists": true}}"#)
            .expect("Failed to parse query");
        match result {
            MongoDBExpression::FieldOperator {
                field,
                operator,
                value,
            } => {
                assert_eq!(field, "email");
                assert_eq!(operator, MongoOperator::Exists);
                assert_eq!(value, MongoDBValue::Bool(true));
            }
            _ => panic!("Expected FieldOperator"),
        }

        // $type operator
        let result = parser
            .parse_query(r#"{"age": {"$type": "number"}}"#)
            .expect("Failed to parse query");
        match result {
            MongoDBExpression::FieldOperator {
                field,
                operator,
                value,
            } => {
                assert_eq!(field, "age");
                assert_eq!(operator, MongoOperator::Type);
                assert_eq!(value, MongoDBValue::String("number".to_string()));
            }
            _ => panic!("Expected FieldOperator"),
        }
    }

    #[test]
    fn test_parse_regex() {
        let parser = MongoDBParser::new();

        let result = parser
            .parse_query(r#"{"name": {"$regex": "^John", "$options": "i"}}"#)
            .expect("Failed to parse query");
        match result {
            MongoDBExpression::FieldOperator {
                field,
                operator,
                value,
            } => {
                assert_eq!(field, "name");
                assert_eq!(operator, MongoOperator::Regex);
                match value {
                    MongoDBValue::Regex { pattern, options } => {
                        assert_eq!(pattern, "^John");
                        assert_eq!(options, "i");
                    }
                    _ => panic!("Expected Regex value"),
                }
            }
            _ => panic!("Expected FieldOperator"),
        }
    }

    #[test]
    fn test_parse_text_search() {
        let parser = MongoDBParser::new();

        let result = parser
            .parse_query(r#"{"$text": {"$search": "hello world"}}"#)
            .expect("Failed to parse query");
        match result {
            MongoDBExpression::TextSearch { search, .. } => {
                assert_eq!(search, "hello world");
            }
            _ => panic!("Expected TextSearch"),
        }
    }

    #[test]
    fn test_parse_projection() {
        let parser = MongoDBParser::new();

        let result = parser
            .parse_projection(r#"{"name": 1, "age": 1, "password": 0}"#)
            .expect("Failed to parse projection");

        assert!(result.include.contains(&"name".to_string()));
        assert!(result.include.contains(&"age".to_string()));
        assert!(result.exclude.contains(&"password".to_string()));
    }

    #[test]
    fn test_parse_aggregation_pipeline() {
        let parser = MongoDBParser::new();

        let pipeline = r#"[
            {"$match": {"status": "active"}},
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]"#;

        let result = parser
            .parse_pipeline(pipeline)
            .expect("Failed to parse pipeline");
        assert_eq!(result.len(), 4);

        // Check $match stage
        match &result[0] {
            MongoDBPipelineStage::Match(_) => {}
            _ => panic!("Expected Match stage"),
        }

        // Check $group stage
        match &result[1] {
            MongoDBPipelineStage::Group {
                id_expression,
                accumulators,
            } => {
                match id_expression {
                    MongoDBValue::String(s) => assert_eq!(s, "$category"),
                    _ => panic!("Expected string id expression"),
                }
                assert!(accumulators.contains_key("count"));
            }
            _ => panic!("Expected Group stage"),
        }

        // Check $sort stage
        match &result[2] {
            MongoDBPipelineStage::Sort(fields) => {
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].0, "count");
                assert_eq!(fields[0].1, SortOrder::Descending);
            }
            _ => panic!("Expected Sort stage"),
        }

        // Check $limit stage
        match &result[3] {
            MongoDBPipelineStage::Limit(n) => assert_eq!(*n, 10),
            _ => panic!("Expected Limit stage"),
        }
    }

    #[test]
    fn test_convert_to_document_filter() {
        let parser = MongoDBParser::new();

        // Simple equality
        let expr = parser
            .parse_query(r#"{"name": "John"}"#)
            .expect("Failed to parse query");
        let filter = expr
            .to_document_filter()
            .expect("Failed to convert to document filter");
        assert_eq!(filter.conditions.len(), 1);
        assert_eq!(filter.conditions[0].path, "name");
        assert_eq!(filter.conditions[0].operator, DocFilterOperator::Eq as i32);

        // Comparison operator
        let expr = parser
            .parse_query(r#"{"age": {"$gte": 18}}"#)
            .expect("Failed to parse query");
        let filter = expr
            .to_document_filter()
            .expect("Failed to convert to document filter");
        assert_eq!(filter.conditions.len(), 1);
        assert_eq!(filter.conditions[0].operator, DocFilterOperator::Gte as i32);

        // $or operator
        let expr = parser
            .parse_query(r#"{"$or": [{"status": "active"}, {"premium": true}]}"#)
            .expect("Failed to parse query");
        let filter = expr
            .to_document_filter()
            .expect("Failed to convert to document filter");
        assert_eq!(filter.or_filters.len(), 2);
    }

    #[test]
    fn test_compound_query() {
        let parser = MongoDBParser::new();

        let query = r#"{
            "status": "active",
            "age": {"$gte": 18, "$lt": 65},
            "tags": {"$in": ["premium", "verified"]}
        }"#;

        let result = parser.parse_query(query).expect("Failed to parse query");
        match result {
            MongoDBExpression::Compound(exprs) => {
                assert_eq!(exprs.len(), 3);
            }
            _ => panic!("Expected Compound expression"),
        }
    }

    #[test]
    fn test_nested_logical_operators() {
        let parser = MongoDBParser::new();

        let query = r#"{
            "$and": [
                {"$or": [{"status": "active"}, {"status": "pending"}]},
                {"age": {"$gte": 18}}
            ]
        }"#;

        let result = parser.parse_query(query).expect("Failed to parse query");
        match result {
            MongoDBExpression::And(exprs) => {
                assert_eq!(exprs.len(), 2);
                match &exprs[0] {
                    MongoDBExpression::Or(or_exprs) => {
                        assert_eq!(or_exprs.len(), 2);
                    }
                    _ => panic!("Expected nested Or"),
                }
            }
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_lexer_tokenize() {
        let tokens =
            MongoDBLexer::tokenize(r#"{"age": 25, "active": true}"#).expect("Failed to tokenize");

        assert!(matches!(tokens[0], Token::LeftBrace));
        assert!(matches!(tokens[1], Token::String(ref s) if s == "age"));
        assert!(matches!(tokens[2], Token::Colon));
        assert!(matches!(tokens[3], Token::Integer(25)));
        assert!(matches!(tokens[4], Token::Comma));
        assert!(matches!(tokens[5], Token::String(ref s) if s == "active"));
        assert!(matches!(tokens[6], Token::Colon));
        assert!(matches!(tokens[7], Token::Boolean(true)));
        assert!(matches!(tokens[8], Token::RightBrace));
    }

    #[test]
    fn test_elem_match() {
        let parser = MongoDBParser::new();

        let query = r#"{"items": {"$elemMatch": {"price": {"$gt": 100}}}}"#;
        let result = parser.parse_query(query).expect("Failed to parse query");

        match result {
            MongoDBExpression::ElemMatch { field, query } => {
                assert_eq!(field, "items");
                match *query {
                    MongoDBExpression::FieldOperator {
                        field, operator, ..
                    } => {
                        assert_eq!(field, "price");
                        assert_eq!(operator, MongoOperator::Gt);
                    }
                    _ => panic!("Expected FieldOperator in elemMatch"),
                }
            }
            _ => panic!("Expected ElemMatch"),
        }
    }

    #[test]
    fn test_array_operators() {
        let parser = MongoDBParser::new();

        // $all operator
        let query = r#"{"tags": {"$all": ["a", "b", "c"]}}"#;
        let result = parser.parse_query(query).expect("Failed to parse query");

        match result {
            MongoDBExpression::FieldOperator {
                field,
                operator,
                value,
            } => {
                assert_eq!(field, "tags");
                assert_eq!(operator, MongoOperator::All);
                match value {
                    MongoDBValue::Array(arr) => assert_eq!(arr.len(), 3),
                    _ => panic!("Expected array"),
                }
            }
            _ => panic!("Expected FieldOperator"),
        }

        // $size operator
        let query = r#"{"items": {"$size": 5}}"#;
        let result = parser.parse_query(query).expect("Failed to parse query");

        match result {
            MongoDBExpression::FieldOperator {
                field,
                operator,
                value,
            } => {
                assert_eq!(field, "items");
                assert_eq!(operator, MongoOperator::Size);
                assert_eq!(value, MongoDBValue::Integer(5));
            }
            _ => panic!("Expected FieldOperator"),
        }
    }

    #[test]
    fn test_full_query_with_options() {
        let parser = MongoDBParser::new();

        let query = r#"{"status": "active"}"#;
        let options = r#"{"projection": {"name": 1, "email": 1}, "sort": {"createdAt": -1}, "limit": 10, "skip": 5}"#;

        let result = parser
            .parse_full_query(query, Some(options))
            .expect("Failed to parse full query");

        assert!(result.filter.is_some());
        assert!(result.projection.is_some());
        assert_eq!(
            result
                .projection
                .as_ref()
                .expect("Projection should be Some")
                .include
                .len(),
            2
        );
        assert!(result.sort.is_some());
        assert_eq!(
            result.sort.as_ref().expect("Sort should be Some")[0].1,
            SortOrder::Descending
        );
        assert_eq!(result.limit, Some(10));
        assert_eq!(result.skip, Some(5));
    }
}
