/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! # Cypher Query Language Parser
//!
//! This module implements a comprehensive Cypher Query Language parser for ProximaDB's
//! graph database. It follows SOLID principles with separate lexer, parser, and AST
//! conversion components.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │              Cypher Query               │
//! │    "MATCH (n:Person) RETURN n"          │
//! └───────────────┬─────────────────────────┘
//!                 │
//! ┌───────────────▼─────────────────────────┐
//! │              Lexer                      │
//! │  • Tokenizes input into tokens          │
//! │  • Handles keywords, identifiers        │
//! │  • Preserves location information       │
//! └───────────────┬─────────────────────────┘
//!                 │
//! ┌───────────────▼─────────────────────────┐
//! │              Parser                     │
//! │  • Parses patterns (nodes, edges)       │
//! │  • Parses clauses (MATCH, WHERE, etc.)  │
//! │  • Parses expressions and functions     │
//! └───────────────┬─────────────────────────┘
//!                 │
//! ┌───────────────▼─────────────────────────┐
//! │           CypherAst                     │
//! │  • Complete AST representation          │
//! │  • Query structure with clauses         │
//! └───────────────┬─────────────────────────┘
//!                 │
//! ┌───────────────▼─────────────────────────┐
//! │        GraphQuery Conversion            │
//! │  • Converts AST to executable query     │
//! │  • Integrates with graph service        │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Supported Cypher Syntax
//!
//! ### Patterns
//! - Nodes: `(n:Label {prop: value})`
//! - Edges: `-[r:TYPE {prop: value}]->`
//! - Variable-length paths: `(a)-[*1..3]->(b)`
//!
//! ### Clauses
//! - MATCH, OPTIONAL MATCH
//! - WHERE with complex conditions
//! - RETURN with aliases
//! - ORDER BY, LIMIT, SKIP
//! - CREATE, MERGE, DELETE, SET
//!
//! ### Expressions
//! - Properties: `n.prop`
//! - Functions: `labels(n)`, `type(r)`, `id(n)`
//! - Aggregations: `count()`, `sum()`, `avg()`, `collect()`

use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, tag_no_case},
    character::complete::{char, digit1, multispace0, multispace1},
    combinator::{map, map_res, opt, value},
    error::VerboseError,
    multi::{many0, many1, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, separated_pair, tuple},
};
use proximadb_kernel::error::ProximaDBError;
// TODO: Move to proximadb-graph crate
// For now, use local definitions
use crate::query_ast::{
    CompiledPattern,
    // TODO: Add other AST types
};
use std::collections::HashMap;

// Shared TD-GRAPH-4 lexeme (string literals for the string predicates).
use proximadb_graph_query::parser::lexemes::string_literal;

// Placeholder types for Cypher AST
pub type EdgeDirection = crate::query_ast::EdgeDirection;
pub type CypherQuery = crate::query_ast::CypherStatement;
pub type MatchPattern = CompiledPattern;
pub type NodePattern = crate::query_ast::PatternNode;
pub type EdgePattern = crate::query_ast::PatternEdge;
pub type ReturnSpec = crate::query_ast::ReturnClause;
pub type WhereClause = crate::query_ast::WhereClause;

// Re-export types from cypher_ast
pub use crate::query_ast::{
    CreateClause, CreateEdgeSpec, CreateNodeSpec, DeleteClause, MergeClause, PropertyConstraint,
    PropertyProjection, RemoveItem, SetClause, SetItem, WithClause,
};
pub use crate::query_ast::{ReadingClause, RemoveClause, UpdatingClause};

type ParseResult<'a, T> = IResult<&'a str, T, VerboseError<&'a str>>;

// ============================================================================
// LEXER - Token Types and Tokenization
// ============================================================================

/// Token types for the Cypher lexer
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// MATCH keyword
    Match,
    /// OPTIONAL MATCH keyword
    OptionalMatch,
    /// WHERE keyword
    Where,
    /// RETURN keyword
    Return,
    /// ORDER BY keyword
    OrderBy,
    /// LIMIT keyword
    Limit,
    /// SKIP keyword
    Skip,
    /// CREATE keyword
    Create,
    /// MERGE keyword
    Merge,
    /// DELETE keyword
    Delete,
    /// DETACH DELETE keyword
    DetachDelete,
    /// SET keyword
    Set,
    /// REMOVE keyword
    Remove,
    /// WITH keyword
    With,
    /// AS keyword
    As,
    /// DISTINCT keyword
    Distinct,
    /// AND logical operator
    And,
    /// OR logical operator
    Or,
    /// NOT logical operator
    Not,
    /// XOR logical operator
    Xor,
    /// IN set membership operator
    In,
    /// IS operator
    Is,
    /// NULL literal
    Null,
    /// TRUE boolean literal
    True,
    /// FALSE boolean literal
    False,
    /// ASC sort direction
    Asc,
    /// DESC sort direction
    Desc,
    /// ON CREATE merge action
    OnCreate,
    /// ON MATCH merge action
    OnMatch,
    /// UNWIND list expansion
    Unwind,
    /// FOREACH iteration
    ForEach,
    /// CALL procedure invocation
    Call,
    /// YIELD for procedure results
    Yield,

    /// Opening parenthesis `(`
    OpenParen,
    /// Closing parenthesis `)`
    CloseParen,
    /// Opening bracket `[`
    OpenBracket,
    /// Closing bracket `]`
    CloseBracket,
    /// Opening brace `{`
    OpenBrace,
    /// Closing brace `}`
    CloseBrace,
    /// Comma separator
    Comma,
    /// Colon separator
    Colon,
    /// Dot property accessor
    Dot,
    /// Semicolon statement separator
    Semicolon,

    /// Right arrow `->`
    Arrow,
    /// Left arrow `<-`
    LeftArrow,
    /// Dash `-`
    Dash,
    /// Equals `=`
    Equals,
    /// Not equals `<>`
    NotEquals,
    /// Less than `<`
    LessThan,
    /// Greater than `>`
    GreaterThan,
    /// Less than or equal `<=`
    LessOrEqual,
    /// Greater than or equal `>=`
    GreaterOrEqual,
    /// Plus `+`
    Plus,
    /// Minus `-`
    Minus,
    /// Asterisk `*`
    Asterisk,
    /// Slash `/`
    Slash,
    /// Percent `%`
    Percent,
    /// Caret `^`
    Caret,
    /// Plus-equals `+=`
    PlusEquals,
    /// Double dot `..` for range
    DoubleDot,

    /// Integer literal
    Integer(i64),
    /// Float literal
    Float(f64),
    /// String literal
    String(String),
    /// Identifier (variable or label name)
    Identifier(String),
    /// Parameter reference
    Parameter(String),

    /// Whitespace token
    Whitespace,
    /// Comment text
    Comment(String),
    /// End of file
    Eof,
}

/// Token with location information
#[derive(Debug, Clone)]
pub struct LocatedToken {
    /// The token value
    pub token: Token,
    /// Line number in the source (1-based)
    pub line: usize,
    /// Column number in the source (1-based)
    pub column: usize,
    /// Byte range in the source string
    pub span: std::ops::Range<usize>,
}

/// Lexer for tokenizing Cypher queries
pub struct CypherLexer {
    /// Current position in input
    position: usize,
    /// Current line number
    line: usize,
    /// Current column number
    column: usize,
}

impl CypherLexer {
    /// Create a new lexer
    pub fn new() -> Self {
        Self {
            position: 0,
            line: 1,
            column: 1,
        }
    }

    /// Tokenize input into a vector of located tokens
    pub fn tokenize(&mut self, input: &str) -> Result<Vec<LocatedToken>, ProximaDBError> {
        let mut tokens = Vec::new();
        self.position = 0;
        self.line = 1;
        self.column = 1;

        let chars: Vec<char> = input.chars().collect();

        while self.position < chars.len() {
            let start_pos = self.position;
            let start_line = self.line;
            let start_col = self.column;

            let token = self.next_token(&chars)?;

            if !matches!(token, Token::Whitespace) {
                tokens.push(LocatedToken {
                    token,
                    line: start_line,
                    column: start_col,
                    span: start_pos..self.position,
                });
            }
        }

        tokens.push(LocatedToken {
            token: Token::Eof,
            line: self.line,
            column: self.column,
            span: self.position..self.position,
        });

        Ok(tokens)
    }

    fn next_token(&mut self, chars: &[char]) -> Result<Token, ProximaDBError> {
        let c = chars[self.position];

        // Whitespace
        if c.is_whitespace() {
            self.consume_whitespace(chars);
            return Ok(Token::Whitespace);
        }

        // Comments
        if c == '/' && self.position + 1 < chars.len() && chars[self.position + 1] == '/' {
            let comment = self.consume_line_comment(chars);
            return Ok(Token::Comment(comment));
        }

        // String literals
        if c == '"' || c == '\'' {
            return self.consume_string(chars, c);
        }

        // Numbers
        if c.is_ascii_digit()
            || (c == '-'
                && self.position + 1 < chars.len()
                && chars[self.position + 1].is_ascii_digit())
        {
            return self.consume_number(chars);
        }

        // Identifiers and keywords
        if c.is_alphabetic() || c == '_' || c == '`' {
            return self.consume_identifier_or_keyword(chars);
        }

        // Parameters
        if c == '$' {
            return self.consume_parameter(chars);
        }

        // Operators and delimiters
        self.consume_operator(chars)
    }

    fn consume_whitespace(&mut self, chars: &[char]) {
        while self.position < chars.len() && chars[self.position].is_whitespace() {
            if chars[self.position] == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            self.position += 1;
        }
    }

    fn consume_line_comment(&mut self, chars: &[char]) -> String {
        let mut comment = String::new();
        self.position += 2; // Skip //
        self.column += 2;

        while self.position < chars.len() && chars[self.position] != '\n' {
            comment.push(chars[self.position]);
            self.position += 1;
            self.column += 1;
        }

        comment
    }

    fn consume_string(&mut self, chars: &[char], quote: char) -> Result<Token, ProximaDBError> {
        let mut s = String::new();
        self.position += 1; // Skip opening quote
        self.column += 1;

        while self.position < chars.len() {
            let c = chars[self.position];

            if c == quote {
                self.position += 1;
                self.column += 1;
                return Ok(Token::String(s));
            }

            if c == '\\' && self.position + 1 < chars.len() {
                self.position += 1;
                self.column += 1;
                let escaped = match chars[self.position] {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '"' => '"',
                    '\'' => '\'',
                    _ => chars[self.position],
                };
                s.push(escaped);
            } else {
                s.push(c);
            }

            self.position += 1;
            self.column += 1;
        }

        Err(ProximaDBError::InvalidInput(
            "Unterminated string literal".to_string(),
        ))
    }

    fn consume_number(&mut self, chars: &[char]) -> Result<Token, ProximaDBError> {
        let mut num_str = String::new();
        let mut is_float = false;

        // Handle negative sign
        if chars[self.position] == '-' {
            num_str.push('-');
            self.position += 1;
            self.column += 1;
        }

        // Integer part
        while self.position < chars.len() && chars[self.position].is_ascii_digit() {
            num_str.push(chars[self.position]);
            self.position += 1;
            self.column += 1;
        }

        // Decimal part
        if self.position < chars.len()
            && chars[self.position] == '.'
            && self.position + 1 < chars.len()
            && chars[self.position + 1].is_ascii_digit()
        {
            is_float = true;
            num_str.push('.');
            self.position += 1;
            self.column += 1;

            while self.position < chars.len() && chars[self.position].is_ascii_digit() {
                num_str.push(chars[self.position]);
                self.position += 1;
                self.column += 1;
            }
        }

        // Exponent part
        if self.position < chars.len()
            && (chars[self.position] == 'e' || chars[self.position] == 'E')
        {
            is_float = true;
            num_str.push('e');
            self.position += 1;
            self.column += 1;

            if self.position < chars.len()
                && (chars[self.position] == '+' || chars[self.position] == '-')
            {
                num_str.push(chars[self.position]);
                self.position += 1;
                self.column += 1;
            }

            while self.position < chars.len() && chars[self.position].is_ascii_digit() {
                num_str.push(chars[self.position]);
                self.position += 1;
                self.column += 1;
            }
        }

        if is_float {
            let f: f64 = num_str
                .parse()
                .map_err(|_| ProximaDBError::InvalidInput(format!("Invalid float: {}", num_str)))?;
            Ok(Token::Float(f))
        } else {
            let i: i64 = num_str.parse().map_err(|_| {
                ProximaDBError::InvalidInput(format!("Invalid integer: {}", num_str))
            })?;
            Ok(Token::Integer(i))
        }
    }

    fn consume_identifier_or_keyword(&mut self, chars: &[char]) -> Result<Token, ProximaDBError> {
        let mut ident = String::new();
        let is_quoted = chars[self.position] == '`';

        if is_quoted {
            self.position += 1;
            self.column += 1;

            while self.position < chars.len() && chars[self.position] != '`' {
                ident.push(chars[self.position]);
                self.position += 1;
                self.column += 1;
            }

            if self.position < chars.len() {
                self.position += 1; // Skip closing backtick
                self.column += 1;
            }

            return Ok(Token::Identifier(ident));
        }

        while self.position < chars.len()
            && (chars[self.position].is_alphanumeric() || chars[self.position] == '_')
        {
            ident.push(chars[self.position]);
            self.position += 1;
            self.column += 1;
        }

        // Check for keywords (case-insensitive)
        let token = match ident.to_uppercase().as_str() {
            "MATCH" => Token::Match,
            "OPTIONAL" => {
                // Check for OPTIONAL MATCH
                let saved_pos = self.position;
                let saved_col = self.column;
                self.consume_whitespace(chars);
                let _next_start = self.position;
                let mut next_word = String::new();
                while self.position < chars.len() && chars[self.position].is_alphabetic() {
                    next_word.push(chars[self.position]);
                    self.position += 1;
                    self.column += 1;
                }
                if next_word.to_uppercase() == "MATCH" {
                    Token::OptionalMatch
                } else {
                    self.position = saved_pos;
                    self.column = saved_col;
                    Token::Identifier(ident)
                }
            }
            "WHERE" => Token::Where,
            "RETURN" => Token::Return,
            "ORDER" => Token::OrderBy,
            "BY" => Token::Identifier(ident), // BY is handled with ORDER
            "LIMIT" => Token::Limit,
            "SKIP" => Token::Skip,
            "CREATE" => Token::Create,
            "MERGE" => Token::Merge,
            "DELETE" => Token::Delete,
            "DETACH" => {
                // Check for DETACH DELETE
                let saved_pos = self.position;
                let saved_col = self.column;
                self.consume_whitespace(chars);
                let mut next_word = String::new();
                while self.position < chars.len() && chars[self.position].is_alphabetic() {
                    next_word.push(chars[self.position]);
                    self.position += 1;
                    self.column += 1;
                }
                if next_word.to_uppercase() == "DELETE" {
                    Token::DetachDelete
                } else {
                    self.position = saved_pos;
                    self.column = saved_col;
                    Token::Identifier(ident)
                }
            }
            "SET" => Token::Set,
            "REMOVE" => Token::Remove,
            "WITH" => Token::With,
            "AS" => Token::As,
            "DISTINCT" => Token::Distinct,
            "AND" => Token::And,
            "OR" => Token::Or,
            "NOT" => Token::Not,
            "XOR" => Token::Xor,
            "IN" => Token::In,
            "IS" => Token::Is,
            "NULL" => Token::Null,
            "TRUE" => Token::True,
            "FALSE" => Token::False,
            "ASC" | "ASCENDING" => Token::Asc,
            "DESC" | "DESCENDING" => Token::Desc,
            "ON" => Token::Identifier(ident), // ON CREATE / ON MATCH handled separately
            "UNWIND" => Token::Unwind,
            "FOREACH" => Token::ForEach,
            "CALL" => Token::Call,
            "YIELD" => Token::Yield,
            _ => Token::Identifier(ident),
        };

        Ok(token)
    }

    fn consume_parameter(&mut self, chars: &[char]) -> Result<Token, ProximaDBError> {
        self.position += 1; // Skip $
        self.column += 1;

        let mut name = String::new();
        while self.position < chars.len()
            && (chars[self.position].is_alphanumeric() || chars[self.position] == '_')
        {
            name.push(chars[self.position]);
            self.position += 1;
            self.column += 1;
        }

        Ok(Token::Parameter(name))
    }

    fn consume_operator(&mut self, chars: &[char]) -> Result<Token, ProximaDBError> {
        let c = chars[self.position];
        let next = if self.position + 1 < chars.len() {
            Some(chars[self.position + 1])
        } else {
            None
        };

        let token = match (c, next) {
            ('-', Some('>')) => {
                self.position += 2;
                self.column += 2;
                Token::Arrow
            }
            ('<', Some('-')) => {
                self.position += 2;
                self.column += 2;
                Token::LeftArrow
            }
            ('<', Some('>')) => {
                self.position += 2;
                self.column += 2;
                Token::NotEquals
            }
            ('<', Some('=')) => {
                self.position += 2;
                self.column += 2;
                Token::LessOrEqual
            }
            ('>', Some('=')) => {
                self.position += 2;
                self.column += 2;
                Token::GreaterOrEqual
            }
            ('+', Some('=')) => {
                self.position += 2;
                self.column += 2;
                Token::PlusEquals
            }
            ('.', Some('.')) => {
                self.position += 2;
                self.column += 2;
                Token::DoubleDot
            }
            ('(', _) => {
                self.position += 1;
                self.column += 1;
                Token::OpenParen
            }
            (')', _) => {
                self.position += 1;
                self.column += 1;
                Token::CloseParen
            }
            ('[', _) => {
                self.position += 1;
                self.column += 1;
                Token::OpenBracket
            }
            (']', _) => {
                self.position += 1;
                self.column += 1;
                Token::CloseBracket
            }
            ('{', _) => {
                self.position += 1;
                self.column += 1;
                Token::OpenBrace
            }
            ('}', _) => {
                self.position += 1;
                self.column += 1;
                Token::CloseBrace
            }
            (',', _) => {
                self.position += 1;
                self.column += 1;
                Token::Comma
            }
            (':', _) => {
                self.position += 1;
                self.column += 1;
                Token::Colon
            }
            ('.', _) => {
                self.position += 1;
                self.column += 1;
                Token::Dot
            }
            (';', _) => {
                self.position += 1;
                self.column += 1;
                Token::Semicolon
            }
            ('-', _) => {
                self.position += 1;
                self.column += 1;
                Token::Dash
            }
            ('=', _) => {
                self.position += 1;
                self.column += 1;
                Token::Equals
            }
            ('<', _) => {
                self.position += 1;
                self.column += 1;
                Token::LessThan
            }
            ('>', _) => {
                self.position += 1;
                self.column += 1;
                Token::GreaterThan
            }
            ('+', _) => {
                self.position += 1;
                self.column += 1;
                Token::Plus
            }
            ('*', _) => {
                self.position += 1;
                self.column += 1;
                Token::Asterisk
            }
            ('/', _) => {
                self.position += 1;
                self.column += 1;
                Token::Slash
            }
            ('%', _) => {
                self.position += 1;
                self.column += 1;
                Token::Percent
            }
            ('^', _) => {
                self.position += 1;
                self.column += 1;
                Token::Caret
            }
            _ => {
                return Err(ProximaDBError::InvalidInput(format!(
                    "Unexpected character: {}",
                    c
                )));
            }
        };

        Ok(token)
    }
}

impl Default for CypherLexer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PARSER - Pattern, Clause, and Expression Parsing
// ============================================================================

/// Main Cypher parser following SOLID principles
pub struct CypherParser;

impl CypherParser {
    /// Create a new parser instance
    pub fn new() -> Self {
        Self
    }

    /// Parse a Cypher query string into a CypherQuery AST
    pub fn parse(&self, input: &str) -> Result<CypherQuery, ProximaDBError> {
        match parse_cypher_query(input.trim()) {
            Ok((remaining, query)) => {
                if remaining.trim().is_empty() {
                    Ok(query)
                } else {
                    Err(ProximaDBError::InvalidInput(format!(
                        "Unexpected input remaining after parsing: '{}'",
                        remaining.chars().take(50).collect::<String>()
                    )))
                }
            }
            Err(e) => Err(ProximaDBError::InvalidInput(format!(
                "Failed to parse Cypher query: {:?}",
                e
            ))),
        }
    }

    /// Parse into CompiledPattern for backward compatibility
    pub fn parse_to_compiled_pattern(
        &self,
        input: &str,
    ) -> Result<CompiledPattern, ProximaDBError> {
        match parse_simple_query(input.trim()) {
            Ok((_, compiled)) => Ok(compiled),
            Err(e) => Err(ProximaDBError::InvalidInput(format!(
                "Failed to parse query pattern: {:?}",
                e
            ))),
        }
    }
}

impl Default for CypherParser {
    fn default() -> Self {
        Self::new()
    }
}

impl super::QueryParser for CypherParser {
    type Output = CypherQuery;

    fn parse(&self, input: &str) -> anyhow::Result<Self::Output> {
        CypherParser::parse(self, input).map_err(|e| anyhow::anyhow!("{}", e))
    }
}

// ============================================================================
// NOM PARSER COMBINATORS
// ============================================================================

/// Parse any identifier (bare or backtick-quoted) — the shared
/// TD-GRAPH-4 lexeme, instantiated with this parser's error type.
fn any_identifier(input: &str) -> ParseResult<'_, String> {
    proximadb_graph_query::parser::lexemes::any_identifier(input)
}

/// Parse a property value (string, float, integer, boolean, null, or
/// array) — the shared TD-GRAPH-4 lexeme.
fn property_value(input: &str) -> ParseResult<'_, serde_json::Value> {
    proximadb_graph_query::parser::lexemes::property_value(input)
}

/// Parse property assignment: key: value
fn property_assignment(input: &str) -> ParseResult<'_, (String, PropertyConstraint)> {
    map(
        separated_pair(
            any_identifier,
            delimited(multispace0, char(':'), multispace0),
            property_value,
        ),
        |(key, value)| (key, PropertyConstraint::Equals(value)),
    )(input)
}

/// Parse property map: {key: value, ...}
fn property_map(input: &str) -> ParseResult<'_, HashMap<String, PropertyConstraint>> {
    map(
        delimited(
            pair(char('{'), multispace0),
            separated_list0(
                delimited(multispace0, char(','), multispace0),
                property_assignment,
            ),
            pair(multispace0, char('}')),
        ),
        |pairs| pairs.into_iter().collect(),
    )(input)
}

/// Parse property value map: {key: value, ...} for CREATE
fn property_value_map(input: &str) -> ParseResult<'_, HashMap<String, serde_json::Value>> {
    proximadb_graph_query::parser::lexemes::property_value_map(input)
}

/// Parse labels: :Label1:Label2
fn parse_labels(input: &str) -> ParseResult<'_, Vec<String>> {
    many1(preceded(char(':'), any_identifier))(input)
}

/// Parse optional labels
fn parse_optional_labels(input: &str) -> ParseResult<'_, Vec<String>> {
    map(opt(parse_labels), |labels| labels.unwrap_or_default())(input)
}

/// Parse node pattern: (variable:Label {props})
fn parse_node_pattern(input: &str) -> ParseResult<'_, NodePattern> {
    map(
        delimited(
            char('('),
            tuple((
                preceded(multispace0, opt(any_identifier)),
                parse_optional_labels,
                preceded(multispace0, opt(property_map)),
                multispace0,
            )),
            char(')'),
        ),
        |(variable_opt, labels, properties_opt, _)| {
            let variable = variable_opt.unwrap_or_else(|| "_anon".to_string());
            let label = labels.first().cloned();
            // Convert PropertyConstraint to Value
            let properties: std::collections::HashMap<String, serde_json::Value> = properties_opt
                .unwrap_or_default()
                .into_iter()
                .map(|(k, v)| (k, property_constraint_to_value(v)))
                .collect();
            NodePattern {
                id: format!("{}_{}", variable, uuid::Uuid::new_v4()),
                variable,
                labels,
                label,
                properties,
                optional: false,
            }
        },
    )(input)
}

/// Convert PropertyConstraint to serde_json::Value
fn property_constraint_to_value(constraint: PropertyConstraint) -> serde_json::Value {
    match constraint {
        PropertyConstraint::Equals(v) => v,
        PropertyConstraint::NotEquals(v) => v,
        PropertyConstraint::GreaterThan(v) => v,
        PropertyConstraint::GreaterOrEqual(v) => v,
        PropertyConstraint::LessThan(v) => v,
        PropertyConstraint::LessOrEqual(v) => v,
        PropertyConstraint::In(vals) => serde_json::Value::Array(vals),
        PropertyConstraint::Contains(s) => serde_json::Value::String(s),
        PropertyConstraint::StartsWith(s) => serde_json::Value::String(s),
        PropertyConstraint::EndsWith(s) => serde_json::Value::String(s),
        PropertyConstraint::Regex(s) => serde_json::Value::String(s),
        PropertyConstraint::NotExists => serde_json::Value::Null,
        PropertyConstraint::Exists => serde_json::Value::Bool(true),
    }
}

/// Parse edge direction left: <- or -
fn parse_edge_left(input: &str) -> ParseResult<'_, bool> {
    alt((value(true, tag("<-")), value(false, tag("-"))))(input)
}

/// Parse edge direction right: -> or -
fn parse_edge_right(input: &str) -> ParseResult<'_, bool> {
    alt((value(true, tag("->")), value(false, tag("-"))))(input)
}

/// Parse variable-length path: *min..max or *max or *
fn parse_var_length(input: &str) -> ParseResult<'_, (u32, u32)> {
    preceded(
        char('*'),
        alt((
            // *min..max
            map(
                separated_pair(
                    map_res(digit1, |s: &str| s.parse::<u32>()),
                    tag(".."),
                    map_res(digit1, |s: &str| s.parse::<u32>()),
                ),
                |(min, max)| (min, max),
            ),
            // *n (exactly n or 1..n)
            map(map_res(digit1, |s: &str| s.parse::<u32>()), |n| (1, n)),
            // * (default: 1..infinity, we use 99 as max)
            value((1, 99), multispace0),
        )),
    )(input)
}

/// Parse edge type: :TYPE1|TYPE2
fn parse_edge_types(input: &str) -> ParseResult<'_, Vec<String>> {
    preceded(char(':'), separated_list1(char('|'), any_identifier))(input)
}

/// Parse edge specification: [variable:TYPE {props}]
#[allow(clippy::type_complexity)] // The labelled return-tuple shape IS the
// documentation here — flattening to a type alias buries the field labels.
fn parse_edge_spec(
    input: &str,
) -> ParseResult<
    '_,
    (
        Option<String>,                      // variable
        Vec<String>,                         // edge types
        HashMap<String, PropertyConstraint>, // properties
        Option<(u32, u32)>,                  // variable length
    ),
> {
    map(
        delimited(
            char('['),
            tuple((
                preceded(multispace0, opt(any_identifier)),
                opt(parse_edge_types),
                preceded(multispace0, opt(parse_var_length)),
                preceded(multispace0, opt(property_map)),
                multispace0,
            )),
            char(']'),
        ),
        |(var, types, var_len, props, _)| {
            (
                var,
                types.unwrap_or_default(),
                props.unwrap_or_default(),
                var_len,
            )
        },
    )(input)
}

/// Parse full edge pattern: <-[spec]- or -[spec]-> or -[spec]-
#[allow(clippy::type_complexity)] // Labelled tuple documents the parser shape.
fn parse_edge_pattern_full(
    input: &str,
) -> ParseResult<
    '_,
    (
        bool,                                // is incoming
        Option<String>,                      // variable
        Vec<String>,                         // edge types
        HashMap<String, PropertyConstraint>, // properties
        bool,                                // is outgoing
        Option<(u32, u32)>,                  // variable length
    ),
> {
    map(
        tuple((parse_edge_left, parse_edge_spec, parse_edge_right)),
        |(left, (var, types, props, var_len), right)| (left, var, types, props, right, var_len),
    )(input)
}

/// Parse a path segment: (node)-[edge]->(node)
fn parse_path_segment(input: &str) -> ParseResult<'_, (NodePattern, EdgePattern, NodePattern)> {
    map(
        tuple((
            parse_node_pattern,
            preceded(multispace0, parse_edge_pattern_full),
            preceded(multispace0, parse_node_pattern),
        )),
        |(
            from_node,
            (is_incoming, edge_var, edge_types, edge_props, is_outgoing, _var_len),
            to_node,
        )| {
            let direction = match (is_incoming, is_outgoing) {
                (false, true) => EdgeDirection::Outgoing,
                (true, false) => EdgeDirection::Incoming,
                _ => EdgeDirection::Bidirectional,
            };

            let edge = EdgePattern {
                variable: edge_var,
                from_variable: from_node.variable.clone(),
                to_variable: to_node.variable.clone(),
                from: from_node.id.clone(),
                to: to_node.id.clone(),
                label: edge_types.first().cloned(),
                edge_types,
                direction,
                properties: edge_props
                    .into_iter()
                    .map(|(k, v)| (k, property_constraint_to_value(v)))
                    .collect(),
                optional: false,
            };

            (from_node, edge, to_node)
        },
    )(input)
}

/// Parse comparison operator
fn parse_comparison_op(input: &str) -> ParseResult<'_, &str> {
    alt((
        tag(">="),
        tag("<="),
        tag("<>"),
        tag("!="),
        tag("=~"),
        tag("="),
        tag("<"),
        tag(">"),
    ))(input)
}

/// Parse a WHERE property condition: variable.property op value
fn parse_where_property(input: &str) -> ParseResult<'_, WhereClause> {
    map(
        tuple((
            any_identifier,
            char('.'),
            any_identifier,
            delimited(multispace0, parse_comparison_op, multispace0),
            property_value,
        )),
        |(variable, _, property, op, value)| WhereClause::Property {
            variable,
            property,
            constraint: match op {
                "=" => PropertyConstraint::Equals(value),
                "<>" | "!=" => PropertyConstraint::NotEquals(value),
                ">" => PropertyConstraint::GreaterThan(value),
                ">=" => PropertyConstraint::GreaterOrEqual(value),
                "<" => PropertyConstraint::LessThan(value),
                "<=" => PropertyConstraint::LessOrEqual(value),
                "=~" => {
                    if let serde_json::Value::String(s) = value {
                        PropertyConstraint::Regex(s)
                    } else {
                        PropertyConstraint::Equals(value)
                    }
                }
                _ => PropertyConstraint::Equals(value),
            },
        },
    )(input)
}

/// Parse IS NULL / IS NOT NULL
fn parse_is_null(input: &str) -> ParseResult<'_, WhereClause> {
    alt((
        map(
            tuple((
                any_identifier,
                char('.'),
                any_identifier,
                delimited(multispace1, tag_no_case("IS"), multispace1),
                tag_no_case("NULL"),
            )),
            |(variable, _, property, _, _)| WhereClause::Property {
                variable,
                property,
                constraint: PropertyConstraint::NotExists,
            },
        ),
        map(
            tuple((
                any_identifier,
                char('.'),
                any_identifier,
                delimited(multispace1, tag_no_case("IS"), multispace1),
                tag_no_case("NOT"),
                multispace1,
                tag_no_case("NULL"),
            )),
            |(variable, _, property, _, _, _, _)| WhereClause::Property {
                variable,
                property,
                constraint: PropertyConstraint::Exists,
            },
        ),
    ))(input)
}

/// Parse IN clause
fn parse_in_clause(input: &str) -> ParseResult<'_, WhereClause> {
    map(
        tuple((
            any_identifier,
            char('.'),
            any_identifier,
            delimited(multispace1, tag_no_case("IN"), multispace1),
            delimited(
                char('['),
                separated_list0(
                    delimited(multispace0, char(','), multispace0),
                    property_value,
                ),
                char(']'),
            ),
        )),
        |(variable, _, property, _, values)| WhereClause::Property {
            variable,
            property,
            constraint: PropertyConstraint::In(values),
        },
    )(input)
}

/// Parse STARTS WITH / ENDS WITH / CONTAINS
fn parse_string_predicates(input: &str) -> ParseResult<'_, WhereClause> {
    alt((
        map(
            tuple((
                any_identifier,
                char('.'),
                any_identifier,
                delimited(multispace1, tag_no_case("STARTS"), multispace1),
                tag_no_case("WITH"),
                multispace1,
                string_literal,
            )),
            |(variable, _, property, _, _, _, value)| WhereClause::Property {
                variable,
                property,
                constraint: PropertyConstraint::StartsWith(value),
            },
        ),
        map(
            tuple((
                any_identifier,
                char('.'),
                any_identifier,
                delimited(multispace1, tag_no_case("ENDS"), multispace1),
                tag_no_case("WITH"),
                multispace1,
                string_literal,
            )),
            |(variable, _, property, _, _, _, value)| WhereClause::Property {
                variable,
                property,
                constraint: PropertyConstraint::EndsWith(value),
            },
        ),
        map(
            tuple((
                any_identifier,
                char('.'),
                any_identifier,
                delimited(multispace1, tag_no_case("CONTAINS"), multispace1),
                string_literal,
            )),
            |(variable, _, property, _, value)| WhereClause::Property {
                variable,
                property,
                constraint: PropertyConstraint::Contains(value),
            },
        ),
    ))(input)
}

/// Parse primary WHERE condition
fn parse_where_primary(input: &str) -> ParseResult<'_, WhereClause> {
    alt((
        // NOT condition
        map(
            preceded(pair(tag_no_case("NOT"), multispace1), parse_where_primary),
            |cond| WhereClause::Not(Box::new(cond)),
        ),
        // Parenthesized condition
        delimited(
            pair(char('('), multispace0),
            parse_where_condition,
            pair(multispace0, char(')')),
        ),
        // IS NULL / IS NOT NULL
        parse_is_null,
        // IN clause
        parse_in_clause,
        // String predicates
        parse_string_predicates,
        // Property comparison
        parse_where_property,
    ))(input)
}

/// Parse WHERE condition with AND/OR
fn parse_where_condition(input: &str) -> ParseResult<'_, WhereClause> {
    let (input, first) = parse_where_primary(input)?;

    // Try to parse chained AND/OR
    let result = many0(tuple((
        delimited(
            multispace1,
            alt((tag_no_case("AND"), tag_no_case("OR"), tag_no_case("XOR"))),
            multispace1,
        ),
        parse_where_primary,
    )))(input);

    match result {
        Ok((remaining, pairs)) => {
            let mut current = first;
            for (op, right) in pairs {
                current = match op.to_uppercase().as_str() {
                    "AND" => WhereClause::And(Box::new(current), Box::new(right)),
                    "OR" => WhereClause::Or(Box::new(current), Box::new(right)),
                    "XOR" => {
                        // XOR = (A OR B) AND NOT (A AND B)
                        let a_or_b =
                            WhereClause::Or(Box::new(current.clone()), Box::new(right.clone()));
                        let a_and_b = WhereClause::And(Box::new(current), Box::new(right));
                        WhereClause::And(
                            Box::new(a_or_b),
                            Box::new(WhereClause::Not(Box::new(a_and_b))),
                        )
                    }
                    _ => unreachable!(),
                };
            }
            Ok((remaining, current))
        }
        Err(e) => Err(e),
    }
}

/// Parse WHERE clause
fn parse_where_clause(input: &str) -> ParseResult<'_, WhereClause> {
    preceded(
        pair(tag_no_case("WHERE"), multispace1),
        parse_where_condition,
    )(input)
}

/// Parse aggregation function
fn parse_aggregation(input: &str) -> ParseResult<'_, PropertyProjection> {
    alt((
        // COUNT(*)
        map(
            tuple((
                tag_no_case("COUNT"),
                multispace0,
                char('('),
                multispace0,
                char('*'),
                multispace0,
                char(')'),
            )),
            |_| PropertyProjection::Count,
        ),
        // COUNT(DISTINCT expr) - we simplify to count
        map(
            tuple((
                tag_no_case("COUNT"),
                multispace0,
                char('('),
                multispace0,
                tag_no_case("DISTINCT"),
                multispace1,
                any_identifier,
                multispace0,
                char(')'),
            )),
            |_| PropertyProjection::Count,
        ),
        // SUM(variable.property)
        map(
            tuple((
                tag_no_case("SUM"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                char('.'),
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, prop, _, _)| PropertyProjection::Sum {
                variable: var,
                property: prop,
            },
        ),
        // AVG(variable.property)
        map(
            tuple((
                tag_no_case("AVG"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                char('.'),
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, prop, _, _)| PropertyProjection::Avg {
                variable: var,
                property: prop,
            },
        ),
        // MIN(variable.property)
        map(
            tuple((
                tag_no_case("MIN"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                char('.'),
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, prop, _, _)| PropertyProjection::Min {
                variable: var,
                property: prop,
            },
        ),
        // MAX(variable.property)
        map(
            tuple((
                tag_no_case("MAX"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                char('.'),
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, prop, _, _)| PropertyProjection::Max {
                variable: var,
                property: prop,
            },
        ),
    ))(input)
}

/// Parse function call (labels, type, id, etc.)
#[allow(dead_code)]
fn parse_function_call(input: &str) -> ParseResult<'_, CypherFunction> {
    alt((
        map(
            tuple((
                tag_no_case("labels"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, _)| CypherFunction::Labels(var),
        ),
        map(
            tuple((
                tag_no_case("type"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, _)| CypherFunction::Type(var),
        ),
        map(
            tuple((
                tag_no_case("id"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, _)| CypherFunction::Id(var),
        ),
        map(
            tuple((
                tag_no_case("properties"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, _)| CypherFunction::Properties(var),
        ),
        map(
            tuple((
                tag_no_case("keys"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, _)| CypherFunction::Keys(var),
        ),
        map(
            tuple((
                tag_no_case("exists"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                char('.'),
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, prop, _, _)| CypherFunction::Exists(var, prop),
        ),
        map(
            tuple((
                tag_no_case("collect"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, _)| CypherFunction::Collect(var),
        ),
        map(
            tuple((
                tag_no_case("collect"),
                multispace0,
                char('('),
                multispace0,
                any_identifier,
                char('.'),
                any_identifier,
                multispace0,
                char(')'),
            )),
            |(_, _, _, _, var, _, prop, _, _)| CypherFunction::CollectProperty(var, prop),
        ),
    ))(input)
}

/// Cypher function types
#[derive(Debug, Clone, PartialEq)]
pub enum CypherFunction {
    /// Get labels of a node
    Labels(String),
    /// Get type of a relationship
    Type(String),
    /// Get internal ID of a node or relationship
    Id(String),
    /// Get all properties as a map
    Properties(String),
    /// Get property keys of a node or relationship
    Keys(String),
    /// Check if a property exists on an element
    Exists(String, String),
    /// Collect values into a list
    Collect(String),
    /// Collect property values into a list
    CollectProperty(String, String),
}

/// Parse return item: variable, variable.property, aggregation, or function
fn parse_return_item(input: &str) -> ParseResult<'_, (String, PropertyProjection)> {
    alt((
        // Aggregation with alias
        map(
            tuple((
                parse_aggregation,
                opt(preceded(
                    delimited(multispace1, tag_no_case("AS"), multispace1),
                    any_identifier,
                )),
            )),
            |(agg, alias)| {
                let name = alias.unwrap_or_else(|| match &agg {
                    PropertyProjection::Count => "count".to_string(),
                    PropertyProjection::Sum { property, .. } => format!("sum_{}", property),
                    PropertyProjection::Avg { property, .. } => format!("avg_{}", property),
                    PropertyProjection::Min { property, .. } => format!("min_{}", property),
                    PropertyProjection::Max { property, .. } => format!("max_{}", property),
                    _ => "result".to_string(),
                });
                (name, agg)
            },
        ),
        // Property with alias: n.name AS person_name
        map(
            tuple((
                any_identifier,
                char('.'),
                any_identifier,
                opt(preceded(
                    delimited(multispace1, tag_no_case("AS"), multispace1),
                    any_identifier,
                )),
            )),
            |(var, _, prop, alias)| {
                let name = alias.unwrap_or_else(|| format!("{}.{}", var, prop));
                (
                    name,
                    PropertyProjection::Property {
                        variable: var,
                        property: prop,
                    },
                )
            },
        ),
        // Simple variable with alias
        map(
            tuple((
                any_identifier,
                opt(preceded(
                    delimited(multispace1, tag_no_case("AS"), multispace1),
                    any_identifier,
                )),
            )),
            |(var, alias)| {
                let name = alias.unwrap_or_else(|| var.clone());
                (name, PropertyProjection::Variable(var))
            },
        ),
    ))(input)
}

/// Parse ORDER BY clause
fn parse_order_by(input: &str) -> ParseResult<'_, Vec<(String, bool)>> {
    preceded(
        tuple((
            tag_no_case("ORDER"),
            multispace1,
            tag_no_case("BY"),
            multispace1,
        )),
        separated_list1(
            delimited(multispace0, char(','), multispace0),
            map(
                tuple((
                    any_identifier,
                    opt(preceded(
                        multispace1,
                        alt((
                            value(true, alt((tag_no_case("ASC"), tag_no_case("ASCENDING")))),
                            value(false, alt((tag_no_case("DESC"), tag_no_case("DESCENDING")))),
                        )),
                    )),
                )),
                |(var, asc)| (var, asc.unwrap_or(true)),
            ),
        ),
    )(input)
}

/// Parse LIMIT clause
fn parse_limit(input: &str) -> ParseResult<'_, usize> {
    preceded(
        pair(tag_no_case("LIMIT"), multispace1),
        map_res(digit1, |s: &str| s.parse::<usize>()),
    )(input)
}

/// Parse SKIP clause
fn parse_skip(input: &str) -> ParseResult<'_, usize> {
    preceded(
        pair(tag_no_case("SKIP"), multispace1),
        map_res(digit1, |s: &str| s.parse::<usize>()),
    )(input)
}

/// Parse RETURN clause
fn parse_return_clause(input: &str) -> ParseResult<'_, ReturnSpec> {
    map(
        tuple((
            tag_no_case("RETURN"),
            opt(preceded(multispace1, tag_no_case("DISTINCT"))),
            multispace1,
            separated_list1(
                delimited(multispace0, char(','), multispace0),
                parse_return_item,
            ),
            opt(preceded(multispace1, parse_order_by)),
            opt(preceded(multispace1, parse_skip)),
            opt(preceded(multispace1, parse_limit)),
        )),
        |(_, distinct, _, items, order_by, skip, limit)| {
            let (vars, projections): (Vec<_>, Vec<_>) = items.into_iter().unzip();
            let order_by_strs: Vec<String> = order_by
                .unwrap_or_default()
                .into_iter()
                .map(|(s, asc)| format!("{}{}", s, if asc { " ASC" } else { " DESC" }))
                .collect();
            ReturnSpec {
                items: vars.clone(),
                variables: Some(vars),
                projections: Some(projections),
                distinct: distinct.is_some(),
                order_by: order_by_strs,
                limit,
                skip,
            }
        },
    )(input)
}

/// Parse MATCH clause
fn parse_match_clause(input: &str) -> ParseResult<'_, (Vec<NodePattern>, Vec<EdgePattern>)> {
    preceded(
        pair(tag_no_case("MATCH"), multispace1),
        map(
            separated_list1(
                delimited(multispace0, char(','), multispace0),
                alt((
                    // Path pattern
                    map(parse_path_segment, |(from, edge, to)| {
                        (vec![from, to], vec![edge])
                    }),
                    // Simple node
                    map(parse_node_pattern, |node| (vec![node], vec![])),
                )),
            ),
            |patterns| {
                let mut nodes = Vec::new();
                let mut edges = Vec::new();
                for (mut ns, mut es) in patterns {
                    nodes.append(&mut ns);
                    edges.append(&mut es);
                }
                (nodes, edges)
            },
        ),
    )(input)
}

/// Parse OPTIONAL MATCH clause
fn parse_optional_match_clause(
    input: &str,
) -> ParseResult<'_, (Vec<NodePattern>, Vec<EdgePattern>)> {
    preceded(
        tuple((
            tag_no_case("OPTIONAL"),
            multispace1,
            tag_no_case("MATCH"),
            multispace1,
        )),
        map(
            separated_list1(
                delimited(multispace0, char(','), multispace0),
                alt((
                    map(parse_path_segment, |(from, edge, to)| {
                        (vec![from, to], vec![edge])
                    }),
                    map(parse_node_pattern, |node| (vec![node], vec![])),
                )),
            ),
            |patterns| {
                let mut nodes = Vec::new();
                let mut edges = Vec::new();
                for (mut ns, mut es) in patterns {
                    for n in &mut ns {
                        n.optional = true;
                    }
                    for e in &mut es {
                        e.optional = true;
                    }
                    nodes.append(&mut ns);
                    edges.append(&mut es);
                }
                (nodes, edges)
            },
        ),
    )(input)
}

/// Parse CREATE node spec
fn parse_create_node_spec(input: &str) -> ParseResult<'_, CreateNodeSpec> {
    map(
        delimited(
            char('('),
            tuple((
                preceded(multispace0, opt(any_identifier)),
                parse_optional_labels,
                preceded(multispace0, opt(property_value_map)),
                multispace0,
            )),
            char(')'),
        ),
        |(var, labels, props, _)| CreateNodeSpec {
            variable: var,
            labels,
            properties: props.unwrap_or_default(),
        },
    )(input)
}

/// Parse CREATE edge spec
#[allow(clippy::type_complexity)] // Labelled tuple documents the parser shape.
fn parse_create_edge_spec(
    input: &str,
) -> ParseResult<
    '_,
    (
        Option<String>,                     // variable
        Option<String>,                     // edge type
        HashMap<String, serde_json::Value>, // properties
        EdgeDirection,
    ),
> {
    map(
        tuple((
            parse_edge_left,
            delimited(
                char('['),
                tuple((
                    preceded(multispace0, opt(any_identifier)),
                    opt(preceded(char(':'), any_identifier)),
                    preceded(multispace0, opt(property_value_map)),
                    multispace0,
                )),
                char(']'),
            ),
            parse_edge_right,
        )),
        |(left, (var, edge_type, props, _), right)| {
            let direction = match (left, right) {
                (false, true) => EdgeDirection::Outgoing,
                (true, false) => EdgeDirection::Incoming,
                _ => EdgeDirection::Bidirectional,
            };
            (var, edge_type, props.unwrap_or_default(), direction)
        },
    )(input)
}

/// Parse CREATE edge pattern: (a)-[r:TYPE]->(b)
fn parse_create_edge_pattern(
    input: &str,
) -> ParseResult<'_, (CreateNodeSpec, CreateEdgeSpec, CreateNodeSpec)> {
    map(
        tuple((
            parse_create_node_spec,
            preceded(multispace0, parse_create_edge_spec),
            preceded(multispace0, parse_create_node_spec),
        )),
        |(from_node, (edge_var, edge_type, edge_props, direction), to_node)| {
            let from_var = from_node
                .variable
                .clone()
                .unwrap_or_else(|| "_from".to_string());
            let to_var = to_node
                .variable
                .clone()
                .unwrap_or_else(|| "_to".to_string());

            let edge = CreateEdgeSpec {
                variable: edge_var,
                from_variable: Some(from_var),
                to_variable: Some(to_var),
                edge_type: Some(edge_type.unwrap_or_default()),
                properties: edge_props,
                direction,
            };

            (from_node, edge, to_node)
        },
    )(input)
}

/// Parse CREATE clause
fn parse_create_clause(input: &str) -> ParseResult<'_, CreateClause> {
    preceded(
        pair(tag_no_case("CREATE"), multispace1),
        map(
            separated_list1(
                delimited(multispace0, char(','), multispace0),
                alt((
                    map(parse_create_edge_pattern, |(from, edge, to)| {
                        (vec![from, to], vec![edge])
                    }),
                    map(parse_create_node_spec, |node| (vec![node], vec![])),
                )),
            ),
            |patterns| {
                let mut nodes = Vec::new();
                let mut edges = Vec::new();
                for (mut ns, mut es) in patterns {
                    nodes.append(&mut ns);
                    edges.append(&mut es);
                }
                CreateClause { nodes, edges }
            },
        ),
    )(input)
}

/// Parse DELETE clause
fn parse_delete_clause(input: &str) -> ParseResult<'_, DeleteClause> {
    alt((
        // DETACH DELETE
        map(
            preceded(
                tuple((
                    tag_no_case("DETACH"),
                    multispace1,
                    tag_no_case("DELETE"),
                    multispace1,
                )),
                separated_list1(
                    delimited(multispace0, char(','), multispace0),
                    any_identifier,
                ),
            ),
            |vars| DeleteClause {
                variables: vars,
                detach: true,
            },
        ),
        // DELETE
        map(
            preceded(
                pair(tag_no_case("DELETE"), multispace1),
                separated_list1(
                    delimited(multispace0, char(','), multispace0),
                    any_identifier,
                ),
            ),
            |vars| DeleteClause {
                variables: vars,
                detach: false,
            },
        ),
    ))(input)
}

/// Parse SET item
fn parse_set_item(input: &str) -> ParseResult<'_, SetItem> {
    alt((
        // n.prop = value
        map(
            tuple((
                any_identifier,
                char('.'),
                any_identifier,
                delimited(multispace0, char('='), multispace0),
                property_value,
            )),
            |(var, _, prop, _, val)| SetItem::Property {
                variable: var,
                property: prop,
                value: val,
            },
        ),
        // n:Label
        map(
            tuple((any_identifier, char(':'), any_identifier)),
            |(var, _, label)| SetItem::AddLabel {
                variable: var,
                label,
            },
        ),
        // n += {props}
        map(
            tuple((
                any_identifier,
                delimited(multispace0, tag("+="), multispace0),
                property_value_map,
            )),
            |(var, _, props)| SetItem::MergeProperties {
                variable: var,
                properties: props,
            },
        ),
        // n = {props}
        map(
            tuple((
                any_identifier,
                delimited(multispace0, char('='), multispace0),
                property_value_map,
            )),
            |(var, _, props)| SetItem::AllProperties {
                variable: var,
                properties: props,
            },
        ),
    ))(input)
}

/// Parse SET clause
fn parse_set_clause(input: &str) -> ParseResult<'_, SetClause> {
    preceded(
        pair(tag_no_case("SET"), multispace1),
        map(
            separated_list1(
                delimited(multispace0, char(','), multispace0),
                parse_set_item,
            ),
            |items| SetClause { items },
        ),
    )(input)
}

/// Parse REMOVE item
fn parse_remove_item(input: &str) -> ParseResult<'_, RemoveItem> {
    alt((
        // n.prop
        map(
            tuple((any_identifier, char('.'), any_identifier)),
            |(var, _, prop)| RemoveItem::Property {
                variable: var,
                property: prop,
            },
        ),
        // n:Label
        map(
            tuple((any_identifier, char(':'), any_identifier)),
            |(var, _, label)| RemoveItem::Label {
                variable: var,
                label,
            },
        ),
    ))(input)
}

/// Parse REMOVE clause
fn parse_remove_clause(input: &str) -> ParseResult<'_, RemoveClause> {
    preceded(
        pair(tag_no_case("REMOVE"), multispace1),
        map(
            separated_list1(
                delimited(multispace0, char(','), multispace0),
                parse_remove_item,
            ),
            |items| RemoveClause { items },
        ),
    )(input)
}

/// Parse MERGE clause
fn parse_merge_clause(input: &str) -> ParseResult<'_, MergeClause> {
    map(
        tuple((
            preceded(
                pair(tag_no_case("MERGE"), multispace1),
                map(
                    separated_list1(
                        delimited(multispace0, char(','), multispace0),
                        alt((
                            map(parse_path_segment, |(from, edge, to)| {
                                (vec![from, to], vec![edge])
                            }),
                            map(parse_node_pattern, |node| (vec![node], vec![])),
                        )),
                    ),
                    |patterns| {
                        let mut nodes = Vec::new();
                        let mut edges = Vec::new();
                        for (mut ns, mut es) in patterns {
                            nodes.append(&mut ns);
                            edges.append(&mut es);
                        }
                        MatchPattern {
                            nodes,
                            edges,
                            paths: vec![],
                            where_clause: None,
                            where_clauses: vec![],
                            with_clauses: vec![],
                        }
                    },
                ),
            ),
            opt(preceded(
                tuple((
                    multispace1,
                    tag_no_case("ON"),
                    multispace1,
                    tag_no_case("CREATE"),
                    multispace1,
                )),
                parse_set_clause,
            )),
            opt(preceded(
                tuple((
                    multispace1,
                    tag_no_case("ON"),
                    multispace1,
                    tag_no_case("MATCH"),
                    multispace1,
                )),
                parse_set_clause,
            )),
        )),
        |(pattern, on_create, on_match)| MergeClause {
            pattern,
            on_create,
            on_match,
        },
    )(input)
}

/// Parse WITH clause
fn parse_with_clause(input: &str) -> ParseResult<'_, WithClause> {
    map(
        tuple((
            tag_no_case("WITH"),
            opt(preceded(multispace1, tag_no_case("DISTINCT"))),
            multispace1,
            separated_list1(
                delimited(multispace0, char(','), multispace0),
                parse_return_item,
            ),
            opt(preceded(multispace1, parse_order_by)),
            opt(preceded(multispace1, parse_skip)),
            opt(preceded(multispace1, parse_limit)),
            opt(preceded(multispace1, parse_where_clause)),
        )),
        |(_, distinct, _, items, order_by, skip, limit, where_clause)| {
            let (_, projections): (Vec<_>, Vec<_>) = items.into_iter().unzip();
            let order_by_strs: Vec<String> = order_by
                .unwrap_or_default()
                .into_iter()
                .map(|(s, asc)| format!("{}{}", s, if asc { " ASC" } else { " DESC" }))
                .collect();
            WithClause {
                projections,
                distinct: distinct.is_some(),
                order_by: order_by_strs,
                limit,
                skip,
                where_clause,
            }
        },
    )(input)
}

/// Parse a complete Cypher query
fn parse_cypher_query(input: &str) -> ParseResult<'_, CypherQuery> {
    map(
        tuple((
            // Reading clauses
            many0(preceded(
                multispace0,
                alt((
                    map(
                        tuple((
                            parse_optional_match_clause,
                            opt(preceded(multispace1, parse_where_clause)),
                        )),
                        |((nodes, edges), where_opt)| ReadingClause::Match {
                            pattern: MatchPattern {
                                nodes,
                                edges,
                                paths: vec![],
                                where_clause: where_opt,
                                where_clauses: vec![],
                                with_clauses: vec![],
                            },
                            optional: true,
                        },
                    ),
                    map(
                        tuple((
                            parse_match_clause,
                            opt(preceded(multispace1, parse_where_clause)),
                        )),
                        |((nodes, edges), where_opt)| ReadingClause::Match {
                            pattern: MatchPattern {
                                nodes,
                                edges,
                                paths: vec![],
                                where_clause: where_opt,
                                where_clauses: vec![],
                                with_clauses: vec![],
                            },
                            optional: false,
                        },
                    ),
                )),
            )),
            // WITH clauses
            many0(preceded(multispace0, parse_with_clause)),
            // Updating clauses
            many0(preceded(
                multispace0,
                alt((
                    map(parse_create_clause, UpdatingClause::Create),
                    map(parse_delete_clause, UpdatingClause::Delete),
                    map(parse_set_clause, UpdatingClause::Set),
                    map(parse_remove_clause, UpdatingClause::Remove),
                    map(parse_merge_clause, UpdatingClause::Merge),
                )),
            )),
            // Optional RETURN
            opt(preceded(multispace0, parse_return_clause)),
            multispace0,
        )),
        |(reading_clauses, with_clauses, updating_clauses, return_spec, _)| CypherQuery {
            clauses: vec![],
            reading_clauses,
            updating_clauses,
            with_clauses,
            union_clauses: vec![],
            return_spec,
        },
    )(input)
}

/// Parse simple MATCH query into CompiledPattern
fn parse_simple_query(input: &str) -> ParseResult<'_, CompiledPattern> {
    map(
        tuple((
            parse_match_clause,
            multispace0,
            opt(parse_where_clause),
            multispace0,
            parse_return_clause,
            multispace0,
        )),
        |((nodes, edges), _, where_opt, _, _return_spec, _)| CompiledPattern {
            nodes,
            edges,
            paths: vec![],
            where_clause: where_opt.clone(),
            where_clauses: where_opt.into_iter().collect(),
            with_clauses: vec![],
        },
    )(input)
}

// ============================================================================
// AST VISITOR PATTERN
// ============================================================================

/// Visitor trait for CypherQuery AST
pub trait CypherVisitor {
    /// Output type produced by visiting each node
    type Output;

    /// Visit a complete Cypher query
    fn visit_query(&mut self, query: &CypherQuery) -> Self::Output;
    /// Visit a reading clause (MATCH, OPTIONAL MATCH)
    fn visit_reading_clause(&mut self, clause: &ReadingClause) -> Self::Output;
    /// Visit an updating clause (CREATE, MERGE, DELETE, SET)
    fn visit_updating_clause(&mut self, clause: &UpdatingClause) -> Self::Output;
    /// Visit a node pattern
    fn visit_node_pattern(&mut self, pattern: &NodePattern) -> Self::Output;
    /// Visit an edge pattern
    fn visit_edge_pattern(&mut self, pattern: &EdgePattern) -> Self::Output;
    /// Visit a WHERE clause
    fn visit_where_clause(&mut self, clause: &WhereClause) -> Self::Output;
    /// Visit a RETURN specification
    fn visit_return_spec(&mut self, spec: &ReturnSpec) -> Self::Output;
}

/// Query validator visitor
pub struct QueryValidator {
    errors: Vec<String>,
}

impl QueryValidator {
    /// Create a new query validator.
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Validate a Cypher query AST for semantic correctness.
    pub fn validate(query: &CypherQuery) -> Result<(), Vec<String>> {
        let mut validator = Self::new();
        validator.visit_query(query);
        if validator.errors.is_empty() {
            Ok(())
        } else {
            Err(validator.errors)
        }
    }
}

impl Default for QueryValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CypherVisitor for QueryValidator {
    type Output = ();

    fn visit_query(&mut self, _query: &CypherQuery) {
        // Stub implementation - visitor pattern not fully implemented
    }

    fn visit_reading_clause(&mut self, _clause: &ReadingClause) {
        // Stub implementation
    }

    fn visit_updating_clause(&mut self, _clause: &UpdatingClause) {
        // Stub implementation
    }

    fn visit_node_pattern(&mut self, _pattern: &NodePattern) {
        // Stub implementation
    }

    fn visit_edge_pattern(&mut self, _pattern: &EdgePattern) {
        // Stub implementation
    }

    fn visit_where_clause(&mut self, _clause: &WhereClause) {
        // Stub implementation
    }

    fn visit_return_spec(&mut self, _spec: &ReturnSpec) {
        // Stub implementation
    }
}

// ============================================================================
// GRAPH QUERY CONVERSION
// ============================================================================

/// GraphQuery represents an executable graph query
#[derive(Debug, Clone)]
pub struct GraphQuery {
    /// Graph ID to execute against
    pub graph_id: String,
    /// Query type
    pub query_type: GraphQueryType,
    /// Query parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Types of graph queries.
///
/// The MATCH variant carries a much heavier payload than the others
/// (multiple `Pattern`/`Constraint`/`Order` lists). Boxing it would
/// touch every constructor and pattern-match site in the parser and
/// the executor; it's intentionally deferred until those sites have
/// settled. See `clippy::large_enum_variant`.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum GraphQueryType {
    /// Read-only pattern match
    Match {
        /// Patterns to match in the graph
        patterns: Vec<MatchPattern>,
        /// Specification for what to return
        return_spec: ReturnSpec,
    },
    /// Create nodes and edges
    Create {
        /// Node specifications to create
        nodes: Vec<CreateNodeSpec>,
        /// Edge specifications to create
        edges: Vec<CreateEdgeSpec>,
        /// Optional return specification
        return_spec: Option<ReturnSpec>,
    },
    /// Merge (create if not exists)
    Merge {
        /// Pattern to merge
        pattern: MatchPattern,
        /// SET clause to apply on CREATE
        on_create: Option<SetClause>,
        /// SET clause to apply on MATCH
        on_match: Option<SetClause>,
        /// Optional return specification
        return_spec: Option<ReturnSpec>,
    },
    /// Delete nodes and edges
    Delete {
        /// Variables to delete
        variables: Vec<String>,
        /// Whether to detach (delete relationships first)
        detach: bool,
        /// Patterns to match for deletion
        patterns: Vec<MatchPattern>,
    },
    /// Update properties
    Update {
        /// Patterns to match for updating
        patterns: Vec<MatchPattern>,
        /// SET clause with property updates
        set_clause: SetClause,
        /// Optional return specification
        return_spec: Option<ReturnSpec>,
    },
}

/// Convert CypherQuery AST to executable GraphQuery
pub fn cypher_to_graph_query(
    cypher: &CypherQuery,
    graph_id: &str,
) -> Result<GraphQuery, ProximaDBError> {
    // Stub implementation - full conversion not yet implemented
    let patterns = cypher
        .reading_clauses
        .iter()
        .filter_map(|clause| match clause {
            ReadingClause::Match { pattern, .. } => Some(pattern.clone()),
            ReadingClause::With(_) => None,
        })
        .collect();

    Ok(GraphQuery {
        graph_id: graph_id.to_string(),
        query_type: GraphQueryType::Match {
            patterns,
            return_spec: cypher.return_spec.clone().unwrap_or_else(|| ReturnSpec {
                items: vec![],
                variables: None,
                projections: None,
                distinct: false,
                order_by: vec![],
                limit: None,
                skip: None,
            }),
        },
        parameters: std::collections::HashMap::new(),
        timeout_ms: None,
    })
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_basic() {
        let mut lexer = CypherLexer::new();
        let tokens = lexer
            .tokenize("MATCH (n) RETURN n")
            .expect("lexer should tokenize basic query");
        assert!(tokens.len() >= 5);
        assert!(matches!(tokens[0].token, Token::Match));
    }

    #[test]
    fn test_lexer_string_literals() {
        let mut lexer = CypherLexer::new();
        let tokens = lexer
            .tokenize("'hello' \"world\"")
            .expect("lexer should tokenize string literals");
        assert!(matches!(&tokens[0].token, Token::String(s) if s == "hello"));
        assert!(matches!(&tokens[1].token, Token::String(s) if s == "world"));
    }

    #[test]
    fn test_lexer_numbers() {
        let mut lexer = CypherLexer::new();
        let tokens = lexer
            .tokenize("42 3.14 -5")
            .expect("lexer should tokenize numbers");
        assert!(matches!(tokens[0].token, Token::Integer(42)));
        assert!(matches!(tokens[1].token, Token::Float(f) if (f - 3.14).abs() < 0.001));
        assert!(matches!(tokens[2].token, Token::Integer(-5)));
    }

    #[test]
    fn test_parse_simple_match() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person) RETURN n")
            .expect("parser should parse simple match");
        assert_eq!(query.reading_clauses.len(), 1);
        assert!(query.is_read_only());
    }

    #[test]
    fn test_parse_match_with_properties() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person {name: 'Alice', age: 30}) RETURN n")
            .expect("parser should parse match with properties");
        assert_eq!(query.reading_clauses.len(), 1);
    }

    #[test]
    fn test_parse_match_with_edge() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b")
            .expect("parser should parse match with edge");
        assert_eq!(query.reading_clauses.len(), 1);
        if let ReadingClause::Match { pattern, .. } = &query.reading_clauses[0] {
            assert_eq!(pattern.nodes.len(), 2);
            assert_eq!(pattern.edges.len(), 1);
        }
    }

    #[test]
    fn test_parse_where_clause() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person) WHERE n.age > 25 RETURN n")
            .expect("parser should parse where clause");
        if let ReadingClause::Match { pattern, .. } = &query.reading_clauses[0] {
            assert!(pattern.where_clause.is_some());
        }
    }

    #[test]
    fn test_parse_where_and() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person) WHERE n.age > 25 AND n.name = 'Alice' RETURN n")
            .expect("parser should parse where with AND");
        if let ReadingClause::Match { pattern, .. } = &query.reading_clauses[0] {
            assert!(pattern.where_clause.is_some());
            assert!(matches!(
                pattern.where_clause.as_ref().unwrap(),
                WhereClause::And(_, _)
            ));
        }
    }

    #[test]
    fn test_parse_create() {
        let parser = CypherParser::new();
        let query = parser
            .parse("CREATE (n:Person {name: 'Alice'}) RETURN n")
            .expect("parser should parse create");
        assert!(!query.is_read_only());
        assert!(query.has_create());
    }

    #[test]
    fn test_parse_delete() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person) WHERE n.age > 100 DELETE n")
            .expect("parser should parse delete");
        assert!(query.has_delete());
    }

    #[test]
    fn test_parse_set() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person {name: 'Alice'}) SET n.age = 31 RETURN n")
            .expect("parser should parse set");
        assert!(!query.is_read_only());
    }

    #[test]
    fn test_parse_merge() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MERGE (n:Person {name: 'Alice'}) RETURN n")
            .expect("parser should parse merge");
        assert!(query.has_merge());
    }

    #[test]
    fn test_parse_aggregation() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person) RETURN COUNT(*)")
            .expect("parser should parse aggregation");
        if let Some(ref return_spec) = query.return_spec {
            assert!(matches!(
                return_spec.projections.as_ref().unwrap()[0],
                PropertyProjection::Count
            ));
        }
    }

    #[test]
    fn test_parse_order_limit() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person) RETURN n ORDER BY n ASC LIMIT 10")
            .expect("parser should parse order by and limit");
        if let Some(ref return_spec) = query.return_spec {
            assert_eq!(return_spec.order_by.len(), 1);
            assert_eq!(return_spec.limit, Some(10));
        }
    }

    #[test]
    fn test_validator() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person) RETURN n")
            .expect("parser should parse for validator test");
        assert!(QueryValidator::validate(&query).is_ok());
    }

    #[test]
    fn test_conversion_to_graph_query() {
        let parser = CypherParser::new();
        let cypher = parser
            .parse("MATCH (n:Person) RETURN n")
            .expect("parser should parse for conversion test");
        let graph_query = cypher_to_graph_query(&cypher, "test_graph")
            .expect("conversion to graph query should succeed");
        assert_eq!(graph_query.graph_id, "test_graph");
        assert!(matches!(
            graph_query.query_type,
            GraphQueryType::Match { .. }
        ));
    }

    #[test]
    fn test_parse_optional_match() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person) OPTIONAL MATCH (n)-[r:KNOWS]->(m) RETURN n, m")
            .expect("parser should parse optional match");
        assert_eq!(query.reading_clauses.len(), 2);
        if let ReadingClause::Match { optional, .. } = &query.reading_clauses[1] {
            assert!(*optional);
        }
    }

    #[test]
    fn test_parse_multiple_labels() {
        let parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person:Employee) RETURN n")
            .expect("parser should parse multiple labels");
        if let ReadingClause::Match { pattern, .. } = &query.reading_clauses[0] {
            assert_eq!(pattern.nodes[0].labels.len(), 2);
        }
    }
}
