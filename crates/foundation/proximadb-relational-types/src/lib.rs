//! Typed relational expressions, rows, and schema — the foundation
//! layer for ProximaDB's SQL query path (ADR-019 L2).
//!
//! This crate is intentionally minimal. It owns:
//!
//! - [`RelationalSchema`] + [`ColumnInfo`] — ordered column metadata.
//! - [`RelationalRow`] — a single row of `ProximaValue`s.
//! - [`ColumnRef`] — a resolved reference (ordinal + type) into a row.
//! - [`Expr`] — the typed expression AST.
//! - [`BinaryOp`] / [`UnaryOp`] — operator enums.
//! - [`Expr::eval`] — SQL-correct expression evaluation against a row,
//!   including three-valued-logic NULL propagation.
//! - [`Expr::result_type`] — pure type inference helper.
//! - [`Expr::type_check`] — validates a freshly-lowered expression
//!   against a target schema before it reaches the executor.
//!
//! The crate builds directly on `proximadb-data-model::{ProximaType,
//! ProximaValue}` (the canonical type system per the multimodal
//! architecture mandate). It does NOT introduce a parallel value
//! system; relational rows are just `Vec<ProximaValue>`.
//!
//! Design constraints driven by ADR-019:
//!
//! - **Three-valued logic.** SQL boolean operators have NULL as a
//!   third state. `TRUE AND NULL = NULL`, `FALSE AND NULL = FALSE`,
//!   `TRUE OR NULL = TRUE`, `FALSE OR NULL = NULL`, `NOT NULL = NULL`.
//!   Comparison operators return NULL if either operand is NULL.
//!   The only way to test for NULL is `IS NULL` / `IS NOT NULL`.
//!
//! - **NULL propagation in arithmetic.** Any arithmetic operand
//!   that is NULL produces a NULL result.
//!
//! - **Strict type discipline.** No silent implicit promotion at MVP
//!   (e.g. `INT + FLOAT` requires an explicit `CAST`). Implicit
//!   numeric promotion is a Phase 3 add-on; the AST and evaluator
//!   are designed so it can be enabled by relaxing
//!   `BinaryOp::resolve_arithmetic_type` without touching the
//!   executor.
//!
//! - **Function calls are extension points.** [`Expr::FuncCall`]
//!   carries an opaque name + arg list; resolution happens at the
//!   executor's [`FunctionRegistry`] (provided by callers, not
//!   shipped here). The foundation crate stays small.
//!
//! - **Subquery expressions are deferred** to the algebra crate so
//!   we don't create a circular type dependency between
//!   `LogicalNode` and `Expr`. The algebra crate adds an `Exists`
//!   / `Scalar(LogicalNode)` extension wrapper around `Expr` when
//!   needed.

use proximadb_data_model::{ProximaType, ProximaValue, TimeUnit};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// =========================================================================
// Errors
// =========================================================================

/// Errors that can arise during expression evaluation or type
/// checking. Distinct from a syntactic SQL error — these are
/// runtime / planning-time issues over a typed AST.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ExprError {
    #[error("column ordinal {ordinal} out of range for row of length {row_len}")]
    ColumnOrdinalOutOfRange { ordinal: usize, row_len: usize },

    #[error("unknown column {name:?} (schema has {available_count} columns)")]
    UnknownColumn {
        name: String,
        available_count: usize,
    },

    #[error("type mismatch: expected {expected:?}, got {actual:?} at {context}")]
    TypeMismatch {
        expected: ProximaType,
        actual: ProximaType,
        context: String,
    },

    #[error("operator {op} is not defined for operand types {left:?} and {right:?}")]
    OperatorNotDefined {
        op: String,
        left: ProximaType,
        right: ProximaType,
    },

    #[error("unary operator {op} is not defined for operand type {operand:?}")]
    UnaryOperatorNotDefined { op: String, operand: ProximaType },

    #[error("cast from {from:?} to {to:?} is not supported")]
    UnsupportedCast { from: ProximaType, to: ProximaType },

    #[error("function {name} is not registered with the evaluator")]
    UnknownFunction { name: String },

    #[error("function {name} called with wrong arity (expected {expected}, got {got})")]
    WrongFunctionArity {
        name: String,
        expected: usize,
        got: usize,
    },

    #[error("division by zero")]
    DivisionByZero,

    /// Text that is not a valid literal of the target type (PostgreSQL's
    /// `invalid_text_representation`, 22P02) — e.g. `CAST('x' AS BOOLEAN)`.
    #[error("invalid input syntax for type {to:?}: \"{value}\"")]
    InvalidTextRepresentation { value: String, to: ProximaType },

    #[error("arithmetic overflow")]
    ArithmeticOverflow,

    #[error("LIKE pattern compile failed: {0}")]
    InvalidLikePattern(String),

    #[error("CASE expression has no matching WHEN branch and no ELSE")]
    UnmatchedCase,

    #[error("{0}")]
    Other(String),
}

// =========================================================================
// Schema and Row
// =========================================================================

/// One column's metadata in a relational schema.
///
/// `nullable` is part of the contract: the type checker uses it to
/// determine whether a downstream operator must handle NULL, and the
/// planner uses it to skip the NULL-propagation wrapper for operators
/// over non-nullable inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub ty: ProximaType,
    pub nullable: bool,
}

impl ColumnInfo {
    pub fn new(name: impl Into<String>, ty: ProximaType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            ty,
            nullable,
        }
    }
}

/// Ordered column metadata. The order is significant: column ordinals
/// in [`RelationalRow`] match the order here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RelationalSchema {
    pub columns: Vec<ColumnInfo>,
}

impl RelationalSchema {
    pub fn new(columns: Vec<ColumnInfo>) -> Self {
        Self { columns }
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Find a column by name. Returns `(ordinal, info)`. Case-
    /// sensitive — the caller normalises identifiers before lookup.
    pub fn column_by_name(&self, name: &str) -> Option<(usize, &ColumnInfo)> {
        self.columns
            .iter()
            .enumerate()
            .find(|(_, c)| c.name == name)
    }

    /// Resolve a column reference. Returns `ExprError::UnknownColumn`
    /// if the name doesn't exist in this schema.
    pub fn resolve_column(&self, name: &str) -> Result<ColumnRef, ExprError> {
        let (ordinal, info) =
            self.column_by_name(name)
                .ok_or_else(|| ExprError::UnknownColumn {
                    name: name.to_string(),
                    available_count: self.columns.len(),
                })?;
        Ok(ColumnRef {
            name: info.name.clone(),
            ordinal,
            ty: info.ty.clone(),
            nullable: info.nullable,
        })
    }

    /// Build a schema that's the result of a projection. Order matches
    /// the input column-ref list.
    pub fn project(&self, refs: &[ColumnRef]) -> RelationalSchema {
        RelationalSchema {
            columns: refs
                .iter()
                .map(|r| ColumnInfo {
                    name: r.name.clone(),
                    ty: r.ty.clone(),
                    nullable: r.nullable,
                })
                .collect(),
        }
    }
}

/// A single relational row. Index aligns with [`RelationalSchema`].
pub type RelationalRow = Vec<ProximaValue>;

// =========================================================================
// Column reference
// =========================================================================

/// A resolved column reference — name kept for display/EXPLAIN, but
/// `ordinal` is the index the evaluator actually uses. Resolution
/// happens at lowering time against a [`RelationalSchema`]; the
/// executor never does name lookup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnRef {
    pub name: String,
    pub ordinal: usize,
    pub ty: ProximaType,
    pub nullable: bool,
}

// =========================================================================
// Operators
// =========================================================================

/// Binary operator on two expressions. Logical operators
/// (`And`, `Or`) have three-valued-logic semantics; arithmetic and
/// comparison propagate NULL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    // Arithmetic
    Plus,
    Minus,
    Mul,
    Div,
    Mod,
    // Comparison
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    // Logical
    And,
    Or,
    // String
    Concat,
}

impl BinaryOp {
    pub fn is_arithmetic(self) -> bool {
        use BinaryOp::*;
        matches!(self, Plus | Minus | Mul | Div | Mod)
    }

    pub fn is_comparison(self) -> bool {
        use BinaryOp::*;
        matches!(self, Eq | NotEq | Lt | LtEq | Gt | GtEq)
    }

    pub fn is_logical(self) -> bool {
        use BinaryOp::*;
        matches!(self, And | Or)
    }

    pub fn as_str(self) -> &'static str {
        use BinaryOp::*;
        match self {
            Plus => "+",
            Minus => "-",
            Mul => "*",
            Div => "/",
            Mod => "%",
            Eq => "=",
            NotEq => "<>",
            Lt => "<",
            LtEq => "<=",
            Gt => ">",
            GtEq => ">=",
            And => "AND",
            Or => "OR",
            Concat => "||",
        }
    }
}

/// Unary operator. `Not` is logical (three-valued); `Neg` is
/// numeric (NULL-propagating).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
}

impl UnaryOp {
    pub fn as_str(self) -> &'static str {
        match self {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "NOT",
        }
    }
}

// =========================================================================
// Expression AST
// =========================================================================

/// Typed relational expression. Every variant carries enough
/// information for the evaluator to produce a [`ProximaValue`]
/// without re-doing name resolution or type inference.
///
/// Subquery expressions (`Exists`, scalar subquery) deliberately do
/// NOT appear here — they would create a circular type dependency
/// between this crate and the algebra crate. The algebra crate
/// wraps `Expr` in its own enum that adds subquery variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Bound column reference. The ordinal is what the evaluator
    /// uses; the name is for display.
    Column(ColumnRef),

    /// Inline literal. The value's type must match `ty`; we carry
    /// `ty` separately for the (rare) NULL literal where the value
    /// alone doesn't determine its type.
    Literal {
        value: ProximaValue,
        ty: ProximaType,
    },

    /// Explicit cast. Failure mode: `UnsupportedCast` for casts that
    /// have no defined rule (e.g. `CAST(json AS int)` for
    /// non-numeric JSON values).
    Cast { expr: Box<Expr>, ty: ProximaType },

    /// Binary operator.
    BinaryOp {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Unary operator.
    UnaryOp { op: UnaryOp, expr: Box<Expr> },

    /// `expr IS NULL` (or `IS NOT NULL` when `not = true`). Note
    /// this is the ONLY way to test a value for NULL — `expr = NULL`
    /// always returns NULL.
    IsNull { expr: Box<Expr>, not: bool },

    /// `expr BETWEEN low AND high` (or `NOT BETWEEN`).
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        not: bool,
    },

    /// `expr IN (v1, v2, ...)` (or `NOT IN`). List form only;
    /// subquery `IN` lives in the algebra crate.
    In {
        expr: Box<Expr>,
        list: Vec<Expr>,
        not: bool,
    },

    /// `expr LIKE pattern` (or `NOT LIKE`). Supports SQL wildcards
    /// `%` (any-length) and `_` (single char). Case-insensitive
    /// when `case_insensitive = true` (corresponds to `ILIKE`).
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        not: bool,
        case_insensitive: bool,
    },

    /// `CASE WHEN c1 THEN r1 [WHEN c2 THEN r2 ...] [ELSE r]`.
    Case {
        branches: Vec<(Expr, Expr)>,
        otherwise: Option<Box<Expr>>,
    },

    /// `COALESCE(e1, e2, ...)` — returns the first non-NULL arg.
    Coalesce(Vec<Expr>),

    /// `NULLIF(a, b)` — returns NULL if `a = b`, else `a`.
    NullIf { left: Box<Expr>, right: Box<Expr> },

    /// `f(args)` for user-defined or builtin functions. Resolution
    /// happens via a [`FunctionRegistry`] supplied by the executor;
    /// the foundation crate doesn't ship any concrete functions.
    /// `return_ty` is what the planner inferred; the evaluator
    /// asserts the function actually returns that type.
    FuncCall {
        name: String,
        args: Vec<Expr>,
        return_ty: ProximaType,
    },
}

// =========================================================================
// Function registry (extension point)
// =========================================================================

/// Function dispatch trait. Executors supply an implementation; the
/// foundation crate doesn't ship any concrete functions because
/// function libraries (string ops, math, date/time, JSON) are
/// per-domain and evolve independently.
pub trait FunctionRegistry {
    /// Look up a function by name. Returns `None` if unregistered.
    /// Returned closure receives the already-evaluated args and
    /// must return the result `ProximaValue` (or an error).
    ///
    /// Implementations are typically `HashMap<&str, fn(&[ProximaValue]) -> Result<ProximaValue, ExprError>>`
    /// or similar.
    fn dispatch(
        &self,
        name: &str,
        args: &[ProximaValue],
    ) -> Option<Result<ProximaValue, ExprError>>;
}

/// A function registry that holds no functions. Useful as the
/// default when no functions are needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoFunctions;

impl FunctionRegistry for NoFunctions {
    fn dispatch(
        &self,
        _name: &str,
        _args: &[ProximaValue],
    ) -> Option<Result<ProximaValue, ExprError>> {
        None
    }
}

// =========================================================================
// Type inference
// =========================================================================

impl Expr {
    /// What ProximaType does this expression's value have? Pure —
    /// no I/O, no row needed. Caller is responsible for ensuring
    /// the expression is well-formed (via [`type_check`] if it's
    /// untrusted input).
    pub fn result_type(&self) -> ProximaType {
        match self {
            Expr::Column(col) => col.ty.clone(),
            Expr::Literal { ty, .. } => ty.clone(),
            Expr::Cast { ty, .. } => ty.clone(),
            Expr::BinaryOp { op, left, right } => {
                let lt = left.result_type();
                let rt = right.result_type();
                if op.is_comparison() || op.is_logical() || matches!(op, BinaryOp::Concat) {
                    if matches!(op, BinaryOp::Concat) {
                        ProximaType::String
                    } else {
                        ProximaType::Boolean
                    }
                } else if op.is_arithmetic() {
                    if lt == rt {
                        lt
                    } else if numeric_type(&lt) && numeric_type(&rt) {
                        // Mixed numeric arithmetic widens both sides to f64 at
                        // evaluation (`eval_arithmetic`, TPC-H Q19) — the plan
                        // schema must agree.
                        ProximaType::Float64
                    } else {
                        // Best-effort for display; evaluation errors loudly.
                        lt
                    }
                } else {
                    lt
                }
            }
            Expr::UnaryOp { op, expr } => match op {
                UnaryOp::Not => ProximaType::Boolean,
                UnaryOp::Neg => expr.result_type(),
            },
            Expr::IsNull { .. } => ProximaType::Boolean,
            Expr::Between { .. } => ProximaType::Boolean,
            Expr::In { .. } => ProximaType::Boolean,
            Expr::Like { .. } => ProximaType::Boolean,
            Expr::Case {
                branches,
                otherwise,
            } => {
                // Result type is the first branch's result. The type
                // checker enforces that all branches agree.
                if let Some((_, then)) = branches.first() {
                    then.result_type()
                } else if let Some(default) = otherwise {
                    default.result_type()
                } else {
                    ProximaType::Boolean
                }
            }
            Expr::Coalesce(args) => args
                .first()
                .map(|e| e.result_type())
                .unwrap_or(ProximaType::Boolean),
            Expr::NullIf { left, .. } => left.result_type(),
            Expr::FuncCall { return_ty, .. } => return_ty.clone(),
        }
    }

    /// Walk the expression and validate against the schema. Returns
    /// the first error found.
    pub fn type_check(&self, schema: &RelationalSchema) -> Result<(), ExprError> {
        match self {
            Expr::Column(col) => {
                if col.ordinal >= schema.len() {
                    return Err(ExprError::ColumnOrdinalOutOfRange {
                        ordinal: col.ordinal,
                        row_len: schema.len(),
                    });
                }
                let actual = &schema.columns[col.ordinal];
                if actual.ty != col.ty {
                    return Err(ExprError::TypeMismatch {
                        expected: col.ty.clone(),
                        actual: actual.ty.clone(),
                        context: format!("column {:?} ordinal {}", col.name, col.ordinal),
                    });
                }
                Ok(())
            }
            Expr::Literal { value, ty } => {
                let actual = proxima_value_type(value);
                if let Some(actual) = actual
                    && &actual != ty
                    && !matches!(value, ProximaValue::Null)
                {
                    return Err(ExprError::TypeMismatch {
                        expected: ty.clone(),
                        actual,
                        context: "literal".into(),
                    });
                }
                Ok(())
            }
            Expr::Cast { expr, .. } => expr.type_check(schema),
            Expr::BinaryOp { left, right, .. } => {
                left.type_check(schema)?;
                right.type_check(schema)?;
                Ok(())
            }
            Expr::UnaryOp { expr, .. } => expr.type_check(schema),
            Expr::IsNull { expr, .. } => expr.type_check(schema),
            Expr::Between {
                expr, low, high, ..
            } => {
                expr.type_check(schema)?;
                low.type_check(schema)?;
                high.type_check(schema)?;
                Ok(())
            }
            Expr::In { expr, list, .. } => {
                expr.type_check(schema)?;
                for e in list {
                    e.type_check(schema)?;
                }
                Ok(())
            }
            Expr::Like { expr, pattern, .. } => {
                expr.type_check(schema)?;
                pattern.type_check(schema)?;
                Ok(())
            }
            Expr::Case {
                branches,
                otherwise,
            } => {
                for (cond, then) in branches {
                    cond.type_check(schema)?;
                    then.type_check(schema)?;
                }
                if let Some(default) = otherwise {
                    default.type_check(schema)?;
                }
                Ok(())
            }
            Expr::Coalesce(args) => {
                for a in args {
                    a.type_check(schema)?;
                }
                Ok(())
            }
            Expr::NullIf { left, right } => {
                left.type_check(schema)?;
                right.type_check(schema)?;
                Ok(())
            }
            Expr::FuncCall { args, .. } => {
                for a in args {
                    a.type_check(schema)?;
                }
                Ok(())
            }
        }
    }

    /// Recursively collect every function-call name referenced in this expression
    /// (in evaluation order). Used to pre-validate that a predicate's functions
    /// resolve against the executor's registry before scanning, so a scan that
    /// cannot evaluate a function fails loudly instead of silently dropping rows.
    pub fn collect_func_names<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Expr::Column(_) | Expr::Literal { .. } => {}
            Expr::Cast { expr, .. } | Expr::UnaryOp { expr, .. } | Expr::IsNull { expr, .. } => {
                expr.collect_func_names(out)
            }
            Expr::BinaryOp { left, right, .. } | Expr::NullIf { left, right } => {
                left.collect_func_names(out);
                right.collect_func_names(out);
            }
            Expr::Between {
                expr, low, high, ..
            } => {
                expr.collect_func_names(out);
                low.collect_func_names(out);
                high.collect_func_names(out);
            }
            Expr::In { expr, list, .. } => {
                expr.collect_func_names(out);
                for e in list {
                    e.collect_func_names(out);
                }
            }
            Expr::Like { expr, pattern, .. } => {
                expr.collect_func_names(out);
                pattern.collect_func_names(out);
            }
            Expr::Case {
                branches,
                otherwise,
            } => {
                for (cond, then) in branches {
                    cond.collect_func_names(out);
                    then.collect_func_names(out);
                }
                if let Some(default) = otherwise {
                    default.collect_func_names(out);
                }
            }
            Expr::Coalesce(args) => {
                for a in args {
                    a.collect_func_names(out);
                }
            }
            Expr::FuncCall { name, args, .. } => {
                out.push(name.as_str());
                for a in args {
                    a.collect_func_names(out);
                }
            }
        }
    }
}

/// Best-effort inference of a `ProximaValue`'s `ProximaType`.
/// Returns `None` for `ProximaValue::Null` (which has no concrete
/// type) and for complex types whose element type isn't carried
/// on the value (e.g. empty Array).
pub fn proxima_value_type(v: &ProximaValue) -> Option<ProximaType> {
    use ProximaValue::*;
    Some(match v {
        Null => return None,
        Boolean(_) => ProximaType::Boolean,
        Int8(_) => ProximaType::Int8,
        Int16(_) => ProximaType::Int16,
        Int32(_) => ProximaType::Int32,
        Int64(_) => ProximaType::Int64,
        UInt8(_) => ProximaType::UInt8,
        UInt16(_) => ProximaType::UInt16,
        UInt32(_) => ProximaType::UInt32,
        UInt64(_) => ProximaType::UInt64,
        Float16(_) => ProximaType::Float16,
        Float32(_) => ProximaType::Float32,
        Float64(_) => ProximaType::Float64,
        Decimal(_) => ProximaType::Decimal {
            precision: 38,
            scale: 10,
        },
        String(_) => ProximaType::String,
        Symbol(_) => ProximaType::Symbol,
        Binary(_) => ProximaType::Binary,
        Date(_) => ProximaType::Date,
        Time(_, u) => ProximaType::Time(*u),
        Timestamp(_, u) => ProximaType::Timestamp(*u),
        TimestampTz(_, u) => ProximaType::TimestampTz(*u),
        Uuid(_) => ProximaType::Uuid,
        ULID(_) => ProximaType::ULID,
        Json(_) => ProximaType::Json,
        Jsonb(_) => ProximaType::Jsonb,
        Array(_) => return None, // element type not recoverable here
        Map(_) => return None,
        Struct(_) => return None,
        DenseVector(_) | SparseVector { .. } | BinaryVector(_) => return None,
    })
}

// =========================================================================
// Evaluation
// =========================================================================

impl Expr {
    /// Evaluate this expression against a row, looking up functions
    /// via the registry. Implements SQL three-valued logic and
    /// NULL propagation.
    pub fn eval<F: FunctionRegistry>(
        &self,
        row: &RelationalRow,
        funcs: &F,
    ) -> Result<ProximaValue, ExprError> {
        match self {
            Expr::Column(col) => {
                row.get(col.ordinal)
                    .cloned()
                    .ok_or(ExprError::ColumnOrdinalOutOfRange {
                        ordinal: col.ordinal,
                        row_len: row.len(),
                    })
            }
            Expr::Literal { value, .. } => Ok(value.clone()),
            Expr::Cast { expr, ty } => {
                let v = expr.eval(row, funcs)?;
                cast_value(&v, ty)
            }
            Expr::BinaryOp { op, left, right } => {
                let l = left.eval(row, funcs)?;
                let r = right.eval(row, funcs)?;
                eval_binary(*op, &l, &r)
            }
            Expr::UnaryOp { op, expr } => {
                let v = expr.eval(row, funcs)?;
                eval_unary(*op, &v)
            }
            Expr::IsNull { expr, not } => {
                let v = expr.eval(row, funcs)?;
                let is_null = matches!(v, ProximaValue::Null);
                Ok(ProximaValue::Boolean(if *not { !is_null } else { is_null }))
            }
            Expr::Between {
                expr,
                low,
                high,
                not,
            } => {
                let v = expr.eval(row, funcs)?;
                let l = low.eval(row, funcs)?;
                let h = high.eval(row, funcs)?;
                if matches!(v, ProximaValue::Null)
                    || matches!(l, ProximaValue::Null)
                    || matches!(h, ProximaValue::Null)
                {
                    return Ok(ProximaValue::Null);
                }
                let ge_low = compare_ord(&v, &l)?.is_ge();
                let le_high = compare_ord(&v, &h)?.is_le();
                let inside = ge_low && le_high;
                Ok(ProximaValue::Boolean(if *not { !inside } else { inside }))
            }
            Expr::In { expr, list, not } => {
                let v = expr.eval(row, funcs)?;
                if matches!(v, ProximaValue::Null) {
                    return Ok(ProximaValue::Null);
                }
                let mut saw_null = false;
                for candidate in list {
                    let c = candidate.eval(row, funcs)?;
                    if matches!(c, ProximaValue::Null) {
                        saw_null = true;
                        continue;
                    }
                    if values_equal(&v, &c)? {
                        return Ok(ProximaValue::Boolean(!*not));
                    }
                }
                // SQL semantics: `x IN (a, NULL)` returns NULL when x ≠ a
                if saw_null {
                    Ok(ProximaValue::Null)
                } else {
                    Ok(ProximaValue::Boolean(*not))
                }
            }
            Expr::Like {
                expr,
                pattern,
                not,
                case_insensitive,
            } => {
                let v = expr.eval(row, funcs)?;
                let p = pattern.eval(row, funcs)?;
                if matches!(v, ProximaValue::Null) || matches!(p, ProximaValue::Null) {
                    return Ok(ProximaValue::Null);
                }
                let haystack = value_as_str(&v)?;
                let pat = value_as_str(&p)?;
                let m = like_match(&haystack, &pat, *case_insensitive)?;
                Ok(ProximaValue::Boolean(if *not { !m } else { m }))
            }
            Expr::Case {
                branches,
                otherwise,
            } => {
                for (cond, then) in branches {
                    let c = cond.eval(row, funcs)?;
                    if let ProximaValue::Boolean(true) = c {
                        return then.eval(row, funcs);
                    }
                    // NULL conditions skip to the next branch.
                }
                if let Some(default) = otherwise {
                    default.eval(row, funcs)
                } else {
                    Ok(ProximaValue::Null)
                }
            }
            Expr::Coalesce(args) => {
                for a in args {
                    let v = a.eval(row, funcs)?;
                    if !matches!(v, ProximaValue::Null) {
                        return Ok(v);
                    }
                }
                Ok(ProximaValue::Null)
            }
            Expr::NullIf { left, right } => {
                let l = left.eval(row, funcs)?;
                let r = right.eval(row, funcs)?;
                if matches!(l, ProximaValue::Null) || matches!(r, ProximaValue::Null) {
                    return Ok(l);
                }
                if values_equal(&l, &r)? {
                    Ok(ProximaValue::Null)
                } else {
                    Ok(l)
                }
            }
            Expr::FuncCall { name, args, .. } => {
                let mut evaluated = Vec::with_capacity(args.len());
                for a in args {
                    evaluated.push(a.eval(row, funcs)?);
                }
                funcs
                    .dispatch(name, &evaluated)
                    .ok_or(ExprError::UnknownFunction { name: name.clone() })?
            }
        }
    }
}

// =========================================================================
// Operator evaluation helpers
// =========================================================================

fn eval_unary(op: UnaryOp, v: &ProximaValue) -> Result<ProximaValue, ExprError> {
    use ProximaValue::*;
    match op {
        UnaryOp::Neg => match v {
            Null => Ok(Null),
            Int8(n) => Ok(Int8(-n)),
            Int16(n) => Ok(Int16(-n)),
            Int32(n) => Ok(Int32(-n)),
            Int64(n) => Ok(Int64(-n)),
            Float32(f) => Ok(Float32(-f)),
            Float64(f) => Ok(Float64(-f)),
            other => Err(ExprError::UnaryOperatorNotDefined {
                op: "-".into(),
                operand: proxima_value_type(other).unwrap_or(ProximaType::Boolean),
            }),
        },
        UnaryOp::Not => match v {
            Null => Ok(Null),
            Boolean(b) => Ok(Boolean(!b)),
            other => Err(ExprError::UnaryOperatorNotDefined {
                op: "NOT".into(),
                operand: proxima_value_type(other).unwrap_or(ProximaType::Boolean),
            }),
        },
    }
}

fn eval_binary(
    op: BinaryOp,
    l: &ProximaValue,
    r: &ProximaValue,
) -> Result<ProximaValue, ExprError> {
    use BinaryOp::*;
    // Three-valued logic for AND/OR has special NULL semantics
    // (not pure NULL propagation), so handle those first.
    if matches!(op, And) {
        return eval_and(l, r);
    }
    if matches!(op, Or) {
        return eval_or(l, r);
    }
    // All other operators propagate NULL.
    if matches!(l, ProximaValue::Null) || matches!(r, ProximaValue::Null) {
        return Ok(ProximaValue::Null);
    }
    match op {
        Plus | Minus | Mul | Div | Mod => eval_arithmetic(op, l, r),
        Eq => Ok(ProximaValue::Boolean(values_equal(l, r)?)),
        NotEq => Ok(ProximaValue::Boolean(!values_equal(l, r)?)),
        Lt => Ok(ProximaValue::Boolean(compare_ord(l, r)?.is_lt())),
        LtEq => Ok(ProximaValue::Boolean(compare_ord(l, r)?.is_le())),
        Gt => Ok(ProximaValue::Boolean(compare_ord(l, r)?.is_gt())),
        GtEq => Ok(ProximaValue::Boolean(compare_ord(l, r)?.is_ge())),
        Concat => match (l, r) {
            (ProximaValue::String(a), ProximaValue::String(b)) => {
                Ok(ProximaValue::String(format!("{a}{b}")))
            }
            _ => Err(ExprError::OperatorNotDefined {
                op: "||".into(),
                left: proxima_value_type(l).unwrap_or(ProximaType::Boolean),
                right: proxima_value_type(r).unwrap_or(ProximaType::Boolean),
            }),
        },
        And | Or => unreachable!("handled above"),
    }
}

fn eval_and(l: &ProximaValue, r: &ProximaValue) -> Result<ProximaValue, ExprError> {
    use ProximaValue::*;
    // Three-valued logic: TRUE AND NULL = NULL, FALSE AND NULL = FALSE.
    match (l, r) {
        (Boolean(false), _) | (_, Boolean(false)) => Ok(Boolean(false)),
        (Boolean(true), Boolean(true)) => Ok(Boolean(true)),
        (Null, _) | (_, Null) => Ok(Null),
        (a, b) => Err(ExprError::OperatorNotDefined {
            op: "AND".into(),
            left: proxima_value_type(a).unwrap_or(ProximaType::Boolean),
            right: proxima_value_type(b).unwrap_or(ProximaType::Boolean),
        }),
    }
}

fn eval_or(l: &ProximaValue, r: &ProximaValue) -> Result<ProximaValue, ExprError> {
    use ProximaValue::*;
    // Three-valued logic: TRUE OR NULL = TRUE, FALSE OR NULL = NULL.
    match (l, r) {
        (Boolean(true), _) | (_, Boolean(true)) => Ok(Boolean(true)),
        (Boolean(false), Boolean(false)) => Ok(Boolean(false)),
        (Null, _) | (_, Null) => Ok(Null),
        (a, b) => Err(ExprError::OperatorNotDefined {
            op: "OR".into(),
            left: proxima_value_type(a).unwrap_or(ProximaType::Boolean),
            right: proxima_value_type(b).unwrap_or(ProximaType::Boolean),
        }),
    }
}

fn eval_arithmetic(
    op: BinaryOp,
    l: &ProximaValue,
    r: &ProximaValue,
) -> Result<ProximaValue, ExprError> {
    use ProximaValue::*;
    macro_rules! int_op {
        ($lt:ty, $a:expr, $b:expr) => {{
            let res: Result<$lt, ExprError> = match op {
                BinaryOp::Plus => $a.checked_add($b).ok_or(ExprError::ArithmeticOverflow),
                BinaryOp::Minus => $a.checked_sub($b).ok_or(ExprError::ArithmeticOverflow),
                BinaryOp::Mul => $a.checked_mul($b).ok_or(ExprError::ArithmeticOverflow),
                BinaryOp::Div => {
                    if $b == 0 {
                        Err(ExprError::DivisionByZero)
                    } else {
                        $a.checked_div($b).ok_or(ExprError::ArithmeticOverflow)
                    }
                }
                BinaryOp::Mod => {
                    if $b == 0 {
                        Err(ExprError::DivisionByZero)
                    } else {
                        $a.checked_rem($b).ok_or(ExprError::ArithmeticOverflow)
                    }
                }
                _ => unreachable!(),
            };
            res
        }};
    }
    match (l, r) {
        (Int64(a), Int64(b)) => int_op!(i64, *a, *b).map(Int64),
        (Int32(a), Int32(b)) => int_op!(i32, *a, *b).map(Int32),
        (Int16(a), Int16(b)) => int_op!(i16, *a, *b).map(Int16),
        (Int8(a), Int8(b)) => int_op!(i8, *a, *b).map(Int8),
        (UInt64(a), UInt64(b)) => int_op!(u64, *a, *b).map(UInt64),
        (UInt32(a), UInt32(b)) => int_op!(u32, *a, *b).map(UInt32),
        (UInt16(a), UInt16(b)) => int_op!(u16, *a, *b).map(UInt16),
        (UInt8(a), UInt8(b)) => int_op!(u8, *a, *b).map(UInt8),
        (Float64(a), Float64(b)) => Ok(Float64(match op {
            BinaryOp::Plus => a + b,
            BinaryOp::Minus => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => {
                if *b == 0.0 {
                    return Err(ExprError::DivisionByZero);
                }
                a / b
            }
            BinaryOp::Mod => {
                if *b == 0.0 {
                    return Err(ExprError::DivisionByZero);
                }
                a % b
            }
            _ => unreachable!(),
        })),
        (Float32(a), Float32(b)) => Ok(Float32(match op {
            BinaryOp::Plus => a + b,
            BinaryOp::Minus => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => {
                if *b == 0.0 {
                    return Err(ExprError::DivisionByZero);
                }
                a / b
            }
            BinaryOp::Mod => {
                if *b == 0.0 {
                    return Err(ExprError::DivisionByZero);
                }
                a % b
            }
            _ => unreachable!(),
        })),
        _ => {
            // Mixed numeric operands (TPC-H Q19: `1 - l_discount`, Int64 vs
            // Float64): widen both sides to f64 and retry — the same SQL
            // promotion `values_equal`/`compare_ord` already apply, with the
            // same documented precision caveat for very large i64. Terminates:
            // the retry hits the (Float64, Float64) arm above. Non-numeric
            // operands still error loudly.
            if let (Some(lf), Some(rf)) = (try_to_f64(l), try_to_f64(r)) {
                return eval_arithmetic(op, &Float64(lf), &Float64(rf));
            }
            Err(ExprError::OperatorNotDefined {
                op: op.as_str().into(),
                left: proxima_value_type(l).unwrap_or(ProximaType::Boolean),
                right: proxima_value_type(r).unwrap_or(ProximaType::Boolean),
            })
        }
    }
}

/// SQL equality. `NULL = NULL` returns NULL — but `values_equal` is
/// only called when neither side is NULL (the caller handles NULL
/// before invoking this). Internal helper.
fn values_equal(l: &ProximaValue, r: &ProximaValue) -> Result<bool, ExprError> {
    use ProximaValue::*;
    Ok(match (l, r) {
        (Boolean(a), Boolean(b)) => a == b,
        (Int64(a), Int64(b)) => a == b,
        (Int32(a), Int32(b)) => a == b,
        (Int16(a), Int16(b)) => a == b,
        (Int8(a), Int8(b)) => a == b,
        (UInt64(a), UInt64(b)) => a == b,
        (UInt32(a), UInt32(b)) => a == b,
        (UInt16(a), UInt16(b)) => a == b,
        (UInt8(a), UInt8(b)) => a == b,
        (Float64(a), Float64(b)) => a == b,
        (Float32(a), Float32(b)) => a == b,
        (String(a), String(b)) => a == b,
        (Symbol(a), Symbol(b)) => a == b,
        (Binary(a), Binary(b)) => a == b,
        (Date(a), Date(b)) => a == b,
        (Time(a, ua), Time(b, ub)) => ua == ub && a == b,
        (Timestamp(a, ua), Timestamp(b, ub)) => ua == ub && a == b,
        (TimestampTz(a, ua), TimestampTz(b, ub)) => ua == ub && a == b,
        (Uuid(a), Uuid(b)) => a == b,
        (ULID(a), ULID(b)) => a == b,
        (Decimal(a), Decimal(b)) => a == b,
        _ => {
            // A dynamically-typed `Json` scalar operand coerces to its natural type so
            // `doc->'price' = 8` works cast-free (ADR-043 zero-copy idiom). Unwrap one
            // (or both) and re-compare; non-Json operands are unchanged.
            let (lc, rc) = (json_scalar(l), json_scalar(r));
            if lc.is_some() || rc.is_some() {
                return values_equal(lc.as_ref().unwrap_or(l), rc.as_ref().unwrap_or(r));
            }
            // Mixed numeric types: widen both sides to f64 and
            // compare. SQL standard: integer-vs-floating-point
            // and integer-vs-integer comparisons must succeed.
            // (Precision loss for very large i64 values is the
            // documented Phase 3 follow-up.)
            if let (Some(lf), Some(rf)) = (try_to_f64(l), try_to_f64(r)) {
                return Ok(lf == rf);
            }
            return Err(ExprError::OperatorNotDefined {
                op: "=".into(),
                left: proxima_value_type(l).unwrap_or(ProximaType::Boolean),
                right: proxima_value_type(r).unwrap_or(ProximaType::Boolean),
            });
        }
    })
}

/// SQL ordering (returns `Ordering`, callers map to bool for <,<=,>,>=).
/// Same caller contract as `values_equal`: NULL handling is upstream.
fn compare_ord(l: &ProximaValue, r: &ProximaValue) -> Result<std::cmp::Ordering, ExprError> {
    use ProximaValue::*;
    Ok(match (l, r) {
        (Boolean(a), Boolean(b)) => a.cmp(b),
        (Int64(a), Int64(b)) => a.cmp(b),
        (Int32(a), Int32(b)) => a.cmp(b),
        (Int16(a), Int16(b)) => a.cmp(b),
        (Int8(a), Int8(b)) => a.cmp(b),
        (UInt64(a), UInt64(b)) => a.cmp(b),
        (UInt32(a), UInt32(b)) => a.cmp(b),
        (UInt16(a), UInt16(b)) => a.cmp(b),
        (UInt8(a), UInt8(b)) => a.cmp(b),
        (Float64(a), Float64(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (Float32(a), Float32(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (String(a), String(b)) => a.cmp(b),
        (Symbol(a), Symbol(b)) => a.cmp(b),
        (Binary(a), Binary(b)) => a.cmp(b),
        (Date(a), Date(b)) => a.cmp(b),
        (Time(a, ua), Time(b, ub)) if ua == ub => a.cmp(b),
        (Timestamp(a, ua), Timestamp(b, ub)) if ua == ub => a.cmp(b),
        (TimestampTz(a, ua), TimestampTz(b, ub)) if ua == ub => a.cmp(b),
        _ => {
            // `Json` scalar operands coerce to their natural type (mirrors
            // `values_equal`) so ordered comparisons over `->` extractions are cast-free.
            let (lc, rc) = (json_scalar(l), json_scalar(r));
            if lc.is_some() || rc.is_some() {
                return compare_ord(lc.as_ref().unwrap_or(l), rc.as_ref().unwrap_or(r));
            }
            // Mixed numeric types: widen to f64 and compare.
            // Mirrors `values_equal`'s widening path.
            if let (Some(lf), Some(rf)) = (try_to_f64(l), try_to_f64(r)) {
                return Ok(lf.partial_cmp(&rf).unwrap_or(std::cmp::Ordering::Equal));
            }
            return Err(ExprError::OperatorNotDefined {
                op: "compare".into(),
                left: proxima_value_type(l).unwrap_or(ProximaType::Boolean),
                right: proxima_value_type(r).unwrap_or(ProximaType::Boolean),
            });
        }
    })
}

/// Best-effort widening of any numeric ProximaValue to `f64`. Used
/// to support SQL's implicit cross-type numeric comparisons
/// (e.g. `int_col > 25` where `25` parses as Int64 and the column
/// is Int32). Returns `None` for non-numeric values.
/// True for the numeric `ProximaType`s that widen to f64 in mixed-type
/// arithmetic/comparison (the `try_to_f64` value-level counterpart).
fn numeric_type(t: &ProximaType) -> bool {
    matches!(
        t,
        ProximaType::Int8
            | ProximaType::Int16
            | ProximaType::Int32
            | ProximaType::Int64
            | ProximaType::UInt8
            | ProximaType::UInt16
            | ProximaType::UInt32
            | ProximaType::UInt64
            | ProximaType::Float16
            | ProximaType::Float32
            | ProximaType::Float64
    )
}

fn try_to_f64(v: &ProximaValue) -> Option<f64> {
    use ProximaValue as V;
    match v {
        V::Int8(x) => Some(*x as f64),
        V::Int16(x) => Some(*x as f64),
        V::Int32(x) => Some(*x as f64),
        V::Int64(x) => Some(*x as f64),
        V::UInt8(x) => Some(*x as f64),
        V::UInt16(x) => Some(*x as f64),
        V::UInt32(x) => Some(*x as f64),
        V::UInt64(x) => Some(*x as f64),
        V::Float16(x) | V::Float32(x) => Some(*x as f64),
        V::Float64(x) => Some(*x),
        _ => None,
    }
}

/// Unwrap a JSON SCALAR (`number`/`bool`/`string`) into the matching `ProximaValue`,
/// so a dynamically-typed `Json` operand coerces naturally in comparisons and casts —
/// the cast-free `doc->'price' > 8` idiom (ADR-043). Objects, arrays, and JSON `null`
/// have no scalar form and return `None` (the caller then errors loudly per Invariant 1
/// rather than guessing). A JSON integer maps to `Int64`, a non-integer to `Float64`.
fn json_scalar(v: &ProximaValue) -> Option<ProximaValue> {
    let (ProximaValue::Json(j) | ProximaValue::Jsonb(j)) = v else {
        return None;
    };
    match j {
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(ProximaValue::Int64)
            .or_else(|| n.as_f64().map(ProximaValue::Float64)),
        serde_json::Value::Bool(b) => Some(ProximaValue::Boolean(*b)),
        serde_json::Value::String(s) => Some(ProximaValue::String(s.clone())),
        _ => None,
    }
}

fn value_as_str(v: &ProximaValue) -> Result<String, ExprError> {
    match v {
        ProximaValue::String(s) | ProximaValue::Symbol(s) => Ok(s.clone()),
        other => Err(ExprError::TypeMismatch {
            expected: ProximaType::String,
            actual: proxima_value_type(other).unwrap_or(ProximaType::Boolean),
            context: "LIKE / string fn arg".into(),
        }),
    }
}

/// SQL `LIKE` matcher. Supports `%` (any-length) and `_` (single
/// char). Other characters are matched literally. Backslash is not
/// a special character — SQL standard uses `ESCAPE` to redefine,
/// which we defer.
fn like_match(haystack: &str, pattern: &str, case_insensitive: bool) -> Result<bool, ExprError> {
    let (h_chars, p_chars): (Vec<char>, Vec<char>) = if case_insensitive {
        (
            haystack.chars().flat_map(|c| c.to_lowercase()).collect(),
            pattern.chars().flat_map(|c| c.to_lowercase()).collect(),
        )
    } else {
        (haystack.chars().collect(), pattern.chars().collect())
    };
    Ok(like_recursive(&h_chars, &p_chars))
}

fn like_recursive(h: &[char], p: &[char]) -> bool {
    if p.is_empty() {
        return h.is_empty();
    }
    match p[0] {
        '%' => {
            // Match zero or more chars; recurse on each possible split.
            if like_recursive(h, &p[1..]) {
                return true;
            }
            if !h.is_empty() && like_recursive(&h[1..], p) {
                return true;
            }
            false
        }
        '_' => !h.is_empty() && like_recursive(&h[1..], &p[1..]),
        c => !h.is_empty() && h[0] == c && like_recursive(&h[1..], &p[1..]),
    }
}

// =========================================================================
// Cast
// =========================================================================

/// Explicit cast. Supports the common SQL conversions. Unsupported
/// pairs return `UnsupportedCast` so the planner can reject them at
/// type-check time.
pub fn cast_value(v: &ProximaValue, target: &ProximaType) -> Result<ProximaValue, ExprError> {
    use ProximaType as PT;
    use ProximaValue as PV;
    // NULL casts to NULL (preserves type signal via the AST node).
    if matches!(v, PV::Null) {
        return Ok(PV::Null);
    }
    // Identity casts.
    if let Some(src_ty) = proxima_value_type(v)
        && &src_ty == target
    {
        return Ok(v.clone());
    }
    // A `Json` scalar casts by its natural type: `CAST(doc->'price' AS INT)` unwraps the
    // JSON number, then casts that. Objects/arrays/JSON-null have no scalar (json_scalar
    // returns None) and fall through to the UnsupportedCast error.
    if let Some(scalar) = json_scalar(v) {
        return cast_value(&scalar, target);
    }
    macro_rules! to_i64 {
        ($n:expr) => {
            i64::try_from($n).map_err(|_| ExprError::ArithmeticOverflow)
        };
    }
    let out = match (v, target) {
        // Integer widening / narrowing
        (PV::Int8(n), PT::Int16) => PV::Int16(*n as i16),
        (PV::Int8(n), PT::Int32) => PV::Int32(*n as i32),
        (PV::Int8(n), PT::Int64) => PV::Int64(*n as i64),
        (PV::Int16(n), PT::Int32) => PV::Int32(*n as i32),
        (PV::Int16(n), PT::Int64) => PV::Int64(*n as i64),
        (PV::Int32(n), PT::Int64) => PV::Int64(*n as i64),
        (PV::Int16(n), PT::Int8) => {
            PV::Int8(i8::try_from(*n).map_err(|_| ExprError::ArithmeticOverflow)?)
        }
        (PV::Int32(n), PT::Int16) => {
            PV::Int16(i16::try_from(*n).map_err(|_| ExprError::ArithmeticOverflow)?)
        }
        (PV::Int32(n), PT::Int8) => {
            PV::Int8(i8::try_from(*n).map_err(|_| ExprError::ArithmeticOverflow)?)
        }
        (PV::Int64(n), PT::Int32) => {
            PV::Int32(i32::try_from(*n).map_err(|_| ExprError::ArithmeticOverflow)?)
        }
        (PV::Int64(n), PT::Int16) => {
            PV::Int16(i16::try_from(*n).map_err(|_| ExprError::ArithmeticOverflow)?)
        }
        (PV::Int64(n), PT::Int8) => {
            PV::Int8(i8::try_from(*n).map_err(|_| ExprError::ArithmeticOverflow)?)
        }
        // Int → Float
        (PV::Int32(n), PT::Float64) => PV::Float64(*n as f64),
        (PV::Int64(n), PT::Float64) => PV::Float64(*n as f64),
        (PV::Int32(n), PT::Float32) => PV::Float32(*n as f32),
        (PV::Int64(n), PT::Float32) => PV::Float32(*n as f32),
        // Float widening
        (PV::Float32(f), PT::Float64) => PV::Float64(*f as f64),
        (PV::Float64(f), PT::Float32) => PV::Float32(*f as f32),
        // String conversions
        (PV::String(s), PT::Int64) => {
            PV::Int64(s.parse::<i64>().map_err(|_| ExprError::UnsupportedCast {
                from: PT::String,
                to: PT::Int64,
            })?)
        }
        // `::int`/`::smallint`/`::tinyint` (now reaching here, no longer string-stripped):
        // parse to the narrower width. Postgres `int4`→Int32 is the common d06 path.
        (PV::String(s), PT::Int32) => {
            PV::Int32(s.parse::<i32>().map_err(|_| ExprError::UnsupportedCast {
                from: PT::String,
                to: PT::Int32,
            })?)
        }
        (PV::String(s), PT::Int16) => {
            PV::Int16(s.parse::<i16>().map_err(|_| ExprError::UnsupportedCast {
                from: PT::String,
                to: PT::Int16,
            })?)
        }
        (PV::String(s), PT::Float32) => {
            PV::Float32(s.parse::<f32>().map_err(|_| ExprError::UnsupportedCast {
                from: PT::String,
                to: PT::Float32,
            })?)
        }
        (PV::String(s), PT::Float64) => {
            PV::Float64(s.parse::<f64>().map_err(|_| ExprError::UnsupportedCast {
                from: PT::String,
                to: PT::Float64,
            })?)
        }
        // PostgreSQL's boolean input: the accepted spellings (case-insensitive,
        // surrounding whitespace ignored); anything else is 22P02, never a
        // silent `false`.
        (PV::String(s), PT::Boolean) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "t" | "yes" | "y" | "on" | "1" => PV::Boolean(true),
            "false" | "f" | "no" | "n" | "off" | "0" => PV::Boolean(false),
            _ => {
                return Err(ExprError::InvalidTextRepresentation {
                    value: s.clone(),
                    to: PT::Boolean,
                });
            }
        },
        // Number → string
        (PV::Int64(n), PT::String) => PV::String(n.to_string()),
        (PV::Int32(n), PT::String) => PV::String(n.to_string()),
        (PV::Float64(f), PT::String) => PV::String(f.to_string()),
        (PV::Float32(f), PT::String) => PV::String(f.to_string()),
        (PV::Boolean(b), PT::String) => PV::String((if *b { "true" } else { "false" }).into()),
        // Identity time-unit conversions retain the value; cross-unit
        // (ms → ns etc) is deferred to a Phase 3 helper.
        (PV::Timestamp(t, u), PT::Timestamp(uu)) if u == uu => PV::Timestamp(*t, *u),
        (PV::Date(d), PT::Date) => PV::Date(*d),
        // Date → Timestamp
        (PV::Date(d), PT::Timestamp(TimeUnit::Millisecond)) => {
            let ms = to_i64!(*d as i64 * 86_400_000)?;
            PV::Timestamp(ms, TimeUnit::Millisecond)
        }
        // Fallback
        _ => {
            let src = proxima_value_type(v).unwrap_or(PT::Boolean);
            return Err(ExprError::UnsupportedCast {
                from: src,
                to: target.clone(),
            });
        }
    };
    Ok(out)
}

// =========================================================================
// Builder helpers (ergonomics — keep callers terse)
// =========================================================================

impl Expr {
    pub fn literal(value: ProximaValue) -> Self {
        let ty = proxima_value_type(&value).unwrap_or(ProximaType::String);
        Expr::Literal { value, ty }
    }

    pub fn null(ty: ProximaType) -> Self {
        Expr::Literal {
            value: ProximaValue::Null,
            ty,
        }
    }

    pub fn column(col: ColumnRef) -> Self {
        Expr::Column(col)
    }

    pub fn bin(op: BinaryOp, left: Expr, right: Expr) -> Self {
        Expr::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn unary(op: UnaryOp, expr: Expr) -> Self {
        Expr::UnaryOp {
            op,
            expr: Box::new(expr),
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_int_str_bool() -> RelationalSchema {
        RelationalSchema::new(vec![
            ColumnInfo::new("id", ProximaType::Int64, false),
            ColumnInfo::new("name", ProximaType::String, true),
            ColumnInfo::new("active", ProximaType::Boolean, true),
        ])
    }

    fn row(id: i64, name: Option<&str>, active: Option<bool>) -> RelationalRow {
        vec![
            ProximaValue::Int64(id),
            name.map(|s| ProximaValue::String(s.into()))
                .unwrap_or(ProximaValue::Null),
            active
                .map(ProximaValue::Boolean)
                .unwrap_or(ProximaValue::Null),
        ]
    }

    // --- Schema -----------------------------------------------------------

    #[test]
    fn schema_resolves_known_column() {
        let s = schema_int_str_bool();
        let c = s.resolve_column("name").unwrap();
        assert_eq!(c.ordinal, 1);
        assert_eq!(c.ty, ProximaType::String);
        assert!(c.nullable);
    }

    #[test]
    fn schema_rejects_unknown_column() {
        let s = schema_int_str_bool();
        let e = s.resolve_column("nope").unwrap_err();
        assert!(matches!(e, ExprError::UnknownColumn { .. }));
    }

    #[test]
    fn schema_projection_preserves_order() {
        let s = schema_int_str_bool();
        let refs = vec![
            s.resolve_column("name").unwrap(),
            s.resolve_column("id").unwrap(),
        ];
        let p = s.project(&refs);
        assert_eq!(p.columns[0].name, "name");
        assert_eq!(p.columns[1].name, "id");
    }

    // --- Literal / column eval ------------------------------------------

    #[test]
    fn literal_evaluates_to_itself() {
        let e = Expr::literal(ProximaValue::Int64(42));
        let v = e.eval(&row(1, None, None), &NoFunctions).unwrap();
        assert_eq!(v, ProximaValue::Int64(42));
    }

    #[test]
    fn column_evaluates_to_row_position() {
        let s = schema_int_str_bool();
        let e = Expr::column(s.resolve_column("id").unwrap());
        let v = e.eval(&row(7, None, None), &NoFunctions).unwrap();
        assert_eq!(v, ProximaValue::Int64(7));
    }

    // --- Arithmetic ------------------------------------------------------

    #[test]
    fn integer_arithmetic_works() {
        let e = Expr::bin(
            BinaryOp::Plus,
            Expr::literal(ProximaValue::Int64(2)),
            Expr::literal(ProximaValue::Int64(3)),
        );
        let v = e.eval(&Vec::new(), &NoFunctions).unwrap();
        assert_eq!(v, ProximaValue::Int64(5));
    }

    #[test]
    fn integer_arithmetic_overflow_is_caught() {
        let e = Expr::bin(
            BinaryOp::Plus,
            Expr::literal(ProximaValue::Int64(i64::MAX)),
            Expr::literal(ProximaValue::Int64(1)),
        );
        let err = e.eval(&Vec::new(), &NoFunctions).unwrap_err();
        assert!(matches!(err, ExprError::ArithmeticOverflow));
    }

    #[test]
    fn division_by_zero_is_caught() {
        let e = Expr::bin(
            BinaryOp::Div,
            Expr::literal(ProximaValue::Int64(1)),
            Expr::literal(ProximaValue::Int64(0)),
        );
        let err = e.eval(&Vec::new(), &NoFunctions).unwrap_err();
        assert!(matches!(err, ExprError::DivisionByZero));
    }

    #[test]
    fn arithmetic_propagates_null() {
        let e = Expr::bin(
            BinaryOp::Plus,
            Expr::literal(ProximaValue::Int64(2)),
            Expr::null(ProximaType::Int64),
        );
        let v = e.eval(&Vec::new(), &NoFunctions).unwrap();
        assert_eq!(v, ProximaValue::Null);
    }

    // --- Three-valued logic ---------------------------------------------

    #[test]
    fn and_three_valued_logic() {
        let t = || Expr::literal(ProximaValue::Boolean(true));
        let f = || Expr::literal(ProximaValue::Boolean(false));
        let n = || Expr::null(ProximaType::Boolean);
        let cases = [
            (
                Expr::bin(BinaryOp::And, t(), t()),
                ProximaValue::Boolean(true),
            ),
            (
                Expr::bin(BinaryOp::And, t(), f()),
                ProximaValue::Boolean(false),
            ),
            (
                Expr::bin(BinaryOp::And, f(), n()),
                ProximaValue::Boolean(false),
            ),
            (
                Expr::bin(BinaryOp::And, n(), f()),
                ProximaValue::Boolean(false),
            ),
            (Expr::bin(BinaryOp::And, t(), n()), ProximaValue::Null),
            (Expr::bin(BinaryOp::And, n(), t()), ProximaValue::Null),
            (Expr::bin(BinaryOp::And, n(), n()), ProximaValue::Null),
        ];
        for (e, expected) in cases {
            assert_eq!(e.eval(&Vec::new(), &NoFunctions).unwrap(), expected);
        }
    }

    #[test]
    fn or_three_valued_logic() {
        let t = || Expr::literal(ProximaValue::Boolean(true));
        let f = || Expr::literal(ProximaValue::Boolean(false));
        let n = || Expr::null(ProximaType::Boolean);
        let cases = [
            (
                Expr::bin(BinaryOp::Or, t(), n()),
                ProximaValue::Boolean(true),
            ),
            (
                Expr::bin(BinaryOp::Or, n(), t()),
                ProximaValue::Boolean(true),
            ),
            (Expr::bin(BinaryOp::Or, f(), n()), ProximaValue::Null),
            (Expr::bin(BinaryOp::Or, n(), f()), ProximaValue::Null),
            (
                Expr::bin(BinaryOp::Or, f(), f()),
                ProximaValue::Boolean(false),
            ),
            (Expr::bin(BinaryOp::Or, n(), n()), ProximaValue::Null),
        ];
        for (e, expected) in cases {
            assert_eq!(e.eval(&Vec::new(), &NoFunctions).unwrap(), expected);
        }
    }

    #[test]
    fn not_three_valued_logic() {
        let e = Expr::unary(UnaryOp::Not, Expr::literal(ProximaValue::Boolean(true)));
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(false)
        );
        let e = Expr::unary(UnaryOp::Not, Expr::null(ProximaType::Boolean));
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Null
        );
    }

    // --- Comparison ------------------------------------------------------

    #[test]
    fn equality_returns_null_when_either_side_null() {
        let e = Expr::bin(
            BinaryOp::Eq,
            Expr::literal(ProximaValue::Int64(1)),
            Expr::null(ProximaType::Int64),
        );
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Null
        );
    }

    #[test]
    fn is_null_does_not_propagate() {
        let e = Expr::IsNull {
            expr: Box::new(Expr::null(ProximaType::Int64)),
            not: false,
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(true)
        );
        let e = Expr::IsNull {
            expr: Box::new(Expr::literal(ProximaValue::Int64(1))),
            not: false,
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(false)
        );
    }

    #[test]
    fn is_not_null_inverts() {
        let e = Expr::IsNull {
            expr: Box::new(Expr::literal(ProximaValue::Int64(1))),
            not: true,
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(true)
        );
    }

    #[test]
    fn ordering_works_for_strings_and_numbers() {
        let lt = Expr::bin(
            BinaryOp::Lt,
            Expr::literal(ProximaValue::String("a".into())),
            Expr::literal(ProximaValue::String("b".into())),
        );
        assert_eq!(
            lt.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(true)
        );
        let gt = Expr::bin(
            BinaryOp::Gt,
            Expr::literal(ProximaValue::Int64(5)),
            Expr::literal(ProximaValue::Int64(3)),
        );
        assert_eq!(
            gt.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(true)
        );
    }

    // --- BETWEEN / IN / LIKE -------------------------------------------

    #[test]
    fn between_inclusive_bounds() {
        let e = Expr::Between {
            expr: Box::new(Expr::literal(ProximaValue::Int64(5))),
            low: Box::new(Expr::literal(ProximaValue::Int64(1))),
            high: Box::new(Expr::literal(ProximaValue::Int64(5))),
            not: false,
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(true)
        );
    }

    #[test]
    fn in_list_membership() {
        let e = Expr::In {
            expr: Box::new(Expr::literal(ProximaValue::Int64(2))),
            list: vec![
                Expr::literal(ProximaValue::Int64(1)),
                Expr::literal(ProximaValue::Int64(2)),
                Expr::literal(ProximaValue::Int64(3)),
            ],
            not: false,
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(true)
        );
        let miss = Expr::In {
            expr: Box::new(Expr::literal(ProximaValue::Int64(99))),
            list: vec![
                Expr::literal(ProximaValue::Int64(1)),
                Expr::literal(ProximaValue::Int64(2)),
            ],
            not: false,
        };
        assert_eq!(
            miss.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(false)
        );
    }

    #[test]
    fn in_list_with_null_member_returns_null_on_miss() {
        // SQL: `x IN (a, NULL)` returns NULL when x ≠ a.
        let e = Expr::In {
            expr: Box::new(Expr::literal(ProximaValue::Int64(99))),
            list: vec![
                Expr::literal(ProximaValue::Int64(1)),
                Expr::null(ProximaType::Int64),
            ],
            not: false,
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Null
        );
    }

    #[test]
    fn like_pattern_matches_percent_and_underscore() {
        let e = Expr::Like {
            expr: Box::new(Expr::literal(ProximaValue::String("hello world".into()))),
            pattern: Box::new(Expr::literal(ProximaValue::String("h%world".into()))),
            not: false,
            case_insensitive: false,
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(true)
        );
        let e = Expr::Like {
            expr: Box::new(Expr::literal(ProximaValue::String("cat".into()))),
            pattern: Box::new(Expr::literal(ProximaValue::String("c_t".into()))),
            not: false,
            case_insensitive: false,
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(true)
        );
        let e = Expr::Like {
            expr: Box::new(Expr::literal(ProximaValue::String("hello".into()))),
            pattern: Box::new(Expr::literal(ProximaValue::String("HELLO".into()))),
            not: false,
            case_insensitive: true,
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Boolean(true)
        );
    }

    // --- CASE / COALESCE / NULLIF --------------------------------------

    #[test]
    fn case_returns_first_matching_branch() {
        let e = Expr::Case {
            branches: vec![
                (
                    Expr::literal(ProximaValue::Boolean(false)),
                    Expr::literal(ProximaValue::String("a".into())),
                ),
                (
                    Expr::literal(ProximaValue::Boolean(true)),
                    Expr::literal(ProximaValue::String("b".into())),
                ),
            ],
            otherwise: Some(Box::new(Expr::literal(ProximaValue::String("c".into())))),
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::String("b".into())
        );
    }

    #[test]
    fn case_else_branch_used_when_no_match() {
        let e = Expr::Case {
            branches: vec![(
                Expr::literal(ProximaValue::Boolean(false)),
                Expr::literal(ProximaValue::Int64(1)),
            )],
            otherwise: Some(Box::new(Expr::literal(ProximaValue::Int64(99)))),
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Int64(99)
        );
    }

    #[test]
    fn case_no_else_returns_null() {
        let e = Expr::Case {
            branches: vec![(
                Expr::literal(ProximaValue::Boolean(false)),
                Expr::literal(ProximaValue::Int64(1)),
            )],
            otherwise: None,
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Null
        );
    }

    #[test]
    fn coalesce_returns_first_non_null() {
        let e = Expr::Coalesce(vec![
            Expr::null(ProximaType::Int64),
            Expr::null(ProximaType::Int64),
            Expr::literal(ProximaValue::Int64(42)),
            Expr::literal(ProximaValue::Int64(99)),
        ]);
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Int64(42)
        );
    }

    #[test]
    fn coalesce_all_null_returns_null() {
        let e = Expr::Coalesce(vec![
            Expr::null(ProximaType::Int64),
            Expr::null(ProximaType::Int64),
        ]);
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Null
        );
    }

    #[test]
    fn nullif_returns_null_when_equal() {
        let e = Expr::NullIf {
            left: Box::new(Expr::literal(ProximaValue::Int64(5))),
            right: Box::new(Expr::literal(ProximaValue::Int64(5))),
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Null
        );
    }

    #[test]
    fn nullif_returns_left_when_different() {
        let e = Expr::NullIf {
            left: Box::new(Expr::literal(ProximaValue::Int64(5))),
            right: Box::new(Expr::literal(ProximaValue::Int64(6))),
        };
        assert_eq!(
            e.eval(&Vec::new(), &NoFunctions).unwrap(),
            ProximaValue::Int64(5)
        );
    }

    // --- Cast -----------------------------------------------------------

    #[test]
    fn cast_widening_int_works() {
        let v = cast_value(&ProximaValue::Int32(123), &ProximaType::Int64).unwrap();
        assert_eq!(v, ProximaValue::Int64(123));
    }

    #[test]
    fn cast_narrowing_overflow_errors() {
        let e = cast_value(&ProximaValue::Int64(i64::MAX), &ProximaType::Int8).unwrap_err();
        assert!(matches!(e, ExprError::ArithmeticOverflow));
    }

    #[test]
    fn cast_int_to_string() {
        let v = cast_value(&ProximaValue::Int64(42), &ProximaType::String).unwrap();
        assert_eq!(v, ProximaValue::String("42".into()));
    }

    #[test]
    fn cast_string_to_int_works_and_fails() {
        let v = cast_value(&ProximaValue::String("42".into()), &ProximaType::Int64).unwrap();
        assert_eq!(v, ProximaValue::Int64(42));
        let e = cast_value(&ProximaValue::String("abc".into()), &ProximaType::Int64).unwrap_err();
        assert!(matches!(e, ExprError::UnsupportedCast { .. }));
    }

    #[test]
    fn cast_null_stays_null() {
        let v = cast_value(&ProximaValue::Null, &ProximaType::Int64).unwrap();
        assert_eq!(v, ProximaValue::Null);
    }

    #[test]
    fn json_scalar_coerces_in_cast_and_compare() {
        use ProximaValue as V;
        // CAST(json number AS int) unwraps the JSON scalar to its natural type.
        let ten = V::Json(serde_json::json!(10));
        assert_eq!(cast_value(&ten, &ProximaType::Int64).unwrap(), V::Int64(10));
        // Cast-free comparison idiom: a Json scalar coerces against a native scalar.
        assert!(values_equal(&V::Json(serde_json::json!(8)), &V::Int64(8)).unwrap());
        assert_eq!(
            compare_ord(&V::Json(serde_json::json!(10)), &V::Int64(8)).unwrap(),
            std::cmp::Ordering::Greater
        );
        assert!(values_equal(&V::Json(serde_json::json!(true)), &V::Boolean(true)).unwrap());
        // An object/array has no scalar form → cast errors loudly (never silent).
        assert!(cast_value(&V::Json(serde_json::json!({ "a": 1 })), &ProximaType::Int64).is_err());
    }

    // --- type_check ----------------------------------------------------

    #[test]
    fn type_check_passes_on_valid_column_ref() {
        let s = schema_int_str_bool();
        let e = Expr::column(s.resolve_column("id").unwrap());
        assert!(e.type_check(&s).is_ok());
    }

    #[test]
    fn type_check_catches_ordinal_out_of_range() {
        let s = schema_int_str_bool();
        let bad = Expr::Column(ColumnRef {
            name: "x".into(),
            ordinal: 99,
            ty: ProximaType::Int64,
            nullable: false,
        });
        let err = bad.type_check(&s).unwrap_err();
        assert!(matches!(err, ExprError::ColumnOrdinalOutOfRange { .. }));
    }

    #[test]
    fn type_check_catches_type_mismatch_on_column_ref() {
        let s = schema_int_str_bool();
        let bad = Expr::Column(ColumnRef {
            name: "id".into(),
            ordinal: 0,
            ty: ProximaType::String, // wrong — actual is Int64
            nullable: false,
        });
        let err = bad.type_check(&s).unwrap_err();
        assert!(matches!(err, ExprError::TypeMismatch { .. }));
    }

    #[test]
    fn type_check_walks_into_subexpressions() {
        let s = schema_int_str_bool();
        let bad = Expr::bin(
            BinaryOp::Plus,
            Expr::column(s.resolve_column("id").unwrap()),
            // Bogus column ref with out-of-range ordinal.
            Expr::Column(ColumnRef {
                name: "x".into(),
                ordinal: 99,
                ty: ProximaType::Int64,
                nullable: false,
            }),
        );
        assert!(bad.type_check(&s).is_err());
    }

    // --- result_type ----------------------------------------------------

    #[test]
    fn result_type_for_common_shapes() {
        assert_eq!(
            Expr::literal(ProximaValue::Int64(1)).result_type(),
            ProximaType::Int64
        );
        assert_eq!(
            Expr::bin(
                BinaryOp::Eq,
                Expr::literal(ProximaValue::Int64(1)),
                Expr::literal(ProximaValue::Int64(2))
            )
            .result_type(),
            ProximaType::Boolean
        );
        assert_eq!(
            Expr::unary(UnaryOp::Not, Expr::literal(ProximaValue::Boolean(true))).result_type(),
            ProximaType::Boolean
        );
    }

    /// TD-REL-LOWER-1 / TPC-H Q19: mixed numeric arithmetic (`1 - l_discount`,
    /// Int64 vs Float64) widens both sides to f64 — the same promotion
    /// comparisons already apply — instead of `OperatorNotDefined`.
    #[test]
    fn mixed_numeric_arithmetic_widens_to_f64() {
        let cases = [
            (
                BinaryOp::Minus,
                ProximaValue::Int64(1),
                ProximaValue::Float64(0.06),
                0.94,
            ),
            (
                BinaryOp::Mul,
                ProximaValue::Float64(2.5),
                ProximaValue::Int32(4),
                10.0,
            ),
            (
                BinaryOp::Plus,
                ProximaValue::Int64(3),
                ProximaValue::Float32(0.5),
                3.5,
            ),
        ];
        for (op, l, r, want) in cases {
            let got = eval_binary(op, &l, &r).expect("mixed numeric arithmetic must evaluate");
            let ProximaValue::Float64(v) = got else {
                panic!("expected Float64, got {got:?}");
            };
            assert!((v - want).abs() < 1e-9, "{op:?}: got {v}, want {want}");
        }
        // Non-numeric mixes still error loudly.
        assert!(
            eval_binary(
                BinaryOp::Minus,
                &ProximaValue::String("a".into()),
                &ProximaValue::Int64(1)
            )
            .is_err()
        );
        // And the static result type agrees with the widened evaluation.
        assert_eq!(
            Expr::bin(
                BinaryOp::Minus,
                Expr::literal(ProximaValue::Int64(1)),
                Expr::literal(ProximaValue::Float64(0.06))
            )
            .result_type(),
            ProximaType::Float64
        );
    }

    // --- FunctionRegistry extension -----------------------------------

    struct UpperFn;
    impl FunctionRegistry for UpperFn {
        fn dispatch(
            &self,
            name: &str,
            args: &[ProximaValue],
        ) -> Option<Result<ProximaValue, ExprError>> {
            if name != "upper" {
                return None;
            }
            if args.len() != 1 {
                return Some(Err(ExprError::WrongFunctionArity {
                    name: name.into(),
                    expected: 1,
                    got: args.len(),
                }));
            }
            match &args[0] {
                ProximaValue::String(s) => Some(Ok(ProximaValue::String(s.to_uppercase()))),
                ProximaValue::Null => Some(Ok(ProximaValue::Null)),
                other => Some(Err(ExprError::TypeMismatch {
                    expected: ProximaType::String,
                    actual: proxima_value_type(other).unwrap_or(ProximaType::Boolean),
                    context: "upper".into(),
                })),
            }
        }
    }

    #[test]
    fn func_call_via_registry() {
        let e = Expr::FuncCall {
            name: "upper".into(),
            args: vec![Expr::literal(ProximaValue::String("hello".into()))],
            return_ty: ProximaType::String,
        };
        let v = e.eval(&Vec::new(), &UpperFn).unwrap();
        assert_eq!(v, ProximaValue::String("HELLO".into()));
    }

    #[test]
    fn func_call_unknown_function_errors() {
        let e = Expr::FuncCall {
            name: "nope".into(),
            args: vec![],
            return_ty: ProximaType::Int64,
        };
        let err = e.eval(&Vec::new(), &NoFunctions).unwrap_err();
        assert!(matches!(err, ExprError::UnknownFunction { .. }));
    }

    // --- End-to-end realistic predicate -------------------------------

    #[test]
    fn realistic_where_predicate() {
        // WHERE id > 5 AND (name IS NOT NULL OR active = TRUE)
        let s = schema_int_str_bool();
        let id = s.resolve_column("id").unwrap();
        let name = s.resolve_column("name").unwrap();
        let active = s.resolve_column("active").unwrap();
        let pred = Expr::bin(
            BinaryOp::And,
            Expr::bin(
                BinaryOp::Gt,
                Expr::column(id.clone()),
                Expr::literal(ProximaValue::Int64(5)),
            ),
            Expr::bin(
                BinaryOp::Or,
                Expr::IsNull {
                    expr: Box::new(Expr::column(name.clone())),
                    not: true,
                },
                Expr::bin(
                    BinaryOp::Eq,
                    Expr::column(active.clone()),
                    Expr::literal(ProximaValue::Boolean(true)),
                ),
            ),
        );
        // type_check passes
        assert!(pred.type_check(&s).is_ok());

        // Row (id=10, name="alice", active=NULL) → true
        assert_eq!(
            pred.eval(&row(10, Some("alice"), None), &NoFunctions)
                .unwrap(),
            ProximaValue::Boolean(true)
        );
        // Row (id=3, name="alice", active=true) → false (id > 5 fails)
        assert_eq!(
            pred.eval(&row(3, Some("alice"), Some(true)), &NoFunctions)
                .unwrap(),
            ProximaValue::Boolean(false)
        );
        // Row (id=10, name=NULL, active=NULL) → NULL (then short-circuits
        // to FALSE? actually: `name IS NOT NULL` = false, `active = true`
        // = NULL, OR = NULL; AND of TRUE and NULL = NULL).
        assert_eq!(
            pred.eval(&row(10, None, None), &NoFunctions).unwrap(),
            ProximaValue::Null
        );
    }
}
