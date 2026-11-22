//! ADR-042 (amended 2026-11-20) / TD-PAXMETA-1: per-block pruning statistics
//! for a PAX segment's filterable columns.
//!
//! The writer records, for every Region-D block and every filterable (P-Shred
//! spec) column, a summary of the literal the filter evaluator resolves that
//! column to on each row ([`resolve_filter_literal`] — the same value
//! `evaluate_filter_proxima` compares, so absent, present-null and
//! wrong-typed values are told apart, unlike the shred stripe whose null
//! conflates them). The summary is:
//!
//! * the row count, the absent count and the present-null count;
//! * the set of JSON classes among the non-null literals;
//! * bounds when every non-null literal is of one comparable class: exact
//!   `i64` bounds for an all-integer block, `f64` bounds for an all-numeric
//!   block, and string bounds for an all-string block (the lower bound is a
//!   prefix of at most [`STRING_BOUND_PREFIX_BYTES`]; an upper bound longer
//!   than that is NOT stored — a truncated upper bound never prunes, the
//!   TD-SPINE-4 rule);
//! * the complete set of distinct non-null literals when it has at most
//!   [`MAX_DISTINCT_VALUES`] entries of at most [`MAX_DISTINCT_VALUE_BYTES`]
//!   encoded bytes each — DROPPED (never truncated) when either cap is
//!   exceeded;
//! * for an all-string block whose distinct set was dropped, a bloom filter
//!   over the strings (the PAX block bloom: `fnv1a` keys, 3 salted probes,
//!   [`BLOOM_BITS_PER_KEY`] bits per key) when it has at most
//!   [`BLOOM_MAX_KEYS`] distinct strings.
//!
//! The stats ride an optional footer section of the segment (written at flush
//! and compaction, in the same pass as Region D) and are copied verbatim into
//! the collection segment directory (PXCD v3), so a filtered query can prune
//! blocks and whole segments without reading a footer.
//!
//! ## Soundness
//!
//! [`BlockPruneStatsTable::block_may_match`] returns `false` only when no row
//! of the block can satisfy the filter. Absent rows, null rows and the
//! distinct-set values are tested by running the evaluator itself
//! ([`evaluate_filter_resolved`]) on that literal, so those verdicts are the
//! evaluator's by construction, for every operator. Bounds and blooms are used
//! only when every non-null literal of the block is in the literal's own class
//! (a class mismatch keeps the block — under the evaluator's total order a
//! string outranks any number, so cross-class range verdicts are never
//! inferred), and only for `=`, `IN`, `<`, `<=`, `>`, `>=` and `BETWEEN`.
//! Float comparisons are strict and equality allows the evaluator's relative
//! epsilon (plus rounding slack). A negation, an empty `OR`, an unknown column,
//! a block without stats and anything else not covered keeps the block.
//!
//! ## Wire format (little-endian, deterministic)
//!
//! ```text
//! [version u8 = 1][n_columns u16]
//!   n_columns × [name_len u16][name utf-8][column_id i32]
//! [n_blocks u32]
//!   n_blocks × n_columns × column stats:
//!     [flags u8][rows u32][absent u32][nulls u32][classes u8]
//!     flags & INT_BOUNDS   → [min i64][max i64]
//!     flags & NUM_BOUNDS   → [min f64][max f64]
//!     flags & STR_MIN      → [len u8][bytes]
//!     flags & STR_MAX      → [len u8][bytes]
//!     flags & DISTINCT     → [n u8] n × [len u16][MessagePack JSON value]
//!     flags & BLOOM        → [len u16][bloom bytes]
//! ```
//!
//! Parsing is fail-closed: an unknown version, unknown flag bits, a
//! truncated or trailing byte, a count that does not add up, or a value that
//! breaks a cap is an error (callers then ignore the stats and keep every
//! block).

#![forbid(unsafe_code)]

use std::collections::HashSet;

use anyhow::{Result, bail};
use proximadb_filter_expression::{ComparisonOperator, FilterExpression};
use proximadb_records::ProximaTree;
use proximadb_search_types::sql_value_filter::{evaluate_filter_resolved, resolve_filter_literal};
use serde_json::Value;

use crate::format_splits::{ColumnBounds, ScalarPredicate, ScalarValue};

/// Wire version of the stats table.
pub const PRUNE_STATS_VERSION: u8 = 1;
/// Largest distinct-value set kept per column per block (dropped above).
pub const MAX_DISTINCT_VALUES: usize = 8;
/// Largest MessagePack encoding of one distinct value (the set is dropped
/// when any value is larger).
pub const MAX_DISTINCT_VALUE_BYTES: usize = 32;
/// String bound prefix length. The lower bound is truncated to it (still a
/// valid lower bound); a longer upper bound is not stored.
pub const STRING_BOUND_PREFIX_BYTES: usize = 16;
/// Largest distinct-string count a bloom is built for.
pub const BLOOM_MAX_KEYS: usize = 128;
/// Bloom bits per distinct string key (~1.7% false positives at 3 probes).
pub const BLOOM_BITS_PER_KEY: usize = 10;
const BLOOM_MIN_BYTES: usize = 8;
/// Largest number of filterable columns carried (more are not summarized).
pub const MAX_COLUMNS: usize = 64;

const CLASS_BOOL: u8 = 1;
const CLASS_INT: u8 = 2;
const CLASS_FLOAT: u8 = 4;
const CLASS_STRING: u8 = 8;
const CLASS_OTHER: u8 = 16;
const CLASS_ALL: u8 = CLASS_BOOL | CLASS_INT | CLASS_FLOAT | CLASS_STRING | CLASS_OTHER;

const F_INT_BOUNDS: u8 = 1;
const F_NUM_BOUNDS: u8 = 2;
const F_STR_MIN: u8 = 4;
const F_STR_MAX: u8 = 8;
const F_DISTINCT: u8 = 16;
const F_BLOOM: u8 = 32;
const F_ALL: u8 = F_INT_BOUNDS | F_NUM_BOUNDS | F_STR_MIN | F_STR_MAX | F_DISTINCT | F_BLOOM;

/// One filterable column the table summarizes: its filter field name and
/// the P-Shred column id the segment stores it under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PruneColumn {
    pub name: String,
    pub column_id: i32,
}

/// One column's statistics within one block.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnBlockStats {
    pub rows: u32,
    /// Rows whose field is absent.
    pub absent: u32,
    /// Rows whose field is present and JSON null.
    pub nulls: u32,
    /// Bitmask of the JSON classes among the non-null literals.
    pub classes: u8,
    /// Exact bounds, when every non-null literal is an `i64` integer.
    pub int_bounds: Option<(i64, i64)>,
    /// `f64` bounds, when every non-null literal is a number.
    pub num_bounds: Option<(f64, f64)>,
    /// String lower bound (a prefix of the minimum), when every non-null
    /// literal is a string.
    pub str_min: Option<String>,
    /// String upper bound — only when the maximum fits
    /// [`STRING_BOUND_PREFIX_BYTES`] untruncated.
    pub str_max: Option<String>,
    /// The complete distinct non-null literal set (`None` = dropped).
    pub distinct: Option<Vec<Value>>,
    /// Bloom over the strings of an all-string block (`None` = none).
    pub bloom: Option<Vec<u8>>,
}

impl ColumnBlockStats {
    fn non_null(&self) -> u32 {
        self.rows
            .saturating_sub(self.absent)
            .saturating_sub(self.nulls)
    }
}

/// Per-block statistics for a segment's filterable columns.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockPruneStatsTable {
    pub columns: Vec<PruneColumn>,
    /// `blocks[b][c]` = column `c`'s stats in block `b`.
    pub blocks: Vec<Vec<ColumnBlockStats>>,
}

impl BlockPruneStatsTable {
    /// Number of blocks summarized.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Fail closed unless the table describes exactly these blocks (one entry
    /// per block, each column's row count equal to the block's).
    pub fn check_block_rows(&self, block_rows: &[u32]) -> Result<()> {
        if self.blocks.len() != block_rows.len() {
            bail!(
                "prune stats: {} blocks summarized, segment has {}",
                self.blocks.len(),
                block_rows.len()
            );
        }
        for (b, (stats, rows)) in self.blocks.iter().zip(block_rows).enumerate() {
            if stats.iter().any(|s| s.rows != *rows) {
                bail!("prune stats: block {b} row count disagrees with the block table");
            }
        }
        Ok(())
    }

    /// Whether block `block` may hold a row matching `filter`. `false` only
    /// when provably not (see the module docs); a block index outside the
    /// table keeps the block.
    pub fn block_may_match(&self, block: usize, filter: &FilterExpression) -> bool {
        match self.blocks.get(block) {
            Some(stats) => self.may_match(stats, filter),
            None => true,
        }
    }

    fn may_match(&self, stats: &[ColumnBlockStats], filter: &FilterExpression) -> bool {
        match filter {
            FilterExpression::And(children) => children.iter().all(|c| self.may_match(stats, c)),
            // `OR []` matches nothing in the evaluator; keep anyway (nothing
            // is gained by pruning on a degenerate filter).
            FilterExpression::Or(children) => {
                children.is_empty() || children.iter().any(|c| self.may_match(stats, c))
            }
            // Negation is not bounded by these summaries: keep.
            FilterExpression::Not(_) => true,
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => match self.columns.iter().position(|c| &c.name == field) {
                Some(idx) => match stats.get(idx) {
                    Some(column) => leaf_may_match(column, filter, operator, value),
                    None => true,
                },
                None => true,
            },
        }
    }

    /// Serialize (see the module docs for the layout).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        out.push(PRUNE_STATS_VERSION);
        let n_columns = u16::try_from(self.columns.len())
            .ok()
            .filter(|n| usize::from(*n) <= MAX_COLUMNS)
            .ok_or_else(|| anyhow::anyhow!("prune stats: too many columns"))?;
        out.extend_from_slice(&n_columns.to_le_bytes());
        for column in &self.columns {
            let len = u16::try_from(column.name.len())
                .map_err(|_| anyhow::anyhow!("prune stats: column name too long"))?;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(column.name.as_bytes());
            out.extend_from_slice(&column.column_id.to_le_bytes());
        }
        let n_blocks = u32::try_from(self.blocks.len())
            .map_err(|_| anyhow::anyhow!("prune stats: too many blocks"))?;
        out.extend_from_slice(&n_blocks.to_le_bytes());
        for block in &self.blocks {
            if block.len() != self.columns.len() {
                bail!("prune stats: block column count disagrees with the column table");
            }
            for stats in block {
                write_column_stats(&mut out, stats)?;
            }
        }
        Ok(out)
    }

    /// Parse; fail-closed on anything malformed.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut p = 0usize;
        let version = take_u8(bytes, &mut p)?;
        if version != PRUNE_STATS_VERSION {
            bail!("unsupported prune stats version {version}");
        }
        let n_columns = usize::from(take_u16(bytes, &mut p)?);
        if n_columns > MAX_COLUMNS {
            bail!("prune stats: {n_columns} columns exceeds {MAX_COLUMNS}");
        }
        let mut columns = Vec::with_capacity(n_columns);
        let mut names = HashSet::new();
        for _ in 0..n_columns {
            let len = usize::from(take_u16(bytes, &mut p)?);
            let name = String::from_utf8(take(bytes, &mut p, len)?.to_vec())
                .map_err(|_| anyhow::anyhow!("prune stats: column name is not UTF-8"))?;
            let column_id = i32::from_le_bytes(take(bytes, &mut p, 4)?.try_into()?);
            if !names.insert(name.clone()) {
                bail!("prune stats: duplicate column {name}");
            }
            columns.push(PruneColumn { name, column_id });
        }
        let n_blocks = take_u32(bytes, &mut p)? as usize;
        // Each column record is at least 14 bytes: bound the allocation.
        if n_columns > 0 && n_blocks > bytes.len().saturating_sub(p) / (14 * n_columns) {
            bail!("prune stats: block count exceeds the remaining bytes");
        }
        let mut blocks = Vec::with_capacity(if n_columns == 0 { 0 } else { n_blocks });
        for _ in 0..n_blocks {
            let mut block = Vec::with_capacity(n_columns);
            for _ in 0..n_columns {
                block.push(read_column_stats(bytes, &mut p)?);
            }
            blocks.push(block);
        }
        if p != bytes.len() {
            bail!("prune stats: trailing bytes");
        }
        Ok(Self { columns, blocks })
    }
}

fn leaf_may_match(
    stats: &ColumnBlockStats,
    leaf: &FilterExpression,
    op: &ComparisonOperator,
    value: &Value,
) -> bool {
    // The evaluator itself decides absent, null and distinct-set rows.
    let matches =
        |literal: Option<&Value>| evaluate_filter_resolved(leaf, &|_: &str| literal.cloned());
    if stats.absent > 0 && matches(None) {
        return true;
    }
    if stats.nulls > 0 && matches(Some(&Value::Null)) {
        return true;
    }
    if stats.non_null() == 0 {
        return false;
    }
    if let Some(distinct) = &stats.distinct {
        return distinct.iter().any(|v| matches(Some(v)));
    }
    !proves_no_non_null_match(stats, op, value)
}

/// Whether the bounds / bloom prove that no non-null row matches. Only for a
/// block whose non-null literals are all of the literal's class.
fn proves_no_non_null_match(
    stats: &ColumnBlockStats,
    op: &ComparisonOperator,
    value: &Value,
) -> bool {
    if stats.classes == CLASS_STRING {
        if let Some(bloom) = &stats.bloom
            && bloom_excludes(bloom, op, value)
        {
            return true;
        }
        if let (Some(min), Some(pred)) = (&stats.str_min, string_predicate(op, value)) {
            let bounds = ColumnBounds {
                min: Some(Value::String(min.clone())),
                max: stats.str_max.clone().map(Value::String),
                null_count: 0,
                distinct_count: None,
            };
            if bounds.can_prune(&pred) {
                return true;
            }
        }
        return false;
    }
    if stats.classes == CLASS_INT
        && let (Some((lo, hi)), Some(pred)) = (stats.int_bounds, int_predicate(op, value))
    {
        let bounds = ColumnBounds {
            min: Some(Value::from(lo)),
            max: Some(Value::from(hi)),
            null_count: 0,
            distinct_count: None,
        };
        if bounds.can_prune(&pred) {
            return true;
        }
    }
    if stats.classes != 0
        && stats.classes & !(CLASS_INT | CLASS_FLOAT) == 0
        && let Some((lo, hi)) = stats.num_bounds
    {
        return numeric_excludes(lo, hi, op, value);
    }
    false
}

fn as_exact_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64(),
        _ => None,
    }
}

fn as_string(v: &Value) -> Option<String> {
    v.as_str().map(str::to_owned)
}

fn two_bounds<T>(value: &Value, f: impl Fn(&Value) -> Option<T>) -> Option<(T, T)> {
    let arr = value.as_array()?;
    if arr.len() != 2 {
        return None;
    }
    Some((f(&arr[0])?, f(&arr[1])?))
}

fn all_of<T>(value: &Value, f: impl Fn(&Value) -> Option<T>) -> Option<Vec<T>> {
    let arr = value.as_array()?;
    if arr.is_empty() {
        return None;
    }
    arr.iter().map(f).collect()
}

/// The TD-SPINE-4 kernel's predicate for an integer literal (exact type).
fn int_predicate(op: &ComparisonOperator, value: &Value) -> Option<ScalarPredicate> {
    let one = |v: &Value| as_exact_i64(v).map(ScalarValue::Int64);
    Some(match op {
        ComparisonOperator::Equals => ScalarPredicate::Equal(one(value)?),
        ComparisonOperator::LessThan => ScalarPredicate::LessThan(one(value)?),
        ComparisonOperator::LessThanOrEqual => ScalarPredicate::LessThanOrEqual(one(value)?),
        ComparisonOperator::GreaterThan => ScalarPredicate::GreaterThan(one(value)?),
        ComparisonOperator::GreaterThanOrEqual => ScalarPredicate::GreaterThanOrEqual(one(value)?),
        ComparisonOperator::Between => {
            let (lo, hi) = two_bounds(value, one)?;
            ScalarPredicate::Between(lo, hi)
        }
        ComparisonOperator::In => ScalarPredicate::In(all_of(value, one)?),
        _ => return None,
    })
}

/// The TD-SPINE-4 kernel's predicate for a string literal (exact type).
fn string_predicate(op: &ComparisonOperator, value: &Value) -> Option<ScalarPredicate> {
    let one = |v: &Value| as_string(v).map(ScalarValue::String);
    Some(match op {
        ComparisonOperator::Equals => ScalarPredicate::Equal(one(value)?),
        ComparisonOperator::LessThan => ScalarPredicate::LessThan(one(value)?),
        ComparisonOperator::LessThanOrEqual => ScalarPredicate::LessThanOrEqual(one(value)?),
        ComparisonOperator::GreaterThan => ScalarPredicate::GreaterThan(one(value)?),
        ComparisonOperator::GreaterThanOrEqual => ScalarPredicate::GreaterThanOrEqual(one(value)?),
        ComparisonOperator::Between => {
            let (lo, hi) = two_bounds(value, one)?;
            ScalarPredicate::Between(lo, hi)
        }
        ComparisonOperator::In => ScalarPredicate::In(all_of(value, one)?),
        _ => return None,
    })
}

fn finite_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64().filter(|f| f.is_finite()),
        _ => None,
    }
}

/// Slack around a float bound: the evaluator's relative epsilon (`1 ×
/// EPSILON × max(|a|, |b|, 1)`) plus rounding of an integer's `f64`
/// projection, with margin.
fn float_slack(a: f64, b: f64) -> f64 {
    4.0 * f64::EPSILON * a.abs().max(b.abs()).max(1.0)
}

fn outside(l: f64, lo: f64, hi: f64) -> bool {
    l < lo - float_slack(l, lo) || l > hi + float_slack(l, hi)
}

/// `f64` bounds of an all-numeric block. Ordering comparisons are strict
/// (the `f64` projection is monotone but not injective on large integers),
/// equality allows [`float_slack`].
fn numeric_excludes(lo: f64, hi: f64, op: &ComparisonOperator, value: &Value) -> bool {
    match op {
        ComparisonOperator::GreaterThan | ComparisonOperator::GreaterThanOrEqual => {
            finite_number(value).is_some_and(|l| hi < l)
        }
        ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual => {
            finite_number(value).is_some_and(|l| lo > l)
        }
        ComparisonOperator::Equals => finite_number(value).is_some_and(|l| outside(l, lo, hi)),
        ComparisonOperator::In => all_of(value, finite_number)
            .is_some_and(|ls| ls.into_iter().all(|l| outside(l, lo, hi))),
        ComparisonOperator::Between => {
            two_bounds(value, finite_number).is_some_and(|(a, b)| hi < a || lo > b)
        }
        _ => false,
    }
}

fn bloom_excludes(bloom: &[u8], op: &ComparisonOperator, value: &Value) -> bool {
    let absent = |s: &str| {
        !proximadb_block_format::bloom_may_contain_hash(
            bloom,
            proximadb_block_format::header::fnv1a_hash(s),
        )
    };
    match op {
        ComparisonOperator::Equals => value.as_str().is_some_and(absent),
        ComparisonOperator::In => all_of(value, |v| v.as_str().map(str::to_owned))
            .is_some_and(|ss| ss.iter().all(|s| absent(s))),
        _ => false,
    }
}

fn class_of(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Bool(_) => CLASS_BOOL,
        Value::Number(n) if n.is_i64() || n.is_u64() => CLASS_INT,
        Value::Number(_) => CLASS_FLOAT,
        Value::String(_) => CLASS_STRING,
        _ => CLASS_OTHER,
    }
}

/// Largest prefix of `s` of at most `max` bytes ending on a char boundary.
fn prefix(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Accumulates one column's stats over the rows of the current block.
#[derive(Debug, Default)]
struct ColumnAccumulator {
    rows: u32,
    absent: u32,
    nulls: u32,
    classes: u8,
    all_i64: bool,
    int_bounds: Option<(i64, i64)>,
    num_bounds: Option<(f64, f64)>,
    str_min: Option<String>,
    str_max: Option<String>,
    /// Distinct values by encoding; `None` once a cap is exceeded.
    distinct: Option<(HashSet<Vec<u8>>, Vec<Value>)>,
    /// Distinct string hashes for the bloom; `None` once over the cap.
    bloom_keys: Option<HashSet<u64>>,
}

impl ColumnAccumulator {
    fn new() -> Self {
        Self {
            all_i64: true,
            distinct: Some((HashSet::new(), Vec::new())),
            bloom_keys: Some(HashSet::new()),
            ..Default::default()
        }
    }

    fn push(&mut self, literal: Option<Value>) {
        self.rows = self.rows.saturating_add(1);
        let value = match literal {
            None => {
                self.absent = self.absent.saturating_add(1);
                return;
            }
            Some(Value::Null) => {
                self.nulls = self.nulls.saturating_add(1);
                return;
            }
            Some(v) => v,
        };
        self.classes |= class_of(&value);
        if let Value::Number(n) = &value {
            match n.as_i64() {
                Some(i) if self.all_i64 => {
                    self.int_bounds = Some(match self.int_bounds {
                        Some((lo, hi)) => (lo.min(i), hi.max(i)),
                        None => (i, i),
                    });
                }
                _ => self.all_i64 = false,
            }
            if let Some(f) = n.as_f64().filter(|f| f.is_finite()) {
                self.num_bounds = Some(match self.num_bounds {
                    Some((lo, hi)) => (lo.min(f), hi.max(f)),
                    None => (f, f),
                });
            }
        }
        if let Value::String(s) = &value {
            if self.str_min.as_deref().is_none_or(|m| s.as_str() < m) {
                self.str_min = Some(s.clone());
            }
            if self.str_max.as_deref().is_none_or(|m| s.as_str() > m) {
                self.str_max = Some(s.clone());
            }
            if let Some(keys) = &mut self.bloom_keys {
                keys.insert(proximadb_block_format::header::fnv1a_hash(s));
                if keys.len() > BLOOM_MAX_KEYS {
                    self.bloom_keys = None;
                }
            }
        }
        if let Some((seen, values)) = &mut self.distinct {
            let exact = rmp_serde::to_vec(&value).ok().filter(|bytes| {
                bytes.len() <= MAX_DISTINCT_VALUE_BYTES
                    && rmp_serde::from_slice::<Value>(bytes).is_ok_and(|back| back == value)
            });
            match exact {
                Some(bytes) => {
                    if seen.insert(bytes) {
                        values.push(value);
                    }
                    if values.len() > MAX_DISTINCT_VALUES {
                        self.distinct = None;
                    }
                }
                None => self.distinct = None,
            }
        }
    }

    fn finish(self) -> ColumnBlockStats {
        let non_null = self
            .rows
            .saturating_sub(self.absent)
            .saturating_sub(self.nulls);
        let numeric = self.classes != 0 && self.classes & !(CLASS_INT | CLASS_FLOAT) == 0;
        let string_only = self.classes == CLASS_STRING;
        let distinct = self.distinct.map(|(_, values)| values);
        let (str_min, str_max) = if string_only && non_null > 0 {
            let min = self
                .str_min
                .map(|m| prefix(&m, STRING_BOUND_PREFIX_BYTES).to_owned());
            // A truncated upper bound is not an upper bound: drop it.
            let max = self
                .str_max
                .filter(|m| m.len() <= STRING_BOUND_PREFIX_BYTES);
            (min, max)
        } else {
            (None, None)
        };
        let bloom = match (&distinct, self.bloom_keys) {
            (None, Some(keys)) if string_only && !keys.is_empty() => {
                let bytes = (keys.len() * BLOOM_BITS_PER_KEY)
                    .div_ceil(8)
                    .max(BLOOM_MIN_BYTES);
                let mut bloom = vec![0u8; bytes];
                for key in keys {
                    proximadb_block_format::bloom_insert_hash(&mut bloom, key);
                }
                Some(bloom)
            }
            _ => None,
        };
        ColumnBlockStats {
            rows: self.rows,
            absent: self.absent,
            nulls: self.nulls,
            classes: self.classes,
            int_bounds: (self.classes == CLASS_INT && self.all_i64)
                .then_some(self.int_bounds)
                .flatten(),
            num_bounds: numeric.then_some(self.num_bounds).flatten(),
            str_min,
            str_max,
            distinct,
            bloom,
        }
    }
}

/// Builds a [`BlockPruneStatsTable`] row by row, in on-disk order, cut at
/// every block boundary.
#[derive(Debug)]
pub struct BlockPruneStatsBuilder {
    columns: Vec<PruneColumn>,
    current: Vec<ColumnAccumulator>,
    blocks: Vec<Vec<ColumnBlockStats>>,
    /// A block cut disagreed with the rows pushed: emit nothing.
    poisoned: bool,
}

impl BlockPruneStatsBuilder {
    /// A builder for `columns` (the P-Shred spec); `None` when there is
    /// nothing to summarize (no column, too many, or a duplicate name).
    pub fn new(columns: Vec<PruneColumn>) -> Option<Self> {
        let names: HashSet<&str> = columns.iter().map(|c| c.name.as_str()).collect();
        if columns.is_empty() || columns.len() > MAX_COLUMNS || names.len() != columns.len() {
            return None;
        }
        let current = columns.iter().map(|_| ColumnAccumulator::new()).collect();
        Some(Self {
            columns,
            current,
            blocks: Vec::new(),
            poisoned: false,
        })
    }

    /// Record one row's props.
    pub fn push_row(&mut self, props: &ProximaTree) {
        for (column, acc) in self.columns.iter().zip(&mut self.current) {
            acc.push(resolve_filter_literal(props, &column.name));
        }
    }

    /// Close the current block, which the writer cut with `rows` rows. A
    /// mismatch poisons the builder (no stats are emitted).
    pub fn cut_block(&mut self, rows: u32) {
        let current = std::mem::replace(
            &mut self.current,
            self.columns
                .iter()
                .map(|_| ColumnAccumulator::new())
                .collect(),
        );
        if current.iter().any(|acc| acc.rows != rows) {
            self.poisoned = true;
        }
        self.blocks
            .push(current.into_iter().map(ColumnAccumulator::finish).collect());
    }

    /// The finished table, or `None` when poisoned, when rows were pushed
    /// after the last cut, or when no block was cut.
    pub fn finish(self) -> Option<BlockPruneStatsTable> {
        if self.poisoned || self.blocks.is_empty() || self.current.iter().any(|acc| acc.rows > 0) {
            return None;
        }
        Some(BlockPruneStatsTable {
            columns: self.columns,
            blocks: self.blocks,
        })
    }
}

fn write_column_stats(out: &mut Vec<u8>, s: &ColumnBlockStats) -> Result<()> {
    let mut flags = 0u8;
    if s.int_bounds.is_some() {
        flags |= F_INT_BOUNDS;
    }
    if s.num_bounds.is_some() {
        flags |= F_NUM_BOUNDS;
    }
    if s.str_min.is_some() {
        flags |= F_STR_MIN;
    }
    if s.str_max.is_some() {
        flags |= F_STR_MAX;
    }
    if s.distinct.is_some() {
        flags |= F_DISTINCT;
    }
    if s.bloom.is_some() {
        flags |= F_BLOOM;
    }
    out.push(flags);
    out.extend_from_slice(&s.rows.to_le_bytes());
    out.extend_from_slice(&s.absent.to_le_bytes());
    out.extend_from_slice(&s.nulls.to_le_bytes());
    out.push(s.classes);
    if let Some((lo, hi)) = s.int_bounds {
        out.extend_from_slice(&lo.to_le_bytes());
        out.extend_from_slice(&hi.to_le_bytes());
    }
    if let Some((lo, hi)) = s.num_bounds {
        out.extend_from_slice(&lo.to_le_bytes());
        out.extend_from_slice(&hi.to_le_bytes());
    }
    for bound in [&s.str_min, &s.str_max].into_iter().flatten() {
        let len = u8::try_from(bound.len())
            .ok()
            .filter(|len| usize::from(*len) <= STRING_BOUND_PREFIX_BYTES)
            .ok_or_else(|| anyhow::anyhow!("prune stats: string bound too long"))?;
        out.push(len);
        out.extend_from_slice(bound.as_bytes());
    }
    if let Some(values) = &s.distinct {
        if values.len() > MAX_DISTINCT_VALUES {
            bail!("prune stats: distinct set too large");
        }
        out.push(values.len() as u8);
        for value in values {
            let bytes = rmp_serde::to_vec(value)?;
            if bytes.len() > MAX_DISTINCT_VALUE_BYTES {
                bail!("prune stats: distinct value too large");
            }
            out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
    }
    if let Some(bloom) = &s.bloom {
        let len = u16::try_from(bloom.len())
            .map_err(|_| anyhow::anyhow!("prune stats: bloom too large"))?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(bloom);
    }
    Ok(())
}

fn read_column_stats(bytes: &[u8], p: &mut usize) -> Result<ColumnBlockStats> {
    let flags = take_u8(bytes, p)?;
    if flags & !F_ALL != 0 {
        bail!("prune stats: unknown flags 0x{flags:02x}");
    }
    let rows = take_u32(bytes, p)?;
    let absent = take_u32(bytes, p)?;
    let nulls = take_u32(bytes, p)?;
    let classes = take_u8(bytes, p)?;
    if classes & !CLASS_ALL != 0 {
        bail!("prune stats: unknown classes 0x{classes:02x}");
    }
    let non_null = rows
        .checked_sub(absent)
        .and_then(|r| r.checked_sub(nulls))
        .ok_or_else(|| anyhow::anyhow!("prune stats: absent + nulls exceed rows"))?;
    if (non_null == 0) != (classes == 0) {
        bail!("prune stats: classes disagree with the non-null count");
    }
    let int_bounds = if flags & F_INT_BOUNDS != 0 {
        let lo = i64::from_le_bytes(take(bytes, p, 8)?.try_into()?);
        let hi = i64::from_le_bytes(take(bytes, p, 8)?.try_into()?);
        if lo > hi || classes != CLASS_INT {
            bail!("prune stats: invalid integer bounds");
        }
        Some((lo, hi))
    } else {
        None
    };
    let num_bounds = if flags & F_NUM_BOUNDS != 0 {
        let lo = f64::from_le_bytes(take(bytes, p, 8)?.try_into()?);
        let hi = f64::from_le_bytes(take(bytes, p, 8)?.try_into()?);
        let numeric = classes != 0 && classes & !(CLASS_INT | CLASS_FLOAT) == 0;
        if !lo.is_finite() || !hi.is_finite() || lo > hi || !numeric {
            bail!("prune stats: invalid numeric bounds");
        }
        Some((lo, hi))
    } else {
        None
    };
    let read_str = |p: &mut usize| -> Result<String> {
        let len = usize::from(take_u8(bytes, p)?);
        if len > STRING_BOUND_PREFIX_BYTES {
            bail!("prune stats: string bound too long");
        }
        String::from_utf8(take(bytes, p, len)?.to_vec())
            .map_err(|_| anyhow::anyhow!("prune stats: string bound is not UTF-8"))
    };
    let str_min = if flags & F_STR_MIN != 0 {
        Some(read_str(p)?)
    } else {
        None
    };
    let str_max = if flags & F_STR_MAX != 0 {
        Some(read_str(p)?)
    } else {
        None
    };
    if (str_min.is_some() || str_max.is_some()) && classes != CLASS_STRING {
        bail!("prune stats: string bounds on a non-string block");
    }
    if str_max.is_some() && str_min.is_none() {
        bail!("prune stats: string upper bound without a lower bound");
    }
    if let (Some(min), Some(max)) = (&str_min, &str_max)
        && min > max
    {
        bail!("prune stats: string bounds out of order");
    }
    let distinct = if flags & F_DISTINCT != 0 {
        let n = usize::from(take_u8(bytes, p)?);
        if n > MAX_DISTINCT_VALUES || n > non_null as usize || (n == 0) != (non_null == 0) {
            bail!("prune stats: invalid distinct-set size {n}");
        }
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            let len = usize::from(take_u16(bytes, p)?);
            if len > MAX_DISTINCT_VALUE_BYTES {
                bail!("prune stats: distinct value too large");
            }
            let value: Value = rmp_serde::from_slice(take(bytes, p, len)?)
                .map_err(|e| anyhow::anyhow!("prune stats: distinct value: {e}"))?;
            if value.is_null() || classes & class_of(&value) == 0 {
                bail!("prune stats: distinct value outside the block's classes");
            }
            values.push(value);
        }
        Some(values)
    } else {
        None
    };
    let bloom = if flags & F_BLOOM != 0 {
        let len = usize::from(take_u16(bytes, p)?);
        if len == 0 || classes != CLASS_STRING {
            bail!("prune stats: invalid bloom");
        }
        Some(take(bytes, p, len)?.to_vec())
    } else {
        None
    };
    Ok(ColumnBlockStats {
        rows,
        absent,
        nulls,
        classes,
        int_bounds,
        num_bounds,
        str_min,
        str_max,
        distinct,
        bloom,
    })
}

fn take<'a>(bytes: &'a [u8], p: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = p
        .checked_add(len)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| anyhow::anyhow!("prune stats truncated"))?;
    let out = &bytes[*p..end];
    *p = end;
    Ok(out)
}

fn take_u8(bytes: &[u8], p: &mut usize) -> Result<u8> {
    Ok(take(bytes, p, 1)?[0])
}

fn take_u16(bytes: &[u8], p: &mut usize) -> Result<u16> {
    Ok(u16::from_le_bytes(take(bytes, p, 2)?.try_into()?))
}

fn take_u32(bytes: &[u8], p: &mut usize) -> Result<u32> {
    Ok(u32::from_le_bytes(take(bytes, p, 4)?.try_into()?))
}

#[cfg(test)]
#[path = "block_prune_stats_tests.rs"]
mod tests;
