//! MLflow search-filter grammar (TD-MLOPS-1 audit finding #3 — one home).
//!
//! The clause lexer (`split_filter_clauses`, quote-aware with doubled-quote
//! escapes), the field/op/value parser, and the per-resource predicate
//! types. Every search surface (experiments, runs, registered models,
//! model versions) dispatches through THESE — adding a grammar field is
//! one edit here, not four hand-rolled dispatchers downstream.

use super::{MlflowError, MlflowResult};
use proximadb_catalog::run_store::{ExperimentRecord, ExperimentStage, RunRecord};

pub(crate) enum ExperimentFilter {
    NameEq(String),
    NameLike(String),
    TagEq(String, String),
    Lifecycle(ExperimentStage),
}

impl ExperimentFilter {
    pub(crate) fn matches(&self, experiment: &ExperimentRecord) -> bool {
        match self {
            Self::NameEq(name) => experiment.name == *name,
            Self::NameLike(pattern) => like_match(&experiment.name, pattern),
            Self::TagEq(key, value) => experiment.tags.get(key) == Some(value),
            Self::Lifecycle(stage) => experiment.stage == *stage,
        }
    }
}

pub(crate) fn parse_experiment_filter(filter: &str) -> MlflowResult<Vec<ExperimentFilter>> {
    let mut predicates = Vec::new();
    for clause in split_filter_clauses(filter)? {
        let clause = clause.trim();
        if let Some(rest) = clause.strip_prefix("tags.") {
            let (key, op, value) = parse_filter_parts(rest, true)?;
            if op != "=" {
                return Err(MlflowError::invalid(
                    "experiment tag filter supports = only",
                ));
            }
            predicates.push(ExperimentFilter::TagEq(key, value));
            continue;
        }
        let (field, op, value) = parse_filter_parts(clause, true)?;
        match field.to_ascii_lowercase().as_str() {
            "name" | "attributes.name" => match op.as_str() {
                "=" => predicates.push(ExperimentFilter::NameEq(value)),
                "LIKE" => predicates.push(ExperimentFilter::NameLike(value)),
                _ => {
                    return Err(MlflowError::invalid(
                        "experiment name filter supports = / LIKE",
                    ));
                }
            },
            "attributes.lifecycle_stage" if op == "=" => {
                let stage = match value.to_ascii_lowercase().as_str() {
                    "active" => ExperimentStage::Active,
                    "deleted" => ExperimentStage::Deleted,
                    _ => {
                        return Err(MlflowError::invalid(format!(
                            "invalid experiment lifecycle_stage '{value}'"
                        )));
                    }
                };
                predicates.push(ExperimentFilter::Lifecycle(stage));
            }
            other => {
                return Err(MlflowError::invalid(format!(
                    "unsupported experiment filter field '{other}'"
                )));
            }
        }
    }
    Ok(predicates)
}

pub(crate) enum FieldFilter {
    ParamEq(String, String),
    ParamNe(String, String),
    ParamLike(String, String),
    TagEq(String, String),
    TagNe(String, String),
    TagLike(String, String),
    MetricCmp(String, f64, fn(f64, f64) -> bool),
}

impl FieldFilter {
    pub(crate) fn matches(&self, run: &RunRecord) -> bool {
        match self {
            FieldFilter::ParamEq(k, v) => run.params.get(k) == Some(v),
            FieldFilter::ParamNe(k, v) => run.params.get(k) != Some(v),
            FieldFilter::ParamLike(k, pattern) => run
                .params
                .get(k)
                .is_some_and(|actual| like_match(actual, pattern)),
            FieldFilter::TagEq(k, v) => run.tags.get(k) == Some(v),
            FieldFilter::TagNe(k, v) => run.tags.get(k) != Some(v),
            FieldFilter::TagLike(k, pattern) => match run.tags.get(k) {
                Some(actual) => like_match(actual, pattern),
                None => false,
            },
            FieldFilter::MetricCmp(k, v, cmp) => run
                .latest_metrics
                .get(k)
                .is_some_and(|point| cmp(point.value, *v)),
        }
    }
}

pub(crate) fn parse_run_filter(filter: &str) -> MlflowResult<Vec<FieldFilter>> {
    let mut out = Vec::new();
    for clause in split_filter_clauses(filter)? {
        let clause = clause.trim();
        let field = if let Some(rest) = clause.strip_prefix("metrics.") {
            let (key, op, value) = parse_filter_parts(rest, false)?;
            let number: f64 = value.parse().map_err(|_| {
                MlflowError::invalid(format!("metric filter needs a number, got {value}"))
            })?;
            if !number.is_finite() {
                return Err(MlflowError::invalid("metric filter value must be finite"));
            }
            let cmp = cmp_fn(&op)?;
            FieldFilter::MetricCmp(key, number, cmp)
        } else if let Some(rest) = clause.strip_prefix("params.") {
            let (key, op, value) = parse_filter_parts(rest, true)?;
            match op.as_str() {
                "=" => FieldFilter::ParamEq(key, value),
                "!=" => FieldFilter::ParamNe(key, value),
                "LIKE" => FieldFilter::ParamLike(key, value),
                other => {
                    return Err(MlflowError::invalid(format!(
                        "params filter supports = / != / LIKE, got '{other}'"
                    )));
                }
            }
        } else if let Some(rest) = clause.strip_prefix("tags.") {
            let (key, op, value) = parse_filter_parts(rest, true)?;
            match op.as_str() {
                "=" => FieldFilter::TagEq(key, value),
                "!=" => FieldFilter::TagNe(key, value),
                "LIKE" => FieldFilter::TagLike(key, value),
                other => {
                    return Err(MlflowError::invalid(format!(
                        "tags filter supports = / != / LIKE, got '{other}'"
                    )));
                }
            }
        } else {
            return Err(MlflowError::invalid(format!(
                "unsupported filter clause '{clause}' (slice 2: params. / metrics. / tags. clauses ANDed together)"
            )));
        };
        out.push(field);
    }
    if out.is_empty() {
        return Err(MlflowError::invalid("empty filter"));
    }
    Ok(out)
}

pub(crate) fn like_match(value: &str, pattern: &str) -> bool {
    // SQL LIKE: % = any run, _ = one char. Case-sensitive (MLflow is).
    let mut regex = String::from("^");
    for c in pattern.chars() {
        match c {
            '%' => regex.push_str(".*"),
            '_' => regex.push('.'),
            c => regex.push_str(&regex::escape(&c.to_string())),
        }
    }
    regex.push('$');
    regex::Regex::new(&regex)
        .map(|re| re.is_match(value))
        .unwrap_or(false)
}

pub(crate) fn split_filter_clauses(filter: &str) -> MlflowResult<Vec<&str>> {
    let bytes = filter.as_bytes();
    let mut clauses = Vec::new();
    let mut quote = None;
    let mut start = 0;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(active_quote) = quote {
            if byte == active_quote {
                if bytes.get(index + 1) == Some(&active_quote) {
                    index += 2;
                    continue;
                }
                quote = None;
            }
            index += 1;
            continue;
        }
        if matches!(byte, b'\'' | b'`' | b'"') {
            quote = Some(byte);
            index += 1;
            continue;
        }
        let is_and = bytes
            .get(index..index.saturating_add(3))
            .is_some_and(|word| word.eq_ignore_ascii_case(b"and"));
        let left_space = index > start && bytes[index - 1].is_ascii_whitespace();
        let right_space = bytes
            .get(index + 3)
            .is_some_and(|byte| byte.is_ascii_whitespace());
        if is_and && left_space && right_space {
            clauses.push(filter[start..index].trim());
            index += 3;
            start = index;
            continue;
        }
        index += 1;
    }
    if quote.is_some() {
        return Err(MlflowError::invalid("unterminated quote in filter"));
    }
    clauses.push(filter[start..].trim());
    if clauses.iter().any(|clause| clause.is_empty()) {
        return Err(MlflowError::invalid("empty filter clause"));
    }
    Ok(clauses)
}

pub(crate) fn parse_filter_parts(
    rest: &str,
    string_value: bool,
) -> MlflowResult<(String, String, String)> {
    let identifier = r#"(?:`(?:``|[^`])+`|"(?:""|[^"])+"|[^\s!<>=]+)"#;
    let pattern = if string_value {
        format!(r#"(?i)^\s*({identifier})\s*(!=|=|LIKE)\s*'((?:''|[^'])*)'\s*$"#)
    } else {
        format!(r#"(?i)^\s*({identifier})\s*(!=|<=|>=|<|>|=)\s*([^\s]+)\s*$"#)
    };
    let regex = regex::Regex::new(&pattern).map_err(|error| {
        MlflowError::internal(format!("invalid built-in filter regex: {error}"))
    })?;
    let captures = regex
        .captures(rest)
        .ok_or_else(|| MlflowError::invalid(format!("invalid filter clause '{rest}'")))?;
    let key = captures
        .get(1)
        .map(|capture| decode_identifier(capture.as_str()))
        .ok_or_else(|| MlflowError::invalid("filter is missing an identifier"))?;
    let op = captures
        .get(2)
        .map(|capture| capture.as_str().to_ascii_uppercase())
        .ok_or_else(|| MlflowError::invalid("filter is missing an operator"))?;
    let value = captures
        .get(3)
        .map(|capture| capture.as_str().replace("''", "'"))
        .ok_or_else(|| MlflowError::invalid("filter is missing a value"))?;
    Ok((key, op, value))
}

fn cmp_fn(op: &str) -> MlflowResult<fn(f64, f64) -> bool> {
    Ok(match op {
        "<" => |a, b| a < b,
        "<=" => |a, b| a <= b,
        ">" => |a, b| a > b,
        ">=" => |a, b| a >= b,
        "=" => |a, b| a == b,
        "!=" => |a, b| a != b,
        other => {
            return Err(MlflowError::invalid(format!(
                "unsupported comparison '{other}'"
            )));
        }
    })
}

fn decode_identifier(raw: &str) -> String {
    if let Some(inner) = raw
        .strip_prefix('`')
        .and_then(|value| value.strip_suffix('`'))
    {
        inner.replace("``", "`")
    } else if let Some(inner) = raw
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
    {
        inner.replace("\"\"", "\"")
    } else {
        raw.to_string()
    }
}
