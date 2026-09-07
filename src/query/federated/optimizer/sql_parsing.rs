use super::{
    AggregateExpr, AggregateFunction, OrderByClause, Predicate, PredicateOp, PredicateValue,
    SelectItem,
};

pub(crate) fn find_top_level_keyword(sql: &str, keyword: &str) -> Option<usize> {
    find_top_level_keyword_from(sql, keyword, 0)
}

pub(crate) fn find_top_level_keyword_from(
    sql: &str,
    keyword: &str,
    start_at: usize,
) -> Option<usize> {
    let sql_upper = sql.to_uppercase();
    let keyword_upper = keyword.to_uppercase();
    let bytes = sql_upper.as_bytes();
    let keyword_len = keyword_upper.len();
    let mut depth = 0usize;
    let mut in_quote = None;

    for (index, ch) in sql.char_indices() {
        if let Some(quote) = in_quote {
            // SQL literals escape by QUOTE DOUBLING (a doubled close-quote
            // stays in-quote); backslash is not an escape — treating it as
            // one desyncs the scanner when a value ends in a backslash.
            if ch == quote {
                in_quote = None;
            }
            continue;
        }

        match ch {
            '\'' | '"' => {
                in_quote = Some(ch);
                continue;
            }
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ => {}
        }

        if index < start_at || depth != 0 || index + keyword_len > sql_upper.len() {
            continue;
        }

        if &sql_upper[index..index + keyword_len] == keyword_upper.as_str() {
            let before_ok = index == 0
                || (!bytes[index - 1].is_ascii_alphanumeric() && bytes[index - 1] != b'_');
            let after_index = index + keyword_len;
            let after_ok = after_index == bytes.len()
                || (!bytes[after_index].is_ascii_alphanumeric() && bytes[after_index] != b'_');
            if before_ok && after_ok {
                return Some(index);
            }
        }
    }

    None
}

pub(crate) fn find_clause_end(sql: &str, start_at: usize, keywords: &[&str]) -> usize {
    keywords
        .iter()
        .filter_map(|keyword| find_top_level_keyword_from(sql, keyword, start_at))
        .min()
        .unwrap_or(sql.len())
}

pub(crate) fn split_top_level_list(input: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;
    let mut in_quote = None;

    for ch in input.chars() {
        if let Some(quote) = in_quote {
            current.push(ch);
            // Quote-doubling escapes; backslash is not an escape (see the
            // scanners above).
            if ch == quote {
                in_quote = None;
            }
            continue;
        }

        match ch {
            '\'' | '"' => {
                in_quote = Some(ch);
                current.push(ch);
            }
            '(' | '[' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' => {
                depth = depth.saturating_sub(1);
                current.push(ch);
            }
            ',' if depth == 0 => {
                if !current.trim().is_empty() {
                    items.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    if !current.trim().is_empty() {
        items.push(current.trim().to_string());
    }

    items
}

pub(crate) fn select_has_distinct(sql: &str) -> bool {
    let Some(select_pos) = find_top_level_keyword(sql, "SELECT") else {
        return false;
    };
    let Some(from_pos) = find_top_level_keyword_from(sql, "FROM", select_pos + 6) else {
        return false;
    };

    sql[select_pos + 6..from_pos]
        .trim_start()
        .to_uppercase()
        .starts_with("DISTINCT ")
}

pub(crate) fn extract_select_items(sql: &str) -> Vec<SelectItem> {
    let Some(select_pos) = find_top_level_keyword(sql, "SELECT") else {
        return vec![];
    };
    let Some(from_pos) = find_top_level_keyword_from(sql, "FROM", select_pos + 6) else {
        return vec![];
    };

    let clause = sql[select_pos + 6..from_pos].trim();
    let clause = clause
        .strip_prefix("DISTINCT ")
        .or_else(|| clause.strip_prefix("distinct "))
        .unwrap_or(clause);

    split_top_level_list(clause)
        .into_iter()
        .map(|item| {
            let upper = item.to_uppercase();
            if let Some(as_pos) = upper.rfind(" AS ") {
                SelectItem {
                    expression: item[..as_pos].trim().to_string(),
                    alias: Some(item[as_pos + 4..].trim().to_string()),
                }
            } else {
                SelectItem {
                    expression: item.trim().to_string(),
                    alias: None,
                }
            }
        })
        .collect()
}

#[allow(dead_code)]
pub(crate) fn extract_select_columns(sql: &str) -> Vec<String> {
    extract_select_items(sql)
        .into_iter()
        .map(|item| item.expression)
        .collect()
}

pub(crate) fn extract_group_by_columns(sql: &str) -> Vec<String> {
    let Some(group_by_pos) = find_top_level_keyword(sql, "GROUP BY") else {
        return vec![];
    };
    let end = find_clause_end(
        sql,
        group_by_pos + 8,
        &["ORDER BY", "LIMIT", "OFFSET", "HAVING", ";"],
    );

    split_top_level_list(sql[group_by_pos + 8..end].trim())
}

pub(crate) fn parse_aggregate_expr(item: &SelectItem) -> Option<AggregateExpr> {
    let expression = item.expression.trim();
    let open_paren = expression.find('(')?;
    if !expression.ends_with(')') {
        return None;
    }

    let function_name = expression[..open_paren].trim().to_uppercase();
    let inner = expression[open_paren + 1..expression.len() - 1].trim();
    let alias = item.alias.clone().unwrap_or_else(|| expression.to_string());

    match function_name.as_str() {
        "COUNT" => {
            if inner == "*" {
                Some(AggregateExpr {
                    function: AggregateFunction::Count,
                    column: None,
                    alias,
                })
            } else if let Some(distinct_column) = inner
                .strip_prefix("DISTINCT ")
                .or_else(|| inner.strip_prefix("distinct "))
            {
                Some(AggregateExpr {
                    function: AggregateFunction::CountDistinct,
                    column: Some(distinct_column.trim().to_string()),
                    alias,
                })
            } else {
                Some(AggregateExpr {
                    function: AggregateFunction::Count,
                    column: Some(inner.to_string()),
                    alias,
                })
            }
        }
        "SUM" => Some(AggregateExpr {
            function: AggregateFunction::Sum,
            column: Some(inner.to_string()),
            alias,
        }),
        "AVG" => Some(AggregateExpr {
            function: AggregateFunction::Avg,
            column: Some(inner.to_string()),
            alias,
        }),
        "MIN" => Some(AggregateExpr {
            function: AggregateFunction::Min,
            column: Some(inner.to_string()),
            alias,
        }),
        "MAX" => Some(AggregateExpr {
            function: AggregateFunction::Max,
            column: Some(inner.to_string()),
            alias,
        }),
        _ => None,
    }
}

pub(crate) fn extract_limit_offset(sql: &str) -> (Option<usize>, usize) {
    let limit = find_top_level_keyword(sql, "LIMIT").and_then(|limit_pos| {
        let end = find_clause_end(sql, limit_pos + 5, &["OFFSET", ";"]);
        sql[limit_pos + 5..end]
            .split_whitespace()
            .next()
            .and_then(|value| value.parse::<usize>().ok())
    });

    let offset = find_top_level_keyword(sql, "OFFSET")
        .and_then(|offset_pos| {
            let end = find_clause_end(sql, offset_pos + 6, &["LIMIT", ";"]);
            sql[offset_pos + 6..end]
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(0);

    (limit, offset)
}

pub(crate) fn extract_order_by(sql: &str) -> Vec<OrderByClause> {
    let Some(order_pos) = find_top_level_keyword(sql, "ORDER BY") else {
        return vec![];
    };
    let end = find_clause_end(sql, order_pos + 8, &["LIMIT", "OFFSET", ";"]);
    let clause = sql[order_pos + 8..end].trim();

    split_top_level_list(clause)
        .into_iter()
        .filter_map(|entry| {
            let upper = entry.to_uppercase();
            let nulls_first = upper.ends_with(" NULLS FIRST");
            let nulls_last = upper.ends_with(" NULLS LAST");
            let trimmed = if nulls_first {
                entry[..entry.len() - "NULLS FIRST".len()].trim()
            } else if nulls_last {
                entry[..entry.len() - "NULLS LAST".len()].trim()
            } else {
                entry.trim()
            };

            let upper_trimmed = trimmed.to_uppercase();
            let ascending = !upper_trimmed.ends_with(" DESC");
            let column = if upper_trimmed.ends_with(" ASC") || upper_trimmed.ends_with(" DESC") {
                trimmed[..trimmed.rfind(' ').unwrap_or(trimmed.len())]
                    .trim()
                    .to_string()
            } else {
                trimmed.to_string()
            };

            if column.is_empty() {
                None
            } else {
                Some(OrderByClause {
                    column,
                    ascending,
                    nulls_first: if nulls_first {
                        true
                    } else if nulls_last {
                        false
                    } else {
                        !ascending
                    },
                })
            }
        })
        .collect()
}

pub(crate) fn find_top_level_operator(input: &str, operator: &str) -> Option<usize> {
    let upper = input.to_uppercase();
    let operator_upper = operator.to_uppercase();
    let mut depth = 0usize;
    let mut in_quote = None;

    for (index, ch) in input.char_indices() {
        if let Some(quote) = in_quote {
            // SQL literals escape by QUOTE DOUBLING (a doubled close-quote
            // stays in-quote); backslash is not an escape — treating it as
            // one desyncs the scanner when a value ends in a backslash.
            if ch == quote {
                in_quote = None;
            }
            continue;
        }

        match ch {
            '\'' | '"' => {
                in_quote = Some(ch);
                continue;
            }
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ => {}
        }

        if depth == 0
            && index + operator_upper.len() <= upper.len()
            && &upper[index..index + operator_upper.len()] == operator_upper.as_str()
        {
            return Some(index);
        }
    }

    None
}

pub(crate) fn parse_predicate_value(raw: &str) -> Option<PredicateValue> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("NULL") {
        return Some(PredicateValue::Null);
    }
    if trimmed.eq_ignore_ascii_case("TRUE") {
        return Some(PredicateValue::Bool(true));
    }
    if trimmed.eq_ignore_ascii_case("FALSE") {
        return Some(PredicateValue::Bool(false));
    }
    if (trimmed.starts_with('\'') && trimmed.ends_with('\''))
        || (trimmed.starts_with('"') && trimmed.ends_with('"'))
    {
        return Some(PredicateValue::String(
            trimmed[1..trimmed.len() - 1].to_string(),
        ));
    }
    if let Ok(value) = trimmed.parse::<i64>() {
        return Some(PredicateValue::Int(value));
    }
    if let Ok(value) = trimmed.parse::<f64>() {
        return Some(PredicateValue::Float(value));
    }
    None
}

pub(crate) fn extract_where_predicate(sql: &str) -> Option<Predicate> {
    let where_pos = find_top_level_keyword(sql, "WHERE")?;
    let end = find_clause_end(
        sql,
        where_pos + 5,
        &["ORDER BY", "GROUP BY", "LIMIT", "OFFSET", "HAVING", ";"],
    );
    let clause = sql[where_pos + 5..end].trim();
    if clause.is_empty()
        || find_top_level_keyword(clause, "AND").is_some()
        || find_top_level_keyword(clause, "OR").is_some()
    {
        return None;
    }

    let upper = clause.to_uppercase();
    if upper.ends_with(" IS NOT NULL") {
        return Some(Predicate {
            column: clause[..clause.len() - "IS NOT NULL".len()]
                .trim()
                .to_string(),
            op: PredicateOp::IsNotNull,
            value: PredicateValue::Null,
        });
    }
    if upper.ends_with(" IS NULL") {
        return Some(Predicate {
            column: clause[..clause.len() - "IS NULL".len()].trim().to_string(),
            op: PredicateOp::IsNull,
            value: PredicateValue::Null,
        });
    }

    if let Some(index) = find_top_level_keyword(clause, "LIKE") {
        return Some(Predicate {
            column: clause[..index].trim().to_string(),
            op: PredicateOp::Like,
            value: parse_predicate_value(clause[index + 4..].trim())?,
        });
    }

    for (operator, predicate_op) in [
        ("!=", PredicateOp::Ne),
        ("<>", PredicateOp::Ne),
        (">=", PredicateOp::Ge),
        ("<=", PredicateOp::Le),
        ("=", PredicateOp::Eq),
        (">", PredicateOp::Gt),
        ("<", PredicateOp::Lt),
    ] {
        if let Some(index) = find_top_level_operator(clause, operator) {
            return Some(Predicate {
                column: clause[..index].trim().to_string(),
                op: predicate_op,
                value: parse_predicate_value(clause[index + operator.len()..].trim())?,
            });
        }
    }

    None
}
