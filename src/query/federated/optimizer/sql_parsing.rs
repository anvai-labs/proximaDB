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
    let bytes = sql.as_bytes();
    let keyword_bytes = keyword.as_bytes();
    let keyword_len = keyword_bytes.len();
    let mut depth = 0usize;
    let mut in_quote = None;

    let mut in_doubled_pair = false;
    for (index, ch) in sql.char_indices() {
        if let Some(quote) = in_quote {
            if in_doubled_pair {
                // second char of a doubled close-quote — literal content
                in_doubled_pair = false;
                continue;
            }
            // POSTGRES dialect (standard_conforming_strings=on): the ONLY
            // in-literal escape is a DOUBLED close-quote (both chars stay
            // in the literal); backslash is a LITERAL character — treating
            // it as an escape desyncs the scanner on values ending in one
            // ('C:\dir\' is a complete literal).
            if ch == quote {
                if sql[(index + ch.len_utf8())..].starts_with(quote) {
                    in_doubled_pair = true;
                    continue;
                }
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

        if index < start_at || depth != 0 {
            continue;
        }

        if bytes
            .get(index..index.saturating_add(keyword_len))
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(keyword_bytes))
        {
            // Identifier boundary via the ONE shared byte class
            // ('$' and non-ASCII continue an identifier — b$from / éfrom
            // must not match FROM).
            let before_ok = index == 0 || !crate::core::utils::is_identifier_byte(bytes[index - 1]);
            let after_index = index + keyword_len;
            let after_ok = after_index == bytes.len()
                || !crate::core::utils::is_identifier_byte(bytes[after_index]);
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

    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if let Some(quote) = in_quote {
            current.push(ch);
            // POSTGRES dialect: a DOUBLED close-quote stays in-quote;
            // backslash is a literal (see the scanners above).
            if ch == quote {
                if chars.peek() == Some(&quote) {
                    chars.next();
                    current.push(quote);
                    continue;
                }
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

fn strip_distinct_prefix(clause: &str) -> Option<&str> {
    let keyword = clause.get(.."DISTINCT".len())?;
    if !keyword.eq_ignore_ascii_case("DISTINCT") {
        return None;
    }
    let rest = clause.get("DISTINCT".len()..)?;
    // Whitespace (any amount) optionally followed by parens that WRAP the
    // whole operand — the Postgres-legal COUNT(DISTINCT(id)) and
    // COUNT(DISTINCT (id)) spellings (the spaced form returned a
    // paren-wrapped column; a mid-operand paren was blindly stripped).
    let trimmed_rest = rest.trim_start();
    if let Some(inner) = trimmed_rest
        .strip_prefix('(')
        .and_then(|r| r.strip_suffix(')'))
        .filter(|inner| !inner.is_empty())
    {
        return Some(inner.trim());
    }
    if trimmed_rest.starts_with(|c: char| c.is_whitespace()) {
        return Some(trimmed_rest);
    }
    None
}

pub(crate) fn select_has_distinct(sql: &str) -> bool {
    let Some(select_pos) = find_top_level_keyword(sql, "SELECT") else {
        return false;
    };
    let Some(from_pos) = find_top_level_keyword_from(sql, "FROM", select_pos + 6) else {
        return false;
    };

    strip_distinct_prefix(sql[select_pos + 6..from_pos].trim_start()).is_some()
}

pub(crate) fn extract_select_items(sql: &str) -> Vec<SelectItem> {
    let Some(select_pos) = find_top_level_keyword(sql, "SELECT") else {
        return vec![];
    };
    let Some(from_pos) = find_top_level_keyword_from(sql, "FROM", select_pos + 6) else {
        return vec![];
    };

    let clause = sql[select_pos + 6..from_pos].trim();
    // Fully case-insensitive and token-boundary-aware: a projected column
    // named `Distinctive` must not lose its prefix.
    let clause = strip_distinct_prefix(clause).unwrap_or(clause);

    split_top_level_list(clause)
        .into_iter()
        .map(|item| {
            // Use the shared top-level scanner: CAST(price AS INT) and
            // literals containing ' AS ' are not alias separators, while
            // every SQL whitespace class is accepted around a real alias.
            let mut as_pos: Option<usize> = None;
            let mut search_from = 0usize;
            while let Some(position) = find_top_level_keyword_from(&item, "AS", search_from) {
                as_pos = Some(position); // keep scanning: LAST wins
                search_from = position + "AS".len();
            }
            if let Some(as_pos) = as_pos {
                // Skip the (whitespace-padded) keyword between expression
                // and alias.
                let after_as = item[as_pos..]
                    .find(|c: char| !c.is_whitespace())
                    .map(|rel| as_pos + rel)
                    .unwrap_or(item.len());
                let alias_start = after_as + 2; // "AS"
                SelectItem {
                    expression: item[..as_pos].trim().to_string(),
                    alias: Some(item[as_pos + "AS".len()..].trim().to_string()),
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
            } else if let Some(distinct_column) = strip_distinct_prefix(inner) {
                // The ONE distinct predicate (any case + any whitespace —
                // 'DISTINCT\tid' silently fell to a non-distinct count
                // with a phantom column).
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
            // ASCII-case tail checks on the ORIGINAL — uppercase can
            // change UTF-8 byte lengths, making len()-needle subtraction
            // slice mid-character (the class the sibling scanners fixed).
            let ends_ci = |needle: &str| {
                // BYTE compare (a str slice could cut mid-character)
                entry.len() >= needle.len()
                    && entry.as_bytes()[entry.len() - needle.len()..]
                        .eq_ignore_ascii_case(needle.as_bytes())
            };
            let strip_ci = |needle: &str| entry[..entry.len() - needle.len()].trim();
            let trimmed = if ends_ci(" NULLS FIRST") {
                strip_ci(" NULLS FIRST")
            } else if ends_ci(" NULLS LAST") {
                strip_ci(" NULLS LAST")
            } else {
                entry.trim()
            };

            let ends_trim_ci = |needle: &str| {
                trimmed.len() >= needle.len()
                    && trimmed.as_bytes()[trimmed.len() - needle.len()..]
                        .eq_ignore_ascii_case(needle.as_bytes())
            };
            let ascending = !ends_trim_ci(" DESC");
            let column = if ends_trim_ci(" ASC") || ends_trim_ci(" DESC") {
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
                    nulls_first: if ends_ci(" NULLS FIRST") {
                        true
                    } else if ends_ci(" NULLS LAST") {
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
    let bytes = input.as_bytes();
    let operator_bytes = operator.as_bytes();
    let mut depth = 0usize;
    let mut in_quote = None;

    let mut in_doubled_pair = false;
    for (index, ch) in input.char_indices() {
        if let Some(quote) = in_quote {
            if in_doubled_pair {
                // second char of a doubled close-quote — literal content
                in_doubled_pair = false;
                continue;
            }
            // POSTGRES dialect: doubled close-quote stays in-quote;
            // backslash is a literal.
            if ch == quote {
                if input[(index + ch.len_utf8())..].starts_with(quote) {
                    in_doubled_pair = true;
                    continue;
                }
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
            && bytes
                .get(index..index.saturating_add(operator_bytes.len()))
                .is_some_and(|candidate| candidate.eq_ignore_ascii_case(operator_bytes))
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
    if trimmed.len() >= 2
        && ((trimmed.starts_with('\'') && trimmed.ends_with('\''))
            || (trimmed.starts_with('"') && trimmed.ends_with('"')))
    {
        let inner = &trimmed[1..trimmed.len() - 1];
        // Decode SQL-standard doubled quotes — this PR's own producers
        // (sql_quote) emit them, and a WHERE literal containing an
        // apostrophe silently matched nothing undecoded.
        let decoded = if trimmed.starts_with('\'') {
            inner.replace("''", "'")
        } else {
            inner.replace("\"\"", "\"")
        };
        return Some(PredicateValue::String(decoded));
    }
    if let Ok(value) = trimmed.parse::<i64>() {
        return Some(PredicateValue::Int(value));
    }
    if let Ok(value) = trimmed.parse::<f64>() {
        return Some(PredicateValue::Float(value));
    }
    None
}

#[cfg(test)]
mod scanner_tests {
    use super::{
        AggregateFunction, extract_order_by, extract_select_items, extract_where_predicate,
        find_top_level_keyword, find_top_level_operator, parse_aggregate_expr,
        parse_predicate_value, select_has_distinct,
    };

    #[test]
    fn top_level_scanners_keep_original_utf8_offsets() {
        // U+FB00 uppercases to two ASCII bytes. Building an uppercased copy
        // and indexing it with offsets from the original UTF-8 string shifts
        // every token that follows this literal.
        let sql = "SELECT 'ﬀ' AS label FROM documents";
        assert_eq!(
            find_top_level_keyword(sql, "FROM"),
            sql.find("FROM"),
            "keyword offsets must refer to the original SQL"
        );
        let items = extract_select_items(sql);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].expression, "'ﬀ'");
        assert_eq!(items[0].alias.as_deref(), Some("label"));

        let order = extract_order_by("SELECT label FROM documents ORDER BY 'ﬀ' DESC NULLS LAST");
        assert_eq!(order.len(), 1);
        assert_eq!(order[0].column, "'ﬀ'");
        assert!(!order[0].ascending);
        assert!(!order[0].nulls_first);

        let predicate = extract_where_predicate("SELECT * FROM documents WHERE 'ﬀ' IS NOT NULL")
            .expect("simple predicate");
        assert_eq!(predicate.column, "'ﬀ'");

        let predicate = "label = 'ﬀ' AND score >= 0.5";
        assert_eq!(
            find_top_level_operator(predicate, ">="),
            predicate.find(">="),
            "operator offsets must refer to the original predicate"
        );
    }

    #[test]
    fn distinct_keyword_requires_a_token_boundary() {
        let sql = "SELECT Distinctive FROM documents";
        assert!(!select_has_distinct(sql));
        let items = extract_select_items(sql);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].expression, "Distinctive");
    }

    #[test]
    fn malformed_single_quote_predicate_fails_closed() {
        assert!(parse_predicate_value("'").is_none());
        assert!(parse_predicate_value("\"").is_none());
    }

    #[test]
    fn select_alias_scanner_accepts_whitespace_classes_only_at_top_level() {
        let items = extract_select_items(
            "SELECT CAST(price AS INT)\tAS\tprice_int, 'kept AS literal'\nAS\nlabel FROM sales",
        );

        assert_eq!(items.len(), 2);
        assert_eq!(items[0].expression, "CAST(price AS INT)");
        assert_eq!(items[0].alias.as_deref(), Some("price_int"));
        assert_eq!(items[1].expression, "'kept AS literal'");
        assert_eq!(items[1].alias.as_deref(), Some("label"));

        let aggregate_items = extract_select_items(
            "SELECT COUNT(DISTINCT(customer_id)) AS unique_customers FROM sales",
        );
        let aggregate = parse_aggregate_expr(&aggregate_items[0])
            .expect("parenthesized distinct operand must parse");
        assert!(matches!(
            aggregate.function,
            AggregateFunction::CountDistinct
        ));
        assert_eq!(aggregate.column.as_deref(), Some("customer_id"));
    }
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

    // ASCII-case tail checks on the ORIGINAL (uppercase-offset class).
    let ends_ci = |needle: &str| {
        clause.len() >= needle.len()
            && clause.as_bytes()[clause.len() - needle.len()..]
                .eq_ignore_ascii_case(needle.as_bytes())
    };
    if ends_ci(" IS NOT NULL") {
        return Some(Predicate {
            column: clause[..clause.len() - " IS NOT NULL".len()]
                .trim()
                .to_string(),
            op: PredicateOp::IsNotNull,
            value: PredicateValue::Null,
        });
    }
    if ends_ci(" IS NULL") {
        return Some(Predicate {
            column: clause[..clause.len() - " IS NULL".len()].trim().to_string(),
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
