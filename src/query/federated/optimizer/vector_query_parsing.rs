use super::{VectorQuery, VectorSource};

pub(crate) fn vector_source_from_query(query: &VectorQuery) -> VectorSource {
    match query {
        VectorQuery::Literal(vector) => VectorSource::Literal(vector.clone()),
        VectorQuery::Expression(expr) => vector_source_from_expression(expr),
    }
}

pub(crate) fn vector_source_from_literal(raw: &str) -> VectorSource {
    parse_vector_literal(raw)
        .map_or_else(|| vector_source_from_expression(raw), VectorSource::Literal)
}

pub(crate) fn vector_source_from_expression(expr: &str) -> VectorSource {
    let trimmed = expr.trim();
    // Identifier-validated (see split_qualified_reference): literal-shaped
    // text — quoted, cast-suffixed, or bare — stays Expression so the
    // executor reports the validation error instead of a bogus
    // correlation split at the first '.' inside the literal.
    if let Some((table, column)) = split_qualified_reference(trimmed) {
        VectorSource::ColumnRef {
            table: table.trim().to_string(),
            column: column.trim().to_string(),
        }
    } else {
        VectorSource::Expression(trimmed.to_string())
    }
}

pub(crate) fn split_qualified_reference(expr: &str) -> Option<(&str, &str)> {
    let mut in_quotes = false;
    let mut chars = expr.char_indices().peekable();
    let mut split: Option<(&str, &str)> = None;

    while let Some((idx, ch)) = chars.next() {
        match ch {
            '"' => {
                if in_quotes && matches!(chars.peek(), Some((_, '"'))) {
                    chars.next();
                    continue;
                }
                in_quotes = !in_quotes;
            }
            '.' if !in_quotes && split.is_none() => {
                split = Some((&expr[..idx], &expr[idx + ch.len_utf8()..]));
            }
            _ => {}
        }
    }
    if in_quotes {
        return None;
    }
    let (table, column) = split?;
    // Both halves must be IDENTIFIER-shaped — vector literals (quoted,
    // cast-suffixed, or bare) reaching this splitter must not be misread
    // as bogus column references.
    if !is_identifier(table) || !is_identifier(column) {
        return None;
    }
    Some((table, column))
}

fn is_identifier(part: &str) -> bool {
    let trimmed = part.trim();
    if let Some(inner) = stripped_quoted_ident(trimmed) {
        return !inner.is_empty() && !inner.contains('"');
    }
    let mut chars = trimmed.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn stripped_quoted_ident(part: &str) -> Option<&str> {
    part.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
}

pub(crate) fn parse_vector_literal(raw: &str) -> Option<Vec<f32>> {
    let trimmed = raw.trim();
    let without_cast = trimmed
        .strip_suffix("::vector")
        .or_else(|| trimmed.strip_suffix("::VECTOR"))
        .unwrap_or(trimmed)
        .trim();
    let unquoted = without_cast.trim_matches('\'').trim_matches('"').trim();

    if !(unquoted.starts_with('[') && unquoted.ends_with(']')) {
        return None;
    }

    let inner = &unquoted[1..unquoted.len() - 1];
    if inner.trim().is_empty() {
        // An empty vector is not a queryable literal (the JSON twins 400
        // on it; the SQL surface must not silently dispatch a
        // 0-dimension vector).
        return None;
    }

    inner
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<f32>()
                .ok()
                .filter(|component| component.is_finite())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::parse_vector_literal;

    #[test]
    fn vector_literals_reject_non_finite_components() {
        assert!(parse_vector_literal("'[NaN]'").is_none());
        assert!(parse_vector_literal("'[inf]'").is_none());
        assert!(parse_vector_literal("'[-inf]'").is_none());
        assert!(parse_vector_literal("'[1e300]'").is_none());
    }
}
