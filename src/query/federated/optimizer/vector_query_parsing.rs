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

    while let Some((idx, ch)) = chars.next() {
        match ch {
            '"' => {
                if in_quotes && matches!(chars.peek(), Some((_, '"'))) {
                    chars.next();
                    continue;
                }
                in_quotes = !in_quotes;
            }
            '.' if !in_quotes => return Some((&expr[..idx], &expr[idx + ch.len_utf8()..])),
            _ => {}
        }
    }

    None
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
        return Some(Vec::new());
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
