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

/// Strip a trailing `::vector` cast (any ASCII case, optional `(dim)`),
/// returning the input unchanged when no cast is present.
fn strip_vector_cast_suffix(input: &str) -> &str {
    // Cheap gate: the lowercase scan allocated a full copy of 14KB-class
    // literals on every per-row call just to discover no cast.
    if !input.contains("::") {
        return input;
    }
    let lower = input.to_ascii_lowercase();
    let Some(cast_start) = lower.rfind("::vector") else {
        return input;
    };
    let after = &input[cast_start + 8..];
    let inner = after
        .trim()
        .strip_prefix('(')
        .and_then(|r| r.strip_suffix(')'));
    // Whitespace-tolerant ('::vector (3)') like every sibling stripper;
    // digits-only dimension or no dimension at all.
    let dimension_ok = after.trim().is_empty()
        || inner.is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()));
    if dimension_ok {
        &input[..cast_start]
    } else {
        input
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
    // The table half must be IDENTIFIER-shaped; the COLUMN half may be a
    // dotted nested path (the documented LATERAL correlation form
    // `p.document.embedding`). Literal-shaped text (quoted, cast-suffixed,
    // bare) must not be misread as column references.
    if !is_identifier(table) || !is_dotted_identifier_path(column) {
        return None;
    }
    Some((table, column))
}

fn is_identifier(part: &str) -> bool {
    let trimmed = part.trim();
    if let Some(inner) = stripped_quoted_ident(trimmed) {
        // Doubled quotes are the escaped-quote spelling INSIDE a quoted
        // identifier (the scan loop above already honors them). Dots ARE
        // allowed: quoted dotted ALIASES are a test-pinned lateral form
        // ("Right.Alias".document.embedding) and the executor's alias
        // resolution handles quoted names.
        let unescaped = inner.replace("\"\"", "");
        return !inner.is_empty() && !unescaped.contains('"');
    }
    let mut chars = trimmed.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_dotted_identifier_path(part: &str) -> bool {
    // Quote-aware split: a quoted segment may CONTAIN dots
    // (p."doc.embedding" is one column, not a nested path).
    let mut segment = String::new();
    let mut in_quotes = false;
    let mut chars = part.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes && matches!(chars.peek(), Some('"')) {
                    chars.next();
                    segment.push('"');
                    segment.push('"');
                    continue;
                }
                in_quotes = !in_quotes;
                segment.push('"');
            }
            '.' if !in_quotes => {
                if !is_identifier(&segment) {
                    return false;
                }
                segment.clear();
            }
            _ => segment.push(ch),
        }
    }
    !in_quotes && is_identifier(&segment)
}

fn stripped_quoted_ident(part: &str) -> Option<&str> {
    part.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
}

pub(crate) fn parse_vector_literal(raw: &str) -> Option<Vec<f32>> {
    let trimmed = raw.trim();
    // Case-insensitive, optionally dimensioned cast suffix ('::vector',
    // '::Vector', '::vector(3)') — the fusion front parser accepts all of
    // these (strip_vector_cast); the executor's re-parse must agree or a
    // front-accepted literal fails here as an 'unsupported expression'.
    let without_cast = strip_vector_cast_suffix(trimmed).trim();
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

    #[test]
    fn vector_literals_accept_case_insensitive_dimensioned_casts() {
        assert_eq!(
            parse_vector_literal("'[0.25, -0.5]'::Vector(2)"),
            Some(vec![0.25, -0.5])
        );
        assert_eq!(parse_vector_literal("'[0.25]'::VeCtOr"), Some(vec![0.25]));
        assert!(parse_vector_literal("'[0.25]'::vector(x)").is_none());
    }
}
