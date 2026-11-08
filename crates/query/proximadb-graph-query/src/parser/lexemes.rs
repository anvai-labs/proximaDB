//! Shared Cypher lexemes — the TD-GRAPH-4 convergence step 1 ("shared
//! lexemes", one grammar behind the seam, not two drifting forks).
//!
//! These are the ONE source of truth for token-level parsing, consumed by
//! BOTH grammars in this crate (the read path in [`super`] and the write
//! seam in [`super::write_seam`]) and by the modality crate's graph-API
//! parser (`proximadb-graph`'s `cypher_parser`), each with its own nom
//! error type — hence generic over `E: ParseError<&str>`.
//!
//! The behavior here is the SUPERSET the modality grammar shipped (the
//! richer live parser): quoted/backtick identifiers, negative integers,
//! floats with exponents, case-insensitive booleans, `null`, arrays,
//! empty strings, and permissive `{ … }` spacing. The read path and the
//! write seam previously accepted strictly less; adopting the superset
//! only makes previously-REJECTED inputs parse (existing parses are
//! unchanged — differential-checked by the modality crate's 41-entry
//! pinned corpus, which must stay byte-identical across this convergence).

use std::collections::HashMap;

use nom::IResult;
use nom::branch::alt;
use nom::bytes::complete::{tag, tag_no_case, take_while, take_while1};
use nom::character::complete::{alpha1, alphanumeric1, char, digit1, multispace0, one_of};
use nom::combinator::{map, map_opt, map_res, opt, recognize, value, verify};
use nom::error::{FromExternalError, ParseError};
use nom::multi::{many0, separated_list0};
use nom::sequence::{delimited, pair, separated_pair, tuple};
use serde_json::Value;

/// Parse a bare identifier: `[A-Za-z_][A-Za-z0-9_]*`.
pub fn identifier<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, String, E> {
    map(
        recognize(tuple((
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        ))),
        String::from,
    )(input)
}

/// Parse a backtick-quoted identifier: `` `any chars` ``.
pub fn quoted_identifier<'a, E: ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, String, E> {
    map(
        delimited(char('`'), take_while1(|c| c != '`'), char('`')),
        String::from,
    )(input)
}

/// Parse any identifier (bare or backtick-quoted).
pub fn any_identifier<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, String, E> {
    alt((quoted_identifier, identifier))(input)
}

/// Parse a string literal (single or double quoted; may be empty).
pub fn string_literal<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, String, E> {
    alt((
        map(
            delimited(char('"'), take_while(|c| c != '"'), char('"')),
            String::from,
        ),
        map(
            delimited(char('\''), take_while(|c| c != '\''), char('\'')),
            String::from,
        ),
    ))(input)
}

/// Parse an integer literal (optionally negative).
pub fn integer_literal<
    'a,
    E: ParseError<&'a str> + FromExternalError<&'a str, std::num::ParseIntError>,
>(
    input: &'a str,
) -> IResult<&'a str, i64, E> {
    map_res(recognize(tuple((opt(char('-')), digit1))), |s: &str| {
        s.parse::<i64>()
    })(input)
}

/// Parse a float literal (`-?d+.d+` with an optional exponent).
pub fn float_literal<
    'a,
    E: ParseError<&'a str> + FromExternalError<&'a str, std::num::ParseFloatError>,
>(
    input: &'a str,
) -> IResult<&'a str, f64, E> {
    map_res(
        recognize(tuple((
            opt(char('-')),
            digit1,
            char('.'),
            digit1,
            opt(tuple((one_of("eE"), opt(one_of("+-")), digit1))),
        ))),
        |s: &str| s.parse::<f64>(),
    )(input)
}

/// Parse a boolean literal (case-insensitive).
pub fn boolean_literal<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, bool, E> {
    alt((
        value(true, tag_no_case("true")),
        value(false, tag_no_case("false")),
    ))(input)
}

/// Parse a `null` literal (case-insensitive).
pub fn null_literal<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Value, E> {
    value(Value::Null, tag_no_case("null"))(input)
}

/// Parse a property value: string, float, integer, boolean, null, or a
/// (recursively typed) array of values.
pub fn property_value<
    'a,
    E: ParseError<&'a str>
        + FromExternalError<&'a str, std::num::ParseIntError>
        + FromExternalError<&'a str, std::num::ParseFloatError>,
>(
    input: &'a str,
) -> IResult<&'a str, Value, E> {
    alt((
        map(string_literal, Value::String),
        // inf/nan do not fit a JSON number — fail the parse rather than
        // silently coercing to 0 (TD-GRAPH-4 step-2 decision, both paths).
        map(
            map_opt(float_literal, serde_json::Number::from_f64),
            Value::Number,
        ),
        map(integer_literal, |i| {
            Value::Number(serde_json::Number::from(i))
        }),
        map(boolean_literal, Value::Bool),
        null_literal,
        map(
            delimited(
                char('['),
                separated_list0(
                    delimited(multispace0, char(','), multispace0),
                    property_value,
                ),
                char(']'),
            ),
            Value::Array,
        ),
    ))(input)
}

/// Parse a property-value map: `{ key: value, … }` with permissive
/// spacing and (backtick-)quoted keys.
pub fn property_value_map<
    'a,
    E: ParseError<&'a str>
        + FromExternalError<&'a str, std::num::ParseIntError>
        + FromExternalError<&'a str, std::num::ParseFloatError>,
>(
    input: &'a str,
) -> IResult<&'a str, HashMap<String, Value>, E> {
    map(
        delimited(
            pair(char('{'), multispace0),
            separated_list0(
                delimited(multispace0, char(','), multispace0),
                separated_pair(
                    any_identifier,
                    delimited(multispace0, char(':'), multispace0),
                    property_value,
                ),
            ),
            pair(multispace0, char('}')),
        ),
        |pairs| pairs.into_iter().collect(),
    )(input)
}

/// Parse a CONSTRAINT-map property assignment: `key: value` →
/// `PropertyConstraint::Equals(value)` — with the two TD-GRAPH-4 step-2
/// decisions applied:
///
/// * `null` is REJECTED in constraint position (MATCH/WHERE maps). Cypher
///   equality-with-null never matches — the idiom is `IS NULL` — and the
///   pre-convergence behavior on both grammars was to accept it and then
///   silently DROP the constraint downstream (an unconstrained scan).
///   The write-path `property_value_map` still accepts null (setting a
///   property to null is meaningful there).
/// * a float that does not fit an `f64`-backed JSON number (`1.0e999`)
///   fails the parse instead of silently coercing to `0` (the
///   pre-convergence `unwrap_or(0)` behavior, which produced a
///   `= 0` equality constraint with no error).
pub fn property_assignment<
    'a,
    E: ParseError<&'a str>
        + FromExternalError<&'a str, std::num::ParseIntError>
        + FromExternalError<&'a str, std::num::ParseFloatError>,
>(
    input: &'a str,
) -> IResult<&'a str, (String, crate::ast::PropertyConstraint), E> {
    map(
        separated_pair(
            any_identifier,
            delimited(multispace0, char(':'), multispace0),
            verify(constraint_value, |v| !v.is_null()),
        ),
        |(key, value)| (key, crate::ast::PropertyConstraint::Equals(value)),
    )(input)
}

/// A property value in CONSTRAINT position: the shared `property_value`
/// with the inf→0 coercion removed (`map_opt` — a non-finite float fails
/// the branch instead of becoming `0`).
fn constraint_value<
    'a,
    E: ParseError<&'a str>
        + FromExternalError<&'a str, std::num::ParseIntError>
        + FromExternalError<&'a str, std::num::ParseFloatError>,
>(
    input: &'a str,
) -> IResult<&'a str, Value, E> {
    alt((
        map(string_literal, Value::String),
        map(
            map_opt(float_literal, serde_json::Number::from_f64),
            Value::Number,
        ),
        map(integer_literal, |i| {
            Value::Number(serde_json::Number::from(i))
        }),
        map(boolean_literal, Value::Bool),
        null_literal,
        map(
            delimited(
                char('['),
                separated_list0(
                    delimited(multispace0, char(','), multispace0),
                    property_value,
                ),
                char(']'),
            ),
            Value::Array,
        ),
    ))(input)
}

/// Parse a constraint map: `{ key: value, … }` → per-key equality
/// constraints (permissive spacing, quoted keys — the shared shape).
pub fn constraint_map<
    'a,
    E: ParseError<&'a str>
        + FromExternalError<&'a str, std::num::ParseIntError>
        + FromExternalError<&'a str, std::num::ParseFloatError>,
>(
    input: &'a str,
) -> IResult<&'a str, HashMap<String, crate::ast::PropertyConstraint>, E> {
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
