//! Characterization tests for the two `ComparisonOperator` vocabularies.
//!
//! Captures the CURRENT serde wire names of both enums before they are
//! consolidated into one canonical enum. After consolidation these tests
//! must keep passing: the canonical enum carries `#[serde(alias = …)]`
//! entries for the legacy variant names, so payloads written by either
//! vocabulary remain deserializable (mixed-read-safe).

use proximadb_filter_expression::ComparisonOperator as Canonical;

fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(
    value: T,
) {
    let json = serde_json::to_string(&value).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(value, back, "round trip through {json}");
}

/// The lib.rs vocabulary — the one every external consumer uses.
#[test]
fn canonical_operator_wire_names_and_round_trips() {
    let all = [
        Canonical::Equals,
        Canonical::NotEquals,
        Canonical::GreaterThan,
        Canonical::GreaterThanOrEqual,
        Canonical::LessThan,
        Canonical::LessThanOrEqual,
        Canonical::In,
        Canonical::NotIn,
        Canonical::Contains,
        Canonical::StartsWith,
        Canonical::EndsWith,
        Canonical::Between,
        Canonical::IsNull,
        Canonical::IsNotNull,
        Canonical::Like,
        Canonical::Exists,
        Canonical::NotExists,
        Canonical::Regex,
    ];
    let expected = [
        "\"Equals\"",
        "\"NotEquals\"",
        "\"GreaterThan\"",
        "\"GreaterThanOrEqual\"",
        "\"LessThan\"",
        "\"LessThanOrEqual\"",
        "\"In\"",
        "\"NotIn\"",
        "\"Contains\"",
        "\"StartsWith\"",
        "\"EndsWith\"",
        "\"Between\"",
        "\"IsNull\"",
        "\"IsNotNull\"",
        "\"Like\"",
        "\"Exists\"",
        "\"NotExists\"",
        "\"Regex\"",
    ];
    for (op, want) in all.iter().zip(expected) {
        let json = serde_json::to_string(op).expect("serialize");
        assert_eq!(json, want);
        round_trip(op.clone());
    }
}

/// The former metadata_query vocabulary is gone; its legacy wire names
/// must deserialize into the canonical variants (mixed-read-safe).
#[test]
fn legacy_metadata_query_names_parse_into_canonical() {
    let cases = [
        ("Equal", Canonical::Equals),
        ("NotEqual", Canonical::NotEquals),
        ("LessThan", Canonical::LessThan),
        ("LessThanOrEqual", Canonical::LessThanOrEqual),
        ("GreaterThan", Canonical::GreaterThan),
        ("GreaterThanOrEqual", Canonical::GreaterThanOrEqual),
        ("Contains", Canonical::Contains),
        ("StartsWith", Canonical::StartsWith),
        ("EndsWith", Canonical::EndsWith),
        ("In", Canonical::In),
        ("NotIn", Canonical::NotIn),
        ("Exists", Canonical::Exists),
        ("NotExists", Canonical::NotExists),
        ("Regex", Canonical::Regex),
    ];
    for (legacy, canonical) in cases {
        let parsed: Canonical = serde_json::from_str(&format!("\"{legacy}\""))
            .unwrap_or_else(|e| panic!("legacy name {legacy} must parse: {e}"));
        assert_eq!(parsed, canonical);
    }
}

/// Whatever the canonical enum becomes, these legacy payload strings must
/// deserialize (the mixed-read-safe contract for persisted filters).
#[test]
fn legacy_payload_strings_still_parse() {
    let legacy = [
        "\"Equal\"",
        "\"NotEqual\"",
        "\"Exists\"",
        "\"NotExists\"",
        "\"Regex\"",
    ];
    for s in legacy {
        let op: Canonical = serde_json::from_str(s)
            .unwrap_or_else(|e| panic!("legacy payload {s} must parse: {e}"));
        round_trip(op);
    }
}
