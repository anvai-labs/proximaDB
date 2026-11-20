//! Pure parsing of default-ON positive-integer env gates.
//!
//! Moved here from the root crate's `services::storage_format` (which
//! re-exports it) so storage crates below the root — the Iceberg engine owns
//! the MATERIALIZE generation window that metadata GC and the bounded
//! `metadata-log` share (TD-SPINE-3) — parse the same gate with the same
//! rules instead of a drifting twin.

/// A positive-integer gate that defaults ON, over its raw value: unset or
/// empty ⇒ `default`; `0` (in any spelling), `off`, `false`, `no`, `none` or
/// `disabled` ⇒ `None` (off); a positive integer ⇒ that value. Anything else
/// keeps the default (a typo must neither disable a feature silently nor pick
/// an absurd value; the env-reading wrappers warn).
pub fn default_on_gate_value(raw: Option<&str>, default: u64) -> Option<u64> {
    let Some(raw) = raw else {
        return Some(default);
    };
    let value = raw.trim();
    if value.is_empty() {
        return Some(default);
    }
    if is_off_like(value) {
        return None;
    }
    match value.parse::<u64>() {
        Ok(0) => None,
        Ok(n) => Some(n),
        _ => Some(default),
    }
}

/// Whether `value` is one of the OFF spellings [`default_on_gate_value`] accepts.
pub fn is_off_like(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "off" | "false" | "no" | "none" | "disabled" | "0"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_on_gate_value_spellings() {
        assert_eq!(default_on_gate_value(None, 3), Some(3));
        assert_eq!(default_on_gate_value(Some(""), 3), Some(3));
        assert_eq!(default_on_gate_value(Some(" 7 "), 3), Some(7));
        assert_eq!(default_on_gate_value(Some("0"), 3), None);
        assert_eq!(default_on_gate_value(Some("OFF"), 3), None);
        assert_eq!(default_on_gate_value(Some("typo"), 3), Some(3));
    }
}
