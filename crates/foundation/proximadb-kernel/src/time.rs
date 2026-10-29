//! Wall-clock helpers shared across the workspace.
//!
//! These exist so each subsystem does not re-implement (and drift on) the
//! `SystemTime::now → UNIX epoch → millis/nanos` idiom.

/// Current UNIX epoch time in milliseconds.
///
/// Returns `0` if the system clock is before the epoch (matches the
/// long-standing `unwrap_or(0)` behaviour of the call sites this replaces).
pub fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Current UNIX epoch time in nanoseconds (`0` before the epoch), the clock
/// basis of `valid_to_ns` / `is_record_dead`.
pub fn now_unix_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_unix_ms_is_monotonic_and_plausible() {
        let a = now_unix_ms();
        let b = now_unix_ms();
        assert!(a > 0, "epoch millis should be positive: {a}");
        assert!(b >= a, "clock must not go backwards: {a} -> {b}");
        // Sanity: later than 2020-01-01, earlier than 2100-01-01.
        assert!(a > 1_577_836_800_000 && a < 4_102_344_000_000);
    }

    #[test]
    fn now_unix_ns_is_plausible_and_agrees_with_ms() {
        let ns = now_unix_ns();
        let ms = now_unix_ms();
        // Later than 2020-01-01 in ns, and within a second of the ms clock.
        assert!(ns > 1_577_836_800_000_000_000, "epoch nanos: {ns}");
        assert!((ns / 1_000_000 - ms).abs() < 1_000, "ns {ns} vs ms {ms}");
    }
}
