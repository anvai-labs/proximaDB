// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! `[oltp]` runtime gates (ADR-095). Default OFF per mandate #8: the atomic
//! commit path writes a new WAL frame version that older binaries cannot read.
//!
//! Resolution: `PROXIMADB_OLTP_ATOMIC_COMMIT` (env) > `[oltp] atomic_commit`
//! (installed once by the server at boot) > `false`.

use std::sync::OnceLock;

/// Env override for `[oltp] atomic_commit`.
pub const ATOMIC_COMMIT_ENV: &str = "PROXIMADB_OLTP_ATOMIC_COMMIT";

static CONFIGURED_ATOMIC_COMMIT: OnceLock<bool> = OnceLock::new();

/// Install the TOML value. First call wins (boot installs it once, before the
/// shared services open the canonical WAL).
pub fn install_atomic_commit(enabled: bool) {
    let _ = CONFIGURED_ATOMIC_COMMIT.set(enabled);
}

fn parse_flag(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

/// Whether pgwire COMMIT (and multi-row appends on the shared canonical WAL)
/// are atomic: one group frame per transaction.
pub fn atomic_commit_enabled() -> bool {
    if let Some(enabled) = std::env::var(ATOMIC_COMMIT_ENV)
        .ok()
        .as_deref()
        .and_then(parse_flag)
    {
        return enabled;
    }
    CONFIGURED_ATOMIC_COMMIT.get().copied().unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_parsing_accepts_common_spellings_and_rejects_noise() {
        assert_eq!(parse_flag("1"), Some(true));
        assert_eq!(parse_flag(" ON "), Some(true));
        assert_eq!(parse_flag("false"), Some(false));
        assert_eq!(parse_flag("maybe"), None);
    }
}
