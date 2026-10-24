//! ADR-097 P2.2/P2.3: the on-disk format of the relational write path, its
//! defaults, and the guard that keeps a binary from serving a data directory
//! written in a format it cannot fully read.
//!
//! **Format 2** (this binary) may seal canonical-WAL segments (P2.1), spill
//! relational rows to Layer A Parquet segments (TD-SEAM-3) and truncate the WAL
//! below what those segments hold (P2.2c). A binary that predates it reads only
//! the active WAL file and no spill segment, so on a format-2 directory it
//! would serve a silent subset of the data. Every binary from this one on
//! records the format it writes in [`FORMAT_MARKER`] before it writes it, and
//! refuses to open a directory whose marker is newer than it understands.
//! Binaries older than the marker cannot check it: rolling back across the
//! format-1 → 2 transition is not supported.
//!
//! Defaults: rotation, truncation and the spill are ON. Each gate's env var
//! still sets the value; `0` or `off` turns it off.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Default canonical-WAL segment size (`PROXIMADB_CANONICAL_WAL_SEGMENT_BYTES`).
pub const DEFAULT_WAL_SEGMENT_BYTES: u64 = 64 * 1024 * 1024;

/// Default seconds between truncator runs
/// (`PROXIMADB_CANONICAL_WAL_TRUNCATE_INTERVAL_SECS`).
pub const DEFAULT_TRUNCATE_INTERVAL_SECS: u64 = 300;

/// Default resident rows per partition before a spill flush
/// (`PROXIMADB_RELATIONAL_SPILL_MAX_RESIDENT`).
pub const DEFAULT_SPILL_MAX_RESIDENT: u64 = 65_536;

/// Default WAL lag, in entries, past which a partition's unflushed rows are
/// flushed so they stop pinning WAL truncation
/// (`PROXIMADB_RELATIONAL_SPILL_FLUSH_LAG_ENTRIES`). Equal to the change-feed
/// retention default, so a lagging partition costs no more WAL than the feed
/// already keeps.
pub const DEFAULT_SPILL_FLUSH_LAG_ENTRIES: u64 = 1_000_000;

/// A positive-integer gate that defaults ON: unset or empty ⇒ `default`;
/// `0` (in any spelling), `off`, `false`, `no`, `none` or `disabled` ⇒ `None`
/// (off); a positive integer ⇒ that value.
/// Anything else keeps the default and warns — a typo must neither disable
/// durability features silently nor pick an absurd value.
pub fn default_on_gate(env: &str, default: u64) -> Option<u64> {
    let Ok(raw) = std::env::var(env) else {
        return Some(default);
    };
    let value = raw.trim();
    if value.is_empty() {
        return Some(default);
    }
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "off" | "false" | "no" | "none" | "disabled"
    ) {
        return None;
    }
    match value.parse::<u64>() {
        Ok(0) => None,
        Ok(n) => Some(n),
        _ => {
            tracing::warn!(
                env,
                value,
                default,
                "unrecognised value; using the default (set 0 or off to disable)"
            );
            Some(default)
        }
    }
}

/// The format this binary writes and the newest it can read.
pub const CURRENT_FORMAT: u32 = 2;

/// Marker file, beside the canonical WAL.
pub const FORMAT_MARKER: &str = "DATA_FORMAT";

#[derive(Debug, Serialize, Deserialize)]
struct Marker {
    format: u32,
    written_by: String,
}

/// Check `dir`'s format marker and record [`CURRENT_FORMAT`] in it. Call before
/// opening anything in `dir`.
///
/// Refuses a marker newer than [`CURRENT_FORMAT`] (written by a newer binary)
/// or one that cannot be read — guessing would risk serving a subset of the
/// data. A missing marker is a format-1 directory, or a new one; either way it
/// is upgraded, durably (temp file, fsync, rename, fsync of the directory),
/// before the caller can write anything format 2.
pub fn ensure_data_format(dir: &Path) -> Result<u32> {
    let path = dir.join(FORMAT_MARKER);
    let found = match std::fs::read(&path) {
        Ok(bytes) => {
            let marker: Marker = serde_json::from_slice(&bytes).with_context(|| {
                format!(
                    "data format marker {} is unreadable; refusing to guess the format",
                    path.display()
                )
            })?;
            Some(marker.format)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error).with_context(|| format!("reading {}", path.display()));
        }
    };
    if let Some(format) = found {
        if format > CURRENT_FORMAT {
            anyhow::bail!(
                "{} was written in data format {format}, newer than this binary's {CURRENT_FORMAT} \
                 (ProximaDB {}); serving it could silently drop data. Run a newer binary.",
                dir.display(),
                crate::version::PROXIMADB_VERSION
            );
        }
        if format == CURRENT_FORMAT {
            return Ok(format);
        }
    }
    write_marker(dir, &path)?;
    tracing::info!(
        dir = %dir.display(),
        from = ?found,
        to = CURRENT_FORMAT,
        "data format marker recorded"
    );
    Ok(CURRENT_FORMAT)
}

fn write_marker(dir: &Path, path: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let bytes = serde_json::to_vec_pretty(&Marker {
        format: CURRENT_FORMAT,
        written_by: crate::version::PROXIMADB_VERSION.to_string(),
    })?;
    // Per-process temp name: two processes booting on one directory must not
    // rename each other's file out from under them.
    let tmp: PathBuf = dir.join(format!("{FORMAT_MARKER}.{}.tmp", std::process::id()));
    {
        use std::io::Write;
        let mut file =
            std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("installing {}", path.display()))?;
    std::fs::File::open(dir)
        .and_then(|d| d.sync_all())
        .with_context(|| format!("syncing {}", dir.display()))
}

/// Whether `spill_root` holds any file from an earlier run (a segment, or a
/// purge obligation still owed). Empty directories — what a DROP purge leaves
/// — are not state. An unreadable directory counts as state: when unsure,
/// keep the data reachable.
pub fn spill_root_has_state(spill_root: &Path) -> bool {
    match std::fs::read_dir(spill_root) {
        Ok(entries) => entries.into_iter().any(|entry| match entry {
            Ok(entry) => match entry.file_type() {
                Ok(kind) if kind.is_dir() => spill_root_has_state(&entry.path()),
                Ok(_) => true,
                Err(_) => true,
            },
            Err(_) => true,
        }),
        Err(error) => error.kind() != std::io::ErrorKind::NotFound,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_on_gates_parse_off_values_and_keep_the_default_on_typos() {
        const ENV: &str = "STORAGE_FORMAT_TEST_DEFAULT_ON_GATE";
        // SAFETY: a variable no other test reads.
        unsafe { std::env::remove_var(ENV) };
        assert_eq!(default_on_gate(ENV, 7), Some(7), "unset");
        for (raw, want) in [
            ("", Some(7)),
            ("0", None),
            ("00", None),
            ("none", None),
            ("disabled", None),
            ("off", None),
            ("OFF", None),
            ("false", None),
            ("no", None),
            ("42", Some(42)),
            (" 42 ", Some(42)),
            ("-1", Some(7)),
            ("abc", Some(7)),
        ] {
            unsafe { std::env::set_var(ENV, raw) };
            assert_eq!(default_on_gate(ENV, 7), want, "{raw:?}");
        }
        unsafe { std::env::remove_var(ENV) };
    }

    #[test]
    fn the_marker_upgrades_old_directories_and_refuses_newer_ones() -> Result<()> {
        let dir = tempfile::tempdir()?;
        // New (or format-1) directory: recorded as the current format.
        assert_eq!(ensure_data_format(dir.path())?, CURRENT_FORMAT);
        assert_eq!(
            ensure_data_format(dir.path())?,
            CURRENT_FORMAT,
            "idempotent"
        );
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "no temp file left behind");

        // An explicit older marker is upgraded.
        std::fs::write(
            dir.path().join(FORMAT_MARKER),
            br#"{"format":1,"written_by":"old"}"#,
        )?;
        assert_eq!(ensure_data_format(dir.path())?, CURRENT_FORMAT);

        // A newer one refuses, and is left as it was.
        let newer = format!(
            r#"{{"format":{},"written_by":"future"}}"#,
            CURRENT_FORMAT + 1
        );
        std::fs::write(dir.path().join(FORMAT_MARKER), &newer)?;
        let err = ensure_data_format(dir.path()).expect_err("newer format");
        assert!(
            format!("{err:#}").contains("newer than this binary"),
            "{err:#}"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join(FORMAT_MARKER))?,
            newer
        );

        // So does an unreadable one.
        std::fs::write(dir.path().join(FORMAT_MARKER), b"not json")?;
        assert!(ensure_data_format(dir.path()).is_err());
        Ok(())
    }

    #[test]
    fn spill_state_is_any_file_under_the_root_not_an_empty_directory() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().join("record-spill");
        assert!(!spill_root_has_state(&root), "missing root");
        std::fs::create_dir_all(root.join("tenant").join("orders"))?;
        assert!(
            !spill_root_has_state(&root),
            "empty directories (what a DROP purge leaves) are not state"
        );
        std::fs::write(
            root.join("tenant").join("orders").join("spill-1.parquet"),
            b"x",
        )?;
        assert!(spill_root_has_state(&root));
        Ok(())
    }
}
