//! Durable state a server writes OUTSIDE its leased storage roots.
//!
//! Some durable locations are chosen by per-process environment (k8s/ECS task
//! spec, systemd `Environment=`) or by config sections the ownership lease does
//! not cover. A cold backup run from an operator's shell cannot see the
//! server's environment, so the SERVING process records what it actually
//! resolved — absolute paths, from its own env + config + cwd — in
//! `<data_dir>/.proximadb-durable-locations.json` at boot. Backup reads that
//! record (and also checks its own environment) and refuses if any recorded
//! local location lies outside every leased root, instead of silently
//! producing a backup that omits it.

use std::path::{Path, PathBuf};

use proximadb::core::Config;
use serde::{Deserialize, Serialize};

pub(crate) const DURABLE_LOCATIONS_FILE: &str = ".proximadb-durable-locations.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DurableLocation {
    /// Where the location came from (config key or env var) — for the operator.
    pub(crate) source: String,
    /// The location as resolved by the process that recorded it: an absolute
    /// local path, or a remote URL (object storage is durable on its own).
    pub(crate) location: String,
}

/// Durable locations the lease does not cover, resolved from `config` and the
/// environment `env` (injected for tests). Relative local paths are made
/// absolute against the CURRENT process's cwd.
pub(crate) fn effective_durable_locations(
    config: &Config,
    env: impl Fn(&str) -> Option<String>,
) -> Vec<DurableLocation> {
    let env = |key: &str| env(key).filter(|value| !value.trim().is_empty());
    let mut found = Vec::new();
    let mut push = |source: &str, location: String| {
        found.push(DurableLocation {
            source: source.to_string(),
            location: absolutize(&location),
        });
    };

    if let Some(root) = config.queue.as_ref().and_then(|queue| queue.root.clone()) {
        push("[queue] root", root);
    }
    if let Some(root) = env("PROXIMADB_QUEUE_ROOT") {
        push("PROXIMADB_QUEUE_ROOT", root);
    }
    if let Some(dir) = env("PROXIMADB_WAL_LOCAL_DIR") {
        push("PROXIMADB_WAL_LOCAL_DIR", dir);
    }
    if let Some(url) = config.storage.wal_config.global_manifest_url.clone() {
        push("[storage.wal_config] global_manifest_url", url);
    }
    if let Some(path) = env("PROXIMADB_CORPUS_VERSION_PATH") {
        push("PROXIMADB_CORPUS_VERSION_PATH", path);
    }
    // io_trace sink spool — mirrors `ResolvedIoTraceSinkConfig::try_resolve`'s
    // env > TOML > default ladder for the enable switch and the directory.
    let sink = config
        .observability
        .as_ref()
        .and_then(|observability| observability.io_trace_sink.clone())
        .unwrap_or_default();
    let sink_enabled = env("PROXIMADB_IO_TRACE_SINK")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(sink.enabled);
    if sink_enabled {
        let dir = env("PROXIMADB_IO_TRACE_SINK_LOCAL_DIR")
            .or(sink.local_dir)
            .unwrap_or_else(|| "./log/io_trace".to_string());
        push("io_trace sink local_dir", dir);
    }
    if let Some(path) = config
        .query
        .as_ref()
        .and_then(|query| query.rl_planner.log_path.clone())
    {
        push("[query.rl_planner] log_path", path);
    }
    found
}

/// Local path of a location, if it is local (`file://` or scheme-less).
pub(crate) fn local_path(location: &str) -> Option<PathBuf> {
    if let Some(path) = location.strip_prefix("file://") {
        Some(PathBuf::from(path))
    } else if !location.contains("://") {
        Some(PathBuf::from(location))
    } else {
        None
    }
}

fn absolutize(location: &str) -> String {
    match local_path(location) {
        Some(path) => std::path::absolute(&path)
            .unwrap_or(path)
            .display()
            .to_string(),
        None => location.to_string(),
    }
}

/// Record this serving process's effective durable locations (atomic write).
/// Called at boot, after the ownership lease is held.
pub(crate) fn record(config: &Config) -> anyhow::Result<()> {
    let locations = effective_durable_locations(config, |key| std::env::var(key).ok());
    write_record(&config.server.data_dir, &locations)
}

pub(crate) fn write_record(data_dir: &Path, locations: &[DurableLocation]) -> anyhow::Result<()> {
    let target = data_dir.join(DURABLE_LOCATIONS_FILE);
    let tmp = data_dir.join(format!("{DURABLE_LOCATIONS_FILE}.tmp"));
    let body = serde_json::to_vec_pretty(locations)?;
    std::fs::write(&tmp, body).map_err(|e| anyhow::anyhow!("write {}: {e}", tmp.display()))?;
    std::fs::File::open(&tmp)
        .and_then(|file| file.sync_all())
        .map_err(|e| anyhow::anyhow!("fsync {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &target)
        .map_err(|e| anyhow::anyhow!("publish {}: {e}", target.display()))?;
    Ok(())
}

/// The record a server left in `data_dir`, if any.
pub(crate) fn read_record(data_dir: &Path) -> anyhow::Result<Option<Vec<DurableLocation>>> {
    let path = data_dir.join(DURABLE_LOCATIONS_FILE);
    match std::fs::read(&path) {
        Ok(bytes) => {
            Ok(Some(serde_json::from_slice(&bytes).map_err(|e| {
                anyhow::anyhow!("parse {}: {e}", path.display())
            })?))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow::anyhow!("read {}: {e}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |key| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| (*v).to_string())
        }
    }

    #[test]
    fn env_overrides_are_recorded_absolute() {
        let config = Config::default();
        let found = effective_durable_locations(
            &config,
            env_of(&[
                ("PROXIMADB_WAL_LOCAL_DIR", "relative/wal"),
                ("PROXIMADB_QUEUE_ROOT", "s3://bucket/queue"),
                ("PROXIMADB_CORPUS_VERSION_PATH", "/abs/corpus"),
            ]),
        );
        let wal = found
            .iter()
            .find(|loc| loc.source == "PROXIMADB_WAL_LOCAL_DIR")
            .expect("wal");
        assert!(Path::new(&wal.location).is_absolute(), "{}", wal.location);
        assert!(found.iter().any(|loc| loc.location == "s3://bucket/queue"));
        assert!(found.iter().any(|loc| loc.location == "/abs/corpus"));
    }

    #[test]
    fn io_trace_spool_recorded_only_when_enabled() {
        let config = Config::default();
        let off = effective_durable_locations(&config, env_of(&[]));
        assert!(off.iter().all(|loc| !loc.source.contains("io_trace")));
        let on = effective_durable_locations(&config, env_of(&[("PROXIMADB_IO_TRACE_SINK", "1")]));
        assert!(
            on.iter().any(
                |loc| loc.source.contains("io_trace") && loc.location.ends_with("log/io_trace")
            )
        );
    }

    #[test]
    fn record_round_trips() {
        let temp = tempfile::tempdir().expect("tempdir");
        let locations = vec![DurableLocation {
            source: "x".into(),
            location: "/y".into(),
        }];
        write_record(temp.path(), &locations).expect("write");
        assert_eq!(read_record(temp.path()).expect("read"), Some(locations));
        assert_eq!(
            read_record(&temp.path().join("missing")).expect("read missing"),
            None
        );
    }
}
