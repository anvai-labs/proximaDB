//! Exclusive ownership of every local writable root used by a server.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::Context;
use proximadb::core::Config;
use proximadb_runtime_common::{AccessMode, FileLockSet};

/// Holds all root locks for the server lifetime. Partial acquisition is safe:
/// already-acquired locks drop if a later overlapping root is unavailable.
pub(crate) struct StorageOwnershipLease {
    _locks: FileLockSet,
}

impl StorageOwnershipLease {
    pub(crate) fn acquire(config: &Config) -> anyhow::Result<Self> {
        let roots = local_writable_roots(config)?;
        let locks = FileLockSet::acquire(&roots, AccessMode::Exclusive).with_context(|| {
            "one or more local storage roots are already owned; connect to the existing ProximaDB process instead of opening its files again"
        })?;
        for root in locks.roots() {
            tracing::info!(root = %root.display(), "acquired local-storage ownership");
        }
        Ok(Self { _locks: locks })
    }

    /// Like [`Self::acquire`], but for a process that must not invent state:
    /// every local root must already exist (cold backup of a stopped server).
    /// A missing root is an error, never silently created-and-locked empty.
    pub(crate) fn acquire_existing(config: &Config) -> anyhow::Result<Self> {
        for (role, root) in root_plan(config).local {
            anyhow::ensure!(
                root.is_dir(),
                "configured {role:?} root {} does not exist; nothing to back up there \
                 (wrong --config / --data-dir, or the server never ran with this config?)",
                root.display()
            );
        }
        Self::acquire(config)
    }
}

/// What a configured local root holds. Cold backup records roles so a restore
/// can prove the target configuration has the same shape as the source.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub(crate) enum RootRole {
    DataDir,
    Metadata,
    StorageLocation(usize),
    WalWriteBuffer,
    WalLocalDir,
    SstData,
    /// Persistent warm-tier cache: derived and rebuildable, so backup skips its
    /// contents (it is still locked like every other root).
    SstCache,
    ViperData,
}

/// Every root the configuration names, labelled by role. `local` roots are
/// paths this process owns; `remote` roots are object-store URLs (durable on
/// their own, and never copied by a local cold backup).
pub(crate) struct RootPlan {
    pub(crate) local: Vec<(RootRole, PathBuf)>,
    pub(crate) remote: Vec<(RootRole, String)>,
}

/// Pure: enumerates roots without touching the filesystem. The ownership lease
/// and cold backup both derive from this one list, so they can never disagree
/// about what the server owns.
pub(crate) fn root_plan(config: &Config) -> RootPlan {
    let mut plan = RootPlan {
        local: vec![(RootRole::DataDir, config.server.data_dir.clone())],
        remote: Vec::new(),
    };
    push_root(&mut plan, RootRole::Metadata, &config.storage.metadata_url);
    for (index, location) in config.storage.storage_locations.iter().enumerate() {
        push_root(&mut plan, RootRole::StorageLocation(index), &location.url);
    }
    // WAL roots are owned UNCONDITIONALLY. `enable_wal` only toggles an unused
    // optimized-writer flag at runtime — the WAL is written to these paths
    // either way — so gating them on it left a live WAL unleased (two servers
    // could share it) and silently missing from cold backups (acknowledged
    // writes lost on restore; found in review of the cold-backup PR).
    push_root(
        &mut plan,
        RootRole::WalWriteBuffer,
        &config.storage.wal_config.write_buffer_directory,
    );
    if let Some(path) = config.storage.wal_config.wal_local_dir.as_deref() {
        push_root(&mut plan, RootRole::WalLocalDir, path);
    }
    if let Some(sst) = config.storage.sst_config.as_ref() {
        push_root(&mut plan, RootRole::SstData, &sst.data_directory);
        if let Some(path) = sst.cache_local_disk_path.as_deref() {
            push_root(&mut plan, RootRole::SstCache, path);
        }
    }
    if let Some(viper) = config.storage.viper_config.as_ref() {
        push_root(&mut plan, RootRole::ViperData, &viper.data_directory);
    }
    plan
}

fn local_writable_roots(config: &Config) -> anyhow::Result<Vec<PathBuf>> {
    let mut roots = BTreeSet::new();
    for (_, candidate) in root_plan(config).local {
        std::fs::create_dir_all(&candidate)
            .with_context(|| format!("create local storage root {}", candidate.display()))?;
        roots.insert(
            candidate
                .canonicalize()
                .with_context(|| format!("canonicalize local root {}", candidate.display()))?,
        );
    }
    Ok(roots.into_iter().collect())
}

fn push_root(plan: &mut RootPlan, role: RootRole, value: &str) {
    if let Some(path) = value.strip_prefix("file://") {
        plan.local.push((role, PathBuf::from(path)));
    } else if !value.contains("://") {
        plan.local.push((role, PathBuf::from(value)));
    } else {
        plan.remote.push((role, value.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_config(root: &std::path::Path) -> Config {
        let mut config = Config::default();
        config.server.data_dir = root.join("instance");
        config.storage.storage_locations[0].url =
            format!("file://{}", root.join("storage").display());
        config.storage.metadata_url = format!("file://{}", root.join("metadata").display());
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}", root.join("wal").display());
        config.storage.sst_config = None;
        config.storage.viper_config = None;
        config
    }

    #[test]
    fn rejects_a_second_owner_for_the_same_roots() {
        let temp = tempfile::tempdir().expect("tempdir");
        let config = local_config(temp.path());
        let first = StorageOwnershipLease::acquire(&config).expect("first owner");
        assert!(StorageOwnershipLease::acquire(&config).is_err());
        drop(first);
        StorageOwnershipLease::acquire(&config).expect("released owner");
    }

    #[test]
    fn shared_root_conflicts_even_when_data_dirs_differ() {
        let temp = tempfile::tempdir().expect("tempdir");
        let shared = temp.path().join("shared");
        let mut first = local_config(&temp.path().join("first"));
        let mut second = local_config(&temp.path().join("second"));
        first.storage.storage_locations[0].url = format!("file://{}", shared.display());
        second.storage.storage_locations[0].url = format!("file://{}", shared.display());
        let _lease = StorageOwnershipLease::acquire(&first).expect("first owner");
        assert!(StorageOwnershipLease::acquire(&second).is_err());
    }

    #[test]
    fn remote_urls_are_not_treated_as_local_paths() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut config = local_config(temp.path());
        config.storage.storage_locations[0].url = "s3://bucket/prefix".into();
        let roots = local_writable_roots(&config).expect("roots");
        assert!(roots.iter().all(|root| !root.ends_with("bucket/prefix")));
    }

    #[test]
    fn root_plan_reports_remote_and_roles() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut config = local_config(temp.path());
        config
            .storage
            .storage_locations
            .push(proximadb::core::config::StorageLocation {
                url: "s3://bucket/prefix".into(),
                ..Default::default()
            });
        let plan = root_plan(&config);
        let roles: Vec<_> = plan.local.iter().map(|(role, _)| role.clone()).collect();
        assert_eq!(
            roles,
            vec![
                RootRole::DataDir,
                RootRole::Metadata,
                RootRole::StorageLocation(0),
                RootRole::WalWriteBuffer,
            ]
        );
        assert_eq!(
            plan.remote,
            vec![(
                RootRole::StorageLocation(1),
                "s3://bucket/prefix".to_string()
            )]
        );
    }

    #[test]
    fn wal_roots_are_owned_even_when_enable_wal_is_false() {
        // Regression (cold-backup review): the WAL is written whether or not
        // `enable_wal` is set, so its roots must be leased + backed up either way.
        let temp = tempfile::tempdir().expect("tempdir");
        let mut config = local_config(temp.path());
        config.storage.wal_config.enable_wal = false;
        config.storage.wal_config.wal_local_dir =
            Some(format!("file://{}", temp.path().join("waldisk").display()));
        let roles: Vec<_> = root_plan(&config)
            .local
            .into_iter()
            .map(|(role, _)| role)
            .collect();
        assert!(roles.contains(&RootRole::WalWriteBuffer));
        assert!(roles.contains(&RootRole::WalLocalDir));
    }

    #[test]
    fn acquire_existing_refuses_missing_roots() {
        let temp = tempfile::tempdir().expect("tempdir");
        let config = local_config(temp.path());
        assert!(StorageOwnershipLease::acquire_existing(&config).is_err());
        // Creating the roots (what a real server run does) makes it succeed.
        drop(StorageOwnershipLease::acquire(&config).expect("create + lock"));
        StorageOwnershipLease::acquire_existing(&config).expect("existing roots");
    }
}
