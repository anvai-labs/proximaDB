//! Offline (cold) backup and restore of every LOCAL storage root a server owns.
//!
//! Consistency comes from the server's own ownership lease, not a new quiesce
//! mechanism: a running server holds an exclusive lock on every root, so if
//! backup can take that lease the server is provably stopped and nothing is
//! being written (fail-closed — a live server makes backup refuse, never tear).
//!
//! Scope, stated rather than implied:
//! * Local roots only. Object-store roots are durable on their own and are
//!   recorded in the manifest as `uncovered`, with a warning.
//! * The persistent warm-tier cache is derived data: locked, not copied, and
//!   restored empty.
//! * Durable state outside the lease (a `[queue]` root, `PROXIMADB_QUEUE_ROOT`,
//!   `PROXIMADB_WAL_LOCAL_DIR`) makes backup refuse instead of silently
//!   producing an incomplete copy.
//! * Restore requires the SAME absolute root layout as the source (catalog and
//!   WAL state can embed absolute `file://` paths); relocation is future work.

mod manifest;
mod walk;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use proximadb::core::Config;
use sha2::{Digest, Sha256};

use crate::runtime_state::RUNTIME_STATE_FILE;
use crate::storage_ownership::{RootRole, StorageOwnershipLease, root_plan};
use manifest::{
    FORMAT, FORMAT_VERSION, MANIFEST_FILE, Manifest, ManifestEntryDir, ManifestFile, ManifestRoot,
    ROOTS_DIR, RootContent, UncoveredRoot,
};

const COPY_BUF: usize = 1024 * 1024;

/// What a completed backup or restore did, for the operator summary.
#[derive(Debug)]
pub(crate) struct Summary {
    pub(crate) roots: usize,
    pub(crate) files: usize,
    pub(crate) bytes: u64,
    pub(crate) warnings: Vec<String>,
}

// --------------------------------------------------------------------------
// Root grouping
// --------------------------------------------------------------------------

/// Local roots grouped by path (several roles may share one directory), with
/// each group's content kind. `canonical` = resolve symlinks via the
/// filesystem (backup: roots exist); otherwise a lexical absolute path
/// (restore: roots may not exist yet).
fn grouped_roots(config: &Config, canonical: bool) -> anyhow::Result<Vec<ManifestRoot>> {
    let mut groups: BTreeMap<PathBuf, Vec<RootRole>> = BTreeMap::new();
    for (role, path) in root_plan(config).local {
        let key = if canonical {
            path.canonicalize()
                .map_err(|e| anyhow::anyhow!("resolve {role:?} root {}: {e}", path.display()))?
        } else {
            absolute(&path)?
        };
        groups.entry(key).or_default().push(role);
    }
    Ok(groups
        .into_iter()
        .enumerate()
        .map(|(index, (path, mut roles))| {
            roles.sort();
            roles.dedup();
            let content = if roles.iter().all(|role| *role == RootRole::SstCache) {
                RootContent::ExcludedCache
            } else {
                RootContent::Full
            };
            ManifestRoot {
                index,
                roles,
                path,
                content,
            }
        })
        .collect())
}

/// Absolute without touching the filesystem — but canonical when the path
/// exists, so an existing target compares equal to its canonical source.
fn absolute(path: &Path) -> anyhow::Result<PathBuf> {
    if let Ok(canonical) = path.canonicalize() {
        return Ok(canonical);
    }
    std::path::absolute(path)
        .map_err(|e| anyhow::anyhow!("absolute path for {}: {e}", path.display()))
}

fn other_roots(roots: &[ManifestRoot], index: usize) -> BTreeSet<PathBuf> {
    roots
        .iter()
        .filter(|root| root.index != index)
        .map(|root| root.path.clone())
        .collect()
}

/// Durable state the ownership lease does not cover. Backup refuses rather
/// than produce a copy that silently omits it.
fn unleased_durable_state(config: &Config) -> Vec<String> {
    let mut found = Vec::new();
    if config
        .queue
        .as_ref()
        .is_some_and(|queue| queue.root.is_some())
    {
        found.push("[queue] root".to_string());
    }
    for var in ["PROXIMADB_QUEUE_ROOT", "PROXIMADB_WAL_LOCAL_DIR"] {
        if std::env::var_os(var).is_some_and(|value| !value.is_empty()) {
            found.push(var.to_string());
        }
    }
    // An explicit WAL manifest location on LOCAL disk holds durable manifest
    // state; it is covered only when it sits inside a leased root (the shipped
    // `config/config.toml` points it at its own directory, outside them all).
    if let Some(url) = config.storage.wal_config.global_manifest_url.as_deref()
        && let Some(path) = local_path(url)
    {
        let leased: Vec<PathBuf> = root_plan(config)
            .local
            .into_iter()
            .filter_map(|(_, root)| absolute(&root).ok())
            .collect();
        let inside = absolute(&path)
            .map(|path| leased.iter().any(|root| path.starts_with(root)))
            .unwrap_or(false);
        if !inside {
            found.push(format!(
                "[storage.wal_config] global_manifest_url = {url} (outside every storage root)"
            ));
        }
    }
    found
}

fn local_path(url: &str) -> Option<PathBuf> {
    if let Some(path) = url.strip_prefix("file://") {
        Some(PathBuf::from(path))
    } else if !url.contains("://") {
        Some(PathBuf::from(url))
    } else {
        None
    }
}

// --------------------------------------------------------------------------
// Hash + copy primitives
// --------------------------------------------------------------------------

/// Stream `src` to `dst` (created exclusively), returning (size, sha256 hex).
/// The digest is computed over the bytes actually written.
fn copy_hashing(src: &Path, dst: &Path) -> anyhow::Result<(u64, String)> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("create {}: {e}", parent.display()))?;
    }
    let mut input = File::open(src).map_err(|e| anyhow::anyhow!("open {}: {e}", src.display()))?;
    let mut output = File::options()
        .write(true)
        .create_new(true)
        .open(dst)
        .map_err(|e| anyhow::anyhow!("create {}: {e}", dst.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; COPY_BUF];
    let mut size = 0u64;
    loop {
        let n = input
            .read(&mut buf)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", src.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        output
            .write_all(&buf[..n])
            .map_err(|e| anyhow::anyhow!("write {}: {e}", dst.display()))?;
        size += n as u64;
    }
    output
        .sync_all()
        .map_err(|e| anyhow::anyhow!("fsync {}: {e}", dst.display()))?;
    Ok((size, hex::encode(hasher.finalize())))
}

fn hash_file(path: &Path) -> anyhow::Result<(u64, String)> {
    let mut input =
        File::open(path).map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; COPY_BUF];
    let mut size = 0u64;
    loop {
        let n = input
            .read(&mut buf)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        size += n as u64;
    }
    Ok((size, hex::encode(hasher.finalize())))
}

fn fsync_dir(dir: &Path) -> anyhow::Result<()> {
    File::open(dir)
        .and_then(|handle| handle.sync_all())
        .map_err(|e| anyhow::anyhow!("fsync directory {}: {e}", dir.display()))
}

fn is_empty_or_missing(dir: &Path) -> anyhow::Result<bool> {
    match std::fs::read_dir(dir) {
        Ok(mut entries) => Ok(entries.next().is_none()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(e) => Err(anyhow::anyhow!("read {}: {e}", dir.display())),
    }
}

// --------------------------------------------------------------------------
// Backup
// --------------------------------------------------------------------------

pub(crate) fn run_backup(config: &Config, to: &Path) -> anyhow::Result<Summary> {
    let unleased = unleased_durable_state(config);
    anyhow::ensure!(
        unleased.is_empty(),
        "refusing cold backup: durable state outside the server's ownership lease is configured \
         ({}); a backup would silently omit it",
        unleased.join(", ")
    );

    // The lease FIRST: if the server is running this fails and nothing at all
    // is written — no partial backup directory, no manifest.
    let _lease = StorageOwnershipLease::acquire_existing(config).map_err(|e| {
        anyhow::anyhow!(
            "cannot take the storage ownership lease — is the server still running? \
             Stop it before a cold backup. ({e:#})"
        )
    })?;
    let roots = grouped_roots(config, true)?;

    anyhow::ensure!(
        is_empty_or_missing(to)?,
        "backup destination {} must be empty or not exist",
        to.display()
    );
    std::fs::create_dir_all(to).map_err(|e| anyhow::anyhow!("create {}: {e}", to.display()))?;
    let to = to
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("resolve {}: {e}", to.display()))?;
    for root in &roots {
        anyhow::ensure!(
            !to.starts_with(&root.path) && !root.path.starts_with(&to),
            "backup destination {} must not be inside (or contain) storage root {}",
            to.display(),
            root.path.display()
        );
    }

    let mut warnings = Vec::new();
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    let mut bytes = 0u64;
    for root in &roots {
        if root.content == RootContent::ExcludedCache {
            continue;
        }
        if root.roles.contains(&RootRole::SstCache) {
            warnings.push(format!(
                "the SST cache shares {} with {:?}; its contents are backed up as part of that root",
                root.path.display(),
                root.roles
            ));
        }
        let listing = walk::list_root(&root.path, &other_roots(&roots, root.index))?;
        let dest_root = to.join(ROOTS_DIR).join(root.index.to_string());
        for dir in listing.dirs {
            dirs.push(ManifestEntryDir {
                root: root.index,
                path: manifest::relative_string(Path::new(""), &dir)?,
            });
        }
        for file in listing.files {
            let relative = manifest::relative_string(&root.path, &file)?;
            let (size, sha256) = copy_hashing(&file, &dest_root.join(&relative))?;
            bytes += size;
            files.push(ManifestFile {
                root: root.index,
                path: relative,
                size,
                sha256,
            });
        }
    }

    let clean_shutdown = !config.server.data_dir.join(RUNTIME_STATE_FILE).exists();
    if !clean_shutdown {
        warnings.push(
            "the server did not shut down cleanly (runtime-state file present): this backup is \
             crash-consistent, and the WAL will replay on restore"
                .to_string(),
        );
    }
    let uncovered: Vec<UncoveredRoot> = root_plan(config)
        .remote
        .into_iter()
        .map(|(role, url)| UncoveredRoot { role, url })
        .collect();
    for root in &uncovered {
        warnings.push(format!(
            "{:?} lives in object storage ({}) and is NOT in this backup",
            root.role, root.url
        ));
    }

    let manifest = Manifest {
        format: FORMAT.to_string(),
        format_version: FORMAT_VERSION,
        proximadb_version: env!("CARGO_PKG_VERSION").to_string(),
        created_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis() as u64)
            .unwrap_or(0),
        clean_shutdown,
        roots: roots.clone(),
        uncovered,
        dirs,
        files,
    };
    // Manifest LAST, atomically: a backup without manifest.json is incomplete
    // by definition and restore refuses it.
    let tmp = to.join(format!("{MANIFEST_FILE}.tmp"));
    {
        let mut out = File::options()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| anyhow::anyhow!("create {}: {e}", tmp.display()))?;
        serde_json::to_writer_pretty(&mut out, &manifest)?;
        out.sync_all()
            .map_err(|e| anyhow::anyhow!("fsync {}: {e}", tmp.display()))?;
    }
    std::fs::rename(&tmp, to.join(MANIFEST_FILE))
        .map_err(|e| anyhow::anyhow!("publish manifest: {e}"))?;
    fsync_dir(&to)?;

    Ok(Summary {
        roots: roots.len(),
        files: manifest.files.len(),
        bytes,
        warnings,
    })
}

// --------------------------------------------------------------------------
// Restore
// --------------------------------------------------------------------------

pub(crate) fn run_restore(config: &Config, from: &Path, force: bool) -> anyhow::Result<Summary> {
    // 1. Structure. Nothing below this point writes until step 5.
    let manifest_path = from.join(MANIFEST_FILE);
    let manifest: Manifest = serde_json::from_reader(File::open(&manifest_path).map_err(|e| {
        anyhow::anyhow!(
            "open {} (no manifest = incomplete or not a cold backup): {e}",
            manifest_path.display()
        )
    })?)
    .map_err(|e| anyhow::anyhow!("parse {}: {e}", manifest_path.display()))?;
    manifest.validate()?;

    // 2. Every checksum, and no unlisted payload.
    let listed: BTreeSet<(usize, String)> = manifest
        .files
        .iter()
        .map(|file| (file.root, file.path.clone()))
        .collect();
    for file in &manifest.files {
        let path = from
            .join(ROOTS_DIR)
            .join(file.root.to_string())
            .join(&file.path);
        let (size, sha256) = hash_file(&path)?;
        anyhow::ensure!(
            size == file.size && sha256 == file.sha256,
            "checksum mismatch: {}/{}/{} (expected {} bytes {}, found {} bytes {})",
            ROOTS_DIR,
            file.root,
            file.path,
            file.size,
            file.sha256,
            size,
            sha256
        );
    }
    for root in &manifest.roots {
        let payload = from.join(ROOTS_DIR).join(root.index.to_string());
        if !payload.exists() {
            continue;
        }
        let listing = walk::list_root(&payload, &BTreeSet::new())?;
        for file in listing.files {
            let relative = manifest::relative_string(&payload, &file)?;
            anyhow::ensure!(
                listed.contains(&(root.index, relative.clone())),
                "backup contains a file the manifest does not list: {}/{}/{relative}",
                ROOTS_DIR,
                root.index
            );
        }
    }

    // 3. The target configuration must have the source's exact root layout.
    let target = grouped_roots(config, false)?;
    let shape = |roots: &[ManifestRoot]| -> Vec<(PathBuf, Vec<RootRole>, RootContent)> {
        roots
            .iter()
            .map(|root| (root.path.clone(), root.roles.clone(), root.content))
            .collect()
    };
    if shape(&manifest.roots) != shape(&target) {
        let mut table = String::from("backup roots:\n");
        for (path, roles, _) in shape(&manifest.roots) {
            table.push_str(&format!("  {roles:?} -> {}\n", path.display()));
        }
        table.push_str("target config roots:\n");
        for (path, roles, _) in shape(&target) {
            table.push_str(&format!("  {roles:?} -> {}\n", path.display()));
        }
        anyhow::bail!(
            "the target configuration's storage roots do not match the backup (restore needs the \
             same absolute layout; relocation is not supported yet)\n{table}"
        );
    }
    let target_remote: Vec<UncoveredRoot> = root_plan(config)
        .remote
        .into_iter()
        .map(|(role, url)| UncoveredRoot { role, url })
        .collect();
    anyhow::ensure!(
        target_remote == manifest.uncovered,
        "the target configuration's object-store roots differ from the backup's"
    );

    // 4. Only now create/lock the roots (refuses under a live server).
    let _lease = StorageOwnershipLease::acquire(config).map_err(|e| {
        anyhow::anyhow!(
            "cannot take the storage ownership lease — is a server running on these roots? ({e:#})"
        )
    })?;
    let roots = grouped_roots(config, true)?;

    // 5. Empty targets, or an explicit --force wipe.
    for root in &roots {
        let others = other_roots(&roots, root.index);
        let listing = walk::list_root(&root.path, &others)?;
        if listing.files.is_empty() {
            continue;
        }
        anyhow::ensure!(
            force,
            "storage root {} ({:?}) is not empty; refusing to overwrite it — re-run with --force \
             to replace its contents with the backup",
            root.path.display(),
            root.roles
        );
        wipe_root(&root.path, &others)?;
    }

    // 6. Copy, re-verifying every byte as it is written.
    let mut bytes = 0u64;
    for dir in &manifest.dirs {
        let root = manifest.root(dir.root)?;
        let path = root.path.join(&dir.path);
        std::fs::create_dir_all(&path)
            .map_err(|e| anyhow::anyhow!("create {}: {e}", path.display()))?;
    }
    for file in &manifest.files {
        let root = manifest.root(file.root)?;
        let src = from
            .join(ROOTS_DIR)
            .join(file.root.to_string())
            .join(&file.path);
        let dst = root.path.join(&file.path);
        let (size, sha256) = copy_hashing(&src, &dst).map_err(|e| {
            anyhow::anyhow!(
                "{e:#} — the target roots are now partially restored; re-run restore with --force"
            )
        })?;
        anyhow::ensure!(
            size == file.size && sha256 == file.sha256,
            "{} changed during restore (checksum mismatch) — the target roots are partially \
             restored; re-run restore with --force",
            src.display()
        );
        bytes += size;
    }
    for root in &roots {
        fsync_dir(&root.path)?;
    }

    let mut warnings = Vec::new();
    if manifest.proximadb_version != env!("CARGO_PKG_VERSION") {
        warnings.push(format!(
            "backup was taken by ProximaDB {}, restoring with {}",
            manifest.proximadb_version,
            env!("CARGO_PKG_VERSION")
        ));
    }
    if !manifest.clean_shutdown {
        warnings.push(
            "the backup is crash-consistent (source did not shut down cleanly); the WAL replays on \
             next start"
                .to_string(),
        );
    }
    for root in &manifest.uncovered {
        warnings.push(format!(
            "{:?} lives in object storage ({}) and was not part of this backup",
            root.role, root.url
        ));
    }
    Ok(Summary {
        roots: roots.len(),
        files: manifest.files.len(),
        bytes,
        warnings,
    })
}

/// Remove a root's contents, keeping coordination lock files (held by this
/// process) and nested roots (handled as their own roots).
fn wipe_root(root: &Path, nested: &BTreeSet<PathBuf>) -> anyhow::Result<()> {
    for entry in
        std::fs::read_dir(root).map_err(|e| anyhow::anyhow!("read {}: {e}", root.display()))?
    {
        let entry = entry.map_err(|e| anyhow::anyhow!("read {}: {e}", root.display()))?;
        let path = entry.path();
        if proximadb_runtime_common::is_coordination_lock_file(&entry.file_name())
            || nested.contains(&path)
        {
            continue;
        }
        let meta = std::fs::symlink_metadata(&path)
            .map_err(|e| anyhow::anyhow!("stat {}: {e}", path.display()))?;
        if meta.is_dir() {
            // A nested root deeper down keeps its own contents; only plain
            // directories are removed wholesale.
            if nested.iter().any(|root| root.starts_with(&path)) {
                wipe_root(&path, nested)?;
            } else {
                std::fs::remove_dir_all(&path)
                    .map_err(|e| anyhow::anyhow!("remove {}: {e}", path.display()))?;
            }
        } else {
            std::fs::remove_file(&path)
                .map_err(|e| anyhow::anyhow!("remove {}: {e}", path.display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_config(base: &Path) -> Config {
        let mut config = Config::default();
        config.server.data_dir = base.join("server_data");
        config.storage.metadata_url = format!("file://{}", base.join("metadata").display());
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", base.join("collections").display()),
            ..Default::default()
        }];
        config.storage.wal_config.enable_wal = true;
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}", base.join("wal").display());
        config.storage.wal_config.wal_local_dir = None;
        config.storage.sst_config = None;
        config.storage.viper_config = None;
        config.queue = None;
        config
    }

    /// Create the roots (as a real server run would) and seed data files.
    fn seed(config: &Config) {
        drop(StorageOwnershipLease::acquire(config).expect("create roots"));
        std::fs::create_dir_all(config.server.data_dir.join("pgwire")).expect("mkdir");
        std::fs::write(
            config.server.data_dir.join("pgwire/canonical-records.wal"),
            b"wal-bytes",
        )
        .expect("write");
        let collections = root_plan(config)
            .local
            .into_iter()
            .find(|(role, _)| *role == RootRole::StorageLocation(0))
            .map(|(_, path)| path)
            .expect("collections root");
        std::fs::create_dir_all(collections.join("c1")).expect("mkdir");
        std::fs::write(collections.join("c1/seg-0.pax"), vec![7u8; 4096]).expect("write");
    }

    fn wipe_all(base: &Path) {
        for name in ["server_data", "metadata", "collections", "wal"] {
            let _ = std::fs::remove_dir_all(base.join(name));
        }
    }

    #[test]
    fn backup_restore_round_trip_preserves_bytes() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let config = local_config(&base);
        seed(&config);
        let backup = temp.path().join("bk");
        let summary = run_backup(&config, &backup).expect("backup");
        assert_eq!(summary.files, 2);

        wipe_all(&base);
        run_restore(&config, &backup, false).expect("restore");
        assert_eq!(
            std::fs::read(config.server.data_dir.join("pgwire/canonical-records.wal"))
                .expect("read"),
            b"wal-bytes"
        );
        assert_eq!(
            std::fs::read(base.join("collections/c1/seg-0.pax")).expect("read"),
            vec![7u8; 4096]
        );
    }

    #[test]
    fn backup_refuses_while_lease_held() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let config = local_config(&base);
        seed(&config);
        let _live = StorageOwnershipLease::acquire(&config).expect("live server lease");
        let backup = temp.path().join("bk");
        assert!(run_backup(&config, &backup).is_err());
        assert!(
            !backup.join(MANIFEST_FILE).exists(),
            "a refused backup must leave no manifest"
        );
    }

    #[test]
    fn restore_refuses_tampered_backup_and_writes_nothing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let config = local_config(&base);
        seed(&config);
        let backup = temp.path().join("bk");
        run_backup(&config, &backup).expect("backup");

        let manifest: Manifest =
            serde_json::from_reader(File::open(backup.join(MANIFEST_FILE)).expect("open"))
                .expect("parse");
        let victim = &manifest.files[0];
        let victim_path = backup
            .join(ROOTS_DIR)
            .join(victim.root.to_string())
            .join(&victim.path);
        let mut bytes = std::fs::read(&victim_path).expect("read");
        bytes[0] ^= 0xff;
        std::fs::write(&victim_path, bytes).expect("write");

        wipe_all(&base);
        let err = run_restore(&config, &backup, false).expect_err("tampered backup");
        assert!(format!("{err:#}").contains("checksum mismatch"), "{err:#}");
        for name in ["server_data", "metadata", "collections", "wal"] {
            assert!(
                !base.join(name).exists(),
                "{name} must not be created by a refused restore"
            );
        }
    }

    #[test]
    fn restore_refuses_non_empty_target_without_force() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let config = local_config(&base);
        seed(&config);
        let backup = temp.path().join("bk");
        run_backup(&config, &backup).expect("backup");

        let stray = base.join("collections/stray.bin");
        std::fs::write(&stray, b"stray").expect("write");
        assert!(run_restore(&config, &backup, false).is_err());
        assert!(
            stray.exists(),
            "a refused restore must not touch the target"
        );

        run_restore(&config, &backup, true).expect("forced restore");
        assert!(!stray.exists(), "--force replaces the root's contents");
        assert_eq!(
            std::fs::read(base.join("collections/c1/seg-0.pax")).expect("read"),
            vec![7u8; 4096]
        );
    }

    #[test]
    fn root_set_mismatch_detected() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let config = local_config(&base);
        seed(&config);
        let backup = temp.path().join("bk");
        run_backup(&config, &backup).expect("backup");

        let mut other = config.clone();
        other
            .storage
            .storage_locations
            .push(proximadb::core::config::StorageLocation {
                url: format!("file://{}", base.join("extra").display()),
                ..Default::default()
            });
        wipe_all(&base);
        let err = run_restore(&other, &backup, false).expect_err("layout mismatch");
        assert!(format!("{err:#}").contains("do not match"), "{err:#}");
    }

    #[test]
    fn remote_locations_recorded_as_uncovered() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let mut config = local_config(&base);
        config
            .storage
            .storage_locations
            .push(proximadb::core::config::StorageLocation {
                url: "s3://bucket/prefix".into(),
                ..Default::default()
            });
        seed(&config);
        let backup = temp.path().join("bk");
        let summary = run_backup(&config, &backup).expect("backup");
        assert!(
            summary
                .warnings
                .iter()
                .any(|w| w.contains("s3://bucket/prefix"))
        );
        let manifest: Manifest =
            serde_json::from_reader(File::open(backup.join(MANIFEST_FILE)).expect("open"))
                .expect("parse");
        assert_eq!(
            manifest.uncovered,
            vec![UncoveredRoot {
                role: RootRole::StorageLocation(1),
                url: "s3://bucket/prefix".into()
            }]
        );
    }

    #[test]
    fn cache_root_contents_excluded() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let mut config = local_config(&base);
        config.storage.sst_config = Some(proximadb::core::config::SstConfig {
            data_directory: base.join("sst").display().to_string(),
            cache_local_disk_path: Some(base.join("cache").display().to_string()),
            ..Default::default()
        });
        seed(&config);
        std::fs::write(base.join("cache/warm.bin"), b"derived").expect("write");

        let backup = temp.path().join("bk");
        run_backup(&config, &backup).expect("backup");
        let manifest: Manifest =
            serde_json::from_reader(File::open(backup.join(MANIFEST_FILE)).expect("open"))
                .expect("parse");
        assert!(manifest.files.iter().all(|f| !f.path.ends_with("warm.bin")));
        assert!(
            manifest
                .roots
                .iter()
                .any(|r| r.content == RootContent::ExcludedCache)
        );
    }

    #[test]
    fn backup_refuses_wal_manifest_outside_roots() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let mut config = local_config(&base);
        seed(&config);
        config.storage.wal_config.global_manifest_url =
            Some(format!("file://{}", temp.path().join("manifest").display()));
        let err = run_backup(&config, &temp.path().join("bk")).expect_err("manifest outside");
        assert!(
            format!("{err:#}").contains("global_manifest_url"),
            "{err:#}"
        );

        // Inside a leased root it is ordinary root content: allowed.
        config.storage.wal_config.global_manifest_url =
            Some(format!("file://{}", base.join("wal/manifest").display()));
        run_backup(&config, &temp.path().join("bk2")).expect("manifest inside a root");
    }

    #[test]
    fn backup_refuses_unleased_queue_root() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let mut config = local_config(&base);
        config.queue = Some(proximadb::core::config::QueueRuntimeConfig {
            root: Some(format!("file://{}", base.join("queue").display())),
            ..Default::default()
        });
        seed(&config);
        let err = run_backup(&config, &temp.path().join("bk")).expect_err("queue root");
        assert!(format!("{err:#}").contains("ownership lease"), "{err:#}");
    }
}
