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

use crate::durable_locations;
use crate::runtime_state::RUNTIME_STATE_FILE;
use crate::storage_ownership::{RootRole, StorageOwnershipLease, root_plan};
use manifest::{
    FORMAT, FORMAT_VERSION, MANIFEST_FILE, Manifest, ManifestEntryDir, ManifestFile, ManifestRoot,
    ROOTS_DIR, RootContent, UncoveredRoot,
};

const COPY_BUF: usize = 1024 * 1024;

/// Present in `data_dir` while a restore is writing. The server refuses to
/// start on it, so a crashed, killed or failed restore can never be booted
/// from silently; it is removed only after every byte is written and fsynced.
pub(crate) const RESTORE_IN_PROGRESS_FILE: &str = ".proximadb-restore-in-progress";

/// Consistency rests on the ownership lease, which is an advisory `flock` on
/// unix. Elsewhere the lock is a no-op (it only warns), so a "server is
/// stopped" proof does not exist — refuse rather than pretend.
fn require_unix_lease() -> anyhow::Result<()> {
    anyhow::ensure!(
        cfg!(unix),
        "cold backup/restore require unix advisory locks to prove the server is stopped; \
         unsupported on this platform"
    );
    Ok(())
}

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

/// Durable state outside every leased root. Backup refuses rather than
/// produce a copy that silently omits it.
///
/// Fail-closed on two sources, because a backup run from an operator's shell
/// cannot see the server's per-pod environment:
/// * the record the SERVING process wrote at boot (its own env + config,
///   absolute paths) — required: a data dir without one cannot be checked;
/// * this process's own env + config (catches a config edited since boot).
///
/// Remote locations are durable on their own and are not refused.
fn unleased_durable_state(config: &Config) -> anyhow::Result<Vec<String>> {
    let recorded = durable_locations::read_record(&config.server.data_dir)?.ok_or_else(|| {
        anyhow::anyhow!(
            "no {} in {}: the server records where it writes durable state at boot, and \
             without that record backup cannot prove nothing lies outside the storage roots. \
             Start the server once with this version (then stop it) before taking a cold backup.",
            durable_locations::DURABLE_LOCATIONS_FILE,
            config.server.data_dir.display()
        )
    })?;
    let mine =
        durable_locations::effective_durable_locations(config, |key| std::env::var(key).ok());

    let leased: Vec<PathBuf> = root_plan(config)
        .local
        .into_iter()
        .filter_map(|(_, root)| absolute(&root).ok())
        .collect();
    let mut found = Vec::new();
    for location in recorded.iter().chain(mine.iter()) {
        let Some(path) = durable_locations::local_path(&location.location) else {
            continue;
        };
        let inside = absolute(&path)
            .map(|path| leased.iter().any(|root| path.starts_with(root)))
            .unwrap_or(false);
        let entry = format!("{} = {}", location.source, location.location);
        if !inside && !found.contains(&entry) {
            found.push(entry);
        }
    }
    Ok(found)
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

fn manifest_dirs_of(root: usize, dirs: &[ManifestEntryDir]) -> Vec<String> {
    dirs.iter()
        .filter(|dir| dir.root == root)
        .map(|dir| dir.path.clone())
        .collect()
}

/// Payload directories under `base`, deepest first (fsync children before
/// the parents that reference them).
fn manifest_dirs_for(base: &Path, relative: &[String]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = relative
        .iter()
        .map(|dir| base.join(dir))
        .filter(|dir| dir.is_dir())
        .collect();
    dirs.sort_by_key(|dir| std::cmp::Reverse(dir.components().count()));
    dirs
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
    require_unix_lease()?;
    anyhow::ensure!(
        !config
            .server
            .data_dir
            .join(RESTORE_IN_PROGRESS_FILE)
            .exists(),
        "a restore into these roots did not complete ({} present); finish it with \
         `restore --force` before taking a backup",
        RESTORE_IN_PROGRESS_FILE
    );
    let unleased = unleased_durable_state(config)?;
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
    // Make the payload's directory entries durable before the manifest that
    // vouches for them is published.
    for root in &roots {
        let base = to.join(ROOTS_DIR).join(root.index.to_string());
        if !base.exists() {
            continue;
        }
        for dir in manifest_dirs_for(&base, &manifest_dirs_of(root.index, &manifest.dirs)) {
            fsync_dir(&dir)?;
        }
        fsync_dir(&base)?;
    }
    if to.join(ROOTS_DIR).exists() {
        fsync_dir(&to.join(ROOTS_DIR))?;
    }

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
    require_unix_lease()?;
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

    // 5. Decide first: refuse non-empty roots without --force BEFORE marking
    //    anything, so a refused restore leaves the target exactly as it was.
    let mut to_wipe = Vec::new();
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
        to_wipe.push((root.path.clone(), others));
    }

    // 6. Mark the roots as mid-restore. From here until the marker is removed
    //    the server refuses to start on them.
    let data_root = roots
        .iter()
        .find(|root| root.roles.contains(&RootRole::DataDir))
        .map(|root| root.path.clone())
        .ok_or_else(|| anyhow::anyhow!("configuration has no data_dir root"))?;
    let sentinel = data_root.join(RESTORE_IN_PROGRESS_FILE);
    std::fs::write(&sentinel, b"restore in progress\n")
        .map_err(|e| anyhow::anyhow!("create {}: {e}", sentinel.display()))?;
    File::open(&sentinel)
        .and_then(|file| file.sync_all())
        .map_err(|e| anyhow::anyhow!("fsync {}: {e}", sentinel.display()))?;
    fsync_dir(&data_root)?;

    for (path, others) in &to_wipe {
        wipe_root(path, others, &sentinel)?;
    }

    // 7. Copy, re-verifying every byte as it is written.
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
                "{e:#} — the target roots are partially restored (the server will refuse to \
                 start on them); re-run restore with --force"
            )
        })?;
        anyhow::ensure!(
            size == file.size && sha256 == file.sha256,
            "{} changed during restore (checksum mismatch) — the target roots are partially \
             restored (the server will refuse to start on them); re-run restore with --force",
            src.display()
        );
        bytes += size;
    }

    // 8. Every directory entry durable, deepest first, then the roots.
    let mut created: Vec<PathBuf> = manifest
        .dirs
        .iter()
        .map(|dir| {
            manifest
                .root(dir.root)
                .map(|root| root.path.join(&dir.path))
        })
        .collect::<anyhow::Result<_>>()?;
    created.sort_by_key(|dir| std::cmp::Reverse(dir.components().count()));
    for dir in &created {
        fsync_dir(dir)?;
    }
    for root in &roots {
        fsync_dir(&root.path)?;
    }

    // 9. Only now is the restore complete: clear the marker durably.
    std::fs::remove_file(&sentinel)
        .map_err(|e| anyhow::anyhow!("remove {}: {e}", sentinel.display()))?;
    fsync_dir(&data_root)?;

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
fn wipe_root(root: &Path, nested: &BTreeSet<PathBuf>, keep: &Path) -> anyhow::Result<()> {
    for entry in
        std::fs::read_dir(root).map_err(|e| anyhow::anyhow!("read {}: {e}", root.display()))?
    {
        let entry = entry.map_err(|e| anyhow::anyhow!("read {}: {e}", root.display()))?;
        let path = entry.path();
        if proximadb_runtime_common::is_coordination_lock_file(&entry.file_name())
            || nested.contains(&path)
            || path == keep
        {
            continue;
        }
        let meta = std::fs::symlink_metadata(&path)
            .map_err(|e| anyhow::anyhow!("stat {}: {e}", path.display()))?;
        if meta.is_dir() {
            // A nested root deeper down keeps its own contents; only plain
            // directories are removed wholesale.
            if nested.iter().any(|root| root.starts_with(&path)) {
                wipe_root(&path, nested, keep)?;
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

    /// Create the roots and the durable-locations record (as a real server
    /// boot would), then seed data files. The record is resolved with an EMPTY
    /// environment so tests never depend on the developer's shell.
    fn seed(config: &Config) {
        drop(StorageOwnershipLease::acquire(config).expect("create roots"));
        durable_locations::write_record(
            &config.server.data_dir,
            &durable_locations::effective_durable_locations(config, |_| None),
        )
        .expect("durable-locations record");
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
        // Two seeded data files + the server's durable-locations record.
        assert_eq!(summary.files, 3);

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
    fn backup_refuses_without_server_record() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let config = local_config(&base);
        seed(&config);
        std::fs::remove_file(
            config
                .server
                .data_dir
                .join(durable_locations::DURABLE_LOCATIONS_FILE),
        )
        .expect("remove record");
        let err = run_backup(&config, &temp.path().join("bk")).expect_err("no record");
        assert!(
            format!("{err:#}").contains(durable_locations::DURABLE_LOCATIONS_FILE),
            "{err:#}"
        );
    }

    /// Review finding: the server's per-pod env (k8s/systemd) is invisible to a
    /// backup run from an operator shell. The server's own record must make
    /// backup refuse even though THIS process's environment is clean.
    #[test]
    fn backup_refuses_location_only_the_server_env_knew() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let config = local_config(&base);
        seed(&config);
        durable_locations::write_record(
            &config.server.data_dir,
            &[durable_locations::DurableLocation {
                source: "PROXIMADB_WAL_LOCAL_DIR".into(),
                location: temp.path().join("pod-wal").display().to_string(),
            }],
        )
        .expect("record");
        let err = run_backup(&config, &temp.path().join("bk")).expect_err("server env WAL");
        assert!(
            format!("{err:#}").contains("PROXIMADB_WAL_LOCAL_DIR"),
            "{err:#}"
        );
    }

    #[test]
    fn restore_sentinel_lifecycle() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = temp.path().join("live");
        let config = local_config(&base);
        seed(&config);
        let backup = temp.path().join("bk");
        run_backup(&config, &backup).expect("backup");
        let sentinel = config.server.data_dir.join(RESTORE_IN_PROGRESS_FILE);

        // A refused restore (non-empty, no --force) must not mark the roots.
        assert!(run_restore(&config, &backup, false).is_err());
        assert!(
            !sentinel.exists(),
            "a refused restore must not leave the marker"
        );

        // A crashed restore leaves the marker: backup refuses on it, and a
        // forced restore finishes the job and clears it.
        std::fs::write(&sentinel, b"crashed").expect("simulate crash");
        let err = run_backup(&config, &temp.path().join("bk2")).expect_err("marker");
        assert!(format!("{err:#}").contains("did not complete"), "{err:#}");
        run_restore(&config, &backup, true).expect("forced restore");
        assert!(!sentinel.exists(), "a completed restore clears the marker");
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
