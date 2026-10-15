//! The cold-backup manifest: the one record of what a backup contains.
//!
//! A backup is complete iff `manifest.json` exists — it is written last, via a
//! fsynced temp file and an atomic rename, so a torn backup has no manifest.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::storage_ownership::RootRole;

pub(crate) const MANIFEST_FILE: &str = "manifest.json";
pub(crate) const FORMAT: &str = "proximadb-cold-backup";
pub(crate) const FORMAT_VERSION: u32 = 1;
/// Root contents live at `<backup>/roots/<index>/<relative path>`.
pub(crate) const ROOTS_DIR: &str = "roots";

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Manifest {
    pub(crate) format: String,
    pub(crate) format_version: u32,
    pub(crate) proximadb_version: String,
    pub(crate) created_unix_ms: u64,
    /// `false` when the server's runtime-state file was present, i.e. the last
    /// shutdown was not clean: the backup is crash-consistent and the WAL
    /// replays on restore.
    pub(crate) clean_shutdown: bool,
    pub(crate) roots: Vec<ManifestRoot>,
    /// Remote (object-store) roots: durable on their own, never copied here.
    pub(crate) uncovered: Vec<UncoveredRoot>,
    pub(crate) dirs: Vec<ManifestEntryDir>,
    pub(crate) files: Vec<ManifestFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ManifestRoot {
    pub(crate) index: usize,
    /// Several roles can share one directory (e.g. SST data + a storage
    /// location); they are one root with every role recorded.
    pub(crate) roles: Vec<RootRole>,
    pub(crate) path: PathBuf,
    pub(crate) content: RootContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RootContent {
    /// Every file is backed up.
    Full,
    /// A derived, rebuildable cache: locked during backup, contents skipped,
    /// and restored empty.
    ExcludedCache,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct UncoveredRoot {
    pub(crate) role: RootRole,
    pub(crate) url: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ManifestEntryDir {
    pub(crate) root: usize,
    pub(crate) path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ManifestFile {
    pub(crate) root: usize,
    pub(crate) path: String,
    pub(crate) size: u64,
    pub(crate) sha256: String,
}

impl Manifest {
    /// Structural validation — everything checkable without reading data.
    /// Restore runs this before touching any file, so a crafted or corrupt
    /// manifest can never direct a write outside a configured root.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.format == FORMAT,
            "not a ProximaDB cold backup (format {:?})",
            self.format
        );
        anyhow::ensure!(
            self.format_version == FORMAT_VERSION,
            "unsupported cold-backup format version {} (this build reads {FORMAT_VERSION})",
            self.format_version
        );
        let mut indexes = HashSet::new();
        for root in &self.roots {
            anyhow::ensure!(
                indexes.insert(root.index),
                "duplicate root index {}",
                root.index
            );
            anyhow::ensure!(!root.roles.is_empty(), "root {} has no roles", root.index);
        }
        let mut seen = HashSet::new();
        for (root, path) in self
            .dirs
            .iter()
            .map(|d| (d.root, &d.path))
            .chain(self.files.iter().map(|f| (f.root, &f.path)))
        {
            anyhow::ensure!(
                indexes.contains(&root),
                "entry {path:?} names unknown root {root}"
            );
            validate_relative(path)?;
            anyhow::ensure!(
                seen.insert((root, path.as_str())),
                "duplicate entry {path:?} in root {root}"
            );
        }
        for file in &self.files {
            anyhow::ensure!(
                file.sha256.len() == 64 && file.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                "malformed sha256 for {:?}",
                file.path
            );
            let root = self.root(file.root)?;
            anyhow::ensure!(
                root.content == RootContent::Full,
                "cache root {} must not list files ({:?})",
                file.root,
                file.path
            );
        }
        Ok(())
    }

    pub(crate) fn root(&self, index: usize) -> anyhow::Result<&ManifestRoot> {
        self.roots
            .iter()
            .find(|root| root.index == index)
            .ok_or_else(|| anyhow::anyhow!("unknown root index {index}"))
    }
}

/// A manifest path must be relative and made only of normal components, so it
/// can only ever resolve INSIDE its root.
pub(crate) fn validate_relative(path: &str) -> anyhow::Result<()> {
    anyhow::ensure!(!path.is_empty(), "empty entry path");
    anyhow::ensure!(
        !path.contains('\\'),
        "entry path {path:?} contains a backslash"
    );
    let parsed = Path::new(path);
    anyhow::ensure!(
        parsed
            .components()
            .all(|component| matches!(component, Component::Normal(_))),
        "entry path {path:?} must be relative with no '.', '..' or root components"
    );
    Ok(())
}

/// The relative path of `path` under `root`, `/`-joined (platform-neutral).
pub(crate) fn relative_string(root: &Path, path: &Path) -> anyhow::Result<String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| anyhow::anyhow!("{} is not under {}", path.display(), root.display()))?;
    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(part) => parts.push(
                part.to_str()
                    .ok_or_else(|| anyhow::anyhow!("non-UTF-8 path {}", path.display()))?
                    .to_string(),
            ),
            other => anyhow::bail!("unexpected component {other:?} in {}", path.display()),
        }
    }
    Ok(parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_with(files: Vec<ManifestFile>) -> Manifest {
        Manifest {
            format: FORMAT.into(),
            format_version: FORMAT_VERSION,
            proximadb_version: "test".into(),
            created_unix_ms: 0,
            clean_shutdown: true,
            roots: vec![ManifestRoot {
                index: 0,
                roles: vec![RootRole::DataDir],
                path: "/data".into(),
                content: RootContent::Full,
            }],
            uncovered: vec![],
            dirs: vec![],
            files,
        }
    }

    fn file(path: &str) -> ManifestFile {
        ManifestFile {
            root: 0,
            path: path.into(),
            size: 0,
            sha256: "0".repeat(64),
        }
    }

    #[test]
    fn manifest_rejects_parent_and_absolute_paths() {
        for bad in ["../escape", "/etc/passwd", "a/../../b", "./a", "a\\b", ""] {
            assert!(
                manifest_with(vec![file(bad)]).validate().is_err(),
                "{bad:?} must be rejected"
            );
        }
        manifest_with(vec![file("wal/segment-1.log")])
            .validate()
            .expect("normal relative path");
    }

    #[test]
    fn manifest_rejects_duplicates_and_unknown_roots() {
        assert!(
            manifest_with(vec![file("a"), file("a")])
                .validate()
                .is_err()
        );
        let mut stray = file("a");
        stray.root = 7;
        assert!(manifest_with(vec![stray]).validate().is_err());
    }

    #[test]
    fn manifest_rejects_wrong_format_and_bad_digest() {
        let mut wrong = manifest_with(vec![]);
        wrong.format_version = FORMAT_VERSION + 1;
        assert!(wrong.validate().is_err());
        let mut digest = file("a");
        digest.sha256 = "zz".into();
        assert!(manifest_with(vec![digest]).validate().is_err());
    }
}
