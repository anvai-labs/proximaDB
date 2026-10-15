//! Enumerate one root's contents for backup, restore-emptiness checks and
//! `--force` wipes.
//!
//! Rules (shared by every caller so they cannot drift):
//! * A file belongs to the DEEPEST root containing it: walking root `R` never
//!   descends into another configured root nested inside `R` — that root is
//!   walked (and locked) on its own.
//! * Coordination lock files (`access.lock`, `leader.lock`) are process state,
//!   never data — skipped at any depth.
//! * The server runtime-state file at the top of a root is skipped: it records
//!   a live process and must never be "restored" into a stopped one.
//! * Symlinks and special files are refused: the engine never creates them, and
//!   following one could copy (or wipe) data outside the root.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use proximadb_runtime_common::is_coordination_lock_file;

use crate::runtime_state::RUNTIME_STATE_FILE;

#[derive(Debug, Default)]
pub(crate) struct RootListing {
    /// Directories (relative to the root), parents before children.
    pub(crate) dirs: Vec<PathBuf>,
    /// Regular files (absolute paths).
    pub(crate) files: Vec<PathBuf>,
}

/// Walk `root`, skipping any directory in `other_roots` (nested roots).
pub(crate) fn list_root(
    root: &Path,
    other_roots: &BTreeSet<PathBuf>,
) -> anyhow::Result<RootListing> {
    let mut listing = RootListing::default();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let mut entries = std::fs::read_dir(&dir)
            .map_err(|e| anyhow::anyhow!("read directory {}: {e}", dir.display()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("read directory {}: {e}", dir.display()))?;
        // Deterministic manifest order.
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let name = entry.file_name();
            let meta = std::fs::symlink_metadata(&path)
                .map_err(|e| anyhow::anyhow!("stat {}: {e}", path.display()))?;
            let file_type = meta.file_type();
            if file_type.is_symlink() {
                anyhow::bail!(
                    "refusing symlink {} inside a storage root (cold backup copies regular files only)",
                    path.display()
                );
            }
            if file_type.is_dir() {
                if other_roots.contains(&path) {
                    continue;
                }
                listing.dirs.push(
                    path.strip_prefix(root)
                        .map_err(|_| anyhow::anyhow!("{} escaped its root", path.display()))?
                        .to_path_buf(),
                );
                stack.push(path);
            } else if file_type.is_file() {
                if is_coordination_lock_file(&name) {
                    continue;
                }
                if dir == root && name == RUNTIME_STATE_FILE {
                    continue;
                }
                listing.files.push(path);
            } else {
                anyhow::bail!(
                    "refusing special file {} inside a storage root",
                    path.display()
                );
            }
        }
    }
    // Pop order is not parent-first; sort so restore can create dirs in order.
    listing.dirs.sort_by_key(|dir| dir.components().count());
    Ok(listing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_root_files_attributed_to_deepest_root() {
        let temp = tempfile::tempdir().expect("tempdir");
        let outer = temp.path().canonicalize().expect("canon");
        let inner = outer.join("inner");
        std::fs::create_dir_all(inner.join("deep")).expect("mkdir");
        std::fs::write(outer.join("a.bin"), b"a").expect("write");
        std::fs::write(inner.join("deep/b.bin"), b"b").expect("write");

        let roots: BTreeSet<PathBuf> = [outer.clone(), inner.clone()].into();
        let others_for_outer: BTreeSet<PathBuf> = [inner.clone()].into();
        let outer_listing = list_root(&outer, &others_for_outer).expect("outer");
        assert_eq!(outer_listing.files, vec![outer.join("a.bin")]);
        assert!(
            outer_listing.dirs.is_empty(),
            "nested root is not a dir of the outer root"
        );

        let others_for_inner: BTreeSet<PathBuf> = roots
            .iter()
            .filter(|root| **root != inner)
            .cloned()
            .collect();
        let inner_listing = list_root(&inner, &others_for_inner).expect("inner");
        assert_eq!(inner_listing.files, vec![inner.join("deep/b.bin")]);
        assert_eq!(inner_listing.dirs, vec![PathBuf::from("deep")]);
    }

    #[test]
    fn lock_and_runtime_state_files_excluded() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().canonicalize().expect("canon");
        std::fs::create_dir_all(root.join("sub")).expect("mkdir");
        for name in ["access.lock", "leader.lock", RUNTIME_STATE_FILE, "keep.bin"] {
            std::fs::write(root.join(name), b"x").expect("write");
        }
        std::fs::write(root.join("sub/access.lock"), b"x").expect("write");
        // A runtime-state-named file BELOW the top level is ordinary data.
        std::fs::write(root.join("sub").join(RUNTIME_STATE_FILE), b"x").expect("write");

        let listing = list_root(&root, &BTreeSet::new()).expect("list");
        let mut files = listing.files.clone();
        files.sort();
        assert_eq!(
            files,
            vec![
                root.join("keep.bin"),
                root.join("sub").join(RUNTIME_STATE_FILE)
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_refused() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().canonicalize().expect("canon");
        std::os::unix::fs::symlink("/etc", root.join("link")).expect("symlink");
        assert!(list_root(&root, &BTreeSet::new()).is_err());
    }
}
