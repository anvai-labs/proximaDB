use proximadb_runtime_common::file_lock::FileLockManager;

#[cfg(unix)]
#[test]
fn conditional_file_checks_sibling_guards_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let offset = dir.path().join("offset.meta");
    let lease = dir.path().join("lease.meta");
    std::fs::write(&offset, b"old").unwrap();
    std::fs::write(&lease, b"successor").unwrap();
    assert!(
        !FileLockManager::compare_exchange_file(
            &offset,
            Some(b"old"),
            b"new",
            &[(&lease, Some(b"predecessor"))]
        )
        .unwrap()
    );
    assert_eq!(std::fs::read(&offset).unwrap(), b"old");
    assert!(
        FileLockManager::compare_exchange_file(
            &offset,
            Some(b"old"),
            b"new",
            &[(&lease, Some(b"successor"))]
        )
        .unwrap()
    );
    assert_eq!(std::fs::read(&offset).unwrap(), b"new");
    assert_eq!(std::fs::read(&lease).unwrap(), b"successor");
}

#[cfg(unix)]
#[test]
fn conditional_file_guard_distinguishes_absent_and_empty() {
    let dir = tempfile::tempdir().unwrap();
    let offset = dir.path().join("offset.meta");
    let lease = dir.path().join("lease.meta");
    assert!(
        !FileLockManager::compare_exchange_file(&offset, None, b"new", &[(&lease, Some(b""))])
            .unwrap()
    );
    assert!(
        FileLockManager::compare_exchange_file(&offset, None, b"old", &[(&lease, None)]).unwrap()
    );
    std::fs::write(&lease, b"").unwrap();
    assert!(
        !FileLockManager::compare_exchange_file(&offset, Some(b"old"), b"new", &[(&lease, None)])
            .unwrap()
    );
    assert_eq!(std::fs::read(&offset).unwrap(), b"old");
}

#[cfg(unix)]
#[test]
fn conditional_file_rejects_unsafe_guard_paths() {
    let dir = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let offset = dir.path().join("offset.meta");
    let lease = dir.path().join("lease.meta");
    std::fs::write(&lease, b"owner").unwrap();
    let alias = dir.path().join("alias");
    std::os::unix::fs::symlink(&lease, &alias).unwrap();
    let subdir = dir.path().join("subdir");
    std::fs::create_dir(&subdir).unwrap();
    for guard in [
        alias,
        subdir,
        other.path().join("lease.meta"),
        dir.path().join("access.lock"),
        dir.path().join("leader.lock"),
    ] {
        assert!(
            FileLockManager::compare_exchange_file(&offset, None, b"new", &[(&guard, None)])
                .is_err(),
            "guard {guard:?}"
        );
        assert!(!offset.exists());
    }
}

#[cfg(unix)]
#[test]
fn conditional_file_child_process() {
    use std::io::{Read, Write};
    let Some(path) = std::env::var_os("LEASE_CAS_TEST_PATH") else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    match std::env::var("LEASE_CAS_TEST_ACTION").unwrap().as_str() {
        "hold" => {
            let _guard = FileLockManager::acquire(
                path.parent().unwrap(),
                proximadb_runtime_common::file_lock::AccessMode::Exclusive,
            )
            .unwrap();
            println!("LOCK_READY");
            std::io::stdout().flush().unwrap();
            let _ = std::io::stdin().read(&mut [0]);
        }
        "contended" => {
            assert!(!FileLockManager::compare_exchange_file(&path, None, b"bad", &[]).unwrap())
        }
        "publish" => {
            assert!(FileLockManager::compare_exchange_file(&path, None, b"new", &[]).unwrap())
        }
        action => panic!("unknown child action {action}"),
    }
}

#[cfg(unix)]
fn child_command(path: &std::path::Path, action: &str) -> std::process::Command {
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "conditional_file_child_process", "--nocapture"])
        .env("LEASE_CAS_TEST_PATH", path)
        .env("LEASE_CAS_TEST_ACTION", action);
    command
}

#[cfg(unix)]
struct ChildGuard(std::process::Child);

#[cfg(unix)]
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[cfg(unix)]
fn run_child(path: &std::path::Path, action: &str) {
    let mut child = ChildGuard(
        child_command(path, action)
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success(), "child {action}: {status}");
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "child {action} did not finish"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[cfg(unix)]
#[test]
fn conditional_file_excludes_separate_process_and_recovers_after_holder_killed() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state");
    let mut holder = ChildGuard(
        child_command(&path, "hold")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut output = BufReader::new(holder.0.stdout.take().unwrap());
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let reader = std::thread::spawn(move || {
        loop {
            let mut line = String::new();
            if output.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            if line.trim().ends_with("LOCK_READY") {
                let _ = ready_tx.send(());
                break;
            }
        }
    });
    let ready = ready_rx.recv_timeout(std::time::Duration::from_secs(5));
    if ready.is_err() {
        drop(holder);
        reader.join().unwrap();
        panic!("holder did not become ready: {ready:?}");
    }
    reader.join().unwrap();
    run_child(&path, "contended");
    drop(holder);
    assert!(!path.exists());
    run_child(&path, "publish");
    assert_eq!(std::fs::read(path).unwrap(), b"new");
}

#[cfg(unix)]
#[test]
fn conditional_file_requires_exact_expected_contents() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state");
    assert!(FileLockManager::compare_exchange_file(&path, None, b"first", &[]).unwrap());
    assert!(!FileLockManager::compare_exchange_file(&path, None, b"overwrite", &[]).unwrap());
    assert!(
        !FileLockManager::compare_exchange_file(&path, Some(b"wrong"), b"overwrite", &[]).unwrap()
    );
    assert!(FileLockManager::compare_exchange_file(&path, Some(b"first"), b"second", &[]).unwrap());
    assert_eq!(std::fs::read(path).unwrap(), b"second");
}

#[cfg(unix)]
#[test]
fn conditional_file_preserves_empty_versus_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state");
    assert!(!FileLockManager::compare_exchange_file(&path, Some(b""), b"wrong", &[]).unwrap());
    assert!(FileLockManager::compare_exchange_file(&path, None, b"", &[]).unwrap());
    assert!(!FileLockManager::compare_exchange_file(&path, None, b"wrong", &[]).unwrap());
    assert!(FileLockManager::compare_exchange_file(&path, Some(b""), b"value", &[]).unwrap());
}

#[cfg(not(unix))]
#[test]
fn conditional_file_rejects_unqualified_platform() {
    let dir = tempfile::tempdir().unwrap();
    let error =
        FileLockManager::compare_exchange_file(&dir.path().join("state"), None, b"bad", &[])
            .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
    assert!(!dir.path().join("state").exists());
}

#[test]
fn conditional_file_cannot_replace_its_lock_inode() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["access.lock", "leader.lock", "Access.lock", "LEADER.LOCK"] {
        assert!(
            FileLockManager::compare_exchange_file(&dir.path().join(name), None, b"bad", &[])
                .is_err()
        );
    }
}

#[cfg(unix)]
#[test]
fn conditional_file_rejects_symbolic_link_target() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("protected");
    let link = dir.path().join("state");
    std::fs::write(&target, b"original").unwrap();
    std::os::unix::fs::symlink(&target, &link).unwrap();
    assert!(FileLockManager::compare_exchange_file(&link, Some(b"original"), b"bad", &[]).is_err());
    assert_eq!(std::fs::read(target).unwrap(), b"original");
}

#[cfg(unix)]
#[test]
fn conditional_file_rejects_linked_lock_without_modifying_target() {
    for hardlink in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let protected = dir.path().join("protected");
        let lock = dir.path().join("access.lock");
        std::fs::write(&protected, b"original").unwrap();
        if hardlink {
            std::fs::hard_link(&protected, &lock).unwrap();
        } else {
            std::os::unix::fs::symlink(&protected, &lock).unwrap();
        }
        assert!(
            FileLockManager::compare_exchange_file(&dir.path().join("state"), None, b"new", &[])
                .is_err()
        );
        assert_eq!(std::fs::read(&protected).unwrap(), b"original");
    }
}
