// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Guard: every use of the anonymous port identity is listed here with its
//! reason. A request path must pass the request tenant (`PortIdentity::for_tenant`
//! or `From<&TenantContext>`) so collection resolution is scoped to it; a new,
//! unlisted anonymous use fails this test.

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    /// `file → (count, reason)`. Keep the reason specific.
    const ALLOWED: &[(&str, usize, &str)] = &[
        (
            "crates/binding/proximadb-embedded/src/lib.rs",
            1,
            "embedded single-process library: no request tenant exists",
        ),
        (
            "crates/platform/proximadb-api/src/lib.rs",
            3,
            "unit tests of the API port wiring",
        ),
        (
            "crates/platform/proximadb-runtime/src/handlers.rs",
            4,
            "the deprecated non-tenant v1 handler (test callers only) and its tests",
        ),
        (
            "src/network/rest/canonical/rank_backend.rs",
            1,
            "vector leg runs after scope_collection resolved the collection within the \
             request tenant",
        ),
        (
            "src/query/facade/strategies/vector.rs",
            1,
            "internal execution outside a request scope (requests carry request_scope)",
        ),
        (
            "src/query/federated/execution/mod.rs",
            1,
            "internal execution outside a request scope (requests carry request_scope)",
        ),
        (
            "src/services/dml/mod.rs",
            1,
            "scan is scoped by the TenantContext argument; identity carries no subject",
        ),
        ("src/services/dml/tests.rs", 5, "unit tests"),
        (
            "src/services/record_store.rs",
            1,
            "get is scoped by the TenantContext argument; identity carries no subject",
        ),
    ];

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }

    fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "target") {
                    continue;
                }
                rust_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    #[test]
    fn anonymous_port_identity_uses_are_all_listed() {
        // Built so this file does not match itself.
        let needle = format!("PortIdentity::{}()", "anonymous");
        let root = workspace_root();
        let mut files = Vec::new();
        for top in ["src", "crates", "apps"] {
            rust_files(&root.join(top), &mut files);
        }
        let mut found: BTreeMap<String, usize> = BTreeMap::new();
        for file in files {
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            let count = text.matches(&needle).count();
            if count > 0 {
                let rel = file
                    .strip_prefix(&root)
                    .map(|p| p.to_string_lossy().replace('\\', "/"))
                    .unwrap_or_else(|_| file.to_string_lossy().into_owned());
                found.insert(rel, count);
            }
        }
        assert!(
            !found.is_empty(),
            "guard scanned nothing under {}: the workspace layout changed",
            root.display()
        );
        let allowed: BTreeMap<String, usize> = ALLOWED
            .iter()
            .map(|(file, count, _)| (file.to_string(), *count))
            .collect();
        let mut problems = Vec::new();
        for (file, count) in &found {
            match allowed.get(file) {
                Some(limit) if count <= limit => {}
                Some(limit) => problems.push(format!(
                    "{file}: {count} uses, {limit} allowed — pass the request tenant instead"
                )),
                None => problems.push(format!(
                    "{file}: {count} unlisted use(s) — pass the request tenant, or list it with a reason"
                )),
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }
}
