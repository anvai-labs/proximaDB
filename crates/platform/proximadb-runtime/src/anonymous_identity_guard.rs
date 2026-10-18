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

    /// Same contract for the owned form's tenant-less default.
    const ALLOWED_OWNED_DEFAULT: &[(&str, usize, &str)] = &[
        (
            "src/datafusion/mod.rs",
            2,
            "create_session_context registers no vector/graph function (the identity is \
             unused); create_session_context_with_vector_ops has no production caller",
        ),
        (
            "src/datafusion/cross_modal.rs",
            2,
            "VectorSearchTableFunction::new (test constructor) and a unit test; production \
             registers with_identity",
        ),
    ];

    /// Unscoped table resolution (`CatalogManager::resolve_table`). Request
    /// paths resolve with `resolve_table_scoped` and the request tenant.
    const ALLOWED_UNSCOPED_RESOLVE: &[(&str, usize, &str)] = &[
        (
            "crates/binding/proximadb-embedded/src/lib.rs",
            1,
            "embedded single-process library: no request tenant exists",
        ),
        (
            "crates/binding/proximadb-embedded/src/python_dataframe.rs",
            1,
            "embedded single-process library: no request tenant exists",
        ),
        (
            "crates/binding/proximadb-embedded/src/tests.rs",
            1,
            "unit tests",
        ),
        (
            "crates/control/proximadb-catalog/src/fc_metamodel.rs",
            1,
            "metamodel name lookup within an explicit namespace (a different resolver)",
        ),
        (
            "crates/control/proximadb-catalog/src/manager.rs",
            1,
            "resolve_table_scoped's own base resolution before tenant prefixing",
        ),
        (
            "src/catalog/index_location_resolver.rs",
            1,
            "storage path resolution for a collection id already resolved by its caller",
        ),
        ("src/catalog/mod.rs", 5, "unit tests"),
        (
            "src/services/collection/manager.rs",
            1,
            "the internal collection lookup (collection_internal); request paths apply the \
             ownership rule on its result",
        ),
        ("src/services/ddl/mod.rs", 5, "unit tests"),
        ("src/services/dml/tests.rs", 20, "unit tests"),
        (
            "src/services/external_collection/service.rs",
            1,
            "unit test",
        ),
        (
            "src/services/snapshot/coordinator.rs",
            2,
            "discovery projections for a collection the caller already resolved within its \
             tenant, and background discovery",
        ),
        (
            "src/storage/persistence/write_ahead_log/mod.rs",
            1,
            "internal WAL/recovery collection lookup; no request reaches it",
        ),
    ];

    /// Direct catalog access (`CatalogManager::default_catalog`) bypasses the
    /// tenant-scoped resolver. Every use is listed; request-facing ones state how
    /// they scope to the request tenant.
    const ALLOWED_DEFAULT_CATALOG: &[(&str, usize, &str)] = &[
        (
            "crates/control/proximadb-catalog/src/iceberg_rest_service.rs",
            14,
            "Iceberg REST: every method maps the namespace into the request tenant's \
             (scoped_namespace) and checks visibility (visible_entry) before use",
        ),
        (
            "crates/control/proximadb-catalog/src/manager.rs",
            4,
            "the resolver itself",
        ),
        (
            "crates/control/proximadb-catalog/src/model_registry_service.rs",
            1,
            "model registry service owns tenant scoping of its own documents",
        ),
        ("src/catalog/mod.rs", 7, "unit tests"),
        (
            "src/cluster/primary_pod_registry.rs",
            5,
            "operator-only placement registry (internal)",
        ),
        ("src/database.rs", 2, "startup wiring"),
        (
            "src/network/rest/canonical/primary_pod.rs",
            6,
            "operator-only API (authorize_operator) over all tenants' placement, by design",
        ),
        ("src/network/shared_services.rs", 2, "startup wiring"),
        (
            "src/services/collection/manager.rs",
            16,
            "collection writer/reader internals; request paths apply the ownership rule",
        ),
        (
            "src/services/ddl/mod.rs",
            4,
            "namespace DDL scoped via tenant_scoped_namespace, and unit tests",
        ),
        ("src/services/discovery/executor.rs", 1, "unit test"),
        ("src/services/snapshot/coordinator.rs", 1, "unit test"),
        (
            "src/storage/persistence/write_ahead_log/mod.rs",
            2,
            "internal WAL/recovery catalog walk",
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

    /// Uses of the borrowed form's tenant-less default (`Default` is derived).
    /// Excludes the owned form, whose name ends with the same text.
    const ALLOWED_BORROWED_DEFAULT: &[(&str, usize, &str)] = &[(
        "src/services/dml/tests.rs",
        4,
        "unit tests of the identity carrier",
    )];

    /// Struct literals of either identity spelling `tenant_id: None`.
    const ALLOWED_TENANTLESS_LITERAL: &[(&str, usize, &str)] = &[];

    fn check(needle: &str, allowlist: &[(&str, usize, &str)]) {
        check_with(|text| text.matches(needle).count(), allowlist);
    }

    /// The borrowed form's `default()` constructor, excluding the owned form.
    fn borrowed_defaults(text: &str) -> usize {
        let needle = format!("PortIdentity::{}()", "default");
        text.match_indices(&needle)
            .filter(|(at, _)| !text[..*at].ends_with("Owned"))
            .count()
    }

    /// Struct literals `PortIdentity { .. }` / `OwnedPortIdentity { .. }` whose
    /// body sets `tenant_id: None` (whitespace-insensitive).
    fn tenantless_literals(text: &str) -> usize {
        let opener = format!("PortIdentity {}", "{");
        let mut count = 0;
        for (at, _) in text.match_indices(&opener) {
            // Skip the type definitions and impl headers.
            let line_start = text[..at].rfind('\n').map_or(0, |i| i + 1);
            let line = &text[line_start..at];
            if line.contains("struct ") || line.contains("impl") || line.contains("for ") {
                continue;
            }
            let body_start = at + opener.len();
            let mut depth = 1usize;
            let mut end = body_start;
            for (offset, c) in text[body_start..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = body_start + offset;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let body: String = text[body_start..end]
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            if body.contains(&format!("tenant_id:{}", "None")) {
                count += 1;
            }
        }
        count
    }

    fn check_with(count_in: impl Fn(&str) -> usize, allowlist: &[(&str, usize, &str)]) {
        let root = workspace_root();
        let mut files = Vec::new();
        for top in ["src", "crates", "apps"] {
            rust_files(&root.join(top), &mut files);
        }
        assert!(
            !files.is_empty(),
            "guard found no sources under {}: the workspace layout changed",
            root.display()
        );
        let mut found: BTreeMap<String, usize> = BTreeMap::new();
        for file in files {
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            let count = count_in(&text);
            if count > 0 {
                let rel = file
                    .strip_prefix(&root)
                    .map(|p| p.to_string_lossy().replace('\\', "/"))
                    .unwrap_or_else(|_| file.to_string_lossy().into_owned());
                found.insert(rel, count);
            }
        }
        // A guard with listed uses must still find them; one with none listed
        // passes on finding none.
        assert!(
            allowlist.is_empty() || !found.is_empty(),
            "guard matched nothing under {}: the pattern or layout changed",
            root.display()
        );
        let allowed: BTreeMap<String, usize> = allowlist
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

    #[test]
    fn anonymous_port_identity_uses_are_all_listed() {
        // Built so this file does not match itself.
        check(&format!("PortIdentity::{}()", "anonymous"), ALLOWED);
    }

    #[test]
    fn tenant_less_owned_identity_defaults_are_all_listed() {
        check(
            &format!("OwnedPortIdentity::{}()", "default"),
            ALLOWED_OWNED_DEFAULT,
        );
    }

    #[test]
    fn tenant_less_borrowed_identity_defaults_are_all_listed() {
        check_with(borrowed_defaults, ALLOWED_BORROWED_DEFAULT);
    }

    #[test]
    fn tenant_less_identity_literals_are_all_listed() {
        check_with(tenantless_literals, ALLOWED_TENANTLESS_LITERAL);
    }

    #[test]
    fn tenantless_literal_detector_catches_the_spellings() {
        let src = format!(
            "let a = PortIdentity {{ tenant_id: {n}, subject }};\n\
             let b = proximadb_runtime::OwnedPortIdentity {{\n    tenant_id:   {n},\n}};\n\
             let c = PortIdentity {{ tenant_id: Some(t) }};\n\
             pub struct PortIdentity {{ pub tenant_id: Option<u8> }}\n",
            n = "None"
        );
        assert_eq!(tenantless_literals(&src), 2);
        let defaults = format!(
            "PortIdentity::{d}(); OwnedPortIdentity::{d}();",
            d = "default"
        );
        assert_eq!(borrowed_defaults(&defaults), 1);
    }

    #[test]
    fn unscoped_table_resolutions_are_all_listed() {
        check(&format!(".resolve_{}(", "table"), ALLOWED_UNSCOPED_RESOLVE);
    }

    #[test]
    fn direct_default_catalog_access_is_all_listed() {
        check(
            &format!(".default_{}()", "catalog"),
            ALLOWED_DEFAULT_CATALOG,
        );
    }
}
