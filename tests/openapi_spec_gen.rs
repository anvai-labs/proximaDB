//! Spec-from-code generator + drift gate for the published REST contract
//! (TD-126 Phase 1).
//!
//! `docs/openapi/proximadb-openapi.yaml` is GENERATED from the annotated axum
//! handlers (see `src/network/rest/openapi.rs`), not hand-maintained. This test
//! is both the generator and the drift gate:
//!
//!   * `UPDATE_OPENAPI_SPEC=1 cargo test --test openapi_spec_gen` regenerates
//!     and writes the committed YAML (run this after changing a handler/DTO).
//!   * `cargo test --test openapi_spec_gen` (the default, and what CI runs)
//!     regenerates in memory and FAILS if it differs from the committed copy.
//!
//! Mirrors the `proto-compat` generated-artifact gate: the spec can never
//! silently diverge from the handler code.

use std::path::PathBuf;

use proximadb::network::rest::openapi::openapi_yaml;

fn spec_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/openapi/proximadb-openapi.yaml")
}

#[test]
fn openapi_spec_matches_handlers() {
    let generated = openapi_yaml().expect("generate OpenAPI document from handlers");
    let path = spec_path();

    if std::env::var_os("UPDATE_OPENAPI_SPEC").is_some() {
        std::fs::write(&path, generated.as_bytes())
            .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        eprintln!("Wrote regenerated OpenAPI spec to {}", path.display());
        return;
    }

    let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read committed OpenAPI spec at {}: {e}\n\
             Run `UPDATE_OPENAPI_SPEC=1 cargo test --test openapi_spec_gen` to generate it.",
            path.display()
        )
    });

    assert!(
        committed == generated,
        "docs/openapi/proximadb-openapi.yaml is out of sync with the annotated REST handlers.\n\
         The OpenAPI spec is generated from code (TD-126); do not hand-edit it.\n\
         Run `UPDATE_OPENAPI_SPEC=1 cargo test --test openapi_spec_gen` and commit the result.\n\
         \n{}",
        describe_drift(&committed, &generated)
    );
}

/// Render the first differing lines so a CI failure says *what* drifted.
///
/// Without this the gate is a bare string compare: the log names the file and
/// the fix command but not the change, so a reviewer cannot tell a handler edit
/// from an emitter formatting shift, and cannot judge whether regenerating is
/// safe. That mattered on #1958, where a Cargo.lock-only dependency bump moved
/// the generated contract and the log could not say how.
///
/// Deliberately a plain line walk rather than a diff crate: the gate must not
/// grow a dev-dependency whose own version could perturb the output it exists
/// to explain.
fn describe_drift(committed: &str, generated: &str) -> String {
    const CONTEXT: usize = 3;
    const MAX_SHOWN: usize = 40;

    let c: Vec<&str> = committed.lines().collect();
    let g: Vec<&str> = generated.lines().collect();

    let first = (0..c.len().max(g.len()))
        .find(|&i| c.get(i) != g.get(i))
        .unwrap_or(0);

    let mut out = format!(
        "first difference at line {} (committed has {} lines, generated {}):\n",
        first + 1,
        c.len(),
        g.len()
    );
    for i in first.saturating_sub(CONTEXT)..first {
        out.push_str(&format!("  {:>5}   {}\n", i + 1, c[i]));
    }
    for i in first..(first + MAX_SHOWN).min(c.len().max(g.len())) {
        match (c.get(i), g.get(i)) {
            (a, b) if a == b => out.push_str(&format!("  {:>5}   {}\n", i + 1, a.unwrap_or(&""))),
            (Some(a), Some(b)) => {
                out.push_str(&format!("  {:>5} - {}\n  {:>5} + {}\n", i + 1, a, i + 1, b));
            }
            (Some(a), None) => out.push_str(&format!("  {:>5} - {}\n", i + 1, a)),
            (None, Some(b)) => out.push_str(&format!("  {:>5} + {}\n", i + 1, b)),
            (None, None) => break,
        }
    }
    out.push_str("  (- committed, + generated from the handlers)\n");
    out
}

// No `#[cfg(test)]`: this file IS a test crate, and gating the module on it
// would make these tests silently vanish if that assumption were ever wrong.
mod drift_report_tests {
    use super::describe_drift;

    #[test]
    fn names_the_line_that_changed_and_both_sides() {
        let report = describe_drift("a\nb\nc\n", "a\nB\nc\n");
        assert!(report.contains("first difference at line 2"), "{report}");
        assert!(report.contains("- b"), "{report}");
        assert!(report.contains("+ B"), "{report}");
    }

    #[test]
    fn reports_a_pure_addition_without_panicking_past_the_shorter_side() {
        // The committed side is shorter: indexing it at the differing line
        // would panic, which would replace the drift report with an unrelated
        // failure — the exact opacity this function exists to remove.
        let report = describe_drift("a\n", "a\nb\nc\n");
        assert!(report.contains("first difference at line 2"), "{report}");
        assert!(report.contains("+ b"), "{report}");
        assert!(report.contains("+ c"), "{report}");
    }

    #[test]
    fn reports_a_pure_deletion_from_the_generated_side() {
        let report = describe_drift("a\nb\n", "a\n");
        assert!(report.contains("- b"), "{report}");
    }

    #[test]
    fn identical_input_is_never_reported_as_drift_at_a_real_line() {
        // Not reachable through the assert, but a wrong `unwrap_or` here would
        // silently point at line 1 for every future failure.
        let report = describe_drift("a\nb\n", "a\nb\n");
        assert!(
            report.contains("committed has 2 lines, generated 2"),
            "{report}"
        );
    }
}
