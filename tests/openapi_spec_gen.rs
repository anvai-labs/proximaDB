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

/// Render what drifted, so a CI failure says more than "the file differs".
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
    // Stop once the sides have re-converged for this many lines: a single
    // changed line in a large file otherwise prints MAX_SHOWN lines of
    // identical trailing context.
    const SETTLE: usize = 3;

    let c: Vec<&str> = committed.lines().collect();
    let g: Vec<&str> = generated.lines().collect();

    // `str::lines()` drops a trailing newline and strips a trailing `\r`, so two
    // byte-unequal strings can produce identical line vectors. The caller only
    // calls this on a byte difference, so reporting "first difference at line 1"
    // and then printing identical lines would be worse than saying nothing —
    // it reads as "the files match and the gate is broken". That case is
    // precisely the emitter formatting shift this function exists to name.
    if c == g {
        let describe = |label: &str, s: &str| {
            format!(
                "  {label}: {} bytes, {}, {}\n",
                s.len(),
                if s.ends_with('\n') {
                    "ends with a newline"
                } else {
                    "no trailing newline"
                },
                if s.contains('\r') {
                    "contains CR (CRLF line endings)"
                } else {
                    "no CR"
                },
            )
        };
        let mut out = String::from(
            "the two sides have IDENTICAL lines but differ as bytes — the drift is              invisible to a line diff (trailing newline and/or CR):\n",
        );
        out.push_str(&describe("committed", committed));
        out.push_str(&describe("generated", generated));
        out.push_str(
            "  This is a line-ending or trailing-newline difference, not a handler \
             change. Check the checkout's line endings before regenerating.\n",
        );
        return out;
    }

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
    let mut settled = 0usize;
    for i in first..(first + MAX_SHOWN).min(c.len().max(g.len())) {
        match (c.get(i), g.get(i)) {
            (a, b) if a == b => {
                out.push_str(&format!("  {:>5}   {}\n", i + 1, a.unwrap_or(&"")));
                settled += 1;
                if settled == SETTLE {
                    break;
                }
            }
            (Some(a), Some(b)) => {
                settled = 0;
                out.push_str(&format!("  {:>5} - {}\n  {:>5} + {}\n", i + 1, a, i + 1, b));
            }
            (Some(a), None) => {
                settled = 0;
                out.push_str(&format!("  {:>5} - {}\n", i + 1, a));
            }
            (None, Some(b)) => {
                settled = 0;
                out.push_str(&format!("  {:>5} + {}\n", i + 1, b));
            }
            // Unreachable in practice — the `a == b` arm above matches it, and
            // within 0..max(len) at least one side is always Some — but a match
            // GUARD does not count toward exhaustiveness, so the compiler
            // requires this arm. Removing it is E0004, not dead-code cleanup.
            (None, None) => break,
        }
    }
    out.push_str("  (- committed, + generated from the handlers)\n");
    out
}

// No `#[cfg(test)]` gate. It would be harmless — `rustc --test` does set
// `cfg(test)` for an integration-test crate — but it buys nothing here and
// reads as if the module might be conditionally absent.
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
    fn prints_preceding_context_lines() {
        // Pins CONTEXT > 0. Without this the context loop — the only place in
        // the function that indexes directly (`c[i]`) — is never executed by
        // any test, and dropping it to 0 passes everything else.
        let report = describe_drift("a\nb\nc\nd\nE\n", "a\nb\nc\nd\nX\n");
        assert!(report.contains("first difference at line 5"), "{report}");
        for ctx in ["    2   b", "    3   c", "    4   d"] {
            assert!(
                report.contains(ctx),
                "missing context {ctx:?} in:\n{report}"
            );
        }
        // CONTEXT is 3, so line 1 is outside the window.
        assert!(!report.contains("    1   a"), "{report}");
    }

    #[test]
    fn a_difference_invisible_to_lines_is_named_as_such() {
        // A trailing-newline-only difference: `lines()` erases it, so the
        // line-walk report would print identical lines with no -/+ markers and
        // read as "the files match". Must say what actually differs instead.
        let report = describe_drift("a\nb", "a\nb\n");
        assert!(
            report.contains("IDENTICAL lines but differ as bytes"),
            "{report}"
        );
        // Assert the WHOLE line, so each fact stays attached to the right side.
        // Asserting the phrases separately passes even when the predicate is
        // inverted and they are merely swapped — confirmed by mutation.
        assert!(
            report.contains("committed: 3 bytes, no trailing newline, no CR"),
            "{report}"
        );
        assert!(
            report.contains("generated: 4 bytes, ends with a newline, no CR"),
            "{report}"
        );
        assert!(!report.contains("first difference at line"), "{report}");
    }

    #[test]
    fn a_crlf_only_difference_is_named_as_such() {
        let report = describe_drift("a\r\nb\r\n", "a\nb\n");
        assert!(
            report.contains("IDENTICAL lines but differ as bytes"),
            "{report}"
        );
        // Whole lines again, so an inverted CR predicate cannot pass by swapping.
        assert!(
            report.contains(
                "committed: 6 bytes, ends with a newline, contains CR (CRLF line endings)"
            ),
            "{report}"
        );
        assert!(
            report.contains("generated: 4 bytes, ends with a newline, no CR"),
            "{report}"
        );
    }

    #[test]
    fn stops_once_the_sides_reconverge() {
        // One changed line in a long file must not dump MAX_SHOWN lines of
        // identical trailing context.
        let committed: String = (0..60).map(|i| format!("line{i}\n")).collect();
        let generated = committed.replace("line2\n", "CHANGED\n");
        let report = describe_drift(&committed, &generated);
        assert!(report.contains("- line2"), "{report}");
        assert!(report.contains("+ CHANGED"), "{report}");
        assert!(
            !report.contains("line40"),
            "ran past re-convergence:\n{report}"
        );
    }
}
