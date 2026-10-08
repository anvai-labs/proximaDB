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
///
/// The output is bounded, and every way it is bounded is STATED in the output.
/// An earlier revision stopped at the first re-converged region with no marker,
/// so a multi-region drift printed one line and read as complete — worse than
/// the bare compare, because it looked authoritative.
fn describe_drift(committed: &str, generated: &str) -> String {
    const CONTEXT: usize = 3;
    const MAX_SHOWN: usize = 40;
    // Stop printing once the sides have re-converged for this many lines, so a
    // one-line change in a large file does not drag MAX_SHOWN lines of identical
    // trailing context. Anything still differing after that is COUNTED and
    // reported, never silently dropped.
    const SETTLE: usize = 3;

    let c: Vec<&str> = committed.lines().collect();
    let g: Vec<&str> = generated.lines().collect();

    // `str::lines()` drops a trailing newline and strips a trailing `\r`, so two
    // byte-unequal strings can produce identical line vectors. Reporting "first
    // difference at line 1" and then printing identical lines would read as "the
    // files match and the gate is broken" — worse than saying nothing. That case
    // is precisely the emitter formatting shift this function exists to name.
    if c == g {
        let describe = |s: &str| {
            format!(
                "{} bytes, {}, {}",
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
        let (dc, dg) = (describe(committed), describe(generated));
        let mut out = String::from(
            "the two sides have IDENTICAL lines but differ as bytes — the drift is \
             invisible to a line diff (trailing newline and/or CR):\n",
        );
        out.push_str(&format!("  committed: {dc}\n"));
        out.push_str(&format!("  generated: {dg}\n"));
        // Those three facts can MATCH while the bytes still differ — e.g. mixed
        // line endings, `"a\r\nb\n"` vs `"a\nb\r\n"`: same length, both end in a
        // newline, both contain CR. Without this fallback the report would name
        // a difference and then describe nothing that differs, which is the very
        // failure the branch exists to remove, relocated one level down.
        if dc == dg {
            let at = committed
                .bytes()
                .zip(generated.bytes())
                .position(|(a, b)| a != b);
            match at {
                Some(off) => out.push_str(&format!(
                    "  those are identical, so the difference is positional: first \
                     differing byte at offset {off} (committed {:?}, generated {:?})\n",
                    committed.as_bytes()[off] as char,
                    generated.as_bytes()[off] as char,
                )),
                None => out.push_str(
                    "  those are identical and no byte differs in the common prefix — \
                     one side is a prefix of the other\n",
                ),
            }
        }
        out.push_str(
            "  This is a line-ending or trailing-newline difference, not a handler \
             change. Check the checkout's line endings before regenerating.\n",
        );
        return out;
    }

    // One pass for both. No `unwrap_or` fallback: past the `c == g` return the
    // sides differ by line, so `differing` is non-empty by construction — an
    // `unwrap_or(0)` here would be an unreachable branch that silently reports
    // line 1 if the invariant ever broke, and nothing could test it.
    let differing: Vec<usize> = (0..c.len().max(g.len()))
        .filter(|&i| c.get(i) != g.get(i))
        .collect();
    let (Some(&first), total_differing) = (differing.first(), differing.len()) else {
        return String::from(
            "  the sides differ as bytes and as line vectors, but no line index \
             differs — describe_drift's invariant is broken; report this\n",
        );
    };

    // Trailing whitespace is invisible in a CI log, so render each line with
    // `{:?}`: a line that gained a trailing space shows as "foo " vs "foo"
    // rather than as two identical-looking rows.
    let mut out = format!(
        "first difference at line {} of {} differing line(s) \
         (committed has {} lines, generated {}):\n",
        first + 1,
        total_differing,
        c.len(),
        g.len()
    );
    for i in first.saturating_sub(CONTEXT)..first {
        out.push_str(&format!("  {:>5}   {:?}\n", i + 1, c[i]));
    }
    let mut settled = 0usize;
    let mut last_printed = first;
    for i in first..(first + MAX_SHOWN).min(c.len().max(g.len())) {
        last_printed = i;
        match (c.get(i), g.get(i)) {
            (a, b) if a == b => {
                out.push_str(&format!("  {:>5}   {:?}\n", i + 1, a.unwrap_or(&"")));
                settled += 1;
                if settled >= SETTLE {
                    break;
                }
            }
            (Some(a), Some(b)) => {
                settled = 0;
                out.push_str(&format!("  {:>5} - {a:?}\n  {:>5} + {b:?}\n", i + 1, i + 1));
            }
            (Some(a), None) => {
                settled = 0;
                out.push_str(&format!("  {:>5} - {a:?}\n", i + 1));
            }
            (None, Some(b)) => {
                settled = 0;
                out.push_str(&format!("  {:>5} + {b:?}\n", i + 1));
            }
            // Unreachable in practice — the `a == b` arm above matches it, and
            // within 0..max(len) at least one side is always Some — but a match
            // GUARD does not count toward exhaustiveness, so the compiler
            // requires this arm. Removing it is E0004, not dead-code cleanup.
            (None, None) => break,
        }
    }
    // Say what was left out. Silence here is what made an earlier revision read
    // as "the drift is one line".
    let remaining = (last_printed + 1..c.len().max(g.len()))
        .filter(|&i| c.get(i) != g.get(i))
        .count();
    if remaining > 0 {
        out.push_str(&format!(
            "  ... {remaining} further differing line(s) not shown (output capped \
             at {MAX_SHOWN} lines, stopping after {SETTLE} re-converged lines)\n"
        ));
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
        assert!(report.contains(r#"- "b""#), "{report}");
        assert!(report.contains(r#"+ "B""#), "{report}");
    }

    #[test]
    fn reports_a_pure_addition_without_panicking_past_the_shorter_side() {
        // The committed side is shorter: indexing it at the differing line
        // would panic, which would replace the drift report with an unrelated
        // failure — the exact opacity this function exists to remove.
        let report = describe_drift("a\n", "a\nb\nc\n");
        assert!(report.contains("first difference at line 2"), "{report}");
        assert!(report.contains(r#"+ "b""#), "{report}");
        assert!(report.contains(r#"+ "c""#), "{report}");
    }

    #[test]
    fn reports_a_pure_deletion_from_the_generated_side() {
        let report = describe_drift("a\nb\n", "a\n");
        assert!(report.contains(r#"- "b""#), "{report}");
    }

    #[test]
    fn names_which_side_is_longer() {
        // Pins the header's side association: swapping c.len()/g.len() would
        // report the wrong side as longer, and no other test looks at it.
        let report = describe_drift("a\n", "a\nb\nc\n");
        assert!(
            report.contains("committed has 1 lines, generated 3"),
            "{report}"
        );
    }

    #[test]
    fn prints_preceding_context_lines() {
        // Pins CONTEXT > 0. Without this the context loop — the only place in
        // the function that indexes directly (`c[i]`) — is never executed by
        // any test, and dropping it to 0 passes everything else.
        let report = describe_drift("a\nb\nc\nd\nE\n", "a\nb\nc\nd\nX\n");
        assert!(report.contains("first difference at line 5"), "{report}");
        for ctx in [r#"    2   "b""#, r#"    3   "c""#, r#"    4   "d""#] {
            assert!(
                report.contains(ctx),
                "missing context {ctx:?} in:\n{report}"
            );
        }
        // CONTEXT is 3, so line 1 is outside the window.
        assert!(!report.contains(r#"    1   "a""#), "{report}");
    }

    #[test]
    fn prints_trailing_context_so_the_change_is_readable_in_place() {
        // Pins the equal-arm print: deleting it entirely passes every other
        // test, leaving a report with no surrounding content.
        let report = describe_drift("a\nB\nc\nd\n", "a\nX\nc\nd\n");
        assert!(report.contains(r#"    3   "c""#), "{report}");
        assert!(report.contains(r#"    4   "d""#), "{report}");
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
    fn identical_descriptions_fall_back_to_a_byte_offset() {
        // Mixed line endings: same byte length, both end in a newline, both
        // contain CR — so all three described facts match and the branch would
        // otherwise name a difference while describing nothing that differs.
        let report = describe_drift("a\r\nb\n", "a\nb\r\n");
        assert!(
            report.contains("IDENTICAL lines but differ as bytes"),
            "{report}"
        );
        assert!(
            report.contains("the difference is positional: first differing byte at offset 1"),
            "{report}"
        );
    }

    #[test]
    fn stops_once_the_sides_reconverge_but_says_what_it_left_out() {
        // One changed line in a long file must not dump MAX_SHOWN lines of
        // identical trailing context...
        let committed: String = (0..60).map(|i| format!("line{i}\n")).collect();
        let generated = committed.replace("line2\n", "CHANGED\n");
        let report = describe_drift(&committed, &generated);
        assert!(report.contains(r#"- "line2""#), "{report}");
        assert!(report.contains(r#"+ "CHANGED""#), "{report}");
        assert!(
            !report.contains(r#""line40""#),
            "ran past re-convergence:\n{report}"
        );
        // ...and with only one differing line there is nothing left to report.
        assert!(report.contains("of 1 differing line(s)"), "{report}");
        assert!(!report.contains("further differing line(s)"), "{report}");
    }

    #[test]
    fn a_second_changed_region_is_counted_not_silently_dropped() {
        // THE REGRESSION THIS PINS: an earlier revision broke out of the loop at
        // the first re-converged region and printed the unconditional legend, so
        // a three-region drift reported one line and read as complete.
        let committed: String = (0..60).map(|i| format!("line{i}\n")).collect();
        let generated = committed
            .replace("line2\n", "X\n")
            .replace("line31\n", "Y\n")
            .replace("line51\n", "Z\n");
        let report = describe_drift(&committed, &generated);
        assert!(report.contains("of 3 differing line(s)"), "{report}");
        assert!(
            report.contains("2 further differing line(s) not shown"),
            "{report}"
        );
    }

    #[test]
    fn trailing_whitespace_is_visible() {
        // Rendered with {:?}, so a gained trailing space is not two
        // identical-looking rows in a CI log.
        let report = describe_drift("foo: bar \n", "foo: bar\n");
        assert!(report.contains(r#"- "foo: bar ""#), "{report}");
        assert!(report.contains(r#"+ "foo: bar""#), "{report}");
    }

    #[test]
    fn a_contiguous_region_is_shown_whole() {
        // Pins MAX_SHOWN from BELOW. The cap test below only pins that output is
        // bounded, so shrinking MAX_SHOWN to 3 passed it — the report would show
        // three lines of a ten-line change and look complete.
        let committed: String = (0..40).map(|i| format!("line{i}\n")).collect();
        let mut generated = committed.clone();
        for i in 5..25 {
            generated = generated.replace(&format!("line{i}\n"), &format!("CH{i}\n"));
        }
        let report = describe_drift(&committed, &generated);
        for i in 5..25 {
            assert!(
                report.contains(&format!(r#"+ "CH{i}""#)),
                "line {i} of a contiguous region missing:\n{report}"
            );
        }
    }

    #[test]
    fn trailing_context_stops_at_settle() {
        // Pins SETTLE from ABOVE. Raising it to 20 passed every other test: the
        // report would trail twenty identical lines after a one-line change,
        // which is the noise SETTLE exists to cut.
        let committed: String = (0..60).map(|i| format!("line{i}\n")).collect();
        let generated = committed.replace("line2\n", "CHANGED\n");
        let report = describe_drift(&committed, &generated);
        // Change at index 2 → trailing context is indices 3,4,5 (SETTLE = 3).
        assert!(report.contains(r#"    6   "line5""#), "{report}");
        assert!(
            !report.contains(r#""line6""#),
            "trailed past SETTLE:\n{report}"
        );
    }

    #[test]
    fn a_region_after_a_short_gap_is_still_shown() {
        // Pins the `settled = 0` resets. Without them `settled` accumulates
        // across changed lines, so regions separated by fewer than SETTLE equal
        // lines make the loop break early and drop the later ones.
        let committed: String = (0..30).map(|i| format!("line{i}\n")).collect();
        let generated = committed
            .replace("line1\n", "P\n")
            .replace("line4\n", "Q\n")
            .replace("line7\n", "R\n");
        let report = describe_drift(&committed, &generated);
        for v in ["P", "Q", "R"] {
            assert!(
                report.contains(&format!(r#"+ "{v}""#)),
                "region {v} dropped:\n{report}"
            );
        }
    }

    #[test]
    fn the_output_is_capped_for_a_wholly_different_file() {
        // Pins MAX_SHOWN: every line differs, so nothing re-converges and only
        // the cap bounds the output. Raising it to 1000 would dump the file.
        let committed: String = (0..500).map(|i| format!("a{i}\n")).collect();
        let generated: String = (0..500).map(|i| format!("b{i}\n")).collect();
        let report = describe_drift(&committed, &generated);
        assert!(report.lines().count() < 100, "uncapped:\n{report}");
        assert!(
            report.contains("further differing line(s) not shown"),
            "{report}"
        );
    }
}
