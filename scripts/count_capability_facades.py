#!/usr/bin/env python3
"""Ratchet the Rust capability-facade sites.

The Rust-side counterpart to `count_silent_failures.py` (Python SDK swallows)
and `count_panic_patterns.sh` (unwrap/expect/panic). This counts a third shape
of the same family -- code that *declares in its own comments* that it is not
the implementation its name and signature promise:

    fn check_bloom_simple(&self, bloom_data: &[u8], key: &[u8]) -> bool {
        // Simplified bloom filter check
        // In practice, would use proper bloom filter implementation
        let hash = self.hash_bytes(key);
        bloom_data[(hash % bloom_data.len() as u64) as usize] != 0
    }

A facade is worse than a missing feature because the caller cannot tell. The
name, the signature and the enum variant all read as working, so the next
integrator wires it and ships the consequence. TD-WAL-6 (the `proto`/`avro` WAL
slots that both emitted bincode, while `SerializationFormat` advertised three
wire formats) and TD-BLOOM-1 (an SST bloom probe reading the file's magic+header
as if it were a filter, while `crates/foundation/proximadb-bloom` carries a
complete implementation) are the same defect. TD-FACADE-1 holds the inventory
and the live/dead tiering.

WHAT THIS COUNTS, PRECISELY: a facade marker appearing in a function's doc
comment or body, in non-test Rust under `src/`, `crates/` and `apps/`. It is a
count of *self-declared non-implementations*, nothing cleverer. Comments are the
signal rather than the body shape on purpose: the motivating defect
(`check_bloom_simple`) has three real statements, uses both parameters and
returns an expression, so every "trivial body" heuristic misses it. Body shape
is reported per site as triage metadata (`ignored_params`, `trivial_body`,
`literal_return`) but never gates the count.

WHAT IT DOES NOT DO: liveness. Whether a facade has production callers is what
separates a bug from debt, and a regex cannot answer it -- `callers` below is an
advisory name-grep for triage only, deliberately not part of the ratchet. The
authoritative tiering lives in TD-FACADE-1. A guard that overclaims its own
precision is the failure this repo keeps relearning.

RECALL IS BOUNDED BY SELF-DECLARATION, and that bound is load-bearing: a facade
that does not admit itself in a comment is invisible here. Two known examples,
both found by review rather than by this gate -- `tokenize_simple`
(`text_search/fulltext_index.rs`, silently `{ self.tokenize_standard(text) }`
behind a doc promising a distinct tokenizer) and `should_read_superblock_bloom`
(`swift/proximablocks_compact_reader.rs`, a `size > 0` test wearing a bloom
name, whose three siblings do honestly say "not yet implemented"). So this gate
raises the cost of *adding* a facade and makes removals visible; it does not
certify that none remain, and no reviewer should read a green run that way.

PRECISION HAS THE MIRROR-IMAGE LIMIT: an *honest refusal* is counted too. A
function that returns `Err("... is not implemented ...")` names its gap to the
caller, so by TD-FACADE-1's own definition it is not a facade -- yet the marker
matches. That is accepted deliberately rather than special-cased: the count
then measures *unimplemented capabilities that announce themselves*, which is
the quantity worth driving down. The consequence for reviewers: converting a
lying facade into an honest refusal is a SEVERITY fix (recorded in the TD's
tiering) and leaves the count unchanged; only implementing or deleting the
capability lowers it. Do not "fix" a site by rewording its message.

Not every site is wrong. A deliberate narrowing is legitimate when it says so
and refuses safely -- `src/cluster/partition_lease.rs` `is_compatible_with`
returns `false` for every pair and documents exactly why ("MVP DML locking uses
the existing single-holder object-store lease ... intentionally not claimed
here"). That is the model: conservative answer, stated reason. What is never
acceptable is a facade that answers *permissively* (claims presence, claims
absence, claims a prune happened), so the count only ever goes DOWN and the
baseline records what remains.

Modes mirror the sibling scripts:
  report         -> metrics only, always exit 0
  no-regression  -> fail if the total differs from the baseline
  baseline       -> rewrite the baseline from the current tree

The check is BIDIRECTIONAL. A total *below* the baseline also fails, because a
baseline that silently drifts down records nothing and stops being a ratchet.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("src", "crates", "apps")
DEFAULT_BASELINE = (
    REPO_ROOT / "docs" / "_internal" / "roadmap" / "CAPABILITY_FACADE_BASELINE.json"
)

#: Phrases that assert "this is not the real implementation". Kept to explicit
#: self-declarations: each one is the author stating the gap, not an inference
#: about code quality. `For now` is included because it is this repo's house
#: idiom for a temporary stand-in ("For now, fall back to individual
#: processing"). A plain `TODO` is NOT, and neither is `Deferred:`, because both
#: routinely mark future *enhancement of working code* rather than a stand-in
#: for absent code -- adding `Deferred:` to this set nearly doubled the count
#: (696 -> 1195) without adding a defect the other markers missed.
#: Trailing `\b` is applied per-alternative, never to the group: a phrase that
#: ends in punctuation (`for now,`) can never satisfy a trailing `\b`, because
#: the boundary would require a word character *after* the comma. That bug cost
#: this guard four of its ten motivating defects on the first run.
MARKER = re.compile(r"""(?ix)
    \b(?:
        placeholders?
      | not \s+ yet \s+ implemented\b
      | not \s+ implemented\b
      | in \s+ a \s+ real \s+ implementation\b
      | in \s+ practice ,? \s+ would\b
      | would \s+ (?: use | implement | need | require | be \s+ implemented )\b
      | simplified
      | stub (?:bed|s)?\b
      | for \s+ now [,.]
      | won't \s+ work \s+ in \s+ production\b
    )
    """)

#: Generated trees are regenerated wholesale; test code is allowed to fake.
EXCLUDED_PARTS = ("_generated", "/proto/", "/tests/", "/target/", "/benches/")
EXCLUDED_SUFFIXES = ("_test.rs", "_tests.rs", "tests.rs")

_FN_RE = re.compile(r"\bfn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^>(]*>)?\s*\(")


def _blank_preserving_lines(text: str, start: int, end: int) -> str:
    """Replace text[start:end] with newlines so line numbers stay correct."""
    return text[:start] + re.sub(r"[^\n]", "", text[start:end]) + text[end:]


def _match_brace(text: str, open_idx: int) -> int:
    """Index just past the `}` matching the `{` at open_idx; len(text) if unbalanced.

    String-, char- and comment-aware: a `{` inside a literal or a comment must
    not move the depth, or every body boundary after it is wrong.
    """
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        two = text[i : i + 2]
        if two == "//":
            j = text.find("\n", i)
            i = n if j == -1 else j + 1
            continue
        if two == "/*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if c == '"':
            # Raw strings (r"..", r#".."#) would need the hash count; they are
            # rare inside fn bodies here and a miscount only shifts a boundary,
            # never silently drops a marker (the file still scans).
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if c == "'":
            # char literal or lifetime; skip one escaped char at most
            if i + 2 < n and text[i + 1] == "\\":
                i += 4
                continue
            if i + 2 < n and text[i + 2] == "'":
                i += 3
                continue
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _strip_cfg_test(text: str) -> str:
    """Blank out `#[cfg(test)] mod ... { ... }` blocks, preserving line count."""
    out = text
    for m in reversed(list(re.finditer(r"#\[cfg\(test\)\]", out))):
        brace = out.find("{", m.end())
        if brace == -1:
            continue
        # Only treat it as a block if a `mod`/`impl`/`fn` intervenes with no `;`
        between = out[m.end() : brace]
        if ";" in between:
            continue
        end = _match_brace(out, brace)
        out = _blank_preserving_lines(out, m.start(), end)
    return out


def _leading_comment_block(text: str, fn_start: int) -> str:
    """The contiguous `///`/`//`/attribute lines immediately above a fn."""
    lines = text[:fn_start].splitlines()
    collected: list[str] = []
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("///") or s.startswith("//") or s.startswith("#["):
            collected.append(s)
            continue
        if s in ("", "pub", "pub(crate)"):
            # a blank line ends the doc block
            if s == "":
                break
            continue
        break
    return "\n".join(reversed(collected))


def _params_of(text: str, paren_idx: int) -> str:
    end = paren_idx
    depth = 0
    while end < len(text):
        if text[end] == "(":
            depth += 1
        elif text[end] == ")":
            depth -= 1
            if depth == 0:
                break
        end += 1
    return text[paren_idx + 1 : end]


_LITERAL_RETURN_RE = re.compile(
    r"(?:^|\n)\s*(?:return\s+)?"
    r"(?:true|false|Ok\(\(\)\)|Ok\(true\)|Ok\(false\)|Ok\(vec!\[\]\)|Ok\(None\)|None|Ok\(0\))"
    r"\s*[;\n}]"
)


def find_sites() -> list[dict]:
    sites: list[dict] = []
    for root in SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.rs")):
            rel = str(path.relative_to(REPO_ROOT))
            posix = "/" + rel
            if any(part in posix for part in EXCLUDED_PARTS):
                continue
            if any(rel.endswith(sfx) for sfx in EXCLUDED_SUFFIXES):
                continue
            raw = path.read_text(errors="ignore")
            text = _strip_cfg_test(raw)
            for m in _FN_RE.finditer(text):
                paren = text.index("(", m.end() - 1)
                params = _params_of(text, paren)
                brace = text.find("{", paren)
                if brace == -1:
                    continue
                # a `;` before the body means a trait signature, not a fn body
                if ";" in text[paren:brace] and text[paren:brace].find(";") < (
                    brace - paren
                ):
                    semi = text.index(";", paren)
                    if semi < brace:
                        continue
                body_end = _match_brace(text, brace)
                body = text[brace:body_end]
                doc = _leading_comment_block(text, m.start())
                hay = doc + "\n" + body
                hit = MARKER.search(hay)
                if not hit:
                    continue
                sites.append(
                    {
                        "file": rel,
                        "line": text[: m.start()].count("\n") + 1,
                        "fn": m.group("name"),
                        "marker": re.sub(r"\s+", " ", hit.group(0)).strip().lower(),
                        "ignored_params": bool(
                            re.search(r"\b_[A-Za-z0-9_]+\s*:", params)
                        ),
                        "trivial_body": body.count(";") <= 3,
                        "literal_return": bool(_LITERAL_RETURN_RE.search(body)),
                    }
                )
    return sites


def summarise(sites: list[dict]) -> dict:
    by_file: dict[str, int] = {}
    for s in sites:
        by_file[s["file"]] = by_file.get(s["file"], 0) + 1
    return {"total": len(sites), "by_file": dict(sorted(by_file.items()))}


def _caller_count(name: str) -> int:
    """Advisory only: workspace name-grep, minus the definition itself.

    Not part of the ratchet. A name-grep cannot resolve trait dispatch,
    re-exports or shadowing, so this is a triage hint and TD-FACADE-1 holds the
    authoritative live/dead call.
    """
    try:
        out = subprocess.run(
            ["git", "grep", "-c", rf"\b{re.escape(name)}\b", "--", "*.rs"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return -1
    total = 0
    for line in out.stdout.splitlines():
        _, _, n = line.rpartition(":")
        if n.isdigit():
            total += int(n)
    return max(total - 1, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode", choices=("report", "no-regression", "baseline"), default="report"
    )
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument(
        "--callers",
        action="store_true",
        help="annotate report with an advisory caller-count per facade (slow)",
    )
    args = ap.parse_args()

    sites = find_sites()
    current = summarise(sites)

    if args.mode == "baseline":
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(current, indent=2) + "\n")
        print(
            f"capability-facade baseline written: {current['total']} site(s) "
            f"-> {args.baseline}"
        )
        return 0

    print(f"Capability-facade sites in {'/, '.join(SCAN_ROOTS)}/: {current['total']}")
    for f, n in current["by_file"].items():
        print(f"  {n:3d}  {f}")

    if args.mode == "report":
        if args.callers:
            print(
                "\nTriage (advisory caller counts; see TD-FACADE-1 for the real tiering):"
            )
            for s in sites:
                flags = ",".join(
                    k
                    for k in ("ignored_params", "trivial_body", "literal_return")
                    if s[k]
                )
                print(
                    f"  {s['file']}:{s['line']} {s['fn']}() "
                    f"[{s['marker']}] callers~{_caller_count(s['fn'])}"
                    + (f" {{{flags}}}" if flags else "")
                )
        return 0

    if not args.baseline.exists():
        print(
            f"::error::no baseline at {args.baseline}; run --mode baseline",
            file=sys.stderr,
        )
        return 1
    baseline = json.loads(args.baseline.read_text())
    expected = baseline["total"]

    if current["total"] > expected:
        added = [
            s
            for s in sites
            if s["file"] not in baseline["by_file"]
            or current["by_file"][s["file"]] > baseline["by_file"].get(s["file"], 0)
        ]
        print(
            f"::error::capability-facade sites rose {expected} -> {current['total']}. "
            "A function whose own comments say it is not the real implementation, "
            "behind a name and signature that read as working, is a trap for the "
            "next integrator. Implement it, or make it refuse explicitly and "
            "conservatively (see src/cluster/partition_lease.rs is_compatible_with), "
            "and record it in TD-FACADE-1.",
            file=sys.stderr,
        )
        for s in added[:10]:
            print(
                f"  candidate: {s['file']}:{s['line']} {s['fn']}() -> {s['marker']}",
                file=sys.stderr,
            )
        return 1

    if current["total"] < expected:
        print(
            f"::error::capability-facade sites FELL {expected} -> {current['total']}. "
            "Good -- now lower the baseline in the same commit: "
            "`python3 scripts/count_capability_facades.py --mode baseline`. "
            "A baseline that drifts silently is not a ratchet.",
            file=sys.stderr,
        )
        return 1

    print(f"capability-facade ratchet holds at {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
