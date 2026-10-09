#!/usr/bin/env bash
# =============================================================================
# ProximaDB Version Sync Script
# =============================================================================
# Ensures all version strings across the codebase are consistent.
#
# Usage:
#   bash scripts/version-sync.sh check          # Validate all files match Cargo.toml
#   bash scripts/version-sync.sh set <version>  # Update all files to <version>
#   bash scripts/version-sync.sh get            # Print current version from Cargo.toml
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Reads [package].version STRUCTURALLY. The old form was
# `grep '^version = ' Cargo.toml | head -1`, which slides down to
# [workspace.package] when [package] has no version line -- so deleting
# Cargo.toml's own version line left `get` reporting the workspace version and
# `check` comparing it against itself, while cargo reported proximadb = 0.0.0.
# Consequence, stated correctly (an earlier revision had the causality
# backwards): release.yml does NOT pick the tag from `version-sync.sh get` -- it
# derives VERSION from the pushed tag ref. `get` is consumed by prerelease-ci.yml,
# which is the gate that would have reported 0.4.0 while cargo built 0.0.0.
get_cargo_version() {
  python3 -c 'import sys,tomllib
with open(sys.argv[1],"rb") as fh: d=tomllib.load(fh)
v=(d.get("package") or {}).get("version")
print(v if isinstance(v,str) else "")' "$REPO_ROOT/Cargo.toml" 2>/dev/null
}

extract_version_from_file() {
  local file="$1"
  local type="$2"

  if [ ! -f "$file" ]; then
    echo ""
    return
  fi

  # Every extractor below returns a `<MISSING-...>` sentinel when the file is
  # present but the field is absent, because an EMPTY return reads as `[SKIP]`
  # and `[SKIP]` passes. That fail-open was found in NINE places — deleting the
  # version from the root pyproject, the SDK pyproject, clients/rust/Cargo.toml,
  # ui/package.json, either Chart.yaml key, either __init__.py, or the root
  # [package] version all made `check` PASS. A deleted field is exactly as much
  # a release defect as a wrong one. (A missing FILE still skips; that is
  # deliberate, since not every checkout has every client.)
  local out
  case "$type" in
    # Structural, for the reason in get_cargo_version: a line-oriented
    # `head -1` cannot tell [package].version from [workspace.package].version,
    # and the root manifest has both. `version.workspace = true` (a member
    # inheriting) is reported as a distinct sentinel rather than missing, since
    # it is a legitimate shape that this gate simply does not own.
    cargo)
      out=$(python3 -c 'import sys,tomllib
with open(sys.argv[1],"rb") as fh: d=tomllib.load(fh)
v=(d.get("package") or {}).get("version")
if isinstance(v,dict) and v.get("workspace") is True: print("<SKIP:inherits-workspace>")
elif isinstance(v,str): print(v)
else: print("")' "$file" 2>/dev/null || echo "<UNPARSEABLE-$file>")
      ;;
    pyproject)
      out=$(grep '^version = ' "$file" | sed 's/.*"\(.*\)".*/\1/')
      ;;
    python_init)
      out=$(grep '__version__' "$file" | head -1 | sed 's/.*"\(.*\)".*/\1/')
      ;;
    # `tr -d '\r'` is load-bearing: `.*` captures to end of line, so on a CRLF
    # file the value carried a trailing CR and a CORRECT version read as a
    # mismatch against the expected one. (`set` normalises line endings, so it
    # self-healed — but a false [FAIL] on an untouched tree is still a gate that
    # cries wolf.)
    yaml_version)
      out=$(grep '^version:' "$file" | sed 's/.*: *\(.*\)/\1/' | tr -d '\r')
      ;;
    yaml_appversion)
      out=$(grep '^appVersion:' "$file" | sed 's/.*"\(.*\)".*/\1/' | head -1 | tr -d '\r')
      ;;
    package_json)
      # The TOP-LEVEL "version", read structurally. Must not be "the first
      # `\"version\"` line": in a lockfile that is the top-level field, and if it
      # is deleted the scan falls through to packages[""] and prints a confident
      # [OK]. An earlier form anchored it to `^  "version"` — exactly two spaces
      # — which made the gate depend on indentation: re-formatting the file to
      # 4-space indent reported <MISSING> on a perfectly valid manifest, and
      # `set` could not repair it because `set` rewrites it fine. Parsing the
      # document has neither failure mode.
      out=$(python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
v=d.get("version")
print(v if isinstance(v,str) else "")' "$file" 2>/dev/null || echo "<UNPARSEABLE-$file>")
      ;;
    # `cargo` takes the FIRST `^version = ` line, which is [package]. The
    # [workspace.package] version is further down the file and was therefore
    # invisible to this gate — the reason a "0.4.0" release shipped 150 crates
    # (and the server binary) still reporting 0.2.0.
    # Parsed with tomllib, not a regex. A hand-rolled awk walk failed OPEN on
    # four valid TOML spellings of the field it exists to watch — `[ workspace.
    # package ]`, an indented `version`, a single-quoted value, and a DELETED
    # version line (empty extraction reads as [SKIP], which does not fail) — and
    # it leaked a later table's version when `[workspace.package]` had none,
    # because the section flag was never reset at the next header.
    cargo_workspace)
      out=$(python3 -c 'import sys,tomllib
with open(sys.argv[1],"rb") as fh: d=tomllib.load(fh)
v=d.get("workspace",{}).get("package",{}).get("version")
print(v if v else "<MISSING-workspace.package.version>")' "$file" 2>/dev/null \
        || echo "<UNPARSEABLE-$file>")
      ;;
    # M2: intra-workspace path dependencies carry an explicit version
    # requirement that MUST track [workspace.package] — cargo fails with
    # "failed to select a version for the requirement" otherwise, a hard build
    # break the release gate was blind to. Reports the first mismatching pin so
    # the comparison fails with a useful value.
    cargo_path_dep_pins)
      out=$(python3 -c 'import sys,tomllib
with open(sys.argv[1],"rb") as fh: d=tomllib.load(fh)
pins={}
def walk(tbl, prefix=""):
    for name,spec in (tbl or {}).items():
        if isinstance(spec,dict) and "path" in spec and "version" in spec:
            pins[prefix+name]=spec["version"]
for table in ("dependencies","dev-dependencies","build-dependencies"):
    walk(d.get(table))
# `set` rewrites every `version = "X", path =` line file-wide, and the root
# manifest keeps its ~107 intra-workspace path deps HERE, under
# [workspace.dependencies] -- the table this walk originally skipped, which is
# where such pins most naturally live.
walk((d.get("workspace") or {}).get("dependencies"), prefix="workspace.")
# `set` rewrites every `version = "X", path =` line in the file, so a pin under
# [target.(cfg).dependencies] is written but was not read here. The root manifest
# already has a target table for macos/aarch64, so this was one line from live.
for cfg,tbl in (d.get("target") or {}).items():
    for table in ("dependencies","dev-dependencies","build-dependencies"):
        walk(tbl.get(table), prefix=f"target.{cfg}.")
vals=set(pins.values())
if not pins: print("<NO-PATH-DEP-PINS>")
elif len(vals)==1: print(next(iter(vals)))
else: print("<MIXED:"+",".join(f"{k}={v}" for k,v in sorted(pins.items()))+">")' "$file" 2>/dev/null \
        || echo "<UNPARSEABLE-$file>")
      ;;
    # M2: the `embedded` extra pins the wheel this release builds. Its own upper
    # bound once excluded that wheel, making `pip install proximadb[embedded]`
    # unsatisfiable. Reports the LOWER bound so a stale pin fails.
    python_extra_lower_bound)
      out=$(python3 -c 'import re,sys
def key(v): return tuple(int(x) for x in re.findall(r"\d+", v))
s=open(sys.argv[1]).read()
m=re.search(r"proximadb_embedded>=([0-9][^,\"]*),<([0-9][^,\"]*)", s)
# The int-tuple key cannot order a PEP 440 prerelease against its release
# (0.4.0-beta.1 vs 0.4.0 both key to (0,4,0,1) vs (0,4,0)), so emptiness is only
# claimed when BOTH bounds are plain releases. That keeps the one direction that
# matters -- never a FALSE failure on a non-empty range -- at the cost of missing
# an empty range that straddles a prerelease, which `set` cannot produce anyway.
plain=lambda v: re.fullmatch(r"\d+(\.\d+)*", v) is not None
if not m: print("<NO-EMBEDDED-EXTRA-BOUND>")
elif plain(m.group(1)) and plain(m.group(2)) and key(m.group(1)) >= key(m.group(2)):
    # The one defect this step exists to prevent: an upper bound at or below the
    # lower bound is an EMPTY set, so `pip install proximadb[embedded]` is
    # unsatisfiable. Reporting only the lower bound could never see it.
    print(f"<EMPTY-EXTRA-RANGE:>={m.group(1)},<{m.group(2)}>")
else: print(m.group(1))' "$file" 2>/dev/null \
        || echo "<UNPARSEABLE-$file>")
      ;;
    # A package-lock.json records the package's own version TWICE: top-level and
    # packages[""]. `package_json` above only sees the first, so the second could
    # drift unnoticed — which is half of what `set` writes.
    # The `|| echo` is load-bearing, not decoration: without it a non-zero python
    # exit propagates out of the command substitution and `set -e` terminates
    # `check` with an EMPTY stderr, mid-list, before the remaining entries run —
    # so the gate dies with no diagnostics on exactly the field it was added to
    # watch. Verified by deleting `packages[""]["version"]`.
    # F1: the lockfile was outside the gate entirely, so `set` produced a tree
    # that `check` PASSED and `cargo metadata --locked` (ci.yml, merge-blocking)
    # REJECTED -- the gate's own remediation message handed you a red PR. Reports
    # every in-workspace `proximadb*` package version, so a stale lock fails.
    cargo_lock_members)
      out=$(python3 -c 'import sys,tomllib
with open(sys.argv[1],"rb") as fh: d=tomllib.load(fh)
vals={}
for pkg in d.get("package") or []:
    name=pkg.get("name","")
    # In-workspace members have no `source`; a registry crate that merely starts
    # with "proximadb" would otherwise be swept in and compared.
    if name.startswith("proximadb") and "source" not in pkg:
        vals.setdefault(pkg.get("version",""), []).append(name)
if not vals: print("<NO-WORKSPACE-PACKAGES-IN-LOCK>")
elif len(vals)==1: print(next(iter(vals)))
else: print("<MIXED-LOCK:"+",".join(f"{v}={len(n)} crate(s)" for v,n in sorted(vals.items()))+">")' "$file" 2>/dev/null \
        || echo "<UNPARSEABLE-$file>")
      ;;
    # F9: a maven client manifest. Was stale at 0.2.0 and invisible, while the
    # script's header claimed "all version strings across the codebase".
    #
    # XML-STRUCTURAL, and it has to be. A positional read ("the first <version>
    # after </parent>") is WRONG, not merely fragile: <project>'s children are an
    # xs:all, so a project that declares <version> after <dependencies> makes the
    # first post-parent <version> a DEPENDENCY's. Reproduced on a parent+deps pom:
    # the positional form reported the junit-jupiter version, and `set` rewrote
    # that dependency while `check` then printed [OK] over the corruption. That is
    # confident-wrong, which is worse than the fail-opens this gate was written
    # for. Reads /project/version only.
    maven_pom)
      out=$(python3 -c 'import sys,xml.etree.ElementTree as ET
try: root=ET.parse(sys.argv[1]).getroot()
except Exception as exc: print(f"<UNPARSEABLE-pom:{exc}>"); raise SystemExit(0)
tag=root.tag
ns=tag[:tag.index("}")+1] if tag.startswith("{") else ""
if tag != ns+"project": print("<NOT-A-POM>"); raise SystemExit(0)
# A direct child only: find() on the root never descends into <dependencies>.
el=root.find(ns+"version")
if el is None or not (el.text or "").strip():
    # Legitimate: the version is inherited from <parent>. Not this gate'"'"'s to own.
    print("<SKIP:pom-inherits-parent>")
else:
    print(el.text.strip())' "$file" 2>/dev/null || echo "<UNPARSEABLE-$file>")
      ;;
    package_lock_own)
      out=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["packages"][""]["version"])' "$file" 2>/dev/null \
        || echo "<UNPARSEABLE-$file>")
      ;;
    *)
      echo ""
      return
      ;;
  esac
  if [ -z "$out" ]; then
    echo "<MISSING-$type-in-$file>"
  else
    echo "$out"
  fi
}

cmd_get() {
  get_cargo_version
}

cmd_check() {
  local expected
  expected=$(get_cargo_version)
  echo "Checking all version files match: $expected"
  echo ""

  local failed=0
  local actual
  local file
  local reason

  # Define files to check: file_path:type
  declare -a files=(
    "Cargo.toml:cargo"
    "Cargo.toml:cargo_workspace"
    "pyproject.toml:pyproject"
    "clients/python/pyproject.toml:pyproject"
    # NOT clients/python-embedded/pyproject.toml: it carries no version by
    # design. #1675 made it non-buildable (the canonical proximadb_embedded
    # wheel builds from the REPO-ROOT pyproject.toml via maturin manifest-path),
    # so it holds only dev-tool config. `check` reported it as [SKIP], which was
    # correct but read like a gap; `set` ran a no-op substitution against it and
    # printed "[SET] ... -> <version>", which was a false claim. The version that
    # ships for that package is the root pyproject.toml's, already checked above.
    "clients/python/src/proximadb_sdk/__init__.py:python_init"
    "clients/python-embedded/src/proximadb_embedded/__init__.py:python_init"
    "clients/rust/Cargo.toml:cargo"
    "deploy/helm/proximadb/Chart.yaml:yaml_version"
    "deploy/helm/proximadb/Chart.yaml:yaml_appversion"
    "ui/package.json:package_json"
    "clients/nodejs-embedded/package.json:package_json"
    # The npm lockfiles record the package's OWN version (top-level and
    # packages[""]), and `npm ci` hard-fails when it disagrees with
    # package.json: "npm error `npm ci` can only install packages when your
    # package.json and package-lock.json are in sync". `set` used to bump only
    # package.json, so a release left the Admin UI build broken. The
    # `package_json` extractor reads the FIRST "version" key, which in a lockfile
    # is the package's own, so it works unchanged here.
    "ui/package-lock.json:package_json"
    "ui/package-lock.json:package_lock_own"
    "clients/nodejs-embedded/package-lock.json:package_json"
    "clients/nodejs-embedded/package-lock.json:package_lock_own"
    "Cargo.toml:cargo_path_dep_pins"
    "clients/python/pyproject.toml:python_extra_lower_bound"
    # `cargo metadata --locked` runs on every PR (ci.yml) and rejects a lock that
    # disagrees with the manifests, so the lock is part of the release contract.
    "Cargo.lock:cargo_lock_members"
    # F9: these two were stale at 0.2.0 and unwatched. python-queue-embedded is a
    # maturin-built [project] WITH a version (unlike python-embedded, which has
    # none by design and is excluded above with its reason).
    "clients/python-queue-embedded/pyproject.toml:pyproject"
    "clients/java-embedded/pom.xml:maven_pom"
  )

  for entry in "${files[@]}"; do
    file="${entry%%:*}"
    type="${entry##*:}"

    actual=$(extract_version_from_file "$REPO_ROOT/$file" "$type")

    # A `<...>` sentinel means the extractor ran and found the field missing or
    # malformed. That is a FAILURE, not a skip: an empty extraction reading as
    # [SKIP] is how a deleted [workspace.package] version passed this gate.
    # A `<SKIP:reason>` sentinel means the extractor ran and found a shape this
    # gate deliberately does not own -- a crate inheriting `version.workspace =
    # true`, or a pom inheriting from <parent>. Treating those as FAIL made the
    # gate reject the idiomatic spelling used by 151 of 152 workspace members,
    # with `set` unable to repair it (it exits 0 and changes nothing), so `check`
    # stayed red forever. The value is already covered by the corresponding
    # workspace/parent entry. Any OTHER `<...>` is still a hard failure.
    if [ "${actual#<SKIP:}" != "$actual" ]; then
      reason="${actual#<SKIP:}"
      echo "  [SKIP] $file ($type): ${reason%>}"
    elif [ "${actual#<}" != "$actual" ]; then
      echo "  [FAIL] $file ($type): $actual"
      ((failed++)) || true
    elif [ -z "$actual" ]; then
      echo "  [SKIP] $file (file not found or no version extracted)"
    elif [ "$actual" != "$expected" ]; then
      echo "  [FAIL] $file: found $actual, expected $expected"
      ((failed++)) || true
    else
      echo "  [OK]   $file ($actual)"
    fi
  done

  echo ""
  if [ "$failed" -gt 0 ]; then
    echo "FAILED: $failed version mismatch(es) found."
    echo "Run 'bash scripts/version-sync.sh set $expected' to fix."
    exit 1
  else
    echo "PASSED: All versions match $expected"
  fi
}

cmd_set() {
  local version="$1"

  # Validate semver format
  if ! echo "$version" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$'; then
    echo "Error: Invalid version format '$version'. Expected semver (e.g., 0.2.0 or 0.3.0-beta.1)"
    exit 1
  fi

  echo "Setting all version files to: $version"
  echo ""

  # Use perl for cross-platform compatibility (supports -i)
  # perl and python3 are REQUIRED, checked before any write.
  #
  # There used to be a `sed` fallback here. It was a second, divergent
  # implementation of the same substitutions: it still patched
  # clients/python-embedded/pyproject.toml (which has no version), never touched
  # the npm lockfiles, and finished by printing "[SET] All files updated" — a
  # blanket false success, the very category of defect this script was being
  # fixed for. #1675 recorded the same lesson about two configs with one
  # correct. Requiring the tools is strictly safer than maintaining a path
  # nobody exercises; perl ships with macOS and every CI image we use.
  local missing=""
  command -v perl >/dev/null 2>&1 || missing="$missing perl"
  command -v python3 >/dev/null 2>&1 || missing="$missing python3"
  if [ -n "$missing" ]; then
    echo "Error: version-sync.sh set requires:$missing" >&2
    echo "       Nothing was modified." >&2
    exit 1
  fi

  # Pre-flight every ASSERTING step's precondition before the first write.
  # Tool presence alone was not enough: steps 2, 3 and 5 abort on a shape they
  # cannot edit, and they run AFTER step 1's perl has already written. Removing
  # [workspace.package] from Cargo.toml left root [package] bumped and the rest
  # of the tree behind, with a message ("no [workspace.package] table — not
  # modified") that was true of the one edit and false of the tree.
  python3 - "$REPO_ROOT" <<'PYEOF' || exit 1
import os
import re
import sys

root = sys.argv[1]
problems = []

cargo = os.path.join(root, "Cargo.toml")
if os.path.isfile(cargo):
    text = open(cargo).read()
    m = re.search(r"^\[workspace\.package\]\s*$", text, re.M)
    if not m:
        problems.append("Cargo.toml: no [workspace.package] table")
    else:
        start = m.end()
        nxt = re.search(r"^\[", text[start:], re.M)
        end = start + (nxt.start() if nxt else len(text) - start)
        if not re.search(r'(?m)^\s*version\s*=\s*["\']', text[start:end]):
            problems.append("Cargo.toml: [workspace.package] has no version key")
    # A pin whose requirement is not a bare x.y.z (e.g. "0.4" or "^0.4.0") is
    # READ by check and SKIPPED by set's regex, so set would report success and
    # leave check failing forever. Refuse up front instead.
    # `set`'s writers match only SINGLE-LINE inline tables. A dep spelled as a
    # sub-table ([workspace.dependencies.foo] with version/path on their own
    # lines) is READ by check's tomllib walk and never written -- `set` exits 0,
    # changes nothing, and check stays red forever. Same shape as the defect the
    # pre-flight exists for, through the one spelling it did not scan.
    try:
        import tomllib

        with open(cargo, "rb") as fh:
            doc = tomllib.load(fh)
    except Exception:
        doc = {}

    def subtable_pins(tbl, prefix=""):
        for name, spec in (tbl or {}).items():
            if not isinstance(spec, dict) or "path" not in spec or "version" not in spec:
                continue
            # Is it written inline on one line? If so `set` handles it.
            inline = re.search(
                r"^\s*" + re.escape(name) + r"\s*=\s*\{[^{}\n]*\}\s*$", text, re.M
            )
            if not inline:
                yield prefix + name, spec["version"]

    for table in ("dependencies", "dev-dependencies", "build-dependencies"):
        for name, ver in subtable_pins(doc.get(table)):
            problems.append(
                f"Cargo.toml: path dep {name!r} is spelled as a sub-table, which "
                "`set` cannot rewrite (use a single-line inline table)"
            )
    for name, ver in subtable_pins((doc.get("workspace") or {}).get("dependencies"),
                                   prefix="workspace."):
        problems.append(
            f"Cargo.toml: path dep {name!r} is spelled as a sub-table, which "
            "`set` cannot rewrite (use a single-line inline table)"
        )

    # HIGH 5: step 1's writer is "the first `^version = ` line", which slides
    # down to [workspace.package] when [package] has no version -- so `set`
    # printed "[SET] Cargo.toml [package]" while retargeting the workspace
    # table. The READERS were made structural in round 5; this refuses the shape
    # the writer still cannot address.
    if not (doc.get("package") or {}).get("version"):
        problems.append(
            "Cargo.toml: [package] has no literal version, so `set` would "
            "mis-target [workspace.package] while reporting [package]"
        )

    for pin in re.finditer(r"\{[^{}\n]*path\s*=[^{}\n]*\}", text):
        body = pin.group(0)
        v = re.search(r'version\s*=\s*["\']([^"\']*)["\']', body)
        if v and not re.fullmatch(r"\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?", v.group(1)):
            problems.append(
                f"Cargo.toml: path-dep pin version {v.group(1)!r} is not a bare "
                "x.y.z, so `set` cannot rewrite it (spell it x.y.z)"
            )

sdk = os.path.join(root, "clients", "python", "pyproject.toml")
if os.path.isfile(sdk):
    # The EXACT predicate step 5 asserts: the full ">=...,<..." shape, occurring
    # exactly once. The weaker `proximadb_embedded>=[0-9]` probe passed on a
    # bound with no upper bound, and step 5 -- which runs after step 1's perl and
    # step 4's -- then aborted with three files already bumped.
    sdk_text = open(sdk).read()
    bounds = re.findall(r'proximadb_embedded>=[0-9][^,"\']*,<[0-9][^"\']*', sdk_text)
    if len(bounds) != 1:
        problems.append(
            f"clients/python/pyproject.toml: expected exactly 1 complete "
            f"proximadb_embedded'>=X,<Y' bound for `set` to rewrite, found "
            f"{len(bounds)}"
        )
if problems:
    for pr in problems:
        print(f"Error: {pr}", file=sys.stderr)
    print("       Nothing was modified.", file=sys.stderr)
    sys.exit(1)
PYEOF

  # ONE implementation of the npm-lockfile edit, run twice: mode=check as a dry
  # run during pre-flight, mode=write for real. Previously the pre-flight was a
  # SECOND, slightly different line scan, and a lockfile whose first `packages`
  # entry is a workspace rather than "" (valid npm-workspaces shape, hand-ordered)
  # satisfied the pre-flight and then mis-targeted in the writer -- leaving eleven
  # files bumped and the lock untouched, which is the half-bumped tree the
  # pre-flight exists to prevent. A dry run of the real thing cannot disagree
  # with the real thing.
  local LOCK_EDIT_PY
  LOCK_EDIT_PY=$(cat <<'PYEOF'
import json
import os
import re
import sys

path, version, mode = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path) as fh:
        doc = json.load(fh)
except (OSError, json.JSONDecodeError) as exc:
    sys.exit(f"{path}: not readable as JSON ({exc})")
if "version" not in doc:
    sys.exit(f'{path}: no top-level "version" key')
if "version" not in doc.get("packages", {}).get("", {}):
    sys.exit(f'{path}: no packages[""]["version"] key (lockfileVersion 1?)')

old_top = doc["version"]
old_own = doc["packages"][""]["version"]

# Edit by line so indentation and key order survive byte-for-byte, but take the
# TARGETS from the parsed document rather than from position. newline="" on BOTH
# read and write: text mode would translate CRLF to LF throughout, rewriting
# every line of a CRLF lockfile and contradicting the byte-faithful intent.
with open(path, newline="") as fh:
    lines = fh.readlines()

done_top = done_own = False
for i, line in enumerate(lines):
    if '"node_modules/' in line:
        break
    if not done_top and re.fullmatch(
        r'\s*"version":\s*"' + re.escape(old_top) + r'",?\s*[\r\n]*', line
    ):
        lines[i] = line.replace(f'"{old_top}"', f'"{version}"', 1)
        done_top = True
        continue
    if done_top and not done_own and re.fullmatch(
        r'\s*"version":\s*"' + re.escape(old_own) + r'",?\s*[\r\n]*', line
    ):
        lines[i] = line.replace(f'"{old_own}"', f'"{version}"', 1)
        done_own = True

if not (done_top and done_own):
    sys.exit(f"{path}: could not locate both own-version lines "
             f"(top={done_top}, packages[''] ={done_own}) — not modified")

# Stage, verify, THEN replace. Verifying after an in-place write left a damaged
# file on disk whenever the line scan mis-targeted. os.replace is atomic within a
# filesystem. In mode=check the staged file is verified and discarded, so the
# pre-flight exercises exactly this path without touching the tree.
tmp = path + ".version-sync.tmp"
try:
    with open(tmp, "w", newline="") as fh:
        fh.writelines(lines)
    with open(tmp) as fh:
        after = json.load(fh)
    if after["version"] != version or after["packages"][""]["version"] != version:
        sys.exit(f"{path}: staged edit did not land structurally — not modified")
    if mode == "write":
        os.replace(tmp, path)
    else:
        os.unlink(tmp)
except BaseException:
    if os.path.exists(tmp):
        os.unlink(tmp)
    raise
PYEOF
)

  # Pre-flight the npm lockfiles: a DRY RUN of the real editor (mode=check),
  # so the predicate cannot drift from the writer's.
  local lock
  for lock in "ui/package-lock.json" "clients/nodejs-embedded/package-lock.json"; do
    [ -f "$REPO_ROOT/$lock" ] || continue
    python3 -c "$LOCK_EDIT_PY" "$REPO_ROOT/$lock" "$version" check || exit 1
  done

  # 1. Cargo.toml [package] version (root, first occurrence only)
  perl -i -pe 's/^version = ".*"/version = "'"$version"'"/ if 1..m/^version = / && /^version = /' "$REPO_ROOT/Cargo.toml"
  echo "  [SET]  Cargo.toml [package] -> $version"

  # 2. Cargo.toml [workspace.package] version — this is what the OTHER 150
  #    workspace crates inherit via `version.workspace = true`, including
  #    apps/proximadb-server (whose startup banner and runtime-state report
  #    env!("CARGO_PKG_VERSION")) and the proximadb-embedded binding (which sets
  #    the Python package's __version__). Missing it meant a "0.4.0" release
  #    whose server said 0.2.0 while its own /health said 0.4.0.
  #    Steps 2 and 3 are python with ASSERTIONS rather than bare perl, because
  #    both failed silently: step 2's regex required `version` to be the FIRST
  #    key after the header (move `edition` above it and the substitution matched
  #    nothing while `[SET]` printed anyway), and step 3's required `version`
  #    BEFORE `path` — so a pin written `{ path = "...", version = "..." }` was
  #    READ by check and never written by set, leaving a gate its own remediation
  #    could not fix.
  python3 - "$REPO_ROOT/Cargo.toml" "$version" <<'PYEOF' || exit 1
import re
import sys

path, version = sys.argv[1], sys.argv[2]
text = open(path).read()

# [workspace.package] version, wherever it sits inside that table.
m = re.search(r"^\[workspace\.package\]\s*$", text, re.M)
if not m:
    sys.exit(f"{path}: no [workspace.package] table — not modified")
start = m.end()
nxt = re.search(r"^\[", text[start:], re.M)
end = start + (nxt.start() if nxt else len(text) - start)
table = text[start:end]
table_new, n = re.subn(
    r'(?m)^(\s*version\s*=\s*)["\'][^"\']*["\']', r'\g<1>"' + version + '"', table, count=1
)
if n != 1:
    sys.exit(f"{path}: [workspace.package] has no version key — not modified")
text = text[:start] + table_new + text[end:]

# Intra-workspace path-dep pins, in EITHER key order.
pin = re.compile(
    r'(\{[^{}\n]*?)version\s*=\s*["\']\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?["\']'
    r'([^{}\n]*?path\s*=)'
)
text, a = pin.subn(r'\g<1>version = "' + version + r'"\g<2>', text)
pin_rev = re.compile(
    r'(\{[^{}\n]*?path\s*=[^{}\n]*?)version\s*=\s*["\']\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?["\']'
)
text, b = pin_rev.subn(r'\g<1>version = "' + version + '"', text)
open(path, "w").write(text)
print(f"  [SET]  Cargo.toml [workspace.package] + {a + b} path-dep pin(s) -> {version}")
PYEOF

  # 4. pyproject.toml (root) — this is the canonical proximadb_embedded wheel.
  perl -i -pe 's/^version = ".*"/version = "'"$version"'"/' "$REPO_ROOT/pyproject.toml"
  echo "  [SET]  pyproject.toml -> $version"

  # 5. clients/python/pyproject.toml, including the self-referential `embedded`
  #    extra: its own upper bound would otherwise exclude the wheel this release
  #    builds, making `pip install 'proximadb[embedded]'` unsatisfiable.
  perl -i -pe 's/^version = ".*"/version = "'"$version"'"/' "$REPO_ROOT/clients/python/pyproject.toml"
  python3 - "$REPO_ROOT/clients/python/pyproject.toml" "$version" <<'PYEOF' || exit 1
import re
import sys

path, version = sys.argv[1], sys.argv[2]
# `[\d.]+` could not match a PRERELEASE lower bound, so once `set 0.4.1-beta.1`
# wrote ">=0.4.1-beta.1,<0.5.0" no later `set` could repair it: perl matched
# nothing, wrote nothing, and the caller's echo still claimed success. At 0.5.0
# that bound excludes the very wheel the release builds — the defect this step
# exists to prevent, made permanent. Match any bound, and ASSERT the rewrite
# applied rather than trusting it.
base = version.split("-", 1)[0]
major, minor, _ = base.split(".")
upper = f"{major}.{int(minor) + 1}.0"
text = open(path).read()
pattern = r'"proximadb_embedded>=[^,"]+,<[^"]+"'
replacement = f'"proximadb_embedded>={version},<{upper}"'
new, n = re.subn(pattern, replacement, text)
if n != 1:
    sys.exit(
        f"{path}: expected exactly 1 proximadb_embedded bound to rewrite, found {n} "
        "— not modified"
    )
open(path, "w").write(new)
PYEOF
  echo "  [SET]  clients/python/pyproject.toml (+ embedded extra) -> $version"

  # (no clients/python-embedded/pyproject.toml step — see the note in
  # check_versions(): that file has no version field, so the substitution
  # matched nothing while the echo claimed success.)

  # 6. clients/python/src/proximadb_sdk/__init__.py
  perl -i -pe 's/__version__ = ".*"/__version__ = "'"$version"'"/' "$REPO_ROOT/clients/python/src/proximadb_sdk/__init__.py"
  echo "  [SET]  clients/python/src/proximadb_sdk/__init__.py -> $version"

  # 7. clients/python-embedded/src/proximadb_embedded/__init__.py
  perl -i -pe 's/__version__ = ".*"/__version__ = "'"$version"'"/' "$REPO_ROOT/clients/python-embedded/src/proximadb_embedded/__init__.py"
  echo "  [SET]  clients/python-embedded/src/proximadb_embedded/__init__.py -> $version"

  # 8. clients/rust/Cargo.toml (first occurrence only)
  perl -i -pe 's/^version = ".*"/version = "'"$version"'"/ if 1..m/^version = / && /^version = /' "$REPO_ROOT/clients/rust/Cargo.toml"
  echo "  [SET]  clients/rust/Cargo.toml -> $version"

  # 9+10. deploy/helm/proximadb/Chart.yaml
  perl -i -pe 's/^version: .*/version: '"$version"'/' "$REPO_ROOT/deploy/helm/proximadb/Chart.yaml"
  perl -i -pe 's/^appVersion: .*/appVersion: "'"$version"'"/' "$REPO_ROOT/deploy/helm/proximadb/Chart.yaml"
  echo "  [SET]  deploy/helm/proximadb/Chart.yaml -> $version"

  # 11. ui/package.json (first version field)
  perl -i -pe 's/"version": ".*"/"version": "'"$version"'"/ if 1..m/"version":/ && /"version":/' "$REPO_ROOT/ui/package.json"
  echo "  [SET]  ui/package.json -> $version"

  # 12. clients/nodejs-embedded/package.json (first version field)
  perl -i -pe 's/"version": ".*"/"version": "'"$version"'"/ if 1..m/"version":/ && /"version":/' "$REPO_ROOT/clients/nodejs-embedded/package.json"
  echo "  [SET]  clients/nodejs-embedded/package.json -> $version"

  # 13. The npm lockfiles' own version fields, LAST so an abort here cannot skip
  #     a package.json above. Identified STRUCTURALLY (doc["version"] and
  #     doc["packages"][""]["version"]), not positionally: a positional "first
  #     two version lines before node_modules/" walk silently corrupted a
  #     dependency pin on a lockfileVersion-1 lockfile (no `packages` object, no
  #     node_modules/ key) and reported success. Pre-flighted above.
  #
  #     12b first: the two manifests F9 found stale at 0.2.0 and unwatched.
  if [ -f "$REPO_ROOT/clients/python-queue-embedded/pyproject.toml" ]; then
    perl -i -pe 's/^version = ".*"/version = "'"$version"'"/' "$REPO_ROOT/clients/python-queue-embedded/pyproject.toml"
    echo "  [SET]  clients/python-queue-embedded/pyproject.toml -> $version"
  fi
  if [ -f "$REPO_ROOT/clients/java-embedded/pom.xml" ]; then
    python3 - "$REPO_ROOT/clients/java-embedded/pom.xml" "$version" <<'PYEOF' || exit 1
import os
import re
import sys
import xml.etree.ElementTree as ET

path, version = sys.argv[1], sys.argv[2]
src = open(path, encoding="utf-8").read()

# Locate /project/version by DEPTH, not by position relative to </parent>.
# <project>'s children are an xs:all, so a project declaring <version> after
# <dependencies> makes the first post-parent <version> a DEPENDENCY's -- an
# earlier revision of this writer did exactly that and rewrote junit-jupiter's
# version while `check` reported [OK] over it. Editing a span of the original
# text (rather than ET.write) keeps comments, entities and formatting intact.
depth = 0
span = None
for m in re.finditer(r"<\?[^>]*\?>|<!--.*?-->|<!\[CDATA\[.*?\]\]>|</?([A-Za-z_][\w.:-]*)([^>]*?)(/?)>", src, re.S):
    name, attrs, selfclose = m.group(1), m.group(2), m.group(3)
    if name is None:          # declaration, comment or CDATA
        continue
    if m.group(0).startswith("</"):
        depth -= 1
        continue
    if selfclose == "/":
        continue
    depth += 1
    # depth 1 == <project>; depth 2 == its direct children.
    if depth == 2 and name.split(":")[-1] == "version":
        close = src.find("</", m.end())
        if close != -1:
            span = (m.end(), close)
        break

if span is None:
    sys.exit(f"{path}: no /project/version element (inherited from <parent>?) — not modified")

staged = src[:span[0]] + version + src[span[1]:]

# Stage, verify by PARSING, then replace. The verify is namespace-aware and
# checks the project version specifically, so an edit that landed on the wrong
# element cannot pass -- which is the failure the previous revision had.
tmp = path + ".version-sync.tmp"
try:
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(staged)
    root = ET.parse(tmp).getroot()
    ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    el = root.find(ns + "version")
    if el is None or (el.text or "").strip() != version:
        sys.exit(f"{path}: staged edit did not land on /project/version — not modified")
    # And nothing else moved: exactly the version span changed.
    if len(staged) - len(src) != len(version) - (span[1] - span[0]):
        sys.exit(f"{path}: staged edit changed the document length unexpectedly — not modified")
    os.replace(tmp, path)
except BaseException:
    if os.path.exists(tmp):
        os.unlink(tmp)
    raise
PYEOF
    echo "  [SET]  clients/java-embedded/pom.xml -> $version"
  fi

  for lock in "ui/package-lock.json" "clients/nodejs-embedded/package-lock.json"; do
    [ -f "$REPO_ROOT/$lock" ] || continue
    python3 -c "$LOCK_EDIT_PY" "$REPO_ROOT/$lock" "$version" write || exit 1
    echo "  [SET]  $lock -> $version"
  done


  # 14. Cargo.lock LAST, because it is derived from every manifest edited above.
  #     Without this, `set` produced a tree that `check` PASSED and
  #     `cargo metadata --locked` (ci.yml, merge-blocking) REJECTED with
  #     "cannot update the lock file ... because --locked was passed" -- so the
  #     remediation message `check` prints handed you a red PR. Offline, so it
  #     cannot reach the network mid-release: every workspace member is a path
  #     dep, so re-resolving their versions needs no registry access.
  if command -v cargo >/dev/null 2>&1; then
    if cargo update --workspace --offline --manifest-path "$REPO_ROOT/Cargo.toml" >/dev/null 2>&1; then
      echo "  [SET]  Cargo.lock (cargo update -w --offline) -> $version"
    else
      echo "Error: Cargo.lock NOT refreshed (cargo update -w --offline failed)." >&2
      echo "       Every manifest above is bumped, so the tree is INCONSISTENT:" >&2
      echo "       cargo metadata --locked is merge-blocking and will reject it." >&2
      echo "       Re-run cargo update -w yourself, then version-sync.sh check." >&2
      exit 1
    fi
  else
    echo "Error: cargo not on PATH, so Cargo.lock was NOT refreshed." >&2
    echo "       The manifests above are bumped and the lock is stale." >&2
    exit 1
  fi

  echo ""
  echo "Done. Run 'bash scripts/version-sync.sh check' to verify."
}

# Main
case "${1:-}" in
  get)
    cmd_get
    ;;
  check)
    cmd_check
    ;;
  set)
    if [ -z "${2:-}" ]; then
      echo "Usage: $0 set <version>"
      echo "Example: $0 set 0.3.0"
      exit 1
    fi
    cmd_set "$2"
    ;;
  *)
    echo "Usage: $0 {check|set <version>|get}"
    echo ""
    echo "Commands:"
    echo "  check          Validate all version files match Cargo.toml"
    echo "  set <version>  Update all version files to <version>"
    echo "  get            Print current version from Cargo.toml"
    exit 1
    ;;
esac
