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

get_cargo_version() {
  grep '^version = ' "$REPO_ROOT/Cargo.toml" | head -1 | sed 's/version = "\(.*\)"/\1/'
}

extract_version_from_file() {
  local file="$1"
  local type="$2"

  if [ ! -f "$file" ]; then
    echo ""
    return
  fi

  case "$type" in
    cargo)
      grep '^version = ' "$file" | head -1 | sed 's/.*"\(.*\)".*/\1/'
      ;;
    pyproject)
      grep '^version = ' "$file" | sed 's/.*"\(.*\)".*/\1/'
      ;;
    python_init)
      grep '__version__' "$file" | head -1 | sed 's/.*"\(.*\)".*/\1/'
      ;;
    yaml_version)
      grep '^version:' "$file" | sed 's/.*: *\(.*\)/\1/'
      ;;
    yaml_appversion)
      grep '^appVersion:' "$file" | sed 's/.*"\(.*\)".*/\1/' | head -1
      ;;
    package_json)
      grep '"version"' "$file" | head -1 | sed 's/.*: *"\(.*\)".*/\1/'
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
      python3 -c 'import sys,tomllib
with open(sys.argv[1],"rb") as fh: d=tomllib.load(fh)
v=d.get("workspace",{}).get("package",{}).get("version")
print(v if v else "<MISSING-workspace.package.version>")' "$file" 2>/dev/null \
        || echo "<UNPARSEABLE-$file>"
      ;;
    # M2: intra-workspace path dependencies carry an explicit version
    # requirement that MUST track [workspace.package] — cargo fails with
    # "failed to select a version for the requirement" otherwise, a hard build
    # break the release gate was blind to. Reports the first mismatching pin so
    # the comparison fails with a useful value.
    cargo_path_dep_pins)
      python3 -c 'import sys,tomllib
with open(sys.argv[1],"rb") as fh: d=tomllib.load(fh)
pins={}
def walk(tbl, prefix=""):
    for name,spec in (tbl or {}).items():
        if isinstance(spec,dict) and "path" in spec and "version" in spec:
            pins[prefix+name]=spec["version"]
for table in ("dependencies","dev-dependencies","build-dependencies"):
    walk(d.get(table))
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
        || echo "<UNPARSEABLE-$file>"
      ;;
    # M2: the `embedded` extra pins the wheel this release builds. Its own upper
    # bound once excluded that wheel, making `pip install proximadb[embedded]`
    # unsatisfiable. Reports the LOWER bound so a stale pin fails.
    python_extra_lower_bound)
      python3 -c 'import re,sys
s=open(sys.argv[1]).read()
m=re.search(r"proximadb_embedded>=([0-9][^,\"]*),<", s)
print(m.group(1) if m else "<NO-EMBEDDED-EXTRA-BOUND>")' "$file" 2>/dev/null \
        || echo "<UNPARSEABLE-$file>"
      ;;
    # A package-lock.json records the package's own version TWICE: top-level and
    # packages[""]. `package_json` above only sees the first, so the second could
    # drift unnoticed — which is half of what `set` writes.
    # The `|| echo` is load-bearing, not decoration: without it a non-zero python
    # exit propagates out of the command substitution and `set -e` terminates
    # `check` with an EMPTY stderr, mid-list, before the remaining entries run —
    # so the gate dies with no diagnostics on exactly the field it was added to
    # watch. Verified by deleting `packages[""]["version"]`.
    package_lock_own)
      python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["packages"][""]["version"])' "$file" 2>/dev/null \
        || echo "<UNPARSEABLE-$file>"
      ;;
    *)
      echo ""
      ;;
  esac
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
  )

  for entry in "${files[@]}"; do
    file="${entry%%:*}"
    type="${entry##*:}"

    actual=$(extract_version_from_file "$REPO_ROOT/$file" "$type")

    # A `<...>` sentinel means the extractor ran and found the field missing or
    # malformed. That is a FAILURE, not a skip: an empty extraction reading as
    # [SKIP] is how a deleted [workspace.package] version passed this gate.
    if [ "${actual#<}" != "$actual" ]; then
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

  # Pre-flight the npm lockfiles BEFORE writing anything, so a malformed lockfile
  # aborts with the tree untouched rather than half-bumped.
  local lock
  for lock in "ui/package-lock.json" "clients/nodejs-embedded/package-lock.json"; do
    [ -f "$REPO_ROOT/$lock" ] || continue
    python3 - "$REPO_ROOT/$lock" <<'PYEOF' || exit 1
import json
import sys

path = sys.argv[1]
try:
    with open(path) as fh:
        doc = json.load(fh)
except (OSError, json.JSONDecodeError) as exc:
    sys.exit(f"{path}: not readable as JSON ({exc})")
if "version" not in doc:
    sys.exit(f'{path}: no top-level "version" key')
if "version" not in doc.get("packages", {}).get("", {}):
    sys.exit(f'{path}: no packages[""]["version"] key (lockfileVersion 1?)')

# Key existence is not enough: the writer edits by LINE, so each own-version
# field must also sit alone on a line it can match. Checking only the keys let a
# minified-but-valid lockfile pass the pre-flight and then fail the writer, which
# is exactly the half-bumped tree this pre-flight promises to prevent.
import re

with open(path, newline="") as fh:
    lines = fh.readlines()
wanted = [doc["version"], doc["packages"][""]["version"]]
found = 0
for line in lines:
    if '"node_modules/' in line:
        break
    if found < len(wanted) and re.fullmatch(
        r'\s*"version":\s*"' + re.escape(wanted[found]) + r'",?\s*[\r\n]*', line
    ):
        found += 1
if found != 2:
    sys.exit(
        f"{path}: the two own-version fields are not on separately matchable lines "
        f"(matched {found}/2) — this writer edits by line, so it would fail "
        "mid-run; not modified"
    )
PYEOF
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
  perl -i -0pe 's/(\[workspace\.package\]\s*\nversion = )".*"/${1}"'"$version"'"/' "$REPO_ROOT/Cargo.toml"
  echo "  [SET]  Cargo.toml [workspace.package] -> $version"

  # 3. Intra-workspace path dependencies carry an explicit version requirement,
  #    which must track [workspace.package] or resolution fails with
  #    "failed to select a version for the requirement".
  perl -i -pe 's/version = "\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?", path =/version = "'"$version"'", path =/g' "$REPO_ROOT/Cargo.toml"
  echo "  [SET]  Cargo.toml intra-workspace path deps -> $version"

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
  for lock in "ui/package-lock.json" "clients/nodejs-embedded/package-lock.json"; do
    [ -f "$REPO_ROOT/$lock" ] || continue
    python3 - "$REPO_ROOT/$lock" "$version" <<'PYEOF' || exit 1
import json
import re
import sys

path, version = sys.argv[1], sys.argv[2]
with open(path) as fh:
    doc = json.load(fh)
old_top = doc["version"]
old_own = doc["packages"][""]["version"]

# Edit by line so indentation and key order survive byte-for-byte, but take the
# TARGETS from the parsed document rather than from position.
# newline="" on BOTH read and write: text mode would translate CRLF to LF
# throughout, rewriting every line of a CRLF lockfile and contradicting the
# byte-faithful intent stated above.
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

with open(path, "w", newline="") as fh:
    fh.writelines(lines)

# Re-read and assert the structural result, so a line edit that landed in the
# wrong place cannot pass.
with open(path) as fh:
    after = json.load(fh)
if after["version"] != version or after["packages"][""]["version"] != version:
    sys.exit(f"{path}: post-write check failed")
PYEOF
    echo "  [SET]  $lock -> $version"
  done


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
