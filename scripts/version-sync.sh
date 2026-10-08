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
    "clients/nodejs-embedded/package-lock.json:package_json"
  )

  for entry in "${files[@]}"; do
    file="${entry%%:*}"
    type="${entry##*:}"

    actual=$(extract_version_from_file "$REPO_ROOT/$file" "$type")

    if [ -z "$actual" ]; then
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
  if command -v perl >/dev/null 2>&1; then
    # 1. Cargo.toml (root, first occurrence only)
    perl -i -pe 's/^version = ".*"/version = "'"$version"'"/ if 1..m/^version = / && /^version = /' "$REPO_ROOT/Cargo.toml"
    echo "  [SET]  Cargo.toml -> $version"

    # 2. pyproject.toml (root)
    perl -i -pe 's/^version = ".*"/version = "'"$version"'"/' "$REPO_ROOT/pyproject.toml"
    echo "  [SET]  pyproject.toml -> $version"

    # 3. clients/python/pyproject.toml
    perl -i -pe 's/^version = ".*"/version = "'"$version"'"/' "$REPO_ROOT/clients/python/pyproject.toml"
    echo "  [SET]  clients/python/pyproject.toml -> $version"

    # (no clients/python-embedded/pyproject.toml step — see the note in
    # check_versions(): that file has no version field, so the substitution
    # matched nothing while the echo claimed success.)

    # 4. clients/python/src/proximadb_sdk/__init__.py
    perl -i -pe 's/__version__ = ".*"/__version__ = "'"$version"'"/' "$REPO_ROOT/clients/python/src/proximadb_sdk/__init__.py"
    echo "  [SET]  clients/python/src/proximadb_sdk/__init__.py -> $version"

    # 5. clients/python-embedded/src/proximadb_embedded/__init__.py
    perl -i -pe 's/__version__ = ".*"/__version__ = "'"$version"'"/' "$REPO_ROOT/clients/python-embedded/src/proximadb_embedded/__init__.py"
    echo "  [SET]  clients/python-embedded/src/proximadb_embedded/__init__.py -> $version"

    # 6. clients/rust/Cargo.toml (first occurrence only)
    perl -i -pe 's/^version = ".*"/version = "'"$version"'"/ if 1..m/^version = / && /^version = /' "$REPO_ROOT/clients/rust/Cargo.toml"
    echo "  [SET]  clients/rust/Cargo.toml -> $version"

    # 7+8. deploy/helm/proximadb/Chart.yaml
    perl -i -pe 's/^version: .*/version: '"$version"'/' "$REPO_ROOT/deploy/helm/proximadb/Chart.yaml"
    perl -i -pe 's/^appVersion: .*/appVersion: "'"$version"'"/' "$REPO_ROOT/deploy/helm/proximadb/Chart.yaml"
    echo "  [SET]  deploy/helm/proximadb/Chart.yaml -> $version"

    # 9. ui/package.json (first version field)
    perl -i -pe 's/"version": ".*"/"version": "'"$version"'"/ if $. <= 10 && /"version"/' "$REPO_ROOT/ui/package.json"
    echo "  [SET]  ui/package.json -> $version"

    # 10. the npm lockfiles' own version fields.
    #
    # A blunt substitution is WRONG here: ui/package-lock.json contains four
    # `"version": "0.3.0"` lines, two of which belong to dependencies that happen
    # to share the release version (@standard-schema/utils at the time of
    # writing). Only the two that precede the first "node_modules/" key are the
    # package's own, so bound the edit to those.
    if command -v python3 >/dev/null 2>&1; then
      for lock in "ui/package-lock.json" "clients/nodejs-embedded/package-lock.json"; do
        [ -f "$REPO_ROOT/$lock" ] || continue
        python3 - "$REPO_ROOT/$lock" "$version" <<'PYEOF'
import re
import sys

path, version = sys.argv[1], sys.argv[2]
with open(path) as fh:
    lines = fh.readlines()

patched = 0
for i, line in enumerate(lines):
    if '"node_modules/' in line:
        break
    new, n = re.subn(r'("version":\s*")[^"]*(")', r'\g<1>' + version + r'\g<2>', line)
    if n:
        lines[i] = new
        patched += 1
    if patched == 2:
        break

if patched != 2:
    sys.exit(f"{path}: expected 2 own-version fields before the first "
             f"node_modules entry, patched {patched}")

with open(path, "w") as fh:
    fh.writelines(lines)
PYEOF
        echo "  [SET]  $lock -> $version"
      done
    else
      echo "  [WARN] python3 not found — package-lock.json versions NOT updated;"
      echo "         'npm ci' will fail until they match package.json."
    fi

    # 11. clients/nodejs-embedded/package.json (first version field)
    perl -i -pe 's/"version": ".*"/"version": "'"$version"'"/ if $. <= 10 && /"version"/' "$REPO_ROOT/clients/nodejs-embedded/package.json"
    echo "  [SET]  clients/nodejs-embedded/package.json -> $version"
  else
    # Fallback: use gsed if available (macOS coreutils)
    if command -v gsed >/dev/null 2>&1; then
      local SED="gsed"
    else
      local SED="sed"
    fi

    # Note: This fallback may not work on all systems due to -i incompatibilities
    echo "  [WARN] perl not found, using sed (may fail on macOS)"

    $SED -i.bak 's/^version = ".*"/version = "'"$version"'"/' "$REPO_ROOT/Cargo.toml"
    rm -f "$REPO_ROOT/Cargo.toml.bak"

    $SED -i.bak 's/^version = ".*"/version = "'"$version"'"/' "$REPO_ROOT/pyproject.toml"
    rm -f "$REPO_ROOT/pyproject.toml.bak"

    $SED -i.bak 's/^version = ".*"/version = "'"$version"'"/' "$REPO_ROOT/clients/python/pyproject.toml"
    rm -f "$REPO_ROOT/clients/python/pyproject.toml.bak"

    $SED -i.bak 's/^version = ".*"/version = "'"$version"'"/' "$REPO_ROOT/clients/python-embedded/pyproject.toml"
    rm -f "$REPO_ROOT/clients/python-embedded/pyproject.toml.bak"

    $SED -i.bak 's/__version__ = ".*"/__version__ = "'"$version"'"/' "$REPO_ROOT/clients/python/src/proximadb_sdk/__init__.py"
    rm -f "$REPO_ROOT/clients/python/src/proximadb_sdk/__init__.py.bak"

    $SED -i.bak 's/__version__ = ".*"/__version__ = "'"$version"'"/' "$REPO_ROOT/clients/python-embedded/src/proximadb_embedded/__init__.py"
    rm -f "$REPO_ROOT/clients/python-embedded/src/proximadb_embedded/__init__.py.bak"

    $SED -i.bak 's/^version = ".*"/version = "'"$version"'"/' "$REPO_ROOT/clients/rust/Cargo.toml"
    rm -f "$REPO_ROOT/clients/rust/Cargo.toml.bak"

    $SED -i.bak 's/^version: .*/version: '"$version"'/' "$REPO_ROOT/deploy/helm/proximadb/Chart.yaml"
    $SED -i.bak 's/^appVersion: .*/appVersion: "'"$version"'"/' "$REPO_ROOT/deploy/helm/proximadb/Chart.yaml"
    rm -f "$REPO_ROOT/deploy/helm/proximadb/Chart.yaml.bak"

    $SED -i.bak 's/"version": ".*"/"version": "'"$version"'"/' "$REPO_ROOT/ui/package.json"
    rm -f "$REPO_ROOT/ui/package.json.bak"

    $SED -i.bak 's/"version": ".*"/"version": "'"$version"'"/' "$REPO_ROOT/clients/nodejs-embedded/package.json"
    rm -f "$REPO_ROOT/clients/nodejs-embedded/package.json.bak"

    echo "  [SET]  All files updated (using sed fallback)"
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
