#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Land a rendered Homebrew formula in the tap through a PR gated on the tap's
# own `CI Success` (brew audit + install tests; required on the tap's main).
#
# Called by release.yml `update-homebrew` with GH_TOKEN = secrets.HOMEBREW_TAP_TOKEN
# (fine-grained PAT on the tap: Contents + Pull requests read/write).
#
#   * No token: says so in the step summary and exits 0. The tap's own 6-hourly
#     update-data-tools workflow opens the same PR; release-smoke.yml's homebrew
#     job stays strict and reports the tap as stale until it lands.
#   * Main already serves exactly this formula (e.g. the tap's updater won the
#     race): done.
#   * Otherwise, up to ATTEMPTS times: reset the branch onto the CURRENT main,
#     write the formula, open/reuse the PR, wait (bounded) for its CI Success,
#     merge. The tap requires branches to be up to date, and its cron updaters
#     move main; a merge refused because main moved is retried from the new
#     main, any other failure is fatal.
#
# Only an HTTP 404 means "absent" for any read here; other API errors fail.
#
# Usage: release_homebrew_publish.sh <tap> <formula_path> <version> <rendered.rb> <run_url>
set -euo pipefail
TAP="${1:?tap}"; FORMULA="${2:?formula path}"; VERSION="${3:?version}"
RENDERED="${4:?rendered formula}"; RUN_URL="${5:?run url}"
ATTEMPTS="${HOMEBREW_PUBLISH_ATTEMPTS:-3}"
CI_WAIT_MIN="${HOMEBREW_CI_WAIT_MINUTES:-75}"
POLL="${HOMEBREW_POLL_SECONDS:-60}"
BRANCH="automation/proximadb-v${VERSION}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

if [ -z "${GH_TOKEN:-}" ]; then
  echo "::warning::HOMEBREW_TAP_TOKEN is not set: ${TAP} was NOT updated to ${VERSION}."
  {
    echo "### Homebrew tap NOT updated"
    echo "Secret \`HOMEBREW_TAP_TOKEN\` is missing (fine-grained PAT on ${TAP}: Contents + Pull requests read/write)."
    echo "The rendered formula is the \`homebrew-formula\` artifact; the tap's update-data-tools workflow opens the same PR within 6h."
  } >> "$SUMMARY"
  exit 0
fi

# read_file <ref> <out>: 0 = written, 4 = absent (404), anything else = fatal.
read_file() {
  local ref="$1" out="$2" err
  err="$(mktemp)"
  if gh api "repos/${TAP}/contents/${FORMULA}?ref=${ref}" --jq .content 2> "$err" | base64 -d > "$out"; then
    rm -f "$err"; return 0
  fi
  if grep -q 'HTTP 404' "$err"; then rm -f "$err"; return 4; fi
  echo "::error::cannot read ${TAP}/${FORMULA}@${ref}: $(cat "$err")"
  rm -f "$err"; exit 1
}

wait_ci() {
  local pr="$1" deadline state
  deadline=$(( $(date +%s) + CI_WAIT_MIN * 60 ))
  while :; do
    state="$(gh pr view -R "$TAP" "$pr" --json statusCheckRollup \
      --jq '[.statusCheckRollup[] | select(.name == "CI Success")] | last | (.conclusion // .status // "")')"
    case "$state" in
      SUCCESS) return 0 ;;
      FAILURE|CANCELLED|TIMED_OUT|ACTION_REQUIRED|STARTUP_FAILURE)
        echo "::error::${TAP} PR #${pr}: CI Success concluded ${state}; the formula did not land"; exit 1 ;;
    esac
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "::error::${TAP} PR #${pr}: CI Success not green within ${CI_WAIT_MIN} min"; exit 1
    fi
    sleep "$POLL"
  done
}

for attempt in $(seq 1 "$ATTEMPTS"); do
  MAIN_SHA="$(gh api "repos/${TAP}/git/ref/heads/main" --jq .object.sha)"
  rc=0; read_file "$MAIN_SHA" main.rb || rc=$?
  if [ "$rc" -eq 0 ] && cmp -s main.rb "$RENDERED"; then
    echo "${TAP} main already serves this formula for ${VERSION}"
    exit 0
  fi

  # Reset the branch onto the current main (create it if absent).
  rc=0; BR_ERR="$(gh api "repos/${TAP}/git/ref/heads/${BRANCH}" 2>&1 >/dev/null)" || rc=$?
  if [ "$rc" -eq 0 ]; then
    gh api -X PATCH "repos/${TAP}/git/refs/heads/${BRANCH}" -f sha="$MAIN_SHA" -F force=true >/dev/null
  elif grep -q 'HTTP 404' <<<"$BR_ERR"; then
    gh api -X POST "repos/${TAP}/git/refs" -f ref="refs/heads/${BRANCH}" -f sha="$MAIN_SHA" >/dev/null
  else
    echo "::error::cannot read branch ${BRANCH}: ${BR_ERR}"; exit 1
  fi

  ARGS=(-f message="proximadb v${VERSION}" -f branch="$BRANCH" -f content="$(base64 -w0 "$RENDERED")")
  rc=0; BLOB_ERR="$(mktemp)"
  BLOB="$(gh api "repos/${TAP}/contents/${FORMULA}?ref=${BRANCH}" --jq .sha 2> "$BLOB_ERR")" || rc=$?
  if [ "$rc" -eq 0 ]; then
    ARGS+=(-f sha="$BLOB")
  elif ! grep -q 'HTTP 404' "$BLOB_ERR"; then
    echo "::error::cannot read ${FORMULA}@${BRANCH}: $(cat "$BLOB_ERR")"; exit 1
  fi
  gh api -X PUT "repos/${TAP}/contents/${FORMULA}" "${ARGS[@]}" >/dev/null

  PR="$(gh pr list -R "$TAP" --head "$BRANCH" --state open --json number --jq '.[0].number // empty')"
  if [ -z "$PR" ]; then
    gh pr create -R "$TAP" --base main --head "$BRANCH" --title "proximadb v${VERSION}" \
      --body "Automated from ${RUN_URL}: url + sha256 for the v${VERSION} linux x86_64 archive, verified against SHA256SUMS.txt and the downloaded bytes."
    PR="$(gh pr list -R "$TAP" --head "$BRANCH" --state open --json number --jq '.[0].number')"
  fi
  echo "attempt ${attempt}/${ATTEMPTS}: ${TAP} PR #${PR}; waiting for CI Success"
  wait_ci "$PR"

  if gh pr merge -R "$TAP" "$PR" --squash --delete-branch; then
    echo "${TAP}: ${FORMULA} now serves ${VERSION} (PR #${PR})"
    exit 0
  fi
  NOW_MAIN="$(gh api "repos/${TAP}/git/ref/heads/main" --jq .object.sha)"
  if [ "$NOW_MAIN" = "$MAIN_SHA" ]; then
    echo "::error::${TAP} PR #${PR} could not be merged and main did not move -- not a race; see the PR"
    exit 1
  fi
  echo "main moved (${MAIN_SHA:0:12} -> ${NOW_MAIN:0:12}) while PR #${PR} was in CI; rebasing and retrying"
done
echo "::error::${TAP} PR for ${VERSION} not merged after ${ATTEMPTS} attempts (main kept moving)"
exit 1
