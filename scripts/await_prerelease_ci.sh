#!/usr/bin/env bash
# Wait for the Pre-Release CI workflow (prerelease-ci.yml) to pass on a commit.
#
# Used by release.yml and release-python.yml before anything is built or
# published. The tag is normally pushed right after the qa->main merge lands,
# while Pre-Release CI for that same commit is still running (~55 min). A
# single "has it passed yet?" query failed the v0.4.1 release outright
# (2026-10-16). This waits for the verdict instead:
#
#   success run for SHA exists        -> exit 0
#   newest run for SHA concluded red  -> exit 1 (fail fast, do not wait)
#   a run for SHA is queued/running   -> keep polling
#   no run for SHA at all             -> poll for a short grace window, then exit 1
#   gh api error (e.g. a transient 502) -> warn and retry until the deadline
#
# Only a run that really builds counts: a push to main, or a dispatch. A push to
# qa also triggers prerelease-ci.yml, but every build job is skipped there
# (`if: !(push && ref_name == 'qa')`), so that run concludes `success` in ~15 s
# having verified nothing, and it must not satisfy this gate.
#
# On success, when GITHUB_OUTPUT is set, writes `run_id=<id>` of the newest
# successful building run, so release.yml can reuse that run's binaries.
#
# Usage: await_prerelease_ci.sh <repo> <sha> [deadline_minutes] [grace_minutes]
# Needs GH_TOKEN with actions:read. Poll interval: PRERELEASE_POLL_SECONDS (60).
set -euo pipefail

REPO="${1:?repo (owner/name) required}"
SHA="${2:?commit sha required}"
DEADLINE_MIN="${3:-110}"
GRACE_MIN="${4:-10}"
POLL="${PRERELEASE_POLL_SECONDS:-60}"
WORKFLOW="prerelease-ci.yml"

start=$(date +%s)
deadline=$((start + DEADLINE_MIN * 60))
grace=$((start + GRACE_MIN * 60))

echo "Waiting for ${WORKFLOW} to pass on ${SHA} (deadline ${DEADLINE_MIN} min)"
while :; do
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    echo "::error::Pre-Release CI did not pass on ${SHA} within ${DEADLINE_MIN} min."
    exit 1
  fi

  # Newest run first. One query gives both the success check and the newest
  # run's state, so a red run is seen without waiting out the deadline.
  if ! runs=$(gh api "repos/${REPO}/actions/workflows/${WORKFLOW}/runs?head_sha=${SHA}&per_page=20" \
    --jq '[.workflow_runs[]
           | select((.event == "push" and .head_branch == "main") or .event == "workflow_dispatch")
           | {id, status, conclusion}]'); then
    # A transient API error must not throw away a long wait; the deadline
    # still bounds the retries.
    echo "::warning::Could not query ${WORKFLOW} runs; retrying."
    sleep "$POLL"
    continue
  fi
  succeeded=$(jq '[.[] | select(.status == "completed" and .conclusion == "success")] | length' <<<"$runs")
  total=$(jq 'length' <<<"$runs")

  if [ "$succeeded" -ge 1 ]; then
    # Newest successful building run: release.yml downloads its binaries
    # instead of rebuilding them (CI redesign PR1).
    run_id=$(jq -r '[.[] | select(.status == "completed" and .conclusion == "success")][0].id' <<<"$runs")
    echo "Pre-Release CI passed on ${SHA} (${succeeded} successful run(s); newest ${run_id})."
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
      echo "run_id=${run_id}" >> "$GITHUB_OUTPUT"
    fi
    exit 0
  fi

  if [ "$total" -eq 0 ]; then
    if [ "$now" -ge "$grace" ]; then
      echo "::error::No ${WORKFLOW} run exists for ${SHA} after ${GRACE_MIN} min."
      echo "::error::A building Pre-Release CI run starts on push to main (a qa push skips every build)."
      echo "::error::Tag the qa->main merge commit on main, or dispatch ${WORKFLOW} on a ref whose head is ${SHA}."
      exit 1
    fi
    echo "No ${WORKFLOW} run for ${SHA} yet; waiting."
  else
    newest_status=$(jq -r '.[0].status' <<<"$runs")
    newest_conclusion=$(jq -r '.[0].conclusion // ""' <<<"$runs")
    newest_id=$(jq -r '.[0].id' <<<"$runs")
    if [ "$newest_status" = "completed" ]; then
      echo "::error::Pre-Release CI run ${newest_id} on ${SHA} concluded '${newest_conclusion}'."
      echo "::error::Fix it (or re-run it) and then re-run this workflow's failed jobs; the tag can stay."
      exit 1
    fi
    echo "Pre-Release CI run ${newest_id} is ${newest_status}; waiting."
  fi
  sleep "$POLL"
done
