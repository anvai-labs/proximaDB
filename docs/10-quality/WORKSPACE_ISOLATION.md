# Workspace Isolation — one task = one worktree = one branch

> **Mandate:** Never edit or commit in the main checkout
> (`/…/proximaDB`). Every unit of work gets its own isolated git **worktree**.
> Use `scripts/worktree.sh`. This is enforced for AI agents (see `CLAUDE.md`)
> and strongly recommended for humans running concurrent sessions.

## Why this exists

ProximaDB is frequently worked on by **several sessions at once** — multiple AI
agents and/or a human, in parallel. When they all share a single git checkout
they also share one **HEAD, index, and working tree** — a single piece of
mutable state that they fight over. Observed failure modes (all real):

| Symptom | Cause |
|---|---|
| A commit lands on the **wrong branch** | Another session ran `git checkout` and moved the shared HEAD between your `checkout -b` and your `commit`. |
| Edits silently clobber someone's work | The shared tree held another session's **uncommitted WIP**. |
| `git stash`/`checkout` destroys WIP | Same shared tree; defensive `git commit -o <file>` was the only safe write. |
| CI head-of-line stalls | Everyone pushes the same shared branches; superseded runs pile up. |

The fix is to **eliminate the shared mutable state**: give every task its own
worktree (its own directory + branch + HEAD + index), branched from a
known-good base. Isolation is then structural, not a matter of discipline.

## The model

- **One task = one worktree = one branch = one PR.** A session may own several.
- Worktrees live in a **sibling** directory next to the repo, so they're
  outside it (no `.gitignore`, no tooling scans them):
  ```
  /…/code/proximaDB/                      ← main checkout (read/run only; never edit)
  /…/code/proximaDB.worktrees/
      feat-cloud-full/                    ← branch feat/cloud-full
      fix-adls-scheme/                    ← branch fix/adls-scheme
  ```
  Override the root with `PROXIMADB_WORKTREE_ROOT`.
- Every branch is cut from **`origin/develop`** (freshly fetched), the
  integration base.

## Commands (`scripts/worktree.sh`)

```bash
# Start a task — creates the worktree + branch off origin/develop and
# prints a `cd`. eval it to land inside:
eval "$(scripts/worktree.sh new feat/my-thing)"

scripts/worktree.sh list                 # all worktrees + which are dirty
scripts/worktree.sh guard                # exits non-zero if run in the main checkout
scripts/worktree.sh check                # cargo check ONLY the crates changed vs develop
scripts/worktree.sh test                 # cargo nextest ONLY the crates changed vs develop
scripts/worktree.sh rm feat/my-thing     # remove after the PR merges (guards dirty)
scripts/worktree.sh clean                # reclaim worktrees merged to develop (defensive: protects active worktrees)
```

`new` refuses to clobber an existing branch/path; `rm` refuses to delete a
dirty worktree (pass `--force` to discard) and only deletes the branch if it's
merged; `clean` is the periodic housekeeping pass. `clean` is **defensive**: it
reclaims only worktrees whose branch has genuinely landed in develop (squash- or
merge-commit) and has a clean tree — it never deletes a *fresh/active* worktree
whose branch tip still equals develop (no commits of its own; possibly mid-build
with gitignored `target/` WIP invisible to `status --porcelain`), nor a dirty
one. A branch later fast-forwarded so develop == its tip is indistinguishable
from a fresh one and is likewise protected — remove such a worktree explicitly
with `rm`.

**Affected-only feedback.** `check`/`test` map your diff vs `develop` to the
owning crates (via `scripts/affected_crates.py` + `cargo metadata`) and build/
test only those — so editing one crate doesn't compile the 66-crate workspace.
It's direct crates only; for a change to a low-level/foundation crate, also run
the full `make test` (a dependent in another crate won't be picked up).

**Pre-push hook.** `new` activates a fast `.githooks/pre-push` (repo-wide, via
`core.hooksPath`) that blocks pushing from the main checkout and on rustfmt
drift — no compile, so it stays quick. Activate manually in an existing clone
with `git config core.hooksPath .githooks`; bypass once with `git push --no-verify`.

**Commit-msg hook (no agent attribution).** `.githooks/commit-msg` (same
`core.hooksPath`) rejects AI-agent authorship attribution in commit messages —
co-author trailers naming an agent, "Generated with ..." taglines, the robot
emoji, and agent signatures (e.g. "Claude Opus", "Claude Code"). The human drives
the code, not the agents. Mere mentions of `CLAUDE.md`/`GEMINI.md` or the
Anthropic/OpenAI APIs are fine — only attribution is blocked. The same check runs
in CI (`layering-check.yml`) over PR commits and the PR title/body (which becomes
the squash-merge message); logic lives in `scripts/check_no_agent_attribution.py`.
Bypass a single commit with `git commit --no-verify`.

## Lifecycle

```
new feat/x  ──▶  edit · commit · push · open PR  ──▶  PR merges  ──▶  rm feat/x
                 (inside the worktree only)                          (or: clean)
```

## Agent mandate (also in `CLAUDE.md`)

1. **Before editing anything, run `scripts/worktree.sh guard`.** If it fails,
   you are in the main checkout — stop and create a worktree.
2. **Never** `checkout`, `reset`, `stash`, or `branch -f` in a tree you don't
   own; never touch another worktree's branch or WIP.
3. Commit only files you created/changed (the worktree makes this natural; if
   ever in a shared tree, fall back to `git commit -o <file>`).
4. Clean up (`rm`) once your PR merges.
5. **Commit WIP to your branch early and often — never carry valuable
   uncommitted work in a worktree.** A worktree is disposable: `rm --force`, a
   concurrent session, or a merge/cleanup pass can remove or reset it at any
   time, and **uncommitted edits are lost with the directory**. Committed work
   survives in the branch ref (recoverable even if the worktree is deleted).
   Commit each coherent increment (it's fine to amend/rebase later); treat
   "still in the editor, not yet committed" as fragile. (`clean` is defensive —
   it protects a no-commit worktree whose tip == develop — but `rm --force` and
   the OS can still drop a worktree, so committing remains the durable guard.)
6. **Re-verify state before resuming or acting.** After any gap — and before
   any destructive or push step — confirm the worktree still exists
   (`scripts/worktree.sh list`) and the branch hasn't advanced under you
   (`git log --oneline -3`; unexpected commits you didn't author mean a
   concurrent session is on the same task — stop and reconcile, don't blindly
   re-edit on top). A worktree whose directory vanishes mid-session is the
   signal that another session merged/cleaned the branch; your uncommitted WIP
   is gone — recover from your last commit, not from memory of the edits.

## Runtime isolation (beyond git)

Git worktrees isolate *source*, but two agents running servers/tests in parallel
also share **runtime** state — TCP ports, data dirs, the Docker daemon,
registries. Isolate those too:

- **Server ports.** The default server binds 5678 (REST, gRPC and Flight
  multiplexed) and 5433 (pgwire); two servers on the same port collide. Give
  each agent's server its own port with `-p <port>` and its own pgwire port in a
  per-agent config (`[api] pg_port`). The `PROXIMADB_*_PORT` environment
  variables are not read by any code.
- **Point tests at your server.** Integration tests honor `REST_API_URL`
  (default `http://127.0.0.1:5678`) — set it to your agent's server so suites in
  different worktrees don't hit one shared instance. (A few legacy tests still
  hardcode the URL; standardizing them onto `REST_API_URL` is a tracked follow-up.)
- **Data dirs.** Use a per-agent `PROXIMADB_DATA_DIR` (tests already use
  `tempfile::TempDir`); never share `/data/proximadb`.
- **Docker.** Use a unique Compose **project name** (`docker compose -p
  pdb-<branch>`) and don't publish fixed host ports — let Docker assign them or
  map to the agent's ephemeral ports, so containers/networks/volumes don't
  collide across agents.
- **Registries / tags.** Use per-PR image tags (`pr-<n>-<sha>`) so concurrent
  publishes don't clobber each other.

Rule of thumb: anything with a *fixed* global name (port, path, container, tag)
needs a per-agent value before two agents can use it at once.

## FAQ

- **Disk cost?** Worktrees share the one `.git` object store; only the checked-
  out files are duplicated. Cheap relative to the safety.
- **Builds?** Each worktree has its own `target/` (no cross-contamination of
  feature flags). That's a feature — a `--features cloud-full` build can't
  poison a default build. To avoid paying N× cold compiles across N worktrees,
  install **sccache** — `worktree.sh new` (and `worktree.sh cache-env` for an
  existing worktree) then emits `RUSTC_WRAPPER`/`SCCACHE_DIR`/`CARGO_INCREMENTAL=0`
  so every worktree shares one content-addressed compilation cache. Setup:
  `cargo install sccache` (or `brew install sccache`); it's opt-in — without it,
  builds work unchanged. (mold linker is a further win but isn't auto-wired yet,
  since setting `RUSTFLAGS` for it would clobber the repo's `codegen-units=4`.)
- **`/tmp` worktrees?** Fine for throwaway/CI, but prefer the sibling root so
  work survives reboots and is discoverable via `worktree.sh list`.
