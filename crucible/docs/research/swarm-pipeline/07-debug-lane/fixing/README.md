# Fixing the debug-lane tasks: how we do it

Status 2026-10-23. Uzair fixed the first four; Haseeb and Azaan take the rest (`haseeb.md`, `azaan.md`). Worked examples,
one per task, are in `notes/`; the bug reports we wrote for the changed tasks are in `prompts/` (copy their shape). Paths
under `crucible/data/debug-uzair/` and `crucible/data/targeting/uzair*` in the notes are on Uzair's machine (gitignored).

## Where things stand

Ahmed's 17 non-GPU debugging tasks (10 training, 7 held out) had **0 kept**. Now:

| Kept | How |
|---|---|
| `db-go-go-toml-a02114e3-e1b` (training) | dropped the bug the student never fixed; student 3/8 |
| `db-python-black-90117fa0-e1` (training) | swapped its slow first bug for a quicker one; student 3/8, no attempt ran out of calls |
| `db-go-urfave-cli-2b0adb14-d1` (benchmark) | applied an undone fix the report never described; added a missing ordering rule to one paragraph |
| `db-node-h3-a6cd4654` (benchmark) | unchanged task; fixed a changelog hint and two Node environment bugs |

Before the lead's call below, pulldown-cmark, unstorage, echo and the tasks whose "link" between bugs turned out fake
(mvdan-sh, black ae5830b7, carbon ×2, excelize ×2) were set aside. Now **no task is set aside**: only mvdan-sh 04056a74 and
black-ec71b918 are dropped (each repeats most of another task's fixes); every other task is changed and measured again
until it is kept.

## The rules now

- **Keep rule, training:** the student (Qwen3.6-35B-A3B) fully passes **10–60%** of 8 attempts at 300 tool calls, a strong
  model (DeepSeek-V4.1-Flash or Qwen3.5-397B) fully passes at least once, and the audit puts most failures on the skills
  the task is meant to train.
- **Keep rule, benchmark (held-out):** the student fully passes **under 30%**, and a strong model fully passes.
- **Ahmed, 2026-10-23:** change the tasks as needed to get them kept; the **"at least one real link between bugs" rule may
  be dropped** (use `subset --no-link`). Making a too-easy task harder (adding a bug) is allowed for echo and unstorage.
- **Stay close to Ahmed's task:** keep his fixes and his paragraphs word for word where you can. Prefer the smallest
  change: drop one bug, swap one bug, add one bug, or fix one defect.
- **No more dropped tasks** (Uzair, 2026-10-23): a quick pass at 0/3 or 3/3 means change the task again (drop, swap or
  add a bug, fix a defect) and re-measure, never stop. Leaving one of a task's bugs applied is still the same task.
- **Held-out tasks** are never made easier; fix real defects only. A held-out task's own failure traces must not shape a
  training task.
- **No reward hacking, no hinting, no mistakes** (Ahmed): run the checklist below before measuring anything.

## Setup (once)

1. Work on a branch from **`tasks/uzair`**, not main. It has `lh_debug.py subset` (build a task from exact fixes),
   `--no-link`, `scripts/lh_debug_links.py`, and fixes for grading and environment bugs that main does not have yet.
2. `RL-for-Real-World-Environments/.env` (gitignored) needs `TINKER_API_KEY` (ask Ahmed or Uzair, never commit it) and
   `CRUCIBLE_DEBUG_ROOT="<abs path>/crucible/data/debug-<you>"`, your own store.
3. Clone the repo you work on into the store: `git clone https://github.com/<owner>/<repo>.git
   "$CRUCIBLE_DEBUG_ROOT/src/<owner>__<repo>"` (registry names: `lanes/windows.py`, e.g. `"mvdan-sh": "mvdan/sh"`).
4. Ahmed's bug reports for every task, held-out included, are in `crucible/data/debug/prompts/<id>.json`.

Every command below runs from `crucible/` after `set -a && source ../.env && set +a`.

## The loop for one task

1. **Rebuild Ahmed's task from its fixes** (fix lists are in your file) and check the id matches:
   `.venv/bin/python scripts/lh_debug.py subset --repo <repo> --fixes a,b,c,d [--since 2025-09-01] [--b <sha>]`
2. **Build the changed task:**
   `.venv/bin/python scripts/lh_debug.py subset --repo <repo> --fixes <kept fixes> --id <old-id>-e1 --parent <old-id> [--no-link] [--hide revert] [--since …] [--b …]`
   A dropped fix stays applied in the start code. Re-running on an existing id skips the build: delete
   `$CRUCIBLE_DEBUG_ROOT/gen/store/<train|heldout>/tasks/<id>` first.
3. **Write the bug report** `$CRUCIBLE_DEBUG_ROOT/prompts/<new-id>.json`: copy Ahmed's (`builder`, `opening`,
   `requirements`, `closing`), drop the dropped bug's paragraph, and key each paragraph by the **new** milestone id. Match
   paragraphs to milestones by the bug (the `subject` in `window.json`), not by number: with `--no-link` the milestones
   follow commit history, so the numbering can change.
4. **Lint:** `.venv/bin/python scripts/lh_debug.py author` → your task must show `findings=0`.
5. **Cheap gates:** `.venv/bin/python scripts/lh_debug.py gates --ids <new-id> --gates S6,S5,S8` → `ok=True` (~3 min).
   S6 the link (or `--no-link`), S5 the real fix passes and doing nothing fails, S8 cheating scores 0 and every bug's
   own revert fails its milestone. The record is partial: measurement refuses it unless asked (`--cheap-gates`).
   A one-off S8 failure under heavy load can be noise: check with `.venv/bin/python data/debug/dbg_grade.py <id>
   <train|heldout>` (grades the real fix and prints what failed), then `gates --ids <id> --gates S6,S5,S8 --redo`.
6. **Checklist** (below).
7. **Quick pass** (~$5–10): `.venv/bin/python scripts/lh_debug.py intent` then
   `.venv/bin/python scripts/lh_debug_target.py student frontier --tasks <id> --k 3 --frontier "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144" --frontier-k 1 --debug "$CRUCIBLE_DEBUG_ROOT" --out data/targeting/<you>-<task>-quick --audit-reserve 0 --cap 30 --cheap-gates`
   Read `data/targeting/<out>/arms/*/<id>/episodes.jsonl` (`passed`, `milestones`, `turns`, `stop`, `gates`). A variant
   that misses the band is changed again without ever paying for S3/S10.
8. **All gates** before the full run: `.venv/bin/python scripts/lh_debug.py gates --ids <id> --gates S3,S10` (~27 min;
   added to the cheap gates' record; S3 is the World, leak probes and 60 reruns of the existing suite, S10 grades the
   real fix 60 times, since 0 false fails in 60 is the fewest that bounds the rate under 5%). Run it on a quiet host:
   one false fail under load (mvdan-sh 02a801cd-e1 beside a full run) fails S10. To rerun one gate after such a noise
   failure: `--gates S10 --redo` (the failed result stays in the record under `earlier`). Then `lh_debug.py intent`.
9. **Full measurement** (~$20–35) if the quick pass looks in range (needs all five gates; never `--cheap-gates`):
   `.venv/bin/python scripts/lh_debug_target.py student frontier disclose audit mine report --tasks <id> --k 8 --frontier-k 2 --debug "$CRUCIBLE_DEBUG_ROOT" --out data/targeting/<you>-<task>-full --cap 90`
   The verdict is in `data/targeting/<out>/report.json` → `tasks.<id>.keep` (`train`, `bench`, `too_hard`, `too_easy`,
   `unfair`).
10. **Write a task note** like `notes/go-toml-a02114e3.md`: before, what changed, after.

## Checklist before measuring

- [ ] **Hidden tests not visible:** no hidden test name in `base/`; for case-file suites (black's `tests/data/cases/*.py`)
      the fixes' new cases and expected outputs are gone. Use the default/`--hide revert` for black.
- [ ] **No stale assertion:** copy `base/`, overwrite it with `window.json`'s `oracle_files` (the fixed source), run the
      whole visible test suite in the task's World image: everything must pass. A failure means a visible test still
      expects the bug (go-toml e1 had one). Revert mode restores the old version of every test a fix changed. Fix:
      rebuild with `subset --hide-tests <file::Test,...>` naming each stale test function (revert mode then keeps the
      table drivers visible; mvdan-sh's `TestNodeEndPos` with its `// TODO: want 22` offsets). When the stale lines sit
      in a table that a whole driver reads (mvdan-sh's printer rows), hiding the driver, or `--hide strip`, leaves the
      agent blind to its regressions (02a801cd-d1: 0/3, every attempt broke other printer cases): leave that bug applied
      or swap it instead.
- [ ] **No hint:** no changelog/NEWS/docs line in `base/` describing an undone fix (the pipeline now strips release
      changelog entries that cite one; check anyway), no `.git`.
- [ ] **Report matches the tests:** read each paragraph against its hidden tests (`lh_debug.py brief <id>`). A test that
      checks something no paragraph says (urfave-cli's flag order) is a defect: add the missing behaviour in words, never
      a file or function name.
- [ ] **Real link** (only if you did not pass `--no-link`): fix one bug alone with the other still broken and check that
      its *own* test still fails. One test holding both bugs' cases side by side is not a link.

## Gotchas that cost us time or money

- `--cap` must cover spend + projection + `--audit-reserve` (default $25), or arms are silently "not started". Use
  `--audit-reserve 0` for quick passes and a generous cap for full runs.
- Re-running an `--out` with a different `--k` fails with IdentityDrift: use a **new** `--out` for the full measurement
  (quick-pass attempts can't be reused).
- `--debug "$CRUCIBLE_DEBUG_ROOT"` is required; "CRUCIBLE_DEBUG_ROOT is set but Crucible reads no such variable" is harmless.
- If a fix "is not a real fix at B" at today's head, the task was built at an older point: retry with `--b` at the commit
  20/45 days before head (black-90117fa0 only builds at `8d5a2d9f`).
- Overlapping measurements slow every World: add `--slots 5` when two runs overlap.
- zsh: write `"${B}:path"`, not `"$B:path"` (`:t` is a zsh modifier), and zsh does not split `$var` into words in loops.
- To find which fixes in a repo really link (or which are usable at all):
  `.venv/bin/python scripts/lh_debug_links.py <repo> <lang> <since> <days-before-head> [task fix prefixes]`.
- An install rule false positive is known: `which pnpm || npm install -g pnpm` is flagged even when the install never ran.
  It does not change verdicts so far; mention it if you see it cost a full pass.
