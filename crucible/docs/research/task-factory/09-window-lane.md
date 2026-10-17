# M17 task factory, report 09: the window lane (built, measured)

Study 08 §9 steps 1-3: long-horizon coding tasks from real release windows. This report says what was built, where it
lives, what it measured on CPU, and where it departs from 08.

## 1. What was built

| Piece | Where | What it does |
|---|---|---|
| Miner | `factory/lanes/windows_git.py:124-213` | release windows between consecutive version tags (creator date) and commit runs that overlap none; 5-25 qualifying units; cap per repo |
| Trees | `windows_git.py:216` (`RepoGit.build`) | A plus any ordered subset of units through a private index (`git apply --3way`), reverts too; parallel-safe |
| Test owners | `windows_git.py:311` | owner = last unit that changed the test function's AST; unchanged or test-only-unit tests are collateral regression checks |
| World image | `windows_world.py:153` (`ImageBook`) | `crucible-lh-w:<repo>-<packaging hash>` from `crucible-lh-base:py312`: A's own deps (`.[test,tests,testing]`, PEP 735 groups, test requirement names); the project stays installed for its metadata, the workspace is first on `PYTHONPATH` |
| Leak probes | `windows_world.py:260` | executed, `--network none`: `pip download <pkg>==B` fails; no B distribution file; no `def <new test>(` in the image or the seeded workspace; `git log --all` = one commit tagged `base` |
| Policy's World | `windows_world.py:287` | `common/git_repo` App seeded at A on the host; every tool runs by `docker exec` in a `--network none` container that bind-mounts the workspace at the same path, so `WorkspaceHarness` collects from the host |
| F2P / P2P | `windows.py:484` (`f2p_stage`) | whole suite at A (B's tests) and at B, `reruns` each; flaky dropped, P2F dropped; milestones = F2P grouped by owner unit, in history order |
| Dependency prober | `windows.py:296` | A + unit k's source diff on k's own tests; if they fail, ddmin over the earlier units (complements, halves first); edges labelled `executed` / `patch-conflict`; `requires`; `independent` windows stop |
| Revert mutants | `windows.py:403` | B with unit k's source diff reverted (3-way); `valid` / `broad` / `uncaught` / `revert_conflict` |
| Task | `windows.py:820` (`WindowTask(ExecTask)`) | LANE_TASK + accept contract (floors, attacks, tamper, property audit reused), `grade_patch`, `milestone_status`, `phi`, `milestones_of` (horizon replay over unit steps), `open/workspace/broken`, `snapshot_patch`, opt-in `episode_sidecar`, `requirement_ids` / `requirement_milestones` |
| Lane | `windows.py:1002` (`WindowLane(ExecLane)`) | materialize, a grading sandbox per World image, machine handlers (spec+author, realism, oracle, deterministic verifiers) |
| Prompt + ledger | `windows.py:658-818` | author brief (qwen3.7-flash) from messages, changelog and public names; one `R<k>` per milestone; two-way ledger (`cards.RequirementLedger` + unmapped milestones); `lint_prompt`; fixed truthful epilogue; realism review (deepseek-v4-flash) of the maintainer's text |
| Held-out | `windows.py:82`, `WindowStore` (`:106`) | 6 repos fixed before mining; a store has one split (`split.json`) and refuses the other's repos |
| Script | `scripts/lh_windows.py` | `mine`, `report`, `author` (paid smoke), `cleanup` |
| Option B | `executable.py:121-160`, `:1326`, `:1517` | `DockerSandbox.extra_args` (default empty: the executable lane is unchanged; windows pass `--continue-on-collection-errors`); `call_supported` (kwargs a client cannot take are dropped), `lane_ask`; spec cap 8000 -> 16000 plus one compact retry (tests by number, one-line JSON) on `cut off` / `finish=length` |

## 2. Repo pool

- **Bring-up (6):** python-semver, parse, boltons, cachetools, humanize, more-itertools.
- **Medium (12), chosen for work that builds on itself, pure Python, pytest on CPU, permissive licences:** click
  (BSD; options, parsing and help build on one core), marshmallow (MIT; fields, schemas, validators), pyparsing (MIT;
  combinators), lark (MIT; grammar -> parser -> tree), markdown (BSD; extension pipeline), arrow (Apache-2.0; parsing and
  formatting share one token layer), glom (BSD; a spec language over its own core), toolz (BSD), funcy (BSD), cerberus
  (ISC; rule engine), typer (MIT; built on click), dateutil (Apache/BSD; parser, rrule, tz).
- **Held-out, fixed before mining (`HELD_OUT_REPOS`):** H1 (same class) sqlparse, tomlkit, python-dotenv, voluptuous;
  H2 (larger app or another test style) jinja (template compiler), mistune (fixture-driven markdown tests).
- Each one's tests ran in its World image: the funnel's `f2p_valid` column is the executed proof (§3).

## 3. Measured (CPU, local Docker, 2026-10-17)

Funnel per repository: windows recorded -> leak probes pass -> F2P/P2P valid (>= 2 milestones) -> >= 1 executed milestone
edge (`dependent`) -> >= 1 valid commit-revert mutant. Raw: `runs/window_funnel.json`.

| split | repo | windows | leak ok | F2P valid | dependent | mutants valid | milestones (dep.) | milestone edges |
|---|---|---|---|---|---|---|---|---|
| train | arrow | 15 | 15 | 12 | 10 | 10 | 71 | 49 |
| train | boltons | 8 | 8 | 7 | 2 | 2 | 19 | 2 |
| train | cachetools | 6 | 6 | 6 | 5 | 5 | 26 | 18 |
| train | cerberus | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| train | click | 8 | 8 | 4 | 2 | 2 | 26 | 3 |
| train | dateutil | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| train | funcy | 4 | 4 | 4 | 1 | 1 | 7 | 1 |
| train | glom | 3 | 3 | 3 | 3 | 2 | 19 | 25 |
| train | humanize | 6 | 6 | 4 | 2 | 2 | 11 | 4 |
| train | lark | 8 | 8 | 7 | 4 | 4 | 22 | 6 |
| train | markdown | 8 | 8 | 5 | 1 | 1 | 3 | 1 |
| train | marshmallow | 8 | 7 | 6 | 2 | 2 | 15 | 4 |
| train | more-itertools | 16 | 16 | 16 | 9 | 9 | 63 | 28 |
| train | parse | 2 | 2 | 2 | 2 | 2 | 15 | 21 |
| train | pyparsing | 6 | 5 | 4 | 4 | 2 | 21 | 11 |
| train | python-semver | 5 | 5 | 5 | 3 | 3 | 19 | 13 |
| train | toolz | 1 | 1 | 1 | 1 | 1 | 8 | 8 |
| train | typer | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| **train** | **all** | **116** | **105** | **86** | **51** | **48** | | |
| heldout | jinja | 26 | 26 | 8 | 6 | 6 | 53 | 60 |
| heldout | mistune | 20 | 13 | 8 | 2 | 2 | 22 | 5 |
| heldout | python-dotenv | 4 | 4 | 3 | 3 | 3 | 24 | 12 |
| heldout | sqlparse | 16 | 13 | 11 | 6 | 5 | 40 | 16 |
| heldout | tomlkit | 11 | 11 | 5 | 1 | 1 | 6 | 1 |
| heldout | voluptuous | 7 | 7 | 2 | 2 | 2 | 11 | 3 |
| **heldout** | **all** | **84** | **74** | **37** | **20** | **19** | | |

- **Goals:** 51 training windows with >= 1 milestone edge (goal 30), 20 held-out (goal 12); 48 + 19 also have a valid
  revert mutant. typer's 8 windows were never reached (the 2 h deadline); jinja's 18 "no test result at B" are pre-3.0
  releases on Python 3.12 (markupsafe API), markdown's and tomlkit's mismatches are 3.12 removals and a git submodule.
- **Dependent windows (71):** median 7 milestones (max 18), median 2 milestone edges. Milestone probes: 482 pass alone,
  264 minimized, 63 `prefix_fails` (no edge counted). Unit-level edges: 77 `executed`, 722 `patch-conflict` (the unit's
  source diff does not apply without the earlier unit: it edits code that unit wrote).
- **Mutants (the 71 dependent windows):** 264 valid, 18 broad, 23 uncaught, 194 revert conflicts, 2 masked.
- **Leak probes:** every probe of every validated window passes; the failures seen in bring-up were probe defects, fixed
  (a shared word in pip's vendored code; A's own dist-info carrying B's version string).
- **Cost:** 0 USD for mining. Prompt smoke test (`lh_windows.py author`): 9 windows, 8 mapped on the first or second
  try, 1 left with an unmapped milestone (sent back, as designed); realism (maintainer text only) passed 5 of 5; total
  spend 0.008 USD.
- **Wall-clock:** about 2 h of mining on 14 CPUs (up to 18 test containers); a window takes 30 s to 35 min (pyparsing's
  prober).

## 4. Deviations from study 08

1. **Unit = first-parent step**, not every non-merge commit. A merged pull request is one unit (its diff against the first
   parent). Non-merge commits on side branches do not compose onto A in order; first-parent steps always do, so the oracle
   replays exactly and the prober never fights the merge structure.
2. **Prober diffs are source-only** (`.py` outside tests and docs). With every writable file, the changelog line each unit
   appends made units "depend" on each other by patch conflict (first python-semver run, largest window: 24 of 26 unit edges were changelog or adjacency conflicts, 2 executed).
3. **Milestone tests run as unit k wrote them** in the prober (B's test files may import names later units add).
   Grading always uses B's tests.
4. **`--continue-on-collection-errors`** for every window run: without it one module that fails to import stops pytest
   before any test runs, so A looked like "nothing passes" and a revert mutant broke every milestone.
5. **Mutant stage passes with >= 1 `valid` mutant.** `broad` (milestone k fails and so does an independent one: the revert
   left code another unit uses) and `uncaught` (a later unit re-did the work) are reported, not gating; `WindowTask.mutants`
   offers `valid` + `broad` to the verifier probes.
6. **Frozen package mirror**: not built. The World installs A's dependencies at image build time and runs with no network;
   `pip download <pkg>==B` failing and no B file in the image are the executed guards. A mirror matters once a task may
   install packages; none may now.
7. **Flake**: 3 reruns at mining (the brief's default); the 30 x 2 gate stays in the machine.
8. **Realism review sees the maintainer's text only.** Shown the whole prompt, deepseek-v4-flash rejected the fixed
   epilogue (sandbox, STATUS.json, GIVE_UP) as "not something a maintainer writes" on the first smoke window; 08 §3 asks
   for that text, so it is outside the review.
9. **Option B (b), the trial adapter bug.** The trial's journals no longer show the TypeError (the verifier state ended on
   unknown test ids). The fix covers the failure the note describes: every lane call to a client goes through
   `call_supported`, and `tests/unit/test_m17_w_optionb.py` drives the spec state at attempt 2 (which adds `sample_key`)
   with a client that has no `**kwargs`.

## 5. Open

- Windows `independent` by the executed test are the majority; the dependent ones are the Type A pool.
- Prompts were authored for a few windows only (the smoke test); the per-window machine has not run on windows.
- `requires` of `prefix_fails` milestones is conservative (every earlier milestone) and is not counted as an edge.

## 6. Changes after the first run (report 11, run 2)

| # | change | where |
|---|---|---|
| F1 | **Type A outcome reward** = `phi_weight * Phi(end state) + (1 - phi_weight) * [every gating check passes]`, default 0.5, configurable (`WindowLane(phi_weight=)`, `WindowTask(phi_weight=)`; 0 = run 1's binary reward). Phi is an end-state check (hidden milestone tests, prerequisite-gated), so the reward still grades outcomes only; an all-fail group keeps a spread when end states differ. Applied on the RL path (`grade_patch`, what `WorkspaceHarness` scores); `grade` stays the binary verdict the accept contract reads (floors 0, a revert mutant is killed only at 0, flake). Turn credit stays ΔΦ through `progress.jsonl` | `windows.py` `WindowTask.__init__`, `grade` (`meta.outcome_reward`, `full_pass`), `grade_patch` |
| F2 | **Only behavioural edges gate Phi.** An `executed` edge (without unit j, unit k's diff applies and k's tests fail) goes in `requires`; a `patch-conflict` edge (k's diff edits code j wrote) is an `order_hint`, metadata only. A `prefix_fails` milestone gets no `requires` (nothing was measured). The prober now runs B's test files, as the grader does | `probe_dependencies` |
| F3 | **Test ownership by prefix**: the owner of a fail-to-pass test is the first unit after whose prefix A + u1..uk the test passes under B's tests. Prefix files are read from unit k's commit (units are first-parent steps), never re-applied. ≤ 25 runs per window: the last prefix first, then the candidate that splits the most unresolved intervals; units that change only prose or only tests are not candidates. A test that never passes on a prefix is a collateral regression check | `candidate_units`, `bisect_owners`, `prefix_owners`, `milestones_from_owners`; `f2p_stage` |
| D1 | Prompt lint: editing verbs ("without modifying") are a process instruction only when their object is the work environment (tests, conftest, CI, packaging, git, history) | `LINT_RULES` |
| — | No regression requirement in the prompt when the window has no regression tests (it was an unchecked requirement) | `author_window` |
| F7 | `open_world(name_prefix=)` (always `crucible-lh-*`), `WindowLane.world_prefix`: a caller finds exactly its own World containers | `windows_world.py`, `test_m17_w_world.py` |

The tool-call limit note and the final STATUS.json turn (F5) are an opt-in of `actors.policy.LLMToolPolicy`
(`tool_limit`, `limit_warning`, `status_file`), not of the lane.
