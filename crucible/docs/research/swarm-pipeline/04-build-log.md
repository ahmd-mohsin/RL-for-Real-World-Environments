# Swarm pipeline, build log

## P1 (built): machine v2, agentic sessions, typed hand-offs, the task repo, receipts

Spec: `00` §3, §4, §13, §15 P1; Prime Agent at 7cbffa77b (`01` facts 4-10). $0 spend: every model is scripted.

**Files.** `factory/machine.py` (extended), `factory/sessions.py`, `factory/handoff.py`, `factory/receipts.py` (new);
tests `tests/unit/test_m17_p1_{machine,handoff,sessions,receipts}.py` (42); `THIRD_PARTY.md` rows (Prime 7cbffa77b PORT,
MIT; receipts and GitSwarm as PATTERN).

**Machine v2** (`machine.py`).
- **Infra vs task.** `is_infra_error` ports Prime's two markers (`F:1095-1116`) and adds our runner's errors (infra-blamed
  failure kinds, lost containers, provider outages). After the quiet period (injected `sleep`), `infra_probe(state, err)`
  answers `alive` (re-run the same session) or `dead` (the zombie is removed first). The state then re-runs once for free:
  same attempt number, no retry, no entry, the dead attempt's spend row kept. A second death takes the retry path. Infra
  that outlives the retries pauses as kind `infra` (V8: masked); `fail_fast` does not apply to it.
- **Capture ownership.** New port type `artifact` = `{path, sha256}` under `artifact_root` (the task repo).
  - A missing port, or an artifact missing or hashing wrong after one re-capture window, fails the producer; its outputs
    stay in `needs_verify`.
  - A consumer whose bound artifact vanished or changed pauses as `capture_failed` on the producer; `resume` re-runs it.
  - A hand-off that breaks a port schema is handled the same way. A port's own `schema` is checked at the producer's settle.
- **Recovery posture.** `mutates: true` requires `recovery: reconcile | reset`, and a writing `session` requires
  `mutates`; the validator refuses both omissions.
  - The run's `trees(state)` gives a `MutableTree`; one snapshot per entry, journaled.
  - Every re-run (retry, free respawn, resumed failed entry, crash replay) recovers first. `reset` restores the
    snapshot; `reconcile` puts `{diff, findings}` in `ctx.recovery`, and the session brief prints both.
- **V7.** `test_exhausted_review_loop_escalates_where_prime_ends_done` runs Prime's pr-manager shape. It pauses
  `exhausted` where Prime's `TF:5450-5477` pins `done`.
- **Budgets.** `run.budget_tokens` and per-state `budget_usd` / `budget_tokens` (all attempts and entries), next to
  `run.budget_usd`. Tokens are booked with `book_tokens` or `Settle.tokens`; `ctx.budget` shows what is left.
  - A state over its cap fails with no retry and pauses as `state_budget`; `resume` grants `extra_usd` / `extra_tokens`.
  - v1 machines keep their hash: the P1 keys are emitted only when set, so their journals still resume.

**Sessions** (`sessions.py`). A `session` state is an M16 `RLMPolicy`:
- **Worktree:** a private clone holding only the source's `base` commit. A `git worktree` would share objects and refs,
  so a verifier's `git log --all` would list the oracle's commits (V2).
- **Sandbox:** one `crucible-p1-*` container, `--network none`, worktree and read-only `/handoff` mounted. The kernel
  runs inside it via `ExecLauncher`; the tools are the World's `GitRepoApp` (the windows lane's `open_world` pattern).
- **Hand-off in, files out:** captures (`json text file dir patch commits`) committed with `informed_by=<hand-off>`.
- **Failure split.** A dead sandbox and infra- or harness-blamed kinds raise `InfraError`; give-up and budget stops
  are `StateFailed`. Solver exposure probes (e.g. writing `tests/hidden`) are recorded, never fatal, in builder sessions.
- **Specialists.** `session.specialist` (e.g. `world_expert:{repo_name}`) resumes by id across runs: same ledger, a new
  `start` window (`kind=specialist_session`), a `[memory]` note in the brief.

**Hand-offs and the task repo** (`handoff.py`).
- **Hand-offs.** Each `Handoff` holds every input with its schema and the full original spec (`params.spec`) with its
  sha256. A missing spec or a summary is refused. Every hand-off is stored in the task repo. `attribute()` returns
  `handoff_loss` (port hash differs from the settle, schema error, spec not original), `grader_error` (re-grade
  disagrees) or `actor_error`.
- **Task repo** (`TaskRepo`, GitSwarm PATTERN): one git repo per family. Commits are immutable (a rewrite raises
  `ImmutableError`) and carry `informed_by=` lineage.
  In-flight announcements carry scope and abort criteria (an overlap raises `DuplicateWork`) and expire; abstention is
  free; the 40% herding guard counts from 5 accepted tasks. Machine handlers: `lease_state` (S0), `accept_state` (S17).

**Receipts** (`receipts.py`). `run_receipt` records versions, image digests, seeds, spend, per-state outcomes from the
journal, `not_verified` and a sha256. The seeded-defect run plants these defects through fakes:
- **Task defects, each caught at its state:** leaky World at S3, unsolvable at S5, hinting prompt at S7, flaky test at
  S10 (bound 0.30 vs 0.049 clean), reachable grader at S12, mislabelled cause at S14 (`audit` seeded controls).
- **Machine-layer defects:** dead worker, missing capture, vanished capture, unbounded loop, no recovery posture,
  state over budget.
- **Result:** the clean control passes every gate; 13/13 cases ok, `check_receipt` = []; the twin tests (9 mutations
  plus an un-rehashed edit) are all rejected.

**Executed.**
- **Docker e2e** (scripted models, real `crucible-p1-*` containers, about 13 s): lease → planner → World expert → oracle
  builder → verifier writer → review loop (reject, then approve) → accept = `done`.
  - The oracle's first attempt left no commit, so the producer failed; the retry reconciled from the diff.
  - The verifier's store has no oracle commit, and its hand-offs carry no oracle canary.
  - The exhaustion variant pauses `exhausted`; the World expert resumed by id; a third run on a live scope abstains; no
    container was left behind.
- **Infra death:** a second Docker test kills the sandbox mid-session. Result: `infra_respawn` verdict `dead`, a reset
  recovery, a fresh container, `done`, no task failure. m17 suite 386 passed; `ruff`, `lint-imports` (4 kept) clean.

**Deviations.**
1. The herding guard and lease are machine handlers, not yet wired into `root.RootCurator` (root.py untouched).
2. The session kernel shares its state's sandbox (one container: kernel, shell and tools). It is not a separate
   `crucible-rlm-*` container.
3. Ignored files are not snapshotted; `read_only` sessions get a fresh clone per attempt instead of a recovery posture.
4. The task-defect detectors are fakes for the production gates (P2-P5 wire those in); `not_verified` says so.
5. The image `crucible-p1-session:py312` (python:3.12-slim plus git) is built by `ensure_image()` and kept.

## P0 (built): the LongTask contract, path checks, import-tolerant grading, D7, D8

Spec: `00` §1, §5, §6, §15 P0; report 11 (D7, D8, F8). $0 spend: no model call.

**Files.** New: `factory/longtask.py`, `factory/pathcheck.py`, `factory/lanes/windows_imports.py`; tests
`tests/unit/test_m17_p0_{longtask,pathcheck,imports,run}.py` (48). Changed: `factory/lanes/windows.py`, `scripts/lh_run.py`,
and two existing tests whose asserted rule P0 changes on purpose (`test_m17_w_task`, `test_m17_w_world`: see below).

**LongTask** (`longtask.py`). A runtime-checkable Protocol with every §1 member, plus:
- `check_longtask`: members, async methods, the World, the milestone DAG with cycles and executed edges, the two-way ledger, the card;
- `ground_truth`: oracle, alternatives = 1; floors = 0 with Phi 0; each mutant fails its milestone;
- `null_agent_score` (S8) and `clean_world_solvable` (S5, optionally through a real World round trip);
- `LongTaskRef` for serialisation.

Deviation: the oracle is the accept contract's `oracle` *attribute* plus `oracle_patch()`, not a method `oracle()`. A
method of that name would break every accept check.

`WindowTask` conforms. It adds `world`, `prompt`, `path_rules`, `oracle_patch`, `floors` (no_op, claims_only,
test_edit, pass_stub), `alternatives` (unit steps), `mutants` with their milestone, `simplify`, `card` and `ref`.
`simplify('prerequisites')` builds A + the *executed* prerequisite units (source only; the prefix when they do not apply
alone) and keeps the remaining milestones, requirements and mutants.

**Path checks** (`pathcheck.py`). Inputs are the diff before restore and the World's own `trace.json`, found next to the
grade dir or in the latest earlier attempt for a re-grade. Shell commands are parsed so the program actually run decides:
`grep curl` is not a network attempt, and a redirect's fd is not a package.

| gate | severity | consequence |
|---|---|---|
| `path:protected` (test, config edits) | gate | no full-pass bonus |
| `path:install` (other packages, requirements) | gate | no full-pass bonus |
| `path:required:*`, `path:steps` | gate | no full-pass bonus |
| `path:plumbing` (conftest, sitecustomize, .pth) | hack | reward 0, `hack_suspect` |
| `path:network` (curl, git clone or fetch, pip download, installing the project itself) | hack | reward 0, `hack_suspect` |
| `path:history` (`git log --all`, reflog, `.git/objects`, grader files) | hack | reward 0, `hack_suspect` |

- **Decision.** Phi is kept for gate violations (protected paths are restored before the hidden tests run, V4); hack
  violations zero the reward. No trail: trail checks unmeasured (listed, never gating). `grade()` is unchanged (the
  accept contract's binary verdict). The epilogue now states these rules (V5).

On 439 recorded run-1/2 trails, `fetch` fired on 10 of 404 solver episodes and 1 of 35 hacker episodes; every one was a
real attempt to get the release (git clone of upstream, `pip download glom==21.1.0`, `pip install more-itertools==10.0.0`).
`git log --all` fired 3 times. Installs fired on 21 solvers (gate only).

**F8.** `windows_imports.tolerant_source` guards each module-level project import per name in the grader's copy of B's
tests. A missing name becomes a placeholder that raises a `BaseException` on use, so a test that never touches it runs as
usual. `parametrize` ids are unchanged.

Every run of a window (f2p, ownership, prober, mutants, grading) uses `grader_tests(rec)` in the window's own mode: new
windows are mined tolerant; stored windows have no flag and stay strict. Measured reason: forcing tolerance on
strict-mined glom-wf853c3e and lark-w293bf07 lets the start state pass a milestone.

**D7.** `EpisodeCfg.max_tool_output` 8,000 -> 32,000 (>= the World's 30,000; `--max-tool-output`). **D8.**
`findings_from_tests` renders only hidden test ids into findings from the test's own asserts, `pytest.raises` and
parameters; free text is refused (`lh_run --author-findings`, `sent_back_refused` in the journal).

**Executed verification** (`lhwork/store/train`, 51 dependent windows, read-only; 542 `grade()` calls; script in scratch).
- **Re-grade identical to stored:** 185/185 mutant milestone vectors match mining; run-1 journal floors match exactly
  (11 windows: 6 grades + mutants each); the pre-P0 code gives the same results on every anomaly below.
- **Pre-existing anomalies P0 now surfaces:**
  - cachetools ×5 and markdown: the oracle has Phi 1 but fails its regression check (p2p ids `tests/__init__.py::...` do
    not reproduce);
  - arrow-wc856ab9: unit-step oracle does not apply (`.github` file exists);
  - marshmallow-wdc3bab4 and pyparsing-w7d4bda2: the pass-stub passes one milestone (Phi 0 through `requires`).
- **RL path:** null agent 51/51; clean World 45/51 (the 6 regression-check windows above); test_edit floor 0 on 51/51.
- **In a real World** (5 authored windows): oracle round trip 5/5, collected patch byte-identical.
- **Conformance:** 9/51 conform; 27 short, 15 with patch-conflict `requires`. This is the pre-F2 store; store2 is the
  re-mined one.
- **F8 re-classification** (16 windows with late module-level imports, tolerant f2p + ownership + prober, in memory):
  - owned tests 738 -> 3,788, mostly strict collateral now owned per unit;
  - 282 owners moved and 18 tests became regression checks;
  - milestones 114 -> 112;
  - 9/16 stay dependent on executed edges.
- **Tolerant re-grade** changes 8 of 53 mutant vectors: a revert no longer fails unrelated milestones through the import.
- **simplify** on 5 authored windows: ground truth holds on 4. humanize fails its floors because a pre-F3 owner passes on
  the prefix; the S8 re-run catches it.

**Open.** Re-mine both stores tolerant (store2 plus F8) before P2; the regression-check id defect in cachetools/markdown.
- Grading containers can now be named (`sandbox_factory(name_prefix=)`); World containers stay `crucible-lh-*` (F7).
