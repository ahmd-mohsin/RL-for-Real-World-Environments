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

## P3 (built): the RLM trace-audit loop, executed acceptance, versions and objectives

Spec: `00` §8 (8.2-8.6), §11.3 item 6, §0 V6/V12, §15 P3. Spend $0.20 of the $5 cap, all auditor (deepseek-v4-flash).

**Files.** New: `factory/traceaudit.py`, `factory/versioning.py`, `scripts/lh_audit.py`; `factory/audit.py` extended; tests
`tests/unit/test_m17_p3_{traceaudit,auditor,versioning,seeded}.py` (30; the seeded ones `slow`, one also `docker`).

**The loop** (`traceaudit.run_loop`) over an `EpisodeBundle` (a runner attempt dir read whole: turns, tool calls with results,
`progress.jsonl` + `checkpoints.jsonl` + checkpoint patches, final diff, GradeReport and grader tail, STATUS claims; path checks
recomputed from `trace.json` when the report predates P0).
1. **Facts, no judge:** masked -> `env_flake`, limit stop -> `budget`, path violations and near misses, reply-before-action,
   World faults (a project module the World's own tool cannot import; a missing module or command; a tool name typed as a
   command is misuse, not a fault), harness misuse, empty diff.
2. **First wrong step:** `lost` (Phi never regains its peak; refined to the tool call that edited a file changed between the two
   checkpoints), `plateau`, `flat`, or `divergence` from passing episodes of the task (an edit outside every passing end state,
   or no edit of their files by their median first gain). Step, turn, position in the episode.
3. **Cause label:** `RLMAuditor`, an M16 `RLMPolicy` in a `crucible-p3-audit-*` container whose `CONTEXT` is the rendered bundle;
   `audit.validate` plus a rule that the quote must come from the attempt, not the factory's `[facts]` lines. Seeded controls in
   every batch; an untrusted batch's labels never make proposals. Same-family judging raises (`AuditConfig.forbid_same_family`).
   10% human kappa queue (`audit.human_queue`, `kappa_vs_human`). Auditor notes change only via `AuditorHarness.refine` ->
   `HarnessState.evaluate`.
4. **Proposals** with artifacts: `verifier_gap` (pass audit: short path vs passing peers or the oracle's changed lines, files
   outside the oracle's, near misses; or a trusted label "passes but incorrect"), `false_fail` (trusted label), `spec_defect`
   (a name the grader says is missing and the prompt the trace saw never states), `world_bug` (a World-fault fact),
   `intent_drift` (>= 3 off-intent traces from >= 2 families, `intent_candidates.jsonl`, `awaiting_human`, never applied).

**Executed acceptance** (§8.3, ports as Protocols: `VerifierPort`, `AlternativesPort`, `SecondVerifierPort`, `WorldPort`; P2's
alternatives and second verifier are faked behind them until P2 lands). `verifier_gap`: demo passes the current verifier and
fails the new check; the check passes the oracle and every alternative; mutants and floors still fail; 60 oracle reruns at
concurrency 1 and 4, Clopper-Pearson <= 5%. `false_fail`: second verifier, relaxed verdict on demo and oracle, mutants and
floors still fail. `spec_defect`: the name is in the failing hidden test's own source as a project name (imports, attribute of
a project module or name), is absent from the start tree, the edit is test-derived (`findings_from_tests`), and the amended
prompt adds no lint or ledger problem; `counterfactual_rerun_needed` is set (P4). `world_bug`: the reproduction (pagers cut,
the World's paths rewritten) fails on the oracle state with the same fault signature, the fix makes it pass, S3 passes again.

**Versions** (`versioning.py`): append-only, hash-chained `VersionStore`; a derived version needs the head as parent, a proposal
id and the evidence hash; task versions are per published prompt (`task_subject`). `TrainingObjective` pins task -> verifier
version (`objective_id`), refuses rewards from another version, `after_change` resumes only from a checkpoint before the change
(`run@step`) or starts `run+obj`; `curve`/`merge_rewards` refuse mixed objectives; `regrade_offline` rows are reporting-only.

**Exit criterion (executed).** `lh_audit.py seeded`: a small durations repo, hidden tests, oracle, two alternatives (one the
faked P2 rewrite), two revert mutants, two floors, all graded by pytest in `crucible-p3-grade-*` containers; six scripted
episodes whose shell results were executed in the World. Lucky pass -> `verifier_gap` accepted (demo pass -> check fail,
oracle and both alternatives pass, mutants and floors fail, flake 0/60, bound 0.049) -> verifier version. Unstated
`format_duration` -> `spec_defect` accepted (grounded in `test_format`/`test_round_trip`, lint and ledger clean, counterfactual
due) -> task version. Broken `tools/run_tests.sh` (`python -I`) -> `world_bug` accepted (fails on the oracle, passes with the
fix, S3 5/5 clean, leak probes hold) -> task version. Fake check (`'90m'` must raise) -> rejected: fails the oracle. Same
outcome locally and in containers (about 2 min). The RLM auditor (3/3 controls) called the lucky pass correct: the executed
pass audit, not the judge, caught it (V6).

**Real run** over the 439 stored run-1/2 trails (`lh_audit.py real`; 120 labelled, stratified by window x family).
- 364 failing, 15 passing (horizon iso tasks), 60 masked. First wrong step: flat 191, plateau 76, divergence 53, lost 15,
  unmeasured 29 (RLM runs without checkpoints). Median position: lost 0.67, plateau 0.67, divergence 0.36 of the episode.
- Facts (episodes): no edit 92, reply-before-action 91, limit stop 81, near miss 44, path violation 36, World fault 17.
- Causes: deterministic budget 81, env_flake 60; trusted labels skill_gap 39, spec_defect 4, budget 4, unstated_format 1
  (skill-gap share 0.81). Batch 3 was untrusted (`ctl-lazy` unlabelled) and its 30 labels were dropped.
- Proposals: 15 spec defects, 1 accepted (`isvalid`, semver), 13 rejected (8 names already exist at the start: the agent's
  breakage, e.g. `parse`, `search`, `VersionInfo`, `bump_build`; 5 not in the failing tests' own source), 1 needs input (horizon prompt in no
  journal). 7 World bugs reproduced on the oracle in the World image and left `reproduced_needs_fix`: setuptools missing (3
  windows), `make` (2), `hexdump`, PyYAML (glom: 2 of the repo's own tests fail on the oracle in the World). Pass audit: 0 of 15
  passes flagged. 17 intent candidates for a human (largest: flat + no edit, 53 traces, 5 families).

**Deviations / open.**
1. 44 of 120 auditor sessions hit the 24-turn cap; 39 gave no verdict (rejected, never guessed). A forced final-verdict turn
   or a tighter brief is next; it is an auditor-harness edit, so it goes through `AuditorHarness.refine`.
2. No `VerifierPort` for windows or horizon iso tasks yet: label-made `verifier_gap`/`false_fail` stay `needs_input`.
3. Window World fixes and their S3 re-run belong to the World expert; P3 reproduces and stops at `reproduced_needs_fix`.
4. The intent-drift clusters are mostly unlabelled signatures; a human reads them before any intent changes.

## P2 (built): second verifier, shadow tests, alternatives, fairness lint, gold set per family

Spec: `00` §0 V2-V6, §1, §4 S5/S8/S9, §5.4-5.8, §15 P2. Spend $0.445 of the $5 cap (one USD ledger, `lhwork/p2/spend.jsonl`):
author (qwen) 0.015, second verifier (deepseek) 0.174, rewriters qwen 0.049 / mistral 0.038 / deepseek 0.168.

**Files.** New: `factory/verifier2.py`, `factory/alternatives.py`, `factory/fairness.py`, `scripts/lh_p2.py`. Extended:
`factory/goldset.py`, `lanes/windows.py` (`normalize_ids`, `volatile_ids`, `remine`), `lanes/windows_world.py` (leak-probe
prefix), `scripts/lh_windows.py` (`--store`, `--from-store`, prefixes). Tests `test_m17_p2_{seeded,verifier2,alternatives,
goldset,fairness,remine}.py` (24).

**Re-mine** (P0 open item). `remine` re-validates exactly store2's windows, import-tolerant, into `store3`.
- **Test ids.** xunit1's `file` names the file that *defines* an inherited test (`tests/__init__.py`,
  `markdown/test_tools.py`). `normalize_ids` keys it by the file that ran it (executed test). No store3 id keeps the old form.
- **Funnel.** Train: 124 windows (store2's 122, plus 2 from a fresh mine) → 123 World → 123 leak → 95 f2p → **28 dependent**.
  Against store2's 28: 24 kept, 3 gained, 4 now independent. Held-out: 84 → 74 → 74 → 40 → **11** (store2: 10), mined
  and graded only.
- **New defect, fixed forward.** marshmallow parametrizes on `datetime.now()`. Concurrent reruns share the second, so two
  volatile ids became regression checks, and S5 rejects the window. `f2p_stage` now drops ids a later oracle run lacks.

**Second verifier** (`verifier2`).
- **Mode.** deepseek writes one pytest module per requirement from the maintainer text, the requirements and A only: the
  file list, the README head, and the modules the request names (or an outline). `leaks` refuses a payload line that exists
  only in B, or a hidden test name. It is one call plus one syntax repair, retried in groups of 6 → 3 → 2.
  - Not a session, because of cost: about 20 k tokens a window against 0.5-1 M. `shadow_session_state` validates but is not run.
- **Qualification.** A test is *sound* when it passes the oracle and every accepted alternative. A requirement *gates*
  only when its sound tests fail the start state and every mutant of its milestone; otherwise it flags.
  - 858 tests on 208 requirements: 336 fail the oracle (confirmed verifier-2 defects), 0 fail an alternative. 60 gate.
- **Differential.** 3,406 (end state, requirement) pairs, 1,180 disagreements, each owned by rule (`assign_owner`).
  - Truth known (oracle, alternatives, floors, start; a mutant by its executed `expected`, a `broad` one only on its own
    milestone): verifier 1 has 0 confirmed defects; verifier 2 has 293 weak and 343 unsound.
  - Candidates for the P3 audit: spec `unstated_format` 75, verifier-1 `verifier_gap` 6, false fail 39 (rewrites the
    real tests fail and verifier 2 passes).

**Alternatives** (`alternatives`).
- **Rewrites.** qwen (imperative), mistral and deepseek (functional) rewrite the changed functions; each is spliced back
  with its name, parameters and decorators kept. A rewrite the real tests fail is **salvaged**: chunks of its definitions
  are kept while the tests pass, with no model call. Acceptance needs the real tests and verifier 2.
- **Results.** 99 candidates, 57 accepted; 22 of 27 windows have ≥ 2 with ≥ 1 outside the author family. glm-4.7-flash
  (empty replies at 16 k) was replaced. No recorded solver run passed: 0 solver alternatives.

**Fairness** (`fairness`).
- **The lint** refuses checks that read source text (`getsource`, `.py` reads, grep) or structure (`__code__`,
  `co_varnames`, `ast.parse`, `dis`), private-name checks and no-requirement checks. It flags unstated free-text equality
  as `unstated_format` (an echoed input is exempt). On the real hidden tests: 435 `unstated_format`, 0 refused.
- **Executed.** `ten_competent_people`, `admit_check` (§8.3) and `recheck_false_fail`. One gating check fails a valid end
  state (marshmallow).

**Gold set** (`goldset`).
- Sources `oracle` and `floor` are added. `executed_label` (`miner:executed`: B's whole suite with strict imports) counts
  only where verifier 2 agrees.
- Also `class_bounds` (0 errors: 35 items give ≤ 10%, 73 give ≤ 5%), `pair_fracs`, `spot_check_queue` (10% to confirm,
  plus every disputed real item to decide) and `verifier_card`.

| family `window` (27 windows, 448 items) | n | errors | Wilson upper |
|---|---|---|---|
| truly failing (false pass) | 342 | 0 | 1.1% (≤ 5% met) |
| truly passing (false fail) | 84 | 1, the marshmallow oracle | 6.4% (not met) |
| milestone unit: failing / passing | 740 / 1,445 | 2 (click pass-stub floor) / 0 | 0.98% / 0.27% |

- **Composition.** 167 real failing (145 executed labels, 22 disputed), 116 mutants, 57 alternatives, 27 oracles, 81
  floors, 0 confirmed hacks; real share 37% (spec ≥ 50%); queue 65 rows. The truly-passing class only partly tests
  verifier 1 (an alternative is accepted only when verifier 1 passes it). Per-repo cards are under-powered (best 6.0%).

**Machine** (`gate_handlers`, `gate_machine`: S5 → S8 → S9, loop to S5, then escalate): of 27 windows, 20 `done`, 6
paused at S8 (< 2 alternatives, or click's pass-stub floor passes a milestone), 1 rejected at S5 (marshmallow).

**Seeded gaps.**
- **Toy window:**
  - (i) a `getsource` name check: the lint refuses it, and an alternative fails it;
  - (ii) `callable(gamma)` in place of the behaviour: a behaviour mutant passes it, which the shadow makes a confirmed
    verifier-1 `verifier_gap`;
  - (iii) a monkeypatch check: the lint passes it, and `ten_competent_people` / `admit_check` refuse it.
- **Real windows** (boltons, click, funcy, cachetools):
  - (i) lint 4/4; execution 0/4 (the name still appears in the source);
  - (ii) 3/3, a confirmed `verifier_gap` at the revert mutant and at the start;
  - (iii) a `co_varnames` pin refused 3/3 (the cachetools seed was invalid: a closure name).

**Deviations.** (1) the second verifier ran as a call, not a session; (2) no judge, so no κ; (3) `WindowTask.alternatives()`
is unchanged (accepted alternatives are in `lhwork/p2/windows/*`); (4) more-itertools-wd23124d has no shadow suite (3 replies
cut off); (5) DESIGN.md is left to the merge.
