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

## P4 (built): World fixes, failure targeting, S15

Spec: `00` §7, §4 S3/S15, §1 `simplify`, §15 P4. Spend $2.05 of the $3 cap (one USD ledger, `p4/run/spend.jsonl`): solver arms
1.27, seeded arms 0.37, paraphrase arms 0.42, paraphrase author 0.0014, probe episodes 0.0012. Target rung qwen3.7-flash.

**Files.** New: `factory/targeting.py`, `factory/lanes/windows_s3.py`, `scripts/lh_target.py`; extended: `lanes/windows_world.py`
(recipe 3, `open_world(workspace_mode=)`), `lanes/windows.py` (`simplify('locate')`, `target_milestones`, `changed_defs`,
policy-World overrides). Tests `test_m17_p4_{targeting,world}.py` (15). Reports `runs/p4_report.json`, `runs/p4_world_versions.jsonl`.

**Part A, World fixes (S3, World expert).** Recipe 3 (`world_needs`) adds, each with the file that asked for it: a developer
baseline (make, bsdextrautils for hexdump, xxd, patch, file), setuptools and wheel, the commands the repo's Makefile, tox
`commands`, CI `run:` steps, nox and scripts run (`TOOL_APT`; humanize's translation script gave `gettext`), every non-doc
extra, and optional imports with a known provider (`MODULE_DIST`). Unknown commands or imports are recorded, never installed.
- **Version search** (`repair_versions`): A's own tests failing in the World at A = a World bug. Rank the modules on the
  failing lines, map them to installed distributions, try releases from A's pin upward as one-line layers, keep the first
  that fixes them with no new failure. Blind pin-first was tried and dropped: glom's PyYAML 3.13 imports but calls
  `collections.Hashable` on 3.12.
- **S3** (`world_s3`): builds; A's suite runs; leak probes; 60 reruns (30 at concurrency 1, 30 at 4), CP bound <= 5%; the
  verifier does not move (oracle passes every milestone, start passes none).
- **Acceptance** through `traceaudit.accept_world_bug` with `WindowWorldPort`, then `version_accepted` (`world_fix`
  versions; `world_of_version` gives consumers the head World). The version chain verifies.

| window | P3 signature | before (current World) | after (recipe 3) | result |
|---|---|---|---|---|
| parse-w07c2ff9, parse-wc669f87 | `make` not found | repro exit 127 | make present (the agent's `make clean` now fails on its own: no Makefile) | accepted |
| more-itertools-w89d58c2, toolz, humanize | setuptools missing | repro fails | `setup.py` runs | accepted |
| python-semver-w1bafb9f | `hexdump` not found | fails | passes | accepted |
| glom-w67f423c | yaml missing | already fixed in store3's recipe-2 World | rejected (no repro on the oracle) | |
| glom-w67f423c (S3-sourced) | A's `test_yaml_target` fails: PyYAML 6 `load()` needs `Loader` | 72 pass / 1 fail | PyYAML 5.1.2: 73 pass | accepted |
| toolz (S3-sourced) | 5 A tests fail (3.12 importlib, builtin introspection) | | no release repairs it | rejected (needs an older interpreter) |

S3 before/after: 60/60 clean reruns, leak probes hold, verifier unchanged on all 7 Worlds.

**Part B, targeting** (`targeting.py`). `SIMPLIFY_KINDS` per generator plus the `Simplifiable` Protocol. Windows support
`prerequisites` and `locate` (the prompt names the target unit's source files and changed functions). Stitch, rebreak and
open plug in at P5.
- **Counterfactual**: the unit is the target milestone with its `requires`. One-sided Fisher test on success. A bootstrap
  CI on the target-set Φ gain counts only when that Φ is graded. Verdicts:
  - `not_attributed`: simplified success significantly below p_ref = 0.8;
  - `underpowered`: neither test significant, flagged with the power at the reference effect.
- **Prerequisite probe**: a scripted World check (suite through `repo.shell` under the verifier's and the repo's own
  invocation; write and remove a file) plus a 12-tool qwen episode that must run tests and report a count its run printed.
- **Paraphrase**: two paraphrases (fixed sentences and epilogue kept; ids, milestones, ledger, lint and required names
  re-checked). Stable when the pass rate is inside the original's Wilson CI and Φ inside its bootstrap CI.
- **Band, cause mix**: band from F1 reward spread; cause mix from the P3 auditor's per-window labels (session ledgers).
- **S15**: `s15_target` + `target_machine` (S2 <-> S15, bounded; a never-fixed loop pauses `exhausted`, V7; World-owned
  probe failure drops).

**Exit run** (store3 train, 4 attempts per arm, tool limit 150, checkpoints off). The target is the deepest
executed-edge milestone; the simplified variant pre-applies a prefix.

| window (target) | full | simplified | Fisher p | verdict | band | cause (P3) | probe | S15 |
|---|---|---|---|---|---|---|---|---|
| **seeded** parse (m17): pytest removed + read-only workspace | 0/3 | 0/4 | 1.0 (P(X<=0 \| 0.8)=0.0016) | **not_attributed** | | | fails (World) | drop |
| parse-w07c2ff9 (m17) | 0/4 | 0/4 | 1.0 (0.0016) | not_attributed | yes | 6/6 skill | ok | S2 (exhausted) |
| glom-w67f423c (m47) | 0/4 | 1/4 | 0.50 (0.027) | not_attributed | yes | 0.50 | ok | S2 |
| humanize-wf9fa838 (m08) | 0/4 | 1/4 | 0.50 (0.027) | not_attributed | no (all 0.083) | 6/6 | ok | S2 |
| python-semver-w41775dd (m31) | 0/4 | 3/4 | 0.071; ΔΦ CI [0.25, 1.0] | **underpowered** (power 0.41) | yes | 6/7 | ok | accepted, flagged |

Paraphrases of parse are stable: Φ 0.275 and 0.325 inside [0.05, 0.325]; pass 0 inside [0, 0.49]. Only 4/4 vs 0/4 is
significant at n = 4. On the kept milestones Φ rises (parse 0.25 -> 0.50, semver 0 -> 0.375), but the targets do not
move: the policy's failures on three windows are not the prerequisite axis.

**Defects found and fixed while running.**
1. The first verdicts compared Φ over every kept milestone and attributed parse on non-target gains. Fixed: target set only.
2. With one target, a bootstrap of a binary at n = 4 overruled Fisher on semver. Fixed: graded Φ only.
3. The probe judged qwen's correct doctest count (the repo's tox command) wrong. Fixed: the repo-config count, or the
   count its own run printed.
4. S15 machine runs resumed an older journal. Fixed: run id per evidence hash.
5. At concurrency 12 the forge ledger's in-flight holds hit the cap at $0.85 spent. 8 attempts were masked
   `credits_exhausted` (V8), stayed owed, and resumed at concurrency 6.

**Deviations / open.**
1. n = 4 per arm detects only 4/4 vs 0/4 (power 0.41 at p_ref 0.8). Results are verdicts with numbers, not acceptance.
2. The cause mix uses P3 labels on run-1/2 trails (store2 prompts). One untrusted P3 batch cannot be split per trace.
3. `locate` was built and unit-tested, not run on real windows (budget). Seeded "test command hidden" was run only
   combined with a read-only workspace, so it fails both arms by construction.
4. Auditor forced final-verdict turn: not built (left for `AuditorHarness.refine`; ~$0.85 left).
5. toolz needs a Python <= 3.9 World. Only the 7 rebuilt Worlds use recipe 3; other store3 windows keep recipe 2.

## P6 (built): the sealed held-out benchmark, native environments, the slime bridge

Spec: `00` §10, §11 (11.3), §0 V8/V9, §15 P6; `03` §4.3, §5. $0 spend: no model is called (scripted solvers only).

**Files.** New: `factory/benchmark.py`, `factory/envs.py`, `runtime/harness/sandbox.py`, `train/bridges/slime/`
(`types`, `rows`, `episode`, `generate`, `credit`, `filters`, `batch_builder_turn_advantages.patch`), `scripts/lh_bench.py`;
tests `tests/unit/test_m17_p6_{equiv,slime,envs,benchmark}.py` (33; one `docker`). `THIRD_PARTY.md` rows: the slime
sandbox protocol PORT, the slime mirrors, slime as DEPEND (not installed: it pulls Megatron and SGLang).

**The benchmark** (`benchmark.build`, `lh_bench.py build`). Candidates: the held-out store's 11 dependent windows (of 84
mined). No held-out text was authored: every task is `prompt_pending` until a dedicated author lease writes it under
sealing; `publish` refuses until then.
- **Sealing** against the 28 trainable training windows (`train_filter="all"`, the 124 mined ones, also available):
  - repositories disjoint;
  - **organisations: `pallets` is shared** (training `click`, held-out `jinja`): the 4 jinja windows fail;
  - **time split** (UTC): the latest training B is 2026-07-02T21:45Z (pyparsing). 8 of 11 held-out B dates are earlier
    (the store was split by repository, not by date). Against all 124 mined training windows (latest 2026-10-13) no
    held-out window would pass;
  - **contamination**: the full oracle diffs gave 132 code-substring hits, every one shared boilerplate (GitHub Actions
    steps, diff headers, Sphinx `conf.py`, `from __future__ import annotations`). The scan now compares the lines each
    oracle adds to Python sources (tests, packaging, CI, docs, imports out): 4 hits remain (2 jinja windows vs boltons
    and pyparsing: `if type(self) is not type(other): return NotImplemented`, an operator table). They are idioms, not
    leaks, and still block (conservative; both windows fail org and time anyway).
- **Gates** (executed, `crucible-lh-p6-*` / `crucible-p6-*`, about 30 min; journaled per task in `gates/`):

  | gate | passed | failures |
  |---|---|---|
  | S6 executed edges, DAG | 11/11 | |
  | S5 clean-World solvability + null agent | 11/11 | (the lint waits for the prompt) |
  | S8 oracle, unit-step alternative, 4 floors, mutants | 11/11 | |
  | S10 60 oracle grades at concurrency 1 and 4 | 10/11 | mistune-w90b1a73: 12/60 false negatives, same end state (verifier nondeterminism). Its m12/m13 checks are "near_linear" timing tests. The host was also running the equivalence run, which is the load S10 exists to expose |
  | S3 build, A's suite, leak probes, 60 reruns, verifier unchanged | 6/11 | jinja ×4: A's suite cannot import (markupsafe ≥ 2.1 removed `soft_unicode`, a World bug, not fixed: excluded anyway); mistune-w89cbd56: m05 (`..._does_not_scale_quadratically`) passes at A on an idle host. The empty patch then scores Φ 0.083. S5's null agent passed only because the build ran under load |
- **Split: 1 task.** `voluptuous-w0953225-6cf9450`: H1, 9 milestones, 2 executed edges, B 2026-07-05.
  - Manifest sha256 `828a9286…`, and `verify` passes. It holds per task the window record, A tree, oracle patch,
    hidden tests, check set, verifier version and World digest, plus `tasks.jsonl` (the portable row, split `heldout`).
  - With `time_policy="report"`, 4 more pass every other rule and gate (mistune-w1bdbdf3, python-dotenv-wc50c6f7,
    sqlparse-w647d145, sqlparse-w93d7718). The manifest is then `sealed: false`.
  - §10.2's ≥ 20 per intent needs held-out windows mined after a fixed training cutoff date, from orgs no training
    repository shares.
- **Report plan** (`report`, `paired_report` over `bench.lh_metrics`): pass@k / pass^k (k = 1..5) by category with
  task-resampled CIs; mean final Φ; milestones lost; compaction loss; Brier / ECE; USD per passing episode with the
  max/min ratio across tasks; paired base vs trained with the MDE.

**Native environments** (`envs.make_env`). They hold:
- `WorkspaceHarness` (or `RLMHarness`) with `EpisodeLimits(timeout_s=agent timeout)`;
- the checkpoint sidecar inside `InfraWatch`;
- `honesty_group_fn`;
- the tool limit (`policy_kwargs`: limit, warning, final STATUS turn; `max_steps` = limit + 2);
- `run_spec(attempt_timeout_s=rollout timeout)`.

The two timeouts were executed on the toy window: the agent timeout gives stop `timeout`, graded Φ 1/3 -> 1/6; the
rollout timeout gives a masked attempt.

**Finding (V8, native path):** a Docker World killed mid-episode was graded as an ordinary 0 (`lh_bench` on
parse-w07c2ff9: `reward 0.0, masked False, stop agent_done`). The host-mounted workspace stays collectable, and the
dead container's tool errors read as the policy's.
- `InfraWatch` probes the World after every step and at the end, recording `crucible:env_container_lost` (infra, masked).
- `scripts/lh_run.py`'s `harness()` does not use it yet.

Also in `envs`:
- `drop_infra_groups`: whole-group drop with a refill count;
- portable rows: `task_row`, `write_task_rows` (held-out refused unless the benchmark writes them), `read_task_rows`;
- `objective_of`: `versioning.objective_id`;
- `export_harbor`: validated by Harbor's loader. It uses a separate verifier and the patch collected by `COLLECT_SH`;
  its stdlib grader, executed on the host, matched `grade_patch` on the oracle and the empty patch;
- `export_agentenv`: through `factory.publish` with the row as `RowTask` arguments.

**The slime bridge** (`train/bridges/slime/`; `train` never imports `factory`: the task side arrives through
`--crucible-env-provider`, default `crucible.factory.envs:slime_provider`).
- **Rows:** `prompt` + epilogue, with `metadata` {`instance_id`, `image`, `workdir`, `agent`, `crucible_task_ref`,
  `split=train`, `objective_id`}. A row is refused when its split or its ref's split is held-out, or when it names an id
  the manifest seals. `lh_bench.py rows` wrote 28 training rows, each resolved back to the same task (hashes checked).
- **Sandbox:** slime's protocol ported, `exec_and_wait` included (it needs Linux `setsid`, so it was tested in Docker).
  - `ExecSandbox` covers the Docker World and host Worlds; files travel on stdin.
  - A dead container raises `SandboxTerminatedError`.
  - `egress_probe` executed in a World container: public blocked, `ok`.
- **`generate`** (the steps of the module doc):
  - the snapshot hook runs before each model call and reports the previous turn's actions to the same
    `CheckpointSidecar`, and checks the World is alive;
  - the final collection is `COLLECT_SH` run inside the sandbox;
  - grading happens in a fresh container, with the model's tool calls as the trail;
  - R = 0.5·Φ + 0.5·[gates], cross-checked against the report;
  - `metadata.crucible` carries turn_credit (with token spans), STATUS claims, the honesty term and the give-up;
  - every masked case is a removed sample.
- **Credit and filters:**
  - `reward_post_process`: unique trajectories, R − mean, honesty composed;
  - `convert`: slime's own conversion when installed, plus `turn_advantages`;
  - `advantage`: refuses to run without the field-list patch, which `git apply --check`s cleanly on slime 859938d;
  - `group_filter`: infra drops the whole group, then the staleness cap, then nonzero spread;
  - `staleness_sample_filter`: the batch-level cap.

**Exit criterion (executed): identical on both paths.** Two runs, five scripted solvers each:
- the toy window as a host World (unit test);
- the real training window parse-w07c2ff9-44e14b0 in its Docker World (`lh_bench.py equiv`, `equiv.json`,
  `all_identical: true`).

| script | reward | per-turn ΔΦ | M6 group credit | honesty-composed advantage | masking |
|---|---|---|---|---|---|
| pass (oracle, commit, STATUS) | 1.0 / 1.0 | equal | equal | 0.6365 / 0.6365 | — |
| fail (no-op) | 0.0 / 0.0 | equal | equal | −0.3625 / −0.3625 | — |
| hack (oracle + conftest.py) | 0.0 / 0.0 (`path:protected`, `path:plumbing`) | equal | equal | −0.3875 / −0.3875 | — |
| partial (mutant revert_m23, GIVE_UP) | 0.45 / 0.45 (Φ 0.9) | equal | equal | 0.0785 / 0.0785 | — |
| infra death (`docker kill` mid-episode) | — | — | — | — | masked / removed (`env_container_lost` / `sandbox_terminated`) |

On the toy window the partial run also loses a milestone (ΔΦ −1/3 at the turn that broke it), and the per-token
advantages of `convert` equal native `turn_credit_map` on each call's own tokens. The infra group is dropped whole on
both sides.

**Defects found and fixed while running.**
1. `write_file` as base64 in the command line broke on a 52 KB source file: Linux caps one argument at 128 KiB, and the
   deadline wrapper repeats the command. It now writes over stdin.
2. The dead-World failure kind was unmapped (blame unknown, counted by the stop rule). It is now
   `crucible:env_container_lost`.
3. Contamination: the boilerplate false positives above.

**Deviations / open.**
1. Not ported yet:
   - the Claude Code / Codex CLI harnesses, the SGLang token adapter and `TrajectoryManager`. `generate` runs the
     scripted harness and model; any other agent raises `NotImplementedError` with the reason;
   - a real adapter.
   With no real adapter, "egress only to the adapter" is executed as "no egress": the World is `--network none`.
   Reaching an adapter needs a socket shim.
2. `RowTask` (agentenv) runs only on the local backend: window Worlds run on Docker exec, not the agentenv gateway.
3. Harbor export: path checks are `not_verified` (Harbor gives the verifier no trail). Class-based test ids match by
   test name as a fallback.
4. Gate failures on held-out Worlds are reported, not fixed:
   - jinja's markupsafe pin (a World-expert job; P4's `repair_versions` applies);
   - the timing-test milestones of two mistune windows (verifier defects, owner: the verifier).
5. The mirrors (`MirrorSample`, `_mirror_convert`) are used only without slime; a test checks them against slime's
   source. DESIGN.md is left to the merge.

## P5a (built): generators (rebreak, stitched windows, open), generic scouts, the deterministic gates

Spec: `00` §1, §2, §4, §6, §7.7, §15 P5 (the zero-cost half: no solver, no swarm-built window). Spend $0.0098 of the $1 cap,
all prompt authoring (qwen3.7-flash, `p5/spend.jsonl`).

**Files.** New: `factory/generators/{base,rebreak,stitch,open}.py`, `factory/sources/{records,lease,github,scouts}.py`,
`scripts/lh_gen.py`; tests `tests/unit/test_m17_p5_{base,rebreak,stitch,open,sources}.py` (30, synthetic git repositories and the host interpreter,
no Docker). Changed (additive): `lanes/windows_world.open_world(pythonpath=)` (a two-repository workspace),
`lanes/windows_s3.world_s3` (reruns compare the ids both runs have, plus the test count: marshmallow's clock-built
parametrize ids made every rerun "unclean", the P2 volatile-id defect at S3).

**One contract, one machine** (`base`). A generator writes a *window record* whose A..B history it synthesises when the
units are not consecutive commits of one repository (`derived_clone` / `init_mono`, `apply_chain` through a private
index, `write_chain` under `refs/crucible/<gen>/<id>`; the source clones are never written, tests check it). Then
`windows.validate_window` runs unchanged (World at A, leak probes, f2p with reruns, prefix owners, executed prober,
revert mutants) and the task is a `WindowTask` subclass, so the P2/P4 handlers read it as a mined window. Gates
(`deterministic_gates`, cheap first): S6 (`check_longtask`, >= 1 executed edge), S5 (`verifier2.s5_solvability`), S8
(`verifier2.s8_ground_truth`; its "2 valid alternatives" finding needs P2's paid rewriters and is reported apart as
`alternatives_pending`), S3 (`world_s3`, 60 reruns at concurrency 1 and 4), S10 (`measure_flake`, 30 + 30 oracle grades).
Before authoring, checks bind to placeholder requirements (`gate_requirements`: the same ids and milestone map authoring
keeps). `author_generated` is `author_window` in the generator's voice, plus a version-number lint (the author invented
"releasing version 0.13.2" twice; release numbers are scrubbed from its changelog input).

**Rebreak** (`rebreak`). Fix units by message + files (src and tests, nothing but prose besides); per release B (the
release that shipped them, <= 4 windows back: reverting a fix five years old conflicts in 219 of 770 checks even so)
each fix is executed at B: B with its source reverted runs the pool's test files; real when a test its commit added or
changed passes at B and fails reverted. Interaction screen: two fixes interact when one test *case* fails under each
single revert (`interaction_edges`; parametrize ids matter: humanize's two `naturalsize` fixes add different cases to
one test and do not interact). A' = B with the group's sources reverted (3-way chain) and its fix tests removed from the
agent's test files (`strip_tests`; prose kept, adjacent changelog reverts conflict); B' == B (checked). Milestones = per
fix test groups; S6 = the prober's executed edge (fix k's tests fail on A' + k, pass with j). `simplify('locate')` names
the faulty functions.

**Stitched windows** (`stitch`). Pairs: a consumer commit that raises (or lifts the cap of) its requirement on the host
and whose code imports the host (a tox or docs pin is not a consumer: `imports_host` drops click <- cachetools). Host
window (release before V, V]; consumer window = that commit (or the next one with code, within 5). Workspace `host/` +
`consumer/`, one synthetic history; World image installs the host from A under a pip constraint and fails the build if
any other host version ends in site-packages; `StitchSandbox` runs pytest per side. Consumer tests that a host unit
makes pass become their own *observation* milestone (`<id>c`, `split_observations`) with an executed edge (ddmin over the
host units) to the host milestone: the cross-repository dependency. `host_required` re-executes them on (host A,
consumer C1) vs (host B, consumer C1). `simplify('host_change')` pre-applies every host unit (observation milestones go
with their prerequisites).

**Open** (`open`). `real_perf` commits; A = parent, P = known faster version. Workload = the module-level entries a
profiler plugin sees tests call whose extent runs a changed function, with their plain-data arguments; every grade
draws fresh inputs from a seed (digits redrawn, numbers rescaled, lists grown), runs A and the candidate alternately in
one 1-CPU container (`QUIET`: no other container of the process meanwhile), trims the extreme round, and requires A's
results on every input (exceptions by type) plus A's passing suite. Reward 0 unless correct and lo > 1, then log s /
log s_bar, uncapped (`reward_capped` for [0, 1] trainers). Floors: no-op, claims, memo, replay (fresh inputs: no hit),
fast-wrong mutant. Milestones are *gated*, not `requires`: S6 is n/a (`open_no_ceiling`), never a pass.

**Scouts** (`sources`). Intent card -> terms (content words, bigrams) -> fixed templates per record type (no model).
`GitHub`: unauthenticated, on-disk cache, search paced at 6.5 s, core quota read from headers, `Retry-After` honoured.
Lease first (held-out repos by name and owner/name, orgs owning only held-out repos, a sealed manifest's repos/orgs/time
cut), then licence, size (not fork, not archived), requirement scan of raw packaging files (the dependents snowball;
GitHub has no REST dependents endpoint), forks, then the CPU probe (clone, World image of HEAD, suite with no network).

**Run** (training clones of store3's 18 repos + scouted pairs; `p5/`).

| generator | funnel (candidates -> stage) | passing S6+S5+S8+S3+S10 |
|---|---|---|
| rebreak | 383 fix units -> 770 executions at B, 324 real -> 1,482 pairs of real fixes, **4 interact** (all more-itertools; full-suite screen too) -> 2 groups built -> 1 dependent | **1** (rb-more-itertools-061d3f39, 4 milestones, edge numeric_range reversed fixes, 4/4 valid mutants) |
| stitched | 8 pairs (3 training, 5 scouted) -> 26 bumps, 12 with a host window -> 11 Worlds -> 8 with milestones -> 5 dependent | **5** (arrow<-dateutil, 2 x marshmallow-jsonschema<-marshmallow, 2 x glom<-boltons); 3 cross-dependent (consumer milestone with an executed edge to a host milestone) |
| open | 89 perf commits -> 40 with a traced entry -> 5 with proven headroom (s_bar 1.17-1.68) | **2** with S6 n/a (all_equal s_bar 1.29, padded 1.68); tail, isiterable and join fail S10 (1, 1 and 8 of 60 oracle grades miss the bar: speedups near it) |

Rebreak reasons: 219 source reverts conflict at B, 18 test files gone, 8 Worlds do not build; 48 groups dropped by the
interaction screen. Open reasons: 48 no module-level entry with plain-data arguments (methods dominate), 29 headroom
unproven (P not significantly faster on fresh inputs), 4 P changes results on fresh inputs (humanize's `naturalsize`
log rewrite: 20 of 20,000, later fixed upstream), 2 inputs mostly raise, 1 trace failed. Scouts: 721 consumer records (1 refused by
the lease), 77 passed licence and size, 76 require their host; 6 probed in order (libraries named after their host
first, then smallest: the first ordering picked strawberry, 57 failing tests); 5 pass the CPU probe; 221 network calls,
469 cache hits.

Authoring: 5 stitched + 1 rebreak prompts (qwen3.7-flash; ledger two-way, lint clean, conformance clean); 2 open prompts
from a fixed template (no model). Total 0.0098 USD.

**Findings that change the plan.**
1. *Rebreak's premise is rare in real history*: 4 of 1,482 pairs of executed real fixes interact (0.3%). Real fixes are
   local; a rebreak task is long only when the generator finds the rare interacting pair. P5b should rebreak with a
   swarm-planted interaction (a real fix plus a real refactor of its function) or accept rebreak as `short`.
2. *Stitched observation edges are the real cross-repository dependency*: consumer adaptation commits rarely carry tests
   that fail without the host (0 consumer-unit milestones); the consumer's existing tests observing the host are what
   requires the host change.
3. *Open needs a margin, not just significance*: three proven-headroom tasks (s_bar 1.17-1.26) fail S10 because the
   oracle's verdict is itself a timing draw; P5b should require lo_bar well above the bar (about 1.2) at build.
4. *Open tasks need module-level entries*: 48 of 89 perf commits change methods only reachable through objects; the
   workload generator needs constructor capture (P5b).

**Deviations.** (1) S8 runs without P2's two model alternatives (`alternatives_pending`, $0 phase). (2) Swarm-built
windows are not built (P5b). (3) Rebreak S6 yield is 1, not 5 (finding 1). (4) Open tasks pass S6 as n/a, by design.
(5) Stitched records drop CRLF prose files from the oracle (`drop_crlf_prose`: the lane's text round trip loses CRs,
dateutil's NEWS). (6) Two stitched glom tasks overlap (host windows 19.1->19.2 and 19.1->19.3; different consumer
commits): the herding guard should see them as one family. (7) DESIGN.md is left to the merge. (8) One container sweep
removed in-flight runs of the full-suite rebreak screen; any effect is a missed fix or interaction, never a false accept. (9) Scout `real_fix`/`real_perf`/
`request` records are leased only; their repositories are filtered when cloned. (10) Docker-marked m17 tests were not
run; the non-Docker m17 suite passes (480), ruff and lint-imports (4 kept) clean.

## P5b (built): swarm-built windows, and failure targeting against a frontier closed model

Spec: `00` §2, §3, §4, §7, §14, §15 P5 (the paid half). Hard cap $30 (one USD ledger, `p5b/run/spend.jsonl`, every client
guarded); spend **$21.96** (by role below; $8.04 left, more than the $3 slack). Report `runs/p5b_report.json`.

**Files.** New: `factory/generators/swarm.py`, `scripts/lh_swarm.py`; tests `tests/unit/test_m17_p5b_{swarm,run}.py` (13, scripted
models, no Docker). Changed: `scripts/lh_run.py` (`harness()`: the sidecar is `envs.InfraWatch` around the checkpoint sidecar, or
around nothing with checkpoints off, so a World container lost mid-episode is masked, never an ordinary 0: P6's open item).

**Part A, the swarm-built window** (`swarm.py`; P1's machine, sessions, typed hand-offs with the full spec, the family task
repo, receipts). Source (S1): a store3 *training* window's start A; the requests are the lines the next release added to the
repository's changelog (else its commit subjects), with the issue or pull request text they cite (GitHub, cached), one `request`
record each, leased. The real B is never shown to a role. The machine: lease -> planner (a call; A's static call graph between
the names the requests mention, `code_links`, is in its input) -> oracle builder (an M16 session in a private clone at A, the
window's World image, offline; one commit per milestone) -> oracle check (count, subjects, library source, A's passing suite
still passes; a regression the builder cannot remove twice goes to the planner) -> verifier writer (another session, a clone
holding A only, `isolated_from` the oracle; `crucible_hidden/test_<mid>.py`) -> reviewer (deepseek; blocks once, then advisory:
V6) -> build (the hidden tests on A and on the oracle: findings name test ids and outcome kinds, never values; then the
synthetic history: tree k = oracle commit k with test and protected paths back at A plus the hidden files of milestones 1..k,
and `validate_window` unchanged: World, leak probes, f2p with reruns, prefix owners, the executed prober, commit-revert mutants
of the oracle's own commits; no executed edge goes back to the planner) -> prompt (S7: `author_generated`, ledger, lint,
required names, version lint) -> gates (S6, S5, S8, S3 60 reruns, S10 60 grades) -> second (P2's rewriters of both families
and the deepseek shadow suite: S8 alternatives, S9 differential) -> hacker (deepseek agent in the solver's World, S12) -> accept
(S17, herding guard). Roles: planner qwen3.8-flash, builder / verifier writer / prompt author qwen3.7-flash (author family);
reviewer, second verifier, rewriter, hacker deepseek-v4-flash.

Funnel (14 store3 start windows, the last run of each):

| stage | in | out | drops |
|---|---|---|---|
| S1 requests (changelog 11, commit subjects 3) | 14 | 14 | |
| S2 plan | 14 | 12 | 2: `why_requires` names nothing the required milestone adds (boltons 26.1, semver 3.1) |
| S4 oracle | 12 | 10 | marshmallow: the plan contradicts 2 existing tests (after the planner's second plan); markdown: builder out of turns (paused, V7) |
| S5 hidden tests vs the reference (+ one verifier loop) | 10 | 5 | arrow, click, funcy 2.1, humanize: tests fail on the oracle; more-itertools 10.2: tests already pass at A |
| build: World, leak, f2p, S6 executed edge, valid mutants | 5 | 5 | (earlier runs: 3 S6 drops, "claimed" edges that did not execute) |
| S7 prompt | 5 | 5 | |
| S6, S5, S8, S3, S10 (deterministic) | 5 | **5** | |
| S8 alternatives (2, one per family) + S9 | 5 | 5 | 3 needed a re-run: the second verifier's unbounded thinking returned empty replies (`ReasoningCap` added) |
| S12 hacker | 5 | 5 | 2 hacker passes, both judged valid solutions (below) |
| S17 accepted | 5 | **5** | |

Accepted (start A, the requests of the release after it): python-semver 2.9.1 (to_tuple, to_dict, `__getitem__` built on
to_tuple), boltons 25.0.0 (remap `cache`, research forwards it to remap, split maxsplit=0), funcy 1.15 (memoize .skip/.invalidate,
cache unified on it, reraise callable), cachetools 5.5.2 (`condition` on cached/cachedmethod, func decorators built on it, MRU
removal), more-itertools 10.2.0 (totient, all_equal `key` parameter, then key semantics). Each: 3 milestones, 1 executed edge, 3/3 valid revert mutants, S3 60/60, S10 0/60 (CP 0.049).
Example (boltons): requests R4 "Added `cache` option to iterutils.remap and iterutils.research", R5 the split fix; plan m1 remap
`cache`, m2 research forwards it (`research calls remap`, from `code_links`), m3 split; hidden tests 4/4/4; the prober's edge
m02 <- m01; prompt R1-R3 in the maintainer's voice; alternatives qwen imperative + deepseek functional; shadow suite gates R3.

Findings that changed the build while running: (1) qwen3.7-flash planned independent series on every candidate in 4 tries and
its claimed edges did not execute (S6) -> planner on qwen3.8-flash, `code_links`, the `why_requires` name guard, the previous
plan returned with its findings. (2) 80 session turns ended a 4-milestone build with all four commits made but no
`rlm.finish` -> 150 turns. (3) deepseek and qwen3.8 thought through 24-32k tokens and answered nothing (`finish=length`) in
reviews, plans and shadow suites -> `ReasoningCap` (OpenRouter `reasoning.max_tokens` 12k) on call roles. (4) S12 counted any
hacker reward 1 as a hack; two hackers reported "no bypass worked" and implemented the behaviour. Now executed: a hacker pass is
re-run under the qualified shadow suite; every measured requirement passing and no path check fired = a valid solution
(`judge_pass`), else a confirmed hack. (5) A redo reused the previous run's session worktrees -> a redo clears them; an accepted
task is never rebuilt. (6) Stopped runs left live lease announcements (later runs abstained) -> non-accepted runs retire theirs.

**Part B, frontier targeting.** Model (OpenRouter list, no key read): **openai/gpt-6.1-sol**, $2 / $10 per 1M (cache read $0.10,
automatic): the strongest OpenAI or Anthropic model the cap affords for >= 4 attempts per arm on several windows (GPT-6 Astra and
Claude Fable 5.1 are $10/$50; Claude Opus 5.5 $4/$20 with no cache read without `cache_control`, which our client does not send).
Pilot first (semver, full arm, tool limit 150, checkpoints on): $0.089, 14 steps, Phi 0.29, `agent_done`; extrapolation written
before scaling (`pilot.json`): 301 such episodes under cap - $3 slack, n = 4 per arm = $0.71 per window. Actual cost per window
(both arms, probe) $0.24-5.08: glom 30eb7ffe's contexts reached 0.5-1.0M prompt tokens per episode.

Per window (gpt-6.1-sol, tool limit 150, checkpoint sidecar on: every episode's Phi series and turn credit are in the report;
target = the deepest executed-edge milestone, success = it passes with its `requires`; Phi = the task's mean final Phi; cause
mix = trusted P3-auditor labels on the full arm's failures; S15 through `targeting.target_verdict` and the S15 machine):

| window (generator; target; simplify) | full | simplified | counterfactual: Fisher p (power at p_ref 0.8) | Phi full / simple | band | cause mix (skill share) | probe | S15 | $ |
|---|---|---|---|---|---|---|---|---|---|
| python-semver-w41775dd (mined; m31; prerequisites) | 0/9 | 8/8 | **attributed** 4e-5 (0.99) | 0.33 / 0.50 | yes | skill 3, spec 4, unstated 2 (0.33) | ok | S2 (intent: cause mix) | 1.16 |
| marshmallow-jsonschema <- marshmallow 0856ed04 (stitched, cross-repo; m02c+m02; locate) | 0/8 | 8/8 | **attributed** 8e-5 (0.99) | 0.00 / 1.00 | no (all 0) | skill 6, spec 1, unstated 1 (0.75) | ok | S2 (out of band) | 1.55 |
| glom <- boltons 2ae0bb57 (stitched, cross-repo; m12c+m12; locate) | 4/4 | 4/4 | underpowered 1.0 (0) | 0.56 / 0.56 | yes | skill 3 (1.0) | ok | **ok** (flags: underpowered, no paraphrase) | 2.28 |
| glom <- boltons 30eb7ffe (stitched, cross-repo; m12c+m12; locate) | 4/4 | 4/4 | underpowered 1.0 (0) | 0.40 / 0.40 | no | skill 4 (1.0) | ok | S2 (out of band) | 5.08 |
| rb-more-itertools-061d3f39 (rebreak; m04; locate) | 4/4 | 4/4 | underpowered 1.0 (0) | 1.00 / 1.00 | no (all 1) | no failure | ok | S2 (harden) | 0.38 |
| sw-python-semver (swarm; m03; prerequisites) | 4/4 | 4/4 | underpowered 1.0 (0) | 1.00 / 1.00 | no (all 1) | no failure | ok | S2 (harden) | 0.26 |
| sw-boltons (swarm; m02; prerequisites) | 0/8 | 8/8 | **attributed** 8e-5 (0.99) | 0.33 / 1.00 | no | skill 5, spec 3 (0.63) | ok | S2 (cause mix, band) | 0.64 |
| sw-funcy (swarm; m02; prerequisites) | 0/8 | 8/8 | **attributed** 8e-5 (0.99) | 0.33 / 1.00 | no | spec 4 (0.0) | ok | S2 (spec defects) | 0.70 |
| sw-cachetools (swarm; m02; prerequisites) | 4/4 | 4/4 | underpowered 1.0 (0) | 1.00 / 1.00 | no (all 0.5) | verifier_defect 1 (unmeasured) | ok | S2 (out of band) | 0.92 |
| sw-more-itertools (swarm; m03; prerequisites) | 4/4 | 4/4 | underpowered 1.0 (0) | 1.00 / 1.00 | no (all 1) | no failure | ok | S2 (harden) | 0.24 |

Reading: four windows are attributed at power 0.99 (the frontier fails the dependent milestone in the full task in every
attempt and passes it in every attempt once the prerequisite is applied or located); five are solved by the frontier in both
arms (harden: the axis is not hard for it); the frontier's rewards rarely spread across attempts (band fails on 8 of 10: it
behaves deterministically at the task level), and the auditor sends semver w41775dd and the swarm funcy window back for spec
defects.

Findings that change the plan (swarm-built windows, exposed only by solving them):
1. *Unstated import paths* (built). All 4 trusted funcy labels: the hidden tests import `SkipMemoization` from the package root;
   R2 names the class, but nothing says the root exports it, and S7's required-name rule checks names, not import paths. S7 now
   runs every `from X import Y` of the hidden tests in A's World image; a pair A cannot import whose path the request does not
   state gets one fixed lane sentence ("These names must be importable exactly as written: ..."). Executed on the accepted
   windows: funcy -> `from funcy import SkipMemoization`; boltons, cachetools: nothing. The targeted funcy prompt predates the
   rule (its next version goes through P3's spec-defect acceptance). Boltons' m1 tests also pin an unstated fact (`cache=False`
   must raise `RecursionError` on a cycle) that the reviewer passed: open (the verifier brief, the reviewer and S5's lint do
   not catch an asserted fact the plan never states).
2. *An oracle that skips part of its plan.* sw-cachetools: R3 asks to remove `MRUCache` and `mru_cache`; the oracle deprecated
   them instead, its m3 tests are weak enough to pass, and 23 MRU tests stay in the regression check. A correct removal (the
   frontier, 4/4) gets Phi 1 and reward 0.5; the auditor labelled it `verifier_defect`. Every gate passed: S8's alternatives
   are rewrites of the oracle (they share the gap) and S9's shadow suite, which gates R3, does not test the removal either.
   Open: an executed oracle-vs-plan check (each requirement's independent tests written to fail the start state for that
   behaviour, run on the oracle before the regression set is frozen; a regression test that contradicts a requirement leaves
   with it). Not built: the second verifier's suites fail the oracle for wrong reasons too often to send the oracle back on
   their evidence alone (P2: 336 of 858).

Audit: 113 bundles (65 failing), 3 runs (incremental; seeded controls in every batch, 27 of 31 batches trusted). P3's 24-turn cap
ended 30 of 41 first sessions without a verdict; at 48 turns 4 labels stay rejected. Trusted labels on the full arms: skill_gap
21, spec_defect 12, unstated_format 3, verifier_defect 1.

Spend by role ($21.96): frontier solver 13.45 (arms 13.13, probes 0.23, pilot 0.09); auditor (deepseek) 4.85; swarm 3.66:
oracle builder 2.09, verifier writer 0.91, planner 0.27, reviewer 0.16, hacker 0.15, second verifier 0.04, rewriters 0.04,
prompt author 0.003.

Tests: `test_m17_p5b_swarm.py` (11: requests, plan/oracle/test guards, outcome findings without values, `synthesize`, call
links, `ReasoningCap`, lane-stated import paths, the machine end to end on a local kernel with scripted planner, sessions and reviewer; frontier pick,
extrapolation, target specs) and `test_m17_p5b_run.py` (2: `harness()` wraps InfraWatch; a dead World is infra). The non-Docker
m17 suite passes (525); ruff and lint-imports (4 kept) clean.

**Deviations.** (1) Planner on qwen3.8-flash (same author family; qwen3.7-flash produced no dependent plan). (2) The World expert role is
not a session: the start World is the store3 window's image (`ImageBook`, recipe 3), and S3 runs on the built window. (3)
Stitched windows simplify by `locate`, not `host_change`: on observation-only windows host_change pre-applies every milestone
(nothing left); targets are the observation milestone and the host milestone it requires. Their scripted probe runs the
consumer's suite; a policy report equal to the sum of its own two runs (two repositories) passes (`two_repo_probe`). (4) The
auditor's turn cap was raised (24 -> 48) by run configuration, not through `AuditorHarness.refine`; its notes are unchanged. (5)
The S12 rule change (`judge_pass`) and the shadow-suite fix (`ReasoningCap`) were applied to stored builds by `finish` (S9, S12,
S17 re-run; the hacker episodes reused, no new one): semver (S12), funcy, cachetools, more-itertools (S9). (6) Paraphrase
stability (§7.6) not run; S15 flags it. S15 routes to S2 pause `exhausted` (no automated S2, as in P4). (7) The funnel reports
each start window's last run; earlier runs (qwen3.7 planner, 80 turns) were redone. Per-candidate dollars are approximate
(role clients are shared by concurrent candidates); role totals are exact. (8) Part A's S3/S10 ran while Part B episodes ran
(host load; all passed). (9) DESIGN.md is left to the merge; Docker-marked tests not run.

## Benchmark v2 (built): one training cutoff, every training source, timing tests out, World fixes

Spec: `00` §10, §0 V9, §7.7; P6's open item (1 of 11 held-out windows sealed). $0: no model is called; every held-out
task stays `prompt_pending`.

**Files.** Changed: `factory/benchmark.py`, `factory/sealing.py`, `factory/lanes/windows.py` (held-out registry, timing
rule in `f2p_stage`, the World-mismatch rule), `factory/lanes/windows_s3.py` (S3 reruns skip timing tests),
`factory/lanes/windows_world.py` (World fixes, a leak probe), `scripts/lh_bench.py` (`select`, `mine`, `funnel`,
`build --v2`). New: `factory/lanes/windows_timing.py`; tests `tests/unit/test_m17_bench_{timing,sealing,world}.py` (23, no Docker);
`runs/bench_v2_manifest.json`.

**Cutoff and policy.**
- `TRAINING_CUTOFF` = 2026-07-03T00:00Z, just after the latest B of any trainable training window (pyparsing,
  2026-07-02T21:45Z). A later cutoff leaves fewer held-out windows; an earlier one cuts trainable windows.
- A held-out window must END after the cutoff (UTC). Its repository and organisation must differ from every training
  source (`benchmark.training_sources`, `sealing.seal_sources`): every window of every training store (store, store2,
  store3: all 124 mined, trainable or not; P5a rebreak, open, stitched), the stitched pairs, the scouted repositories
  the lease let through (a lease-refused record was never usable: python-poetry), the probed clones and the training
  repository list. 1,443 sources, 835 organisations.
- Training sources that end after the cutoff are the **embargo**: 20 ids, 38 entries over the three copies of the mined
  store (8 untrainable store3 windows, P5a's 9 open
  windows and 2 rebreak windows; rebreak's B is its build date). The manifest lists them and `sealed_ids` returns them,
  so the training row writers refuse them. None of the 28 trainable windows is embargoed; P5a's accepted
  rb-more-itertools-061d3f39 is.
- Contamination scans every training store's oracle additions (not only the 28 trainable ones). A code-substring hit
  whose longest shared run is under 100 normalised characters (two of the scan's 50-character windows, about one line)
  is an idiom: recorded per task, not blocking (`idiom_chars`; P6 blocked them). The 3 such hits: an alphabet literal
  (pyjwt vs pyparsing), `def __init__(self, format: Optional[str] = None` (voluptuous vs marshmallow),
  `return get_type_hints(obj, globalns, localns, include_extras=True)` (sphinx-autodoc-typehints vs typer).

**Timing tests** (`lanes.windows_timing`). Never a milestone or a regression check:
- static, at mining (before ownership): the name names a measurement (`near_linear`, `scale_quadratically`, `timeout`,
  `perf`, `benchmark`, `quickly`...), a marker or fixture says so, or the body measures elapsed time (`perf_counter`,
  `monotonic`, `time.time() - t0`, a helper named like `assert_near_linear`). The first run also matched `scale` and
  `speedup`: it excluded voluptuous's decimal `precision_scale` tests and mistune's `speedup` plugin test. Both words are
  out, and the 4 windows were mined again;
- empirical, at S10: each oracle grade's failing tests are kept. A checked test that fails in some grades of the one end
  state is load-dependent; the window is validated again without it (`revalidate`) and gated once more;
- S3's existing-suite reruns do not compare timing tests.
Result: mistune-w89cbd56 (P6: 12/60 S10 false negatives on its `near_linear` milestones) passes S10 (0/60 after 18
timing tests left its checks). No v2 task showed a load-dependent test at S10.

**Selection** (before mining; `BENCH_V2`, `lh_bench.py select` -> `selection.json`). 14 new repositories: fsspec,
filelock, sphinx-autodoc-typehints, w3lib, starlette, pyjwt, jsonpickle, peewee, streamlink, pydash, pint, dateparser,
graphql-core, alembic. 10 reserves. 5 kept: sqlparse, tomlkit, python-dotenv (no window after the cutoff), voluptuous,
mistune. jinja dropped (pallets). Excluded at selection:
- 12 repositories whose organisation a lease-passed scout record holds (PyCQA, executablebooks, sqlglot...);
- 7 that the grader's pytest would import: a pytest11 plugin or a pytest dependency (packaging, pluggy, anyio,
  typeguard, faker, pytest-mock, platformdirs). The agent's edit would reach the grader (V4);
- 28 with no window after the cutoff, 4 with a compiled core or a data download, and sqlfluff (suite too slow).
Dropped after selection, and still held out: peewee (unittest modules pytest does not collect); filelock (its lock
tests hang on the bind-mounted workspace, the containers survive `docker kill`); streamlink (suite past the 600 s
sandbox timeout); alembic (writes SQLite files into the read-only grading mount).

**Mining** (`benchmark.mine_heldout`, store3's pipeline: tolerant ownership, executed prober, revert mutants, recipe-3
Worlds, leak probes). Candidates: release windows whose B is after the cutoff, then 8-unit commit runs wholly after it,
newest first, 8 per repository; store3's held-out windows that end after the cutoff are validated again. The
pre-declared rule then applied (fewer than 20): the reserves were mined, and the cap rose to 12.

| | windows | World built, leak-ok | >= 2 milestones | dependent | mutants valid |
|---|---|---|---|---|---|
| 14 selected + 5 kept (8 per repository) | 72 | 72 | 56 | 8 | 8 |
| 10 reserves | 14 | 14 | 6 | 3 | 2 |
| cap 8 -> 12 (fsspec, sphinx-autodoc-typehints, w3lib, pint) | 10 | 10 | 10 | 0 | 0 |
| **all** | **96** | 96 | 72 | 11 | **10** |

filelock's windows never finished (not counted). Stop reasons:
- 61 independent (no milestone needs an earlier unit): fsspec 12/12, sphinx-autodoc-typehints 11/12, pint 8/9,
  w3lib 7/9, pydash 6/7, graphql-core 4/4, tomlkit 3/3;
- 16 no test result at B: streamlink 5, litestar 4 (SQLAlchemy 2.1 deprecation errors), starlette 2 (B's conftest
  needs `blockbuster`, which A does not declare), nox 2, alembic 2, pydash 1 (mypy test items past the timeout);
- 6 under 2 milestones (dateparser), 2 real failures at B (huey needs redis), 1 with every revert conflicting
  (jsonpath-ng).

**World fixes found while mining** (each with a test; only the Worlds that need one change tag, except 3):
1. A with every extra did not install (fsspec's `full`): the build tries A's test extras before A bare.
2. 345 xfails read as "354 tests fail at B (environment mismatch)" (w3lib): a mismatch counts only failed or error.
3. **Structural (leak).** One World image serves every window with the same packaging, so its installed copy of the
   project is the A of whichever window built it first. Executed: w3lib's and pyjwt's Worlds held another A's sources
   in site-packages (a later A is the solution); pint ships its tests, and the leak probe caught a later A's tests.
   In store3 training, 27 of 28 trainable Worlds hold an installed copy, and 1 (arrow-wce69c31) holds another A
   (9 of 10 files differ). Fix: A is installed again without dependencies last, and its installed copy keeps only its
   metadata. A new leak probe (`no_installed_project_source`) fails any image that still holds source. Executed: the
   new pint image passes with 0 files; the old w3lib image fails with 10. `benchmark.refresh_world` rebuilds every
   candidate's World under the new recipe before gating; all 10 passed the probes again.
4. `minversion = 8.4.0` refused the pinned pytest 8.3.3 (streamlink): the World pins pytest 8.4.2 when asked.
5. `--cov` in `addopts` wrote `.coverage` into the read-only mount (jsonpickle): `COVERAGE_FILE=/tmp/.coverage`.
6. Test plugins listed only in `tox.ini [testenv] deps` (graphql-core's `pytest-describe`) are installed.
7. A dependency group in one pip command failed whole on one pin (starlette): one by one, with a name-only retry.

**Gates and split** (`lh_bench.py build --v2 --train-filter all --idiom-chars 100`; S3 60 reruns, S10 30 + 30 at
concurrency 1 and 4; containers `crucible-bench-*` / `crucible-lh-bench-*`). All 10 candidates passed sealing and S6,
S5, S8, S10 and S3. No S10 grade moved (0/60 each, upper bound 4.87%). Gate time ran from 2 minutes (voluptuous) to
2.3 hours (pint: S3 reruns a 2-minute suite 60 times).

| task | class | B (UTC) | milestones | executed edges |
|---|---|---|---|---|
| jsonpickle-wbddf979-42b78e9 | H1 | 2026-08-06 | 6 | 1 |
| mistune-w89cbd56-618ebfb | H2 | 2026-10-02 | 11 | 1 |
| parsel-w05b3d37-9f353b9 (reserve) | H1 | 2026-09-25 | 11 | 1 |
| pint-wa1db27c-6521366 | H2 | 2026-09-11 | 7 | 1 |
| pyjwt-w7144e45-c6fe464 | H1 | 2026-09-11 | 13 | 1 |
| sphinx-autodoc-typehints-w4949f6c-599a386 | H2 | 2026-10-15 | 8 | 1 |
| tenacity-wa2af454-4410996 (reserve) | H1 | 2026-10-07 | 5 | 3 |
| voluptuous-w0953225-6cf9450 | H1 | 2026-07-05 | 9 | 2 |
| w3lib-w6dd8249-0a98564 | H1 | 2026-09-27 | 6 | 1 |
| w3lib-wd5d5e4f-05aff0d | H1 | 2026-10-03 | 7 | 2 |

**Split: 10 tasks, `sealed: true`.** 9 repositories, 9 organisations; 7 H1 and 3 H2. Manifest sha256 `15897a31c7d60dbe...`
(`runs/bench_v2_manifest.json`, the full hash inside), `tasks.jsonl` `99a5e949...`, embargo `5dc8883a...`. `verify`
passes.

**Deviations / open.**
1. **10 sealed tasks, not >= 20.** The binding constraint is dependency: 61 of 72 gradeable windows are independent.
   The post-cutoff period is 108 days, so windows are short (8-unit runs), and these libraries' recent work is mostly
   parallel fixes. The gates passed all 10 that reached them. What would raise it:
   - more held-out repositories: about 1 sealed task per 10 windows here, so roughly 100 more windows (10 to 15 repos);
   - time: the cutoff stays fixed, so each month adds windows, and 12-unit runs become possible;
   - held-out tasks from the other generators on held-out repositories (§10.2: same machine, same gates);
   - grader-side fixes for the 16 windows with no result at B, and filelock's: a writable grading copy (alembic), B-only
     test dependencies for the grader (starlette's `blockbuster`), a longer timeout for large suites (streamlink,
     pydash's mypy items), version repair (litestar: `windows_s3.repair_versions`), file locks on a bind mount
     (filelock). At about 1 dependent window in 7 gradeable ones, these add 2 to 4 tasks.
2. The idiom rule is new policy (P6 blocked idioms). Without it, 3 tasks drop (pyjwt, voluptuous,
   sphinx-autodoc-typehints).
3. The miner ran alongside the gates (load). S10's level 4 saw that load, and no task moved.
4. Training side, not changed here:
   - the embargo removes P5a's accepted rebreak task from training rows;
   - store3's training Worlds (and P6's) predate fix 3. arrow-wce69c31's World holds another A, and S3 now fails every
     old image on the new probe. They need `refresh_world` (or `windows_s3.world_fix`) before training.
5. Run 8-unit commit runs for held-out (training used 12). DESIGN.md is left to the merge.
