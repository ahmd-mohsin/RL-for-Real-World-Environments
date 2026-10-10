# M8 `forge` — research: agent-swarm task and world synthesis (2026-10-09)

**Scope.** What eight local sources can give `crucible.forge` (ARCHITECTURE §M8; DESIGN §7, §9, §4.3, §4.4), and how M8 should be built. Every claim below was read in source; the load-bearing ones were re-checked by hand (marked ✔). Nothing was run except greps. No code was changed.

**Path prefixes.**

| Prefix | Path |
|---|---|
| `FW/` | `RL-Environments/06-frameworks-and-env-code/` |
| `SS/` | `FW/swe-bench-swe-smith/swesmith/` |
| `VF/` | `FW/primeintellect-ai-verifiers/` |
| `RE/` | `FW/primeintellect-ai-research-environments/` |
| `PE/` | `FW/primeintellect-ai-prime-environments/` |
| `OE/` | `FW/meta-pytorch-openenv/envs/agent_world_model_env/` |
| `EOG/` | `FW/servicenow-enterpriseops-gym/` |
| `MD/` | `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/general/envs/` |
| `HB/` | `FW/harbor-framework-harbor/` |
| `AE/` | `agentenv-framework/src/agent_env/` |
| `R2E/` | `FW/huggingface-repo2rlenv/src/repo2rlenv/` |
| `ART/` | `RL-Environments/07-articles/` |
| `CR/` | `crucible/src/crucible/` |

**Licences at a glance.**

| Source | Licence | What that allows |
|---|---|---|
| SWE-smith | MIT | PORT with notice |
| verifiers | MIT | PORT with notice |
| research-environments, prime-environments | Apache-2.0 | PORT with notice |
| Recursive Task Synthesis (RTS) dataset | CC-BY-4.0 | Reference only, with attribution |
| OpenEnv AWM code | BSD-3 (already vendored) | PORT with notice |
| AWM HF dataset | not recorded locally | Check before shipping derivatives |
| EnterpriseOps-Gym | Apache-2.0 | PORT with notice |
| MiMo v2.6 release | Apache-2.0 | PORT with notice |
| Harbor | Apache-2.0 | PORT with notice |
| agentenv-framework | Apache-2.0 + NOTICE | PORT with notice |
| Repo2RLEnv | Apache-2.0 (per-recipe upstream licences in `R2E/pipelines/recipes/catalog.json`) | PORT with notice |
| FineEnvs | Apache-2.0 | PORT with notice |
| git-swarm repo | **no licence; website only** | Patterns from the paper only |

---

## 0. TL;DR

1. **No source has an M8.** Each one has one or two pieces:
   - SWE-smith: mutate, validate, combine.
   - verifiers: a 12-line learnability reward.
   - AWM: generate the tasks before the seed data.
   - EnterpriseOps-Gym (EOG): distractor tools and infeasible tasks.
   - MiMo: difficulty = number of Apps, decoy documents, footprint gates.
   - Harbor: compile, check, analyze, and installed agents run as sandboxed roles.
   - agentenv: a human review store.
   - Repo2RLEnv: a review→repair loop, semantic probes, dedup, and a budget ledger.
   - GitSwarm: coordination through a shared repo.

   Nobody publishes their synthesis generators locally: AWM, MiMo `office_gen`/`db_gen`/`make_tasks.py`, the RTS generator, the general_agent synth, and the GitSwarm harness are all missing. What Crucible must build is the asymmetric-information orchestration and the signal-cited traps; no source has either.
2. **Orchestrator: build our own.** Use a deterministic DAG of about 400 lines on `CR/core/store`, combining these patterns:
   - RTS `publish.py`: an fsync'd events journal and resume keyed by input hash.
   - agentenv `Task`: `depends_on`, the rule that a retry may only resume from an ancestor step, and `preflight()`.
   - Repo2RLEnv `campaigns/budget.py`: a reserve/settle dollar ledger. PORT this one.
   - FrontierSmith: hash-bound task states.
   - GitSwarm: commits with `informed_by` provenance.

   Neither agentenv's step runner (static, no dollar budgets, retries without feedback) nor Harbor `exec` (fixed map→reduce) fits as a dependency.
3. **Run sandboxed roles as Harbor installed agents inside wrapper tasks, as Harbor's own `check`/`analyze`/`annotate` already do.** The Environment builder, Oracle builder, Test writer and Adversary need a sandbox for coding work.
   - "Cannot see" is enforced by what is copied into the wrapper's `environment/`, plus no network except the model API.
   - Single-shot JSON roles (Architect, Miner ranking, Curator) call `CR/actors/llm.py` directly.
4. **SWE-smith PORT list (MIT):**
   - procedural modifiers (Adversary defect-class mutants for A3), with `.cu` mapped to the C++ modifiers;
   - the status-transition report, **with its states renamed**;
   - the repair rule;
   - the combine loop (G-compose);
   - the size gate and recovery prompt from PR mirror.

   ✔ **The F2P direction does not match locally.** `SS/harness/valid.py:144` counts `FAIL_TO_PASS` (fails on buggy, passes on clean), but `SS/harness/gather.py:328-348` reads `PASS_TO_FAIL` as F2P. Port only with named states and a direction fixture test. The history leak (per-task branches) is a red-team fixture for A5.
5. **learnability.py: PATTERN (MIT).** `4p(1−p)` from `VF/environments/proposer_solver/.../taskset.py:151-168` ✔.
   - Upstream uses n=4 with no confidence interval.
   - Ours: compute it on the binary outcome O (not R), per (policy, harness).
   - Sample sequentially, 4 → 8 → 16 runs, with Wilson bounds (`CR/credit/operators/branch_value.py:40`).
   - Bands: train p ∈ [0.1, 0.7]; for bench, the upper bound of frontier pass@1 ≤ 0.2.
   - Output advice on which knob to turn, never an instruction rewrite.
6. **world_synth: PATTERN.** The stage order:
   - from AWM: scenario → **tasks first** → schema → **seed data written to satisfy the tasks, one reason per row** → tools → checks;
   - from general_agent: the `gold_check` rule (the oracle replays, the state changes, verify = 0 at the start and 1 at the end) ✔;
   - from uuid_ctf and enigmata: a seeded `DifficultySpec`;
   - from MiMo and EOG: Apps used, distractor tools and decoy documents as the difficulty ladder.

   The toy variant world and its 5 tasks must also run offline in CI: scripted or cached roles, no paid calls.
7. **Traps.** G-trap composes three things:
   - M4 signals (polarity −1, with a reliability score);
   - C2 `defect_to_check.from_correction` (`CR/credit/operators/defect_to_check.py:51`), which already produces a validated static check plus a PR-mirror mutant;
   - a behavioural check from the Test writer and a surface hook from the Architect.

   For the toy pack, traps are EOG-style infeasible requests and MiMo-style superseded documents, each check citing its signal or policy id.
8. **dedup.py: build it ourselves.** Layers:
   - content hash;
   - Repo2RLEnv two-hash fingerprints;
   - state identity (start and oracle-end World fingerprints);
   - milestone-set Jaccard;
   - optional semantic embeddings;
   - FrontierSmith behavioural divergence over calibration pass vectors.

   No source has semantic dedup.
9. **The M9 subset that must exist before M8 can pass its gate:** A1, A2, A3, A5, A7, A8, plus the A4-lite footprint/collateral gate (§9.8). M8 depends on M9, so build `accept/` checks A1/A2/A3/A5/A7/A8 first, deterministic, with Harbor `check` only as A7 support.

---

## 1. What M8 must deliver, and what Crucible already gives it

**Spec.** ARCHITECTURE §M8 (l.565-583) and DESIGN §7.2-7.4. The gate:
- every derived span task, and at least one trap task per seed, passes M9;
- each trap cites its signal;
- the toy variant world and its tasks pass M9.

**Interfaces M8 consumes (already built):**

| Need | Crucible API | Notes for forge |
|---|---|---|
| Task container | `CR/core/schema/task.py`: `CrucibleTask{id, family, horizon, world, milestones, card, negatives[Negative{id, path, expected_gate, provenance}], oracle="solution/solution.patch", harbor_dir, seed_of, derived_from}`; `TaskCard{grounding{traces, oracle_source, checks_with_provenance}, value_proof{frontier[], oracle, mutation_score, negatives_killed, flake, leak_audit}, splits}` | `derived_from` and `seed_of` already model span lineage. **Gap:** `oracle` and `Negative.path` assume patches. Non-coding worlds need action scripts or state blobs (§9.2). |
| World | `CR/core/schema/world.py` `WorldSpec{apps[AppSpec{id, template, state, params}], actors[ActorSpec{tools (fnmatch), grounded_on, policy_doc}], clock, events[EventSpec{when{at,after,every,on_tool}, action{nl,tool,permission}}], policies, network, isolation, limits, seed}` with reference validation | The world_synth output is a `WorldSpec` plus a data bundle. Distractor tools come free: widen `ActorSpec.tools`. |
| Snapshots | `World.snapshot()` / `restore(WorldRef)` (`CR/world/world.py:536,564`); `WorldRef.fingerprint` excludes the step count | Span start states (G-span); fingerprints for state-identity dedup; replaying the oracle three times for A1. |
| Verifier reads | `VerifierView.query(app, Query)`, `.fingerprints()`, `.unchanged(*apps)` (`world.py:95-126`) | Milestone checks; collateral gate. `unchanged` is per App, so row-level `unchanged_except` is new (§9.8). |
| Milestones | `CR/core/schema/milestone.py` `Milestone{id, check{app, query{op,args}, expect}, requires, weight, cost, gate, provenance[{signal, source, note}]}`, with a DAG validator | The Test writer emits a `MilestoneSet`; A8 counts `provenance`. |
| Evaluate | `CR/workflow/milestones/evaluate.py` `evaluate`, `evaluate_series(milestones, states)`; `FileStateView`; `LogStateView` | Span goal = the milestones newly at 1 between steps i and j. |
| Mine | `CR/workflow/milestones/mine.py` `from_diff`, `from_ci`, `from_signals` (l.118, 173, 195); `filters.screen` | Candidate checks for the Test writer and Miner. |
| Graph | `CR/workflow/graph.py` `WorkflowGraph{milestones, stages, edges, verdicts, sources}` | The Miner uses stages to choose (i, j) span boundaries. |
| Signals | `CR/core/schema/signal.py` `Signal{id, type, polarity, grades[], source_event, target{app, object, span}, reliability}`; `signals/attribute.py`, `reliability.py` | Trap seeds = negative signals with reliability ≥ θ and a located target. |
| C2 | `CR/credit/operators/defect_to_check.py` `from_correction(signal_id, patch, wrong, end_state) -> CheckCandidate{checks, mutant_patch, validated}`; `reverse_patch` | Already does "validate that the check separates wrong from corrected" and emits the PR-mirror negative. The trap generator wraps it. |
| R-properties | `CR/credit/properties.py` r1_oracle … r6_state_only | A1/A2/A3 can reuse r1/r2/r3 directly. |
| World audit | `CR/world/audit.py` (RFC-008-style check ids; pass/fail/skip) | A5/A6 base; forge calls it on every synthesized world. |
| Leaks | `CR/world/leaks.py` `scrub_text`, `leak_tokens` | A5 instruction and document scan. |
| LLM | `CR/actors/llm.py` `LLMClient(model, budget_usd)`, `.complete_json`, per-sample cache, `BudgetExceeded`, `UsageBlank` | Single-shot roles. The cache gives offline CI replay. |
| Store | `CR/core/store` `put_doc/get_doc/put_file/tag/resolve/list_docs` | Artifacts by content hash; orchestrator state. |
| Harbor I/O | `CR/interop/harbor/task_io.py` `load_harbor_task` (Harbor's own config, packager hash, ignored keys) | Validates every forged task dir. |
| Toy world | `crucible/packs/_toy/worlds/helpdesk.yaml`: tickets (sqlite: customers, tickets, orders, refunds, notes), kb (refund_policy.md, escalation.md), mail; grounded customers dana and lee; a rule-based supervisor; ops events (sla_reminder 1h, month_end_freeze 3h revokes `tickets.issue_refund`, refund_audit on_tool) | The base for world_synth. Tool rules live in `data/helpdesk/tools/tickets.py` (refund > 50 needs `approved_by`; same customer; amount ≤ order). |
| Toy signals | `packs/_toy/data/ticket_logs/tickets.jsonl` (T1001-T1024): `policy_flag` (refund_over_50_needs_approval, retention_offer_before_closure, identity_check_before_deletion), `sla_breach`, CSAT, reopen, supervisor_edit, refund_reversed | Trap provenance for the toy pack. |
| Coding seeds | `packs/coding/families/legacy_gpu_migration/seeds/task0*/crucible/{task,world,card}.yaml` (overlays; `harbor_dir` points to `migration_tasks/`); `seed_of: []` | The H4 seed is task04 v3. Its traces are the upstream PCL GPU PRs and the construction transcript (M3). |

---

## 2. Source 1: SWE-smith (MIT)

### 2.1 Pipeline and data shapes

| Stage | Files | Output shape |
|---|---|---|
| S0 profile + mirror | `SS/profiles/base.py:80-131` (owner, repo, commit, test_cmd, exts, timeout=90, timeout_ref=900, min_testing, min_pregold); `create_mirror` `:340-407` (`rm -rf .git`, `git init`, one "Initial commit", strips `.github/workflows`); image build `SS/profiles/python.py:46-91`; `log_parser` `:93-102` | `{test_name: PASSED\|FAILED\|...}` |
| S1 entities | `SS/profiles/base.py:438-491`; skips test paths `:549-558`; `CodeEntity` in `SS/constants.py:88-135` (file_path, line span, src_code, `_tags: set[CodeProperty]`, complexity); Python complexity `SS/bug_gen/adapters/python.py:73-99` | in memory |
| S2 bug gen | procedural `SS/bug_gen/procedural/generate.py:67-185`; LM modify `SS/bug_gen/llm/modify.py:57-197`; LM rewrite `SS/bug_gen/llm/rewrite.py:58-168`; PR mirror `SS/bug_gen/mirror/generate.py` | `logs/bug_gen/<repo>/<path>/<entity>_<h8>/{bug__<strategy>__<h8>.diff, metadata__...json{cost, explanation, output, rewrite, strategy}}` (`SS/constants.py:138-176`) |
| S3 collect | `SS/bug_gen/collect_patches.py:18-77` | `[{..., instance_id: "<repo>.<strategy>__<h8>", patch, repo}]` |
| S4 validate | `SS/harness/valid.py:150-252`; reference run once per repo `:198-219` (per task if `min_pregold`, `:74-100`); `run_patch_in_container` `SS/harness/utils.py:101-284` (git apply → `--reject` → `patch --fuzz=5`; eval.sh with start/end markers; 10g memory) | `report.json{FAIL_TO_PASS, PASS_TO_PASS, FAIL_TO_FAIL, PASS_TO_FAIL}` or `{timed_out}`; status ∈ fail, timeout, 0_f2p, 1+_f2p |
| S5 gather | `SS/harness/gather.py:279-583`; drops timeouts, 0 F2P, 0 P2P (`:333-343`, PR-mirror exception) | `task_insts/<run>.json [{instance_id, patch, FAIL_TO_PASS, PASS_TO_PASS, image_name, repo}]` and **a public 2-commit branch per task** (`:491-569`) |
| S6 gold eval + repair | `SS/harness/eval.py`, `utils.py:192-238` (checks out `HEAD~1`, reverses the bug, reverts test edits); `SS/harness/grading.py:198-238`; `SS/harness/repair.py:82-256` | `{patch_exists, resolved, tests_status{F2P{success,failure}, P2P{...}}}`; flaky tests dropped |
| S7 issue text | `SS/issue_gen/generate.py:154-525` (configs `ig_v2.yaml`, `ig_tests.yaml`); `get_static.py`, `get_from_tests.py`, `get_from_pr.py` | `{messages, cost, responses{model:[text]}}` merged as `problem_statement` |
| S8 difficulty | `SS/train/difficulty_rater/` (predicts human fix time; no rollouts) | — |

### 2.2 Defects that a port must fix

1. ✔ **F2P direction mismatch.**
   - What happens:
     - `valid.py:69-71` takes the clean repo reference as `val_postgold_path` and the bug run as `val_pregold_path` (`:117`).
     - `grading.py:get_valid_report` (`:40-92`) labels "FAILED on buggy and PASSED on clean" as `FAIL_TO_PASS`, and `valid.py:144` keys status on it.
     - `gather.py:328-348` then selects `PASS_TO_FAIL` (passes on buggy, fails on clean) as F2P, which matches only the Modal path (`scripts/bug_gen_modal.py:222-231`).
   - Fix: a port names the states explicitly (`before`/`after`) and pins the direction with a fixture test.
2. **History leak.** Each task branch holds commit 1 (the bug) and commit 2 (the F2P test files deleted). The image runs `git clone -o origin` (`SS/profiles/python.py:57`), and `--repush_image` rebuilds with every branch (`gather.py:617-621`). So `git log --all` shows every task's answer. This is a good red-team fixture: A5 must fail such a task dir.
3. **Combine filter bug.** `SS/bug_gen/combine/same_module.py:167-172` produces an empty list under `--include_invalid_patches`.
4. **PR-mirror direct path probably inverted.** `SS/bug_gen/mirror/generate.py:283-306` stores the forward fix as the "bug" when it applies to HEAD.
5. **Flake handling is two samples.** One buggy run, one shared reference run, then one gold re-run. `repair` drops tests that ever fail under gold, and the reference is never re-run. Crucible's A4 needs ≥ 40 draws.
6. **Others:**
   - `ig_v1.yaml:13-17` describes the diff backwards;
   - the README cites a `criteria.py` that does not exist;
   - dedup is only the identical-rewrite hash in the filename;
   - global `os.chdir`, shared clones, and GitHub pushes as stage handoff.

### 2.3 Procedural modifiers (Adversary mutants for A3)

- **Base class** `SS/bug_gen/procedural/base.py:14-52`: `max_attempts=5`, `min_complexity=3`, `conditions` (tags the entity must have), `likelihood` (default 0.2, `SS/constants.py:15`), a per-modifier `random.Random(24)`. Each eligible node flips its own coin.
- **Families** `base.py:58-145`:
  - class: remove a base, remove methods, shuffle methods;
  - control flow: invert if/else, shuffle statements;
  - operations: change, flip (`python/operations.py:7-33`), swap operands, break chains, change a constant by ±1, increment ↔ decrement, swap augmented assignments, negate a boolean;
  - removal: loop, conditional, assignment, wrapper;
  - swap break ↔ continue.
- **Java and Ruby extras:** null returns, off-by-one, argument swaps, ternaries, string typos, guard inversion, nil-guard / safe navigation.
- **Likelihoods:** Python 0.25 / 0.4 (`python/__init__.py:25-39`), Java 0.3-0.75, C++ 0.5. `MAP_EXT_TO_MODIFIERS` (`procedural/__init__.py:20-35`) has no `.cu`/`.cuh`; adding them is one line.
- **No subtlety or difficulty filter.** The only downstream filter is ≥ 1 F2P, ≥ 1 P2P and no timeout.

### 2.4 LM prompts and combine

- **`configs/bug_gen/lm_modify.yml:21-52`:** "You are a software developer doing chaos monkey testing. Your job is to rewrite a function such that it introduces a logical bug that will break existing unit test(s)." It samples 3 of 9 bug examples with a jinja shuffle (`llm/modify.py:64-81`) and adds "DO NOT INCLUDE COMMENTS … indicating the bug location". Sampled at temperature 1, n = n_bugs.
- **`func_fun.yml:6-30`:** "a tired, deadline-pressured developer who has just worked 14 consecutive hours… a genuine cognitive error, not a contrived modification". Use it to build A2 "obvious naive" floors and realistic negatives.
- **`lm_rewrite.yml`** (`llm/rewrite.py:95-155`): blank the body, reimplement at temperature 0, keep the result if tests break. This is the best plausible-but-wrong floor generator.
- **Mirror `RECOVERY_PROMPT`** (`mirror/prompts.py:1-19`): "rewrite the entire source code while reversing the changes indicated by the diff patch… DO NOT MAKE ANY OTHER CHANGES". Use it as a fallback when `reverse_patch` will not apply to drifted code. Size gate: `should_attempt_recovery` (`mirror/generate.py:70-101`): 1-8 files, ≤ 500 lines, no file over 10k lines.
- **Combine:**
  - `combine/same_file.py:36-126`: pairs of validated bugs from one file, excluding func_basic, combine and pr_mirror; at most 100 combinations (`bug_gen/utils.py:185-192`); patches applied in sequence and the merged diff re-checked (`:149-172`); every combination reusing a patch is dropped.
  - `same_module.py:117-224`: groups by directory prefix at depth 3 and requires more than one file.
  - "Compatible" means only "applies together". Nothing checks that the combination is harder, and re-validation is not automatic.

### 2.5 Issue generation (instruction writer)

- **`ig_v2.yaml`:** "Mimic the style … DO NOT explain the fix… focus on how to reproduce… Do not mention pytest or what exact test failed… DO NOT GIVE AWAY THE FIX! THE SOLUTION CODE SHOULD NEVER APPEAR IN YOUR RESPONSE."
- **Inputs:** 2 SWE-bench Verified demos, the **full bug patch**, test output truncated to 10k tokens (`generate.py:145-151`), and up to 5 F2P test sources.
- **`ig_tests.yaml`** never shows the patch. That is the safer input set.
- **Leak risks:**
  - nothing checks the issue against the diff or the test names;
  - `get_from_pr` reuses public issues that link the fix;
  - the `get_static` tiers name gold files and tests.

### 2.6 Recommendations

| M8 piece | Call | What |
|---|---|---|
| Adversary | **PORT** P1 modifiers + P3 transition report (renamed) | Defect-class mutants for A3 ≥ 90%. A mutant counts only if it kills ≥ 1 check and leaves ≥ 1 check passing (no equivalent or total-break mutants). Add `.cu` → C++ modifiers. |
| Adversary prompts | **PATTERN** | `lm_modify` (seeded with the reviewer's defect text when trap-directed), `func_fun` and `lm_rewrite` for A2 naive floors. |
| Test writer | **PORT** P4 repair rule, stronger | K ≥ 3 oracle runs plus ≥ 40 fresh draws. Drop checks that are not constant. |
| Architect (instruction) | **PATTERN** `ig_tests` rules | Default to patch-free inputs. Mandatory n-gram leak scan against the oracle diff and hidden check names (`world/leaks.py`). |
| G-compose | **PORT** P5 (filter bug fixed) + new interaction check | Re-validate is mandatory. Keep the combination only if its pass rate is below each part's and its check set is not a mere union. |
| G-trap | **PATTERN** mirror | C2's `reverse_patch` is already the mirror. The LLM recovery prompt is the fallback. |
| G-transplant | **PATTERN** mirror recovery + size gate | Applying a change to drifted code is the closest analog. |
| Auditor / A5 | **SKIP** (code); use as a **fixture** | A SWE-smith-style dir with task branches must fail A5. |
| Calibrator, dedup, learnability, G-span, G-tighten, world_synth | **SKIP** | Nothing usable. |
| Orchestrator | **PATTERN** | Artifacts keyed by content hash, resume when a report exists, success/failed/skipped status with a reason. SKIP the script chaining, `chdir` and GitHub handoff. |

**Generalizing to a non-coding world.**
- **Tool code:** the procedural modifiers run unchanged on `packs/_toy/data/helpdesk/tools/tickets.py`. They can turn `<=` into `<` in the refund-amount check, drop the > 50 approval guard, or move the constant 50 to 49 or 51. Those are world-dynamics mutants for the Auditor's "do the checks notice a broken world" test.
- **State modifiers:** a parallel registry over `sql`/`fs` adapters. Conditions are table tags (foreign key, status enum, amount); each modifier has a likelihood. Operations: delete a row; swap a status within the enum; change an amount by ±0.01; drop `approved_by`; swap `customer_id`; move a note to another ticket.
- **F2P/P2P becomes:** for a state mutant M of the oracle end state O, `transition(checks(M) → checks(O))` must show ≥ 1 check failing on M and passing on O, and ≥ 1 check passing on both.

---

## 3. Source 2: verifiers proposer_solver, research-environments, prime-environments

### 3.1 verifiers (MIT)

- **`VF/environments/proposer_solver/proposer_solver/taskset.py`:**
  - six seed topics (`:171-192`);
  - the `PROPOSE` prompt (`:34-41`): "Invent ONE self-contained, genuinely hard {topic} word problem whose final answer is a single integer. Use your tools: write and run code to construct the numbers and to VERIFY the answer end-to-end before you commit to it. Do not reveal the solution path in the problem text. Your FINAL reply must end with exactly one JSON object…";
  - `SolveTask.from_trace` (`:78-116`) scans the reply in reverse, then falls back to a greedy `{.*}` match, then retries once leniently on LaTeX backslashes (`_loads_contract`, `:58-75`). A contract violation **raises**: the episode fails and is retryable, and is never scored.
  - 1 proposer then n = 4 solvers in a `TaskGroup` (`:137-149`).
- ✔ **Learnability** (`:151-168`):
  ```python
  rate = self._solve_rate(episode.traces)          # mean binary "correct" over solver traces
  for trace in episode.traces:
      if trace.agent.name == "proposer":
          trace.record_metric("solve_rate", rate)
          trace.record_reward("learnability", 4.0 * rate * (1.0 - rate))
  ```
  - **Holes:**
    - p comes from one episode, so it is one of {0, .25, .5, .75, 1}, with no confidence interval;
    - the proposer's answer is never checked independently;
    - nothing checks the problem for leaks, and nothing dedups;
    - an ambiguous problem cracked half the time by luck scores best.
- **Other useful pieces:**
  - **`best_of_n`** (`VF/verifiers/v1/envs/best_of_n/env.py:26-44`): records pass@n. Calibrator building block.
  - **`agentic_judge`** (`.../agentic_judge/env.py`):
    - "Verify the work EMPIRICALLY … never take the trace's word for an outcome you can check" (`:37-40`);
    - the reason comes before the verdict (`:55-75`);
    - "references are context, not your standard: recorded scores can be wrong" (`:86-96`);
    - the judge can run in a fresh box holding only the collected artifacts (`:353-361`), and it is never trained (`:318-320`).
  - **`vf-validate`** (`VF/verifiers/v1/cli/validate.py:51-53, 242-263, 349-367`): setup, gold and noop checks, each in a fresh runtime, with `--resume`.
  - **Two retry levels** (`VF/verifiers/v1/utils/retries.py`): one re-runs a single agent ("a flaky grader retries without re-burning the solver"), the other the whole episode. Off by default.

### 3.2 research-environments (Apache-2.0)

- **recursive_task_synthesis:** a **consumer and publisher only**; the generator is not local.
  - The corpus is 36,249 Harbor task dirs. Its README records 1,104 unresolved VM validation failures and says it is "not a fully gold-verified dataset". Use it as a reference corpus for dedup and contamination only.
  - **`publish.py`** is the orchestration pattern to copy:
    - `append_event` / `read_events` (`:64-85`): a JSONL journal written with sorted keys, flush and fsync; the latest event wins.
    - Resume (`:261-285`): identity = content hash + image ref. Completed tasks are skipped. Submitted tasks are polled again, never resubmitted. Failed tasks are retried only with `--retry-failed`, once per run.
    - The manifest is validated before any work (`:121-146`). Dockerfiles are compared to metadata to catch tampering (`:148-174`).
- **general_agent:** the closest analog to toy world_synth. 4,417 tasks, each `task.toml[metadata]{tier 0-4, pass_rates[{model, solver, value}]}` + `instruction.md` + `tools.py{TaskDB, TaskTools(@tool), verify(db)->float}` + `db.json` + `gold.json [[tool, kwargs]]`.
  - ✔ `gold_check` (`.../general_agent/corpus.py:78-105`) passes only if the gold replay does not raise, the DB hash changes, `verify(initial) == 0` and `verify(gold) == 1`.
  - Curation filters (`taskset.py:96-144`, `corpus.py:111-126`): tier band; pass-rate band per (model, solver); unmeasured tasks dropped when the band is narrower than [0, 1].
  - Holes: `verify` comes from the same generator as the gold chain; pass rates have no n or confidence interval; reward = max(exact hash match, verify).
- **uuid_ctf** (`.../uuid_ctf/taskset.py:44-82, 272-281, 531-780`): `DifficultySpec{noise files, decoys per file, guidance}` with 4 presets, seeded per sample, plus decoy and noise generators. Caveat: the easier presets add hints to the instruction. Treat those as a curriculum scaffold, not a world knob (DESIGN §7.4 forbids difficulty through the instruction).
- **terminal_lego / swelego:** "don't look up the answer key" is prompt-only and network access is restored. The Auditor must fail tasks like this.

### 3.3 prime-environments (Apache-2.0)

- **No generators.** `enigmata` (`PE/environments/enigmata/enigmata.py:28-100, 200-240`) is the generator interface to mirror: `generate(n, difficulty, language, split)` + `verify`, seeded; a difficulty a generator does not support is skipped, never faked.

### 3.4 Recommendations

| Piece | Call |
|---|---|
| learnability.py | **PATTERN** (reimplemented; the core may not import verifiers, which also conflicts on `mcp`). Optional **DEPEND** later via `interop/verifiers` if the Architect is trained online. |
| Architect | **PATTERN**: the proposer contract ("VERIFY end-to-end", "do not reveal the solution path", a typed final JSON block) and the `from_trace` parsing. **Add** an independent answer check (Oracle builder or Test writer) and a leak scan. |
| Calibrator | **PATTERN**: best_of_n + general_agent `pass_rates`, extended to `{model, harness, n, k, p, ci, task_hash}`. |
| Curator | **PATTERN**: tier, pass-rate band, RTS-style `exclusions.json` / `patches.json` with evidence. |
| Auditor (judge items) | **PATTERN**: agentic_judge (fresh box, reason first, "unverified is not solved"); `vf-validate` gold/noop as the model-free A1/A2 core. |
| Orchestrator | **PATTERN**: RTS `publish.py` journal and resume; verifiers' two retry levels; "contract violation raises, never scored". |
| world_synth | **PATTERN**: general_agent task layout + `gold_check` + uuid_ctf `DifficultySpec` + the enigmata API. |

---

## 4. Source 3: OpenEnv agent_world_model_env (AWM, BSD-3)

- **Already in Crucible:**
  - vendored: `CR/_vendor/openenv_awm/{data_loader,db_manager,patch}.py`;
  - wrapped: `CR/interop/openenv/awm.py:37,102`;
  - 100/100 sweep passes (M1).
- **Synthesis code is not local:** `OE/README.md:252-253` points to `github.com/Snowflake-Labs/agent-world-model` (arXiv 2602.10090).
- **Pipeline order** (from `DATA_FILES`, `OE/server/data_loader.py:23-31`, and the cached JSONL in `crucible/data/cache/awm/`):
  1. **Scenario** `{name, description}`: 1,000 rows, 371 categories.
  2. **Tasks** `{scenario, tasks[10]}`: fully specified, with literal ids ("For ticket ID 8754 … change status from 'new' to 'open'").
  3. **Schema** `{db_schema{tables[{name, ddl, indexes}]}}`: about 18 tables.
  4. **Seed data, written for the tasks** `{sample_data{tables[{table_name, reasoning, insert_statements}]}}`: about 129 inserts. Each table's `reasoning` names the task literals it seeds ("Includes task-specific IDs 8754 and 5621…"). Every task literal is present in the seed data. **This is the most portable idea:** tasks come first, then data generated to make them feasible.
  5. **Env code** `{full_code}`: FastAPI + SQLAlchemy, about 1.8k lines. `operation_id` becomes the MCP tool name; about 35 tools per env (16-87).
  6. **Verifier** `{verification{code, raw_response{reasoning, success_criteria, failure_criteria}}}` (`OE/server/verifier.py:200-214`).
     - Code mode compares initial and final DBs.
     - SQL mode runs an LLM judge over the trajectory plus a state diff (`:277-318`). That violates R6.
     - Sandbox: CPU 20 s, 512 MiB, no fork, builtins and import allowlist, read-only DB (`OE/server/_verifier_runner.py:1-60, 160-242`).
     - Rewards: complete 1.0, incomplete **0.1**, format error −1.
- **Holes:**
  - wall clock: 529/1000 seed sets use `datetime('now')`, and 992/1000 env codes call `now()` in tools (already defect B.7);
  - no oracle, no negatives, no collateral checks;
  - instructions name every literal, so there is no discovery;
  - the "incomplete" partial credit rewards trying.
- **Recommendations:**
  - **PATTERN: the world_synth stage order**, plus an automatic **literal-grounding audit**: every literal in an instruction exists in the seed data or is created by the oracle.
  - **DEPEND (built):** AWM worlds as base worlds for OOD variants, always with time frozen and our own milestones, oracle and negatives.
  - **PORT (small):** the verifier sandbox recipe, to run check code the Test writer writes.
  - **SKIP:** SQL-mode judge reward; 0.1 partial credit.

## 5. Source 4: EnterpriseOps-Gym (Apache-2.0)

- **Local:** 13 tasks (`EOG/data/revised/{csm,itsm}/*.json`), `gym_dbs.zip`, the runner.
- **Not local:** the full 1,150 tasks, the `plus_N_tools` configs, the infeasible subset (HF dataset, `evaluate.py:284-298`). **No generation code**; the tasks are expert-curated.
- **Task shape** (`EOG/benchmark/models.py:6-80`): `gym_servers_config[{mcp_server_name, seed_database_file, context, user_info}]` (more than one entry = hybrid / cross-app), `system_prompt` (role table plus "If a user request violates policy, do not act on it…"), `user_prompt`, `selected_tools`, `restricted_tools`, `verifiers[{verifier_type, name, validation_config}]`.
- **Verifiers** (`EOG/benchmark/verifier.py`):
  - `database_state{query, expected_value, comparison_type ∈ equals, gt, lt, contains}` (`:81-127, 291-372`);
  - `response_check`: an LLM scores 1-10 and passes at ≥ 7 (`:129-164, 412-483`);
  - `tool_execution`: each selected tool called at least `minimum_tool_calls` times (`:166-193`);
  - success = all verifiers pass (`executor.py:371-395`).
- **Distractors are tools.** Modes `oracle` / `plus_5|10|15_tools` (`README.md:185`; `executor.py:276-303`) and `restricted_tools` (`:98-103`). The example drops from 72% to 65% at +5.
- **Infeasible tasks:** the best model refuses correctly 53.9% of the time (site). There are no local examples, so we cannot see how refusal is graded.
- **Holes, measured:**
  - all 102 local verifiers are `database_state equals`, and only 1 checks absence;
  - **checks hard-code predicted auto-increment ids** (`WHERE case_id = 1233`, `csm/task_20251209_132736_…json`), so a correct agent that creates an extra row first fails;
  - `tool_execution` grades actions, not outcomes.
- **Recommendations:**
  - **PATTERN, infeasible trap:** the correct end state = protected state unchanged + an escalation or refusal record. Checks: absence predicates, `unchanged_except`, and an escalation or reply predicate.
  - **PATTERN, `distractor_tools` knob:** widen `ActorSpec.tools` with +5/+10/+15 tools from other Apps. It costs nothing.
  - **PATTERN, one named state predicate per milestone**, which is exactly our `MilestoneCheck`. **Lint:** no surrogate-id literal for rows the policy creates; match on natural keys.
  - **DEPEND (planned):** `interop/eog` as an external OOD benchmark.
  - **SKIP:** `tool_execution` and `response_check` as reward channels.

## 6. Source 5: MiMo v2.6 general envs (Apache-2.0)

- **Layout:**
  - `manifest.json` (uploads, `mcp_servers`, verifier cmd);
  - `tools/<app>.py` over `system/<app>/{schema.sql, state.db}`, the style our toy `tickets.py` already uses;
  - `workspace/` (docx, xlsx, pdf, csv);
  - `verifier_meta.json{items, check_code, gold_answer, input ∈ db|workspace|both, act ∈ none|mutate_db|edit_workspace|both, output, reward, src_protect?}`.
- **Synthesis is referenced but not released:**
  - `verify.py:410` imports `synthesis.office_gen.grade_rubric`;
  - `_helpers.py:1-4` says the helpers are shared between authoring-time `gt_code` and `verify.py`, generated by `make_tasks.py`;
  - `_helpers.py:563` mentions `db_gen`.
- **Tiers = number of Apps** (counted over 267 envs):

  | tier | envs | Apps (range, mean) | rubric items (mean) |
  |---|---|---|---|
  | t1 | 28 | 3-4, 3.5 | 5.1 |
  | t2 | 98 | 4-6, 4.8 | 5.6 |
  | t3 | 92 | 6-8, 6.9 | 5.3 |
  | t4 | 40 | 10-14, 11.6 | 5.2 |
  | t5 | 9 | 14-17, 15.9 | 5.0 |

  Requirement count and instruction length stay flat across tiers. Difficulty comes from search breadth. Many Apps are pure distractors: `s3k_0462_…_t4` has 11 Apps and reads only the workspace.
- **Task mix:**
  - about 75% are read-only answers (201/266 have `act=none`; 101 have no state check);
  - items are 1,212 LLM-judged and 215 rule-based.
- **Superseded and decoy documents as traps:**
  - 10 envs have `99_Archive/` folders;
  - 92 items grade avoiding a stale source, and their `pass_anchor` names both the authoritative and the decoy source. `s3k_0010` `party_rollup`: "Using the authoritative 31-row … ledger-2025Q3.csv (not the stale/different CSV … in the workbook)".
- **Collateral and footprint checks:**
  - `s3k_0013`: "Do not alter any of the other eight … cases", plus `case_integrity` (sanity).
  - `s3k_0018` `check_no_unrelated_mutation`: SHA-256 over non-target rows (target rows minus mutable columns), compared with a digest pinned at authoring time. 32 envs have such checks.
  - **`src_protect` / `protect_diff`** (`MD/*/_helpers.py:536-560`; grader `verify.py:351-366`): files and tables outside the legal solution's footprint must be unchanged, or the score is 0; new files are allowed. Zero false kills by construction. **Implemented but never populated in the 266 shipped envs.**
- **Grading** (`s3k_0013/verify.py:311-393`): gate items first (any failure → 0, LLM items skipped); score = Σw·s/Σw; LLM items get one call each with median votes; **judge unavailable → masked, not 0**.
- **Holes:**
  - mostly read-only, LLM-judged answers;
  - tiers raised with Apps the task never uses;
  - hand-set weights;
  - pinned digests break on rebuild;
  - no negatives, no clock, no actors.
- **Recommendations:**
  - **PATTERN, difficulty ladder:** Apps *used* + distractor Apps/tools/docs/records.
  - **PATTERN, trap as decoy + citing check.**
  - **PORT:** `protect_diff` and the non-target-row digest, as a new `unchanged_except` query op on the `sql`/`fs` adapters, with the footprint derived from the oracle's `StateDiff` (§9.8).
  - **PATTERN, `gt_code`:** the oracle replays in the World, and checks are validated against that end state through the same State Adapters.
  - **DEPEND (built):** `interop/mimo/importer.py` for base worlds.
  - **SKIP:** tier inflation; read-only-dominated mixes.

## 7. Source 6: Harbor (Apache-2.0)

### 7.1 `compile` (scaffolding)

- **Models:** `HB/src/harbor/models/compile/config.py:11-152`. `CompileConfig{dataset_name, task_name_prefix, output_dir, instructions[{text|path}], task_template, artifacts, environments[{path|paths, docker_image="ubuntu:latest", workdir="/app"}], verifiers[{path|auto_verifier{required_artifacts, reward_artifact, artifact_json_schemas}}]}`.
- **`Compiler(config).compile() -> list[Path]`** (`HB/src/harbor/compile/compiler.py:52-95`):
  - **cross product** of instructions × environments × verifiers; dirs named `<slug>-NNNN`;
  - copies the template, writes the parts, then checks `Task.is_valid_dir`;
  - the auto-verifier (`:173-252`) checks only that artifacts exist and match their schemas.
- **Holes:**
  - no index pairing, so call it once per forged task with single-element lists;
  - knows nothing of `environment_mode="separate"`, `Dockerfile.verifier` or `crucible/`, so the template must carry them;
  - **deletes an existing output dir**, so never point it at a seed;
  - `ubuntu:latest` breaks A9.
- **Related:**
  - `harbor task init` (`HB/src/harbor/cli/init.py:73-245`, `cli/template-task/`) has a canary-string option.
  - The adapter template `cli/template-adapter/adapter.py.tmpl` has the shape `Adapter(output_dir, limit, overwrite, task_ids).run()`.
  - **`harbor exec`** (`HB/src/harbor/exec/executor.py:59-179`) runs Compiler → `Job.run()` → reduce. Map trial artifacts are staged into the reduce task's `environment/artifacts/NNNN-<trial>`. This is Harbor's only artifact handoff.
  - `RetryConfig` (`HB/src/harbor/models/job/config.py:291-321`): max_retries, include/exclude exceptions, backoff.
- **Recommendation: DEPEND** through a new `CR/interop/harbor/compile_bridge.py` with a Crucible `task_template`:
  - schema 1.4 `task.toml` with `[verifier] environment_mode="separate"`, `network_mode="no-network"`, `[[verifier.collect]]`;
  - a `Dockerfile.verifier` stub, a `crucible/` skeleton, a canary header.

  Then pin digests and validate with `load_harbor_task` (`CR/interop/harbor/task_io.py:75`). The auto-verifier is for **role wrapper tasks only**.

### 7.2 `check` (A7) and `analyze` (Auditor / A3)

- **`run_checks(path, agent="claude-code", model="claude-sonnet-4-6", rubric_path, prompt_path, n_concurrent=4) -> (CheckReport, job_dir)`** (`HB/src/harbor/analyze/checker.py:40-113`).
  - `assemble_check_task` (`:144-207`) builds a wrapper task: the reviewed task under `environment/task/` (without `.git`), `python:3.13-slim`, artifact `/app/check-result.json`, agent timeout 1800 s.
  - `validate.py` gives **reward 1 = a valid review was produced**, not "the task is good".
  - Output: `CheckReport{results[{checks{name:{outcome ∈ pass|fail|not_applicable, explanation}}, cost_usd}]}` (`quality_checker/models.py:31-46`).
- **Default rubric** (`HB/src/harbor/cli/quality_checker/default-rubric.toml:1-54`), 11 criteria: behavior_in_task_description, behavior_in_tests, informative_test_structure, anti_cheating_measures ("ensure that the agent won't see newer commits"), structured_data_schema, pinned_dependencies, typos, tests_or_solution_in_image, test_deps_in_image, hardcoded_solution, file_reference_mentioned.
- **`run_analyze(path, agent, model="claude-haiku-4-5", filter_passing)`** (`HB/src/harbor/analyze/analyzer.py:94-125`). `assemble_analyze_task` (`:204-292`) **excludes `analysis.json`/`analysis.md` so the agent cannot read its own output**, an explicit information-hiding move. The `analyze-rubric.toml` criteria:
  - `reward_hacking`: "modifications to test files…, writes to /logs/verifier/reward.txt…, accessing or copying from the solution/ directory, manipulating the grading mechanism";
  - `task_specification`: "implicit expectations… hardcoded test expectations".
- **Holes:**
  - static LLM judgement;
  - blind to `crucible/`, milestones, separate verifier images, history beyond "newer commits", and signal provenance;
  - the hack list misses history spelunking, network answer sources, answer-key probes (already detected in M1, P27) and A13 cheats;
  - wrapper network is public by default;
  - default models drift (pin them).
- **Recommendations:**
  - **A7: DEPEND** on `run_checks` with a Crucible rubric. Add: `milestones_in_instruction`, `nothing_graded_unstated`, `nothing_stated_ungraded`, `reference_only_in_verifier_image`, `no_history_leak`, `world_spec_consistent`, `traps_cite_signals` (supporting only; A8 is computed). It **supports** the deterministic coverage check in §9.9, never replaces it.
  - **A3 / dynamic Auditor: DEPEND** on `run_analyze` with `filter_passing=True` and the A13 cheat list, on Calibrator passes and Adversary cheat runs. A detected hack becomes a registry incident plus a new A13 mutant.

### 7.3 Installed agents as builder roles

- About 45 harnesses under `HB/src/harbor/agents/installed/`.
- `BaseInstalledAgent(logs_dir, prompt_template_path, version, extra_env, config)` (`base.py:594-670`). `@with_prompt_template` renders through sandboxed Jinja and must contain `{{instruction}}`. `convert_trajectory` gives ATIF. There is an API-error taxonomy (`:34-159`) for telling infra failures from role failures.
- `ClaudeCodeOptions` (`claude_code.py:53-90`): `max_turns`, `max_budget_usd`, `append_system_prompt`, `allowed_tools`/`disallowed_tools`, `disable_web_search`. Exact `total_cost_usd` (`:1001-1031`).
- `AgentConfig` (`models/trial/config.py:63-140`): `extra_allowed_hosts`, which applies only while the agent runs.
- **Recipe**, the one Harbor already uses three times:
  1. Build a wrapper task. `environment/` holds only the role's visible inputs. The instruction is the role prompt + output schema + "X is your only deliverable". Artifact: `/app/<role>-out.json`. `tests/validate.py` checks it against the role's pydantic schema.
  2. Run one Harbor Job per role batch.
  3. Parse the artifact on reward 1; otherwise retry with the validator output as feedback.
- **Recommendation: DEPEND** through a new `CR/interop/harbor/role_runner.py` (`assemble_role_task`, `run_role_job`). It PORTs about 100 lines of the `assemble_check_task` / `_run_check_job` / `_extract_check_result` structure (Apache-2.0, credited in THIRD_PARTY.md).

## 8. Source 7: git-swarm and the articles; Repo2RLEnv code

### 8.1 GitSwarm (arXiv 2610.04862; repo = website only, no licence)

- **Mechanics** (`ART/gitswarm-2610.04862.txt:144-227`):
  - the task is a git repo with the spec at its root commit;
  - every contribution is an immutable commit with one parent and a mandatory `informed_by=` set (semantic inheritance without merges, `:159-179`); message format `[<type>|informed_by=<sha>,<sha>] <summary>` (`:3604-3613`);
  - B charged episodes; 5 concurrent workers;
  - each episode ends in Contribute, Nominate or Abstain (abstaining is free, `:186-196`);
  - validation covers publication rules only (required files, allowed paths);
  - read-only `gs_dashboard` / `gs_inflight` views, "neither view assigns work" (`:198-210`);
  - final answer = most nominations (`:212-224`).
- **Evidence:**
  - forcing a central agent to synthesise the final answer cost 3.87 points, but a central agent that only *selects* gained 2.34 (`:3240-3251`). This supports "no role grades its own output" and selection as a separate step.
  - About 1 worker in 6 abstains; in 97.1% of collisions the duplicate commit is avoided but the compute is already spent (`:2870-2905`).
  - Workers inspect only 5-7 commits (`:723-725`).
- **Prompt passages to adapt** (`:3338-3625`):
  - "Do not treat commit subjects as substitutes for reading the work".
  - Announce scope first: `"Gap: … | Deliverable: … | Success: <observable evidence> | Abort: <invalidating evidence>"` (`:3445`).
  - Write `EVALUATION.md` ("candid, evidence-based assessment … what you actually checked").
  - "agreement can be correlated or redundant… Do not infer quality from nomination count" (`:3404-3410`).
  - "Pass counts from different candidate-authored suites are not directly comparable" (`:3534-3536`).
- **Holes:**
  - everyone sees everything, the reference included (`:3667`);
  - no correctness check at publication;
  - votes are correlated;
  - the harness is not published.
- **Use: PATTERN only.** Keep the deterministic DAG on the acceptance path; use the GitSwarm mechanics *inside* open-ended stages (Miner exploration, Adversary search, Calibrator failure taxonomy). Use per-task forge repos with `Informed-By:` trailers for provenance (A8 for free). Dedup at announce time.

### 8.2 Articles

- **Prime Intellect, "On the Nature of the Swarm"** (`ART/pi-on-the-nature-of-the-swarm.md:21-99`):
  - compaction is a bet; persistent sub-agents beat one-shot ones;
  - a mesh beats a tree because the root of a tree becomes the planner that needs every context;
  - seed the swarm with "a shared git repo, an issue tracker".
  - → keep one persistent Auditor and one persistent Adversary per family (DESIGN P28), with the defect registry as their memory; the orchestrator holds handles, not content.
- **Preference Model, "Building RL Environments for Superintelligence"** (`ART/pm-building-rl-environments-for-superintelligence.md:15-39`):
  - "If the grader can't check it, it cannot go into the prompt"; ban the Triton example. → A7 runs **both ways**, and the Architect writes no process constraints.
  - Prefer security over realism.
  - Deep tasks with an honest way to give up. → infeasible tasks with a graded "give up / escalate" path.
- **Repo2RLEnv, "How not to build coding environments"** (`ART/hf-repo2rlenv-how-not-to-build-coding-environments.md:22-69`):
  - three leaks produced perfect scores: a PR link in the instruction, `origin/main` in `.git`, and `pip download`. The honest solve rate was 0.
  - The first defect was a gold patch scoring 0 because unrelated tests needed the network.
  - "A prompt instruction is a nudge, not a control."
  - → the Auditor **runs** these probes on every forged task; M1 already implements the defences.
- **FineEnvs multi-harness** (`ART/hf-space-fineenvs-multi-harness-rl.md:255-290, 1192-1203`):
  - the harness moved scores 13 points, against 2.5-5 for swapping models;
  - 35-58% of optimizer steps had zero reward contrast;
  - adding 500 hard tasks did not help.
  - → p is measured per (policy, **harness**); the Curator balances by harness and task; learnability bands matter.

### 8.3 Repo2RLEnv code (Apache-2.0): the most reusable M8 pieces in the survey

| Piece | Path | Call |
|---|---|---|
| `Review{assessments, issues[category, blocking], rollout label ∈ legitimate success/failure, reward hack, task defect, infra}`, `Citation`, `.sound` | `R2E/quality/loop/models.py:21-100` | **PORT** as the Auditor output schema |
| `SemanticProbe` (wrong solution **or valid alternative**) | `models.py:56-63` | **PORT** for the Adversary. Valid-alternative probes catch over-fitted checks, which our negatives-only A3 misses. |
| `Repair{edits old→new ≤ 16, issue refs}` | `models.py:102-133` | PATTERN for builder repair rounds |
| Citation check (a quote must appear literally in the cited doc) | `R2E/quality/loop/protocol.py:14-150` | **PORT**, extended with `signal_id` resolved against the M4 store (A8) |
| Probe outcomes (valid alternative passes, wrong fails, an uninstalled probe is a failure, not a kill) | `R2E/quality/loop/runner.py:93-112` | **PORT** into A3 |
| Label vs reward consistency | `runner.py:114-138` | PORT into the Calibrator |
| Review prompt | `R2E/quality/loop/prompts/review.md:11-220` | PATTERN (excerpts in §10) |
| Repair prompt: "Never make a task easier just to pass a particular rollout"; ≤ 3 rounds | `prompts/repair.md:1-70` | PATTERN |
| CodeMidas design/tests: remove the feature body, keep the original as a private oracle, expected values from running the original, the original passes and the starter fails, "the contract is frozen" | `R2E/pipelines/recipes/codemidas/{design,tests}.md` | PATTERN for G-hindsight/G-span and the Test writer |
| Soundness separate from difficulty: `curriculum() -> all_pass\|all_fail\|mixed`; exploit attempt first, then 4 solves; a provider refusal is not a defect | `codemidas/audit.py:82-89, 328-385, 525-565` | **PORT** (Auditor vs Calibrator split) |
| Dedup fingerprints (first 80 normalised chars; sorted public symbols of the oracle) | `R2E/pipelines/_oss_instruct.py:548-578` | **PORT** as dedup L1 |
| Test-strength gate | `_oss_instruct.py:427-545` | PATTERN pre-filter |
| `behavioral_divergence` | `R2E/pipelines/recipes/frontiersmith/pipeline.py:50-58` | **PORT** as dedup L5 |
| FrontierSmith prompts ("Do not describe the strong solution algorithm"; difficulty "not from unclear requirements"; "NEVER need a hidden optimum") | `frontiersmith/prompts.py:3-60` | PATTERN for G-tighten and the Architect |
| Retention binding (a label is written only if the bundle hash, construction report and reference receipt match; otherwise `unverified`) | `frontiersmith/retention.py:12-60` | **PORT** as hash-bound task states |
| SETA strategies (change_context ≈ transplant; increase_difficulty ≈ tighten) | `R2E/pipelines/recipes/seta_evol/strategies/*.md` | PATTERN |
| Budget ledger (SQLite reserve/settle, integer micro-dollars, overrun blocks dispatch) | `R2E/campaigns/budget.py:19-257` | **PORT** |

## 9. Source 8: agentenv-framework (Apache-2.0)

- **Review step** (`AE/task_step/task_steps/review.py:1-152`; store `AE/task_step/review_store.py:1-118`):
  - `ReviewTaskStep(timeout_seconds=10800, poll_interval_seconds=5)` registers an awaiting review with upstream structured output (truncated to 16k) and polls for `continue|abort`. Decisions go to `review_decisions[step_id]={decision, decided_by}`; abort raises `ReviewAbortedError`.
  - The store is a `task_reviews` collection keyed on `(instance_id, step_id)` with a 7-day TTL; `set_decision` changes only awaiting reviews; `list_awaiting` feeds the hub UI.
  - **Holes:** binary decisions, no reasons or per-check comments; needs the agentenv document store.
  - **DEPEND** behind a new `CR/interop/agentenv/review.py`: `put_awaiting("forge:<task_hash>", "human_audit", upstream={card, findings, check report})`.
    - Deterministic sample: `sha256(task_id) % 10 == 0`, plus every H4+ task.
    - Reasons are recorded in Crucible's own registry.
    - Fallback: the file-queue human actor (`CR/actors/human.py`).
- **Step runner** (`AE/task/task.py:296-808`; `AE/task_step/task_step.py:1-179`):
  - `TaskStep(id, depends_on, fail_task_on_error, retry_config{retry_from_step_id, max_retries=1})`; the resume step must be an ancestor (`_validate_dag`, `task.py:317-348`);
  - `preflight()` runs before any sandbox exists (`task_step.py:120-125`);
  - concurrent independent steps (`_drive_dag`, `:500-718`);
  - the failed span is rolled back through a step journal (`:166-276`);
  - handoff via `TaskStepContext{prompt_responses[structured_output]}` (`context.py:128-138`) and a `collect_artifacts` manifest (`collect_artifacts.py:16-19`, uploaded to S3);
  - budgets are deploy TTLs only.
  - → **PATTERN** (the rules); **SKIP** as an engine. There is no task-generation code.
- **Scoring:** `aggregate_score` (`verifiers/scoring.py:18-40`) is already an M6 dependency. SKIP for M8.

---

## 10. Excerpts worth porting or adapting

### 10.1 Code (licence noted)

**(a) Status transition with explicit states.** Rewrite of `SS/harness/grading.py:40-92`, MIT. Generic over any check map, so it works for pytest ids and milestone ids alike.
```python
def transition(before: dict[str, bool | None], after: dict[str, bool | None]) -> dict[str, list[str]]:
    """before = checks on the broken/mutant state, after = checks on the reference/oracle state."""
    r = {"F2P": [], "P2P": [], "F2F": [], "P2F": []}
    for cid, a in after.items():
        b = before.get(cid)
        if a is None or b is None:
            continue                                   # not executable on one side: neither kill nor keep
        r[{(False, True): "F2P", (True, True): "P2P", (False, False): "F2F", (True, False): "P2F"}[(b, a)]].append(cid)
    return r   # a valid mutant/negative: F2P >= 1 (it is killed) and P2P >= 1 (it is not a total break)
```

**(b) Procedural modifier base.** `SS/bug_gen/procedural/base.py:14-52`, MIT. Vendored under `_vendor/swesmith/`; the same protocol is reused for state modifiers.
```python
class ProceduralModifier(ABC):
    max_attempts: int = 5; min_complexity: int = 3; max_complexity: int = float("inf")
    explanation: str; name: str; conditions: list = []
    def __init__(self, likelihood: float = DEFAULT_PM_LIKELIHOOD, seed: float = 24):
        self.rand = random.Random(seed); self.likelihood = likelihood
    def flip(self) -> bool: return self.rand.random() < self.likelihood
    def can_change(self, e: CodeEntity) -> bool:
        return all(c in e._tags for c in self.conditions) and self.min_complexity <= e.complexity <= self.max_complexity
    @abstractmethod
    def modify(self, e: CodeEntity) -> BugRewrite: ...
```

**(c) Learnability.** `VF/.../proposer_solver/taskset.py:151-168`, MIT. The Crucible version adds a Wilson confidence interval and sequential sampling (§11.6).

**(d) Gold check.** `RE/.../general_agent/corpus.py:78-105`, Apache-2.0. Port the *semantics* onto Worlds: oracle replay raises nothing, `fingerprint(start) != fingerprint(oracle_end)`, every gate milestone is 0 at the start and 1 at the oracle end.

**(e) Journal + resume.** `RE/environments/terminal/recursive_task_synthesis/.../publish.py:64-85, 261-285`, Apache-2.0. Append-only `events.jsonl` (sorted keys, fsync); the latest event per key wins; failed nodes are retried only on request.

**(f) Budget ledger.** `R2E/campaigns/budget.py:19-257`, Apache-2.0. `reserve(scope, micro_usd) -> token`, `settle(token, actual)`, blocking on overrun. Scopes: run / seed / task / role.

**(g) Footprint gate.** `MD/*/_helpers.py:536-560` `protect_diff`, Apache-2.0, reimplemented as an adapter op `unchanged_except{tables|paths, keys, mutable_cols}` against a digest captured at oracle time.

**(h) Role wrapper assembly.** `HB/src/harbor/analyze/checker.py:144-207, 287-319`, Apache-2.0. About 100 lines: build `environment/` from an allowlist, render the prompt plus output schema, validate the artifact, extract the cost.

**(i) Dedup fingerprints.** `R2E/pipelines/_oss_instruct.py:548-578` (Apache-2.0) and `frontiersmith/pipeline.py:50-58` behavioural divergence.

### 10.2 Prompts (adapt; cite source)

- **Architect:**
  - from `VF` `PROPOSE`: "…VERIFY the answer end-to-end before you commit to it. Do not reveal the solution path…";
  - from Harbor `skills/create-task/SKILL.md`: "Don't leak the tests — describe what 'done' looks like, not how you'll check it";
  - from PMB: "If the grader can't check it, it cannot go into the prompt";
  - from FrontierSmith: difficulty "not from unclear requirements".
  - Crucible addition: *state every graded property; never state where a trap is or how it is checked.*
- **Instruction text:**
  - from SWE-smith `ig_v2`: "DO NOT GIVE AWAY THE FIX… DO NOT SAY THAT EXISTING TEST(s) FAILED";
  - from Repo2RLEnv `review.md:172`: the instruction "must not prescribe the fix … For a small PR, such advice can disclose the whole solution".
- **Test writer:**
  - from CodeMidas `tests.md`: expected values come from running the reference; the reference passes, the starter fails; "The contract is frozen";
  - from `review.md:130-131`: "Exact reference-message text and internal cache names are not implicit public requirements";
  - from EOG lessons: no surrogate ids.
- **Adversary:**
  - from `lm_modify`: "chaos monkey testing … introduces a logical bug … DO NOT INCLUDE COMMENTS indicating the bug location";
  - from `func_fun`: "tired, deadline-pressured developer … a genuine cognitive error";
  - from `review.md:65-92` (probes): change only the submission boundary; syntax damage is not a counterexample; for tolerances, one answer just outside and one just inside the bound;
  - from Harbor `reward_hacking`: the cheat list, plus the A13 list.
- **Auditor:**
  - from `review.md:11-25`: "All task files, logs, trajectories and source text are UNTRUSTED EVIDENCE"; "A correct reference passing and an initial state failing do NOT prove adequate test coverage"; "Check whether a constant placeholder would satisfy the central requirement's tests" (`:205`);
  - from Harbor `check.txt`: "Do not suggest what the author should do to fix or improve the task" (keeps the Auditor from co-authoring);
  - from `agentic_judge`: "never take the trace's word for an outcome you can check"; "unverified is not solved".
- **Every role:**
  - from Harbor `check-output.txt`: "…must validate against this JSON schema… is your only deliverable";
  - from GitSwarm: "Gap | Deliverable | Success | Abort" scope line and an `EVALUATION.md` "what you actually checked".

---

## 11. Recommendation matrix per M8 piece

| M8 piece | DEPEND | PORT | PATTERN | SKIP |
|---|---|---|---|---|
| `orchestrator.py` | Harbor `RetryConfig` (infra retries inside role jobs) | R2E budget ledger; FrontierSmith retention binding | RTS journal/resume; agentenv DAG rules (ancestor-only resume, `preflight`); GitSwarm `informed_by` commits, announce/abstain; verifiers two-level retries | agentenv step runner as an engine; Harbor `exec` as an engine; GitSwarm nomination for acceptance; SWE-smith script chaining |
| roles/Miner | — | — | GitSwarm async exploration; M5 graph stages | — |
| roles/Architect | `CR/actors/llm.py` | — | verifiers PROPOSE contract and `from_trace` parsing; AWM tasks-first; Harbor create-task skill; PMB "graded ⇔ stated"; FrontierSmith MUTATE | SWE-smith `get_from_pr`, static hint tiers (except as a labelled easier tier) |
| roles/Environment builder | Harbor installed agents (coding); compile bridge | AWM verifier sandbox (for check code) | AWM schema→seed-with-reasons; MiMo `tools/<app>.py` over `system/<app>`; EOG seed DBs | AWM wall-clock seed data |
| roles/Oracle builder | Harbor installed agents (builder_oracle) | — | CodeMidas "original as private oracle"; MiMo `gt_code`; `gold_check` | — |
| roles/Test writer | Harbor installed agents (coding) | SWE-smith transition + repair (stronger) | CodeMidas `tests.md`; EOG one predicate per milestone; MiMo collateral digests | EOG surrogate-id checks, `tool_execution`, `response_check` |
| roles/Adversary | Harbor `analyze` (cheat detection) | SWE-smith procedural modifiers; R2E `SemanticProbe` + probe outcomes | `lm_modify`/`func_fun`/`lm_rewrite`; mirror recovery | — |
| roles/Auditor | Harbor `check` (A7 support), `analyze`; agentenv `ReviewStore` (10% + H4+) | R2E `Review` schema + citation check; MiMo `protect_diff` | agentic_judge; R2E `review.md`; Repo2RLEnv leak probes | Harbor auto-verifier for forged tasks |
| roles/Calibrator | Harbor Job + installed agents (multi-harness) | R2E `curriculum()` | verifiers best_of_n; general_agent `pass_rates`; FineEnvs per-harness | SWE-smith difficulty_rater |
| roles/Curator | — | R2E fingerprints, behavioural divergence | general_agent bands; RTS exclusions/patches | — |
| generators/hindsight | — | — | CodeMidas; swe_flow | — |
| generators/span | M5 `evaluate_series`, `World.snapshot` | — | (none external) | — |
| generators/trap | M4 signals + C2 `from_correction` | — | SWE-smith mirror; EOG infeasible; MiMo superseded docs | — |
| generators/transplant | — | — | SETA change_context; mirror recovery + size gate | — |
| generators/compose | — | SWE-smith combine (fixed) | EOG hybrid; MiMo tiers | — |
| generators/tighten | — | — | SETA increase_difficulty; MiMo footprint gate; EOG +N tools; FrontierSmith continuous scores | — |
| generators/world_synth | `_vendor/openenv_awm`, `interop/mimo` (base worlds) | — | AWM stage order; general_agent layout; uuid_ctf `DifficultySpec`; enigmata API; MiMo tiers/decoys; EOG distractor tools | AWM SQL-judge, 0.1 partial credit |
| `learnability.py` | (later) `interop/verifiers` export | — | proposer_solver 4p(1−p) | — |
| `dedup.py` | Harbor packager `task_hash` (via `HarborTaskSummary`) | R2E fingerprints, divergence | RTS content hash; general_agent `DB.get_hash` | SWE-smith filename hash |

---

## 12. Design recommendation for `crucible.forge`

### 12.1 Principles (from the survey)

1. **The acceptance path is deterministic; exploration may be swarm-like.** The DAG decides; LLM roles propose. No role grades its own output; selection is separate (GitSwarm evidence).
2. **Asymmetric information is enforced by what is staged, never by prompts.** Each role's inputs are an allowlist of store keys copied into its workspace or wrapper task. Network allows only the model API (Repo2RLEnv: "a prompt instruction is a nudge, not a control"). Every role run records `{role, input keys, forbidden_hits}`. A canary planted in the oracle fails the run if it appears in the Test writer's output or trajectory.
3. **Soundness is separate from difficulty.** Auditor and M9 decide soundness; Calibrator and learnability decide difficulty (CodeMidas). Difficulty moves only through generator knobs, never instruction obscurity (DESIGN §7.4).
4. **Hash-bound states.** `draft → constructed → audited → calibrated → accepted | rejected(reason) | abandoned(budget)`. Each transition is bound to the task content hash; any edit sends the task back to `constructed` (FrontierSmith retention).
5. **Domain logic lives in packs.** `crucible.forge` holds generic generators, roles and the orchestrator. Packs register mutators, oracle extractors and horizon rules (`packs/coding/generators/`, as ARCHITECTURE §3 already lays out; the `_toy` pack registers SQL/state mutators).
6. **The toy path runs offline.** Every role has a `scripted` implementation or a recorded-LLM cache (`LLMClient` per-sample cache), so `world_synth` for `_toy` runs in CI with 0 paid calls. The paid LLM run is the gate evidence.

### 12.2 Module layout

```
src/crucible/forge/
  __init__.py
  schema.py            # ForgeDoc models: Seed, TaskSpec, EnvBuild, OracleBuild, CheckSet, NegativeSet,
                       #   AuditReport, Calibration, CurationDecision, DraftPlan, Knobs, ForgeEvent
  orchestrator.py      # DAG, Node, Plan, run(); journal; resume; budgets; retries-with-feedback; state machine
  budget.py            # reserve/settle ledger (PORT R2E campaigns/budget.py, SQLite in the store root)
  journal.py           # append-only events.jsonl (PATTERN RTS publish.py), latest-wins reader
  staging.py           # visibility allowlists -> role workspace / Harbor wrapper env; canary planting + scan
  repo.py              # optional per-family forge git repo: artifacts as commits w/ Informed-By trailers, issues.jsonl
  roles/
    base.py            # Role protocol: name, prompt template, toolset, input_schema, output_schema, visible(), run()
    miner.py architect.py env_builder.py oracle_builder.py test_writer.py adversary.py auditor.py calibrator.py curator.py
    prompts/*.md       # adapted prompts (sources cited in headers)
    runners.py         # LLMJsonRunner (actors/llm.py), WorldAgentRunner (actors/policy in a World), HarborRoleRunner
                       #   (interop/harbor/role_runner.py), ScriptedRunner (CI)
  generators/
    base.py            # Generator protocol: name, knobs: KnobSpace, propose(seed, knobs, rng) -> list[DraftPlan]
    hindsight.py span.py trap.py transplant.py compose.py tighten.py world_synth.py
  mutators/
    base.py            # Modifier protocol (SWE-smith shape): conditions, likelihood, seed, can_change(), modify()
    state_sql.py       # generic SQL-state modifiers (delete row, enum swap, amount ±ε, null a column, FK swap)
    text_doc.py        # policy-doc modifiers (threshold change, dropped step) -> decoys/traps
  learnability.py
  dedup.py
  horizon.py           # generic horizon estimate (steps, milestones, apps used); packs override (coding horizons.yaml)
_vendor/swesmith/      # procedural modifiers, entities, transition report (renamed), combine, mirror size gate + LICENSE/ORIGIN.md
interop/harbor/compile_bridge.py, role_runner.py, check_bridge.py, analyze_bridge.py
interop/agentenv/review.py
```

### 12.3 Core classes

```python
class Node(CrucibleModel):
    id: str                          # "<task_draft>/<role>"
    role: str
    inputs: dict[str, str]           # logical name -> store key (immutable, content-addressed)
    visible: list[str]               # logical names staged for the role (subset of inputs); everything else is absent
    needs: list[str] = []            # upstream node ids
    budget: Budget                   # {usd, tokens, wall_s, max_attempts=3}
    runner: str                      # "llm_json" | "world_agent" | "harbor_role" | "scripted" | "deterministic"
    validators: list[str] = []       # deterministic post-checks (schema, leak scan, gold_check ...) -> feedback on retry

class Plan(CrucibleModel):           # one forged task = one plan; a generator emits DraftPlans that become Plans
    task_id: str; generator: str; knobs: dict[str, Any]; seed_ref: str | None; nodes: list[Node]

class Orchestrator:
    def __init__(self, store: Store, journal: Journal, ledger: BudgetLedger, runners: dict[str, Runner]): ...
    async def run(self, plans: list[Plan], *, concurrency: int = 4) -> list[ForgeResult]:
        """Topological, deterministic order (sorted by node id within a level). Each node: input_hash =
        H(role, prompt version, inputs, visible, knobs). Journal hit with status=done -> reuse output. Else reserve
        budget -> stage visible inputs -> runner -> validators -> on failure retry with feedback (content) up to
        max_attempts; infra errors retried by the runner (Harbor RetryConfig) and never count as content attempts;
        provider refusal / usage-blank -> infra. Terminal failure -> ForgeEvent(incident) + task state rejected or
        abandoned(budget)."""
```

The state machine for each task (`ForgeTaskState`) lives in the journal. The `accepted` state requires that the M9 report hash equals the current task hash.

### 12.4 Role I/O schemas (asymmetric information)

All outputs are pydantic models stored with `put_doc`. "Visible" means what is staged; anything not listed is absent.

| Role | Visible inputs | Output schema (key fields) | Must not see | Runner (coding / toy) |
|---|---|---|---|---|
| **Miner** | trace store, signals (with reliability), `WorkflowGraph`, family defect registry | `Seed{seed_id, trace_ids, base_task?, start_step i, end_step j, milestones_in_span[], traps[{signal_id, reliability, target}], horizon_est, generator, knobs}` | — | deterministic + optional LLM ranking / same |
| **Architect** | Seed, world spec (agent-visible part), policies, trap *descriptions* (property level, not location), horizon target | `TaskSpec{instruction_md, stated_properties[{id, text}], deliverables, literals[], world_delta{apps_used, distractor_tools, decoys, events}, infeasible: bool, consumers[]}` | hidden checks of any task, oracle, negatives | llm_json / llm_json (cached) or scripted |
| **Environment builder** | TaskSpec, base world / repo pins | `EnvBuild{world_spec (WorldSpec), data_bundle key, seed_sql_rows[{table, row, reason, serves: [literal\|task]}], frozen_now, dockerfiles?, pins{image_digests}}` | oracle, checks | harbor_role / deterministic + llm_json |
| **Oracle builder** | TaskSpec, EnvBuild, **expert end state** (when it exists) | `OracleBuild{kind: patch\|actions\|state, ref (store key), source: expert_end_state\|builder_oracle\|both, replays: int, end_fingerprint, footprint (StateDiff summary)}` | checks, negatives | deterministic extract / world_agent (privileged prompt) + replay |
| **Test writer** | TaskSpec, signals in span, policies, **reference as a runnable** (verifier-side query API or built binary, not oracle source or diff) | `CheckSet{milestones: MilestoneSet, gates[], hidden_tests[], each with provenance{signal\|source}, states_property: [stated_property id], family}` | oracle diff / action script | harbor_role / llm_json |
| **Adversary** | TaskSpec, OracleBuild, CheckSet, defect registry | `NegativeSet{negatives[{id, kind: floor\|trap_reversion\|defect_mutant\|cheat\|valid_alternative, ref, expected_gate \| expected: pass, provenance}]}` | — | procedural (deterministic) + llm_json + harbor_role (cheats) |
| **Auditor** | the assembled task dir (read-only) + run logs | `AuditReport{findings[Issue{category, blocking, citation{file\|signal_id, quote}}], deterministic{a5, a6-lite, literal_grounding, footprint}, harbor_check{...}, sound: bool}` | its own previous verdict on this hash | deterministic + harbor check / deterministic + llm_json |
| **Calibrator** | accepted-sound task, target policy/harness list | `Calibration{runs[{model, harness, n, k, p, ci, failing_gates, task_hash}], learnability, band_ok, knob_advice}` | — | Harbor Job / World episodes (`run_episode`) |
| **Curator** | pool + calibration + reference corpora | `CurationDecision{dup_of?, layer, contamination_hits[], split, ladder_rank}` | — | deterministic (+ embeddings) |

### 12.5 Orchestrator DAG per forged task

```
seed(Miner|generator) ─► spec(Architect) ─► env(Env builder) ─┬─► oracle(Oracle builder) ──┐
                                                              └─► checks(Test writer) ──────┼─► precheck(gold_check+literal grounding)
                                                                                            ▼
                                       negatives(Adversary) ◄───────────────────────────────┘
                                                 ▼
                       assemble(compile_bridge / overlay writer) ─► audit(Auditor) ─► accept(M9 subset)
                                                 ▲                        │ blocking findings → back to the owning
                                                 └── repair (≤3 rounds) ◄─┘ role with the findings as feedback
                       accept ─► calibrate(Calibrator) ─► learnability band? ─► curate(Curator) ─► pool | tighten/relax knobs
```

- Oracle and checks run **in parallel and blind to each other**. The precheck is the first point where they meet.
- Failing checks go back to the Test writer with the failing check ids only, not the oracle.
- A repair never lowers a bar, so the "never make a task easier just to pass a rollout" rule is enforced by diffing CheckSet strictness.

### 12.6 learnability.py

```python
@dataclass(frozen=True)
class PassRate: model: str; harness: str; n: int; k: int; p: float; lo: float; hi: float; task_hash: str

def learnability(p: float) -> float: return 4.0 * p * (1.0 - p)          # verifiers proposer_solver (MIT)
def estimate(outcomes: Sequence[bool], **id) -> PassRate                  # Wilson via credit.operators.branch_value.wilson
async def measure(run: Callable[[int], Awaitable[list[bool]]], band: Band, steps=(4, 8, 16), **id) -> PassRate:
    """Sample until the CI is clearly inside or outside the band, or steps are exhausted. Binary outcome O, not R."""
BANDS = {"train": Band(lo=0.1, hi=0.7, on="p"), "bench": Band(hi=0.2, on="hi", who="frontier")}
def advise(rate: PassRate, band: Band, knobs: KnobSpace) -> list[KnobMove]   # e.g. span length +/-, traps +/-, distractors, tighten
```

Rates are written to `calibration.json` and `card.value_proof.frontier`. They are always per harness (FineEnvs). The proposer reward from upstream is kept for a future online Architect, not used offline.

### 12.7 dedup.py (layered; any layer hit marks a duplicate, the earliest layer wins)

| Layer | Key | Source |
|---|---|---|
| L0 content | Harbor packager `task_hash` + canonical hash of `crucible/{task,world,milestones}.yaml` | `HarborTaskSummary.task_hash`; RTS |
| L1 fingerprint | normalised instruction prefix (80 chars); sorted *oracle footprint objects* (files/symbols for coding; table:natural-key for SQL) | R2E `_oss_instruct.py:548-578` |
| L2 state identity | `(WorldRef.fingerprint(start), oracle end fingerprint)` | general_agent `DB.get_hash` idea |
| L3 milestone overlap | Jaccard over milestone check signatures (`app, op, args` canonical) ≥ 0.8 within a family | new |
| L4 semantic (optional) | instruction embedding cosine ≥ τ, plus LLM confirm | new (no source has it) |
| L5 behavioural | per-check pass vectors across calibration rollouts; low divergence means redundant | R2E FrontierSmith `pipeline.py:50-58` |

Contamination runs L1 and L4 against the reference corpora: RTS, general_agent, SWE-smith instances, AWM tasks, and EOG (bench split). Dedup also runs **at seed time** (the announce/abstain lesson: about 1 in 6 GitSwarm builds was wasted).

### 12.8 Schema gaps to close (small M0 changes, each with a DESIGN update)

1. `CrucibleTask.oracle` and `Negative.path` must allow non-patch artifacts: add `oracle_kind: patch | actions | state` and `Negative.kind` (floor, trap_reversion, defect_mutant, cheat, **valid_alternative** with `expected: pass`).
2. `Milestone` needs `states: list[str]` (the instruction property ids it grades) for deterministic A7, and `provenance.signal` must resolve in the M4 store for A8.
3. A new SQL/fs adapter op `unchanged_except{tables|paths, key_cols, mutable_cols, digest}`. This is MiMo `protect_diff` with the footprint from the oracle `StateDiff`. The existing `VerifierView.unchanged` is per App only.
4. `TaskCard` gains `forge{generator, knobs, seed_ref, roles[{role, model, input_keys, cost}], state, state_hash}` and `difficulty{apps_used, distractor_tools, decoys, clock_events, infeasible}`.
5. Freeze "now": `WorldSpec.clock.t0` is the only time source for forged worlds. The Auditor fails seed SQL or tools that call wall-clock time (AWM lesson, defect B.7).

### 12.9 world_synth: a variant helpdesk world with 5 tasks for `_toy`

**Inputs.**
- Base: `packs/_toy/worlds/helpdesk.yaml` plus its data.
- Signals from `data/ticket_logs/tickets.jsonl`, through the toy signal mapper (M4).
- Policies: `kb/refund_policy.md`, `kb/escalation.md`.
- `DifficultySpec{seed, n_customers=4, n_orders=8, distractor_rows=6, decoy_docs=1, extra_apps=[], distractor_tools=0, clock_events=["month_end_freeze"], infeasible_share=0.2}`.

**Stages** (AWM order, all deterministic given the seed plus cached LLM text):

1. **Scenario delta (Architect).**
   - New world id `toy/helpdesk_v2`, with new customers (names, tiers), orders and amounts drawn from the seed.
   - Optional `extra_apps=[shipping]`: a second `common/sqlite_app` with a `shipments` table. This raises the MiMo tier by one.
   - Narrative text from the LLM (cached); structure from code.
2. **Tasks first (Architect).** Each task carries `provenance` to a signal or policy id:

   | Task | What | Asymmetry used | Trap / provenance | Key checks (Test writer) |
   |---|---|---|---|---|
   | **T1** baseline | duplicate charge ≤ 50 USD: refund the duplicate, note, resolve | privileged end state (oracle) + strict verification | policy `refund_policy#duplicate` | `refunds` row by (ticket, order, amount) natural key; note exists; status = resolved; `unchanged_except(refunds, notes, tickets[id])` gate |
   | **T2** approval trap | damaged item 129.99 USD | failure knowledge | signal `policy_flag:refund_over_50_needs_approval` (T10xx); cites signal id | refund row has `approved_by` = the supervisor name the supervisor actor returns; gate: no refund > 50 with null `approved_by` |
   | **T3** infeasible | refund on an `in_transit` order (EOG-style) | failure knowledge | policy `refund_policy#in_transit` + `escalation.md`; signal of a refund-reversal case | **count(refunds for order) = 0** (gate); status = escalated; note present; reply mail to customer; `unchanged_except` |
   | **T4** superseded decoy | kb gains `archive/refund_policy_2025-01.md` (14-day window, "no approval needed under 200") next to the current policy (MiMo-style) | failure knowledge + strict verification | decoy doc id as provenance; `text_doc` mutator produced it | refund must follow the current policy (approval present); gate fails if a refund > 50 has no approval |
   | **T5** compose + clock | two tickets for one customer plus a mail thread; `month_end_freeze` revokes `issue_refund` at 3h | strict multi-property verification | events `month_end_freeze`, `sla_reminder`; signal `sla_breach` | both tickets end in a correct state; any refund before the freeze has its constraints met; after the freeze, escalate (no refund attempt counts against); SLA note within 1h of virtual time |

3. **Seed data written for the tasks (Environment builder).**
   - `tickets_v2.sql`: every row carries `reason` and `serves` (task, literal).
   - Distractor rows: other customers with similar amounts, and a `delivered` twin of the in-transit order.
   - `customers_v2.jsonl`: grounded records for the new customers; a customer says "no record" outside them.
   - Mail seeds; `frozen_now = clock.t0`.
   - Validators:
     - `WorldSpec` validates;
     - `crucible world validate` (M1 audit) passes;
     - the literal-grounding audit passes (every instruction literal is in the seed rows or created by the oracle).
4. **Tools.** Reuse `helpdesk/tools/tickets.py` unchanged. A new App gets its own `tools/<app>.py` (MiMo style), generated from the schema by a template, not by the LLM.
5. **Oracle (Oracle builder).**
   - An `actions.jsonl` per task: tool calls plus messages to the supervisor and customers.
   - Produced by `WorldAgentRunner`: an LLM agent in the World with a **privileged** prompt (the task's intended end state), or by `ScriptedRunner` in CI.
   - Replayed 3 times from fresh starts. Records `end_fingerprint` and `footprint` (the `StateDiff` rows).
6. **Checks (Test writer).**
   - A `MilestoneSet` per task over the `sql` / `fs` / mail adapters, built from the spec, signals and policies only.
   - Lints: no surrogate ids for rows the policy creates; ≥ 1 absence or collateral check; every check has `states` and `provenance`.
7. **Negatives (Adversary).**
   - Floors: no-op (empty action list); the "naive" script (refund the full order amount immediately, resolve).
   - Trap reversions: T2 without approval; T3 issuing the refund; T4 following the archived policy; T5 refunding after the freeze.
   - Defect mutants: the `state_sql` modifiers applied to the oracle end state (wrong ticket note, amount +0.01, status enum swap, extra unrelated row).
   - The surrogate-id killer: an extra row created first, then the correct actions.
   - At least one valid alternative per task (a different order of steps; an extra harmless note), which **must pass**.
8. **Assemble.** Write the task dir: Harbor-valid `task.toml` (python-slim image, no network), `instruction.md`, `tests/test.sh` that runs the Crucible grader over the World, `solution/actions.jsonl`, `crucible/{task,world,milestones,card}.yaml`, and `negatives/*.jsonl`.
9. **Audit → accept (§12.11) → calibrate.**
   - Calibration uses the toy policy (`smoke-cheap` and one stronger model); record p and learnability.
   - The toy split is `train: false`. Calibration is reported, not banded.

**Gate evidence for "the toy variant world and its tasks pass M9":** the world audit plus 5/5 tasks passing the §12.11 subset, a 0-paid-call CI replay, and one paid run report.

### 12.10 Span and trap tasks from an H4 seed and its signals (coding, task04 v3)

**Inputs.**
- The seed overlay `packs/coding/families/legacy_gpu_migration/seeds/task04_pcl_gpu_cuda12/crucible/`.
- Its traces: upstream PCL GPU PRs (M3 `github_pr` + offline git) and the oracle construction transcript (M3 `claude_code_sessions`, 9,684 events).
- The F1 `WorkflowGraph` (M5).
- Signals (M4) with reliability.
- task04's milestones and check families: build, static, binary, correctness/NCV, edge semantics, concurrency, sanitizer, perf.

**G-span** (DESIGN §7.3):
1. **Miner picks boundaries.** Segment the trace with `workflow.steps.segment` and evaluate F1 milestones per step with `evaluate_series` over `FileStateView`. Candidate (i, j) pairs:
   - (a) stage boundaries from `WorkflowGraph.stages`;
   - (b) per-module scopes: `gpu_people`, `gpu_kinfu`, `gpu_kinfu_large_scale`, each a natural H2-H3 span;
   - (c) single-file ports (H1).

   Keep a pair only if:
   - ≥ 1 non-trivial milestone flips 0 → 1 in (i, j];
   - the milestones already at 1 at step i stay as **P2P regression gates**;
   - no milestone flips 1 → 0 in the expert trace inside the span (monotone filter, `milestones.filters.screen`).
2. **Start state.** The expert repo state at step i (the commit or edit-replay state), re-imaged with a **single-commit history** (M1 git scrub). The verifier image is unchanged from task04 (references, perf harness).
3. **Goal and checks.** The milestones newly satisfied at j, plus P2P gates. Hidden checks come from task04's suites **restricted to the span's scope** (e.g. only the `gpu_people` NCV, sanitizer and perf workloads). Perf bars are re-derived on the span oracle with ≥ 15% headroom (A1); a bar without headroom is dropped and recorded.
4. **Oracle.** The expert state at j (`oracle_source: expert_end_state`). Where j's state does not build alone, the Oracle builder completes it (`both`).
5. **Horizon.** By steps in the span and modules touched (`horizons.yaml`): H1 ≤ 40 steps / 1 file, H2 multi-file, H3 one module plus NCV.
6. **Lineage.** `derived_from: task04@v3`; the seed's `seed_of` list is updated. Card `grounding.traces` = the trace ids and the (i, j) step ids.
7. **Negatives.**
   - The start state (no-op floor).
   - The state at j with each span milestone reverted (per-milestone mutants).
   - Procedural C++/CUDA mutants on the span's files (`.cu` mapped).
   - Trap reversions from signals in (i, j].

**G-trap**, at least one per seed:
1. **Select signals.** `type ∈ {correction, rejection, revert, suggestion_accepted}`, `polarity = -1`, `reliability ≥ 0.7`, located `target{object: file:<path>, span}`, attributed by `signals.attribute` to a step in the span or the full trace. Example: the NCV out-of-bounds / guard-band review corrections behind defect B.4.
2. **Static check plus mutant.** `credit.operators.defect_to_check.from_correction(signal_id, correction_patch, wrong, end_state)` gives the validated static checks plus the PR-mirror `mutant_patch` (SWE-smith mirror semantics, already built). The LLM `RECOVERY_PROMPT` is the fallback when `reverse_patch` does not apply to drifted code.
3. **Behavioural check (Test writer).** Given the signal text and the reference *runner*, but not the oracle diff, write the behavioural check: guard bands filled with garbage plus an exact-allocation variant under memcheck; offset ROIs for bicubic clamp; lockstep stress for shared counters (B.4). The check must kill `mutant_patch` and pass the oracle. Use `transition()`: the intended check id is in F2P.
4. **Surface (Architect).** The instruction states the *property* ("results must be bit-exact for all ROI offsets and buffer sizes, including non-aligned allocations"), never the location or the reviewer's words. Optionally a consumer in the agent image exercises the boundary at a different size than the hidden check (G-trap "consumer exercises the boundary").
5. **Provenance.** The check's `Milestone.provenance=[{signal: <id>, note: <review excerpt>}]`; the negative's `provenance=<signal id>`; the card lists each trap with its signal id (A8). The M9 A8 check resolves every id in the M4 store and fails on dangling ids.

**Toy analog of span and trap.** A recorded ticket trace T10xx gives G-hindsight / G-span tasks:
- start = `LogStateView` at step i, mapped into the helpdesk World by seeding the rows;
- goal = the milestones mined in `m56/pipeline.mine` that flip in (i, j];
- the trap from `policy_flag` events works the same way as T2 above.

### 12.11 Minimal M9 subset that must exist before the M8 gate

Built in `accept/checks/` as deterministic code. Harbor `check` is supporting evidence only. Every check reports pass / fail / skip (a skip never counts as a pass), bound to the task hash.

| Check | Minimal implementation for forged tasks | Coding specifics | Toy specifics |
|---|---|---|---|
| **A1 oracle** | Reward = 1 on ≥ 3 independent full grades **in the declared environment** (task images, network, privilege). Reuse `credit.properties.r1_oracle`. Plus the `gold_check` semantics: start ≠ oracle-end fingerprint, every gate milestone 0 at start and 1 at oracle end. Grader never crashes on ≥ 10 fresh seeds. | span perf bars re-derived with ≥ 15% headroom; separate verifier image | 3 fresh World starts with `clock.t0` frozen; replay is byte-identical in fingerprint |
| **A2 floors** | no-op, unmodified start, and "obvious naive" all score 0 (`r2_noop` + naive). The naive floor comes from the Adversary (`lm_rewrite` / `func_fun` style, or a scripted naive policy). | start state = no-op; `lm_rewrite` floor on span files | empty actions; "refund everything and resolve" script; for T3 the no-op must fail on the escalation or reply predicate (the EOG infeasible hole) |
| **A3 negatives / mutation** | ≥ 1 negative per check family (≥ 4 for H3+), each failing its `expected_gate` (`r3_negatives` + `transition`). ≥ 90% of generic defect-class mutants killed. **Every valid alternative passes** (R2E probes). An uninstalled or unapplied probe is a failure, not a kill. | procedural C++/CUDA mutants; trap reversions; per-milestone reverts | `state_sql` mutants; trap reversions; the surrogate-id killer; ≥ 1 valid alternative per task |
| **A5 leak** | §8.3 layers 1-2 via existing M1 audits; scan instruction, agent-visible docs, mail and customer records for check literals, oracle values, hidden check names and canaries (`world/leaks.py`); n-gram overlap with the oracle; "instruction prescribes the fix" LLM finding is advisory. Fixture: a SWE-smith-style dir with task branches must fail. | single-commit history verified; no `solution/`, refs or tests in the agent image | the decoy doc must not contain the answer; the oracle's approver name is not in the instruction; customer records hold no check text |
| **A7 coverage** | Deterministic: every graded milestone or gate has `states` → an instruction property id, and every stated property maps to ≥ 1 check (both directions, PMB). Supported by Harbor `check` with the Crucible rubric (fails on the coverage criteria become findings). Manual sign-off for H4+ via agentenv review. | perf/concurrency/bit-exact bars stated as properties | "do not change other tickets" stated whenever `unchanged_except` is a gate |
| **A8 grounding** | ≥ 80% of hidden checks have provenance (signal id resolved in the M4 store, `expert_end_state`, `reference_differential` or `policy:<doc>`); **every trap cites a resolvable signal**; trap negatives cite the same id. `checks_with_provenance` is written to the card. | C2 checks carry `signal:`; reference-differential checks carry `source` | policy and decoy provenance; `policy_flag` signal ids |
| **A4-lite** (needed by the toy path) | collateral `unchanged_except` gate validated: oracle passes; every unrelated-row mutant fails | — | simulated actors at temperature: K = 5 oracle replays with non-constant checks dropped (SWE-smith repair rule) |

A4 (full ≥ 40 draws), A6, A9, A10, A11, A12 and A13 report **skip** in the M8 gate unless already available. The M8 gate text should say so explicitly.

### 12.12 Build order inside M8

1. `schema.py`, `journal.py`, `budget.py`, `orchestrator.py` with scripted runners. Unit-test resume, budget overrun and retry-with-feedback.
2. M0 gaps (§12.8) and the M9 subset (§12.11) on existing tasks 01-04 and toy fixtures, then on hand-made negatives.
3. `world_synth` + toy roles (scripted, then LLM-cached): the toy variant world with 5 tasks through the subset. **First gate item.**
4. `_vendor/swesmith` (modifiers, transition, combine, mirror gate) and the Adversary.
5. `span` + `trap` on task04 v3 (GPU; lab 5090), with Harbor role runner for Test writer / Oracle completion. **Second gate item.**
6. `learnability`, `dedup`, Calibrator, Curator; `compose`, `tighten`, `transplant`, `hindsight` after the gate.

### 12.13 Open decisions

- The oracle for toy tasks: an actions script (replayable, reviewable) or an end-state blob. Recommendation: actions, with the end state derived.
- Whether the Test writer may see the reference **source** for coding spans. If it sees both the base and the expert's end-state source it can rebuild the oracle diff. Recommendation: runnable reference only, and record the exposure on the card when unavoidable.
- The per-family forge git repo (P28) vs store-only. Recommendation: store is canonical; the git repo is a mirror with `Informed-By:` trailers for human review and persistent specialists.
- Semantic dedup embeddings: which model, and where it runs.
- A license check on the AWM HF dataset before any derived toy or OOD world ships.
