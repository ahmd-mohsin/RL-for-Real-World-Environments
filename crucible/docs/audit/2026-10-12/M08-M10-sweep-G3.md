# M8 / M9 / M10 framework sweep, group 3: NeMo-Gym, NeMo-RL, EnterpriseOps-Gym, Karotte, MiMo (verl fork + v2.6 release)

Read-only sweep at Crucible HEAD `526fdb5` (2026-10-12). Every Crucible citation was opened. External citations were opened by me or by two read-only sub-agents. I re-opened the load-bearing ones with `sed -n` (marked ✓).

Path roots:
- `NG` = `RL-Environments/06-frameworks-and-env-code/nvidia-nemo-gym/` (Apache-2.0)
- `NR` = `RL-Environments/06-frameworks-and-env-code/infra-nvidia-nemo-rl/` (Apache-2.0)
- `EOG` = `RL-Environments/06-frameworks-and-env-code/servicenow-enterpriseops-gym/` (Apache-2.0; tasks are data only)
- `KA` = `RL-Environments/06-frameworks-and-env-code/preferencemodel-karotte/src/karotte/` (MIT); `KAD` = its `docs/`
- `XV` = `RL-Environments/06-frameworks-and-env-code/xiaomimimo-verl/` (Apache-2.0); `MA` = `XV/third_party/mimoagent-osr/src/mimoagent/` (MIT, Xiaomi + mini-swe-agent authors)
- `ME` = `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/` (README front matter says `license: apache-2.0`; there is no LICENSE file and no split between code and data). `S` = `ME/general/envs/s3k_0000_accounting_audit_tax_en_t1_rl_008/`
- `C` = `crucible/src/crucible/`

Probes live in `scratchpad/m810sweep/g3/` (`p1`..`p9`). They are run with `crucible/.venv/bin/python` from `crucible/`. `[ran]` means a probe reproduced it. `[read]` means I found it by reading code only.

Verdicts: **DEPEND** (use their code or contract as is), **PORT** (copy and adapt; license given), **PATTERN** (reimplement the idea), **nothing** (Crucible is already as good or better), **anti-pattern** (do not copy).

---

## 0. Corrections and confirmations of the prior Crucible studies

| Prior claim | What the code says now |
|---|---|
| `docs/research/improvements/M09-M11.md:38-39`: "Harness exceptions are not contained. `run.py:139` has no try/except." | **Still open.** `C/runtime/run.py:139` calls `harness.run` bare. `NativeHarness.run` catches only `StageFailure` (`C/runtime/harness/native.py:289`). `save_episode`, `rule_flags` and `attempt.task.broken` run outside any `boundary()` (`native.py:257-271`). `[ran p1]`: one `KeyError` makes `Runner.run` raise. No `RunSummary` is returned, and that attempt's ledger op stays `reserved`. |
| `M09-M11.md:40-46`: no per-attempt deadline, immediate retries, one global semaphore held across retries, no run lock, health rules run-global only | **Still open.** The `while True` retry loop sits inside `async with sem` (`run.py:118-157`). There is no backoff and no `asyncio.timeout`. Health is one global `Health` object (`run.py:95`; `C/runtime/infra.py:147-171`). |
| `M09-M11.md:47`: "`over_uncertain_tolerance` is never called; `release()` is unused" | **Still open** (`C/runtime/budget.py:61-74`). `Failure.delivery` exists (`C/core/schema/run.py:110`), but nothing maps `not_sent` to `release`. |
| `M09-M11.md:50`: checkpoints carry no identity | **Still open.** `checkpoint.json` holds step, `WorldRef`, log length and policy/observer state, but no spec hash, code version or blob digests (`C/runtime/checkpoint.py:40-46`). |
| `M09-M11.md:33`, item 9: grade-store writes are non-atomic | **Still open** (`C/accept/checks/base.py:107-110`, plain `write_text`). |
| `M09-M11.md:30`, item 6: `accept/run.py` is serial | **Still open** (`C/accept/run.py:45-71`). |
| `M09-M11.md:26`, item 2: a cost-gated short circuit | **Partly there.** Expensive checks depend on A1, so they go `pending` when A1 fails (`C/accept/run.py:56-57`). But A3 and A4 (GPU) still run after A2 (no-op floor) fails, because they do not depend on A2 (`C/accept/checks/core.py:73,88,154`). |
| `M09-M11.md:58`, item 2, and `docs/research/m9-m10/02-runtime.md`: "PORT NeMo-Gym walltime drain from `rollout_collection.py:1140-1225`" | **Line range moved.** The drain is `dispatch_budget_s`, `NG/nemo_gym/rollout_collection.py:1543-1565` ✓. It is a dispatch deadline plus a p75 margin. NG has **no per-rollout timeout and no in-process backoff** in the collector; a retry happens only on resume (`:903-914`, `:2236-2300`). Do not cite it as the source of a per-attempt deadline. |
| `M09-M11.md:62`, item 6: "Circuit breaker per task and per harness (NeMo-Gym `rollout_health.py:315`)" | **Mis-cited.** `NG/nemo_gym/rollout_health.py` is a post-run JSONL audit ("Deterministic post-run rollout quality verification", `:4-8`). It has no circuit breaker. `:315-357` are the per-task checks `task_consistently_unhealthy` (≥ 2 repeats, all unhealthy) and `task_no_successful_model_calls`. Nothing stops dispatch. A circuit breaker is our own design. |
| `M09-M11.md:68`, item 12: EOG `ray_experiment_queue.py:36-68` as a multi-host scheduler | **It is a keyed resource pool, not a scheduler.** Ray custom resources `llm_<name>` = `num_llm_instances` and `domain_<d>` = 1 (`EOG/ray_experiment_queue.py:36-50,68-70`). Each mode is one `subprocess.run(..., timeout=3600*5)` (`:144-180`). Take it as the pattern for keyed concurrency pools (item 7), not for multi-host. |
| `docs/research/m9-m10/02-runtime.md:629-636` (Karotte): "PORT the per-limit `error\|score` policy, which becomes our `LimitPolicy`" | **Confirmed, with one gap.** Karotte's default is `error` (`KA/schemas/evaluation_run_config.py:52,67`). A limit raises (`KA/agents/message_loop.py:88-109`). The run ends `TaskCompletedEvent(status="error")` with no `ScoringEvent` (`KA/evaluation_runner.py:88-103`), and nothing retries it. Crucible's `mask` mode carries `blame="harness"`, so `RetryRule` re-runs the agent twice more (`C/runtime/harness/native.py:272-273`, `:187-188`; `C/runtime/infra.py:93-96`) `[ran p6]`. Karotte never re-samples a limit stop. See defect D6. |
| `m9-m10/01-acceptance.md:516-520` and `m1-taxonomy/04…md:524-525`: "Karotte: misbehavior = 0 vs ErrorEvent" as the infra/policy split | **Incomplete.** Karotte's `ExecutableJudge` scores a **grader crash** 0: a non-zero exit, a missing executable and unreadable output each return `Scoring(0, …, False)` (`KA/judges/executable_judge.py:154-185`), and the docs say so (`KAD/tasks/scoring.md:51-52`). Only exceptions *outside* the judge become `status="error"`. Do not port Karotte judges. Port only the runner split. |
| `m9-m10/02-runtime.md:679-686` and `m1-taxonomy/04…md:357` (EOG errors stay in the denominator) | **Confirmed** (`EOG/benchmark/executor.py:514-522`: an exception becomes `overall_success: False`; `:565-575` divides by all runs). This contradicts `EOG/README.md:273` ("excluded from averages"); `EOG/compute_score.py:35-41` counts errors but still averages their rates. **New:** a verifier whose `gym_name` is not configured is **skipped** (`executor.py:440-451` ✓, `continue`), and `overall_success = all(...)` (`:371` ✓) is True over an empty dict. A task whose verifiers are all mis-addressed passes vacuously. |
| `m7-m8/02-forge-synthesis.md:325-366` (EOG: no generation code; distractors are HF configs; infeasible subset not local; predicted auto-increment ids in checks) | **Confirmed.** Modes are HF dataset configs (`EOG/evaluate.py:284-298`). The 13 local tasks have 102 `database_state` verifiers, 1 with `expected_value: 0`, and every task has one gym (my count over `EOG/data/revised/*/*.json`). Infeasibility appears only on the site (`EOG/docs/index.html:222-224`). The code has no abstain verifier type: only `database_state`, `response_check` and `tool_execution` (`EOG/benchmark/verifier.py:48-80`). |
| `m7-m8/02-forge-synthesis.md` §6 (MiMo tiers = number of Apps; synthesis not released) | **Confirmed.** Tier is encoded only in the directory name `…_t{1-5}_rl_NNN`. Counts t1 28, t2 98, t3 92, t4 40, t5 9, with t1 3-4 Apps up to t5 14-17. Generator code is absent: `S/verify.py:409` imports an unreleased `synthesis.office_gen` and falls back to `_grade_sc`. **New:** the rubric `tier` (critical / important / sanity) is **not read** by `verify.py`; only `weight` scores. The source-protection gate `protect_diff` (`S/_helpers.py:536-560`, used at `S/verify.py:352-366`) is **dormant**: 0 of 266 metas set `src_protect`, and 0 items set `gate: true`. The README states no pass-rate, flaky-task or dedup rule (`ME/README.md`, 790 bytes). It names `general/train.parquet`, `webdev.parquet` and `music.parquet`, which are not in the release. One env (`s3k_0559_…_zh_t4_rl_004`) has no `verify.py` or `verifier_meta.json`. |
| `m1-taxonomy/04…md:444-445` (MiMo masking contract: no `reward` key → `testbed_corrupted` → −999) | **Confirmed.** `S/run_verify.py:141` ✓ writes `{"reward_error": …}` instead of a reward. `XV/recipes/general/general_agent/environment.py:572-578` ✓ masks `missing_or_invalid_reward_json` and `reward_out_of_range` (non-finite, or outside [0, 1]). `XV/recipes/general/agent_loop.py:76-87` maps six categories to −999. `XV/verl/trainer/ppo/core_algos.py:304-350` ✓ fills an invalid row with the group's valid mean, then zeroes its advantage (`scores[invalid_mask] = 0.0`). **Crucible's importer breaks this contract (D9).** |
| `M05-M07-study-sweep.md:98` (D2: Crucible's NeMo-Gym `reward_components` do not sum to `reward`) | **Fixed at HEAD.** `C/interop/nemo_gym/server.py:147-148` ✓ now sends `reward_components = {"crucible": reward}`, with channels under `crucible_channels`, and writes `mask_sample` at the top level **and** in `instance_config`. |
| — (new; matters for the export) | NeMo-RL reads `mask_sample` **only** from `instance_config` (`NR/nemo_rl/experience/rollouts.py:314-322` ✓). NeMo-Gym's contract puts it at the top level (`NG/nemo_gym/base_resources_server.py:110-122`). Only NG's `judge_failsafe` copies it down, "to keep older NeMo RL consumers working" (`NG/nemo_gym/judge.py:105-115` ✓). Crucible writes both places, so its export is correct. Any NG server that sets only the top-level flag is silently trained on in NR. NR's dynamic sampling decides "trivial group" with `torch.ones_like(rewards)` as the valid mask (`NR/nemo_rl/algorithms/grpo.py:3455-3462` ✓), so masked rows still count toward group variance. MiMo's DAPO filter drops infra rows first (`XV/verl/trainer/ppo/v1/replay_buffer.py:257-268` ✓). |

---

## 1. Per repo × module

### 1.1 NeMo-Gym (`NG`, Apache-2.0)

#### M8 forge

| Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|
| Pivot-candidate selection: ≥ 8 local rollouts per candidate; keep mixed-reward candidates; drop all-0 and all-1; drop the easiest first. This is guidance only; `validate_pivot_dataset.py` checks the schema | `.agents/skills/nemo-gym-pivot-datasets/SKILL.md:96-110` | `C/forge/learnability.py:69-79` samples 4 → 8 → 16 with a Wilson interval (stronger). There is no pivot/branch-point generator | nothing (ours is stronger); PATTERN for G-pivot (already proposed in `improvements/M06-M08.md` M8 #5) |
| Pass-rate statistics only: pass@k, majority@k, std across runs. **No pass-rate filter in code** | `nemo_gym/reward_profile.py:515-660` | `C/forge/ladder.py`, `C/forge/curator.py:58-83` place and filter tasks | nothing |
| `prepare_data`: validate samples and aggregate metrics; `task_data_validation` off/warn/error/auto; train and validation datasets must declare a license | `nemo_gym/cli/dataset.py:88-96`; `nemo_gym/train_data_utils.py:91-100,691`; `config_types.py:482-486` | None for forged-task *datasets*. Tasks are checked one by one (M9) | PATTERN (S): license and provenance required on every exported split |

#### M9 accept

| Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|
| **Verifier fixture**: `full_reward`, `zero_reward` and `malformed` are required; `determinism` is required only when the manifest says `seeded`. Each runs in-process on a fresh server. A malformed case must *raise* with `expected_error`. Full/zero must hit the declared reward-range endpoints. Non-finite rewards are errors | `nemo_gym/verifier_fixture.py:21,49-54,95-98,190-244` | A1 (oracle = 1), A2 (floors = 0), A4 (determinism). **Nothing requires that a malformed submission raises or masks, and never scores > 0.** Nothing checks that the reward is finite and in range (`C/accept/checks/core.py`, no such check) | PATTERN (S): add a malformed-submission check to A2 |
| **Environment validation**: hard errors for a stale manifest (mirror fields ≠ config, unless `sync=True` rewrites it atomically), unordered reward range, missing benchmark prepare script, bad dataset rows, empty dataset, async `prepare()`. One warning class only (profile inference) | `nemo_gym/environment/validation.py:448-456,500-562,636-711`; `environment/manifest.py:105-139,270-362` | A9 lints the task (`C/accept/checks/extended.py:53-81`). There is no manifest-staleness concept for exported envs | PATTERN (S) for the NeMo-Gym export (`improvements/M09-M11.md` M11 #2 stands) |
| **Harness capabilities TE-1..TE-9** with a P0 gate; "absence never implies N/A"; post-hoc CLI exit 0/1/2 only | `nemo_gym/harness_capabilities/checker.py:21-48`; `cli.py:278,292-301` | `C/accept/checks/base.py:6-8,41-45` (`MissingCapability` → fail) already ports "absence never implies N/A" | nothing (done) |
| **Golden-patch buckets**: inconclusive / supported / broken / flaky; evidence-based failure causes | `resources_servers/swemer_v2/aggregate_golden_patch.py:51-64`; `diagnose_failures.py:42-94,160-175` | Confirmed as G2 sweep pick D5. A4 here never sees `None`: `ctx.grade` retries masked grades and raises `InfraUnavailable` (`C/accept/checks/base.py:97-112`) | PORT (S, Apache-2.0); already ranked in G2 |

#### M10 runtime

| Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|
| `mask_sample` ⟂ `failure_kind`; 22 shared kinds; `<server>:<kind>` extension; unknown names warn once and never raise | `nemo_gym/base_resources_server.py:110-148`; `failure_kinds.py:49,64-130,148-165` | `C/core/schema/run.py:20-94` mirrors it, with `register_failure_kind`. **But `runtime.infra.normalize` renames registered pack kinds to `agent_run_error`** (D5) `[ran p5]` | DEPEND (contract) + fix D5 |
| `RolloutFailure{source, delivery}`; "no reward is inferred from a failure" | `nemo_gym/rollout_outcomes.py:14-53` | `Failure.delivery` exists (`C/core/schema/run.py:110`) but does not drive `BudgetLedger.release` | PATTERN (S) |
| **Failure sidecar**: successes go to the main JSONL; every failed attempt goes to `<stem>_failures.jsonl` with `_ng_failure_class` / `_ng_failure_terminal`; drained rows write nothing | `nemo_gym/rollout_collection.py:173-210` | `C/runtime/journal.py:81-89` (`failures.jsonl` + `results.jsonl`) | nothing (equivalent) |
| **Torn-tail repair before resume**: scan back from the end; if the last line fails to parse, `truncate`; else append `\n`. Runs on the main file and the sidecar | `rollout_collection.py:872-900` ✓, called at `:2383-2385` | `C/runtime/journal.py:36-47` only *skips* a bad line on read. The next `_append` concatenates onto the fragment, so the **first record written after a crash is lost** (D2) `[ran p2]`. The forge journal does not even tolerate it (D7) `[ran p7]` | **PORT (S, Apache-2.0)** |
| **Walltime drain** `dispatch_budget_s` + p75 `drain_margin_s` after 5 completions; undispatched rows write nothing | `rollout_collection.py:1515-1565,3395` | `run.py:106-107` stops only at the next dispatch; it has no drain margin, so in-flight attempts can overrun | PORT (S, Apache-2.0) |
| Resume: main file = done keys; sidecar = attempt counts; `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS` (3); terminal-stamped rows never retried | `rollout_collection.py:903-914,1243-1270,2236-2300` | `C/runtime/journal.py:91-108`; `RetryRule.max_attempts=3` | nothing (equivalent) |
| Health checks: hollow turn, zero completion tokens, missing token counts, capture mismatch, runaway generation, token mismatch, per-task consistently unhealthy | `nemo_gym/health/checks.py:26-110`; `rollout_health.py:212,292-357` | Only `crucible:capture_incomplete` (`C/runtime/run.py:219-220`) | PATTERN (M); `improvements/M09-M11.md` M10 #5 stands |

### 1.2 NeMo-RL (`NR`, Apache-2.0)

| Module | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M10 | `mask_sample` read only from `instance_config`; `loss_multiplier[mask] = 0`; metric `num_mask_sample_filtered`; opt-out `should_mask_flagged_samples` | `nemo_rl/experience/rollouts.py:314-322,1913-1925`; `algorithms/grpo.py:2524-2537` | `C/interop/nemo_gym/server.py:147-148` writes both locations | DEPEND (contract; done) |
| M10 | Typed Gym failures; infra attempts 5, data attempts 2; backoff `min(2^(n-1), 30)` s; `max_skipped_prompts=0` | `nemo_rl/environments/nemo_gym.py:251-275`; `single_controller_utils/config.py:106-150`; `experience/rollout_manager.py:159-211` | `RetryRule` (no backoff, no split between infra and data budgets; `C/runtime/infra.py:87-96`) | PATTERN (S): separate infra and task retry budgets, exponential backoff capped at 30 s |
| M10 | One env exception fails the whole batch (`return_exceptions=False`) | `rollouts.py:1612-1617` | Same flaw in `C/runtime/run.py:159` (D1) | anti-pattern |
| M10 | `max_turns_reached` and `truncated` recorded as rates, never as masks | `rollouts.py:934,974-975,1085-1160` | `HarnessResult.truncated`, `LimitPolicy` (`C/runtime/infra.py:99-113`) | nothing |
| M10 | Dataloader state checkpointed (`train_dataloader.pt`); rollout recovery state (`rollout_recovery.pt`) | `algorithms/grpo.py:4071-4080`; `experience/rollout_recovery.py` | `C/runtime/checkpoint.py` (episode-level). Its M12 equivalent is not in scope here | nothing |
| M8 | Dynamic sampling: exact-equality "trivial" groups, but masked rows count | `grpo.py:2056-2122,3455-3462`; `algorithms/utils.py:231-252` | The forge learnability `run(n) -> list[bool]` cannot express a mask (D8) | anti-pattern (counting masks); see MiMo for the right way |

### 1.3 EnterpriseOps-Gym (`EOG`, Apache-2.0)

| Module | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M8 | Task = `gym_servers_config[{seed_database_file, context, user_info}]` + `system_prompt` (policy) + `user_prompt` + `selected_tools` / `restricted_tools` + SQL verifiers. No generator code | `EOG/benchmark/models.py:6-97`; `EOG/data/revised/*/*.json` | `C/core/schema/task.py:121-122` (`apps_used`, `distractor_tools`); toy scenario (`packs/_toy/.../forge/scenario.py:88-115`) | nothing (ours already has the shape) |
| M8 | Distractor tools as dataset modes `oracle` / `plus_{5,10,15}_tools`; `restricted_tools` removes tools | `EOG/evaluate.py:284-298`; `EOG/benchmark/executor.py:98-103,276-303` | `C/world/world.py:384-398` draws `distractors` reproducibly from the World seed (better: seeded, and not tied to a dataset config) | nothing |
| M8 | Infeasible tasks (policy-violating requests); only on the site, no code | `EOG/docs/index.html:222-224` | `C/forge/schema.py:42` `infeasible`; toy T3 (`scenario.py:99`); DESIGN §7.5 `AbstainPolicy` | nothing |
| M9 | Fresh DB clone per run from a seed SQL; deleted in `finally` | `executor.py:320-345,555-563` | World reset + snapshot (M1) | nothing |
| M9 | **Verifier with an unknown `gym_name` is skipped; `all({})` is True** | `executor.py:440-451,371` ✓ | `C/accept/checks/core.py:21-70` (A1 fails when oracle reward ≠ 1). Crucible's plan has no "every declared check produced a verdict" assertion either; see D11 | anti-pattern (fail-open) |
| M9 | SQL query error or judge exception → `passed: False` (scored) | `EOG/benchmark/verifier.py:94-107,150-154,481-483` | `C/verify/runtime/*`: infra → masked | anti-pattern |
| M10 | Whole-sample retry ×5 with linear backoff on any run `error`; then the errored result is saved and counted | `EOG/evaluate.py:221-234`; `executor.py:514-522,565-575` | `RetryRule` retries masks only; policy failures never | anti-pattern (survivorship bias + 0 scoring) |
| M10 | Resume skips any existing result file, including error ones | `EOG/benchmark_utils.py:51-57` | `C/runtime/journal.py:91-96` (only terminal rows are done) | nothing |
| M10 | Keyed resource pools (`llm_<name>` = N, `domain_<d>` = 1) | `EOG/ray_experiment_queue.py:36-50,68-70` | One global semaphore (`C/runtime/run.py:101`) | PATTERN (S): keyed pools per provider and per grading host |

### 1.4 Karotte (`KA`, MIT)

| Module | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M10 | **Limit policy**: `turn_limit` always errors; time/context limits default `error`, opt-in `score`; checked between turns; an error ends with no score and is never retried | `KA/schemas/evaluation_run_config.py:43,52,67`; `KA/agents/message_loop.py:88-109`; `KA/evaluation_runner.py:88-103`; `KAD/running/run-config.md:145-168` | `C/runtime/infra.py:99-113` `LimitPolicy` (default `score`, by design: `m9-m10/02-runtime.md:826`). `mask` is retried (D6) | PORT done; fix D6 |
| M10 | **Misbehavior = 0, not infra**: `StudentMisbehaviorError` caught in `Step.score` | `KA/student_misbehavior.py:5-18`; `KA/step.py:83-92` | `crucible:policy_misbehavior` (`C/core/schema/run.py:45,56`); custody violations (`native.py:131-133`) | nothing (done) |
| M10 | **Grader crash scored 0** (`ExecutableJudge`) | `KA/judges/executable_judge.py:154-185`; `KAD/tasks/scoring.md:51-52` | Crucible masks grader crashes (`native.py:282-283`, `classify_verifier`) | anti-pattern |
| M9 | **Submission custody**: `save_submission` copies with `O_NOFOLLOW` dir/file fds, refuses symlinks and special files, caps bytes / entries / depth, raises misbehavior; `FileExistsError` on an existing destination | `KA/save_submission.py:47-90`; `KA/untrusted_paths.py:12-38` | `C/world/custody.py` (hand-off checks, `CustodyViolation` → scored 0 at `native.py:131-133`) | PATTERN (S): compare the caps (entries, depth, total bytes) with our custody limits |
| M9 | **`karotte check` at image build**: credential material in home dirs (netrc, git-credentials, gh, gcloud, aws, kube, HF, keyrings, SSH keys; config files only if the secret is inline), unsafe loader env paths, protected-store permissions | `KA/cli/check.py:26-90`; `KA/check_credentials.py:19-20,220`; `KA/check_paths.py:21-27`; `KAD/running/development-loop.md:92-108` | DESIGN §9 lists "credentials in the agent environment" as an import check (I-series), **not built** (`ARCHITECTURE.md` M9 "Imports") ; A5 scans hidden strings only (`core.py:195-222`) | **PORT (S, MIT)**: a credential scan in the A5 image audit |
| M9 | **Confinement audit**: observations vs per-sandbox expectations (uid ≠ 0, cgroup v2, memory/pids/disk limits, firewall canaries); every probe is undone | `KA/confinement_check.py:1-6,188-398` | A6 `isolation_audit()` is adapter-defined (`extended.py:31-50`) | PATTERN (confirmed from `m9-m10/01-acceptance.md:520`) |
| M8 | No task generation, oracle, no-op or flake check (absent) | — | — | nothing |

### 1.5 MiMo verl fork (`XV`, Apache-2.0) and mimoagent (`MA`, MIT)

| Module | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M10 | **Reward-phase masks**: `judge_key_missing`, `mcp_backend_down` (liveness probe), `missing_run_verify`, `missing_or_invalid_reward_json`, `reward_out_of_range` (non-finite or outside [0, 1]) | `XV/recipes/general/general_agent/environment.py:483-578` ✓ | `C/interop/mimo/verify.py:86-93`: a missing `score` is read as 0.0; nothing checks range or finiteness (D9) | **PORT (S)** |
| M10 | Six infra categories → −999; the row stays in the batch, its score is replaced by the group's valid mean, its advantage is zeroed; excluded from metrics | `XV/recipes/general/agent_loop.py:76-87,699-700,926-942`; `XV/verl/trainer/ppo/core_algos.py:304-350` ✓; `metric_utils.py:477-492` | `C/runtime/infra.py:119-144` drops masked rows from every aggregate | nothing (M10); M12 concern |
| M8/M10 | **DAPO group filter ignores infra rows** before judging zero variance | `XV/verl/trainer/ppo/v1/replay_buffer.py:257-268` ✓ | Forge calibration counts masked episodes as failures (D8) | **PATTERN (S)** |
| M10 | Tool retries only for exec-stream-open errors (3 attempts, 2·n s backoff); any infra error skips the reward | `XV/recipes/general/env_actor.py:268-285,302-341` | `RetryRule` is per attempt, not per tool call | PATTERN (low) |
| M10 | Code recipe: `InfraError` status is "ungradable" (raise); reward POST retried 6× (5-120 s backoff) | `XV/recipes/code/mimoagent_runner.py:42,244-407` | — | nothing |
| M9/M7 | Code reward: reset the paths the patch touches, `git apply test_patch` (no `--reject`), run `test_command`; a reset or apply failure → `REWARD_TESTBED_CORRUPTED` (masked) | `MA/environments/datasets/opensource_code.py:24-45,211-250` | Root-owned outside copy of hidden tests (stronger for in-kit tasks) | PATTERN (import path; same as the G2 DeepSWE pick) |
| M10 | Compaction is a model-called `Compact` tool; history becomes [system, boundary, verbatim issue anchor, summary]; token-threshold compaction explicitly absent | `MA/compaction/__init__.py:1-6`; `MA/compaction/surgery.py:4-7,161` | `actors/policy.ContextConfig` (policy-written compaction, DESIGN §10.4) | nothing (confirmed) |
| M9 | **Anti-hack defaults are off in the general recipe**: `_GIT_LEAK_PREVENTION_DEFAULT = "none"`, `_ANTI_HACK_CLEANUP_DEFAULT = False`; the answer-leak domain blocklist is opt-in | `XV/recipes/general/general_agent/environment.py:97-100` ✓; `MA/environments/datasets/base.py:46,95,101,197-208,767-792` | Crucible's leak/isolation checks (A5/A6) run on every task | anti-pattern (defaults); a note for the MiMo importer: an imported general env must get A5/A6 |

### 1.6 MiMo v2.6 release (`ME`)

| Module | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M8 | Difficulty tier = number of Apps, encoded only in the env name | `ME/general/envs/*_t{1..5}_rl_*` | `TaskCard.difficulty.apps_used` (`C/core/schema/task.py:121`) | nothing (the pattern is already taken) |
| M8 | Rubric items `{tier, method, weight, gate, check_code}`; 1,212 llm / 215 rule; tier unused in scoring | `S/verifier_meta.json:1-70`; `S/verify.py:343-405` | Gates + weighted checks (M7) | nothing |
| M9 | Rule items run in a subprocess with a 300 s timeout and a `random` ban; a timeout or bad output scores 0 (not masked) | `S/verify.py:15-48` | Crucible masks a crashing check (`check_infra`) | anti-pattern (scores grader crashes as 0) |
| M9/M10 | **"No parseable reward → masked"**: a crash or judge failure writes `reward_error`, never `reward` | `S/run_verify.py:102-151` (✓ `:141`); `S/verify.py:336-337,377-381` | `C/interop/mimo/verify.py:86-90` maps `error` / `reward_error` to masks (good) but not a missing score (D9) | DEPEND (contract) |
| M9 | Source-protection gate (`protect_diff`: any change outside the solve footprint zeroes the task), dormant in every env | `S/_helpers.py:536-560`; `S/verify.py:352-366` | DESIGN §8 isolation, ToolSandbox guardrail idea (G2) | PATTERN (an `unchanged_except` gate for forged World tasks) |
| M9 | Read-only DB opens (`mode=ro`); answer = last assistant message | `S/run_verify.py:14-31,42`; `S/verify.py:420-433` | Separate verifier World (`C/world/verifier.py`) | nothing |

---

## 2. Best picks per module (ranked)

Effort: S = under a day, M = a few days, L = a week or more.

### M8 forge

1. **Never count a masked rollout as a failure in calibration and learnability (MiMo DAPO pattern).**
   - **Source:** `XV/verl/trainer/ppo/v1/replay_buffer.py:257-268` (`valid = [v for v, is_infra in metrics if is_infra < 0.5]`). NR's opposite (`grpo.py:3455-3462`, masks counted) is the anti-pattern.
   - **Change:**
     - `C/forge/learnability.py:54-79`: `run(n)` returns `list[bool | None]`; `estimate` drops `None` and records `masked` on `PassRate`.
     - `measure` keeps sampling until it has `target` scored outcomes. It caps the number of masked draws and returns "undecided: masked" past the cap, never a lower p.
     - `C/forge/ladder.py` rungs inherit the same.
     - `packs/_toy/src/crucible_pack_toy/forge/calibrate.py:41-42,60-63`: return `None` when `res.masked` or `rep.infra` is masked, instead of `rep.reward == 1.0`. Better, route calibration through the M10 `Runner`, whose aggregates already drop masks (`C/runtime/infra.py:119-144`).
   - **Why it beats Crucible:** today a judge outage or tool crash during calibration lowers p. `advise` then says "easier" (`learnability.py:91-92`), so the forge moves knobs on infrastructure noise (D8).
   - **Effort:** S.
2. **Make the forge journal crash-safe and fail-closed on budget overrun.**
   - **Source:** NG torn-tail repair (`NG/nemo_gym/rollout_collection.py:872-900`, Apache-2.0); Repo2RLEnv `EventJournal` (already named for PORT in `m9-m10/02-runtime.md`).
   - **Change:**
     - `C/forge/orchestrator.py:127-153`: `Journal.__init__` truncates a torn tail (P7), and `append` fsyncs. Better, reuse `C/runtime/journal.py` once D2 is fixed there.
     - `C/forge/orchestrator.py:320-341`: move `ledger.settle` into the `try` and turn `BudgetExhausted` from `settle` into `NodeResult("abandoned")`, so a plan ends `abandoned`, never crashes (P8).
     - Catch any other exception from `role.run` as `RoleInfraError`-like `abandoned(crash)`, with the ledger settled.
     - Replace the in-memory `C/forge/budget.py:20-41` with the SQLite `C/runtime/budget.py` ledger (open since `m9-m10/02-runtime.md` §2.10).
   - **Why:** `[ran p7]` one torn line makes `Journal()` raise, so a crashed forge cannot resume at all. `[ran p8]` a role that overspends its estimate raises `BudgetExhausted` out of `run_plan`. The task stays `draft`, and `run_plans` (`gather`) loses every other plan in flight.
   - **Effort:** S.
3. **Key cached role outputs on the validators too.**
   - **Source:** Crucible's own M9 rule (registry version in every acceptance key, `C/accept/key.py:47-52`). NG does the same for stale manifests (`NG/nemo_gym/environment/validation.py:636-711`).
   - **Change:** `C/forge/orchestrator.py:302`: add a validator fingerprint (`[v.__qualname__ + inspect.getsource hash]`, or an explicit `Node.validator_version`) to `_hash(...)`.
   - **Why:** a reused `done` output is never re-validated (`:303-307`). Tightening a leak or schema validator does not re-check old drafts.
   - **Effort:** S.
4. **An `unchanged_except` gate generator for forged World tasks (MiMo `protect_diff`, dormant there).**
   - **Source:** `S/_helpers.py:536-560`, `S/verify.py:352-366`.
   - **Change:** a forge validator in `C/forge/` that emits, for each task, a hard gate "no table or file outside the oracle's write footprint changed". The footprint comes from the oracle's end-state diff, which the Oracle builder already has. The Adversary adds a collateral-edit negative to kill it in A3.
   - **Why:** the m14 live run found candidates that pass by making collateral edits (`C/accept/repair.py:199-200` comment). EOG checks only the rows the task names (`EOG/benchmark/verifier.py:81-127`).
   - **Effort:** M.
5. **Natural-key lint for checks on rows the policy creates.** Confirmed from `m7-m8/02-forge-synthesis.md` §5 (EOG `WHERE case_id = 1233`). Put it in the Test-writer validators. **Effort:** S.

### M9 accept

1. **A malformed-submission and reward-range check in A2 (NeMo-Gym verifier fixture).**
   - **Source:** `NG/nemo_gym/verifier_fixture.py:95-98,199-211` (Apache-2.0; PATTERN); MiMo's range mask (`XV/.../environment.py:577-578`).
   - **Change:** `C/accept/checks/core.py:73-85`: add solutions `malformed:*` from an adapter `malformed_solutions()`, with a closed N/A reason when the task has no submission surface. Each must grade masked (`check_infra`) or exactly 0, and never crash the grader. Every grade A1-A4 sees must have a finite `reward` within the task's declared range (`math.isfinite` plus a range check in `AcceptContext.grade`, `base.py:97-111`).
   - **Why:** no Crucible check covers a broken submission (truncated patch, binary garbage, wrong schema). A grader that crashes on one is a mask the policy can trigger (the M7 audit class).
   - **Effort:** S.
2. **A3 must fail when it scored no mutant.**
   - **Change:** `C/accept/checks/core.py:142-145`: if `mutants is not None` and `scored == 0`, add the reason "no scorable mutant (generator yielded none or all equivalent)". Require ≥ 1 scored mutant per gate family where a mutant source exists.
   - **Why:** `[ran p4]` A3 passes with `score: None, killed: 0`. DESIGN §9 A3 says "≥ 90% of generic defect-class mutants are killed", which is vacuous when there are none.
   - **Effort:** S.
3. **Fail-closed backfill and honest detector claims.**
   - **Change:**
     - `C/accept/registry/run.py:47-49`: `quarantined` unless the new check is `pass` or `not_applicable`. Today `error` and `pending` are `kept`, so a detector that crashes on a task leaves it in the pool (D10).
     - Add a unit test that every `CheckDef.detectors` id exists in `load_registry()`, and that every registry entry's `detector.check` lists it back (`[ran p9]`: 8 phantom ids, 7 unlisted entries).
   - **Effort:** S.
4. **Credential scan in the A5 image audit (Karotte `karotte check`).**
   - **Source:** `KA/check_credentials.py` (MIT; PORT the path list and the "inline secret only" rule for config files), `KA/check_paths.py` (loader env paths).
   - **Change:** `C/verify/leak_audit.py` `audit_image` gains a `credentials` pass; A5 (`C/accept/checks/core.py:212-221`) fails on a hit. This also delivers the DESIGN §9 import check "credentials in the agent environment".
   - **Why:** A5 searches only for hidden strings the task declares. A leaked HF or cloud token in an agent image is not a declared string.
   - **Effort:** S.
5. **Repair commits only on the full required set.**
   - **Change:** `C/accept/repair.py:201,300-331`: `commit` re-runs `accept_task` with the task's full `Policy.required` (or at least A1, A2, A3, A7, A16), and records which checks the verification ran. `verify` refuses a narrower `checks=` for commit purposes.
   - **Why:** the default verification set is A1, A7, A16 (`:201`). A repaired version can tighten a check until it fails the **valid alternative** (A3) or lift a floor (A2), and still commit.
   - **Effort:** S.
6. **Verdict completeness.** Every declared check must produce a verdict, or the grade is masked (EOG's skip-on-unknown-gym is the counterexample, `EOG/benchmark/executor.py:440-451`). Assert it in A1. **Effort:** S.

### M10 runtime

1. **Contain every attempt (the boundary plus a deadline).**
   - **Source:** NR typed failures + capped backoff (`NR/nemo_rl/environments/nemo_gym.py:251-275`; `rollout_manager.py:159-211`); NG drain (`rollout_collection.py:1543-1565`).
   - **Change:** `C/runtime/run.py:139-142`:
     - wrap `harness.run` (and `_record` / `_rollout`) in `try/except Exception` → a `Failure(kind="agent_run_error", blame="unknown", stage=<last phase>)`;
     - always `ledger.settle(..., uncertain=estimate)` in `finally`;
     - add `asyncio.timeout(spec.attempt_deadline_s)` → masked `crucible:attempt_deadline` with a sandbox kill;
     - stop dispatching when `remaining < p75(attempt seconds)` (NG's margin).
   - **Why:** `[ran p1]` one unclassified exception kills the whole run, returns no summary, and leaves the reservation `reserved` forever. A crash on resume re-marks it as uncertain spend (`budget.py:34`).
   - **Effort:** S.
2. **Torn-tail repair in the run journal (PORT from NG, Apache-2.0).**
   - **Change:** `C/runtime/journal.py:60-73`: on open, truncate `results.jsonl`, `failures.jsonl` and `journal.jsonl` back to the last full line (`NG/nemo_gym/rollout_collection.py:872-900`). Optionally write each row with a leading `\n` guard.
   - **Why:** `[ran p2]` after a torn write, the next terminal record (`s3`) is unreadable, so a finished sample is re-dispatched. That is a duplicate attempt, and for a paid frontier run a duplicate charge. The docstring claims NeMo-Gym's "torn-tail repair" (`journal.py:14`), but only the skip half was ported.
   - **Effort:** S.
3. **Make `LimitPolicy` "mask" non-retryable (Karotte semantics).**
   - **Change:** `C/runtime/harness/native.py:187-188,272-273`: `Failure(..., mask=True, blame="harness", retryable=False)`.
   - **Why:** `[ran p6]` a policy that habitually overruns is re-sampled up to 3 times. Then it is masked, which drops it from pass@k. That is "retry until ok", the survivorship bias `infra.py:12` forbids. Karotte ends such a run with `status="error"` and never re-runs it (`KA/evaluation_runner.py:96-103`).
   - **Effort:** S.
4. **Keep pack failure kinds through `normalize`.**
   - **Change:** `C/runtime/infra.py:46-52`: `if is_failure_kind(kind): return kind, None` (use `crucible.core.schema.run.is_failure_kind`). Delete the duplicate `infra.default_blame` (`:188-189`), which ignores `_REGISTERED`.
   - **Why:** `[ran p5]` a pack kind registered with `blame="policy"` becomes `agent_run_error` (harness, masked) when it passes through `failure()`. `native.py:183-184,266-268` and `workspace.py:119-121` all route World failures that way. So a pack's policy-outcome kind is masked and retried instead of scored 0.
   - **Effort:** S.
5. **Re-collect from the recorded base, not the tag.**
   - **Change:** `C/runtime/harness/workspace.py:95-96`: read `prior[1].parent / "base_sha"` (the previous attempt's dir). Copy it forward to the new attempt dir. Refuse to diff against the tag `base` when no sha was recorded (`crucible:empty_collection` with `retryable=False`).
   - **Why:** `[ran p3]` an agent that committed its fix and moved `base` gets a 0-byte patch on the re-collect retry. It is graded 0 and scored, although attempt 1 recorded the true base, which would yield the 122-byte fix. This bypasses the M7-audit fix the comment at `:108-109` describes.
   - **Effort:** S.
6. **Keyed concurrency pools and a per-task circuit breaker** (EOG resource keys `ray_experiment_queue.py:36-50`; NG per-task health `rollout_health.py:315-357`, used here as a *live* rule). Release the semaphore while a retry waits. **Effort:** M.
7. **`delivery` drives the ledger** (NG `rollout_outcomes.py:14-53`): `not_sent` → `BudgetLedger.release`; enforce `over_uncertain_tolerance` as a stop rule. **Effort:** S.

---

## 3. Crucible defects and spec gaps

| # | Defect | Where | Evidence |
|---|---|---|---|
| D1 | An unclassified harness exception aborts `Runner.run`; no summary; the ledger op stays `reserved` | `C/runtime/run.py:139,159`; `C/runtime/harness/native.py:257-271,289` | [ran p1] |
| D2 | The run journal loses the first record appended after a torn tail (skip-on-read, no truncate) | `C/runtime/journal.py:29-47,81-87` | [ran p2] |
| D3 | The re-collect-only retry diffs against the agent-mutable tag `base`: `base_sha` is read from the new attempt dir, which never has it | `C/runtime/harness/workspace.py:94-96` vs `:110-112` | [ran p3] |
| D4 | A3 passes with zero scored mutants | `C/accept/checks/core.py:121-145` | [ran p4] |
| D5 | `runtime.infra.normalize` renames registered pack kinds to `agent_run_error` (harness, masked); `infra.default_blame` ignores the pack registry | `C/runtime/infra.py:46-52,188-189` vs `C/core/schema/run.py:78-94` | [ran p5] |
| D6 | `LimitPolicy` "mask" stops are retried (harness blame, `retryable` default True): the agent is re-sampled until it fits, then masked | `C/runtime/harness/native.py:187-188,272-273`; `C/runtime/infra.py:93-96` | [ran p6] |
| D7 | The forge `Journal` raises on a torn last line (no resume after a crash) and does not fsync | `C/forge/orchestrator.py:135-153` | [ran p7] |
| D8 | Forge calibration counts masked episodes and infra-masked grades as failures; `learnability.measure` cannot represent a mask | `packs/_toy/src/crucible_pack_toy/forge/calibrate.py:41-42,60-63`; `C/forge/learnability.py:54-79` | [read] |
| D9 | The MiMo importer scores a missing `score` as 0.0 and does not check finiteness or [0, 1]. MiMo's own runner masks both | `C/interop/mimo/verify.py:91` vs `XV/recipes/general/general_agent/environment.py:572-578` | [read] |
| D10 | Registry backfill keeps a task whose new detector `error`s or is `pending` (fail-open). The docstring says it runs only on accepted tasks; the code does not check | `C/accept/registry/run.py:35-50` | [read] |
| D11 | Detector bookkeeping drifts: 8 ids that checks claim are not in the registry (`B.10-doctest-disable`, `B.10-os-exit`, `B.15-forged-approval`, the `G-*` ids), and 7 registry entries are not listed by their check (e.g. `HF.D1-oracle-noop` under A1). A11 claims `B.5-empty-patch` and `B.5-zombie`, but runs only the native harness cases (blank usage, credits, contention, grader crash, two controls) | `C/accept/checks/*.py` `@register(... detectors=...)`; `C/accept/checks/extended.py:136-189`; `C/accept/registry/defects/*.yaml` | [ran p9] |
| D12 | A forge role that overspends past the cap crashes the plan (`settle` raises outside the `try`); the task stays `draft`; `run_plans` loses every concurrent plan | `C/forge/orchestrator.py:339-341`; `C/forge/budget.py:36-41` | [ran p8] |
| D13 | Cached forge outputs are keyed without validators, so a tightened validator never re-checks reused drafts | `C/forge/orchestrator.py:302-307` | [read] |
| D14 | The repair loop commits after A1, A7 and A16 only. A2 (floors) and A3 (valid alternatives) are not re-run on a changed check set | `C/accept/repair.py:201,300-331,454-456` | [read] |
| D15 | `PerfGuard` counts every running `crucible*` container. In `WorkspaceHarness` the grade runs before the agent's World closes (`_grade` at `:133`, `close` in `finally` at `:136-141`). A live perf task whose agent sandbox (`crucible-sbx-*`, `C/world/sandbox.py:81`) shares the grading host therefore waits `wait_s`, then masks `node_contention`. The R1 gate uses `ImportedHarness` (no World), so it never exercised this | `C/runtime/perf_guard.py:49-50,78-79`; `C/runtime/harness/workspace.py:63-76,133-141` | [read] |
| D16 | Still open from `improvements/M09-M11.md`: global semaphore held across retries, no backoff, no per-attempt deadline, no run lock or pins, `release()` and `over_uncertain_tolerance` unused, checkpoint without identity, non-atomic grade store, serial checks | `C/runtime/run.py:101,118-157`; `C/runtime/budget.py:61-74`; `C/runtime/checkpoint.py:40-46`; `C/accept/checks/base.py:107-110`; `C/accept/run.py:45` | [read] |
| G1 (spec) | ARCHITECTURE M9 promises `imports/` (I0-I10, `import_report.json`) and the credential check; none is built. Karotte's `check_credentials` is the cheapest start | `ARCHITECTURE.md` M9 "Imports"; `DESIGN.md` §9 import checks | [read] |
| G2 (spec) | DESIGN §9 has no malformed-submission or reward-range requirement (NG verifier fixture; MiMo `reward_out_of_range`) | `DESIGN.md` §9 table | [read] |
| G3 (spec) | DESIGN §10.1 says a masked sample is "retried (never a policy failure)"; it does not say that a limit stop masked by `LimitPolicy` must not be re-sampled | `DESIGN.md` §10.1 "As built (M10)" | [read] |

---

## 4. Probe index (`scratchpad/m810sweep/g3/`)

| Probe | Shows | Result |
|---|---|---|
| `p1_runner_unhandled.py` | A harness `KeyError` in `Runner.run` | `Runner.run raised: KeyError`; ledger row `('…-a1', 'reserved', 0.5)` |
| `p2_journal_torn_tail.py` | Torn tail, then `finish(s3)` | `terminal samples read back: ['s1']`; `s3 lost: True` |
| `p3_recollect_base.py` | Re-collect retry after the agent committed and moved tag `base` | `stop: recollected reward: 0.0`; graded patch 0 bytes; with attempt 1's `base_sha`, 122 bytes |
| `p4_a3_zero_mutants.py` | A3 with `mutants() == []` | `A3 status: pass`, `score: None, killed: 0` |
| `p5_pack_kind_normalized.py` | `register_failure_kind("mypack:invalid_submission", blame="policy")`, then `infra.failure()` | becomes `agent_run_error`, `who=harness`, `masks=True`; `infra.default_blame` → `unknown` |
| `p6_limit_mask_retried.py` | The `LimitPolicy` mask failure through `RetryRule` | `masks: True`; retried after a1 and a2 |
| `p7_forge_journal_torn.py` | Forge `Journal()` on a torn last line | `JSONDecodeError` |
| `p8_forge_overrun.py` | A role spends 0.06 against cap 0.05 with estimate 0.01 | `run_plan raised: BudgetExhausted`; journal state stays `draft` |
| `p9_detector_ids.py` | Check `detectors=` vs registry ids | 8 phantom ids, 7 unlisted entries |
