# M8-M10 sweep, group 2: Prime Intellect repos, GitSwarm and autoresearch vs Crucible forge, accept, runtime (2026-10-12)

Read-only, 2026-10-12, Crucible HEAD `526fdb5`. Two helper agents read prime-rl with prime-agent, and prime-environments
with research-environments. I read verifiers, git-swarm, autoresearch and the Crucible code, checked the helpers' key
citations, and ran the probes.

**Concurrent edits.** While this sweep ran, another agent had uncommitted changes in `forge/budget.py` and
`forge/orchestrator.py`. Line numbers for those two files, and probes P2a and P2b, refer to HEAD `526fdb5`. Re-run
`p2_forge.py` after those changes land to see whether D6 and D7 are still open.

REF = `RL-Environments/06-frameworks-and-env-code/`. ART = `RL-Environments/07-articles/`. C = `crucible/src/crucible/`.

| Repo | Pin | Licence |
|---|---|---|
| `primeintellect-ai-verifiers` (VF) | `ae819de` | MIT |
| `primeintellect-ai-prime-environments` (PE, 109 envs) | `cf220ee` | Apache-2.0 |
| `primeintellect-ai-research-environments` (RE, 106 tasksets) | `d4fb80a` | Apache-2.0 |
| `infra-primeintellect-ai-prime-rl` (PRL) | `f7e8ab6` | Apache-2.0 |
| `primeintellect-ai-prime-agent` (PA, Rust) | `2ee7e623a` | MIT |
| `veds12-git-swarm` (GS) | `9f66af0` | **no licence**; a static project page. The mechanisms are in `ART/gitswarm-2610.04862.txt` |
| `karpathy-autoresearch` (AR) | `228791f` | **no licence** |

**Import limits.** Core may not import `verifiers` (`crucible/pyproject.toml:130, 143`). prime-rl is not a dependency,
and prime-agent is Rust. So DEPEND below always means "in `crucible.interop` or a bridge". In M8, M9 and M10, every pick
is a PORT or a PATTERN. Only MIT and Apache-2.0 code is marked PORT. GS and AR are PATTERN only.

Probes are in `scratchpad/m810sweep/g2/` (index in §4). They run with `crucible/.venv/bin/python` from `crucible/` and
read `crucible/src` only. verifiers cannot be imported in the Crucible venv (`pydantic_config` and `renderers` are
missing), so every verifiers claim is [read].

---

## 0. Corrections and confirmations of the earlier Crucible studies

| # | Earlier claim | Status | Evidence |
|---|---|---|---|
| 0.1 | `vf-validate` gold and noop are "the model-free A1/A2 core" (`docs/research/m7-m8/02-forge-synthesis.md:260, 290`; `m9-m10/01-acceptance.md:322`) | **Correct it.** The gold check calls `Task.validate`, which returns `None` by default (VF `verifiers/v1/task.py:195-197`). In all of VF only `gsm8k` overrides it (`environments/gsm8k/gsm8k/taskset.py:46-56`). The Harbor taskset has no `validate`, so its gold check is always "unchecked". In `all` mode a task with an unchecked gold is never "valid" (`cli/validate.py:329-339`). RE has 15 `validate()` hooks (14 SWE plus prolog), and none of them runs a no-op. The noop check passes when reward < 1.0 (`cli/validate.py:51-53, 260`), so it is weaker than Crucible's exact 0. Take the pattern (a fresh runtime per check, resume by content key, owed counts), not the coverage | [read] |
| 0.2 | verifiers gives "a run-level budget" for retries (`improvements/M09-M11.md:60`) | **Correct it.** `RetryState` is created per rollout (`agent.py:671`) or per episode (`utils/retries.py:129`). Its docstring says "per-run budgets … never across concurrent runs" (`utils/retries.py:77-78`). The part worth porting is real: the semaphore is released before the backoff (`env.py:315-319, 336-344`) | [read] |
| 0.3 | Runtime bugs: no try/except around `harness.run`, immediate retries, the semaphore held across retries, `over_uncertain_tolerance` never called (`improvements/M09-M11.md:37-49`) | **Confirmed, still open.** `runtime/run.py:118-157`; `budget.py:72` has no caller | [ran P1a, P1b] |
| 0.4 | "grader-only retry ×2 (verifiers), never the agent" (`m9-m10/02-runtime.md:844`) | **Partly right.** Only `IsolatedVerifierEnv` retries the verifier box alone, on any exception (`envs/isolated_verifier/env.py:179-182`). `Agent.run` retries rerun the whole rollout, agent included (`agent.py:673-676`). "A flaky grader retries without re-burning the solver" (`utils/retries.py:5-7`) refers to a separate grader *agent* in a multi-agent episode | [read] |
| 0.5 | `Trace.reward` sums non-None rewards (`m9-m10/02-runtime.md:58`) | **Confirmed, and it goes further.** Rewards are seeded as `None` (`task.py:253-255`). A scoring timeout is "a stop, not a failure" (`rollout.py:211-215, 571-572`), so the trace keeps `ok=True` (`rollout.py:584`) and `reward` sums to 0.0 (`trace.py:473-474`). The vf-eval headline mean is taken over non-errored traces (`cli/dashboard/eval.py:339-342`), so it counts a grader timeout as a 0 | [read] |
| 0.6 | `RetryRule{type, status_code, message, max_retries}` (`m9-m10/02-runtime.md:460`, `:15-35`) | Confirmed (`configs/retries.py:15-48`). An exhausted rule shadows later rules for that error (`utils/retries.py:113-116`) | [read] |
| 0.7 | proposer_solver `4p(1−p)` at `taskset.py:151-168`; the proposer's answer is never checked independently (`m7-m8/02-forge-synthesis.md:76, 250`) | **Confirmed, with an addition.** `_solve_rate` counts an errored solver as 0 in the denominator (`proposer_solver/taskset.py:152-158`), so infra noise lowers p. The answer is trusted from the proposer's own JSON (`:84-116`) | [read] |
| 0.8 | Training turns an agent timeout into a dropped sample (M5-M7 sweep G3 D13, PRL `dispatcher.py:668-674`) | **Correct the consequence.** The timed-out trace and its episode are marked not ok, but the group is **not** dropped. GRPO's baseline uses the survivors (`algo/base.py:15`; `algo/grpo.py:29-46`). The group is dropped only when nothing survives (`train_sink.py:243-255`). The curriculum pool ignores errored traces (`curriculum/samplers/pool.py:66-78`), so a task that always times out keeps weight 1.0 forever. Eval keeps timeouts scored (`metrics.py:541-546`; `docs/training.md:122`) | helper [read], spot-checked |
| 0.9 | prime-rl curriculum `pool.py:18-90`, `gates/adv.py:15-25` (`improvements/M06-M08.md:68`) | Confirmed. The file is 99 lines; the gate class spans `adv.py:15-36` | [read] |
| 0.10 | git-swarm is a website and the mechanisms are in the paper (M5-M7 sweep §1.5) | Confirmed (GS `README.md:1-6`; `source/build.py:1-49` builds a page from figure JSON). No swarm, economy or merge code exists in the repo. Nothing GS-derived in Crucible can be a PORT | [read] |
| 0.11 | autoresearch: the metric runs in the candidate's process; keep or discard on one run (M5-M7 sweep §1.6) | Confirmed (`train.py:26, 611-613`; `program.md:103-104`). Addition: a crash is logged as `val_bpb 0.000000` (`program.md:75`), in a column where lower is better. Only the status filter in `analysis.ipynb` cell 5 keeps crashes out of "best" | [read] |
| 0.12 | ARCHITECTURE M8 takes "PATTERN: research-environments recursive synthesis" (`ARCHITECTURE.md:605`) | **Overstated.** `recursive_task_synthesis` only consumes a published corpus. The synthesis code is not in the repo, and the README says it is "not a fully gold-verified dataset" (RE `recursive_task_synthesis/README.md:12, 61-68`). The only thing to take is its honest label for the verification level | helper [read] |
| 0.13 | ARCHITECTURE M10 takes "DEPEND: verifiers interception proxy for tokens" (`ARCHITECTURE.md:643`) | **Narrow it to interop only.** Core cannot import VF (0 above). Crucible's own capture proxy already collapses SDK retries by `x-stainless-retry-count` and `idempotency-key` (`C/connectors/capture_proxy.py:31, 58, 392`), as VF's server does (`interception/server.py:96-100, 634-662`) | [read] |
| 0.14 | prime-agent kernel has no per-cell timeout and RLM depth is 2 (`m14-rlm/01-prime-agent-study.md:69, 87`) | Confirmed (PA `crates/pa-core/src/kernel/manager/mod.rs:591-594`; `crates/pa-daemon/src/rlm_children.rs:34`) | [read] |
| 0.15 | DESIGN §10 as built: "Retries reuse what is still valid … an empty collection as a re-collect of the preserved tree" | **The re-collect diffs against the wrong base.** See D5 | [ran P3] |
| 0.16 | VF and RE CI smoke tests assert only the exit code (`m9-m10/01-acceptance.md:307-309`) | Confirmed (VF `tests/v1/test_envs.py:80`; RE `tests/test_envs.py:55-107`). PE's CI runs no rollout at all; it checks metadata, the README, `outputs/`, install and import (PE `tests/test_envs.py:27-106`) | [read] |

---

## 1. Mechanism tables

Verdicts: DEPEND (in interop only), PORT (licence allows), PATTERN, nothing (Crucible has it or better),
anti-pattern (use as a negative fixture).

### 1.1 verifiers (MIT)

| Mod | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M10 | One error vocabulary (Provider, Harness, Toolset, Env, Sandbox, Task, Interception). `boundary()` types any escaping error and lets typed errors through. A bad rollout is data and never cancels its siblings | `v1/errors.py:1-22, 30-95` | `runtime/infra.py:68-84` (`boundary`), but `runtime/run.py:139` calls `harness.run` outside any boundary | PATTERN (Crucible copied the boundary; it must also wrap the runner) |
| M10 | `ProviderError.status_code` is relayed so the harness SDK retries 5xx/429/timeout and not 4xx; exit code 97 is reserved for an exhausted model transport | `errors.py:34-47, 106-132` | `infra.NORMALIZE` (`infra.py:27-43`) has no status | PORT (S): keep the status on `Failure` for retry matching |
| M10 | `RetryRule` matches type AND status class AND message regex, with per-rule budgets. An explicit 0 denies. An exhausted rule shadows later ones for that error | `configs/retries.py:15-48`; `utils/retries.py:77-117` | `infra.RetryRule` matches blame and kind only (`infra.py:87-96`) | PORT (S) |
| M10 | Full-jitter backoff `min(0.5·2^a, 30)·(0.5+U)`. The error history is prepended to the final trace | `utils/retries.py:40-42, 120-158`; `agent.py:688-701` | none: retry is immediate (`run.py:154-156`) [ran P1b] | PORT (S) |
| M10 | The concurrency permit is held per attempt and released before a retry's backoff | `env.py:315-319, 336-344` | `run.py:118-157` holds the semaphore across all retries [ran P1b] | PORT (S) |
| M10 | Never retry into a borrowed box: its state is no longer the task's start state | `agent.py:679-684` | the re-collect and re-grade reuse (`harness/workspace.py:35-47`) | nothing (Crucible's reuse is the better idea, see D5) |
| M10 | Stage deadlines (setup, agent segment, finalize, scoring). An expired deadline is a timeout stop; a `TimeoutError` from the harness's own I/O with no expired deadline is a failure | `rollout.py:436-463, 541-572` | none: no per-attempt deadline in `runtime/` (grep finds no `asyncio.timeout` or `wait_for`) | PORT (M) |
| M10 | A scoring timeout leaves `ok=True` with every reward `None`, so the reward sums to 0 and counts in the headline mean | `rollout.py:571-572, 584`; `task.py:253-255`; `trace.py:473-474`; `cli/dashboard/eval.py:339-342` | `verify/stage.py` masks `verifier_timeout` | anti-pattern (fixture for the M11 export: a masked grade must not reach vf-eval as 0) |
| M10 | Any error makes the trace not ok; errored rollouts drop out of the mean and are re-run on `--resume`. There is no blame, so a policy-caused `TaskError` is resampled too | `cli/eval/resume.py:1-11, 42, 69-70`; `trace.py:742-758` | `infra.py:87-96` never retries policy blame | nothing (Crucible is better; VF's form is survivorship bias) |
| M10 | Resume keys rows by the task content hash; a torn last line is "owed", not a crash; rewrite by temp file and rename | `cli/eval/resume.py:24-89` | `journal.py:36-47` drops torn lines but never re-terminates them [ran P4a] | PORT (S) for the guard |
| M10 | The interception server replays a completed response or coalesces an in-flight one for an SDK retry or a reused Idempotency-Key, so one logical call commits one turn | `interception/server.py:90-100, 151-206, 615-700` | `connectors/capture_proxy.py:31, 58, 392` | nothing (parity) |
| M8 | proposer_solver: a proposer writes a problem with a JSON contract; n=4 solvers; proposer reward `4p(1−p)`. An off-contract reply raises (episode error, retryable) instead of scoring | `environments/proposer_solver/proposer_solver/taskset.py:78-116, 144-168` | `forge/learnability.py:21-93` (Wilson, 4→8→16) | PATTERN (taken). Two anti-patterns become fixtures: errored solvers count as 0 in p (`:152-158`), and the answer is never verified (`:104-115`) |
| M8 | `best_of_n`: siblings under a `TaskGroup`, so one raising attempt cancels and awaits the rest | `envs/best_of_n/env.py:26-44` | `forge/swarm.py:199-211` re-raises the first worker exception and leaves the others running [ran P2c] | PATTERN (S) |
| M9 | vf-validate: setup, gold and noop checks, each in a fresh runtime; separate reasons (valid, invalid, unchecked, error, timeout); owed counts; resume by task hash; bounded concurrency | `cli/validate.py:166-194, 210-220, 267-320, 420-475` | `accept/run.py:45-71` runs checks one by one; grades cached by content (`checks/base.py:78-112`) | PATTERN (S): bounded concurrency of independent checks; an "owed" count in the report |
| M9 | gsm8k `validate`: run the real verifier on the gold answer formatted as a model would write it | `gsm8k/taskset.py:46-56` | A1 (`accept/checks/core.py:21-70`) | nothing (A1 is stronger) |
| M9 | Harbor taskset: `rm -f` reward files before `test.sh`; strict reward parsing | `tasksets/harbor/taskset.py:345-354, 397-417` | `verify/runtime/plan.py` | nothing (parity, M5-M7 sweep) |

### 1.2 prime-environments (Apache-2.0; counts from a grep over all 109 envs, approximate)

| Mod | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M10 | 24 envs return 0 from an `except` (16 of them catch bare `Exception`); 41 counting False/None. 12 of 24 sandbox envs have an except→0 path. 0 envs raise a typed framework error. This contradicts AGENTS.md "fail fast" | `AGENTS.md:59-64`; e.g. `kernelbench/kernelbench.py:247-258`, `aider_polyglot/aider_polyglot.py:139-153`, `enigmata/enigmata.py:375-382` | `runtime/infra.py:1-15` | anti-pattern (I-check fixtures) |
| M9 | Fixed stdout marker: the grader appends `print("SUCCESS")` after `problem.run()` and scores `"SUCCESS" in stdout`, so a candidate that prints SUCCESS and exits wins. Any other exception scores 0.0 | `gpu_puzzles/gpu_puzzles.py:130-134, 158-167` (also `gpu_puzzles_modal.py:145, 163`) | A13 stdout-row injection (DESIGN §9) | anti-pattern → I7/A13 fixture |
| M10 | A sandbox exception sets exit sentinel 124; the candidate can `sys.exit(124)` itself, so the reward raises instead of scoring 0 | `humaneval/humaneval.py:48-53, 101-108` | `infra.classify_verifier` (`infra.py:174-185`) | anti-pattern → "policy launders a failure into infra" fixture |
| M9 | `correctness or compiled` counts a compiled-but-wrong kernel; dormant today because only the exception dict reaches it | `kernelbench/kernelbench.py:151, 262` | A3 mutants | fixture |
| M8 | A procedural generator with a difficulty knob; generator exceptions are swallowed; train and eval differ only by seed vs seed+1, with no overlap check | `enigmata/enigmata.py:78-92, 350-364` | `forge/dedup.py`; curator lineage seal (`forge/curator.py:10-13`) | anti-pattern |
| M8 | Reward scaled by the label's difficulty | `longbench_v2/longbench_v2.py:55-66` | none | anti-pattern (the reward should not depend on unstated label metadata) |
| M9 | CI checks metadata, the README, `outputs/` existence, LFS, install and import; no rollout or reward | `tests/test_envs.py:27-106` | `accept/*` | nothing |
| M9 | Publishing: README says a version bump triggers it, but the workflow publishes on any path change and dedups by content hash; 60 of 109 envs are still at 0.1.0 | `README.md:115` vs `.github/workflows/env-sync.yml:4-10, 93-101` | `accept/key.py` (content-bound key) | nothing (Crucible keys on content already) |

### 1.3 research-environments (Apache-2.0)

| Mod | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M8 | Pass-rate band per (model, solver) stored in task metadata. A narrower band drops unmeasured tasks (fail closed) | `general_agent/taskset.py:106-112`; `general_agent/corpus.py:118-126` | `forge/learnability.py:25-34`: `PassRate` is measured live and never persisted | PORT (S) |
| M8 | Train sets drop exact matches against eval statement hashes | `lean_common/scripts/eval_hashes.py:17-41`; `lean_common/taskset.py:186` | `bench/contamination.py` exists, but `forge/curator.py` never calls it (grep: only the docstring at `:13`) | PORT (M) |
| M8 | The decontamination index fails open: a missing file gives an empty set | `lean_common/taskset.py:65-66` | none | anti-pattern (Crucible must fail closed) |
| M8 | Rejection lists with reasons (misformalized, unported); publisher exclusion lists, `apply_audit_fixes` | `lean_common/taskset.py:91-96`; `pi_programbench/corrections.py:10-13`; `pi_programbench/taskset.py:29-32` | none in forge; I10 is spec only (DESIGN §9) | PATTERN (S) |
| M8/M9 | Gold check reused as the synthesis reward: gold must change the DB hash, verify(initial)=0, verify(gold)=1 | `general_agent/corpus.py:78-108`; `general_agent/taskset.py:89-93` | A1 (`accept/checks/core.py:21-40`) | nothing (reusing A1 as a cheap generator gate is a pattern Crucible already has in `accept/contract.py`) |
| M9 | A negative control must fail for the right reason ("reward 0 specifically for missing_compile_sh"), with full case accounting | `pi_programbench/README.md:75-82` | A2 checks the reward only (`accept/checks/core.py:73-85`) | PORT (S) |
| M9 | Grader attack suite with 30 bypass and 25 **honest** fixtures (disk fill, many processes, a daemon); grading refuses the agent's runtime | `lean_common/tests/fixtures/cases.json`; `lean_common/task.py:43-44, 123-126, 155-158` | A13 runs attacks only (`accept/checks/extended.py:248-271`) | PATTERN (M): honest stress controls must score 1 unmasked |
| M9 | Validation order including "one realistic concurrency check" | `BENCHMARK_PORTING.md:244-264` | A4 has no concurrency axis (`accept/checks/core.py:154-193`) | PATTERN (S) |
| M9 | Harbor `registry.json`: `name@date`, per-task git commit and path; tasks pinned by SHA | `registry.json`; `skills/prime-tasks/SKILL.md:15-25` | none | PORT (M) in interop/harbor |
| M9 | Status-line parsing where the last line wins: `test_status_map[name] = status` over merged stdout and stderr, so a candidate conftest that prints `PASSED <id>` overwrites FAILED | `swerebench_v2/taskset.py:208, 241-247`; `swerebench_v2/log_parsers.py:57-66` | hidden tests use an fd channel (`verify/runtime/patterns/hidden_tests.py`) | anti-pattern → A13/I7 fixture |
| M9 | Verifier in the agent's box; no reward-file removal; `reward_txt or 0`; the judge's OpenAI key is in `test.sh`'s env inside the agent container | `senior_swe_bench/taskset.py:108-110, 115-120, 124-126, 134-136` | `verify/stage.py` (separate container) | anti-pattern → I5/A13 fixture |
| M10 | A missing answer file or bad JSON counts as a policy failure (NO_ANSWER); other errors propagate | `longpdfs.py:213-219`; `nl2repobench/taskset.py:210-215` | `infra.boundary` maps every `OSError`, including `FileNotFoundError`, to `transport_unreachable` (infra), wrapping the grader (`infra.py:82-83`; `harness/native.py:139-140`) | PATTERN (S); see D15 |
| M10 | Typed framework errors raised by 9 tasksets (`SandboxError`, `HarnessError`, `ToolsetError`) | e.g. `apex_agents/harness.py:70, 133`; `enterprise_ops_gym/toolset.py:167, 239` | `NORMALIZE` has none of these names (`infra.py:27-43`), so they become `unknown` and count toward the 5% stop | PORT (S) |
| M10 | `trace.has_error` → reward 0.0 | `lean_common/task.py:150-153` | `infra.py:119-129` drops masked runs | anti-pattern |

### 1.4 prime-rl (Apache-2.0)

| Mod | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M10 | One terminal output per attempt (episode, failure or cancellation); an exception becomes a `DispatchFailure`, never aborts the loop | `orchestrator/dispatcher.py:9-12, 632-666` | `runtime/run.py:139` [ran P1a] | PATTERN (S) |
| M10 | Train timeouts become errors and are excluded; eval scores them | `dispatcher.py:668-674`; `docs/training.md:122` | `infra.LimitPolicy` default `score` (`infra.py:99-113`) | nothing in core; the M11 bridge must keep Crucible's choice (0.8) |
| M10 | A group is final once episodes + failures + cancellations ≥ group size; the baseline is over survivors | `train_sink.py:149-153, 205-255`; `algo/grpo.py:29-46` | k samples are independent (`infra.py:132-144`) | PORT (M): flag groups whose survivors < k |
| M10 | Failure rate by env and error type | `metrics.py:18-28`; `dispatcher.py:80-113` | `infra.Health` keeps a global `by_kind` only (`infra.py:147-171`) | PORT (M): per-task health so one broken task is quarantined instead of stopping the run |
| M10 | Admission burst cap `max(group_size, cap/10)` per 5 s; `tasks_per_minute` limiter | `dispatcher.py:170-182, 271-278, 620-621` | none | PATTERN (S) |
| M10 | Orchestrator checkpoint: progress, source RNG and curricula by atomic rename; in-flight work is not saved | `ckpt.py:30-47`; `train_source.py:67-85` | `runtime/journal.py` (per attempt) | nothing (Crucible is finer) |
| M10 | Seeds: `Random(42)` source mix and a seeded pool sampler | `train_source.py:21, 67-71`; `samplers/pool.py:41, 80-88` | `runtime/keys.derive_seed` per sample | nothing (Crucible is better) |
| M8 | Difficulty pools by the latest group mean: hard ≤ 0.25 (weight 0.2), normal ≤ 0.75 (1.0), easy (0.2); unseen tasks weight 1; checkpointed | `curriculum/samplers/pool.py:18-99`; config `packages/prime-rl-configs/.../orchestrator.py:231-262` | `forge/curator.py` places and retires offline only | PORT (M), with Wilson bounds instead of a point mean of about 8 |
| M8 | The sampler ignores errored traces and does not update when nothing is valid | `samplers/pool.py:66-78` | `forge/ladder.py:37, 62-77`: `run(n) -> list[bool]` has no masked slot [ran P2e] | PORT (S) |
| M8 | Zero-advantage admission gate `[0, 0]` | `curriculum/gates/adv.py:15-36` | the learnability band drops p ∈ {0, 1} before training | nothing for M8 (an M6/M12 concern) |
| — | Docs disagree with code: `docs/training.md:124` names `empty_rollouts` and `errored_rollouts` metrics that do not exist (the code emits `dispatcher/errored/*`, `metrics.py:22-27`); `docs/algorithms.md:142` gives a wrong module path; `docs/algorithms.md:166-176` says exactly one step behind, while the code allows a 1-batch lead plus in-flight staleness up to 8 (`orchestrator.py:84-87`; `dispatcher.py:401-439`) | as listed | — | note |

### 1.5 prime-agent (MIT, Rust)

| Mod | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M10 | Provider retry: 3 retries, 2 s base ×2, ±20% jitter, Retry-After honoured as `max(exp, retry_after)`, give up if Retry-After > 60 s | `crates/pa-core/src/session_engine/provider_retry.rs:11-31, 161-170, 191-209`; `auto_retry.rs:64-213` | none | PORT (S) |
| M10 | Permanent kinds: invalid request (except 404), refusal, permission, safety, payment required; auth retried once; context overflow and unsupported tool never retried; unknown is not retryable | `provider_retry.rs:142-155`; `auto_retry.rs:101-111`; `error_classify.rs:180-193, 226-296` | `Failure.retryable` defaults True and nothing sets it False (`core/schema/run.py:107`; only `runtime/collect.py:60` sets it, to True) [ran P4b] | PORT (S) |
| M10 | Quota park: sleep until the provider's reset time, then resume | `provider_park.rs:1-4, 94-125` | `run.py:144-148` (credits exhausted: stop, the sample stays owed) | PATTERN (S): record the reset time in the stop event |
| M10 | Failover to another provider serving the same model | `provider_failover.rs:1-4, 91` | none | anti-pattern for evaluation unless the provider is recorded per call |
| M10 | Append + fsync with a torn-tail newline guard; repair at open (zero-fill, unterminated row) by atomic rename | `pa-core/src/session/window_cache.rs:200-213`; `manager/repair.rs:85-145` | `journal.py:29-47` [ran P4a] | PORT (S) |
| M10 | Single-writer lease per session; a stale owner (pid + process start id) is reclaimed | `pa-daemon/src/lease.rs:1-4, 150-152, 194-218` | none (`journal.py:61-73` takes no lock) | PORT (S) |
| M10 | Tool errors go back to the model as `is_error` results | `pa-agent/src/agent_loop/tool_call.rs:28-53, 209-236` | `crucible:tool_crashed`, format errors | nothing |
| M10 | No-progress cap: 3 consecutive turns | `goal_driver/progress.rs:13, 17`; `goal_driver.rs:524-549` | `harness/native.py:36` LIMIT_STOPS | PATTERN (S): a `no_progress` limit stop |
| M10 | Child-agent usage rolled up to the parent per origin | `pa-daemon/src/rlm_children/usage.rs:1-15` | `harness/rlm.py:117` (shared `inner.spend`) | PATTERN, tied to D18 |
| M9 | Goal completion "audit" is a self-check written into the prompt | `pa-core/src/goals.rs:344, 352-356` | hidden verifier | anti-pattern as a reward source |

### 1.6 GitSwarm (paper only; no licence)

| Mod | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M8 | Budget B of charged episodes, concurrency C; a voluntary abstention is not charged | `ART/gitswarm-2610.04862.txt:197-201` | `forge/swarm.py:5-8, 196-202` | nothing (built) |
| M8 | API failures, turn limits and protocol violations are **charged unsuccessful episodes** (21 of 20,000) that the run absorbs | `ART:2677-2680` | `swarm.py:208-211` re-raises the first worker exception and aborts the stage [ran P2c] | PATTERN (S) |
| M8 | Announce scope (gap, deliverable, success, abort) before editing via `gs_inflight` | `ART:206-212` | `swarm.py:103-112`, but announcing is optional: a worker that never announces still commits (`swarm.py:182-195`) [ran P2c] | PATTERN (S): require an announcement before `Contribute` |
| M8 | Publication validation enforces requirements, not correctness; a commit with an extra file outside the allowed paths is rejected | `ART:187-189, 2215-2216` | `swarm.py:184-186` (`validate` defaults to accept-all, `:145`); zone rule in `forge/repo.py` | PATTERN (S): an allowed-paths default |
| M8 | Plurality readout of nominations, "votes are not evidence of correctness" | `ART:202-203, 213-224` | `swarm.py:215-220`; an author may nominate its own commit [ran P2c] | nothing in core (acceptance is M9); PATTERN (S): drop self-nominations |
| M8 | `informed_by` semantic dependencies beside the git parent | `ART:163-176` | `forge/repo.py:13-20` | nothing (built) |
| M8 | A record is "a best observation rather than a verified final architecture" until an exact-source repeat confirms it | `ART:2213-2219` | record gates (`verify/runtime/patterns/record.py`) | PATTERN (already a queued M7 pick) |
| — | No economics: no bounties, payments or reputation exist in the paper. Crucible's `forge/economy.py` (yield per generator, cost caps, diversity quota) is its own design; only the herding statistic is GitSwarm's (`economy.py:17-18`) | `ART:107-112, 197-201` | `forge/economy.py:1-21` | note |

### 1.7 autoresearch (no licence)

| Mod | Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|---|
| M8 | Keep on strict improvement, else `git reset`; one run per idea; no repeat | `program.md:94-106` | `forge/tighten.py`, A16 hacker search | anti-pattern for any hill-climb on a noisy metric (same as GS 2213-2219) |
| M10 | Three statuses (keep, discard, crash) and a 10-minute kill treated as a failure | `program.md:66-88, 108-110` | `runtime/infra.py` blame | nothing (Crucible separates infra from policy; AR does not) |
| M10 | Crash logged as `0.000000` in a lower-is-better metric column | `program.md:75`; `analysis.ipynb` cell 5 (filters CRASH), cell 7 (`kept.min()`) | `reward=None` for masked (`infra.py:119-125`) | anti-pattern (sentinel in the metric) |
| M9 | The evaluation runs inside the candidate's script, and the candidate's `model(x, y, reduction='none')` produces the loss it is scored on | `train.py:26, 611-613`; `prepare.py:344-365`; `program.md:28-31` | differential and outside-timing patterns (P26) | anti-pattern (A13 fixture: candidate-controlled loss) |

---

## 2. Best picks, ranked per module

### M10 runtime

1. **Contain every attempt.** Source: PRL `dispatcher.py:632-666` (one terminal output per attempt); VF `errors.py:80-95`. Change `runtime/run.py` `Runner.run.one`: wrap `harness.run` (`:139`) and `_rollout` (`:142`) in `try/except Exception`. Record a masked `agent_run_error` with blame `unknown`. Call `ledger.settle(aid, 0, uncertain=<reserved>)` and continue. Also wrap the unguarded calls in `harness/workspace.py:89-131` (`task.workspace`, `collect`, `shutil.move`) and `harness/native.py` (`save_episode`, `task.broken`). Effort S. Today one bad task kills the run and leaves every in-flight reservation `reserved` [ran P1a].
2. **Retry outside the semaphore, with backoff and rule matching.** Source: VF `utils/retries.py:40-42, 77-117`, `env.py:336-344`, `configs/retries.py:15-48`. In `run.py:116-157`, move the `while True` loop outside `async with sem`, sleep `backoff(n)` between attempts, and extend `infra.RetryRule` with `status_code` and `message` fields and per-rule budgets. Effort S. Today a sample's three attempts run back to back while every other sample waits [ran P1b].
3. **A retryable verdict per kind.** Source: PA `error_classify.rs:180-193`, `provider_retry.rs:142-170`. Add `DEFAULT_RETRYABLE` next to `DEFAULT_BLAME` in `core/schema/run.py`, and default `Failure.retryable` from it: `crucible:stale_image`, `crucible:stale_build`, `crucible:answer_exposed` and `provider_quota_exhausted` are not retryable. Honour Retry-After for `provider_unavailable`. Effort S. Fixes D3 [ran P4b].
4. **Stage deadlines.** Source: VF `rollout.py:436-463, 541-572` (an expired deadline vs a raw `TimeoutError`). Add `asyncio.timeout` per phase in `harness/native.py` and `harness/workspace.py`. On agent expiry, apply `LimitPolicy.on_timeout`. On grader expiry, mask as `verifier_timeout`. Effort M.
5. **A durable journal and a single writer.** Source: PA `window_cache.rs:200-213`, `manager/repair.rs:85-145`, `pa-daemon/src/lease.rs:150-218`. In `runtime/journal.py:_append`, write a leading newline when the file does not end in one, and repair at open. Take `fcntl.flock` on `run.lock` in `RunJournal.__init__`. Effort S. Fixes D4 [ran P4a] and the two-runner race.
6. **Blame from the boundary that failed, not from the exception class.** Source: RE `longpdfs.py:213-219`; VF error names. Add `SandboxError`, `TaskError`, `HarnessError`, `ToolsetError`, `InterceptionError` and `ProviderError` to `infra.NORMALIZE`. In `infra.boundary("verification")`, stop mapping every `OSError` to `transport_unreachable`: a `FileNotFoundError` on a candidate output is a policy fail. Effort S. Fixes D15 and D16.
7. **Per-task health and group accounting.** Source: PRL `metrics.py:18-28`, `train_sink.py:149-153`. Extend `infra.Health` with a per-(task, harness) breakdown that quarantines a task with a high mask share instead of stopping the run, and a "survivors < k" flag per group. Effort M.
8. **Re-collect from the recorded base.** Own fix, D5: `harness/workspace.py:95` should read `prior[1].parent / "base_sha"`. Effort S.

### M8 forge

1. **Masked-aware outcomes in the ladder and learnability.** Source: PRL `samplers/pool.py:66-78`. Change `Rung.run(n)` to `list[bool | None]` (`forge/ladder.py:37`). `learnability.estimate` drops `None`. `measure` tops up to valid counts with a cap. `n == 0` is "pending", never p=0. `accept` and `steer` return pending on no evidence. In the toy calibrator, pass a mask as `None` (`packs/_toy/.../forge/calibrate.py:42, 62-65`). Effort S. Fixes D10 and D13 [ran P2e].
2. **A forge orchestrator that never aborts on one node.** Source: VF `errors.py:80-95`; PRL `dispatcher.py:632-666`. In `forge/orchestrator.py:_run_node`, move each `ledger.settle` inside the try. Turn a `BudgetExhausted` from settle into `abandoned(budget)`. Turn any other exception into `RoleInfraError` (abandoned after 3), with backoff instead of `asyncio.sleep(0)` (`:328`). In `run_plans` (`:286-295`), use `return_exceptions` and record each plan's terminal state. Effort S. Fixes D6 and D7 [ran P2a, P2b].
3. **A swarm stage that absorbs worker failures.** Source: GS `ART:2677-2680` (failures are charged, unsuccessful episodes); VF `envs/best_of_n/env.py:28-32` (`TaskGroup` cancels siblings). In `forge/swarm.py:162-211`, catch worker exceptions as a charged `failed` episode. On a stage abort, cancel and await in-flight tasks. Reject a `Contribute` whose episode never announced, and ignore nominations by the author of the nominated commit. Effort S. Fixes D8 [ran P2c].
4. **Decontaminate at curation, fail closed.** Source: RE `lean_common/scripts/eval_hashes.py:17-41`, `taskset.py:186`; the anti-pattern `taskset.py:65-66`. Call `bench/contamination.py` from `forge/curator.py` `place` against the bench split and public benchmarks. A missing index rejects. Effort M.
5. **A train-time difficulty pool.** Source: PRL `samplers/pool.py:18-99`. Add it beside `Curator` and feed it from M10 group results. Use the band (hard < 0.1, normal 0.1-0.7, easy > 0.7 → `retire`) on Wilson bounds, with masked samples excluded. Effort M. Today difficulty is measured once, offline.
6. **Persist the pass rate on the card.** Source: RE `general_agent/corpus.py:118-126`. Write `PassRate` keyed by (model, harness, task_hash) into `crucible/card.yaml`. Loaders band-filter and drop unmeasured tasks. Effort S.
7. **Economy resume that survives a restart.** Own fix, D9: `forge/economy.py:62-69` must handle `resume` events (unpause and drop earlier rows of that generator). Effort S.
8. **Fixtures from proposer_solver.** Source: VF `proposer_solver/taskset.py:104-115, 152-158`. A proposer reward test where errored solvers must not count, and an "unverified answer" Architect output that the Oracle-builder check must reject. Target `tests/unit/test_forge.py`. Effort S.

### M9 accept

1. **A10 on the right success predicate and band.** In `accept/checks/extended.py:104-105`, count success with the task's own predicate (`outcome == 1` or every hard gate passing), not `reward == 1.0`. A record-setting score gate pays more than 1 (DESIGN §7.5) and is a success. Use `learnability.BANDS["bench"]` (upper bound ≤ 0.2) and `learnability.decided` instead of `Band(lo=0.0, hi=0.2)` on the point estimate (`:102, 114, 129`). Apply the same to A15 (`accept/checks/composition.py:68-69`). Effort S. Fixes D11 and D12 [ran P2f, P4c].
2. **A floor must fail where it is expected to.** Source: RE `pi_programbench/README.md:75-82`. Each floor negative declares its expected failing gate. A2 (`accept/checks/core.py:73-85`) fails when the floor scores 0 for another reason. That catches a verifier that scores everything 0. Effort S.
3. **Honest stress controls in A13.** Source: RE `lean_common/tests/fixtures/cases.json` (25 honest cases: disk fill, process fan-out, a daemon). Add controls that must score 1 and stay unmasked next to the attacks (`accept/checks/extended.py:248-271`). Effort M.
4. **Build I0-I10 with the counterexamples in §1.2-1.3 as fixtures.** Marker spoof (PE `gpu_puzzles.py:133, 160`); last-line-wins status (RE `swerebench_v2/log_parsers.py:57-66`); stale reward plus a credential in the agent box (RE `senior_swe_bench/taskset.py:108-136`); infra scored 0 (PE `kernelbench.py:247-258`); exit-code laundering (PE `humaneval.py:53, 106`); gold reruns at two concurrency levels (RE `BENCHMARK_PORTING.md:251`). Target: a new `accept/checks/imports.py`. Effort M/L. Nothing exists today (no `import_report` in `src/`).
5. **Bounded concurrency and an owed count.** Source: VF `cli/validate.py:166-194, 420-475`. Run independent checks of one plan level concurrently in `accept/run.py:45-71`, and report owed checks (pending + error) beside the status. Effort S.
6. **The hacker fails closed.** In `forge/hacker.py:145-152`, `reward > 0` with `audit_ok is None` is `unaudited`, not `legitimate_success`. Effort S. D14 [ran P2g].

---

## 3. Crucible defects and spec gaps

| # | Finding | Where | Evidence |
|---|---|---|---|
| D1 | An unclassified harness exception aborts `Runner.run` through `asyncio.gather`. No results are written, and every in-flight reservation stays `reserved` | `runtime/run.py:139, 159` | [ran P1a]: `KeyError` raised; ledger rows `[('…-a1','reserved') ×3]`; 0 result rows |
| D2 | Masked retries are immediate and hold the semaphore. With concurrency 1, sample 1 starts only after sample 0's three attempts | `runtime/run.py:118-157` | [ran P1b]: attempts at t = 0, 0.067, 0.12 (sample 0), then 0.173 … (sample 1) |
| D3 | No failure is ever non-retryable: `Failure.retryable` defaults True and nothing sets it False. A deterministic `crucible:stale_image` runs three attempts | `core/schema/run.py:107`; `runtime/infra.py:94`; grep finds only `runtime/collect.py:60` | [ran P4b]: 3 attempts |
| D4 | A torn last line in `results.jsonl` swallows the first row written after a restart, so a finished sample is lost and re-run | `runtime/journal.py:29-47` | [ran P4a]: rows read back `['s1']`; `s3` was written after the crash |
| D5 | A re-collect retry reads `base_sha` from the **new** attempt directory, so it always falls back to the agent-movable `base` tag the M7 audit removed | `runtime/harness/workspace.py:95-96` (written at `:111-112` into the prior attempt) | [ran P3]: prior recorded `0123456789ab`; the re-collect used `base` |
| D6 | In the forge orchestrator, `Ledger.settle` raises `BudgetExhausted` outside the try, so one overspending node aborts `run_plans`. Other plans are left in `draft` | `forge/orchestrator.py:324, 334, 341`; `forge/budget.py:40-41` | [ran P2a]: `run_plans` raised; journal states `t0: draft` only |
| D7 | Any role exception other than `RoleInfraError`, `LLMOutputError` or `ValidationError` aborts `run_plans`, and reservations stay held | `forge/orchestrator.py:314-338, 286-295` | [ran P2b]: `KeyError` raised; `ledger.reserved` non-empty |
| D8 | Swarm stage: one worker exception aborts the stage and leaves its siblings running; announcing is optional, so a worker that never announces bypasses collision checks; an author's self-nominations count | `forge/swarm.py:162-211, 182-195, 215-220` | [ran P2c]: stage raised; `proposer-4`, `proposer-5` finished afterwards; an unannounced contribution committed with 0 collisions; self-voted commit wins |
| D9 | `YieldBook` ignores `resume` events on reload: a generator a person resumed is paused again after a restart, and its old rows return | `forge/economy.py:62-69, 138-143` | [ran P2d]: same process paused=False rows=0; after reload paused=True rows=2 |
| D10 | Zero evidence reads as "too hard": `estimate([])` gives p=0 and `wilson(0,0) = (0,1)`. The ladder rejects and steers `drop_outermost` on train, `compose_one_more` on bench. `Rung.run -> list[bool]` cannot carry a mask | `forge/learnability.py:54-57`; `forge/ladder.py:37, 92-122`; `credit/operators/branch_value.py:41-42` | [ran P2e] |
| D11 | A10 and A15 count only `reward == 1.0` as success, so a record-setting score gate (reward > 1, DESIGN §7.5) counts as a failure | `accept/checks/extended.py:104-105`; `accept/checks/composition.py:68-69` | [ran P2f]: rewards [1.3, 1.12, 0] → p = 0.0 |
| D12 | A10 checks bench on the point estimate in [0, 0.2]; the forge and curator need the upper bound ≤ 0.2 | `accept/checks/extended.py:102, 114, 129` vs `forge/learnability.py:51`, `forge/ladder.py:101-105` | [ran P4c]: 1/8 passes A10 bench (hi 0.47), fails the forge band |
| D13 | The toy calibrator counts masked episodes and masked grades as failures (`None == 1.0` is False) | `packs/_toy/src/crucible_pack_toy/forge/calibrate.py:39-42, 61-65` | [read] |
| D14 | The hacker labels `reward > 0` with an undecided audit (`None`) as `legitimate_success` (fails open) | `forge/hacker.py:145-152, 166-169` | [ran P2g] |
| D15 | `boundary("verification")` maps every `OSError`, including a `FileNotFoundError` a grader raises on a missing candidate output, to `transport_unreachable` (infra, masked). A policy that deletes its output may be masked instead of scored 0 | `runtime/infra.py:82-83`; `runtime/harness/native.py:139-140`; `harness/workspace.py:72-76` | [read]; a risk, since it depends on each grader |
| D16 | `NORMALIZE` lacks verifiers v1 error names, so they map to `unknown` and count toward the 5% unknown stop | `runtime/infra.py:27-52, 164-171` | [read] |
| D17 | A2 checks only the reward, not the gate a floor fails on; a verifier that scores everything 0 passes A2 | `accept/checks/core.py:73-85` | [read] |
| D18 | The workspace harness costs an attempt as the delta of a client shared across concurrent attempts; the own-call fix in `NativeHarness` (`native.py:301-303`) is not applied there, and `cost_uncertain_usd` is still a delta in both | `runtime/harness/workspace.py:89, 142-143`; `runtime/single_turn.py:204-205`; `harness/rlm.py:117` | [read] |
| D19 | Ledger hygiene: `release` and `Failure.delivery` are dead; `over_uncertain_tolerance` is never called; `Runner.regrade` neither reserves nor settles; no single-writer lock on a run | `runtime/budget.py:61-74`; `core/schema/run.py:110`; `runtime/run.py:168-200`; `runtime/journal.py:61-73` | [read] (grep) |
| D20 | The curator never runs contamination checks, although `bench/contamination.py` exists and the docstring promises it | `forge/curator.py:13, 58-64` | [read] |
| D21 | I0-I10 and `import_report.json` are spec only (DESIGN §9; ARCHITECTURE M9 "Imports") | no `import_report` in `src/` | [read] |
| D22 | A4 has no concurrency axis, though I-checks and RE's porting guide both ask for one | `accept/checks/core.py:154-193` | [read] |

---

## 4. Probe index (`scratchpad/m810sweep/g2/`)

| Probe | Output | What it shows |
|---|---|---|
| `p1_runner.py` | `p1_runner.out` | P1a: D1 (a harness exception aborts the run; reservations leak). P1b: D2 (immediate retries hold the semaphore) |
| `p2_forge.py` | `p2_forge.out` | P2a: D6. P2b: D7. P2c: D8. P2d: D9. P2e: D10. P2f: D11. P2g: D14 |
| `p3_recollect.py` | `p3_recollect.out` | D5: the re-collect uses `base`, not the recorded sha |
| `p4_runtime.py` | `p4_runtime.out` | P4a: D4 (torn tail). P4b: D3 (stale image retried ×3). P4c: D12 (A10 bench band) |
| `stubs/pydantic_config.py` | — | a failed attempt to import verifiers in the Crucible venv (`renderers` is missing too); verifiers claims stay [read] |
