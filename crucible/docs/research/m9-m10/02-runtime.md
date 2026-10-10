# 02 — Runtime (M10): running policies and agents on tasks at scale, and classifying every failure

**Scope.** This report covers how every local framework runs agents against environments, and what Crucible's M10 `runtime` should depend on, port, take as a pattern, or skip. The topics are:
- job and trial orchestration, k samples and seeds, concurrency, retries, timeouts;
- sandbox lifecycle and GPU sandboxes;
- cost and budget stops;
- failure classification and masking;
- trajectory formats (ATIF) and step ids;
- harness adapters (installed agents) and capture of tokens and logprobs;
- resumability.

It closes with a design for `crucible.runtime` and an exit-gate plan.

**Read on 2026-10-09 (read-only, nothing modified):**
- Crucible: `ARCHITECTURE.md` M10/M12 (incl. the RL-infrastructure subsection), `DESIGN.md` §10 and Appendix B.5–B.13, and the code under `actors/`, `world/`, `connectors/capture_proxy.py`, `verify/` and `credit/operators/infra_mask.py`;
- `migration_tasks/*/run_agent.sh`, `common/run_mini.sh`, `common/mswea/ext.py`, and the historical task02 run directory;
- the eleven frameworks listed in §2;
- `docs/research/2026-10-09-agentic-rl-infra-survey.md` and `RL-Environments/07-articles`.

**Path prefixes:**
- `C/` = `crucible/src/crucible/`
- `MT/` = `migration_tasks/`
- `FW/` = `RL-Environments/06-frameworks-and-env-code/`
- `H/` = `FW/harbor-framework-harbor/src/harbor/`
- `AE/` = `agentenv-framework/src/agent_env/`
- `AEP/` = `agentenv-framework/packages/agentenv-protocol/src/agentenv_protocol/`
- `NG/` = `FW/nvidia-nemo-gym/`
- `OE/` = `FW/meta-pytorch-openenv/`
- `VF/` = `FW/primeintellect-ai-verifiers/verifiers/v1/`
- `RE/` = `FW/primeintellect-ai-research-environments/environments/`
- `PE/` = `FW/primeintellect-ai-prime-environments/environments/`
- `VL/` = `FW/xiaomimimo-verl/`
- `MM/` = `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/`
- `MS/` = `FW/swe-agent-mini-swe-agent/src/minisweagent/`
- `FE/` = `FW/adithya-s-k-fineenvs/05-multi-harness-rl/`
- `KAR/` = `FW/preferencemodel-karotte/src/karotte/`
- `R2E/` = `FW/huggingface-repo2rlenv/src/repo2rlenv/`
- `EOG/` = `FW/servicenow-enterpriseops-gym/`
- `SS/` = `FW/swe-bench-swe-smith/swesmith/`

Line numbers refer to the pinned commits in ARCHITECTURE §2.2.

---

## 0. TL;DR

1. **No framework has the runtime we need. Each has one or two pieces that are best in class.** Crucible's M10 is therefore mostly assembled, not invented:
   - **Harbor** (DEPEND): the harness adapters, i.e. 45 installed agents with base-URL overrides and usage parsing; `Trial`; `BaseEnvironment`; the ATIF models; regrade-from-artifacts.
   - **NeMo-Gym** (PORT/PATTERN): the failure vocabulary we already use; `mask_sample` semantics; the success-JSONL plus failure-sidecar resume model; attempt-qualified ids; the mask-fraction abort; reverify modes.
   - **verifiers v1** (PORT): the boundary error types with `boundary()`, the `alive()` probe and the stash-real-error trick; `RetryRule`; hash-keyed resume with a keep-verdict; the run-vs-run_program split.
   - **Repo2RLEnv** (PORT): the reserve/uncertain/settled `BudgetLedger` and the metered-call receipt protocol.
   - **OpenEnv RFC 006** (DEPEND or PORT): the black-box capture proxy, with 4 dialects, session-as-key routing, canonical token ids and capture levels.
   - **MiMo verl / uni_agent** (PORT shapes): `TokenTrace`, `chat_delta`, and the gateway's per-generation version marks.
   - **FineEnvs** (PORT): an evaluate loop with atomic per-unit files, an identity hash, coverage, and `reward=None` masking.
   - **Karotte** (PORT): the misbehavior-vs-infra split and the per-limit `error|score` policy.
2. **Every surveyed runner scores at least one class of infrastructure failure as 0.** This is the single most important finding for the exit gate:
   - Harbor: `0 if reward is None` (`H/metrics/base.py:23-33`; `H/utils/pass_at_k.py:41-43`).
   - verifiers: `Trace.reward` sums non-None rewards (`VF/trace.py:472-474`).
   - agentenv: a missing reward file scores 0.0 (`AE/task_step/task_steps/verifiers/run_container_unit_tests_verifier.py:333-339`).
   - mini-swe-agent batch: an env-start failure writes an empty patch to `preds.json`, and resume then skips it forever (`MS/run/benchmarks/swebench.py:157-160,176`).
   - SWE-smith: an exception returns `(logger, False)` (`SS/harness/utils.py:273-284`).
   - EnterpriseOps-Gym: errors stay in the denominator (`EOG/benchmark/executor.py:567-572`).
   - MiMo verl general: `pod_conn_timeout` never matches its mask set (`VL/recipes/general/env_actor.py:282` vs `agent_loop.py:78-87,934`).

   Crucible's runtime must therefore make "masked" a property of the record that **no reward accessor can turn into a number**, and every adapter must be tested against these exact anti-patterns.
3. **Nobody has a per-sample seed or a real (task × policy × harness × k) scheduler with resume keyed on that identity.** Harbor's attempts are unindexed identical configs, matched as a multiset on resume (`H/job.py:355-385`). agentenv's k runs are not even grouped (`AE/task/task.py:432`). verifiers' only seed is the task shuffle (`VF/configs/select.py:100-115`). We build this ourselves, around a `SampleKey`.
4. **Nobody keeps a policy version per generated span on the default path.**
   - verifiers has it per episode only (`VF/episode.py:31-61`).
   - NeMo-Gym has it only on its staging path (`NG/nemo_gym/token_id_capture/staging/rebuild.py:44-52`).
   - The uni_agent gateway records it per generation, then collapses it at finalize (`VL/third_party/uni_agent/uni_agent/session/session.py:50-65` vs `:893-897`).
   - OpenEnv lists it as an open question (`OE/rfcs/006-agentic-rl-harness-interception.md:290-292`).

   `crucible.rollout/v1` must stamp `policy_version` on every generated span at capture time. That means the endpoint/proxy, not a post-hoc join.
5. **Crucible today has the episode loop and the vocabulary, but no runner.**
   - `run_episode` (`C/actors/policy.py:194-275`) runs one episode. Nothing schedules k samples, journals, resumes, enforces a run budget, or writes the existing `RunRecord` (`C/core/schema/run.py:69-95`).
   - Several verifier failure kinds are outside the vocabulary and would raise in `RunRecord.mark_failure` (§1.3).
   - Two of the eight Appendix B.5 classes have no failure kind at all: the ABI/stale-build mismatch and the self-matching `pkill`.
6. **Exit gate (§6).** Two parts:
   - reproduce the GPT-5.6-sol / mini-swe-agent 2.4.6 run on task02 at three levels: regrade the recorded patch, replay the recorded 88 commands, and a live rerun through Harbor's `mini-swe-agent` installed agent on the lab 5090;
   - inject each of the 8 B.5 classes with a deterministic fault injector, plus 5 policy-caused controls. Every injected run must come out `outcome="masked"`, `reward=None`, excluded from pass@k, and retried when retryable. Every control must come out scored 0, not masked.

---

## 1. What Crucible already has (and the holes M10 must close)

### 1.1 Episode loop, policies, LLM client

**`run_episode(world, policies, limits=EpisodeLimits)`** (`C/actors/policy.py:194-275`):
- Drives the policies round-robin. `asyncio.wait_for` enforces the episode wall clock (`:243-246`).
- Takes a quiescing final snapshot. If that snapshot raises `WorldError`, the result is `crucible:policy_misbehavior` with `mask=False`, i.e. scored 0 (`:250-254`).
- Adds `crucible:answer_exposed` (`mask=True`) when a probe hit (`:256-258`).
- Picks the stop condition: `agent_done | all_done | max_steps | too_many_errors | timeout | context_overflow | <failure kind>` (`:188`, `:259-270`).
- Sets `truncated` for `max_steps/timeout/context_overflow/too_many_errors` (`:274`).
- `_masks` treats every kind outside `POLICY_OUTCOME_KINDS` as masked (`:278-283`).
- This is a good single-episode core. M10 wraps it and does not replace it.

**`EpisodeLimits`** (`:168-176`): `max_steps`, `max_tool_errors`, `timeout_s`, `end_on`, `snapshot_every`.

**`LLMClient`** (`C/actors/llm.py:115-224`):
- Per-client hard USD cap, which raises `BudgetExceeded`, kind `crucible:credits_exhausted` (`:25-26`, `:162-163`). A 402 or `insufficient_quota` response also raises `BudgetExceeded` (`:184-185`).
- `UsageBlank` (`crucible:usage_blank`) is raised when the response has no usage (`:39-42`, `:200-201`).
- `ContextOverflow` has `failure_kind=None`, so the episode ends truncated and scored (`:45-49`, `:186-187`).
- `RequestRejected` (`agent_request_failed`) is raised for non-retryable 4xx (`:33-36`, `:188-189`).
- Retry-After is honoured, with capped exponential backoff plus jitter (`:99-108`).
- The record/replay cache is keyed with `sample_key` (`:148-161`).

**`BashLoopPolicy`** (`C/actors/harness.py:80-150`): the mini-swe-agent v2 loop ported over World shell tools. The format-error stop is scored; credits exhaustion is masked.

**`LLMToolPolicy`** (`C/actors/policy.py:62-165`) carries the cut-off and empty-turn nudges.

**Holes:**
- **H1. No orchestration.**
  - No k, no seeds (the policy's `seed` is `None` when temperature > 0, `policy.py:121`), no concurrency limits, no journal, no resume.
  - No run-level budget. Each `LLMClient` has its own cap, and `forge/budget.py:20-44` is a forge-only `Ledger` with no persistence and no "uncertain" state.
  - The `RunRecord` schema exists, but nothing writes it.
- **H2. `UsageBlank` ends the episode masked.**
  - B.5's fix was "ignore cost-tracking errors, classify as infra". The legacy launcher literally did `MSWEA_COST_TRACKING=ignore_errors` (`MT/task04_pcl_gpu_cuda12/run_agent.sh:18`).
  - Masking on the first blank response throws away long, expensive episodes.
  - M10 should **tolerate** blank usage. Account for the call with a conservative estimate, held as *uncertain* in the ledger. Mask only when the uncertain share of the sample's spend exceeds a threshold, or when the hard cap could have been crossed.
- **H3. `RequestRejected` (`agent_request_failed`) masks.** It is not a policy-outcome kind. That is right for 401/403/404 (infra/config), but a 400 caused by a malformed history *our* harness built is a harness bug, which is also infra. Keep masking. Record `blame="harness"` so that the share of harness bugs is visible and alarmable.

### 1.2 World, sandboxes, guard, exec

**`World._call_tool`** (`C/world/world.py:295-339`) is the single choke point:
- it labels the fault `policy | tool | infra`;
- a tool timeout becomes `transport_timeout` (infra), and an App crash becomes `crucible:tool_crashed` (infra);
- `_infra` appends a `Failure(stage="agent")` to `world.failures` (`:341-344`).

**Sandboxes:**
- `DockerSandbox` (`C/world/sandbox.py:75-117`) always passes `--init`, `--network none` for no-network worlds, `--pids-limit`, and `--runtime=nvidia` (or `--gpus`).
- `image_id()` digests RootFS layers (`:49-63`), and `check_image()` refuses a restore into a different image (`:66-72`). This is the stale-image primitive, but it is used only for WorldRefs, not before grading.
- Harbor: `C/interop/harbor/gpu_docker.py:30-100` (`runtime: nvidia`, `init: true`, `network_mode: none`, pids/mem/cap limits), and `C/interop/harbor/world_backend.py:90-246` (any Harbor provider).
- agentenv: `C/interop/agentenv/sandbox.py:108-159` (Modal GPUs and any agentenv provider).
- A2A agents: `C/interop/agentenv/a2a.py:28-115`.

### 1.3 Verify, collect, capture

**`docker_grade`** (`C/verify/stage.py:67-102`):
- runs the separate verifier with `--network none --init`;
- with no result file it writes `{"infra": {"status": "masked", "failure_kind": "no_reward_json"}}` (`:99`).

Holes in `docker_grade`:
- **H4.** `no_reward_json`, and the runtime's `check_infra` / `grader_exception` (`C/verify/runtime/plan.py:194,196`), are **not in `FAILURE_KINDS`** (`C/core/schema/run.py:18-47`). `RunRecord.mark_failure` would raise `ValueError` on them (`run.py:90-92`). `mask_for` silently maps unknown kinds to `verifier_error` (`C/credit/operators/infra_mask.py:28`). M10 needs one normalizer: kind `verifier_error`, with the original name kept as a `detail` subkind.
- **H5.** `subprocess.run(..., timeout=timeout)` (`stage.py:97`) raises `TimeoutExpired`, which nothing catches. The unnamed `--rm` container keeps running on the daemon after the client is killed. No image fingerprint is checked before grading (the B.5 stale-image case).

**`verify/collect.py`** (`C/verify/collect.py:14-53`): a tolerant collect hook and `COLLECT_SH`.
- It never aborts on one path, adds untracked files, excludes derived artifacts and reports errors. This is the B.5 "empty patch" fix.
- **H6.** It cannot tell "the hook failed" from "the agent changed nothing". Both produce an empty patch.
- task02's own `[[verifier.collect]]` still uses the fragile `git add … && git diff` chain (`MT/task02_pcops_triton/task.toml`, `[[verifier.collect]]`).

**Perf patterns** (`C/verify/runtime/patterns/perf.py:40-128`):
- `LoadGuard` (load per core ≤ 0.5, GPU utilization ≤ 20%, waits up to 900 s, then `InfraError`);
- the round-spread check marks contamination → `Verdict(infra=…)`.
- This is the grader-side half of `perf_guard.py`.

**`CaptureProxy`** (`C/connectors/capture_proxy.py:49-148`):
- an OpenAI chat / Anthropic messages recording proxy with per-session keys;
- logprobs are recorded "when the upstream returns them".
- **H7.** No token ids, no token-in/token-out (TITO) path, no policy version, no Responses dialect, no stream replay from a non-streamed engine call, no call budget.

### 1.4 Trajectories

**`episode_to_atif`** (`C/interop/harbor/atif.py:23-80`): `step_id = len(steps)+1`, built on Harbor's own models.

**`StepRow`** (`C/core/schema/reward.py:62-74`) declares `atif_step_id`, `call_id` and `harness`.

**H8.**
- `atif_step_id` is **never set**. `C/credit/steps.py:21-30` sets only `call_id`.
- `episode_to_atif` emits no `Metrics` (tokens, cost, logprobs), no `FinalMetrics`, no `extra.crucible_step` (World step index) and no `call_id`.
- LLM calls that produced only a format error produce no ATIF step, so ATIF steps, World steps and model calls cannot be joined.

### 1.5 The historical launcher (`run_agent.sh`) and its fixes

| Script | Fix it encodes | Gap |
|---|---|---|
| `MT/task04_pcl_gpu_cuda12/run_agent.sh:14` | `docker run -d --init --network none --runtime=nvidia --cpus --memory … sleep infinity` | the container is left running "for inspection" (`:28`); no cleanup on interrupt |
| `:18` | `MSWEA_COST_TRACKING=ignore_errors` (blank usage) | the cost of blank calls is invisible |
| `:26` | `\|\| true` after `mini`, so a crashed agent still gets collected | no classification of *why* it crashed |
| `:27` | collect with `{ git add … \|\| true; }` | still a hand-written per-task exclude list |
| `MT/task03_mdlite_pallas/run_agent.sh:14` | — | **no `--init`** (the zombie class), no cpu/memory limits, a fragile `&&` collect (`:26`) |
| `MT/common/run_mini.sh:25-27` | collect over SSH | the `&&` chain bug (the empty-patch class) |
| `MT/common/mswea/ext.py:44-72` | `SSHEnvironment` (docker exec over ssh, `timeout` wrapper, `ServerAliveInterval`), regex network guard for pods that cannot be cut off | a timeout or ssh drop becomes an observation (`returncode -1`), never infra |
| `MT/common/mswea/ext.py:75-100` | `LongOpenRouterModel` (request timeout 1200 s instead of 60 s) | 401/429 typed; other errors are generic `OpenRouterAPIError` |

PORT into M10:
- `--init` and limits everywhere (already in `DockerSandbox` and `GpuDockerEnvironment`);
- tolerant collect;
- cost-tracking tolerance;
- long request timeouts;
- the "always collect, then classify" ordering.

**What the historical task02 run is** (`MT/task02_pcops_triton/runs/gpt-5.6-sol-run1/`):
- `trajectory.json` is mini-swe-agent-1.1 format: `info.model_stats = {instance_cost: 2.287, api_calls: 88}`, 191 messages.
- Config: `step_limit=350`, `cost_limit=80`, `max_consecutive_format_errors=3`, `environment.docker_container=pcops-gpt` over ssh to the lab host.
- `model.patch`: 31 files, 3,020 lines. `grade_report.json`: `reward 0.0`; gates `static ✓ install ✓ ops ✗ (132/138) hard ✗ (36/37) app ✓ perf ✗`.
- The grader was later tightened (commits `2ce6ffd`, `5aa2b63`). The historical grade therefore reproduces only against task02 at `5e53057`. M10's regrade must pin the task digest of the original run, not the current task.

---

## 2. Per-framework findings

Each subsection lists the files, interfaces, what is good, the holes, the license, and a recommendation per M10 piece. Ratings: **DEPEND** (import), **PORT** (copy with license and attribution), **PATTERN** (re-implement the idea), **SKIP**.

### 2.1 Harbor (Apache-2.0, `harbor` 0.23.0, Python ≥ 3.12; heavy deps: litellm, supabase, fastapi; providers as extras)

**Job / Trial.**

`JobConfig` (`H/models/job/config.py:397-460`):
- `n_attempts` (k), `timeout_multiplier` plus four per-phase multipliers, `n_concurrent_trials=4`, `retry: RetryConfig`;
- `environment`, `verifier`, `agents`, `datasets`, `tasks`, `artifacts`, `source_jobs`.
- Equality ignores `job_name`/`debug` (`:547-554`). Resume depends on this.

`TrialConfig` (`H/models/trial/config.py:448-498`):
- `task`, `trial_name` (`{task[:32]}__{ShortUUID(7)}`, `:545-548`), `agent`, `environment`, `verifier`, `job_id`, `source_trial`, and timeout multipliers.

`AgentConfig` (`:63-150`): `name`, `import_path`, `model_name`, `n_concurrent`, `concurrency_group`, override and max timeouts, `resume_trajectory`, `load_trajectory`, `kwargs`, `env`, `mcp_servers`.

**k samples and concurrency:**
- k is a plain Cartesian product with no sample index (`H/job_plan.py:140-168`). There is **no seed anywhere** in Job, Trial or Agent configs.
- Scheduling is a single-process asyncio `TrialQueue`: a global semaphore (`H/trial/queue.py:45`), optional per-agent pools keyed by a config hash, acquired only around `agent.run()` (`:114-192`).

**Disk layout:**
- Job dir: `config.json`, `lock.json` (harbor version and git hash, task/skill/compose digests: `H/models/job/lock.py:60-195`), and a `result.json` rewritten after each trial (`H/job.py:964-998`).
- Trial dir: `result.json`, `exception.txt`, `agent/`, `verifier/{reward.txt,reward.json,test-stdout.txt}`, `artifacts/manifest.json`, `steps/` (`H/models/trial/paths.py:190-339`).

**`TrialResult`** (`H/models/trial/result.py:70-97`): `task_checksum`, `agent_info`, `agent_result: AgentContext`, `verifier_result.rewards`, `exception_info`, four `TimingInfo`s, `step_results`.

**Resume:**
- Deletes trial dirs without `result.json` (`H/job.py:286-288`) and matches finished trials by config equality as a multiset (`:355-385`).
- **Errored trials count as done.** Only `harbor jobs resume --filter-error-type` deletes them, and its default list is `["CancelledError"]` (`H/cli/jobs.py:2303-2312,2422-2466`).

**Regrade:** `SourceJobConfig(action: regrade|diff, …)` (`H/models/job/config.py:325-352`). It restores recorded `agent/` and `artifacts/`, then runs separate-mode verifiers. A trial is regradable if and only if the verifier's inputs are in the artifact manifest (`H/trial/regrade.py:1-13`). This is exactly what perf re-grades need.

**Retries and errors.**

`RetryConfig` (`H/models/job/config.py:291-322`):
- `max_retries=0`;
- `include_exceptions`;
- `exclude_exceptions` defaults to `{AgentTimeoutError, VerifierTimeoutError, RewardFileNotFoundError, RewardFileEmptyError, VerifierOutputParseError, ApiUsageLimitError, AgentSafetyRefusalError, AgentAuthenticationError, ModelNotFoundError}`;
- backoff `min_wait·mult^attempt`, with no jitter (`H/trial/queue.py:105-110`).

Behaviour:
- Exceptions are matched by **class-name string** (`:83-103`).
- On retry, the failed trial dir is `rmtree`d (`:222`), so the evidence is gone.

Exception hierarchy:
- Timeouts: `trial/errors.py:4-17`.
- Verifier: `verifier/verifier.py:20-37`.
- Environment: `environments/base.py:70-84`.
- API: `agents/installed/base.py:34-162`, classified by the `ERROR_PATTERNS` regex table over stdout/stderr, where the last match wins (`:487-582,912-948`).

`ExceptionInfo{exception_type, message, traceback, occurred_at}` keeps the first exception only (`H/models/trial/result.py:21-36`; `H/trial/trial.py:557-566`).

**Holes:**
1. Docker exec and compose failures surface as a bare `RuntimeError` (`H/environments/docker/docker.py:716-720,741-743`).
2. `wait_for(agent.run)` turns *any* inner timeout into `AgentTimeoutError` (`H/trial/trial.py:648-659`).
3. The verifier ignores the test script's exit code (`H/verifier/verifier.py:226-229`). A crashed grader becomes `RewardFileNotFoundError`, which is not retried and is scored 0.
4. A teardown failure after a good verify overrides the reward (`H/trial/trial.py:1875-1881`).
5. `ContextWindowExceededError` is retried by default.
6. Errored trials are scored 0 (`H/metrics/base.py:23-33`; `H/utils/pass_at_k.py:41-43`). The pass@k estimator itself is the standard unbiased one (`:87-94`).

**Timeouts:** `min(base, max)·multiplier` (`H/trial/trial.py:568-578`). The task agent timeout defaults to **None** (`H/models/task/config.py:340-341`), verifier 600 s, setup 360 s, build 600 s.

**Environments.**
- `BaseEnvironment` (`H/environments/base.py:93-1420`): `start`, `stop(delete)`, `upload/download_file|dir`, `exec(command, cwd, env, timeout_sec, user) -> ExecResult`, health checks, network policies.
- 31 providers (`H/models/environment_type.py:5-35`).
- Docker declares no `gpus` (`H/environments/docker/docker.py:317-331`). Hence our `GpuDockerEnvironment`. Custom providers load through `EnvironmentConfig.import_path`.

**Installed agents.**
- `BaseInstalledAgent` (`H/agents/installed/base.py:349-1240`), with 45 agents and about 35 ATIF converters.
- Pointing an agent at our endpoint goes through `resolve_model_connection` (`H/agents/model_connection.py:153-200`):
  - claude-code: `ANTHROPIC_BASE_URL`, with every alias pinned to `ANTHROPIC_MODEL` (`claude_code.py:124-128,1873-1880`);
  - codex: `openai_base_url` written into its config (`codex.py:1568-1575`);
  - mini-swe-agent: `OPENAI_BASE_URL`/`OPENAI_API_BASE` (`mini_swe_agent.py:508-526`), installed with `uv tool install` (`:637-667`), `--cost-limit` (`:476`);
  - openhands: `LLM_BASE_URL`;
  - terminus-2: `api_base`.
- `AgentContext` (`H/models/agent/context.py:21-52`): `n_input_tokens`, `n_cache_tokens`, `n_output_tokens`, `cost_usd`, `model_usage`, `rollout_details`.

**ATIF and RL data.**
- `Trajectory` v1.0–v1.8, `extra="forbid"`, `step_id == i+1` enforced (`H/models/trajectories/trajectory.py:121-131`).
- `Step` (`step.py:14-86`): `step_id`, `source`, `message`, `tool_calls`, `observation`, `metrics`, `is_copied_context`, `llm_call_count`, `extra`.
- `Metrics` (`metrics.py:8-39`) carries `prompt_token_ids`, `completion_token_ids` and `logprobs`.
- `RolloutDetail` (`H/models/agent/rollout_detail.py:4-27`) is filled only by terminus-2 (`H/llms/lite_llm.py:313-321,415-448`) and openhands_sdk. The CLI agents never return token ids, so we capture through our proxy.

**Budget:** none at job level. Only per-agent flags (claude-code `--max-budget-usd`, mini-swe-agent `--cost-limit`, openhands `MAX_BUDGET_PER_TASK`).

**Recommendations:**
- `run.py`: **DEPEND on `Trial` only** (`Trial.create(TrialConfig).run()` per attempt) and own the outer loop. PATTERN `lock.json` provenance, live `result.json`, per-agent pools and regrade. **PORT** `utils/pass_at_k.py` (the unbiased estimator) and compute it over *scored* samples only.
- `harness/`: **DEPEND** (installed agents, model-connection resolution, ATIF converters).
- `infra.py`: **PATTERN**. Use `ERROR_PATTERNS` and the exception names as *inputs* to a phase-aware classifier. Never trust `exception_type` alone.
- `budget.py`: **SKIP**. Read `AgentContext.cost_usd` as a meter only.
- `gpu.py`: **DEPEND** on `BaseEnvironment` plus our `GpuDockerEnvironment`.
- `trajectories.py`: **DEPEND** on the ATIF models.

### 2.2 agentenv-framework (Apache-2.0 with NOTICE "agent-env, Copyright 2026 Scale AI")

Packages:
- `agentenv-framework` 0.9.1301: heavy (litellm, an exact `a2a-sdk==0.3.26` pin, `mcp<2`, modal, e2b, pymongo).
- `agentenv-framework-protocol` 0.1.316: light (pydantic, starlette, httpx).

**`eval run --k`** (`AE/cli/eval/run.py:76-185`):
- builds `len(tasks)·k` coroutines under one `asyncio.Semaphore(max_concurrency)` (`:139-148`);
- **no per-sample seed**. "Seed" means a string-substitution dict (`AE/task/task.py:91-119`);
- each run gets its own `run_group_id` (`AE/task/task.py:432`), so the k samples are not grouped;
- **bug:** it reads `metadata["score"]`, which nothing writes (`:38,163` vs `AE/task_step/task_steps/verifiers/scoring.py:3-4`);
- a run that raises writes no output file (`:61-72`).

**`LocalRunner`** (`AE/runner/local_runner.py`):
- in-process asyncio with `Semaphore(workers=2)` (`:160`);
- a single terminal writer with CAS-guarded status (`:154-217`; `AE/runner/store.py:20-22,83-94`);
- on restart, it marks QUEUED/RUNNING records FAILED and does not resume them (`:51-60`);
- no per-run wall timeout. Teardown is bounded (`:30-36`), and `teardown_run` always runs (`AE/task/teardown.py:31-49`).

**Task DAG and journal:**
- `TaskStep(id, version, depends_on, fail_task_on_error, retry_config)` (`AE/task_step/task_step.py:99-118`);
- `RetryConfig(retry_from_step_id, max_retries)` (`:51-81`);
- span rollback with an `attempt_epoch` fence (`AE/task/task.py:166-275`);
- journal rows `{instance_id, step_id, seq, generation, status, recorded_at, ops}` (`AE/task/store.py:436-443`), built from a `ContextUpdateOps` diff (`AE/task_step/context_ops.py:19-31`);
- replay from the journal (`AE/task/step_journal.py:75-95`);
- SQLite under `$XDG_STATE_HOME/agent-env/` (`AE/config/paths.py:12-25`);
- resume only through `--start-step` plus `--context-json` (`AE/cli/task/run.py:415-416`). The journal is for in-run rollback, not crash resume.

**Sandboxes:**
- `SandboxProvider.create_sandbox/create_vm/create_container` (`AE/providers/sandbox_providers/sandbox_provider.py:38-130`), with the `EXTRA_CONTAINER_RUN_ARGS` hook (`:136-206`);
- built-ins `local`, `modal`, `modal_vm`, `e2b`, `sail_vm`;
- local ignores cpu, memory and disk (`local_sandbox.py:436-458`);
- only Modal supports GPUs (`modal_sandbox.py:195-206`);
- custom providers through `[sandbox.providers.<name>] impl=` (`sandbox_provider.py:263-314`).

**A2A:**
- JSON-RPC `message/send` → poll `tasks/get` (`AE/a2a_agent/protocol.py:100-204`);
- `AgentUnreachableError` after 60 s and 3 unanswered polls;
- third-party harnesses through the `install_agent` step with `urn:agentenv:install/v1` (`AE/task_step/task_steps/install_agent.py:1-88`).

**Failure handling:**
- `TaskError.error_type: Literal["agent_error","infra_error"]` (`AEP/a2a_agent/tasks/v1.py:161-170`) is **flattened into `RuntimeError`** at the step (`AE/task_step/task_steps/prompt_agent.py:697-718`), so the blame is lost.
- `Usage(input_tokens, output_tokens, cost_usd, …)` (`AEP/a2a_agent/tasks/v1.py:186-243`) is dropped by the client, which keeps only `tool_call_count` (`AE/a2a_agent/protocol.py:257-260`).
- No masking, no budgets, no logprobs. Trajectories are an opaque `NativeTrajectory(format, payload)` (`AEP/a2a_agent/tasks/v1.py:173-183`).

**Recommendations:**
- `run.py`: **PATTERN**. The ARCHITECTURE line "DEPEND: agentenv `eval run --k`, LocalRunner" should be revised. The CLI's holes (no grouping, a broken score readout, failed runs not persisted, no seeds) make it unsuitable as our scheduler. **PORT** the `Runner`/`RunRecord`/`runner/store` trio (about 480 LOC) for its single-terminal-writer and CAS design. **PATTERN** the journal: diff ops, epoch fence, generation counter.
- `harness/` (A2A): **DEPEND** on `agentenv-framework-protocol` (already used in `C/interop/agentenv/a2a.py`). Read `TaskError.error_type` and `Usage` directly from the wire, bypassing the lossy client.
- `gpu.py`: **DEPEND** on the provider registry and register a `gpu_docker` provider (a subclass that sets `--runtime=nvidia` plus limits), as DESIGN §10.1 already plans.
- `infra.py`, `budget.py`, `trajectories.py`: **SKIP**.

### 2.3 NeMo-Gym (Apache-2.0)

**Rollout collection** (`NG/nemo_gym/rollout_collection.py`, 3,813 lines; CLI `gym eval run`):
- `num_repeats` (int, or a dict per agent with `_default`), `num_repeats_add_seed` (a per-rollout seed in `extra_body`, `:1444-1458`), `interleave_repeats`;
- `num_samples_in_parallel` → semaphore (`:935-942`);
- `max_resident_rollout_tasks` → bounded admission (`:1869-1965`);
- walltime: `dispatch_budget_s`, plus `drain_margin_s` that adapts to the observed p75 after 5 completions (`:1543-1565`). Drained rows are `cancelled` and persisted nowhere (`:1207-1226`).

**Resume** (`:2236-2348`):
- a success JSONL plus a `<stem>_failures.jsonl` sidecar;
- `gated = successes ∪ terminal ∪ maxed_out` (`:2296-2300`), with `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=3` (`:217`);
- attempts stamped `_ng_attempt_index`, so capture keys become `rollout_id-a<N>` (`NG/nemo_gym/rollout_correlation.py:41-75`);
- a torn tail is repaired (`:872-899`);
- "kill_shaped failures go NOWHERE: the absence of a row is the canonical signal" (`:162-192`).

**Persistence routing** (`:2615-2729`): a failure row never enters the score. The capture snapshot is retired only after fsync.

**Failure taxonomy:**
- `NG/nemo_gym/failure_kinds.py` (165 lines, stdlib only) is the vocabulary we already copied (`C/core/schema/run.py:18-27`):
  - `FailureStage` (`:49`);
  - the `<server>:<kind>` extension (`:56-60`);
  - doctrine: retry, replay and mask are properties of the *occurrence*, not of the name (`:21-29`).
- `mask_sample` on `BaseVerifyResponse` (`NG/nemo_gym/base_resources_server.py:107-148`).
- `judge_failsafe` → `mask_sample=True, failure_kind="judge_failed"` (`NG/nemo_gym/judge.py:85-119`).
- Token capture masks on a missing, ambiguous or incomplete capture (`NG/nemo_gym/token_id_capture/delivery.py:74-84`).
- **The run aborts above `max_mask_fraction`** (`rollout_collection.py:2664-2676`).
- `select_measured` drops masked rows from the metrics but keeps them in coverage (`NG/nemo_gym/reward_profile.py:842-891`).
- Zero-counting is opt-in (`:990-1018`).
- Failure records: `EpisodeFailure{failure_reason, terminal, failure_kind, stage}` (`NG/nemo_gym/episode_types.py:57-86`), and `RolloutFailureRecord.delivery ∈ {not_sent, possibly_delivered, delivered}` (`NG/nemo_gym/rollout_outcomes.py:37-53`).

**Token capture:**
- the rollout id rides in the URL path `/ng-rollout/<id>/v1/...` (`NG/nemo_gym/server_utils.py:1552-1557`);
- vLLM injection of `logprobs`, `return_tokens_as_token_ids`, `return_token_ids` (`NG/responses_api_models/vllm_model/app.py:683-695`);
- `TokenEntry` (`NG/nemo_gym/token_id_capture/records.py:108-180`): `prompt_token_ids`, `generation_token_ids`, `generation_log_probs`, `routed_experts`, `parent_call_id`, `digest`, a schema version that rejects newer records;
- `WeightVersionSpan` exists only on the staging path (`staging/rebuild.py:44-52`).

**`rollout_health`:**
- offline and post-collection, 14 checks (`NG/nemo_gym/health/checks.py:26-111`), e.g. `model_call_zero_completion_tokens`, `trajectory_capture_mismatch`;
- verdict `healthy|unhealthy|unobserved`;
- **it does not set masks**.

**Reverification:**
- `REVERIFY_MODE ∈ STATELESS|UNSUPPORTED|UNKNOWN` (`base_resources_server.py:75-85`);
- refused unless forced, in which case outputs get an `unsafe_` prefix (`NG/nemo_gym/rollout_reverification.py:779-819`);
- `judge_failed_only` (`:84-91`).

**Holes:**
- two taxonomies coexist (legacy `_ng_failure_class` labels are unregistered: `:1255-1263`);
- no `blame` axis (`agent_run_error` mixes infra 500s and agent bugs, `:1614-1640`);
- `ClientSession(timeout=ClientTimeout())`, i.e. no timeout at all, and retries **forever** on disconnect (`NG/nemo_gym/server_utils.py:390-394,581-649`);
- `route_failures_to_sidecar=False` by default (`:1020-1027`).

**Recommendations:**
- `infra.py`: **PORT** `failure_kinds.py` (already done) plus the `EpisodeFailure`/`RolloutFailureRecord` shapes and the `delivery` field. **PATTERN** `mask_sample`, `judge_failsafe` and the mask-fraction abort.
- `run.py`: **PATTERN** the success-plus-sidecar files, absence-as-resume-signal, `-a<N>` attempt ids, torn-tail repair, bounded admission and the adaptive drain margin. Do not depend: the driver is welded to Hydra and its head server.
- Rollout record: **PATTERN** `TokenEntry` and `WeightVersionSpan`.
- Health: **PATTERN** the check ids, but wire their verdicts into masks.
- Reverify: **PATTERN** for regrade.

### 2.4 OpenEnv (BSD-3-Clause)

**RFC 006** (`OE/rfcs/006-agentic-rl-harness-interception.md`; code in `OE/src/openenv/core/harness/capture/`, about 6.1k lines, depending on httpx plus fastapi/starlette):
- the harness's `base_url` points at the proxy and its API key is the per-rollout session id (D5, `:272`);
- the proxy injects `return_token_ids` and logprobs and records the engine's canonical `prompt_token_ids`, with no retokenization (D3);
- 4 wire dialects (D19);
- non-streamed upstream calls, with SSE replayed to the harness (D6);
- capture levels `tokens` (trainable) vs `logprobs`/`text` (eval only) (D2, `:219-224`).

**Data shapes:**
- `TurnNode{node_id, prompt_ids, sampled_ids, sampled_logprobs, parent_id, finish_reason, harness_session_id, sampling_params, …}` (`capture/graph.py:158-206`);
- longest-prefix parent linking, where a prefix break opens a new root (D4);
- `TrainingSequence{input_ids, loss_mask, logprobs, node_ids, prompt_len, root_id, n_turns}` (`graph.py:209-236`);
- graded ingest validation `FATAL/WARN/INFO` (`capture/validate.py:27,70-369`). The RFC calls `degenerate_rollout` FATAL; the code emits WARN (`:345-353`).

**Session limits and failure handling:**
- `Session.max_model_calls` → a clean stop (`capture/sessions.py:139-184`);
- upstream retries on 408/409/429/5xx, honouring Retry-After (`capture/upstream.py:396-412`);
- **policy version is open** (D12, `:290-292`).

**Other parts:**
- Harbor's retry is deliberately off: one attempt is one session (`OE/src/openenv/harbor/rollout.py:45-50`);
- `env_reward: float|None`, with "must not synthesize" (`OE/src/openenv/core/harness/rollout.py:49-59`);
- `ContainerProvider` ABC (`OE/src/openenv/core/containers/runtime/providers.py:18-117`) is sync, one container each;
- `EnvClient` drops the WebSocket on a timeout so a stale frame cannot pair with the next request (`OE/src/openenv/core/env_client.py:680-713`);
- `CollectRunner` is sequential, and failures are only printed (`OE/src/openenv/core/harness/collect.py:333-339`).

**Recommendations:**
- `harness/endpoint.py` (capture during training): **DEPEND** on `openenv-core`'s `capture` package, or **PORT** it (vendor it, BSD-3) if gradio and fastmcp are too heavy. We add:
  - a policy version stamp per node;
  - NeMo-Gym's URL-path routing as a second mode;
  - aggregate token and wall budgets;
  - a vLLM pause/resume fence.
- Our existing `CaptureProxy` stays as the **eval-mode** recorder for hosted providers, at capture level `text`/`logprobs`.
- Validation: **PATTERN** (ingest-time FATAL masks the sample).
- Providers and collector: **SKIP**.

### 2.5 verifiers v1 (MIT; this checkout has only v1)

**Errors** (`VF/errors.py`, 132 LOC):
- `RolloutError` and boundary subclasses: `ProviderError(status_code)`, `HarnessError`, `ToolsetError`, `EnvError`, `SandboxError`, `TaskError`, `InterceptionError`, `TunnelError` (`:30-76`);
- reserved harness exit code 97 = model transport (`:45`);
- `boundary(error_cls, what)` types any escaping exception and passes typed ones through (`:80-95`);
- `stop_condition()` → `"<boundary>_error"` (`:98-103`);
- provider status mapping: timeout → 504, connection → 503 (`:106-117`).

**Harness exit classification** (`VF/harness.py:173-193`): exit 0 → OK; otherwise probe `runtime.alive()` first, and a dead box → `SandboxError`; exit 97 → `InterceptionError`; anything else → `HarnessError`. The interception server stashes the *real* model error on the session, and the rollout re-raises it (`VF/interception/server.py:850-864`; `VF/rollout.py:464-472`).

**Masking story:**
- a failed run skips scoring, but a timed-out run scores its partial trajectory (`VF/rollout.py:510-514,538`);
- reward keys are pre-seeded to `None` (`VF/task.py:253-255`);
- **but `Trace.reward` sums non-None values, so a failed trace reads 0.0** (`VF/trace.py:472-474`);
- no `masked` field, no blame axis.

**Retries:**
- `RetryRule{type, status_code: [int|"5xx"], message: regex, max_retries}` (`VF/configs/retries.py:15-35`);
- per-run budgets, first match wins (`VF/utils/retries.py:77-117`);
- full-jitter backoff capped at 30 s (`:40-42`);
- **never retries into a borrowed box** (`VF/agent.py:679-684`);
- prior errors are kept (`VF/utils/retries.py:153-157`).

**Rollout stages:**
- `RolloutTimeouts{setup, agent, finalize, scoring}` with a cumulative agent budget (`VF/rollout.py:42-53,171-175`);
- an expired deadline counts as a timeout only if *its own* context expired (`:385-392`);
- `close()` is idempotent and always tears down (`:510-621`);
- `timing{boot, setup, agent{model, harness}, finalize, scoring}` (`VF/trace.py:53-81`);
- `RolloutLimits{max_turns, max_*_tokens}` (`VF/session.py:83-113`);
- derived `is_truncated` (`VF/trace.py:620-632`).

**Eval:**
- `num_rollouts`, `max_concurrent=128` episodes (`VF/configs/cli/eval.py:91-104`);
- `push=True` by default uploads to Prime (`:116-119`) — **turn it off**;
- `traces.jsonl` with one Episode per line under a lock (`VF/utils/trace_store.py:18-60`);
- **resume keyed on `task.hash`**, with keep-verdict `env.complete`, torn lines owed again, and an atomic rewrite (`VF/cli/eval/resume.py:24-89`);
- seed: only the task shuffle.

**Interception:**
- per-rollout bearer secret multiplexing (`VF/interception/server.py:368-395`);
- SDK-retry coalescing on `x-stainless-retry-count`/`Idempotency-Key` (`:88-100,669-701`);
- 409 for stragglers, a `ModelCall` record in `finally` (`:518-581`);
- `TrainClient` does client-side tokenization and **bridges the previous prompt+completion ids** into vLLM `/inference/v1/generate`, returning `TurnTokens{prompt_ids, completion_ids, completion_logprobs}` (`VF/clients/train.py:404-457`; `VF/types.py:219-253`);
- chat dialect only (`VF/clients/train.py:354-362`).

**Trajectory graph:**
- `MessageNode{parent, message, sampled, token_ids (delta), mask, logprobs, advantages, …}` (`VF/graph.py:83-143`);
- `ModelCall{node, model, sampling, endpoint, finish_reason, usage, policy}` (`VF/trace.py:153-175`);
- `PolicySpan{start, end}` per episode (`VF/episode.py:31-61`);
- disk JSON rounds logprobs to 4 dp (`VF/graph.py:75`).

**Runtimes and harnesses:**
- `Runtime` ABC with `run` (idempotent) vs `run_program` (never replayed), `alive`, shielded `stop`, atexit backstop (`VF/runtimes/base.py:112-252`);
- shielded create through id capture (`VF/runtimes/prime.py:203-219`);
- cross-process creation limiter (`VF/runtimes/limiters.py:1-40`);
- harness contract `launch(ctx, trace, runtime, endpoint, secret, mcp_urls, data)` (`VF/harness.py:295-317`), with pinned versions (e.g. Claude Code 2.1.278);
- `HarborTaskset`: a missing reward → `TaskError` (unscored) (`VF/tasksets/harbor/taskset.py:370-403`);
- grader-only retries `verifier.retries=2` (`VF/envs/isolated_verifier/env.py:124-158`).

**research-environments:** `pi_deepswe`'s `evaluation_valid` raises when the evidence is incomplete (`RE/swe/pi_deepswe/pi_deepswe/taskset.py:147-166`), and retries only timed-out tests while keeping every attempt (`verifier.py:587-646`).

**prime-environments (legacy API):** anti-patterns.
- `mini_swe_agent_bench` returns `{}` on an sb-cli failure, i.e. scored 0 (`PE/mini_swe_agent_bench/mini_swe_agent_bench.py:261-278`).
- HumanEval assumes a timeout means model failure (`PE/humaneval/humaneval.py:42-54`).

**Recommendations:**
- `infra.py`: **PORT** `errors.py` plus `boundary()`, the `alive()` probe and stash-real-error.
- Retries: **PORT** `RetryRule`/`RetryState`, keyed on blame = infra.
- `run.py`: **PORT** the hash-keyed resume algorithm and the episode-semaphore discipline.
- `harness/verifiers`: **DEPEND**, in a *separate process/env*, because of the `mcp==2.0.0` vs `<2` conflict (`crucible/pyproject.toml`, `[tool.uv] conflicts`).
- `trajectories.py`/rollout: **PATTERN** `MessageNode` and add `policy_version` per `ModelCall`.
- Sandboxes: **PATTERN** the `Runtime` ABC.
- PE: **SKIP**, except as negative tests.

### 2.6 MiMo verl fork and MiMo v2.6 release (Apache-2.0; `third_party/mimoagent-osr` MIT; `uni_agent` Apache-2.0)

**General recipe** (`VL/recipes/general/`): mimoagent owns the loop, and only the model (`_VerlRolloutModel`, TITO) and `execute_action` (a Ray call) are swapped (`agent_loop.py:14-35`). Config: `invalid_reward_value: -999`, `filter_groups.enable`, `rollout.n: 8` (`config/general.yaml:24-120`); `env_setup_timeout` 3600, `trajectory_timeout` 1200 (`config/general_agent_loop.yaml:7-17`).

**`TokenTrace`** (`token_trace.py:72-174`):
- `token_ids`, `response_mask` (1 = policy, 0 = env), `response_logprobs`, `prompt_len`, `truncated`;
- `check_budget` at message boundaries;
- assistant turns are never re-rendered (`select_delta_messages`, `:49-69`).
- **Hole:** missing logprobs are padded with 0.0 silently (`:126-128`).

**`chat_delta`** (`chat_delta.py:46-108`): the anchor-render trick for template quirks.

**Env actor** (`env_actor.py:75-389`):
- `@ray.remote DatasetEnvActor` with setup retries, a tagged-dict result taxonomy `ok|tool_exception|transport_error|unexpected|limits_exceeded` (`:47-51`), and a `cleanup` that never raises;
- the caller maps `transport_error`/`RayError` → `InfraError` (`agent_loop.py:401-424`).

**-999 masking:**
- `_INFRA_ERROR_CATEGORIES` (`agent_loop.py:78-87`);
- `_failure_output` stubs (`:635-727`), where every field exists because a production batch crashed without it;
- GRPO substitutes the group's valid mean, then forces the row's advantage to 0 (`VL/verl/trainer/ppo/core_algos.py:306-350`);
- metrics exclude masked rows (`VL/verl/trainer/ppo/metric_utils.py:477-486`);
- `DROP_INFRA_FROM_GROUP` (`VL/verl/trainer/ppo/v1/trainer_base.py:2023-2043`);
- infra-aware `filter_groups` (`VL/verl/trainer/ppo/v1/replay_buffer.py:255-268`).

**Bugs (all verified or reported with line refs):**
1. `pod_conn_timeout: {e}` never matches the frozenset, so it scores 0 (`VL/recipes/general/env_actor.py:282` vs `agent_loop.py:934`). Design fixed it; general and arvo did not.
2. design/arvo omit `reward/testbed_corrupted`.
3. general never sets `is_infra`, so `[1, 1, -999]` passes the variance filter.
4. Inference-server failures are graded as agent failures (`VL/third_party/mimoagent-osr/src/mimoagent/agents/base.py:247-250`).
5. `asyncio.gather` without `return_exceptions` kills the batch (`VL/verl/experimental/agent_loop/agent_loop.py:673`).
6. `seq_timeout` is masked, so slow agents escape their penalty.
7. Mean substitution shrinks the group std.

**uni_agent gateway** (`VL/third_party/uni_agent/uni_agent/`):
- `/sessions/{id}/v1/chat/completions`, `/v1/messages`, `/v1/responses` (`gateway/gateway.py:133-158`);
- `generation_spans` plus logprobs (`session/types.py:44-90`);
- **per-generation `generation_versions`** (`session/session.py:50-65,281-288`), collapsed at finalize (`:893-897`);
- prefix-hash chains and rollback (`:443-692`);
- enforced logprob alignment (`:293-301`);
- `abort()` returns an unscorable snapshot (`:369-400`);
- `gradable: False` and drop-and-refill for failures (`framework/framework.py:219-249,1095-1140`);
- an idle timeout for external agents (`:207-217`).

**MiMo general env runtime:**
- `manifest.json` with uploads, a sidecar that owns state, MCP and the verifier, and late-uploaded verifier files (`VL/recipes/general/general_agent/environment.py:37-92`);
- "a missing `reward` key means a testbed fault" (`MM/general/envs/*/run_verify.py:102-151`);
- the sidecar supervisor restarts MCP servers silently (a hole).

**Recommendations:**
- Rollout record: **PORT** the `TokenTrace` and uni_agent `Trajectory` shapes, keeping per-span versions (a two-line change) — the ARCHITECTURE M12 line already says DEPEND on recipes/general for training.
- `harness/endpoint.py`: **PATTERN** the uni_agent gateway (it is welded to Ray and TransferQueue): per-session base_url, three dialects, prefix chains, enforced logprobs, abort snapshot, plus a typed infra header instead of a bare 500.
- `infra.py`: **PATTERN**, with an explicit enum matched by kind, never by message string, and a test that every infra path reaches the mask.
- `run.py`: **PATTERN** the separate setup/trajectory/reward timeouts, the dedicated thread pool for blocking agents, a cooperative stop plus kill in `finally`, and `gather(return_exceptions=True)`.

### 2.7 mini-swe-agent (MIT)

**Protocols** (`MS/__init__.py:43-80`): `Model.query`, `Environment.execute/serialize`, `Agent.run/save`.

**Environments** (`MS/environments/__init__.py:8-16`): docker, singularity, local, swerex_docker, swerex_modal, bubblewrap, contree.

**Docker env:**
- `docker run -d … sleep 2h` (`MS/environments/docker.py:74-99`);
- a command timeout or exception becomes `returncode -1` plus `exception_info`, i.e. an observation (`:126-136`);
- cleanup only from `__del__` (`:153-161`).

**Exit taxonomy** (`MS/exceptions.py:1-26`): `Submitted`, `LimitsExceeded`, `TimeExceeded`, `UserInterruption`, `FormatError`. Other exceptions set `exit_status=type(e).__name__` (`MS/agents/default.py:74-86`).

**Agent loop:**
- the trajectory is saved after every step in `finally` (`:120-121`);
- defaults `step_limit=0, cost_limit=3.0` (`:26-33`);
- litellm cost raises unless `cost_tracking=ignore_errors` (`MS/models/litellm_model.py:36,108-126`);
- `GlobalModelStats` raises an untyped `RuntimeError` at the global limit (`MS/models/__init__.py:13-42`).

**Batch** (`MS/run/benchmarks/swebench.py`):
- `ThreadPoolExecutor` (`:257`), Ctrl-C drain (`:266-271`);
- skip-existing over `preds.json` (`:229-232`), deleting the stale entry first (`:131-133`);
- **an env-start failure writes an empty patch, scored 0 forever** (`:157-160,176`).

**Trajectory:** `{info:{model_stats, config, mini_version, exit_status, submission}, messages, trajectory_format:"mini-swe-agent-1.1"}` (`MS/agents/default.py:159-180`). This is the format of our historical runs. Harbor converts it to ATIF (`C/interop/harbor/atif_import.py:104`).

**Recommendations:**
- `harness/`: **DEPEND via Harbor's installed agent** for frontier runs, pinned to 2.4.6 as in the historical runs.
- Native loop: already **PORTed** (`BashLoopPolicy`).
- `run.py`: **PATTERN** the Ctrl-C drain and delete-stale-before-start, but never its empty-pred-on-infra behaviour.
- Exit taxonomy: **PATTERN**, mapped into our stop conditions.

### 2.8 FineEnvs multi-harness RL (Apache-2.0; the HF space is CC-BY-4.0 prose only)

**Harnesses and capture:**
- `HARNESSES=("opencode","claude-code","codex","mini-swe-agent")`, with `HARNESSES[index % 4]`, so one harness per GRPO group (`FE/envs/harbor/smoldataenv_harbor/environment.py:9,74`);
- a Daytona Harbor session per rollout;
- tasks matched by prompt, not seed, because "seeds can change after a restart" (`:66-67`);
- `harness_adapter=None`: the OpenEnv capture proxy captures every harness (`FE/train/multi_harness.py:145`);
- ATIF tool-call validation returns `None` (not 0) on a malformed ATIF (`environment.py:12-28`);
- proxy-vs-ATIF cross-check per call (turns, completion tokens, prompt_len), where a mismatch fails the rollout (HF space `openenv.mdx:289-298`);
- `reward()` returns `None` when `env_reward is None`: "Infrastructure failures do not become wrong answers" (`FE/train/multi_harness.py:78-80`).

**Eval runner** (`FE/eval/evaluate.py`):
- atomic `write_json` (`:17-21`);
- the grade must be exactly 0/1 (`:49-50`);
- **a run-identity file** (protocol sha256, checkpoint, harnesses, task names, instruction hash) refuses a mismatched out dir (`:191-208`);
- one file per (task, harness), reused only if `correctness in (0,1)` (`:214-220`);
- exceptions persisted as `correctness=None` with secrets redacted (`:234-243`);
- `coverage = graded/expected`, exit code 2 if incomplete (`:140-157,288`).

**Recommendations:**
- `run.py`: **PORT** the evaluate skeleton (identity, per-unit atomic files, final-only reuse, coverage).
- `trajectories.py`: **PATTERN** the proxy-vs-ATIF cross-check.
- Frontier calibration: **PATTERN** one harness per group, harness by index.

### 2.9 Karotte (MIT)

**Run status and failure split:**
- `RunStatus = pending|running|passed|failed|error` (`KAR/schemas/__init__.py:8`);
- an exception → `ErrorEvent` plus `status="error"`, transcript saved in `finally` (`KAR/evaluation_runner.py:88-106`);
- `StudentMisbehaviorError` "scores the run 0 instead of treating it as an infra error" (`KAR/student_misbehavior.py:5-18`; caught in `KAR/step.py:83-92`).

**Limits:**
- `TurnLimitReachedError`, `StepTimeLimitReachedError`, `CliAgentExitedError`, `StepContextWindowLimitReachedError`, `EmptyTurnLimitReachedError` (`KAR/agents/agent.py:13-32`);
- **per-limit policy `on_step_time_limit` / `on_step_context_window_limit: "error"|"score"`** (`KAR/schemas/evaluation_run_config.py:52,67`).

**LLM retry:** 16 attempts honouring Retry-After up to 300 s; separate budgets for auth (3) and unreachable (4) (`KAR/agents/builtin_source.py:46-90`).

**Other:**
- GPU forces docker (`KAR/hardware.py:45-47,117-123`);
- cgroup and cohort kill — we already PORTed this into `world/guard`;
- typed event-stream transcript (`KAR/schemas/transcript.py:19-180`).

**Recommendations:**
- `infra.py`: **PORT** the misbehavior/limit split and the `error|score` per-limit policy, which becomes our `LimitPolicy`.
- LLM retry classifier: **PATTERN** (our `LLMClient` already follows it).
- Runner: **SKIP**.

### 2.10 Repo2RLEnv (Apache-2.0)

**`BudgetLedger`** (`R2E/campaigns/budget.py`):
- SQLite with `BEGIN IMMEDIATE` (`:108-120`) and integer micro-dollars rounded up (`:16,72-76`);
- `operations(id, description, reserved_micros, actual_micros, status ∈ {reserved, uncertain, settled}, evidence)` (`:88-97`);
- `reserve(op_id, usd, desc, scopes=((prefix, limit), …))` with typed denials `request_exceeds_limit|accounted_exhaustion|uncertain_reservations|reservation_pressure` (`:27-53,132-182`);
- a duplicate op id raises `OperationAlreadyRecorded` (`:151-152`);
- `settle` requires evidence and is idempotent (`:226-245`); `mark_uncertain` (`:214-224`);
- `increase_limit(expected, evidence)` (`:184-212`).

**Metered calls** (`R2E/campaigns/llm.py:17-100`):
- reserve → receipt `dispatched` → call → receipt `completed` → settle, or mark uncertain when usage is missing;
- resume replays a completed receipt only on a matching request hash.

**Sandboxes as spend** (`R2E/execution/lifecycle.py:34-97`):
- `creation_uncertain`/`termination_uncertain`;
- a GPU cost estimate labelled `conservative_estimate_not_invoice` (`R2E/execution/harbor_modal.py:45-66`).

**Durability and retries:**
- `save_record` with fsync and `os.replace` (`R2E/execution/lifecycle.py:19-31`);
- an fsynced JSONL `EventJournal` that tolerates a torn tail (`R2E/campaigns/events.py:26-53`);
- `retry_read` only for observational operations (`R2E/execution/read_retry.py:1-37`).

**Holes:** 402/insufficient-quota is not classified, and there is no wall-clock stop.

**Recommendations:**
- `budget.py`: **PORT** the ledger and the receipt protocol (about 250 LOC). It supersedes `C/forge/budget.py`, and forge should move onto it.
- Journal: **PORT** `save_record` and `EventJournal`.

### 2.11 EnterpriseOps-Gym (Apache-2.0)

- `TaskQueueWorker`: an asyncio `FIRST_COMPLETED` pump (`EOG/utils/task_queue_worker.py:9-39`);
- `run_{i}/` folders for k (`EOG/evaluate.py:316-329`);
- resume skips existing result files, **even error ones** (`EOG/benchmark_utils.py:51-57`);
- whole-sample retry ×5 on any error (`EOG/evaluate.py:221-233`);
- every exception retryable in the LLM client (`EOG/benchmark/llm_client.py:363-368`);
- errors counted in `overall_success_rate` (`EOG/benchmark/executor.py:567-572`).

**Recommendation:** **PATTERN** the queue pump and the per-run DB clone. **SKIP** the scoring, which is a negative test.

### 2.12 SWE-smith (MIT)

- `run_patch_in_container`: container named `swesmith.{eval|val}.{run_id}.{iid}`, `mem_limit="10g"`, a `GIT_APPLY_CMDS` ladder, `eval.sh` with start/end markers, `exec_run_with_timeout`, cleanup in `finally` (`SS/harness/utils.py:70-283`);
- **an exception returns `(logger, False)`, the same as a clean run** (`:273-284`);
- missing output is written as `timed_out`, after which resume skips it (`SS/harness/valid.py:117-127,177-185`).

**Recommendation:** **PATTERN** the log markers, report-as-done marker, apply ladder and deterministic container names. Use a tri-state result and never write a done marker on an infra error.

### 2.13 Cross-framework matrix (M10 piece × framework)

| M10 piece | Harbor | agentenv | NeMo-Gym | OpenEnv | verifiers | MiMo verl | mini-swe | FineEnvs | Karotte | Repo2RLEnv | EOG / SWE-smith |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `run.py` (k, seeds, concurrency, resume) | DEPEND `Trial`; PATTERN lock/regrade; PORT pass@k | PORT Runner/store; PATTERN journal | PATTERN resume/sidecar/drain | SKIP | PORT resume | PATTERN timeouts | PATTERN drain | PORT evaluate | SKIP | PORT journal | PATTERN pump / markers |
| `harness/` installed agents | **DEPEND** | DEPEND protocol (A2A) | — | — | DEPEND (separate env) | — | via Harbor | PATTERN 1 harness/group | — | — | — |
| `harness/endpoint.py` (training) | — | — | PATTERN URL routing | **DEPEND/PORT capture** | PATTERN TITO bridge, coalescing | PATTERN uni_agent | — | PATTERN cross-check | — | — | — |
| `collect.py` | PATTERN `[[verifier.collect]]` | — | — | — | PATTERN late staging | PATTERN late upload | — | — | PORTed no-follow | — | PATTERN apply ladder |
| `infra.py` | PATTERN `ERROR_PATTERNS` | wire `error_type` | **PORT kinds + records** | PATTERN validate | **PORT errors/boundary/RetryRule** | PATTERN stubs | PATTERN exits | PORT reward=None | **PORT misbehavior/limits** | PATTERN completed rule | negative tests |
| `budget.py` | meter only | — | PATTERN drain/mask-fraction | PATTERN call cap | — | — | PATTERN cost_limit | — | — | **PORT ledger + receipts** | — |
| `perf_guard.py` | PATTERN regrade | — | — | — | PATTERN grader-only retry | — | — | — | — | — | — |
| `gpu.py` | DEPEND env + GpuDocker | DEPEND provider + gpu_docker | — | — | PATTERN Runtime ABC | — | — | — | PATTERN GPU→docker | PATTERN GPU cost estimate | — |
| `trajectories.py` | **DEPEND ATIF** | SKIP | — | PATTERN graph | PATTERN MessageNode | — | via Harbor conv. | PATTERN cross-check | PATTERN event stream | — | — |
| rollout record v1 | `RolloutDetail` | — | PATTERN `TokenEntry` + versions | PATTERN `TurnNode`/levels | PATTERN node/PolicySpan | **PORT** TokenTrace / gateway versions | — | — | — | — | — |

---

## 3. Design recommendation for `crucible.runtime`

### 3.1 Module layout

ARCHITECTURE rule: `runtime/` never imports a framework; all framework calls go through `interop/`.

```
src/crucible/runtime/
  __init__.py
  keys.py          # SampleKey, AttemptKey, run identity hash
  run.py           # RunSpec, Runner (scheduler: k × seeds × harnesses, concurrency pools, resume, stop rules)
  journal.py       # append-only fsynced journal + atomic record files (Repo2RLEnv EventJournal/save_record, NeMo-Gym torn-tail repair)
  infra.py         # Failure taxonomy (kind × stage × blame × mask × retry), classify(), RetryRule (verifiers), LimitPolicy (Karotte), normalizer for verifier kinds
  budget.py        # BudgetLedger (Repo2RLEnv PORT): run/task/policy scopes, reserve/uncertain/settle, credit-exhaustion stop
  collect.py       # tolerant collect + empty-collection discrimination (wraps verify/collect.py COLLECT_SH)
  perf_guard.py    # host load guard, GPU-grade serialization per host, contamination → node_contention, regrade queue
  gpu.py           # GpuHost registry (lab5090, runpod, modal), preflight probe (arch, driver, PID-1 init), placement
  trajectories.py  # ATIF v1.8 emission + step-id map aligned with steps.jsonl; proxy-vs-ATIF cross-check
  record.py        # producer side of crucible.rollout/v1 (schema owned by M12 train/rollout.py; M10 writes it)
  inject.py        # fault injectors for the 8 B.5 classes + policy-caused controls (exit gate, also CI)
  harness/
    base.py        # Harness protocol + HarnessResult
    native.py      # seam W: ScriptedPolicy / LLMToolPolicy / BashLoopPolicy through run_episode
    harbor.py      # seam S: installed agents via interop/harbor/job_bridge.py (Trial per attempt)
    agentenv.py    # A2A agents via interop/agentenv (wire error_type + Usage)
    verifiers.py   # verifiers harnesses in a subprocess env (mcp conflict); JSONL over stdio
    endpoint.py    # OpenAI/Anthropic/Responses-compatible endpoint for policies in training (capture proxy, TITO, policy versions)
interop/harbor/job_bridge.py     # TrialConfig construction, GpuDockerEnvironment, AgentContext/ExceptionInfo extraction, regrade
interop/agentenv/gpu_provider.py # [sandbox.providers.gpu_docker]
```

### 3.2 Core classes

```python
class SampleKey(CrucibleModel):          # the identity resume keys on (no framework has one)
    task: str            # task content hash (Harbor checksum / C/core/ids.py), pinned version
    world: str           # world spec hash
    policy: str          # model id or checkpoint@policy_version
    harness: str         # "harbor:mini-swe-agent@2.4.6" | "native:bash-loop@1" | "a2a:<url>@<card-version>" | "verifiers:<name>@<ver>"
    sample_idx: int      # 0..k-1
    seed: int            # derive(run_seed, task, policy, harness, sample_idx): sampling seed sent to the endpoint
    tier: Literal["fast","full"] = "full"
    def id(self) -> str  # sha256 of the canonical JSON, 16 hex
# AttemptKey = f"{sample_id}-a{n}" (NeMo-Gym rollout_id-a<N>): capture sessions, sandboxes, container names and logs use it.

class RunSpec(CrucibleModel):
    run_id: str; tasks: list[TaskRef]; policies: list[PolicyRef]; harnesses: list[HarnessRef]
    k: int; run_seed: int
    concurrency: Concurrency      # global, per-harness, per-gpu_host, per-task (length-bias control, M12)
    limits: EpisodeLimits | HarnessLimits; limit_policy: LimitPolicy   # Karotte: on_timeout / on_context_window → "score"|"mask"
    retry: list[RetryRule]        # verifiers RetryRule; only blame ∈ {infra, harness} rules may retry
    budget: BudgetSpec            # cap_usd, scopes, uncertain_tolerance, wall_clock_s, max_mask_fraction
    tier: Literal["fast","full"]; regrade: RegradePolicy

class Runner:
    async def run(self, spec: RunSpec) -> RunSummary       # idempotent: re-running = resume
    async def regrade(self, run_id, selector) -> RunSummary # from stored artifacts (Harbor regrade pattern)
    def status(self, run_id) -> RunStatus                   # coverage, masked share, spend, ETA

class Harness(Protocol):
    id: str; capabilities: HarnessCaps     # tokens (capture level), snapshots (seam W), gpu, resume
    async def run(self, attempt: Attempt, env: SandboxLease, endpoint: EndpointLease, budget: BudgetLease) -> HarnessResult

class HarnessResult(CrucibleModel):
    stop: str; truncated: bool; failures: list[Failure]
    trajectory: TrajectoryRef        # ATIF document (store key)
    capture: CaptureRef | None       # proxy log / rollout record
    usage: Usage; cost_usd: float; cost_uncertain_usd: float
    artifacts: dict[str, str]        # model.patch etc. (store keys)
    timings: dict[str, float]        # boot / setup / agent{model,harness} / collect / verify / teardown (verifiers timing)
```

The run loop for one attempt is a fixed phase machine. Each phase runs under `infra.boundary(stage=…)`, which follows verifiers' `boundary()`:
1. **admission:** ledger reserve, then gpu placement plus preflight.
2. **seed:** sandbox start, image fingerprint check, world reset, collect baseline.
3. **agent:** harness run, through the endpoint, with the budget lease.
4. **collect:** tolerant collect plus empty-collection discrimination.
5. **verification:** fingerprint the verifier image, then `docker_grade` (fast or full tier) with perf_guard, then `mask_for`.
6. **cleanup:** shielded teardown. A failure here can never override a reward, unlike Harbor's `trial.py:1875-1881`.

**Ordering rule from `run_agent.sh`: always collect before classifying.** A crashed agent's workspace is still collected and graded unless the crash was infra. Grading a workspace that infra damaged would produce a measurement nobody trusts.

### 3.3 Failure taxonomy (`infra.py`)

One record type extends `C/core/schema/run.py:57-66` with two new fields, `blame` and `delivery`:

```python
class Failure(CrucibleModel):
    kind: str                     # FAILURE_KINDS (NeMo-Gym verbatim + crucible:*); never a free string
    stage: FailureStage           # admission | seed | agent | verification | cleanup
    blame: Literal["infra", "harness", "task", "policy", "unknown"]
    mask: bool                    # derived from blame unless overridden by LimitPolicy; policy ⇒ False
    retryable: bool               # matched against RetryRule; policy ⇒ False ("retry until ok" is survivorship bias)
    delivery: Literal["not_sent", "possibly_delivered", "delivered"] = "delivered"   # NeMo-Gym; drives budget settle
    detail: str | None = None     # original subkind, e.g. "no_reward_json", "RewardFileNotFoundError"
    reason: str = ""
```

Rules:
- `blame=policy` → scored (normally 0), never masked, never retried. `blame ∈ {infra, harness, task, unknown}` → masked, with `reward=None`.
- `task` blame (a broken grader or task) also files a defect with M9.
- `unknown` is masked **and** counted against `max_mask_fraction`. If too many samples are unknown, the run aborts instead of silently masking.

**Classifier inputs, in order of trust:**
1. Typed exceptions from our own code.
2. The `ExecResult`/rc of the phase, plus an `alive()` probe of the sandbox. This is verifiers' rule: a non-zero exit from a dead box is infra.
3. The verifier's `infra.status` and `failure_kind`, normalized.
4. Harbor `ExceptionInfo.exception_type` and the `ERROR_PATTERNS` class, with the phase known.
5. A2A `TaskError.error_type`.
6. Capture-proxy evidence: upstream status codes, blank usage, missing token records.

One deliberate deviation from MiMo: **a wall-clock or step-limit stop of the agent is `policy` by default** (`agent_timeout`, scored). `LimitPolicy` can switch it to `mask`, as Karotte's `on_step_time_limit` does.

`truncated` is a stop condition, not a failure. M12 decides whether a truncated sample is masked or Φ-bootstrapped (ARCHITECTURE M12 `credit.py`).

**Taxonomy table.** The first 8 rows are the Appendix B.5 classes; rows marked "new" are kinds to add to `CRUCIBLE_FAILURE_KINDS`.

| # | Class (B.5) | Detection in M10 (stage) | kind | blame | mask | retry | Remedy recorded |
|---|---|---|---|---|---|---|---|
| 1 | Empty patch: a gitignored `build/` made `git add` fail inside `&&` | `collect.py`: patch empty **and** (`git status --porcelain` non-empty, or the hook's error list non-empty, or the patch is empty while the tree fingerprint ≠ the baseline) (collect) | `crucible:empty_collection` | harness | yes | yes, **re-collect only** from the preserved sandbox (or from the WorldRef); never re-run the agent | tolerant hook used; legacy chains rejected at import (A11) |
| 2 | Blank-usage API response aborted the run | endpoint/proxy/LLMClient: 200 with no `usage` → cost estimated from tokens (or `max_tokens` × price) and held **uncertain** in the ledger; the run continues. Masked only if the uncertain share exceeds `uncertain_tolerance` or the cap could have been crossed (agent) | `crucible:usage_blank` | infra | only above tolerance | yes | `cost_uncertain_usd` on the record |
| 3 | Out of API credits mid-run | 402, or 429 `insufficient_quota`, or a ledger denial `accounted_exhaustion` (agent/admission) | `crucible:credits_exhausted` (provider side: `provider_quota_exhausted`) | infra | yes | yes, **after the stop rule clears**: the run halts dispatch, in-flight attempts finish as masked, and resume continues | stop rule fired, recorded in the journal |
| 4 | Zombie wait loops (`sleep infinity` as PID 1) | preflight: `cat /proc/1/comm` ∈ {`docker-init`, `tini`, `init`}, else refuse to start; during the run: exec deadline hit while `ps -eo stat` shows `Z` children of PID 1 (seed/agent) | `crucible:zombie_wait` | infra | yes | yes, fresh sandbox with `--init` | sandbox argv audit (A11) |
| 5 | Stale verifier image | before grading: `image_id()` RootFS-layer digest (`C/world/sandbox.py:49-63`) ≠ the digest pinned in the task card / `lock.json`; grading never starts (verification) | `crucible:stale_image` | infra | yes | yes, after rebuild/pull | expected vs actual digest |
| 6 | ABI mismatch after a header change | the grader records `consumer_build_inputs` (hash of headers/config the consumers compiled against) and the runtime compares it with the candidate tree's hash; also the M7 rule "always rebuild consumers". A mismatch is infra (verification) | **new** `crucible:stale_build` | task | yes | yes, regrade with a forced rebuild; files an M9 defect | build-input hashes |
| 7 | GPU/CPU contention from other users | `perf_guard`: host guard before dispatching a perf grade (foreign `nvidia-smi --query-compute-apps` PIDs, GPU utilization, load/core) plus the grader's `LoadGuard`/spread `InfraError` (`perf.py:40-128`) → normalized (verification) | `crucible:node_contention` | infra | yes | yes, **regrade from artifacts** later (Harbor regrade); at most N regrades, then `unknown` | load samples, spread table |
| 8 | Self-matching `pkill -f` over ssh | the harness's own control channel dies: ssh rc 255 / SIGTERM with no remote result while `alive()` says the box is up; orchestrator processes are tracked by **pidfile** and `pgrep` patterns use `[x]yz`; any `pkill -f`/`pgrep -f` in runtime code fails a lint (agent/cleanup) | **new** `crucible:harness_self_kill` | harness | yes | yes | pidfile path |
| — | App crashed / tool timeout | `World._call_tool` (`world.py:330-339`) | `crucible:tool_crashed` / `transport_timeout` | infra | yes | yes | traceback in the store |
| — | Sandbox/provider failed to start | phase seed; `alive()` false | `crucible:backend_error` / `provider_unavailable` | infra | yes | yes | — |
| — | Verifier crashed / no reward file / `check_infra` / `grader_exception` | normalized from `stage.py:99`, `plan.py:194-196`, Harbor `RewardFile*Error` with the script's rc kept | `verifier_error` (detail = original) | task | yes | grader-only retry ×2 (verifiers), never the agent | — |
| — | Verifier timeout | `subprocess.TimeoutExpired` around `docker_grade` (fixes H5), named container killed | `verifier_error` (detail `verifier_timeout`) | task | yes | grader-only | — |
| — | Token record missing / FATAL capture validation | `record.py` + OpenEnv-style validate | **new** `crucible:capture_incomplete` | infra | yes (training) / no (eval: ignore) | yes | validation codes |
| — | Answer exposed (P27) | `run_episode` probe (`policy.py:256-258`) | `crucible:answer_exposed` | task | yes | no | M9 defect |
| — | Policy defeated a verification boundary | `policy.py:250-254`, Karotte misbehavior | `crucible:policy_misbehavior` | policy | **no** (score 0) | no | — |
| — | Agent broke the environment (deleted the repo, killed its own sandbox's services) | the World/sandbox is alive but its state invariants fail after a policy action (the survey's `agent_broke_env`) | **new** `crucible:agent_broke_env` | policy | **no** (score 0) | no | — |
| — | Agent hit step/wall/cost limit | stop `max_steps`/`timeout`; mini-swe `LimitsExceeded` | `agent_timeout` (stop) | policy (LimitPolicy may mask) | no by default | no | — |
| — | Context overflow | `ContextOverflow` (`llm.py:45-49`) | stop `context_overflow` | policy | no (M12 decides truncation handling) | no | — |
| — | Provider 5xx/timeout after retries | `LLMError`, Harbor `ApiInternalServerError`/`ApiOverloadedError` | `provider_unavailable` | infra | yes | yes | — |
| — | Harness built an invalid request (4xx) | `RequestRejected` | `agent_request_failed` | harness | yes | no (deterministic) → alarm | — |

**Normalizer.** `infra.normalize(kind)` maps every legacy or foreign name onto `FAILURE_KINDS`:
- `no_reward_json`, `check_infra` and `grader_exception` become `verifier_error` with `detail` set;
- Harbor exception names map through a table;
- the MiMo categories `setup/failed → crucible:backend_error`, `rollout/pod_conn_timeout → transport_peer_drop`, and so on.

`RunRecord.mark_failure` then never raises on a known source.

**Reward accessor.** `RunRecord.reward` is `None` whenever `mask_sample`. Every aggregate takes `Iterable[RunRecord]` and drops masked records from numerators and denominators, then reports coverage = scored / planned. This applies to `metrics.pass_at_k` (Harbor's unbiased estimator per sample group, over scored samples), mean reward and M8 Calibrator rates. A test asserts that no aggregate reads a masked record as 0, which is the guard against Harbor's `metrics/base.py:25`.

### 3.4 Journaling and resume

**On-disk layout**, under `data/runs/<run_id>/`. The design combines FineEnvs identity, NeMo-Gym sidecars, Harbor lock, verifiers' hash-keyed resume and Repo2RLEnv's journal.

```
run.json            # RunSpec + identity: sha256(RunSpec minus cosmetic fields), crucible git sha, framework pins (Harbor 0.23.0@a2fafbc, …),
                    #   task digests, image layer digests, harness versions, endpoint/model ids. Mismatch on resume ⇒ refuse (FineEnvs evaluate.py:191-208)
journal.jsonl       # append-only, fsync on state change; events: planned | dispatched{attempt} | phase{stage,t} | budget{op,status} |
                    #   finished{attempt,outcome,reward|null,failure} | retry_scheduled | stop_rule{name} | regrade_*; torn tail tolerated/repaired
results.jsonl       # one RunRecord per *terminal* sample (scored, policy-failed, or masked-terminal); written after the attempt dir is complete
failures.jsonl      # every masked attempt (NeMo-Gym sidecar): never read by metrics, read by status/coverage/retry
ledger.sqlite       # BudgetLedger (Repo2RLEnv), op ids = attempt keys and "<attempt>/call-<n>"
samples/<sample_id>/a<N>/
    record.json            # RunRecord (atomic tmp+fsync+rename)
    trajectory.atif.json   # ATIF v1.8 (Harbor models), steps.jsonl alongside
    steps.jsonl
    rollout.jsonl          # crucible.rollout/v1 (training runs; capture level ≥ tokens)
    capture/<session>.jsonl  # raw proxy log (eval) / TurnNode graph (training)
    artifacts/model.patch, artifacts/manifest.json
    verifier/{reward.json, crucible_reward.json, report.json, console.log}
    harbor/                # the Trial dir when seam S (kept, never rmtree'd on retry — Harbor queue.py:222 is avoided by owning retries)
```

**Resume rule.** Re-running `Runner.run(spec)` with the same identity resumes:
- **done** = terminal in `results.jsonl`: scored, policy-failed, or masked with `retryable=False` or attempts exhausted (NeMo-Gym's `gated = success ∪ terminal ∪ maxed`).
- **owed** = planned keys with no terminal row. Their next attempt index is 1 + max(a in `failures.jsonl`) for that sample.
- Attempt dirs without `record.json` are moved to `samples/<id>/aborted/` (kept as evidence, as opposed to Harbor's delete).
- The ledger reconciles `reserved` operations left without a settle into `uncertain`. Proxy logs or `AgentContext` provide evidence to settle them; nothing is re-dispatched blindly (Repo2RLEnv).
- Infra-masked samples are **retried on resume by default**. Harbor's default keeps them as done.

**Seam W snapshots.** For native harnesses, a mid-episode crash can also resume *within* an attempt from the last journaled `WorldRef` plus the policy message prefix (`EpisodeLimits.snapshot_every`). This is for C3 branching. For evaluation, the attempt restarts cleanly; this is verifiers' "never retry into a borrowed box" rule.

**Stop rules**, checked by the scheduler before each dispatch:
1. Budget: a ledger denial `accounted_exhaustion` at run scope, or a provider credit exhaustion. Dispatch stops. The run ends `paused:budget` with exit code 3, and is resumable after `increase_limit`.
2. Wall clock: NeMo-Gym's dispatch budget plus an adaptive drain margin (p75 of observed attempt durations).
3. **Mask fraction:** more than `max_mask_fraction` (default 0.2) masked after 20 terminal samples. Pause and alarm: an environment that masks a fifth of its samples is broken, not noisy.
4. **Unknown-blame fraction > 0.05:** pause, because the classifier needs a rule.
5. Ctrl-C: the first press drains in-flight attempts; the second cancels them into `cancelled`, retryable.

### 3.5 Budget (`budget.py`)

- PORT the `BudgetLedger` with scopes `run`, `run/policy/<id>`, `run/task/<hash>` and `sample/<id>`.
- Every metered thing is an operation:
  - model calls, settled from LLMClient spend / proxy usage / `AgentContext.cost_usd`;
  - sandbox-seconds per GPU host, using a GPU rate (`conservative_estimate_not_invoice`, as R2E does);
  - verifier GPU time.
- Per-attempt reservation = `harness.estimate(task)`, e.g. the historical $2.29 for task02 sol, with a floor.
- The harness-side caps (`--cost-limit`, `--max-budget-usd`) are set to the reservation, so the agent stops itself before the ledger has to stop it. Agent-side `LimitsExceeded` is a policy stop.
- `forge/budget.py` is migrated onto this ledger, so M8 and M10 share one source of truth.

### 3.6 Perf guard (`perf_guard.py`)

- **Per-host GPU semaphore**, default 1 grade at a time on `lab5090`. Agent sandboxes and perf grades never overlap on a shared host.
- **Pre-dispatch host check:** foreign compute PIDs (`nvidia-smi --query-compute-apps=pid,used_memory`), load per core and GPU utilization. If busy, wait up to `wait_s`, then defer the grade, not fail it.
- The grader-side `LoadGuard` plus round spread stays in M7. `perf_guard` maps its `InfraError` to `crucible:node_contention` and enqueues a regrade from artifacts.
- After `max_regrades` (default 3), the grade becomes `unknown` and is counted against the stop rule.
- **Round spread across grades:** when two grades of the same artifact disagree on a perf gate, both become infra. This reproduces the `astra_v3` → `astra_v3b` case (B.12).

### 3.7 GPU placement (`gpu.py`)

`GpuHost` comes from `crucible.toml [gpu_hosts]` and has a `kind`:

| kind | Sandbox | GPU flags | Network | Notes |
|---|---|---|---|---|
| `docker_ssh` (lab5090) | `DockerSandbox` / Harbor `GpuDockerEnvironment` / agentenv `gpu_docker` over `DOCKER_HOST=ssh://` | `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` (`--gpus` fails on this host) | `--network none` | `shared=true` → polite mode (perf_guard semaphore, idle wait, label-based cleanup sweep `crucible.sandbox=1`) |
| `ssh_exec` (Runpod) | **no Docker inside the pod** (no user namespaces): the pod *is* the sandbox; `SSHExec` (port of `MT/common/mswea/ext.py` `SSHEnvironment`) | native | cannot be cut off → regex network guard + post-run audit; the record carries `network_enforced=false`, so it is ineligible for calibration (A1 "declared environment") | one sample per pod |
| `harbor_cloud` / `agentenv_cloud` (Modal) | Harbor `environment_type=modal` or `C/interop/agentenv/sandbox.py` with `gpu=` | provider-side | provider policy | datacenter GPUs only: a task pinned to `gpu_arch=sm_120` refuses placement there unless the card allows it |

**Preflight on every lease** (fixes B.7 "GPU silently CPU-only"):
- `nvidia-smi --query-gpu=name,compute_cap,driver_version` inside the sandbox must match the task's `gpu_arch`;
- the PID-1 init check from row 4;
- the image digest check from row 5.

A preflight failure is `crucible:backend_error` (infra) and never reaches the agent.

### 3.8 Trajectories (`trajectories.py`): ATIF with step ids that line up with `steps.jsonl`

**One canonical map per attempt.** `StepMap` rows `(world_step, atif_step_id, call_id, harness, actor)` are written as `extra.crucible_map` in the ATIF and used to fill `StepRow.atif_step_id` and `StepRow.call_id`. This fixes H8.

**Seam W (native).** `episode_to_atif` is extended:
- every model call becomes an ATIF agent step, including format-error turns with `extra.format_error=true` and no tool calls, so `llm_call_count`, metrics and call ids are complete;
- `Step.metrics = {prompt_tokens, completion_tokens, cached_tokens, cost_usd}` from `model_calls`, plus `prompt_token_ids`/`completion_token_ids`/`logprobs` when the endpoint captured them;
- `Step.extra = {crucible_step, call_id, harness}`;
- `FinalMetrics` totals.

**Seam S (installed agents).**
- The ATIF comes from Harbor's converter.
- `call_id`s come from the capture proxy log. They are joined by order and by request digest, then cross-checked per call on turn count, completion tokens and prompt length (FineEnvs). A mismatch at training capture level marks the record `crucible:capture_incomplete`; at eval it only warns.
- World step indices come from re-segmenting the trajectory with M5 (shell-effect replay), so `steps.jsonl` rows point at ATIF step ids.

**Contract test.** For each harness, `len({r.atif_step_id for r in steps.jsonl}) == number of scored ATIF steps`, and every `call_id` resolves to a proxy record.

### 3.9 `crucible.rollout/v1` (produced by M10, owned by M12)

```
RolloutRecord {
  schema: "crucible.rollout/v1", attempt: AttemptKey, sample: SampleKey,
  harness: "name@version", capture_level: "tokens"|"logprobs"|"text", trainable: bool,
  world_trace_id, atif_ref, steps_ref,
  infra: {status: ok|masked, failure_kind, blame}, stop, truncated,
  timings: {boot, setup, agent: {model, harness}, collect, verify},
  segments: [{segment_id, parent_segment, kind: "main"|"compaction"|"subagent", root_call}],   # compaction as recorded segments
  calls: [{
    call_id, parent_call_id, segment_id, atif_step_id,
    prompt_token_ids (full, canonical from the engine) | prompt_delta_from: call_id + delta_ids,
    completion_token_ids, behaviour_logprobs (float32, full precision; never rounded),
    policy_version: {min, max} per generated span (resumed/partial generations keep every span: [(start_tok, end_tok, version)]),
    sampling: {temperature, top_p, top_k, seed, max_tokens}, finish_reason,
    routed_experts?: ref, top_p_candidates?: ref,
    digest: sha256(prompt_ids+completion_ids), endpoint, latency_s, usage
  }],
  oldest_version: int   # w0 for the staleness drop rule
}
```

Rules:
- **No call may lack `completion_token_ids` and `behaviour_logprobs` when `trainable=true`.** Otherwise the record is masked (`crucible:capture_incomplete`). Logprobs are never padded with 0.0, unlike `TokenTrace`.
- The endpoint (`harness/endpoint.py`) is the only producer of token fields. It is OpenEnv's capture core plus three additions:
  - the policy version read from the inference server at **each generated span**. A partial rollout that resumes after a weight swap appends a new span, which is the uni_agent `generation_versions` design without the collapse;
  - the verifiers TITO bridge for chat-dialect native harnesses, with NeMo-Gym's URL-path routing as a fallback for harnesses that cannot set a key;
  - a typed infra status header back to the harness instead of a bare 500, in the spirit of verifiers' `ProviderError` status relay.
- Hosted providers (frontier calibration) are capture level `text`. Their records are eval-only by construction.

### 3.10 Frontier calibration across several Harbor harnesses

`crucible calibrate --tasks <set> --harnesses mini-swe-agent@2.4.6,claude-code@<pin>,codex@<pin>,openhands@<pin>,terminus-2 --models <...> --k 4 --gpu-host lab5090`

Mechanics:
- **One Harbor `Trial` per attempt** through `interop/harbor/job_bridge.py`. We own k, seeds, retries and resume, so Harbor's `n_attempts=1, max_retries=0`. This is OpenEnv's "one attempt = one session" rule.
- Environment: `GpuDockerEnvironment(network_none=True, init=True)` on the lab host. The separate verifier runs through Crucible's `docker_grade`, because Harbor ignores `verifier.environment.dockerfile` (B.7).
- **Model traffic goes through the capture proxy**, by setting the agent's base-URL env (`ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, codex's `openai_base_url`, `LLM_BASE_URL`) to the proxy with a per-attempt session key. Usage and cost then come from one place even when a harness's own accounting breaks (blank usage), and the upstream key never enters the sandbox.
- Concurrency: harness pools of 1–2 per frontier provider (rate limits), and per-host GPU semaphore = 1. **One harness per group** (FineEnvs). The harness is part of `SampleKey`.
- Budget: per-attempt reservations. Harness flags are set from the reservation. A run-level cap pauses the run, which can be resumed.
- Output for the M8 Calibrator:
  - per (task, model, harness): scored n, masked n, coverage, pass rate (scored only), unbiased pass@k, gate profile, cost and steps distributions;
  - the A1 "declared environment" flag: `network_enforced`, image digest, GPU arch;
  - harness disagreement: the same model under different harnesses, cf. the FineEnvs finding that ranks don't transfer across harnesses.

### 3.11 The toy: scripted plus LLM policies

- `_toy` helpdesk world on `LocalBackend` (CPU, CI).
- Policies:
  - `ScriptedPolicy` oracle (expects reward 1);
  - `ScriptedPolicy` no-op (expects 0);
  - a replay of a recorded trace;
  - `LLMToolPolicy("smoke-cheap")` with record/replay cache (`CRUCIBLE_LLM_CACHE=replay` in CI).
- `RunSpec(k=4, run_seed=7, concurrency=4)`.
- CI asserts:
  - identity and resume: kill the runner after 5 of 16 samples, re-run, and get byte-identical `results.jsonl` (CI uses replay mode, so LLM samples are deterministic);
  - seeds: the four LLM samples have distinct `seed` and distinct cache keys (`sample_key = attempt id`), so they do not collapse to one replayed completion;
  - fault injection for the CPU-reproducible classes (2, 3, 8, tool crash, verifier crash) and all five policy controls (§6.2);
  - the toy run writes ATIF plus `steps.jsonl`, with the step map contract satisfied.

---

## 4. Concrete changes to existing Crucible code (inputs for the M10 build)

| File | Change |
|---|---|
| `C/core/schema/run.py` | Add the `blame` and `delivery` fields to `Failure`. Add the kinds `crucible:stale_build`, `crucible:harness_self_kill`, `crucible:capture_incomplete`, `crucible:agent_broke_env` (the last is policy blame, so it joins `POLICY_OUTCOME_KINDS`). Add `RunRecord.sample: SampleKey`, `attempt`, `cost_uncertain_usd`, and a `reward` that is None when masked. |
| `C/verify/stage.py` | Use a named container. Catch `TimeoutExpired`, then `docker rm -f` and return `verifier_error/verifier_timeout`. Check the image digest before `docker run`. Replace `no_reward_json` with the normalized kind plus `detail`. |
| `C/verify/runtime/plan.py` | Emit `verifier_error` with `detail` (`check_infra`/`grader_exception`) and keep the rc. |
| `C/verify/collect.py` | Return `tree_changed: bool` (porcelain or fingerprint vs baseline) so the runtime can separate rows 1 and "agent changed nothing". |
| `C/actors/llm.py` | `UsageBlank` → tolerance mode (estimate cost, count uncertain spend, continue). Keep the raise only for `require_usage=strict`. |
| `C/actors/policy.py` | `begin_episode` receives the `AttemptKey` (cache key) and the seed. Record `call_id` per model call for the step map. |
| `C/interop/harbor/atif.py` | Add metrics, format-error steps, `extra.crucible_step/call_id/harness`, and `FinalMetrics`. |
| `C/connectors/capture_proxy.py` | Becomes the eval-capture backend of `harness/endpoint.py`. Training capture goes through OpenEnv capture (DEPEND/PORT) with policy-version stamping. |
| `C/forge/budget.py` | Replace with `runtime/budget.py` (ledger). |
| `MT/task02_pcops_triton/task.toml` `[[verifier.collect]]` | Replace with `COLLECT_SH`. Add an import-time A11 check that rejects `&&`-chained collect commands. |
| ARCHITECTURE M10 "Takes" | agentenv `eval run --k`/LocalRunner: DEPEND → PORT/PATTERN (§2.2). Add PORT for Repo2RLEnv `budget.py`, verifiers `errors.py`/`RetryRule`/resume, FineEnvs `evaluate.py`, and Karotte limit policy. Add DEPEND/PORT for OpenEnv capture (endpoint). |

---

## 5. Holes no framework fills (Crucible-specific work)

1. A `SampleKey` with per-sample seeds and resume keyed on it.
2. A blame axis plus an un-zeroable masked reward across every aggregate.
3. Credit exhaustion as a **run** stop rule. Elsewhere it is either per-task or unhandled.
4. Distinguishing "collect failed" from "the agent did nothing" (row 1).
5. The ABI/stale-build and self-kill classes (rows 6, 8).
6. Policy version per generated span through a black-box proxy.
7. A shared-GPU perf guard spanning dispatch and grading, with regrade-from-artifacts.
8. A step map ATIF ↔ World step ↔ model call ↔ `steps.jsonl`.
9. Mask-fraction and unknown-fraction stop rules wired to masks. NeMo-Gym's health checks are diagnostic only.

---

## 6. Exit-gate plan

### 6.1 Reproduce one historical frontier run end to end (mini-swe-agent on task02)

**Reference** (`MT/task02_pcops_triton/runs/gpt-5.6-sol-run1/`):
- `openai/gpt-5.6-sol`, mini-swe-agent 2.4.6, `migration.yaml` + `sandbox_docker.yaml`, 350 steps / $80;
- container from `task02-pcops-triton:v1`, `--network none`, on the lab 5090;
- 88 API calls, $2.287, patch of 31 files / 3,020 lines;
- grade: reward 0, gates `static✓ install✓ ops✗(132/138) hard✗(36/37) app✓ perf✗`.

The grader must be pinned to task02 at `5e53057` (the version the run was graded with). Later commits tightened the suite.

Three levels, all driven by `crucible run`/`regrade` with the run journaled:

| Level | What runs | Pass criterion | Cost |
|---|---|---|---|
| **R1 regrade** | `Runner.regrade` on an imported run: `model.patch` as the artifact, task digest = task02@5e53057, verifier image digest pinned, `docker_grade` on lab5090 with perf_guard | gate profile identical (`ops` 6 failures with the same test ids, `hard` 36/37, `perf` skipped after the failed gate); `reward 0.0`; `infra.status ok`; journal shows exactly one attempt, not masked | GPU minutes |
| **R2 replay** | native harness `ReplayPolicy` that issues the 88 recorded bash commands in order (from `trajectory.json` via the Harbor `convert_mini_swe_agent` → ATIF → actions) through a World on `HarborBackend(gpu_docker, network_none)` | collected patch equals the recorded patch byte for byte after `DERIVED` excludes (else a diff limited to timing/log files, reported); R1 criteria on the regrade of the replayed patch; ATIF step map has 88 agent steps and `steps.jsonl` rows reference them | GPU ~1 h, $0 API |
| **R3 live** | `harness/harbor.py` with Harbor installed agent `mini-swe-agent@2.4.6`, same templates (`instance_template` from `sandbox_docker.yaml`), `step_limit=350`, `cost_limit=80`, model `openai/gpt-5.6-sol` through the capture proxy, `GpuDockerEnvironment` on lab5090, k=1 (k=3 if budget allows) | completes with **no infra failure** (or, if one occurs, it is classified and the attempt retried); ATIF valid (Harbor validator) and cross-checked against the proxy; usage and cost recorded from the proxy (blank usage tolerated); outcome is a scored record. The reward is not required to equal 0, since sampling differs; the report lists gate profile, steps and cost against the historical run | ≈ $3–10 API + GPU |

R1 and R2 are deterministic reproductions. R3 shows that the live path is the same pipeline the historical bash scripts approximated, now with the B.5 fixes:
- `--init`;
- tolerant collect;
- cost tolerance;
- budget;
- fingerprinting;
- perf guard.

### 6.2 Inject all 8 Appendix B.5 classes; each is classified infra and masked, never scored 0

`runtime/inject.py` provides deterministic injectors. Each one runs an attempt that *would* score 0 if mis-handled. Each class is run twice: once on the toy (where reproducible on CPU) and once on task02 or the GPU golden on lab5090 (classes 4–7).

| # | Injector | Where | Expected record |
|---|---|---|---|
| 1 | Workspace with a gitignored, root-owned `build/`. The agent edits a source file. Collect runs the **legacy** `git add … && git diff` chain from task02's `task.toml` | toy fs_git world + task02 sandbox | `empty_collection`, blame harness, masked, re-collect retry with `COLLECT_SH` succeeds and yields the edit |
| 2 | Fake upstream (`CaptureProxy(transport=…)`) returns 200 without `usage` on calls 3 and 7 | toy LLM policy; R3 harness via proxy | the run continues and `cost_uncertain_usd > 0`. With `uncertain_tolerance=0` (strict), the record is `usage_blank`, masked, retry |
| 3 | Fake upstream returns 402 at call 5, plus a ledger with cap < the second sample's reservation | toy k=4 | the first attempt is `credits_exhausted`, masked. The stop rule pauses the run (exit 3), with no further dispatch. After `increase_limit` and resume, the remaining samples run and the masked one is retried |
| 4 | Sandbox started **without** `--init`. The agent command leaves orphaned children (`(sleep 600 &) ; exit`) | DockerSandbox local + lab | preflight refuses (`zombie_wait`, masked). With preflight disabled, the deadline is hit with `Z` children → `zombie_wait`, masked |
| 5 | The verifier image tag is re-pointed to a rebuilt image with one changed layer | task02 verifier | `stale_image`, masked, before any grading process starts |
| 6 | A candidate patch that changes a public header, graded with "rebuild consumers" disabled | task04/task02 consumer build | `stale_build`, blame task, masked, plus an M9 defect. Regrade with forced rebuild yields a valid verdict |
| 7 | A GPU burner container (`--runtime=nvidia`, a busy kernel loop) plus `stress-ng` started during perf rounds | lab5090, task04 golden perf gate | `node_contention`, masked, regrade queued. The regrade after the burner stops gives the oracle's pass. Cross-check: this reproduces `astra_v3` → `astra_v3b` |
| 8 | A cleanup step uses the legacy `ssh host pkill -f <pattern>`, which matches its own ssh command line | toy over local ssh (or lab) | `harness_self_kill`, masked. The lint test flags any `pkill -f` in runtime code. The pidfile variant passes |

**Assertions for every injected attempt:**
- `outcome == "masked"`, `mask_sample`, `reward is None`, `failure.kind` and `blame` as tabled;
- the sample is absent from the pass@k and mean numerators **and** denominators, and coverage drops accordingly;
- `retryable` samples are re-dispatched on resume with attempt index +1, and the prior attempt dir is kept;
- the flat `reward.json` written for Harbor consumers carries `infra/masked = 1.0`.

A meta-test feeds the same records through the **Harbor** aggregation path (`metrics/base.py`) to show it *would* score them 0. This documents why Crucible aggregates its own.

**Policy-caused controls** (each must be **scored 0, not masked**):
- the agent deletes the repo → `agent_broke_env`;
- a fork bomb or unkillable cohort → `policy_misbehavior`;
- the agent exceeds its step limit → `agent_timeout`/`max_steps` (truncated, scored);
- context overflow → truncated, scored;
- the agent makes no change → empty patch with `tree_changed=false`, scored 0.

Without these controls a classifier that masks everything would pass the gate.

**Gate report.** A table of 8 + 5 rows with the expected and observed record, journal excerpts and coverage numbers, saved in `docs/modules/M10-runtime.md`.
