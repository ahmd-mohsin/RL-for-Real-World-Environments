# M11 research 05: interop contracts, cross-runtime identity, and environment diversity

Status: research (2026-10-10). Scope: Crucible M11 `interop`. For every framework: its exact task/environment contract, what M11 must build so the **same task + same patch/trajectory gives identical `reward.json` and `steps.jsonl`** in every runtime (the M11 exit gate), the agentenv plugin extension points, and a catalog of environment kinds with what a Crucible World must support to host each. Everything below was read from code at the pins in §0.3. Claims are cited as `path:line`. Where something was inferred rather than read, it says so.

---

## 0. Summary, method, pins

### 0.1 Findings that change the M11 plan

1. **Crucible has two different `reward.json` producers today, and they disagree.** The in-container grader runtime writes `flat_reward()` (`CR/verify/runtime/plan.py:264-273`: `reward, outcome, gate/*, check/*, infra/masked, progress`). The host grader writes `RewardJson.flat()` (`CR/verify/grader.py:66-73` → `CR/core/schema/reward.py:44-60`: it adds `rubric/*`, `efficiency`, `kappa/alpha/beta`, `plan_reward`). The host also **recomposes the scalar** with the κ(1−o)Φ shaping term (`CR/credit/compose.py:70`), so a failed outcome with Φ=0.5 gets `reward` 0 in the container and 0.1 on the host. Harbor, verifiers, NeMo-Gym's `harbor_agent`, OpenEnv's Harbor bridge and TRL's `HarborEnv` all read the container's file, never the host's. **M11 needs one canonical producer** (§2.1, N1).
2. **A masked grade currently looks like a real 0 to every foreign consumer.** The masked `reward.json` is `{"reward": 0.0, …, "infra/masked": 1.0}` (`CR/verify/runtime/plan.py:276-283` pre-write; `CR/credit/compose.py:65-67`). Every consumer surveyed reads only `reward`: agentenv `reward_path` (`AE/task_step/task_steps/verifiers/run_container_unit_tests_verifier.py:334-346`), verifiers `HarborTask` (`VF/tasksets/harbor/taskset.py:380-386`), NeMo-Gym `harbor_agent` (`NG/responses_api_agents/harbor_agent/utils.py:92-110`), OpenEnv Harbor (`OE/harbor/rollout.py:151-190`), TRL (`TRL/experimental/harbor/_env.py:163-183`) and rLLM's Harbor evaluator. Each of them trains on a false 0. The fix must be **runtime-specific mask signals plus a reward file that cannot be parsed as 0** (§2.3, N3).
3. **Dropping the `reward` key does not fix it.** verifiers turns every key of a `reward.json` that lacks `reward` into a separate reward of weight 1 and **sums** them (`VF/tasksets/harbor/taskset.py:380-386` → `VF/task.py:264-273` → `VF/trace.py:473-475`). A file holding only `{"infra/masked": 1.0}` scores **1.0**. MiMo uses `{"reward_error": "..."}` (`MM/general/envs/*/run_verify.py:140`), which Harbor rejects as non-numeric (`HB/verifier/verifier.py:104-121`). That rejection is the safe outcome.
4. **`reward.txt` loses precision.** It is written as `f"{reward:g}"`, six significant digits (`CR/verify/grader.py:72`, `CR/verify/runtime/plan.py:301`), and agentenv's `reward_path` path reads that text. Any reward that is not a short decimal (a ΔΦ sum, a record ratio) differs in the 7th digit between runtimes.
5. **Wall-clock values are part of the graded documents.** `crucible_reward.json` components include `time/<check>` seconds (`CR/verify/runtime/plan.py:106`), and `StepRow.cost_s` is a wall time (`CR/core/schema/reward.py:73`). They can never be identical across runtimes and must move out of the compared documents (§2.1, N5).
6. **`steps.jsonl` cannot come from foreign runtimes.** Only Crucible computes it, from per-step state. A Harbor or NeMo-Gym run exposes only the final state and a transcript. So `steps.jsonl` identity has to be built as **trajectory normalization → deterministic World replay → Crucible progress**. Identity then reduces to canonical `call_id`/step segmentation plus replay determinism (§2.4).
7. **agentenv's built-in verifier step is unusable as-is.** It `docker exec`s into a named container on the sandbox (`…/run_container_unit_tests_verifier.py:278-283`). It parses `reward_path` as a scalar only, so a JSON dict becomes `None` and then score 0 (`:396-410`). It clamps to [0,1] (`:410`), turns a missing reward into a hard 0 (`:336-342`), and has no `steps.jsonl`. `crucible_verify` must be our own step (§4.3).
8. **The agentenv `gpu_docker` provider does not exist yet.** Crucible's agentenv sandbox adapter refuses GPU for any provider except Modal (`CR/interop/agentenv/sandbox.py:119-127`). The extension point is `SandboxProvider.EXTRA_CONTAINER_RUN_ARGS` (`AE/providers/sandbox_providers/sandbox_provider.py:116-124,143`). That point does **not** cover `run_docker_container` steps, which build their own `docker run` string (`AE/task_step/task_steps/run_docker_container.py:250-253`) (§4.1).
9. **Stock OpenEnv rubric containers cannot express Crucible's formula.** `WeightedSum` weights must sum to 1 (`OE/core/rubrics/containers.py:256-258`). `Sequential` returns the *last* child's score and short-circuits on 0 (`:57-65`). So `gate·(o·(1+αr+βe)+κ(1−o)Φ)` needs one custom `Rubric`, with gates, outcome and Φ exposed only as reporting children (§3.5).
10. **Per-step reward support varies by trainer.** SkyRL places per-step rewards on each turn's last token (`SK/skyrl/train/generators/skyrl_gym_generator.py:712-719`), but its GRPO sums token rewards (`SK/skyrl/backends/skyrl_train/utils/ppo_utils.py:1367`). Telescoping ΔΦ therefore collapses to the terminal reward under GRPO. Only rLLM accepts precomputed per-token advantages from the environment side (`RL/types.py:135`, `RL/trainer/config/rllm/base.yaml:79`). prime-rl assigns advantages only inside an `Algorithm`, and `ALGORITHM_CLASSES` is a closed dict (`PR/orchestrator/algo/__init__.py:42-53`). Details in §3.6.
11. **PortSimEnv v1 is not a discrete-event simulation.** All information is given up front (`PS/core/berth_core/dock.py:1-3`). The "simulation" is a deterministic rules evaluator over an hour × quay-section grid (`PS/core/berth_core/check.py:59-120`), a seeded disruption generator (`dock.py:92`, `generate.py:180`) and a CP-SAT optimum (`solve.py:77-175`). The 3D traffic replay never affects reward (`portsim-v1/DESIGN.md:111`). This makes it easy to host natively today. A dynamic v2 (disruptions revealed over simulated time) is where Crucible's clock and events add something OpenEnv lacks (§5.4).
12. **FineEnvs `00-environments-101` already runs one environment core under six framework adapters** (OpenEnv, ORS, NeMo-Gym, verifiers, skyrl-gym, GEM), with per-adapter tests (`FE/00-environments-101/project.yaml:16-50`; `envs/wordle/{core,openenv,nemo_gym,verifiers,skyrl_gym,gem,ors}`). It also shows the exact drift M11 must prevent: the NeMo-Gym adapter **re-derives the reward from the transcript** using a constant `DEFAULT_MAX_GUESSES` instead of the game state (`envs/wordle/nemo_gym/server.py:155-184` vs `core/game.py:108-121`). Its docstring says "1.0 if win else 0.0" while the code pays up to 1.5. Rule: exporters grade **state through the VerifierView, server-side, keyed by session**, never by re-parsing transcripts (N6).

### 0.2 Path abbreviations

| Abbrev. | Path (relative to `/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/`) |
|---|---|
| `CR/` | `crucible/src/crucible/` |
| `AE/` | `agentenv-framework/src/agent_env/` |
| `FW/` | `RL-Environments/06-frameworks-and-env-code/` |
| `HB/` | `FW/harbor-framework-harbor/src/harbor/` |
| `VF/` | `FW/primeintellect-ai-verifiers/verifiers/v1/` |
| `PR/` | `FW/infra-primeintellect-ai-prime-rl/src/prime_rl/` |
| `NG/` | `FW/nvidia-nemo-gym/` |
| `NR/` | `FW/infra-nvidia-nemo-rl/nemo_rl/` |
| `OE/` | `FW/meta-pytorch-openenv/src/openenv/` |
| `TRL/` | `FW/infra-huggingface-trl/trl/` |
| `SK/` | `FW/infra-novasky-ai-skyrl/` |
| `RL/` | `FW/infra-rllm-org-rllm/rllm/` |
| `RO/` | `FW/infra-alibaba-roll/roll/pipeline/agentic/` |
| `VE/` | `FW/infra-volcengine-verl/verl/` |
| `MV/` | `FW/xiaomimimo-verl/` |
| `MM/` | `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/` |
| `FE/` | `FW/adithya-s-k-fineenvs/` |
| `PS/` | `FE/07-simulation-environments/portsim-v1/envs/berth_planning/` |
| `EOG/` | `FW/servicenow-enterpriseops-gym/` |
| `ART/` | `FW/hf-space-fineenvs-simulation-rl-environments/app/src/content/chapters/` |

### 0.3 Pins (local clones, read-only)

Harbor a2fafbc (2026-10-03), agentenv 041afd7 (10-08), verifiers ae819de (10-04), prime-rl f7e8ab6 (10-09), NeMo-Gym 3ef478d (10-03), NeMo-RL c9193bc (10-10), OpenEnv 49aa302 (10-02), TRL 72cf37d (10-09), SkyRL 3d0ae22 (10-09), rLLM 4e828c8 (10-08), ROLL 581046a (09-28), verl 5eb4826 (10-10), MiMo verl fork a2ad9f6 (09-26), FineEnvs 5e0b46e (10-08), simulation article 46bb2be (10-06). Others are as in `ARCHITECTURE.md` §2.2 and `00-source-pins.txt`.

---

## 1. The Crucible side of the boundary

### 1.1 What is canonical

- **Task:** a Harbor dir plus `crucible/` (DESIGN §4.3).
- **Grade:** `crucible_reward.json` (`RewardJson`, `CR/core/schema/reward.py:24-60`) and its flat projection `reward.json`. `reward` is the training scalar. Any 0 gate forces reward 0 unless masked (`:35-42`). `progress` is Φ ∈ [0,1].
- **Dense channel:** `steps.jsonl` rows `StepRow` (`:63-76`), written by `CR/credit/steps.py`.
- **Failure vocabulary:** the NeMo-Gym names verbatim plus `crucible:*` names, each with a default blame (`CR/core/schema/run.py:17-67`). `POLICY_OUTCOME_KINDS` are scored; everything else is masked.
- **World API:** `start/reset/step/snapshot/restore/close` (`CR/world/world.py:172-611`); `VerifierView` (`:95-125`).
- **Clock:** `off | step | wall`. Step mode advances `step_dt_s` per world step; `advance()` moves time explicitly (`CR/world/clock.py:50-85`; `CR/core/schema/world.py:90-101`). `WorldSpec.seed` exists (`CR/core/schema/world.py:215`).

### 1.2 What already exists under `CR/interop/` vs what M11 still needs

| Folder | Exists (file) | Missing for M11 |
|---|---|---|
| `harbor/` | task I/O (`task_io.py`), World→ATIF (`atif.py`), ATIF→trace (`atif_import.py`), GPU compose overlay (`gpu_docker.py`), Harbor sandbox backend (`world_backend.py`), grade-with-task's-own-verifier (`verify_dir.py`) | job bridge (Harbor `Job` → M10 journal); export identity check; ATIF `RolloutDetail` → `crucible.rollout/v1`; `[[steps]]` export rules (§3.1) |
| `agentenv/` | gateway World backend (`world_backend.py`, 487 lines), provider sandboxes without GPU except Modal (`sandbox.py`), A2A serve/client/relay (`a2a.py`) | the **plugin**: `gpu_docker` provider, `crucible_world` + `crucible_verify` steps, artifact mirror, entry points (§4) |
| `mimo/` | env → WorldSpec importer (`importer.py`), sweep checker (`check.py`) | verifier import (`verify.py` → RewardJson), **byte-identical export**, verl-fork dataset export (§3.7) |
| `openenv/` | AWM-1K import as Apps (`awm.py`, `awm_check.py`) | Environment export, rubric, `openenv.yaml`, generic OpenEnv-App import (§3.5, §5.4) |
| `verifiers/`, `verl/`, `nemo_gym/`, `eog/` | none | everything (§3.3, §3.4, §3.7, §3.8) |

No agentenv entry points are declared (`crucible/pyproject.toml` has only `[project.entry-points."crucible.packs"]`). verifiers (mcp==2.0.0) and agentenv (mcp<2) conflict, so they are declared as conflicting extras (`pyproject.toml` `[tool.uv] conflicts`).

---

## 2. Cross-runtime identity: the normalization contract (core of the exit gate)

### 2.1 What "identical" means, and rules N1 to N6

The gate compares two artifacts per (task, patch-or-trajectory, runtime):

- `reward.json`: compared as **canonical JSON** (RFC 8785 JCS: sorted keys, shortest round-trip float repr, no whitespace). Byte equality of on-disk files is not required, because each runtime's writer formats differently.
- `steps.jsonl`: compared as a list of canonical rows with the volatile fields removed.

**N1. One producer.** `reward.json` is written by exactly one function, inside the verifier container, and nothing re-composes it afterwards. Today `flat_reward` (`CR/verify/runtime/plan.py:264-273`) and `RewardJson.flat` (`CR/core/schema/reward.py:44-60`) differ in their key sets, and `to_reward` re-composes the scalar (`CR/verify/grader.py:38-49`). Choose one of two options:
- **(a) recommended:** move the composer (`CR/credit/compose.py`, standard library only) into the container runtime, and pass the milestone Φ of the final state in from the plan. The container can evaluate Φ of the *final* state, because the milestones are state predicates.
- **(b):** keep the container's `reward` as the pure gated outcome and publish Φ only as the `progress` channel and in `steps.jsonl`.

Under either option the host grader becomes a reader, not a writer.

**N2. Precision.** Floats are IEEE-754 double. `reward.txt` is written with `repr()`, not `:g`. No adapter reads `reward.txt` when `reward.json` exists (Harbor already prefers JSON: `HB/verifier/verifier.py:257-262`; verifiers too: `VF/tasksets/harbor/taskset.py:378-401`). Rounding happens only where a source framework defines it, and is then recorded as a component. Examples: MiMo `round(wsum/tsum, 4)` (`MM/general/envs/*/verify.py:400`); PortSim `round(reward, 6)` (`PS/core/berth_core/reward.py:106`); verifiers rounds per-token float streams to 4 decimals in `to_record` (`VF/graph.py:75`, `VF/trace.py:760-771`).

**N3. Masks.**
- A masked grade must never yield a parseable scalar `reward`. Write `crucible_reward.json` (status masked), and either write **no** `reward.json`/`reward.txt` or write an **empty** `reward.json`.
- Harbor raises `RewardFileNotFoundError`/`RewardFileEmptyError` (`HB/verifier/verifier.py:66-71, 89-93, 257-266`). That is a trial exception, which M10 already classifies.
- verifiers' strict adapter rejects an empty file (`VF/tasksets/harbor/taskset.py:50-54, 405-417`) → `TaskError` → errored trace → prime-rl drops it (`PR/orchestrator/algo/base.py:11-19`).
- The pre-write of `{"reward": 0.0}` (`CR/verify/runtime/plan.py:276-283`) must become an empty-file pre-write. An interrupted grade then reads as infra in every runtime instead of a 0.
- Crucible-native exporters then add the runtime's explicit mask signal (table in §2.2).

**N4. Truncation.** Truncation is mapped by the policy chosen per export, mask or Φ-bootstrap (DESIGN §10.3 item 6), never silently to 0. Where the foreign environment itself defines "no submission = 0" as a task rule (PortSim `tool_call_limit` → reward 0.0, `PS/openenv/berth_openenv/environment.py:274-278`; PortSim agent "an episode without `submit_plan` scores 0", `PS/openenv/berth_openenv/agent.py:9-10`), import keeps it as a **policy outcome** (scored). Faithful import beats Crucible's preference.

**N5. Volatile fields are excluded or made deterministic.**
- `time/*` components move to `report.json`.
- `StepRow.cost_s` is excluded from identity, or taken from the source trajectory in replay mode.
- `efficiency` is computed only from trace counts (tool calls, tokens), never from wall time.
- `harness` is the canonical `harness@version` string.
- `extra` is namespaced by runtime and excluded.
- LLM-judge verdicts come from the M9 grade cache keyed by (item, evidence hash), so a re-grade through another runtime replays the same verdict.

**N6. Grade state, not transcripts.**
- Exporters keep the World alive per session (NeMo-Gym `seed_session` → `verify` → `close_session`; OpenEnv WebSocket session; agentenv gateway).
- The grade comes from the `VerifierView`.
- Transcripts are used only for `call_id` alignment.
- The wordle counter-example is in §0.1 item 12. NeMo-Gym's own `verify` receives only the Responses transcript (`NG/nemo_gym/base_resources_server.py:103-108`), which is why its adapters drift.

### 2.2 Divergence matrix

| Aspect | Crucible | Harbor | agentenv | verifiers → prime-rl | NeMo-Gym → NeMo-RL | OpenEnv → TRL/SkyRL | MiMo/verl fork |
|---|---|---|---|---|---|---|---|
| Scalar location | `reward.json["reward"]` | `reward.json` (dict of finite numbers) wins over `reward.txt` (`HB/verifier/verifier.py:89-124,257-268`) | `reward_path` text → float, **clamped [0,1]** (`…unit_tests_verifier.py:396-410`) | Σ score×weight of reward channels (`VF/trace.py:134-141,473-475`) | `BaseVerifyResponse.reward: float` (`NG/nemo_gym/base_resources_server.py:107-108`) | `Observation.reward` per step, `bool\|int\|float\|None` (`OE/core/env_server/types.py:68-88`) | `reward_score: float` (`MV/recipes/general/agent_loop.py:1069`) |
| Range | [0, 1+α+2β] by default (`CR/credit/compose.py:36-70`); record-style score gates unbounded (`CR/verify/runtime/plan.py:11-13`) | unbounded | [0,1] by clamp; aggregate also clamps (`AE/task_step/task_steps/verifiers/scoring.py:39`) | unbounded; `max_rl` assumes ≥0 (`PR/orchestrator/algo/max_rl.py:11-27`) | declared in manifest `reward.range` (`NG/nemo_gym/environment/manifest.py:100-102`) | declared in `openenv.yaml validation.reward.range` | MiMo items Σw·s/Σw ∈ [0,1] |
| Multi-component | `reward_components()` | extra numeric keys | none (single `score`) | metrics (unweighted) + rewards (weighted) | `reward_components` (multi-reward subclass, `:151-164`); metrics read top-level numbers only | rubric children `named_rubrics()` | `reward_detail.json` (separate file) |
| Masked/infra | `infra.status=masked`, reward None in runtime | exception in TrialResult | none (missing reward → **0**) | errored trace excluded (`PR/orchestrator/algo/base.py:15`) | `mask_sample=True` (+ legacy `instance_config.mask_sample`, read by `NR/experience/rollouts.py:314-322`) | TRL: reward `None` → NaN → unscorable (`TRL/trainer/grpo_trainer.py:2685-2693`); OpenEnv TrainingTrace `reward: None` = unscored (`OE/core/harness/rollout.py:184`) | −999 sentinel → group mean + advantage 0 (`MV/verl/trainer/config/ppo_trainer.yaml:133-139`; `MV/verl/trainer/ppo/core_algos.py:306-309`) |
| Truncation | mask or Φ-bootstrap | `AgentTimeoutError` in exception_info | `timed_out` → raise | `is_truncated` (`VF/trace.py:621-633`); trace still trainable | Gym flags max-iteration rollouts as `mask_sample`; NeMo-RL honours it unless `should_mask_flagged_samples: false` (`NR/experience/rollouts.py:1914-1928`) | `done`; harness `max_turns` | `rollout/seq_timeout` infra category (`MV/recipes/general/agent_loop.py:78-87`) |
| Failure kinds | NeMo vocab + `crucible:*` | Harbor exception class names | `exit_code/timed_out` | boundary error types (`VF/errors.py:1-60`) | shared vocab (`NG/nemo_gym/failure_kinds.py`) | free-form `error` | `_INFRA_ERROR_CATEGORIES` (`MV/recipes/general/agent_loop.py:78-87`) |
| Per-step reward | `steps.jsonl` ΔΦ per call_id | `[[steps]]` (task stages, not turns) | none | node `advantages` set by trainer only (`VF/graph.py:138-142`) | none | yes (`TrajectoryRubric.compute_step_rewards`, `OE/core/rubrics/trajectory.py:108-117`) | none (TokenTrace masks only) |
| Tool-call format | ToolSpec `<app>.<tool>` + JSON args | harness-native | MCP / REST `/step` (`AE/env/gateway/gateway.py:727-829`) | OpenAI chat via renderer/interception | OpenAI Responses `function_call` items → `POST /{tool}` (`NG/responses_api_agents/simple_agent/app.py:171-247`) | MCP `CallToolAction` | MCP over HTTP via sidecar + `mcp_proxy` (`MV/recipes/general/mcp_proxy.py:72-140`) |
| Sync/async | async | async | async | async | async HTTP | sync+async (`OE/core/env_server/interfaces.py:262-305`) | async Ray actor |

### 2.3 Normalization per runtime (import side ← / export side →)

| Runtime | Import → Crucible | Export Crucible → runtime |
|---|---|---|
| Harbor | `VerifierResult.rewards` → RewardJson only if it contains Crucible keys; otherwise `reward` → outcome-only RewardJson. Exceptions → M10 classifier | identity, plus N1–N3; `multi_step_reward_strategy="final"` for staged tasks (the mean treats missing keys as 0: `HB/models/task/config.py:813-820`) |
| agentenv | `verifications[vid].score` alone is lossy; read `extracted_files[crucible_reward.json]` | `crucible_verify` records the full RewardJson; masked → typed error after recording, no score row (§4.3) |
| verifiers | trace.rewards → components; `trace.has_error` → masked; `is_truncated` → N4 | `CrucibleTask.run_verifier` reads `crucible_reward.json`; masked → `raise TaskError`; return `{"reward": r}` only; components as metrics with a `crucible/` prefix |
| NeMo-Gym | `reward`, `reward_components`, `mask_sample`, `failure_kind` map 1:1 (vocabulary already shared) | `BaseMultiRewardVerifyResponse` + `mask_sample` + `instance_config.mask_sample` + top-level numeric copies; `crucible_steps` extra field |
| OpenEnv | Σ step rewards vs last non-null reward: pick per env (the wordle env documents both, `FE/00-environments-101/envs/wordle/openenv/server/wordle_environment.py:161-171`) | Ψ-telescoping per-step rewards (§3.5), `reward=None` when masked |
| MiMo | Σw·s/Σw → reward; items → `rubric/<id>`; `reward_error` → masked (`judge_failed`/`verifier_error`) | bytes verbatim (§3.7); for the verl fork, masked → −999 |

### 2.4 How `steps.jsonl` becomes runtime-independent

1. **Canonical trajectory.** Every runtime's transcript (ATIF from Harbor; gateway JSONL plus harness log from agentenv; verifiers `Trace` graph; NeMo-Gym Responses `output` items; OpenEnv `TrainingTrace`) normalizes to `crucible.trace/v1` events. `call_id` is **derived**, not copied: `call_id = H(episode_key, model_call_ordinal)`, where the ordinal counts sampled assistant messages in the main branch. Runtime IDs differ across runtimes: Responses item ids, verifiers node ids (`VF/graph.py:83-160`), ATIF step ids, gateway `event_N` (`AE/env/gateway/gateway.py:992-1006`). Runtime IDs are kept in `extra` for joins.
2. **Canonical actions.** `(qualified_tool, canonical JSON args)` after the per-runtime name map: NeMo-Gym strips its MCP namespace (`NG/nemo_gym/base_resources_server.py:50-73`); agentenv uses `<app>__<tool>` (`CR/interop/agentenv/world_backend.py:54`); Crucible uses `<app>.<tool>`. Parse failures are policy errors (an observation is shown), never environment errors.
3. **Deterministic replay.** Re-issue the canonical actions into a fresh `World.restore(start)` with `ScriptedPolicy`, evaluate milestones online (`CR/verify/progress.py`), and emit rows (`CR/credit/steps.py`). A trace that does not replay is quarantined (`CR/workflow/replay.py:1-15`).
4. **Gate assertion.** For runtimes that execute tools themselves (agentenv gateway, NeMo `POST /{tool}`, OpenEnv MCP), the replay is also run **through that runtime's tool transport**. The final `VerifierView.fingerprints()` (`CR/world/world.py:111-115`) must equal the native replay's. That proves the transport preserves semantics, and makes reward identity follow from state identity.

---

## 3. Per-framework contracts

### 3.1 Harbor (DEPEND, TARGET; Apache-2.0)

**Contract.**
- `task.toml` schema 1.4 (`HB/models/task/config.py:801-802`).
- `[verifier]` with `environment_mode` shared or separate (`:562-610`).
- `[[steps]]` with `min_reward` as a float or a per-key dict (`:760-790`) and `multi_step_reward_strategy` mean or final (`:794-820`). Per-step gating is `_min_reward_failure` (`HB/trial/multi_step.py:204-241`).

**Reward parsing.**
- `reward.txt` must be a finite float (`HB/verifier/verifier.py:66-88`).
- `reward.json` must be a dict of finite numbers; non-numeric values raise (`:89-124`).
- JSON wins over text (`:257-268`) → `VerifierResult(rewards=…)` (`HB/models/verifier/result.py:4-5`).

**Trajectory.** ATIF v1.0–v1.8 (`HB/models/trajectories/trajectory.py:13-26`). `Metrics` carry token ids (`HB/models/trajectories/metrics.py:8-35`). `RolloutDetail` gives per-turn `prompt_token_ids`, `completion_token_ids`, `logprobs` and `extra` (`HB/models/agent/rollout_detail.py:4-27`), produced by `HB/llms/{lite_llm,tinker}.py`.

**Trainer consumers.** NeMo-Gym `harbor_agent`, OpenEnv `harbor/`, TRL `experimental/harbor`, rLLM `integrations/harbor`, SkyRL `examples/train_integrations/harbor*`, verifiers `HarborTaskset`. Each takes `rewards["reward"]` (or the first value) and drops the rest. NeMo-Gym's `harbor_agent` scores **any exception as 0.0** (`NG/responses_api_agents/harbor_agent/app.py:297-319`).

**M11 must build.**
- Export identity plus N1–N3.
- Staged tasks: `multi_step_reward_strategy="final"`, and `min_reward={"gate/<g>":1.0}` to express Crucible gates per stage.
- Job bridge into the M10 journal.
- ATIF `RolloutDetail` → `crucible.rollout/v1`.
- A one-page "Harbor consumer hazard" note shipped with every exported dataset: masked = missing reward file.

### 3.2 agentenv-framework (DEPEND + plugin, TARGET; Apache-2.0)

**Contract.**
- **Task:** a JSON list of step dicts, each `{id, type, depends_on?, retry_config?, fail_task_on_error?, …}` (`AE/examples/hello/tasks/hello.json`; `AE/task_step/task_step.py:84-160`).
- **DAG:** `depends_on` defaults to all earlier steps (`:31-38`).
- **Retry:** `RetryConfig{retry_from_step_id, max_retries}`, which rolls back to an ancestor and re-dispatches on a redeployed world (`:52-80`).
- **Steps** read and write a shared `TaskStepContext{deployed_envs, deployed_agents, deployed_sandboxes, prompt_responses, metadata, instance_id, …}` (`AE/task_step/context.py:207-217`).

**Envs.**
- `MCPServerEnv(docker_image_artifact, environment_name)` (`AE/env/envs/mcp_server.py:44-60`).
- `MultiEnv(mcp_server_envs, website_envs, env_provider_type="gateway")` (`AE/env/envs/multi_env.py:41-60`).
- `ServiceDBEnv`, `WebsiteEnv` and the gateway.
- Deploy produces `DeployedGatewayEnv{gateway_url, gateway_mode, db_*_url, website_frontend_urls, …}` (`AE/env/env.py:159-187`).

**Gateway protocol** (`AE/env/gateway/gateway.py:150-291`):

| Endpoint | Purpose |
|---|---|
| MCP streamable HTTP | tool calls |
| `POST /step` `{action: list_tools\|call_tool, tool_name, arguments}` + header `AgentEnv-Role` | role-filtered tool calls (`:727-829`; `constants.py:98`) |
| `GET /state` | tools per server, changelog id, roles |
| `/tools/{disable,enable}` | per-role tool gating |
| `/triggers/{register,remove,clear,state}` | trigger engine |
| `/clock/{set-time,clear,time,state}` | virtual clock |
| `GET /trajectory` | JSONL events `{event_type: tool_call\|tool_call_result, event_id, timestamp_utc, virtual_time, changelog_id}` (`:992-1006`) |
| `/.well-known/agent-env.json` | env card |
| `POST /agentenv` | JSON-RPC `data/reset\|add\|get` (`constants.py:64-69`) |

Tool timeout is 600 s (`gateway.py:74`).

**Eval and run.**
- `--k` parallel runs (`AE/cli/task/run.py:412`, `AE/cli/eval/run.py:80`).
- `RunRecord.status` ∈ QUEUED, RUNNING, COMPLETED, FAILED, CANCELED (`AE/runner/runner.py:16-55`).
- Scores live in `context.metadata["verifications"][vid].score`, aggregated by `all_pass`, `any_pass` or `weighted_average` (clamped; `AE/task_step/task_steps/verifiers/scoring.py:12-40`; `aggregate_verifiers.py:63-90` drops `skipped` rows).

**Trainer integration: none.** No token capture, logprobs or rollout API exist in core (searched: only explorer UI and snapshot utilities mention rollouts). A trainer would need tokens from a capture proxy on the agent's model endpoint (verifiers interception or OpenEnv RFC 006), plus Crucible's grade. agentenv is therefore an **evaluation and enterprise-world runtime**, and its training path goes through M12 using Crucible records.

**Semantic differences.** See §0.1 item 7 and the §2.2 table. Also:
- Gateway trajectories carry no model `call_id`. Join them with the harness trajectory from `PromptResponse.agent_trajectory_file_path`/`object_url` (`AE/task_step/context.py:148-168`).
- `gateway_mode` defaults to `performance` (`AE/task_step/task_steps/deploy_env.py` constructor). Crucible's backend uses `consistent` for determinism (`CR/interop/agentenv/world_backend.py` header). Exports must pin `consistent`.

M11 work is §4.

### 3.3 verifiers v1 → prime-rl (verifiers DEPEND/TARGET, MIT; prime-rl Apache-2.0)

**Contract.**
- `TaskData{prompt, system_prompt, image, workdir, network_allow/block, artifacts, timeout{setup,agent,finalize,scoring}, resources{cpu,memory,gpu,disk}}` (`VF/task.py:51-137`).
- `Task` hooks: `setup`, `finalize`, `stage_verifier`, `validate` (`:180-210`), and `score()` (`:213-290`). Score runs `@vf.metric` (unweighted) and `@vf.reward` (weight `_vf_weight`, default 1), plus plugged judges. Each value becomes `trace.record_reward(name, value, weight)`, and the scalar is `Σ score·weight` (`VF/trace.py:134-141,473-475,692-695`).
- Errors are classified by boundary (`VF/errors.py:1-60`). `record_error` stops the trace with `ok=False` (`VF/trace.py:742-758`).
- `is_truncated` covers max_turns, token limits, compaction failure and `finish_reason=length` (`:621-633`).
- Env server: ZMQ ROUTER + msgpack, `RunRequest{task_data, client, model, sampling}` → `RunResponse{traces}` (`VF/serve/server.py:28-61`; `VF/serve/types.py:42-61`).

**HarborTaskset.**
- Downloads or caches Harbor dirs. It **requires a pullable `[environment].docker_image`** and rejects Dockerfile-only tasks unless `ignore_dockerfile` (`VF/tasksets/harbor/taskset.py:9-15`).
- Maps GPU to `"type:count"` (`:454-475`).
- Stages tests and deletes stale reward files first (`:327-355`).
- `solved` (`@reward(weight=1.0)`, `:357-368`) → `run_verifier` (`:370-403`). If reward.json is a dict with `reward`, it records the other keys as metrics and returns `{"reward": r}`; any other dict is returned whole (summed, §0.1 item 3). Missing reward → `TaskError`.
- Separate verifier box via `HarborEnv` (`VF/tasksets/harbor/env.py:1-12`) built on `IsolatedVerifierEnv` (`VF/envs/isolated_verifier/env.py:1-6`).

**Cross-framework bridges inside verifiers.**
- `tasksets/nemo_gym`: posts `seed_session`, then `verify` with the trace rendered as a NeMo response. It returns `float(reward)` and records numeric fields as metrics, **ignoring `mask_sample`/`failure_kind`** (`VF/tasksets/nemo_gym/taskset.py:55-87`).
- `tasksets/openenv`: **sums per-step rewards** into `openenv_reward` (`VF/tasksets/openenv/taskset.py:5-7,119-134`).

**prime-rl consumption.**
- Errored or non-trainable traces are skipped (`PR/orchestrator/algo/base.py:11-19`). Algorithms assign advantages: GRPO uses reward minus group mean (`PR/orchestrator/algo/grpo.py:26-52`) via `assign_advantages(trace, float|list)` (`PR/orchestrator/algo/routing.py:17-35`), and the batch takes those advantages (`PR/orchestrator/batch.py:350-360`).
- Discards are counted as stale, errored or no-signal (`PR/orchestrator/orchestrator.py:706-727`).

**M11 builds.**
- `CrucibleTaskset(HarborTaskset)` / `CrucibleTask(HarborTask)` overriding `run_verifier`: read `crucible_reward.json`; masked → `TaskError("crucible:masked:<kind>")`; return `{"reward": r}`; components → `record_metric("crucible/<k>", v)`.
- A pinned image per task (Crucible already pins digests in `verify/stage.py`).
- `steps.jsonl` attached as `trace.info["crucible_steps"]`.
- For prime-rl, an `Algorithm` subclass (M12) that maps rows by `call_id` to sampled nodes and calls `assign_advantages(trace, per_token_list)`. Registration needs a small upstream change (an import-path `algo.type`) or a launcher-side insertion into `ALGORITHM_CLASSES`.
- Keep verifiers in its own uv environment (the mcp conflict).

### 3.4 NeMo-Gym → NeMo-RL (TARGET, PATTERN; Apache-2.0)

**Contract.**
- Resources server routes: `POST /seed_session`, `/close_session`, `/verify` (wrapped in `judge_failsafe`), `/aggregate_metrics`, `GET /reverify_mode` (`NG/nemo_gym/base_resources_server.py:238-256`).
- Requests/responses:
  - `ResourcesSeedSessionRequest{resources_session_id, episode_id, task_id, task_data}` → `ResourcesSeedSessionResponse{resources_session_id, resources_tools: MCPServerMetadata{server_name,url_path="/mcp",transport,headers}, sandbox_access}` (`:175-205`).
  - `BaseVerifyRequest{responses_create_params, response}` → `BaseVerifyResponse{reward, mask_sample=False, failure_kind?, failure_reason?}` (`:103-149`).
  - `BaseMultiRewardVerifyResponse{reward_components}`; aggregate metrics read only top-level numbers (`:151-164`).
- Judge failure → `reward 0.0, mask_sample True, failure_kind judge_failed`, mirrored into the legacy `instance_config.mask_sample` (`NG/nemo_gym/judge.py:95-117`).
- Failure vocabulary and stages (`NG/nemo_gym/failure_kinds.py`); `RolloutFailure{source, delivery, failure}` (`NG/nemo_gym/rollout_outcomes.py`).
- Agent loop: Responses API. `function_call` items are dispatched as `POST /{name}` with JSON args and session cookies, producing `function_call_output` items, with optional `max_steps` (`NG/responses_api_agents/simple_agent/app.py:65,171-257,306-371`).
- Manifest: `EnvironmentManifest{name, version, kind, integration_profile, domain, description, modality, licensing, authors, reward{range, higher_is_better}, determinism, config_path, resources_server, agent_server}` (`NG/nemo_gym/environment/manifest.py:41-250`); `Domain` enum (`NG/nemo_gym/config_types.py:574-610`).
- 145 resources servers exist (domains in §5.1).

**NeMo-RL consumption.**
- `run_rollouts` → postprocess. Token ids are taken from the output items' `prompt_token_ids/generation_token_ids/generation_log_probs` with a prefix check (`NR/environments/nemo_gym.py:1137-1295`).
- `reward=float(result.get("reward") or 0.0)` (`:991`). A None reward becomes 0: the mask has to travel in `mask_sample`.
- Masks are read from `instance_config.mask_sample` (`NR/experience/rollouts.py:314-322`) and honoured by default (`:1914-1928`).

**M11 builds.**
- `CrucibleResourcesServer(SimpleResourcesServer)`. `seed_session` starts or restores a World session keyed by `resources_session_id` and returns MCP metadata for `World.tools_for(policy)`; MCP is preferred over `POST /{tool}` so tool names match OpenEnv and agentenv.
- `verify` grades from the `VerifierView` (N6) and returns `BaseMultiRewardVerifyResponse` with `reward_components=rj.reward_components()`, `mask_sample=masked` (both top-level and `instance_config`), `failure_kind` (Crucible names are already NeMo-legal, `CR/core/schema/run.py:1-6`), top-level numeric copies of the gates, and a `crucible_steps` extra.
- `close_session` closes the World.
- `manifest.yaml` with `integration_profile: custom-gym-verifier`, `determinism: seeded`, and the true `reward.range`.
- **Do not export through `harbor_agent`** (exceptions → 0.0).

### 3.5 OpenEnv → TRL / SkyRL / torchforge (PATTERN, TARGET, DATA; BSD-3-Clause)

**Contract.**
- `Environment.reset(seed, episode_id, **kw)`, `step(action, timeout_s, **kw)` (each with an async twin), the `state` property, and `SUPPORTS_CONCURRENT_SESSIONS` (`OE/core/env_server/interfaces.py:213-351`).
- `Observation{done, reward, metadata}`, `ResetRequest{seed, episode_id}`, `StepRequest{action, timeout_s, request_id}` (`OE/core/env_server/types.py:50-170`).

**Transport** (`OE/core/env_server/http_server.py`):
- Simulation mode registers `/reset` and `/step` (`:1504-1545`) and `/state` (`:1631-1640`). Production mode exposes only safe endpoints (`:859-860`).
- `/mcp` over HTTP (`:1701`) and WebSocket (`:1395`).
- `/ws` gives one environment instance per connection (`:1731-1741`).
- Plain HTTP handlers construct and close an environment per request. OpenEnv's own Harbor bridge says so and exposes a rollout as one long MCP tool instead (`OE/harbor/environment.py:1-5`). Stateful exports must use `/ws` or an MCP session.

**Rubrics.**
- `Rubric.forward(action, observation)`; `Sequential` (fail-fast on 0, returns the last score), `Gate(threshold)`, `WeightedSum` (weights sum to 1) (`OE/core/rubrics/containers.py:27-270`).
- `TrajectoryRubric`: `intermediate_reward` until done, then `score_trajectory`; `compute_step_rewards` (`OE/core/rubrics/trajectory.py:22-136`).

**Other surfaces.**
- `openenv.yaml`: `spec_version, name, type, runtime, app, port, validation{reward{range, oracle_tolerance, floor_margin}, resources, network, capabilities{verifier, oracle, rubric_tree, task_api, declared_tools}}` (examples: `FW/meta-pytorch-openenv/envs/echo_env/openenv.yaml`, `PS/openenv/openenv.yaml`).
- Capture: `TrainingTrace` turns carry token ids, logps and `loss_mask`, with `reward: None` meaning unscored (`OE/core/harness/rollout.py:173-185,220`).

**Trainer consumption.**
- **TRL `GRPOTrainer`:**
  - `environment_factory` builds a class whose public methods are tools, with `reset()` and an optional `get_reward()` (weight 1) (`TRL/trainer/grpo_trainer.py:137,266-286,650-712`).
  - `rollout_func(prompts, trainer) → {prompt_ids, completion_ids, logprobs, env_mask?, …extra → reward funcs}` (`:130,258-265,2134-2201`).
  - `tool_mask` zeroes tool tokens (`:1892-2096`).
  - A None reward is unscorable and excluded from the baseline (`:2685-2693`).
  - Advantages are sequence-level only.
  - Async path: `TRL/experimental/async_grpo/openenv_harness.py`.
- **SkyRL:** `examples/train_integrations/openenv/env.py` wraps OpenEnv clients as `BaseTextEnv`.
- **torchforge:** not in the local corpus (not verified).

**M11 builds** (`interop/openenv/export.py`):

`CrucibleEnvironment(MCPEnvironment)`:
- `SUPPORTS_CONCURRENT_SESSIONS=True`.
- `reset(seed, episode_id, task_id=…)` → `World.reset`.
- MCP tools come from `tools_for(policy)`.
- `step(CallToolAction)` → `World.step`.
- **Per-step reward = Ψ-telescoping.** Define Ψ_t = Φ_t for t < T and Ψ_T = `rj.reward`, and set `Observation.reward_t = Ψ_t − Ψ_{t−1}`. Every consumer then gets the same total whether it sums per-step rewards (verifiers' OpenEnv bridge) or reads the last value.
- `metadata["crucible"] = {phi, delta_phi, milestones_new, call_id}`. At done, the full `crucible_reward.json`. Masked → `reward=None`, `metadata.masked=True`.
- `CrucibleRewardRubric(Rubric)` calls the Crucible composer and exposes `gates.<g>`, `outcome`, `progress` as children for `named_rubrics()` only (§0.1 item 9).
- `openenv.yaml` declares `range: [0, 1+α+2β]`, or the record cap for score tasks, `oracle_tolerance: 0`, `network: no-network`, `declared_tools`, `task_api: true`.

### 3.6 Trainer-side environment APIs

| Trainer | Env contract | Reward/mask/truncation | Dense credit entry | Crucible bridge (M12, specified here so M11 exports carry what it needs) |
|---|---|---|---|---|
| **skyrl-gym / skyrl-train** (Apache-2.0) | `Env.init(prompt)→(obs, info)`, `step(action:str)→{observations, reward, done, metadata, postprocessed_action?}`, `close`, `get_metrics`, tool groups (`SK/skyrl-gym/skyrl_gym/core.py:12-93`; `envs/base_text_env.py:9-93`); registry `envs/registration.py` | per-step reward at each turn's last token, or step-wise samples (`SK/skyrl/train/generators/skyrl_gym_generator.py:712-731`); `GeneratorOutput{rewards: List[float]\|List[List[float]], loss_masks, stop_reasons, rollout_logprobs, is_last_step}` (`SK/skyrl/train/generators/base.py:40-65`); GRPO **sums** token rewards (`SK/skyrl/backends/skyrl_train/utils/ppo_utils.py:1344-1367`); env.step runs in an executor (sync) (`skyrl_gym_generator.py:549`) | per-token rewards are honoured only by token-level estimators (GAE etc.), not by GRPO | `BaseTextEnv` around a World running its own event loop thread; text action → tool-call parse; per-step reward Ψ deltas; masked → drop the trajectory (no mask field) or zero `loss_mask` in a custom generator hook |
| **rLLM** (Apache-2.0) | `BaseEnv.reset()→(obs, info)`, `step(a)→(obs, r, done, info)`, `from_dict`, `is_multithread_safe` (`RL/environments/base/base_env.py:33-80`); `Step{prompt_ids, response_ids, logprobs, reward, mc_return, advantage: list[float]\|float\|None}` (`RL/types.py:100-136`); `Trajectory{steps, reward, signals}`, `Episode{termination_reason, …}` (`:241-330`) | `TerminationReason` ∈ {max prompt/response length, env_done, max_turns, timeout, unknown, error} (`RL/workflows/workflow.py:18-25`); per-reason masking via `compact_filtering` (`RL/trainer/algorithms/config.py:122-161`) | **`use_precomputed_advantage`** (`RL/trainer/config/rllm/base.yaml:79`; `RL/trainer/algorithms/config.py:282`; used in `RL/trainer/tinker/transform.py:105-112`) | the easiest dense bridge: `Step.advantage` = per-token list from `steps.jsonl` ΔΦ on the call's tokens + group-relative outcome; masked → `ERROR` with `mask_error: true` |
| **TRL** (Apache-2.0) | §3.5 | §3.5 | none (sequence-level) | shaping only through the reward; dense credit needs a custom loss (out of M11 scope) |
| **ROLL + GEM** (ROLL Apache-2.0; GEM is a pip dependency, its license not verified locally) | `gem.register(id, entry_point)` (`RO/env/__init__.py:8-21`); `reset(seed)`, `step(a)→(obs, reward, terminated, truncated, info)` (e.g. `RO/env/sokoban/env.py:69-147`) | trajectory manager sums step rewards into `episode_score` at the last token and keeps `step_scores` (`RO/env_manager/traj_env_manager.py:307-385`); `truncated` set on max steps (`:190-203`); masks are response masks only | `step_env_manager.py` builds per-step samples | GEM `Env` wrapper; no infra mask → drop |
| **verl agent loop** (Apache-2.0) | `BaseTool.create/execute(→ToolResponse, reward, metrics)/calc_reward/release` (`VE/tools/base_tool.py:36-87`); `AgentLoopOutput{prompt_ids, response_ids, response_mask, response_logprobs, reward_score?, num_turns, extra_fields}` (`VE/experimental/agent_loop/agent_loop.py:88-115`); tool parsers hermes, gpt-oss, qwen3_coder, glm (`VE/experimental/agent_loop/tool_parser.py:48-378`) | scalar `reward_score` | none in vanilla | use the MiMo-fork recipe (§3.7) |
| **slime** (Apache-2.0) | custom `rollout_function_path` (`FW/infra-thudm-slime/slime/utils/arguments.py:867,2147`) | `Sample.Status` ∈ {PENDING, COMPLETED, TRUNCATED, ABORTED, FAILED}; `reward: float\|dict`; `loss_mask` (`FW/infra-thudm-slime/slime/utils/types.py:131-158`) | via a custom rollout | later |
| **AReaL** (Apache-2.0) | `RolloutWorkflow.arun_episode(engine, data)`; returning `None` rejects the trajectory (`FW/infra-inclusionai-areal/areal/api/workflow_api.py:14-30`) | rejection = mask | token-level rows | later |

**Chat templates and tool formats.** The World works at the ToolSpec level, so each bridge owns the conversion:
- OpenAI chat (verifiers renderer, SkyRL, TRL `parse_response`, `TRL/trainer/grpo_trainer.py:2178-2187`);
- OpenAI Responses (NeMo-Gym);
- MCP (OpenEnv, agentenv, MiMo);
- text XML/tags (verl parsers; GEM `<guess>…</guess>`, `FE/00-environments-101/envs/wordle/gem/env.py:17`).

The invariant everywhere is token-in-token-out: never re-render a sampled assistant turn (`MV/recipes/general/token_trace.py:14-31`; `VF/graph.py:13-19`).

### 3.7 MiMo format and the XiaomiMiMo verl fork (DEPEND, PORT, TARGET; fork Apache-2.0, mimoagent MIT, uni_agent Apache-2.0, release Apache-2.0)

**MiMo general env** (267 envs). Each has `manifest.json`:
- uploads per container (`main`/`sidecar`);
- `setup.command` starts `sidecar_entrypoint.py --start-and-detach`;
- `wait_ports` 39101+;
- `mcp_servers[{name, url}]`.

The `manifest.json` files are 1-space-indented JSON (example `MM/general/envs/s3k_0000_accounting_audit_tax_en_t1_rl_008/manifest.json`). The sidecar starts one MCP HTTP server per `tools/<app>.py` over `system/<app>/{schema.sql,state.db}` and a health server. Its docstring promises `GET /dump, POST /verify` on :39100 (`…/sidecar_entrypoint.py:11`), but the handler implements only `GET /health` (`:106-119`). **None of the 267 sidecars defines `do_POST`** (grep count 0). Grading goes through `run_verify.py`, which writes `reward.json` = `{"reward": s}`, or `{"reward_error": e}` when masked, plus `reward_detail.json` (`…/run_verify.py:95-150`).

**Scoring.**
- `_grade_sc` runs the `src_protect` gate and `gate:true` rule items first; any failure → 0. LLM items are skipped after a gate failure.
- Atoms are scored `round(Σw·s/Σw, 4)`; agent and vision items are skipped and not counted in the denominator.
- Judge unavailable → `reward_error: judge_crashed` (masked).
- All of this is in `…/verify.py:343-406, 458-469`.
- `tier` (critical/important/sanity) is **metadata only** in this scorer. Across the 267 envs: 723 llm/critical, 489 llm/important, 124 rule/critical, 89 rule/sanity, 2 rule/important, and **0 `gate:true` items** (counted from all `verifier_meta.json`).
- The scorer prefers `synthesis.office_gen.grade_rubric` when importable (`…/verify.py:408-417`), so **the same env grades differently** depending on what is installed (agent/vision items included or skipped). Import must pin the self-contained path.

**Catalog.** accounting_audit_tax 157, consulting_bizops_analytics 89, construction_pm 14, agriculture_coop 7. Languages: en 157, zh 110. Tiers t1–t5: 28/98/92/40/9. Apps per env: 0–17, mean 6.7.

**verl fork consumption.**
- `DatasetEnvActor` owns a mimoagent k8s pod and its tools (`MV/recipes/general/env_actor.py:14-45`). `calculate_reward` optionally binarizes (`:101-105, 301-354`).
- `GeneralAgentLoop` uses TokenTrace (generated tokens 1, environment-injected 0).
- An infra failure scores `INVALID_REWARD_VALUE = -999` (`MV/recipes/general/agent_loop.py:76-87, 640-700`). The trainer replaces it with the group's valid mean and a zero advantage (`MV/verl/trainer/config/ppo_trainer.yaml:133-139`; `MV/verl/trainer/ppo/core_algos.py:306-309`).
- `filter_groups` and a length penalty are configured in `MV/recipes/general/config/general.yaml:84-104`.

**Code recipe.** `extra_info.instance_json` with `docker_image, cwd, base_commit, test_command, test_patch, verifier_timeout_sec, language, instance_id` (`MV/recipes/code/dataset.py:14-45`; field reads in `MV/third_party/mimoagent-osr/src/mimoagent/environments/datasets/{base,opensource_code}.py`).

**Rubric judge.** Optional `rubric.rubrics` items folded by `combine ∈ {programmatic, rubric, product, weighted}` and `solution_combine ∈ {weighted, product}` (`MV/third_party/mimoagent-osr/src/mimoagent/environments/rubric_judge.py:1-75, 125-145`).

**M11 builds.**
1. **Verifier import.** Run `run_verify.py` against the World's final state. `reward` → `rj.reward`, items → `rubric/<id>`, `reward_error` → masked with `judge_failed` or `verifier_error`. Use cached judge verdicts for identity.
2. **Byte-identical export.**
   - At import, record a per-file sha256 manifest of the pristine env dir. Today's importer copies it as a bundle and ignores the verifier files (`CR/interop/mimo/importer.py:56-97`).
   - At export, emit the pristine bytes for every file, including the ignored verifier files and `state.db` (**never** regenerate SQLite from a dump; any write changes its bytes). Then check the manifest.
   - Edited Worlds (forged variants) export as new envs with new hashes, not as "round-trips".
3. **verl export.** Dataset rows for `recipes/code` and MiMo-format env dirs for `recipes/general`. Masked → −999.
4. **Dense credit into verl.** Signed-span via the PORTed `signed_rebalance` (M12) on TokenTrace spans keyed by `call_id`.

### 3.8 EnterpriseOps-Gym (PATTERN, DATA; Apache-2.0) → `interop/eog`

- Tasks are `BenchmarkConfig{system_prompt, user_prompt, verifiers[], number_of_runs, gym_servers_config[{mcp_server_name, mcp_server_url, mcp_endpoint="/mcp", seed_database_file, auth_config, context}], selected_tools, restricted_tools, temperature, reset_database_between_runs}` (`EOG/benchmark/models.py:30-84`).
- Verifiers are `database_state` (SQL via MCP, `comparison_type`), `response_check` (LLM) and `tool_execution`, routed by `gym_name` (`EOG/benchmark/verifier.py:18-195`).
- Success = all verifiers pass. `verifier_level_pass_rate` = Σpassed/Σtotal (`EOG/benchmark/executor.py:565-636`).
- Domains: calendar, csm, drive, email, hr, hybrid, itsm, teams. The MCP servers ship only as Docker images (`EOG/README.md:100-135`).

**Importer:** one App per gym (MCP server App from the image; state through the SQL adapter on the seed DB); verifiers → gates (`database_state` → SQL check; `tool_execution` → trace check; `response_check` → judge item, cached); success → outcome; verifier-level pass rate → `components["eog/verifier_pass_rate"]`.

### 3.9 Other sources

| Source | What it is (contract) | M11 relevance | License, mode |
|---|---|---|---|
| SWE-smith | SWE-bench instance schema (instance_id, repo, patch, FAIL_TO_PASS, PASS_TO_PASS); harness grading + repo profiles | maps 1:1 onto a Harbor dir; already PORTed in M8 (history-leak fix) | MIT, PORT |
| Repo2RLEnv | repo → **Harbor-format** verifiable tasks (README badge "spec: Harbor") | import through Harbor; leak patterns already PORTed | Apache-2.0 + MIT, PORT |
| Karotte | VM-confined eval framework: `Task.system_prompt/configure_tools/pre_hook/steps/tools/required_hardware/submission_paths` (`FW/preferencemodel-karotte/src/karotte/task.py:39-120`) | PATTERN for confinement (already PORTed) | MIT |
| mini-swe-agent | bash-only harness; trajectories | harness calibration (M10) | MIT |
| karpathy/autoresearch | agent edits `train.py`, 5-min fixed GPU budget, metric `val_bpb` (README) | an environment kind: ML research with a record-relative reward | **no LICENSE file** → PATTERN only |
| veds12/git-swarm | project website only (README) | none | no license |
| FineEnvs 00/05 | one core × six framework adapters; multi-harness (TRL whitebox, native OpenCode, Harbor × 4 harnesses) | PATTERN for M11's test layout and the harness rotation in M12 | Apache-2.0 |

---

## 4. agentenv plugin: exact extension points

All four pieces live in `CR/interop/agentenv/plugin/` (import-linter: only `interop/` imports agentenv). Registration is by entry point. The group names are in `AE/plugins/_registration.py:24-30`, and the docstring rule "group = kind, entry-point name = registry key, value = class" is in `AE/plugins/__init__.py:1-15`. Plugin settings live under `[plugins.<dist>]` (`AE/config/plugin_tables.py:79-95`).

```toml
# crucible/pyproject.toml (agentenv extra)
[project.entry-points."agent_env.sandbox_providers"]
gpu_docker = "crucible.interop.agentenv.plugin.gpu_docker:GpuDockerSandboxProvider"
[project.entry-points."agent_env.task_steps"]
crucible_world  = "crucible.interop.agentenv.plugin.steps:CrucibleWorldStep"    # name == cls.type
crucible_verify = "crucible.interop.agentenv.plugin.steps:CrucibleVerifyStep"
[project.entry-points."agent_env.artifacts"]
crucible_ref = "crucible.interop.agentenv.plugin.artifacts:CrucibleRefArtifact"
[project.entry-points."agent_env.cli_plugins"]
crucible = "crucible.interop.agentenv.plugin.cli:crucible"                       # AE/plugins/_cli.py:1-31
```

The same can be done from config without packaging:
- `[sandbox.providers.gpu_docker] impl="module:Class"` with an optional `config` table (`AE/providers/sandbox_providers/sandbox_provider.py:263-313`);
- `[task_steps] impls=["module:Class"]` (`AE/task_step/registry.py:145-175`);
- `[artifacts] impls` (`AE/artifact/registry.py:45-57`).

Built-in names cannot be overridden (`sandbox_provider.py:283-300`). Conflicting plugin names are dropped and reported (`AE/plugins/_registration.py:93-120`); `agent-env plugin` reports the inventory (`AE/plugins/_inventory.py`).

### 4.1 `gpu_docker` sandbox provider

**Base class.** `SandboxProvider` (`AE/providers/sandbox_providers/sandbox_provider.py:38-197`):
- abstract `create_sandbox(image_name, port, env, cpu, memory, disk_size_gb, timeout, attribution, network_policy)`;
- optional `create_vm`, `create_container` (default: `create_vm` → `pull_images` → `_start_container` → `docker run -d {_container_args}`, `:72-114`);
- `get_sandbox`, `close`, `from_config(**cfg)` (`:133-135`);
- class variables `URL_REWRITES` (`:138`), `EXTRA_CONTAINER_RUN_ARGS` (`:143`), `CONTAINER_ENV` (`:172`), `REQUEST_HEADERS`, `EGRESS_HOSTS`, `supports_network_policy`.
- `_validate_plugin` requires a subclass with no unimplemented abstract methods (`:250-254`).
- Fallback chains are comma-separated specs → `ChainedSandboxProvider` (`:531-554`).

**Where GPU flags go.** `_container_args` splices `EXTRA_CONTAINER_RUN_ARGS` into the agent/env container's `docker run` (`:116-124`). Implementation:

```python
class GpuDockerSandboxProvider(LocalSandboxProvider):          # AE/providers/sandbox_providers/local_sandbox.py:431
    EXTRA_CONTAINER_RUN_ARGS = LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS + " --runtime=nvidia --gpus all --init --shm-size 8g"
    def __init__(self, docker_host: str | None = None, gpus: str = "all", runtime: str = "nvidia"): ...
    @classmethod
    def from_config(cls, **cfg): return cls(**cfg)            # [sandbox.providers.gpu_docker.config]
```

Three caveats:
1. **`run_docker_container` steps bypass the provider.** They build `docker run` themselves with only `devices/cap_add/privileged/volumes/shm_size` (`AE/task_step/task_steps/run_docker_container.py:31-56, 236-253`). A task whose verifier or app container is built that way gets no GPU. Either ship a `crucible_run_container` step (same params plus `gpus`, `runtime`), or (fragile) wrap the provider's `LocalSandbox.exec` to rewrite `docker run`.
2. **Remote daemon.** `LocalSandbox.exec` inherits `os.environ` (`local_sandbox.py:283-288`), so `DOCKER_HOST=ssh://…` reaches the lab RTX 5090. But port publishing and `tunnel_urls` assume loopback (`:135-162`), and the provider says local sandboxes never share a network (`:551-557`). A remote host therefore needs `URL_REWRITES`, an SSH tunnel or Tailscale address mapping. Crucible already solves this for Harbor (`DOCKER_HOST` inheritance, `CR/interop/harbor/world_backend.py` header) and can reuse that code.
3. **Verifier env injection.** The provider's `CONTAINER_ENV` is injected into every verifier's environment (`…unit_tests_verifier.py:181-183, 254`). Keep it empty: no secrets in verifier containers.

### 4.2 `crucible_world` step (a World served as MultiEnv plus gateway)

**Base class.** `TaskStep` (`AE/task_step/task_step.py:84-160`):
- `type: ClassVar[str]` must be unique and must not inherit the base default (`AE/task_step/registry.py:166-171`);
- `async execute(context) -> context`;
- `preflight() -> list[str]` (`:120-125`, no sandbox);
- `step_param_overrides(context)` (`:127-133`);
- `to_dict`/`from_dict`, where `from_dict` is mandatory (`registry.py:14`);
- `entity_refs` for ids.

**Design A (build first).**
- `execute` builds the World with Crucible's existing `AgentEnvBackend`. That backend already runs a real agentenv `Gateway` subprocess with one internal MCP server per App, tool names `<app>__<tool>`, per-role ACL, clock and triggers (`CR/interop/agentenv/world_backend.py:1-21, 84-246`).
- The step then appends a `DeployedGatewayEnv(env_id=params.env_id, env_version=0, env_provider_type="gateway", gateway_url=backend.gateway_url, gateway_mode="consistent", sandbox_id=…)` to `context.deployed_envs` (fields at `AE/env/env.py:63-187`).
- Effect: the stock `deploy_agent(env_ids=[…])`, `prompt_agent`, `sync_env_clock`, `register_env_triggers` and `modify_env_tool_access` steps work unchanged. They look envs up by `env_id` in `context.deployed_envs` (`AE/task_step/task_steps/deploy_agent.py:67,431`).
- Clock: `sync_env_clock` arms the real gateway clock via `/clock/set-time` (`AE/task_step/task_steps/sync_env_clock.py:1-26`; `gateway.py:247-262`). Crucible's backend keeps its step-mode clock as the authority and mirrors it to the gateway (`world_backend.py:292-298`).
- Snapshot: `World.snapshot()` → `WorldRef`; store `world_ref` in `context.metadata["crucible"]`, ids only, because the context is persisted.

**Lifetime.** The World object lives in the runner process. That is fine for agentenv's local runner (`AE/runner/local_runner.py`). For a hub or remote runner, run the World as a Crucible World server and put its URL in the `DeployedEnv`.

**Design B (later).** Register an `Env` type `crucible_world` (group `agent_env.envs`; base `AE/env/env.py:213-320` with `deploy`, `from_deployed_env`, `reset`) plus an `EnvironmentProvider` (group `agent_env.env_providers`; `AE/providers/env_providers/env_provider.py:28-40`). Then `deploy_env`, `reset_env`, `snapshot_env` and the explorer treat Crucible Worlds natively.

### 4.3 `crucible_verify` step (separate verifier container)

**Why not compose built-ins.** The obvious composition is `run_docker_container` (verifier image) → `load_artifact` → `run_container_unit_tests_verifier(container_name=verifier, reward_path=/logs/verifier/reward.txt)`. It fails on several counts:
- no GPU (§4.1);
- the reward path must be the lossy `reward.txt` (N2), clamped to [0,1] (`…unit_tests_verifier.py:410`);
- a mask cannot be expressed: missing → score 0 (`:336-342`);
- there is no `steps.jsonl`.

**Step parameters.** `{sandbox_name, task_dir | task_ref, verifier_image (digest), agent_container | collected_artifacts_step_id, artifact_paths, gpu, timeout_sec, verifier_id, write_steps: bool}`.

**`execute`:**
1. Collect the candidate. Take the patch via `git diff` in the agent container, the same transport `collect_artifacts` uses (`AE/task_step/task_steps/collect_artifacts.py:1-40, 226`), or take files from a prior step's artifacts.
2. Grade in a **fresh container from the pinned image** using `CR/interop/harbor/verify_dir.HarborDirTask.grade_patch` (`CR/interop/harbor/verify_dir.py:49-135`). It already does Harbor layout, pinned image digest, network, limits, GPU and a remote `DOCKER_HOST`.
3. Read `crucible_reward.json`, `reward.json` and `report.json`, then build `steps.jsonl` by replay (§2.4) from the gateway `/trajectory` plus the harness trajectory.
4. Record `context.metadata["verifications"][verifier_id]` in the aggregate-compatible shape (`…unit_tests_verifier.py:353-379`):
   - `results` = one row per gate/check `{id:"gate/<g>", score, result, weight}`;
   - `score = rj.reward` (not clamped);
   - `crucible: {reward_json, masked, failure_kind, steps_artifact}`.
   - When masked: write rows with `skipped: True`, so `aggregate_verifiers` drops them (`AE/task_step/task_steps/verifiers/aggregate_verifiers.py:77`), then **raise a typed error after recording**. The run becomes FAILED (infra), never score 0. This is the same pattern the built-in uses for timeouts (`…unit_tests_verifier.py:381-392`).
5. Upload `reward.json`, `crucible_reward.json`, `steps.jsonl` and `report.json` as `FileArtifact.put_at(...)` (pattern at `…unit_tests_verifier.py:412-429`).

Do not feed Crucible verifications into `aggregate_verifiers`: it clamps to [0,1] (`scoring.py:39`).

### 4.4 Artifact mirroring

- Artifacts are immutable, versioned pydantic models `{id, version, type}` (`AE/artifact/artifact.py:24-80`) with `put`/`get`/`query`. `FileArtifact.put_at/put_attempt/load` live at `AE/artifact/artifacts/file.py:140-160`. Built-in types: file, file_artifact_universe, environment_universe, docker_image, skill (`AE/artifact/artifacts/`).
- Plugin type `CrucibleRefArtifact(Artifact)` with `type="crucible_ref"` and `{store_key (sha256), kind: world_ref|state_ref|grade|trace|rollout, derived_from}`.
- Mirror rules:
  - Crucible `WorldRef`/`StateRef` → `crucible_ref`, plus a `FileArtifactUniverse` for fs Apps.
  - agentenv `snapshot_env` → `EnvironmentUniverseArtifact` (`AE/task_step/task_steps/snapshot_env.py:1,158`) → a Crucible `StateRef` for SQL Apps.
  - Gateway trajectory and harness trajectory → Crucible trace store.
- Every mirror records `derived_from` both ways (`CR/core/prov.py`).

---

## 5. Environment diversity

### 5.1 Catalog (counts are local directories; examples are paths)

| Kind | Examples (framework: path) | State | Clock | Seeding | Reward type | Observation | Step latency, resources |
|---|---|---|---|---|---|---|---|
| **Coding / SWE** | research-env `environments/swe` (19: deep_swe, multiswe, openswe, r2e_gym, scaleswe, senior_swe_bench, swe_atlas_*); NeMo coding (25: swebench_pro, swe_rebench, swemer_v1/v2/oml, deepswe, scale_swe…); MiMo `code.parquet` (2,698); SWE-smith; Repo2RLEnv; Crucible tasks 01–04 | fs + git, build artifacts | none | test-order flake | gates (build/tests), F2P/P2P, partial tests, rubric | text | s–min (builds); GPU for CUDA tasks |
| **Kernels / perf** | prime-env kernelbench, backend_bench, gpu_puzzles(_modal), triton/tensor puzzles; OpenEnv `envs/kernrl`; Crucible task04 | fs + process + GPU counters | none | perf noise | correctness gate × speedup tiers (record-relative) | text | min; GPU, quiet host (PerfGuard) |
| **Terminal / shell** | research-env terminal (terminal_bench_2, tmax, terminal_lego, recursive_task_synthesis, openthoughts_tblite); NeMo terminal_bench_2_1/4, terminal_multi_harness; OpenEnv tbench2_env, terminus_env; ROLL rock_tb_native_env | container fs | none | low | test script (Harbor-style) | text | s |
| **Browser / web / computer use** | OpenEnv browsergym_env, helium_browser_env; NeMo interactive_browser, osworld, gui_coordinate; prime-env web_voyager, androidworld, seeclick; FineEnvs desktop (E2B, 19 tools, `FE/00-environments-101/project.yaml:41-50`) | DOM / a11y tree, VM disk | wall (live sites) | low (live web) | task checks, judges | **image** + DOM | 0.1–3 s/action; VM or browser |
| **Search / deep research** | research-env search (11: browsecomp, deepdive, s1_deepresearch, wideseek…); NeMo browsecomp_advanced_harness, google_search, tavily_search; prime-env browsecomp_plus, webexplorer, wiki_race | external (web) or cached corpus | wall | none (live) | exact answer / judge | text | s; egress allowlist |
| **Enterprise SaaS / tool use (MCP + DB)** | EOG (8 domains); MiMo general (267, 4 domains, 6.7 apps/env); OpenEnv agent_world_model_env (1,000 envs, 10k tasks), calendar_env, openapp_env, finqa_env; NeMo workplace_assistant, calendar, finance_agent_v2, indian_banking, legal_agent_bench, toolsandbox, xlam_fc; research-env tool_use (tau2/tau3, bfcl_v3, mcp_atlas, apex_agents, enterprise_ops_gym); prime-env tau_bench, github_mcp, mcp_universe, art_e; agentenv MultiEnv/ServiceDB/Website universes | SQL (sqlite/postgres), docs, mail | virtual (calendars, SLAs) | user-sim randomness | SQL state checks + rubric/judge + tool-trace checks | text / JSON | ms–s; MCP sidecars |
| **Simulation / digital twins / OR** | **PortSimEnv** (§5.3); OpenEnv sumo_rl_env (traffic), finrl_env (markets), carla_env (driving), wildfire_env, dm_control_env, unity_env; NeMo openair_congestion; Crucible toy `helpdesk_backlog` (DESIGN §10.4) | in-memory sim / JSON | **sim time** | seeded generators | optimization gap vs baseline/optimum; per-step env reward | structured JSON, sometimes image | ms (pure Python) to 100 ms+ (CARLA, GPU) |
| **Games** | OpenEnv atari, chess, connect4, openspiel, textarena, snake, maze, grid_world; NeMo blackjack, grl_sokoban, grl_tetris, wordle, tales; ROLL sokoban, frozen_lake, webshop (`RO/env/__init__.py:8-21`); prime-env balrog_bench, hanabi, mastermind, text_quests, nyt_connections; FineEnvs wordle | in-memory | turns | seeded | per-step or terminal score | text / grid / image (Atari) | ms; in-process |
| **OCR / vision / multimodal** | FineEnvs latex-ocr (Qwen3-VL), watercolour (headless Chromium render + aesthetic PM + pairwise judge), geoguesser (km error); research-env multimodal (charxiv, mmk12, mmmu_pro, virl39k); NeMo labbench2_vlm, vlm_eval_kit, image_tools, circle_click/count; ROLL deepeyes, geo3k (geometry); prime-env vision_sr1, penrose_diagrams | stateless or renderer | none | low | continuous (edit distance, km, PM score) | **image in / image out** | 0.1–10 s; GPU for PM/judge |
| **Geo** | FineEnvs 03-geoguesser (street panoramas, haversine km) | dataset | none | seeded draw | continuous distance | image | s |
| **Data science / SQL / spreadsheets** | FineEnvs SmolDataEnvs (5,394 Kaggle tasks; ladder grader: exact → numeric tolerance → list/percent → symbolic, `FE/04-smoldataenvs/README.md`); NeMo bird_sql, spider2_lite, text_to_sql; skyrl-gym sql; prime-env dabstep, polars_env, bixbench | sandbox fs with data | none | low | exact/tolerance answer match | text | s; Daytona/E2B |
| **Math / reasoning (single-turn)** | research-env math (9), lean (12), reasoning (7), science (5), knowledge (8); NeMo math (15), knowledge (26); skyrl-gym aime, gsm8k, lcb | none | none | none | verifier function | text | ms |
| **Multi-agent** | prime-rl `hierarchical_grpo` (proposer/solver, `PR/orchestrator/algo/hierarchical_grpo.py:19-42`); OpenEnv chess/connect4; prime-env hanabi; agentenv `peer_agents`/A2A; Crucible actors | shared | turns | seeded | per-actor rewards | text | varies |
| **Human-in-the-loop / user simulation** | verifiers `envs/user_sim`; agentenv `deploy_human_agent`, `review` step; tau-bench user sim; NeMo conversational_tool_use_simulation; mimoagent `agents/user_agent.py` | conversation | wall/virtual | LLM sampling | task + user satisfaction | text | s–min (human) |
| **MCP-server envs** | agentenv MCPServerEnv; OpenEnv MCPEnvironment (RFC 003: the agent sees only MCP tools); NeMo `expose_tools_over_mcp`, example_mcp_weather; MiMo sidecars | per server | — | — | — | text | ms |
| **Security / CTF / safety** | MiMo `cyber.parquet` (1,000 ARVO); research-env reasoning/uuid_ctf; NeMo safety (6: indirect_prompt_injection, jailbreak_detection…); prime-env agentharm, agent_dojo, wmdp | container | none | low | exploit success / refusal | text | s; strict isolation |
| **Science / lab** | NeMo assaybench, bunsenbench_chemistry_mcq, ether0, moldetox, newton_bench; research-env drug_discovery_bench, frontierscience; OpenEnv dipg_safety_env | tools / sim | none | — | judge / exact | text | s |
| **Long context** | research-env long_context (13); NeMo ruler, ruler2, mrcr, longmemeval | none | none | none | exact | text | ms |
| **ML research automation** | karpathy/autoresearch (`val_bpb` under a fixed 5-min budget) | fs + GPU run | wall budget | training noise | record-relative metric | text | 5 min/step; GPU |

NeMo-Gym's 145 servers by `domain`: knowledge 26, agent 26, other 25, coding 25, math 15, instruction_following 11, safety 6, games 6, rlhf 4, long_context 1.

### 5.2 What a Crucible World or App template must support, per kind

| Requirement | Present today | Needed by | Gap / action |
|---|---|---|---|
| fs/git, SQL (sqlite/postgres), JSON-doc, process State Adapters | yes (`CR/world/state/{fs,sql,postgres,json_doc,process}.py`) | coding, enterprise, sims | none |
| browser DOM / a11y state adapter | App exists (`CR/world/apps/browser.py`), no `state/browser.py` | browser, computer use | add a DOM/a11y fingerprint adapter |
| image observations and image tool results | not verified in `ToolResult` | vision, desktop, Atari, geo | confirm multi-part content in `core/schema/actions`; carry image blobs by store ref |
| step-mode clock + explicit `advance` + Events | yes (`CR/world/clock.py:50-85`) | enterprise SLAs, sims, long horizon | add a **sim-owned clock** mode: an App advances time (DES) |
| world seed → per-App RNG | `WorldSpec.seed` (`CR/core/schema/world.py:215`) | sims, games, user-sim | derive `seed_app = H(seed, app_id)` in `AppContext` and record it in the trace |
| per-tool quotas (e.g. `check_plan` 10/episode) | `LimitsSpec` has only timeouts and output caps (`CR/core/schema/world.py:168-173`) | PortSim, budgeted tools | either the App enforces quotas (faithful import), or add `LimitsSpec.tool_quota` |
| record-relative / optimization reward | `score` gates, unbounded above 1 (`CR/verify/runtime/plan.py:11-13`) | kernels, sims, autoresearch | none; declare caps for exports |
| opaque external env (no state access) | AWM via fastapi services (`CR/interop/openenv/awm.py`) | OpenEnv/NeMo imports | generic `openenv_app` with **replay-restore** (deterministic re-execution from reset), capability flag `restore: replay` |
| MCP server App | yes (`CR/world/apps/mcp_server.py`) | all MCP envs | none |
| in-process fast path (ms steps) | local backend | games, math, sims | a benchmark gate: `World.step` overhead < 5 ms on toy |

### 5.3 PortSimEnv v1: deep dive

**What it is.** "Re-plan a broken week of container-ship dockings at the Port of Barcelona, graded against a proven optimum" (`FE/07-simulation-environments/portsim-v1/project.yaml:3-4`). An OpenEnv environment built from the port's 1,784 real 2024 container calls (`ART/portsim.mdx`; data in `portsim-v1/data/barcelona/container_calls_2024.csv`, 1,785 lines with header; loader `PS/core/berth_core/data.py:12-13,55`).

**Environment API** (`PS/openenv/berth_openenv/environment.py`):
- `BerthPlanningEnvironment(MCPEnvironment)`, `SUPPORTS_CONCURRENT_SESSIONS = True` (`:119-122`).
- Three FastMCP tools:
  - `get_situation()`;
  - `check_plan(plan)`, budgeted `BERTH_MAX_CHECKS=10`, returns violations and cost, never a score (`:148-169`);
  - `submit_plan(plan)`, which ends the episode and grades once (`:171-191`).
- `MAX_TOOL_CALLS=24` (`:38-39`). Plan entries are `{ship, berth_hour, section, cranes?}` (`:42-51`).
- `reset(seed, episode_id, split, index, task_id)` → observation metadata `{instructions, situation, max_checks, max_tool_calls, tools, prompt_version, …}` (`:217-244`).
- `_after_step` sets `obs.reward = rubric(action)` (0.0 unless submit). Hitting the tool-call limit → `done, reward 0.0, end_reason tool_call_limit` (`:259-279`).
- Task API: `list_splits/num_tasks/get_task/get_task_range` (`:198-213`).
- `openenv.yaml` (`PS/openenv/openenv.yaml`): `range [0,1]`, `oracle_tolerance 0.001`, `network no-network`, `rubric_tree`, `task_api`, `declared_task_count` eval 50 / train 1050, `MAX_CONCURRENT_ENVS 64`.

**Simulation core.**
- **Model** (`PS/core/berth_core/model.py`): `Ship{arrival, handling, planned_hour/section, due, workload, min/max/std_cranes, weight, berth_deadline, deadline_penalty, …}` (`:25-46`); `Block{first,last,start,end,kind,cranes}` (`:60-68`); `Task{ships, blocks, notices, disruptions, difficulty, reference (hidden), rules (crane_pool, outages, max_moves_per_hour, reward)}` (`:72-88`).
- **Evaluator** `check.evaluate`: berth at or after arrival, stay on the quay, no section-hour overlap with ships, alongside blocks or closures, crane pool per hour minus outages, movements per hour, wind no-move windows. Cost = Σ sections × hours late (priority ×3) + emergency penalty + 5 per moved ship (`PS/core/berth_core/check.py:59-170`). Provable floor: `unavoidable_cost` (`:172-187`).
- **References:** `naive_replan` (`PS/core/berth_core/solve.py:55`) and the CP-SAT `optimal_plan` (`:77-175`, OR-Tools). Baselines `published/greedy/serial` (`PS/core/berth_core/baselines.py:16-47`).
- **Generator** (dock-v1): `build_dock_task(…, tier, seed)` with `rng = random.Random(f"dock-{quay}-{week}-{n_weeks}-{tier}-{seed}")` (`PS/core/berth_core/dock.py:89-92`); disruptions are closure, late, bunching, outage, gale, divert, extra, emergency, priority (`:1-16`); `HELD_OUT` eval weeks (`:33`). A task is kept only if CP-SAT is within 1% of its bound, naive ≤ 0.6, greedy ≤ 0.85 and ≥ 12 ships (`:35-38, 287`; `portsim-v1/DESIGN.md:44-49`).
- **Not dynamic.** "All information is given up front; the difficulty is the size and the coupling of the problem" (`dock.py:1-3`). The 3D twin's kinematic traffic controller is visualization only (`DESIGN.md:111-116`).

**Reward** (`PS/core/berth_core/reward.py`, v3 for dock-v1): infeasible → `0.2 × clean_fraction` (entry problems count as unclean); feasible → `0.2 + 0.8·exp(−g/0.5)` with `g = max(0, cost−opt) / (max(0, opt−floor) + gap_k)`, `gap_k=100` (`:74-88`). Graded once on submit as the rubric `BerthPlanRubric` with children `ships_clean` and `cost_quality` (`PS/openenv/berth_openenv/rubric.py:58-103`). Rounded to 6 decimals (`reward.py:106`). Honesty checks: an independent occupancy-grid checker, hostile plans, no plan beats the optimum, lazy policies stay low; 331 tests (`DESIGN.md:89-96`).

**Data.** Real calls plus filled gaps (crane fleets, 28 moves/crane-hour, wind and traffic rules) plus seeded disruptions modelled on real events (Red Sea bunching, ordinance wind limits), then CP-SAT, then filtering. 1,050 train / 50 eval with no shared week (`ART/data.mdx`; `DESIGN.md:14-49`). License: **CC BY-SA 4.0** for data and derived packs; OSM data for the twin is ODbL; code Apache-2.0 (`portsim-v1/README.md`, "Data and licence").

**Rollouts.** Eval only: 6 models × 50 tasks, 12 turns, 32k output tokens per turn. GPT-6.1 Sol scores 0.888 and is optimal on 26/50. Open models lose most of their score by never calling `submit_plan` (`portsim-v1/README.md` results; `ART/rollouts.mdx`). The harness is `PS/openenv/berth_openenv/agent.py`: Anthropic, OpenAI Responses and the HF router over OpenEnv WebSocket sessions; two turns without a tool call end the episode; no submit = 0 (`:1-14`). Runner: `portsim-v1/eval/run_eval.py`. Status `wip`; **no training run yet**.

**Why it counts as an "enterprise simulation RL environment."** Per the article's intro (`ART/introduction.mdx`: "digital twins of real work… rebuild [a job] from the records it leaves behind, and grade the agent against an answer that can be proven best"):
1. Real operational records and real written rules (port ordinance, crane fleets).
2. Disruptions grounded in real events.
3. A provably optimal answer key, with the reward measured against the *avoidable* cost.
4. A difficulty filter against naive and greedy operators.
5. Splits held out by week.
6. A deterministic grader with an independent checker.
7. Budgeted, realistic tooling: a planner can "check" drafts a limited number of times.

**How Crucible hosts it.**

*Import via OpenEnv (now, PATTERN + DATA):*
- Generic `openenv_app` App: the MCP client over `/ws` session, tools `get_situation/check_plan/submit_plan` passed through.
- State adapter `opaque_replay`: snapshot = (task_id, action log), restore = re-run from `reset(task_id)`. This is valid because the env is deterministic: `rng` is seeded from the task id and the grade is a pure function.
- Reward: terminal `Observation.reward` → `rj.reward`; `ships_clean`/`cost_quality` → components; `feasible` → `check/feasible`. It must not become a gate: a gate would force reward 0 and break the 0.2 floor for infeasible plans.
- `tool_call_limit` → policy outcome 0 (N4).
- `steps.jsonl`: Φ is 0 until submit (state changes only at submit). Dense credit degenerates, which is faithful.

*Native port (next, PORT `berth_core`, Apache-2.0, into `packs/ops_logistics`):*
- App template `ops/berth_planner` with `json_doc` state `{task (agent view without reference), working_plan, checks_used, submitted}`.
- Two tool surfaces. **Faithful:** the original three tools. **Editor variant:** `set_entry`, `move_ship`, `assign_cranes`, `check`, `submit`, where `working_plan` is real state.
- Verifier plan: a `score` gate calling `berth_core.reward.grade`, plus subgoals per ship (`clean(ship)`) and a cost-quality subgoal.
- With the editor variant, Φ(s) = 0.2·clean_fraction(working_plan) while infeasible, and the full reward once feasible. Φ telescopes to the final grade and **is never shown to the policy**, which matches PortSim's own rule that checks never reveal a score (`environment.py:150-151`). This gives honest dense credit on a combinatorial task.
- Clock `off` for v1.

*Dynamic v2 (Crucible-only value):*
- Disruptions (late arrival, gale, outage) become `EventSpec`s revealed at simulated times.
- The step-mode clock advances when the planner commits a rolling-horizon decision.
- A simulated harbour master and terminal operator Actors issue notices.
- Reward = realized cost of the executed schedule vs the CP-SAT hindsight optimum.
- This needs the sim-owned clock (§5.2) and seeded per-App RNG.

*License guard.* Derived tasks keep CC BY-SA 4.0 (share-alike). DATA mode only: attribute, keep out of proprietary benchmark splits, and record the license on the task card.

---

## 6. Recommendations

### 6.1 Mode per framework

| Framework | License (read) | Mode | Why |
|---|---|---|---|
| Harbor | Apache-2.0 | DEPEND, TARGET | canonical task format; adapters everywhere read it |
| agentenv | Apache-2.0 | DEPEND + plugin, TARGET | enterprise worlds, A2A, clock/triggers; no trainer path |
| verifiers v1 | MIT | DEPEND, TARGET | prime-rl path; separate env (mcp conflict) |
| prime-rl | Apache-2.0 | DEPEND (trainer), small upstream PR for pluggable algorithms | dense advantages need an `Algorithm` |
| prime-/research-environments | Apache-2.0 | PATTERN, DATA | catalog, porting guides |
| NeMo-Gym | Apache-2.0 | TARGET, PATTERN (failure vocab already adopted) | NeMo-RL path |
| NeMo-RL | Apache-2.0 | consumer (no code taken) | reads legacy `instance_config.mask_sample` |
| OpenEnv | BSD-3-Clause | TARGET, PATTERN, DATA | TRL/SkyRL path; AWM corpus |
| TRL, SkyRL, rLLM, ROLL, verl, slime, AReaL | Apache-2.0 | bridge targets (M12) | rLLM first for dense credit |
| GEM | pip dependency of ROLL, not vendored; license not verified locally | later | — |
| MiMo verl fork | Apache-2.0 (mimoagent MIT, uni_agent Apache-2.0) | DEPEND, PORT (`signed_rebalance`), TARGET | training path 2 |
| MiMo release | Apache-2.0 (README front matter) | PORT (sidecar), TARGET, DATA | byte-identical round-trip gate |
| EnterpriseOps-Gym | Apache-2.0 | DATA, PATTERN | OOD benchmark importer |
| SWE-smith / Repo2RLEnv / Karotte / mini-swe-agent | MIT / Apache+MIT / MIT / MIT | PORT (done in M8/M10) | — |
| FineEnvs (code) | Apache-2.0 | PATTERN (six-adapter layout), PORT (`berth_core`) | test layout; diversity proof |
| PortSim data | CC BY-SA 4.0 (+ ODbL twin) | DATA with share-alike | license propagates to tasks |
| autoresearch, git-swarm | no license file | PATTERN only / none | — |

### 6.2 Prioritized M11 build list

**P0: gate-blocking (fix Crucible's own side first).**
1. **Single reward producer** (N1). Unify `flat_reward` and `RewardJson.flat`. Move the composer into the container runtime (`CR/verify/runtime`). Move `time/*` to `report.json`. Write `reward.txt` with `repr()` (N2, N5). Size S.
2. **Mask encoding** (N3). The masked grade writes `crucible_reward.json` only. The pre-write becomes an empty `reward.json` with no `reward.txt`. Add tests per consumer: Harbor exception; verifiers `TaskError`; empty-file behaviour. Size S.
3. **`interop/canonical.py`**: JCS canonicalization, the volatile-field strip list, derived `call_id`, tool-name maps per runtime, and an `identity_report(runtime_a, runtime_b)`. Size S.
4. **Replay-based `steps.jsonl`** (§2.4): trajectory normalizers for ATIF, agentenv gateway + harness log, verifiers Trace, NeMo Responses output and OpenEnv TrainingTrace → canonical trace → `ScriptedPolicy` replay → progress → rows. Size M.
5. **agentenv plugin** (§4): `gpu_docker` + `crucible_run_container`, `crucible_world` (design A), `crucible_verify`, `crucible_ref` artifact, entry points, `agent-env plugin` check in CI. Size M.
6. **verifiers export** (§3.3): `CrucibleTaskset/CrucibleTask`; pinned images; GPU resources. Size S–M.
7. **NeMo-Gym export** (§3.4): `CrucibleResourcesServer` + `manifest.yaml`. Size M.
8. **OpenEnv export** (§3.5): `CrucibleEnvironment`, Ψ-telescoping rewards, `CrucibleRewardRubric`, `openenv.yaml`. Size M.
9. **MiMo round-trip** (§3.7): pristine-bundle sha manifest, verbatim export, verifier import with cached judges. Size S–M.
10. **Gate harness.** Matrix {task02 patch, toy trajectory} × {Harbor, agentenv, verifiers, NeMo-Gym, OpenEnv}:
    - assert canonical `reward.json` and `steps.jsonl` equality;
    - assert state-fingerprint equality after replay through each runtime's tool transport;
    - negative controls: a forced masked grade must surface as each runtime's native mask and **never as 0**; a truncated episode must follow N4.

    Also: one MiMo env imported → run → re-exported byte-identical. Size M.

**P1: breadth.**

11. Trainer bridges carried by M11 exports: rLLM precomputed advantages; SkyRL per-step rewards; TRL `environment_factory`/`rollout_func` with `env_mask` and a None reward when masked; verl-fork −999.
12. `interop/eog` importer.
13. Generic `openenv_app` with replay-restore; import PortSimEnv; then port `berth_core` into `packs/ops_logistics` (editor variant with honest Φ).
14. World template gaps from §5.2: browser state adapter, image observations, sim-owned clock, per-App seeds, optional tool quotas.

**P2: later.** ROLL/GEM wrapper, slime/AReaL rollout functions, ORS, design B for agentenv (native `Env` type + provider), PortSim v2 dynamic.

### 6.3 Risks and open decisions

- **N1 choice (a) vs (b).** (a) puts the κΦ shaping into the scalar that foreign runtimes train on. That is intended (DESIGN §6.4), but it means a Harbor leaderboard's "pass rate" must read `outcome`, not `reward`. Record this in DESIGN.
- **Masked-as-missing for Harbor.** Some consumers turn a missing reward into 0 anyway (NeMo `harbor_agent`, rLLM's Harbor evaluator fallback, agentenv `reward_path`). Crucible can only control its own exporters. Document per consumer and never ship through those paths.
- **LLM-judge determinism.** Identity requires cached verdicts. A cache miss on another runtime must mask (`judge_failed`), not re-judge.
- **verifiers GPU on the lab host.** Not verified whether verifiers' docker runtime passes GPU flags for `TaskResources.gpu`. Check before relying on it for task02-class GPU tasks.
- **OpenEnv HTTP is stateless per request.** Exported environments must document `/ws` or MCP-session use (`OE/harbor/environment.py:1-5`).
