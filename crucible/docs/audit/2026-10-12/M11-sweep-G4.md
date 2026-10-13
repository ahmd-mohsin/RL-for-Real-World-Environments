# M11 sweep, group 4: agentenv and the RL trainers' environment interfaces vs Crucible M11 `interop` (2026-10-12)

Read-only. Crucible HEAD `af402b4` (only `crucible/.coverage` modified). Roots:
- `C` = `crucible/src/crucible`; `T` = `crucible/tests/unit`.
- `AE` = `agentenv-framework/src/agent_env`, `AP` = `agentenv-framework/packages/agentenv-protocol/src/agentenv_protocol` (041afd7, v0.9.1301, Apache-2.0 with NOTICE "agent-env, Copyright 2026 Scale AI").
- `REF` = `RL-Environments/06-frameworks-and-env-code`: `UV` verl (5eb4826), `MV` XiaomiMiMo verl fork (a2ad9f6), `SK` SkyRL (3d0ae22), `RL` rLLM (4e828c8), `AR` AReaL (298412a), `RO` ROLL (581046a), `SL` slime (c933787), `OR` OpenRLHF (dc2a7ad), `TR` TRL (72cf37d), all Apache-2.0; `CE` checkpoint-engine (84631a2, MIT).

Verdicts: **DEPEND** (call their code), **PORT** (copy; keep headers and NOTICE), **PATTERN** (re-implement the idea), **TARGET** (a contract our export must meet), **nothing**, **anti-pattern**.

How this was done. I read the agentenv plugin, harness, sandbox and replay code in Crucible and the agentenv registries, gateway, teardown, verifier and bundle code directly. Two read-only sub-sweeps read the trainer repos. I re-opened a sample of their citations and all matched: SkyRL `trainer.py:1068-1070`, `trainer_utils.py:700-701`; TRL `grpo_trainer.py:2144-2149,1268-1272,2690-2694`, `vllm_generation.py:550-572`; verl `agent_loop.py:103-108` (`metrics` required). Eight probes ran (`scratchpad/m11sweep/g4/`, section 4). No Docker, GPU or paid calls.

Scope note. The trainer adapters live in `C/train/bridges/` (M12 code). They are listed here because they are the only code that hands a Crucible rollout to verl, SkyRL, TRL and rLLM, which is the M11 export contract (ARCHITECTURE.md:660-661).

## The picks (summary)

| Rank | Take | From | Mode | Effort |
|---|---|---|---|---|
| 1 | Make every `crucible_world` record teardown-able by agentenv's own `teardown_run`, and key Worlds per run, not per `env_id` | `AE/task/teardown.py:46-153` | DEPEND | S |
| 2 | Report the reward as one verifier row (`id="reward"`, `score=reward`), gates out of `results` | `AE/task_step/task_steps/verifiers/run_container_unit_tests_verifier.py:331-362` | PATTERN | S |
| 3 | Never hand the policy the gateway root; serve one MCP-only, role-pinned URL | own (agentenv has no auth on `/tools/*`, `/clock/*`, `/triggers/*`) | own + anti-pattern | M |
| 4 | Route external tool calls through `World.step` (clock, events, trace, choke point) and check replay against the live World | own; agentenv gateway trajectory `AE/env/gateway/gateway.py:786-816` | own | M |
| 5 | TRL export: one row per input prompt, placeholder rows with reward None, `lp[0]` logprobs | `TR/trl/trainer/grpo_trainer.py:2144-2149,2685-2694`, `vllm_generation.py:566-572` | TARGET | S |
| 6 | verl export as a `@register` AgentLoop plus a reward manager that writes per-token `rm_scores`; masked rollouts as `[]`; MiMo `-999` handling | `UV/verl/experimental/agent_loop/agent_loop.py:88-115,466,487-495`, `agent_loop_tq.py:154-161`; `MV/verl/trainer/ppo/core_algos.py:304-321` | DEPEND + TARGET | M |
| 7 | Build the spec'd verl exports: MiMo `recipes/general` env dir and `recipes/code` `instance_json` (`opensource-code`) | `MV/recipes/general/general_agent/environment.py:47-90,171-259`; `MV/.../datasets/opensource_code.py:63-71,211-236` | TARGET | M |
| 8 | SkyRL export: all `GeneratorOutput` keys, real `stop_reasons`, `None` logprobs, step-wise mode for split rollouts | `SK/skyrl/train/generators/base.py:40-69`, `trainer_utils.py:680-851` | TARGET | S |
| 9 | A2A harness: own deadline, cancel the remote task, timeout as a policy limit | `AE/a2a_agent/protocol.py:145-207` (`cancel_a2a_task`) | DEPEND | S |
| 10 | Per-link provision deadline in the agentenv sandbox fallback chain | `AE/providers/sandbox_providers/chained_sandbox_provider.py:12,61` | PORT/PATTERN | S |

## 0. Corrections and confirmations of the prior studies

Confirmed still true:
- **Entry points load clean.** agentenv's own `plugins.inventory()` lists `crucible_world`, `crucible_verify`, `crucible_ref` and the explorer plugin as `active`, with no failures [ran p1]. The `typed_validator` rules hold: own `type`, entry-point name equals `type` (`AE/plugins/_registration.py:190-206`).
- **C1 (no GPU provider plugin)** holds: no `agent_env.sandbox_providers` entry point (`pyproject.toml:55-63`).
- **agentenv `eval run` reads `metadata["score"]`** (`AE/cli/eval/run.py:38,163`), not the per-verifier score. Prior G4 0.x; still true at 041afd7 (D16).

Corrections:
- 0.1 **DESIGN.md:866 "the oracle trajectory gives byte-identical ... through ... agentenv ... with the same final state fingerprint".** The agentenv gate test strips `VOLATILE_ROW_KEYS` and `harness` from the agentenv rows only (`T/test_interop_gate.py:219-222`) and never compares fingerprints. The live agentenv World and its replay have different fingerprints on the gold trajectory [ran p5] (D3).
- 0.2 **ARCHITECTURE.md:659 "`crucible_verify` step (separate verifier container)".** As built it grades the live World in-process on the host (`C/interop/agentenv/plugin/steps.py:95`). There is no container. Fine for state-graded Worlds, but the spec line is wrong.
- 0.3 **ARCHITECTURE.md:659 "artifact mirroring".** Not built. `crucible_ref` is registered but never produced (grep: only its definition), and reward files go to a `mkdtemp` (D15).
- 0.4 **DESIGN B.17 "a missing reward is an error in every runtime".** TRL's Harbor env returns the first value or `0.0` when the verifier's rewards dict is empty (`TR/trl/experimental/harbor/_env.py:182-183`) and `0.0` for a never-reset env (`:93-94`). Prior G4 0.9 already noted it. B.17 should name the exception.
- 0.5 **`C/train/bridges/adapters.py:6` says `extra_fields` carry token advantages "for a registered estimator".** verl never passes `extra_fields` to an estimator: v1 selects fixed fields (`UV/verl/trainer/ppo/v1/trainer_base.py:1745`); legacy passes `non_tensor_batch` only to GDPO (`UV/verl/trainer/ppo/ray_trainer.py:261-263`). Dead path (D9).
- 0.6 **`C/train/bridges/trl_bridge.py:6-7` "TRL treats None as unscorable".** True per row (`TR/trl/trainer/grpo_trainer.py:2685-2694,2734`), but only if the row exists. The bridge drops it (D2).
- 0.7 **verl `interactions/` does not exist at 5eb4826**, and `BaseTool.calc_reward` is never called; tool step rewards land only in `extra_fields["tool_rewards"]` (`UV/verl/experimental/agent_loop/tool_agent_loop.py:212,376,486-496`). A stateful World fits verl as a custom AgentLoop, not as a `BaseTool` (it is created and released around every call).
- 0.8 **rLLM `ToolEnvironment` is docs-only at 4e828c8.** The code has `BaseEnv` (`RL/rllm/environments/base/base_env.py:33-79`).

## 1. Contracts, by framework

### 1.1 agentenv (`AE`, `AP`; HEAD 041afd7)

| Contract | agentenv | Crucible | Match |
|---|---|---|---|
| Step registry | `agent_env.task_steps`; subclass of `TaskStep`, own `type` = entry name, abstract `execute`, `from_dict` (`AE/plugins/_registration.py:26,190-206`; `AE/task_step/task_step.py:84-153`) | `CrucibleWorldStep`, `CrucibleVerifyStep` (`C/interop/agentenv/plugin/steps.py:39,80`) | yes [ran p1] |
| `to_dict`/`from_dict`, `retry_config` | base `to_dict` adds `retry_config`; `put` hydrates it (`task_step.py:135-174`) | `**kw` forwards it; `from_dict` uses `_base_from_dict` | yes |
| `entity_refs` | undeclared (None) steps pass through bundles but env ids they name can't be rewritten (`AE/bundle/resolve.py:185-187,365-379`); `deploy_env` declares `EntityRef.env("env_id")` | none declared | no (D13) |
| Artifact registry | `agent_env.artifacts`, `Artifact` subclass with `type` Literal | `CrucibleRefArtifact` (`plugin/artifacts.py:12-16`) | yes, unused (D15) |
| Deployed env record | `DeployedGatewayEnv(DeployedSandboxEnv)`: `sandbox_id`, `sandbox_type` drive teardown, restore and URL reachability (`AE/env/env.py:141-165`) | `sandbox_type="crucible_world"`, `sandbox_id="crucible-local:<env_id>"` (`steps.py:63-67`) | no: not a registered backend (D6) |
| Teardown | `teardown_run` terminates every recorded sandbox through `build_sandbox_provider(sandbox_type)` (`AE/task/teardown.py:75-142`); `_env_sandbox_ids` skips records with empty `sandbox_id` (`AE/task_step/task_steps/teardown_sandboxes.py:143-145`) | no close path; `WORLDS` is process-global | no (D1, D6) [ran p2, p4] |
| Verifier record | `metadata["verifications"][id] = {results: [{id, score, result, weight, skipped?}], score}`; aggregators fold `results` (`AE/task_step/task_steps/verifiers/scoring.py:18-40`, `aggregate_verifiers.py:66-82`) | one row per gate, `score` = reward (`steps.py:115-120`) | shape yes, meaning no (D7) [ran p6] |
| Gateway role model | role = `AgentEnv-Role` header, default `"default"`; no auth on `/tools/enable|disable`, `/clock/*`, `/triggers/*`, `/trajectory` (`AE/env/gateway/gateway.py:161-291,296-313,740`) | ACL per actor role; gateway root published as the agent's MCP URL (`steps.py:63-67`; `harness.py:121,136`) | anti-pattern exposed (D5) [ran p3] |
| Gateway trajectory | `tool_call` events carry `function_name`, `arguments`, `source`; **no role**; logged before the call, so failed calls appear too (`gateway.py:786-816,837-841,916-922`) | `from_gateway_jsonl` replays every `tool_call` as actor `agent` (`C/interop/replay.py:159-170`) | lossy (D4) |
| Sandbox providers | `create_sandbox`/`create_container(image_name, port, env, cpu, memory, disk_size_gb, timeout, attribution, network_policy)`; `supports_network_policy` classmethod; `Attribution = dict[str,str]` (`AE/providers/sandbox_providers/sandbox_provider.py:42-84,182-191`; `AE/attribution.py:10`) | `AgentEnvSandbox._create` filters kwargs by signature (`C/interop/agentenv/sandbox.py:150-166`); uses private `_build_sandbox_provider_from_section` (`:132`) | yes; private API dependency noted |
| Fallback chain | 180 s per link (`chained_sandbox_provider.py:12,61`) | no per-link deadline (`sandbox.py:173-183`) | no (D14) |
| GPU on local provider | class attribute `LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS` (`local_sandbox.py:434`) | mutated process-wide, restored by value (`plugin/gpu_docker.py:34-52`) | fragile (D12) [ran p8] |
| A2A client | `send_a2a_message`, `poll_a2a_task(url, task_id, timeout_seconds, poll_interval_seconds)` raises `TimeoutError` out of time; `cancel_a2a_task` (`AE/a2a_agent/protocol.py:100-207`); terminal states exclude `input-required` (`:17-19`) | `A2AHarness._ask` (`C/interop/agentenv/harness.py:102-108`) | signatures yes; timeout semantics no (D8) |
| Extension routes | `/ext/mcp-config`, `/ext/agent-config` (`AP/a2a_agent/extensions.py:881-920`); data plane `data/reset|add|get` (`AP/types.py:25-27`) | hard-coded the same (`harness.py:136,141`; `mcp_env.py:41`) | yes |

Mechanisms worth taking: `teardown_run` and `TeardownReport` as the lifecycle contract (DEPEND: make records it can act on); reward-mode verifier rows (PATTERN); `cancel_a2a_task` (DEPEND); chained provider deadline (PATTERN, 10 lines). Anti-pattern: the gateway's self-declared role header and open control routes. Do not expose them to a policy.

### 1.2 verl (`UV`, 5eb4826) and the MiMo fork (`MV`, a2ad9f6)

- **Estimator registry:** `register_adv_est(name)` takes any string; config `adv_estimator: str` has no enum check (`UV/verl/trainer/ppo/core_algos.py:116-131`; `UV/verl/trainer/config/algorithm.py:734`). Called as `fn(token_level_rewards, response_mask, config, index=uid, ...)`, returns `(advantages, returns)` (`ray_trainer.py:250-281`). Crucible's `crucible_delta` matches (`C/train/bridges/verl_bridge.py:36-60`). Caveat: `use_kl_in_reward` rewrites rewards as `scores - beta*kl` (`ray_trainer.py:106`), which breaks the R-on-last-token encoding.
- **`AgentLoopOutput`** (`agent_loop.py:88-115`): `metrics: AgentLoopMetrics` is required. `reward_score` is a scalar placed at the last response token, mask or not (`:144-148`; legacy `:1110-1116`). `extra_fields` never reach an estimator. `@register(name)` + `async run(sampling_params, **kwargs)` (`:466,487-495`).
- **Masking:** v1 skips a rollout only when `run()` returns `[]` (`agent_loop_tq.py:158-161`); `None` crashes in `_compute_score` (`agent_loop.py:988-989`). No `-999` in UV.
- **MV `recipes/general`:** row needs `extra_info.instance_json`; instance has `instance_id`, `dataset_type: "general_agent"`, `docker_image`, `env_task_dir`, `cwd`, `problem_statement`|`queries` (`MV/recipes/general/general_agent/environment.py:47-57`). The env dir holds `workspace/ tools/ system/`, sidecar payload and verifier files; `manifest.json` has `uploads`, `wait_ports`, `mcp_servers`, `setup`, `verifier{command, container, reward_file, timeout_sec}` (`:82-90,171-259`). A verifier must write `{"reward": float in [0,1]}`, anything else is `testbed_corrupted` (`:574-582`). Infra categories get `INVALID_REWARD_VALUE = -999.0` (`agent_loop.py:76-87,933-942`); GRPO replaces a `-999` row sum by the group's valid mean (`MV/verl/trainer/ppo/core_algos.py:304-321`).
- **MV `recipes/code`:** `instance_json` with `dataset_type: "opensource-code"` needs `cwd, docker_image, instance_id, problem_statement, test_command, test_patch, verifier_timeout_sec`; grading resets test files, applies `test_patch`, runs `test_command`, 1.0 on rc 0 (`MV/third_party/mimoagent-osr/.../datasets/opensource_code.py:63-71,211-236`). `train.yaml` sets no `invalid_reward_value`, so infra failures there score 0.0.
- Picks: DEPEND `@register` AgentLoop + `register_adv_est`; TARGET MV general env dir and code `instance_json`; PATTERN the `-999` + valid-mean replacement (it is how MV masks without dropping rows).

### 1.3 TRL (`TR`, 72cf37d)

- **`rollout_func(prompts, trainer) -> dict`** (`grpo_trainer.py:130,2144`). Required keys `prompt_ids, completion_ids, logprobs`; `env_mask` becomes the tool mask (`:2201`); other list keys go to reward functions by row index (`:2670-2676`). `prompts` are already repeated `num_generations` times by `RepeatSampler` (`:1268-1272`); rewards are reshaped `view(-1, num_generations)` (`:2693`). So the output must have exactly one row per input prompt, in order.
- **`vllm_generation.generate(prompts, images, num_generations, profiler=None)`** returns `(prompt_ids, completion_ids, logprobs, logprob_token_ids)`, logprobs `list[list[list[float|None]]]` (`TR/trl/generation/vllm_generation.py:550-572`); TRL takes `lp[0]` (`grpo_trainer.py:1742`). It uses collectives in server mode and TP>1 colocate (`:594,690`).
- **None rewards:** a row where every reward function returns None is unscorable: NaN, out of the group mean and std, advantage 0 (`grpo_trainer.py:1591-1634,2685-2694,2734`).
- **`environment_factory`:** `reset(**row) -> str|None`, public methods as tools, optional `get_reward()` (`:134,270-275,2303-2318`). Harbor env: empty rewards → 0.0 (0.4).
- Picks: TARGET the row contract above. `environment_factory` is a second, simpler export shape (a World as an env object whose methods are its tools, `get_reward` returning None on a mask).

### 1.4 SkyRL (`SK`, 3d0ae22)

- **Env:** `skyrl_gym.Env.init/step/close`; `BaseTextEnv.step(str) -> BaseTextEnvStepOutput{observations, reward, done, metadata, postprocessed_action}`; `register(id, entry_point)` (`skyrl-gym/skyrl_gym/core.py:12-71`; `envs/base_text_env.py:9-100`; `envs/registration.py:183,233`).
- **`GeneratorOutput`** (`skyrl/train/generators/base.py:40-69`): Optional-typed keys must still be present; `rollout_metrics` is indexed unconditionally (`skyrl/train/trainer.py:1068`). Validation: `num_prompts == num_responses` unless step-wise (`trainer_utils.py:700-701`); per-row lengths equal `response_ids` (`:708-746`); step-wise needs `is_last_step` and contiguous `trajectory_ids` (`:823-851`).
- **Masking:** keyed by `stop_reasons`: `zero_reward_on_non_stop`, overlong filtering (`skyrl_gym_generator.py:1141-1194`; `generators/utils.py:396-418`). Rows can't be dropped; "reject" is an all-zero loss mask. Non-None `rollout_logprobs` drive the off-policy path (`trainer.py:925,1371,1456`).
- Picks: TARGET `GeneratorOutput` in step-wise mode for split rollouts; PATTERN stop-reason-driven masking.

### 1.5 rLLM (`RL`, 4e828c8)

- `BaseEnv.reset() -> (obs, info)`, `step(action) -> (obs, reward, done, info)`, `from_dict`, `is_multithread_safe` (`RL/rllm/environments/base/base_env.py:33-79`).
- `Step.advantage: list[float] | float | None`; a list must match `len(response_ids)` or it is zeroed with a warning (`RL/rllm/types.py:100-135`; `advantage.py:152-161`); only `broadcast` mode (`:191`). `Episode.id = "task:idx"`, trajectory names without `:` (`transform.py:116-120`). Masking: CompactFiltering by termination reason; zero-step trajectories skipped (`transform.py:124-131`).
- Crucible `rllm_steps` matches the keys and lengths (`C/train/bridges/adapters.py:50-56`). Masked rollouts give `[]`, a valid skip. Picks: TARGET only.

### 1.6 AReaL (`AR`, 298412a)

- `RolloutWorkflow.arun_episode(engine, data) -> dict | None`; `None` rejects (`AR/areal/api/workflow_api.py:14-40`). Full-sequence tensors with batch 1; `versions = -1` on prompt tokens (`AR/areal/workflow/rlvr.py:165-183`).
- Agent path: `async run(data, base_url, http_client, api_key) -> float | dict`; agent errors reject unless classified `model_failure_zero` (`AR/areal/experimental/openai/proxy/workflow.py:475-481,559-616`). Proxy routes include `rl/set_reward` (`proxy/server.py:309-317`).
- Crucible has no AReaL adapter; `TrainSeq.versions` uses `None` where AReaL wants `-1` (`C/train/bridges/common.py:72`). Picks: TARGET (`None` = reject maps 1:1 to a Crucible mask); PATTERN the `model_failure_zero` split (policy-caused model failures scored, the rest rejected) matches Crucible's blame rule.

### 1.7 slime (`SL`), ROLL (`RO`), OpenRLHF (`OR`), checkpoint-engine (`CE`)

- **slime:** `Sample{tokens, response_length, loss_mask, reward, status, rollout_log_probs, remove_sample, rollout_id:int}` (`SL/slime/utils/types.py:107-170`); `remove_sample` zeroes the loss mask but still counts in normalization (`SL/slime/data/batch_builder.py:282-283`; `arguments.py:1538-1546`); ABORTED is recycled (`sglang_rollout.py:462-468`). `custom_rm` must be async despite the help text (`rm_hub/__init__.py:61-63`). TARGET.
- **ROLL:** GEM `reset(seed) -> (obs, info)`, `step(action) -> (obs, r, terminated, truncated, info)` (`RO/roll/pipeline/agentic/env/cli_env/env.py:77-188`). A failed reset gives a zero-mask placeholder with `drop_flag`; `GroupFilter` drops the group, capped at 50 % (`env_manager/agent_native_env_manager.py:48-52,433-453,499-520`). PATTERN: the drop cap.
- **OpenRLHF:** `AgentInstanceBase.reset/step` returning `rewards` (tensor), `environment_feedback` (string, re-tokenized), `done` (`OR/openrlhf/utils/agent.py:13-184`). No mask channel; exceptions propagate (`vllm_engine.py:194-206`). nothing (string observations break exact token-in/token-out).
- **checkpoint-engine:** weight sync only (`CE/checkpoint_engine/ps.py:176`). nothing for M11.

Common shape. An OpenAI-compatible chat endpoint with token capture plus a per-episode reward call fits AReaL, ROLL's proxy, slime's OpenAI adapter and SkyRL's inference servers. Crucible's `EngineLLM` is already that shape (`C/train/world_env.py:42-112`). One HTTP export of it would cover four trainers.

## 2. Crucible defects

| # | Sev | Where | Defect | Probe | Fix |
|---|---|---|---|---|---|
| D1 | P0 | `C/interop/agentenv/plugin/steps.py:27,60,93` | `WORLDS` is keyed by `env_id` alone. agentenv runs `eval run --k` and runner workers concurrently in one process (`AE/cli/eval/run.py:139-148`; `AE/runner/local_runner.py:42,160`), each run with the same `env_id`. Run 2's `crucible_world` overwrites run 1's entry, so run 1 is graded on run 2's World. Run 1 solved its task and scored 0.0; its World leaked. | [ran p2] | Key by a per-run handle: a uuid stored in the record (`DeployedEnv.instance_id` or `metadata`), looked up by the verify step from its own context. Refuse to overwrite. |
| D2 | P0 | `C/train/bridges/trl_bridge.py:24-25,36,44-50` | TRL `rollout_func` breaks three ways. (a) It loops `num_generations` per prompt, but TRL already repeats prompts, so it returns N× rows. (b) Masked rows are dropped from `prompt_ids`/`completion_ids` while `crucible_reward` keeps all rows: rewards shift onto other completions (completion 2 got reward None, completion 3 got 2.0). (c) `trl_gen` does `float(x)` on TRL's per-token logprob lists: TypeError. Concurrent `generate` calls also break TRL's collectives in server or TP>1 mode. | [ran p7]; contract [read] `TR/grpo_trainer.py:1268-1272,2144-2149,2693`; `vllm_generation.py:566-572` | One rollout per input prompt. Emit a placeholder row for a mask (one token, `env_mask=[0]`, reward None). Take `lp[0]`. Serialize `generate` (one lock) or batch it. |
| D3 | P1 | `C/interop/agentenv/world_backend.py:208-216`; `steps.py:63-67`; `harness.py:121,136` | An external agent's tool calls go gateway → App MCP server → `App.call`. They never pass `World.step`. So in the agentenv export and the A2A harness: no trace events (0 of 8 calls), no step clock, no World events (`month_end_freeze` never revokes; `refund_audit` never mails), no choke-point infra classification. The agentenv task is easier than the native one, and the A2A rollout's trace has no actions for M4/M6. | [ran p5] | Make the App caller for non-system roles call a World entry point that does what `step` does (log, clock, events, choke point). Or always delegate events to the gateway in this path. Gate test: compare live vs replay fingerprints. |
| D4 | P1 | `steps.py:95-112`; `C/interop/replay.py:159-170` | `crucible_verify` takes the reward from the live World and `steps.jsonl` from a replay, and never checks they describe the same episode (the replay's reward and fingerprint are discarded). The gateway trajectory has no role field, so every actor's calls, failed calls (logged before the call) and gateway tools (`get_time`) are replayed as `agent`. | [ran p5] (fingerprints differ on the gold run) | Compare `replay()`'s reward and fingerprint to the live grade; mask `crucible:replay_divergence` on mismatch. Source the action list from Crucible's own App-caller log (it knows the signed actor and the result), not the gateway file. |
| D5 | P1 | `steps.py:63-67`; `harness.py:121,136`; `world_backend.py:191-198` | The policy gets the gateway root. Its control routes have no auth, and the role is a self-declared header. After `modify_env_tool_access` revoked `issue_refund`, the agent re-enabled it with `POST /tools/enable` and refunded (200). It can also `PUT /clock/set-time`, register triggers and read `/trajectory`. The native path blocks the gateway port from the policy for this reason (`world_backend.py:204-206`). | [ran p3] | Serve the agent a Crucible proxy that forwards only `/mcp` and injects (and strips any client) `AgentEnv-Role`. Keep the gateway on loopback. Upstream ask: a token on agentenv's admin routes. |
| D6 | P1 | `steps.py:63-67` | Nothing closes a `crucible_world`. `teardown_run` calls `build_sandbox_provider("crucible_world")` and fails ("Unknown sandbox backend"); the gateway process, App servers, keepalive task and tempdir stay. A step retry or `retry_from_step_id` builds a new World and leaks the old one. A resumed run in a new process hits `KeyError` in `WORLDS`. | [ran p4] | Record `sandbox_id=""` (agentenv then skips it, `teardown_sandboxes.py:143-145`) and close the World in `crucible_verify` by default, plus a `crucible_teardown` step for runs that fail before verify. Or register a tiny `crucible_world` sandbox provider whose `terminate()` closes the World. Fail resume with a typed "World not resumable" error. |
| D7 | P1 | `steps.py:113-120` | `results` holds one row per gate. agentenv's `aggregate_verifiers` (and any row-folding reader) then scores the gate mean, not the gated reward. Partial episode: Crucible reward 0.0; `aggregate_verifiers` weighted_average 1.0 and all_pass 1.0. | [ran p6] | One row `{"id": "reward", "score": reward, "result": reward >= 1.0, "weight": 1.0}`, as agentenv's own reward-mode verifier does; gates move to `crucible.gates`. |
| D8 | P1 | `C/interop/agentenv/harness.py:102-108`; `C/runtime/infra.py:84-85` | An agent that overruns `timeout_s`, or parks in A2A `input-required` (not terminal, `AE/a2a_agent/protocol.py:17-19`), makes `poll_a2a_task` raise `TimeoutError`. `boundary` maps it to `transport_timeout`, blame infra: masked and retried. A policy's own overrun becomes a mask (survivorship). The remote task is not cancelled on a harness-side exception. | [read] | Own deadline in the harness; on expiry `cancel_a2a_task` and raise the limit failure the native harness uses (`agent_timeout` through `LimitPolicy`). `input-required` is a policy outcome. |
| D9 | P1 | `C/train/bridges/adapters.py:23-26`; `verl_bridge.py:23-55` | verl export: `verl_agent_loop_output` lacks the required `metrics`, so `AgentLoopOutput(**d)` fails. The per-token encoding (`token_rewards`) has no way into `rm_scores`: `reward_score` is one scalar at the last token and `extra_fields` never reach the estimator (0.5). With a frozen actor's call last, verl writes R at index -1 while `crucible_delta` reads the last sampled token: R is lost. On MV, `crucible_delta` ignores `invalid_reward_value` and the c_k tokens change the row sum away from -999. No Crucible AgentLoop or reward manager exists. | [read] | A `@register("crucible")` AgentLoop that runs `world_rollout` and returns `AgentLoopOutput` with `metrics`, a reward manager that writes per-token `rm_scores`, `[]` for a mask (v1), and `-999` on MV with the estimator honouring `config.invalid_reward_value`. |
| D10 | P1 | `adapters.py:29-41`; `common.py:51-53,63-65,78-79` | SkyRL export: no `rollout_metrics` (KeyError at `SK/trainer.py:1068`), no `trajectory_ids`/`is_last_step`; `stop_reasons` hard-coded `"stop"` (disables SkyRL's truncation masking); zero-filled logprobs instead of `None` (fake behaviour logprobs); masked rollouts dropped and multi-segment rollouts split, which breaks `num_prompts == num_responses`. Every segment carries the full reward. | [read] (sub-sweep; `trainer.py:1068` and `trainer_utils.py:700-701` re-opened) | Emit all keys; carry the finish reason into `TrainSeq`; `None` logprobs when missing; zero-loss-mask placeholder for a mask; step-wise mode with `trajectory_ids` for splits, reward on the last step only. |
| D11 | P1 | `C/studio/export.py:18`; `C/interop/mimo/export.py:14-24` | Spec gap. ARCHITECTURE.md:661 promises Worlds exported in MiMo format for verl `recipes/general` and coding tasks as `recipes/code` rows. Only the byte-identical re-export of an imported MiMo env exists. `crucible export` targets are `canonical, atif, nemo-gym, openenv`: no verl, MiMo, agentenv or verifiers target. | [read] | MiMo env-dir writer (manifest, sidecar payload, `run_verify.py` that calls the single reward producer and writes `{"reward"}`, or nothing on a mask, which MV reports as `testbed_corrupted`); `instance_json` writer for `opensource-code`. Add the targets to `crucible export`. |
| D12 | P2 | `C/interop/agentenv/plugin/gpu_docker.py:34-52` | GPU flags mutate a process-wide class attribute and are restored by value. Two overlapping GPU runs that end out of order strand flags: after both ended the GPU flags stayed on for every later container, including CPU runs. | [ran p8] | Refcounted, set-based composition (each run adds a token; the attribute is rebuilt from live tokens). |
| D13 | P2 | `steps.py:39,80` | No `entity_refs`. In an agentenv bundle the `env_id` the World step outputs can't be rewritten, and stock steps naming it are flagged (`AE/bundle/resolve.py:365-379`). No `preflight()` (task spec not importable is found only at run time). | [read] | `entity_refs = (EntityRef.env("env_id", role=RefRole.OUTPUT),)` on the World step, `(EntityRef.env("env_id"),)` on verify; `preflight` imports `task`. |
| D14 | P2 | `C/interop/agentenv/sandbox.py:173-183` | The fallback chain has no per-link deadline (agentenv: 180 s, `chained_sandbox_provider.py:12,61`). A hung provider stalls the World forever; the next link is never tried. | [read] | `asyncio.timeout(180)` around `_create`, configurable. |
| D15 | P2 | `plugin/artifacts.py`; `steps.py:54,97,120` | Artifact mirroring is not built. `crucible_ref` is never produced. `reward.json`, `steps.jsonl` and the grade live in a `mkdtemp` that nothing cleans, outside agentenv's output dir; `out_dir` in the record points there. | [read] | Write into the run's output dir and publish a `crucible_ref` artifact per grade. |
| D16 | P2 | `steps.py:117-120` | agentenv `eval run` prints `metadata["score"]` (`AE/cli/eval/run.py:38,163`); `crucible_verify` writes only `verifications`, so evals show no score. | [read] | Also set `context.metadata["score"]` (None on a mask). |
| D17 | P2 | `C/train/bridges/common.py:72` | No AReaL, slime, ROLL or OpenRLHF adapter; `versions` are `None` on observation tokens where AReaL wants `-1`; truncation (`limit_policy`) is not exported as `TRUNCATED` / `is_truncated` / `"length"` (`C/train/world_env.py:181-184`). | [read] | One OpenAI-compatible export with a reward call (section 1.7) covers four of them. |

Not defects (checked): plugin registration and validators [ran p1]; agentenv sandbox kwargs and network-policy refusal (`sandbox.py:150-176`); A2A extension routes and data-plane method names; rLLM `Step` mapping; verl estimator signature and string registration.

## 3. Picks table

| Framework | Take | Mode | For |
|---|---|---|---|
| agentenv | `teardown_run` / `TeardownReport` as the lifecycle contract of every record we create | DEPEND | D1, D6 |
| agentenv | Reward-mode verifier row (`run_container_unit_tests_verifier.py:331-362`) | PATTERN | D7 |
| agentenv | `entity_refs` declarations, `preflight` | DEPEND | D13 |
| agentenv | `cancel_a2a_task`, poll deadline | DEPEND | D8 |
| agentenv | Chained-provider per-link deadline (`chained_sandbox_provider.py:12,61`, Apache-2.0) | PATTERN | D14 |
| agentenv | Gateway control plane and header roles | anti-pattern (never expose) | D5 |
| verl | `@register` AgentLoop, `register_adv_est`, `[]` = masked | DEPEND, TARGET | D9 |
| MiMo verl | `recipes/general` env dir + manifest + `run_verify`; `recipes/code` `instance_json`; `-999` + group valid mean | TARGET, PATTERN | D9, D11 |
| TRL | `rollout_func` one row per prompt, None = unscorable; `environment_factory` env object | TARGET | D2 |
| TRL | Harbor env reward 0.0 on empty | anti-pattern | 0.4 |
| SkyRL | `GeneratorOutput` step-wise mode; stop-reason masking | TARGET, PATTERN | D10 |
| AReaL | `arun_episode` None = reject; `model_failure_zero` split; proxy `rl/set_reward` | TARGET, PATTERN | D17 |
| slime | `Sample.status`, `remove_sample` | TARGET | D17 |
| ROLL | `drop_flag` group filter with a 50 % cap | PATTERN | D17 |
| rLLM | `Step.advantage` list, CompactFiltering | TARGET | none open |
| OpenRLHF | string observations, no mask channel | nothing | |
| checkpoint-engine | no env interface | nothing | |

## 4. Probe index (`/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m11sweep/g4/`)

All run with `crucible/.venv/bin/python` from `crucible/`; local processes only.

| Probe | Shows | Result |
|---|---|---|
| `p1_inventory.py` | agentenv's inventory of the crucible dist | 4 contributions `active`, `load_failures() == {}` |
| `p2_worlds_collision.py` | two runs, same `env_id` | solved run 1 scored 0.0, replayed 0 actions; its World leaked (D1) |
| `p3_gateway_admin.py` | agent-held gateway URL | revoked refund 403 → `POST /tools/enable` 200 → refund 200; clock set 200; trajectory 200 (D5) |
| `p4_teardown.py` | `teardown_run` on a `crucible_world` context | failed "Unknown sandbox backend 'crucible_world'"; gateway alive; tempdir kept (D6) |
| `p5_bypass.py` | gold trajectory via the gateway | 0 of 8 calls in the World trace; clock unmoved; no audit mails; live fingerprint ≠ replay fingerprint (D3, D4) |
| `p6_aggregate.py` | `aggregate_verifiers` over crucible rows | reward 0.0 vs aggregate 1.0 (weighted_average and all_pass) (D7) |
| `p7_trl_align.py` | `make_rollout_func`, one masked of four | 3 completion rows vs 4 rewards, shifted (D2) |
| `p8_gpu_flags.py` | overlapping `enable/disable_local_gpu` | run B's flags lost while running; GPU flags stranded after both end (D12) |
