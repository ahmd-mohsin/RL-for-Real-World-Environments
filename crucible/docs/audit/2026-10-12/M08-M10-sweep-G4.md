# M8-M10 sweep, group 4: agentenv and the RL trainer infrastructure vs Crucible M8 `forge`, M9 `accept`, M10 `runtime` (2026-10-12)

Read-only. Crucible HEAD `526fdb5`; `C/forge/orchestrator.py`, `C/forge/budget.py`, `C/interop/harbor/role_runner.py` and `C/accept/contract.py` were being edited (uncommitted) by the concurrent M8 audit during this sweep, so citations into them are to HEAD. Roots:
- `C` = `crucible/src/crucible`;
- `AE` = `agentenv-framework/src/agent_env`, `AP` = `agentenv-framework/packages/agentenv-protocol/src/agentenv_protocol` (041afd7, Apache-2.0 with NOTICE "agent-env, Copyright 2026 Scale AI");
- `REF` = `RL-Environments/06-frameworks-and-env-code`: `UV` verl (5eb4826), `SK` SkyRL (3d0ae22), `RL` rLLM (4e828c8), `AR` AReaL (298412a), `RO` ROLL (581046a), `SL` slime (c933787), `OR` OpenRLHF (dc2a7ad), `TR` TRL (72cf37d), all Apache-2.0; `CE` checkpoint-engine (84631a2, MIT).

Verdicts: **DEPEND** (call their code), **PORT** (copy; every source here is Apache-2.0 or MIT, keep headers and NOTICE), **PATTERN** (re-implement the idea), **nothing** (Crucible equal or ahead), **anti-pattern** (do not copy).

How this was done. agentenv and Crucible were read directly. The nine trainer repos were read by two read-only sub-sweeps; I re-opened a sample of their citations (verl `replay_buffer.py:296-312`, `agent_loop.py:664-671`; rLLM `workflow.py:80-105`, `eval/curation.py:88-130`, `sandbox/warm_queue.py:128-175`; SkyRL `trainer_utils.py:650-677`; slime `dynamic_sampling_filters.py:9-23`, `base_types.py:14-24`; ROLL `generate_scheduler.py:636-641`, `user_defined_rollout_loop.py:29-36`; AReaL `reward_api.py:155-162`, `remote_inf_engine.py:195-232`) and all matched. Six probes ran against Crucible (`scratchpad/m810sweep/g4/`, §4).

## The picks (summary)

| Module | Rank | Take | From | Effort |
|---|---|---|---|---|
| M10 | 1 | Contain every attempt: exception wrapper, per-attempt deadline, settle, keep going | agentenv `_run_safe` + `teardown_run`; rLLM `run_with_termination_handling` | S |
| M10 | 2 | Same containment for the trainer's rollout paths (token env, `world_rollout`) with a failure kind on every mask | rLLM `workflow.py:80-105` | S |
| M10 | 3 | Durable crash resume of long episodes: keep the newest checkpoint across aborted attempts | own (D4) + slime buffer checkpoint | S |
| M10 | 4 | Sandbox hygiene: liveness check on hand-out, start-failure accounting, TTL, orphan sweep by label, teardown report | rLLM WarmQueue; ROLL `auto_clear_seconds`; agentenv `teardown_run` | S–M |
| M10 | 5 | Retry rules with backoff and an idempotency flag | slime E2B classifier; ROLL reset backoff | S |
| M10 | 6 | Group survival rule: `min_usable_group_size` and up-front redundancy | AReaL `GroupedRolloutWorkflow`; ROLL `group_size_redundancy` | S |
| M9 | 1 | One bench band (Wilson upper bound) for A10 and the ladder | own (D15) | S |
| M9 | 2 | World round-trip admission check (A19) | agentenv universe round trip | M |
| M9 | 3 | Preflight + dry run before any paid or GPU check | agentenv bundle plan/preflight | S |
| M9 | 4 | Tool-surface conformance of forged Worlds (spec as a floor) | agentenv `verify_spec_conformance` | S |
| M9 | 5 | Wire the flaky-task screen into admission of training sources | own `train/admission.screen` (MiMo 8 reruns) | S |
| M8 | 1 | Per-prompt pass-rate history fed back from training; liveness-aware zero-variance filter with fallback and bounded refill | SkyRL `trainer_utils.py:637-677`; slime `_with_fallback`; verl bounded refill; SkyRL filtered-UID resume | M |
| M8 | 2 | Masked-aware outcome adapter for learnability and the ladder | own (D18, D19); `C/studio/datasets.py` | S |
| M8 | 3 | Forge node robustness and retry-from-ancestor | agentenv `RetryConfig` + span rollback | M |

## 0. Corrections and confirmations of the prior studies

Confirmed still open:
- **Harness exceptions are not contained** (`docs/research/improvements/M09-M11.md:38-39`; `C/runtime/run.py:139,159`) [ran p1a]. Worse than stated: the other in-flight samples are orphaned and no row reaches `results.jsonl`.
- **No per-attempt deadline** (`M09-M11.md:40`; `C/runtime/run.py:103-120`) [ran p1d].
- **Semaphore held across retries, no backoff** (`M09-M11.md:41`; `C/runtime/run.py:118-157`).
- **`over_uncertain_tolerance` and `release` unused** (`M09-M11.md:49`; `C/runtime/budget.py:61-74`). Also `Failure.delivery` is never read (`C/core/schema/run.py:110`; grep).
- **agentenv chained failover** (`M09-M11.md:66`): confirmed at `AE/providers/sandbox_providers/chained_sandbox_provider.py:50-73`, plus a 180 s provision deadline per link (`:12,61`) and network-policy filtering (`:27-45`).
- **`eval run` score readout is broken** (`docs/research/m9-m10/02-runtime.md:305`): `AE/cli/eval/run.py:38,163` read `metadata["score"]`; scores are written under `metadata["verifications"][id]["score"]` (`AE/task_step/task_steps/verifiers/aggregate_verifiers.py:79-84`).
- **`error_type` flattened at the `prompt_agent` step** (`02-runtime.md:336`): `AE/task_step/task_steps/prompt_agent.py:697-718`. Crucible bypasses it (`C/interop/agentenv/harness.py:105-106,157-162`).
- **Resume only by hand in agentenv** (`02-runtime.md:321`): `AE/cli/task/run.py:415-416,435`.
- **Curriculum and the dynamic-sampling fallback** (`docs/research/improvements/M12-train.md:26-27`): no per-prompt history (grep), no fallback (`C/train/scheduler.py:172-178`).
- **`overlong_penalty` never called** (`M12-train.md:35`): only the definition (`C/train/objectives.py:136`).
- **verl `max_num_gen_batches` ignored by v1** (`docs/research/m11-m12/01-verl-slime.md:345`): `UV/verl/trainer/ppo/v1/trainer_base.py:205-211`; and the DAPO doc still describes it as enforced (`UV/docs/algo/dapo.md:78-108`).
- **slime zero-std filter** (`01-verl-slime.md:549`): `SL/slime/rollout/filter_hub/dynamic_sampling_filters.py:9-23`. The `_with_fallback` variant (`:18-23`, `base_types.py:14-24`) is exactly the fallback `M12-train.md:27` asks for.
- **AReaL rejected rollouts free their slot** (`docs/research/m11-m12/02-areal-roll.md:159`): `AR/areal/infra/staleness_manager.py:159-166`.

Corrections:
- 0.1 **`docs/research/m12-infra/00-option-matrix.md:91` marks the DAPO overlong penalty "Have"**: it is defined and never called (above).
- 0.2 **`00-option-matrix.md:117` marks the flaky-task filter "Have (D)" for training sources**: `train/admission.screen` has no caller in `src/` or `packs/` (grep `admission import|screen(`). It is a library, not a gate.
- 0.3 **`00-option-matrix.md:38` lists the circuit breaker as "Have"**: the token-env path bypasses it; a `ValueError` in `env.step` leaves `failures={}` and hangs the group [ran p2].
- 0.4 **`02-runtime.md:928` describes a "label-based cleanup sweep `crucible.sandbox=1`"**: the labels exist, the sweep does not (D6).
- 0.5 **DESIGN.md:2600 "Crashed on attempt 1, it resumed from the step-3000 checkpoint"**: the gate's crash is an in-process `ConnectionError` that writes a masked record (`packs/_toy/src/crucible_pack_toy/longhorizon/gate.py:37`). A killed process does not resume [ran p5] (D4).
- 0.6 **`02-runtime.md` predates agentenv's second runner.** `AE/bundle/` (plan, preflight, materialize, ledger, run; 4,008 lines) is not mentioned (grep `bundle`: 0 hits). Its outcome taxonomy keeps a raised run FAILED, never "scored 0" (`AE/bundle/run.py:45-97`), and its plan/preflight/ledger are M9 patterns (§1.1).
- 0.7 **The A2A blame rule is weaker than `harness.py` assumes.** `02-runtime.md:342` recommends reading `TaskError.error_type` from the wire, which Crucible does. But the protocol defaults `error_type` to `agent_error` (`AP/a2a_agent/tasks/v1.py:165`); only unhandled framework exceptions become `infra_error` (`AP/a2a_agent/framework.py:1166-1179`). An agent that reports its upstream outage as a plain failure is scored 0 (D7).
- 0.8 **ARCHITECTURE.md:223 (and M8 "Takes") list agentenv's review step as DEPEND for "M8 (10% human audit)" and M9 sign-off.** No Crucible code uses it (grep `review` in `C/interop/agentenv`, `C/accept`: none relevant). agentenv's review has no sampling rate: it blocks one run instance until continue/abort, with a 3 h timeout (`AE/task_step/task_steps/review.py:22,95-129`). S1 is Crucible's own and stricter: bound to the acceptance key (`C/accept/checks/extended.py:275-287`). The 10% sampling has to be built (a `sample_for_review(rate, seed)` in `C/forge/curator.py`), or the claim removed.
- 0.9 **TRL path:** `trl/experimental/openenv` does not exist at 72cf37d; the harness is `TR/trl/experimental/async_grpo/openenv_harness.py`. The prior TRL Harbor DEPEND (`M09-M11.md:78`) needs one caveat: `TR/trl/experimental/harbor/_env.py:183-190` turns empty verifier rewards into 0.0, so a Crucible mask must reach TRL as an explicit `None` reward, not as "no reward file".
- 0.10 **rLLM has no `AgentExecutionEngine` at 4e828c8** (only "legacy" mentions, `RL/cookbooks/frozenlake/README.md:3,133`); the live engines are `AgentWorkflowEngine`, `UnifiedWorkflowEngine`, `AgentFlowEngine`, `RemoteAgentFlowEngine`. CompactFiltering is **off by default** (`RL/rllm/trainer/config/rllm/base.yaml:112-120`) and never applied on the legacy verl trainer path (`RL/rllm/trainer/verl/agent_workflow_trainer.py:440-443`), although `RL/docs/backends/verl.mdx:329-341` says it is. The prior group-4 table (`docs/audit/2026-10-12/M05-M07-study-sweep.md`, section 1f, CompactFiltering row) does not say it is off by default or missing on that path.

## 1. Mechanism tables

### 1.1 agentenv (`AE` = `agentenv-framework/src/agent_env`, `AP` = `agentenv-framework/packages/agentenv-protocol/src/agentenv_protocol`; HEAD 041afd7; Apache-2.0 + NOTICE)

#### M10 runtime

| Mechanism | Source file:line | Crucible equivalent file:line | Verdict |
|---|---|---|---|
| `eval run --k`: tasks × k coroutines under an optional semaphore; each run wrapped so an exception comes back with its task id instead of killing the batch (`_run_safe`) | `AE/cli/eval/run.py:118-148` | `C/runtime/run.py:116-159`: no wrapper; one harness exception aborts `asyncio.gather` [ran p1] | PATTERN (pick M10-1) |
| `eval run` result readout: reads `context.metadata["score"]`, which nothing writes (scores live under `metadata["verifications"][id]["score"]`); any run that did not raise prints PASSED, even at score 0; no seeds, no pass@k, no retry, no resume, output file named by a random uuid | `AE/cli/eval/run.py:38,49,163-170`; writer `AE/task_step/task_steps/verifiers/aggregate_verifiers.py:79-84` | `C/runtime/run.py:79-88` (SampleKey, derived seeds), `C/runtime/infra.py:119-144` (masked-aware aggregates) | nothing (anti-pattern; Crucible ahead) |
| Bundle runner outcome taxonomy: PASSED / BELOW_ONE / UNSCORED / FAILED / CANCELLED; a raised run is FAILED, never "scored 0"; failures a retry recovered are not counted (`failed_steps` minus `retried`) | `AE/bundle/run.py:45-97` | `C/core/schema/run.py:126-139` (blame → masked or scored) | nothing (Crucible's blame split is finer) |
| `teardown_run`: terminate every sandbox the run's context recorded, all at once, each under a 120 s timeout; best effort; returns `TeardownReport(terminated, failed, left)` with `still_up` | `AE/task/teardown.py:27,36-50,75-118` | `C/runtime/harness/native.py:291-297`, `workspace.py:136-141`: `world.close()` errors swallowed (`except Exception: pass`), nothing records what is still up | PATTERN (pick M10-4) |
| `Interrupts`: first SIGINT/SIGTERM cancels runs not yet tearing down, a second cancels the teardown too; runs mark themselves `tearing_down()` | `AE/task/interrupts.py:25-29,64-77` | none (Runner has no signal handling; a Ctrl-C leaves reservations `reserved`, `C/runtime/budget.py:34` turns them `uncertain` on reopen) | PATTERN-low |
| LocalRunner orphan reconciliation: at start, every run a dead process left QUEUED/RUNNING is marked FAILED "interrupted by a hub restart"; terminal states are never overwritten | `AE/runner/local_runner.py:51-60`; `AE/runner/store.py:20-22,83-94` | `C/runtime/journal.py:98-108` moves record-less attempt dirs to `aborted/`; no process lock (prior M09-M11.md:46) | PATTERN-low |
| Local sandbox cleanup by label: containers labelled with the sandbox id are removed with `docker rm -f` on terminate; process trees are SIGKILLed via `ps` and reaped | `AE/providers/sandbox_providers/local_sandbox.py:229-260,320-345` | labels set (`C/verify/stage.py:111`, `C/world/sandbox.py:87`, `C/interop/agentenv/mcp_env.py:124`) but no sweep lists or removes them (grep `crucible.sandbox` outside `--label`: none) | PATTERN (pick M10-4) |
| `ChainedSandboxProvider`: providers tried in order, each under a 180 s provision deadline; providers that cannot enforce the network policy are skipped; all failures reported together | `AE/providers/sandbox_providers/chained_sandbox_provider.py:12,27-73` | `C/interop/agentenv/sandbox.py:132-169` (one provider) | PATTERN (prior M09-M11.md:66 confirmed) |
| Step retry with span rollback: `RetryConfig(retry_from_step_id, max_retries)` must name an ancestor; the span (+ dependents + cancelled siblings) is undone by replaying the journal over the seed in one CAS; refuses when a surviving step overlapped the span; audit in `retry_resets`, recovered failures flagged `retried` | `AE/task_step/task_step.py:52-81`; `AE/task/task.py:166-275,317-340,640-700`; `AE/task/step_journal.py:75-95` | attempt-level reuse only: re-collect (`C/runtime/harness/workspace.py:35-47`), re-grade (`:49-61`), checkpoint resume (`C/runtime/harness/native.py:106-120`) | PATTERN-low (M10 phases are fixed; useful for M8, pick M8-3) |
| Attempt-epoch fence: a completion written under an older epoch than the stored one no-ops | `AE/task/store.py:183-186,422-431,632-690` | `C/runtime/journal.py:91-96` (newest terminal row wins, no fence); `C/train/runtime.py:400-405` cancels late group stages | PATTERN-low (a sample's attempts are sequential today; needed only if regrade runs beside a live run) |
| Crash resume of a task run: only by hand (`--start-step` + `--context-json`); the journal serves in-run rollback | `AE/cli/task/run.py:415-416,435` | `C/runtime/run.py:90-98` resumes by SampleKey automatically | nothing (Crucible ahead; confirms 02-runtime.md:321) |
| A2A terminal parsing: `TerminalResponse.error_type/code/class/message`, `tool_call_count`; `usage` read only for the tool-call count | `AE/a2a_agent/protocol.py:228-265` | `C/interop/agentenv/harness.py:105-106,157-163` polls the wire directly and reads `error_type` | DEPEND (in use) |
| `prompt_agent` step flattens `error_type` into `RuntimeError` text | `AE/task_step/task_steps/prompt_agent.py:697-718` | bypassed (above) | nothing (confirms 02-runtime.md:336) |
| Protocol default: `TaskError.error_type="agent_error"` unless the agent says otherwise; only an unhandled framework exception becomes `infra_error` | `AP/a2a_agent/tasks/v1.py:162-170`; `AP/a2a_agent/framework.py:1166-1179` | `C/interop/agentenv/harness.py:157-162` scores every `agent_error` as policy 0 | caution (defect D7) |
| Review step: registers `awaiting`, polls every 5 s, heartbeats, 3 h timeout raises `TimeoutError`; continue/abort with `decided_by`; no sampling rate | `AE/task_step/task_steps/review.py:22-23,95-129,131-145`; `AE/task_step/review_store.py:17-25` | S1 sign-off bound to the acceptance key (`C/accept/checks/extended.py:275-287`) | nothing (ARCHITECTURE "10% human audit" is not in agentenv; correction 0.8) |

#### M9 accept

| Mechanism | Source file:line | Crucible equivalent file:line | Verdict |
|---|---|---|---|
| `aggregate_gate`: required gates; a gate with no verdict **fails** (fail-closed); a gate that reported `skipped` is advisory; reports failed / missing / skipped lists | `AE/task_step/task_steps/mcp_env_validator/validation_gate_aggregator.py:1-49`; test `tst/unit/task_step/test_validation_gate_aggregator.py` | `C/accept/run.py:49-66` (N/A from a closed vocabulary, missing capability = fail, unavailable need = pending) | nothing (equal) |
| Spec conformance as a floor: the baked OpenAPI spec vs live `list_tools()`; extra tools/params reported, an undeclared *required* param fails; record-only unless opted in | `AE/task_step/task_steps/mcp_env_validator/verify_spec_conformance.py:1-15` | none for forged World tool surfaces (A7 covers instruction↔test only, `C/accept/checks/core.py:225-241`) | PATTERN (pick M9-4) |
| Universe round trip: load the universe, export (1), reload export 1, export (2); compare original↔export1 (load fidelity) and export1↔export2 (idempotency); count/value mismatches and non-empty dropped fields are critical | `AE/task_step/task_steps/multienv_validator/verify_universe_roundtrip.py:97-151`; `universe_comparison.py:131-242` | per-restore fingerprint checks at run time (`C/world/state/fs.py:115-116`, `json_doc.py:64-66`, `postgres.py:129-131`); A9 checks only that two fresh starts agree (`C/accept/checks/extended.py:76-80`) | PATTERN (pick M9-2) |
| Single authoritative writer of a combined verdict, written last, so a crash leaves no half-written result | `AE/task_step/task_steps/multienv_validator/combine_universe_verdicts.py:1-8` | `C/accept/run.py:67-71` writes evidence per check; `accepted` derived on read | nothing |
| Bundle plan/preflight/dry run: everything knowable before a write is reported together (id collisions, reference loops, missing store ids, tasks whose steps don't build); each deploy checked against its provider (local-only image on a remote provider, VM on a provider that can't, no Docker daemon) | `AE/bundle/plan.py:1-8,98`; `AE/bundle/preflight.py:1-14,83`; `AE/bundle/run.py:232-257` | `C/accept/plan.py:12-32` orders by cost; needs checked per check at run time (`C/accept/run.py:54-55`) | PATTERN (pick M9-3) |
| Bundle ledger: a digest per input; a rewrite says why ("files changed: server.py", "config changed", "type changed") | `AE/bundle/ledger.py:1-8,76-112,224-241` | `C/accept/key.py:47-52` (structured key, no "why re-run" diff) | PATTERN-low |
| Validators for agents/envs: A2A card, core protocol, MCP, install, snapshot, trajectory; env card, env core protocol (`reset/add/get`) | `AE/task_step/task_steps/a2a_agent_validator/*.py`; `env_card_validator/verify_env_core_protocol.py:1-20` | `C/interop/agentenv/plugin/steps.py` exports Worlds behind a real gateway | DEPEND (as an M11 conformance test of the export, not M9) |

#### M8 forge

| Mechanism | Source file:line | Crucible equivalent file:line | Verdict |
|---|---|---|---|
| `mcp_cli_builder`: generate a CLI from a deployed server's manifest and live MCP tools | `AE/task_step/task_steps/mcp_cli_builder/build_mcp_cli.py:1`, `codegen.py` (627 lines) | none | nothing (tooling, not task synthesis) |
| Seed substitution into every step field (`<key>` in any string leaf), into a local copy so `--k` runs never share a rewrite | `AE/task/task.py:91-119,430-447` | `C/forge/orchestrator.py:60-73` (knobs into the input hash) | nothing |
| Retry from a declared ancestor (see M10 row) | `AE/task_step/task_step.py:52-81`; `AE/task/task.py:166-275` | `C/forge/orchestrator.py:313-363` retries the same node only | PATTERN (pick M8-3) |
| Task synthesis, difficulty, dedup, learnability | — (grep `synthes / difficulty / learnab / dedup / generate_task` in `AE/src`: only unrelated hits in document-store update code and list dedupe) | `C/forge/*` | absent in agentenv |

### 1.2 verl (`UV` = `REF/infra-volcengine-verl`, 5eb4826, Apache-2.0)

`UV/recipe/` is an empty submodule at this HEAD (`.gitmodules`), so the DAPO recipe is not present; the live DAPO path is the v1 replay buffer.

| Mod | Mechanism | Source file:line | Crucible equivalent file:line | Verdict |
|---|---|---|---|---|
| M10 | Agent loop manager: `num_workers` Ray actors (default 8), batch split per worker, joined by a plain `asyncio.gather` | `UV/verl/experimental/agent_loop/agent_loop.py:1226-1249,1262-1270`; `trainer/config/rollout/rollout.yaml:258` | `C/train/runtime.py:525-536` (one task per rollout) | nothing |
| M10 | Per-worker: one task per sample, plain `gather`, no semaphore: **any agent-loop exception fails the whole batch**; no rollout deadline (grep `wait_for / asyncio.timeout` in `experimental/agent_loop`: none) | `agent_loop.py:657-671` | `C/train/runtime.py:298-303` isolates `rollout_fn`; the token path does not (D10) | anti-pattern |
| M10 | v1 sessions joined with `return_exceptions=True`; any failed session marks the group `failure` | `UV/verl/trainer/ppo/v1/agent_loop_tq.py:47-49,133-151` | `C/train/pool.py:189-207` (per-rollout status; masked members excluded) | nothing (Crucible's per-member mask is finer) |
| M10 | Failed groups stay sampleable by default; refill only with `sync_refill_failed_groups` (needs `gen_batch_size=1`) | `UV/verl/trainer/ppo/v1/replay_buffer.py:91,109-113,198-199,326-334` | `C/train/scheduler.py:18-20,56` (`group_backfill = pad / refill_member / drop_group`) | nothing (anti-pattern upstream) |
| M10 | Tool errors (unknown tool, bad JSON, tool exception) become an observation with tool reward 0, loop continues | `UV/verl/experimental/agent_loop/tool_agent_loop.py:455-493` | `C/world/world.py` choke point: `crucible:tool_crashed` masks (`C/core/schema/run.py:43`) | anti-pattern |
| M10 | Abort-resume: an aborted generation is re-issued with the tokens so far, 1 s sleep, no cap | `UV/verl/workers/rollout/llm_server.py:290-357` (346-355) | `C/train/runtime.py:179-199` (`version_switch` call/token, also no cap) | nothing |
| M10 | Rollout resume: only dataloader state; "in-flight samples… will be lost… TODO" | `UV/verl/experimental/fully_async_policy/fully_async_rollouter.py:655-669` | `C/train/state.py` (run checkpoint at the barrier; in-flight restarted, `C/train/runtime.py:145`) | nothing |
| M10 | Sandbox Fusion: 3 attempts on HTTP 504 only, linear backoff; but API error (−1) and timeout (−3) count as wrong; exception → 0.0 | `UV/verl/utils/reward_score/sandbox_fusion/utils.py:26-27,115-171,341-345`; `__init__.py:104-118` | `C/verify/runtime/plan.py:242-263` (InfraError → masked) | PATTERN for the 504-only retry rule; anti-pattern for the scoring |
| M10 | Rate-limited reward manager: token-bucket RPM/TPM; timeout (300 s) or exception → reward 0 with a flag | `UV/verl/experimental/reward_loop/reward_manager/limited.py:306,365,451-466` | `C/train/runtime.py:452-466` (reward timeout masks) | PATTERN for the bucket (M2 LLM client); anti-pattern for the 0 |
| M10 | Reward router: 16 retries, none on 4xx; comment says "safer to have a timeout", sets `total=None` | `UV/verl/experimental/reward_loop/reward_loop.py:157-170` | `C/train/runtime.py:118` (`reward_timeout_s=600`) | anti-pattern (confirms prior G4 1d) |
| M10 | Rollout dump `rollout_data_dir/{step}.jsonl` (input, output, gts, score, step, extra); no termination or error fields | `UV/verl/trainer/ppo/ray_trainer.py:454-477,1722-1724` | `rollout.json` per attempt, `crucible.rollout/v1` with failure and blame (`C/runtime/run.py:202-221`) | nothing |
| M8/M12 | DAPO `filter_groups` (v1): drop finished groups with `np.std(metric) == 0` (exact) over `rm_scores` or `reward_extra_info[metric]`; a missing metric raises | `replay_buffer.py:261-315` | `C/train/scheduler.py:172-178` (exact `len(set(rs)) == 1`) | nothing |
| M8/M12 | Bounded refill: evicting k groups grants 2k refill credits up to `max_inflight_gen_batches`; surplus discarded | `replay_buffer.py:94-114,470-500` | none: admission refills from sources with no bound or fallback (D13) | PATTERN (pick M8-1, the bound) |
| M8/M12 | `max_num_gen_batches` documented as "raise ValueError … Generated too many" but **ignored** in v1 (warning) and absent from `ray_trainer.py` | doc `UV/docs/algo/dapo.md:78-108`; code `UV/verl/trainer/ppo/v1/trainer_base.py:205-211` | — | doc ≠ code (confirms `m11-m12/01-verl-slime.md:345`) |
| M9 | Overlong prompt filter at dataset load (default on) | `UV/verl/utils/dataset/rl_dataset.py:119,197-199` | none at admission | PATTERN-low |
| M8 | Pass-rate prefilter or per-prompt history | absent (grep `pass_rate / curriculum / difficulty`) | — | — |

### 1.3 SkyRL (`SK` = `REF/infra-novasky-ai-skyrl`, 3d0ae22, Apache-2.0)

| Mod | Mechanism | Source file:line | Crucible equivalent file:line | Verdict |
|---|---|---|---|---|
| M10 | Generator fan-out: one `agent_loop` per prompt by `tqdm.gather` without `return_exceptions`; `agent_loop` has `try/finally` for the session only and `env.close()` outside the finally (leak on exception) | `SK/skyrl/train/generators/skyrl_gym_generator.py:386-743` (`:642`), `:1008-1028` | same defect class as D10 (`C/train/runtime.py:480-523`) | anti-pattern |
| M10 | Env worker pool: env `init/step/close` on a thread pool (`max_env_workers=32`), a new env per trajectory | `skyrl_gym_generator.py:200-205,286-291,394-396`; `skyrl/train/config/config.py:1424` | `C/train/envpool.py` (`env_workers`) | nothing |
| M10 | Tool timeouts become an error string shown to the policy; search retries on 5xx/timeout | `SK/skyrl-gym/skyrl_gym/tools/python.py:6-20`; `tools/sql.py:15-71`; `tools/search.py:25-120` | tool infra → masked (`crucible:tool_crashed`) | anti-pattern |
| M10 | **Harbor generator**: 2 attempts with a fresh Trial, no backoff; `AgentTimeoutError` masked without retry; `ContextLengthExceededError` trained with reward 0; missing verifier result or exception retried; if any rollout of a prompt failed, every rollout of that prompt gets loss mask 0 and `stop_reason="error"` | `SK/examples/train_integrations/harbor/harbor_generator.py:39,76-121,204-206,362-450` | `C/runtime/infra.py:87-96` (RetryRule by blame), `C/core/schema/run.py:60` (`agent_timeout` is policy, scored) | PATTERN-low (Crucible's split is finer; SkyRL masks the agent's own timeout) |
| M10 | Inference HTTP client: 30 retries, fixed 1 s, on disconnect/OSError/non-JSON 5xx; docstring says "retry + backoff" | `SK/skyrl/backends/skyrl_train/inference_servers/remote_inference_client.py:97,325-355,861` | `C/train/vllm_http.py` (prior M12-train.md:41 #16 "retry with backoff" still open) | doc ≠ code |
| M10 | `RateLimiter` around `trial.run()` | `SK/skyrl/train/utils/rate_limiter.py:181`; `harbor_generator.py:293-294,392-393` | none in `C/runtime/run.py` (one semaphore) | PATTERN-low |
| M10 | Fully-async resume restores consumed and filtered UIDs and staleness state; in-flight groups regenerated; one generator exception stops training | `SK/skyrl/train/fully_async_trainer.py:441-465,994-998` | `C/train/state.py`; `C/train/runtime.py:145` | nothing |
| M8/M12 | **Liveness-aware zero-variance filter**: groups with more than one *live* trajectory (loss mask not all 0) and spread ≤ `tol` (1e-6) are loss-masked; dead trajectories are left out of the variance test | `SK/skyrl/train/utils/trainer_utils.py:637-677`; `trainer.py:1156-1171`; `config.py:976-978` | `C/train/scheduler.py:174-175` reads members in `{"finished","failure"}` and compares with exact equality | PORT (pick M8-1; S) |
| M8/M12 | DAPO "filter": keep std > 0 groups, accumulate until `train_batch_size`, raise past `max_sample_batches` (30); **not liveness-aware** | `trainer_utils.py:525-600` (`:564`); `trainer.py:385-386,1657-1716`; `config.py:801` | `C/train/scheduler.py:172-178` | PATTERN (the cap + error) |
| M8/M12 | POLARIS "replace": std == 0 groups replaced by copies of good groups when > 30% are good | `trainer_utils.py:440-522` | none | nothing (duplicates samples) |
| M8/M12 | Fully-async `sample_full_batch` drops zero-variance groups and records their UIDs as filtered, so a resume skips them | `fully_async_trainer.py:803-821,853-867` | none (no per-prompt memory, D21) | PATTERN (pick M8-1) |
| M10 | skyrl-agent `mask_out_reason` masks infra and policy terminations alike | `SK/skyrl-agent/skyrl_agent/agents/base.py:321-329,464-488` | `C/core/schema/run.py:56-62` | anti-pattern (confirmed) |
| M9 | Dataset validation: prompt-length filter only; RL curriculum absent (only SFT `curriculum_sampler.py:37`) | `SK/skyrl/train/dataset/dataset.py:65-81` | `C/train/admission.py` (unwired, D14) | nothing |

Doc vs code: `SK/docs/content/docs/tutorials/fully_async.mdx:124` says dynamic sampling is unsupported in fully-async, while `:94-97` and `config.py:1041-1042` offer `sample_full_batch`; `harbor_generator.py:37-38` says a failed prompt is "skipped altogether", the code masks it (`:116-121`).

### 1.4 rLLM (`RL` = `REF/infra-rllm-org-rllm`, 4e828c8, Apache-2.0)

`agent_execution_engine.py` does not exist at this HEAD (only "legacy" mentions, `RL/cookbooks/frozenlake/README.md:3,133`).

| Mod | Mechanism | Source file:line | Crucible equivalent file:line | Verdict |
|---|---|---|---|---|
| M10 | `TerminationReason` {MAX_PROMPT_LENGTH_EXCEEDED, MAX_RESPONSE_LENGTH_EXCEEDED, ENV_DONE, MAX_TURNS_EXCEEDED, TIMEOUT, UNKNOWN, ERROR}; `run_with_termination_handling`: `wait_for(timeout)` → TIMEOUT, any exception → ERROR with type, message and traceback, partial trajectories kept | `RL/rllm/workflows/workflow.py:18-25,80-105` | `C/runtime/infra.py:68-84` (`boundary`), `C/core/schema/run.py:23-62` (blame) | PATTERN (pick M10-2: wrap the train rollout the same way) |
| M10 | Default workflow timeout `1e6` s (effectively none) | `workflow.py:39,59`; `RL/rllm/trainer/config/rllm/base.yaml:33` | `C/train/world_env.py:171` (none, D11) | anti-pattern |
| M10 | `AgentWorkflowEngine`: retries the whole workflow only on ERROR, up to `retry_limit` (3), no backoff; TIMEOUT not retried; `raise_on_error=True` (default) makes one permanent ERROR kill the step | `RL/rllm/engine/agent_workflow_engine.py:98-149`; `base.yaml:36,38` | `C/runtime/infra.py:87-96` (by blame, never policy) | nothing (Crucible stricter) |
| M10 | Episodes with no steps dropped and listed in `dropped_episodes` | `agent_workflow_engine.py:237-276` | masked records stay in `failures.jsonl` (`C/runtime/journal.py:84-85`) | nothing |
| M10 | AgentFlowEngine: semaphore; any exception (flow, trace fetch, evaluator) re-runs the whole rollout after deleting old traces | `RL/rllm/engine/agentflow_engine.py:346-377,458-524` | `C/runtime/harness/workspace.py:35-61` (re-grade, never re-run the agent) | anti-pattern (re-samples the policy when only the grader failed) |
| M10 | Sandbox lifecycle: teardown in `finally`, closed on setup error; install timeout 600 s | `agentflow_engine.py:556-568,638-643`; `RL/rllm/hooks.py:245-291` | `C/runtime/harness/native.py:291-297` | nothing |
| M10 | **WarmQueue**: prefetch up to `size` sandboxes, `is_alive()` before handing one out, dead ones closed and replaced, one prefetch retry after 15 s, tail closed on shutdown | `RL/rllm/sandbox/warm_queue.py:1-55,71-88,128-175,227-256` | `C/world/pool.py` (sandbox pool; no liveness check before hand-out found: grep `is_alive / alive` in `C/world/pool.py`) | PATTERN (pick M10-4) |
| M10 | RemoteAgentFlowEngine: 900 s session timeout; unfinished → reward 0.0, no retry despite `process_task_with_retry` | `RL/rllm/engine/remote_agent_flow_engine.py:34,96-150,203-211` | — | anti-pattern |
| M10 | CompactFiltering by termination reason: off by default; infra and policy reasons toggled together; legacy verl trainer never applies it though `docs/backends/verl.mdx:329-341` says it does | `RL/rllm/trainer/algorithms/config.py:110-161`; `base.yaml:112-120`; `RL/rllm/trainer/verl/agent_workflow_trainer.py:440-443` | `C/train/credit.py:68`, `C/train/trainer.py:217,312` (masked never trained) | nothing (doc ≠ code noted) |
| M10 | Async rollout executor: an exception prints and stores None, the group is built from the rest (silent shrink) | `RL/rllm/experimental/fully_async/rollout_executor.py:306-320` | `C/train/scheduler.py:18-20` (backfill policy explicit) | anti-pattern |
| M10 | Dead config keys `trajectory_timeout`, `overlong_filter` | `base.yaml:14-15` (no consumer) | D14 (Crucible has the same class of dead knob) | — |
| M10 | **Episode logger**: one JSON per episode with `termination_reason`, `metadata.error`, per-step data, batch summary | `RL/rllm/utils/episode_logger.py:45-58,79-133,161-193` | `rollout.json` + `record.json` per attempt | nothing |
| M10 | ShellScriptEvaluator: missing tests dir → 0, verifier exception logged at debug and ignored, missing reward file → 0 | `RL/rllm/eval/script_evaluator.py:58-63,90-100,133-134` | prior G4 pick M7-5 (export guard) | anti-pattern (confirmed) |
| M8/M12 | Rejection sampling (verl path): drop all-solved / none-solved tasks, loop until `solve_partial ≥ train_batch_size`, **no cap**; ERROR episodes count as solve-none | `agent_workflow_trainer.py:219-287` | `C/train/scheduler.py:172-178` | anti-pattern (the uncapped loop) |
| M8/M12 | `filter_uniform_groups` in the async buffer drops groups whose advantages are all ≤ 1e-8 (the config comment says "same is_correct") | `RL/rllm/trainer/buffer.py:235-240`; `config.py:204-206` | — | doc ≠ code |
| M8 | **Pass-rate DSL** `rllm dataset from-eval --filter "0 < avg < 1"` pooled across runs; errored attempts stay in `n` and `avg` | `RL/rllm/cli/dataset.py:304-349`; `RL/rllm/eval/curation.py:88-130,217-245` | `C/studio/datasets.py:96-101,197-220` (same DSL; masked episodes skipped) | nothing (Crucible ahead; reuse it as the learnability input, pick M8-2) |
| M9 | Oracle harness: `solution/solve.sh` as a manual smoke test (`rllm eval --agent oracle`) | `RL/rllm/harnesses/oracle.py:1-24,47-75` | A1 (`C/accept/checks/core.py:21-72`) | nothing |

### 1.5 AReaL (`AR` = `REF/infra-inclusionai-areal`, 298412a, Apache-2.0)

| Mod | Mechanism | Source file:line | Crucible equivalent file:line | Verdict |
|---|---|---|---|---|
| M10 | Workflow contract: `arun_episode` returns a trajectory or `None` (= rejected, never trained) | `AR/areal/api/workflow_api.py:14-23` | `rollout_fn → None` = masked (`C/train/runtime.py:186-188`) | nothing |
| M10 | Executor: `None` → "returned_none", `should_accept_fn` False → "rejected"; both free the staleness slot and clear remote shards; any other exception is logged, counted as rejected and dropped (no retry, no infra/policy label) | `AR/areal/infra/workflow_executor.py:1407-1490` | `C/train/runtime.py:298-303` (counted by blame) | nothing |
| M10/M12 | **`GroupedRolloutWorkflow`**: `None` members removed; `drop_incomplete_group` drops the group; a group below `min_usable_group_size` (default 2 under group normalization) is dropped; usable/trainable slot yields recorded | `AR/areal/infra/remote_inf_engine.py:75-90,182-232`; `AR/areal/api/cli_args.py:1914-1920` | `C/train/scheduler.py:18-20` (`pad` trains any survivors; only an all-masked group is evicted) | PATTERN (pick M10-6; S) |
| M10 | Interruptible generation: paused requests resubmitted with accumulated tokens, each segment tagged with its weight version; a final abort relabelled "length" | `remote_inf_engine.py:1105-1198` | `C/train/runtime.py:179-199` (`version_spans`) | nothing (prior M12-train "parity" confirmed) |
| M10 | Reward wrapper: 15 s timeout, 3 retries, broken process pool recreated, **after the last timeout returns 0.0** | `AR/areal/api/reward_api.py:76-82,141-183` (`:160-161`) | `C/train/runtime.py:452-459` masks `reward_timeout` | anti-pattern |
| M10 | Staleness capacity = min(max_concurrent − running, (max_head_offpolicyness + version + 1)·batch − (accepted + running)); rejected rollouts free their slot | `AR/areal/infra/staleness_manager.py:79-113,159-166` | `C/train/scheduler.py:9-16` (S×B×n untrained cap) | nothing |
| M10 | Daytona sandbox: create 60 s / run 30 s; a `DaytonaError` returns exit code 1 and a timeout 124, as ordinary results | `AR/areal/infra/sandbox/runner.py:112-113,268-294,339-347` | backend errors → `crucible:backend_error`, masked (`C/core/schema/run.py:41`) | anti-pattern |
| M10 | Default wait 7 days: no per-episode wall clock in the executor | `workflow_executor.py:297-298` | D3, D11 | anti-pattern |
| M10 | Resume: dataloader state per rank; staleness counters adjusted; in-flight rollouts not persisted | `AR/areal/utils/recover.py:45-97,293-327` | `C/train/state.py` | nothing |
| M8/M12 | `should_accept_fn` / `dynamic_filter_fn` on the finished group; examples keep mean ≤ 0.95 (one-sided), docs say `0 < mean < 1`; `dynamic_bs` returns a variable batch | `workflow_executor.py:794-845,1411-1417`; `AR/examples/tau2/train.py:62-63`; `AR/docs/en/algorithms/grpo_series.md:291-298` | `C/train/scheduler.py:172-178` | doc ≠ examples |
| M10 | `mask_no_eos_with_zero`: help says "exclude from training", code sets the reward of truncated sequences to 0 | `cli_args.py:1854-1860` vs `AR/areal/trainer/ppo/actor.py:489-492` | `C/runtime/infra.py:99-113` (`LimitPolicy` explicit) | doc ≠ code |
| M10 | Trajectory dump `{log}/rollout/{version}/{task_id}.jsonl`, rejected ones included | `workflow_executor.py:934-948,1024-1111` | `rollout.json` | nothing |

### 1.6 ROLL (`RO` = `REF/infra-alibaba-roll`, 581046a, Apache-2.0)

| Mod | Mechanism | Source file:line | Crucible equivalent file:line | Verdict |
|---|---|---|---|---|
| M10/M12 | `GroupQueue` over-sampling: hands out episodes while running < `group_size + group_size_redundancy`; `group_filter` True drops the group and opens a new one (help text says "return false to filter out") | `RO/roll/distributed/scheduler/rollout_scheduler.py:217-364` (`:307,330-349`); help `RO/roll/pipeline/agentic/agentic_config.py:144-147` | `C/train/scheduler.py` `refill_member` (replacement after failure, no redundancy up front) | PATTERN (pick M10-6) |
| M10 | Hung-env monitor: every 30 s logs envs idle longer than `hung_timeout` (3600 s); log only | `rollout_scheduler.py:31-207`; `agentic_config.py:178-182` | none (D3, D11) | PATTERN (with an action) |
| M10 | Per-env-tag step semaphore (Ray actor) `max_env_step_concurrent` | `RO/roll/utils/env_action_limiter.py:7-50,117-122`; `RO/roll/pipeline/agentic/env_manager/traj_env_manager.py:62-67` | none (one global semaphore, `C/runtime/run.py:101`) | PATTERN-low (prior M09-M11.md:63 keyed pools) |
| M10 | Reset failure → placeholder rollout with zero masks and `drop_flag`; 10 s sleep, then 10·n s backoff capped at 3600 s after 3 failures | `RO/roll/pipeline/agentic/env_manager/agent_native_env_manager.py:46-66,433-488` | `C/runtime/infra.py:87-96` (immediate retries, D9) | PATTERN (backoff) |
| M10 | Drop cap: flagged groups are dropped only while the filtered share ≤ 50%; beyond it placeholder groups (score 0) are trained | `agent_native_env_manager.py:492-521` | `C/train/runtime.py:326-337` (breaker stops the run instead) | anti-pattern |
| M10 | Mid-episode agent-request failure → SESSION_END → tests run and scored; test-session/upload failure returns reward False with `failure_mode` logged only; proxy env: missing score → 0.0, no `drop_flag` | `RO/roll/pipeline/agentic/env/sandbox/rock_tb_native_env.py:294-301,352-402,516-525`; `RO/roll/pipeline/agentic/env/rock/sanbox_manager.py:1222-1226`; `RO/roll/pipeline/agentic/env_manager/proxy_env_manager.py:128-138,374-385,401-409` | P16 (`C/core/schema/run.py:1-7`) | anti-pattern |
| M10 | Sandbox lifecycle: `auto_clear_seconds=3600` server-side TTL; 3 init attempts with 120 s·n backoff; `run_in_session` up to 3 retries; v2 polls an in-sandbox `/health` for 60 s | `sanbox_manager.py:170,198-230,326-361`; `RO/roll/pipeline/agentic/env/rock/sandbox_manager_v2.py:49-51,76-77,340-363` | `C/world/pool.py:95-186` (no liveness, no TTL; D22) | PATTERN (TTL + health; pick M10-4) |
| M8/M12 | RLVR `mean_filter` / `std_filter` (off by default); returns keep when given one item, and the only call passes a one-element list, so the filter never fires in `get_batch_opt_level_0` | `RO/roll/distributed/scheduler/user_defined_rollout_loop.py:29-52`; `RO/roll/distributed/scheduler/generate_scheduler.py:636-641` | — | anti-pattern (bug upstream) |
| M8/M12 | In-loss difficulty mask: groups with mean ≤ low or ≥ high get loss mask 0 | `RO/roll/utils/functionals.py:833-840,924-947`; `RO/roll/pipeline/rlvr/rlvr_config.py:149-153` | `filter_zero_variance` (`C/train/scheduler.py:172-178`) | nothing |
| M9 | `DatasetFilterConfig(min_difficulty, max_difficulty)` declared, no consumer | `rlvr_config.py:13-18,91-94` | `C/train/admission.py` (also unwired) | — |
| M10 | Dumps: `rollout_dump_dir` per step; `trajectory_data` JSON with `failure_info`, `stop_reason`, `test_output` | `RO/roll/pipeline/agentic/utils.py:299-330`; `agent_native_env_manager.py:368-421` | `rollout.json`, `record.json` | nothing |

### 1.7 slime (`SL` = `REF/infra-thudm-slime`, c933787, Apache-2.0)

| Mod | Mechanism | Source file:line | Crucible equivalent file:line | Verdict |
|---|---|---|---|---|
| M10 | `Sample.Status` PENDING/COMPLETED/TRUNCATED/ABORTED/FAILED; the FAILED comment says "can be retried or handled gracefully", nothing consumes it | `SL/slime/utils/types.py:148-158,490-495` | `C/core/schema/run.py:23-62` | doc ≠ code |
| M10 | Abort + recycle: on batch-full, tasks cancelled and servers aborted; with `--partial-rollout`, groups holding ABORTED samples go back to the data buffer; generation continues from the partial response | `SL/slime/rollout/sglang_rollout.py:158-179,426-473,754-774` | `C/train/runtime.py:179-199` | nothing |
| M10 | **Buffer checkpoint**: `state_dict` includes the partial-rollout buffer and cursors (`rollout/global_dataset_state_dict_{id}.pt`) | `SL/slime/data/data_source.py:135-182,220-233` | `C/train/state.py` (in-flight restarted, `C/train/runtime.py:145`) | PATTERN-low |
| M10 | `remove_sample` zeroes the loss mask but "does not determine whether the sample participates in advantage normalization"; the coding-agent example sets reward 0.0 + `remove_sample` + ABORTED on a wall-clock guard or exception, so 0.0 enters the std filter and the baseline | `SL/slime/utils/types.py:145`; `SL/slime/data/batch_builder.py:282-283`; `SL/slime/utils/arguments.py:1538-1547`; `SL/examples/coding_agent_rl/generate.py:202,270-280,316-333` | masked rows excluded from group statistics (`C/train/trainer.py:312`) | anti-pattern |
| M10 | Fully async: an ABORTED group is requeued with no retry cap | `SL/slime/rollout/fully_async_rollout.py:247-262` | `C/train/runtime.py:59-62` (`max_cuts` σ evicts) | nothing |
| M10 | E2B sandbox: retries only transient RPC errors with exponential backoff and jitter, **never a non-idempotent exec** | `SL/slime/agent/sandbox.py:207-271,296-307` | `C/runtime/infra.py:87-96` (no idempotency notion) | PATTERN (pick M10-5) |
| M10 | Engine health: `/health_generate` every 600 s, a failed engine killed and recovered | `SL/slime/utils/arguments.py:658-683` | `C/train/gateway.py:15-20` (health loop evicts) | nothing |
| M10 | SGLang HTTP: 60 retries, fixed 1 s, client timeout None | `SL/slime/utils/http_utils.py:189-223,256` | — | anti-pattern |
| M8/M12 | `check_reward_nonzero_std` (std > 1e-6) with over-sampling; **`_with_fallback`** keeps a zero-std group when the remaining pool is short (`keep_when_insufficient`) | `SL/slime/rollout/filter_hub/dynamic_sampling_filters.py:9-23`; `SL/slime/rollout/filter_hub/base_types.py:14-24`; `sglang_rollout.py:513-554` | `C/train/scheduler.py:172-178` (no fallback, D13) | PORT (pick M8-1; S) |
| M10 | Debug dump `--save-debug-rollout-data` with a `{rollout_id}` template; required when trainer fault tolerance is on | `SL/slime/observability/rollout_data_utils.py:136-160`; `arguments.py:1364-1371,2266-2271` | `C/train/runtime.py:101-104,598` (`record_rollouts`) | nothing |

### 1.8 OpenRLHF (`OR` = `REF/infra-openrlhf-openrlhf`, dc2a7ad, Apache-2.0)

| Mod | Mechanism | Source file:line | Crucible equivalent file:line | Verdict |
|---|---|---|---|---|
| M10 | Agent loop until `done` or token budget; no turn cap, no timeout, no try around `step()`; samples fanned out with `gather` without `return_exceptions` | `OR/openrlhf/utils/agent.py:92-169`; `OR/openrlhf/trainer/ray/vllm_engine.py:183-205` | — | anti-pattern |
| M10 | Remote reward: 180 s, 5 tries, 1 s sleep, then `RuntimeError` that the executor swallows (info log), leaving `reward=None`; downstream raises if the first sample's reward is None | `agent.py:286-304,331,347-362`; `OR/openrlhf/trainer/ppo_utils/experience_maker.py:139-142` | `C/train/runtime.py:452-466` | anti-pattern |
| M8/M12 | Dynamic filtering: keep a group only if `min_r < mean < max_r` (strict, default (0,1)), n > 1, replacement prompt dispatched; **skipped if any score is None** | `OR/openrlhf/trainer/ppo_utils/samples_generator.py:169-190`; `OR/openrlhf/cli/train_ppo_ray.py:600-604,746-755` | `C/train/scheduler.py:172-178` | nothing (masked rows bypassing the filter is the anti-pattern) |
| M10 | Partial rollout pauses vLLM with `mode="keep"`: old and new weights mix inside a sample, no staleness bound | `train_ppo_ray.py:293-302,717-718`; `vllm_engine.py:130-134` | `C/train/runtime.py:179-199` (spans recorded) | anti-pattern |
| M10 | Resume: dataloader state and consumed prompts saved; the dynamic-filter `_sample_buffer` is not | `OR/openrlhf/trainer/ppo_trainer.py:512-518,555-559`; `samples_generator.py:96-131` | — | nothing |

### 1.9 TRL (`TR` = `REF/infra-huggingface-trl`, 72cf37d, Apache-2.0)

| Mod | Mechanism | Source file:line | Crucible equivalent file:line | Verdict |
|---|---|---|---|---|
| M10 | Reward `None` → NaN; rows where every reward function returned NaN are left out of the group mean/std (`nanmean`/`nanstd`) and get advantage 0, but stay in the loss mask (docs say "ignored") | `TR/trl/trainer/grpo_trainer.py:1589-1591,1608,1626-1638,2685-2739`; `TR/docs/source/grpo_trainer.md:592-594` | `C/train/trainer.py:217,312,423` (masked rows dropped from items) | nothing (Crucible stricter; the M11 TRL export maps a mask to `None`, prior M09-M11.md:78) |
| M8/M12 | Zero-std groups only logged (`frac_reward_zero_std`); `scale_rewards` group/batch/none | `grpo_trainer.py:2717,2759`; `TR/trl/trainer/grpo_config.py:226` | `C/train/monitors.py:87` | nothing |
| M10 | `mask_truncated_completions` zeroes the completion mask without EOS (default off) | `grpo_trainer.py:2416-2424`; `grpo_config.py:267` | `LimitPolicy` (`C/runtime/infra.py:99-113`) | nothing |
| M10 | Tools: `max_tool_calling_iterations` default unbounded; a tool exception is shown to the model as `{"error": ...}` | `grpo_trainer.py:1899,1929-1952`; `grpo_config.py:317` | — | anti-pattern |
| M10 | Async GRPO: environment instances pooled and reset per rollout; retries only aiohttp/timeout errors, 3 attempts, backoff min(2^n, 16); a task exception kills the worker | `TR/trl/experimental/async_grpo/async_rollout_worker.py:57,429-451,643-645,784-801,1112-1123` | `C/train/envpool.py`; `C/train/runtime.py:298-303` | PATTERN-low (backoff shape) |
| M10 | Harbor env: exec timeout 180 s; empty verifier rewards → 0.0; a never-reset env → 0.0 | `TR/trl/experimental/harbor/_env.py:78-84,94-95,183-190` | `C/verify/runtime/plan.py:267-275` | anti-pattern (relevant to the M11 TRL export: a Crucible mask must not reach this path as "no reward") |
| — | `trl/experimental/openenv` does not exist at this HEAD; the OpenEnv harness is `TR/trl/experimental/async_grpo/openenv_harness.py` (`max_turns` default 8, `:133`) | — | — | path correction |

### 1.10 checkpoint-engine (`CE` = `REF/infra-moonshotai-checkpoint-engine`, 84631a2, MIT)

Weight broadcast only (`CE/README.md:1-4`): a 300 s request timeout (`CE/checkpoint_engine/api.py:18`) and 8 port-bind retries (`CE/checkpoint_engine/p2p_store.py:22-33`). Nothing for M8, M9 or M10.

Cross-repo absences (grep `pass_rate|curriculum|difficulty` in all nine trainer repos): no per-prompt pass-rate history across epochs and no offline pass-rate prefilter for RL. The only band tooling is rLLM's SFT curation DSL, which Crucible already has with masks excluded (`C/studio/datasets.py:96-101,197-220`). No repo validates that a task is solvable before training.

## 2. Best picks

### M10 runtime

**M10-1. Contain every attempt (PATTERN, S).** Fixes D1, D2, D3, D8.
- Sources: `AE/cli/eval/run.py:118-131` (`_run_safe`: the exception comes back with the run's identity, the batch goes on); `AE/task/teardown.py:75-118` (bounded, best-effort teardown that reports what is still up); `RL/rllm/workflows/workflow.py:80-105` (`wait_for` → TIMEOUT, any exception → ERROR with type and traceback, partial trajectory kept).
- Change in `C/runtime/run.py`:
  - `RunSpec.attempt_timeout_s: float | None` (default from the harness: `limits.timeout_s` + grade budget).
  - In `one()` (`:139`): `async with asyncio.timeout(spec.attempt_timeout_s)` around `harness.run`; on `TimeoutError` build `HarnessResult(failure=Failure(kind="crucible:attempt_deadline", stage=<last phase from the harness timings>, blame="harness"))`; on any other `Exception` build `agent_run_error` with `blame="unknown"` and the traceback tail in `reason`. Both settle the ledger as `uncertain` (`ledger.settle(aid, 0.0, uncertain=policy.estimate_usd)`).
  - `asyncio.gather(..., return_exceptions=True)` at `:159`, and a final sweep that turns a returned exception into a journal event instead of re-raising.
  - Release the semaphore while a retry waits (D9): move `async with sem` inside the `while` loop.
- `C/core/schema/run.py:31-50`: register `crucible:attempt_deadline` (harness) and `crucible:harness_unsupported` (task). `RunRecord.mark_failure` (`:168-170`) should normalize an unknown kind through `runtime.infra.failure` with blame `unknown` rather than raise, so a typo is counted by the unknown-blame stop rule instead of killing the run.
- `C/runtime/infra.py:46-52`: an unmapped foreign name gets blame `unknown`, not `harness` (D8).
- Why it beats Crucible: today one `OSError` in `collect()` ends a multi-hour run with every in-flight sample orphaned [ran p1a]; agentenv and rLLM both isolate per run.
- Test: re-run p1; expect (a) 5 scored + 1 masked `unknown`, ledger 0 `reserved`; (b) masked `crucible:harness_unsupported`; (d) masked `crucible:attempt_deadline` within the deadline.

**M10-2. The same containment in the trainer's rollout paths (PATTERN, S).** Fixes D10, D11, D12.
- Source: `RL/rllm/workflows/workflow.py:18-25,80-105` (termination reasons, wrapper); AReaL's executor drops and counts any exception (`AR/areal/infra/workflow_executor.py:1462-1490`).
- Change:
  - `C/train/runtime.py:480-523` (`_rollout`): `except Exception as e: self._failure(_blame(e), e); rec = None`, as `_rollout_fn` already does (`:298-303`); `finally: await env.close()` when the env has one.
  - `RuntimeConfig.rollout_timeout_s` (default e.g. 3600) applied with `asyncio.timeout` in both `_rollout` and `_rollout_fn`; a timeout masks with kind `crucible:attempt_deadline` (harness) and counts in the breaker window.
  - `C/train/world_env.py:176-180`: return a typed mask instead of bare `None`, e.g. `MaskedRollout(kind, stage, blame)`, and have `_complete` record `self.report.failures[f"{blame}:{kind}"]`, so the harness-share breaker sees harness failures; honor `LimitPolicy.for_stop(...) == "mask"` as `NativeHarness` does (`C/runtime/harness/native.py:272-273`); remove the temp dir (`:165`) in the `finally`.
- Why: p2 shows a 5% non-infra env exception turns a 0.5 s, 6-step run into an 8 s stall with 0 steps and nothing in `failures`.

**M10-3. Crash-durable checkpoint resume (own fix + slime PATTERN-low, S).** Fixes D4.
- Source: slime saves the partial-rollout buffer and cursors with the run state (`SL/slime/data/data_source.py:135-182,220-233`); Crucible needs the per-episode analogue.
- Change: `C/runtime/harness/native.py:106-120` (`_resumable`) looks for the newest `checkpoint.json` among `a{n-1}`, `aborted/a*` of the same sample, and accepts a missing `record.json` (a crash) as resumable; `C/runtime/journal.py:103-107` keeps moving the dir to `aborted/` but the harness resolves the World store path from there (`checkpoint.resume(..., prev_dir=...)` already takes the directory). Record `resumed_from: "aborted/a1@step3000"`. Add identity to `checkpoint.json` (world spec hash, task id, harness id; prior `M09-M11.md:50`) and refuse a mismatch.
- Test: p5 must print a resumable checkpoint; extend `longhorizon/gate.py` with a real `os.kill` of a child runner.
- Why: §10.4 exists for exactly the killed-worker case, which today restarts from step 0.

**M10-4. Sandbox hygiene (PATTERN, S–M).** Fixes D6, D22.
- Sources: `RL/rllm/sandbox/warm_queue.py:128-175` (`is_alive()` before hand-out, dead ones closed and replaced; one prefetch retry; tail closed on shutdown); `RO/roll/pipeline/agentic/env/rock/sanbox_manager.py:170,198-230` (server-side `auto_clear_seconds` TTL; 3 init attempts with backoff); `AE/providers/sandbox_providers/local_sandbox.py:229-260` (label-scoped `docker rm -f`); `AE/task/teardown.py:36-50` (`still_up`).
- Change:
  - `C/world/pool.py:133-143` (`acquire`): check the queued container is running (`docker inspect -f {{.State.Running}}`) before handing it out; close and replace a dead one.
  - `C/world/pool.py:108-114` (`_start_one`): count start failures; `warm()` (`:127-131`) raises after N consecutive failures instead of waiting forever.
  - Every container gets `--label crucible.run=<run_id>` and `--label crucible.attempt=<attempt_id>` (`C/verify/stage.py:111`, `C/world/sandbox.py:87`, `C/interop/agentenv/mcp_env.py:124`, `C/interop/harbor/verify_dir.py:101`); `Runner.run` start sweeps `crucible.run=<run_id>` containers whose attempt has no live owner; the harness `finally` records a `teardown.json` (terminated / failed / still up) instead of `except Exception: pass` (`C/runtime/harness/native.py:291-297`).
- Why: p6 shows `warm()` hangs when the daemon is down and a dead pre-warmed container is handed out.

**M10-5. Retry rules with backoff and idempotency (PATTERN, S).**
- Sources: `SL/slime/agent/sandbox.py:207-271` (retry only transient RPC errors, exponential backoff with jitter, never a non-idempotent exec); `RO/roll/pipeline/agentic/env_manager/agent_native_env_manager.py:433-488` (reset backoff 10·n s capped at 3600 s); verl Sandbox Fusion retries only HTTP 504 (`UV/verl/utils/reward_score/sandbox_fusion/utils.py:115-171`).
- Change: `C/runtime/infra.py:87-96` `RetryRule` gains `backoff_s=(base, cap)`, jitter, and `idempotent_stages=("seed","verification")`: an `agent`-stage failure with `delivery != "not_sent"` (`C/core/schema/run.py:110`) is retried only when the harness can reuse prior work (re-collect, re-grade, checkpoint) and otherwise counts once against `max_attempts`. The runner sleeps outside the semaphore. `delivery` also drives `BudgetLedger.release` vs `uncertain` (D5).
- Why: today retries are immediate and a `not_sent` reservation is charged.

**M10-6. Group survival rule (PATTERN, S).** Fixes D23.
- Sources: `AR/areal/infra/remote_inf_engine.py:182-232` (`min_usable_group_size`, default 2 under group normalization; usable/trainable yields recorded); `RO/roll/distributed/scheduler/rollout_scheduler.py:307` (`group_size_redundancy`: dispatch n + r, keep the first n).
- Change: `C/train/scheduler.py` `SchedulerConfig.min_usable_group_size: int = 2` applied in backfill `pad` (evict a group with fewer live members), and `group_redundancy: int = 0` in admission; report `group/usable_yield`. Same rule in `C/runtime/infra.pass_at_k` (`:132-144`), which already skips groups with n < k.
- Why: with `pad`, a group of 8 with 7 infra-masked members trains one rollout against a baseline of itself.

Do not copy: reward timeout or missing reward → 0 (AReaL `reward_api.py:160-161`, verl `limited.py:451-466`, ROLL `sanbox_manager.py:1222-1226`, TRL `harbor/_env.py:183-190`, rLLM `script_evaluator.py:133-134`); tool infra errors shown to the policy (verl `tool_agent_loop.py:455-493`, SkyRL `tools/python.py:6-20`, TRL `grpo_trainer.py:1929-1952`); re-running the policy when only the grader failed (rLLM `agentflow_engine.py:458-524`); masked samples left in the group baseline (slime `remove_sample`, `arguments.py:1538-1547`); training placeholder groups past a drop cap (ROLL `agent_native_env_manager.py:492-521`).

### M9 accept

**M9-1. One bench band (own fix, S).** Fixes D15 [ran p3].
- Change: `C/accept/checks/extended.py:102,114,129` use `learnability.BANDS[split]` and `decided()` (`C/forge/learnability.py:51,60-66`) instead of a local point-estimate band; for bench, accept only when the Wilson upper bound ≤ 0.2, top up while undecided, and raise `a10_max_runs` (`C/accept/policy.py:31`) to ≥ 16 for bench (0/15 → 0.204, 0/16 → 0.194). Or route bench admission through `ladder.accept(split="bench")` (`C/forge/ladder.py:92-106`) so there is one rule.
- Why: A10 admits a bench task on 0/2 frontier passes; the forge would reject the same evidence.

**M9-2. World round-trip check A19 (PATTERN, M).** Fixes D17.
- Source: `AE/task_step/task_steps/multienv_validator/verify_universe_roundtrip.py:97-151` (load → export 1 → reload → export 2; load fidelity and idempotency) and `universe_comparison.py:131-242` (critical vs benign issue classes).
- Change: `C/accept/checks/extended.py` new `A19` (cost class `cpu`, depends on A1): replay the oracle to step k (k = 1/3, 2/3 of its length), `World.snapshot()`, restore into a fresh World (`C/world/world.py:939-964`), snapshot again, require identical `WorldRef.fingerprint` and identical next-step tool results; plus snapshot → restore → snapshot idempotency at the end state. Detectors: `crucible:snapshot_mismatch` incidents.
- Why: M10 checkpoint resume and M6 branching both restore mid-episode; today a non-restorable App is discovered as masks during training.

**M9-3. Preflight and dry run (PATTERN, S).**
- Source: `AE/bundle/plan.py:1-8,98` (report every pre-write failure together), `AE/bundle/preflight.py:1-14,83` (each deploy checked against its provider), `AE/bundle/run.py:232-257` (`dry_run_bundle`).
- Change: `C/accept/run.py:33-75` gets `preflight(task, policy, needs)` run before the loop: dependency closure resolvable, every needed capability present (`MissingCapability` now surfaces only when the check runs, `:61-62`), image refs resolvable (`C/accept/key.py:39-44`), GPU arch available; `accept_task(..., dry_run=True)` returns the plan and the preflight report without running anything. Fold in the cost-gated short circuit already proposed (`M09-M11.md:26`).
- Why: an unresolvable image or a missing adapter should fail in seconds, not after A1's GPU grades.

**M9-4. Tool-surface conformance of forged Worlds (PATTERN, S).**
- Source: `AE/task_step/task_steps/mcp_env_validator/verify_spec_conformance.py:1-15` (spec as a floor; extra tools reported, an undeclared required param fails); `AE/task_step/task_steps/mcp_env_validator/validation_gate_aggregator.py:27-49` (missing verdict fails, skipped is advisory).
- Change: for `world_synth` tasks (`packs/_toy/src/crucible_pack_toy/forge/world_synth.py:87-123`), a static check in A9's lint list that the World's `ToolSpec`s match the architect's declared tool surface (names, required params, enums); extra tools reported.
- Why: the toy forge builds tools from an LLM-written spec; nothing checks the built surface against it.

**M9-5. Wire the flaky screen and make admission of training sources a gate (own, S).** Fixes D14 (half).
- Change: call `C/train/admission.screen` from the M12 launch path for every source (`C/train/launch.py`) with the task's oracle as `rerun`; `infra` verdicts are listed for a retry, never dropped as broken. Mark the option matrix "Part" until then.

### M8 forge

**M8-1. Close the loop from training to the forge (PATTERN + small PORTs, M).** Fixes D13, D21.
- Sources:
  - SkyRL liveness-aware zero-variance test (`SK/skyrl/train/utils/trainer_utils.py:637-677`; tolerance `config.py:976-978`): only live members count, spread ≤ `tol`. PORT the 15-line predicate.
  - slime fallback (`SL/slime/rollout/filter_hub/dynamic_sampling_filters.py:18-23`, `base_types.py:14-24`): keep a zero-variance group when the pool is short. PORT the rule.
  - verl bounded refill (`UV/verl/trainer/ppo/v1/replay_buffer.py:94-114,470-500`): evictions grant refill credits up to a cap.
  - SkyRL records filtered UIDs so a resume skips them (`SK/skyrl/train/fully_async_trainer.py:803-821,853-867`).
- Change:
  - `C/train/scheduler.py:172-178`: compare live members only (`{"finished"}`, not `"failure"`), with `tol`; `SchedulerConfig.zero_var_fallback_after: int` blocked barriers, then keep; `max_refill_per_step`.
  - New `C/train/curriculum.py`: `PromptStats(task_id, n, k, last_step)` updated at group terminal (`C/train/runtime.py:365-368`), masked members excluded; persisted in the run checkpoint (`C/train/state.py`); `retired` and `too_hard` side pools as in prime-rl (`M12-train.md:26`).
  - Export per-task training rates to the forge: `C/forge/curator.py:89` calls `ladder.retire(rate.p)` from `PromptStats`; `advise()` (`C/forge/learnability.py:82-93`) gets the same rates for L2.
- Why: every trainer here discards the information that a prompt was all-solved or all-failed; DESIGN §7.4's curriculum refresh needs it, and none of the nine repos keeps per-prompt history (grep), so Crucible must build it.

**M8-2. Masked-aware outcomes for learnability and the ladder (own, S).** Fixes D18, D19.
- Change: `measure(run)` and `Rung.run` accept `list[bool | None]`; `None` is dropped and the step target counts only scored samples (cap the extra draws at 2× to stop a broken environment from looping); `PassRate` gets `masked: int`. Add `runtime.infra.outcomes(records, threshold)` that maps `RunRecord.scored_reward()` to bool and skips masked ones, and reuse the per-task aggregation already in `C/studio/datasets.py:96-101,197-220` (same DSL as rLLM's curation, which counts errored attempts: `RL/rllm/eval/curation.py:88-130`). Rewrite `packs/_toy/src/crucible_pack_toy/forge/calibrate.py:32-68` on the M10 Runner with that adapter.
- Why: today a masked toy calibration attempt steers the generator as a failure.

**M8-3. Forge node robustness and retry from an ancestor (PATTERN, M).** Completes D20 (the concurrent M8 audit already contains the plan-level containment; the timeout, backoff and retry-from-ancestor parts remain).
- Source: `AE/task_step/task_step.py:52-81` (`RetryConfig(retry_from_step_id, max_retries)`, validated to name an ancestor: `AE/task/task.py:317-340`); span rollback by journal replay with an epoch fence (`AE/task/task.py:166-275`; `AE/task/store.py:422-431,632-690`); the audit trail `retry_resets` and `retried` flags.
- Change in `C/forge/orchestrator.py`:
  - `_run_node` (`:313-363`): `asyncio.timeout(node.timeout_s)`; any exception outside the content set becomes `RoleInfraError` (infra), with exponential backoff instead of `sleep(0)`; the reservation is settled in a `finally`; infra events journaled.
  - `run_plans` (`:286-295`): `return_exceptions=True`; a crashed plan becomes `abandoned(infra)`.
  - `Node.retry_from: str | None`: when a node's validators fail with findings that blame an input (e.g. the Test writer reports an untestable spec), re-run from the named ancestor with the findings as feedback; outputs of the rolled-back span are marked superseded in the journal (input-hash reuse already makes the replay cheap).
- Why: p4 shows one transport error loses every concurrent plan; and today a bad Architect output can only be retried by the node that consumes it.

Nothing to take for M8 from the trainers' synthesis side: none of the nine repos or agentenv generates tasks (agentenv grep `synthes|difficulty|learnab|dedup|generate_task`: only unrelated hits in document-store update code and list dedupe).

## 3. Crucible defects and spec gaps

M10 runtime:
- **D1 [ran p1a] A harness exception kills the whole run.** `Runner.one` awaits `harness.run` with no try (`C/runtime/run.py:139`), under `asyncio.gather` without `return_exceptions` (`:159`). Code outside `boundary()` can raise: `collect()` (`C/runtime/harness/workspace.py:127`), `save_episode`/`rule_flags`/`format_errors_by_call` (`C/runtime/harness/native.py:261-263`), `task.broken` (`native.py:269`, `workspace.py:122`). Probe: one `OSError` among 6 samples → `Runner.run` raises, 6 ledger ops left `reserved`, 0 rows in `results.jsonl`, the other 5 attempts orphaned. Prior M09-M11.md:38-39 still open.
- **D2 [ran p1b] An unregistered failure kind crashes the run.** `C/interop/agentenv/harness.py:123` raises `crucible:harness_unsupported`, which is not in `CRUCIBLE_FAILURE_KINDS` (`C/core/schema/run.py:31-50`); `RunRecord.mark_failure` raises `ValueError("unknown failure kind")` (`run.py:168-170`) inside `Runner._record` (`C/runtime/run.py:235-236`). Any typo in a harness's kind does the same. Fix: register the kind (blame `task` or `harness`) and make `_record` fall back to `failure(kind)` normalization (`C/runtime/infra.py:46-57`) instead of raising.
- **D3 [ran p1d] No per-attempt deadline.** `wall_clock_s` is only checked before dispatch (`C/runtime/run.py:103-114,120`); a hung `task.open`, grade or harness never returns. `EpisodeLimits.timeout_s` defaults to None (`C/actors/policy.py:467`) and covers only the agent loop (`:586`). Probe: `wall_clock_s=0.2`, still running after 2 s. Prior M09-M11.md:40 still open.
- **D4 [ran p5] Mid-episode checkpoints do not survive a process crash.** A crashed attempt has `checkpoint.json` but no `record.json`; `RunJournal.next_attempt` moves it to `aborted/` (`C/runtime/journal.py:103-107`), and `NativeHarness._resumable` reads only `a{n-1}` and needs a masked agent-stage record (`C/runtime/harness/native.py:106-120`). Probe: resumable → None. The long-horizon gate's "crash" is an in-process `ConnectionError` that writes a masked record (`packs/_toy/src/crucible_pack_toy/longhorizon/gate.py:37`), so DESIGN.md:2600 ("Crashed on attempt 1, it resumed from the step-3000 checkpoint") does not cover a killed worker, the case §10.4 is for.
- **D5 [read] Uncertain spend has no stop rule, delivery is unread.** `BudgetLedger.over_uncertain_tolerance` and `release` have no caller (`C/runtime/budget.py:61-74`; grep); `Failure.delivery` (`C/core/schema/run.py:110`) is never read (grep `delivery` in `src/`). Prior M09-M11.md:49 still open.
- **D6 [read] Orphaned sandboxes are never swept.** Containers carry `crucible.sandbox=1` (`C/verify/stage.py:111`, `C/world/sandbox.py:87`, `C/interop/agentenv/mcp_env.py:124`, `C/interop/harbor/verify_dir.py:101`), but nothing lists or removes them; `world.close()` errors are swallowed (`C/runtime/harness/native.py:291-297`). The sweep promised at `docs/research/m9-m10/02-runtime.md:928` does not exist.
- **D7 [read] An A2A agent's provider outage is scored 0.** The protocol defaults `TaskError.error_type` to `agent_error` (`AP/a2a_agent/tasks/v1.py:165`); `C/interop/agentenv/harness.py:157-162` scores every `agent_error` as policy. An agent that catches an upstream 5xx or quota error and returns a plain failure is trained as a 0. The capture log the harness already reads (`harness.py:152-153`) shows whether the last upstream call failed; use it to override blame to infra.
- **D8 [read] Unmapped foreign kinds bypass the unknown-blame stop rule.** `normalize` maps any unknown name to `agent_run_error` with blame `harness` (`C/runtime/infra.py:46-52`; `C/core/schema/run.py:56`), so `Health.unknown` never counts it (`C/runtime/infra.py:161-162`) [ran p1c: 40/40 unmapped → only the mask-fraction rule fires]. The "classifier needs a rule" alarm (`:169-170`) cannot see the case it exists for.
- **D9 [read] The semaphore is held across retries and there is no backoff** (`C/runtime/run.py:118-157`): a sample that retries 3 times holds a slot through all of them, and retries are immediate. Prior M09-M11.md:41.

M12 runtime (the consumer of M10 records):
- **D10 [ran p2] A non-`EnvInfraError` exception in a token env hangs its group and is invisible.** `Runtime._rollout` catches only `EnvInfraError` (`C/train/runtime.py:517`); the `rollout_fn` path catches everything (`:298-303`). Probe (5% `ValueError` in `env.step`): 0 training steps, stop `wall_clock` after 8 s, `failures={}`, 3 rollouts left `running`, "Task exception was never retrieved"; the same rate as `EnvInfraError` trains 6 steps in 0.5 s with 18 masked. The env is never closed (no `finally`). The circuit breaker the option matrix lists as "Have" (`docs/research/m12-infra/00-option-matrix.md:38`) never sees these.
- **D11 [read] No per-rollout deadline in training.** `RuntimeConfig` has reward, aux, group and teacher deadlines (`C/train/runtime.py:80-120`) but none for the episode; `world_rollout` runs `EpisodeLimits(max_steps=...)` with no `timeout_s` (`C/train/world_env.py:171`); the token loop is `while True` until the env says `done` (`C/train/runtime.py:494-512`).
- **D12 [read] `world_rollout` drops the reason for a mask and ignores `LimitPolicy` "mask".** A masked episode or grade returns bare `None` (`C/train/world_env.py:176-180`), so the runtime records "masked" with no kind (`C/train/runtime.py:355-361`) and the harness-share breaker cannot separate harness from infra. Only `limit_policy.penalty` is consulted (`world_env.py:182`); `on_steps="mask"` scores the episode, unlike `NativeHarness` (`C/runtime/harness/native.py:272-273`). The per-rollout temp dir (`world_env.py:165`) is never removed.
- **D13 [read] Dynamic sampling can starve.** `filter_zero_variance` evicts every all-equal group (`C/train/scheduler.py:172-178`) with no fallback after repeated `need_more` barriers and no per-prompt memory (grep `keep_when_insufficient|per_prompt|curriculum` in `C/train`: only the noise curriculum). Prior M12-train.md:26-27 still open.
- **D14 [read] Built but unwired.** `train/admission.screen` (MiMo's 8-rerun flaky filter) has no caller in `src/` or `packs/` (grep `admission import|screen(`); `objectives.overlong_penalty` has none either (`C/train/objectives.py:136`). The option matrix marks both "Have" (`00-option-matrix.md:91,117`).

M9 accept:
- **D15 [ran p3] Bench admission has two incompatible bands.** A10 uses `Band(lo=0.0, hi=0.2)` on the point estimate (`C/accept/checks/extended.py:102,129`) while the forge uses the Wilson upper bound ≤ 0.2 (`C/forge/learnability.py:51`, `C/forge/ladder.py:100-104`). 0/2 frontier passes: A10 in band, ladder rejects (upper bound 0.66). Worse, `a10_max_runs = 8` (`C/accept/policy.py:31`) can never reach the 15 straight failures the Wilson rule needs (0/15 → 0.204), and the top-up loop never runs for p = 0 because 0 is "inside" [0, 0.2] (`extended.py:114`).
- **D16 [read] A11 never injects two of the B.5 classes it claims.** A11 registers detectors `B.5-empty-patch` and `B.5-zombie` (`C/accept/checks/extended.py:136`) but its cases are blank usage, credits, contention, grader crash and two controls (`:176-187`), all through `NativeHarness` (`:166`), never `WorkspaceHarness`. The registry routes both to per-kit unit tests (`C/accept/registry/defects/B.5-empty-patch.yaml:9-12`, `B.5-zombie.yaml:9-12`) under check names `A11.empty_collection` / `A11.zombie` that do not exist in A11. So a coding task's own collect path is never exercised at admission.
- **D17 [read] No World round-trip admission check.** A9 compares two fresh starts (`C/accept/checks/extended.py:76-80`); restore fidelity is checked only at run time, where a failure becomes a masked `crucible:snapshot_mismatch` during training or calibration. M10 checkpoint resume and M6 branching both depend on it.

M8 forge:
- **D18 [read] Masked calibration attempts are counted as failures.** `packs/_toy/src/crucible_pack_toy/forge/calibrate.py:41-42` returns `rep.reward == 1.0` for a masked episode or grade (it logs `masked` but keeps the `False`), and `estimate`/`advise` steer the generator on it (`:65-68`). This bypasses the M10 runner, the journal and blame. P16 violation in the steering loop.
- **D19 [read] The learnability and ladder interfaces cannot express a masked sample.** `measure(run)` and `Rung.run` take `list[bool]` (`C/forge/learnability.py:69-79`; `C/forge/ladder.py:35-40,62-77`); `run_ladder`, `measure` and `retire` have no production caller (grep: `C/forge/curator.py:89` only), and no adapter turns `RunRecord`s into outcomes with masks dropped.
- **D20 [ran p4 at HEAD; fixed in the uncommitted M8-audit tree] One role's unmapped exception aborts every concurrent forge plan.** `_run_node` maps only `RoleInfraError`, `BudgetExhausted`, `LLMOutputError`, `ValidationError` (`C/forge/orchestrator.py:313-336`); `run_plans` gathers without `return_exceptions` (`:286-295`). Probe: a `ConnectionError` in one of two plans → `run_plans` raises, both tasks stay `draft`, both reservations stay in `Ledger.reserved`. Toy roles call `LLMClient` directly (`packs/_toy/src/crucible_pack_toy/forge/roles.py:79,156,174,339`); only the Harbor role runner raises `RoleInfraError` (`C/interop/harbor/role_runner.py:228-229`). No per-node timeout; infra retries are immediate (`orchestrator.py:322-328`). Re-run of p4 on the working tree late in this sweep: `[('t-ok', 'constructed'), ('t-flaky', 'abandoned')]`; the concurrent edit wraps `run_plans` and classifies role exceptions ("M8 audit V4"). The per-node timeout and the `sleep(0)` infra retry are still there in that tree.
- **D21 [read] Training never feeds the forge.** `ladder.retire(l2_training_rate)` (`C/forge/ladder.py:125-127`) needs a per-task training pass rate; the trainer keeps none (zero-variance groups are evicted and counted, `C/train/scheduler.py:172-178`), so DESIGN §7.4's curriculum refresh and ARCHITECTURE M8's retirement loop are open.
- **D22 [ran p6] The World sandbox pool hangs or hands out dead containers.** `SandboxPool._start_one` swallows a start failure into an unretrieved task exception (`C/world/pool.py:108-114`); `warm()` waits until the queue is full with no limit (`:127-131`); `acquire()` hands out a queued container with no liveness check (`:133-143`). Probe: daemon down → `warm()` still waiting after 2 s with `starting=0, ready=0`; a container that died in the queue is handed out.
- **D23 [read] No minimum usable group size.** `group_backfill="pad"` trains whatever members survive (`C/train/scheduler.py:18-20,56`); only an all-masked group is evicted. AReaL drops groups below `min_usable_group_size` (`AR/areal/infra/remote_inf_engine.py:228-232`).

## 4. Probe index (`/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m810sweep/g4/`)

Run from `crucible/` with `.venv/bin/python <probe>`. No network, no GPU, no paid calls.

| Probe | What it shows | Result |
|---|---|---|
| `p1_runner_exceptions.py` | (a) one `OSError` from a harness among 6 samples; (b) `crucible:harness_unsupported`; (c) blame of an unmapped foreign kind; (d) a hung harness with `wall_clock_s=0.2` | (a) `Runner.run` raises, 6 ledger ops `reserved`, 0 results rows (D1); (b) `ValueError: unknown failure kind` (D2); (c) `agent_run_error`, blame `harness`; 40/40 trip only the mask-fraction rule (D8); (d) still running after 2 s (D3) |
| `p2_train_env_exception.py` | M12 token env raising `EnvInfraError` vs `ValueError` at 5% of steps | `EnvInfraError`: 6 steps, 0.5 s, 18 masked; `ValueError`: 0 steps, `wall_clock` after 8 s, `failures={}`, 3 rollouts stuck `running` (D10) |
| `p3_bench_band.py` | A10's bench band vs `ladder.accept(split="bench")` on 0/2, 0/8, 0/15 | A10 in band every time; ladder rejects all three (upper bounds 0.66, 0.32, 0.204) (D15) |
| `p4_forge_role_exception.py` | `ConnectionError` in one of two concurrent forge plans | at HEAD: `run_plans` raises; both tasks stay `draft`; both reservations stay in `Ledger.reserved` (D20). On the later uncommitted M8-audit tree: t-ok constructed, t-flaky abandoned |
| `p5_crash_resume.py` | a crashed attempt with `checkpoint.json` and no `record.json` | moved to `aborted/a1`; `_resumable` → None; control (masked agent-stage record) → resumable (D4) |
| `p6_sandbox_pool.py` | `SandboxPool.warm()` with failing starts; `acquire()` after a queued container died | `warm()` waits forever; the dead container is handed out (D22) |
