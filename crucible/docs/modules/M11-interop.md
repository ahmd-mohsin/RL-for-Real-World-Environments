# M11 `interop` — the boundary with every framework

**Status: built in parts (2026-10-10 / 10-11); audited 2026-10-12 (`docs/audit/2026-10-12/M11-interop.md`).**
- **Audit:** every foreign path graded a World with a recorded infra fault and scored it; imported frameworks' own
  failures scored 0; the policy could reach the grader (`verify` as a NeMo-Gym tool), the agentenv control plane, and
  planted reward / mask / test files; separate-verifier Harbor tasks graded in the agent's container. All fixed, with
  25 regression tests.
- **Exit gate ⏳:** the toy cross-runtime gate passes (native, Harbor ATIF replay, agentenv, NeMo-Gym, OpenEnv; verifiers
  keeps the reward and raises on a mask). task02 through foreign runtimes, a live `vf-eval` episode, NeMo-Gym's own
  environment server driving the export, and a MiMo export from a World are not built or not run.
- **Code:** `src/crucible/interop/` (the only package that imports external frameworks), `src/crucible_vf/` (the
  verifiers plugin id). Framework Pythons: `crucible/.envs/{openenv,verifiers,nemo}`.
- **Research:** `docs/research/m11-m12/05-interop-diversity.md`; sweeps `docs/audit/2026-10-12/M11-sweep-G1..G5.md`.

## Rules every path follows

1. **One reward producer.** Grades go through the task's own grader and `verify.runtime.plan.write_reward_files`; a
   masked grade writes no `reward.json` (B.17).
2. **Masks travel as masks.** An infrastructure fault recorded in the World (`World.failures`), a grading exception, a
   foreign server's own failure, a non-finite or out-of-range foreign reward: each is a mask, carried in the target's
   own vocabulary (`mask_sample`, `TaskError`, a raised `CrucibleMaskedEpisode`, a failed agentenv run). Where a target
   cannot carry a mask (a consumer that sums step rewards and reads `None` as 0), the episode raises.
3. **The policy's failures are scored.** A rejected action, a 4xx tool reply, an answer that breaks a verifier, a
   container the policy broke: scored 0. A health check decides when it is ambiguous (OpenEnv: replay the journal; AWM:
   run the verifier on the untouched DB; Harbor in place: can the verifier user still write its reward).
4. **The policy never reaches the grader or the control plane.** Agents get role-pinned `/mcp` (agentenv proxy; the
   agentenv gateway itself listens on loopback only, and the actor tokens Crucible carries to App servers are tool-bound
   and single-use: M11 lab audit R-2), outside agents are cut before every grade (`agentenv/cut.py`: every policy actor
   done, later calls refused as `late_call`, quiesce, A2A cancel; `quiesce.json`), one grade per session ends it (NeMo-Gym), the grader's directories are emptied and grader-only files removed before a
   grade (Harbor, verifiers), separate-verifier tasks are refused in place, policy SQL cannot ATTACH or VACUUM INTO.
5. **Framework semantics are the framework's.** Harbor's reward rules (`harbor/rewards.py`), network phases, verifier
   env and user; NeMo-Gym's pydantic models; verifiers' `ok` and branches; EnterpriseOps-Gym's `_compare_values`;
   MiMo's `detail.passed` and range mask.

## Masks per target

What a masked episode becomes in each target's vocabulary (rule 2). A mask is never a number in an optimizer baseline, and
no row reaches a trainer with a `None` where the trainer does arithmetic.

| Target | A masked episode becomes | Where |
|---|---|---|
| NeMo-Gym export | `mask_sample: true` at the top level and in `instance_config` (NeMo-RL reads the latter) | `nemo_gym/server.py` |
| verifiers (`crucible-vf`) | `TaskError("crucible:masked:<kind>")`: an errored trace, dropped by prime-rl | `verifiers/vf_taskset.py` |
| OpenEnv export | a raised `CrucibleMaskedEpisode`, never a reward | `openenv/export.py` |
| agentenv | a failed run; the verification row has `skipped: true` | `agentenv/plugin/steps.py` |
| Harbor | no `reward.json`, a `mask.json`; Harbor's own Mean scores a missing reward 0, so read `metrics.MASKED_MEAN_PY` (no production caller yet) | `verify/runtime/plan.py`, `harbor/metrics.py` |
| verl v1, `crucible_delta` | one placeholder row with no trainable token (`response_mask=[0]`, reward 0.0), out of the optimizer's group mean (verl's reported score mean still counts its 0.0); `min/max_global_steps` = the trainer's current step | `train/bridges/verl_agent_loop.py` |
| verl v1, any other estimator | `[]`: v1 skips the session and the uid group stays finished, so the 0.0 never joins a stock group mean. A uid whose every session is masked has no row, so `algorithm.filter_groups` (which raises for such a uid) is refused with these estimators; a batch whose every uid is fully masked was not checked downstream | same |
| verl legacy worker (`use_v1: false`, or an entrypoint that runs it under `use_v1: true`) | refused unless the estimator is `crucible_delta` (no skip channel; deprecated upstream): when verl builds the loop for the first sample, or at the first legacy call | same |
| TRL | a placeholder row with `crucible_reward=None`: NaN, out of the baseline, advantage 0 | `train/bridges/adapters.py` (`trl_rollout`) |
| SkyRL | `step_wise`: no row; one row per input: a loss-masked row whose reward is the group's live mean (exact for Dr.GRPO, approximate for GRPO std and RLOO) | `train/bridges/skyrl_bridge.py` |
| rLLM | left out of every `TrajectoryGroup` | `train/bridges/rllm_bridge.py` |

verl rows of frozen actors only (no sampled token, so no policy version) take the min and max steps of their session's
other rows, else the current step: v1 converts every tag with `dtype=int` after the actor update (M11 lab audit R-1).

## Pieces

| Path | Direction | What |
|---|---|---|
| `canonical.py`, `replay.py` | both | canonical JSON and tool names, derived `call_id`s; replay of any transcript (World trace, ATIF, Responses, OpenEnv calls, agentenv events) in a fresh World, final answers included; an incomplete replay masks `crucible:replay_incomplete` |
| `harbor/` | import + export + harness | `agent_harness` (Harbor installed agents as a harness, never Harbor Trial/Job: M10 lab audit R-1), `patch_task` (the separate-verifier patch seam: collect hooks, then `verify_dir` in a fresh container), `reconcile` (OpenEnv's ATIF reconcile, ported: capture vs ATIF), `task_io` (task dirs), `inplace` (agent and Harbor's verifier in one container; dirs reset, agent phase applied, agent-phase hosts admitted for the capture proxy, separate verifiers refused in place (`patch_task` grades them from a collected patch); the grader's toolchain held to the pristine image host-side and container loss / grade-time network loss masked: M13 audit B1, B2), `grader_guard` (that toolchain check: `crucible:grader_tampered`), `multi_step`, `verify_dir` (patch mode with Harbor's rules), `rewards` (Harbor's reward parser), `atif` / `atif_import` / `trajectories`, `metrics` (masked mean), `world_backend` / `gpu_docker`, `role_runner` |
| `agentenv/` | export + harness | `world_backend` (Apps behind a real gateway; outside calls are World steps; `AgentProxy`), `plugin/steps` (`crucible_world`, `crucible_verify`), `harness` (A2A agents as a harness; blame from capture evidence, agent liveness: M10 lab audit R-1), `cut` (the pre-grade cut and its `quiesce.json` receipt: M11 lab audit R-2), `agents`, `mcp_env`, `sandbox`, `a2a` |
| `verifiers/` | export + import | `vf_taskset` (the `crucible-vf` plugin: `CrucibleTaskset`; separate-verifier tasks only in verifiers' fresh box, shared ones after a quiesce, `box`; gold via `validate`; built at a pinned verifiers by `scripts/build_verifiers_env.sh`: M11 lab audit R-3), `taskset` (`build()`), `v1_import` (verifiers traces as `crucible.rollout/v1`) |
| `nemo_gym/` | export + import | `server` (resources server), `client` (remote Worlds for the M12 runtime), `import_server` (any NeMo-Gym server as an App + task), `serve_resource` |
| `openenv/` | import + export | `import_env` (any OpenEnv `Environment` as an App, replay-journal state), `export` (a Crucible task as an `Environment`), `awm` / `awm_task` / `awm_check` (AgentWorldModel), `catalog` |
| `mimo/` | import | `importer`, `verify` (MiMo's own `verify.py` in an isolated subprocess), `check`, `export` (source bytes only) |
| `eog/` | import | EnterpriseOps-Gym tasks as SQLite Worlds with typed columns and EOG's comparison rules; `oracle` witnesses |

## Open

- `interop/imports/` (I0-I10): dataset provenance, floors, flake screening, leak audit, import reports (design:
  `M11-sweep-G5.md` §4.2).
- A separate-verifier grade (Harbor's `_run_separate_verifier` path) for in-place tasks.
- NeMo-Gym packaging (`manifest.yaml`, config, dataset rows), ATIF v1.7, a gate driven by `single_agent_turn`.
- Exports: MiMo from a World (`DatasetEnvActor` rows), verl AgentLoop and reward manager, SkyRL step-wise, TRL
  `rollout_func`; the NeMo-RL bridge must fill masked rows before GRPO baselines (M12).
- Replay `call_id`s per model call (they are per action); locked framework environments.
