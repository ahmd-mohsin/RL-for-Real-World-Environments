# M11 framework sweep, group 1: Harbor, OpenEnv, FineEnvs and the SWE task ecosystem (2026-10-12)

Read-only, Crucible HEAD `af402b4`. One reader. Every external file:line below was opened. Eight probes reproduce the
defects marked [ran]. They live in `scratchpad/m11sweep/g1/`. Probes p1 to p4 and p7 run with
`crucible/.venv/bin/python` from `crucible/`. Probes p5, p6 and p8 run with `crucible/.envs/openenv/bin/python`. No
Docker, GPU, network or paid API was used. Docker and the Harbor environment are stubbed where a probe needs them.

Scope: what M11 must honour to import from and export to Harbor and OpenEnv, the FineEnvs PortSim import, and the
task and trajectory formats of Repo2RLEnv, SWE-smith and mini-swe-agent. agentenv, verifiers, NeMo-Gym, MiMo and
EnterpriseOps-Gym belong to other groups. They appear here only where one of my sources depends on them.

Roots and pins:

| Root | Repo | Commit | Licence | How Crucible gets it |
|---|---|---|---|---|
| `H` | `RL-Environments/06-frameworks-and-env-code/harbor-framework-harbor/src/harbor` | a2fafbc (2026-10-03), v0.23.0 | Apache-2.0 | editable path in `crucible/.venv` (`pyproject.toml:90`); the requirement is a floor only, `harbor>=0.22.0` |
| `HA` | `.../harbor-framework-harbor/adapters` | a2fafbc | Apache-2.0 | not used |
| `OE` | `.../meta-pytorch-openenv/src/openenv` | 49aa302 (2026-10-02), 0.7.1.dev0 | BSD-3-Clause | editable path in `crucible/.envs/openenv`; no pin or lock anywhere |
| `FE` | `.../adithya-s-k-fineenvs` | 5e0b46e (2026-10-08) | Apache-2.0 code; PortSim data CC BY-SA 4.0 | `berth-core`, `berth-openenv` 0.1.0 editable in `.envs/openenv` |
| `PS` | `FE/07-simulation-environments/portsim-v1/envs/berth_planning` | 5e0b46e | as `FE` | as `FE` |
| `R2` | `.../huggingface-repo2rlenv/src/repo2rlenv` | 5555443 | Apache-2.0 | not used (probe only) |
| `SS` | `.../swe-bench-swe-smith` | 9b74ac0 | MIT | not used |
| `MS` | `.../swe-agent-mini-swe-agent/src/minisweagent` | 04d809c | MIT | through Harbor's converter only |
| `SP1` `SP2` `SP3` | the three HF Spaces (simulation, multi-harness, guide) | 46bb2be, b780039, ead51a4 | CC BY 4.0 (articles) | no code |
| `VF` | `.../primeintellect-ai-verifiers/verifiers/v1` (cited for one consumer rule) | ae819de | MIT | `.envs/verifiers` |
| `CR` | `crucible/src/crucible` | af402b4 | ours | |

All framework trees were clean (`git status` empty), so the editable installs equal the pinned commits today. Nothing
records them for the gate run: `.envs/{openenv,verifiers,nemo}` have no lock file and no build recipe in the repo.
`fastmcp` 3.4.8 and `mcp` 1.30.0 come from PyPI unpinned.

Verdicts: **DEPEND** (call their code), **PORT** (copy; licence permits), **PATTERN** (rebuild the idea), **nothing**
(Crucible is equal or ahead), **anti-pattern** (do not copy). Severity: **P0** wrong reward or a reward hack on a
path the spec names; **P1** a contract break, a vacuous gate or an import that grades the wrong thing; **P2** a gap,
a leak or a smaller divergence.

## 0. Corrections and confirmations of the spec

Confirmed:
- **One reward producer** (DESIGN B.17). `write_reward_files` (`CR/verify/runtime/plan.py:359-386`) is the only writer
  on the replay, OpenEnv export and gate paths. A masked grade writes no `reward.json` or `reward.txt`, and a
  non-finite value masks (`:366-371`). The file is a flat dict of finite floats (`:324-341`), which Harbor accepts
  (`H/verifier/verifier.py:89-124`).
- **ATIF out and in use Harbor's own models** (`CR/interop/harbor/atif.py:15`, `atif_import.py:25`). Exports are v1.8
  and the studio export runs Harbor's `TrajectoryValidator` (`CR/studio/export.py:94-108`).
- **The OpenEnv import fixes Harbor-style async forwarding** (`CR/interop/openenv/import_env.py:129-146`). The base
  `reset_async` forwards `seed` and `episode_id` to the sync method (`OE/core/env_server/interfaces.py:268-279`).
- **The PortSim catalogue entry matches the environment.** Tools, the 10-check and 24-call budgets, `submit_plan`
  ending the episode, `tool_call_limit` → reward 0.0, and `metadata.grade.feasible` all agree
  (`PS/openenv/berth_openenv/environment.py:148-191,259-279` vs `CR/interop/openenv/catalog.py:20-40`).

Corrections:
- **"Gate (toy) ✅ … byte-identical through … Harbor ATIF … and OpenEnv, with the same final state fingerprint"**
  (DESIGN M11 part 1, line ~866) overstates it. The OpenEnv leg writes `reward.json` and `steps.jsonl` by replaying
  the sent actions in a fresh native World (`packs/_toy/src/crucible_pack_toy/m11/gate.py:50`). The environment's
  own reward document is never compared. The fingerprint assertion compares `path.json` with itself
  (`tests/unit/test_interop_gate.py:183`). The Harbor leg never runs Harbor: it builds an in-memory ATIF model and
  replays it (`:38-40`). See D3 [ran p6].
- **"Masked grade → reward None, never a 0"** for the OpenEnv export (DESIGN M11 part 1; `CR/interop/openenv/export.py:11-12`)
  holds for the last observation only. The earlier per-step rewards are already with the consumer. A summing
  consumer scores a masked episode as Φ before the mask. See D1 [ran p5].
- **"Steps that need a separate verifier environment are refused"** (DESIGN M11 importers, line ~2569) holds for
  `HarborMultiStepTask` only (`CR/interop/harbor/multi_step.py:71-73`). The single-step `HarborInPlaceTask` takes them
  and grades in the agent's container. See D4 [ran p2].
- **ARCHITECTURE M11 "openenv: an Environment plus rubric tree plus `openenv.yaml`"** and **"harbor: … job bridge"**
  are not built. `grep` finds no rubric, no `openenv.yaml` writer and no Harbor `Job`/`Trial` use outside the forge
  role runner (`CR/interop/harbor/role_runner.py:204-207`). Harbor `RolloutDetail` is never read.
- **Research §6.2 item 10 negative control** ("a forced masked grade must surface as each runtime's native mask and
  never as 0") is tested for NeMo-Gym (`test_interop_gate.py:73-90`) and verifiers (`:134-168`) only. Not for OpenEnv
  and not for Harbor.
- **The task02 leg of the exit gate cannot run on stock Harbor.** task02 declares a separate verifier with no image
  and no `tests/Dockerfile`. See D5 [ran p2].

## 1. Contracts and mechanisms

### 1.1 Harbor (`H`): task format, verifier, ATIF, jobs

Contract, field by field. "In place" is `HarborInPlaceTask` / `HarborMultiStepTask`; "patch path" is `HarborDirTask`.

| Harbor field or rule | Source | Crucible | Match |
|---|---|---|---|
| `schema_version` default 1.4; `version` renamed; no feature gating by version | `H/models/task/config.py:801-802,846-848` | recorded in the summary (`CR/interop/harbor/task_io.py:91`) | yes (task02 says 1.3 and uses 1.4 fields; Harbor does not care) |
| Unknown keys are dropped silently | pydantic default | reported as `ignored_keys` (`task_io.py:27-48,105`); task02 drops `verifier.command` and `environment.dockerfile` | reported, not refused (D5) |
| `[environment].network_mode` default **public**; `[agent]` / `[verifier]` `network_mode` are phase overrides | `config.py:219-270` | in place: Harbor's `resolve_trial_network_plan` (`inplace.py:122-134`); patch path: `[verifier].network_mode` or **no-network** (`verify_dir.py:66,103-104`) | in place yes; patch path no (D6) |
| `[verifier] environment_mode` shared or separate; separate builds its image from `tests/` unless `[verifier.environment].docker_image`; tests are in the image (`skip_tests_upload`) | `config.py:555-614`; `H/models/task/verifier_mode.py:43-64`; `H/trial/trial.py:885-1022` | multi-step refuses separate; single-step in place grades in the agent container; patch path mounts `tests/` into the agent image | no (D4, D6) |
| `[[verifier.collect]]` hooks run before artifact collection, in any mode | `config.py:590-599,726-757`; `trial.py:1417-1464` | not run anywhere in `CR/interop/harbor/` (p2: "referenced by inplace.py: False") | no (D4) |
| `[verifier].env`, `user`, `timeout_sec` | `config.py:562-568`; `verifier.py:182-197` | in place: Harbor's `Verifier` merges `env`; `user` set (`inplace.py:211`); patch path ignores `env` and `user` | in place yes; patch path no |
| Reward: `reward.json` wins over `reward.txt`; JSON must be a dict of finite numbers; text one finite float; empty or missing file raises | `verifier.py:66-124,257-268` | in place: DEPEND on `Verifier`; patch path: own parser (`verify_dir.py:121-134`) | in place yes; patch path no (D6 [ran p3]) |
| Shared single-step verifier does not empty `/logs/verifier` first; multi-step does (`_reset_shared_step_verifier_dirs`) | `H/trial/single_step.py:89-114`; `trial.py:768-776,826-833` | multi-step copies it (`multi_step.py:181-182`); single-step does not (`inplace.py:190-213`) | Crucible copied Harbor's hole (D2 [ran p1]) |
| `[[steps]]`, `min_reward` (float on `reward`, dict per key, missing = −inf), `multi_step_reward_strategy` mean (per key, executed steps) or final (last executed step) | `config.py:760-829`; `H/trial/multi_step.py:224-241`, `trial.py:787-824` | DEPEND on `_min_reward_failure` (`multi_step.py:56-60`); `mean_executed` over the scalar; `final` scores 0 when stopped early | documented divergence (`multi_step.py:36-38`); Harbor averages per key, Crucible the chosen scalar |
| Per-step `agent.timeout_sec`, `artifacts`, `trajectory.json` load, resume | `H/trial/multi_step.py:394-418` | kept in `step_meta`, not enforced (`multi_step.py:89-93`) | documented |
| ATIF v1.0 to v1.8; step ids from 1; `is_copied_context` steps are not training data; `llm_call_count` 0/1/>1 | `H/models/trajectories/trajectory.py:15-26,128-138`; `step.py:63-87` | export v1.8 (`atif.py`, `trajectories.py`); trace import skips copied context (`atif_import.py:85-86`); replay import does not (`CR/interop/replay.py:127-137`) | partly (D10 [ran p7]) |
| Job metrics: `Mean` and friends count a missing reward as 0; multi-key dicts are averaged per key | `H/metrics/base.py:14-35` | `MASKED_MEAN_PY` exists (`CR/interop/harbor/metrics.py:19-44`) but nothing ships it | open (D11) |
| pass@k: returns `{}` unless every trial has one 0/1 key | `H/utils/pass_at_k.py:30-66` | Crucible `reward.json` has many keys (`plan.py:336-341`), so Harbor reports no pass@k for any Crucible task | silent (D11) |
| `UvScript` metric: one JSON row per trial, `null` for no reward | `H/metrics/uv_script.py:17-47` | `MASKED_MEAN_PY` reads that format | yes |

Mechanisms:

| Mechanism | Source file:line | Crucible (file:line) | Verdict |
|---|---|---|---|
| Typed reward parsing with finite checks | `H/verifier/verifier.py:66-124` | used in place; re-implemented loosely in `verify_dir.py:121-134` | DEPEND everywhere (pick 2) |
| Separate verifier: fresh env from `tests/`, artifacts uploaded, only `/logs/verifier` mounted, per-phase network | `H/trial/trial.py:885-1022` | `world/verifier.py` custody exists (THIRD_PARTY row 29) but is not wired to Harbor imports | PATTERN (pick 3) |
| Shared-dir hygiene between phases | `H/trial/trial.py:826-833`; `multi_step.py:186-193` | multi-step only | DEPEND, also for single step (pick 4) |
| Adapters: 88 benchmark → Harbor converters (swebench, swesmith, swegym, swebenchpro, …) | `HA/` (e.g. `HA/swesmith/src/swesmith_adapter/task-template/`) | none | DEPEND as the SWE import route (pick 10) |
| Adapter `test.sh` installs tools from the network at grade time (`apt-get`, `curl … uv`, `uv add`) | `HA/swesmith/src/swesmith_adapter/task-template/tests/test.sh:3-17` | patch path forces no network (D6) | anti-pattern for hermetic grading; still the contract imports must meet |
| `Mean` / `pass@k` treat no reward as 0 | `H/metrics/base.py:24-35`; `H/utils/pass_at_k.py:39-41` | masked mean exists, unwired | anti-pattern (D11) |
| ATIF converters for 40+ harnesses | `H/agents/installed/` | DEPEND (`atif_import.py:132-145`) | nothing |

### 1.2 OpenEnv (`OE`): Environment, MCP, rubric, manifest

Export contract (`CR/interop/openenv/export.py`):

| OpenEnv rule | Source | Crucible export | Match |
|---|---|---|---|
| `reset(seed, episode_id, **kw)`; `seed` for reproducible episodes | `OE/core/env_server/interfaces.py:262-279`; `types.py:90-104` | `seed` only names the episode (`export.py:74`); the World ignores it | no (D8) |
| `step(action, timeout_s, **kw)` | `interfaces.py:281-302` | `timeout_s` ignored (`export.py:124-128`) | no (D8) |
| `state` property; `SUPPORTS_CONCURRENT_SESSIONS` | `interfaces.py:247,306-309` | `State(episode_id, step_count)`; `True` | yes; but each instance leaks a thread and a temp dir (D8 [ran p5]) |
| `Observation{done, reward: bool/int/float/None, metadata}` | `types.py:72-88` | yes | yes |
| `CallToolObservation.error` carries transport and framework failures | `OE/core/env_server/mcp_types.py:198-219,269-284` | never set; `ok` lives in `metadata.crucible.ok` (`export.py:112-113`) | no (D8 [ran p5]) |
| MCP JSON-RPC works on a plain `Environment` whose action class is `CallToolAction`, by routing through `step` | `OE/core/env_server/http_server.py:1121-1160,1214-1240` | plain `Environment` (`export.py:47`) | yes |
| Plain HTTP `/reset` and `/step` build a new env per request | `http_server.py:1113-1115`; `OE/harbor/environment.py:1-5` | `step` before `reset` dereferences `self._world = None` | undocumented (D8) |
| `rubric` attribute, `named_rubrics()` for trainers | `interfaces.py:213-260`; `OE/core/rubrics/` | `rubric` is None (p5 d) | no (D12) |
| `openenv.yaml` with `validation.reward.range`, `oracle_tolerance`, capabilities | `OE/validation/manifest.py:41-70,285-302`; `PS/openenv/openenv.yaml` | not written (`CR/studio/export.py:53-68` writes a Python launcher only) | no (D12) |
| A reward of None means unscored | `OE/core/harness/rollout.py` (TrainingTrace); research §2.2 | final reward None when masked (`export.py:119-122`) | yes for the last step; earlier steps leak (D1) |

Import contract (`CR/interop/openenv/import_env.py`):

| OpenEnv rule | Source | Crucible import | Match |
|---|---|---|---|
| MCP errors: `execution_error`, `invalid_args`, `transport_error`, `tool_not_found`, `timeout`; picked by keywords in the exception text | `mcp_types.py:198-206`; `mcp_environment.py:562-596` | policy for invalid args / not found, tool for execution, **infra for timeout and the rest** (`import_env.py:474`) | partly: a policy that makes a tool hang masks its own episode (D9) |
| Default MCP tool timeout, raised only by shadowing `step` | `OE/harbor/environment.py:21-23` | `_call(self.env, "step", action)` passes no timeout (`import_env.py:365`) | gap |
| `done` with `reward=None` | `types.py:79-81` | scored 0 with reason "never terminated" (`import_env.py:577-592`) | no (D9 [ran p8]) |
| Rewards may be negative (games, AWM −1.0) | `OE/envs/agent_world_model_env/server/config.py:31-36` | clamped into `reward_range`, default (0, 1) (`import_env.py:592`) | lossy by default (D9 [ran p8]) |

Mechanisms:

| Mechanism | Source file:line | Crucible (file:line) | Verdict |
|---|---|---|---|
| Harbor tasks served as one long MCP tool `run_rollout`, because HTTP handlers are per request | `OE/harbor/environment.py:1-5,21-23,44-60` | none | PATTERN for a Harbor-task OpenEnv export (D12) |
| "Nothing raises out of `run_rollout`"; per-key finite filter; an unusable key is excluded, never 0; refuse to pick among several keys without `reward_key` | `OE/harbor/rollout.py:130-193,505-553` | `scalar_reward` (`inplace.py:82-92`) checks `reward` before `reward_key` | PORT the selection order and per-key filter (pick 8; D10 [ran p7]) |
| ATIF ↔ capture reconcile: per-turn token counts, subsequence gaps, `llm_call_count > 1` = retries, `merge_into_atif` | `OE/harbor/atif.py:1-40,111-232,498` | `cross_check` counts only `llm_call_count == 1` steps (`CR/interop/harbor/trajectories.py:117-127`) | PATTERN (pick 7; D15) |
| `TrajectoryRubric.compute_step_rewards` and `ExponentialDiscountingTrajectoryRubric` | `OE/core/rubrics/trajectory.py:22-136,138-207` | Ψ-telescoping in the env itself | TARGET for D12 |
| Manifest v2 `RewardDeclaration` (range strictly increasing, oracle tolerance, variance tolerance for judged envs) | `OE/validation/manifest.py:41-70,285-302` | none | TARGET (pick 6) |
| AWM keyword rewards (−1.0 for "invalid"), `judge_error` → 0.0 | `OE/envs/agent_world_model_env/server/awm_environment.py` (prior G1 §1.6) | AWM used as data | anti-pattern; an import must not inherit it |

### 1.3 FineEnvs (`FE`) and PortSim (`PS`)

| Mechanism | Source file:line | Crucible (file:line) | Verdict |
|---|---|---|---|
| One environment core, six framework adapters, each with its own `pyproject.toml`, `uv.lock` and `tests/` | `FE/00-environments-101/envs/wordle/{core,openenv,nemo_gym,verifiers,skyrl_gym,gem,ors}` | `.envs/*` are hand-built, unlocked | PATTERN (pick 5) |
| Adapter drift: the NeMo-Gym adapter re-derives the reward from the transcript with a constant | `FE/00-environments-101/envs/wordle/nemo_gym/server.py:155-184` | Crucible grades server-side state (N6) | anti-pattern (Crucible ahead) |
| Import pins: instruction and grader hashes, per-file hash manifest, dataset revision | `FE/05-multi-harness-rl/prepare.py:46-121` | MiMo import has a manifest; OpenEnv and Harbor imports record none | PATTERN (pick 9; D13) |
| Eval identity: protocol source sha256, instruction hash | `FE/05-multi-harness-rl/eval/evaluate.py:191-210` | none for imports | PATTERN (pick 9) |
| PortSim: MCP tools, check budget, `submit_plan` ends and grades once, tool-call limit → done with 0.0, global `STORE` of episodes | `PS/openenv/berth_openenv/environment.py:119-191,217-244,259-279` | catalogue matches (§0); every restore adds one more episode to PortSim's process-global `STORE` | nothing; note the growth on long runs |
| PortSim data licence CC BY-SA 4.0 (share-alike) | `FE/07-simulation-environments/portsim-v1/README.md` | named in a docstring only (`catalog.py:4`); not in `THIRD_PARTY.md`, not on any task card | gap (D13) |

### 1.4 Repo2RLEnv (`R2`), SWE-smith (`SS`), mini-swe-agent (`MS`)

| Format or mechanism | Source file:line | Crucible (file:line) | Verdict |
|---|---|---|---|
| Repo2RLEnv emits Harbor tasks: `version = "1.0"`, no `network_mode` (Harbor's public default), `[metadata.repo2env]` provenance; lite tasks have no `tests/` or `environment/` | `R2/emitter/harbor.py:75-191` | in-place import keeps public network and drops `remote_code_fetch` interception (D7 [ran p4]) | import route: DEPEND through Harbor; network rule needed |
| SWE-smith instance schema (instance_id, repo, patch, FAIL_TO_PASS, PASS_TO_PASS) reaches Harbor through Harbor's adapter | `HA/swesmith/src/swesmith_adapter/task-template/` | no SWE-bench-format importer; none needed if the adapter route is used | DEPEND (pick 10) |
| SWE-smith mirrors repos on GitHub; the history leak through a public branch | prior `docs/research/m7-m8/02-forge-synthesis.md:183-200` | fetch interception is off for public in-place imports (D7) | fixture for D7 |
| mini-swe-agent trajectory `mini-swe-agent-1.1`: `info.exit_status`, `info.submission`, `role: "exit"` messages; an uncaught exception's class name becomes the exit status | `MS/agents/default.py:74-86,100-150,160-179` | Harbor's converter (DEPEND) drops `role: "exit"` and `exit_status` (`H/agents/installed/mini_swe_agent.py:306-424`); the coding pack reads `exit_status` itself (`packs/coding/.../connectors/agent_runs.py:91-103`) | D14 |

### 1.5 The three Spaces

`SP1`, `SP3`: articles. `SP2` (`app/src/content/chapters/training.mdx:340-386`) is useful for M11 only as evidence:
Harbor-served multi-harness runs record per-harness tokens, and an honest report bounds the score "even counting every
ungraded cell as correct". Already picked as M10-6 in the M8–M10 sweep. Nothing new to take.

## 2. Defects (Crucible, HEAD af402b4)

- **D1 P0 [ran p5] A masked OpenEnv export episode scores a full reward under a summing consumer.**
  `CR/interop/openenv/export.py:108-122`. Each step returns `reward = ΔΦ`. Only the final step knows the grade, and a
  mask sets that one to None. verifiers' OpenEnv taskset sums `result.reward or 0.0`
  (`VF/tasksets/openenv/taskset.py:128-129`). Probe: the toy gold episode with a masked grade gives step rewards
  `[0.4, 0, 0.2, 0, 0, 0.2, 0, 0.2]`, final None, `masked=True`, and the summing consumer scores **1.000**. This is
  the B.17 hazard again on a named export path. Fix: keep ΔΦ out of `Observation.reward` by default (put it in
  `metadata.crucible`; intermediate `reward=None`; the final step carries the whole reward). Offer telescoping only
  as an opt-in for consumers that drop an episode whose last reward is None. On a mask, also raise a typed error from
  the finishing `step`, so a summing consumer errors the trace. Add the masked negative control for OpenEnv to the
  gate.

- **D2 P0 [ran p1] A policy can plant `reward.json` in a single-step in-place Harbor task.**
  `CR/interop/harbor/inplace.py:190-213` runs Harbor's `Verifier` without emptying `/logs/verifier`.
  `HarborMultiStepTask._grade_step` does (`multi_step.py:181-182`). Harbor reads `reward.json` before `reward.txt`
  (`H/verifier/verifier.py:257-262`), so a planted `{"reward": 1}` beats the test's `reward.txt = 0`. The
  `grader_paths` rule is a regex over argument text (`CR/world/intercept.py:55-57`): `cd /logs && … verifier/reward.json`,
  `$d/verifier/…` and `/logs//verifier/…` all pass. The in-place spec sets no `isolation.policy_user`
  (`inplace.py:136-152`), so the policy runs as the task's agent user, root by default. Probe: Harbor's own `Verifier`
  returns `{'reward': 1.0}`; three of four planting commands pass the interceptor. Stock Harbor has the same hole in
  single-step shared mode (`H/trial/single_step.py:89-114`): do not copy it. Fix: call `env.empty_dirs([verifier_dir],
  chmod=True)` and empty `/tests` before verifying, as `multi_step._reset_shared_dirs` does; set a non-root policy user
  where the image allows; add a red-team case for the in-place path.

- **D3 P1 [ran p6] The exit gate's OpenEnv and Harbor legs pass by construction.**
  `packs/_toy/src/crucible_pack_toy/m11/gate.py:50` writes the OpenEnv leg's `reward.json` and `steps.jsonl` with
  `replay()` over the sent actions in a fresh native World. The environment's own grade (`metadata.crucible_reward`)
  is never compared with native. `tests/unit/test_interop_gate.py:183` compares `path.json`'s fingerprint with
  itself. The Harbor leg (`:38-40`) never runs Harbor or serializes ATIF to JSON. Probe: the exported environment pays
  0.5 instead of 1.0; every assertion of `test_openenv_export_identical_and_telescoping` still holds (environment
  final 0.5, native `reward.json` 1.0). Fix: compare the runtime's own outputs. For OpenEnv, the environment's
  `write_reward_files` directory and its per-step rows, plus `fingerprint` against the native run. For Harbor, run
  Harbor's `Verifier` (or a `Trial` with the oracle agent) on the toy exported as a Harbor task, and round-trip ATIF
  through `model_dump_json` / `model_validate_json`.

- **D4 P1 [ran p2] `HarborInPlaceTask` grades separate-verifier tasks in the agent's container and never runs
  `[[verifier.collect]]`.** `CR/interop/harbor/inplace.py:95-121,190-213`. It resolves the verifier mode only for the
  network plan. task02 (`environment_mode = "separate"`, one collect hook) constructs fine and would be graded by
  `Verifier(environment=world.backend.env)`. Harbor grades it in a fresh container after the collect hooks and the
  artifact upload (`H/trial/single_step.py:38-55`; `trial.py:885-1022`). Harbor also runs collect hooks in shared mode
  (`trial.py:1423-1464`); neither in-place class runs them. Fix: refuse separate mode as `multi_step.py:71-73` does,
  or route it through `world/verifier.py` custody with Harbor's rules. Run `[[verifier.collect]]` before grading in
  both modes.

- **D5 P1 [ran p2] task02 is not a runnable Harbor task, yet `load_harbor_task` accepts it.**
  `migration_tasks/task02_pcops_triton/task.toml` has a separate verifier with no `[verifier.environment].docker_image`
  and no `tests/Dockerfile`. Harbor builds the separate verifier from `tests/` (`trial.py:1011-1021`) and refuses a
  context with no definition (`H/environments/definition.py:56-73`). Probe: `has_agent_environment_definition(tests/)`
  is False. `[verifier].command` is a Crucible key Harbor drops (`ignored_keys` lists it). So the "task02 through
  Harbor" leg of the exit gate cannot start. Fix: `load_harbor_task` runs Harbor's definition check on the agent env
  and on the effective verifier env, and an export mode turns `ignored_keys` into errors. Give task02 a verifier image
  or `tests/Dockerfile`.

- **D6 P1 [ran p3] The patch path (`HarborDirTask`) parses rewards and picks the network unlike Harbor.**
  `CR/interop/harbor/verify_dir.py:59-66,103-104,121-134`. Probe, same files through Harbor's parser and ours:
  a multi-key `reward.json` with no `reward` key and a single `score` key both crash with `TypeError` (Harbor accepts
  them); `{"reward": NaN}` gives `reward=nan, infra=ok` (Harbor raises); a non-numeric extra key and a two-line
  `reward.txt` both give 1.0 (Harbor raises); an empty `reward.json` crashes with `JSONDecodeError` (Harbor raises
  `RewardFileEmptyError`). Network: a task with no verifier network (Harbor's swesmith adapter, Repo2RLEnv) is public
  in Harbor and `no-network` here, while its `test.sh` installs tools with `curl` and `uv`; every grade would mask.
  `[verifier].env`, `user` and `[verifier.environment].docker_image` are ignored. Fix: DEPEND on
  `Verifier._parse_reward_json` / `_parse_reward_text` plus `scalar_reward`, masking on their errors; take the network
  from `resolve_trial_network_plan`; pass `env` and `user`; use the verifier image when set.

- **D7 P1 [ran p4] Public-network in-place imports lose fetch interception.** `CR/interop/harbor/inplace.py:146-147`
  keeps only `grader_paths` when the World network is public. Harbor's default network is public
  (`H/models/task/config.py:247-257`), and both Repo2RLEnv's emitter (`R2/emitter/harbor.py:124-136`) and Harbor's
  swesmith adapter declare none. Probe: `git fetch https://github.com/swesmith/…`, `pip download …==2.0.1` and
  `curl … /pull/7.diff | git apply` all pass for both task shapes. For repo-fix tasks this hands the policy the
  upstream fix (Repo2RLEnv's own contamination rule; SWE-smith's mirrored history). Fix: keep `remote_code_fetch` with
  an allowlist of package indexes, or import SWE-type tasks with the agent phase at no-network unless a flag says the
  internet is the task. Record the choice on the task card.

- **D8 P2 [ran p5] OpenEnv export contract gaps.** `CR/interop/openenv/export.py`. Each instance starts an event-loop
  thread that `close()` never stops and a `mkdtemp` it never removes (`:26-34,66,134-137`); probe: 5 reset/close
  cycles add 20 threads and 5 temp dirs. `seed` is ignored and, without `episode_id`, every episode shares the key
  `openenv|None` and the same call_ids (`:74`). `timeout_s` is ignored (`:124-128`). A failed call never sets
  `CallToolObservation.error` (probe: `error=None`, `ok=False` only in metadata). `step` before `reset`, which plain
  HTTP does (`OE/core/env_server/http_server.py:1113-1115`), fails on `self._world = None`; nothing documents "use `/ws`
  or an MCP session". Fix: stop the loop and remove the dir in `close`; seed the World; refuse a missing episode id or
  derive a unique one; map tool failures to `ToolError`; document the session rule.

- **D9 P2 [ran p8] OpenEnv import reward edges.** `CR/interop/openenv/import_env.py:474,577-592`. An environment that
  ends with `reward=None` (unscored in OpenEnv) is graded 0 with the reason "never terminated". With the default
  `reward_range` (0, 1), a −1 terminal reward and a 0 grade the same (probe: both 0.0). An MCP `timeout` error is infra
  and masks, while native tool timeouts are scored, so a policy that makes a tool hang masks its own episode. A
  two-seat game writes one scalar (the last mover's reward, `:585-597`) into `reward.json`; the losing seat's reward
  lives only in check detail. Fix: done with None reward → masked (`verifier_error`), or scored only under an explicit
  per-environment rule; require `reward_range` when a negative reward is seen; map `timeout` by blame like native
  tools; write per-seat rewards into `reward.json` components.

- **D10 P2 [ran p7] ATIF replay and reward-key selection.** `CR/interop/replay.py:127-137` replays agent steps marked
  `is_copied_context` (probe: the copied `issue_refund` runs again; `trajectory_to_trace` skips it). Each tool call
  becomes its own replay step, so one model call with two tool calls gets two derived call_ids (`replay.py:97`).
  `scalar_reward({'reward': 0.2, 'correctness': 1.0}, reward_key='correctness')` returns 0.2
  (`CR/interop/harbor/inplace.py:86-91`); OpenEnv's `_pick_reward` honours the explicit key first
  (`OE/harbor/rollout.py:151-193`). Fix: skip copied context in `from_atif`; derive call_ids per agent step (model
  call), not per action; check `reward_key` before `reward`.

- **D11 P2 [read] Harbor job metrics on Crucible exports.** `MASKED_MEAN_PY` (`CR/interop/harbor/metrics.py:19-61`) is
  used only by `tests/unit/test_m7_audit.py:463-469`; no export writes it into a job config. Stock `Mean` counts a mask
  as 0 (`H/metrics/base.py:24-35`). Harbor's pass@k returns `{}` for any multi-key reward (`H/utils/pass_at_k.py:42-43`),
  so it is silently absent for every Crucible task. Fix: the Harbor dataset export writes the metric script and a job
  config that names it; ship a Crucible pass@k uv-script that reads `reward` and skips nulls.

- **D12 P2 [read] Spec items not built.** No `openenv.yaml` and no rubric on the exported environment (ARCHITECTURE
  M11). No Harbor job bridge or Harbor harness for evaluation (only `role_runner.py`), no ATIF `RolloutDetail` →
  `crucible.rollout/v1`. No Harbor-task OpenEnv export (OpenEnv's own `run_rollout` shape). No task02 leg in
  `test_interop_gate.py`. DESIGN's "Gate (toy) ✅" does not claim task02, but ARCHITECTURE's exit gate does.

- **D13 P2 [read] Imports are not pinned.** `OpenEnvTask` records the factory string only (`import_env.py:484-526`).
  `HarborInPlaceTask` never computes the content hash that `load_harbor_task` computes (`task_io.py:85`). A change in
  PortSim's reward version or a Harbor task file changes grades silently. `.envs/*` have no lock or recipe. PortSim's
  CC BY-SA 4.0 licence is not on any task card or in `THIRD_PARTY.md`. Fix: FineEnvs-style pins (pick 9).

- **D14 P2 [read] mini-swe-agent imports lose the exit status.** Harbor's converter ignores `role: "exit"` messages and
  `info.exit_status` (`H/agents/installed/mini_swe_agent.py:306-424`). The coding pack reads `exit_status` itself but
  treats only `LimitsExceeded` and `TimeExceeded` as unfinished (`packs/coding/src/crucible_pack_coding/connectors/agent_runs.py:91-95`).
  A run that died on an uncaught exception (`exit_status` = its class name, `MS/agents/default.py:74-86`) with reward
  < 1 is labelled `bad`: an API outage becomes a judgement of the work. Fix: any exit status other than `Submitted`
  and `RepeatedFormatError` with reward < 1 is `unknown`; keep the `exit` message in the trace.

- **D15 P2 [read] `cross_check` ignores retried calls.** `CR/interop/harbor/trajectories.py:120` counts steps with
  `llm_call_count == 1`. Harbor harness ATIF can carry `llm_call_count > 1` (retries) or null
  (`H/models/trajectories/step.py:73-85`). Either miscounts against a capture log. OpenEnv's `reconcile` treats `> 1`
  as retries (`OE/harbor/atif.py:15-25,232`). Fix: sum `llm_call_count` (null = unknown, then skip the check).

Documented divergences, not counted: `multi_step` scores 0 under `final` when stopped early and masks a failed step
verifier (`multi_step.py:36-38`). Both are stricter than Harbor. They mean a Harbor import's reward can differ from
Harbor's own for the same trial; the gate must not claim identity for those cases.

## 3. Picks

| # | Pick | Mode | Source | Change in Crucible | Fixes |
|---|---|---|---|---|---|
| 1 | Mask-safe step rewards for OpenEnv | PATTERN | `OE/harbor/rollout.py:505-553` (an unusable value is an exclusion); `VF/tasksets/openenv/taskset.py:128-129` (the consumer rule to defend against) | `export.py`: ΔΦ in metadata, final step carries the reward, typed error on mask; gate negative control | D1 |
| 2 | Harbor's reward parser and network plan on the patch path | DEPEND | `H/verifier/verifier.py:66-124`; `H/trial/network_policy.py` | `verify_dir.py` | D6 |
| 3 | Separate verifier with Harbor's rules | PATTERN | `H/trial/trial.py:885-1022`; `H/models/task/verifier_mode.py:43-64` | `inplace.py` via `world/verifier.py` custody; refuse until built | D4, D5 |
| 4 | Verifier-dir hygiene before every grade | DEPEND | `H/trial/trial.py:826-833` | `inplace.py:190-213` | D2 |
| 5 | Per-framework env with its own lock and tests | PATTERN | `FE/00-environments-101/envs/wordle/*/{pyproject.toml,uv.lock,tests}` | `crucible/envs/<fw>/pyproject.toml` + `uv.lock`, built by a script the gate calls | D3, D13 |
| 6 | `openenv.yaml` and a reporting rubric | TARGET | `OE/validation/manifest.py:41-70,285-302`; `OE/core/rubrics/trajectory.py:22-136`; `PS/openenv/openenv.yaml` | `studio/export.py` openenv target writes the manifest; `CrucibleRewardRubric` exposes gates, outcome, Φ | D12 |
| 7 | ATIF ↔ capture reconcile | PATTERN (BSD-3) | `OE/harbor/atif.py:111-232,498` | `trajectories.cross_check` | D15 |
| 8 | Reward-key order and per-key finite filter | PORT (BSD-3) | `OE/harbor/rollout.py:130-193,527-546` | `inplace.scalar_reward` | D10 |
| 9 | Import pins and licence propagation | PATTERN | `FE/05-multi-harness-rl/prepare.py:46-121`; `eval/evaluate.py:191-210` | `OpenEnvTask` and `HarborInPlaceTask` record source hashes and licence; grade meta carries them | D13 |
| 10 | SWE-family imports through Harbor adapters | DEPEND | `HA/swesmith`, `HA/swebench`, `HA/swegym`, `HA/swebenchpro` | no own SWE-bench importer; an import flag for agent network | D7 |
| — | Do not copy | anti-pattern | `H/metrics/base.py:24-35` and `H/utils/pass_at_k.py:39-41` (no reward = 0); `H/trial/single_step.py:89-114` (no dir hygiene); `OE/core/env_server/mcp_environment.py:576-590` (blame from message keywords); `R2/emitter/harbor.py:124-136` (no network rule); `HA/swesmith/.../tests/test.sh:3-17` (grade-time installs) | | |

## 4. Probe index (`scratchpad/m11sweep/g1/`)

| Probe | Shows | Result |
|---|---|---|
| `p1_inplace_planted_reward.py` | D2 | planting commands: 1 refused, 3 pass; Harbor `Verifier` returns `{'reward': 1.0}` with test `reward.txt` = 0; `inplace.grade` empties `/logs/verifier`: False; `multi_step._grade_step`: True |
| `p2_separate_verifier.py` | D4, D5 | task02 mode `separate`, `HarborInPlaceTask` built, 1 collect hook never referenced; verifier image None, no `tests/Dockerfile`, Harbor definition check False; `load_harbor_task` ok with `ignored_keys ['verifier.command', 'environment.dockerfile']` |
| `p3_verify_dir_vs_harbor.py` | D6 | six reward-file cases differ (2 `TypeError`, NaN accepted, 2 wrong 1.0, 1 `JSONDecodeError`); swesmith adapter task: Harbor verifier network public, ours `no-network`, `test.sh` needs the network |
| `p4_public_import_fetch.py` | D7 | Repo2RLEnv and swesmith-adapter tasks: network public, intercept `['grader_paths']`; `git fetch`, `pip download`, `curl … .diff` all pass |
| `p5_openenv_export.py` (openenv env) | D1, D8 | masked episode: step rewards sum to 1.000 under the verifiers rule; threads 7 → 27 and temp dirs +5 after 5 cycles; `rubric None`, `error None` on a bad call |
| `p6_gate_vacuous.py` (openenv env) | D3 | environment pays 0.5; gate assertions True True True; native `reward.json` 1.0 |
| `p7_atif_and_reward_key.py` | D10 | copied-context action replayed; two call_ids for one model call; `scalar_reward(..., reward_key='correctness')` = 0.2 |
| `p8_openenv_import_none.py` (openenv env) | D9 | `Unscored` → 0.0, ok, "never terminated"; `Lost` (−1) → 0.0; `Zero` → 0.0 |
