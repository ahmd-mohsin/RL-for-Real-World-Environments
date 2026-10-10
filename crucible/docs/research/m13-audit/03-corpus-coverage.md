# M13 audit 03: corpus coverage. What Crucible can run from the local RL-environment corpus

Status: audit (2026-10-10). Read-only audit of the local corpus against Crucible `main` @ 90200ec. This doc extends and corrects the catalog in `m11-m12/05-interop-diversity.md` §5, which is not repeated here. That catalog listed environment *kinds*. This one lists environment *families* (a dataset, a harness or a framework's env collection). For each family it gives a run status through Crucible, the concrete work to make it run, and an acceptance test. Claims about Crucible are cited `path:line`. Corpus counts are local directory counts. Nothing was run in Docker. "Verified" means a test in `crucible/tests` passes on that family.

Path abbreviations: `CR/` = `crucible/src/crucible/`, `FW/` = `RL-Environments/06-frameworks-and-env-code/`, `DS/` = `RL-Environments/08-hf-rl-env-datasets/`, `MM/` = `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/`, `FE/` = `FW/adithya-s-k-fineenvs/`.

---

## 0. Summary

1. **4 of 46 environment families run through Crucible today**:
   - Crucible's own coding pack (verified);
   - the toy helpdesk pack (verified);
   - FineEnvs PortSimEnv v1 through the OpenEnv importer (verified: native = import = replay = re-export, and it runs through the M12 runtime);
   - MiMo v2.6 *general*, which runs but is graded only when an LLM judge is configured (otherwise it masks, which is correct).

   Of the remaining 42, 25 have an adapter that is either untested on the family (A?, 11) or **does not fit the family's grading semantics** (A✗, 14). 17 have no adapter (M). Counts are in §4.
2. **The largest single blocker is Harbor.** Fourteen Harbor dataset families (rows 6–19) and 85 Harbor benchmark adapters ship Harbor task directories. Crucible's only foreign-Harbor grader, `HarborDirTask`, does three things (`CR/interop/harbor/verify_dir.py:92-111`):
   - writes the agent's work to `/logs/artifacts/model.patch`;
   - starts a **fresh** container of the task image;
   - runs `test.sh`.

   It never applies the patch. Crucible's own kit applies it (`CR/verify/runtime/run.py:63`). Foreign `test.sh` files grade the *container state* the agent left behind instead: TB2.1 runs pytest against `/app`, BeyondSWE uses `/workspace/<repo>`, MiMo-harbor-code diffs `/testbed` against `/var/lib/mimo/base`. So through Crucible, an oracle and a no-op both get the pristine image's grade. That is a silent wrong reward, not a crash. The fix is an in-place Harbor task: the agent acts in the task container (the Harbor World backend already exists), then `tests/` is uploaded and the verifier runs in the same container. Details in G1.
3. **Two more Harbor bugs turn infra failures into scored outcomes:**
   - **Grade-time network.** `HarborDirTask` reads only `[verifier].network_mode` (`verify_dir.py:66`). It ignores `[environment].allow_internet` (schema 1.0/1.1) and `[environment].network_mode` (1.4). TB2.1's `test.sh` installs curl and uv at grade time, so offline it writes `0` to `reward.txt`. That is a false 0.
   - **Multi-key `reward.json`.** SmolDataEnvs writes `{"correctness","submission","tool_efficiency"}`, and `float(...get("reward"))` (`verify_dir.py:127`) raises `TypeError` on it.
4. **There is no import side for NeMo-Gym or verifiers.** `CR/interop/nemo_gym/server.py` and `CR/interop/verifiers/taskset.py` are *exports* only. That leaves 145 NeMo-Gym resources servers, 7 local Nemotron datasets, 109 prime-environments (verifiers **v0** API) and 106 research-environments (verifiers **v1** Tasksets) unreachable. verifiers' own `tasksets/nemo_gym` (≈470 lines) is the pattern to port for a NeMo-Gym import.
5. **`interop/eog` does not exist** (the task brief lists it). EnterpriseOps-Gym, Crucible's closest design target, has no adapter. AWM-1K worlds load (100/100 world checks) but no AWM task or verifier is imported, so they produce no reward.
6. **Vision is evaluable but not trainable.** `ToolResult.images` and vision policies work (`CR/actors/policy.py:114,207-214`; `tests/unit/test_world_diversity.py:136`). The M12 trainer's chat codec is text-only. OpenEnv non-MCP observations with pixel fields are serialized as JSON text (`CR/interop/openenv/import_env.py:219`): only MCP `ImageContent` becomes an image (`:237`).

---

## 1. Status legend and Crucible entry routes

| Code | Meaning |
|---|---|
| **R✓** | Runs through Crucible today, and a test verifies its reward against an oracle or the native harness |
| **R~** | Runs, but is only partially verified, or its grading is conditional (judge, credentials) |
| **A?** | An adapter exists but is untested on this family. Expected to work after configuration or small fixes |
| **A✗** | An adapter exists, but its semantics do not fit the family. Running it today gives wrong rewards |
| **M** | No adapter |

Crucible today has four ways to run a foreign environment:
- **(a) Native `RunTask`** (`open` → World, `grade`; `CR/runtime/harness/native.py:6`). Used by the packs, `OpenEnvTask` (`CR/interop/openenv/import_env.py:272`) and the MiMo importer with MiMo's own `verify()` (`CR/interop/mimo/verify.py`).
- **(b) Workspace harness + `HarborDirTask`.** Patch seam only (`verify_dir.py`).
- **(c) AWM service Apps.** World only, no grader (`CR/interop/openenv/awm.py`).
- **(d) Harbor World backend.** Hosts Apps inside a Harbor environment (`CR/interop/harbor/world_backend.py`). No task or grader is wired to it.

The exports (NeMo-Gym, verifiers, OpenEnv, MiMo, agentenv) carry Crucible tasks *out*; none imports.

---

## 2. Cross-cutting gaps (one fix unlocks many families)

| ID | Gap | Evidence | Unblocks | Work |
|---|---|---|---|---|
| **G1** | No in-place Harbor execution: the agent runs in the task container, then the verifier runs in that container. No oracle (`solution/solve.sh`) runner | `verify_dir.py:92-111` (patch written, never applied); no `solve.sh` reference in `CR/` | 14 Harbor dataset families; 85 Harbor adapters; Harbor-shaped research-env SWE/terminal tasksets (42 `HarborTaskset` uses) | `HarborTask` RunTask over the Harbor backend: build/pull image → terminal tools (workspace/toolchain App via `HarborExec`) → episode → upload `tests/` → `[verifier].command` with `[verifier.env]` and `user` → read reward. Add `environment_mode="separate"` with the `[[artifacts]]` copy (Tasksmith 1.3) and an oracle mode that runs `solve.sh` |
| **G2** | Harbor schema 1.0–1.4 coverage | `verify_dir.py:66` reads only `[verifier].network_mode` | TB2.1, MiMo-harbor, SmolDataEnvs, LHTB, NatureBench | Honour `allow_internet` and `[environment].network_mode/allowed_hosts`. Run `[environment.healthcheck]` as a setup hook with retries (SmolDataEnvs data pull; MiMo code setup tarball). Support `[[verifier.collect]]`, multi-service compose, `artifacts=[...]` and LHTB `continue_until_timeout` (an opt-in that is visible to the policy) |
| **G3** | Reward-file contract for foreign Harbor files | `verify_dir.py:127` `float(None)` on a multi-key file | SmolDataEnvs, data-agent, any multi-reward task | Per-dataset declared mapping (`reward_key` or weights) recorded in the task overlay. Without a declaration, mask `reward_schema_unknown` instead of crashing |
| **G4** | The Harbor backend requires `python3` + `git` inside the task image | `world_backend.py:190,202` | Foreign images (R, Go, MIPS toolchains) | Bind-mount a static state shim (busybox + a `git` binary) or fall back to the fs adapter without git |
| **G5** | No NeMo-Gym import | export only (`CR/interop/nemo_gym/server.py`) | 145 servers, 7 Nemotron datasets, NeMo's `openenv`/`gymnasium` bridges | `nemo_gym/import_server.py`: launch the server, `/seed_session` → tools as an App → episode → rebuild a Responses-API `response` from the World trace (port verifiers' `trace_to_nemo_response`) → `/verify`. Read `mask_sample`. Python 3.13 runs in its own `.envs/nemo_gym` |
| **G6** | No verifiers import, and the two APIs differ | export only (`CR/interop/verifiers/taskset.py`) | research-environments (v1, 106), prime-environments (v0 `load_environment`, 109), verifiers examples (19) | v1: wrap `Taskset`/`Task` + toolsets as Apps, use the task's `@reward` functions as a score gate, and map `TaskError` → mask. v0: `SingleTurnEnv`/`ToolEnv`/`StatefulToolEnv`/`SandboxEnv` via `env.rubric.score_rollout` on the replayed transcript |
| **G7** | OpenEnv import is in-process only, and image fields of non-MCP observations are lost | `import_env.py:43-80` (factory `module:attr`), `:219`, `:237-238` | browsergym, atari, dm_control, carla, unity, tbench2, git_env, E2B-backed envs | Add `EnvClient` mode (HTTP/WebSocket to the env's Docker image, as OpenEnv's own clients do), and keep replay-restore. Add `image_fields=[...]` → `ToolResult.images` |
| **G8** | No VLM training path | text-only `CR/train/chat.py` | every vision family | Multimodal chat codec (Qwen-VL processor) with image-token accounting and TRL/vLLM `images` in `gen` (the TRL bridge already passes `images`, `CR/train/bridges/trl_bridge.py:22`) |
| **G9** | No single-turn "prompt + checker" task type | no such type in `CR/` | math, IF, knowledge, long-context rows (Reverse-Text, procedural-pile, openai-math rows) | A zero-tool World whose episode is one completion plus an `answer` App. Usually reached via G5/G6 instead |
| **G10** | Judge-graded families need a pinned judge | MiMo masks without a judge (`tests/unit/test_interop_gate.py:262-266`) | MiMo general, NatureBench, CFBench/SysBench, watercolour | Record judge model and version in `crucible.toml`, run a judge agreement probe, and record the judge on every rollout |
| **G11** | No EnterpriseOps-Gym adapter | no `CR/interop/eog` | EOG 1,150 tasks / 8 domains | §3 row 36 |
| **G12** | AWM worlds have no grader | `awm.py:1-9` ("AWM's own verifiers belong to M7") | AWM-1K (10k tasks) | Import AWM task JSON and verifier code as a score gate over the `sql` adapter |

---

## 3. Family inventory

Resources key: **G** = GPU, **N** = network at run or grade time, **I** = image observations, **L** = long horizon (≥1 h agent budget or ≥100 steps), **J** = LLM judge, **V** = VM/KVM.

### 3.1 Already inside Crucible, or verified through an importer

| # | Family | Path | Interface | Kind | Res. | Status | Remaining work / acceptance |
|---|---|---|---|---|---|---|---|
| 1 | Crucible coding pack (legacy GPU migration, tasks 01–04) | `crucible/packs/coding` | Harbor dir + `crucible/` | coding, kernels | G | **R✓** | none for this audit; the M12 GPU gate rerun (90200ec) is still open |
| 2 | Toy helpdesk | `crucible/packs/_toy` | native World | enterprise (toy) | — | **R✓** | none |
| 3 | FineEnvs PortSimEnv v1 | `FE/07-simulation-environments/portsim-v1` | OpenEnv MCP | sim/OR | — | **R✓** | `test_openenv_import.py:26,50` passes: native = import = replay = re-export |
| 4 | MiMo v2.6 general (267 dirs; 266 in the M1 gate) | `MM/general/envs` | MiMo env (MCP sidecars, `verify.py`) | enterprise knowledge work | J | **R~** | Configure a judge (G10). Acceptance: on 20 envs, Crucible's reward = MiMo `run_verify.py` with the same judge seed, and reruns agree ≥ 0.95. Explain the 267 vs 266 count |
| 5 | OpenEnv AgentWorldModel-1K | `FW/meta-pytorch-openenv/envs/agent_world_model_env` | FastAPI+MCP over SQLite | enterprise (synthetic) | — | **A?** (worlds only) | G12. Acceptance: on 50 scenarios, AWM's verifier result from OpenEnv's own runner = the Crucible grade; no-op scores the AWM floor |

### 3.2 Harbor-format datasets (`DS/`), all blocked on G1, plus G2/G3 where noted

| # | Family | Interface (schema) | Kind | Res. | Status | Work / acceptance |
|---|---|---|---|---|---|---|
| 6 | terminal-bench-2.1 (89 tasks) + terminal-bench | Harbor 1.1, prebuilt `docker_image` | terminal | N (grade) | **A✗** | G1, G2 (`allow_internet`). Acceptance: for every task, Crucible oracle reward = `harbor run --agent oracle` reward; no-op = 0; with grade-time network blocked the run *masks*, never 0 |
| 7 | terminal-bench-science | Harbor 1.1 | science, terminal | N | **A✗** | Same runner as #6. Acceptance: identical to `harbor run` on the oracle |
| 8 | Long-Horizon-Terminal-Bench | Harbor 1.1 + `continue_until_timeout`; daemons inside the image | terminal, game (2048), OR (matpower) | L (14,400 s agent) | **A✗** | G1, G2. Acceptance: the 2048 task's replay-scored band equals the native harness for a scripted move log. Feedback-continuation is off in eval |
| 9 | Terminal-Lego-15k (Lego-X + PI exclusions) | Harbor 1.0 | terminal | — | **A✗** | G1. Acceptance: Crucible reproduces PI's 1,224 exclusion labels (no-op pass, gold-apply fail, …) on a 300-task stratified sample |
| 10 | MiMo-V2.6 Harbor conversions (code sampled; terminal/cyber/general/music/webdev exist on HF) | Harbor 1.4, digest-pinned image, healthcheck setup | coding, security, multimodal (webdev/music) | N | **A✗** | G1, G2 (healthcheck). Acceptance: Crucible reward = MiMo `opensource_code.py` reward on the gold patch (1) and on no-op (0). A failed testbed reset writes no reward → masked |
| 11 | BeyondSWE-harbor | Harbor 1.0, compose | coding | N | **A✗** | G1. Negative control: all 500 must be rejected by acceptance (no-op oracle, unreachable `hub.byted.org` images) |
| 12 | SmolDataEnvs-harbor (train/test/eval) + data-agent-harbor | Harbor 1.2, healthcheck pulls HF bucket with `HF_TOKEN` | data science | N, secret | **A✗** | G1, G2, G3. Keep the token out of the agent phase. Acceptance: gold answer → `correctness` 1, empty answer → 0, through the declared mapping |
| 13 | HF_ML_Tasksmith | Harbor 1.3: separate verifier, `[[artifacts]]`, learner/root users | ML research | G (11 tasks) | **A✗** | G1 (separate mode), lab GPU host. Acceptance: Tasksmith's published oracle/probe `result.json` rewards reproduced |
| 14 | NatureBench-Harbor | Harbor 1.4, judge reads ATIF | science | J, G, ~939 GB | **A✗** | G1, G10, lazy per-task fetch. Acceptance: ATIF export accepted by its judge script; 5 tasks match the native harness within judge variance |
| 15 | repo2rlenv-* (21 datasets: swe-flow, equivalence-tests, cve-patches, …) | Harbor 1.x | coding, security | N | **A✗** | G1. equivalence-tests is a negative control (all degenerate) |
| 16 | Lego-RL-2699 / Lego-RL-SWE-Bench-Verified | Harbor, `network_mode=allowlist` | coding | N (allowlist) | **A✗** | G1, G2 (allowlist), restore the shipped-without-repo `environment/repo` |
| 17 | termgrade-environments | Harbor | terminal | — | **A✗** | G1 |
| 18 | FineEnvs openai-math | Harbor (oracles fetched from GitHub) | math | — | **A?** | G1 is small here: a stateless check on `answer.txt`. Acceptance: oracle 1.0, no-op 0 |
| 19 | harbor-mix | Harbor (gated) | mixed | — | **M** | Gated (accepting the terms is the user's decision); then same as #6 |

### 3.3 Row datasets whose environment code lives elsewhere

| # | Family | Interface | Kind | Status | Work / acceptance |
|---|---|---|---|---|---|
| 20 | PI SWE row tasksets (Multi-SWE-bench, Multi-SWE-RL-Verified, R2E-Gym-Subset-Verified, Scale-SWE-Verified, SWE-Lego-Real-Data-Verified, SWE-rebench-V2-Filtered-Verified) | verifiers v1 rows → `FW/primeintellect-ai-research-environments/environments/swe/*` | coding | **M** | G6 (v1 import), or a row→Harbor converter feeding G1. Acceptance: gold patch → 1 and no-op → 0 on PI's "Verified" subset; PI's exclusion lists reproduced |
| 21 | PI Reverse-Text-RL | verifiers v1 single-turn | text transform | **M** | G6/G9. Acceptance: reward equals `vf-eval` on fixed completions |
| 22 | Nemotron-RL datasets (workplace_assistant, calendar_scheduling, instruction_following, IF-structured, CFBench, SysBench, ReasoningGym) | NeMo-Gym JSONL + `FW/nvidia-nemo-gym/resources_servers/*` | enterprise, IF, reasoning | **M** | G5. Acceptance: workplace_assistant ground-truth actions → 1.0 = NeMo `/verify`; no-op reproduces the known 13% floor exactly |
| 23 | LiteFold RetroEnv | OpenEnv (env code not local) | science (retrosynthesis) | **M** | Fetch env code, then #3's route. Acceptance: its published probe rewards (`first_step_only` 0.595, `empty` 0, oracle 1.0) |
| 24 | reasoning-core procedural-pile | SFT rows + scorers | math, reasoning | **M** | G9 + the reasoning-core scorer. Acceptance: solver answers score 1 |

### 3.4 Framework environment collections

| # | Family | Path | Interface | Kind | Res. | Status | Work / acceptance |
|---|---|---|---|---|---|---|---|
| 25 | OpenEnv text games: chess, connect4, openspiel, textarena, snake, maze, grid_world | `FW/meta-pytorch-openenv/envs/*` | OpenEnv Action envs (non-MCP) | game, multi-agent | — | **A?** | Set `action_cls`, a seeded `reset`, and `reward_range` per env. For two-player envs, map the opponent to a Crucible actor. Acceptance: native reward = imported reward on 100 seeded scripted games; replay-restore digests match; a non-seeded env refuses restore |
| 26 | OpenEnv simulations: sumo_rl, finrl, wildfire | same | Action envs | sim/OR | — (SUMO binary) | **A?** | Same as #25, plus `clock_from` for sim time and `reward_mode="sum"` for per-step rewards. Acceptance: Σ step rewards = native episode return; a sim-clock event fires on sim time |
| 27 | OpenEnv pixel/physics: atari, dm_control, carla, unity | same | Action envs with image observations | game, sim | I, G (carla/unity) | **M** | G7 (client mode + `image_fields`), G8. Acceptance: frames arrive as `ToolResult.images` with matching digests; native return = imported return |
| 28 | OpenEnv browsers: browsergym (MiniWoB), helium_browser, openapp | same | Action envs, screenshot + DOM | browser/GUI | I, N | **M** | G7, G8. Acceptance: MiniWoB oracle scripts → native reward; a11y-only policy path for text models |
| 29 | OpenEnv MCP services: calendar_env, finqa_env, websearch_env, echo | same | MCP / Action | enterprise, search | N (websearch) | **A?** | Factory config; websearch needs a cached corpus for determinism. Acceptance: native = import |
| 30 | OpenEnv code/terminal: coding_env, coding_tools_env, jupyter_env, terminus_env, tbench2_env, git_env, julia_env, repl_env, opencode_env, pi_env, harbor_env, kernrl | same | MCP / Action; E2B or Docker inside | coding, terminal | N, G (kernrl) | **A?** | Prefer Crucible's own Harbor/World route for these. Import only kernrl (GPU) and julia. E2B envs need an API key and are not replay-deterministic |
| 31 | OpenEnv misc: reasoning_gym_env, qed_math_env, dipg_safety, sophistry_bench_sprint, thinkingbox, pelican_svg, chat_env | same | MCP / Action | math, safety, vision (svg render) | — | **A?** | Factory config. Acceptance: native = import on 50 seeded items |
| 32 | NeMo-Gym resources servers (145: knowledge 26, agent 26, coding 25, math 15, IF 11, safety 6, games 6, long-context incl. ruler/ruler2/mrcr/longmemeval, VLM: circle_click, labbench2_vlm, gui_coordinate; GUI: osworld) | `FW/nvidia-nemo-gym/resources_servers` | NeMo-Gym resources server | all kinds | J (many), I (VLM), V (osworld) | **M** | G5. Acceptance per server: Crucible reward = NeMo-Gym `/verify` on the same transcript; `mask_sample` → mask |
| 33 | prime-environments (109) | `FW/primeintellect-ai-prime-environments/environments` | verifiers **v0** (`load_environment`; SingleTurn 89, MultiTurn 37, Sandbox 19, StatefulTool 13, Tool 12, MCP 2 references) | all kinds | varies | **M** | G6-v0. Acceptance: `vf-eval` reward = Crucible reward on scripted rollouts |
| 34 | research-environments (106: swe 18, long_context 12, lean 11, search 10, tool_use 10, math 8, code 7, knowledge 7, terminal 6, reasoning 6, multimodal 4, science 4, if 2, judges 1) | `FW/primeintellect-ai-research-environments/environments` | verifiers **v1** Tasksets (42 `HarborTaskset` uses) | all kinds | varies | **M** (Harbor-shaped part waits on G1) | G6-v1. Acceptance: same as #33 |
| 35 | verifiers examples (19: wordle, kuhn_poker, proposer_solver, nemo_gym_weather, openenv_wordle, …) | `FW/primeintellect-ai-verifiers/environments` | v1 | game, multi-agent | — | **M** | First fixtures for G6 |
| 36 | EnterpriseOps-Gym (1,150 tasks; 12 local samples) | `FW/servicenow-enterpriseops-gym` | custom: per-domain MCP server Docker images + SQL state verifiers + `gym_dbs.zip` seeds | enterprise SaaS | N (Docker Hub), L (avg context 89k) | **M** | G11: per-domain `mcp_server` App on the published image, a postgres/sql adapter on its DB (seeded from `gym_dbs.zip`), task JSON → instruction + `selected_tools` ACL, `verifiers[]` → SQL gates. Acceptance: EOG's `compute_score.py` = Crucible grade on recorded trajectories; no-op → 0 on tasks with create/update checks |
| 37 | Harbor adapters (85: osworld, theagentcompany, crmarena, gaia/gaia2, tau3-bench, textarena, cybergym, mlgym-bench, swelancer, spreadsheetbench-verified, spider2-dbt, scienceagentbench, …) | `FW/harbor-framework-harbor/adapters` | benchmark → Harbor dirs | all kinds | V (osworld), N | **A✗** | G1/G2, then generate the datasets. `parity_summary.csv` gives native-parity reference numbers to match |
| 38 | FineEnvs 00-environments-101: wordle, jupyter, desktop (each × 6 adapters) | `FE/00-environments-101/envs` | OpenEnv MCP (+NeMo, verifiers, skyrl-gym, GEM, ORS) | game, data, GUI | I, N (E2B) for desktop/jupyter | **A?** | wordle via the OpenEnv import now. Acceptance: one game core, identical reward via OpenEnv import and via the (future) NeMo import. This must expose the known NeMo adapter drift (05 §0.1 #12) |
| 39 | FineEnvs latex-ocr | `FE/01-latex-ocr/envs/latex_ocr` | OpenEnv, single-turn image→LaTeX | vision | I, G (train) | **A?** + G8 | Acceptance: gold LaTeX → native reward |
| 40 | FineEnvs watercolour | `FE/02-watercolour/envs/watercolour` | OpenEnv; p5.brush code → headless render → HPSv3 PM + judge | vision (image out), creative | G (PM), J, I | **M** | A GPU PM service as a sidecar App, plus G10. Acceptance: PM score = native rubric on published rollouts (`watercolour-rollouts-*`) |
| 41 | FineEnvs geoguesser | `FE/03-geoguesser/env` | OpenEnv **MCP** with image content | vision, geo | I | **A?** + G8 | Works with API vision policies today. Acceptance: native km-error reward = imported reward for fixed guesses |
| 42 | FineEnvs 05-multi-harness-rl (whitebox / opencode / harbor servers over SmolDataEnvs) | `FE/05-multi-harness-rl/envs` | OpenEnv MCP | data science | N | **A?** | Acceptance: the same task's reward via whitebox import = via #12 |
| 43 | SWE-smith | `FW/swe-bench-swe-smith` | generator → SWE-bench instances + images | coding | N | **M** | Route through the Harbor `swesmith` adapter / repo2rlenv-swe-smith → G1 |
| 44 | Repo2RLEnv | `FW/huggingface-repo2rlenv` | generator → Harbor | coding | N | **A✗** (via G1) | Acceptance: equals #15 |
| 45 | karpathy autoresearch | `FW/karpathy-autoresearch` | custom: edit `train.py`, 5-min run, `val_bpb` | ML research | G, L | **M** (cheap) | Author it as a Crucible coding-pack task: record-relative score gate, PerfGuard, lab GPU. Acceptance: the baseline `train.py` reproduces its recorded `val_bpb` ±noise band; the no-op scores the baseline tier |
| 46 | Trainer-bundled envs (ROLL: sokoban, frozen_lake, webshop, deepeyes, geo3k, rock, gem; SkyRL-gym: sql, aime, gsm8k, lcb, search) | `FW/infra-*` | gym-like `step` | game, web, vision, SQL, math | I (deepeyes, geo3k) | **M** | Low priority: they duplicate OpenEnv/NeMo families |

Not environments, and not counted:
- Karotte (`FW/preferencemodel-karotte`) is a framework with templates and a VM runtime, and no env corpus. Crucible already ported its isolation patterns.
- MiMo verl recipes (`FW/xiaomimimo-verl/recipes/*`) are the MiMo reference harness.
- mini-swe-agent is a harness. Crucible imports its trajectories (`CR/workflow/replay.py:71`) and can drive it as an external harness.
- git-swarm is a paper site only.
- hf-space articles and `infra-*` trainers.

---

## 4. Counts

| Status | Families | Rows |
|---|---|---|
| **R✓** runnable, verified | 3 | 1, 2, 3 |
| **R~** runnable, conditional | 1 | 4 |
| **A?** adapter exists, untested | 11 | 5, 18, 25, 26, 29, 30, 31, 38, 39, 41, 42 |
| **A✗** adapter exists, wrong semantics | 14 | 6–17, 37, 44 |
| **M** missing | 17 | 19–24, 27, 28, 32–36, 40, 43, 45, 46 |
| **Total** | **46** | |

**Runnable today: 4 / 46 (≈9%). Need work: 42.** By task volume the picture is worse: the runnable families hold about 1,400 tasks (PortSim 1,050 train + 50 eval, MiMo general 267, packs ≈10). Blocked tasks number in the hundreds of thousands (Terminal-Lego 15k, SmolDataEnvs 5.4k, MiMo code 2.7k, AWM 10k, PI SWE rows ≈30k+). G1+G2+G3 alone move 13 A✗ Harbor families (all except #37's not-yet-generated datasets) and the 85 adapters to runnable. G5 and G6 move about 380 framework environments from M to A?.

---

## 5. The 10 families to wire next ("extremely diverse": at least one per kind)

Sim/OR is already covered by PortSim (R✓). Its next step (a dynamic sim with sim-clock events, e.g. OpenEnv `sumo_rl_env`/`wildfire_env`) rides on pick 8's work at no adapter cost. Each pick below names the kinds it covers and the gaps it closes. Together they cover all 16 kinds.

| Rank | Family (row) | Kinds covered | Gaps closed | Why this one | Acceptance test |
|---|---|---|---|---|---|
| 1 | **terminal-bench-2.1 + terminal-bench-science** (#6, #7) | terminal, science | G1, G2, G4 | Most-used tagged dataset (162k downloads); all oracles and images prebuilt; its fix is the runner for every Harbor family. Today A✗ with a silent false 0 | For all 89 TB2.1 tasks: Crucible oracle reward = `harbor run --agent oracle`, no-op = native no-op, and offline grading masks (`check_infra`) instead of scoring 0 |
| 2 | **MiMo-V2.6 Harbor code + cyber** (#10) | coding, security | G2 healthcheck | A real lab training set with digest-pinned images. Crucible already owns the MiMo general round trip, so the native-identity reference (`recipes/code/reward.py`, `opensource_code.py`) is in hand. ARVO cyber adds security with rule-check grading | On 50 code and 50 cyber tasks: gold → 1 and no-op → 0, both equal to MiMo's native grader. A testbed failure → masked |
| 3 | **SmolDataEnvs-harbor** (#12) | data science | G2 healthcheck, G3, secret hygiene | 5,394 tasks, an exact/tolerance grader ladder, and multi-key rewards. The same tasks exist as OpenEnv servers (#42), which gives a free cross-runtime check | Gold → `correctness` 1, empty → 0. Whitebox-OpenEnv reward = Harbor reward on 100 tasks. `HF_TOKEN` absent from the agent environment |
| 4 | **HF_ML_Tasksmith** (#13) (+ autoresearch #45 as a native task) | ML research | G1 separate-verifier mode, `[[artifacts]]`, GPU on lab host | The only ML-engineering family with oracles, probes and hash-bound evidence. Exercises the GPU path end to end | Tasksmith's published oracle and probe `result.json` rewards reproduced on 1×/2×GPU tasks on the lab RTX 5090 |
| 5 | **EnterpriseOps-Gym** (#36) | enterprise SaaS (+ long context, 89k avg) | G11 | Crucible's thesis domain: state-checked SQL verifiers, 512 tools, 8 domains. It fits the existing `mcp_server` + postgres Apps directly | `compute_score.py` = Crucible grade on recorded trajectories for 12 local tasks, then 100 HF tasks. No-op → 0 where a create/update check exists. `restricted_tools` enforced by ACL |
| 6 | **tau2/tau3-bench + BrowseComp-Plus** via a verifiers v1 import (#34) | user-sim, search | G6-v1 | Canonical user-sim benchmark plus a search benchmark over a *cached* corpus (deterministic). Unlocks 106 v1 envs | tau2: the gold action sequence → 1.0 = verifiers reward, with a fixed user-sim seed. BrowseComp-Plus: gold answer → 1, wrong → 0, equal to `vf-eval` on recorded transcripts |
| 7 | **NeMo-Gym import: workplace_assistant, ruler/mrcr, math_with_autograder** (#22, #32) | math, long context (+ enterprise) | G5, G9 | 145 servers and the Nemotron datasets behind a single adapter. workplace_assistant has a built-in oracle and a documented no-op floor | Oracle → 1.0 = NeMo `/verify`. No-op reproduces the 13% floor. ruler/mrcr exact match equals NeMo on 200 rows. `mask_sample` → mask |
| 8 | **OpenEnv two-player games: chess, connect4, textarena** (#25) (+ verifiers kuhn_poker #35) | game, multi-agent | small: `action_cls` config, opponent as an actor | Cheapest diversity gain (adapter exists); first true multi-agent family: the opponent maps onto Crucible actors with per-actor rewards | 100 seeded scripted games: native reward = imported reward. Replay-restore digests match. A self-play episode records per-actor rewards |
| 9 | **FineEnvs geoguesser** (#41) (+ latex-ocr #39) | vision/multimodal | G8 (and G7 `image_fields` for latex-ocr) | An MCP image env that already passes through `ToolResult.images`. The continuous km reward is a good dense signal. Published GRPO runs give a training reference | Native reward = imported reward for fixed guesses. A Qwen-VL policy trains through M12 with image tokens accounted. Reward curve directionally matches the published run 1 within 100 steps |
| 10 | **OpenEnv browsergym (MiniWoB)** (#28) | browser/GUI | G7 client mode + `image_fields`, G8 | GUI coverage without a VM (OSWorld needs KVM: later, via Harbor or NeMo `osworld`). Complements Crucible's Playwright-MCP browser App, which hosts only Crucible-authored sites | MiniWoB oracle scripts → native reward through the imported client. Screenshot digests are stable under replay-restore. An a11y-only text policy scores the same as natively |

Kind coverage check: coding 2, terminal 1, browser/GUI 10, search 6, enterprise SaaS 5, sim/OR (PortSim, R✓), game 8, vision 9, data science 3, math 7, multi-agent 8, user-sim 6, security 2, science 1, long context 7 (+5), ML research 4. Picks 1, 2, 3 and 4 share G1/G2. Build order: **G1+G2+G3 first** (picks 1–4), then G11, G6, G5 in parallel, then G7+G8 (picks 9–10).

---

## 6. Corrections to `m11-m12/05-interop-diversity.md`

| 05 location | Said | Correct as of 90200ec |
|---|---|---|
| §1.2 table, `eog/` | listed as an interop folder to build | Still does not exist. No EOG code anywhere in `CR/` |
| §1.2 `verifiers/`, `nemo_gym/` | "none / everything" | Export now exists (`taskset.py`, `server.py`, gate `test_interop_gate.py:92,133`). **Import** does not (G5, G6) |
| §5.1 Coding row: "Crucible tasks 01–04", Harbor datasets | implied Harbor datasets are importable via `HarborDirTask` (DESIGN §10, line 713: "Harbor task directories run as they are") | Only for Crucible-authored tasks whose `test.sh` applies `model.patch`. Foreign Harbor tasks grade container state (G1). DESIGN's claim needs the same correction |
| §5.1 Games row: "ROLL … webshop" | a game | WebShop is a text-web shopping env (browser/search) |
| §5.1 Enterprise row: "prime-env tau_bench, github_mcp …" | treated as one verifiers family | prime-environments is verifiers **v0** (`load_environment`, 111 files); research-environments is **v1** (`Taskset`, 135 files). Two adapters are needed |
| §5.1 (missing) | — | Harbor's 85 benchmark adapters (osworld, theagentcompany, crmarena, gaia, tau3, cybergym, mlgym-bench, swelancer, …) are the largest diversity lever and were not catalogued. Also missing: NeMo-Gym's own `openenv`/`gymnasium` bridge servers and OpenEnv's `harbor_env`/`opencode_env`/`pi_env` |
| §5.1 "Geo" as a separate kind | separate row | Part of vision (geoguesser is an MCP image env) |
| §5.2 browser state adapter | "no `state/browser.py`" | `BrowserStateAdapter` exists in `CR/world/apps/browser.py:91` (URL, cookies, localStorage) |
| §5.2 image observations | "not verified in `ToolResult`" | `ToolResult.images` exists and is tested (`test_world_diversity.py:136`). The remaining gaps are non-MCP OpenEnv image fields (G7) and trainer support (G8) |
| §5.2 tool quotas, sim clock, per-App seeds, replay-restore, step overhead | gaps | All built and tested (`test_world_diversity.py:71,103,123,167`; `import_env.py` replay journal, `test_openenv_import.py:32,41`) |
| §5.1 MiMo "general (267 …)" | 267 | 267 directories. The M1 gate reports 266/266. One directory is unaccounted for and should be explained or excluded explicitly |
| §6.2 | AWM listed as done (100/100) | World checks only. AWM tasks have no grader (G12), so AWM produces no reward |
