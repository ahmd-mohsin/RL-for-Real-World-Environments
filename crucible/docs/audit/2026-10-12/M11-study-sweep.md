# M11 study: every local source, read for interop, and the picks

Five sweeps read every local source for M11 `interop` at code level, with offline probes. Three defect hunts audited
the module itself. Each sweep is its own document:

| Group | Sources | Document |
|---|---|---|
| G1 | Harbor, Repo2RLEnv, SWE-smith, mini-swe-agent, FineEnvs and its HF spaces, OpenEnv | `M11-sweep-G1.md` |
| G2 | verifiers, prime-environments, research-environments, prime-agent, prime-rl, GitSwarm, autoresearch | `M11-sweep-G2.md` |
| G3 | NeMo-Gym, NeMo-RL, EnterpriseOps-Gym, Karotte, Xiaomi's verl fork, the MiMo v2.6 RL release | `M11-sweep-G3.md` |
| G4 | agentenv, OpenRLHF, ROLL, slime, SkyRL, checkpoint-engine, AReaL, TRL, verl, rLLM (their env interfaces) | `M11-sweep-G4.md` |
| G5 | every article in `07-articles`, all datasets in `08-hf-rl-env-datasets` (coverage matrix, I0-I10 design) | `M11-sweep-G5.md` |
| hunts | `interop/harbor`, `canonical`, `replay` (H1); `agentenv`, `verifiers`, `nemo_gym` (H2); `openenv`, `mimo`, `eog` (H3) | `M11-study-defects-H1.md`, `-H2.md`, `-H3.md` |

Every group found the same few defects independently. They are the most certain findings:

- An infrastructure fault recorded in the World during an episode was graded and scored on every foreign path: the
  NeMo-Gym export (G3 D1, H2 V1), the OpenEnv export (G1 D1, H3 V4), the agentenv gateway path (G4 D3, H2 V2-V3).
- An imported framework's own failure was scored 0: the NeMo-Gym import (G3 D10, G5 D1, H2 V4), the Harbor in-place
  verifier reading planted files (G1 D2, H1 V1), OpenEnv's raising environments (H3 V2).
- The policy could reach the grader or the control plane: `verify` as a NeMo-Gym tool (H2 V5), the agentenv gateway's
  open `/tools/enable` (G4 D5, H2 V6), planted `mask.json` / `steps.jsonl` in the verifiers export (G2 D1, H2 V8).
- Separate-verifier Harbor tasks were graded in the agent's container (G1 D4, G5 D3, H1 V2).
- The exit gate passed by construction: the OpenEnv and Harbor legs compared a replay with itself, the NeMo-Gym leg
  ran no NeMo-Gym code, the verifiers leg used stubs (G1 D3, G2 D9, G3 D17).

## The picks

DEPEND = use the framework, PORT = copy code (licence in the group document), PATTERN = rebuild the idea, TARGET = a
format we export to. "Built" means built in this audit round.

| # | Pick | Source | Kind, effort | Status |
|---|---|---|---|---|
| 1 | Every export masks an episode with a World infra fault; a grading exception masks (`judge_failsafe`) | NeMo-Gym `mask_sample` contract, `judge.py` | PATTERN, S | built (NeMo-Gym, OpenEnv, agentenv) |
| 2 | NeMo-Gym's own pydantic models validate every export route | `base_resources_server.py`, `config_types.AggregateMetrics` | DEPEND (test), S | built (seed, verify masked, close, reverify, aggregate) |
| 3 | Idempotent seed, legacy agent seed body, one grade per session | NeMo-Gym seed contract, `simple_agent` | PATTERN, S | built |
| 4 | Harbor's reward rules and verifier dir reset, everywhere a Harbor reward is read | Harbor `verifier.py`, `_reset_shared_step_verifier_dirs` | PORT (Apache-2.0), S | built (`interop/harbor/rewards.py`) |
| 5 | Refuse separate-verifier tasks in place until a separate grade exists | Harbor `trial.py` separate path | PATTERN, S | built (refusal); separate grade queued |
| 6 | verifiers plugin packaging: `__all__` with `HarborEnv`, local `path`, task timeouts kept | research-environments `apex_agents` | PORT (Apache-2.0), S | built (`crucible-vf`) |
| 7 | verifiers traces by `ok`, one trainable branch, exact logprobs from our generate server | verifiers `trace.py` | PATTERN, S | built |
| 8 | Agent proxy: `/mcp` only, role pinned; the gateway control plane never reaches the agent | agentenv gateway routes | PATTERN, S | built |
| 9 | Outside agents' calls are World steps (choke point, log, clock, events) | our World | PATTERN, M | built |
| 10 | agentenv's reward-mode verifier row; `metadata["score"]`; per-run World handle | agentenv `aggregate_verifiers`, `eval run` | TARGET, S | built |
| 11 | OpenEnv: full-observation digests; health replay decides policy vs infra; latched final | OpenEnv `Environment` contract | PATTERN, S | built |
| 12 | MiMo `verify.py` in an isolated subprocess, `detail.passed` as outcome, range mask | MiMo `run_verify.py`, verl `environment.py:575-578` | PATTERN, S | built |
| 13 | EOG comparison verbatim, typed columns, refusal only when it holds at the seed | EnterpriseOps-Gym `verifier.py:374-399` | PORT (Apache-2.0), S | built |
| 14 | SQLite authorizer on policy SQL (no ATTACH, no VACUUM INTO, no setting PRAGMAs) | SQLite authorizer | PATTERN, S | built |
| 15 | `interop/imports/` I0-I10 adapters over the existing RunTasks; Harbor `DatasetSpec` for provenance | G5 §4.2 | DEPEND + PATTERN, M-L | queued |
| 16 | Separate-verifier grade on Harbor's `_run_separate_verifier` path (collect hooks, artifacts) | Harbor `trial.py:885-925` | DEPEND, M | queued |
| 17 | NeMo-Gym packaging: `manifest.yaml`, config, dataset rows; ATIF v1.7; gate through `single_agent_turn` | NeMo-Gym `manifest.py`, `atif_v1_7.py` | DEPEND, M | queued |
| 18 | MiMo export from a World (`DatasetEnvActor` row), verl AgentLoop + reward manager, SkyRL step-wise output, TRL `rollout_func` fixes | XV `environment.py`, verl, SkyRL, TRL | TARGET, M | queued (M12 bridges) |
| 19 | Valid-mean fill for masked rows in the NeMo-RL bridge (NeMo-RL counts a mask's 0.0 in GRPO baselines) | XV `core_algos.py:304-350` | PATTERN, S | queued (M12) |
| 20 | Pinned, locked framework envs (`.envs/*` from each upstream's lock file) | FineEnvs `prepare.py` | PATTERN, S | queued |
