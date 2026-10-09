# Third-party code, frameworks and data

Crucible builds on open-source RL and agent frameworks. This file records exactly what is used, how, at which revision, and under which license. Paths are relative to the repository root.

## Frameworks we depend on (installed from the local clones, editable)

| Framework | Local path | Commit (date) | License | Used for (module) | Install |
|---|---|---|---|---|---|
| Harbor 0.23.0 | `RL-Environments/06-frameworks-and-env-code/harbor-framework-harbor` | a2fafbc (2026-10-03) | Apache-2.0 | task model + content hash (M0); sandbox environments, GPU subclass, remote Docker (M1) | extra `harbor` |
| agentenv-framework 0.9.1301 + agentenv-protocol 0.1.316 | `agentenv-framework`, `agentenv-framework/packages/agentenv-protocol` | 041afd7 (2026-10-08) | Apache-2.0 | gateway (role ACL, clock, triggers, trajectory) as a world backend (M1); A2A agents, peers, humans (M2) | extra `agentenv` |
| verifiers v1 | `RL-Environments/06-frameworks-and-env-code/primeintellect-ai-verifiers` | ae819de (2026-10-04) | MIT | training path (M11/M12, later) | extra `verifiers`, **conflicts** with `agentenv`/`mcp` (it pins `mcp==2.0.0`; agentenv needs `mcp<2`), so it lives in a separate environment |
| MCP Python SDK 1.30 | PyPI | — | MIT | App tool serving/clients (M1) | extra `mcp` |
| fastapi-mcp 0.4, SQLAlchemy 2.1, huggingface-hub 2.2 | PyPI | — | MIT | running OpenEnv AWM services (M1 generality) | extra `openenv-awm` |

## Code ported into `src/crucible/_vendor/` (each folder has `ORIGIN.md`; changed lines are tagged `# [crucible]`)

| Folder | Source | Revision | License | Changes |
|---|---|---|---|---|
| `mimo_sidecar/` | Xiaomi MiMo v2.6 RL release, `general/envs/*/{mcp_http.py, mcp_bridge.py}` | md5 69484c5f…, 98d327a9… | Apache-2.0 | `mcp_http.py`: bind address parameter (default 127.0.0.1 instead of 0.0.0.0). `sidecar_entrypoint.py` not ported (Linux-only readiness probe, `/dump` and `/verify` not implemented upstream) |
| `openenv_awm/` | OpenEnv `envs/agent_world_model_env/server/{data_loader.py, db_manager.py}` + `_patch_env_code` from `scenario_manager.py` | 49aa302 | BSD-3-Clause | per-file lazy download (upstream fetches all 7 files, ~500 MB, on first access) |

## Patterns re-implemented (no code copied)

| Pattern | Source | Where in Crucible |
|---|---|---|
| failure vocabulary (names reused verbatim for compatibility) | NeMo-Gym `nemo_gym/failure_kinds.py` | `core/schema/run.py` |
| content hash of a task directory (`rel\0sha256\n`) | Harbor `publisher/packager.py` | `core/ids.py` |
| `module:Class` registries, `env:` secret references, fail-loud duplicate registration | agentenv `config/runtime.py` | `core/registry.py`, `core/config.py` |
| virtual clock (t0 + elapsed·rate, ≤ 86400) and nl / tool / permission trigger actions | agentenv `env/gateway/{clock,triggers}.py` | `world/clock.py` (step mode added) |
| GPU compose overlay appended to the compose stack | Harbor `environments/kata.py` | `interop/harbor/gpu_docker.py` |
| user-simulator persona and done marker | verifiers v1 `envs/user_sim/env.py` | `actors/simulated.py` |
| scenario fields, per-message consistency judge, simulator failures mask the sample | NeMo-Gym `resources_servers/conversational_tool_use_simulation` | `actors/simulated.py`, `core/schema/run.py` |
| "never hallucinate; unknown info is unavailable; disclose progressively; out-of-scope token" | tau2 guidelines (via NeMo-Gym `indian_banking`) | `actors/simulated.py` |

## Data used for tests and gates (never in training or benchmark splits)

| Data | Source | License / terms | Use |
|---|---|---|---|
| 267 MiMo "general" environments (266 complete) | `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/general/envs` | Apache-2.0 (release README) | M1 generality gate |
| AgentWorldModel-1K (1,000 scenarios) | HF `Snowflake/AgentWorldModel-1K` → `crucible/data/cache/awm` (gitignored) | see dataset card | M1 generality gate |
| Public review comments | GitHub `PointCloudLibrary/pcl`, `NVIDIA/cccl` (via the REST API + raw.githubusercontent.com) → `crucible/data/cache/github` (gitignored) | public repositories | M2 reviewer-fidelity and grounding gates |
