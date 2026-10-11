# M1 taxonomy 00: what Crucible's World can express today (from the code, 2026-10-12)

The other half of the capability matrix. Every entry names the code that provides it; "absent" means no code does.
Paths are relative to `crucible/src/crucible/`.

## State (what an environment can hold)

| Kind | Provided by |
|---|---|
| Files with full git history (managed set = tracked + untracked − ignored) | `world/state/fs.py` (`fs`, `fs_git`), snippets `world/state/snippets/fs.py` |
| SQLite database (online backup, restore into the live DB, canonical dump) | `world/state/sql.py` |
| Postgres server in its own container (schema + per-table row hashes; `pg_dump`/`pg_restore`) | `world/state/postgres.py`, `world/apps/postgres_app.py` |
| JSON documents | `world/state/json_doc.py` |
| Processes, builds, tests, GPU probes | `world/state/process.py` |
| Browser (URL, cookies, localStorage) | `world/apps/browser.py` (`BrowserStateAdapter`) |
| Website over a SQLite state | `world/apps/website.py` |
| agentenv data services | `interop/agentenv/mcp_env.py` (`AgentEnvDataAdapter`) |
| Mailbox, document library (docx/xlsx/pptx/pdf extraction), JSON API, toolchain shell | `world/apps/{mailbox,doc_library,json_api,toolchain}.py` |
| Any MCP server, stdio or streamable HTTP, one persistent session per App | `world/apps/mcp_server.py`, `world/packaging/session.py` |
| Hidden-check CI runner (separate runner, results only) | `world/apps/ci.py` |
| Simulator- or game-owned state | through imported environments only (OpenEnv in-process, NeMo-Gym, AWM); no native simulator App template |
| HTTP mock | declared in `AdapterKind` (`core/schema/state.py:17`); no adapter registered |

## Actions and protocols

| Kind | Provided by |
|---|---|
| Tool calls with JSON-schema validation, ACL, deadlines, output cap, per-tool quotas (one gate) | `world/world.py` (`World._call_tool`) |
| MCP (in-process byte-identical to served) | `world/packaging/mcp_tools.py`, `app_server.py` |
| Bash and file editing (demoted user, process-group cleanup) | `world/apps/workspace.py`, `world/guard/guard.py` |
| Code cells in a persistent kernel (recursive agent) | `actors/rlm/kernel.py`, `actors/rlm/policy.py` |
| Online interception of dangerous calls | `world/intercept.py` |
| Free-text single completion (no tools) | **absent** as a task type |
| Keyboard / mouse / pixel actions, continuous control | **absent** (only through an imported environment's own action type) |

## Actors

| Kind | Provided by |
|---|---|
| Policy actors (one or several) | `core/schema/world.py` (`ActorKind`), `actors/policy.py` |
| Grounded simulated people (quote mode, temporal masking, citation checks) | `actors/grounded.py` |
| Scenario-driven simulated users (persona, consistency judge, masking on simulator failure) | `actors/simulated.py` |
| Humans and peers (file queue, callable, A2A transports) | `actors/human.py`, `actors/transports.py`, `interop/agentenv/a2a.py` |
| System actors (CI runner, relay) | `actors/scripted.py` |
| Turn-taking multi-agent (seats, `not_your_turn`, per-seat rewards) | imported OpenEnv games only (`interop/openenv/import_env.py:32-40`); no native turn-order primitive in `WorldSpec` |
| Simultaneous-move or team games | **absent** |

## Time

| Kind | Provided by |
|---|---|
| Step time (`step_dt_s` per world step), wall time (rate ≤ 86400, pause-safe), simulation time (App-advanced) | `core/schema/world.py` (`ClockSpec`), `world/clock.py` |
| Events at a time, after a duration or event, every N (fixed or exponential), on a tool call, on a state predicate; nl / tool / permission actions | `core/schema/world.py` (`EventWhen`, `EventAction`), `core/triggers.py` |
| Virtual clock and seed inside tool modules (deterministic replay) | `world/packaging/virtual_env.py` |
| Environment-terminated episodes, step limits, timeouts, context-overflow stops | `core/schema/actions.py` (`ends_episode`), `actors/policy.py` (`EpisodeLimits`) |
| Long horizons: compaction segments, checkpoints every K steps, resume mid-episode | `actors/policy.py` (`ContextConfig`), `runtime/checkpoint.py` |

## Observations

| Kind | Provided by |
|---|---|
| Text tool results (head and tail cut with a banner) | `world/world.py` |
| Images on tool results (refs in traces, bytes live only) | `core/schema/actions.py` (`ImageRef`), `world/world.py:457` |
| Images or audio in messages and instructions | **absent** (`Message.text` is plain text) |
| Vision training (multimodal chat codec) | **absent** (`train/chat.py` is text-only) |

## Grading and reward

| Kind | Provided by |
|---|---|
| Hidden verifiers over a verifier-only view; trusted channels; differential checks; noise-aware performance | `verify/` (M7) |
| Multi-key reward document with gates (a failed gate forces 0), progress, rubric, efficiency, named components | `core/schema/reward.py` |
| Dense per-step progress Φ from milestones, ΔΦ per step, mapped onto each turn's tokens | `credit/`, `workflow/potential.py`, `runtime/record.py` |
| Rubric and LLM-judge graders | `verify/` (judges with specialists); judge model and version not pinned per rollout |
| Env-native per-step reward (from an imported environment) | `interop/openenv/import_env.py` (OpenEnv step reward) |
| Per-token rewards and advantages | `train/` (dense turn credit to tokens); no external per-token reward input |
| Single-turn prompt plus checker | **absent** |

## Infrastructure and isolation

| Kind | Provided by |
|---|---|
| Local process backend | `world/backends/local.py` |
| SSH exec backend | `world/backends/ssh.py` |
| Docker sandbox, gVisor sandbox, warm pools | `world/sandbox.py`, `world/pool.py` |
| Harbor environments (Docker, remote Docker over SSH with GPUs, any Harbor provider) and in-place Harbor tasks with oracle | `interop/harbor/world_backend.py`, `gpu_docker.py`, `inplace.py` |
| agentenv gateway backend and agentenv sandbox providers (cloud GPUs) | `interop/agentenv/world_backend.py`, `sandbox.py` |
| Isolation guard: fresh account, verified git scrub, ownership, pinned verifier side, demoted tools, egress self-test; red-team suite | `world/guard/`, `tests/redteam/` |
| Snapshot and restore with fingerprints, image-layer digest check | `world/world.py`, `world/state/*` |
| World audit command | `world/audit.py` |

## Importers (foreign environments running inside Crucible)

| Source | Adapter | Status |
|---|---|---|
| Harbor task directories | `interop/harbor/inplace.py` (in place, oracle), `verify_dir.py` (patch seam) | built; per-family verification open |
| OpenEnv (in-process factories), two-player games | `interop/openenv/import_env.py` | built; HTTP/WebSocket client mode absent |
| AgentWorldModel-1K worlds and graders | `interop/openenv/awm.py`, `awm_task.py` | built |
| MiMo v2.6 environments | `interop/mimo/` | built; judge-graded families need a judge |
| NeMo-Gym resources servers | `interop/nemo_gym/import_server.py` | built for workplace_assistant, ruler/mrcr |
| verifiers v1 | `interop/verifiers/v1_import.py` | built; v0 absent |
| EnterpriseOps-Gym | none | absent |
