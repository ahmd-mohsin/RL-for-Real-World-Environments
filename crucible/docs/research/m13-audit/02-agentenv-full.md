# M13 audit 02: agentenv-framework capabilities vs Crucible's use of them

**Date:** 2026-10-10.
**Versions:** agentenv-framework `041afd7` (v0.9.1301); Crucible `90200ec`.
**Method:** code read (agentenv source, protocol package, README/AGENTS.md; Crucible `interop/agentenv/`, `world/`, M01, DESIGN). Two claims were run in Crucible's `.venv`: P1 creates a sandbox through `gpu_docker`; P2 calls a stock step's extension on a `crucible_world` deployment.

**Path prefixes:**
- `AE/` = `agentenv-framework/src/agent_env/`
- `PP/` = `agentenv-framework/packages/agentenv-protocol/src/agentenv_protocol/`
- `CR/` = `crucible/src/crucible/`

**Use:** **a** = Crucible uses it; **b** = own equivalent; **c** = neither. "Matters?" judges each **c** against long horizons, horizontal scale, all sandbox kinds, GPUs, many harnesses, human/simulated actors.

## 0. Findings in one page

1. **Crucible uses a thin slice:** the gateway process (ACL, clock, a trigger subset, trajectory), four sandbox providers via `create_container`, A2A client/server primitives, three plugin entry points. No `Env` type, `EnvironmentProvider`, runner, store, written artifact, stock step, verifier or A2A extension.
2. **Two plugin pieces described as done do not work.**
   - **`gpu_docker` can never create a sandbox.** agentenv's type guard kills every sandbox it makes (probe P1).
   - **Stock agentenv steps fail on `crucible_world`.** The step publishes a `DeployedGatewayEnv` with no env card and no `mcp_url`. As a result, `sync_env_clock`, `modify_env_tool_access`, `register_env_triggers` and `collect_artifacts` all raise, and `deploy_agent` hands the agent a `None` MCP URL (probe P2).

   The gate missed both: it calls `execute()` directly, never `Task.run()` (`tests/unit/test_interop_gate.py:199-218`).
3. **agentenv is not a scale runtime:** one 2-worker `LocalRunner`, no aggregation in `eval run`, no token or logprob capture, no sandbox snapshot/pause/resume, GPUs on Modal only. Crucible's runtime, snapshot and capture layers (**b**) own all of this. agentenv's real value that Crucible does not yet use: the in-sandbox gateway (`EnvironmentGatewayProvider`) for scale; the rest of the trigger language (state triggers, predicates, seeded stochastic arrivals, barriers) for long-horizon dynamics; A2A agents as harnesses; agentenv-protocol MCP env images as Apps.

## 1. Capability matrix

### 1.1 Environment types and the env protocol

| agentenv capability | Where | Crucible | Use | Matters? |
|---|---|---|---|---|
| `MCPServerEnv` (`mcp_server`): an MCP server from a `DockerImageArtifact`, with data-plane loads | `AE/env/envs/mcp_server.py:37-206` | Its App templates (`common/mcp_server` spawns a host process, `CR/world/apps/mcp_server.py:54`) cannot run an agentenv env image | c | **Yes, for diversity.** agentenv-protocol images are Scale's enterprise env catalogue → C9 |
| `MultiEnv` (`multi`): N MCP servers plus websites; universe loads with derived concurrency, resumable loads and snapshot bake | `AE/env/envs/multi_env.py:42-547` | A multi-App `World` | b | — |
| `ServiceDBEnv` (Postgres) plus the `_changelog` audit table | `AE/env/envs/service_db.py:33`; `AE/providers/env_state/env_state_provider.py:262-395` | `common/postgres_app` and SQL state adapters | b | Changelog ids: low |
| `WebsiteEnv` plus the `website_browser` MCP | `AE/env/envs/website.py:35`; `AE/providers/env_providers/env_gateway_provider.py:660-667` | `common/website`, `common/browser` | b | — |
| `GatewayEnv` (`gateway_server`) | `AE/env/envs/gateway_server.py:11` | Starts `agent_env.env.gateway.entrypoint` as a subprocess (`CR/interop/agentenv/world_backend.py:206-226`) | a (partial) | — |
| Env server SDK: `@tool`, `@extension`, JSON-RPC `data/reset\|add\|get` at `/agentenv`, env card at `/.well-known/agent-env.json` | `PP/agent_env_environment.py:44-127,229-269` | `AppMcpServer` serves tools only, with no card and no data plane | c | Medium: needed to export Crucible Apps to agentenv hubs → C15 |
| `EnvironmentUniverseArtifact`; `snapshot_env` (service state via `get_data`); restore via `load_artifact` | `AE/task_step/task_steps/snapshot_env.py:408-688`; `load_artifact.py:577-590` | State adapters plus `WorldRef` (`CR/world/world.py:574-630`) | b | Mirroring (research 05 §4.4) is unbuilt → C11 |
| Env-state providers (acquire → prepare → load; `local_postgres`) | `AE/providers/env_state/env_state_provider.py:1-164` | `StateAdapter` | b | — |

### 1.2 Gateway

| agentenv capability | Where | Crucible | Use | Matters? |
|---|---|---|---|---|
| Aggregates tools over internal MCP servers; names must be unique | `AE/env/gateway/gateway.py:649-677` | One MCP server per App, with tools named `<app>__<tool>` (`world_backend.py:54,193`) | a | — |
| Per-role ACL `/tools/{enable,disable}` with a rule chain | `gateway.py:183-189,340-348,689-725` | `_program_acl` and restore of allowed sets (`world_backend.py:265-286,459-463`) | a | — |
| REST `/step` with the `AgentEnv-Role` header | `gateway.py:727-829` | `GatewayRouter` (`world_backend.py:58-81`) | a | — |
| MCP streamable HTTP, with the role taken from the header | `gateway.py:295-330` | `mcp_url` for external harnesses (`world_backend.py:173-176`) | a | — |
| `consistent` mode: a global tool-call lock | `gateway.py:114,422-427` | Default `consistent` (`world_backend.py:213`) | a | See U2 |
| Virtual clock, rate-scaled, rate ≤ 86400; `get_time` tool registered when the clock is armed | `AE/env/gateway/clock.py:114-184`; `gateway.py:247-262,868-907` | Wall mode delegates to it (`world_backend.py:199-200,292-298`); step and sim clocks are Crucible's own | a / b | — |
| `sync_time` push to backing servers (`env_get_time_url`) | `AE/task_step/task_steps/sync_env_clock.py:81-129` | Apps read Crucible's own time | b | — |
| Triggers, core: time triggers (`at`, `every`, `count`), action triggers (`tool`, `repeat`), and `nl`/`tool`/`permission` actions with an A2A executor | `AE/env/gateway/triggers.py:401-575,1034-1180` | `_trigger`/`_register_triggers`/`RelayAgent` (`world_backend.py:300-357`; `CR/interop/agentenv/a2a.py:83-99`) | a | — |
| Triggers, the rest: `state` triggers with check pipelines, `where` predicates, `${args.*}`/`${result.*}` templating, `barrier` at `provoking_call`, stochastic `every` (`dist: exp`, `seed`), `after`+`offset` anchoring, `until`, per-action `verify` | `triggers.py:602-722,745-790` | `EventWhen` and `EventAction` cannot express any of these (`CR/core/schema/world.py:112-150`) | c | **Yes.** Reactive, stochastic and reproducible world dynamics over long horizons → C8 |
| Trigger firing log, `/triggers/state` | `triggers.py:723-744` | Settled snapshot and restore (`world_backend.py:373-431`) | a | — |
| Env trajectory JSONL, `GET /trajectory` | `gateway.py:992-1020` | Replayed by `crucible_verify` (`CR/interop/agentenv/plugin/steps.py:83-91`) | a | See B4 and B5 |
| Env card, data-plane forwarding, `/svc` and `/website` REST proxies | `gateway.py:160-181,546-600` | Not read and not used | c | Card: **yes** (B2). Proxies: low |

### 1.3 Sandbox providers

| agentenv capability | Where | Crucible | Use | Matters? |
|---|---|---|---|---|
| `local`: Docker on this host. Ignores cpu, memory and timeout; no GPU; allow-all network only | `AE/providers/sandbox_providers/local_sandbox.py:431-539` | Its own `DockerSandbox` (pids, memory, `--network none`, nvidia runtime, remote `DOCKER_HOST`; `CR/world/sandbox.py:75-117`) | b | — |
| `modal`: containers; GPU through the provider-level `gpu=`; domain/CIDR allowlist | `modal_sandbox.py:180-400` (GPU `:195-206,326-332`; network `:57-68,188-190`) | `AgentEnvSandbox` builds `ModalSandboxProvider(gpu=…)` (`CR/interop/agentenv/sandbox.py:119-123`) | a | — |
| `modal_vm`, `e2b` (network changeable at runtime), `sail_vm` (auto-sleep, model-key injection) | `modal_vm_sandbox.py:275-402`; `e2b/provider.py:32-382`; `sail_vm/provider.py:130-394` | Reachable through `build_sandbox_provider` (`sandbox.py:126-128`), but none is tested live. Sail `auto_sleep` is not exposed | a / c | **Auto-sleep: yes** for long, idle, human-paced episodes → C12 |
| Chained fallback (`"modal,e2b"`) | `chained_sandbox_provider.py:50-91` | Breaks: Crucible calls `create_container`, which the chain lacks (B7) | c | Yes: capacity and availability across clouds |
| `NetworkPolicy` (`ALLOW_ALL` or `ALLOWLIST`), refused when a provider cannot enforce it | `sandbox.py:76-134`; `sandbox_provider.py:474-479` | `network_policy()` (`CR/interop/agentenv/sandbox.py:42-49`) | a | — |
| Attribution tags | `sandbox_provider.py:41-54` | Sends only `{crucible_world}` (`sandbox.py:136`) | a (thin) | Yes: per-sample cost attribution → C5 |
| `EnvironmentGatewayProvider`: the gateway, MCP servers, servicedb and websites in one sandbox (compose in VM mode; i6pn containers on Modal) | `AE/providers/env_providers/env_gateway_provider.py:614-958,1190-1207` | Gateway and App servers run **on the orchestrator host**, and every tool call is an exec RPC into the sandbox (`world_backend.py:143-157,187-196`) | c | **Yes, for horizontal scale.** Host processes, ports and exec round-trips grow with N → C10 |
| Snapshot, pause, resume or fork of a sandbox | absent in every provider | `WorldRef` through state adapters, restored into fresh containers | b | — |
| GPU outside Modal; remote or multi-host Docker | absent | `DockerSandbox` with `gpu_host` / `DOCKER_HOST=ssh://` | b | — |
| Rate limits or concurrency caps on provisioning | absent | `RunSpec.concurrency` (`CR/runtime/run.py:42,101`), `EnvWorkerPool` (`CR/train/envpool.py:38`) | b | — |

### 1.4 Task steps

| agentenv step(s) | Where (`AE/task_step/task_steps/`) | Crucible | Use | Matters? |
|---|---|---|---|---|
| `deploy_env`, `reset_env`, `snapshot_env`, `teardown_sandboxes` | `deploy_env.py:20`, `reset_env.py:53-70`, `snapshot_env.py:158`, `teardown_sandboxes.py:38` | `crucible_world` replaces `deploy_env`. The other steps resolve `Env.get(env_id)` from the store, which holds no Crucible env (`reset_env.py:62`) | c | Yes: stock lifecycle steps and the explorer → C6 |
| `deploy_agent`, `install_agent`, `add_skills`, `peer_agents` (A2A agents with mcp-config, skill-config and agent-config) | `deploy_agent.py:88-410`, `install_agent.py:91`, `add_skills.py:169`, `peer_agents.py:30` | Unused, and broken against `crucible_world` (B2) | c | **Yes, for many harnesses** → C2, C7 |
| `prompt_agent`: multi-turn, user-sim `{"message","done"}`, trigger `/decide`, per-turn native trajectory, `SnapshotSeries` | `prompt_agent.py:113-650` | Crucible's own episode loop and actors | b | Reuse through an A2A harness (C7) |
| `deploy_human_agent` (hub-routed human A2A peer), `review` (hub decision gate, 3 h) | `deploy_human_agent.py:22-100`; `review.py:48-144` | `HumanActor` and `PeerActor` over `A2ATransport` (`CR/actors/human.py`; `CR/interop/agentenv/a2a.py:102-115`) | b | `review` needs Scale's hub (no console in the repo): low |
| `register_agent_triggers` (user-sim schedules) | `register_agent_triggers.py:44-108` | Simulated actors plus `World` events | b | — |
| `register_env_triggers`, `modify_env_tool_access`, `sync_env_clock`, `apply_server_config` | `register_env_triggers.py:91`, `modify_env_tool_access.py:66`, `sync_env_clock.py:86`, `apply_server_config.py:75` | Done inside the backend. The stock steps raise on `crucible_world` (B2) | b | Must work for agentenv-authored tasks → C2 |
| `collect_artifacts`, `load_artifact`, `run_code`, `run_docker_container` | `collect_artifacts.py:226`, `load_artifact.py:131`, `run_code.py:59`, `run_docker_container.py:22-287` | `CR/verify/collect.py`, `CR/verify/stage.py` | b | — |
| Verifiers: `rubrics_verifier` (LLM judge, frames, 5 formats), `run_container_unit_tests_verifier`, `verify_sandbox`, `env_outcome_verifier`, `agent_prompt_response_verifier`, `aggregate_verifiers` | `verifiers/*` | The single reward producer and `verify/judge_item.py` | b (deliberately not used) | Do not adopt: they clamp scores and turn infra failures into 0 (U9) |
| Validators: `verify_a2a_*` (12), MCP spec conformance, env card and core protocol | `a2a_agent_validator/`, `mcp_env_validator/`, `env_card_validator/` | none | c | Medium: CI conformance for exported envs and the relay → C14 |
| `Task.run`: DAG, concurrent ready steps, `retry_config` rollback through the journal, resume from `start_step`, redaction | `AE/task/task.py:339-749`; `AE/task_step/context.py:13-15,207-230` | M10 journal and phase machine (`CR/runtime/`) | b | — |

### 1.5 Runners, eval, scale, plugins, stores, UI, A2A

| agentenv capability | Where | Crucible | Use | Matters? |
|---|---|---|---|---|
| `Runner` seam; `LocalRunner(workers=2)` (in-memory; runs orphaned by a restart are marked FAILED) | `AE/runner/runner.py:89-127`; `local_runner.py:38-60` | `crucible.runtime` (`k`, concurrency, resumable journal, blame) | b | — (no remote runner upstream) |
| `eval run --k --max-concurrency`; `task run --k`; `task run-batch`; bundles (`MAX_CONCURRENT_RUNS=4`, ledger) | `AE/cli/eval/run.py:76-181`; `AE/cli/task/run.py:412,545-609`; `AE/bundle/run.py:42` | Runtime with pass@k and masking | b | Bundle export: low → C13 |
| Plugin entry points: `envs`, `env_providers`, `state_providers`, `explorer_plugins`, `cli_plugins`, `bundles` | `AE/plugins/_registration.py:24-30`; `AE/plugins/_cli.py:22-23` | Registers only `sandbox_providers`, `task_steps`, `artifacts` (`pyproject.toml:47-55`) | c | `envs` and `env_providers`: **yes** (C6). The rest: low |
| Stores: document (SQLite/Mongo/Dynamo/Firestore), object (FS/S3/GCS), image (OCI/ECR), secret | `AE/store/*`; `AE/config/runtime.py:933-942` | Local content-addressed store (`CR/core/store/__init__.py:1-10`) | b (local only) | Yes for multi-host. Could back the Crucible Store with agentenv's ObjectStore → C11 |
| Versioned artifacts (file, docker_image, universes, skill, cli) | `AE/artifact/artifacts/*` | `crucible_ref` type is declared but **never written** (`CR/interop/agentenv/plugin/artifacts.py:12-16`) | c | Medium → C11 |
| Explorer UI (runs, trajectory viewer, trigger timeline, rubric matrix) | `AE/explorer/app.py:1-98`; `routers/*` | none | c | Medium: debugging multi-actor episodes. Needs C6 |
| A2A framework (`AgentEnvAgent.create_app`), client (`send_a2a_message`, `poll_a2a_task`) | `PP/a2a_agent/framework.py:232-278`; `AE/a2a_agent/protocol.py:145-262` | `ServedAgent`, `A2AClient`, `RelayAgent` | a | — |
| A2A extensions: mcp-config, agent-config, skill-config, trajectory/v1, snapshot/v1, peer-agents, triggers, install | `PP/a2a_agent/extensions.py:879-1116` | none | c | **Yes:** configure and inspect any A2A harness; snapshot/restore an agent mid-episode → C7 |
| Token, logprob or model-proxy capture | absent; usage tokens dropped (`AE/a2a_agent/protocol.py:257-261`) | Capture proxy, `rollout.json` (`CR/runtime/record.py:4,96`) | b | — |

## 2. Bugs and limitations

### 2.1 In Crucible's integration (found by this audit)

| # | Defect | Evidence | Effect |
|---|---|---|---|
| B1 | **`gpu_docker` cannot create a sandbox.** The provider is not built in, so agentenv wraps `create_*` with a type guard. The guard terminates any sandbox whose `.type != "gpu_docker"`, and `LocalSandbox.type` is `"local"`. | `AE/providers/sandbox_providers/sandbox_provider.py:482-528`; `local_sandbox.py:132`; `CR/interop/agentenv/plugin/gpu_docker.py:20-22`. **Probe P1:** `build_sandbox_provider('gpu_docker').create_vm(exposed_ports=[])` → `SandboxProviderTypeError`. No test references the class. | No GPU path through agentenv's own deploys |
| B2 | **`crucible_world` publishes an incomplete env record.** `DeployedGatewayEnv` is created without `environment_card(_url)`, `mcp_url` or `mcp_server_name`. | `CR/interop/agentenv/plugin/steps.py:51-52`; `AE/env/env.py:83-121` (`invoke` needs the card). **Probe P2:** `invoke(clock/v1, set_time)` → `EnvCapabilityUnsupported`; with the card fetched from `/.well-known/agent-env.json`, the same call succeeds. | `sync_env_clock`, `modify_env_tool_access`, `register_env_triggers` and `collect_artifacts` raise. `deploy_agent` registers `url=None` (`deploy_agent.py:50-55,334`). `reset_env` and `sync_env_clock._service_names` call `Env.get` on an id the store does not hold. The DESIGN.md:729 claim ("stock agentenv steps work unchanged") is false |
| B3 | Gate rows use `"result": "pass"/"fail"` strings, but agentenv's aggregators test `is True`. | `steps.py:93`; `AE/task_step/task_steps/verifiers/scoring.py:21-24` | `aggregate_verifiers(all_pass)` scores every Crucible gate as failed |
| B4 | The gateway trajectory is append-only for the life of the process, and `on_reset` does not mark an episode boundary. `crucible_verify` replays the whole file. | `gateway.py:121-122,1008-1020`; `world_backend.py:359-371`; `steps.py:83-91` | Correct only for one episode per World. A reused World replays every earlier episode |
| B5 | The HMAC actor token rides inside tool `arguments`, which the gateway logs verbatim. Nothing strips it before replay. | `world_backend.py:70`; `gateway.py:777-783`; no `__actor` handling in `CR/interop/replay.py` | Replays of `GatewayRouter`-driven episodes carry `__actor`. The token is disclosed in `gateway_trajectory.jsonl` |
| B6 | The gateway binds `127.0.0.1`. | `world_backend.py:210`; agentenv rewrites loopback to `host.docker.internal` = the docker0 bridge on Linux (`local_sandbox.py:558-569`) | Agent containers on Linux cannot reach the World |
| B7 | `AgentEnvSandbox` always calls `create_container`. A chained spec provides only `create_sandbox`. Modal is constructed directly, bypassing `[sandbox.providers.modal]` config. | `CR/interop/agentenv/sandbox.py:119-143`; `chained_sandbox_provider.py:50` | Fallback chains fail, and per-deployment Modal config (app name, tags) is ignored |
| B8 | `A2ATransport` returns `TerminalResponse.response_text`. That keeps only the last text part and drops usage tokens. A `failed` or `canceled` task comes back as ordinary text. | `AE/a2a_agent/protocol.py:241-262`; `CR/interop/agentenv/a2a.py:73-77,108-109` | Human and peer actor failures are not blamed; peer cost is invisible |
| B9 | `crucible_verify` rebuilds the replay task from `instruction` only, so the step's `task_kwargs` are lost. | `steps.py:88` vs `steps.py:47` | Parametrized tasks replay against a different World |
| B10 | Store and plugin docs claim artifact mirroring. No code writes `crucible_ref`. | `CR/core/store/__init__.py:9-10`; `plugin/artifacts.py` | Documentation is wrong |

### 2.2 Upstream agentenv limitations Crucible must work around

| # | Limitation | Evidence | Crucible status / workaround |
|---|---|---|---|
| U1 | The time-trigger driver runs only inside the FastMCP lifespan, i.e. while an MCP session is open. When an external client disconnects, the driver stops. | `gateway.py:126-150` | Keepalive session exists (`world_backend.py:228-252`). The disconnect case is still open (`docs/modules/M01-world.md:193`) |
| U2 | `consistent` mode serializes every role's tool calls behind one lock. The tool timeout is fixed at 600 s, and the barrier timeout at 30 s. | `gateway.py:74-75,114,422-427` | Multi-actor worlds run serially. Long tools (builds, GPU jobs) get a 504, so model them as submit-and-poll tools |
| U3 | Trajectory events carry no role and no episode id. | `gateway.py:777-783,992-1006` | Attribute calls from the World trace, never from the gateway log (B4, B5) |
| U4 | Tools are discovered once; duplicate names raise. The data plane accepts only a single child. | `gateway.py:576-600,626-677` | Namespace tool names (done). Apps cannot be added mid-episode |
| U5 | The clock has only a wall rate (no step or advance API). The driver polls every 1 real second, so at rate 86400 an action can land up to one virtual day late. `watch_roles` and the executor are fixed until `clear`. | `clock.py:114-184`; `triggers.py:398,548-575` | Step and sim clocks stay in Crucible. Cap the rate for tasks that are sensitive to trigger timing |
| U6 | GPU only on Modal, and GPU+i6pn is rejected. Env deploys never pass a `network_policy`. `local` ignores limits. Modal's `get_sandbox` drops the policy. The chain's deadline is 180 s. Nothing can snapshot or pause a sandbox. | `modal_sandbox.py:195-206,308-312,394-400`; `env_gateway_provider.py:647,924`; `local_sandbox.py:436-458`; `chained_sandbox_provider.py:12` | GPUs come from `DockerSandbox`; isolation and egress checks from `guard/isolate.py`; snapshots from `WorldRef` |
| U7 | Graders clamp scores to [0,1]. A missing reward becomes 0, and so does an aggregate with no rows. An infra `error_type` is scored 0. Cancelled turns count as success. `env_outcome_verifier` runs `exec_module` in the orchestrator. Graders run in the agent's container. | `run_container_unit_tests_verifier.py:336-345,410`; `scoring.py:19-20,39`; `agent_prompt_response_verifier.py:86-99`; `prompt_agent.py:543`; `env_outcome_verifier.py:107-113` | `crucible_verify` stays the only scorer: unclamped, and masked grades raise |
| U8 | `run_docker_container` builds its own `docker run`: no GPU, no provider involvement, unquoted env values. | `run_docker_container.py:195-258` | Use Crucible's `verify/stage.py` for any container that needs a GPU |
| U9 | `eval run` reads `metadata["score"]`, which nothing writes, and prints PASSED for any run that did not raise. LocalRunner runs die on restart. SQLite runs on one connection with a busy timeout. | `AE/cli/eval/run.py:38,158-170`; `local_runner.py:51-60`; `sqlite_document_store.py:105` | Use the M10 runtime for every count. `crucible_verify` may also set `metadata["score"]` (C3) |
| U10 | No token ids or logprobs anywhere. The native trajectory lives in a 1024-entry in-memory LRU. | `AE/a2a_agent/protocol.py:257-261`; `PP/a2a_agent/framework.py:92,1158` | Point every A2A harness at Crucible's capture proxy (C7) |
| U11 | Compose YAML is written with unescaped env values. | `env_gateway_provider.py:361-363,490-493` | Validate values before C10 relies on it |

## 3. Changes needed to exploit agentenv fully

| ID | Change | Acceptance test | P | Effort |
|---|---|---|---|---|
| C1 | **Fix `gpu_docker`.** Add `GpuDockerSandbox(LocalSandbox)` with `type="gpu_docker"`, returned by `create_vm` and `get_sandbox`. Make `shares_network_with("gpu_docker")` true. Read `docker_host`, `runtime` and `shm` from `from_config`. | Unit: `build_sandbox_provider("gpu_docker").create_vm()` returns `.type=="gpu_docker"` (the probe P1 reversal). Lab 5090: a stock `deploy_env` of an `MCPServerEnv` with `sandbox_type=gpu_docker` runs `nvidia-smi` in its container | P0 | S |
| C2 | **Complete the `crucible_world` record.** Fetch the gateway card, then set `environment_card_url`, `environment_card`, `mcp_url=<gw>/mcp` and `mcp_server_name`. Make the bind host configurable (docker0 bridge on Linux). Register an `Env` stub so `Env.get(env_id)` resolves. | A task JSON run through **`Task.run()`** completes: `crucible_world → sync_env_clock → modify_env_tool_access → register_env_triggers → deploy_agent` (agentenv's echo A2A test agent, `local`) `→ prompt_agent → crucible_verify`. Asserts that the agent listed the role-filtered World tools. Runs on Linux CI and macOS | P0 | M |
| C3 | **`crucible_verify` correctness.** Rows `result: bool`. Replay only the episode's window: record `X-Trajectory-Total-Bytes` at reset, and give each step an `episode_key`. Strip `__actor` in `from_gateway_jsonl`. Rebuild the replay task with `task_kwargs`. Also write `metadata["score"]` (unclamped; `None` when masked). Correct DESIGN.md:729 and the store docstring. | `aggregate_verifiers(all_pass)` = 1 iff reward = 1. Two episodes on one World yield two disjoint step lists, each identical to native. A `GatewayRouter`-driven episode replays byte-identically. A parametrized task gate | P0 | S |
| C4 | **A2A transport fidelity.** Parse the terminal message as a whole: all text parts, `usage` tokens and cost, `error_*`. Map `failed`/`canceled`/`rejected` to a typed actor failure with blame. | Fake A2A peer returning failed, multi-part and usage replies → the trace has a failure with blame `harness`/`infra`, the full text, and usage booked to the run budget | P1 | S |
| C5 | **Sandbox adapter breadth.** Use `create_container` when the provider offers it; otherwise use `create_sandbox` (chains). Build Modal through the registry: `from_config(**cfg, gpu=…)`. Attribution `{world, run_id, sample_key, harness}`. Expose `timeout_s` above 2 h and Sail `auto_sleep`. | Spec `"modal,e2b"` with fakes: the first raises, the second serves the World. Attribution tags are visible on the sandbox. A Sail sandbox sleeps during a 30-min idle human wait and resumes with identical fingerprints | P1 | S |
| C6 | **Design B: native agentenv Env.** Register the `crucible_world` Env type (`agent_env.envs`) and an `EnvironmentProvider` (`agent_env.env_providers`) that starts a **Crucible World server** (not the in-process `WORLDS` dict), so any runner process can attach. Map `reset` → `World.reset`, `snapshot_env` → `WorldRef` plus a `crucible_ref` artifact, and `load_artifact` → `World.restore`. | Stock task `deploy_env → reset_env → prompt_agent → snapshot_env → load_artifact → crucible_verify` under `LocalRunner(workers=4)` with 8 concurrent runs: restored fingerprints are equal, the explorer lists the runs, and teardown leaves no processes | P1 | L |
| C7 | **A2A harness.** Any `AgentEnvAgent` (URL, or `deploy_agent`/`install_agent`) as a Crucible harness: mcp-config adds the World MCP URL with the actor's role header; agent-config sets model/`max_turns`; `LITELLM_BASE_URL` → Crucible's capture proxy; trajectory/v1 into the trace; snapshot/v1 for mid-episode restore | The echo agent plus one real coding agent produce `rollout.json` with complete token records in `rollout_mode="train"`. Without the proxy the rollout is masked `capture_incomplete`. Restoring from a mid-episode WorldRef together with the agent snapshot continues the episode | P1 | M |
| C8 | **Full trigger language.** Extend `EventWhen`/`EventAction` with state triggers (check pipelines), `where`, templated args, `barrier`, seeded stochastic `every`, `after`+`offset`, `until`, `verify`; delegate in wall mode, same semantics in Crucible's step-mode scheduler. | A shared conformance suite of about 20 specs gives identical firing logs (ids, virtual instants, actions) on the local backend (step clock) and the agentenv backend (wall clock). Seeded arrivals reproduce after snapshot/restore. A barrier makes the provoking call observe the action's effect | P1 | L |
| C9 | **agentenv env images as Apps.** App template `agentenv/mcp_env`: runs an agentenv-protocol image (`MCPServerEnv`'s `DockerImageArtifact`) inside the World sandbox. The state adapter goes over the v1 data plane: `data/get` to snapshot, `data/reset`+`data/add` to restore. Tools are cross-checked against the env card. | agentenv's `tst/data/slack_mcp` and `email_mcp` envs load as Apps. reset → mutate → snapshot → restore → equal fingerprints. Tool list = card. Passes `verify_env_core_protocol` | P1 | M |
| C10 | **Gateway inside the sandbox.** Option `gateway_placement: sandbox`: the gateway and App servers start next to the state, through `EnvironmentGatewayProvider`-style compose (VM providers) or the i6pn container mode (Modal). The orchestrator talks to one tunnelled URL per World. | 64 concurrent agentenv-backend Worlds on Modal from one orchestrator: host process count and open ports stay O(1) per World, and p50/p95 tool latency and exec RPC counts are reported against host placement. The red-team suite still passes (egress check includes the tunnel) | P1 | L |
| C11 | **Store and artifact mirroring.** `WorldRef`/`StateRef`/grade/trace become `crucible_ref` artifacts with `derived_from` both ways. Add an optional Crucible Store backend over agentenv's `ObjectStore` (S3/GCS) for multi-host runs. | Putting a WorldRef → `agent-env artifact` queries it, and its provenance resolves back. Two hosts with a shared S3 store resume each other's journal | P2 | M |
| C12 | **Long-horizon sandbox lifetime.** Lease renewal and a `max_lifetime` above the episode budget. Idle-sleep hooks for human waits (Sail `auto_sleep`; Modal `timeout` re-create from WorldRef). | An 8 h simulated-time episode with two 20-min human waits completes on Sail and on Modal (resume via WorldRef), with the same final grade as an uninterrupted run | P2 | M |
| C13 | **Surface area.** Add `agent_env.cli_plugins` (`agent-env crucible run/verify/export`, as research 05 §4 planned) and an `agent_env.bundles` export of Crucible tasks. | `agent-env plugin check` is clean. `agent-env run <crucible-bundle>` runs the toy task end to end | P2 | S |
| C14 | **Protocol conformance in CI.** Run agentenv's `verify_env_card`, `verify_env_core_protocol` and `verify_a2a_*` against the `crucible_world` gateway, the relay agent and the C7 harness adapter. | A CI job that is green, and fails on a deliberately broken card | P2 | S |
| C15 | **Export Apps as agentenv envs.** Serve each App through the env server SDK (card plus data plane) so Scale-side hubs deploy Crucible Apps as `MCPServerEnv`s. | The exported toy App is deployed by stock `deploy_env` and passes C14 | P2 | M |

## 4. What not to adopt

agentenv's verifiers and aggregation (U7, U9: clamping, infra failures scored 0), `LocalRunner`/`eval run`/bundles (single-process, no aggregation), `run_docker_container` for GPU work (U8), and the `local` provider for isolation (U6). Crucible's runtime, verifier and sandbox layers own these. agentenv's leverage is the gateway and its trigger language, the provider zoo (Modal, E2B, Sail), A2A agents and extensions, and the env-image catalogue; the P0/P1 work goes there.
