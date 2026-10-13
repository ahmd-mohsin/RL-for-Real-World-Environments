# M11 study, hunt H2: defects in `interop` (agentenv, verifiers, NeMo-Gym)

Scope: `C/interop/agentenv/` (`world_backend`, `sandbox`, `harness`, `mcp_env`, `a2a`, `agents`, `plugin/{steps,gpu_docker,artifacts}`), `C/interop/verifiers/` (`taskset`, `v1_import`), `C/interop/nemo_gym/` (`server`, `import_server`, `client`, `serve_resource`), and `C/interop/replay.py` and `C/interop/canonical.py` as these paths use them.

Spec: ARCHITECTURE "### M11 `interop`" and the §3 rows tagged M11; DESIGN "As built, M11 part 1" (with agentenv corrections C1 to C3), "C9: agentenv env images as Apps", "E: verifiers v1 import", "E: NeMo-Gym import", Appendix B.17; `docs/research/m11-m12/05-interop-diversity.md` (§2.2, §2.3, §3.2 to §3.4, §4).

Paths: `C` = `crucible/src/crucible/`. `AE` = `agentenv-framework/`. `NG` = `RL-Environments/06-frameworks-and-env-code/nvidia-nemo-gym/`. `VF` = `RL-Environments/06-frameworks-and-env-code/primeintellect-ai-verifiers/verifiers/`.

Method: a read-only study on HEAD `af402b4`, with `crucible/.venv` (Python 3.12.9) on macOS, and `crucible/.envs/verifiers` for the verifiers export. **[ran]** means reproduced with a probe. **[read]** means established from the code alone. The probes are in the session scratchpad at `m11hunt/h2/p1_nemo_server.py … p12_agentenv_fault.py` (plus `aetasks.py`, a task factory for the plugin steps). All probes are offline. They use scripted policies, in-process ASGI clients, loopback uvicorn servers, a local agentenv gateway subprocess and temp directories. No Docker, GPU, network or paid call was used. The only credential is a synthetic canary built at run time.

## 1. Requirement ledger

Status: **live** = built and behaves as specified. **partial** = built, but a specified part is missing or a hole was reproduced. **wrong** = contradicts the spec. **missing** = not built.

| # | Requirement (source) | Status | Evidence |
|---|---|---|---|
| L1 | Every infra fault is masked, never scored 0, on every runtime path (P16; B.17; research N3) | **wrong** | NeMo export scores an in-episode tool crash (V1). The agentenv backend drops every App fault label (V2). The gateway path never records faults (V3). NeMo import scores its server's own failures (V4) |
| L2 | agentenv: per-role tool access; "the agent sees exactly the role-filtered tools the native harness would" (`harness.py:6-7`; ARCH agentenv) | **wrong** | The agent can re-grant itself a revoked tool through the gateway's open control plane (V6). Catalog `hide`/distractors, RBAC, interceptors, quotas and schema checks live only in `World._call_tool`, which the gateway path bypasses (V3) |
| L3 | `crucible_world` step: World behind a real gateway, `DeployedGatewayEnv` with card and MCP URL (C2) | partial | Works (gate test). Nothing ever closes the World it opens (V16) |
| L4 | `crucible_verify`: single producer, unclamped, masked grade raises (C3; `steps.py:8-13`) | partial | Correct when the grade itself masks. It never sees in-episode faults, so a crash scores 0 (V3, p4) |
| L5 | `crucible_verify` steps from only this episode's gateway slice, `__actor` stripped (C3) | partial | The slice and the strip work. Calls the gateway refused are missing, and calls carry no role, so `steps.jsonl` differs from native (V9, p6) |
| L6 | Spoof-proof actor identity at the App (`world_backend.py:107`, `:210-213`) | partial | The HMAC check works. The token is static, readable at `GET /trajectory`, and an unsigned client acts as the first policy actor (V17) |
| L7 | `gpu_docker` opt-in on agentenv's local provider (C1) | live | Flags splice, type guard avoided (tests). A process-global toggle (V20) |
| L8 | `crucible_ref` artifact (ARCH "artifact mirroring") | partial | The type is registered. No producer in `src/` or `packs/` (grep), 0% coverage |
| L9 | verifiers export: masked → `TaskError`; only `reward` returned; components as `crucible/` metrics (DESIGN M11 part 1; research §2.3) | live | Gate test passes in `.envs/verifiers`. B.17 "verifiers sums keys" fix holds |
| L10 | verifiers export: dense credit (ARCH: "written into `MessageNode.advantages`") | partial | `steps.jsonl` goes to `trace.info["crucible_steps"]`. Nothing reads it (grep). The file is not cleared before grading, so an agent can plant it (V8) |
| L11 | NeMo-Gym resources server: mask at both NeMo-RL locations; `reward == sum(reward_components)` | live | Gate tests pass |
| L12 | NeMo-Gym server: session lifecycle (seed, verify, close; admission; reaper) | partial | A re-seed of a live id leaks the World and its slot (V10). Temp dirs never removed (V18) |
| L13 | NeMo-Gym server: works with NeMo-Gym's own agents; MCP metadata in the seed reply; `manifest.yaml` (ARCH; research §3.4) | partial / missing | The stock `simple_agent` seed body gets 500 (V11). No MCP metadata. No `manifest.yaml` writer (grep) |
| L14 | No oracle or grader state reachable by the policy through exported tools (P21; research N6) | **wrong** | A model function call named `verify` returns the grader's per-check verdict mid-episode (V5) |
| L15 | Remote client: admission and lost servers masked (`client.py:11-12`) | partial | Those two work (tests). An in-episode crash comes back scored (V1). A session lost at `/verify` raises. Messages and `Done` never reach the World (V12) |
| L16 | NeMo-Gym import: the server's reward is the score gate; `mask_sample` masks (DESIGN "E: NeMo-Gym import") | partial | `mask_sample` masks (tests). A 500, a dead server, a missing or null reward all score 0 (V4) |
| L17 | verifiers v1 import: reward = weighted rubric; "trace errors mask"; research §2.3: `trace.has_error` → masked | partial | `errors` holds every attempt's errors, so a recovered `ok=True` trace is masked. A `None` reward crashes. Only the last leaf's turns are kept (V13) |
| L18 | A2A harness: the agent's own error is scored, infra is masked (`harness.py:11-13`) | partial | Mapping works for typed results. Any exception in the agent becomes `infra_error`, so malformed model output is masked (V15). `agent_url` reuse breaks isolation (V7) |
| L19 | No secrets in records (M10 V12 rule) | **wrong** for A2A | `a2a_reply.txt` and `agent_trajectory.json` keep a provider-echoed key (V14) |
| L20 | `agentenv/mcp_env` App (C9) | live | Tests pass. Edge leaks (V20) |
| L21 | agentenv sandbox providers (Modal, E2B, Sail) | partial | 24% coverage. Teardown ordering can skip `sandbox.close` (V16) |
| L22 | Exit gate: identical `reward.json` and `steps.jsonl` through Harbor, agentenv, verifiers, NeMo-Gym, OpenEnv on task02 and a toy task | partial | The toy gate passes for the clean oracle transcript. task02 is not in the gate. With policy errors the agentenv dense channel differs (V9). With infra faults the mask differs by runtime (V1 to V4) |

## 2. Defect candidates

Severity: **P0** = silently wrong reward, a mask scored as 0 (or a 0 masked), an answer or oracle leaked to the agent, a cross-runtime mismatch, or an auth bypass. **P1** = a wrong classification, a crash or a leak in normal use. **P2** = an edge case.

### Masks that become 0

**V1 [P0][ran] NeMo-Gym export: an infra fault during the episode is scored, not masked.**
- `server.py:124-151` grades the World state. It never reads `world.failures`. `call_tool` (`:170-180`) returns only `{output, ok}`, so the client cannot see the fault either.
- p1 (a): the tickets App raises once (a synthetic lost DB connection).
  - native `run_episode`: `masked=True stop=crucible:tool_crashed`;
  - server: the World holds `['crucible:tool_crashed']`, yet `/verify` gives `reward=0.0 mask_sample=False`.
- p2 (a): `remote_episode` over the same server returns `reward=0.0 masked=False`. The M12 remote rollout trains on it.
- Fix:
  - in `/verify`, mask when any `world.failures` entry masks (`policy._masks`), with that failure kind;
  - return `fault` in the tool reply so clients can stop early.

**V2 [P0][ran] The agentenv backend drops every App fault label, even through `World.step`.**
- `AppMcpServer._call` raises `ValueError(res.content)` for every `ok=False` result (`world/packaging/app_server.py:68-70`). An App exception also becomes `isError`. `GatewayRouter.call` maps `isError` to `error="tool_error"` (`world_backend.py:80-81`). `_call_tool` then sets fault `tool` (`world.py:485`), which is scored.
- p12, the same fault on each backend with GOLD:

  | App fault | local | agentenv |
  |---|---|---|
  | raises | masked `crucible:tool_crashed` | **reward 0.0** |
  | `ok=False, error="transport"` (what sidecar and MCP-session Apps return) | masked `transport_unreachable` | **reward 0.0** |

- Every World on `backend="agentenv"` is affected, including native harness runs. This is also a cross-runtime mismatch.
- Fix:
  - carry `error` and `fault` across the MCP hop (in `structuredContent` or `_meta`), and rebuild them in `GatewayRouter.call`;
  - in the App server caller, call `world._infra(...)` for exceptions, or return a typed infra marker;
  - add the p12 table as a test for both backends.

**V3 [P0][ran] The agentenv gateway path bypasses the World choke point.**
- The A2A agent (`harness.py:136`), the plugin's `crucible_world` users (`deploy_agent`, `prompt_agent`) and the gate test all call `/step` or `/mcp` on the gateway. The gateway calls `AppMcpServer`, whose caller calls `app.call` directly (`world_backend.py:210-213`). `World._call_tool` and `World.step` never run.
- Lost on that path (p4, p6):
  - fault recording: a crash gives `World.failures: []`, and `crucible_verify` records `score 0.0, masked False`;
  - the World log: 0 `tool_call` events against 8 natively, so trace probes, `stream_trace`, answer-exposure checks and `trace.json` see nothing;
  - Crucible-scheduled events: `refund_audit` (on_tool) never fires, so the mail state differs (`3416…` vs `1325…`);
  - the step clock: `step_index` stays 0, so `month_end_freeze` (after 3h) never revokes `issue_refund`;
  - schema checks, catalog hides and distractors, RBAC, interceptors, tool quotas and output bounding [read, `world.py:425-503`];
  - p6: a call missing a required argument reached the App (HTTP 200). Natively it is refused as `invalid_arguments`.
- Fix:
  - route gateway calls through the World: the App-server caller should call `world._call_tool(actor, ToolCall(...))` and log the action, so the choke point, the clock and the scheduler run;
  - or serve agents a Crucible MCP front that calls `World.step`, and keep the gateway internal;
  - `A2AHarness` and `crucible_verify` must mask on `world.failures`.

**V4 [P0][ran] NeMo-Gym import: the imported server's own failures score 0.**
- `nemo_verify` (`import_server.py:243-253`) lets an `httpx` error escape. `_run_check` treats any non-`InfraError` exception as the candidate's failure, scored 0 (`verify/runtime/plan.py:313-314`). A missing or null reward becomes `float(doc.get("reward", 0.0))`.
- p3, a fake resources server on loopback:

  | `/verify` behaviour | result |
  |---|---|
  | `{"reward": 1.0}` | 1.0 |
  | HTTP 500 | **0.0, infra ok** |
  | no `reward` key | **0.0, infra ok** |
  | `{"reward": null}` | **0.0, infra ok** |
  | server down after the episode | **0.0, infra ok** |
  | `{"reward": 7.0}` | 1.0 (clamped silently) |

- NeMo-Gym wraps `/verify` in `judge_failsafe`, so judge servers (gap G10) fail exactly this way.
- Fix:
  - raise `InfraError` on transport errors, non-2xx and a missing or non-finite reward;
  - flag a reward outside the server's declared range instead of clamping it.

### Oracle and access control

**V5 [P0][ran] NeMo-Gym export: the grader is a tool. A model call named `verify` returns the verdict mid-episode.**
- NeMo-Gym's `simple_agent` posts every model function call to `/{name}` with the session cookie. It does not check the name against the advertised tools (`NG/responses_api_agents/simple_agent/app.py:203-223`). The server's `/verify`, `/close_session` and `/seed_session` share that namespace (`server.py:99-168`).
- p11 (b): after 3 of 8 GOLD calls, a POST to `/verify` with `{}` (what the agent sends) returns status 200, `reward 0.0`, `check/t1 1.0, check/t2 0.0, check/t3 0.0`, both gate bits and 4 dense rows. The policy can search against the grader. `close_session` as a tool ends its own episode.
- Fix:
  - serve tools under a separate prefix (for example `/tools/<name>`, or the MCP endpoint research §3.4 prefers) and refuse control routes for tool traffic;
  - make `/verify` one-shot per session, after which tools refuse; or require a server-held verify token the agent never sees.

**V6 [P0][ran] agentenv: the gateway control plane is open to the agent.**
- The gateway serves `/tools/enable`, `/tools/disable`, `/clock/set-time`, `/triggers/*` and `/trajectory` on the same port as `/mcp`, with no auth (`AE/src/agent_env/env/gateway/gateway.py:161-291`). The A2A agent gets that base URL (`harness.py:136`). A host-side shell App can reach loopback too.
- p4 (c): with `issue_refund` revoked for `agent`, the call gives 403. The agent POSTs `/tools/enable {"role":"agent","tools":["tickets__issue_refund"]}` and gets 200. The same call then gives 200.
- The role header is self-asserted too. Any client can claim another actor's role.
- The `isolation.policy_user` firewall only covers sandboxed Worlds (`world_backend.py:202-207`).
- Fix:
  - put a reverse proxy in front of the gateway that exposes only `/mcp` (and `/step` `call_tool`/`list_tools`), and pins the role per agent with a per-attempt bearer token;
  - keep the control port private to the World process.

**V7 [P0][ran] A2A harness with `agent_url`: attempts share the first attempt's World.**
- The second `mcp-config` for the name `world` gets 409 "already registered" (`AE/packages/agentenv-protocol/.../framework.py:704-716`). The harness accepts 409 (`harness.py:137`), so the agent keeps the first URL.
- p5:
  - two concurrent attempts: both agents used the same MCP URL. Both attempts scored **0.0**, unmasked: one World got GOLD twice (over-refund), the other was untouched;
  - a later sequential attempt: World 1 is closed, so the agent fails as `framework.unhandled_exception` (masked `infra`). Every attempt after the first is masked.
- Fix:
  - register a unique name per attempt (`world-<attempt tag>`) and remove it at the end (`DELETE`/`ext/mcp-remove`), or treat 409 as a harness error;
  - pass the server name in the prompt metadata, so the agent uses only this attempt's server;
  - refuse concurrent attempts on one `agent_url` unless the agent advertises multi-session support.

### Dense channel and cross-runtime identity

**V8 [P1][ran] verifiers export: an agent-planted `steps.jsonl` is attached as the dense channel.**
- verifiers' `stage_tests` removes only `reward.json` and `reward.txt` (`VF/v1/tasksets/harbor/taskset.py:345-347`). Crucible's in-container grader writes reward files only (`verify/runtime/run.py:113`). `CrucibleTask.run_verifier` attaches any `/logs/verifier/steps.jsonl` (`taskset.py:46-50`).
- p8 (in `.envs/verifiers`, shared-verifier layout): an agent writes three rows with `delta_phi 1.0`. The grade is `reward 0.0`, and `trace.info["crucible_steps"]` holds the planted rows.
- No consumer reads `crucible_steps` yet (grep). It becomes P0 when the M12 prime-rl bridge maps rows to nodes.
- Fix: write `steps.jsonl` in the grader, or delete it before `test.sh`. Accept rows only with a grader digest written into `crucible_reward.json`.

**V9 [P1][ran] agentenv `steps.jsonl` differs from native when the policy makes errors.**
- The gateway refuses some calls before logging them (unknown or disabled tool: 403, `gateway.py:775-782`). `from_gateway_jsonl` replays only logged calls. Tool-call events carry no role, so calls by other actors through the World router replay as the agent's (`replay.py:159-170`).
- p6, two bad calls then GOLD:
  - native: 11 rows;
  - agentenv: 10 rows, and the `call_id` ordinals shift after the first bad call;
  - `same dense channel: False`. The rewards agree (1.0).
- Fix:
  - log the agent's actions on the Crucible side (see V3) and replay from the World trace;
  - or record role and outcome per gateway event, and filter to the agent's role.

**V12 [P1][ran] NeMo remote client: messages and `Done` never reach the World; errors at `/verify` raise.**
- `remote_episode` drops `Message` (`client.py:86-87`) and never sends `Done` (`:72-74`).
- p2 (c): after `Message(to="supervisor")` and `Done(answer="42")`, the server World has no agent events. Responders never run. Tasks graded on messages, approvals or `Done.answer` score differently remotely.
- p2 (b): a session reaped before `/verify` raises `HTTPStatusError 404`. Non-404 tool errors raise too (`:83`). The M12 runtime catches it as a "bug" failure, not `env_session_lost`.
- Fix:
  - add `/message` and `/done` routes that call `World.step`;
  - map every non-2xx and transport error at `/verify` to a masked `RemoteEpisode` with a kind.

**V13 [P1][ran] verifiers v1 import: mask rule, `None` rewards, branches.**
- `masked = bool(errors)` (`v1_import.py:77`). `Trace.errors` holds "every error captured across attempts" (`VF/v1/trace.py:425`). Research §2.3 says `trace.has_error` (= `not ok`). p7 (b): an `ok=True` trace with a recovered 429 is masked.
- `rewards` values may be `None` ("scoring didn't run", `VF/v1/trace.py:407`). p7 (a): `AttributeError`, not a mask. The M12 runtime catches it as a bug.
- `_chain` follows only `nodes[-1]` (`v1_import.py:34-46`). verifiers traces have several leaves: subagents, parallel agents, compaction attempts (`VF/v1/trace.py:515-545`). p7 (c): one of 3 sampled nodes is kept, with the full reward.
- Policy-caused errors in `errors` (an overlong prompt, an empty model reply) mask the sample, which the policy can reach [read].
- Fix:
  - mask on `not trace["ok"]`, skip `None` rewards as verifiers does (`trace.py:473-475`);
  - build one record per trainable branch (verifiers' own `branches`);
  - map verifiers error types to blame, and score policy ones.

### Sessions, leaks, failures

**V10 [P1][ran] NeMo-Gym export: a re-seed of a live session id leaks its World and slot.**
- `seed_session` overwrites `sessions[sid]` (`server.py:116`) without closing the old World or releasing its slot.
- p1 (b), capacity 2: after two seeds of `dup`, 1 session and 2 admitted. After `close_session("dup")`, the first World is still open, `x1` seeds, and `x2` gets **503 queue_timeout**. Leaked slots never come back, so the server ends up refusing every episode.
- p1 (c): session ids are client-chosen and unauthenticated. A second client drove `x1`'s World (200). NeMo's new seed contract does take a client id (`NG/nemo_gym/base_resources_server.py:189`), so the fix is to refuse reuse, not to change the contract.
- Fix:
  - refuse a live id (409), or close it first and release its slot;
  - bind the session to a server-minted cookie at seed (NeMo's own signed-session pattern), and require it on tool calls.

**V11 [P1][ran] NeMo-Gym export: NeMo-Gym's own agent cannot seed.**
- `simple_agent.run` seeds with its run body (`NG/responses_api_agents/simple_agent/app.py:309-314`): no `resources_session_id`, no `task_id`.
- p11 (a): our server answers **500** (`server.py:102` `body["resources_session_id"]`). The gate validates response shapes only.
- The seed reply has no `resources_tools` MCP endpoint that research §3.4 prefers. No `manifest.yaml` is written (ARCH; grep).
- Malformed tool bodies (`not json`, a JSON list) give 500 (p1 d), not a scored tool error.
- Fix:
  - accept the legacy body: mint the session id, take the task from `responses_create_params.metadata` or a single-task server;
  - serve MCP;
  - turn bad bodies into a policy-fault tool reply.

**V14 [P1][ran] A2A harness: secrets reach the attempt directory.**
- `harness.py:146`, `:150`, `:155` and `:166` write `a2a_reply.txt`, `agent_trajectory.json`, `calls.json` and `trace.json` raw. The native harness scrubs these since M10 V12 (`runtime/harness/native.py:70-86`).
- p9: a synthetic `sk-proj-…` canary in the agent's reply and trajectory (a provider 401 that echoes the key) is in `a2a_reply.txt` and `agent_trajectory.json`.
- Fix: write all four through `scrub_json` / `scrub_secrets`.

**V15 [P1][ran] A2A path: policy-caused failures become infra masks.**
- agentenv's framework turns any exception from `run()` into `infra_error` (`AE/.../framework.py:1167-1180`). The harness masks `infra_error` (`harness.py:160-162`).
- p5: an agent that hits a malformed model reply (`KeyError`) is masked `agent_run_error/infra`.
- `LLMToolAgent` (`agents.py`):
  - maps any non-200 from the model, including a 400 caused by its own context, to `infra_error` (`:88-89`);
  - indexes `c["function"]["name"]` unguarded (`:101-106`), so malformed output becomes a framework exception and a mask;
  - replaces unparseable arguments with `{}` and runs the call (`:102-105`). Natively that is a format error.
- Fix:
  - catch model-output errors inside `LLMToolAgent` and return `failed(..., error_type="agent_error")`;
  - split 4xx caused by request content from auth and server errors (as M10 V7 proposes).

**V16 [P1][read] agentenv plugin and backend teardown.**
- `crucible_world` puts the World in the process-global `WORLDS` (`steps.py:60`). Nothing pops or closes it (grep: only the test does). Each World holds a gateway subprocess, one MCP server per App, temp dirs and possibly a sandbox. A training or eval loop leaks one set per episode.
- `AgentEnvBackend.teardown` (`world_backend.py:518-539`) has no `try` around `relay.stop()` or the App servers' `stop()`. One failure skips `sandbox.close()`, which can be a paid Modal GPU sandbox.
- The gateway gets SIGTERM only, with no SIGKILL after the 5 s wait (`:523-527`).
- Fix:
  - add a `crucible_close` step (or close in `crucible_verify` with an opt-out), and a `WORLDS` reaper;
  - run each teardown stage in its own `try`, and close the sandbox last in a `finally`.

**V17 [P1][ran] agentenv actor identity is a static bearer token, and unsigned clients act as the first policy actor.**
- `sign_actor` is `HMAC(actor)` with no nonce or call binding (`world_backend.py:111-112`). The World router puts it inside the tool arguments (`:70`), which the gateway logs.
- p4 (d): `GET /trajectory` (unauthenticated) shows `agent:0ae1…`. In a World where simulated actors call tools through the router, their tokens are readable too, and any client can replay them to act as that actor at the App.
- An external agent on role `B` that sends no token is `default_actor` at the App (`:212`), so per-actor App state (inbox, ownership) is misattributed in multi-actor Worlds.
- Fix:
  - sign `(actor, call nonce)` and verify one-shot;
  - strip `__actor` before the gateway logs (carry it in a header the App server reads);
  - bind role to token for external agents.

### Edge cases

**V18 [P2][ran/read] NeMo-Gym export, minor.**
- `nemo-sess-*` (`server.py:110`) and replay's `replay-*` (`replay.py:81`) temp dirs are never removed. Each holds a World store.
- `episode_key` defaults to `e0` (`server.py:138`), and `remote_episode` never sends one. All sessions of a task get the same `call_id`s, so rows of a GRPO group collide on join.
- The reaper clock is set only at request entry (`server.py:92`). A grade longer than `idle_timeout_s` is reaped mid-grade. p10: a grade over a closed World masks `check_infra`, so a correct episode is lost (not scored 0).
- `/verify` runs without the step lock, so a concurrent tool call can change the state being graded [read].

**V19 [P2][read] NeMo-Gym import, minor.**
- `NemoGymApp.stop` and `_seed` never call `/close_session` (`import_server.py:97-115`). Servers that keep per-session state (workplace_assistant) grow for the whole run.
- The journal keeps `output[:65536]` (`:138`), and `trace_to_response` sends that cut text to `/verify`. A server that checks tool outputs sees a different response than natively.
- `NemoGymApp.call` raises on any non-2xx (`:131`). A tool that answers 4xx for the policy's input masks the episode as `tool_crashed`.

**V20 [P2][read] agentenv utilities.**
- `local_gpu` toggles a class attribute (`plugin/gpu_docker.py:34-43`). Overlapping contexts restore the wrong value, and non-GPU tasks running at the same time get GPU flags.
- `AgentEnvMcpApp`: `-e k=v` puts env values on the `docker run` command line (`mcp_env.py:125-126`). If `super().stop()` raises, `docker rm -f` is skipped (`:163-165`).
- `AgentEnvSandbox.providers()` builds every provider in the chain before trying the first (`sandbox.py:145`). A missing credential for a later fallback aborts the whole chain. `close()` keeps `_sandbox` set when `terminate` raises.

**V21 [P2][read] verifiers v1 rollouts.**
- `vf_rollout` ignores `seed` (`v1_import.py:169`).
- Versions are matched to sampled nodes by call order (`:70`). A retried `generate` request appends an extra version, and every later turn gets the wrong stamp.

## 3. Coverage

Run from `crucible/` with `COVERAGE_FILE` in the scratchpad (the repo's `.coverage` is untouched):

```
pytest -q -p no:cacheprovider --cov=crucible.interop.{agentenv,verifiers,nemo_gym,replay,canonical} \
  tests/unit/{test_agentenv_backend,test_agentenv_mcp_env,test_agentenv_plugin_fixes,test_agentenv_sandbox,
  test_a2a_harness,test_verifiers_v1_import,test_nemo_gym_import,test_interop_gate}.py
29 passed in 120.9 s; TOTAL 1810 stmts, 72%
+ tests/unit/{test_remote_env,test_llm_tool_agent}.py: 4 passed; client.py 90%, agents.py 90%
```

| Module | Cover | Untested risky paths |
|---|---|---|
| `verifiers/taskset.py` | 0% here | runs only in `.envs/verifiers` via subprocess (the gate test passes there). No test plants files in `/logs/verifier` (V8) |
| `nemo_gym/serve_resource.py` | 0% | the NeMo-side launcher (exercised by `test_nemo_gym_import` in a subprocess) |
| `agentenv/sandbox.py` | 24% | the provider chain, exec, teardown (V16, V20) |
| `agentenv/a2a.py` | 69% | `A2AClient`, `A2ATransport`, `PeerFailed` |
| `nemo_gym/server.py` | 79% | the reaper, admission timeout, `_close` failure, `aggregate_metrics` |
| `agentenv/mcp_env.py` | 88% | start failures after `docker run` |
| `agentenv/world_backend.py` | 90% | trigger translation branches, `restore_state` trigger restore |
| `replay.py` 91%, `nemo_gym/import_server.py` 93%, `plugin/steps.py` 94%, `harness.py` 95%, `v1_import.py` 96% | | |

High coverage hides the defects above. No test injects an infra fault on a foreign path (V1 to V4). No test runs the agentenv backend with a failing App (V2). No test calls a control route as a tool (V5) or the gateway control plane as the agent (V6). No test reuses an `agent_url` (V7). No test sends a transcript with errors through agentenv (V9). p1, p3, p11 and p12 are the cheapest regression seeds; each runs in under 30 s offline.

## 4. Offline gates

- **Toy interop gate** (`tests/unit/test_interop_gate.py`, 7 tests, MiMo included) passes, including the verifiers, NeMo and OpenEnv subprocess paths. It checks only the clean oracle transcript with no faults. It does not compare the agentenv final state fingerprint, which differs (V3: mail state).
- **NeMo-Gym import** (`test_nemo_gym_import.py`, real workplace_assistant, MRCR and RULER servers) passes.
- **verifiers v1 import** (`test_verifiers_v1_import.py`, real `glossary` env) passes.
- **task02** cross-runtime identity (ARCH exit gate) is not in any test and was not run (it needs Docker and the lab host).
