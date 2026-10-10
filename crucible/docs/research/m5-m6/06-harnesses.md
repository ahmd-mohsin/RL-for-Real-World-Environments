# 06 — Agent harnesses for Crucible Worlds (C3 branch values, calibration, baselines, RL)

Scope: which harness(es) Crucible should implement natively as a `Policy`, which to run as Harbor installed agents behind our capture proxy, and exactly how each plugs into `World` / `WorldRef` / traces. Sources read (read-only): mini-swe-agent 2.4.6, our task03/04 frontier runs, Harbor installed agents and terminus-2, verifiers v1 harnesses/runtimes/interception, karpathy-autoresearch, FineEnvs multi-harness RL (and the OpenEnv capture code it depends on), and Crucible's current policy layer.

Layout:
- Brief structure: §1 TL;DR, §2 per-source mechanisms, §6 mechanism table, §7 open questions and risks.
- Task deliverables: §3 = (a) comparison table, §4 = (b) recommendation and integration design, §5 = (c) robustness lessons.

Path prefixes used in citations:
- `MS/` = `RL-Environments/06-frameworks-and-env-code/swe-agent-mini-swe-agent/src/minisweagent/`
- `MT/` = `migration_tasks/`
- `H/` = `RL-Environments/06-frameworks-and-env-code/harbor-framework-harbor/src/harbor/`
- `VF/` = `RL-Environments/06-frameworks-and-env-code/primeintellect-ai-verifiers/verifiers/`
- `AR/` = `RL-Environments/06-frameworks-and-env-code/karpathy-autoresearch/`
- `MH/` = `RL-Environments/06-frameworks-and-env-code/adithya-s-k-fineenvs/05-multi-harness-rl/`; `SP/` = `.../hf-space-fineenvs-multi-harness-rl/app/src/content/`; `OE/` = `.../meta-pytorch-openenv/`
- `C/` = `crucible/src/crucible/`

---

## 1. TL;DR — the takeaways, each with a "Crucible should…"

1. **There are two integration seams, and they serve different purposes.**
   - **Seam W, the native loop outside the sandbox.** The harness loop runs in Crucible's process, and every action is a `World.step` through `World._call_tool` (`C/world/world.py:295-339`). That gives a step-level trace, snapshots at any step boundary and replay of the conversation prefix.
   - **Seam S, installed agents inside the sandbox.** Claude Code, Codex and OpenCode run in the container with their own native tools. The World sees only the final state plus the capture-proxy log.
   - **Crucible should use seam W for C3, PRM labels, cheap baselines and RL, and seam S for frontier calibration and ceilings.** Seam S step boundaries exist only as proxy request boundaries (§4.4).

2. **Crucible should build `BashLoopPolicy`, a port of the mini-swe-agent v2 loop, as its primary native harness.**
   - Shape: one `bash` tool, every command in a fresh subshell, an explicit submit sentinel, Jinja format-error and observation templates, step/cost/wall limits, and a cap on consecutive format errors (`MS/agents/default.py:19-35,88-157`; `MS/models/utils/actions_toolcall.py:11-76`).
   - It is the harness behind all 7 of our task04 frontier runs. Across 680 model calls those runs had 0 format errors (§2.2).
   - Its stateless subshell is what makes a `WorldRef` a complete description of the state: no tmux pane, cwd or env vars hidden outside the snapshot.

3. **Crucible should fix two protocol bugs in `LLMToolPolicy` before using it for anything long-horizon.**
   - **"No tool call means done."** `C/actors/policy.py:135-138` turns any text-only reply into `Done`. That includes a reply cut off at `finish_reason == "length"` and a mid-task status message.
   - **Deterministic sampling.** The defaults are `temperature=0.0` (`policy.py:67`) and `seed=0` (`C/actors/llm.py:149`), so K branches are near-identical.
   - **Fix:** make submission explicit, the way mini-swe-agent (`MS/environments/local.py:45-56`) and terminus-2 (double confirm, `H/agents/terminus_2/terminus_2.py:1529-1545,1655-1662`) do. Make `finish_reason=="length"` a format error with the "you were cut off" template (`MS/config/mini.yaml:129-131`).

4. **Crucible should define a C3 branch as the pair `(WorldRef_t, PolicyPrefix_t)`.**
   - `WorldRef` has no policy context (`C/core/schema/state.py:90-104`).
   - For a native policy the prefix is its `messages` list at step t (a deep copy). For an installed agent it is an ATIF prefix that Harbor seeds into the agent's native session: `load_trajectory` leads to `claude --resume <id>` (`H/agents/installed/base.py:1192-1213`; `claude_code.py:1929-1939`) or `codex exec resume --last` (`codex.py:1585-1610`).
   - Mark the prefix steps `is_copied_context=True` (`H/models/trajectories/step.py:64-73`) so they are never trained on.

5. **Crucible should branch only at quiescent step boundaries.**
   - Mid-episode snapshots deliberately skip quiescing (`C/actors/policy.py:203`).
   - Our task04 prompt tells the agent to "build in the background and poll a log file" (`MT/task04_pcl_gpu_cuda12/mswea_sandbox.yaml:14-15`). A background build is invisible to `WorldRef`, so a branch restored from it silently diverges.
   - Under `fs_git`, files that `.gitignore` hides (build outputs) are "derived state, not state" (`C/world/state/fs.py:1-5`). Every branch must rebuild them, which costs minutes on PCL.
   - **Rule:** a step is branchable iff the guard finds no live policy process at the snapshot. Record `branchable: false` otherwise.

6. **Crucible should give every LLM-driven policy context compaction and prompt caching.**
   - Without compaction, cost is quadratic in the number of steps. Our longest task04 run (GPT 6 Astra, 134 calls) reached 328,854 prompt tokens on one call. Its last third cost $24.28 against $5.30 for the first third, even with 99.99% cache hits. The total was $39.75.
   - An open model with a 128k context would hit `ContextOverflow`, which `LLMToolPolicy` scores as a truncated failure (`policy.py:121-123`).
   - **Port verifiers' compaction**, which is the simpler of the two: trigger at context window − 16,384 tokens, one summary call with `tool_choice="none"`, ≤3 attempts, then a framed restart (`VF/v1/harnesses/utils/compaction.py:11-38,155-256`).
   - Keep terminus-2's three-step summary → questions → answers (free-token threshold 8000, `H/agents/terminus_2/terminus_2.py:278,865-960`) as the alternative if summaries lose too much.
   - Mark the replaced span as copied context.

7. **Crucible should separate an episode's cost limit (a policy outcome) from an account or credit stop (infra).**
   - mini-swe-agent's `LimitsExceeded` is the agent's own stop (`MS/agents/default.py:132-139`).
   - Two of our 7 task04 runs died on infrastructure:
     - an HTTP 402 "maximum cost exceeds your available credits" (`gpt-6-astra-v2-run1`);
     - "No valid cost information… Usage {}" (`gpt-6-astra-v3-aborted`).
   - **Mapping:**
     - episode cap → `stop_condition="cost_limit"`, `truncated=True`, scored;
     - 402, blank usage or global cap → `crucible:credits_exhausted` / `crucible:usage_blank`, masked (`C/core/schema/run.py:30-49`).

8. **For installed agents, Crucible should enforce budgets in the capture proxy and not trust the harness.**
   - Current state of the proxy (`C/connectors/capture_proxy.py:87-141`):
     - no per-session USD cap and no request cap;
     - only `/v1/chat/completions` and `/v1/messages`;
     - no Responses API, no `count_tokens`;
     - no token-id capture.
   - Add per-session `budget_usd`, `max_requests` and `deadline`. When one is exceeded, answer with a dialect-correct error and record `stop=cost_limit`.
   - Belt and braces: Claude Code's own `--max-budget-usd` and `--max-turns` (`H/agents/installed/claude_code.py:54-67`).

9. **Crucible should open a proxy-only network lane for installed agents.**
   - Our Harbor backend starts containers with `--network none` (`C/interop/harbor/world_backend.py:105-107,192-197`), which also blocks the proxy.
   - Harbor supports an agent-phase allowlist. `extra_allowed_hosts` is "merged into the effective agent phase allowlist during agent.run() only" (`H/models/trial/config.py:125-131`; `H/trial/network_policy.py:23-42`), enforced on Docker by an egress-control sidecar (`H/environments/docker/docker.py:210-217`).
   - Allowlist only the proxy host:port. Give the agent the **session key only**: the proxy's key is the capability (`capture_proxy.py:5-8,62-66`).

10. **Crucible should bake pinned harness binaries into the World images.** No-network sandboxes cannot `npm install` or `curl bootstrap.sh`.
    - Harbor's Claude Code install is skipped when the installed version already matches (`claude_code.py:466-471`). The same check-then-install pattern exists for mini-swe-agent (`mini_swe_agent.py:625-670`).
    - Pin the versions FineEnvs validated: opencode 1.18.31, claude-code 2.1.270, codex 0.154.0, mini-swe-agent 2.4.6 (`MH/envs/harbor/smoldataenv_harbor/server.py:17-20`).

11. **Crucible should treat each harness as part of the policy: record it, calibrate per harness, and never mix harnesses inside one GRPO group.**
    - The same base model solves 62% of FineEnvs tasks under mini-swe-agent and 33% under Claude Code (`SP/chapters/training.mdx:54-56`).
    - FineEnvs assigns one harness per task so "the advantage compared actions rather than harnesses" (`MH/envs/harbor/smoldataenv_harbor/environment.py:74`; `SP/chapters/openenv.mdx:383-384`).
    - Put `harness={name, version, config_hash}` in `RunRecord` and in the trace source.

12. **For RL training data, Crucible should capture engine token ids at the proxy and never retokenize.**
    - Rebuild the turns with OpenEnv's longest-token-prefix graph (`OE/src/openenv/core/harness/capture/graph.py:1-35,299-325`). Claude Code "rewrites its own history", so one rollout becomes about 8 training rows (`SP/chapters/training.mdx:195-197`).
    - Set `top_p=1, top_k=-1` at the tokens tier (`OE/.../capture/upstream.py:139-148`).
    - Verifiers' interception does the same for its v1 harnesses (§2.4).

13. **Crucible should cross-check every installed-agent rollout's capture log against the harness's own ATIF.**
    - Turn counts and per-call completion tokens must match, or the sample is masked. This check caught a harness that sent an empty tools array and got a 400 (`SP/chapters/openenv.mdx:290-303`).
    - We already have both converters: `CaptureLogConnector` (`capture_proxy.py:227-…`) and Harbor's `convert_trajectory` (`C/interop/harbor/atif_import.py:104-117`).

14. **Crucible should keep `World._call_tool` the single choke point, including for MCP-driven harnesses.**
    - The agentenv backend's MCP endpoint calls `_app.call(...)` directly (`C/interop/agentenv/world_backend.py:186-192`). An external harness on `mcp_url` therefore bypasses the trace log, step index, bounding and scheduler.
    - Add a World MCP facade whose handler is `await world.step(actor, ToolCall(...))`.

15. **Crucible should keep the shell's timeout strictly below `tool_timeout_s`.**
    - `WorkspaceApp.tool_shell` allows up to `max_timeout_s` = 3600 s (`C/world/apps/workspace.py:138`). The World's own deadline is also 3600 s (`C/core/schema/world.py:171`).
    - So a policy's 3600 s build is killed by the World first and recorded as `transport_timeout`, which is infra and masked (`world.py:322-324`), when it should be a policy-scored `[TIMEOUT]`.
    - Clamp to `tool_timeout_s - 60`.

16. **Crucible should copy autoresearch's long-horizon hygiene into its coding-World prompts.**
    - Redirect long output to a log and `grep`/`tail` it (`AR/program.md:99-101`).
    - Commit before an experiment, `git reset` on a regression (`:102-104`).
    - Keep a results log outside both context and git (`:66-88`; `.gitignore:23`).
    - Use a fixed time budget per experiment (`AR/prepare.py:31`).
    - These are the habits that keep a 100-step GPU task within context. They also give the M5 milestone miner clean, commit-shaped step boundaries.

17. **Crucible should use FineEnvs' reward lessons when it designs harness-facing rewards and limits.**
    - A "+0.2 for submitting anything" term collapsed held-out accuracy from 0.740 to 0.178 (`SP/chapters/openenv.mdx:285-287`).
    - A correctness-only reward let tool calls creep from 13 to 41 (`SP/chapters/training.mdx:67-69`).
    - A training/eval output-token cap mismatch raised truncated cells from 9 to 556 of 1,000 (`training.mdx:350-354`). Use the same `max_tokens` in calibration and in RL.

---

## 2. Per-source mechanisms

### 2.1 mini-swe-agent 2.4.6 (`MS/`, `__version__="2.4.6"` at `MS/__init__.py:11`)

**Protocols.** `Model.query/format_message/format_observation_messages/serialize` and `Environment.execute(action, cwd)/get_template_vars/serialize` (`MS/__init__.py:42-71`). Duck-typed, so a Crucible World can stand in for `Environment`.

**Agent loop** (`MS/agents/default.py`):
- **`AgentConfig` defaults** (`:19-35`): `step_limit=0` (unlimited), `cost_limit=3.0`, `wall_time_limit_seconds=0`, `max_consecutive_format_errors=3`, `output_path`.
- **`run()`** (`:88-124`): renders the system message and the instance message (Jinja, `StrictUndefined`, `:66-67`), then calls `step()` until the last message has `role == "exit"`.
  - `FormatError`: the call was already billed, so it charges `e.messages[0].extra.cost` (`:101-102`) and increments the streak. At the cap it appends an `exit` message with `RepeatedFormatError` (`:103-112`); otherwise it appends the error message as a user turn (`:113-114`).
  - Any clean step resets the streak (`:99`).
  - `InterruptAgentFlow`, whose subclasses are `Submitted`, `LimitsExceeded`, `TimeExceeded` and `UserInterruption` (`MS/exceptions.py:1-26`), appends its messages (`:115-116`).
  - An uncaught exception appends an `exit` message carrying `exit_status=type(e).__name__` and the traceback, then re-raises (`:74-86,117-119`).
  - **The trajectory is saved after every step** in `finally` (`:120-121`). This is why our aborted runs still have usable trajectories.
- **`query()`** (`:130-152`) checks limits **before** calling the model: `0 < step_limit <= n_calls or 0 < cost_limit <= cost` raises `LimitsExceeded`, and the wall clock raises `TimeExceeded`. The check is a pre-check, so the last call can overshoot `cost_limit`.
- **`execute_actions()`** (`:154-157`) runs every action in the message, in order, then formats one observation per action.
- **Trajectory**: `{"info": {model_stats: {instance_cost, api_calls}, config, mini_version, exit_status, submission}, "messages": [...], "trajectory_format": "mini-swe-agent-1.1"}` (`:159-180`). Each message keeps `extra` (raw response, cost, timestamp, actions, raw_output, returncode).

**Submission protocol.** `_check_finished` (`MS/environments/local.py:45-56`, duplicated in `docker.py:140-151` and `swerex_docker.py`) raises `Submitted` iff:
- the **first line of the output** is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, and
- `returncode == 0`.

The rest of the output becomes the `submission`. The SWE-bench config makes the patch part of it: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt` (`MS/config/benchmarks/swebench.yaml:99-103`). If the command fails, it does not submit (`:103`).

**Environments** (`MS/environments/__init__.py:8-16`): docker, singularity, local, swerex_docker, swerex_modal, bubblewrap, contree. An environment can be given as a full import path (`:19-27`); our `mswea.ext.SSHEnvironment` is loaded this way.
- **local** (`local.py:72-92`): `Popen(..., start_new_session=True)`. On timeout it calls `os.killpg(pid, SIGKILL)`, so no children are orphaned, and re-raises `TimeoutExpired` *with the partial output*.
  - The exception path (`:31-41`) returns `returncode=-1`, the partial output and `exception_info="An error occurred while executing the command: …"`. **A timeout is an observation, not a crash.**
- **docker** (`docker.py:74-99`): `docker run -d … sleep 2h`, then `docker exec -w cwd [-e …] bash -lc <cmd>` per action (`:101-138`). `stderr` is merged into `stdout` (`:123`). Cleanup runs `docker stop` then `rm -f` in the background (`:153-157`).

**Models and parsing.**
- **Tool-call mode (the v2 default).** One tool, `BASH_TOOL = {name: "bash", parameters: {command: string}}` (`MS/models/utils/actions_toolcall.py:11-27`). `parse_toolcall_actions` (`:30-76`) raises `FormatError` when:
  - there are no tool calls ("Every response MUST include at least one tool call", `:40-52`);
  - the arguments are not valid JSON;
  - the tool name is not `bash`;
  - `command` is missing (`:54-74`).

  Observations whose action never ran are padded with `exception_info: "action was not executed"` (`:88-89`).
- **LiteLLM model** (`MS/models/litellm_model.py`):
  - Retries with tenacity: 10 attempts, exponential wait 4–60 s (`MS/models/utils/retry.py:19-25`).
  - It **does not retry** auth, not-found, permission, unsupported-params or `ContextWindowExceededError` (`litellm_model.py:50-57`).
  - On a `FormatError` it still persists the raw response and the cost (`:87-98`).
  - It refuses a response with cost ≤ 0 unless `cost_tracking: ignore_errors` (`:108-126`). That is exactly what aborted `gpt-6-astra-v3`.
  - It passes `finish_reason` to the format-error template (`:128-135`).
- **Cache control.** Anthropic-named models automatically get `set_cache_control="default_end"` (`MS/models/__init__.py:55-60`; `cache_control.py:49-60`).
- **Global caps.** `MSWEA_GLOBAL_COST_LIMIT` and `MSWEA_GLOBAL_CALL_LIMIT`, process-wide and thread-safe (`MS/models/__init__.py:13-42`).

**Templates** (`MS/config/mini.yaml`):
- The **observation** is the full output under 10,000 chars. Above that it becomes `output_head` (5,000) + `output_tail` (5,000) + `elided_chars` + `"warning": "Output too long."` (`:112-128`).
- The **format error template** branches on truncation (`:129-149`):
  ```jinja
  {% if finish_reason is defined and (finish_reason == "length" or (finish_reason == "tool_calls" and not has_tool_calls)) -%}
  Your previous response reached the output token limit ... Respond more concisely and finish with exactly one bash tool call.
  ```
- The **instance template** states: "Directory or environment variable changes are not persistent. Every action is executed in a new subshell" (`:39-40`). It also gives the submit rule "Do not combine it with any other command" (`:18-19`).
- **SWE-bench config**: `step_limit: 250`, `cost_limit: 3`, env `timeout: 60`, `interpreter: ["bash","-c"]`, plus `BASH_ENV=/root/.bashrc`, because the non-login shell would otherwise miss the conda env (`MS/config/benchmarks/swebench.yaml:112-128`). An optional `env_startup_command` must return 0 or the instance fails (`MS/run/benchmarks/swebench.py:79-95`).

### 2.2 Our frontier runs (`MT/`)

**Extensions** (`MT/common/mswea/ext.py`):
- **`SSHEnvironment`** (`:34-72`) runs each action as `docker exec -i … timeout {t} bash -s` over SSH. The command is sent on stdin, which avoids quoting bugs.
  - `rc == 124` maps to `exception_info="command timed out after {t}s"` (`:64-67`).
  - The client-side timeout is `t + 120` (`:64`).
  - An optional `NETWORK` regex guard refuses curl/wget/pip install/git clone/… (`:28-31,50-53`).
- **`LongOpenRouterModel`** sets a 1200 s request timeout. Upstream hard-codes 60 s, too short for high-effort reasoning turns (`:75-100`).

**Config** (`MT/common/mswea/migration.yaml`):
- Limits: `step_limit: 250`, `cost_limit: 60.0`, yolo mode (`:25-27`).
- Model: `max_tokens: 32000`, `reasoning.effort: high`, `set_cache_control: default_end` (`:44-49`).
- Observation cut: 12,000 chars → 6,000 head + 6,000 tail (`:50-71`).
- Task04 overrides: `/workspace`, 1800 s per command, "build in the background and poll a log file" (`MT/task04_pcl_gpu_cuda12/mswea_sandbox.yaml:9-20`).

**Launcher** (`MT/task04_pcl_gpu_cuda12/run_agent.sh`):
- The container is `docker run -d --init --network none --runtime=nvidia … sleep infinity` (`:14`).
- `MSWEA_COST_TRACKING=ignore_errors` with the comment "a transient OpenRouter response without usage must not abort a run" (`:18`).
- The agent runs on the host, with SSH plus `docker exec` as the environment (`:22-26`).
- Patch collection is tolerant: `git add -A -- . ':(exclude,glob)**/build*/**' … || true; git diff --binary --cached base` (`:27`).

**Empirical data**, computed from the 7 `runs/*/trajectory.json` files:

| run | exit_status | api_calls | cost $ | format errors | timeouts | outputs ≥12k chars |
|---|---|---|---|---|---|---|
| gpt-5.6-sol-run1 | Submitted | 128 | 3.63 | 0 | 1 | 14 |
| gpt-5.6-sol-v2-run1 | Submitted | 125 | 3.91 | 0 | 0 | 15 |
| gpt-5.6-sol-v3-run1 | Submitted | 99 | 3.08 | 0 | 0 | 15 |
| gpt-6-astra-run1 | Submitted | 71 | 11.08 | 0 | 0 | 12 |
| gpt-6-astra-v2-run1 | **OpenRouterAPIError (HTTP 402 credits)** | 49 | 5.38 | 0 | 0 | 13 |
| gpt-6-astra-v3-aborted | **RuntimeError (no cost info, Usage {})** | 47 | 4.94 | 0 | 0 | 13 |
| gpt-6-astra-v3-run2 | Submitted | 134 | 39.75 | 0 | 0 | 17 |

What the runs show:
- Infra failures were 2 of 7 runs, and both were billing/accounting failures. Neither was a policy failure.
- About 10–13% of observations needed head/tail truncation.
- GPT 5.6 sol issued parallel tool calls in 6 turns (143 tool messages for 128 calls).
- The longest run's final prompt was 328,854 tokens (cached 328,851), costing $0.69 for that one call.
- Cost by third of the run: $5.30 for the first third, $24.28 for the last.

### 2.3 Harbor installed agents and terminus-2 (`H/`)

**Agent contract** (`H/agents/base.py`):
- `setup(env)` then `run(instruction, env, context)`. `run` should "populate the context … as the agent executes in case of a timeout" (`:342-382`).
- `resume`, `load` and `populate_context_post_run` (`:384-417`).
- Default setup timeout 360 s (`:58-59`).

**Capabilities** (`H/agents/capabilities.py:8-54`): `atif`, `resume`, `load_native_trajectory`, `load_atif_trajectory`, `handoff`, `native_config`, `skills`, `mcp_servers`, `bridges`.

**`BaseInstalledAgent`** (`H/agents/installed/base.py`):
- **Typed failures** (`:34-160`): `ApiRateLimitError`, `ApiUsageLimitError`, `ApiOverloadedError`, `ApiResponseStalledError`, `OutputTokenExceededError`, `ContextWindowExceededError`, `AgentSafetyRefusalError`, `AgentAuthenticationError`, `NetworkConnectionError`, …
  - They are classified by `ErrorPattern` regexes, and "the match furthest toward the end of the output wins" (`:223-230,912-948`).
  - This is a ready-made infra-vs-policy classifier for M10.
- **Install** goes through `ensure_system_dependencies` (apt/dnf/apk maps, `:361-445,723-788`) and `exec_as_root`/`exec_as_agent` (`:1010-1034`).
- **ATIF hooks**: `remote_session_logs_dir` and `convert_trajectory(logs_dir)` (`:352-358`).
- **Prefix seeding** (`:1109-1213`): `load_trajectory` (an ATIF `.json` or a native file) is validated at construction. `_seed_load_trajectory` converts ATIF via `atif_to_native_trajectory` and uploads it where the agent stores sessions.
  - The Trial resolves a run-level `agent.load_trajectory`, or a task-shipped `trajectory.json` beside the instruction (`H/trial/trial.py:1195-1235`).
  - It fails fast if the agent lacks the capability (`:1246-…`).

**Claude Code** (`H/agents/installed/claude_code.py`):
- **Options** (`:53-110`): `--max-turns`, `--effort`, `--max-budget-usd`, `--fallback-model`, `--append-system-prompt`, `--allowedTools`/`--disallowedTools`, `disable_web_search` (which adds WebSearch and WebFetch to the disallowed tools), `--permission-mode` (default `bypassPermissions`), `MAX_THINKING_TOKENS`.
- **Capabilities** (`:112-128`): atif, resume, load native and ATIF trajectories, handoff, native_config, skills, mcp_servers, ACP. Connection: `ANTHROPIC_API_KEY|ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`.
- **Install** (`:466-492`): skipped if the installed version matches. Otherwise it uses `npm install -g @anthropic-ai/claude-code` on apk, or `curl … bootstrap.sh | bash` elsewhere. **Network is required**, unless the binary is baked into the image.
- **Run** (`:1862-1960`):
  - With a custom base URL, **every tier alias** (`ANTHROPIC_DEFAULT_{SONNET,OPUS,HAIKU}_MODEL` and `CLAUDE_CODE_SUBAGENT_MODEL`) is pinned to `ANTHROPIC_MODEL` (`:1876-1880`). Otherwise background calls hit other models.
  - It sets `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` (`:1887`) and `IS_SANDBOX=1` (`:1890`).
  - `CLAUDE_CONFIG_DIR` lives under the logs dir (`:1895`).
  - The command is `printf "%s" "$INSTR" | claude --verbose --output-format=stream-json [--resume <id>] --print | tee claude-code.txt`.
- **Resume caveat**: seeded sessions are uploaded to `sessions/projects/-app/<id>.jsonl` (`:192-200`), so the agent's cwd must be `/app`.
- **Cost**: authoritative `total_cost_usd` from the final stream-json `result` event (`:1001-1034`). If that is missing, a LiteLLM per-step estimate, which returns `None` rather than a partial sum (`:1036-…`).
- **Failure classification** reads only the terminal `result`/`error` events, because assistant text can echo error phrases (`:427-449`).
- **ATIF conversion** de-duplicates tool calls replayed after compaction (`:1335-1340,1436-1450`).

**Codex** (`H/agents/installed/codex.py`):
- Capabilities: atif, resume, load native and ATIF (`:96-100`).
- Install: `npm install -g @openai/codex` (`:391-415`).
- Run: `codex exec [resume --last] --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --model M --json --enable unified_exec -- <instr>` (`:1603-1620`).
- "codex 0.118.0 only honors openai_base_url from config.toml, not the env var", so Harbor writes `$CODEX_HOME/config.toml` (`:1572-1580,1411-1424`).
- Resume copies the previous `sessions/` dir back into `$CODEX_HOME` (`:1585-1595`).
- The OpenEnv seam notes that Codex speaks the **Responses** dialect and "prefers a Responses WEBSOCKET transport, which the capture proxy does not serve", falling back to HTTPS only after five failed upgrades (`OE/src/openenv/harbor/seams.py:389-402`).

**OpenCode** (`H/agents/installed/opencode.py`):
- Capabilities: atif, resume, skills, mcp, ACP. **No load**, so no C3 prefix seeding (`:72-78`).
- Install: `npm i -g opencode-ai@ver` (`:130-150`).
- Run: `opencode --model=provider/model run --format=json [--continue] --thinking --dangerously-skip-permissions` (`:620-627`).
- The base URL goes in the provider config `options.baseURL` (`:546-553`).
- Harbor captures the instruction itself because "OpenCode's run stream omits [the user turn]" (`:586-588`).
- Any error event in the stream raises (`:630-634`).
- Lesson from the OpenEnv seam: name the provider something other than `openai` so the session key never reaches `OPENAI_API_KEY`. A grader read it as its judge key, and the result was "reward corruption, not a crash" (`OE/.../seams.py:336-358`).

**mini-swe-agent as an installed agent** (`H/agents/installed/mini_swe_agent.py`):
- Runs **inside** the sandbox: `uv tool install mini-swe-agent` (`:637-670`), then `mini-swe-agent --yolo --model … --task … --output … -c mini … --exit-immediately` (`:844-853`).
- Forces `MSWEA_COST_TRACKING=ignore_errors` (`:751-755`).
- Raises the LiteLLM timeout over the 600 s default (`:763-764`).
- Sends an `X-Session-ID` header (`:795-806`).
- Routes OpenAI reasoning models to `litellm_response` (`:808-826`).
- Note: "any -c suppresses the packaged config/mini.yaml", hence the prepended `-c mini` (`:838-842`).
- ATIF converter: `convert_mini_swe_agent_to_atif` (`:256-435`). We already import it (`C/interop/harbor/atif_import.py:104-107`).
- Capabilities: `atif`, `mcp_servers` only. No resume or load (`:507`).

**terminus-2** (`H/agents/terminus_2/terminus_2.py`), a Harbor-native, non-installed agent:
- **Tool surface**: tmux keystrokes. JSON or XML `{analysis, plan, commands:[{keystrokes, duration}], task_complete}` (`templates/terminus-json-plain.txt:1-46`).
- **Duration** is clamped to `min(duration, 60)` (`:1310`). Long builds therefore become polling turns.
- **Output** is cut to 10,000 **bytes**, half head and half tail, with an "[... output limited …]" banner (`:655-691`).
- **Timeout template**: "The previous command timed out after {timeout_sec} seconds … It is possible that the command is not yet finished executing" (`templates/timeout.txt`; `:1323-1353`).
- **Parse errors** re-prompt with "Previous response had parsing errors… Please fix these issues" and are still recorded as a step (`:1479-1500`).
- **Completion needs two consecutive `task_complete=true`**, the second after a confirmation prompt that shows the terminal state (`:626-647,1529-1545,1655-1662`).
- **Context**:
  - Proactive summarization when free tokens fall below 8000 (`:120-123,278`).
  - Three-step summarization (`:865-960`): the full-context model writes a summary; a fresh model, given only task + summary + screen, asks what is missing; the full-context model answers. Chat history is replaced with `[system, question_prompt (contains the summary), model_questions]`, and the answers arrive as the next prompt, "Here are the answers the other agent provided… Continue working on this task" (`:889-891,1060-1075`).
  - Reactive unwinding pops user+assistant pairs until 4000 tokens are free (`:693-717`).
  - If every step fails, an "ultimate fallback" sends instruction + last 1000 chars of the screen (`:1184-1191`).
  - No retry on `ContextLengthExceededError` when summarization is off (`:1107-1118`).
- **Truncated responses**: `OutputLengthExceededError` salvage, for the XML parser only (`:1211-1240`).
- **Limits**: `max_turns` defaults to 1,000,000 (`:440`). **No cost limit was found.**
- **Training data**: `collect_rollout_details` records token ids and logprobs, "incomplete if context summarization occurs" (`:108-115,307-309,719-741`).
- **Persistence**: the trajectory is dumped after every episode (`:1650-1653`).

**ATIF fields we need**:
- `Step.is_copied_context`: "should not be included in training data" (`H/models/trajectories/step.py:64-73`).
- `Metrics.prompt_token_ids`, `completion_token_ids`, `logprobs` (`H/models/trajectories/metrics.py:27-38`).

**Network**: agent-phase and verifier-phase policies, and `extra_allowed_hosts` merged into an ALLOWLIST for `agent.run()` only (`H/trial/network_policy.py:23-60`; `H/models/trial/config.py:125-131`). Docker enforces this with an egress-control sidecar when the kernel supports it (`H/environments/docker/docker.py:210-217`).

### 2.4 verifiers v1 harnesses, runtimes, interception (`VF/v1/`)

**Contract** (`VF/v1/harness.py:40-393`):
- Capability flags: `APPENDS_SYSTEM_PROMPT`, `SUPPORTS_MCP`, `SUPPORTS_TOOL_INTERCEPTION`, `SUPPORTS_RESUME`, `EXECUTES_CODE`, `NEEDS_CONTAINER` (`:41-65`).
- `launch(ctx, trace, runtime, endpoint, secret, mcp_urls, data) -> ProgramResult`: "model calls must reach the interception server at `endpoint` (bearer token `secret`)" (`:294-316`).
- `_check_result` classifies a nonzero exit (`:173-193`):
  - if a `stop_condition` was already set → clean;
  - runtime dead → `SandboxError`;
  - exit code 97 → `InterceptionError`;
  - otherwise → `HarnessError` with a 2000-char stderr tail.

**Harness families** (`VF/v1/harnesses/`):
- **Built-in chat loops, `bash` and `null`.** PEP 723 uv scripts run *inside* the runtime (`utils/launch.py:17-32`).
  - Tools: `bash`, `edit` (unique-string replace, on by default), optional web `search`, and MCP tools (`bash/harness.py:41-127`).
  - **Loop** (`utils/core.py:357-439`): `while True`: complete → append → `if not message.tool_calls: return`. **A text-only reply ends the episode.**
  - **Bash** runs `subprocess.run(["bash","-c",cmd], timeout=3600)` and returns stdout+stderr. Exceptions become `"error: …"` text (`core.py:160-171`).
  - **Malformed calls** get repair messages: "error: invalid JSON in tool arguments (…); resend the call with valid JSON" and `error: unknown tool` (`core.py:403-432`).
  - **Tool output** is truncated middle-out to `TOOL_OUTPUT_MAX_BYTES = 20_000` (`utils/compaction.py:18`).
  - **Streaming robustness**: a stream that ends without `finish_reason` is treated as a connection error and retried (`core.py:315-323`). Retries send `x-stainless-retry-count` so the server replays instead of double-committing (`:255-293`). When retries are exhausted the program exits 97, which is classified as infra (`:290-293`).
  - **Compaction** (`utils/compaction.py`):
    - Threshold = context window − `RESERVE_TOKENS = 16_384` (`:11`, window read from `/v1/models`).
    - It also triggers on an overflow error or `finish_reason=="length"`.
    - The summary call uses `tool_choice="none"`, with `COMPACTION_ATTEMPTS = 3`. Empty or tool-calling replies are resampled, and a rejected request falls back to the last good message count (`:14-16,155-256`).
    - The rebuilt context is `[system, user(POST_COMPACTION_FRAMING + summary)]`. The framing starts "Another language model started to solve this problem and produced a summary…" (`:33-38`).
  - **Overflow with compaction off** ends the run cleanly as a budget limit (`core.py:369-377`).
  - **Secrets** go "on argv so tools do not inherit them" (`launch.py:24-26`).
- **Third-party programs**:
  - **`mini_swe_agent`** pins `mini-swe-agent==2.4.6` and runs `--model-class litellm --exit-immediately --yolo -c mini -c agent.cost_limit=0 -c environment.timeout=86400 -c model.cost_tracking=ignore_errors -c …api_base={endpoint} -c …api_key={secret}` (`mini_swe_agent/harness.py:40-66`). Verifiers owns every limit.
  - **`terminus_2`** pins `harbor==0.23.0`. Harbor's environment is replaced by a local `subprocess.run` shim, and the tmux server is killed in a `finally` (`terminus_2/program.py:18-90`; `harness.py:61-77`).
- **CLI agents over the Agent Client Protocol (ACP)**: claude_code `@2.1.278`, codex `@0.155.1`, pi, kimi_code, openclaw, hermes, prime_agent, and `rlm` (`VF/v1/acp/__init__.py:64-419`).
  - **Transport**: length-prefixed JSON over stdio to a bundled runner, using `new_session(cwd, mcp_servers)`.
  - **Tool gating**: permission requests go to a `/tool` gate. If the gate is unreachable, the call is denied.
  - **Empty-run guard**: "ACP agent completed without committing a model turn" (`:168-176`).
  - **Shutdown** escalates graceful 10 s → terminate 5 s → kill 5 s (`:375-419`).
  - **Claude Code wiring**: `ANTHROPIC_BASE_URL` = endpoint without `/v1`, the per-trace `CLAUDE_CONFIG_DIR`, `DISABLE_AUTOUPDATER`, `IS_SANDBOX` (`claude_code/harness.py:91-101`).
  - **`rlm` budget knobs** include `max_total_turns`, `max_total_tokens` (default 10M), `max_tool_output_bytes` (20 KB) and `exec_timeout` (300). It maps `last_stop_reason` onto the trace stop (`rlm/harness.py:48-53,109-113,196-253`).

**Runtimes** (`VF/v1/runtimes/`): subprocess, docker, podman, apptainer, prime, modal, e2b.
- **Interface**: `start/stop/run/run_program/open_process/run_background/read/write/host_url/expose/prepare_execution` (`base.py:54-436`).
- **No snapshot or restore method exists.** The only mention is future-tense: "Sandbox snapshotting will make this free" (`utils/git.py:94-96`).
- **No per-command timeouts at runtime level.** The rollout stages (setup, rollout, finalize, scoring) own the deadlines (`rollout.py:268-282,436-463`).
- **Secret hygiene**: the subprocess runtime strips any env var containing `API_KEY` (`subprocess.py:26-29`).

**Interception** (`VF/v1/interception/server.py`):
- **Routes** (`:423-437`): `/v1/chat/completions`, **`/v1/responses`**, `/v1/messages`, **`/v1/messages/count_tokens`**, `/v1/models`, `/state`, `/task`, `/tool`.
- **`register()`** mints separate model and state secrets per rollout (`:368-377`). The run's model and sampling settings are forced onto every request (`dialects/chat.py:583-597`).
- **Request pipeline** (`:583-902`):
  1. Retry-replay guard (body digest plus retry or idempotency header).
  2. `session.refused()` checks limits.
  3. `@intercept` hooks.
  4. Graph `prepare_turn`.
  5. Sample.
  6. `commit`.
  7. Tool gate.
  8. Record.
  9. JSON or SSE response, with keepalives after 60 s.
- **Limits**: `RolloutLimits(max_turns, max_input_tokens, max_output_tokens, max_total_tokens)` are checked before each turn and are "soft by one turn". When a limit trips, the harness receives an HTTP 400 "rollout stopped: max_turns" and the trace stop is set (`VF/v1/session.py:83-115`). **There is no USD cap**; cost is only tracked.
- **Training client** (`clients/train.py:322-473`):
  - Renders messages to `prompt_ids` through `renderers`. When the new tail is only tool/user messages, it extends the previous turn's tokens incrementally (`bridge_to_next_turn`).
  - Calls vLLM `/inference/v1/generate` and stores `TurnTokens(prompt_ids, completion_ids, completion_logprobs, …)`.
- **Token graph** (`graph.py:9-18,83-166,816-819`): each `MessageNode` stores the token ids it adds, plus `mask`, `logprobs` and `advantages`. A bridged prefix that doesn't match raises an error. Divergent prompts (compaction, sub-agents) become new branches (`graph.py:1-8`).
- **Networking**: loopback without a tunnel. Remote runtimes use a frp tunnel. Docker rewrites `host_url` to an egress-proxy callback (`docker/__init__.py:330-338`), and a restricted network allowlists only the interception and MCP routes (`rollout.py:316`).
- **Prefill and resume**:
  - A prompt can be `str | Messages`; chat programs take it via `--initial-messages-file` (`task.py:93`; `launch.py:72-80`).
  - ACP agents accept user-role messages only (`acp/__init__.py:313-316`).
  - `Harness.resume` relaunches on the trace's current branch, for null, bash and browser_use only (`harness.py:240-288`).
- **ATIF: not found.** The native format is a message-graph `Trace` with `ModelCall` records (`trace.py:38,153,380`).

**What this means for Crucible**:
- PORT the bash loop's repair messages, stream-completion check, idempotent retry header and compaction.
- PATTERN the interception pipeline: refuse-before-serve limits, a dialect set that includes Responses and count_tokens, token-graph bridging. Our capture proxy needs exactly this.
- verifiers cannot snapshot. Its harnesses run their tools inside the runtime, so they bypass `World._call_tool`, which is the same limitation as seam S.
- The `mcp==2.0.0` pin conflicts with agentenv (`crucible/ARCHITECTURE.md:42`), so verifiers can be a DEPEND only in the separate training environment.

### 2.5 karpathy-autoresearch (`AR/`)

- **There is no Python agent loop.** `program.md` is a prompt "skill" handed to Claude or Codex: "Simply spin up your Claude/Codex … in this repo (and disable all permissions)" (`AR/README.md:44-50`).
- **Scope**: the agent may edit only `train.py`. `prepare.py` and the metric are read-only (`AR/program.md:26-31`; `prepare.py:340` "DO NOT CHANGE").
- **Loop** (`AR/program.md:94-104`), "LOOP FOREVER":
  1. hack `train.py`;
  2. `git commit`;
  3. `uv run train.py > run.log 2>&1` ("do NOT use tee or let output flood your context");
  4. `grep "^val_bpb:\|^peak_vram_mb:" run.log`, with `tail -n 50` on a crash;
  5. append to `results.tsv`;
  6. if `val_bpb` improved, keep the commit; if it is equal or worse, `git reset`.
- **Budgets**:
  - `TIME_BUDGET = 300` s of training, excluding the first 10 steps of compile (`AR/prepare.py:31`; `train.py:578-579,603`).
  - A prompt-level kill at 10 minutes, counted as a failure (`program.md:108`).
  - A NaN or loss > 100 makes the script fail fast (`train.py:570-572`).
- **Durable memory outside the context**: the git branch plus the untracked `results.tsv`, kept out of git so it survives resets (`program.md:66-88`; `.gitignore:23`).
- **"NEVER STOP … continue working indefinitely"** (`program.md:112`). This is the opposite of a submit protocol: the episode is bounded by the operator.
- **Crucible relevance**:
  - This is a protocol to inject into prompts, not a harness. Each keep/discard commit is a clean step boundary with a measured metric, so a natural Φ milestone for perf-type checks.
  - "Redirect, then grep" is the cheapest form of context management for build- and benchmark-heavy GPU tasks.

### 2.6 FineEnvs multi-harness RL + OpenEnv capture (`MH/`, `SP/`, `OE/`)

- **Harnesses** `("opencode","claude-code","codex","mini-swe-agent")`, assigned deterministically per task (`MH/envs/harbor/smoldataenv_harbor/environment.py:9,74`). The task is matched by prompt, not seed, because "seeds can change after a restart" (`:66`).
- **Limits per rollout**: `agent_step_limit=17`, `agent_timeout_sec=600` (`:80-81`). WebSocket message timeout 1800 s (`:69`).
- **Proxy**:
  - The harness reaches the proxy through a base URL plus a per-rollout session key. An unknown key gets a 401 (`SP/chapters/openenv.mdx:54-62`).
  - The proxy converts four dialects (OpenAI chat, Responses, Anthropic, Google) to chat completions and replays the answer in the caller's format. It never streams from the engine and replays as SSE when the caller asked to stream (`:71-86`).
  - Per-harness env wiring is in `OE/src/openenv/harbor/seams.py`:
    - claude-code `:374-388` (no `/v1` suffix);
    - codex `:389-402`;
    - mini-swe-agent `:454-470`;
    - opencode `:336-360` (provider block, no env vars).
- **Token capture**:
  - The proxy injects `return_token_ids=True` and `top_logprobs=0` ("must be SET, not omitted"), and forces `top_p=1.0, top_k=-1` at the tokens tier (`OE/src/openenv/core/harness/capture/upstream.py:14-33,139-148`).
  - Moving `top_p` to 1.0 took the importance ratio from 0.985–0.993 to 0.9984–0.9999 (`SP/chapters/openenv.mdx:90-110`).
  - **Reconstruction**: each call is parented to the earlier call whose prompt+completion is the longest exact token prefix. Retries become siblings, and compaction or subagents start new roots (`OE/.../capture/graph.py:1-35,299-325`).
  - The loss mask is 1 only where a real behaviour logprob exists (`graph.py:443-484`).
- **Robustness**:
  - Infra failures return `reward=None`/`NaN`, "not scored as wrong" (`MH/README.md:80`).
  - An in-process design once let an agent exception crash one rank and hang NCCL; failures are now values (`SP/chapters/openenv.mdx:213-217`).
  - The proxy's health check starved at about 200 concurrent sessions and crashed at 320 (`:160-161`).
  - The default `max_concurrent_envs=4` is smaller than a GRPO group of 8, which produced `CAPACITY_REACHED` (`:317-319`).
- **Results**:
  - The harness matters: LFM2.5 scores 62% under mini-swe-agent vs 33% under Claude Code (`SP/chapters/training.mdx:54-56`).
  - Multi-harness RL reached 54.2% vs 52.3% for OpenCode-only, within noise, but with better transfer to Claude Code and Codex (`:115-155`).
  - Claude Code produced 77% of training rows because of history rewrites (`:380`).

### 2.7 Crucible's current policy layer (`C/`)

**`LLMToolPolicy`** (`C/actors/policy.py:61-158`):
- OpenAI function calling over the World's tools, plus a `send_message` tool (`:56-57,86-95`).
- Parallel calls are queued and executed one per `World.step`. Their results are appended before any message that arrived meanwhile (`:98-112,139-158`).
- Provider reasoning fields are replayed verbatim (`:150-151`).
- Optional head/tail cut (2/3 + 1/3) at `max_tool_output` (`:99-103`).
- Empty reply: up to 2 `NUDGE`s (`:60,130-134`).
- **Any text-only reply means `Done`** (`:135-138`).
- `ContextOverflow` → `stop="context_overflow"`, scored (`:121-123`). `LLMError`/`BudgetExceeded` → `failure_kind`, masked (`:124-127`).
- Per-call accounting in `model_calls` (`:128-129`).
- Defaults: `max_tokens=1200`, `temperature=0.0` (`:66-67`).

**`run_episode`** (`:187-268`):
- Round-robin over the policies.
- `EpisodeLimits(max_steps=50, max_tool_errors=25, timeout_s, end_on, snapshot_every)` (`:161-169`).
- Snapshots every N steps, de-duplicated by fingerprint, **without quiesce** (`:196-205`).
- A final `snapshot(quiesce=True)`. A `WorldError` there means the policy outlived the boundary, recorded as `crucible:policy_misbehavior`, scored (`:243-247`).
- An answer exposure masks the episode (`:249-251`).
- **With `reset=False`, the observation carries no tools and no instruction** (`:190`). A branched policy must therefore be pre-seeded.

**`LLMClient`** (`C/actors/llm.py`):
- Hard USD budget, **pre-checked** before each call (`:162-163`).
- Record/replay cache keyed by `sha256(model, messages, tools, params[, sample_key])` (`:139-161`).
- Error mapping:
  - 402 or `insufficient_quota` → `BudgetExceeded` (`:184-185`);
  - overflow markers ported from verifiers → `ContextOverflow` (`:52-62,186-187`);
  - non-retryable 4xx → `RequestRejected` (`:188-189`);
  - no `choices` → retry (`:177-178`);
  - missing usage → `UsageBlank` (`:200-201`).
- Retry delay: `Retry-After`, else exponential backoff with jitter (`:99-108`).

**`World`** (`C/world/world.py`):
- `_call_tool` is the one choke point (`:295-339`). It handles name, ACL and schema checks with "Did you mean" (`:314-321`), applies the `tool_timeout_s` deadline as infra (`:322-327`), contains crashes as infra (`:328-332`), labels faults (`:335-338`) and calls `_bound`.
- `_bound` makes a single head/tail cut of 2/3 + 1/3 with a banner and stores the full text by content hash (`:346-355`). Default 30,000 chars (`C/core/schema/world.py:172`).
- `reset(start=ref)` restores without re-delivering the prefill (`:358-389`).
- `snapshot()`/`restore()` cover App states, clock, events, actor memory, ACL, inboxes, done set and backend state (`:536-587`). The fingerprint excludes `step` (`:557-562`).
- `to_trace()` attaches `data.writes` for M4 attribution (`:593-609`).

**`WorkspaceApp`** (`C/world/apps/workspace.py`):
- Tools: `list_dir`, `read_file(offset, limit≤2000)`, `search`, `write_file`, `edit_file(old,new,replace_all)`, and optionally `shell(command, timeout_s=600)` (`:78-98`).
- `shell` runs one fresh `run()` per call (stateless). It is refused on the local backend without `allow_unsafe_local_shell`, clamps the timeout to `max_timeout_s` (default 3600), appends `[stderr]` separately, applies its own half/half cut at `max_output` (default 30,000), and prefixes the result with `[exit N]` or `[TIMEOUT]` (`:133-143`).
- `GitRepoApp` adds `git_status` and `git_diff(ref="base")` (`:146-172`).

**Deadline**: `with_deadline` runs `timeout -s KILL N bash -c …` inside the sandbox, because killing the client leaves the command running (`C/world/exec.py:117-127`).

**`CaptureProxy`** (`C/connectors/capture_proxy.py`):
- Session key = capability; the upstream key stays on the proxy (`:5-8,62-66`).
- Forwards unchanged, assembles SSE, appends a JSONL record with a digest so retries are kept last-wins (`:87-128`).
- Routes: `/v1/chat/completions`, `/v1/messages`, `/health` (`:140-141`).
- The connector adds `context_reset` events when a request does not extend the previous one (`:14-17`).
- **Missing**: budget/turn caps, `/v1/responses`, `/v1/messages/count_tokens`, token-id injection, a snapshot hook.

**Harbor interop**:
- `HarborBackend` uses Harbor's `EnvironmentFactory`. For Docker it creates our `GpuDockerEnvironment` with `network_none` (`C/interop/harbor/world_backend.py:154-202`).
- `episode_to_atif` builds one ATIF trajectory per actor (`C/interop/harbor/atif.py:23-60`).
- Mini-swe-agent and Claude Code session imports exist (`C/interop/harbor/atif_import.py:104-117`).

---

## 3. (a) Harness comparison

"Seam" says where actions execute: **W** = through `World.step` (step-level trace and snapshots); **S** = inside the sandbox with the harness's own tools (the World sees only the final state and the proxy log).

| Harness | Seam / where the loop runs | Tool surface | Loop and termination | Context management | Robustness features | Cost control | ATIF | Arbitrary environment | Long-horizon coding evidence | C3-branchable |
|---|---|---|---|---|---|---|---|---|---|---|
| **mini-swe-agent 2.4.6** | Host loop; `Environment.execute` is any object (our SSH/docker exec) | `bash{command}`, one fresh subshell per call (`MS/models/utils/actions_toolcall.py:11-27`) | Every reply must contain ≥1 tool call; all of them execute. Submit = sentinel first line with rc 0 (`MS/environments/local.py:45-56`) | **None.** Only the head/tail observation cut (`MS/config/mini.yaml:112-128`); context-window error aborts (`litellm_model.py:54`) | Format-error templates aware of `finish_reason`; streak cap 3; trajectory saved every step; process group killed on timeout; partial output kept; 10 retries | `cost_limit`, `step_limit`, wall time (pre-check); global env caps | Via Harbor converter (`H/agents/installed/mini_swe_agent.py:256`) | Trivial: duck-typed `Environment` | **Ours:** 5/7 task04 runs submitted; the 2 failures were billing infra; 0 format errors in 680 calls. FineEnvs: best harness for a 2.6B model (62%) | Yes, if driven through the World (stateless shell); prefix = `messages` |
| **Crucible `LLMToolPolicy`** (today) | W | World tools (`list_dir`/`read_file`/`search`/`write_file`/`edit_file`/`shell`/git_*) + `send_message` | Text-only reply = Done, **even when `finish=length`** (`C/actors/policy.py:135-138`) | None; overflow → truncated, scored (`:121-123`) | Up to 2 empty-reply nudges; parallel batching; raw provider fields replayed; World "Did you mean" | Client USD budget (masked); `max_steps` | `episode_to_atif` | Native | None yet | Not yet: no `fork()`; `temperature=0`/`seed=0` |
| **terminus-2** (Harbor) | Harbor process; tmux inside the sandbox | Keystrokes + duration (≤60 s) (`H/agents/terminus_2/terminus_2.py:1310`) | `task_complete` twice (confirmation prompt) | Proactive 3-step summarization (<8000 free tokens) and unwind (`:693-717,865-960`) | Parse-error re-prompt; timeout template; 10 KB head/tail; salvage of truncated XML | `max_turns` only (no USD cap found) | Native ATIF, token ids optional | Any Harbor environment | Not measured in our sources | **No:** the tmux pane, cwd and running processes are outside `WorldRef` |
| **Claude Code** (Harbor installed) | S | Native tools (Bash/Edit/Read/…, subagents), MCP, skills | CLI `--print` run to completion | Native auto-compaction (history rewrites, about 8 rows per rollout, `SP/chapters/training.mdx:195-197`) | Typed error classes from stream-json `result`/`error` events | `--max-budget-usd`, `--max-turns` (`claude_code.py:54-67`) | Yes; also ATIF→native for load | Needs Node/the binary in the image and a network lane to the proxy | Frontier default; not measured in our sources | Partly: `load_trajectory` → `--resume` with its own prefix; cwd must be `/app` |
| **Codex** (Harbor installed) | S | Native (unified_exec, apply_patch), MCP | `codex exec --json` | Native | Typed errors | Not found in Harbor options; enforce at the proxy | Yes, plus load | Needs the **Responses** dialect; base URL via config.toml; WebSocket caveat (`OE/.../seams.py:389-402`) | Not measured | Partly: `resume --last`, load |
| **OpenCode** (Harbor installed) | S | Native tools, MCP | `opencode run --format=json` | Native | Error events raise | Not found; proxy | Yes | Provider `baseURL`; use an aliased provider name | FineEnvs RL target | **No:** resume only, no load |
| **verifiers `bash`/`null`** | Inside the runtime (uv script) | `bash`, `edit`, MCP | Text-only reply ends the run | Summary compaction at window − 16k (`VF/v1/harnesses/utils/compaction.py`) | Stream-finish check; idempotent retries; JSON-repair messages; exit 97 = infra | Interception turn/token caps (no USD) | No (native Trace) | Any verifiers runtime | Not measured | No runtime snapshots; prefill via initial-messages file |
| **verifiers ACP agents** (claude_code/codex/pi/rlm/…) | S | Native agent tools behind a `/tool` permission gate | ACP session turns; empty-run guard | Native; rlm has its own compaction | Graceful → kill shutdown; deny when the gate is down | Interception caps; rlm `max_total_tokens` | No | Needs a runtime with live processes | Not measured | No |
| **autoresearch** (protocol, not code) | S (Claude/Codex) | Whatever the agent has | "NEVER STOP"; keep/revert per experiment | Redirect to a log and grep it; state kept in git + `results.tsv` | Crash → tail the log; 10-minute kill; NaN fast-fail | Fixed time budget per experiment | No | Any repo | About 100 experiments overnight (`AR/README.md:64`) | n/a (its commits are natural step boundaries) |

---

## 4. (b) Recommendation and integration design

### 4.1 Decision

| Purpose | Harness | Seam | Why |
|---|---|---|---|
| C3 V̂(s_t), PRM labels, corr(C3, ΔΦ) on task04 | **`BashLoopPolicy`** (native mini-swe-agent v2 port) | W | Every action passes the choke point. The prefix is a plain message list. The bash-only profile accepts the existing task04 mini-swe-agent trajectories as prefixes **verbatim**: same `bash{command}` schema, same sentinel. |
| Cheap baselines (the "beats an LLM judge" gate, R1–R6 smoke) | `BashLoopPolicy` with a cheap model; `ScriptedPolicy` for oracle and no-op | W | Deterministic replay through `LLMClient` record/replay (`C/actors/llm.py:5-7`). |
| RL rollouts | `BashLoopPolicy` against a vLLM endpoint behind the capture proxy, with token ids injected. Optional: verifiers `bash` harness + interception in the separate verifiers environment | W | Token-exact data. One harness per GRPO group. |
| Frontier calibration and ceilings | Harbor installed **Claude Code, Codex, mini-swe-agent (installed), OpenCode**, plus **terminus-2** | S | Measures each harness as shipped. Per-harness calibration curves. |
| C3 on frontier installed agents (optional, expensive) | Claude Code / Codex via `load_trajectory` | S | Only with each agent's own prefix. OpenCode cannot load. |

**Not chosen as native**:
- **terminus-2**: its tmux state is outside `WorldRef`, durations are capped at 60 s (so GPU builds become polling turns), and it has no USD cap.
- **verifiers chat loops as DEPEND in core**: they run inside the runtime, have no snapshots, and carry the `mcp` pin conflict. We **PORT** their compaction and repair logic instead.

### 4.2 Interfaces (M10 `runtime/harness/`)

```python
@dataclass(frozen=True)
class HarnessId:            # recorded in RunRecord + Trace.source (TL;DR 11)
    name: str; version: str; config_hash: str      # e.g. ("crucible-bash", "0.1", sha256(prompts+limits))

@dataclass
class RolloutLimits:        # superset of EpisodeLimits (C/actors/policy.py:161-169)
    max_steps: int = 250                 # model calls; mswea step_limit (MT/common/mswea/migration.yaml:25)
    cost_usd: float | None = None        # per episode/branch; hitting it = policy outcome "cost_limit" (truncated, scored)
    wall_s: float | None = None
    max_consecutive_format_errors: int = 3          # MS/agents/default.py:32
    max_tool_errors: int | None = 25
    max_tokens_per_call: int = 16000     # identical in calibration and RL (SP training.mdx:350-354)
    request_timeout_s: float = 1200      # MT/common/mswea/ext.py:75-76; LLMClient default 120 s is too short
    compaction: CompactionCfg | None = CompactionCfg(reserve_tokens=16_384, attempts=3)

@dataclass
class PolicyPrefix:
    kind: Literal["none", "messages", "atif"]
    messages: list[dict] | None = None   # native: exact message list incl. tool_call ids and provider reasoning fields
    atif: Trajectory | None = None       # installed: the agent's own trajectory up to step t
    harness: HarnessId | None = None     # prefix must come from the same harness family
    step: int = 0                        # World.step_index where the prefix ends

@dataclass
class Branch:
    ref: WorldRef                        # C/core/schema/state.py:90-104
    prefix: PolicyPrefix
    branchable: bool                     # False if a policy process was alive at snapshot (TL;DR 5)
    reason: str | None = None

class Harness(Protocol):
    id: HarnessId
    seam: Literal["world", "sandbox"]
    caps: HarnessCaps                    # step_snapshots, prefix_resume, token_capture, atif, usd_cap
    async def rollout(self, world: World, actor: str, instruction: str, *, start: Branch | None,
                      limits: RolloutLimits, sample_key: str, temperature: float | None) -> RolloutResult

@dataclass
class RolloutResult(EpisodeResult):      # C/actors/policy.py:172-184 + :
    harness: HarnessId
    prefix_steps: int                    # steps copied from the prefix: never trained on (is_copied_context)
    spend: Spend                         # usd, calls, prompt/completion/cached tokens
    policy_states: dict[int, str]        # World step -> store key of the policy's messages at that step (for later branching)
    capture_log: Path | None             # seam S
    atif: Trajectory | None
    consistency: dict | None             # capture-vs-ATIF check (seam S)
```

`run_episode` gets `start: Branch | None`. When `start` is set:
1. Call `world.reset(start=start.ref)`.
2. **Check that the restored fingerprint equals `start.ref.fingerprint`.** A mismatch is `crucible:snapshot_mismatch`, masked (`C/core/schema/run.py:38`).
3. Call `policy.fork(start.prefix, sample_key)`.
4. Hand the forked policy a first observation that carries `tools` but **not** the instruction, since the instruction is already in the prefix.

Today `reset=False` yields observations without tools (`policy.py:190`), and `reset(start=…)` re-sends the instruction (`world.py:389`), so neither works as-is.

### 4.3 `BashLoopPolicy`: exact spec (PORT mini-swe-agent v2 + verifiers repair/compaction)

- **Tool profiles**:
  - `mswea` (default): a single `bash{command}` mapped to the workspace App's `shell` tool.
  - `editor`: `bash` + `read_file` + `edit_file` (verifiers' `bash`+`edit`).
  - Tool schemas are fixed per `HarnessId`.
- **Prompts**: the system and instance templates from `MT/common/mswea/migration.yaml:4-24` (stateless subshell, timeout, no network, submit rule), plus the autoresearch hygiene lines:
  - "redirect long output to a file and grep/tail it";
  - "commit before risky experiments; reset on regression".
- **Turn handling**, in order:
  1. Pre-check limits: `step >= max_steps` → `max_steps`; `episode_cost >= cost_usd` → `cost_limit`; wall time → `timeout`. All three are policy outcomes, truncated (`MS/agents/default.py:130-147`).
  2. Call the LLM.
     - `ContextOverflow` → compact once, then retry. If that fails → `context_overflow` (truncated, scored).
     - `BudgetExceeded`/`UsageBlank`/`LLMError` → masked infra (`C/actors/llm.py:25-49`).
  3. If `finish_reason == "length"`, or there are no tool calls (whether or not text is present), or JSON is malformed, or the tool is unknown, or `command` is missing → append the **format-error** user message (`MS/config/mini.yaml:129-149`; verifiers' JSON-repair wording, `VF/v1/harnesses/utils/core.py:403-432`). `streak += 1`. At `streak >= 3` → stop `format_error`, a policy outcome, scored.
     - Empty replies count toward the streak, replacing the separate nudge counter.
     - Billed cost is still charged (`MS/agents/default.py:101-102`).
  4. Otherwise execute all tool calls in order through `World.step` and reset the streak. Pad with "action was not executed" if the episode stops mid-batch (`MS/models/utils/actions_toolcall.py:88-89`).
- **Submission**: if a `shell` result's first output line is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` and rc == 0 → `Done(reason="submitted", submission=rest)`. The prompt requires it "on its own" (`MS/config/mini.yaml:18-19`).
  - An optional `termination="text"` mode (the verifiers convention) exists for chat-like Worlds only.
- **Observation**: a **single** head/tail cut at the World (`_bound`, `C/world/world.py:346-355`). Drop the shell's own second cut (`C/world/apps/workspace.py:140-141`).
  - Format: `<returncode>N</returncode>[<exception>…</exception>]<output>…</output>`.
  - On timeout: `exception = "command timed out after Ns"` with the partial output. `HarborExec` keeps it when the in-sandbox deadline fires (`C/interop/harbor/world_backend.py:60-69`).
- **Compaction** (PORT `VF/v1/harnesses/utils/compaction.py`):
  - Trigger when `prompt_tokens + est(tool_results) >= window − 16384`, or on overflow or `finish=length`.
  - The summary call uses `tool_choice="none"` with ≤3 attempts. The conversation becomes `[system, user(FRAMING + summary)]`.
  - Emit a `context_compacted` trace event. In ATIF, the replaced span is `is_copied_context`.
  - For RL, each compaction starts a new token root (OpenEnv/verifiers graphs).
- **Caching**: set Anthropic `cache_control` on the last message (`MS/models/utils/cache_control.py:49-…`, auto-on for Claude models, `MS/models/__init__.py:55-60`). For C3, run branch 1 first and then branches 2..K concurrently, so they hit the provider's prefix cache.
- **Persistence**: after every step, write `messages` to the store (content-addressed) and keep the key in `policy_states[step]`. That gives free prefixes for later branching, and an aborted run is still usable (`MS/agents/default.py:120-121`).
- **Sampling**: C3 and RL use `temperature >= 0.7` (or the provider default) and `sample_key = f"{traj}:{t}:{k}"`. With the current `temperature=0.0` and `seed=0` defaults and a shared cache key, the K branches would replay one completion (`C/actors/llm.py:150-153`).
- **Retries**: raise `LLMClient.max_retries` from 3 to 8 for calibration and C3, with exponential backoff capped at 60 s (`MS/models/utils/retry.py:19-25`). Retry a stream that ends without `finish_reason` (`VF/v1/harnesses/utils/core.py:315-323`).

### 4.4 Installed agents (seam S): Harbor agent + capture proxy

1. **World**: `HarborBackend` (`C/interop/harbor/world_backend.py`) with the harness binary pinned in the image (TL;DR 10). The policy user is set as `environment.default_user`.
2. **Proxy session**: `key = proxy.register(f"{episode}-{k}")`, with new per-session limits `{budget_usd, max_requests, deadline}`.
   - When a limit is exceeded, the proxy answers in the caller's dialect with HTTP 400 `"rollout stopped: cost_limit"` (the verifiers convention, `VF/v1/session.py:83-115`) and records `stop` in the session metadata.
   - **Do not use 402**: we classify 402 as credits exhausted, i.e. infra.
   - New routes: `/v1/responses` (Codex), `/v1/messages/count_tokens` (Claude Code), and `/v1/models`.
3. **Network lane**: agent phase = `ALLOWLIST[proxy_host]`, verifier phase = no network. Two options:
   - Harbor's egress-control sidecar (`H/trial/network_policy.py:23-42`);
   - a `GpuDockerEnvironment` option `agent_lane=proxy_host:port` that replaces `--network none` with an internal network whose only route is the proxy. Verifiers' Docker runtime does the equivalent with its egress-proxy callback URL (`VF/v1/runtimes/docker/__init__.py:330-338`).
4. **Agent**: build the Harbor agent class directly (DEPEND), e.g. `ClaudeCode(logs_dir, model_name="anthropic/<m>", extra_env={"ANTHROPIC_BASE_URL": proxy, "ANTHROPIC_API_KEY": key}, max_turns=…, max_budget_usd=…, disable_web_search=True)`. Then call `await agent.setup(world.backend.env)` and `await agent.run(instruction, world.backend.env, ctx)`, or `agent.load(...)` when branching.
   - Codex: `OPENAI_BASE_URL={proxy}/v1` only through the agent's env (`OE/.../seams.py:389-395`).
   - OpenCode: an aliased provider name (`OE/.../seams.py:336-358`).
5. **Step snapshots**: a proxy `on_request(session, body)` hook calls `world.snapshot()` *before* forwarding a main-chain request. The agent is blocked on the model at that moment, so the sandbox sits between tool actions. The snapshot is keyed by request index.
   - **Skip** `count_tokens`, small-model background calls and subagent roots. Main chain = the request extends the previous main-chain prompt (the `context_reset` logic in `capture_proxy.py:14-17`).
   - Mark a snapshot `branchable=False` when policy processes are alive.
6. **Close**:
   - Take `world.snapshot(quiesce=True)`.
   - The primary trace comes from the capture log (`CaptureLogConnector`). The cross-check comes from `agent.populate_context_post_run` → `trajectory.json` (`H/agents/installed/claude_code.py:1657-1686`).
   - If turn counts or completion tokens disagree, mask the sample as `crucible:capture_mismatch`, a new kind.
   - Cost comes from the proxy's metered usage. Harbor's `total_cost_usd` is the cross-check (`claude_code.py:1001-1034`).
7. **Failure mapping** (Harbor typed errors, `H/agents/installed/base.py:34-160`):

   | Harbor error | Crucible outcome |
   |---|---|
   | RateLimit / Overloaded / InternalServer / ConnectionClosed / ResponseStalled / NetworkConnection | `provider_unavailable` / `transport_*`, masked |
   | UsageLimit | `crucible:credits_exhausted`, masked |
   | Authentication / ModelNotFound / ProviderResourceNotFound | `agent_request_failed`, masked (a config bug) |
   | ContextWindowExceeded / OutputTokenExceeded | `context_overflow`, truncated and scored (matches `C/actors/llm.py:45-49`) |
   | Proxy `rollout stopped` | `cost_limit` / `max_steps`, truncated and scored |
   | Generic NonZeroAgentExitCode | `agent_run_error`, masked |
   | SafetyRefusal | open question (§7) |

### 4.5 C3 branch-rollout algorithm (M6 `credit/branch.py`)

```
input: trajectory τ with per-step (WorldRef_t, PolicyPrefix_t), verifier O(·) (M7 gated outcome), harness h, K
T  = branch points: steps where the M5 tracker's milestone set changes ∪ uniform sample (|T| ≈ 6–8), filtered by branchable(t)
for t in T:
    for k in 1..K:                              # k=1 first, then 2..K concurrently (prefix cache)
        world.reset(start=ref_t); assert fingerprint == ref_t.fingerprint   else mask(snapshot_mismatch)
        π_k = h.fork(prefix_t, sample_key=f"{τ.id}:{t}:{k}", temperature=T_roll)
        r_k = run_episode(world, {actor: π_k}, start=Branch(ref_t, prefix_t),
                          limits=RolloutLimits(max_steps=⌈1.5·(len(τ)−t)⌉+10, cost_usd=c_branch))
        o_k = None if r_k.masked else O(r_k.final)        # truncated branches score their final state (usually 0)
    V̂(s_t) = mean(o_k over unmasked);  n_eff = #unmasked;  report Wilson 95% CI;  drop t if n_eff < K/2
A_t = V̂(s_t) − V̂(s_{t−1})           (DESIGN.md:435);   report Pearson/Spearman corr(A_t, ΔΦ_t) per trajectory
```

- **V̂ depends on the policy.** For the task04 trajectories, which came from mini-swe-agent with GPT 5.6 sol and GPT 6 Astra, run `BashLoopPolicy(mswea profile)` with a fixed π_ref and record π_ref in the report.
- **Getting `WorldRef_t` for a trajectory recorded outside Crucible**: M5 replay re-executes the recorded commands into a fresh World. Each step is quarantined if its return code differs from the recording. Prefix = `trajectory.json["messages"]` up to the assistant message of step t, with `extra` stripped and tool_call ids kept.
- **Budget for the gate**, from our data: GPT 5.6 sol averages ≈ $0.028 per call ($3.63 / 128). With |T| = 6, K = 4 and about 64 remaining calls per branch: 6 × 4 × 64 × 0.028 ≈ **$43 per trajectory, ≈ $130 for the 3 task04 trajectories**. The same plan with GPT 6 Astra ($39.75 / 134 calls ≈ $0.30 per call, rising with context) would cost ≈ $19 per branch and ≈ $460 per trajectory. Use the cheaper model as π_ref.
- **Wall-clock**: every branch rebuilds the ignored build tree. Budget about 5–10 minutes per PCL build per branch, and run branches concurrently, each on its own container. GPU contention is a known source of perf noise (B.5), so use correctness-tier outcomes for V̂ and report perf separately.

### 4.6 Cost capping: five layers

1. **Per call**: `max_tokens` (the same value in calibration and RL).
2. **Per episode or branch**: `RolloutLimits.cost_usd`.
   - Native: checked before the call, as in mini-swe-agent.
   - Seam S: the proxy session budget.
   - Plus the harness's own flags (`--max-budget-usd`, `--max-turns`).
   - Hitting it is a policy outcome.
3. **Per job**: the `LLMClient(budget_usd)` or a global proxy cap (like `MSWEA_GLOBAL_COST_LIMIT`, `MS/models/__init__.py:13-42`). Hitting it is infra: stop scheduling and mask in-flight samples.
4. **Context**: compaction (§4.3) turns quadratic growth into roughly linear growth. Our evidence: 4.6× cost growth from the first third of the run to the last.
5. **Caching**: `cache_control` plus branch ordering for prefix reuse.

### 4.7 Trajectories → traces → steps

- **Native (seam W)**:
  - `World.to_trace()` is canonical: events with `writes` for M4 (`C/world/world.py:593-609`). Step ids = `World.step_index` = the `steps.jsonl` ids.
  - Policy metadata (`model_calls`: usage, finish, cost, latency) joins action events by step.
  - `episode_to_atif` (`C/interop/harbor/atif.py:23-60`) gains per-step `metrics`, and `is_copied_context` on prefix and compacted steps.
  - Also export the mini-swe-agent-1.1 format, so Harbor's converter and viewer work unchanged.
- **Installed (seam S)**:
  - The capture log becomes a Trace, with step id = main-chain request index. Snapshots are keyed the same way.
  - Tool calls parsed from responses feed `writes_of` (already used by the capture connector).
  - The harness's ATIF is attached as evidence.
- **Training**:
  - Engine token ids and logprobs are captured at the proxy (inject `return_token_ids`, `top_logprobs=0`, `top_p=1`), or come from verifiers' `TrainClient`.
  - Turns are rebuilt by token prefix; loss mask = sampled tokens with real logprobs (`OE/.../capture/graph.py:443-484`; `VF/v1/graph.py:83-166`).

---

## 5. (c) Robustness lessons to copy

| # | Failure mode | What the best harness does (evidence) | Crucible today | Do |
|---|---|---|---|---|
| 1 | Reply has no tool call | `FormatError` "Every response MUST include at least one tool call" (`MS/models/utils/actions_toolcall.py:40-52`) | Treated as Done (`C/actors/policy.py:135-138`) | Format error + streak; Done only on submit |
| 2 | Reply cut off at max tokens | Template branches on `finish_reason=="length"`: "Respond more concisely…" (`MS/config/mini.yaml:130-131`); verifiers compacts on `length` (`VF/v1/harnesses/utils/compaction.py`); terminus salvages XML (`H/agents/terminus_2/terminus_2.py:1211-1240`) | Partial text = Done | Cut-off template; count toward streak |
| 3 | Malformed JSON args / unknown tool / missing arg | Error naming the fault, then "resend the call with valid JSON" (`VF/v1/harnesses/utils/core.py:403-432`; `MS/.../actions_toolcall.py:54-74`) | World answers "Invalid arguments… Resend…", "Did you mean" (`C/world/world.py:314-321`) | Keep; count toward streak |
| 4 | Repeated format errors | Exit `RepeatedFormatError` after 3 in a row; streak resets on a clean step (`MS/agents/default.py:99-112`) | 25 consecutive tool errors (`policy.py:166`) | Separate format-error cap (3) from tool-error cap (25) |
| 5 | Empty reply (reasoning spent the budget) | ACP: "completed without committing a model turn" is an error (`VF/v1/acp/__init__.py:168-176`); verifiers resamples empty summaries | 2 nudges (`policy.py:130-134`) | Keep the nudge; count it in the streak; never write an empty assistant turn |
| 6 | Billing on a failed parse | Charge the cost even on `FormatError` (`MS/agents/default.py:101-102`; `litellm_model.py:87-98`) | Charged by the client | Keep |
| 7 | Command timeout | Kill the **process group**, return partial output + `exception_info`, keep going (`MS/environments/local.py:72-92`); our SSH env maps rc 124 → "timed out after Ns" (`MT/common/mswea/ext.py:64-70`) | In-sandbox `timeout -s KILL` (`C/world/exec.py:117-127`) | Keep; clamp the shell to `tool_timeout_s − 60` (TL;DR 15) |
| 8 | Client-side kill leaves the command running | The in-container deadline is the real one; the client is a backstop at +15 s (`C/interop/harbor/world_backend.py:56-60`); our SSH env used +120 s | Done | Keep |
| 9 | Huge output | Head/tail with an explicit elision count: 5k+5k (`MS/config/mini.yaml:112-128`), 6k+6k (`MT/.../migration.yaml:50-71`), 10 KB bytes (terminus), 20 KB middle-out (verifiers) | 2/3 + 1/3 at 30k and full text stored (`world.py:346-355`), but the shell cuts again (`workspace.py:140-141`) | One cut, at the World |
| 10 | Long-running jobs flood the context | "redirect everything… do NOT use tee", then grep/tail (`AR/program.md:99-101`) | n/a | Put it in coding prompts |
| 11 | Submission detection | Sentinel on the first output line, rc 0, as a separate command (`MS/environments/local.py:45-56`; `mini.yaml:18-19`); terminus asks twice (`terminus_2.py:626-647`) | Implicit | Sentinel; optional confirm turn for expensive verifiers |
| 12 | Context overflow | mini-swe-agent: abort, no retry (`litellm_model.py:54`); terminus: summarize or unwind (`:693-717,865-960`); verifiers: compaction, else clean stop (`core.py:369-377`) | Truncated, scored (`policy.py:121-123`) | Compaction first, then truncated |
| 13 | Transient provider errors | 10 attempts, exponential 4–60 s (`MS/models/utils/retry.py:19-25`); idempotent retry header (`VF/.../core.py:255-293`); stream with no `finish_reason` → retry (`:315-323`) | 3 attempts, Retry-After (`C/actors/llm.py:99-108,174-193`) | 8 attempts; check stream completion |
| 14 | Blank usage / cost 0 | `cost_tracking: ignore_errors` (`MT/task04/run_agent.sh:18`); it aborted `gpt-6-astra-v3` | `UsageBlank`, masked (`llm.py:39-42,200-201`) | Keep the mask, but meter through the proxy so the run continues |
| 15 | Out of credits (402) | — (it killed `gpt-6-astra-v2`) | `BudgetExceeded`, masked (`llm.py:184-185`) | Keep; check credits before launching a K-batch |
| 16 | Long reasoning turns time out | Request timeout 1200 s (`MT/common/mswea/ext.py:75-90`); LiteLLM default raised (`H/.../mini_swe_agent.py:763-764`) | 120 s (`llm.py:117`) | 1200 s for reasoning models |
| 17 | Crash loses the run | Save the trajectory in `finally` every step (`MS/agents/default.py:120-121`); terminus dumps every episode (`:1650-1653`) | Trace in memory until the end | Persist messages and events every step |
| 18 | Error text echoed by the model misclassified | Classify only from terminal `result`/`error` events (`H/agents/installed/claude_code.py:427-449`); regex last match wins (`H/agents/installed/base.py:223-230`) | n/a | Use for seam S |
| 19 | Secrets reach tools or graders | Session key as capability (`C/connectors/capture_proxy.py:5-8`); secrets on argv (`VF/v1/harnesses/utils/launch.py:24-26`); strip `*API_KEY*` env (`VF/v1/runtimes/subprocess.py:26-29`); aliased provider name (`OE/.../seams.py:336-358`) | Proxy keys | Never set `OPENAI_API_KEY` in the sandbox |
| 20 | Hidden model switching | Pin every Claude tier alias to the model under test (`claude_code.py:1876-1880`); disable non-essential traffic and the autoupdater (`:1887`; `VF/.../claude_code/harness.py:91-101`) | n/a | Use for seam S |
| 21 | No network in the sandbox | Regex guard on commands for pods that cannot be cut off (`MT/common/mswea/ext.py:28-31`); proxy-only allowlist (`VF/v1/rollout.py:316`) | `--network none` | Proxy-only lane for seam S |
| 22 | Patch collection fails | `git add … ':(exclude,glob)**/build*/**' … \|\| true` (`MT/task04/run_agent.sh:27`) | fs_git ignores build output | Keep (B.5) |
| 23 | Zombie PID 1 | `docker run --init` (`MT/task04/run_agent.sh:14`) | B.5 | Keep |
| 24 | Proxy saturation | Health check starved at ~200 sessions, crash at 320; capacity < group size → `CAPACITY_REACHED` (`SP/chapters/openenv.mdx:160-161,317-319`) | n/a | Load-test the proxy; capacity ≥ K × concurrency |
| 25 | Agent crash takes down training | Failures are values (`reward=None`), not exceptions (`SP/chapters/openenv.mdx:213-217`; `MH/README.md:80`) | Masked failures | Keep |

---

## 6. Mechanism → Crucible table

| Mechanism | Source path:line | Use in Crucible | DEPEND/PORT/PATTERN |
|---|---|---|---|
| Agent loop: limits pre-checked, format-error streak, save every step | `MS/agents/default.py:88-157` | harness: `BashLoopPolicy` | PORT |
| Single bash tool + parser + padding of unexecuted actions | `MS/models/utils/actions_toolcall.py:11-113` | harness | PORT |
| Submission sentinel (first line, rc 0) | `MS/environments/local.py:45-56` | harness; M5 (submit = terminal step) | PORT |
| Process-group kill on timeout, partial output kept | `MS/environments/local.py:72-92` | harness (local exec) | PATTERN |
| `finish_reason`-aware format-error and head/tail observation templates | `MS/config/mini.yaml:112-149` | harness | PORT |
| Abort-not-retry exception list; cost guard | `MS/models/litellm_model.py:50-57,108-126` | `LLMClient` | PATTERN |
| Global cost and call caps | `MS/models/__init__.py:13-42` | M10 `budget.py` | PATTERN |
| Auto `cache_control` for Anthropic | `MS/models/__init__.py:55-60`; `cache_control.py:49-…` | harness cost | PORT |
| mini-swe-agent → ATIF | `H/agents/installed/mini_swe_agent.py:256-435` | M3 import, harness export | DEPEND (already, `C/interop/harbor/atif_import.py:104-107`) |
| SSH/`docker exec` env, rc 124 → timeout, network guard | `MT/common/mswea/ext.py:28-72` | B.5 lessons; frontier re-runs | PATTERN |
| Long request timeout | `MT/common/mswea/ext.py:75-100` | `LLMClient.timeout_s` | PORT |
| Installed agent base: install, typed errors, last-match classification | `H/agents/installed/base.py:34-160,223-230,912-948` | seam S; M10 `infra.py` | DEPEND |
| ATIF prefix seeding (`load_trajectory` → native session) | `H/agents/installed/base.py:1109-1213`; `H/trial/trial.py:1195-1235` | C3 on installed agents | DEPEND |
| Claude Code options, env pinning, resume/load, stream-json cost | `H/agents/installed/claude_code.py:53-128,1001-1034,1862-1960` | seam S calibration and C3 | DEPEND |
| Codex exec flags, config.toml base URL, resume | `H/agents/installed/codex.py:96-100,1572-1620` | seam S | DEPEND |
| OpenCode run, provider `baseURL` | `H/agents/installed/opencode.py:72-78,546-553,578-634` | seam S | DEPEND |
| terminus-2 completion double-confirm | `H/agents/terminus_2/terminus_2.py:626-647,1529-1545` | harness option for expensive verifiers | PATTERN |
| terminus-2 three-step summarization and unwind | `H/agents/terminus_2/terminus_2.py:693-717,865-960` | harness compaction (alternative) | PATTERN |
| ATIF `is_copied_context`, token ids and logprobs | `H/models/trajectories/step.py:64-73`; `metrics.py:27-38` | C3 prefixes, compaction, RL export | DEPEND |
| Agent-phase allowlist (`extra_allowed_hosts`) + Docker egress sidecar | `H/trial/network_policy.py:23-42`; `H/models/trial/config.py:125-131`; `H/environments/docker/docker.py:210-217` | seam S network lane | DEPEND |
| verifiers chat loop: text-only end, JSON-repair messages, stream-finish check, idempotent retry | `VF/v1/harnesses/utils/core.py:160-171,255-323,357-439` | harness | PORT |
| verifiers compaction (window − 16384, `tool_choice=none`, 3 attempts, framing) | `VF/v1/harnesses/utils/compaction.py:11-38,155-256` | harness | PORT |
| Interception routes (Responses, count_tokens), refuse-before-serve limits | `VF/v1/interception/server.py:423-437,583-902`; `VF/v1/session.py:83-115` | capture proxy upgrade | PATTERN |
| Token graph with bridged prefixes and mask | `VF/v1/graph.py:1-18,83-166`; `VF/v1/interception/clients/train.py:322-473` | RL data (M12) | PATTERN (DEPEND in the verifiers env) |
| Proxy-side token-id injection, `top_p=1` | `OE/src/openenv/core/harness/capture/upstream.py:14-33,139-148` | capture proxy, RL | PATTERN |
| Longest-token-prefix turn graph | `OE/src/openenv/core/harness/capture/graph.py:1-35,299-325,443-484` | RL data; seam S traces | PATTERN |
| Per-harness env seams (Codex Responses/WebSocket, OpenCode alias) | `OE/src/openenv/harbor/seams.py:336-470` | seam S wiring | PATTERN |
| One harness per task / GRPO group | `MH/envs/harbor/smoldataenv_harbor/environment.py:9,66,74` | RL grouping; calibration design | PATTERN |
| Capture vs ATIF consistency check | `SP/chapters/openenv.mdx:290-303` | seam S mask `crucible:capture_mismatch` | PATTERN |
| Keep/revert experiment loop; log outside context | `AR/program.md:66-112` | coding prompts; M5 step boundaries | PATTERN |
| World choke point, bound, fault labels | `C/world/world.py:295-355` | all seam W harnesses; World MCP facade | (ours) |
| WorldRef snapshot/restore/fingerprint | `C/world/world.py:536-587`; `C/core/schema/state.py:90-104` | C3 branches | (ours) |
| Capture proxy (session keys, SSE assembly, retry digest) | `C/connectors/capture_proxy.py:44-141` | seam S; add budget, routes, snapshot hook | (ours) |

---

## 7. Open questions and risks

1. **V̂ depends on the policy.** C3 under π_ref estimates V^{π_ref}(s_t), not the value under the policy that produced τ. For the gate's corr(C3, ΔΦ), fix π_ref (proposal: GPT 5.6 sol in `BashLoopPolicy(mswea)`, ≈ $0.03 per call) and report it. Open: does ΔΦ correlate across policies?
2. **Branch fidelity.**
   - Background processes and gitignored build trees are not in `WorldRef` (`C/actors/policy.py:203`; `C/world/state/fs.py:1-5`).
   - Branches restored from a non-quiescent snapshot diverge silently. Mitigations: `branchable` flag, fingerprint check after restore, rebuild cost budgeted.
   - Open: should `fs_git` optionally snapshot build dirs (fast restore, larger store) for C3 only?
3. **Replaying recorded trajectories into `WorldRef_t`.** The task04 runs were not executed in a Crucible World, so states come from M5 replay. Re-running builds and benchmarks is slow, and GPU timing is nondeterministic. Return-code mismatches must quarantine steps, or V̂ is computed from a different state.
4. **Mismatched prefixes are off-distribution.** A mini-swe-agent prefix fed to Claude Code (or the reverse) is not the Claude Code policy. Only same-harness prefixes are valid. Seeded Claude Code sessions must live under `projects/-app`, so the cwd must be `/app` (`H/agents/installed/claude_code.py:192-200`). OpenCode cannot load. Codex needs the Responses route and has the WebSocket fallback issue (`OE/.../seams.py:396-401`).
5. **Snapshot-on-request for seam S** assumes the agent is idle while it waits on the model. Claude Code's concurrent background and subagent calls, and any process the agent left running, break this. Main-chain detection must be validated against the ATIF before any seam-S C3.
6. **Which side of the line are refusals and caps on?**
   - Is `AgentSafetyRefusalError` a policy outcome (scored 0) or a mask? Proposal: scored, but flagged and excluded from C3 statistics.
   - Is hitting an episode `cost_usd` a fair failure in calibration? Proposal: yes, reported as truncated, with the cap fixed per task family.
7. **Text-only termination vs an explicit submit.** verifiers' built-in loops and our `LLMToolPolicy` end on a text-only reply. mini-swe-agent requires a sentinel. RL-trained open models will learn whichever convention training uses, so it must be fixed per `HarnessId`, and calibration must use the same one. Not measured in our sources: how often frontier models end prematurely under text-only termination.
8. **Tool timeouts and masking.** `tool_timeout_s` (3600) equals the shell's `max_timeout_s` (3600), so a policy's long command can be mislabelled as infra. Fix: clamp. A policy could in principle drive masking (e.g. a hanging App). Watch the masked fraction per policy and alert when it is above 5%.
9. **Proxy scale and security.**
   - The OpenEnv proxy fell over at about 200–320 sessions.
   - The proxy host is reachable from the sandbox in seam S, so it must accept only registered keys (it does) and must never expose admin or weight routes (the FineEnvs inference proxy allows only `GET models` and `POST chat/completions`, `MH/jobs/inference_proxy.py:33-37`).
10. **The agentenv MCP path bypasses `World._call_tool`** (`C/interop/agentenv/world_backend.py:186-192`). Any harness wired to `mcp_url` today produces no World events. Build the World MCP facade before using MCP-only harnesses (Claude Code with built-ins disabled) as a seam-W variant.
11. **Dependency conflicts.** verifiers v1 pins `mcp==2.0.0`, while agentenv needs `mcp<2` (`crucible/ARCHITECTURE.md:42`). The verifiers harness and interception can therefore run only in the separate environment, with traces exchanged as files.
12. **Cost at frontier scale.** A single long task04 run reached $39.75 with a 329k-token prompt. Without compaction and prompt caching, K × |T| branches on frontier models are unaffordable. With compaction, the summary quality becomes part of the policy, and its effect on V̂ is untested.
