<!-- M1 taxonomy 02: OpenEnv, FineEnvs and the FineEnvs article Spaces. Read-only study by a subagent, 2026-10-12. -->

# Env-family catalog: OpenEnv, FineEnvs, and three FineEnvs article Spaces

ROOT = `RL-Environments/06-frameworks-and-env-code`. All paths below are relative to ROOT. Snapshots: OpenEnv at commit 49aa302 (2026-10-02), FineEnvs at 5e0b46e (2026-10-08).

**Licenses:**
- **OpenEnv:** BSD-3-Clause (`meta-pytorch-openenv/LICENSE`). Every `envs/*/pyproject.toml` header says BSD-style.
- **Wrapped upstream libraries keep their own licenses.** The only one vendored in the repo is the MIT CARLA `agents` code (`envs/carla_env/server/carla_agents/LICENSE`). The licenses I give for other upstream libraries are from general knowledge; I did not check them in the repo.
- **FineEnvs:** Apache-2.0 (`adithya-s-k-fineenvs/LICENSE`). Some files carry an SPDX BSD-3 header, e.g. `02-watercolour/envs/watercolour/core/quality.py:1`.
- **The 3 hf-space-* repos:** Astro article sites under a CC-BY-4.0 template license ("Copyright (c) 2024 Thibaud Frere"). They contain **no environment code**. Their value is the definitions (harness, white-box vs black-box) and the reward-architecture taxonomy, summarised further down.

Abbreviations: ENV = `meta-pytorch-openenv/envs`, SRC = `meta-pytorch-openenv/src/openenv`, FE = `adithya-s-k-fineenvs`.

Unless a family says otherwise, its infrastructure is: a FastAPI server built with `create_app`, a Docker image `server/Dockerfile` on `ghcr.io/huggingface/openenv-base`, and a plain manifest `openenv.yaml` (spec_version 1, type space, runtime fastapi, app, port).

---

## Part A. OpenEnv environment families (41 env dirs; every one is listed)

### A1. Games and classic discrete RL

**1. atari_env** (ALE / ale-py; upstream license unverified)
1. Any ALE ROM; the game is picked per action with `game_name` (`ENV/atari_env/models.py:28-31`).
2. State: an emulator.
3. Actions: typed `AtariAction{action_id, game_name, obs_type, full_action_space}` sent over WebSocket/HTTP.
4. Actors: single agent.
5. Time: frameskip=4 and sticky actions (`atari_environment.py:73-94`). The episode ends on `ale.game_over()` (:233). No step cap.
6. Observations: **pixels** as a flattened `screen: List[int]` plus `screen_shape`, or grayscale, or RAM; also `legal_actions` and `lives` (`models.py:51-56`).
7. Reward: native per-frame reward summed over the skipped frames (`atari_environment.py:184-194`).
8. Dense.
9. CPU only.
10. Isolation: none.
11. Pixels as int lists are inefficient; no image encoding.

**2. chess_env** (python-chess; upstream license unverified)
1. Chess.
2. State: board as FEN.
3. Actions: `ChessAction{move: str}` in UCI (`models.py:31`).
4. Actors: agent against a built-in opponent, either moonfish engine at depth N, random, or `None` = **self-play where one client moves both colours** (`chess_environment.py:38-56`). The opponent moves inside `step()` (:150-153).
5. Time: turn-based, with a `max_moves` cap. Ends on checkmate, stalemate, 50-move rule, threefold repetition or insufficient material (:199-223).
6. Observations: FEN, legal moves, check flag, result (`models.py:47-50`).
7. Reward: terminal +1/-1/0, and -0.1 for an illegal move (:133-137). At game end it also returns `metadata["discounted_rewards"]` per agent move (:183-186, 231).
8. Sparse terminal reward, plus a trajectory-credit rubric `ChessWinLossRubric` (`server/rubrics.py:50`).
9. CPU only.
11. Port: `ENV/chess_env/server/rubrics.py`, a terminal reward turned into per-move discounted credit (BSD).

**3. connect4_env**
1. Connect-4 on a 6x7 board.
2. State: numpy board.
3. Actions: `Connect4Action{column}`.
4. Actors: **two-player hot-seat.** `next_player` flips each step (`connect4_environment.py:59`). An opponent hook exists (:21) but the app passes none (`server/app.py:19-20`), so one client plays both seats.
5. Time: turn-based. Win, draw or invalid move ends the episode.
6. Observations: board plus legal actions; `metadata.next_player`.
7. Reward: win/draw from `_check_win_or_draw`; an invalid move gives -1 and ends the game (:47-48).
8. Sparse.

**4. grid_world_env**
1. Toy grid.
2-4. Actions: `MoveAction` enum; single agent.
5. The goal ends the episode.
6. Observations: x, y and a message.
7. Reward: -0.1 per step, +1 at the goal (`grid_world_environment.py:86-92`).
8. Dense shaped.

**5. maze_env**
1. Maze.
2-3. Actions: integer.
4. Single agent.
5. Status is WIN or LOSE (`maze_env_environment.py:151-153`).
6. Observations: legal actions, current and previous position.
7. Reward comes from the internal maze module (`server/maze.py`).
8. Per-step.

**6. snake_env** (wraps marlenv Snake-v1)
- Multi-agent capable, but instantiated with `num_snakes=1` (`snake_environment.py:174-181`).
- Actions: integer.
- Observations: `grid` plus an **HxWxC tensor** `observation` (`models.py:71-72`).
- Reward: configurable `reward_dict` with defaults fruit 1, lose -1, win 100, time 0.001 (:160-167). Dense.

**7. openspiel_env** (OpenSpiel; upstream license unverified)
1. Any OpenSpiel game, chosen by env var `OPENSPIEL_GAME` (`server/app.py:47`). The examples cover catch, tic_tac_toe, kuhn_poker, cliff_walking, 2048 and blackjack (`examples/openspiel_all_games.py`).
2. State: game-engine state.
3. Actions: `OpenSpielAction{action_id, game_name, game_params}`.
4. Actors: **N-player.** The agent holds seat `agent_player`. The others are server-side `OpponentPolicy` objects, random or fixed (`opponent_policies.py:18-78`), auto-played until it is the agent's turn (`openspiel_environment.py:202-234`). Simultaneous-move games only allow agent seat 0 (:170-186). Chance nodes are handled by the engine.
5. Time: turn-based, ending at `time_step.last()`.
6. Observations: `info_state` vector, `legal_actions`, `current_player_id`, `opponent_last_action` (`models.py:53-57`).
7. Reward: the agent seat's reward, `time_step.rewards[agent_player]` (:259-262).
8. Mostly terminal; game-dependent.
9. Image built from `Dockerfile.openspiel-base`.
10. Has a white-box harness, `openspiel_env/harness.py:125` (`OpenSpielSessionFactory`).
11. Port: `opponent_policies.py` plus `_auto_play_opponents`, the "seat plus scripted opponents" pattern.

**8. textarena_env** (TextArena; upstream license unverified)
1. Any `ta.make(env_id)` game; default Wordle-v0. Env vars `TEXTARENA_ENV_ID`, `TEXTARENA_NUM_PLAYERS`, `TEXTARENA_MAX_TURNS` (`server/app.py:41-48`).
2. State: text-game state.
3. Actions: `TextArenaAction{message: str}`.
4. Actors: `num_players` ≥1. The observation exposes `current_player_id` and `legal_players` (`models.py:39-43`). One client speaks for whichever player is current (hot-seat). Multi-agent therefore means hot-seat, not separate clients.
5. Time: turn-based; game `done`, optional `max_turns`.
6. Observations: prompt, message log `[{sender_id, content, category}]`.
7. Reward: the current player's reward from `ta_env.state.rewards` (`environment.py:355-361`), plus **multi-component `reward_signals`** in metadata. For Wordle these are greens, yellows, repetitions and correct (`rewards.py:78-95`).
8. Wordle signals are per-step.
9. Up to 8 concurrent sessions by default.
11. Port: the `RewardProvider` protocol in `ENV/textarena_env/rewards.py:11-28`, auxiliary named reward channels next to the scalar.

### A2. Physics, simulators and control

**9. dm_control_env** (DeepMind Control Suite and MuJoCo; upstream licenses unverified)
1. All suite domain/task pairs. You can switch per episode with `reset(domain_name, task_name)` (`dm_control_environment.py:261-285`).
2. State: **MuJoCo physics.**
3. Actions: **continuous** `values: List[float]` (`models.py:43`).
4. Single agent.
5. Time: simulator-owned clock; `physics_timestep` and `control_timestep` are in State (`models.py:113-117`). Ends at `StepType.LAST`, the suite's time limit (:251).
6. Observations: dict of float vectors, plus optional **base64 PNG `pixels`** when `reset(render=True)` (:234-247). Uses EGL headless rendering (:21-24).
7. Reward: native suite reward each step (:252).
8. Dense.
9. Docker with EGL/OSMesa.
11. Port: the action/observation spec export (`_get_action_spec_info`, :200-221), which gives a schema for continuous control.

**10. unity_env** (Unity ML-Agents; upstream license unverified)
1. Every `mlagents_envs.default_registry` env (3DBall, PushBlock, ...); switch with `reset(env_id)` (`unity_environment.py:197-276`).
2. State: **game-engine binaries** downloaded and cached in `~/.mlagents-cache` (:56).
3. Actions: `discrete_actions` and/or `continuous_actions` (`models.py:53-57`).
4. Single behaviour, first agent only (:323-328).
5. Time: engine time with a `time_scale` multiplier (:166); ends on terminal steps.
6. Observations: vector observations, optional **base64 visual observations**, spec info.
7-8. Engine reward per step; dense.
9. `no_graphics` headless option (:159-164). A GPU is optional.

**11. carla_env** (CARLA 0.9/0.10; vendored agents are MIT)
1. Benchmark scenarios: trolley_micro (ethical dilemmas), action_bias, maze, free_roam (`server/benchmark_scenarios/*`). Configured by `ScenarioConfig{max_steps=50, weather, camera...}` (`base.py:22-38`).
2. State: **3D driving simulator.** Two modes: "real", a client to an external CARLA server, and "mock", simulated physics with no CARLA (`carla_environment.py:12, 229-273`).
3. Actions: `CarlaAction{action_type ∈ control | emergency_stop | lane_change | observe | maintain_speed | brake_vehicle | init_navigation_agent | set_destination | follow_route | capture_image, throttle, steer, brake, ...}` (`models.py:34-95`). This mixes continuous control with high-level tool-like macros.
4. Actors: single ego vehicle. NPC walkers and vehicles are spawned by the scenario (`trolley_micro.py:210`).
5. Time: **simulator-owned synchronous clock**, `fixed_delta_seconds=0.05` with `world.tick()` (:735-742). A scenario hook `ticks_after_tool()` decides how many ticks each action advances, and `idle_ticks` lets inaction be modelled (`base.py:31-33, 94`). `simulation_time` is in the observation.
6. Observations: text `scene_description`, speed, pose, goal distance, `nearby_actors`, collision info, optional **JPEG `camera_image`** (`models.py:110-166`).
7. Reward: per-step scenario reward. Free-roam is progress + arrival bonus 10 − collision 5 − time 0.01; maze is ±1. `CarlaNavigationRubric` (`rubrics.py:63-87`) and trolley `CarlaTrolleyRubric` with discounted trajectory credit (`rubrics.py:27-45`). Exposed as `rubric_reward` (:454-455).
8. Dense for navigation, terminal for trolley.
9. `Dockerfile` uses nvidia/cuda and needs a T4/A10G GPU. `Dockerfile.real` is a slim client for an external CARLA host.
11. Port: `ENV/carla_env/server/benchmark_scenarios/base.py`, a scenario ABC with `ticks_after_tool` (tool-call → sim-time policy) and `is_done`/`compute_outcome`. Maps directly onto a sim-mode Clock.

**12. sumo_rl_env** (SUMO / sumo-rl; upstream license unverified)
1. Traffic-signal control on a `.net.xml` network.
2. State: **SUMO microsimulator.**
3. Actions: `SumoAction{phase_id, ts_id}`.
4. Actors: single agent (`single_agent=True`, :126-133). sumo-rl itself is multi-agent capable.
5. Time: **sim seconds.** `num_seconds=20000`, `delta_time=5` s between decisions, yellow/min/max green (`models.py:95-100`); `sim_time` in the observation.
6. Observations: vector, `action_mask`, `sim_time`.
7. Reward: configurable `reward_fn`, default "diff-waiting-time" (:106). Dense.

**13. wildfire_env** (custom; BSD)
1. Fire containment on a grid.
2. State: a cellular-automaton fire model with wind and humidity (`wildfire_environment.py:40-50`).
3. Actions: `{action: break | water | wait, x, y}` with limited water and break resources (`models.py:11-20, 35-36`).
4. Single agent.
5. Time: tick-based, `max_steps=128`; the fire spreads every step.
6. Observations: flattened grid, wind direction, humidity, counts, remaining resources, `reward_hint`.
7. Reward: shaped each step. −0.15·spread, +0.10·shrink, −0.05·new burned, −0.01 time, −0.05 invalid action. Terminal containment bonus 0.5+0.5·saved plus 0.2·(1−burned) (:150-215).
8. Dense shaped.
11. Port: the whole file (`wildfire_environment.py`, ~300 LoC), a tiny exemplar of an env-owned stochastic world that advances regardless of the agent.

**14. finrl_env** (FinRL; upstream license unverified)
- Wraps any FinRL env class, e.g. StockTradingEnv, using a config with a price DataFrame (`finrl_environment.py:100-123`).
- Actions: continuous `actions: list[float]`, one per stock (`models.py:34`).
- Time: simulated trading days.
- Observations: state vector, portfolio value, date.
- Reward: per-day FinRL reward, scaled. Dense.

### A3. Browsers and GUI

**15. browsergym_env** (BrowserGym and Playwright; upstream license unverified)
1. Benchmarks miniwob, webarena, visualwebarena, workarena; `task_name` can change per reset (`browsergym_environment.py:122-252`). The Dockerfile bakes in MiniWoB++ served on :8888 (`server/Dockerfile:31-47`).
2. State: **real Chromium** plus benchmark web apps. WebArena needs its external site containers.
3. Actions: `BrowserGymAction{action_str}`, BrowserGym's action DSL such as `click('bid')`.
4. Single agent.
5. Time: `max_steps`, `terminated or truncated`.
6. Observations: text, URL, goal, **axtree_txt**, pruned HTML, optional **screenshot as an int array** (`models.py:40-66`).
7. Reward: the benchmark's task validator, per step. `cum_reward` is in State.
8. Mostly sparse.
9. Image `mcr.microsoft.com/playwright/python`.
11. Has a harness `browsergym_env/harness.py:173` that maps tool calls to action strings (:78).

**16. helium_browser_env**
1. Open-web visual browsing; the start URL is given at `reset(start_url)` (:76).
2. State: Chromium on an **Xvfb virtual display** at 801x601 (`server/desktop.py:10-30`), driven by Helium and Selenium.
3. Actions: `BrowserAction{op: click | type | key | scroll | back | wait | finish, x, y, text, dy}` with pixel coordinates in an 800x600 viewport (`models.py:9-14`).
4. Single agent.
5. Time: **real wall-clock waits after each action**, `BROWSER_ACTION_WAIT_SECONDS=1.5`, wait=3 s (:20-31); `BROWSER_MAX_STEPS=20` (:58-60).
6. Observations: **base64 screenshot**, URL, `controls`, `page_text`.
7. Reward: **none; always 0** (:186). This is a pure tool env for external grading.
9. Refuses to run outside an HF sandbox (`BROWSER_SANDBOX=hf`, :51). Launched via `HFSandboxProvider` (`hf_sandbox.py:28`). `REQUIRES_SINGLE_THREAD_EXECUTOR=True` (:47).
10. Detects Cloudflare/Vercel bot challenges and ends the episode (:34-46).

**17. openapp_env** (Meta OpenApps and BrowserGym)
1. Simulated web apps: calendar, todo, messenger, maps (`openapp_environment.py:10-12`). Optional `task_name`.
2. State: an external **OpenApps web server** that must be started manually (:221-247); `app_state` is in the observation.
3. Actions: `OpenAppAction{action_type: click | fill | select_option | goto | scroll | send_keys | ..., bid, text, value, url, direction}`.
4. Single agent.
5. Time: `max_steps` (:135).
6. Observations: HTML, URL, open pages, optional screenshot, axtree, `app_state`, `task_info`.
7. Reward: BrowserGym task validation. The default `GenericOpenAppsTask` never completes (:110-125), and errors cost −0.1.
8. Mostly zero.

### A4. Code execution and software sandboxes

**18. coding_env**
1. Free-form Python CodeAct.
2. State: an in-process `smolagents.LocalPythonExecutor` with an import allowlist (`server/python_executor.py:29-50`).
3. Actions: `CodeAction{code}`.
4. Single agent.
5. No episode end.
6. Observations: stdout, stderr, exit code.
7. Reward comes from **Transforms**, not a grader: a safety penalty of −1 for regexes like `import subprocess`, `eval(`, `exec(`, plus a conciseness/syntax quality score (`server/transforms.py:15-93`).
8. Per-step shaping only.
10. Isolation: interpreter-level only.

**19. coding_tools_env** ("SETA-style", E2B)
1. Task = instruction + setup commands + verify commands.
2. State: a **fresh E2B sandbox per episode** (`coding_tools_env_environment.py:219-252`).
3. Actions: MCP tools bash, read, write, edit, multi_edit, glob, grep, ls, todo_write, submit_solution (:43-216).
4. Single agent.
5. Ends at submit.
6. Observations: text.
7. Reward: verify commands' pass fraction (`passed/total`), unless a verify command writes `/home/user/logs/verifier/reward.txt` as an override (:28, 375-403, 414-431).
8. Terminal.
9. Needs `E2B_API_KEY`.
10. The reward file is **deleted before verification**, so an agent-written file is ignored (`_CLEAR_REWARD_FILE`, :29, 379-381).
11. Port: the reward.txt-override-with-pre-clear pattern (BSD).

**20. julia_env**
- Julia code plus tests: `JuliaAction{core_code, test_code}`.
- Runs on a persistent Julia process pool (`julia_process_pool.py`).
- Observations: `tests_passed` and `tests_failed`.
- Reward per step: −1 if it does not compile, 0 if no tests ran, otherwise the pass rate, with **1.5 for a perfect pass** (`julia_codeact_env.py:450-469`). −1 on an infinite-loop timeout (:234). A safety transform subtracts 3 for dangerous calls (`julia_transforms.py:32-72`).

**21. jupyter_env** (E2B Code Interpreter)
- MCP tools: add_and_execute_code_cell, edit_and_execute_current_cell, execute_shell_command, get_notebook_state(include_images), final_answer (`jupyter_environment.py:70-178`).
- State: a notebook, with cells holding **base64 PNG images** (`models.py:9-20`).
- Reward: the same verify-commands + reward.txt scheme as #19 (:32); `done` is set after final_answer (:353-355).

**22. repl_env** (Recursive-LM style)
- Python REPL with a large `context` loaded as a variable.
- Actions: `REPLAction{code, is_final, final_answer}`.
- Finishing: `FINAL(...)` or `FINAL_VAR(...)`, or `max_iterations=30` (`repl_environment.py:68-127`).
- Optional **recursive LLM calls** inside the sandbox via `llm_query()` and `llm_query_batched()` against HF inference (:135-209).
- Reward is a `REPLRubric`: an outcome rubric on the terminal step (ExactMatch, Fuzzy or CustomMetric against `expected_answer` passed at reset), plus a per-step process rubric (`CodeExecutionRubric`, −0.05 on error) (`repl_env/rubrics.py:26-187`).
- Port: the REPLRubric outcome+process composition.

**23. git_env**
- A shared **Gitea** service plus per-task workspaces. Fast reset to a pinned (repo, commit) through `task_repos` (`git_task_environment.py:78-145`).
- Actions: `GitAction{action_type: list_repos | clone_repo | exec ..., command, working_dir}`.
- **No reward.** This is a substrate env.
- Port idea: commit-pinned reset ("fast! just git reset").

**24. terminus_env** (E2B)
- A single MCP tool `terminal(command, final_answer)` (`terminus_env_environment.py:43-65`).
- Reward: verify commands + reward.txt override, as in #19.

**25. tbench2_env** (Terminal-Bench 2)
1. Downloads the TB2 repo zip at runtime (`tbench2_env_environment.py:97-103`). Default task "headless-terminal".
2. State: a **local terminal toolkit** (the server and agent share one container) or **Docker mode**, one container per task (`Tbench2DockerEnvironment`, :669).
3. Actions: `Tbench2Action{action_type ∈ exec | write | view | wait | kill | write_file | evaluate | close, command, session_id, block, wait_seconds, ...}` (`models.py:26-32`; :409-448). Supports non-blocking sessions and **real-time waits**.
5. Time: `TB2_COMMAND_TIMEOUT_S`; the task timeout comes from task.toml (:156).
7. Reward: the official `tests/test.sh` writes `/logs/verifier/reward.txt`, which is echoed through a marker `__TB2_REWARD__:` (:171-233). If the verdict is missing, the reward is an error, not 0 (:226-233).
8. Terminal.
10. `TB2_WITHHOLD_TESTS=1` moves `tests/` and `solution/` out of the agent's reach at reset (:278-284, 523-550). Docker mode excludes them from the copied task dir (:844-848).
11. Port: the `_withhold_verifier_assets` pattern for hidden verifiers on a shared filesystem.

**26. kernrl** (GPU kernel optimisation)
1. **89 problems across 10 levels** (`ENV/kernrl/problems/level1..level10`), from basic ops to DeepSeek MLA, MoE, FP8 and GatedDeltaNet.
2. State: **a real GPU.** `LocalGPUEvaluator` compiles, checks correctness against a PyTorch reference (atol/rtol), and benchmarks, with nsys, ncu and compute-sanitizer profiling (`server/evaluator.py:201-310`).
3. Actions: `KernelAction{code}`.
4. Single agent.
5. Time: `max_turns=10`. Ends early when solved, i.e. correct and speedup > 1.05 (`kernrl_environment.py:306-318`).
6. Observations: reference code, GPU info, feedback (profiler summaries), compile errors, `max_diff`, `speedup`.
7. Reward per turn: −0.25 if incorrect. If speedup > 1, `min(speedup − 1, 2)`; otherwise `(speedup − 1)·0.5` (:342-356).
8. Dense per turn, iterative refinement.
9. GPU required.
11. Port: `server/evaluator.py` + `profiler.py`, a GPU correctness+speed harness emitting agent-readable profiler feedback.

### A5. Agent-harness-in-sandbox ("black-box") families

**27. opencode_env**
- A single MCP tool `run_rollout(instruction, setup, verify, model endpoint, top_logprobs, ...)` (`opencode_environment.py:7-19, 117-181`). It runs the **OpenCode CLI agent inside an E2B or HF sandbox** (`opencode_env/sandbox/{e2b,hf}.py`).
- A **transparent OpenAI-compatible proxy** on :7000 captures every request/response/logprob (`sandbox/interception.py:7-30`).
- Reward: verify pass-fraction or reward.txt override (:377-401).
- Returns a `RolloutResult` with `proxy_turns` carrying token ids and logps (`models.py:23-82`).
- One tool call is 30-600 s of wall time.
- Also provides a `ResourceSessionFactory` (`opencode_env/harness.py:84-262`).
- Deprecated in favour of Harbor, per FE 05 README.

**28. pi_env**
- Same design as #27, but runs the "pi" coding-agent CLI (`pi_env/server/pi_environment.py:7-19, 117`).
- Reward file is `/root/logs/verifier/reward.txt`.

**29. harbor_env** (with the framework in `SRC/harbor/*`)
1. Serves **any Harbor task dataset**: an HF dataset of `tasks/<name>/`, a local dir, or `name@version` from the Harbor registry (`SRC/harbor/tasks.py:1-25`). Exposed through the Task API.
2. State: a sandbox per rollout on E2B, Modal or Daytona (`ENV/harbor_env/server/app.py:1-40`).
3. Actions: MCP tools `run_rollout` and `capabilities` (`SRC/harbor/environment.py:1-6, 43`). There is deliberately no step(), because "a Harbor rollout is one stateful 60-600s run that the harness drives". The timeout is raised to 1800 s (:21-23).
4. Actors: a **choice of agent harness per rollout**, e.g. opencode, claude-code, codex, mini-swe-agent, with host-side agents terminus-2, computer-1, oracle, nop and dspy-rlm (`SRC/harbor/capabilities.py:27`; versions pinned in FE `05-multi-harness-rl/envs/harbor/smoldataenv_harbor/server.py:17-20`).
5. Time: the agent's own loop, with `agent_step_limit` and `agent_timeout_sec`.
7. Reward: the task's own verifier, read from the reward key (`OPENENV_HARBOR_REWARD_KEY`).
9. Capture proxy mounted at `/capture` on the same port (app.py docstring). Exports token-level training traces (`SRC/harbor/contract.py:58-183`).
11. Port: `SRC/harbor/contract.py` + `SRC/core/harness/capture/*`, which converts multi-dialect model traffic (Anthropic, OpenAI chat/responses, Google) into one training trace (`capture/dialects/*`).

### A6. Tool-use, API, DB and simulated-user families

**30. calendar_env** (Google Calendar clone, ~27k LoC)
1. A full Calendar API: calendars, events, ACL, freebusy, settings, colors, users (`server/apis/*`). About 40 MCP tools (`server/calendar_mcp/tools/*.py`). One bundled scenario (`calendar_env/scenario_config.json`).
2. State: **SQLite per `database_id`**, `calendar_{db_id}.sqlite` (`server/database/session_manager.py:27-34`). Seed, reset, clone and SQL-runner routes are at `server/apis/database_router.py:54-550`.
3. Actions: `MCPAction{ListToolsAction | ToolCallAction}` (`models.py:25-58`), carrying context headers for the acting user and an access token.
4. Actors: single agent acting as a **named user** in a multi-user seed DB (`scenario_config.json` context).
5. Time: steps; the scenario's `defaultTimeZone` is given.
6. Observations: tool results as JSON.
7. Reward: per-step **heuristic** (+0.1 discovery, −0.5/−1 errors) from `server/openenv_wrapper/mcp_env_environment.py:127-307`. The real grading is **client-side verifiers**: `database_state` (SQL query vs expected value), `response_check` (SQL + LLM comparison), `tool_execution` (`calendar_env/client.py:355-470, 787`).
8. Terminal, multi-check.
11. Port: the scenario JSON schema with `verifiers[{verifier_type, validation_config{query, expected_value, comparison_type}}]` plus `expected_tools` and `restricted_tools`.

**31. agent_world_model_env** (Snowflake AgentWorldModel-1K; README says academic use)
1. **1,000 synthetic MCP tool environments, 10,000 tasks** (`README.md:16`; `server/data_loader.py:20` from HF `Snowflake/AgentWorldModel-1K`).
2. State: each sub-env is a generated **FastAPI app over SQLite**, launched as a subprocess on a random port and patched to mount FastApiMCP (`scenario_manager.py:50-85`). The DB is built from schema + sample data (`db_manager.py:17-104`). An **initial snapshot is saved for verifier diffing** (`save_snapshot`, :106-110).
3. Actions: `ListToolsAction` and `CallToolAction` proxied to the sub-env. The special tools are `verify` (verifier_mode sql or code), `done` and `__list_scenarios__` (`awm_environment.py:28-35, 472-507`). `reset(scenario, task_idx)`.
4. Single agent.
5. Steps.
6. Observations: tool result plus `reward_type` (`models.py:39-92`).
7. Reward by outcome class: complete=1.0, incomplete=0.1, format_error=−1.0 (`server/config.py:31-35`). Two verifier modes: a **code verifier over the initial vs final DB** run in a restricted subprocess (`verifier.py:80-140`; `_verifier_runner.py`), and an **SQL+LLM judge** (`verifier.py:245-275`).
8. Terminal, plus per-step format errors.
9. Up to 10,000 concurrent sessions; idle cleanup.
10. Verifier sandbox uses rlimits and an import allowlist. Scenario code is not sandboxed (README:243).
11. Port: `db_manager.save_snapshot` + `execute_code_verifier(initial_db, final_db)`, i.e. state-diff verification.

**32. finqa_env**
- Financial QA over company tables.
- MCP tools: get_descriptions, get_table_info, `sql_query` (in-memory SQLite, `SELECT *` refused), submit_answer (`finqa_environment.py:65-112`; `server/tools.py:187-259`).
- Time: `max_steps=50` tool calls, then reward 0 (:266-275).
- Reward: tolerant numeric matching with \boxed{}, %, fractions, 1% relative plus absolute tolerance (`server/rewards.py:178-217`). Terminal.

**33. websearch_env**
- `WebSearchAction{query, temp_api_key}`; uses the **Serper.dev API** plus page fetch (`web_search_tool.py:42-209`).
- **No reward.** Tool-only env, external API.

**34. thinkingbox_env** (Microsoft ThinkingBox-Bench)
1. Benchmark test cases ("file.py:test_name"), with pinned data-release provenance (`models.py:708-723`).
2. State: external MCP benchmark servers behind a ThinkingBox **Session Proxy**, plus Typesense. All are externally managed (`README.md:129-135`).
3. Actions: discriminated `CallToolAction` (including **parallel tool calls**), `SubmitMessageAction{content, terminal_tool_calls}`, and an internal `_FinishAction` (`models.py:586-674`).
4. Actors: agent plus an **LLM simulated user** (`UserSimulator`, `can_end_conversation`, `max_user_sim_turns`) (`thinkingbox_environment.py:51, 181-203, 450, 816-832`).
5. Time: user-turn budget.
6. Observations: `kind ∈ reset | tools | tool | tool_batch | user | terminal | error` with user messages (`models.py:737-760`).
7. Reward: **binary pass/fail** from assertions on backend state and side effects plus an **LLM Judge** (`thinkingbox_environment.py:917, 958-967`). `reward_type` is pass, fail or system_error.
8. Terminal.
9. Needs external services and agent/user/judge endpoints. Evaluation only.
10. Hydration, simulated-user context, side effects, assertions and judging all stay inside the trusted server (docstring :1-5).
11. Port: the observation-kind protocol with an explicit system_error that is not scored as fail.

### A7. Single-turn and judge-scored text families

**35. reasoning_gym_env**
- Any reasoning-gym dataset or composite, chosen at `reset(dataset_name, dataset_config | dataset_specs, seed, size)` (`reasoning_gym_environment.py:81-182`).
- Single step; reward = `dataset.score_answer` (:238-250). Partial credit is possible.

**36. qed_math_env**
- Proof mode: MCP tools get_problem, submit_proof, get_grading_guidelines.
- Reward: an **LLM-judge rubric scoring 0-7, normalised /7**, with optional thresholding (`openenv.yaml` rubric; `server/rubric.py:112-158, 392`).
- Answer mode: a process-pool `math_verify_service` with per-status reward presets (`qed_math_environment.py:75-117`).
- A length penalty uses `output_length_tokens` injected by the trainer, not exposed to the agent (`openenv.yaml` tools comment).
- Per-call verifier metrics.

**37. dipg_safety_env** (medical-safety reasoning)
- Single step, `DIPGAction{llm_response}`, with required analysis/proof/final channel formatting (`dipg_environment.py:48-90`).
- Reward: about **17 named reward/penalty components**, including hallucination, proof inconsistency, abstention, conflict and format (:26-78); terminal (:164-178).

**38. sophistry_bench_sprint_env**
- Advocacy on QuALITY passages: the policy defends an assigned answer.
- Reward: a weighted sum over **8 components**, among them aggregate_reward, the *hidden* correctness_reward, claim and citation counts, and four **anti-gaming canaries** (`sophistry_bench_sprint_environment.py:44-56, 253-285`).
- correctness_reward is deleted from the surfaced observation, so a harness cannot leak it (:276-279). Weights come from env vars (:105).

**39. pelican_svg_env**
- "Blind" SVG drawing: animal × vehicle grid (`server/tasks.py:81,120`).
- Reward rubric: `Sequential(Gate(GatePassed), WeightedSum([Structure 0.35, Semantic 0.65]))` (`server/rubric.py:46-80`).
  - Gate: anti-gaming rules such as no embedded raster, no text labels, no external refs (`openenv.yaml` `gate_violations`).
  - Structure: geometry of wheels and rider.
  - Semantic: a **vision judge** (Qwen2.5-VL) doing a blind caption plus a checklist; `judged=false` when no token is set.
- Optional rendered PNG in the observation. Single step.
- Port: Gate→WeightedSum + `judged` flag (BSD).

**40. chat_env**
- Token-level chat: `ChatAction{tokens: list[int]}`. The env owns the tokenizer and both message and token histories (`chat_environment.py:37-120`; `models.py:36-78`).
- Reward comes only from an optional transform.

**41. echo_env**
- The reference MCP env, with tools echo_message and echo_with_length. Reward 0.
- The only env with a full `validation:` manifest block (`ENV/echo_env/openenv.yaml`).

---

## Part B. FineEnvs (Apache-2.0) environment families

### B0. The "one env, six frameworks" matrix (FE/00-environments-101)
- Each env keeps its domain logic in `core/` and wraps it six ways: `openenv/`, `ors/`, `nemo_gym/`, `verifiers/`, `skyrl_gym/`, `gem/` (`FE/00-environments-101/README.md` Layout and matrix).
- Reward placement differs by framework for the **same env**. The README cheat sheet lists OpenEnv as "external", ORS as per-tool-call `ToolOutput.reward`, NeMo Gym as post-episode `/verify`, Verifiers as `Rubric`, and SkyRL/GEM as returned from `step()`.
- Example: the Jupyter OpenEnv server returns `reward=None` (`envs/jupyter/openenv/server/jupyter_environment.py:232,250`), while the ORS version computes a reward per call (`envs/jupyter/ors/server.py:230-247`).

**B1. Wordle** (`FE/00-environments-101/envs/wordle`)
- Pure-Python game state (`core/game.py:34`).
- Tools: `guess` (all variants); the OpenEnv variant adds get_history and **reset_game** (`openenv/server/wordle_environment.py:55-95`). This is an agent-visible game reset, inside the episode.
- 6 guesses.
- Reward: win = 1 + 0.5·(1 − guesses/6). Loss = 0.1 × best green count (partial credit) (`core/game.py:107-122`).
- Reported once on the ending step and never repeated (`wordle_environment.py:160-166`; ORS `_rewarded_game` at `ors/server.py:121-131`).

**B2. Jupyter agent** (`FE/00-environments-101/envs/jupyter`)
- **46 tasks** with `expected_output` (`core/tasks.py`). E2B sandbox; 4 tools.
- Reward (ORS): 1.0 if the expected substring appears, + up to 0.2 efficiency, − 0.05 per error (`ors/server.py:230-247`).
- NeMo variant: `/verify` (`nemo_gym/server.py:199-239`).

**B3. Desktop computer-use** (`FE/00-environments-101/envs/desktop`)
1. 3 tasks: Firefox, terminal, LibreOffice Calc (`core/tasks.py`).
2. State: an **E2B Desktop VM** (`core/desktop.py:81-330`).
3. Actions: **19 tools mirroring Anthropic `computer_20251124`**: screenshot, clicks, drag, mouse down/up, scroll, type, key, hold_key, wait, terminate, run_command, cursor_position, get_screen_size. Pixel coordinates (`openenv/server/desktop_environment.py:1-41`).
6. Observations: **MCP image blocks** (PNG).
7. Reward: **self-reported `terminate(status)`, 1/0** (:526-548; ORS :260-261, 489). There is no environment-side check against `expected_output`, so it is gameable as written.
9. `wait`/`hold_key` take real time.

**B4. LaTeX OCR** (`FE/01-latex-ocr/envs/latex_ocr`)
- Single step, dataset `unsloth/LaTeX_OCR` served through the Task API (`server/latex_ocr_environment.py:54-60, 176-218`).
- **Image observation** (base64 PNG).
- Reward: `[(1−w)(1−CER) + w·exact] × length_factor` with w=0.4. The length guard stops padding hacks (`server/rubric.py:106-185`; `openenv.yaml`).
- Hidden ground truth.

**B5. Watercolour** (`FE/02-watercolour/envs/watercolour`)
- The model writes a p5.brush sketch, which is rendered in a **headless Chromium pool** (`core/render.py:278-323`).
- Reward: `Sequential(Gate, WeightedSum[gate 0.05, length 0.05, pairwise judge 0.60, quality 0.30])` (`openenv/rubric.py:67-130`).
  - Judge: a VLM comparing against a reference pool **in both orders** to cancel position bias (`openenv.yaml` judge layer; `core/pairwise_judge.py`).
  - Quality: reference-free, filling the HPSv3 slot (`core/quality.py`). A separate GPU service is provided (`hpsv3/app.py`).
- Gate violations are listed in `openenv.yaml`.

**B6. GeoGuesser** (`FE/03-geoguesser/env`)
1. 100 + 200 tasks (`tasks/pano_v1.jsonl`, `eval_pano_v3.jsonl`). Real **Mapillary panoramas** (`server/backends/panorama.py:33-396`).
3. Typed actions look, pan, zoom, move (forward/backward metres), pin, view_map, measure, guess (`models.py:58-232`). EpisodeMode is single_shot, agentic or NMPZ (:19-34).
5. Time: `max_steps=24`, plus a free-tool budget (`server/geoguesser_environment.py:160, 869-879`).
6. Observations: **rendered view or minimap image** (`image_base64`, `image_kind`), heading, pins, steps remaining (`models.py:325-404`).
7. Reward: GeoGuessr curve `exp(−d/1492.7)`, optional hierarchical country/region credit, and an action cost (subtract, or **multiply, capped at 0.5**) (`server/scoring.py:5-200`).
   - Terminal, with the cost giving intra-episode shaping.
10. `hide_task_identity` removes contributor and ID metadata, which are a measured reward-hacking channel: 74% of tasks leak the country (`geoguesser_environment.py:660-675`).
11. Port: `server/scoring.py`, a distance reward with a multiplicative capped action cost.

**B7. SmolDataEnvs** (FE/04 + FE/05)
1. **5,394 Kaggle data-analysis tasks**: 5000 train, 250 test, 144 eval (`FE/04-smoldataenvs/README.md`). Harbor task dirs: `task.toml`, `environment/pull_bucket.py`, `tests/grader.py`.
2. State: a **Daytona sandbox** with the data pulled in. Only data credentials enter the sandbox (`FE/05-multi-harness-rl/envs/whitebox/smoldataenv_whitebox/tasks.py:30-50`).
3. Tools (whitebox): bash, read, write, edit, ls, grep, glob, submit_solution (`environment.py:50-134`).
7. Reward: a **host-side grader**, an exact → numeric-tolerance → normalisation → symbolic ladder giving binary correctness (`tasks.py:53-75`). Then `correctness × (1 + 0.1·15/(15+tool_calls))`. Grader failure gives **NaN**, i.e. excluded rather than wrong (`environment.py:136-144`; `train/multi_harness.py:78-86`).
10. The answer stays on the host; `/workdir/answer.txt` is read back.

**B8. PortSimEnv berth planning** (`FE/07-simulation-environments/portsim-v1/envs/berth_planning`)
1. 1,050 train and 50 eval tasks across four task packs (`tasks/*`). Built from real 2024 Port of Barcelona records (README).
2. State: a deterministic scheduling world (quay sections, cranes, wind windows, closures) (`core/berth_core/*`).
3. MCP tools get_situation, check_plan (**budget of 10**), submit_plan (`openenv/berth_openenv/environment.py:142-186, 239`).
4. Single agent.
5. Turn budget; ends at submit (12 turns in the eval).
7. Reward: once, at submit. Infeasible plans get 0.2 × clean fraction. Feasible plans get `0.2 + 0.8·exp(−g/0.5)`, where g is the gap to the **CP-SAT-proven optimum** normalised by avoidable cost (`core/berth_core/reward.py:74-88`; solver `solve.py:81-175`). Every other step returns 0.0 (`environment.py:8-10, 265`).
8. Sparse, but check_plan is a dense feedback oracle.
9. Includes a 3D digital-twin replay (`tools/twin/`).
11. Port: `reward.py` score_v3 together with the oracle `openenv/oracle/solve.py`, a gap-to-proven-optimum grader.

**FE tooling (not env families):**
- `FE/tooling/hf-rl-explorer/app/envs/*` reads Harbor, OpenEnv, NeMo Gym, Verifiers, verl and MiMo datasets. It withholds answer-like columns using the regex at `base.py:16-19`.
- `FE/tooling/mimo-rl-explorer/mimo_harbor` converts 7,780 MiMo envs into Harbor format.

---

## Part C. OpenEnv interface contract (exact)

### Types (`SRC/core/env_server/types.py`)
- **`Action(BaseModel)`**, :50-65. `extra="forbid"`, `validate_assignment`, `arbitrary_types_allowed`; field `metadata: dict`.
- **`Observation`**, :68-87. Fields `done: bool=False`, `reward: bool|int|float|None=None`, `metadata: dict`. **The reward is one scalar.** Multi-component rewards travel in `metadata`, as in textarena, sophistry and pelican.
- **`State`**, :180-199. `extra="allow"`; `episode_id: str|None`, `step_count: int≥0`.
- **Requests and responses:**
  - `ResetRequest{seed≥0, episode_id, **extra}`, :90-103.
  - `StepRequest{action: dict, timeout_s>0, request_id, **extra}`, :125-151.
  - `ResetResponse` and `StepResponse{observation: dict, reward: float|None, done, metadata}`, :106-168.
- **WebSocket messages:** `{type: reset|step|state|close, data}`, :277-314. Responses are `observation`, `state` and `error`, with `WSErrorCode` values (:33-42).
- **Other models:** `ConcurrencyConfig{max_concurrent_envs, session_timeout}` :346; `ServerMode`, simulation or production, :18-22; `EnvironmentMetadata`, `SchemaResponse`, `HealthResponse`.
- **Serialization:** `serialize_observation` lifts `reward` and `done` to the top level and excludes them from `observation` (`serialization.py:147-181`).

### Environment ABC (`SRC/core/env_server/interfaces.py:213-430`)
- **Signatures:**
  - `reset(seed=None, episode_id=None, **kwargs) -> ObsT`, :262-270.
  - `step(action, timeout_s=None, **kwargs) -> ObsT`, :284-292.
  - Async defaults `reset_async` and `step_async`, :272, 294.
  - `@property state -> StateT`, :306-310.
  - `get_metadata()`, :312.
  - `close()`, :424.
- **Class flags:** `SUPPORTS_CONCURRENT_SESSIONS=False` (:247) and `REQUIRES_SINGLE_THREAD_EXECUTOR` (:249).
- **Transform and rubric:** an optional `Transform` (obs→obs, TorchRL-style, :191-210) and an optional `rubric`. `_apply_rubric(action, obs) -> float` is called **by the env inside step()** (:334-386); `_reset_rubric()` (:388).
- **Task API (optional, structural):** `TaskProvider` with `list_splits`, `list_tasks(split)`, `num_tasks`, `get_task(split, i)`, `get_task_range` (:80-188). Task selection happens through `reset(split=, index=)`.

### MCP layer
- **`MCPEnvironment(Environment)`** (`mcp_environment.py:133`) wraps a FastMCP server. Actions are `ListToolsAction` and `CallToolAction{tool_name, arguments}`; observations are `ListToolsObservation` and `CallToolObservation` (`mcp_types.py:225-289`).
- **Reserved tool names** {reset, step, state, close} are rejected (`mcp_types.py:316`; `mcp_environment.py:320-368`). This enforces the "agents cannot reset" invariant from CLAUDE.md.
- **"Code mode":** `execute_code` exposes the tools as Python callables (:246-318).
- **Default MCP tool timeout is 30 s.** Harbor works around it by shadowing step (`SRC/harbor/environment.py:21-23`).

### Rubrics (`SRC/core/rubrics/`)
- `Rubric` is nn.Module-like: auto-registered children, hooks, `last_score`, sync or async `forward(action, obs)` (`base.py:17-80`).
- Containers: `Sequential`, `Gate`, `WeightedSum`, `RubricList`, `RubricDict` (`containers.py:27, 167, 223, 326, 386`).
- `TrajectoryRubric` returns 0 until done and offers `compute_step_rewards()` for credit assignment (`trajectory.py:22-60`); `ExponentialDiscountingTrajectoryRubric` is at :138.
- `LLMJudge(prompt_template, client, score_pattern, normalize)` is at `llm_judge.py:28-46`.

### Server (`SRC/core/env_server/http_server.py`)
- `HTTPEnvServer` is at :140, `create_app(env_factory, action_cls, observation_cls, env_name, max_concurrent_envs | concurrency_config, gradio_builder, ..., mode)` at :1982.
- **HTTP `/reset` and `/step` are stateless.** Each call builds and closes a new env from the factory (:897-918, 932). They are only registered in **simulation** mode (:1505-1560).
- **Stateful sessions run over WebSocket `/ws`** (:1731), with message cases reset, step, state, close and **mcp** (:1814-1893).
- Other routes:
  - `/state` (:1636), `/metadata` (:1609), `/health` (:1622), `/schema` returning the JSON schemas for action, observation and state (:1652).
  - `POST /mcp` (:1701) and WebSocket `/mcp` (:1395).
  - Task API: `/{env}/splits | tasks | num_tasks | task | task_range` (:1343-1383).
  - **production** mode with a `HarnessEnvironment` factory adds WebSocket `/harness` (:680, 1500-1502).
- More than one concurrent session requires the env class to set `SUPPORTS_CONCURRENT_SESSIONS` (:294-312).

### Clients
- **`EnvClient`** (async, WebSocket, persistent session) is at `SRC/core/env_client.py:267`.
  - Constructor options: `mode` ∈ simulation (Gym API) or production (MCP JSON-RPC), also settable through `OPENENV_CLIENT_MODE`. Max message size 100 MB ("screenshots, DOM") (:308-374).
  - The WebSocket URL is `{base}/ws` (:390-393).
  - `from_docker_image(image, provider)` is at :760. `from_env(repo_id, use_docker)` pulls `registry.hf.space/...` or runs the Space with uv (:803-941).
  - Subclasses implement `_step_payload`, `_parse_result → StepResult` and `_parse_state` (:943-956).
  - `reset`, `step`, `state` and `close` send the WebSocket messages (:958-1014); `.sync()` is at :1146.
- **`StepResult{observation, reward: float|None, done, metadata}`** is at `client_types.py:10-29`.
- **`SyncEnvClient`** (`sync_client.py:43`), **`GenericEnvClient`** (dict actions, `generic_client.py:17`), **`MCPToolClient.list_tools/call_tool`** (`mcp_client.py:271, 438-483`).
- **In-process mode:** construct the `Environment` subclass directly and call reset/step. Every env's server class is a plain Python object, and `create_app` takes the class as factory.
- **Runtime providers:** LocalDocker, DockerSwarm, Kubernetes (`containers/runtime/providers.py:120, 329, 647`), Daytona, Modal, ACA, Novita, HFSandbox, and UVProvider (`RuntimeProvider`, :660).
- **Importers** turn Prime Intellect Verifiers or Open Reward Standard sources into OpenEnv packages (`SRC/cli/importers/verifiers.py`, `ors.py`).

### Manifest `openenv.yaml`
- **Base fields:** `spec_version: 1, name, type: space, runtime: fastapi, app: <module>:app, port` (e.g. `ENV/atari_env/openenv.yaml`).
- **Legacy shape:** `name, version, description, action, observation` (`ENV/browsergym_env`, `coding_env`, `julia_env`).
- **Free-form extensions** seen in the wild: `description`, `variables` (Space env vars), `environment` (constructor defaults), `rubric`, `tools`, `metrics`, `gate_violations` (`ENV/pelican_svg_env/openenv.yaml`, `ENV/qed_math_env/openenv.yaml`).
- **Validated block `validation:`** (RFC 008) normalizes to `NormalizedManifest` (`SRC/validation/manifest.py:271`):
  - **`reward`** {range, oracle_tolerance, floor_margin, variance_tolerance} (:39).
  - **`judge`** pin {model, version, params}, required if and only if `llm_judged` (:75, 295-309).
  - **`resources`** {cpu, memory_mb, disk_mb, episode_timeout_s, gpus, gpu_types} (:139).
  - **`network`** {mode: public | no-network | allowlist, allowed_hosts} (:164).
  - **`task_distribution`** pins (:192).
  - **`capabilities`** {oracle{form: injected_state | script, location}, verifier{kind: reward_channel | script, entry}, set_state, llm_judged, rubric_tree, task_api, canaries, declared_tools, declared_task_count} (:202-256).
  - **`types.tags`** (:258).
  - Schema v2 adds `execution{kind: openenv_ws, probe_path, dockerfile, context, agent_boundary: api}` (:311-386).
  - Example: `ENV/echo_env/openenv.yaml`.

### Validation checks
- **Severity policy** `SRC/validation/policies/severity-v2.json`.
  - Bounds: max_oracle_tolerance 0.1, min_floor_margin 0.1, max_variance 0.2, max_episode_timeout 3600 s.
  - **static.*:** manifest, reproducible_build, image_hygiene, layout, sbom, oci_labels, resource_declaration, timeout_ceiling, dependency_pinning, task_distribution_pinning.
  - **runtime.*:** startup, reward_well_formed, rubric_introspectable, observation_schema, state_contract, trajectory_record, reward_attribution, tool/task_declaration_accuracy, seed_control, episode_determinism, network_policy, host_containment, resource_bounds, episode_isolation, oracle_containment.
  - **semantic.*:** oracle_max, floor_gap, canary_floor, no_solution_leakage, verifier_determinism, verifier_portability, resource_envelope, replayability.
  - **hub.*** and **statistical.*:** reward_reachability, difficulty_separation, headroom, variance_bound, training_signal, adversarial_floor, gameability_gap.
- **Packages local validation refuses:** **multi-agent**, **simulated-user** and **hosted-verifier** packages are explicitly "unsupported" (`SRC/validation/signature.py:12-28`).
- **Live-server checks** (`SRC/cli/_validation.py:99`): openapi_version_available, health_endpoint, metadata_endpoint, schema_endpoint, mcp_endpoint, mode_endpoint_consistency.
- **Deployment checks** (`validate_multi_mode_deployment`, :505-564): pyproject, uv.lock, `[project.scripts] server=...:main`, openenv dependency, `server/app.py` with `main()` and a `__main__` guard.

### Harness contract (RFC 005/006; `SRC/core/harness/*`)
- **`HarnessEnvironment(MCPEnvironment)`**: one `step(HarnessAction{message})` = **one conversational turn** of an external harness's own ReAct loop. Env MCP tools are injected into the harness; rubrics run after each turn (`environment.py:34-90`).
- **`HarnessConfig{name, command, transport: stdio | http | mcp, ...}`** (`config.py:13-40`).
- **Rollout API:**
  - `ResourceSession{initial_messages, list_tools, call_tool, verify, close}` (`rollout.py:223-255`).
  - `ResourceSessionFactory.create(task, seed, episode_id)` (:258).
  - `HarnessAdapter.run_white_box(model_step, session)` vs `run_black_box(session)` (:277-295).
- **Reward resolution:** `_resolve_env_reward` forbids synthesising a reward outside the env. A verify reward must equal the trace reward (:329-360).

---

## Part D. The FineEnvs multi-harness idea

- **Harness definition:** "the software layer between the model and the task that constructs the context the model sees, mediates its tool calls, validates its outputs, and decides when to retry, escalate, or stop". To a trainer it is a program it cannot change (`hf-space-fineenvs-multi-harness-rl/app/src/content/chapters/basics.mdx:6-12`).
- **Division of responsibility:** the env owns state, transitions and reward; the harness runs the loop (:50-55).
- **White-box** = the trainer owns the loop (env.step); **black-box** = the harness owns the loop inside a sandbox, and the trainer sees only model request/response pairs (:70-92).
- **Same tasks and grader, three loop owners** (`FE/05-multi-harness-rl/README.md`):
  1. **Whitebox SETA:** TRL `GRPOTrainer(environment_factory=BashEnvironment)` calls Python tools directly (`envs/whitebox/smoldataenv_whitebox/environment.py:14-144`).
  2. **Native OpenCode** in a sandbox (`envs/opencode/`, using OpenEnv `opencode_env`; deprecated).
  3. **Harbor**, which rotates opencode, claude-code, codex and mini-swe-agent. `TaskFactory.create` assigns `harness=HARNESSES[index % len]` per task, so all 8 GRPO rollouts of a task share that harness. Daytona sandbox, `agent_step_limit=17`, `agent_timeout_sec=600`, `reward_key="correctness,reward"` (`envs/harbor/smoldataenv_harbor/environment.py:53-85`).
- **How tokens and reward flow:** OpenEnv's **capture proxy** receives the harness's model calls, authenticated by a per-rollout session id used as the API key, and returns a 401 for unknown keys. It detects 4 API dialects and converts them to chat-completions (`chapters/openenv.mdx:40-80`). It returns engine token ids, behaviour logprobs and loss masks via `fetch_training_trace()` for `AsyncGRPOTrainer` + `HarnessRolloutWorker` (05 README).
- **Reward is identical across harnesses:** the env grade, a tool-count bonus, and None on infrastructure failure (`train/multi_harness.py:78-86`).
- **Evaluation:** pass@1 in every harness, 1,000 attempts = 250 tasks × 4 harnesses.

**Reward-architecture taxonomy** from the guide article (`hf-space-rl-environments-guide/app/src/content/chapters/dimensions.mdx:66-92`):
1. External, as in SkyRL Gym and GEM.
2. Server-embedded per tool call, as in ORS.
3. Post-episode `/verify`, as in NeMo Gym.
4. Environment-embedded composable Rubric, as in OpenEnv and Verifiers.

---

## Part E. Primitives a framework needs to express all of these

Deduplicated. Families are cited by short name.

1. **Typed action schemas with JSON-schema export**, discriminated unions included. Every OpenEnv env; AWM (`AWMAction` union), ThinkingBox, GeoGuesser.
2. **MCP tool surface as an action space, with reserved control names hidden from the agent.** echo, finqa, calendar, AWM, coding_tools, jupyter, terminus, qed_math, PortSim, FE desktop/jupyter/wordle, harbor, opencode, pi.
3. **Parallel tool calls in one step.** ThinkingBox.
4. **Discrete integer or enum actions with legal-action masks.** atari, openspiel, connect4, chess, maze, snake, sumo (`action_mask`).
5. **Continuous control vectors**, with action/observation spec export. dm_control, unity, finrl, carla (throttle/steer/brake).
6. **Mixed continuous plus macro/tool actions in one space.** carla (control + set_destination/follow_route), unity (discrete + continuous).
7. **Pixel and image observations**, as raw arrays or encoded PNG/JPEG, plus image ToolResults or MCP image blocks.
   - Raw arrays: atari, snake (HxWxC), browsergym.
   - Encoded: dm_control, unity, carla camera, helium, openapp, jupyter cells, geoguesser, latex_ocr, FE desktop, pelican/watercolour.
   - **Large-message transport is needed** (100 MB WebSocket frames in EnvClient).
8. **Mouse/keyboard GUI control with pixel coordinates** and a virtual display or remote VM. helium (Xvfb), FE desktop (E2B Desktop, 19 computer_20251124 tools), openapp and browsergym (bid-based DOM actions, a different primitive).
9. **Accessibility tree and DOM observations.** browsergym, openapp.
10. **Structured, vector or tensor observations.** openspiel `info_state`, sumo, dm_control dict, wildfire grid, finrl.
11. **Token-level actions and observations, with the env owning the tokenizer.** chat_env.
12. **Simulator-owned time:** fixed timestep, `ticks_after_tool`, idle ticks while the agent deliberates, sim_time in the observation. carla, sumo (delta_time 5 s), dm_control (control/physics timestep), unity (time_scale), wildfire (fire spreads each step), finrl (trading days).
13. **Real wall-clock time inside actions.** helium (1.5-3 s waits), FE desktop `wait`/`hold_key`, tbench2 (`wait_seconds`, non-blocking sessions), harness rollouts lasting 60-1800 s (harbor, opencode, pi).
14. **Step and turn budgets with distinct end reasons.**
    - Budgets: browsergym, openapp, finqa (50), kernrl (10 turns), geoguesser (24 + free-tool budget), PortSim (check budget 10), repl (30), helium (20).
    - End reasons: truncated vs terminated (browsergym, sumo); system_error vs fail (thinkingbox, tbench2 missing verdict); NaN or None for infrastructure failure (SmolDataEnvs).
15. **Per-step env-computed reward (dense).** atari, dm_control, unity, sumo, wildfire (shaped), grid_world, snake, finrl, julia, kernrl, carla navigation, textarena Wordle signals, coding_env transforms.
16. **Terminal-only grading on submit, with 0 on intermediate steps.** finqa, PortSim, reasoning_gym, dipg, pelican, watercolour, latex_ocr, coding_tools, jupyter, terminus, tbench2, harbor, opencode, SmolDataEnvs, thinkingbox, chess.
17. **Multi-component reward vectors next to the scalar, with weights and hidden components withheld from the observation.** textarena `reward_signals`, sophistry (8 components, hidden correctness), dipg (~17 terms), pelican and watercolour breakdowns, geoguesser (distance, region, cost), latex_ocr (CER + exact + length).
18. **Composable rubric trees**, Gate→WeightedSum and Sequential, including async LLM/VLM judges, a judge-availability flag, and order-swapped pairwise judging. pelican, watercolour, qed_math (0-7 LLM rubric), AWM (SQL+LLM judge), thinkingbox (Judge), calendar (response_check).
19. **Trajectory-level credit assignment**, e.g. discounted per-step rewards from a terminal outcome. chess, carla trolley (`TrajectoryRubric`).
20. **Process plus outcome reward composition.** repl (CodeExecutionRubric + ExactMatch), jupyter-101 ORS (efficiency bonus, error penalty), SmolDataEnvs tool-count bonus.
21. **Hidden verifiers outside the agent's reach.**
    - Mechanisms: withholding or moving tests and solutions (tbench2); host-side grader with only data creds in the sandbox (SmolDataEnvs); pre-clearing the reward.txt override (coding_tools, jupyter, terminus, opencode, pi); answers never in observations (latex_ocr, geoguesser `hide_task_identity`, hf-rl-explorer answer regex).
22. **State-diff verification:** an initial DB snapshot plus a final DB fed to a verifier. AWM (`save_snapshot`, code verifier), calendar (SQL `database_state` checks).
23. **Proven-optimum or oracle grading** (gap-to-optimum), with oracle declaration in the manifest. PortSim (CP-SAT); OpenEnv `capabilities.oracle` with form injected_state or script.
24. **Two-player and N-player turn-taking**: seat assignment, scripted or engine opponents, simultaneous moves, chance nodes, and hot-seat where one client plays all seats. openspiel, chess (moonfish or self-play), connect4, textarena (`current_player_id`, `legal_players`). Multi-agent sim capability is also present but unused in sumo and snake.
    - OpenEnv validation **explicitly does not support multi-agent packages** (`signature.py`).
25. **Simulated users (LLM user simulator)** that can end the conversation, with user-turn budgets. thinkingbox; calendar's scenario "user_prompt" is static.
    - OpenEnv validation also refuses **simulated-user** packages.
26. **Acting-user and identity context** in a multi-user backend. calendar (actingUserId, access token headers).
27. **Remote environment client over WebSocket** (persistent session) plus stateless HTTP and MCP JSON-RPC (production mode). All OpenEnv envs; HTTP `/reset` and `/step` cannot carry state (http_server.py:897-932).
28. **In-process environment mode** (no server). OpenEnv classes used directly; FE Verifiers, SkyRL and GEM variants; whitebox `BashEnvironment` for TRL `environment_factory`.
29. **Per-session isolation and a concurrency declaration.** `SUPPORTS_CONCURRENT_SESSIONS`, `max_concurrent_envs` (AWM 10,000, textarena 8, harbor 40), `REQUIRES_SINGLE_THREAD_EXECUTOR` (helium).
30. **Env-side state snapshot and inspection**, a `state()` separate from the observation; reset with injected state (`set_state`); fast reset to pinned states. All envs (`State`); git_env commit-pinned reset; AWM per-session DB; manifest `set_state` capability.
31. **Task API**: splits, counts, get_task, ranges, and reset(split, index). latex_ocr, harbor, PortSim, AWM (`scenario`, `task_idx`), qed_math, reasoning_gym (procedural `dataset_config`, seed, size).
32. **Procedurally generated tasks with seeds.** reasoning_gym, wildfire (seed, forced wind), pelican (subject × vehicle grid), dm_control and openspiel params.
33. **Per-episode cloud sandboxes**: E2B, E2B Desktop, Daytona, Modal, HF Sandbox, Novita, ACA, plus local Docker, Swarm and K8s. coding_tools, jupyter, terminus, opencode, pi, harbor, helium, FE desktop/jupyter, SmolDataEnvs.
34. **GPU-backed state or grading.** kernrl (GPU benchmarking, nsys/ncu/sanitizer), carla (CUDA image), unity and dm_control rendering, watercolour HPSv3 service; manifest `resources.gpus`/`gpu_types`.
35. **External service dependencies, declared and health-checked.** Web search API (Serper), WebArena sites, OpenApps server, Gitea, ThinkingBox proxy and Typesense, LLM judge endpoints, Mapillary, HF datasets; manifest `network` policy (public, no-network, allowlist).
36. **Black-box agent harness as the environment's driver**: the env launches an agent CLI in a sandbox; a model-call capture proxy records token ids, logprobs and masks across API dialects; per-task harness assignment. harbor, opencode, pi, `HarnessEnvironment`, FE 05.
37. **Recursive or sub-LLM calls from inside the env.** repl `llm_query`, judges in qed/pelican/watercolour/AWM/thinkingbox.
38. **Rendering pipelines inside the env for grading**: headless browser and SVG rasteriser. watercolour (Chromium pool), pelican (render.py), latex_ocr (image dataset).
39. **Anti-gaming gates and canaries before scoring.** pelican and watercolour `gate_violations`, sophistry canaries, latex_ocr length guard, coding safety transforms, geoguesser identity hiding, Wordle "reward once" semantics.
40. **Manifest-declared contracts plus automated validation**: reward range, oracle, floor margin, judge pin, determinism and variance, resources, network, declared tools and task counts, canary trajectories. OpenEnv RFC 008 (`manifest.py`, `severity-v2.json`).
41. **Rich trace export for training** (per-turn tokens and logprobs, tool traces). opencode `RolloutResult.proxy_turns`, harbor `contract.py`, `TrainingTrace`.

**Fit notes for Crucible, from the evidence above:**
- Items 24-26 (multi-agent, simulated users) and 21-23 (hidden or state-diff verifiers) are exactly the categories OpenEnv's own validator refuses or leaves to individual envs. They are where a World/Actors/hidden-verifier model adds value.
- Item 12 needs a sim Clock that the *environment* advances per tool call (carla `ticks_after_tool`).
- Item 36 needs an "Actor = external harness" whose model traffic is captured, not a policy calling tools through one gate.
- Item 7 needs both raw-array and encoded image transport.
- Item 27: HTTP-stateless vs WebSocket-stateful matters for any OpenEnv importer. Only `/ws` (or in-process) preserves episode state.
