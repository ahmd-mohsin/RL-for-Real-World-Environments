<!-- M1 taxonomy 05: environment interfaces of 11 RL trainers. Read-only study by a subagent, 2026-10-12. -->

# How 11 RL trainers connect to environments

All paths are relative to `ROOT=/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/RL-Environments/06-frameworks-and-env-code`.

**Licenses:** every repo is Apache-2.0 except `infra-moonshotai-checkpoint-engine`, which is MIT (`LICENCE`). Some envs pull in third-party packages that carry their own licenses (gymnasium, gym-sokoban, webshop-minimal, tau2, harbor, openenv, verifiers, ROCK, E2B, Daytona, Modal).

**Scope notes:**
- `infra-volcengine-verl/recipe/` is empty here (the verl-recipe submodule is not checked out). verl's SWE, ReTool and LangGraph recipes are therefore not visible. Only core agent loops and tools are covered.
- `checkpoint-engine` contains no environment code. It is weight-sync middleware only.
- prime-rl ships no env implementations. Its envs are `verifiers.v1` packages served by an env server.

---

## A. Environments shipped or natively supported

Reward density key: **T** = terminal only, **S** = per-step, **F** = format/penalty shaping.

### SkyRL (`infra-novasky-ai-skyrl`)

| Env (registry id) | State / observation | Actions | Horizon | Reward | Infra | file:line |
|---|---|---|---|---|---|---|
| `aime` | prompt only | final text answer | 1 | T (math verify against `reward_model.ground_truth`) | none | `skyrl-gym/skyrl_gym/envs/aime/env.py:6` |
| `gsm8k` | prompt | answer text | 1 | T | none | `skyrl-gym/skyrl_gym/envs/gsm8k/env.py:6` |
| `gsm8k_multi_turn` | chat; "N turns remaining" user msg | retry answers | ≤5 | **S** (per turn: 1.0 correct, else 0.2/max_turns format) | none | `skyrl-gym/skyrl_gym/envs/gsm8k/multi_turn_env.py:7,46-66` |
| `text2sql` | chat plus SQL result rows | `<sql>` tool call / `<solution>` | ≤5 | T (0 for intermediate steps) | local SQLite with thread timeout | `skyrl-gym/skyrl_gym/envs/sql/env.py:17`, `tools/sql.py:15` |
| `search` | chat plus retrieved passages | `<search>` / `<answer>` | ≤2 (configurable) | T (EM) | HTTP retriever at `127.0.0.1:8000/retrieve`, timeout 30 | `envs/search/env.py:19`, `tools/search.py:140` |
| `lcb` | prompt | code | 1 | T (unit tests) | local exec | `envs/lcb/env.py:7` |
| `searchcode` | chat | search or python tool | ≤2 | T | search plus subprocess python | `envs/searchcode/env.py:9`, `tools/python.py:5` |
| RLM (`BaseRLMEnv`) | persistent REPL plus context metadata | python code blocks, `FINAL()`/`FINAL_VAR` | ≤10 | T (`_get_reward(final_answer)`) | PersistentREPL, `repl_timeout=180` | `envs/rlm/env.py:180-402` |
| VisGym (example) | image plus text | parsed action | multi | T (`reward if done`) | VisGym, VLM generator | `examples/train/visgym/env_instruct.py:16` |
| OpenEnv bridge | serialized OpenEnv observation | env-specific action | max_turns | S (env reward; -1 on parse failure) | OpenEnv HTTP/Docker | `examples/train_integrations/openenv/env.py:48` |
| OpenReward bridge | tool outputs | JSON tool call | multi | **S** (per tool-call reward) | OpenReward API, retries on 5xx/429 | `examples/train_integrations/openreward/env.py:52,106` |
| Harbor (custom generator) | containerized trial | agent harness | long | T; timeouts masked, context overflow gets 0 | Harbor Trial (Docker/Modal) | `examples/train_integrations/harbor/harbor_generator.py:223,354` |
| Verifiers bridge | verifiers env | n/a | n/a | n/a | verifiers | `examples/train_integrations/verifiers/verifiers_generator.py` |
| mini-swe-agent, thunder_agent, llm_as_a_judge, multiply, geometry3k | n/a | n/a | n/a | n/a | n/a | `examples/train/*` |

SkyRL-Agent (`skyrl-agent/`) adds the following:
- Tasks: SWE-bench/R2E, web research, general ReAct, MemAgent (`skyrl_agent/tasks/*`).
- Tools: browser, search engines, SandboxFusion, str_replace_editor, finish (`skyrl_agent/tools/*`).
- Agents: OpenHands CodeAct and ReAct.
- Backends: verl, SkyRL-train, Tinker, OpenAI (`skyrl_agent/integrations/*`).

### verl (`infra-volcengine-verl`)

There are no stateful envs. Interaction happens through agent loops and tools.

- **`single_turn_agent`**: one LLM call; reward from `verl/utils/reward_score/__init__.py:19 default_compute_score`, dispatched by `data_source`. Sources covered: gsm8k, MATH, DAPO/AIME, codecontests/apps/codeforces/taco (via SandboxFusion), geometry3k, search-r1 QA EM, and rlla tool use. File: `verl/experimental/agent_loop/single_turn_agent_loop.py:28`.
- **`tool_agent`**: a multi-turn ReAct/tool state machine. Its envs are the tools themselves (`BaseTool` or `@function_tool`).
  - Example tool: SandboxFusion code interpreter at `examples/tutorial/agent_loop_get_started/sandbox.py:22`.
  - Data preprocessors for gsm8k, AIME, geo3k and DAPO multi-turn tool tasks live in `examples/data_preprocess/*_w_tool.py`.
- **Reward:** trajectory-level `reward_score`, placed on the last response token as `rm_scores[-1]` (`agent_loop.py:141-146`). Per-turn `turn_scores` and `tool_rewards` are only exposed in `extra_fields` (`tool_agent_loop.py:212`).
- **SandboxFusion client:** `verl/utils/reward_score/sandbox_fusion/utils.py:67`, with retries and separate compile/run timeouts.

### rLLM (`infra-rllm-org-rllm`)

Envs are written as **AgentFlows**: plain Python functions that call an OpenAI-compatible endpoint and return an `Episode`.

| Env | State / obs | Actions | Horizon | Reward | Infra | file:line |
|---|---|---|---|---|---|---|
| FrozenLake | text grid P/_/O/G | Up/Down/Left/Right in a code block | ≤max(2·size,16) | T via evaluator (`is_correct=won`) | gymnasium, seeded map | `cookbooks/frozenlake/frozenlake_flow.py:150-245` |
| Math, math tool agent | prompt (plus calculator) | answer / tool calls | 1..n | T | none | `cookbooks/math/math_flow.py`, `cookbooks/math_tool_agent/` |
| DeepCoder | prompt | code | 1 | T (tests) | code sandbox | `cookbooks/deepcoder/deepcoder_flow.py` |
| FinQA | tables plus tools | tool calls | multi | T | local tools | `cookbooks/finqa/finqa_flow.py`, `finqa_tools.py` |
| Geo3k (VLM) | image plus question | answer | 1 | T | none | `cookbooks/geo3k/geo3k_flow.py` |
| Solver-judge (multi-agent) | n/a | solver plus judge trajectories | 2 roles | per-trajectory rewards | none | `cookbooks/solver_judge_flow/solver_judge_flow.py` |
| Agent frameworks | n/a | LangGraph, Strands, smolagents and OpenAI Agents SDK flows | n/a | n/a | n/a | `cookbooks/agent_frameworks/agentflow/*.py` |
| MigrationBench, AgentCore math | n/a | Bedrock AgentCore remote runtime | n/a | n/a | n/a | `cookbooks/migrationbench/`, `examples/agentcore_math/`, `rllm/engine/remote_runtime/agentcore_runtime.py` |
| Countdown | prompt | arithmetic answer | 1 | T | none | `examples/countdown/`, `rllm/rewards/countdown_reward.py` |
| DeepResearch (fully async) | search results | search/refine tools | multi | T | RAG server | `examples/fully_async/deepresearch/` |
| Harbor SWE | container | CLI harness | long | T (verifier) | Harbor | `examples/harbor_swe/`, `rllm/integrations/harbor/` |
| Verifiers env | n/a | n/a | n/a | n/a | n/a | `examples/verifiers_env/workflow.py` |
| CLI harnesses (12) | sandbox | n/a | n/a | n/a | sandbox backends: docker, daytona, modal, local (`rllm/sandbox/backends/*`) | `rllm/registry/agents.json`; `rllm/harnesses/*.py` (`BaseCliHarness` at `cli_harness.py:44`) |

- **CLI harnesses:** react, bash, claude-code, opencode, mini-swe-agent, terminus2, codex, qwen-code, aider, kimi-cli, oracle, zeroclaw.
- **Dataset registry:** 66 datasets in `rllm/registry/datasets.json`. They include SWE-bench Verified/Pro, R2E-Gym, DeepSWE, BrowseComp, HLE, BFCL, MMMU, OCRBench, SkillsBench and claw_eval.

### AReaL (`infra-inclusionai-areal`)

| Env / workflow | Protocol | Horizon | Reward | Infra | file:line |
|---|---|---|---|---|---|
| RLVR / vision RLVR | single generate | 1 | T (`reward_fn`) | `areal/reward/{gsm8k,geometry3k,clevr_count_70k}.py` | `areal/workflow/rlvr.py`, `vision_rlvr.py` |
| Multi-turn math (retry until correct) | generate, then "wrong, retry" | ≤max_turns | T × `turn_discount^t` | none | `areal/workflow/multi_turn.py:20-142` |
| tau2-bench airline | OpenAI-SDK agent vs user simulator | max_steps | T (tau2 `evaluate_simulation`) | tau2; `asyncio.wait_for` 600s | `examples/tau2/agent.py:65,275,374` |
| Terminal-Bench 1.0 (SETA port) | CAMEL terminal agent | long | T (TB tests) | Docker TB tasks | `examples/terminal_bench/workflow/camel_rlvr_workflow.py` |
| SWE-bench (AReaL-SWEAgent, Arena) | external agent via proxy | long | T (patch graded) | external sandbox | `examples/swe/agent.py`, `arena_agent.py` |
| TIR math | python/calculator markers | multi | T | local or Daytona python | `examples/tir/tir_workflow.py`, `tool_manager.py:32` |
| Search agent (Tongyi DeepResearch) | search tools | multi | T | search APIs | `examples/search_agent/` |
| CAMEL, OpenAI Agents, LangChain, Anthropic, Hermes, OpenClaw/ZeroClaw | any OpenAI- or Anthropic-speaking agent via proxy | multi | T; can be set externally through `/rl/set_reward` | proxy gateway | `examples/{camel,openai_agents,hermes,openclaw}`, `areal/workflow/{langchain,anthropic,openai_agent}` |
| Countdown, PRM, scaffolding, DTE, VLM | n/a | n/a | n/a | n/a | `examples/*` |
| Daytona sandbox reward | n/a | n/a | n/a | n/a | `examples/sandbox_daytona/reward_example.py` |

### ROLL (`infra-alibaba-roll`)

All envs are registered through **GEM**: `gem.register` in `roll/pipeline/agentic/env/__init__.py:9-40`.

| id | State / obs | Actions | Horizon | Reward | Infra | file:line |
|---|---|---|---|---|---|---|
| `sokoban` | text grid | parsed direction | max_steps=20 | S plus F (`format_penalty`) | gym-sokoban | `env/sokoban/env.py:14,90` |
| `sokoban_native_env`, `sokoban_tool_call` | messages | tool call | 10–20 | S plus F | n/a | `env/sokoban/native_env.py`, `tool_call_env.py:16,151` |
| `sokoban_mcp` | MCP server | MCP tool | 20 | S plus F | MCP client | `env/mcp/sokoban_mcp_env.py:12` |
| `sokoban_sandbox` | remote game server | HTTP | n/a | S | ROCK sandbox | `env/sandbox/sokoban_sandbox_env.py:37` |
| `frozen_lake` | grid | direction | 20 | S plus F | gym | `env/frozen_lake/env.py:13` |
| `webshop` | HTML/text page | search/click | 10 | T plus F | webshop-minimal (third_party) | `env/webshop/env.py:15` |
| `roll_math`, `roll_code`, `roll_qa` | prompt | answer | 1 | T (math grading via process pool, timeout=1) | GEM | `env/gem/{math,code,qa}_env.py` |
| `deepeyes` (VLM) | image plus zoom tool | tool/answer | 10 | T (LLM judge) | reward model | `env/deepeyes/env.py:158` |
| `geo3k` | image plus tools | `calc_score` tool | 3 | T | n/a | `env/geo3k/env.py:78` |
| `rock_tb_native_env` | Terminal-Bench in a ROCK sandbox | bash | 80 | T (tests; agent timeout 40 min, test timeout 10 min) | ROCK | `env/sandbox/rock_tb_native_env.py:14` |
| CLIEnv | sandbox shell | commands | 20 | T plus F (-0.1) | ROCK/iflow | `env/cli_env/env.py:22` |
| `openreward_env` | tool calls | n/a | 16 | T plus `nonterminal_reward` | OpenReward | `env/openreward/openreward_env.py:35` |
| `atropos_env` | Atropos (Nous) env bridge | n/a | 16 | S | async executor, timeout 120 | `env/atropos/atropos_env.py:38` |

Tool wrappers: python code tool, MCP tool, `tool_wrapper` (`roll/pipeline/agentic/tools/*`).

### NeMo-RL (`infra-nvidia-nemo-rl`)

Registry: `nemo_rl/environments/utils.py:36`.

| id | Protocol | Reward | Infra | file:line |
|---|---|---|---|---|
| `math` / `math_default` / `math_multi_reward` | batched message logs | T (HF math-verify / DAPO); multi-reward dict for GDPO | Ray worker pool (`num_workers`) | `math_environment.py:86-259` |
| `code` | code exec with feedback | S/T | Ray `CodeExecutionWorker` | `code_environment.py:50,191` |
| `code_jaccard` | n/a | T (similarity) | Ray, `max_restarts=-1` | `code_jaccard_environment.py:148` |
| `reward_model` | n/a | T (RM policy, e.g. Skywork) | GPU actor | `reward_model_environment.py:72` |
| `vlm` | n/a | T (weighted reward fns: GIoU, format, geo3k…) | Ray | `vlm_environment.py:59`, `rewards.py` |
| `nemo_gym` | HTTP `/run` against a NeMo-Gym head server; whole rollouts done remotely | T plus multi-reward | NeMo-Gym servers, token capture, shards | `nemo_gym.py:457,776`, `nemo_gym_shards.py:1690` |
| Sliding puzzle | multi-turn game | T (1.0 on win) | Ray | `games/sliding_puzzle.py:41,346` |
| RAG (BM25) | retriever tool | 0 | Ray | `tools/retriever.py:135` |

NeMo-Gym example configs include SWE, workplace assistant, Nemotron super/ultra/omni env suites, and GDPO tool-call multi-reward (`examples/nemo_gym/*.yaml`).

### slime (`infra-thudm-slime`)

Envs are custom `generate()` functions.

| Example | Protocol | Reward | file |
|---|---|---|---|
| search-r1 | `<search>` turns; obs tokens get loss_mask 0 | T (EM) | `examples/search-r1/generate_with_search.py:145,243,277` |
| ReTool | code sandbox tool | T | `examples/retool/generate_with_retool.py`, `tool_sandbox.py` |
| tau-bench | tool-calling agent plus user sim | T | `examples/tau-bench/generate_with_tau.py:100` |
| geo3k VLM multi-turn | `BaseInteractionEnv` reset/step/close | T | `examples/geo3k_vlm_multi_turn/base_env.py`, `env_geo3k.py` |
| multi-agent | parallel solver/rewriter agents with weighted rewards | T | `examples/multi_agent/agent_system.py` |
| coding agent (SWE) | Claude Code / Codex harness in an E2B sandbox | T | `examples/coding_agent_rl/swe.py`, `slime/agent/harness/*`, `slime/agent/sandbox.py:160` |
| Strands | Strands agents SDK | T | `examples/strands_sglang/` |

Reward-model hub: deepscaler, f1, gpqa, ifbench, math (`slime/rollout/rm_hub/`).

### OpenRLHF (`infra-openrlhf-openrlhf`)

| Example | Protocol | Reward | file |
|---|---|---|---|
| Random n-step env | `AgentInstance` reset/step | T | `examples/python/agent_func.py` |
| GEM GuessTheNumber | `gem.make("game:GuessTheNumber-v0")` | S, summed | `examples/python/agent_func_gem_multiturn.py:48` |
| OpenAI-server executor | local FastAPI `/v1/chat/completions` stitched into sessions | T | `examples/python/agent_func_openai_server_executor.py:52` |
| VLM multi-turn | images in feedback | T | `examples/python/vlm_multiturn_agent.py` |
| Remote RM / reward_func | HTTP or Python | T | `examples/python/reward_func.py`, `math_reward_func.py` |

### TRL (`infra-huggingface-trl`)

Most envs use `environment_factory`.
- **OpenEnv-backed:** Wordle, Sudoku (4 reward components), Catch (OpenSpiel), BrowserGym (VLM), CARLA (VLM), Echo, multi-env (Wordle + Catch). Files: `examples/grpo_{wordle,sudoku,catch,browsergym,carla,echo,multi_env}/`.
- **In-process:** 2048 (`examples/grpo_2048/grpo_2048.py:37`) and the SQL agent.
- **Harbor:** `trl/experimental/harbor/_env.py:37` (`HarborEnv`, `HarborBashEnv:210`, docker/E2B).
- **OpenReward:** `trl/experimental/openreward/environment.py:104`.
- **SETA:** `examples/grpo_seta`.
- **Async GRPO with an OpenEnv Harbor server:** `examples/async_grpo_harbor/async_grpo_harbor.py`, `trl/experimental/async_grpo/openenv_harness.py:336`.

### prime-rl (`infra-primeintellect-ai-prime-rl`)

- Envs are `verifiers.v1` tasksets and players served by an env server.
- Examples: alphabet-sort, reverse-text, wiki-search, wordle, hendrycks (`examples/basic/*/rl.toml`).
- The RAE algorithm covers multi-agent self-play envs such as kuhn-poker (`src/prime_rl/orchestrator/algo/rae.py`).

---

## B. Environment interface contracts per trainer

### SkyRL
- **Base class:** `Env.step(action) -> EnvStepOutput{observations, reward, done, metadata}`, plus `init(...) -> (obs, info)` and `close()` (`skyrl-gym/skyrl_gym/core.py:12-80`).
- **Text env:** `BaseTextEnv(Env[ConversationType,str])`. `init(prompt)` returns `(prompt, {})`. `step(str)` returns `BaseTextEnvStepOutput{observations: list[msg], reward: float, done, metadata, postprocessed_action}`. It also exposes `get_metrics()` and the static `aggregate_metrics()` (`envs/base_text_env.py:11-104`).
- **Tools:** `ToolGroup` with an `@tool` decorator (`tools/core.py:4-58`). Tools run synchronously inside `step`.
- **Registration:** a gym-style `register(id, entry_point)` / `make(id, env_config, extras)` (`envs/registration.py:183,233`; list at `envs/__init__.py`). Per-row selection uses the dataset's `env_class` and `extras` (`skyrl/train/generators/base.py:31-38`).
- **Driver:** `SkyRLGymGenerator.agent_loop` (`skyrl/train/generators/skyrl_gym_generator.py:339`).
  - It calls `skyrl_gym.make` (line 397), then `env.init`, `env.step` and `env.close`, each offloaded to a `ThreadPoolExecutor` (`max_env_workers=32`; lines 200, 286, 549, 642; config at `config.py:1424`).
- **Token handling:** token-in/token-out.
  - Observations are tokenized by applying the chat template to `[*base_conversation, *new_obs]` and slicing off the base prefix (`get_obs_ids_from_obs`, line 783).
  - Observation tokens get `loss_mask=0`. A `retokenize_chat_history` mode with a custom chat template uses `assistant_masks` instead (lines 654-666).
  - `rollout_logprobs` are carried through.
- **Rewards:**
  - Per-step rewards are placed at each turn's last assistant token. The result is a token-level reward list (`_build_per_token_rewards`, lines 745-781). `GeneratorOutput.rewards` is `List[float] | List[List[float]]` (`base.py:43`).
  - With `step_wise_trajectories`, each turn becomes its own sample with an `is_last_step` flag, and the advantage is propagated from the last step (`config.py:1391`; generator lines 583-598, 712-717).
- **Async:** a fully async trainer with `max_staleness_steps=4` (`config.py:1035`). Timing splits are recorded as `trajectory_time_splits` `{llm, env}`.
- **Failures:** handled in custom generators. Harbor masks agent timeouts and errors, and gives reward 0 on context overflow (`harbor_generator.py:43-212`).

### verl
- **Agent-loop base:** `AgentLoopBase.run(sampling_params, **dataset_fields) -> AgentLoopOutput` (`verl/experimental/agent_loop/agent_loop.py:227,466`).
  - Registration: `@register(name)` (line 487), or a Hydra YAML via `rollout.agent.agent_loop_config_path`. The dataset's `agent_name` column selects the loop; the default is `single_turn_agent` (`verl/trainer/config/rollout/rollout.yaml:258-273`).
- **Output contract** (`agent_loop.py:88-180`):
  - `prompt_ids`, `response_ids`, `response_mask` (1 = LLM token, 0 = tool token), `response_logprobs`, `routed_experts`, `multi_modal_data`.
  - `reward_score: float|None`, which is placed at `rm_scores[-1]`.
  - `num_turns`, `metrics`, `extra_fields`.
- **LLM access:** token-level only. `server_manager.generate(request_id, prompt_ids, sampling_params, image_data…)` returns a `TokenOutput` (`tool_agent_loop.py:244`; `verl/workers/rollout/llm_server.py:43,99`). It has sticky sessions and least-in-flight load balancing.
- **Tokenization:** the "Continuous Token" builder (`ct_merge_assistant_token` / `ct_merge_context_msg`; `agent_loop.py:335-454`; `verl/utils/tokenizer/continuous_token_wiring.py:218`). Config: `tokenization_sanity_check_mode` (`rollout.yaml`).
- **Tools:**
  - `BaseTool` lifecycle: `create(instance_id) -> (id, ToolResponse)`, `execute(instance_id, params) -> (ToolResponse, step_reward, metrics)`, `calc_reward`, `release` (`verl/tools/base_tool.py:24-93`).
  - `ToolResponse` can carry text, image or video, so VLM tools work.
  - Stateless tools use `@function_tool` (`function_tool.py:67`).
  - Config: `multi_turn.tool_config_path` / `function_tool_path`, `max_parallel_calls`, `max_tool_response_length` with a truncate side, `max_assistant_turns` / `max_user_turns`, and parser `format: hermes` (`rollout.yaml:184-240`).
  - `ToolAgentLoop` runs the PENDING → GENERATING → PROCESSING_TOOLS state machine (`tool_agent_loop.py:101-411`). Tool exceptions become error text and the instance is released (`tool_agent_loop.py:443-497`).
- **Reward:** computed in the agent loop worker through a `RewardLoopWorker` or a remote reward manager when `reward_score` is None (`agent_loop.py:984`; `verl/experimental/reward_loop/reward_loop.py:93,157`).
- **Async:** fully async policy with `staleness_threshold` and `partial_rollout` (`verl/experimental/fully_async_policy/README.md:53-149`). One-step-off-policy mode is also available.

### rLLM
- **Primary contract:** `AgentFlow.run(task, config: AgentConfig) -> Episode | Trajectory | None` (sync `run` or async `arun`) (`rllm/types.py:432-455`).
  - `AgentConfig{base_url, model, session_uid, sampling_params}` (line 418).
  - A flow can set `needs_env=True` and receive `env: Sandbox` as a keyword argument.
  - The `@rllm.rollout(name=…)` and `@rllm.evaluator` decorators live in `rllm/eval/rollout_decorator.py:139,206`.
- **Data model:** `Step` carries `prompt_ids`, `response_ids`, `logprobs`, `reward`, `done`, `advantage: list|float`, `weight_version` (`types.py:100-136`). Then `Trajectory{steps, reward, signals}` and `Episode{trajectories, is_correct, termination_reason, artifacts}` (lines 241-330).
- **Token capture:** the rllm-model-gateway proxy, at `/sessions/{id}/v1`, records vLLM `prompt_token_ids`, `completion_token_ids` and logprobs (`rllm-model-gateway/README.md`; `client.py:42-102`). The engine then enriches episodes from these traces, strictly during training: empty token ids trigger a retry (`rllm/engine/agentflow_engine.py:646-705`).
- **Reward:** `Evaluator.evaluate(task, episode) -> EvalOutput{reward, is_correct, signals}` (`types.py:493`; `eval/types.py:24`). The reward is written to each trajectory unless that trajectory already has one.
- **Legacy contract:** `BaseEnv.reset() -> (obs, info)`, `step(action) -> (obs, reward, done, info)`, `from_dict`, `is_multithread_safe` (`rllm/environments/base/base_env.py`), and `BaseAgent.update_from_env/update_from_model` (`rllm/agents/agent.py:40-88`).
- **Workflow:** `run(task, uid)`, plus `TerminationReason` (ENV_DONE, MAX_TURNS, TIMEOUT, MAX_PROMPT/RESPONSE_LENGTH, ERROR), reward shaping (`reward_bonus_coeff`), and MC returns with `gamma` (`rllm/workflows/workflow.py:18-200`).
- **Step-wise advantage:** `stepwise_advantage_mode: broadcast|per_step` (`rllm/trainer/algorithms/config.py:280`; `transform.py:70-100`).
- **Sandbox protocol:** `exec(cmd, timeout, user)`, `upload_file`, `upload_dir`, `close`, `is_alive` (`rllm/sandbox/protocol.py:17-44`). `WarmQueue` prefetches sandboxes and replaces dead ones (`rllm/sandbox/warm_queue.py`).

### AReaL
- **Workflow contract:** `RolloutWorkflow.arun_episode(engine, data) -> dict[str,Tensor] | None | dict[str, InteractionWithTokenLogpReward]` (`areal/api/workflow_api.py:14-56`). Returning `None` rejects the sample (`areal/infra/workflow_executor.py:64`).
- **Tensor dict:** `input_ids`, `logprobs`, `loss_mask`, `versions` (per-token weight version), `turn_ids`, `rewards`, and an optional `is_truncated` / `rollout_reward` (`areal/workflow/multi_turn.py:105-142`).
- **LLM call:** `engine.agenerate(ModelRequest) -> ModelResponse` (`areal/api/engine_api.py:635`).
- **Agent path:** any object with `async run(data, base_url=, http_client=, api_key=) -> float | dict[resp_id, float]` (`workflow_api.py:80-118`). It is wrapped by `OpenAIProxyWorkflow` (`areal/experimental/openai/proxy/workflow.py:102`).
- **Proxy endpoints:** start_session, chat/completions, responses, anthropic messages, **set_reward**, end_session (`proxy/proxy_gateway.py:352-685`).
- **Dense rewards:** the interaction cache supports `set_reward`, `apply_reward_discount(turn_discount)` and `export_interactions(style='concat'|'individual')` (`areal/experimental/openai/client.py:1903-1977`).
  - It also has **per-token rewards** through `set_token_rewards` / `add_token_rewards`, which are shape-checked against the turn's `output_len` (`areal/experimental/openai/cache.py:201-245`). The PPO actor consumes these, including a `token_rewards_as_adv` option (`areal/trainer/ppo/actor.py:205,505`).
- **Staleness:** `max_head_offpolicyness`, `max_concurrent_rollouts`, `request_timeout` (`areal/api/cli_args.py:2835-2918`).
- **v2 Agent Service:** Gateway → Router → DataProxy → Worker microservices (`areal/v2/agent_service/README.md`).

### ROLL
- **Env contract:** GEM `Env.reset(seed) -> (obs, info)` and `step(action:str) -> (obs, reward, terminated, truncated, info)`.
  - The `info` dict carries `metrics` plus a `metrics_agg_mode` such as `{"format_penalty":"mean"}`, and an optional `suffix` (`env/sokoban/env.py:90-135`; `env_manager/traj_env_manager.py:190-216`).
- **Env manager:** `BaseEnvManager.run_rollout_loop/reset/step/make_decision/format_messages/formulate_rollouts` (`env_manager/base_env_manager.py:23-65`).
  - Variants: `TrajEnvManager` (whole trajectory), `StepEnvManager` (per step, GiGPO `state_hash`; `step_env_manager.py:16,~63`), `AgentNativeStepEnvManager` with a `GroupFilter`, plus VL, TIR, geo3k and proxy managers.
- **Token handling:**
  - LLM output tokens are decoded to text for `env.step` (`traj_env_manager.py:191`).
  - Per-turn `prompt_ids` / `response_ids` are concatenated. `response_mask` is set on LLM tokens.
  - The episode score (sum of step rewards) goes on the last token. `step_scores` and `episode_scores` are kept as non-tensor fields (`traj_env_manager.py:307-386`).
- **Workers:** `EnvironmentWorker` Ray actors host up to `max_env_num_per_worker` env managers each, threaded (`environment_worker.py:24-171`; `agentic_config.py:119-170`).
  - Seeding: `group_seed = base + group_id`, so all members of a GRPO group share a seed (`traj_env_manager.py:94-108`).
  - Output goes through a Ray `output_queue`. `group_size_redundancy` over-provisions groups.
- **Advantages:** `adv_estimator: gigpo | step_reinforce` with discounted step returns and `episode_reward_weight` / `step_reward_weight` (`roll/pipeline/agentic/utils.py:60-90,183`; `agentic_config.py:206`).
- **Async:** `async_generation_ratio` (`roll/configs/base_config.py:552`).
- **Config:** `custom_envs:` YAML maps a name to `env_type` and `env_config` (e.g. `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake_gigpo.yaml:154`).

### NeMo-RL
- **Env contract:** `EnvironmentInterface.step(message_log_batch, metadata_batch) -> EnvironmentReturn(observations, metadata, next_stop_strings, rewards: Tensor[B] | dict[name,Tensor], terminateds, answers)`, plus `global_post_process_and_metrics(batch)` (`nemo_rl/environments/interfaces.py`). The interface is **batched** and each env is a Ray actor.
- **Registration:** `ENV_REGISTRY` / `register_env(name, fqn)` / `create_env` (`environments/utils.py:36-144`). Each env actor can run in its own uv venv.
- **Routing:** samples go to envs by `task_name` through `task_to_env` (`experience/rollouts.py:793-890`).
- **Rollout loop:** `run_multi_turn_rollout(policy, batch, tokenizer, task_to_env, max_seq_len, max_rollout_turns)` (`rollouts.py:928`).
  - Observations are tokenized with `tokenizer(...)` and stored as `token_ids` per message.
  - Over-length observations are truncated and the sample is flagged (`rollouts.py:1040-1100`).
  - Rewards accumulate across turns into a scalar or a multi-reward dict (GDPO).
  - Async variants: `run_async_multi_turn_rollout` (1673) and `run_async_nemo_gym_rollout` (2557).
- **Failure taxonomy:** infra failures (retry on another shard) vs data failures (small retry budget) (`experience/failures.py`). Reward penalties and effort shaping live in `rollouts.py:476,2083`.

### slime
- **Sample contract:** `Sample{tokens, response, response_length, loss_mask, rollout_log_probs, reward: float|dict, weight_versions, status (PENDING/COMPLETED/TRUNCATED/ABORTED), metadata, generate_function_path, custom_rm_path, session_id}` (`slime/utils/types.py:107-170`).
- **Hooks:**
  - `--custom-generate-function-path`: `async generate(args, sample, sampling_params) -> Sample | list[Sample]`.
  - `--custom-rm-path`: `async reward_func(args, sample)`.
  - `--group-rm`, `--custom-reward-post-process-path`, `--rollout-function-path`, and dynamic sampling filters (`slime/utils/arguments.py:361-1480`; dispatch in `slime/rollout/sglang_rollout.py:289`).
- **Masking:** env and observation tokens get loss_mask 0, written manually by the generate function (`examples/search-r1/generate_with_search.py:219,243`).
- **Partial rollout:** `--partial-rollout` with `--mask-offpolicy-in-partial-rollout`; aborted samples are recycled (`sglang_rollout.py:295,426`).
- **Other async pieces:** fully async rollout (`fully_async_rollout.py`), staleness metrics (`slime/utils/staleness.py`), external rollout buffer (`slime_plugins/rollout_buffer/`).
- **Agent stack:** async `Sandbox` protocol `exec/write_file/read_file` with an E2B implementation, RPC retries and lifetime limits (`slime/agent/sandbox.py:28-390`). The trajectory builder handles TITO token drift (CLEAN / REALIGN / FORK) for agents behind OpenAI/Anthropic adapters (`slime/agent/trajectory.py:130-216`).

### OpenRLHF
- **Agent contract:** `AgentInstanceBase.reset(states) -> {"observation"}` and `step(states{observation_text, action_text, label, sampling_params}) -> {rewards, scores, environment_feedback, done, sampling_params, extra_logs}` (`openrlhf/utils/agent.py:19-30`).
- **Executor:** `AgentExecutorBase.execute(prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images)` (line 13).
  - `MultiTurnAgentExecutor` tokenizes feedback with `add_special_tokens=False`, appends it, records `action_ranges` (the loss spans), gives feedback tokens logprob 0, sums step rewards, and flags truncation (lines 32-186).
  - The executor is loaded from `--train.agent_func_path` (a .py file exporting `AgentExecutor`) (`openrlhf/trainer/ray/vllm_engine.py:19-27`).
- **Async:** `--train.async_enable`, `--train.partial_rollout_enable`, dynamic filtering on `scores` (`openrlhf/cli/train_ppo_ray.py:296,601,717`).

### TRL
- **`environment_factory`** (a callable, or a dict of name → callable for multiple envs) (`trl/trainer/grpo_trainer.py:266-289,637-716`).
  - The env needs `reset(**kwargs) -> str|None`; a returned string is appended to the last user message.
  - Public methods are exposed as tools, with schemas built from type hints and docstrings.
  - An optional `get_reward() -> float` (sync or async) makes the env the reward source.
  - One instance is created per rollout. Reward funcs receive `environments=`.
- **`rollout_func(prompts, trainer) -> {prompt_ids, completion_ids, logprobs, [env_mask], …}`** (lines 258-265, 2134-2201).
- **Tool loop:** `_tool_call_loop` builds a `tool_mask` with 0 on tool-result tokens (line 1887-2125). Tool exceptions become `{"error": str(e)}` and increment `tool_failure_count`. Async tools run through `asyncio.gather(return_exceptions=True)`. Iterations are capped by `max_tool_calling_iterations` (line 761).
- **Async GRPO:** `max_staleness`, `max_inflight_tasks` (`trl/experimental/async_grpo/async_grpo_config.py:120-358`). Its `_SampleBuilder` token-drift handling mirrors slime's (`async_rollout_worker.py:102-198`).

### prime-rl
- **Env server:** envs are `verifiers.v1` packages run by a separate process, `env-server` (`src/prime_rl/entrypoints/env_server.py`). It has a pooled, elastic `serve_env` with `max_concurrent` and publishes its address to a file.
- **Orchestrator side:** the orchestrator owns the taskset (`vf.load_taskset(...).select`) and sends `task_data` with each request. The server is stateless about data (`src/prime_rl/orchestrator/envs.py:1-137`).
  - Call: `EnvClient.run(task_data, client, model, sampling, on_delta)` streams turn deltas and returns a `WireEpisode`.
  - Startup timeout: 600s.
- **Traces:** each `vf.Trace` is a message graph whose branches carry `token_ids`, `sampled_mask` and `logprobs`; one branch becomes one sample (`orchestrator/trajectories.py`).
- **Rewards:** `trace.reward` is the input, but advantages can be scalar or **per-sampled-token lists** (`orchestrator/algo/routing.py:17-37`).
- **Staleness:** `max_off_policy_steps` cancels stale groups (`orchestrator/dispatcher.py:143,402-437`). A curriculum module provides gates and samplers.

---

## C. Training features and porting candidates

**Features worth supporting, with where each exists:**
- **Per-token reward injection:** AReaL `set_token_rewards` (`cache.py:222`); SkyRL token-level rewards at turn ends (`skyrl_gym_generator.py:745`); prime-rl per-token advantages (`routing.py:17`).
- **Step-wise samples / GiGPO / step-REINFORCE:**
  - SkyRL `step_wise_trajectories` with `is_last_step` and `merge_stepwise_output`.
  - ROLL `StepEnvManager` with `state_hash` and gigpo.
  - rLLM `per_step` / broadcast modes.
  - AReaL `turn_discount` and `individual` export.
- **Multi-reward (GDPO):** NeMo-RL `rewards: dict[name,Tensor]`; slime `reward: dict`.
- **Env pools and parallelism:**
  - ROLL `EnvironmentWorker` (N env managers per Ray actor, group-shared seeds, redundancy).
  - NeMo-RL Ray worker pools per env.
  - SkyRL `max_env_workers` thread pool.
  - prime-rl elastic env-server pool.
  - NeMo-Gym shards.
- **Remote / HTTP env servers:** prime-rl env-server, NeMo-Gym `/run`, OpenEnv (TRL, SkyRL), OpenReward (SkyRL, ROLL, TRL), Atropos (ROLL), Harbor (SkyRL, rLLM, TRL).
- **OpenAI-proxy token capture for black-box agents:** rllm-model-gateway, AReaL proxy gateway (`/rl/set_reward`), OpenRLHF server executor, slime adapters plus drift builder.
- **Partial rollouts and abort/resume:** slime, verl fully_async, OpenRLHF. Per-token `versions` / `weight_versions` for staleness: AReaL, slime, rLLM `Step.weight_version`.
- **Multi-agent:** rLLM multi-trajectory Episodes (solver-judge); slime multi_agent; prime-rl RAE per-agent baselines.
- **VLM envs:** verl `ToolResponse` image/video; SkyRL VLM generator; ROLL `vl_traj_env_manager` and deepeyes; NeMo-RL `vlm`; TRL BrowserGym/CARLA.
- **Sandboxes and code execution:**
  - verl SandboxFusion client.
  - SkyRL-Agent `sandbox_fusion` tool.
  - rLLM `Sandbox` protocol with docker/daytona/modal/local backends and a warm queue.
  - slime E2B.
  - ROLL ROCK.
  - AReaL Daytona.
- **Failure classification:** NeMo-RL infra vs data; Harbor timeout masking (SkyRL); rLLM `TerminationReason`; AReaL return-`None` rejection.
- **Dynamic/group filtering:** slime filter_hub, OpenRLHF `dynamic_filtering`, ROLL `GroupFilter`.
- **Per-turn stop strings:** NeMo-RL `next_stop_strings`.

**Porting candidates (1–3 per codebase; all Apache-2.0 unless noted):**

| Codebase | Component | Purpose |
|---|---|---|
| SkyRL | `skyrl-gym/skyrl_gym/{core.py,envs/base_text_env.py,envs/registration.py,tools/core.py}` | Minimal text-env, ToolGroup and registry contract; easiest bridge target. |
| SkyRL | `skyrl/train/generators/skyrl_gym_generator.py:745-830` | Observation tokenization by chat-template delta, plus turn-end token reward placement. |
| SkyRL | `examples/train_integrations/harbor/harbor_generator.py:43-212` | Step-wise output with timeout/error masking policy. |
| verl | `verl/tools/base_tool.py` + `verl/experimental/agent_loop/tool_agent_loop.py` | Tool lifecycle (create/execute→step reward/release), parallel calls, response truncation, multimodal tool output. |
| verl | `verl/utils/reward_score/sandbox_fusion/utils.py` | SandboxFusion code-exec client with retries and timeouts. |
| verl | `verl/utils/tokenizer/continuous_token_wiring.py` | Model-family-aware incremental tokenization with a sanity check. |
| rLLM | `rllm-model-gateway/` | Session-scoped OpenAI proxy capturing token ids and logprobs; agents need zero changes. |
| rLLM | `rllm/sandbox/{protocol.py,warm_queue.py,backends/}` | Sandbox protocol plus prefetch queue with dead-sandbox replacement. |
| rLLM | `rllm/types.py` (Step/Trajectory/Episode/AgentFlow/Evaluator) | Multi-trajectory episode schema. |
| AReaL | `areal/experimental/openai/cache.py` (`set_token_rewards`, `apply_reward_discount`, `export_interactions`) | Per-turn and per-token reward assignment on proxy-captured interactions. |
| AReaL | `areal/experimental/openai/proxy/proxy_gateway.py` | Session plus `set_reward` HTTP API for external or online agents. |
| AReaL | `areal/v2/agent_service/` | Gateway/Router/DataProxy/Worker agent microservice pattern. |
| ROLL | `roll/pipeline/agentic/environment_worker.py` + `env_manager/traj_env_manager.py` | Env worker actors, group-seeded resets, output queue. |
| ROLL | `roll/pipeline/agentic/env_manager/step_env_manager.py` + `utils.py:60-200` | GiGPO state hashing and step-return computation. |
| ROLL | `roll/pipeline/agentic/env/{sokoban,frozen_lake,webshop}` | Ready GEM envs with format-penalty and metric-aggregation conventions. Note the gym-sokoban and webshop dependency licenses. |
| NeMo-RL | `nemo_rl/environments/interfaces.py` | Batched env contract with multi-reward and per-turn stop strings. |
| NeMo-RL | `nemo_rl/experience/failures.py` | Infra vs data failure taxonomy and retry policy. |
| NeMo-RL | `nemo_rl/environments/nemo_gym.py` | Remote whole-rollout env server bridge with token capture. |
| slime | `slime/agent/trajectory.py` | TITO drift classification (CLEAN/REALIGN/FORK) for proxy-captured multi-turn traces. |
| slime | `slime/agent/sandbox.py` | Async sandbox protocol; E2B with RPC retry and lifetime. |
| slime | `slime/rollout/sglang_rollout.py:289-470` | Partial rollout, abort and recycle. |
| OpenRLHF | `openrlhf/utils/agent.py` | Tiny reset/step agent contract; rewards vs scores split (scores used for filtering). |
| TRL | `environment_factory` logic in `trl/trainer/grpo_trainer.py:637-716,1887-2125` | Methods-as-tools env, `get_reward`, multi-env dispatch, `tool_mask`. |
| TRL | `trl/experimental/async_grpo/openenv_harness.py` | Async harness rollout worker against OpenEnv servers. |
| prime-rl | `src/prime_rl/entrypoints/env_server.py` + `orchestrator/envs.py` | Stateless pooled env server; orchestrator owns tasks; streaming deltas. |
| prime-rl | `orchestrator/algo/{rae.py,routing.py}` | Per-agent baselines; per-token advantage routing over trace graphs. |
| checkpoint-engine (MIT) | `checkpoint_engine/ps.py` | Not env-related. Only useful for fast weight broadcast to remote rollout workers. |

---

## Requirements checklist: what a framework needs so envs plug into all these trainers

| # | Requirement | Needed by |
|---|---|---|
| 1 | **Text-in/text-out step adapter**: `init/reset(prompt or seed) -> obs`, `step(str) -> (obs messages, reward, done, info)`, `close()`. | SkyRL, rLLM (legacy BaseEnv), ROLL (GEM 5-tuple with terminated/truncated), OpenRLHF (dict), slime examples, NeMo-RL (batched) |
| 2 | **Batched step variant** over message logs and metadata; Ray-actor safe and serializable. | NeMo-RL |
| 3 | **Observations as OpenAI-style chat messages** (role user/tool), optionally multimodal content parts. | SkyRL, verl, NeMo-RL, TRL, AReaL, rLLM, slime VLM, ROLL VL |
| 4 | **Raw-string feedback mode** (no chat template; trainer concatenates text). | OpenRLHF, SkyRL `use_conversation_multi_turn=False` |
| 5 | **Tools as callables with JSON schemas** (type hints plus docstring) and an env-owned `get_reward()`; one instance per rollout. | TRL, verl `@function_tool`, SkyRL ToolGroup |
| 6 | **Stateful tool lifecycle** (`create` / `execute → (response, step_reward, metrics)` / `release`). | verl |
| 7 | **OpenAI-compatible endpoint mode**: the env/agent calls a `base_url` per session, and the framework or gateway captures prompt/completion token ids and logprobs. | rLLM, AReaL, prime-rl (verifiers client), slime agent adapters, OpenRLHF server executor, TRL async harness |
| 8 | **Token-in/token-out records**: `prompt_ids`, `response_ids`, `loss/response_mask` (env tokens = 0), `rollout_logprobs` aligned to the mask. | verl, SkyRL, AReaL, slime, OpenRLHF (`action_ranges`), TRL (`env_mask`/`tool_mask`), prime-rl (`sampled_mask`), NeMo-RL (per-message `token_ids`), ROLL |
| 9 | **Tokenization-drift handling** when re-rendering chat templates (prefix check, realign or fork). | verl sanity check, slime, TRL async, rLLM strict enrich |
| 10 | **Scalar trajectory reward** as the baseline output. | All |
| 11 | **Per-step rewards** mappable to turn-end tokens, or to separate step samples with `is_last_step`. | SkyRL, ROLL (`step_scores`, GiGPO), rLLM (`Step.reward`, per_step), verl (`turn_scores`/`tool_rewards` extras), OpenRLHF (summed), NeMo-RL (accumulated) |
| 12 | **Per-token rewards/advantages** aligned to sampled tokens. | AReaL `token_rewards`, prime-rl advantages list, SkyRL `List[List[float]]` |
| 13 | **Named multi-component rewards** (dict) plus a separate filtering "score". | NeMo-RL (GDPO), slime (`reward: dict`), OpenRLHF (`scores`), TRL (multiple reward funcs), rLLM `signals` |
| 14 | **Hashable state key per step** for anchor-state grouping. | ROLL GiGPO (`state_hash`) |
| 15 | **Deterministic seeding / task_data replay** so GRPO group members share the initial state. | ROLL (group_seed), rLLM (seeded tasks), prime-rl (task_data shipped per request), TRL |
| 16 | **Per-turn control knobs**: next-turn stop strings, sampling params, max turns, observation-length truncation. | NeMo-RL, OpenRLHF, verl, SkyRL |
| 17 | **Termination taxonomy**: done vs truncated vs timeout vs error vs context overflow, with mask-or-zero policy; ability to reject a sample (return None). | ROLL, rLLM, SkyRL Harbor, AReaL, NeMo-RL failures, slime `Sample.Status` |
| 18 | **Env-side timeouts and retries** (tool exec, sandbox RPC, HTTP 5xx/429), surfaced as observation errors rather than crashes. | verl, TRL, SkyRL tools, slime E2B, OpenReward bridges, AReaL tau2 `wait_for` |
| 19 | **Async, non-blocking env calls** (async methods, or thread-safe sync calls run in executors); `is_multithread_safe` declaration. | SkyRL, verl, rLLM, AReaL, TRL, OpenRLHF |
| 20 | **Remote env server and worker pool** (HTTP/ZMQ, stateless, elastic, health check, address discovery). | prime-rl, NeMo-Gym, OpenEnv (TRL/SkyRL), Atropos (ROLL), ROLL EnvironmentWorker |
| 21 | **Sandbox provisioning** (exec/upload/read, liveness, warm prefetch, lifetime) behind a backend-agnostic protocol. | rLLM, slime, ROLL ROCK, AReaL Daytona, TRL/SkyRL Harbor, verl SandboxFusion |
| 22 | **Weight/policy version tagging per token or sample**, plus abort/resume for partial rollouts. | AReaL `versions`, slime `weight_versions`, rLLM `weight_version`, verl/OpenRLHF/slime partial rollout, SkyRL/TRL/prime-rl staleness |
| 23 | **Multi-trajectory / multi-agent episodes** with per-agent identity and rewards. | rLLM Episode, prime-rl `trace.agent`, slime multi_agent |
| 24 | **Multimodal observations and tool outputs** (images/video) carried with token alignment. | verl, SkyRL VLM, ROLL deepeyes/geo3k, NeMo-RL vlm, TRL BrowserGym/CARLA, prime-rl |
| 25 | **Registration by string id / FQN** with per-row env selection (`env_class`, `agent_name`, `task_name`, `environment` column) and YAML config. | SkyRL, verl, NeMo-RL, ROLL, TRL (dict factory), rLLM registry, OpenRLHF (file path), slime (function path) |
| 26 | **Env metrics** with an aggregation mode (mean/last/sum) and timing splits (llm vs env). | SkyRL (`get_metrics`/`aggregate_metrics`, time_splits), ROLL (`metrics_agg_mode`), NeMo-RL (`global_post_process_and_metrics`), verl `AgentLoopMetrics` |
