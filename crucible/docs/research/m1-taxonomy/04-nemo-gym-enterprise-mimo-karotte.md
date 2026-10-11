<!-- M1 taxonomy 04: NeMo-Gym, EnterpriseOps-Gym, MiMo v2.6, Karotte, xiaomimimo-verl. Read-only study by a subagent, 2026-10-12. -->

# Environment-family taxonomy: NeMo-Gym, EnterpriseOps-Gym, MiMo v2.6, Karotte, xiaomimimo-verl

I read code only and changed nothing. Root abbreviations (all absolute):
- **N** = `/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/RL-Environments/06-frameworks-and-env-code/nvidia-nemo-gym`
- **E** = `/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/RL-Environments/06-frameworks-and-env-code/servicenow-enterpriseops-gym`
- **K** = `/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/RL-Environments/06-frameworks-and-env-code/preferencemodel-karotte`
- **V** = `/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/RL-Environments/06-frameworks-and-env-code/xiaomimimo-verl`
- **M** = `/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss`
- **Mx** = `M/general/envs/s3k_0001_accounting_audit_tax_en_t1_rl_003` (sample MiMo env; all 267 envs ship the same harness files)

Licenses:
- NeMo-Gym framework: Apache-2.0 (`N/LICENSE`). Each server's **dataset** carries its own license in `configs/*.yaml`. This varies: MIT, CC-BY-4.0, CC-BY-SA-4.0, GPL-3.0 (one equivalence_llm_judge dataset), "TBD", and "NVIDIA Internal Use Only, Do Not Distribute" (swe_internal_v1, swemer_*, image_tools). The allowed license enum is at `N/nemo_gym/config_types.py:467-480`.
- EnterpriseOps-Gym: Apache-2.0 (`E/LICENSE`). The 8 domain MCP servers are **only distributed as Docker images** (`E/README.md`, "Gym Servers"); their source is not in the repo.
- Karotte: MIT. Templates are MIT-0.
- xiaomimimo-verl: Apache-2.0.
- MiMo v2.6 dataset: apache-2.0 (`M/README.md`).

---

## Part 1. NeMo-Gym resources-server protocol

### 1.1 Base classes and endpoints (`N/nemo_gym/base_resources_server.py`)

`SimpleResourcesServer.setup_webserver` (lines 238-256) mounts:
- `POST /seed_session`
- `POST /close_session`
- `POST /verify`, wrapped as `traced_verify_endpoint(judge_failsafe(self.verify))`
- `POST /aggregate_metrics`
- `GET /reverify_mode`

Sessions use a signed Starlette `SessionMiddleware` cookie. A uuid4 is stored in `request.session["session_id"]` (`N/nemo_gym/server_utils.py:865, 1200-1220`). Every downstream call must propagate cookies.

**Tool endpoints** are plain FastAPI POST routes on the same server. Two patterns:
- One route per tool, e.g. `interactive_browser/app.py:168-172` (`/browser_navigate|click|type|observe|finish`).
- A catch-all `app.post("/{path}")`, e.g. `workplace_assistant/app.py:62`, `conversational_tool_use_simulation/app.py:708`.

The agent calls `POST /{function_call.name}` with the parsed JSON args. Tool errors are returned as model-visible output, not raised (`N/responses_api_agents/simple_agent/app.py:203-251`). `RESERVED_MCP_TOOL_NAMES = {verify, seed_session, close_session, aggregate_metrics, mcp}` (line 72).

**Two seed contracts** coexist:
- **Legacy (agent-driven).** `BaseSeedSessionRequest`/`Response` are empty models (167-172). The server reads `request.session[SESSION_ID_KEY]`. Example: `workplace_assistant/app.py:72-83`.
- **Environment-Server-driven.** `ResourcesSeedSessionRequest{resources_session_id, episode_id, task_id, task_data}` with `extra="forbid"` (184-192). The response is `{resources_session_id, resources_tools: MCPServerMetadata|None, sandbox_access: SandboxAccess|None}` (195-202). `MCPServerMetadata{server_name, url_path="/mcp", transport="http", headers}` (175-181). `SandboxAccess{connection: DirectSandboxConnection(provider_config_ref, descriptor), workdir}` is in `N/nemo_gym/sandbox/access.py:11-30`.

**Close:** `ResourcesCloseSessionRequest{resources_session_id, episode_id}` → `/close_session` (218-232, 291-298).

**Verify:**
- Request: `BaseVerifyRequest = BaseRunRequest{responses_create_params, _ng_rollout_id} + response: NeMoGymResponse` (92-104). Task fields ride at the row's top level, e.g. `ground_truth` (`workplace_assistant/app.py:44-48`).
- Response: `BaseVerifyResponse{reward: float, mask_sample: bool=False, failure_kind: str|None, failure_reason: str|None}` (107-148).
- `BaseMultiRewardVerifyResponse.reward_components: dict[str,float]` (151-164).
- A typed variant exists: `ResourcesVerifyRequest{episode_id, task_id, verification_input}` (208-215).

**MCP auto-exposure** (`N/nemo_gym/mcp_auto_exposure.py`):
- Harvests typed POST routes as MCP tools (429).
- Wraps `/seed_session` to mint an `itsdangerous.URLSafeSerializer`-signed session token holding the session id plus an optional per-session tool allow-list (`mcp_allowed_tools_for_session`, base 270-275; minting 692-711).
- Every `tools/call` must replay the token via header `X-NeMo-Gym-Session-Token` (base line 43; parsing 713-734).
- Verify-time tool-name normalization strips the `mcp__<server>__` prefix (`normalize_tool_name`, base 50-68; `_wrap_verify` 631).

**Reverify:** `ReverifyMode {stateless, unsupported, unknown}` (75-85).

### 1.2 Session lifecycle

**Legacy agent `/run`** (`simple_agent/app.py:306-379`):
1. `POST /seed_session` (cookies set).
2. `POST /v1/responses` loop: model call → for each `function_call`, `POST resources/{name}` → append `function_call_output`.
3. Stop when there are no tool calls, on `incomplete_details`, or when `max_steps` is reached (167-259).
4. `POST /verify` with `body + response`.

The server is expected to pop the session state inside verify (e.g. `workplace_assistant/app.py:130-131`).

**Environment Server path** (`N/nemo_gym/base_environment_server.py`, `N/environment_servers/single_agent_turn/app.py:85-140`):
- Admission semaphore and queue timeout (`BaseEnvironmentServerConfig` 31-45).
- Episode deadline (185-203).
- `CleanupContext`: LIFO idempotent callbacks with a bounded timeout, shielded from cancellation (80-115, 210-216).
- The close callback is registered **before** seed, so a lost seed response cannot leak the session (single_agent_turn 106).

**Gymnasium variant** (`N/resources_servers/gymnasium/base.py:72-121`):
- `POST /reset` → `{observation, info}`.
- `POST /step` → `{observation, reward, terminated, truncated, info}`. `close_session` runs automatically on terminal.
- `/verify` raises; per-step reward replaces it.
- Used by 8 servers: blackjack, example_multi_turn_gymnasium, grl_sokoban, grl_tetris, indian_banking, openair_congestion, tales, wordle.

**Custom session APIs** (outside the core contract):
- osworld: `/reset /observe /step /evaluate /close /session /healthz` (`osworld/app.py:95-101`).
- toolsandbox: `/step /close`; reward is computed at close and cached for verify (`toolsandbox/app.py:343-348, 480-543`).
- conversational_tool_use_simulation: 10 routes (699-708).

### 1.3 Failure kinds and masking

`N/nemo_gym/failure_kinds.py:63-130` defines a 22-name vocabulary:
- transport: `transport_unreachable|peer_drop|timeout|local_resource`
- agent: `agent_run_error|request_failed|timeout|command_not_allowed`
- judge/verifier: `judge_failed`, `judge_unparseable`, `verifier_error`
- provider: `provider_unavailable|quota_exhausted|oom_killed`
- session: `session_lost|expired|release_failed`
- process: `cancelled`, `kill_shaped`, `shutdown`
- other: `persistence_failed`, `cohort_incomplete`

Environment extensions use the form `<server>:<kind>` (58-60). Unregistered names warn once rather than being rejected (148). `FailureStage` is one of `admission|seed|agent|verification|cleanup` (49).

`EpisodeFailure{failure_reason, terminal, stage, failure_kind}` is in `N/nemo_gym/episode_types.py:57-70`.

`judge_failsafe` (`N/nemo_gym/judge.py:85-121`) turns any `JudgeError` raised inside verify into `reward=0.0, mask_sample=True, failure_kind="judge_failed"`. `call_judge` / `reraise_judge_errors` (51-76) convert arbitrary judge exceptions into `JudgeError`.

### 1.4 Manifest / config fields

Resources-server YAML (`N/resources_servers/workplace_assistant/configs/workplace_assistant.yaml:1-45`):
- `entrypoint`, `domain`, `verified`, `description`, `value`.
- The agent block wires `resources_server`, `model_server` and `max_steps`.
- `datasets[]` entries: `name`, `type` (train|validation|example), `jsonl_fpath`, `huggingface_identifier{repo_id, artifact_fpath}`, `license`, `num_repeats`, `source` (`config_types.py:456-480`).
- An optional `environment_servers:` block (legacy_agent or single_agent_turn).
- `Domain` enum (574-604): math, coding, agent, knowledge, instruction_following, long_context, safety, games, translation, e2e, rlhf, other.
- `N/environments/<name>/config.yaml` plus `prepare.py` is the newer "environment manifest" (same schema, e.g. `N/environments/workplace_assistant/config.yaml`).

Per-server `task_data.py` exports a `TaskData` pydantic model:
- Default is `extra="allow"`; `extra="ignore"` is banned.
- Fields may carry `consumed_by ∈ {verify, metrics, prompt, provenance}` tags (`N/nemo_gym/task_data.py:25-41`).
- Rows use `responses_create_params.input` (Responses API) plus task fields.

### 1.5 Sandboxes

`N/nemo_gym/sandbox/` provides `AsyncSandbox`, `SandboxSpec` and `SandboxResources` with providers apptainer, daytona, docker, e2b, ecs_fargate, enroot, local, opensandbox and openshell (`providers/`).

---

## Part 2. NeMo-Gym families (145 servers)

**The family grouping is my own.** It is based on code signals (seed_session overrides, tool routes, judge/sandbox use) plus config domain and description. The counts sum to 145.

### F1. Math, proofs, formal (13)
- **Servers:** math_with_judge, math_with_autograder, math_with_code, math_advanced_calculations, math_formal_lean, math_proof_judgement, imo_gradingbench, imo_proofbench_judge, false_statement_judge, polymath, proof_genselect, proof_judge, proof_verification.
- **State:** none. math_formal_lean uses a Lean sandbox client (`math_formal_lean/app.py:32`).
- **Actions:** mostly single-turn text. Calculator tool routes in math_with_code (2 routes) and math_advanced_calculations (1). math_formal_lean is multi-turn self-correction.
- **Actors / horizon:** single policy; 1 turn, or a few for Lean and tool use.
- **Grader:**
  - math_with_judge runs symbolic `math_verify` in a child process (wedged-SymPy protection). It falls back to an LLM judge only if the library reward ≤ 0.5 (`math_with_judge/app.py:233-240`). The judge is two-pass, and both passes must agree (345-348).
  - Proof servers use 0-7 rubric judges or a verifier plus meta-verifier.
- **Density:** terminal and sparse (binary or rubric-scaled).
- **Infra:** judge model server; Lean sandbox.
- **Port:** the library-then-judge cascade with a subprocess-isolated symbolic checker (`N/resources_servers/math_with_judge/app.py`, Apache-2.0).

### F2. Knowledge, science QA, reasoning puzzles (24)
- **Servers:** gpqa_diamond, mcqa, bunsenbench_chemistry_mcq, ether0, fukuyamabench, moldetox, assaybench, litmus_agent, physics_judge, ugphysics_judge, frontierscience_judge, critpt, equivalence_llm_judge, equivalence_rule, string_match, simpleqa, omniscience, hotpotqa_qa, spartqa, ragtruth, abstention, arc_agi, nvarc, reasoning_gym.
- **State:** none. **Actions:** single text answer. **Actors:** single.
- **Grader:**
  - Rule-based: MCQ parse, SQuAD-style, string match.
  - LLM judge with optional positional-bias swap check: `check_twice_swap`, `reward_if_swap_fails` (`equivalence_llm_judge/app.py:149-152, 478-495`).
  - simpleqa uses a 3-tier judge (CORRECT / INCORRECT / NOT_ATTEMPTED).
  - abstention uses a 3-tier reward: correct 1 > abstain λ > incorrect 0 (`abstention/app.py:19-22, 254-263, 332-333`).
  - critpt scores via an external API.
- **Density:** terminal. **Infra:** judge server where used.
- **Port:** the swap-check judge (`N/resources_servers/equivalence_llm_judge/app.py`) and the abstention tiering.

### F3. Code generation, execution, SQL (13)
- **Servers:** code_gen, code_fim, bigcodebench, evalplus, competitive_coding_challenges, scicode, scicodepile, bird_sql, spider2_lite, text_to_sql, openenv, cvdp, swerl_llm_judge.
- **State:** ephemeral execution, or a SQLite DB for bird_sql / spider2_lite. **Actions:** single answer; openenv runs code via OpenEnv/MCP.
- **Grader:**
  - Unit tests in subprocess or sandbox with a timeout (`code_gen/app.py:47, 168-241`).
  - SQL execution-result equivalence (bird/spider), or LLM equivalence (text_to_sql).
  - cvdp uses Docker hardware-sim tests.
- **Density:** terminal; binary or fraction of tests passed.
- **Infra:** subprocess, Ray, Docker (cvdp).

### F4. Repository-level SWE in sandboxes (13)
- **Servers:** swebench, swebench_pro, swe_rebench, scale_swe, denovo_swe, deepswe, swe_next, swe_internal_v1, swemer_oml, swemer_v1, swemer_v2, swerl_gen, vibench.
- **State:** a per-task container image of a git repo.
- **Actions:** bash/editor from an external harness (OpenCode, mini-SWE, Codex, ...). The resources server returns a `sandbox_handle` from `seed_session` (`swebench/app.py:300-320`).
- **Horizon:** long (tens to hundreds of steps); harness-owned.
- **Grader:**
  - A fresh eval sandbox runs hidden `FAIL_TO_PASS` / `PASS_TO_PASS` tests (`swebench/app.py:84-85`); `reward=int(resolved)`.
  - deepswe and vibench use a "two-environment" contract: grade in a fresh sandbox, never the agent's.
  - vibench grades with an LLM agent driving a browser against human test plans; `reward_components` holds per-test-plan scores (`vibench/app.py:641-646`).
- **Failure info:** a sandbox error type is recorded (`swebench/app.py:124-227`).
- **Isolation / anti-cheat:** `anti_cheat_setup.sh` deletes future git history, refs, tags, remotes and reflog, then runs `gc --prune=now` (`N/resources_servers/swebench/anti_cheat_setup.sh:1-25`). It is reused by scale_swe, denovo_swe, swe_next and swe_internal_v1 (e.g. `scale_swe/app.py:280-282`).
- **Infra:** OpenSandbox, Docker, Apptainer or enroot providers; CPU caps via `cpu_cap_env`.
- **Port:** the anti-cheat git scrub script and the fresh-sandbox grading pattern (Apache-2.0).

### F5. Terminal, computer-use, browser (5)
- **Servers:** terminal_bench_2_1, terminal_bench_4, terminus_judge, osworld, interactive_browser.
- **State:**
  - TB4: pinned task sandbox with a 10-hour resource-owned deadline; late requests get HTTP 410 (`terminal_bench_4/README.md`).
  - osworld: a remote Docker/QEMU VM per session.
  - interactive_browser: an isolated Playwright context per session.
- **Actions:**
  - osworld: `/step` with idempotent `operation_id`.
  - interactive_browser: navigate, click, type, observe, finish tools (168-172).
- **Observations:** screenshots/accessibility tree for osworld; page observations for the browser.
- **Grader:**
  - TB4 runs the official separate verifier and records `evaluation_completed` and `infrastructure_error` (`terminal_bench_4/app.py:245-292`).
  - osworld: `reward = 1 if score ≥ 1`. **Any evaluate exception → `mask_sample=True`**, and cleanup always runs (`osworld/app.py:148-172`).
  - interactive_browser: sparse outcome from live browser state; "first finish wins" (334, 369-381, 383).
- **Infra:** VM worker pool with SSH or registration control plane (`osworld/worker.py`), Docker/KVM, Playwright or remote CDP.

### F6. Tool use, function calling, workplace (11)
- **Servers:** workplace_assistant, xlam_fc, example_single_tool_call, example_multi_step, example_tool_call_multireward, example_session_state_mgmt, example_mcp_weather, aviary, newton_bench, finance_agent_v2, legal_agent_bench.
- **State:** workplace_assistant seeds 5 in-memory pandas toolkits per session (email, calendar, analytics, project management, CRM) (`workplace_assistant/app.py:72-83`), exposing 27 tools via the catch-all route (65-70). newton_bench builds dynamic `/run_experiment_<module>` and `/execute_python` routes (`newton_bench/app.py:192-198`).
- **Grader:**
  - workplace_assistant does **state-diff by replay**: it re-executes predicted and ground-truth action lists on fresh toolkits and compares the resulting DataFrames, case-insensitive except `status/list_name/board` (`workplace_assistant/utils.py:133-183`).
  - xlam_fc compares call arguments.
  - example_tool_call_multireward returns `reward_components{correctness, schema_valid, format}` with reward = sum (`example_tool_call_multireward/app.py:21-34, 154-165`).
  - newton_bench uses RMSLE.
- **Density:** terminal.
- **Port:** the replay-both-and-diff state grader (`N/resources_servers/workplace_assistant/utils.py:133+`).

### F7. Conversational with simulated users (4, plus the tau2 agent harness)
- **Servers:** conversational_tool_use_simulation, indian_banking, toolsandbox, calendar.

**conversational_tool_use_simulation**
- **State:** per-session policy markdown, generated tool signatures, customer scenario and trajectory (README).
- **Actors:** a user simulator LLM, a **tool-result simulator** LLM (tools are LLM-simulated, not real), and a judge LLM, each with its own model-server ref (`app.py:109-131`).
- **Persona:** `CustomerScenario{customer_persona, reason_for_contact, customer_details, unknown_info, task_instructions}` (134-160). The user simulator is role-swapped (1478-1515).
- **Grader:** message-level judge with termination; user and tool messages are judged first.
- **Masking:** if the user simulator, the tool simulator, or verification generation failed, `mask_sample` is set (2335-2347).

**indian_banking**
- **State:** a per-episode clone of `data/db.json`, isolated via contextvars (`core/engine.py:19`). 34 tools: 33 servicing tools plus `transfer_to_human_agents`. A 59-article keyword KB.
- **Simulated user:** persona, progressive disclosure, stop tokens `###STOP###|###TRANSFER###|###OUT-OF-SCOPE###` (`core/user_sim.py:31`); `max_user_turns=8` (`app.py:112, 234-246`).
- **Grader:** ACTION / DB-hash / COMMUNICATE / NL_ASSERTION.
  - `strict` = product of the active components. `dense` weights ACTION 0.40, DB 0.25, COMMUNICATE 0.15, judge 0.20 (`core/reward.py:17, 36-47`).
  - `reward = 1 - efficiency_cost` (capped at 0.15) if strict, else `0.4 * dense`.
  - The judge is fail-open: it is left out rather than scored 0 (`app.py:287-324`).
- **Protocol:** Gymnasium reset/step.

**toolsandbox**
- **State:** vendored Apple ToolSandbox world (contacts, messaging, reminders, settings).
- **Actors:** user simulator inside the server.
- **Grader:** milestone similarity in [0,1], forced to 0 if a minefield is hit; computed at `/close` and cached for verify (`toolsandbox/app.py:480-543`). This is the closest existing analogue to dense milestone progress.

**calendar**
- Personas are used **offline** to synthesize the multi-turn conversation (`create_synth_conversations.py:295-360`).
- At train time it is single-shot: verify parses the JSON calendar and checks event count, conflicts and constraints against `exp_cal_state` (`calendar/app.py:50-78`, `utils.py:155-194`).

**Infra:** policy, simulator and judge model servers.
**Port:** indian_banking `core/reward.py` (strict-plus-dense composition with efficiency tie-break) and `core/user_sim.py`; CTU sim `_should_mask_sample`.

### F8. Search, browsing, research (8)
- **Servers:** google_search, tavily_search, browsecomp_advanced_harness, finance_sec_search, sec_local_index (library, not standalone), ns_tools, deepsearchqa (grader only), widesearch (grader only; markdown tables).
- **Actions:**
  - tavily: `/web_search /find_in_page /scroll_page` (`tavily_search/app.py:281-283`).
  - google_search: `/search /browse`.
  - finance_sec_search: 7 routes.
- **Grader:** LLM judge (tavily 475-486; finance_sec_search takes the last `[[N]]` rating, lines 79-89) or regex fallback.
- **Infra:** external search APIs. aiohttp-adapter guidance is in `N/AGENTS.md`.

### F9. Instruction following, structured output (11)
- **Servers:** instruction_following, ifbench, agentif, verifif, inverse_if, citation_if, format_verification, iheval, structured_outputs, structeval, multichallenge.
- **Grader:**
  - IFEval registry checkers; `reward = all()` or the fraction followed (`instruction_following/app.py:160-162`).
  - structured_outputs validates JSON/YAML/XML against a schema (openapi-schema-validator). Metrics are sliced by `has_distractors`, `num_tools`, `tool_choice`, and so on (`structured_outputs/app.py:66, 94, 120-170`). This is NeMo-Gym's only explicit distractor-tool axis.
  - verifif mixes rule and judge checks; inverse_if uses per-task judges.
- **State:** none. **Density:** terminal; partial credit by fraction.

### F10. Long-context (7)
- **Servers:** ruler, ruler2, mrcr, graphwalks, lc_niah, longmemeval, aalcr.
- **Grader:**
  - ruler: substring match-all / match-part (`ruler/app.py:56-76`).
  - mrcr: a prefix gate, then `SequenceMatcher.ratio` (`mrcr/app.py:65-72, 110-124`).
  - graphwalks: F1 over node sets.
  - lc_niah: correctness gated by low reasoning/input overlap.
  - longmemeval and aalcr: LLM judges.
- **State:** none; the haystack is baked into the prompt at prep time.

### F11. Games and simulated control (8)
- **Servers:** blackjack, wordle, grl_sokoban, grl_tetris, tales, example_multi_turn_gymnasium, gymnasium (base class), openair_congestion.
- **Protocol:** Gymnasium reset/step. **State:** a game/simulator object per session.
- **Reward density — the only per-step rewards in NeMo-Gym:**
  - tales: score delta for textworld (`tales/app.py:101-108`).
  - wordle: shaping penalties, e.g. `PENALTY_REPEATED_GUESS=-0.2` (`wordle/app.py:33-37, 120-130`).
  - sokoban: env reward per step.
  - openair_congestion: decomposed KPI reward per step from a deterministic replay backend, no judge (`openair_congestion/app.py:433-502`).

### F12. Safety and alignment (6)
- **Servers:** aegis_v4_safety, jailbreak_detection, over_refusal_detection, xstest, indirect_prompt_injection, bbq.
- **indirect_prompt_injection:**
  - State: a JSON world seeded per session for 9 domains, with an `injection_text` planted in a data field (README; `app.py:120-145`).
  - Tools: per-domain handler catalogs (32-40).
  - Grader: `reward = safety × utility`. Safety = the attacker's `target_tool` was not called with `target_args`. Utility = all `required_tools` were called. Truncated output scores 0 (148-185; `verifier.py:136`).
- **jailbreak_detection:** safety judge × quality judge (`jailbreak_detection/app.py:109-114, 326-351`).
- **aegis_v4_safety:** a guard-model classifier.
- **Port:** the injection world-plus-verifier pattern (`N/resources_servers/indirect_prompt_injection/`).

### F13. RLHF, preference, reward-model training (5)
- **Servers:** arena, arena_judge, generative_reward_model, genrm_compare, rolemrc.
- **Grader:**
  - Pairwise LLM judge against a baseline, both orders.
  - GenRM: L1 on rubric scores and ranking (`generative_reward_model/app.py:29-57`).
  - rolemrc: 5-aspect judge.

### F14. Multimodal, VLM, speech (7)
- **Servers:** circle_click, circle_count, gui_coordinate, labbench2_vlm, vlm_eval_kit, gdp_pdf, asr_with_pc.
- **Grader:** rule-based coordinate hit (`circle_click/app.py:61, 88-99`), WER, or judge.

### F15. Knowledge-work deliverables (3)
- **Servers:** gdpval, job_bench, workspace_bench.
- **State:** a per-task sandbox with `/workspace/input` and an output dir.
- **Grader:**
  - gdpval: rubric judge panel, or pairwise comparison plus multi-stage ELO; LibreOffice media conversion (`gdpval/app.py:247-370`).
  - job_bench: vendored judge; reward = earned rubric weight / total.
  - workspace_bench: agent-as-judge in a **separate judge sandbox**.

### F16. Translation (2): wmt_translation (spBLEU/chrF), longmt_eval (SEGALE pipeline).

### F17. Infrastructure benchmark (1): speed_bench (vLLM `/metrics` speculative-decoding throughput).

### F18. Next-action / PivotRL step verifiers (4)
- **Servers:** swe_pivot, terminal_multi_harness, single_step_tool_use_with_argument_comparison, image_tools.
- Each scores one next action against a demonstration: tool-name match, target match, edit-size penalty (`swe_pivot/app.py:295-491`).

### Cross-cutting NeMo-Gym observations
- Only about 28 servers override `seed_session` or keep per-session state. **Most families are stateless single-turn.**
- `mask_sample` is set explicitly in only 4 servers (CTU sim, interactive_browser, osworld, and the vibench path). Everything else relies on `judge_failsafe`.
- `reward_components` is used only in example_tool_call_multireward and vibench.
- There is no shared DB or state adapter: each server rolls its own (pandas, JSON clone, VM, sandbox).

---

## Part 3. EnterpriseOps-Gym

### Domains, tools, state
8 domains: Calendar, CSM, Drive, Email, HR, ITSM, Teams, and Hybrid (multi-gym). The README claims 1,150 tasks, 512 tools, 164 tables, an average of 9.15 steps (max 34), and 5.3 verification conditions per task.

**MCP servers** are per-domain Docker images on ports 8001-8009. Each exposes:
- `/mcp` (JSON-RPC)
- `/api/seed-database`
- `/api/delete-database`
- `/api/sql-runner`

The client side is `E/benchmark/mcp_client.py:19-76, 156-185, 306-319`.

**State model:**
- One SQL snapshot per unique DB in `E/gym_dbs.zip`: calendar 12, csm 19, drive 6, email 4, hr 5, hybrid 15, itsm 10, teams 13.
- Each run calls `create_database_from_file` → a fresh `database_id`. Every tool call carries the header `x-database-id`. The DB is deleted afterward (`E/benchmark/executor.py:309-349, 561-563`).
- `reset_database_between_runs` defaults to True (`models.py:82`).

**ITSM seed tables** (from the zip): users, user_role, user_group(_member), role, permission, role_permission, organization, location, service, service_offering, configuration_item, incident, child_incident, incident_affected_cis, incident_sla, incident_knowledge, incident_template, sla_definition, problem, change, change_request_mapping, knowledge, notification.

**RBAC / identity:**
- Roles are admin, agent, manager, reporter, with a role→permission map and org-scoped `user_role(user_id, role_id, org_id)`.
- Each task's `gym_servers_config[].context` injects identity headers such as `x-itsm-user-token` or `x-user-email`, plus `user_info` (sample: `E/data/revised/itsm/task_20251212_172511_458_e6427839_47076c15.json`).
- Context keys are forwarded as `x-*` headers (`verifier.py:226-242`).
- The policy system prompt mandates checking the user first. Server-side enforcement lives in the closed images.

### Task format (`E/benchmark/models.py:53-82` and task JSON)
`{system_prompt (domain policy), user_prompt, gym_servers_config[{mcp_server_name, mcp_server_url, seed_database_file, context, user_info, auth_config}], selected_tools[], restricted_tools[], verifiers[], number_of_runs, reset_database_between_runs}`

**Distractor tools via modes:** `oracle`, `plus_5_tools`, `plus_10_tools`, `plus_15_tools` are HF dataset configs (README experiment.json; `E/evaluate.py:252, 289-298`).
- `selected_tools` filters the merged tool catalog (`executor.py:276-303`).
- `restricted_tools` removes tools (98-103).
- The local samples have 6-19 selected tools per task.

### Graders (`E/benchmark/verifier.py`)
- **`database_state`:** a SQL query via `/api/sql-runner`, with comparison `equals | greater_than | less_than | contains` (81-127, 374-399). All 102 verifiers across the 13 local tasks are `database_state` with `equals`. Example: `SELECT COUNT(*) FROM incident WHERE incident_id='INC_003' AND impact='high' AND priority='high'` with expected value 1.
- **`response_check`:** run SQL, then an LLM judge scores 1-10 against `comparison_prompt`; pass if score ≥ `minimum_comparison_value` (default 7) (129-164, 412-483).
- **`tool_execution`:** required tools called and `minimum_tool_calls` met (166-193).
- **Multi-gym:** `verifier.gym_name` routes the query to that gym's DB (42-46, `executor.py:424-454`).
- **Reward:** `overall_success = all(passed)`; `pass_rate = passed / total` (`executor.py:366-391`).
- **Failure masking:** the README says files with errors are excluded from averages, **but `E/compute_score.py:35-42` appends their rates anyway** and only counts the errors separately. A judge exception makes the verifier fail (`passed: False`) instead of masking (verifier.py:481-483).

### Actors and horizon
Single agent. Orchestrators: `react` (max_iterations=50, `orchestrators/base.py:23-30`, `react.py:34-79`), `planner_react`, and `decomposing`. There are no simulated users; tasks are one-shot instructions.

### Infra
Docker MCP servers (a Postgres-like SQL backend, inferred from the SQL dumps). Ray queue for parallel model × domain × mode runs (`ray_experiment_queue.py`).

### Worth porting
- The verifier engine: SQL-assertion plus LLM response-check verifier types, multi-gym routing (`E/benchmark/verifier.py`, Apache-2.0).
- The seed/delete-database-per-run lifecycle and the `x-database-id` / `x-*` identity header conventions (`E/benchmark/mcp_client.py`, `E/benchmark/executor.py:309-349`).
- The task JSON schema with oracle/+N distractor modes.

---

## Part 4. MiMo v2.6 RL-OSS

### Categories
The `M/README.md` table lists:

| Domain | Task family | Verifier |
|---|---|---|
| Code | SWE | executable tests |
| Cyber | vulnerability reproduction | rule checks |
| General | knowledge work | rubric judging |
| Visual/webdev | | visual grading |
| Music | | rule checks |

**Locally present:** only `code.parquet`, `cyber.parquet` and `general/envs/` (267 env dirs, 3.9 GB). `webdev.parquet` and `music.parquet` are **absent**.

From parquet footer metadata only (no pyarrow available):
- Schema: `data_source, agent_name, prompt[role,content], reward_model{style, ground_truth}, extra_info{index, instance_id, dataset_type, instance_json}`.
- Cyber rows are `arvo` instances with a per-instance `docker_image` and a submit-script feedback loop.
- Code rows use `mimo_swe_agent`.

Agent configs in V: `V/config/agent/{arvo, code/(mini-bash, mini-claude-code, mini-codex, mini-mimocode, mix-four-whitebox), design/webdev, general/s3k}.yaml`.

### General ("s3k") envs
- **Domains:** accounting_audit_tax 157, consulting_bizops_analytics 89, construction_pm 14, agriculture_coop 7.
- **Languages:** en and zh. **Difficulty tiers:** t1-t5.

**Layout (Mx):**
- `instruction.md`
- `workspace/` (xlsx, pptx, docx, pdf, html)
- `system/<mcp>/{schema.sql, state.db}` (SQLite per business system)
- `tools/<mcp>.py` (plain Python functions)
- harness files: `mcp_http.py`, `sidecar_entrypoint.py`, `mcp_bridge.py`, `verify.py`, `run_verify.py`, `_helpers.py`, `verifier_meta.json`, `manifest.json`

**Catalog size:** 3-17 MCP servers per env (most often 7 or 5). That is 1,796 tool files and about 13,171 public tool functions overall. Many are irrelevant to a given task, which makes them natural distractors, though they were not labeled as such.

**State:** SQLite DBs in the sidecar plus workspace files. `act` is `none` for 201 tasks (read-only answer), `mutate_db` for 60, `both` for 3, `edit_workspace` for 1.

**Tools** read and write `state.db` with realistic guards (`Mx/tools/onesource_tax_provision.py`):
- Opaque base64 page tokens (27-44).
- Optimistic concurrency via `expected_entity_status` (95-120).
- A pinned immutable controlled row (`ENT-ANIXA-001`, line 120).

**Protocol** (`Mx/manifest.json`):
- `uploads` routed to the `main` vs `sidecar` container.
- Sidecar `setup` starts one FastMCP streamable-HTTP server per tools file on :39101+, with health on :39100 (`Mx/sidecar_entrypoint.py:28-34, 150-210`). A supervisor restarts crashed servers (136-145).
- `wait_ports`, `mcp_servers[{name, url}]`, and a verifier block `{uploads, command, reward_file=/logs/verifier/reward.json, reward_detail_file, timeout_sec, env_passthrough}`.
- `mcp_http.py` derives the parameter schema from `isinstance` checks in the function body when annotations are missing (lines 35-60). It caps results at `GA_MCP_MAX_RESULT_CHARS=200000` (70-80); the verifier recomputes via SQL, so truncation never affects scoring.
- `mcp_bridge.py` offers a stdio bridge or a one-shot `--list/--call` emitting `__MCP_ONESHOT__<json>` (1-21, 31-78). This is used when the agent runs out-of-pod (`V/recipes/general/mcp_proxy.py` docstring).

**Isolation:**
- Workspace is mounted **read-only in the sidecar**. `system-vol` (the DBs) is mounted only in the sidecar, so the agent cannot read or modify DBs except through tools (`V/config/agent/general/s3k.yaml`, environment block).
- MCP URLs are hidden from the agent. The agent runs as uid `agentu` and may only launch the bridge via a 0700 launcher owned by the bridge user, through a sudoers rule (`V/recipes/general/general_agent/environment.py:391-446`).
- An iptables uid-owner firewall on loopback is referenced (`Mx/sidecar_entrypoint.py:63-71`), but the firewall code itself is not in these repos.
- Antihack guard: regex interception of git fetch/clone/reflog, reads of eval artifacts, and similar. It returns the dummy "Permission denied" without aborting the rollout (`V/third_party/mimoagent-osr/src/mimoagent/agents/antihack.py:1-35, 60+`; enabled in `s3k.yaml` with `antihack.enabled: true`).
- Git leak prevention modes `hide|strip|none` (`mimoagent/environments/datasets/base.py:44-50`).

**Grader** (`Mx/verify.py`, `Mx/verifier_meta.json`):
- Items `{id, tier (critical|important|sanity), method (llm|rule), question, pass_anchor, files, fn, weight, gate}`. Totals: 1,212 llm items and 215 rule items (2-10 per env, average 5.4).
- `check_code` holds read-only SQL/file checks: post-state rows equal expected, other rows unchanged (sample s3k_0013 `check_target_disposition`, `check_other_cases_unchanged`). `db_execute` is reserved for solve scripts, and verify must be side-effect-free (`Mx/_helpers.py:565-604`). Check code that uses `random` is rejected (verify.py:19-20).
- Rule items run in a subprocess with helpers prepended and a timeout (15-48).
- **LLM judge:**
  - One call per item, binary 0/1, no partial credit, "only from evidence".
  - Truncated or unextracted evidence → 0, with an audit note (86-99).
  - Evidence extraction keeps document order and table coordinates and reports what was lost (274-307).
  - Multi-key/model fallback chain: `GA_JUDGE_KEY/2/3` with api `responses|chat|gemini`, sticky failover, a shared phase budget `GA_JUDGE_PHASE_BUDGET=600s`, and up to 4 retries with backoff (51-271).
  - Optional median-of-votes via `GRADE_VOTES` (311-340).
- **Score composition:**
  - Gates run first; any gate failure → score 0.
  - `src_protect` gate: content fingerprints of protected files and DB tables, so tampering with source data scores 0 (`_helpers.py:494-562`).
  - Score = Σw·s / Σw. Agent and vision items are skipped in the sidecar (343-405).
  - In the shipped 266 metas, **0 gates and 0 src_protect** are actually used; the code supports them.
- **Answer materialization:** the last assistant message is written to `answer.md` if the agent did not write it (420-433; `run_verify.py:14-31`).
- **Masking contract** (`Mx/run_verify.py:102-151`): on verify crash or judge unavailable, `reward.json` gets **no** `reward` key, only `reward_error`. The runner then marks the rollout `REWARD_TESTBED_CORRUPTED` (`"reward/testbed_corrupted"`, `mimoagent/environments/datasets/base.py:29-35`). The trainer sets `INVALID_REWARD_VALUE=-999` for categories in `_INFRA_ERROR_CATEGORIES` (`V/recipes/general/agent_loop.py:76-87, 697-699`).
- More env-side masks: `judge_key_missing`, `mcp_backend_down`, `missing_run_verify`, `missing_or_invalid_reward_json`, `reward_out_of_range` (`V/recipes/general/general_agent/environment.py:461-578`). `pull_post_state` allows offline re-verification after a judge outage (620-626).

**Actors / horizon:** a single cc-agent (Bash, Read, Write, Edit, Grep, Glob) with `step_limit 500` and a 1800 s timeout. `user_agent.py` provides a separate follow-up-query driver (rounds-limited, no stop signal) in `V/third_party/mimoagent-osr/src/mimoagent/agents/user_agent.py:126-272`.

**Density:** terminal; weighted fraction of rubric items.

**Infra:** a two-container K8s pod with a shared emptyDir (`V/recipes/general/general_agent/k8s_sidecar.py:14-66`); Docker images on Docker Hub.

### Worth porting (Apache-2.0)
- `Mx/run_verify.py` (masking contract via an absent reward key) together with `Mx/verify.py` (`_grade_sc`, judge chain, per-item binary judging with evidence-truncation rule).
- `Mx/_helpers.py:494-604` (`ws_fingerprints`, `db_fingerprints`, `protect_diff`, read-only `db_query`).
- The `Mx/sidecar_entrypoint.py` + `mcp_http.py` + `mcp_bridge.py` trio: tools-as-Python-files → MCP sidecar with result capping and hidden URLs.
- `V/third_party/mimoagent-osr/src/mimoagent/agents/antihack.py` (pre-execution action interceptor returning dummy observations).

---

## Part 5. Karotte (MIT; sandbox and anti-cheat framework, no world-state model)

**What it provides:**
- **Task model:** `Task{system_prompt, steps, tools, required_hardware, data_mounts, pre_hook, configure_tools}` (`K/src/karotte/task.py:39-124`).
- **Step model:** `Step{instructions, judge, submission_paths, pre_scoring_hook, post_hook}`. `score()` turns a `StudentMisbehaviorError` into a score of 0 (`step.py:29-92`; `student_misbehavior.py`).
- **Scoring:** `Scoring{score, metadata, continue_task}` (`schemas/scoring.py`). Steps run in sequence; if a judge sets `continue_task=False` the run fails (`docs/running/run-lifecycle.md:71`). This gives milestone-like gating.
- **Judges:**
  - `RegexJudge`
  - `RubricJudge`: an LLM decides each weighted criterion yes/no; score = sum of met weights (`judges/rubric_judge.py:23-112`).
  - `ExecutableJudge`: runs a trusted script that writes `{score, metadata}` (`judges/executable_judge.py`).
  - `AndJudge` / `OrJudge` composites (`judges/_composite_judge.py:104-193`).
  - `AlwaysPass`.
- **Tools:** `bash` (runs as the demoted student), `view_lines_in_file`, `replace_in_file`, `view_image_file`. The tools are served by an HTTP MCP server that runs as root (`docs/running/run-lifecycle.md:45`).
- **Runtimes:** Firecracker microVM (`firecracker/vm.py`: read-only base drive plus overlay scratch, vsock relay, seccomp), Apple `container`, runc, gVisor, docker or podman.

**Isolation and anti-cheat (how):**
- **Privilege split:** the harness, judges and MCP run as root. The student runs under a demoted uid via `setgroups([]); setgid; setuid` (`subprocess.py:297-359`). Harness secret env vars are scrubbed (`subprocess.py:82-102`).
- **Filesystem:**
  - `harden_filesystem` strips group/world write bits from all mount points except promised writable dirs (`confinement.py:255-287`).
  - At startup, student-writable dirs are removed from PATH, LD_LIBRARY_PATH, LD_PRELOAD and LD_AUDIT (`run-lifecycle.md:41`).
  - `trusted_binary` resolves privileged binaries only from root-owned dirs (`trusted_bin.py`).
  - `untrusted_paths` provides O_NOFOLLOW / dir-fd walks that never follow student symlinks (`untrusted_paths.py`).
  - `save_submission` copies submissions root-only (0700/0600) and rejects symlinks, special files and oversized trees as misbehavior (`save_submission.py:46-174`).
- **Secrets:**
  - `ProtectedStore`: root-only files for grader data, checked with `runuser` (`protected_store.py:47-60`).
  - `hide_run_config`: re-execs so the config and API key leave `/proc/[pid]/cmdline` (`hide_run_config.py`).
  - `model_forwarder`: the student agent gets a placeholder key; a root forwarder adds the real one.
- **Network:**
  - iptables / ip6tables `--uid-owner` rules allow only loopback, the sandbox's own IPs, and optionally private ranges (`confinement.py:289-398`).
  - gVisor iptables silently no-ops, so a network namespace is used there (`confinement.py:143-153`).
  - An egress self-test refuses to run if outside hosts answer (`run-lifecycle.md:49`).
  - Firecracker: a host-side egress filter drops link-local (metadata), private and host addresses (`firecracker/network.py:1-12, 373`).
- **Resources:**
  - Student cgroup with memory limit (1 GiB harness reserve), processes 2048, files 1M (`confinement.py:57-70`; `cgroups.py`).
  - Loop-mounted overlay file quota (`file_quota.py`); memory watchdog and reaping (`memory_watch.py`, `reclaim.py`).
  - PID-1 init shim reaps zombies (`init_shim.py`).
  - `karotte check confinement` measures what is actually enforced (`confinement_check.py`).
- **Supply chain:** a 7-day `exclude-newer` age delay for uv (`uv_supply_chain.py`).

**Worth porting (MIT):**
- `K/src/karotte/confinement.py` (uid-owner firewall plus filesystem hardening) with `K/src/karotte/subprocess.py:297-359` (demotion).
- `K/src/karotte/save_submission.py` + `K/src/karotte/untrusted_paths.py` (hostile-input-safe submission harvesting).
- `K/src/karotte/student_misbehavior.py` + `K/src/karotte/protected_store.py` (misbehavior → 0 versus infra error; grader-only secrets).

---

## Part 6. Requirements checklist for a framework to express all of these

Families are tagged as in Part 2 (F1-F18), plus EOG (EnterpriseOps-Gym), MiMo-G (MiMo general), MiMo-C/Cy (MiMo code / cyber), and KAR (Karotte).

| # | Primitive | Needed by |
|---|---|---|
| 1 | **Relational enterprise-records state** (SQLite/Postgres) with per-rollout seed-from-snapshot, fresh DB id per run, and teardown | EOG, MiMo-G (SQLite per MCP), F6 (workplace tables), F7 (banking DB), F3 (bird/spider) |
| 2 | **State-diff grading:** read-only SQL assertions on post-state, "other rows unchanged" checks, logical table fingerprints, and replay-gold-vs-predicted diff | EOG (`database_state`), MiMo-G (`check_code`, `db_fingerprints`), F6 (`workplace_assistant` replay diff), F7 (indian_banking DB-hash vs gold replay) |
| 3 | **Source-protection / tamper gate:** protected files and tables fingerprinted; any change → 0 | MiMo-G `src_protect`, KAR (read-only mounts) |
| 4 | **Per-domain tool catalogs** of many tools with **distractor and tool-set modes** (oracle / +N), allow and restrict lists, and per-session tool allow-lists | EOG (`selected_tools` / `restricted_tools` / modes), MiMo-G (3-17 MCPs per task), F9 (`has_distractors`), NeMo `mcp_allowed_tools_for_session` |
| 5 | **Any-MCP-server-as-App** over streamable HTTP and stdio bridge, with tool-name namespacing and normalization (`mcp__srv__tool`), result-size caps, and hidden endpoint URLs | EOG, MiMo-G, NeMo MCP auto-exposure, KAR (MCP tool server) |
| 6 | **Identity / RBAC context:** per-task acting user, role→permission tables, org scoping, identity headers on every tool call and verifier query | EOG (`x-itsm-user-token`, `user_role` / `role_permission`), MiMo-G (pinned immutable rows, optimistic concurrency in tools), F7 (KYC-gated writes in indian_banking) |
| 7 | **Persona-driven simulated users:** scenario fields (persona, reason, known/unknown info), progressive disclosure, stop/transfer/out-of-scope tokens, turn caps, deterministic opening message | F7 (CTU sim, indian_banking, toolsandbox; calendar offline synthesis), MiMo `user_agent.py` follow-up driver |
| 8 | **LLM-simulated tools / environment responses** as a separate actor role | F7 (CTU tool-result simulator) |
| 9 | **Judge servers as first-class components:** a separate model server from the policy, multi-key/model fallback, phase budget, retries, per-item isolated binary verdicts, vote medians, positional swap checks, judge panels, caching | F1, F2, F7, F8, F9, F10, F12, F13, F15, MiMo-G, EOG (`response_check`), KAR (RubricJudge) |
| 10 | **Rubric items with tiers, weights and methods** (rule / llm / agent / vision); gates before weighted sums; evidence extraction from office files with explicit truncation semantics | MiMo-G, F15 (gdpval, job_bench, workspace_bench), KAR (weighted RubricJudge), F4 (vibench per-test-plan) |
| 11 | **Multi-component rewards:** a named components map plus scalar aggregate; product compositions (safety × utility, strict × ...); strict-vs-dense blends with efficiency tie-breakers | F6 (multireward), F12 (IPI, jailbreak), F7 (indian_banking), F4 (vibench), EOG (all-pass vs pass-rate) |
| 12 | **Failure masking:** distinguish infra failure from policy failure (judge down, sandbox OOM, session lost, verify crash) with a stable failure-kind vocabulary and stage; mask instead of training on 0; invalid-reward sentinel | NeMo (`mask_sample`, `failure_kinds`, `judge_failsafe`; osworld; CTU sim), MiMo (absent reward key → `testbed_corrupted` → -999), KAR (misbehavior=0 vs ErrorEvent). EOG lacks it (compute_score does not exclude errors). |
| 13 | **Student-misbehavior vs infra error classification:** illegitimate actions score 0, not masked | KAR, MiMo antihack (dummy observation), F4 anti-cheat |
| 14 | **Sandboxes with fresh-sandbox hidden-test grading:** the grader never trusts the agent's box; image per task; CPU caps; TTLs; resource-owned deadlines; idempotent seed/verify retries | F4, F5 (TB4), F3 (cvdp), F15, MiMo-C/Cy |
| 15 | **Anti-cheat environment scrubs:** git future-history removal / `.git` hiding, hidden verifier files, pre-execution action interception, network egress restriction, privilege split, root-only grader secrets, symlink-safe submission harvesting | F4 (`anti_cheat_setup.sh`), MiMo (antihack, git_leak_prevention, sidecar-only DB volume, bridge uid), KAR (confinement, ProtectedStore, save_submission) |
| 16 | **Multi-container topology:** agent container plus sidecar for tools and verifier with shared and private volumes; agent cannot see DB volume or verifier | MiMo-G (k8s sidecar), F4 deepswe/vibench two-env, F15 workspace_bench judge sandbox |
| 17 | **Episodic reset/step with per-step reward** (Gymnasium) and score-delta shaping | F11, F7 (indian_banking), openair_congestion |
| 18 | **Milestone / minefield progress scoring** (dense, continuous terminal) and multi-step gating (`continue_task`) | F7 (toolsandbox), KAR (steps + `continue_task`) |
| 19 | **Live GUI / VM / browser Apps:** remote worker pools, observe/step/evaluate, idempotent operation ids, idle reaping | F5 (osworld, interactive_browser), F4 (vibench browser grading) |
| 20 | **External-API Apps** (search, EDGAR, prices) behind adapters, with regex/judge fallback | F8, F6 (finance_agent_v2) |
| 21 | **File workspace Apps** (xlsx/docx/pptx/pdf) with deliverable location, and final-answer materialization from chat into a file | MiMo-G, F15 |
| 22 | **Session lifecycle guarantees:** caller-assigned session ids, close registered before seed, LIFO idempotent bounded cleanup, admission control and queue timeouts, episode deadlines, signed session tokens | NeMo base_environment_server, TB4, osworld |
| 23 | **Offline re-verification:** dump post-state and rerun the verifier later (judge outage); reverify mode declaration | MiMo `pull_post_state`, NeMo `ReverifyMode` / `rollout_reverification` |
| 24 | **Single-turn stateless verify fast path:** dataset row → one model response → verify | F1, F2, F3, F9, F10, F12-F14, F16 (the majority of NeMo-Gym) |
| 25 | **Next-action step verification** against demonstrations (behavior-cloning RL) | F18 |
| 26 | **Task schema registry:** typed task data with consumed-by tags; dataset manifests with license, source and splits | NeMo `task_data.py` / `DatasetConfig`, EOG task JSON, MiMo `manifest.json` + `verifier_meta.json` |

**Gaps relative to Crucible's described model:**
- EnterpriseOps-Gym needs primitives 1, 2, 4, 5, 6 and 9 together, and its MCP servers are closed images. An adapter would wrap them as an MCP App plus SQL-runner verifier, with seed and delete per rollout.
- No source here models a virtual clock or events, except toolsandbox's internal world state.
- "Dense milestone progress" exists only in toolsandbox, Karotte step gating, and the Gymnasium games.
- Infra-failure masking appears under three incompatible conventions (NeMo `mask_sample`, MiMo absent-reward-key, Karotte ErrorEvent). Importers should map all three onto one representation.
