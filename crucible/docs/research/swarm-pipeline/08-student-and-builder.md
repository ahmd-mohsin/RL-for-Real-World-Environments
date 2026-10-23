# 08 Student and builder: the plumbing of the MVP loop

Spec: `00` §3.2 (roles and families), §7 (targeting), §8 (traces and the trace audit), §11 (the env wrapper); `04` (as
built). The MVP (2026-10-20): long-horizon multi-language debugging tasks. Claude builds them, Tinker samples every other
role, and the tasks are calibrated against the exact student that is post-trained.

The loop:
1. **Run the student** on candidate tasks through the normal env/harness path (`WorkspaceHarness` + `LLMToolPolicy`).
2. **Read its traces** for where and why it fails: the trace-audit facts pass, the first wrong step, a cause, a verbatim quote.
3. **Ask the builder** for tasks that target those failures: a builder-intent request per student model, approved by a
   human, consumed by the swarm planner or the root curator.

## 1. The role map (`factory/models.py`, the default preset `tinker`)

| role | model | family |
|---|---|---|
| planner, oracle builder, first verifier writer, prompt writer (author, World expert, mutant builder) | `claude-code:opus-5.5` (headless Claude Code, the user's plan) | anthropic |
| second verifier writer (S9) | `tinker:deepseek-ai/DeepSeek-V4.1-Flash:peft:262144` | deepseek |
| reviewer; trace auditor of the Qwen student | `tinker:zai-org/GLM-5.3:peft:262144` | glm |
| cheap judges, cause labels (`judge`, `realism`) | `tinker:zai-org/GLM-5.3-Flash:peft:262144` | glm |
| hackers (`hacker`, `hacker_2`) | DeepSeek-V4.1-Flash, `tinker:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144` | deepseek, nemotron |
| alternatives / oracle rewrites | DeepSeek-V4.1-Flash, GLM-5.3 | deepseek, glm |
| tie-break | Nemotron (a third family) | nemotron |
| fairness / frontier solvers (`frontier`) | `tinker:Qwen/Qwen3.5-397B-A17B:peft:262144`, DeepSeek-V4.1-Flash | qwen, deepseek |
| difficulty ladder (`solvers`) | `tinker:Qwen/Qwen3.5-4B`, `tinker:Qwen/Qwen3.5-9B`, `tinker:Qwen/Qwen3.8-27B` (not Qwen3.6-27B: it retires Oct 23) | qwen |
| **target student** | `tinker:Qwen/Qwen3.6-35B-A3B` (MoE, about 3B active, 262K context) | qwen |

- **The family rule is enforced in the registry** (`ModelBook.family_problems`, `SameFamilyError` at construction). An
  independent role (reviewer, realism, second verifier, auditor, hackers, tie-break, judge):
  - is never on Claude Code;
  - is never on a builder's family, because it would grade a task its own family built;
  - is never on a student's family, because it would audit its own family's traces (the Qwen student is never audited by
    a Qwen model).

  The two hackers must also come from two families. The machine validator's `different_family_from` is the second line.
  `client(role)` refuses Claude Code on an independent role again after a later edit of `roles`.
  `SwarmRunner(clients=...)` overrides go through the same check (`checked_overrides`).
- **`tinker` is a registry slot.** It is not an HTTP provider: `models.register_backend("tinker", make, final=True)` is
  where the Tinker bridge (`train/bridges/tinker`, owned by another agent) registers its client class. Every
  `tinker:<model>` role or rung is then built as `make(model, role=..., thinking=...)`.
  - Until the bridge registers, `client()` raises a `KeyError` that names the bridge.
  - Nothing here calls Tinker.
- **OpenRouter is a fallback only.** The v1 map is the preset `openrouter`, and a bare `org/model` id still resolves to
  the `openrouter` provider. The default never uses it.

## 1.1 The swarm map (preset `swarm`, 2026-10-23): no Claude in any role

From 2026-10-23 the user's swarm builds new tasks on Tinker: "all next steps should be done by my swarm intelligence not
claude". On the builder: "all three [DeepSeek, GLM, Kimi] but for different things". `ModelBook(preset="swarm")`
refuses any role on Claude Code.

| role | model | family | why this one |
|---|---|---|---|
| oracle builder, oracle author | DeepSeek-V4.1-Flash (256K) | deepseek | the long sessions in the repository: the only model we measured at Φ = 1 on a 300-call debugging task (mvdan-sh, `07` §6.1); the cheapest 256K builder ($0.60 / $3.60 per M) |
| planner, first (blind) verifier writer | GLM-5.3 (256K) | glm | reasoning-heavy, short-output roles; a family other than the oracle builder's (V2 by family) |
| prompt writer (`author`, `prompt_author`), World expert, mutant builder | Kimi-K2.6 (`:peft:131072`) | kimi | text over a brief, not 300-call sessions; the 32K base cannot hold a session; the text protocol now also reads its native tool section (§1.2) |
| reviewer, second verifier, trace auditor | Nemotron-3-Ultra-550B (256K) | nemotron | judgement roles on a family nothing builds; never Qwen on the Qwen student |
| hacker, alternative (`rewriter_nemotron`) | Nemotron-3-Super-120B | nemotron | |
| judge, realism, second hacker, tie-break | gpt-oss-120b (`:peft:131072`) | openai | cheap; a third checker family (the tie-break differs from the reviewer and second verifier) |
| alternative (`rewriter_qwen`) | Qwen3.5-397B | qwen | an alternative solution by a family that neither built the oracle nor wrote the verifier |
| fairness solvers | Qwen3.5-397B, Nemotron-3-Ultra | qwen, nemotron | never a builder's family: a builder's family solving its own family's task is no fairness check |
| target student | Qwen3.6-35B-A3B | qwen | unchanged |

**Rules:** the role rule (`family_problems`) and the strict-preset rules hold, and the provenance rule
(`factory/provenance.py`) applies per artifact: no checker reads an artifact its own family produced (`13` §2).

### 1.2 Measured in the smoke (2026-10-23, `scripts/swarm.py smoke`, $1.07 of Tinker in all)

- **One JSON call per distinct model, $0.0024 for all seven:** GLM-5.3 $0.00033 (43 prompt / 14 completion tokens),
  DeepSeek $0.00008, Kimi $0.0015 (262 completion tokens: it opens `<think>` whatever the switch says), Nemotron-Ultra
  $0.00017, Nemotron-Super $0.00005, gpt-oss-120b $0.00006, Qwen3.5-397B $0.00019. All returned valid JSON.
- **One prompt-writer session per builder family**, on the RLM harness in an offline sandbox, with the potential
  `prompt_draft`. This is the final run, after the fixes in `13` §6:

| family | turns | tokens | USD | compaction | Φ end | Brier (STATUS.json) |
|---|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 7 | 46,313 | 0.0236 | at turn 7, accepted | 1.0 | 0.0025 |
| GLM-5.3 | 7 | 22,622 | 0.0389 | none: it printed only the first 3,000 characters of the long file | 1.0 | 0.0025 |
| Kimi-K2.6 | 19 | 126,911 | 0.1340 | at turn 9 (2 attempts) | 1.0 | 0.0025 |

## 2. The builder client (`actors/claude_code.py`)

`ClaudeCodeClient` has the `LLMClient` / `JsonClient` shape that every factory role already takes. Each call is one
`claude -p --output-format json` subprocess.

- **JSON roles.**
  - The CLI runs with no tools (`--tools ""`), and the system message replaces Claude Code's own prompt.
  - With `schema=`, the CLI also gets `--json-schema`, and the reply is validated here with `jsonschema`.
  - A bad reply is retried with the error fed back. After that the call raises `LLMOutputError`.
- **Agentic roles.** `run_session(task, cwd=<private clone>, add_dirs=[handoff], sandbox=<container>)` runs Claude Code
  with these limits:
  - `--restricted`: file tools stay inside the working directories;
  - `--permission-mode acceptEdits --permission-prompts none`: anything not allowed is denied, never asked;
  - the only allowed commands are read and commit `git` and `docker exec <the session's offline sandbox> ...`, so code
    never runs on the host.

  A `SessionPool` state whose client has `run_session` uses this path instead of the RLM agent. Captures, hand-off
  commits and the infra/task split do not change. A reply of `GIVE UP: <reason>` fails the state.
- **Kept apart from the operator's setup.** Every call runs with `--safe-mode`, `--strict-mcp-config` and
  `--no-session-persistence`. The child process never gets `ANTHROPIC_API_KEY`: the plan pays.
- **Plan usage, not dollars.**
  - `spend.usd` stays 0.
  - Each receipt holds `num_turns`, `duration_ms`, `duration_api_ms`, `total_cost_usd` (an API-equivalent estimate),
    `usage`, `modelUsage` and `session_id`. Receipts can also be written to a JSONL file.
  - Receipts are summed in `ModelBook.plan_usage()`, and `run_receipt(..., plan_usage=)` stores them under `spend.plan`.
  - `budget_usd` caps the API-equivalent total, and each call gets `--max-budget-usd` for what is left.
  - When the plan's limit is reached, the call raises `BudgetExceeded`. That is infrastructure, so the attempt is masked.
- **The one real smoke call** (2026-10-20): a short JSON prompt with a schema.
  - Result: `{"milestones": 3, "ok": true}`.
  - Cost: 3.0 s wall time, 2 turns (structured output), API-equivalent $0.011, model `claude-opus-5-5`.

## 3. The student (`factory/student.py`, `scripts/student_serve.sh`)

- **Target.** `student_spec()` is `tinker:Qwen/Qwen3.6-35B-A3B`.
  - `TARGET_STUDENTS` lists the models whose numbers count; there is one calibration per model.
  - Only a target model, sampled by a final backend (`tinker`, or one registered `final=True`), gives an S15 verdict or
    a benchmark report.
- **Local dev model.** `dev_spec()` is `student:Qwen/Qwen3.5-4B`, used for harness development and the pipeline smoke.
  - A 35B MoE does not fit this 24 GB Mac at bf16, so the target is never served locally.
  - `Qwen3-4B-Instruct-2507` (retired on Tinker) is also a dev model only.
- **Local serving** (dev only):
  - **mac:** `crucible/scripts/student_serve.sh --backend mlx`.
    - mlx-lm 0.32.2 runs in `~/.venvs/mlx`, not the project's venv.
    - bf16, text only: the `qwen3_5` model drops the vision tower on load.
    - The tool parser is `qwen3_coder`, from the template's `<function=...>`.
    - The prompt cache is 2 entries / 1.5 GB, prefill runs in 512-token steps, and one request is served at a time.
  - **CUDA box:** `crucible/scripts/student_serve.sh --backend vllm`. This runs `vllm serve --dtype bfloat16
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image":0,"video":0}'` with
    `VLLM_USE_FLASHINFER_SAMPLER=0`, plus `--reasoning-parser qwen3` only when thinking is on. Not run here: there is no
    CUDA box (the lab 5090 is unreachable).
  - `--stop` and `--status` manage the server. Its descriptor is `~/.cache/crucible/student/student-<port>.json`:
    backend, model, dtype, engine version, tool parser, thinking and pid.
- **Tool calling: native, verified on mlx.**
  - Qwen3.5-4B returned an OpenAI `tool_calls` entry (finish `tool_calls`).
  - Given a tool result, it answered in text.
  - In the smoke, 61 of 61 tool calls parsed. No text tool protocol is needed.
- **Thinking.** Thinking is an explicit setting recorded with every run, and it is off for the debugging agent.
  - The `[llm_providers.student]` entry sends `extra_body.chat_template_kwargs.enable_thinking = false` with every
    request.
  - The serve script makes the same value the server default.
  - `lh_run` refuses a student provider that does not set it. It records the value in the policy config and in
    `runs/<run>/student.json`.
- **Backend stamp.** `lh_run` stamps the backend into the policy id (`solver:student:Qwen/Qwen3.5-4B@mlx`;
  `solver:tinker:<model>@tinker` for a registered backend).
  - `assert_countable` raises in `benchmark.report` and `paired_report`.
  - `target_verdict` returns `ok: false, dev_only: true, route: null`, and `lh_target` then skips the S15 machine.
  - `lh_run` metrics give the window the verdict `dev_only`.

## 4. Failure mining (`factory/failures.py`)

A finding is built from the trace audit's own pieces, with no new trace schema. It is keyed by the **student model id**
and holds:
- the task and the policy;
- the milestones missed at the end, and the milestones lost after they were reached;
- the first wrong step;
- the cause: the auditor's label when there is one, else the deterministic cause from the facts, else unlabelled;
- a quote checked to be a verbatim substring of the rendered trace;
- the facts, the cluster signature and the numerics.

The **mode** comes from the facts and the first wrong step alone. These modes are the student's skill:
- `lost_progress`
- `stalled`
- `never_progressed`
- `no_edit`
- `claimed_done_without_progress`

Each one maps to the behaviour a new task should test. `harness_misuse`, `world_fault`, `budget` and `env_flake` are
routed to their owner, never turned into intents. So is any cause other than skill gap.

`builder_intents` clusters skill findings **per student and per mode** into a `BuilderIntent` (status
`awaiting_human`). Three views of it serve the next step:
- `intent_card()`: an `IntentCard` with status `candidate`;
- `planner_intent()`: the swarm planner's `intent`;
- `cell(approved_by="human:...")`: a root `Cell`.

## 5. The smoke (2026-10-20, Mac, dev-only numerics)

Numbers and trace findings are in `04` §"Student and builder".

**Data loss.** The host rebooted mid-session and `/private/tmp` was cleared. The scratch window stores, clones,
authored prompts and run-1/2 trails were lost.
- The two windows were mined again from GitHub with the same code, and came back as the same windows:
  `more-itertools-w15407c7-41a6d0f` (4 milestones) and `python-semver-w41775dd-1bafb9f` (7), both `mutants_valid`.
- The prompts were rebuilt with no model call, one requirement per milestone from the release's commit text. They are
  weaker than the authored prompts, so this smoke tests the plumbing, not the tasks.
