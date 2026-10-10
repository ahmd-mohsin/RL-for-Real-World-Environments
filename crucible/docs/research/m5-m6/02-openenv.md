# M5/M6/harness research, slice 02a: OpenEnv (rubrics, step rewards, harness capture, Harbor rollout, snapshots)

Source: `RL-Environments/06-frameworks-and-env-code/meta-pytorch-openenv/` @ 49aa302. Read-only; no network. All paths
below are relative to that repo. (Harbor installed agents and FineEnvs code are covered in slice 02b.)

## TL;DR
- **Step rewards do not cross the wire.** `TrajectoryRubric.compute_step_rewards()` is in-process only on `env.rubric`;
  HTTP/WS serialize one scalar `reward` + free-form `metadata` per step (`env_server/serialization.py:147-185`;
  `http_server.py` never mentions rubric). Chess works around it with `metadata["discounted_rewards"]`.
- **No core snapshot/restore/fork.** Mid-trajectory starts exist only as env-specific `reset(**kwargs)` (chess `fen=`);
  RFC 008 `injected_state`/`set_state` is declared in the manifest and never invoked.
- **No milestone / progress-reward construct**, only `TrajectoryRubric` + `ExponentialDiscountingTrajectoryRubric`
  (`r_t = γ^(T-1-t)·R_final`). Harbor multi-step `step_results` are recorded, never trained on.
- **The reusable part is the harness capture proxy** (`core/harness/capture/`): session-id-as-API-key, 4 dialects,
  engine token ids + logprobs, prefix-graph stitching, graded validation, Harbor reward join; seams for claude-code,
  codex, opencode, mini-swe-agent (+ pi, gemini-cli, terminus-2, qwen-coder, vibe, openhands-sdk).
- **On the harness path, reward is one terminal scalar per rollout**; nothing assigns turn-level rewards.
- **RFC vs code drift:** RFC 004 says every env must have a rubric (code: optional, unvalidated, `interfaces.py:252-260`);
  RFC 004 LLMJudge via MCP (code: `LLMClient`); RFC 004 EnvPool absent; RFC 006 `degenerate_rollout` FATAL (code: WARN,
  `capture/validate.py:346-354`).

## 1. Rubrics (`src/openenv/core/rubrics/`)
**Base `Rubric`** (`base.py`): nn.Module-style; child rubrics auto-registered from attributes (50-54); `__call__`
dispatches sync/async forward with pre/post hooks (56-127); `last_score` set after forward (95-103); `forward` abstract,
range not enforced (129-140). Navigation `children/named_children/rubrics/named_rubrics/get_rubric("a.b")` (164-204);
container children named `rubric_0..n` (`tests/core/test_rubrics/test_containers.py:395-410`). **Defects:** `reset()`
is a no-op and does not recurse (206-208; no container overrides it), so a nested `TrajectoryRubric` is never cleared by
`env._reset_rubric()`; `last_score` is not cleared, so fail-fast-skipped children keep stale scores; `state_dict()` = {}.

**Containers** (`containers.py`):
- `Sequential` (27-165): `result=1.0; for r: s=r(a,o); if s == 0.0: return 0.0; result = s; return result` (57-65) —
  returns the **last** child's score (not product/min); empty → 1.0; negative scores do not fail-fast.
- `Gate(rubric, threshold=1.0)` (167-220): `0.0 if score < threshold else score` (raw pass-through).
- `WeightedSum(rubrics, weights)` (223-323): weights must sum to 1 ± 1e-6 (251-257), negatives unchecked; all children
  evaluated; async children `asyncio.gather`. Test: `[Gate(0.8,.5), Gate(0.3,.5)]·[.6,.4] = 0.48`.
- `RubricList` / `RubricDict` (326-465): containers without forward.
- Known bug: in container sync paths pre-hooks run after children (`test_pre_hook_bugs.py:1-11`, "currently FAIL").

**`LLMJudge(prompt_template, client, score_pattern=None, default_score=0.0, normalize=True)`** (`llm_judge.py:28-102`):
async; `prompt_template.format(action, observation)`; first regex match `(\d+\.?\d*)`; no match → default; normalize
clamps to [0,1] ("Score: 8" → 1.0); cannot parse negatives. Model/version not in state; the validation manifest adds
`JudgePin{model, version, params}` (`validation/manifest.py:75-88`). `OpenAIClient` defaults temperature 0.0, max_tokens 256.

**`TrajectoryRubric(intermediate_reward=0.0)`** (`trajectory.py:22-135`): accumulates (action, obs); scores only when
`obs.done`; abstract `score_trajectory` and `compute_step_rewards() -> list[float]`; sync only; `compute_step_rewards`
re-calls `score_trajectory` (judge scored twice). Only built-in strategy: `ExponentialDiscountingTrajectoryRubric(gamma)`
(138-211). Pinned values: γ=1 → [1,1,1]; γ=0 → [0,0,1]; γ=0.5, 3 steps → [0.25,0.5,1]. Envs using it: chess, carla.

**Wiring:** `Environment(transform, rubric)`; `_apply_rubric` (0.0 with no rubric) / async / `_reset_rubric`
(`interfaces.py:254-422`); env author assigns `observation.reward`. Chess puts `discounted_rewards`/`gamma` in metadata
(`chess_environment.py:183-254`); Carla uses `obs.rubric_reward`. `MCPEnvironment` applies no rubric.

## 2. RFC 004 intent
Server-side rubrics; gatekeeper = `Sequential(hard…, WeightedSum(soft…))`, defense rubrics = `Gate` (677-689);
per-component logging via hooks over `named_rubrics` (162-177); process rewards only as `compute_step_rewards`
overrides (plan mode 484-533; Codenames 30% clue + 70% outcome 534-583); docs: prefer `TrajectoryRubric`, keep shaping a
small-weight `WeightedSum` component (`docs/source/guides/rewards.md:27-43`). Envs: REPL process rubric (error −0.05)
on non-terminal steps + outcome on done (`envs/repl_env/rubrics.py:125-187`); Pelican
`Sequential(Gate(GatePassed,1.0), WeightedSum(...))`.

## 3. RFCs 005 / 006 / 012 and others
- **RFC 005 (harness inside env):** `step()` = one harness turn; inner calls only as `turn_events`; shaping after each
  tool call out of scope (795-797). `HarnessEnvironment` timeout/crash → done, reward 0.0 (`harness/environment.py:356-372`).
  Trainer protocols in `harness/rollout.py`: `ModelStep` (80-121) returning prompt_ids/completion_ids/logprobs,
  `ResourceSession{initial_messages, list_tools, call_tool, verify, close}` (223-255), `VerifyResult.env_reward` must be
  forwarded, never synthesized; trace vs verify reward mismatch > 1e-6 raises (329-360). `TraceEntry` (124-185): engine
  prompt ids ("MUST NOT re-derive" — re-rendering matched 0/28 turns), completion ids, per-token logps, loss mask, reward
  (None = unscored). `MCPHarnessAdapter.run_white_box` naively `.extend`s prompt ids (618-624).
- **RFC 006 (interception):** ownership split (95-112): OpenEnv interception/graph/validation, Harbor sandbox, env/verifier
  reward, TRL generation/advantages. D2 train needs own engine (hosted = eval-only); capture levels tokens/logprobs/text.
  D3 no tokenizer: `context_ids = prompt_ids[len(parent.end_ids):]`; processed_logprobs + top_p<1 bias (−1.3292 vs −1.2546).
  D4 stitching: parent = node whose prompt+sampled ids is the longest exact prefix of the child's prompt ids; break → new
  root; retries childless siblings; subagents/compaction new roots; claude-code swaps system prompt at call 6 (2 roots).
  D5 session-id-as-key (require_registered default True). D6 synthetic SSE replay. D7 reward dict verbatim, explicit
  scalar rule, None ≠ 0. D9 missing logprobs → mask-0 context. D13 `max_model_calls` → stop message. D15 graded
  validation on ingest. D17 roles + ATIF cross-check. D18 sandbox dials a published proxy (Direct/Gradio/Cloudflare).
  D19 dialects: chat (opencode, mini-swe-agent), Responses (codex), Anthropic (claude-code), Google (gemini-cli).
- **RFC 012 (Harbor capture providers):** purpose eval/train/auto; training policy requires temperature > 0, full vocab;
  per-session `max_output_tokens`; prompt mask 0, logprobs ≤ 0, never zero-filled; zero reward valid, missing ≠ zero.
- **Others:** RFC 001 reset is never a tool; initial state snapshot separate from dataset. RFC 002 snapshots are
  orchestration ops, never MCP tools. RFC 008 oracle contract `reward(oracle) ≥ max − tol`, `max − floor ≥ margin`;
  `RewardDeclaration{range, oracle_tolerance, floor_margin, variance_tolerance}`. RFC 010 proposes
  `L = GRPO + λ·CE(env-output tokens)`, λ 0.005–0.05, as verifier-free dense supervision.

## 4. Harbor rollout path (`cli/commands/harbor.py`, `harbor/rollout.py`)
`openenv harbor rollout --llm-url (required) --model --dataset --task-index -n --harness (opencode) --sandbox (e2b)
--expose gradio|cloudflare|direct --reward-key …`. `run_rollout` never raises (failure → ok=False, reward=None):
session `s<hex24>`; seam fills agent env with intercept URL + session id; Harbor retry off; reward resolution
(`rollout.py:174-191`): explicit key → single key → `reward` → raise; per-step `step_results` recorded as
`HarborStepResult{name, rewards, passed}` but not used; ATIF reconcile (match / MISMATCH / none); ok gated on FATAL
findings + trainable; session deleted in finally.

**Seams** (`harbor/seams.py`): key is always the session id; agent env preferred over process env (prevents leaking
`OPENAI_API_KEY` into an LLM-judge grader, 114-150).

| Harness | Dialect | Wiring |
|---|---|---|
| opencode | chat | config provider `@ai-sdk/openai-compatible`, baseURL `{base}/v1`, apiKey = session |
| claude-code | anthropic | `ANTHROPIC_BASE_URL={base}` (no /v1), `ANTHROPIC_API_KEY=session`; `count_tokens` aux route |
| codex | responses | `OPENAI_BASE_URL={base}/v1`, `OPENAI_API_KEY=session` |
| mini-swe-agent | chat | `MSWEA_API_KEY`, `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_BASE_URL`; model `openai/{model}` |

**Capture proxy** (`core/harness/capture/`): routes `/health`, admin `/sessions` CRUD, `/sessions/{id}/rollout`,
`/trace_entries`, `/v1/models`, catch-all POST for chat/messages/responses/generateContent; auth Bearer → x-api-key →
x-goog-api-key, unknown 401. Token capture injects `logprobs, return_token_ids, top_logprobs≥0` and NON_POLICY_SAMPLING
`{top_p 1, top_k −1, min_p 0, freq 0, presence 0, repetition 1}` (`upstream.py:146-286`); requests forced `stream=False`,
empty tools dropped, n≠1 → 400. `TurnNode{node_id, prompt_ids, sampled_ids, sampled_logprobs, parent_id, index, model,
finish_reason, harness_session_id, system_digest, n_tools, request_messages, request_tools, sampling_params,
response_message}` (`graph.py:157-187`); parent = longest end-ids prefix; orphan adoption makes arrival order irrelevant.
Flattening: context mask 0 / logprob 0; sampled mask 1 + real logprobs only when present and aligned. Export:
`{session_id, metadata, budget_stop_count, turns, stats, sequences[role, input_ids, loss_mask, logprobs, trainable,
validation], validation, trainable, capture_level, rollout_type}`. Validation FATAL: positive_logprob,
logprob_misalign, no_logprobs, masked_has_logprob, nothing_trainable, sampling_policy_changed; WARN: token_strings,
degenerate_rollout, multiple_roots. Budget: "Step budget exhausted; stopping." returned as a completion, never ingested.

## 5. Snapshots
None in core (`reset/step/state/close` only, `interfaces.py:262-310`). Env-local: chess `fen`, DIPG `set_state`, AWM
`save_snapshot` (verifier DB diff), calendar `/clone-database`. REST is stateless (fresh env per call); only WS sessions
persist.

## DEPEND / PORT / PATTERN for Crucible
- **DEPEND:** `openenv.core.harness.capture` (CaptureServer, SessionRegistry, export_session, to_trace_entries,
  PortForwarder) as the model proxy for black-box harnesses (training needs vLLM with `--return-tokens-as-token-ids
  --logprobs-mode processed_logprobs`; eval works on hosted APIs); `openenv.harbor` seams + `run_rollout` if we drive Harbor.
- **PORT:** rubric tree with fixes (recursive reset clearing `last_score`, async `score_trajectory`, cached trajectory
  score, threshold fail-fast, step-reward strategy registry incl. milestone, `component_scores()` into metadata);
  capture-graph stitching (`graph.py:258-487`) + validation catalogue; `_pick_reward`; NON_POLICY_SAMPLING; ATIF reconcile
  (`atif.py:327-563`); `TraceEntry`/`TrainingTurn` schema.
- **PATTERN:** rewards computed server-side and forwarded, never synthesized (with a consistency check); None ≠ 0; rows
  labelled, never filtered; session-id-as-key; non-streaming capture + synthetic SSE; graded FATAL/WARN validation on
  ingest; budget stop as a message; agent env isolated from the grader's env; oracle/floor/margin validation.
- **Crucible must build itself:** a wire export for dense step rewards and per-component scores (`steps.jsonl` +
  `reward.json` components); turn-level reward on harness rollouts (join keys: ATIF `tool_call_id ↔ source_call_id`, or
  `TurnNode.index`); snapshot/restore/fork for mid-trajectory starts (we have `WorldRef`); harness resume from a
  prefilled conversation; milestone gating beyond Harbor `step_results` / `min_reward`.
