# Long-horizon agent trajectories in existing RL-environment frameworks

- **Question:** how do existing frameworks support *extremely* long agent episodes (thousands of tool calls, 1M+ cumulative tokens)? What should Crucible reuse for compaction inside RL, dense rewards, mid-episode resume, observation offloading and trace storage?
- **Read:** 2026-10-09, from the local checkouts. Every claim cites file:line.
- **Root abbreviations:**
  - `NG` = `RL-Environments/06-frameworks-and-env-code/nvidia-nemo-gym`
  - `VF` = `…/primeintellect-ai-verifiers`
  - `XV` = `…/xiaomimimo-verl`, with `MO` = `XV/third_party/mimoagent-osr/src/mimoagent` and `UA` = `XV/third_party/uni_agent/uni_agent`
  - `HB` = `…/harbor-framework-harbor`
  - `OE` = `…/meta-pytorch-openenv`
  - `AE` = `agentenv-framework` (repo root; there are no agentenv directories under RL-Environments)
  - `DS` = `RL-Environments/08-hf-rl-env-datasets`
- **Licenses (all permissive, so PORT with attribution in THIRD_PARTY.md is allowed):**
  - NG: Apache-2.0
  - VF: MIT
  - XV: Apache-2.0; mimoagent-osr is MIT (a fork of mini-swe-agent 1.9.0, `XV/third_party/mimoagent-osr/LICENSE.md:1-4`); uni_agent is Apache-2.0
  - HB: Apache-2.0
  - OE: BSD-3-Clause
  - AE: Apache-2.0
  - LHTB dataset: apache-2.0

## 0. What Crucible has today (the baseline the gaps are measured against)

| Piece | State | Long-horizon consequence |
|---|---|---|
| `LLMToolPolicy` (`src/crucible/actors/policy.py:63-188`) | Keeps the whole message list in memory, with no compaction. Tool output is head/tail-truncated by `max_tool_output` (`:103-106`). `ContextOverflow` ends the episode as `context_overflow`, which is truncated and scored (`:125-127`, `:281`). | The horizon is capped by one context window. |
| `run_episode` / `EpisodeLimits` (`policy.py:191-281`) | Has `max_steps`, `max_tool_errors`, `timeout_s` and `snapshot_every` (a WorldRef series deduplicated by fingerprint). | World snapshots exist mid-episode, but the **policy context is not part of them**: `WorldRef.actors` (`core/schema/state.py:98`) serializes simulated actors, not `LLMToolPolicy.messages`. |
| `crucible.rollout/v1` (`src/crucible/train/rollout.py:25-85`) | `Turn` stores full `prompt_token_ids` per call. `Segment` has only `start_call`, `end_call` and `summary_tokens`. | Storage is **O(calls × context)**: 1,000 calls at a 200k context is about 2×10⁸ ints per rollout. `Segment` cannot express a boundary reason, the summary call or trainability. |
| `RunJournal` (`src/crucible/runtime/journal.py:1-60`) | Append-only fsynced events plus atomic per-attempt records. Resume works at **sample** granularity. | A crash at hour 3 of a 4-hour episode reruns the whole episode. |
| M6 credit (`src/crucible/credit/compose.py:89-130`, `credit/steps.py`) | Per-turn advantage A_t = A^GRPO + η(ΔΦ_t − c)·σ_R/σ_ΔΦ, keyed by `call_id`. | Already **segment-agnostic**: credit attaches to calls, not to sequences. This is the key enabler for cross-segment credit (§(a)). |
| DESIGN §10.3 point 7 and ARCHITECTURE M12 `compaction.py` | "Compaction is a recorded segment structure, and credit crosses segments" is designed but not built. | This report fills it in. |

---

## 1. NeMo-Gym: `simple_agent_with_compaction` and `vllm_model_with_compaction`

**What it is.** NG's compaction is **deterministic history *rewriting*, not LLM summarization.** The built-in policies are:
- `identity`;
- `recency`: drop old image groups and old reasoning blocks (`NG/responses_api_agents/simple_agent_with_compaction/compaction/config.py:13-79`, `compaction/policies.py:112-200`).

Agents can register other policies (`policies.py:46-70`). The real contribution is the **evidence model**, which keeps token ids and logprobs exact across rewrites.

### Mechanism
- **Semantic history vs. materialized view.**
  - `SemanticHistory` keeps every event. Media is deduplicated in a content-addressed `MediaArena` (`compaction/history.py:172-227`, `:658-790`).
  - Before each call, a `HistoryController` asks the policy for a `HistoryViewPlan` (keep refs plus omission artifacts) and materializes the request view (`compaction/controller.py:61-115`).
  - A test runs 100 turns with bounded active images while the arena retains all 100 payloads (`tests/test_compaction.py:854-901`).
- **Segment boundary detection.**
  - If the new view's descriptor is not append-compatible with the last acknowledged view, the controller emits a `RewriteBoundaryEvent` and bumps `context_epoch` and `segment_index` (`controller.py:88-103`).
  - The event records the policy name/version, config digest, previous and current view digests, changed ranges, and retained/omitted counts (`controller.py:129-165`).
  - The boundary stays *pending* until the call is acknowledged, and a retry that would produce a different view raises (`controller.py:69-86`, `:117-127`). The rewrite is therefore retry-safe.
- **Turn-chunked schedule.** `TurnChunkedHistoryController` freezes one compacted base and appends a raw tail for K actions, then recompacts (`controller.py:168-306`). This amortizes prefix-cache breaks to one per K actions. It emits `FinalizedChunkRecord`s (`:324-359`).
- **Exact-prefix enforcement inside a segment.**
  - When the view is append-compatible, the session sends `required_prefix_token_ids` = the previous call's `prompt_token_ids + sampled_token_ids` (`compaction/session.py:345-351`).
  - `capture_observed_completion` refuses a prompt that does not start with that prefix (`history.py:907-913`) and requires exactly one generation-evidence item carrying `prompt_token_ids`, `generation_token_ids` and `generation_log_probs` (`history.py:885-903`).
  - The vLLM adapter **refuses post-hoc retokenization** (`NG/responses_api_models/vllm_model_with_compaction/app.py:62-70`). Its `/tokenize` is only for pre-generation guards (`README.md`, `app.py:92-109`).
- **Per-call evidence.**
  - `ObservedCompletion` binds the token arrays to `segment_index`, `segment_id`, `context_epoch`, `compaction_event_id`, the policy decision and a `PolicyOutputSpan` that covers the whole sampled completion (`history.py:525-607`).
  - The transport form drops duplicated payloads after verification and keeps only a digest of the token arrays (`session.py:84-127`, `:158-188`).
- **Context guards.** `max_total_tokens + reserved_generation_tokens`, `max_active_images` and `max_vision_tokens` are evaluated between turns using the server tokenizer (`controller.py:385-428`; `app.py:190-220`).
  - An exceeded guard first closes the chunk early and recompacts.
  - If the guard still fails, the outcome is `reject` → `RuntimeError` (`session.py:406-477`).
  - Decisions are recorded as `GuardOutcomeRecord`s (`controller.py:448-484`).
- **Generation contract.** It hashes the model, sampling and compaction-policy config, and sets `training_eligible=False` with explicit reasons when the server does not report tokenizer, template or processor identity (`session.py:191-243`). It fails closed.
- **Reward attribution: one logical rollout, one reward.**
  - `/verify` is called once on the full logical output (`app.py:597-608`). The compaction evidence rides along in the response (`app.py:620-635`).
  - From the OSWorld agent docs: "NeMo-RL can therefore reconstruct prefix-contiguous physical traces while one logical rollout retains one reward and advantage" (`NG/responses_api_agents/osworld_agent/README.md:59-63`).
  - At `max_steps`, the last action executes, the state is evaluated and `horizon_reached=true` is recorded. Horizon runs remain trainable (`osworld_agent/README.md:93-98`).
- **History offload with a pointer back** (a different agent, `langchain_deepagents_agent`): deepagents' SummarizationMiddleware writes evicted history to `/conversation_history/{thread_id}.md`, and "the agent can and does read_file that path mid-rollout to recover context" (`NG/responses_api_agents/langchain_deepagents_agent/README.md:170-175`; the thread id is the rollout id, `app.py:122-131`).

### Reuse for Crucible
- **PATTERN:**
  - semantic history vs. materialized view;
  - boundary event = an append-compatibility break, pending until acknowledged;
  - per-call evidence keyed by `(segment_id, context_epoch, compaction_event_id)`;
  - the generation contract with `training_eligible` and reasons;
  - guards evaluated on server tokens;
  - turn-chunked recompaction.
- **PORT:** the `required_prefix_token_ids` check (about 10 lines) into the capture path. It is the cheapest proof that a segment's token record is exact.

### Gaps
- No LLM summary policy ships.
- No per-segment reward.
- No resume of a live rollout.

---

## 2. verifiers v1: compaction, the message graph, limits and truncation

### Mechanism
- **Compactor** (`VF/verifiers/v1/harnesses/utils/compaction.py`):
  - **Trigger.** Proactive when `prompt+completion` usage ≥ threshold. The threshold is window − 16,384 (`RESERVE_TOKENS`, `:11`), floored at half the window (`default_threshold`, `:83-85`), and is discovered from the provider's model card (`:88-106`). It is checked after the tool results are appended (`utils/core.py:438-439`). There is also a reactive path on a provider overflow error, matched against 12 provider-specific markers (`compaction.py:40-80`, `:179-217`).
  - **Who writes the summary.** The **same policy model**, through the same client (and so through the interception proxy, which makes the summary a captured, sampled turn), with `tool_choice="none"` (`:226-237`). The prompt is a "CONTEXT CHECKPOINT COMPACTION … handoff summary for another LLM" (`:21-31`).
  - **What is kept.** **Only the system messages plus one user message**: `POST_COMPACTION_FRAMING + summary` (`:249-250`). The original task message is *not* kept verbatim. Reasoning never enters the summary, and a reasoning-only reply is resampled (`:243-248`).
  - **Robustness.** Up to 3 attempts. An overflowing checkpoint request falls back to the last usage-verified good prefix (`last_good`, `:165-177`, `:238-242`). After a failure the error is `CompactionFailed`, which becomes stop condition `compaction_failed`.
  - **Tool output bounding.** Middle-out truncation to 20 KB with a header giving the original size (`:18`, `:119-140`). It is applied to every tool message, including rewrites (`core.py:397`, `:435`).
- **Message graph** (`VF/verifiers/v1/graph.py`):
  - A rollout is a graph of `MessageNode`s. Each node stores only the **delta tokens it adds** to the cumulative sequence, so storage is linear in turns, and "concat(node.token_ids along a path) reproduces the exact prompt_ids + completion_ids the model saw" (`graph.py:1-19`).
  - Compaction, subagents and rewrites yield multiple leaves, and each root→leaf path is one **physical, exact-prefix training view** (`:3-11`).
  - Per-node fields (`:83-160`):
    - `sampled`: provenance. Fabricated assistant context is `False`.
    - `mask`, `logprobs`.
    - `advantages`: per-token. `None` means unassigned, which is distinct from 0.
    - `loss_weights`: named streams (`:152`).
    - `is_content`: body vs. scaffold.
    - `routed_experts`.
- **Branches and compaction trainability** (`VF/verifiers/v1/trace.py`):
  - `Branch` = one training sample. `advantages` and `logprobs` are spread onto sampled positions (`trace.py:192-266`).
  - `Trace.branches`: "A compaction attempt is a dangling physical leaf. It is trainable only when the final semantic graph shows that the harness resumed from it". A leaf with a `compaction_attempt` edge is trainable only if some node has a `compaction` semantic parent pointing at it (`trace.py:516-560`).
  - Harnesses publish typed semantic edges: `continuation`, `compaction_attempt`, `compaction`, `subagent_call`, `subagent_return`, `agent_message` and `subagent_cancel`. These record causation, not ancestry (`VF/verifiers/v1/semantic.py:61-72`).
- **Limits and truncation.**
  - `RolloutLimits{max_turns, max_input_tokens, max_output_tokens, max_total_tokens}` are checked between turns, and the turn that crosses a cap completes (`VF/verifiers/v1/session.py:83-115`).
  - `Trace.is_truncated` is true for any limit, for `compaction_failed`, or for a final `finish_reason == "length"` (`trace.py:621-632`).
  - The RLM harness exposes `summarize_at_tokens`, `max_compactions`, `max_compaction_attempts`, `max_total_turns`, `max_total_tokens` and `max_tool_output_bytes` (`VF/verifiers/v1/harnesses/rlm/harness.py:195-218`).
- **Resume.**
  - `Harness.resume` relaunches the program on the accreted conversation, meaning the current branch plus new messages (`VF/verifiers/v1/harness.py:240-275`). It is a continuation for multi-segment exchanges, not crash recovery.
  - `--resume` re-runs episodes that are not `complete` (`VF/verifiers/v1/env.py:175-179`). This is episode granularity.
- **Per-turn reward.** None is computed by the framework. Consumers write `MessageNode.advantages` / `loss_weights`, which is Crucible's injection point (DESIGN §6.4).

### Reuse for Crucible
- **DEPEND:** `graph.py` + `trace.Branch`. ARCHITECTURE M12 already lists this, and it is the best existing answer to both "token ids across compaction" and "linear storage".
- **PATTERN:**
  - the "compaction attempt trainable only if accepted" rule;
  - typed semantic edges;
  - `None` ≠ 0 for advantages;
  - the reserve-based threshold plus last-good fallback;
  - truncation that includes `compaction_failed`.
- **PORT (MIT):** `compaction.py` (256 lines) as Crucible's default summary compactor, **changed to keep the original task verbatim** (§(a)).

### Gaps
- No advantage computation across branches.
- No mid-episode resume.
- The framing prompt pretends that "another language model" wrote the summary. That is fine at inference but is a distribution choice that should be recorded as part of `harness@version`.

---

## 3. MiMo (xiaomimimo-verl + mimoagent-osr + uni_agent)

### Mechanism: two compaction styles, both written by the policy itself
- **A. Model-decided (`Compact` tool; CC and mimocode agents).**
  - There is no automatic threshold (`MO/compaction/surgery.py:1-8`).
  - The policy calls `Compact`, which only sets a flag. Compaction runs at the top of the next query, so the tool-call/response pair stays intact (`MO/tools/cc/compact.py:70-71`; `MO/agents/cc/cc_agent.py:146-155`).
  - **Pressure signal.** At 80% of (window − buffer), every tool result gets a `<context_usage>~used/~budget (pct%). Call Compact …` footer, explicitly so the model "can learn to act on" it under RL (`cc_agent.py:227-244`). The budget is 100k−10k for CC (`:64-65`) and 1M−10k for mimocode (`MO/agents/mimocode/mimocode_agent.py:71-72`). Tokens are estimated at about 4 characters per token (`MO/utils/cal_token.py:19-23`).
  - **Summary.** The same model, in a text-only turn with no tools (`cc_agent.py:174-177`). It uses a Claude-Code-style 9-section prompt with an `<analysis>` scratchpad that is stripped before storage (`MO/compaction/prompt.py:7-120`, `:148-155`). The model may append its own `instructions` (`prompt.py:133-138`).
  - **Rebuilt context** = `[system…, boundary marker(trigger, count, pre-tokens), <original_user_issue> verbatim anchor, summary + "if details are missing, state the gap"]` (`cc_agent.py:186-196`; `surgery.py:47-77`, `:108-132`). Recent turns are **not** kept verbatim.
  - An empty summary or an error leaves the history unchanged (`cc_agent.py:173-184`).
- **B. Codex-style automatic compaction.**
  - Triggered at `compaction_trigger_tokens`, or window − buffer, before each query, with a one-query cooldown (`MO/agents/codex/codex_agent.py:338-350`, `:378`). The example uses 900k (`XV/third_party/mimoagent-osr/example_configs/swe_codex_agent.yaml:19-21`).
  - The rebuild keeps **real user messages, newest first, up to 20k tokens** plus a handoff summary, and drops all assistant and tool messages (`surgery.py:161-200`).
- **Turning compacted rollouts into training rows (uni_agent gateway).**
  - Each request's messages are prefix-hashed into chains (`UA/gateway/session/session.py:678-702`).
  - A request extends a chain only if it matches the chain's prefix and uses the same tool schemas (`:611-663`, tool gate `:621`). Otherwise it starts a **new chain**; the test is "context compaction starts new chain" (`XV/third_party/uni_agent/tests/uni_agent/gateway/test_session_multiple_chains_on_cpu.py:367-387`).
  - Inside a chain, generated tokens get mask 1 plus engine logprobs and injected tokens get mask 0 (`session.py:279-299`, `:548-552`).
  - Each chain is written as its own row `{uid}_{session}_{chain}` (`UA/framework/framework.py:1592-1600`).
- **Reward and advantage.**
  - The session reward is copied to every chain and placed on the last response token (`framework.py:1512-1565`, `:1666-1669`).
  - Under GRPO, group statistics use only each session's **final** chain, and that scalar is broadcast to every chain's response tokens (`XV/verl/trainer/ppo/v1/utils.py:204-286`, called at `trainer_base.py:2053`).
  - Non-GRPO estimators (GAE) fall back to plain `compute_advantage`. Chains are then independent, so there is **no cross-segment GAE** (`utils.py:224-233`).
- **Loss normalization.** `loss_agg_mode: prompt-mean` (`XV/recipes/code/config/train.yaml:97`): weight = 1/(active prompts × total action tokens across **all sessions and chains of that uid**) (`XV/verl/trainer/ppo/core_algos.py:1160-1182`, `:1217-1221`). This is a token-mean pooled over segments, then a mean over prompts.
- **Surprises.**
  1. The shipped code recipe does **not** enable compaction: CC has no `Compact` tool and codex has `compaction_enabled: false` (`XV/config/agent/code/mini-claude-code.yaml:5-14`, `mini-codex.yaml:6`). It relies on a 256k window instead (`train.yaml:21-22`).
  2. `trajectory_selection: longest` (`train.yaml:81`) keeps only the chain with the most mask-1 tokens (`framework.py:253-280`). After a compaction, that is usually the *pre*-compaction segment, which then receives a reward it did not produce.
  3. The one shipped recipe that trains multiple segments is MemAgent. It processes chunks with a fresh context each, carries at most 1,024 memory tokens, uses `trajectory_selection=all` and copies the reward to every chain (`UA/agents/mem_agent/agent.py:26-39`, `:305-338`; `XV/third_party/uni_agent/examples/mem_agent/train_mem_agent.sh:182`).
- **Limits and truncation.**
  - `step_limit: 500`, session wall clock 7,200 s, idle timeout 720 s (`mini-claude-code.yaml:45`; `train.yaml:82-83`).
  - Truncated runs are **graded, not masked** (`finished: True, termination_kind: "truncated"`; `XV/recipes/code/mimoagent_runner.py:283-291`). Only `InfraError` is ungradable (`:42`).
  - A chain at the length cap is kept with `materialization_reason=max_trajectory_length` (`session.py:812-843`).
- **Dense signals.** No per-turn reward. Token-level penalty masks instead: tool-call errors, repetition, and an experimental `deep_failure_mask` (`XV/verl/trainer/ppo/v1/trainer_base.py:102-160`, `:205-250`).
- **Offload.** None. Tool output is middle-truncated at 40k characters (`MO/agents/default.py:69`; `MO/utils/truncate.py:32-41`). The mimocode bash tool explicitly has no spool file (`MO/tools/mimocode/bash.py:7-13`).
- **Resume.** Batch-level only (`MO/run/extra/batch.py:491`, `:612`). There is no resume of an in-flight rollout.

### Reuse for Crucible
- **PORT (MIT):**
  - the `Compact` tool + `<context_usage>` footer (`compact.py`, `cc_agent.py:227-244`) as a "model-decided" compaction mode;
  - `surgery.py` original-issue anchor and boundary marker;
  - the prompt (`prompt.py`).
- **PATTERN:**
  - a prefix-hash chain split at a compaction;
  - final-chain GRPO statistics broadcast to all chains;
  - prompt-mean pooled normalization.
- **AVOID:** `trajectory_selection: longest` with compaction.
- **DEPEND:** verl `recipes/general` is already planned (M12). It needs `trajectory_selection=all` + Crucible's per-call advantages.

---

## 4. Harbor: computer-1 compaction, terminus-2 summarization, ATIF continuation, load/resume

### Mechanism
- **computer-1 compactor** (`HB/src/harbor/agents/computer_1/compaction.py`):
  - Image-aware accounting: litellm undercounts screenshots, so each image is charged 1,300 tokens (`:30-34`, `:222-234`).
  - Stage 1 strips all but the last 3 screenshot turns (`:36-40`, `:168-195`).
  - Stage 2 is a proactive LLM summary when free tokens < threshold (`:121-154`). The summary is skipped if the history is already above 90% of the window (`:261-269`).
  - **Reactive path:** drop the oldest assistant/user pairs until the target free space is reached, then summarize (`:156-166`, `:236-253`).
  - Fallbacks: full summary → 2–3 sentence summary → a raw "Task + last 1,000 characters" (`:197-220`).
  - The rebuilt chat is `[first instruction (images stripped), user "Summary of previous work", assistant "Understood. I will continue…"]` (`:308-341`). The last message is a **fabricated assistant turn**, which a trainer must mask.
  - Tests: `HB/tests/unit/agents/computer_1/test_compaction.py:106-257`.
- **terminus-2 three-subagent summarization** (`HB/src/harbor/agents/terminus_2/terminus_2.py:865-1080`):
  1. A summary subagent receives the full (possibly unwound) history.
  2. A *questions* subagent with **no history** asks at least 5 questions about what is missing.
  3. An *answers* subagent with the full history answers them.
  - The main chat becomes `[system, question_prompt(+summary), model_questions]`, followed by the answers as a handoff prompt (`:1056-1080`).
  - The proactive threshold is 8,000 free tokens (`:120-122`, `:278`, `:1083-1107`).
  - Each subagent is saved as its own ATIF trajectory and linked through `subagent_trajectory_ref` (`:984-995`).
- **ATIF support for segments:**
  - `Step.is_copied_context`: "Steps marked as copied context should not be included in training data" (`HB/src/harbor/models/trajectories/step.py:64-72`).
  - `Trajectory.continued_trajectory_ref` links continuation files (`trajectory.py:79-82`).
  - With `linear_history`, terminus-2 dumps the segment and starts `session-cont-N`. All carried-over steps are marked copied (`terminus_2.py:1441-1456`, `:1997-2023`).
- **Load and resume.**
  - `agent.load_trajectory` seeds the first step's session from a native `.jsonl` or an ATIF `.json` (converted with `atif_to_native_trajectory`) (`HB/src/harbor/trial/trial.py:1195-1272`; `HB/src/harbor/agents/installed/base.py:1170-1200`).
  - `agent.resume_trajectory` continues the previous step's native session in `[[steps]]` tasks (`HB/src/harbor/trial/multi_step.py:53-67`, `:393-397`). An agent without the capability fails before any spend.
  - `BaseAgent.resume`/`load` raise `NotImplementedError` unless overridden (`HB/src/harbor/agents/base.py:384-401`).
- **Timeouts.** `asyncio.wait_for(run, timeout_sec)` → `AgentTimeoutError`. The verifier still runs (`trial.py:630-659`).
- **Live trajectory.** `sync_trajectory` polls over SSH every 2 s, tars **every** session `.jsonl`, reconverts it and atomically rewrites `trajectory.json` (`HB/src/harbor/trial/sync_trajectory.py:19-100`). It is crash-tolerant but **O(n²)** over a long run.
- **`continue_until_timeout`** (LHTB) is **not implemented** in Harbor. `AgentConfig` has no such field and pydantic silently drops it (`HB/src/harbor/models/task/config.py:219`, `:340-345`). The dataset patch lives in the external repo `zli12321/LHTB` (`DS/IntelligenceLab__Long-Horizon-Terminal-Bench/README.md:40-58`).
- **Cross-check from OpenEnv's Harbor seam:** terminus-2 summarization is turned **off** for training captures because "a summariser is a different task. Training its turns with this rollout's reward is exactly the contamination to avoid. Compaction also breaks prefix stitching" (`OE/src/openenv/harbor/seams.py:313-329`).

### Reuse for Crucible
- **DEPEND:**
  - ATIF `is_copied_context`, `continued_trajectory_ref`, `subagent_trajectory_ref` (Crucible already uses ATIF);
  - `load_trajectory` for C3 branch rollouts and for crash resume of installed agents.
- **PATTERN:**
  - the staged fallback chain (cheap strip → summary → short summary → raw), so compaction never kills the episode;
  - image token charging.
- **Gap to fix when importing Harbor runs:** mark fabricated "Understood…" turns `sampled=False`.

---

## 5. agentenv-framework (AE)

### Mechanism
- **Step journal.**
  - Pipeline steps (deploy, prompt, verify), *not* agent turns. Each completed step journals a `ContextUpdateOps` diff (`AE/src/agent_env/task/step_journal.py:1-102`).
  - Atomic sequence bump, journal-before-completion, and a generation epoch that retires entries without deleting them (`AE/src/agent_env/task/store.py:41-46`, `:412-433`, `:641-651`).
  - Rollback of a failed step plus its dependents, with `retry_from_step_id` (`AE/src/agent_env/task/task.py:165-200`; `task_step/task_step.py:53-81`).
- **Snapshot extension `urn:agentenv:snapshot/v1` + per-tool-call changelog.** This is the most relevant piece.
  - `save`/`load` of an opaque trajectory blob plus a workspace blob.
  - `changelog.apply{increments[sequence, object], resume_conversation, target_context_id}` with strictly increasing sequences (`AE/packages/agentenv-protocol/src/agentenv_protocol/a2a_agent/extensions.py:135-220`, `:981-1024`).
  - `changelog_apply_call(..., up_to_tool_call_exclusive=N)` replays a capture up to tool call N (`AE/src/agent_env/a2a_agent/object_transfer.py:533-570`). Partial applies are refused (`:573-582`).
  - `DeployAgentTaskStep` can **rewind a fresh agent to tool call N (filesystem + conversation) and resume** (`AE/src/agent_env/task_step/task_steps/deploy_agent.py:513-560`).
  - Increments are drained to the object store every 5 s while the agent runs, so loss is bounded by that interval (`AE/src/agent_env/a2a_agent/staging.py:1-9`, `:45-50`).
  - Limits: 10k objects, 1 GiB each, 10 GiB total (`object_transfer.py:84-90`).
- **Mid-run capture series.**
  - `interval_seconds` / `per_turn` / `at_end`, at most 24 snapshots. Each capture holds a workspace snapshot, the cumulative partial trajectory and optionally the env DB (`AE/src/agent_env/task_step/snapshot_utils/snapshot_series.py:1-85`; `agent_state_capture.py:58-209`).
  - The index is published only at teardown (`snapshot_series.py:593-596`), so a crash orphans the objects.
  - Rubric criteria can be tagged `intermediate`/`process` (`AE/src/agent_env/task_step/task_steps/verifiers/judge_utils/judge_output_format.py:224-231`).
- **Env-side trace.** The MCP gateway writes an **append-only JSONL**, flushed per event, of `tool_call`/`tool_call_result`/trigger events with `event_id`, UTC time and virtual time. It is served as a chunked stream (`AE/src/agent_env/env/gateway/gateway.py:120-123`, `:968-1020`). The agent-side trajectory, by contrast, is one JSON blob fetched per turn (`object_transfer.py:597-675`).
- **Judge-side offload.** Tool results over 1 KB or containing images are externalized to side files before judging (`AE/src/agent_env/task_step/task_steps/verifiers/.../trajectory_filter.py:1-37`).
- **Limits.** `prompt_agent.timeout_seconds=600` per send. `max_turns` is passed through to the harness but not enforced. There is no token, tool-call or global wall-clock budget.

### Reuse for Crucible
- **DEPEND:** the snapshot/v1 + changelog protocol models (`agentenv-protocol`), for A2A agents in M10.
- **PATTERN:**
  - tool-call-indexed incremental checkpoints with "replay to N, resume conversation";
  - the 5 s staging drain;
  - a gateway JSONL that does not depend on the agent.
- **PORT:** the about 100-line pure `step_journal.py` replay rules. Crucible's `RunJournal` already covers most of this.

---

## 6. OpenEnv (OE)

### Mechanism
- **Capture graph** (`OE/src/openenv/core/harness/capture/graph.py`):
  - Every model call is a node. Its parent is the node whose prompt+completion is the longest exact token prefix of the new prompt (`:299-324`), with a message-based fallback (`:326+`).
  - **"Compaction is representable. A rewritten history is not a prefix extension, so it opens a new root instead of corrupting the chain"** (`:1-25`; RFC `OE/rfcs/006-agentic-rl-harness-interception.md:257-259`, "Fork, don't truncate").
  - Retries show up as childless siblings (`:408-441`).
  - `sequence_for` builds `input_ids`/`loss_mask`/`logprobs` per path (`:443-484`).
  - Limits:
    - The graph is **in memory only**, per session (`capture/sessions.py:140-147`).
    - Parent search is O(nodes × prompt length) (`graph.py:310-323`).
    - Token storage is full prompt per node, so quadratic.
- **Model-call budget.** On overflow, the proxy returns a synthetic `finish_reason=stop` reply in the caller's dialect, which is **never ingested into training data**. Rationale: "NO AGENT HARNESS HONOURS ITS OWN STEP CONFIG" (`sessions.py:158-178`; `server.py:965-1000`).
- **White-box loop.** `HarnessRunLimits{max_turns, max_tool_calls_per_turn, max_total_tool_calls}`. `truncated` is set only on the tool-call cap, and `max_turns` sets neither done nor truncated (`OE/src/openenv/core/harness/rollout.py:103-110`, `:617-701`). `prompt_ids.extend(...)` per turn makes memory quadratic (`:621`).
- **Rubrics.**
  - `TrajectoryRubric` returns `intermediate_reward` until done, then the trajectory score. `compute_step_rewards` assigns credit, with an exponential-discount variant (`OE/src/openenv/core/rubrics/trajectory.py:1-211`).
  - The code itself warns: "Very long episodes (thousands of steps)… consider streaming rubrics" (`:32-37`; `OE/rfcs/004-rubrics.md:354-355`).
  - Containers: Sequential, Gate, WeightedSum and others (`rubrics/containers.py:27-386`).
- **State and checkpointing.** The `Environment` interface has no save/restore (`OE/src/openenv/core/env_server/interfaces.py:262-310`). Suspend/resume is "orchestration-only" and tool-aware checkpointing is future work (`OE/rfcs/002-env-spec.md:195-197`, `:539`).
- **Persistence.** `RolloutSerializer` writes an append-only `results.jsonl` with a torn-tail-safe separator, one line per **episode**. `CollectRunner.run(resume=True)` skips finished episode ids (`OE/src/openenv/core/harness/collect.py:108-185`, `:265-320`).
- **Offload pattern.** repl_env (RLM-style) keeps the long context in REPL variables that sub-LLMs query (`OE/envs/repl_env/prompts.py:65`). Output is capped at 8,192 characters (`envs/repl_env/server/app.py:60`).

### Reuse for Crucible
- **PORT (BSD-3):** `RolloutGraph` (about 500 lines) as a fallback when only a proxy is available, plus the synthetic-stop budget technique.
- **PATTERN:**
  - compaction = new root;
  - streaming rubrics are what OE says it lacks;
  - torn-tail-safe JSONL.
- **Prefer** VF's delta-token graph for storage.

---

## 7. Long-horizon datasets (`DS`)

- **IntelligenceLab Long-Horizon-Terminal-Bench (LHTB)**
  - **Size and coverage.** 46 tasks (`DS/IntelligenceLab__Long-Horizon-Terminal-Bench/README.md`, `files/eval.yaml`). `tests/` and `solution/` are held out on purpose, so the local mirror has only the card plus 2 sample tasks.
  - **Horizons.** Agent timeouts run 1–4 h; for example 2048 is 14,400 s and matpower is 3,600 s + `continue_until_timeout` (`sample_task/tasks/2048/task.toml:18-34`; `matpower-opf-regression/task.toml:26-43`). Expert estimates are 30–240 min. The tasks are framed as "hundreds of steps", and **no token budget** is stated.
  - **Grading.** Continuous 0–1 reward, but the leaderboard metric `lhtb_solved` counts tasks with reward ≥ 0.95. It replaced mean reward (`files/eval.yaml`). Styles vary:
    - 2048: replay `moves.log` in a fresh seeded engine and take the **peak** banded score (`environment/harness/game2048_engine.py:39-41`, `:140-155`).
    - apex-*: 38 strictly ordered stages unlocked through `inbox`/`validate`/`submit` tools; the score is the fraction of 146 numeric fields within tolerance, revisions are allowed and there is no LLM judge (`first_rows.json`). These are de-facto **checkpoint graders**.
    - alp: 9 gates, and partial progress scores 0.
  - **`continue_until_timeout`** (30 of 46 tasks). When the agent stops, the harness runs the hidden verifier and, if it has not passed, **resumes the same session with verifier feedback** until timeout or pass (`README.md:40-58`). The agent is told this in its instruction. Stock Harbor ignores it (§4).
- **FrontisAI NatureBench-Harbor**
  - 12 h agents with 1 GPU.
  - An eval sidecar accepts unlimited `/evaluate` calls and keeps `best_aggregate_improvement` in `/state/best_score.json`. The final reward is the **best attempt**, read by a separate verifier that writes `reward.txt` + `reward.json` (`DS/FrontisAI__NatureBench-Harbor/sample_task/s41587-024-02428-4/task.toml:16-48`, `instruction.md:18,48,67`, `tests/verifier.py:231-306`).
- **Other long-budget sets:**
  - terminal-bench-science: 8 h.
  - openai-math: 4 h, binary Lean reward.
  - BeyondSWE, MiMo-harbor-code: 1 h.
  - No other dataset uses `continue_until_timeout` or `[[steps]]`.

---

## (a) Compaction inside RL so that training and inference match

| Decision | Evidence | Recommendation for Crucible |
|---|---|---|
| **Who writes the summary** | The policy itself in VF (`compaction.py:226-237`), MO (`cc_agent.py:174-177`), HB computer-1 (`compaction.py:282-285`) and HB terminus-2 (three subagents of the same LLM). OE turns it off rather than train it with the wrong reward (`seams.py:313-329`). | **The policy writes it, and the summary turn is trained** with the episode's advantage. It is an action whose quality decides everything downstream; CompactionRL's "summaries share the task reward" says the same (survey §3.6). A separate summarizer model would make the deployed distribution differ. |
| **Trigger** | Threshold + reactive in VF (window − 16k) and HB (8k free). Codex-style in MO (window − buffer). Model-decided in MO (`Compact` tool + `<context_usage>` footer). | Make both modes available and record which one ran in `harness@version`. Use the model-decided mode for training-time learning of *when* to compact, always with a hard threshold backstop. Measure on the **server tokenizer** (NG guards, `app.py:190-220`), not 4 characters per token. |
| **What is kept** | VF: system + summary only (`compaction.py:250`), **so the original task is lost**. MO CC: system + boundary + verbatim original issue + summary (`surgery.py:47-132`). MO codex: user messages ≤ 20k + summary. HB: first instruction + summary + a fabricated ack. NG: deterministic recency (images/reasoning). | System prompt + **verbatim task anchor** + summary + a boundary marker (count, pre-tokens). Do **not** fabricate assistant turns; if a harness does, mark them `sampled=False` / `is_copied_context`. Recency pruning of images and reasoning (NG/HB) is a cheap first stage before summarizing. Environment-provided state (files, the World) stays reachable through tools, so the summary need not carry it. |
| **Token ids across a compaction** | Every framework does the same thing: **a compaction starts a new exact-prefix sequence**. That is a new root in OE `graph.py:1-25`, a new leaf/branch in VF `graph.py:3-11`, a new chain in UA `session.py:467-477`, and a new `segment_index` in NG `controller.py:88-103`. Exactness inside a segment is enforced by NG's `required_prefix_token_ids` (`history.py:907-913`). | A segment = one root→leaf path. Every model call carries `segment_id` + `parent_call_id`. The boundary event records reason (`threshold` / `model` / `reactive` / `policy_rewrite`), pre/post tokens, summary call id and view digests. Check the prefix inside a segment (port NG's check); a mismatch becomes `capture_incomplete` → masked. |
| **Is the summary segment trained** | VF: a compaction *attempt* branch is trainable only if the harness resumed from it (`trace.py:516-560`). MO/UA: the summary is its own chain, trained only with `trajectory_selection=all`. HB: summary subagent trajectories are linked but untrained. | Adopt VF's rule: a summary is trainable only if it was **accepted** (the next segment's root contains it), and rejected or resampled attempts are masked. All segments are trained (never `longest`). |
| **Loss normalization** | UA `prompt-mean`: 1/(prompts × total action tokens over all chains of the prompt) (`core_algos.py:1160-1221`). DAPO token-level loss (survey §3.5). | **Token-mean over the whole logical rollout group** (all segments pooled), then a mean over prompts. Per-segment seq-mean would overweight short summary segments and penalize episodes that compact more often. Per-segment `loss_weights` streams (VF `graph.py:152`) let the summary's weight be ablated. |
| **Advantage across segments (GRPO)** | UA: group statistics on the final chain per session, broadcast to all chains (`utils.py:204-286`). NG: one reward and advantage per logical rollout (`osworld_agent/README.md:59-63`). | Group statistics come from **logical rollouts**, never physical segments: group size = n rollouts regardless of compaction count. M6's per-turn term A_t = A^GRPO + η(ΔΦ_t − c)·σ_R/σ_ΔΦ is keyed by `call_id`, so it **already crosses segments** once each call knows its segment. The summary call gets A^GRPO plus the ΔΦ of its step (normally 0, since a summary changes no world state). |
| **Cross-segment GAE (critic mode)** | **No framework implements it.** UA falls back to per-chain `compute_advantage` (`utils.py:224-233`). It exists only in CompactionRL (survey §3.5). | Implement in M12 `compaction.py`. Order sampled tokens by logical time across segments with skip-observation GAE (SAO), and **bootstrap across the boundary**: the TD target of segment k's last token uses V(first sampled token of segment k+1), which the critic computes on the *compacted* context. γ=1 and λ apply over the logical chain. A truncated final segment bootstraps with Φ or V (DESIGN §10.3 point 6). |
| **Harness variance** | Kimi varies the context mechanism per group (survey §3.8). | Sample the compaction mode and threshold per GRPO group (one per group), recorded in `harness@version`. This matches DESIGN §10.3 point 7. |

## (b) Dense but hack-resistant reward on very long trajectories

**What already exists:**
- **Per-step rubric reward:** OE `TrajectoryRubric.intermediate_reward` / `compute_step_rewards` with discounting (`trajectory.py:1-211`). It keeps the whole trajectory in memory, and streaming rubrics are an admitted gap (`:32-37`).
- **Checkpoint graders inside the environment:**
  - LHTB apex: ordered stages unlocked by a `validate`/`submit` tool, scored as the fraction of fields correct.
  - LHTB 2048: the peak band, from a deterministic replay of the action log.
  - NatureBench: an unlimited `/evaluate` sidecar with best-so-far kept in `/state` and read by a separate verifier.
- **Multi-stage tasks:** Harbor `[[steps]]` with per-step verifier, `min_reward` gate and `multi_step_reward_strategy=mean|final` (`HB/src/harbor/models/task/config.py:760-825`).
- **Verifier-feedback loop:** LHTB `continue_until_timeout` (`README.md:40-58`).
- **Mid-run captures with intermediate rubric criteria:** AE (`snapshot_series.py`, `judge_output_format.py:224-231`).
- **Token-level penalty channels (not rewards):** MiMo tool-error, repetition and deep-failure masks (`trainer_base.py:102-250`).
- **Nobody implements potential-based shaping with a telescoping guarantee.** Crucible's M6 (DESIGN §6.4) is already ahead here.

**Recommendation (an extension of DESIGN §6.2–6.4, nothing new in kind):**
1. **Φ over state predicates evaluated on World snapshots, not on agent text** (R6). Cadence follows milestone `cost`:
   - `cheap` predicates on every step;
   - `medium` every N steps (`snapshot_every`);
   - `expensive` terminal only, or on a sampled checkpoint.

   ΔΦ telescopes, so oscillation earns nothing (R4), and the term sits inside the per-turn advantage with small η. The `steps.jsonl` row is keyed by `call_id`, so it crosses segments.
2. **Checkpoint graders = milestone predicates with prerequisites** (`requires:`). This is the apex-style ordered unlock, but evaluated by the hidden verifier on a snapshot, so the policy cannot claim a stage. Stubbing a milestone without its prerequisites earns ≤ (R5).
3. **Per-segment returns are derived, not separate rewards.** A segment's return = Σ ΔΦ over its calls + (final segment) the gated terminal R. Use it for monitoring (which segment lost progress? did compaction lose state?) and as the critic-mode bootstrap target. Never use it as an independent GRPO reward, because that would multiply group statistics by segment count.
4. **Hack resistance specific to long horizons:**
   - **Feedback loops leak the grader.** `continue_until_timeout` and NatureBench's unlimited `/evaluate` turn verifier output into observations. Only the **fast-tier public checks** may be fed back. The hidden full tier stays terminal-only, and a feedback channel counts as a declared tool in the A13 attack suite.
   - **Best-of/peak rewards** (NatureBench best attempt, 2048 peak) are acceptable only when the measured state is re-derived by a replay in a clean environment, as 2048 does, and never read from agent-writable files.
   - **Compaction can hide hacks from judges:** the semantic trace must keep the full uncompacted history (the NG `SemanticHistory` pattern), and judges and A13 monitors read that, not the policy's view.
   - **Truncation ≠ failure:** a truncated rollout is Φ-bootstrapped or masked (DESIGN §10.3 point 6). This matters more at 1M tokens, where truncation is common. MiMo grades truncated runs as failures (`mimoagent_runner.py:283-291`), which biases toward short tasks.

## (c) Resuming a long episode mid-way after a crash

**What exists:**
- **Episode-granularity resume everywhere:**
  - VF `--resume` (`env.py:175-179`);
  - OE `CollectRunner.run(resume=True)` (`collect.py:265-320`);
  - Harbor job resume (`HB/src/harbor/cli/jobs.py:2294+`);
  - MiMo batch resume;
  - Crucible `RunJournal`.
- **Mid-episode, but manual or agent-dependent:**
  - **AE changelog rewind:** replay workspace + conversation up to tool call N, then `resume_conversation=True` (`deploy_agent.py:513-560`; `object_transfer.py:533-582`). Increments are drained every 5 s. The agent runtime must implement the extension.
  - **Harbor `load_trajectory`:** seed a native or ATIF session into a fresh environment (`trial.py:1195-1272`). It restores the *conversation only*; the environment must be restored separately.
  - **NG:** retry-safe pending boundaries make the *compaction step* idempotent under retry (`controller.py:69-86`), but there is no persistence.
- **Nobody** checkpoints (world state + policy context + rollout record) atomically and resumes automatically.

**Recommendation.** Compaction makes cheap resume possible: the policy context at any step is **only the current segment** (≤ one window). Restoring it never means replaying millions of tokens. An **episode checkpoint** at step t holds:
1. `WorldRef` at t, which already exists (`run_episode` `snapshot_every`, quiesce off);
2. the **current segment's** policy context: the messages and the exact token ids of the segment prefix (the last call's `prompt_ids + completion_ids`), plus queue and pending tool-call state;
3. the rollout-record prefix, already on disk under the streaming layout in (e);
4. sample key, seed counters and the policy version of each span.

These go to the attempt directory atomically (the `journal.atomic_write` pattern) with a `checkpoint` journal event.

On resume:
- restore the World;
- re-prefill **one segment** (≤ window, prefix-cached);
- continue.

Spans generated after the resume carry the new `policy_version`, which is exactly the partial-rollout case of DESIGN §10.3 point 3, so the staleness cap applies.

If the token prefix cannot be reproduced byte-exactly (a different engine or template), keep the pre-crash segments trainable and **mask only the resumed segment**. Never guess.

Black-box harnesses (Harbor installed agents, A2A agents) use their native mechanisms instead: Harbor `load_trajectory`, AE snapshot/changelog. The World is restored from Crucible's own snapshot at the matching tool-call index.

## (d) Observation offloading and memory tools

| Framework | What it does |
|---|---|
| VF | Middle-out truncation to 20 KB with the original size stated (`compaction.py:119-140`). |
| MO | Middle truncation at 40k characters with a banner, and explicitly no spool file (`truncate.py:32-41`; `bash.py:7-13`). |
| Crucible | Head/tail `max_tool_output` (`policy.py:103-106`). |
| NG deepagents | Evicted history goes to `/conversation_history/{rollout_id}.md`, and the agent reads it back with `read_file` (`langchain_deepagents_agent/README.md:170-175`). This is the one real "offload + pointer" in the corpus. |
| AE | Offloads large tool results to side files, but only for judges (`trajectory_filter.py:1-37`). |
| OE repl_env | Context lives in REPL variables queried by sub-LLMs (`prompts.py:65`). |
| MiMo MemAgent | A fixed 1,024-token memory carried across fresh contexts (`mem_agent/agent.py:26-39`). |
| NG / HB | Recency pruning of images and reasoning, with lossless lineage in a media arena (`policies.py:112-200`; `computer_1/compaction.py:168-195`). |

**Recommendation.**
- **Offload at the World's tool choke point.** Outputs above N tokens are written to a file in the agent's workspace, `/.crucible/obs/<call_id>.txt`. The policy sees head + tail + path + size + a hint to read slices. The full output goes into the trace as a content-addressed blob (sha256), so graders and judges always see everything.
- **Expose memory as an ordinary World app.** A notes file or `memory.*` tools, whose state the snapshot captures, so resume and C3 branching restore it.
- **Never credit memory writes through Φ.** Milestone predicates must read task state, not notes, or "write that I did X" becomes a hack.
- **Treat the offload threshold and memory tool as harness parameters** (`harness@version`), varied across groups like compaction.

## (e) Storage for multi-million-token episodes

**What exists:**
- **Linear storage:**
  - VF `MessageNode` stores **delta tokens only**, so a branch is a concatenation (`graph.py:1-19`).
  - NG dedupes media by content and keeps only a digest of the token arrays in transport (`history.py:172-227`; `session.py:84-127`).
  - UA writes `trajectory.json` + compressed `.npz` per session (`framework.py:1346-1360`).
- **Quadratic storage:**
  - OE graph and white-box loop (`rollout.py:621`).
  - Crucible `Turn.prompt_token_ids` (`rollout.py:30`).
  - Harbor `sync_trajectory`, which re-tars and rewrites everything every 2 s (`sync_trajectory.py:19-100`).
- **Streaming:**
  - AE gateway JSONL, flushed per event and served chunked (`gateway.py:968-1020`).
  - Crucible `RunJournal`, fsynced with torn-tail tolerance.
  - OE `RolloutSerializer`, but per episode (`collect.py:108-185`).
- **Segment files:** Harbor `linear_history` writes one ATIF file per segment, linked by `continued_trajectory_ref` (`terminus_2.py:1997-2023`).

**Recommendation: a streaming, segment-sharded `crucible.rollout/v1` layout under `samples/<id>/a<N>/`.**
- `calls.jsonl`: one fsynced line per model call: `call_id`, `segment_id`, `parent_call_id`, **delta** `token_ids` (base64 int32), logprobs (base64 float32), mask run-lengths, `policy_version`, sampling, finish, `atif_step_ids`. Size is linear in tokens generated plus observed. The full prompt of call k = the concatenation along its parent chain, as in VF.
- `segments.jsonl`: boundary events (the NG `RewriteBoundaryEvent` fields + summary call id + trainable flag).
- `trace.jsonl`: World events, appended live instead of `to_trace()` at the end.
- `blobs/<sha256>`: large observations, screenshots and offloaded outputs.
- `steps.jsonl`: unchanged.
- `checkpoints/<step>.json`: per (c).
- `rollout.json`: becomes a small **manifest** (ids, reward, gates, truncation, failure/blame, per-file digests). `valid_for_training` checks token completeness by streaming `calls.jsonl`.
- ATIF export stays one file per segment, linked by `continued_trajectory_ref`, with copied context marked.
- A trainer bridge reads branches lazily, segment by segment.
- Rough budget: 1M tokens of int32 ids + float32 logprobs (sampled only) ≈ 4–8 MB per rollout, compared with gigabytes for the current per-call full prompts.

---

## Recommendation table

| Crucible piece | Mode | Source |
|---|---|---|
| Segment model: `segment_id`, `parent_call_id`, boundary event (reason, pre/post tokens, view digests, summary call, trainable) replacing `rollout.Segment` | PATTERN | NG `controller.py:88-165`, `history.py:407-430`; VF `semantic.py:61-72` |
| Exact-prefix check inside a segment → `capture_incomplete` on mismatch | PORT (Apache-2.0) | NG `history.py:885-913`, `session.py:345-351` |
| Token record storage: delta tokens per call, branch = concatenation | DEPEND (or PORT, MIT) | VF `graph.py` `MessageNode`, `trace.py:192-266` |
| Proxy-only capture across compaction/subagents (black-box harnesses) | PORT (BSD-3), with delta storage added | OE `capture/graph.py` (compaction = new root) |
| Default threshold compactor with last-good fallback, reactive overflow detection | PORT (MIT), modified to keep the verbatim task anchor | VF `harnesses/utils/compaction.py` |
| Model-decided compaction: `Compact` tool + `<context_usage>` footer; 9-section prompt; original-issue anchor | PORT (MIT) | MO `tools/cc/compact.py`, `agents/cc/cc_agent.py:146-244`, `compaction/prompt.py`, `surgery.py:47-132` |
| Staged fallback (strip images → summary → short summary → raw) | PATTERN | HB `computer_1/compaction.py:121-220` |
| Summary turn trainable only if accepted; fabricated turns `sampled=False` | PATTERN | VF `trace.py:516-560`; HB `step.py:64-72` |
| Group statistics on logical rollouts; summary shares the episode advantage; M6 per-call ΔΦ crosses segments | PATTERN (credit is Crucible's own) | UA `verl/trainer/ppo/v1/utils.py:204-286`; NG osworld README:59-63; Crucible `credit/compose.py` |
| Token-mean loss pooled over all segments of a prompt | PATTERN | XV `core_algos.py:1160-1221` (`prompt-mean`) |
| Cross-segment GAE with boundary bootstrap (critic mode) | NEW (no framework has it) | CompactionRL (survey §3.5); SAO skip-observation GAE |
| Compaction mode/threshold varied per GRPO group, recorded in `harness@version` | PATTERN | survey §3.8 (Kimi); DESIGN §10.3 point 7 |
| Context guards on server tokens (`max_total_tokens + reserved_generation`) | PATTERN | NG `controller.py:385-428`, `app.py:190-220` |
| Limits incl. token caps; `compaction_failed` and `length` count as truncation | PATTERN | VF `session.py:83-115`, `trace.py:621-632` |
| Model-call budget enforced at the proxy by a synthetic stop that is never trained | PORT (BSD-3) | OE `capture/sessions.py:158-178`, `server.py:965-1000` |
| Dense reward: Φ over snapshot predicates at milestone cost cadence; ordered checkpoint milestones | Crucible M6 (keep); PATTERN for staged unlock | LHTB apex (`first_rows.json`); Harbor `[[steps]]` `min_reward` (`config.py:760-825`) |
| Verifier-feedback loop (`continue_until_timeout`) using the fast tier only, declared to A13 | NEW in M10 (Harbor lacks it) | LHTB `README.md:40-58` |
| Streaming per-step rubric (OE's admitted gap) | NEW | OE `rubrics/trajectory.py:32-37` |
| Episode checkpoint = WorldRef + current-segment context + rollout prefix + versions; resume re-prefills one segment | NEW, on Crucible `World.snapshot` + `RunJournal` | Pattern from AE snapshot/v1 changelog (`object_transfer.py:533-582`, `staging.py`) |
| Resume for black-box agents | DEPEND | Harbor `load_trajectory` (`trial.py:1195-1272`); AE `urn:agentenv:snapshot/v1` (`extensions.py:981-1024`) |
| Observation offload to workspace file + pointer, full blob in trace | PATTERN | NG deepagents (`README.md:170-175`); AE `trajectory_filter.py` |
| Memory as a World app, never credited by Φ | PATTERN | MiMo MemAgent (`mem_agent/agent.py`); OE repl_env |
| Image/reasoning recency pruning with a lossless semantic history | PATTERN | NG `policies.py:112-200`, `history.py:172-227`; HB `computer_1/compaction.py:30-40` |
| Streaming storage: `calls.jsonl` (delta tokens) + `segments.jsonl` + `trace.jsonl` + `blobs/` + manifest `rollout.json` | PATTERN | AE `gateway.py:968-1020`; Crucible `journal.py`; UA `.npz`; Harbor `continued_trajectory_ref` |
| ATIF export per segment with copied context marked | DEPEND | Harbor `trajectory.py:79-82`, `step.py:64-72`, terminus-2 `linear_history` |
| Avoid: `trajectory_selection: longest` with compaction; a single trajectory blob rewritten on every poll; full prompt ids per call | (anti-pattern) | XV `recipes/code/config/train.yaml:81`; HB `sync_trajectory.py`; Crucible `rollout.py:30` |
