# M5/M6/harness research, slice 03: Prime Intellect code (verifiers v1, prime-environments, research-environments)

Date: 2026-10-09. Read-only; no network, no installs. Every mechanism below is cited `path:line` against the local checkouts.

**Path aliases** (all under `RL-Environments/06-frameworks-and-env-code/`):

| Alias | Path |
|---|---|
| `VF/` | `primeintellect-ai-verifiers/verifiers/v1/` (the package; v0 `Rubric`/`JudgeRubric`/`RubricGroup` are **not in this checkout**: `verifiers/__init__.py` only re-exports v1, and the migration table maps v0 `Rubric` → `@vf.reward`, `primeintellect-ai-verifiers/skills/create-environments/SKILL.md:229-240`) |
| `VFR/` | `primeintellect-ai-verifiers/` (repo root: `environments/`, `docs/`, `skills/`) |
| `PE/` | `primeintellect-ai-prime-environments/environments/` (109 community envs, v0 API: `vf.Rubric(funcs, weights)`, `vf.JudgeRubric`, `vf.RubricGroup`) |
| `RE/` | `primeintellect-ai-research-environments/` (PI research team, v1 API) |
| DES | `crucible/DESIGN.md` |

**Not found locally:** prime-rl (the trainer that consumes `MessageNode.advantages`), `prime_runs.projection.build_samples` (imported at `VF/utils/platform.py:10-13`), and the v0 `verifiers.rubrics` sources. Anything I say about the trainer side is inferred from the verifiers-side contract and is marked as such.

---

## 1. TL;DR: the most valuable takeaways

**Reward plumbing (M6 `compose.py`, export adapters)**

1. **verifiers composes rewards as a plain weighted sum, so it cannot express a gate.**
   - `Trace.reward = Σ score·weight` over named rewards (`VF/trace.py:135-141,472-474`). Weights come from `@vf.reward(weight=…)` (`VF/utils/decorators.py:118-131`, applied at `VF/task.py:264-273`). Metrics are recorded but never summed (`VF/utils/decorators.py:105-115`).
   - **Crucible should** compute the whole gated R = G·[O·(1+αp+βe) + κ(1−O)Φ] inside **one** `@vf.reward` (weight 1), and emit every channel (gates, O, Φ, p, e, Φ_mid) as `@vf.metric`s. Never export channels as separate weighted rewards: the sum would let a failed gate be traded for partial credit.

2. **Crucible's `reward.json` as specified in DES §4.3.1 will be rejected by verifiers' Harbor taskset.**
   - The reader is strict: `float | dict[str, float]` with `strict=True` (`VF/tasksets/harbor/taskset.py:51-55`). Crucible's nested `"gates": {...}` and `"infra": {"status": "ok"}` fail validation. The reader then falls back to `reward.txt` and raises `TaskError` if that is missing too (`:370-403`).
   - With a `reward` key, that key is the score and every other key becomes a metric. **Without** a `reward` key, every key becomes a separate weight-1 reward and is summed (`:379-386`; `VFR/docs/v1/harbor.md:176`).
   - **Crucible should** write a flat `reward.json` (`{"reward": R, "gate.build": 1.0, "outcome": 0.0, "progress": 0.86, "infra.ok": 1.0, …}`, every value finite) and put the rich nested record in a second file, read by a `HarborTask` subclass into `trace.info`.

3. **The dense channel has no home in verifiers' reward path; it must ride on the trace.**
   - Nothing in verifiers computes or assigns `advantages`. The field is only declared (`VF/graph.py:138-142`) and read (`VF/trace.py:258-266`). The docstring says it is "`None` until a consumer's RL algorithm assigns it".
   - `Episode.group` is "consumer-assigned" (`VF/episode.py:25-29,96-97`). The env cannot see group statistics, so the A_t of DES §6.4 (which needs group mean and std of shaping-return-to-go) **must be stamped consumer-side**.
   - **Crucible should** put `steps.jsonl`-equivalent rows into `trace.info["crucible"]["steps"]`, keyed by **message-node index** (plus `tool_call_id`), and ship a small consumer hook that stamps `node.advantages`. The hook goes either into prime-rl (not in tree, so it needs a PR or plugin) or into Crucible's own trainer adapter (§2.1.3 gives the exact recipe).

4. **The `advantages` and `loss_weights` layouts differ, and a partially stamped branch silently trains with zero advantage.**
   - `advantages` is **compact over sampled tokens**: the same layout as `logprobs`, with length = Σ`mask` (`VF/graph.py:135-142`).
   - `loss_weights` is a dict of named streams aligned to **all** `token_ids` of the node; a length mismatch raises (`VF/graph.py:152-153`; `VF/trace.py:289-308`).
   - `Branch.advantages` returns `None` only if *no* node on the path was assigned. Otherwise every unassigned node reads 0.0 (`VF/trace.py:258-266`, via `spread` `:229-250`).
   - **Crucible should** stamp every sampled node on every trainable branch, and assert this in the adapter test.

**Online tracking and replay (M5 tracker, `steps.py`)**

5. **The step boundary is the sampled assistant node, and nodes are shared across branches.**
   - A rollout is a message DAG. Each node stores only the tokens it adds, and nodes are deduplicated by `(parent, tools-hash-at-root, message_hash)` (`VF/graph.py:1-18,324-331`).
   - Compaction and subagents create extra leaves. Each leaf is one training sample (`VF/trace.py:192-199,515-553`). A compaction *attempt* is non-trainable unless a `compaction` semantic edge shows the harness resumed from it (`:519-549`).
   - **Crucible should** key Φ, ΔΦ and milestones by node index, not by "turn k of branch b". One node, one credit, even when it appears in several branches.

6. **verifiers has a per-tool-result hook that fits the Workflow State Tracker.**
   - `@vf.intercept` request hooks run at **each new tool-result boundary**, parallel calls included, before the next model call (`VF/session.py:262-268,276-284`). They receive only `Request` and `Trace`, so a runtime handle must be stashed by `Task.setup(trace, runtime)`; that is a pattern, not an official API.
   - Any exception in a hook becomes `TaskError("request interception failed")` and fails the rollout (`VF/session.py:345-350`). **Wrap the tracker in try/except and record a C6 flag instead of raising.**
   - `@vf.stop(trace)` runs before every model call (`VF/session.py:589-607`), which gives early termination on a gate violation such as test tampering.
   - Map the cost tiers onto the hooks:
     - **cheap checks** → intercept hook;
     - **medium checks** → `Task.finalize`;
     - **expensive checks** → `IsolatedVerifierEnv` / Harbor separate verifier (`VFR/docs/v1/env.md:23-86`).

7. **Do not take per-step state with verifiers' `capture_patch` mid-rollout.**
   - `capture_patch` runs `git add -A … git reset -q` (`VF/utils/git.py:39-66`), which would wipe the agent's own staging area.
   - **Crucible should** snapshot each step boundary with an alternate index: `GIT_INDEX_FILE=$tmp git add -A && git diff --cached --binary $BASE`, stored host-side and diffed against a host-held base SHA (the `resolve_head` pattern, `VF/utils/git.py:73-84`).
   - Then **replay** those cumulative diffs inside the isolated verifier box to compute Φ(s_t) for every step. This is exactly M5's replay. Hidden checks never enter the agent box.

**Branching (C3)**

8. **verifiers has no environment-state snapshot or fork.**
   - "Sandbox snapshotting will make this free: once runtimes can snapshot…" (`VF/utils/git.py:94-96`). Trace "branches" are context branches, not world branches.
   - The runtime set is closed: a discriminated union plus a hard-coded class map (`VF/runtimes/__init__.py:42-74`). A Crucible runtime cannot be plugged in without a fork.
   - **Crucible should** implement C3 as follows:
     - materialize `WorldRef_t` as a container **image**;
     - set `TaskData.image` to it, or provision a box and pass it as `agent.run(task, runtime=box)` (`VF/agent.py:649-662`; there are no retries on a borrowed box, `:680-685`);
     - run K continuations with `-r K` or `BestOfNEnv(n=K)` (`VF/envs/best_of_n/env.py:26-44`), then compute V̂ = mean O consumer-side. `BestOfNEnv` records only `best` and `pass_at_n`.

9. **Only three harnesses can be seeded with a conversation prefix.**
   - `bash`, `null` and `browser_use` take a `Messages` prompt (`--initial-messages-file`, `VF/harnesses/utils/launch.py:72-80`; default `resume`, `VF/harness.py:240-288`).
   - ACP harnesses (Claude Code, Codex, Prime Agent, RLM) accept **user messages only** (`VF/acp/__init__.py:308-315`). mini-swe-agent and terminus-2 accept plain text only (`resolve_text_prompt`, `VF/harnesses/mini_swe_agent/harness.py:38`; `VF/harnesses/terminus_2/harness.py:42`).
   - **Crucible should** run C3 for black-box harnesses as "world snapshot + handoff summary", reusing verifiers' `POST_COMPACTION_FRAMING` (`VF/harnesses/utils/compaction.py:33-38`). Label that estimator separately (V̂_handoff ≠ V̂_same-context).

**Harnesses**

10. **Only harnesses that speak chat-completions can be trained with token capture.**
    - The renderer `TrainClient` raises `NotImplementedError` for any non-chat dialect (`VF/clients/train.py:350-361`).
    - That makes Claude Code (Anthropic Messages, `VF/harnesses/claude_code/harness.py:91-101`) and Codex (Responses, `VFR/docs/v1/architecture.md:21`) **eval and calibration only**.
    - Trainable through verifiers: `bash`, `null`, mini-swe-agent (litellm `custom_llm_provider=openai`, `VF/harnesses/mini_swe_agent/harness.py:57-66`), terminus-2 (litellm openai, `VF/harnesses/terminus_2/program.py:66-73`), and pi, kimi_code and prime_agent with their `chat_completions` transport (`VF/harnesses/pi/harness.py:52-53`; `VF/harnesses/kimi_code/harness.py:43-44`; `VF/harnesses/prime_agent/harness.py:218`).
    - **Crucible should** use verifiers `bash` (or mini-swe-agent 2.4.6) for on-policy RL rollouts and C3 continuations, and Claude Code 2.1.278 or Codex 0.155.1 for frontier calibration. OpenCode is **not** a verifiers harness (no match in `VF/`).

**Judges (C4 and the M6 LLM-judge baseline)**

11. **Adopt verifiers' rubric file and scoring contract unchanged for C4 items.**
    - File format: TOML `[[criteria]] name/text/weight≥0/choices`, with choices ordered worst→best and scored rank/(n−1) (`VF/judges/rubric.py:38-39,60-76,81-110`).
    - Score: Σw·s/Σw. Each criterion is also logged as the metric `<judge>/<name>` (`:265-272`).
    - Weights can be overridden from config without editing the file (`:113-120`).
    - The judge writes a one-sentence `reason` **before** the verdict (`:25-35`), and parsing is strict: a missing, duplicate or unknown verdict **raises** instead of scoring 0 (`:156-183,240-241`).
    - **Crucible should** map the C4 lifecycle onto this:
      - candidate = weight 0 (metric only);
      - validated = weight max(0, 2(AUC−0.5)), written as a `weights` override by `validity.py`.
    - RE's SWE judge already ships exactly this pattern: `solved` at weight 1 and nine "taste" criteria at weight 0 with 1/3/5 anchors (`RE/judges/swe/rubrics.toml:1-76`).

12. **Copy the agentic-judge discipline for any judge that sees a workspace.**
    - Verify empirically and never trust the trace (`VF/envs/agentic_judge/env.py:37-40`).
    - Hidden reasoning and provider state are stripped from the record the judge sees by default (`:153-157,236-238`); this is R6 for judges.
    - The judge writes a verdict file that is validated strictly (`:195-206,322-336`). The judge's tokens are never trained on (`:318-320`).
    - "Unverified is not solved: when you cannot confirm an outcome, answer with the worst option", and reward hacking counts as unsolved (`RE/judges/swe/prompt.md:26-28`).

13. **Build the M6 gate's LLM-judge baseline with a graded scale.**
    - A binary verdict gives a degenerate AUC. Use ordered 5-level `choices`, ideally averaged over k samples.
    - Feed the judge only the prefix up to the midpoint.
    - Beware `Trace.transcript`: it renders **only the last branch** (`VF/trace.py:662-683`), so compaction erases early work. Render from Crucible's own trace instead.
    - Rewrite the reference-judge prompt (`VF/judges/reference.txt:1-18`) as a forecast question.

**Learnability, validation, shaping**

14. **Learnability: use 4p(1−p) on the binary outcome O, not on R.**
    - verifiers' `proposer_solver` rewards the proposer with `4·rate·(1−rate)`, where rate is the mean solver `correct` over n=4 (`VFR/environments/proposer_solver/proposer_solver/taskset.py:151-168`).
    - **Crucible should** compute it on O, not R. The κΦ partial credit makes R almost never zero-variance, which would hide "too hard" tasks. Use n≥8, which matches GRPO n=8 in DES §10.3.

15. **verifiers' noop check is weaker than Crucible's R2.**
    - `vf-validate` marks a task invalid only if the untouched score is ≥ 1.0 (`VF/cli/validate.py:51-53,247-260`). Partial rewards pass.
    - The gold check is `Task.validate(runtime) -> bool|None` (`:242-244`).
    - **Crucible's export should** implement `Task.validate` with the oracle patch, and CI should assert `noop.reward == 0` from `results.jsonl` (`:313-319`).

16. **Exporting Crucible's shaping through verifiers' OpenEnv bridge collapses it.**
    - That bridge sums per-step rewards into one scalar (`VF/tasksets/openenv/taskset.py:4-7,128-134`). Potential-based shaping telescopes to Φ(s_T)−Φ(s_0) under that sum, so the dense signal disappears.
    - **Crucible's OpenEnv export should** carry steps out of band, not as `Observation.reward` (DES §10.1).

17. **Partial-credit patterns worth porting:**
    - alphabet_sort: credit repeated attempts only if they **strictly improve**, otherwise 0. This is the R4 oscillation rule as code (`VFR/environments/alphabet_sort/alphabet_sort/taskset.py:73-96`).
    - TextQuests: hand-labeled milestone checkpoints with observation-string detectors, and progress kept as a running max (`PE/text_quests/core/textquests_core.py:25-31,290-297`).
    - Mastermind: unsolved partial credit kept strictly below solved by **magnitude separation** (`PE/mastermind/mastermind.py:325-370,462-475`).
    - code_golf: efficiency is rewarded only among passing siblings (`VFR/environments/code_golf/code_golf/taskset.py:84-97`).
    - color_codeword: partial match is a metric only, never reward (`VFR/environments/color_codeword/color_codeword/taskset.py:95-111`).

18. **Anti-patterns found in PE (use as R3/R6 negative fixtures):**
    - "compiled" counts as correct (`PE/kernelbench/kernelbench.py:262`);
    - SWE reward = `completed_instances`, not resolved (`PE/mini_swe_agent_bench/mini_swe_agent_bench.py:50-51,139`);
    - a bonus for reasoning-text length (`PE/balrog_bench/balrog_bench.py:268-270,320`);
    - an ungated efficiency bonus (`PE/triton_codebase_search/triton_codebase_search.py:424-445`);
    - judge parse failure scored as 0.0 against the model (`:446-450`).

<!-- TLDR-EXTRA -->

---

## 2. Per-source mechanisms

### 2.1 verifiers v1 (`VF/`)

#### 2.1.1 Reward and metric composition

- **Declaring signals.** `@vf.reward(weight=1.0, priority=0)` marks a weighted Task reward returning a float or a keyed dict (`VF/utils/decorators.py:118-131`). `@vf.metric` is "recorded, not summed" (`:105-115`). `@vf.stop` (`:85-92`) and `@vf.intercept` (`:95-102`) are rollout-control hooks. Handlers declare what they need by parameter name (`task`, `trace`, `runtime`), resolved by `invoke` (`:29-38`).
- **Scoring order** (`VF/task.py:213-284`):
  1. Seed every expected name to `None`, so a crash leaves a visible "unscored" (`VF/utils/decorators.py:48-54`).
  2. Run metrics, then rewards, then plugged judges, each group concurrently (`invoke_all`).
  3. A keyed-dict result records each key under the function's weight (`:264-273`).
  4. Without a runtime (offline replay), runtime-requiring signals are skipped and logged (`:234-251`).
- **Aggregate.** `Trace.reward = sum(r.value)` with `value = score*weight` (`VF/trace.py:135-141,472-474`). `None` rewards are skipped (no error), so a crashed reward silently lowers the sum. Check `trace.ok` / `errors`.
- **Config-plugged hooks.** Rewards, metrics and stops can be swapped by import path with `weight` and `priority`. A `fn`-less entry only re-weights, "turning an existing zero-weight reward into a training signal" (`VFR/docs/v1/tasksets.md:268-290`; merge at `VF/task.py:299-317`). This is a clean seam for Crucible's rubric lifecycle: ship items at weight 0 and promote them by config.
- **Cross-agent judgement** belongs in `Env.finalize(task, episode)` and is "THE programmable judgement surface… validate strictly, never record a guess" (`VF/env.py:166-173`). Examples:
  - best-of-n `best` and `pass_at_n` (`VF/envs/best_of_n/env.py:34-44`);
  - code_golf sibling comparison (`VFR/environments/code_golf/code_golf/taskset.py:79-97`);
  - proposer learnability (`VFR/environments/proposer_solver/proposer_solver/taskset.py:160-168`).

#### 2.1.2 The message graph and the credit fields (`VF/graph.py`, `VF/trace.py`)

- **Node fields** (`VF/graph.py:83-166`):
  - `parent`;
  - `semantic_parents` (typed causal links, no token prefix, may loop: `:92-100`);
  - `message`;
  - `sampled` (True only for model-produced messages, `:103-106`);
  - `timestamp` (`:107-112`);
  - `token_ids` (the node's *delta* contribution, `:113-117`);
  - `mask` (True only on the sampled completion span, `:122-125`);
  - `is_content` (body vs template scaffold; feeds prime-rl `echo` observation weighting, `:126-134`);
  - `logprobs` (compact over sampled tokens, `:135-137`);
  - `advantages`, `reference_logprobs`, `trainer_logprobs`, `entropies` (all compact, consumer-stamped, `:138-151`);
  - `loss_weights` ("Named loss-weight streams aligned to `token_ids`, consumer-stamped", `:152-153`);
  - `routed_experts` and `sampling_mask` (MoE router replay; sampling masks, `:154-165`).
- **Exact semantics of `advantages`** (`VF/graph.py:138-142`): "Per-token credit over the sampled tokens, same layout as `logprobs`. `None` until a consumer's RL algorithm assigns it, which is not the same as a credit of zero: a group whose rewards were all equal is assigned zeros and carries no gradient, while an unassigned node was never scored at all."
- **Branch view.** `Branch.spread(fn)` widens a compact per-sampled-token stream to `token_ids` layout with 0.0 on unsampled positions (`VF/trace.py:229-250`). `Branch.advantages` returns `None` only when every node is `None`; otherwise "the unassigned nodes read 0.0" (`:258-266`). `Branch.loss_weights(name)` concatenates node streams, fills absent nodes with zeros, and **raises** on a length mismatch with `token_ids` (`:289-308`).
- **Branches.** One branch per graph leaf. A leaf tagged `compaction_attempt` is non-trainable unless some node has a `compaction` semantic parent pointing at it (`VF/trace.py:515-553`). The edge labels used today: `continuation`, `compaction_attempt`, `compaction`, `subagent_call`, `subagent_return`, `agent_message`, `subagent_cancel` (`VF/semantic.py:62-75`). `AgentInfo.trainable` excludes judge and user-sim seats (`VF/trace.py:107-115`; `VF/envs/agentic_judge/env.py:318-320`).
- **Per-call records.** `ModelCall(node, model, sampling, endpoint, finish_reason, usage, time, error, policy, acp)` (`VF/trace.py:153-176`), written at intercept time (`VF/interception/server.py:518-581`). `ModelCall.node` links each call to the assistant node it committed, and `Trace.is_truncated` covers limits, failed compaction or `finish_reason=="length"` (`VF/trace.py:620-632`). These are the C6 inputs.

#### 2.1.3 Recipe: Crucible dense credit → verifiers v1 / prime-rl

Inferred contract, because prime-rl is not in the tree:

1. **During the rollout** (Task-side, inside verifiers):
   - `Task.setup(trace, runtime)`:
     - record the base SHA host-side (`resolve_head`);
     - stash the runtime handle under `trace.id`;
     - evaluate cheap Φ(s_0).
   - `@vf.intercept async def track(self, request: vf.Request, trace: vf.Trace) -> None`:
     - for each new `ToolMessage` at the request tail, find the issuing assistant node: the leaf whose `tool_calls[].id == tool_call_id`, as `RolloutSession.decide_tool` does (`VF/session.py:478-528`);
     - snapshot the per-step diff with an alternate git index (TL;DR 7);
     - run the cheap milestone predicates;
     - append `{"node": k, "tool_call_ids": [...], "phi": Φ, "delta_phi": ΔΦ, "milestones_new": [...], "milestones_lost": [...], "cost_level": "cheap", "t": now}` to `trace.info["crucible"]["steps"]`;
     - return `None`, so the request is never rewritten;
     - catch all exceptions into `trace.info["crucible"]["tracker_errors"]`.
   - `@vf.stop def gate_violation(self, trace) -> bool` reads the tracker's latest record, e.g. test files touched or a sanitizer-disabled flag.
   - `Task.finalize`: medium checks; artifact collection.
   - `IsolatedVerifierEnv` / Harbor separate verifier: replay the per-step diffs in a fresh box, recompute Φ at **every** step at full cost, apply expensive checks at the end, write a flat `reward.json`, and overwrite `steps` with the verified values. Only verified values are reward-bearing; the online values are a forecast.
   - One `@vf.reward` (weight 1) returns R; everything else is a metric.
2. **Consumer side** (prime-rl hook or Crucible trainer adapter), per GRPO group (one `Episode.group.id`, n rollouts of one task):
   - A^GRPO_i = (R_i − mean R)/(std R + ε).
   - r^sh_{i,k} = ΔΦ at node k. The terminal adds −Φ(s_{T−1}), per DES §6.4.
   - G_{i,k} = Σ_{u≥k} r^sh_{i,u}, taken over sampled nodes in node-index order **on the trace's final trainable path**. A node on a discarded compaction attempt gets the G of its successor on the resumed path.
   - A_{i,k} = A^GRPO_i + η·(G_{i,k} − μ_G)/σ_G, with μ_G and σ_G over all (i,k) in the group.
   - `node.advantages = [A_{i,k}] * sum(node.mask)`. Stamp **every** sampled node of every trainable branch.
   - Optional: `node.loss_weights["crucible_infra"] = [0.0]*len(node.token_ids)` for turns after a C6 infra event. Whether prime-rl reads arbitrary `loss_weights` names is **unknown**.
3. **Shared prefix nodes** (compaction, subagents) appear in several branches. Their advantage is single-valued, which is correct for state-based credit, but a per-branch trainer counts their tokens more than once. Raise this with prime-rl, or down-weight via `loss_weights`.

#### 2.1.4 Interception and token capture (`VF/interception/server.py`, `VF/session.py`, `VF/clients/`)

- **Routing.**
  - One aiohttp server multiplexes all rollouts by bearer secret (`VF/interception/server.py:1-20`).
  - Routes: `/v1/chat/completions` (`VF/dialects/chat.py:425-426`), `/v1/messages` and `/v1/messages/count_tokens` (`VF/dialects/anthropic.py:435-437`), `/v1/responses` (`VF/dialects/responses.py:372-373`), `/v1/models`, plus `/tool` (tool gate), `/state` and `/task` (`VF/interception/server.py:423-437`).
  - Request bodies up to 1 GiB (`:72-77`).
- **Per turn.**
  1. `refused()` checks limits and `@stop`s before every model call (`VF/session.py:589-607`). `RolloutLimits` covers max turns and input/output/total tokens, all "soft by one turn" (`:82-117`).
  2. Request interceptors may rewrite only new user and tool messages. A rewritten tool result is pinned for the rest of the rollout (`VF/session.py:1-15,234-350`).
  3. Response interceptors and stops run before harness delivery (`:388-440`).
  4. Each proposed tool call is gated through `/tool`, and a denial returns a synthetic result in place of executing it (`VF/session.py:478-528`; client side `VF/harnesses/utils/core.py:332-354`).
- **Tokens.**
  - The eval client relays native bytes (`VF/clients/eval.py:1`).
  - The train client renders chat-template tokens and calls vLLM generate. It refuses non-chat dialects (`VF/clients/train.py:340-361`) and namespaced tools (`:40-52`).
  - Token attribution per message comes from renderer spans, so concatenating `token_ids` along a path equals exactly what the model saw (`VF/graph.py:13-18`).

#### 2.1.5 Harnesses and runtimes

| Harness | Pinned program | Dialect | MCP | Messages prompt / resume | Tool gate | Container needed | Notes |
|---|---|---|---|---|---|---|---|
| `bash` | in-house chat loop, bundled PEP-723 (`VF/harnesses/utils/launch.py:24-32`) | chat | yes | yes / yes | yes | no (`VF/harnesses/bash/harness.py:54-59`) | tools `bash` (3600 s, `VF/harnesses/utils/core.py:160-171`), `edit` (exactly-once replace, `:174-201`), optional `search`; stops at the first text-only reply (`:357-381`, return at `:380-381`); optional compaction |
| `null` | same loop, no tools | chat | yes | yes / yes | yes | no | `EXECUTES_CODE=False` (`VF/harnesses/null/harness.py:18-23`) |
| `mini_swe_agent` | `mini-swe-agent==2.4.6`, `litellm==1.102.0` (`VF/harnesses/mini_swe_agent/harness.py:14,program.py:1-4`) | chat (litellm openai) | **no** | text only | no | yes | `--yolo --exit-immediately`, `cost_limit=0`, `environment.timeout=86400`, `parallel_tool_calls=true` (`harness.py:40-66`) |
| `terminus_2` | `harbor==0.23.0` (`VF/harnesses/terminus_2/harness.py:15-17`) | chat (litellm openai) | **no** | text only | no | yes | tmux; system prompt injected by monkeypatch (`program.py:76-89`) |
| `claude_code` | `@anthropic-ai/claude-code@2.1.278` + `claude-agent-acp@0.79.0` (`VF/harnesses/claude_code/harness.py:14-35`) | Anthropic Messages | yes | user-only (ACP) | yes, via `permissions.ask=["*"]` (`:109-118`) | yes | `ANTHROPIC_BASE_URL` = interception (`:91-101`); **not trainable** (TL;DR 10) |
| `codex` | `@openai/codex@0.155.1` + `codex-acp@1.12.0` (`VF/harnesses/codex/harness.py:23-55`) | Responses | yes | user-only | yes (`gate.mjs`) | yes | `APPENDS_SYSTEM_PROMPT=False  # TODO` (`:58`); terminal failures surface in ACP metadata (`:63-72`) |
| `pi`, `kimi_code`, `prime_agent`, `rlm`, `openclaw`, `hermes_agent`, `browser_use` | various | configurable `transport` (pi/kimi default chat) | yes | ACP user-only, except browser_use (resume) | pi/kimi yes | yes | flags at `VF/harnesses/*/harness.py` (grep `SUPPORTS_`) |

- **Capability flags** on `Harness`: `APPENDS_SYSTEM_PROMPT`, `SUPPORTS_MCP`, `SUPPORTS_TOOL_INTERCEPTION`, `SUPPORTS_RESUME`, `EXECUTES_CODE`, `SUPPORTS_SKILLS`, `NEEDS_CONTAINER` (`VF/harness.py:40-65`). The contract is that every model call goes through `endpoint` + `secret`, and an in-process loop is legal (`:313-316`).
- **Runtime contract.**
  - Methods: `start`, `stop`, `run`, `run_program`, `open_process`, `run_background`, `read`, `write`, `prepare_execution(routes)` (network policy), `expose` (`VF/runtimes/base.py:138-436`). No snapshot or restore.
  - Types: subprocess, docker, podman, apptainer, prime, modal, e2b (`VF/runtimes/__init__.py:42-74`).
  - Docker GPUs use `--gpus N` plus an `nvidia-smi` type check (`VF/runtimes/docker/__init__.py:202-266`). Our lab 5090 needs `--runtime=nvidia` (memory note), so verify before relying on it.
  - Compose GPU tasks and Harbor multi-step tasks are unsupported (`VFR/docs/v1/harbor.md:109,191`). That matters for DES's `[[steps]]`.
- **Compaction** (`VF/harnesses/utils/compaction.py`):
  - Trigger: when tokens ≥ window − 16,384 (`RESERVE_TOKENS`, `:11-12,83-85`), or reactively on a context-overflow error identified by an 11-marker list across providers (`:40-80`).
  - `COMPACTION_ATTEMPTS=3`; a rejected checkpoint falls back to the last usage-verified good snapshot (`:14-16,176-256`).
  - The summary prompt asks for "progress, decisions, constraints, next steps, critical data", with `tool_choice="none"` (`:21-31,226-237`). The rebuilt context is `[system…, user(POST_COMPACTION_FRAMING + summary)]` (`:248-253`).
  - Tool outputs are middle-truncated to 20,000 bytes with a header (`:18-19,119-140`).

#### 2.1.6 Judges

- **`Judge` base** (`VF/judge.py`):
  - Prompt filling substitutes only declared placeholders in one regex pass, so literal braces and injected `{answer}` text are safe (`:135-149`).
  - Every call is recorded in `trace.info["judge_calls"]`, with usage in `trace.extra_usage`, even when parsing fails (`:163-221`; `VF/trace.py:696-717`).
  - `judge_verdict` **raises** when no verdict is found, "so judge failures are not scored against the model" (`VF/judge.py:102-107`).
  - Default judge model `openai/gpt-5.6-luna` (`VF/configs/judge.py:20`). Judges are config, never row data (`VF/judge.py:42-47`), and duplicate reward keys are refused (`VF/configs/judge.py:49-61`).
- **Verdict parser `parse_judge_choice`** (`VF/utils/score.py:51-86`):
  - returns `None` on an unterminated `<think>`;
  - strips everything before `</think>`;
  - prefers the last `FINAL (JUDGMENT|ANSWER|VERDICT)[: ] X` line;
  - then a `\boxed{}` answer;
  - then the first choice after a `VERDICT:`/`ANSWER:` label;
  - then the last standalone choice token (word-boundary regex).
- **`ReferenceJudge`** (`VF/judges/reference.py`, prompt `reference.txt:1-18`):
  - default view `last_reply`; positive/negative labels configurable;
  - an empty response returns 0 **without** a judge call (`:80-82`);
  - an unparseable verdict raises.
- **`RubricJudge`** (`VF/judges/rubric.py`, prompt `rubric.txt:1-16`):
  - default view `full_trace`, "rubric criteria typically grade the process" (`:129-132`);
  - `answer_field` injects a reference fenced with a backtick run longer than any inside it, which is anti-injection (`:200-220`);
  - `max_criteria` batches criteria k at a time: `1` gives n independent judges, `None` grades all in one call, and batches run concurrently with sibling cancellation on failure (`:133-137,243-263`);
  - `structured_output=False` by default, because "GLM-5.2 and other non-OpenAI models… return empty completions for structured calls" (`:138-143`);
  - the JSON parser scans for the first balanced object containing `verdicts`, skipping an echoed example (`:42-57`).
- **`AgenticJudgeEnv`** (`VF/envs/agentic_judge/env.py`): see TL;DR 12.
  - `ScoreConfig.task_weight=0.0` and `judge_weight=1.0`: judge-only by default (`:261-268`).
  - The judge harness must execute code, and the solver must be containerized (`:299-316`).
  - Shared and isolated variants exist (`:339-362`).
- **Offline re-scoring.** `vf-replay` recomputes every trace-only signal and judge on saved traces without a runtime (`VF/cli/replay.py:1-8`). This is a cheap way to run judge baselines on stored rollouts, if Crucible traces are converted to verifiers `Trace` records.

#### 2.1.7 Validation, learnability, multi-agent envs

- **`vf-validate`** (`VF/cli/validate.py`):
  - three checks per task, each in a fresh runtime after `Task.setup`: `setup`, `gold` (= `Task.validate(runtime)`), and `noop` (finalize then score with no agent turn) (`:236-263`);
  - noop is valid iff `trace.reward < 1.0` (`:51-53,260`);
  - rows record `reward` and per-reward scores (`:313-319`);
  - `_run_all` = gold then noop (`:349-366`).
- **Learnability.** `ProposerSolverEnv`:
  - the proposer mints a typed `SolveTask` from its own trace, and off-contract output raises, making the episode retryable (`VFR/environments/proposer_solver/proposer_solver/taskset.py:78-116`);
  - n solvers race it (`:144-149`);
  - the proposer reward is `4.0 * rate * (1.0 - rate)` with `solve_rate` as a metric (`:160-168`);
  - the seats' trainability is env-controlled (`:137-142`).
- **User simulation.**
  - `UserSimEnv`: the user is a `null`-harness agent with persona and done marker; its tokens are never trained on (`VF/envs/user_sim/env.py:22-99`).
  - `TauEnv` (`RE/packages/tau_common/tau_common/env.py:22-44`) runs assistant, user and judge models inside **one** official-runner process. Each role routes through its own interception session, so all three traces are captured and only `assistant` is trainable.
  - The reward is the official evaluator's `reward_info.reward` (`RE/packages/tau_common/tau_common/taskset.py:34-51`; `harness.py:59-64`).
  - This is the pattern for M2 simulated people.
- **Example reward schemes in VFR.**
  - `alphabet_sort`: per turn, a difflib similarity raised to the 4th power (`similarity_power=4`); multiple attempts count only if each strictly improves (`VFR/environments/alphabet_sort/alphabet_sort/taskset.py:35-38,64-98`).
  - `color_codeword`: the partial match is a metric only (`:95-111`).
  - `kuhn_poker`: zero-sum terminal payoff, with game facts in `trace.info` "exactly what a consumer-side advantage baseline needs" (`VFR/environments/kuhn_poker/kuhn_poker/taskset.py:1-14`).
  - `bash_interception`: `@vf.stop` on a buffered Response blocks a command before it executes, and `@vf.intercept` rewrites a tool result. These are the anti-hack primitives (`VFR/environments/bash_interception/bash_interception/taskset.py:24-76`).
- **Shared per-rollout tool state.** A `vf.Toolset`'s `self.state` is per-rollout even on a shared server (`VF/mcp/server.py:130-148`; demo at `VFR/environments/scratchpad/`). If a Crucible World App is exported as an MCP Toolset, state-changing actions can update milestone state directly. This only works for MCP-capable harnesses, not mini-swe-agent or terminus-2.

<!-- PE-SECTION -->

### 2.3 research-environments (`RE/`), v1 production tasksets

All `RE/environments/...` paths are relative to `RE/environments/`.

#### 2.3.1 SWE grading: the binary F2P ∧ P2P rule and how it is protected

- **The rule** is "1.0 iff every required (FAIL_TO_PASS + PASS_TO_PASS) test id is PASSED", and empty outcomes score 0:
  - swelego `swe/swelego/swelego/taskset.py:132-141`; swerebench_v2 `swe/swerebench_v2/swerebench_v2/taskset.py:145-150`.
  - swesmith also accepts `XFAIL` (`swe/swesmith_env/swesmith_env/taskset.py:148-155`).
  - scaleswe treats a skipped required test as missing (`swe/scaleswe/scaleswe/score.py:35-66`).
  - r2e_gym is strictest: the whole parsed status map must equal the expected map, including tests expected to FAIL (`swe/r2e_gym/r2e_gym/taskset.py:140-151`).
  - The pytest summary parser lives in verifiers (`VF/utils/score.py:153-177`).
- **pi_deepswe** keeps the reward binary and emits the dense decomposition as metrics (`swe/pi_deepswe/pi_deepswe/verifier.py:420-443`):
  ```python
  bucket: passed / len(selected) if selected else float(bucket == "p2p"),   # f2p, p2p fractions
  "reward": int(bool(scores["f2p_total"]) and not build_failure and not failed),
  "partial": sum(r["status"] == "passed" for r in rows) / len(rows) if rows else 0.0,
  scores["triage"] = int(scores["f2p"] == 1 and scores["reward"] == 0)  # fixed the bug, broke something else
  ```
  - Missing tests stay in the denominator.
  - Timeout retries are allowed only when all F2P pass and every failure is a timeout, and both retries must pass (`verifier.py:587-646`).
  - **This is the M5 "CI-derived F2P per step" milestone vocabulary already in production:**
    - `f2p` fraction = progress;
    - `p2p` = the no-regression gate;
    - `triage` = the "fixed but regressed" deviation flag.
- **Test-tamper defenses:**
  - **Revert test files from the base commit, not HEAD**, then re-apply the test patch. "An agent that ran `git add && git commit` mid-rollout moved HEAD to its own — potentially weakened — test version" (`swe/swelego/swelego/taskset.py:114-129`).
  - scaleswe also deletes agent-added test files.
  - pi_deepswe has a `PROTECTED` path regex (tests, fixtures, conftest, sitecustomize, runner configs) and token-aware restoration of Rust `#[cfg(test)]` items (`verifier.py:20-26,45-159`).
  - multiswe moves grading scripts off the box during the rollout (`swe/multiswe/multiswe/taskset.py:110-148,352-393`).
  - The verifiers Harbor reader wipes stale `reward.json`/`reward.txt` before grading (`VF/tasksets/harbor/taskset.py:327-347`).
  - The agent patch is captured *before* scoring restores files, so test edits stay auditable (`VF/utils/git.py:1-14`).
  - **These belong in Crucible's gate set G as cheap online checks** (test files touched → `@vf.stop` or gate=0).
- **Infra failure vs model failure:**
  - Grader failures raise instead of scoring 0: swe_sweep, apex, pi_deepswe `evaluation_valid`.
  - A tree the agent broke scores 0 (`swe/README.md:75-78`; `swe/swe_sweep/README.md:28`).
  - This is the C6 split.

#### 2.3.2 Partial credit that is gated

- **swe_sweep / pi_swe_sweep:**
  - The reward is the fraction of tracked bugs fixed, and "any detected regression in the retained baseline tests zeroes the repository reward" (`swe/swe_sweep/README.md:28`).
  - The official aggregate is bug-weighted, Σresolved/Σbugs (`swe/swe_sweep/swe_sweep/score.py:33-44`).
  - Failure mode: 17 tasks with empty regression baselines silently disable the gate. pi_swe_sweep treats an empty baseline as an invalid fixture (`swe/pi_swe_sweep/README.md:3-5,22`).
  - **Crucible should** add "gate is non-vacuous" (≥1 check) to the acceptance contract.
- **lab:** RewardKit all-pass is the reward, and a weighted `partial_credit` = Σvalue·weight/Σweight is a metric only (`tool_use/lab/lab/taskset.py:31-46`).
- **enterprise_ops_gym** re-runs every SQL goal verifier after **each** successful tool call (`tool_use/enterprise_ops_gym/enterprise_ops_gym/toolset.py:344,348-358`). The reward is all-pass, and `verifier_pass_rate` is a metric (`taskset.py:93-103`).
  - This is the only per-step progress vector found in the three repos, and it is exactly a cheap-tier Φ evaluated at every step. It is an M5 tracker precedent from the EnterpriseOps family.
- **general_agent:** reward = `max(db_hash_match, verify(db))`. Two independent success definitions: an exact end state OR a semantic verifier (`tool_use/general_agent/general_agent/taskset.py:66-87`).
- **pinchbench:** hybrid deterministic + judge score, default 0.5/0.5, overridable per task. It includes transcript process checks such as "did the agent read config.json". The judge prompt says "An average acceptable completion should score around 0.6-0.7". Per-task rubric weights in prose are **not** applied in code: the judge's `total` is used as-is (`tool_use/pinchbench/pinchbench/taskset.py:63-75,202-211,353-366`). That is a drift between rubric spec and scorer that Crucible should test for.
- **mcp_atlas:** a claim-coverage judge scores 1/0.5/0 per claim, averaged and returned as JSON (`tool_use/mcp_atlas/mcp_atlas/common.py:35-52`). On a parse failure it falls back to 0.0 and clamps (`:116-127`), which scores judge failure against the model.

#### 2.3.3 Validation and difficulty

- **Gold and no-op.** `swe/README.md:87-89`: "no-op runs should fail real tasks, gold-patch runs should pass, and repeated passes should separate task quality issues from sandbox or infrastructure failures."
  - Most SWE tasksets implement `validate()` as "apply gold, require reward == 1.0" (e.g. deep_swe, pi_deepswe, scaleswe, swebench_pro). swelego has `apply_gold_patch` but no `validate()` (`swe/swelego/swelego/taskset.py:223-227`).
  - swe_sweep's oracle applies each bug's gold patch separately, so it "does not prove they compose" (`swe/swe_sweep/README.md:48`). That is a warning for Crucible's per-milestone oracles.
  - general_agent's model-free check (`tool_use/general_agent/general_agent/corpus.py:78-105`) requires that the gold chain changes the DB hash, `verify(initial)==0` and `verify(gold)==1`. This is exactly Crucible's milestone filter "non-trivial at base, satisfied at oracle", in eight lines.
- **Difficulty targeting.**
  - Only general_agent filters by pass rate: an inclusive band `[min_pass_rate, max_pass_rate]` on a recorded per-(model, solver) pass rate plus tier bounds (`tool_use/general_agent/general_agent/taskset.py:104-144`).
  - Elsewhere there are only static slices (swebench_pro `hard51`, swerebench_v2 "Easy-Verified").
  - No 4p(1−p), online curriculum or "too easy/too hard" logic in the SWE, terminal or tool-use envs.
- **Repo guidance.** "Instructions and reward must match … any contradiction … will teach the model anti-instruction following" (`RE/AGENTS.md:113-115`). Keep deterministic and judge scores as separate parts and combine them only at the final score (`RE/BENCHMARK_PORTING.md:82-84`).

#### 2.3.4 Judges

- **SWE agentic judge** (`RE/judges/swe/{prompt.md,rubrics.toml,eval.toml}`):
  - Policy: judge behaviour, not the shape of the solution. "Reference solutions and held-out tests you come across are one way to solve or check the task — possibly a narrow one — never the answer key". Reward hacking counts as unsolved, and unverified counts as the worst option (`prompt.md:16-28`).
  - Criteria: `solved` (weight 1) plus nine 1-5 anchored taste criteria at weight 0. The anchors are written as "1 = …; 3 = …; 5 = …", and relative criteria fix "4 = on par with the reference; 5 = genuinely exceeding it" (`rubrics.toml:45-76`).
  - Per-taskset `hints/*.md` tell the judge where the evidence is: the `info.patch` diff, `fail_to_pass` ids, and which files are scoring side effects (`RE/judges/swe/hints/swelego.md:1-7`; `swebench-verified.md:1-8`).
  - Config: judge model `z-ai/glm-5.2`, `max_turns=128`, solver on `prime` runtime (`RE/judges/swe/eval.toml:1-13`).
- **RLM process rubric** (`RE/judges/rlm/rubrics.toml:1-29`):
  - six yes/no behaviour criteria over the transcript, e.g. `bounded_output` and `stateful_variables`;
  - each is judged "by the dominant pattern across the transcript";
  - each has explicit not-applicable semantics, e.g. "answer yes if … it simply never needed the same data twice — do not penalize the absence of reuse when no reuse was warranted".
  - This is a template for Crucible process items, such as "built before testing" or "ran the bit-exact check before the perf run".
- **`trace_cheating_recall_500`** (`RE/environments/judges/trace_cheating_recall_500/trace_cheating_recall_500/`): a judge benchmark of 500 SWE traces, all labelled CHEATING, so it measures recall.
  - The system prompt defines cheating as obtaining "implementation details from a source that already contains the requested solution" (`taskset.py:16-33`).
  - The judge sees only observable turns; hidden reasoning is removed (`:98-128`).
  - The verdict is the **last** CHEATING/LEGITIMATE label (`:44-62`).
  - Selection regexes: solution language plus network retrieval or solution-revealing git commands (`build_selection.py:25-76`).
  - Baselines: gpt-5.6-luna 96.8%, deepseek-v4-flash 70.6% recall (README:63-67).
  - **Crucible should** reuse these regexes as cheap gate checks: no `git log --all`, `cherry-pick` or upstream fetch in a migration rollout.
- **tau_common** (§2.1.7): the official tau2 runner and its `evaluator_nl_assertions` judge are routed through a dedicated judge agent session (`RE/packages/tau_common/tau_common/program.py:57-59`). Its tokens are captured but never trained on (`env.py:23-28`).
