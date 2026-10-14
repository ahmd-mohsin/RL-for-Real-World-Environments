# Long-horizon RLM harness, study 01: how every local framework manages an agent's context

- **Question.** How does each local framework and environment manage an agent's context on very long tasks
  (thousands of steps, up to millions of tokens)? The study looks at what stays in the prompt, what is offloaded and
  how it is retrieved, compaction, memory across steps and sub-agents, recursion budgets, training across context
  rewrites, failure modes and measured results. It ends with ranked requirements for Crucible's very-long-horizon
  RLM harness, where the model works inside a persistent Python kernel.
- **Read:** 2026-10-13, from the local checkouts, against Crucible HEAD 775f30f. Every claim cites `file:line`.
  Seven parallel readers covered the sources. Their notes are in
  `scratchpad/lh/frameworks/g{1..7}-*.md`. I read nano-rlm myself (`g0-nano-rlm.md`) and spot-checked the
  claims that carry the most weight (marked ✓ where I re-read the code).
- **One source is not in the workspace.** verifiers' RLM harness is now a thin wrapper around **nano-rlm**
  (github.com/PrimeIntellect-ai/nano-rlm, MIT). It is pinned to `c38e20b60c90bbf625274d3fde2b2a384f390a66`
  (`VF/verifiers/v1/harnesses/rlm/harness.py:70-75` ✓). I cloned that commit into the scratchpad because it is the
  engine prime-rl trains RLMs with. It was read there and is not vendored into the repo.

## Root abbreviations and licences

All paths under `RL-Environments/06-frameworks-and-env-code/` unless stated.

| Prefix | Source (commit) | Licence |
|---|---|---|
| `NR/` | nano-rlm `c38e20b` (scratchpad clone) | MIT |
| `R/` | primeintellect-ai-prime-agent `2ee7e623a` (`SE/`=`R/crates/pa-core/src/session_engine/`, `K/`=`R/crates/pa-core/src/kernel/`, `D/`=`R/crates/pa-daemon/src/`, `PY/`=`R/prime-agent-runtime/src/rlm/`) | MIT |
| `VF/` | primeintellect-ai-verifiers `ae819de` | MIT |
| `PE/`, `RE/` | prime-environments `cf220ee`, research-environments `d4fb80a` | Apache-2.0 |
| `PR/` | infra-primeintellect-ai-prime-rl `f7e8ab6` | Apache-2.0 |
| `XV/` (`MO/`=mimoagent-osr, `UA/`=uni_agent) | xiaomimimo-verl `a2ad9f6` | Apache-2.0; mimoagent-osr MIT |
| `MR/` | `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss` | dataset |
| `VL/`, `SL/`, `AR/` | verl `5eb4826`, slime `c933787`, AReaL `298412a` | Apache-2.0 |
| `SK/`, `RL/` | SkyRL `3d0ae22`, rLLM `4e828c8` | Apache-2.0 |
| `NG/` | nvidia-nemo-gym `3ef478d` | Apache-2.0 |
| `HB/` | harbor-framework-harbor `a2fafbc` | Apache-2.0 |
| `DS/` | `RL-Environments/08-hf-rl-env-datasets` | per card |
| `MS/` | swe-agent-mini-swe-agent `04d809c` | MIT |
| `OE/` | meta-pytorch-openenv `49aa302` | BSD-3 |
| `FE/`, `FM/`, `FS/` | fineenvs `5e0b46e`, the two hf-space-fineenvs spaces | Apache-2.0; spaces CC-BY-4.0 |
| `AE/` | `agentenv-framework` (repo root) | Apache-2.0 |
| `KA/` | karpathy-autoresearch `228791f` | MIT stated in README, no LICENSE file: pattern only |
| `GS/` | veds12-git-swarm `9f66af0` (website only) + `07-articles/gitswarm-2610.04862.txt` ("paper") | no licence: pattern only |
| PA, CH, RLM, RUST | `07-articles/prime-agent/{2608.23552,2605.09998,rlm,prime-agent-rust}.txt` | articles |
| ART | `07-articles/hf-space-fineenvs-multi-harness-rl.md` | article |

---

## 0. Findings in one page

1. **Two families.**
   - **The RLM family** (nano-rlm, Prime Agent, OpenEnv `repl_env`, SkyRL RLM, Harbor `dspy_rlm`, Crucible
     `RLMPolicy`) keeps data and history *outside* the prompt: in kernel variables, files and an append-only ledger.
     The model pulls slices with code.
   - **The transcript family** (terminus-2, mini-swe-agent, MiMo agents, verl/slime/AReaL loops, rLLM) keeps a
     linear chat. It either truncates observations, summarizes, or ends at the window.
   - Only the RLM family has run horizons of millions of tokens and days:
     - Prime Agent's 7-day Factorio run used 23.4M output tokens and 633 sub-agents (PA:623-631 ✓).
     - ARC-AGI-3 reached 95.5% at up to 3M output tokens per game (PA:364-376 ✓).
     - The Rust rewrite swarm used 228.7B tokens across 2,209 agents (RUST:28).
2. **Every RLM still compacts.** None relies on the variables alone:
   - nano-rlm compacts 16,384 tokens below the window (`NR/src/rlm/compaction.py:41,90-92`).
   - Prime Agent compacts at `min(0.95·W, W − max_out − 16,384)` (`SE/compaction.rs:8-17,152-163`).
   - Crucible compacts at 0.8 × `budget_tokens` (`crucible/src/crucible/actors/policy.py:81-101`).

   The variables survive a compaction because the kernel keeps running. Every implementation tells the summarizer to
   name the variables worth remembering (`NR/.../compaction.py:24-31`; `SE/compaction.rs:436`).
3. **The best design keeps history addressable.** nano-rlm logs every message, including the *full* tool output
   next to the shortened context version. It numbers context windows (`start`/`compaction`/`rollback`) and gives the
   kernel a `history()` API (`h.windows[w].messages[i]`, `h.user_messages()`). The post-compaction message points
   to that API (`NR/src/rlm/engine.py:1371-1388` ✓; `NR/src/rlm/session.py:91-156` ✓; `NR/src/rlm/prompt.py:313-326` ✓).
   - Prime Agent keeps the JSONL tree on disk but exposes **no** kernel API to read its own pre-compaction history.
   - Crucible's `HISTORY` archive holds only the **capped** outputs (`policy.py:623-644`), so what was cut is lost.
4. **Prime Agent's compaction has three gaps the earlier study missed:**
   - the summarizer **never sees tool results** (`SE/compact_session.rs:63-68` ✓);
   - threshold compaction runs **only at run boundaries**, so a model that keeps calling `ipython` grows its context
     until the provider overflows (`D/auto_compaction.rs:32-62` ✓; `R/crates/pa-agent/src/agent_loop/run.rs:45-189`);
   - only **one** overflow recovery is allowed per run (`D/overflow_compaction.rs:1-19` ✓).
5. **Training sees every context rewrite as a new row.** The mechanism differs:
   - verifiers/prime-rl message-graph branches;
   - MiMo prefix-hash chains;
   - slime message-tree forks;
   - AReaL concat leaves;
   - SkyRL/rLLM prefix-merge then split;
   - OpenEnv capture-graph new roots;
   - the NeMo-Gym boundary contract.

   **Every one copies the episode's scalar advantage onto all rows.** None gives a compaction boundary its own
   credit or bootstrap. NeMo-RL *refuses* non-monotonic traces outside that contract
   (`NG/fern/versions/latest/pages/reference/faq.mdx:451-465` ✓).
6. **A summary is trainable only when it is accepted.** verifiers trains a compaction attempt only if a
   `compaction` edge shows the run resumed from it (`VF/verifiers/v1/trace.py:515-553`). nano-rlm emits
   `compaction_attempt` per summary request and `compaction` on success (`NR/src/rlm/semantic.py:112-201`). prime-rl
   trains a node shared by several branches once (`PR/src/prime_rl/orchestrator/trajectories.py:105-128`).
   - **Trap:** prime-rl cuts any sample longer than `seq_len` at the tail, **silently**, and the kept prefix keeps
     the full advantage (`PR/src/prime_rl/orchestrator/batch.py:385-406` ✓). Production RLM configs therefore set the
     compaction threshold to 98,304 tokens under a 131,072 `seq_len` (`PR/examples/advanced/glm-4.5-air/search.toml:6,98,176`).
7. **Compaction multiplies rows and skews the data.** Under Claude Code, compaction made about 8 rows per rollout:
   77% of rows and 35% of supervised tokens, and 451M tokens trained against 162M (ART:1012-1014 ✓, 1196-1197,
   1274-1277). Weighting must be per logical rollout, not per row.
8. **Budgets must cover the whole tree, and few do.**
   - nano-rlm enforces tree-wide new-token, turn, spawn and concurrency caps. It reserves per-depth capacity so
     nested calls cannot deadlock (`NR/src/rlm/supervisor.py:146-158,439-456` ✓; `NR/src/rlm/config.py:51-93`).
   - Prime Agent has **no** per-child or global budget (only the factory's `max_parallel`).
   - SkyRL's RLM has **no depth cap at all** (`SK/examples/train/rlm/rlm_generator.py:362-373`).
   - OpenEnv's caps default to unbounded (`OE/envs/repl_env/recursive_backends.py:45-61` ✓).
9. **Measured failure modes recur across sources:**
   - **Stalls.** CH saw a 1,003-turn stall where the same payload was sent 842 times. Self-repair happens only in
     the first 50-200 turns (CH:1020-1106, 1777-1779).
   - **Tool-format junk loops.** On Qwen3.5-9B, 76.6% of tool calls named non-existent tools, and 89.1% of runs died
     on the 128K budget (`XV/recipes/general/agent_loop.py:286-291` ✓).
   - **Compaction that drops evidence.** terminus-2 unwinds the *newest* turns and discards the pending observation
     at handoff (`HB/src/harbor/agents/terminus_2/terminus_2.py:693-718` ✓, 1385-1395).
   - **Weak small models.** They ignore the REPL scaffold or regress with self-improvement (rlm.txt:540; CH:443-451).
10. **Crucible's `RLMPolicy` already has the core.**
    - Context as variables (`TASK`/`TOOLS`/`OBS`/`INBOX`).
    - The answer as a call (`rlm.finish`).
    - An 8,192-char output cap.
    - A verbatim task anchor, a summary and `[python-state]` at compaction, plus the kernel `HISTORY` archive.
    - Sub-call budgets and journal-replay restore.

    The gaps for very long horizons (§7):
    - estimated rather than usage-anchored accounting;
    - no reactive overflow compaction;
    - one summary attempt;
    - no `max_compactions`;
    - archived outputs are the truncated ones;
    - no depth-2 / persistent children;
    - `max_turns=500`;
    - journal replay that grows with every cell.

---

## 1. Crucible's baseline (what the requirements are measured against)

| Piece | State at 775f30f | Long-horizon consequence |
|---|---|---|
| Prompt | system + task (with a one-line tool index) + the model's own turns. Long data lives in `TASK`, `TOOLS`, `OBS`, `INBOX` (`actors/rlm/policy.py:56-72`) | Good: the RLM pattern |
| Output cap | `max_output_chars=8192`, head 3/4 + tail 1/4 with a size marker (`policy.py:413-420`) | Full cell stdout is not kept anywhere once capped |
| Compaction trigger | `approx_tokens(messages) ≥ trigger × budget_tokens` (0.8 × 128,000), checked before each root call (`policy.py:446-456,578-581`; `actors/policy.py:81-101`) | A chars/4 estimate, not server usage. No reactive path: `ContextOverflow` ends the episode as `context_overflow` (`policy.py:463-466`) |
| Summary | One call by the policy model, `max_tokens=2000`, prompt `COMPACT` with the kernel note (`policy.py:123-126,582-592`) | One attempt, no shrink-and-retry, no `max_compactions`, no failure stop |
| Rebuilt context | system + verbatim task + summary + `[python-state]` (live names) + the last `keep_recent_turns=3` turns (`policy.py:599-607`) | Correct anchor. The names list has no types or sizes and no cap |
| Archive | dropped turns go into kernel `HISTORY[i]` = {summary, turns:[{code, output}]} (`policy.py:623-644`) | Outputs are the *capped* tool messages. The archive lives in kernel memory, so it grows with the episode |
| Sub-LLMs | `rlm.query` / `rlm.batch`, depth 1. Caps: `max_sub_calls=256`, `max_sub_calls_per_cell=64`, `sub_concurrency=4`, `max_sub_tokens=2,000,000` (`policy.py:134-149,325-368`) | No persistent children, no depth 2, no tree budget beyond sub tokens |
| Stops | `max_turns=500`, continuations 2, stalls 3 (`policy.py:134-138`) | 500 turns is far below "thousands of steps" |
| Restore | journal replay of every cell with recorded host replies (`kernel.py`; `policy.py:646-690`) | Replay cost grows linearly with cells. At thousands of cells a restore re-runs them all |
| Training records | `model_calls` with `kind ∈ {turn, sub, compaction}`, `segment`, `parent`; a segment per compaction or engine re-render (`policy.py:473-477,588-611`) | The segment model exists. Edges (accepted vs attempted) are implicit |

---

## 2. Corrections to Crucible's earlier studies (verify, don't trust)

| Study claim | Verdict | Evidence |
|---|---|---|
| long-horizon/01 §2: verifiers' RLM harness exposes `summarize_at_tokens`… and compaction is `harnesses/utils/compaction.py` | **Stale.** The v1 RLM harness forwards a policy to nano-rlm over ACP. `compaction.py` is only used by the bundled chat program (bash harness) | `VF/verifiers/v1/harnesses/rlm/harness.py:32,56-123,161-253`; `VF/.../utils/launch.py:9,32` |
| §2: legacy `MultiTurnEnv`/`RLMEnv` | **Removed.** "The legacy v0 stack … has been removed" | `VF/docs/overview.md:7` |
| §2: "12 provider overflow markers" | **Wrong:** 9, matched only on HTTP 400/413 | `VF/.../compaction.py:40-61,75-80`; nano-rlm identical (`NR/src/rlm/compaction.py:53-87` ✓) |
| §2: `CompactionFailed` → stop `compaction_failed` | **Wrong for the bash program.** It is uncaught → non-zero exit → `HarnessError` (an errored rollout). Correct for nano-rlm, which ends cleanly | `VF/.../core.py:367-377`; `VF/.../harness.py:190-193`; `NR/src/rlm/engine.py:762-768` ✓ |
| §2: 3 compaction attempts | 3 in the chat program. nano-rlm defaults to **5** (its README still says "three", which is stale) | `VF/.../compaction.py:14`; `NR/src/rlm/config.py:87` ✓ |
| §2: trainability rule `trace.py:516-560` | Lines drifted to `:515-553` | `VF/verifiers/v1/trace.py:515-553` |
| §1 NeMo-Gym: `recency` bounds history | **Imprecise.** It drops only image groups and reasoning blocks (keep 3). Text is always kept | `NG/responses_api_agents/simple_agent_with_compaction/compaction/policies.py:193-195`; `config.py:20,40` |
| §1: `training_eligible=False` when identity is unreported | **Always** False, with three hard-coded reasons. The trainer owns admission | `.../compaction/session.py:237-242`; `osworld_agent/README.md:82-89` |
| §1: "no LLM summary policy ships" | True only for the compaction agent. terminus_2_agent, stirrup_agent and deepagents summarize with an LLM | `NG/responses_api_agents/terminus_2_agent/app.py:277-296`; `stirrup_agent/app.py:1045-1051` |
| §3 MiMo: MemAgent config at `agent.py:26-39` | That range is prompt templates. Config is at `:99-104` (`max_memorization_length 1024`, `max_chunks 8`, `max_steps 50`) | `UA/agents/mem_agent/agent.py:99-104` |
| §3: 40k truncation everywhere | Codex overrides it to 48,000. The 9-section prompt is CC only (Codex has a 4-bullet checkpoint prompt) | `MO/agents/codex/codex_agent.py:84`; `MO/compaction/prompt.py:122-130` |
| §4 Harbor: terminus-2 keeps `[system, question_prompt, questions]` | `messages[0]` is the initial **user** prompt. The answers arrive as the next user turn | `HB/.../terminus_2.py:1064-1078,1724-1745` |
| §4: unwinding drops the oldest pairs | Only computer-1 does. **terminus-2 drops the newest** (`chat.messages[:-2]`) | `HB/.../terminus_2.py:693-718` ✓; `HB/.../computer_1/compaction.py:236-253` |
| §4: the verifier still runs after a timeout (`trial.py:630-659`) | The error is raised at `trial.py:648-660`. Continuation is in `single_step.py:84-87` | as cited |
| §5 agentenv: 5 s staging drain | Only for remote agents reached through staging/v1. A series holds ≤24 interior captures + 1 final | `AE/.../staging.py:1-8,47`; `snapshot_series.py:52-54` |
| m14/01: anchor, messaging and child-result paths | Values right, paths wrong: anchor `SE/compact_session/recent_state_anchor.rs:8`; messaging limits `SE/agent_messaging.rs:16-17`; `answer_text` cap `rlm_child_model.rs:22`; revival `D/revival_gate.rs:12` | g1 notes |
| m14/01: `[python-state]` lists surviving names | Also lists variables **pruned for exceeding 16 MiB**. No cap on the list; 5 s listing timeout | `SE/ipython_state.rs:21,119-198`; `K/manager/mod.rs:792-796` |
| m14/01: output capped at 65,536 per stream | Head-first cut, no spool. The kernel caps the repr at 1 MiB first | `K/manager/requests.rs:14-40`; `PY/repl.py:53` |

---

## 3. Per-source findings

Each source is answered on seven points: (1) context policy; (2) memory and persistence across steps and
sub-agents; (3) truncation, offload and retrieval; (4) recursion and budgets; (5) training; (6) failure modes;
(7) measured results.

### 3.1 nano-rlm (the engine behind verifiers' `rlm` harness; what prime-rl trains)

1. **Context policy.**
   - **The prompt** is the system prompt plus the user task and the turns. One tool, `ipython`, is the default.
   - **Compaction trigger.**
     - Fires when `usage.total + est(tool output)` ≥ `summarize_at_tokens`. It is checked after every tool result,
       *inside* the run (`NR/src/rlm/engine.py:995-1000,1133-1146` ✓).
     - The threshold is discovered from `/models` (10 s timeout) as `max(W − 16,384, W//2)` (`compaction.py:90-136`).
       If it cannot be discovered, proactive compaction is off.
   - **Reactive path.** A 400/413 overflow, or a `length` stop past the threshold, discards the reply, compacts, and
     retries once on the rebuilt context (`engine.py:1195-1246` ✓).
   - **The summary.**
     - Written by the **same policy model**, with the tool schemas still forwarded and `tool_choice="none"` "to keep
       … prime-rl's trajectory extension property" (`engine.py:1283-1286`).
     - The prompt (`compaction.py:8-22` ✓) asks for: objective, exact requirements, completed work and evidence,
       orchestration state (child ids, jobs, subscriptions, unread events, output cursors, history references),
       interrupted operations, and "do not invent … state". `REPL_NOTE` adds: name useful variables, but not ones
       lost in a kernel restart (`:24-31`).
   - **Rebuilt context** = `[system, user(SUMMARY_FRAMING + summary + path to messages.jsonl + the history() API)]`
     (`engine.py:1366-1388` ✓). **The original task is not kept verbatim**; `h.user_messages()` recovers it. No recent
     tail is kept.
   - **If the summary fails:**
     - up to `max_compaction_attempts=5` (`config.py:87`);
     - on an overflow or `length`, `hollow_middle` shrinks the summary *input*: it keeps the beginning and the end
       and whole tool exchanges, and over-removes ×1.25 (`compaction.py:175-223` ✓);
     - empty or tool-calling replies are resampled; reasoning is never part of the summary;
     - after the last attempt, `stop_reason="compaction_failed"`, "end the run cleanly … still a trainable sample"
       (`engine.py:762-768`).
   - **Accounting.** The summary is "housekeeping, not a work turn": it is excluded from `max_total_turns` but
     counted in tokens (`engine.py:1276-1281`).
2. **Memory.**
   - The kernel keeps running across compaction. Kernel recovery is limited to 3 attempts; after a restart the
     variables are gone and a notice is carried into the next summary (`tools/ipython.py:254`; `engine.py:1294-1298`).
   - **Ledger:**
     - an append-only `messages.jsonl` per agent, in a session directory tree that mirrors the call tree;
     - `context_window` records with a `reason`;
     - closed windows never change;
     - full tool outputs and shortened context versions have separate indices linked by `source_message_index`
       (`session.py:91-156` ✓; README "Compaction").
   - **What a child inherits:** a fresh engine with its own kernel and session directory. It shares cwd, filesystem
     and MCP servers, and gets the configuration by `model_copy` (`supervisor.py:457-470,1203-1214` ✓).
   - **No `session/load`.** "An arbitrary live Python kernel cannot be reconstructed after the ACP process exits", and
     supervisor crash recovery is unsupported (README "Agent Client Protocol", "Compaction").
3. **Truncation.**
   - Tool results are cut to 20,000 bytes, head + tail, with the original token and line counts
     (`compaction.py:44,139-151`; applied at `engine.py:975`).
   - `rlm.shell.run` returns `.text` ≤16 KiB (8+8) but **retains 16 MiB per job**, read with
     `job.read(cursor, max_bytes)` (`prompt.py:158-164`).
   - Completion events carry the last 4 KiB (`supervisor.py:55`).
   - The prompt says: "Prefer `grep -n`, `sed -n`, and `head` over printing whole files; print the parts needed for
     your next decision" (`prompt.py:160-161`).
4. **Recursion.**
   - `await rlm.agent.spawn(task, name, persistent=False)` returns a handle; `result(yield_after)` defaults to
     300 s.
   - Further calls: `send`/`steer` (delivered at answer or tool boundaries), `history()` on a child, `cancel()`
     (cascades), and `watch.agent(child, every_turns=10 | every_tokens=50000)` progress events (`prompt.py:262-303` ✓).
   - **Policy defaults** (`config.py:51-93` ✓):
     - `max_depth=1`
     - `max_total_tokens=1,000,000` NEW tokens, tree-wide; verifiers raises this to 10M
     - `max_total_turns=None`
     - `exec_timeout=300` (hard cap 600, `tools/ipython.py:64`)
     - `max_concurrent_subagents=4`
     - `max_subagent_calls=None`
     - `max_compactions=None`
   - `depth_capacities` reserves semaphore capacity per depth "so nested calls cannot deadlock"
     (`supervisor.py:146-158` ✓). A spawn is refused at the depth, call, turn or token cap (`:439-456` ✓).
   - Payloads ≤65,536 bytes; an inbox holds ≤10,000 events (`supervisor.py:88-89`).
   - When a cap stops an agent, it salvages its last assistant text, then the last handoff summary
     (`engine.py:740-756` ✓).
5. **Training.**
   - Each model call carries `X-ACP-Model-Request-ID`. nano-rlm publishes edges: `continuation`,
     `compaction_attempt` (each summary request), `compaction` (accepted), `subagent_call`, `subagent_return` (when
     the parent retrieves the result), and `subagent_cancel` (`semantic.py:112-256`).
   - Metrics: `num_compactions`, `turns_between_compactions_mean`, chars dropped and summary chars, programmatic
     tool-call counts, sub-RLM calls (`types.py:137-173`).
6. **Failure modes handled in the prompt:**
   - "Never blindly repeat an interrupted cell" (`prompt.py:119-122`);
   - after compaction, "refresh resource metadata and inbox state before deciding whether to repeat work"
     (`compaction.py:29-30`);
   - "do not duplicate work merely because its full conversation is absent" (`:38`).
7. **Measured results.** None in the repo.

### 3.2 Prime Agent (Rust daemon + Python kernel)

1. **Context policy.**
   - **What the prompt holds:**
     - a cache-stable system prefix plus a dynamic tail (`R/crates/pa-core/src/prompts/system_prompt.rs:104-245`);
     - after compaction, one user row `[compaction-summary] … The retained messages below are authoritative; this
       summary may lag behind them` (`SE/messages.rs:12-17,353-367`);
     - the retained tail of about 20,000 estimated tokens. The cut is never at a tool result; a split turn gets a
       second, prefix summary (`SE/compaction.rs:9,297-432`; `SE/compact_session.rs:230-361`);
     - the `[python-state]` row (`SE/compaction_arms.rs:217-226`).
   - **No pruning** of old tool outputs, reasoning or images (`SE/engine.rs:644-649`).
   - **Accounting:** the last assistant usage, plus chars/4 for later messages. Usage from before a compaction never
     re-triggers it (`SE/compaction.rs:39-47,199-295`).
   - **The summarizer** is the session model, or an optional `auxiliaryModel` because "on the session model they can
     never hit the cached prefix" (`SE/compact_session.rs:230-272`).
     - Tool-less call, with a summarizer system prompt.
     - Max tokens: 0.8 × reserve = 13,107 for the summary; 8,192 for the prefix summary
       (`SE/compact_session/summarization.rs:9-16`).
     - Update mode: `<previous-summary>` plus a `<recent-state-anchor>` holding the newest assistant text, ≤2,000
       chars.
     - The summarizer **sees no tool results** (`compact_session.rs:63-68` ✓). It sees tool-call code and thinking.
   - **Triggers:**
     - at run boundaries only (`D/agent_engine/turn/run_loop.rs:40-49,158-162`; `D/auto_compaction.rs:32-62` ✓);
     - overflow: one compact-and-retry per run, after which the active goal fails (`D/overflow_compaction.rs:1-19` ✓;
       `run_loop.rs:131-133`);
     - model-requested: `compact.status()` / `compact.run(instructions)`, with the rule "Compact at a natural boundary
       when context usage is high and substantial work remains, instead of becoming terse or returning to the user
       early" (`R/skills/compact/SKILL.md`; `SE/turn_boundary.rs:236-291`).
2. **Memory.**
   - **The REPL namespace survives compaction**, except variables whose serialized form exceeds 16 MiB
     (`K/manager/mod.rs:792-796`). The prompt says so ("core.md:139").
   - The namespace is dill-snapshotted after each successful cell (1,500 ms debounce), 256 MiB total and 16 MiB per
     variable (`PY/repl.py:40-41`; `K/shared.rs:15-18`).
   - Goals are a durable `thread_goal_state` entry outside the LLM context. The objective returns at every
     continuation (`R/crates/pa-core/src/goals.rs:6,271-294,341-350`).
   - Harness memories are versioned in local or global stores, injected as a `[harness-digest]` user row with
     ≤3 entries per kind, 140 chars each (`R/crates/pa-core/src/refinement/mod.rs:17-19`).
   - **A child inherits** the configured cwd, model and thinking, and the daemon environment. It gets its own kernel
     and worker, and **no variables and no conversation** (`D/rlm_children/lifecycle.rs:78-150`). Siblings share the
     filesystem: "Have children write files and read those files for fan-in" (usage.md:32).
3. **Truncation.**
   - Per-cell output is cut head-first at 65,536 per stream, with no spool (`K/manager/requests.rs:14-40`).
   - `bash()` in the kernel keeps 512 KiB head + 1.5 MiB tail in its handle (`PY/bash.py:35-36`).
   - The native bash tool spills to a temp file and reports `fullOutputPath` (`R/crates/pa-core/src/tools/output_accumulator.rs:1-60`).
   - The prompt never states the cap. Its guidance is "Agents always assign read/search results to named variables
     so they can revisit them later" (usage.md:12).
4. **Recursion.**
   - `rlm.spawn(prompt, *, name, model, thinking, kind)`, depth limit 2 (`D/rlm_children.rs:34`).
   - Results come back three ways:
     - `agent_message` (≤16,384 chars, ≤20 pending);
     - automatic no-reply / failed notices;
     - `rlm.collect`, with `answer_text` ≤65,536.
   - **No per-child token, turn, time or cost budget, and no concurrency cap** outside the factory (`max_parallel`
     1-64, default 8).
   - Child usage is billed but excluded from the parent's context count (`SE/rlm_usage.rs:32-38`).
5. **Training.** No token ids or logprobs anywhere. CH trains on text tuples (CH:2061-2080).
6. **Failure modes.**
   - From the code: context grows unbounded within a run; a second overflow kills the goal; the summarizer is blind
     to tool output; no cell timeout.
   - **Mitigations in the code:**
     - audit-before-complete continuation;
     - no-progress cap of 3 (`goal_driver/progress.rs:11-21`);
     - automatic child notices.
   - **From the papers:**
     - a 1,003-turn stall with 842 identical payloads (CH:1020-1106);
     - self-repair only in the first 50-200 turns (CH:1777-1779);
     - create-and-forget: 99 skills, 5 used (CH:1798-1806);
     - an exploit preserved as a skill (PA:632-638).
7. **Measured results.**
   - ARC-AGI-3: Opus 5 at 95.5% vs 30.2% on the ARC harness, up to 3M output tokens per game (PA:364-376 ✓).
   - Factorio: 7 days, 23.4M output tokens, 633 depth-1 sub-agents in 149 waves, ≤7 concurrent (PA:623-631 ✓).
   - nanoGPT: an 85.5 h run (PA:126).
   - OOLONG at 128k: 0.70 → 0.94 (PA:396-411).
   - CH: GPP runs past turn 170,000 (CH:1095).
   - **No source reports compaction counts.**

### 3.3 verifiers v1 (outside the RLM harness)

1. **Context policy.**
   - The bash/null chat program appends forever (`VF/verifiers/v1/harnesses/utils/core.py:357-439`).
   - The opt-in compactor (bash harness only) keeps **system + one user(framing + summary)**, so the task is lost.
     It also uses the framing "Another language model started to solve this problem…" (`compaction.py:33-38,249-250`).
   - The proactive check runs only after tool turns, and inside `complete()` only on a `length` finish (`:199-206`).
2. **Memory.** The `environments/compact` demo resets context every turn: "your saved notes are your entire memory"
   (`compact/program.py:23-33`).
3. **Truncation.** 20 KB middle-out on every tool message, including rewritten ones (`compaction.py:18,128-140`;
   `session.py:311-314`).
4. **Budgets.** `RolloutLimits{max_turns, max_input_tokens, max_output_tokens, max_total_tokens}` count every
   sampled node, sub-agent and summary calls included, and are soft by one turn (`session.py:83-115,589-599`).
5. **Training.**
   - One sample per root→leaf branch of a delta-token message graph (`graph.py:1-19,83-160`).
   - `advantages=None` is distinct from 0.
   - Semantic edges resolve to the last committed retry (`trace.py:555-597`).
   - **Only ACP harnesses publish edges**, so in the bash harness every summary branch is trainable, accepted or not
     (`acp/__init__.py:78-86`; g2 probe).
6. **Failure modes.** `CompactionFailed` crashes a bash rollout; there is no `max_compactions`; turns after
   `last_good` never reach the summary.
7. **Measured results.** From the RLM blog:
   - Oolong real data: the RLM beats the plain LLM at ~1.5M chars (~300-400k tokens); "Beyond that, no model can
     succeed" (rlm.txt:492 ✓).
   - GLM 4.6 "almost a doubling" on DeepDive (rlm.txt:540).
   - Math gets worse under the RLM (rlm.txt:372,464).

### 3.4 research-environments and prime-environments (the long-context tasks)

1. **RE puts the long input in a sandbox file, never in the prompt.**
   - `/workspace/context.txt`, written in `Task.setup`, for example `RE/environments/long_context/oolong_synth/taskset.py:22-24,102-104`.
   - The file avoids spending tokens on the whole document and "sidesteps the E2BIG exec-argv limit"
     (`longbenchpro/taskset.py:5-7`).
   - The prompt names the path.
   - Answers go to a file, with the last reply as fallback (`oolong_synth/taskset.py:108`).
2. **Ladders:**
   - oolong_synth 1,024 → 4,194,304 tokens, default 262,144 (`taskset.py:28,125`);
   - mrcr_v2 "4k-8k" → "4m-8m", default "1m-2m" (`mrcr_v2/taskset.py:31-44`);
   - oolong_pairs up to 1M (gold file ~1.7 GB);
   - longbenchpro 8k-256k;
   - longpdfs: 10 documents per task, "too large to all fit in context, forcing programmatic tool calling,
     compaction, and/or sub-agents" (`longpdfs.py:1-9`).
3. **Scaffold-use rubric** (`RE/judges/rlm/rubrics.toml:1-29`): repl_native_shell, programmatic_tool_calling,
   bounded_output, stateful_variables, self_formatted_status, subagent_delegation. It is a weight-0 monitor in PR,
   and that monitor is commented out.
4. **PE** has no RLM code.
   - balrog keeps the last 16 text observations (`balrog_bench.py:338-340`).
   - browsecomp_plus forces a finish when one (tool, args) pair appears ≥6 times in the last 10 calls
     (`browsecomp_plus.py:490-503`).
   - webexplorer summarizes pages with a frozen, untrained model (`webexplorer/tools.py:103-114`).
   - tau_bench `max_turns=1000`.
5. **Measured results.** Only browsecomp_plus reports numbers (`README.md:20`). No RLM long-context scores.

### 3.5 prime-rl (training long RLM rollouts)

- **Samples.**
  - One `TrainingSample` per trainable branch. A node shared by several branches is trained only in the first
    (`PR/src/prime_rl/orchestrator/trajectories.py:105-191`).
  - Turns merge while the exact-prefix "extension property" holds. When it breaks ("environment compacts context, an
    agent hands off to a sub-agent") a new sample starts; the worst case is O(T²) (`docs/algorithms.md:489-531,563-565`).
- **Credit.** One GRPO scalar per trace, applied to every sampled token on every trainable branch: root, sub-agents
  and accepted summaries (`orchestrator/algo/grpo.py:26-54`; `algo/routing.py:11-40`). `hierarchical_grpo` separates
  only roles.
- **Truncation.**
  - `seq_len` default 2048, "longer samples are truncated" (`configs/orchestrator.py:579-580`). This is a **silent
    tail cut** that keeps the full advantage (`batch.py:385-406` ✓).
  - Turn- and token-limit stops are trained with their reward; `is_truncated` is only a metric.
  - Timeouts become errors and are dropped (`dispatcher.py:668-674`).
- **Off-policy.** `max_off_policy_steps=8` by default; "a rollout can span several weight updates"
  (`orchestrator.py:591-592`).
- **Long context.** Context parallelism `cp` ring or ulysses; the fused LM head is chunked at 8,192
  (`configs/trainer.py:312-316,344`).
- **RLM configs.**
  - glm-4.5-air search: `seq_len=131072`, `cp=4`, group 16, `max_off_policy_steps=32`, length penalty 0.1 on
    input tokens, `rlm` with `compaction.summarize_at_tokens=98304` (`search.toml:6,45-46,66,69,82-86,95-98`).
  - terminal: the same compaction setting plus a 0.1 turn penalty.
  - intellect-3.1 deepdive with `rlm`.

### 3.6 MiMo (mimoagent-osr, uni_agent gateway, xiaomimimo-verl)

1. **Context policy.**
   - **CC agent, model-decided compaction.**
     - A `Compact` tool sets a flag; the compaction runs at the next query (`MO/tools/cc/compact.py:70-71`;
       `MO/agents/cc/cc_agent.py:146-155`).
     - At 80% of (window − buffer), every tool result carries a `<context_usage>` footer so the policy "can learn to
       act on" it (`cc_agent.py:72,227-244`).
     - Windows: 100k−10k for CC; 1M−10k for mimocode.
     - Tokens estimated at ~4 chars each.
     - Rebuild = system + boundary marker + verbatim `<original_user_issue>` + summary ("if details are missing,
       state the gap") (`surgery.py:47-132`).
   - **Codex agent, automatic compaction.**
     - Trigger at 900k (`swe_codex_agent.yaml:19-21`).
     - Keeps real user messages ≤20k tokens plus the summary.
     - Wipes the on-disk transcript and keeps no anchor (`codex_agent.py:354-380`).
     - The authors removed a "time never runs out" promise because it "changes rollout length vs truncation" (`:60-62`).
2. **Memory.**
   - Codex PTC mode: the model sees only `exec`/`wait`. `store(key,value)` / `load(key)` persist across cells, and a
     long cell yields and is resumed with `wait` (`MO/tools/codex/code_mode.py:1-7,41,63,73`). It is the closest
     thing to an RLM in MiMo.
   - MemAgent carries ≤1,024 memory tokens across fresh contexts (`UA/agents/mem_agent/agent.py:99-104`).
3. **Truncation.** Middle cut at 40k chars, 48k for Codex, with a banner and no spool (`MO/agents/default.py:69`;
   `truncate.py:44-60`).
4. **Recursion.** CC sub-agents, depth 1: "Subagents cannot spawn further subagents". Each has a fresh context,
   `step_limit` 500, and its own gateway chain (`MO/tools/cc/agent.py:55,78`; `subagents.py:145`).
5. **Training.**
   - The gateway prefix-hashes requests into chains. Compaction or a tool-schema change starts a new chain; a
     rewritten last assistant message rolls the chain back (`UA/gateway/session/session.py:477-526,610-703`).
   - GRPO statistics come from each session's final chain and are broadcast to all chains
     (`XV/verl/trainer/ppo/v1/utils.py:203-285`).
   - Loss is `prompt-mean` (`core_algos.py:1160-1221`).
   - **`trajectory_selection: longest`** (`recipes/code/config/train.yaml:81`) usually keeps the *pre*-compaction chain.
   - Truncated runs are graded, never masked (`mimoagent_runner.py:283-291`).
   - `recipes/general` rejects `Compact` and `Agent` because its append-only buffer "cannot represent" them
     (`XV/recipes/general/tools/__init__.py:21-24`).
6. **Measured failure.** The junk tool-call loop: 76.6% non-existent tools, 89.1% of runs died on the 128K budget
   (`XV/recipes/general/agent_loop.py:286-291` ✓).
7. **Note.** The shipped code recipe enables neither compaction mode; it relies on a 256k window
   (`train.yaml:21-22`; `mini-codex.yaml:6`). `MR/` (the RL dataset release) has no turn-count or context-length fields.

### 3.7 verl, slime, AReaL

- **verl.**
  - No compaction. Prompts are left-truncated (`VL/verl/experimental/agent_loop/agent_loop.py:454-464`).
  - Tool output is cut to **256 chars** by default (`rollout.yaml:214-220` ✓).
  - Extra parallel tool calls are silently dropped (`max_parallel_calls=1`).
  - `[:response_length]` truncates silently (`tool_agent_loop.py:183-206`).
  - **Partial rollout inside `generate()`:** `FullyAsyncLLMServerClient` resumes an aborted generation with a
    cumulative token budget and records the policy-version range (`VL/verl/workers/rollout/llm_server.py:172-359`).
  - Measured: 2.35-2.67× faster; multi-turn ReTool 1.55-1.60× at equal accuracy (`fully_async_policy/README.md:9,564-565`).
- **slime.**
  - Black-box harnesses. `TrajectoryManager` decides per turn whether to append, realign a small token drift, or fork
    a new row (fork threshold 1,024) (`SL/slime/agent/trajectory.py:140-188,271`).
  - A shared turn is trained once (`:456-477`).
  - **Every row gets the full reward** (`:327-340` ✓); the README's `reward/K` is wrong.
  - `max_sample_tokens` cuts silently.
  - `--partial-rollout` is asserted off in every multi-turn example.
- **AReaL.**
  - `InteractionCache` links calls by longest prefix. `concat` gives one row per branch; `individual` (the default)
    gives one row per call with a turn discount, 0.9 in terminal-bench (`AR/areal/experimental/openai/cache.py:35-40,369-419,564-669`).
  - Interruptible generation records a policy version per token (`AR/areal/infra/remote_inf_engine.py:1105-1198`).
  - The Tongyi deep-research example forces an answer at 80% of 32,768 tokens and summarizes pages with an untrained
    judge model (`react_agent.py:69-80,191-216`).

### 3.8 SkyRL (RLM env, step-wise training, MemAgent)

1. **The RLM env puts the root context in the REPL.**
   - `context` / `context_0` hold the data. The prompt has only a metadata line: "Your context is a {type} with {N}
     total characters, and is broken up into chunks of char lengths: [...]" (`SK/skyrl-gym/skyrl_gym/envs/rlm/env.py:137-159,276-297`).
   - The per-turn instruction is **popped** from history every step (`env.py:201-220,345`), which makes every step a
     prefix break.
   - No compaction. REPL output is cut *head-only* at 20,000 chars (`env.py:369-373` ✓).
2. **Memory.**
   - In-process `exec` over a safe-builtins namespace. Scaffold names are restored after each exec
     (`repl.py:17-120,353-432`).
   - SIGALRM works only on the main thread; a timed-out thread is abandoned (`repl.py:376-385,434-460`).
4. **Recursion.**
   - `llm_query`/`llm_query_batched` send a single user message to the policy or to a frozen model and are **never
     trained** (`rlm_generator.py:229-276`).
   - `rlm_query`/`rlm_query_batched` spawn a full child agent loop with its own `context=`. The child returns
     `final_answer`; timeout 600 s; `max_workers=min(2, n)` (`rlm_generator.py:294-334`; `repl.py:280-344`).
   - **No depth cap and no child budget** (`rlm_generator.py:362-373`).
5. **Training.**
   - `step_wise_trajectories=True`: one sample per turn.
   - With `train_child_trajectories`, all descendants' steps are prepended to the root's steps in DFS preorder and
     share the root's GRPO advantage (`rlm_generator.py:186-198` ✓; `SK/skyrl/train/trainer.py:1212-1250`).
   - A judge failure scores 0.0 instead of being masked (`evidence_rewards.py:345-361`).
   - Overflow is a `length` stop with reward 0.
6. **Failure modes.**
   - The prompt says the engine runs the first code block; the code runs the last (`evidence_rlm_env.py:43` vs
     `env.py:86-89`).
   - `generator.max_turns=6` in the script against "HARD LIMIT of 10 rounds" in the prompt.
- **skyrl-agent `next_with_summary` (MemAgent style).** The policy writes the memory as a tool argument; the history
  resets to system + `<problem><memory><section>`. The summary tokens are trained; context-exceeded and
  max-iteration runs are masked (`SK/skyrl-agent/skyrl_agent/tools/next_memagent.py:150-199`; `agents/base.py:321-325,406-418`).

### 3.9 rLLM

- No RLM and no compaction in the core.
- `CumulativeWorkflow` shrinks `max_tokens` as the prompt grows (`RL/rllm/workflows/cumulative_workflow.py:43-56`).
- **Archived deepresearch.**
  - `MAX_LLM_CALL_PER_RUN=100`; at 90% of the window it keeps system + question + the last 4 messages and forces an
    answer (`examples/archive/deepresearch/deepresearch_agent.py:21,162,577-607`).
  - **Bug:** "used tokens" sums the prompt length over all calls, so the limit fires early (`:296-299,322`).
- **Training.**
  - Prefix-merge into datums, with two health metrics: `steps_per_traj` (">1 indicates a prefix break") and
    `merge_compression_ratio` (`RL/rllm/trainer/tinker/transform.py:42-182`).
  - GRPO groups keyed `task_id:role`, so roles get separate baselines (`RL/rllm/trainer/algorithms/transform.py:115-132`).
  - `CompactFilteringConfig` can mask any termination reason (`config.py:110-128`).

### 3.10 NeMo-Gym (and how NeMo-RL consumes traces)

- **NeMo-RL requires monotonic token sequences.** Summarization, sliding windows and observation pruning are listed
  as violations (`NG/fern/versions/latest/pages/reference/faq.mdx:451-465` ✓). The only exception is the compaction
  agent's contract (semantic history, a boundary event per append-compatibility break, `required_prefix_token_ids`
  inside a segment, one `/verify` per logical rollout).
- **finance_agent** truncates on overflow "for eval only — during training the full trajectory must be preserved"
  (`NG/responses_api_agents/finance_agent/app.py:159-164`).
- **terminus_2_agent** enables summarization with an 8,000-token proactive threshold, 262,144 max input
  (`terminus_2_agent/app.py:277-296`).
- **stirrup_agent**: window 262,144, `agent_max_turns 250`. Measured: truncation-recovery turns "burned 15.8% of all
  model time … median 34.6k tokens" on GDPVal (`stirrup_agent/app.py:907,997-1051`).
- **deepagents** offloads evicted history to `/conversation_history/{rollout_id}.md`, and the agent reads it back with
  `read_file` (`langchain_deepagents_agent/README.md:170-174`).
- **prime_agent** runs as a black box: usage only, no token evidence (`prime_agent/app.py:123-153,262-263`).

### 3.11 Harbor (terminus-2, computer-1, dspy_rlm) and the long-horizon datasets

1. **terminus-2.**
   - A linear chat until free tokens < 8,000, counted with litellm against the model limit (`terminus_2.py:120-123,650-653`).
   - **Summary, three steps:**
     1. A summary sub-agent sees the history ("Major Actions Completed / Important Information Learned / Challenging
        Problems Addressed / Current Status").
     2. A questions sub-agent **with no history** asks ≥5 questions.
     3. An answers sub-agent with history answers them (`:865-1079`).
   - **On overflow** it unwinds the **newest** turns to 4,000 free tokens (`:693-718` ✓). The fallbacks: a short
     summary, then instruction + screen, then "Technical difficulties. Please continue with the task." (`:1138-1207`).
   - **The pending observation is discarded at handoff** (`:1385-1395`).
   - Observations are cut to 10,000 bytes, half head and half tail (`:655-691`).
   - `max_turns` 1,000,000.
   - "Rollout details will be incomplete if context summarization occurs" (`:307-308`).
2. **computer-1.**
   - Charges 1,300 tokens per image; keeps the last 3 screenshots; proactive at 8k and unwind to 4k.
   - Skips the full summary above 0.9 × the limit.
   - The fallback chain ends in "Task + last 1,000 chars".
   - Fabricates an "Understood…" assistant turn (`computer_1/compaction.py:34-38,197-269,319-329`).
3. **dspy_rlm.** `dspy.RLM` host-side, tools bridged into the sandbox. `max_iterations=20`, `max_llm_calls=50`,
   `max_output_chars=10_000`, tool timeout 30 s, root 16k / sub-LM 8k max tokens. The initial context is a
   `find … | head -200` file tree (`HB/src/harbor/agents/dspy_rlm.py:62-215,343-355`).
4. **ATIF.**
   - `is_copied_context`: "should not be included in training data".
   - `continued_trajectory_ref` and `subagent_trajectory_ref`.
   - A `context_management{type:compaction, boundary:replace}` step (`HB/src/harbor/models/trajectories/step.py:64-73`;
     `trajectory.py:79-82`; `terminus_2.py:1418-1438`).
5. **Datasets.** Agent timeouts:

   | Dataset | Agent timeout | Notes |
   |---|---|---|
   | TB2.1 | 900 s | |
   | TB-science | 28,800 s | |
   | LHTB | 3,600-14,400 s | `continue_until_timeout` on 30/46 tasks: on a premature stop, resume the same session with verifier feedback (`DS/IntelligenceLab__Long-Horizon-Terminal-Bench/README.md:40-56`) |
   | NatureBench | 43,200 s | best-so-far `/evaluate` sidecar and `/time_remaining` |

   No instruction anywhere gives context-window guidance.

### 3.12 mini-swe-agent

- **History.** "A completely linear history … no difference between the trajectory and the messages that you pass on
  to the LM" (`MS/docs/index.md:49-51`). There is no summarizer. `ContextWindowExceededError` aborts.
- **Output template.** At ≥10,000 chars: `<warning>…you can redirect output to a file and then search in that
  file</warning>`, then `<output_head>` 5,000 + `<elided_chars>` + `<output_tail>` 5,000 (`MS/src/minisweagent/config/default.yaml:114-140`).
- **ProgramBench config.**
  - 1,000 steps.
  - Within 20 steps of the limit: "write them to a document "AGENT_REPORT.md". Focus on handing off to the next agent".
  - A time warning under 600 s (`benchmarks/programbench.yaml:181-186,238-255`).
- **Format errors.** 3 consecutive format errors exit.

### 3.13 OpenEnv `repl_env`

- **Context.** The context is a variable. The model sees a 500-char preview and the length.
- **Caps.** stdout/stderr at 8,192 chars with the total stated; variable reprs at 500 chars. Up to 30 iterations, then
  a forced-answer call (`OE/envs/repl_env/server/repl_environment.py:92-108,465-476`; `python_executor.py:214-240`;
  `runner.py:229-261`).
- **Recursion.** `BackendLimits` defaults: depth 1, 8 workers, everything else `None`. The timeout is "cooperative:
  checked between iterations, not during LLM calls" (`recursive_backends.py:45-61` ✓). The controller never sets the
  caps.
- **System prompt.** "You will only be able to see truncated outputs … figure out a chunking strategy, query sub-LLMs
  over useful batches, save answers to buffers, and aggregate them" (`prompts.py:36-67`).
- **Runner.** The root history is linear and never compacted (`runner.py:201-214`).
- **Capture graph.** "A rewritten history is not a prefix extension, so it opens a new root"; a turn without logprobs
  is context only (`OE/src/openenv/core/harness/capture/graph.py:1-35,443-484`).
- **Pull-on-demand memory.** `jupyter_env`'s `get_notebook_state` returns the last 10 cells (code 120 / output 300
  chars) (`envs/jupyter_env/server/notebook_tracker.py:69-94`).

### 3.14 FineEnvs multi-harness RL

- **Harnesses as context policies.** mini-swe-agent is a white-box loop with no compression; Claude Code, Codex,
  OpenClaw and OpenHands "compress and reorganise their context" (ART:157-162).
- **Compaction breaks token renderers** (ART:227-232).
- **Measured.**
  - Held-out 42% → 54% with 31% fewer tool calls (`FE/05-multi-harness-rl/RESULTS.md:66-73`).
  - The same base model scores 62% under mini-swe-agent and 33% under Claude Code (ART:872).
  - Swapping the harness moved GLM-5.1 by 13 points (ART:255-270).
  - Claude Code compaction gave ~8 rows per rollout, 77% of rows (ART:1012-1014 ✓).
  - 35-58% of optimizer steps had no reward contrast.

### 3.15 agentenv (AE)

- **Context ownership.** AE never compacts. It hands context to the agent runtime: each A2A `context_id` maps to an
  opaque `session_ref` in an LRU of 1,024 (`AE/packages/agentenv-protocol/src/agentenv_protocol/a2a_agent/framework.py:91,134-157` ✓),
  and turns on one context are serialized.
- **Limits.** `max_turns` is passed through, not enforced; 600 s per send; a 2 h Modal sandbox default.
- **For resume.** The snapshot/v1 changelog keyed by six-digit tool-call position (`000042.tar`) supports "rewind to
  tool call N and resume" (`object_transfer.py:533-582`; `deploy_agent.py:513-560`).
- **For judges.** `trajectory_filter` offloads results over 1 KB to side files, for judges only (`trajectory_filter.py:21-22,461-468`).
- **Delegation.** `peer-agents/v1` routes delegation without depth or budget limits (`extensions.py:1026-1042`).

### 3.16 karpathy-autoresearch

- **Observation offload by redirection.** "`uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or
  let output flood your context)", then `grep "^val_bpb:\|^peak_vram_mb:" run.log`, and `tail -n 50` on a crash
  (`KA/program.md:99-101` ✓).
- **Memory = git + an untracked `results.tsv` ledger.** keep/discard/crash; advance the branch or `git reset`.
- **Never stop, with a fixed budget.** "NEVER STOP … The loop runs until the human interrupts you, period.", with a
  fixed 5-min budget and a kill at 10 min (`:106-112` ✓).
- **Measured.** 83 experiments, 15 kept. One kept step was a seed change (42→137), so greedy keep-if-better accepts
  noise.

### 3.17 GitSwarm (paper) and "On the nature of the swarm"

- **Short episodes on a shared graph.**
  - Many short, fresh episodes rebuild context from a shared git graph.
  - Each contribution is an immutable commit with `informed_by=` provenance plus mandatory `MY_THOUGHTS.md` and
    `EVALUATION.md` handoffs (paper:159-196, 3480-3500).
  - Workers announce scope (Gap | Deliverable | Success | Abort); the dashboard is pulled on demand.
- **Workers read selectively:** about 5-7 commits for code, about 3 for proofs. Input still grows with the
  repository (paper:344-345, 721-724).
- **Measured.**
  - ProgramBench 79.4% vs 65.1% for a single agent forced to keep going (paper:339-342).
  - 94.7% of code contributions are built upon.
  - Shared persistent notes kept coverage high, while independent workers reached 20-75% (paper:1440-1444).
- **Failure modes.** Seed noise as large as real gains; search collapse onto one idea (~90% of commits); predicted
  command outputs.
- **The pi essay.** "Every compaction is a bet" (pi:23). A sub-agent's summary is written "after the question is
  known" (pi:41). Persistent sub-agents are "parallel over context" (pi:61).

---

## 4. Comparison table

| Framework | Kept in the prompt | Offload and retrieval | Compaction trigger | Summary writer / trained? | Sub-agents and budgets | Long-rollout training |
|---|---|---|---|---|---|---|
| **nano-rlm** (VF `rlm`) | system + `[framing+summary+history pointer]`; no tail, no verbatim task | kernel vars; `messages.jsonl` with full outputs; `history()` windows; shell jobs retain 16 MiB | `W−16,384` (floor `W/2`) after every tool result; reactive 400/413 and `length`; up to 5 attempts with hollow-middle | policy, tool_choice none / yes if accepted (`compaction` edge) | spawn/result/send/steer/inbox; depth 1; tree tokens 1M (10M in VF), concurrency 4 with per-depth reserve; exec 300 s | VF graph branches + PR one scalar per trace |
| **Prime Agent** | cached system + summary + ~20k tail + `[python-state]` + digest | kernel vars (≤16 MiB each survive); JSONL on disk, no self-read API; native bash spill file | `min(0.95W, W−max_out−16,384)` at run boundaries only; 1 overflow retry per run; `compact.run` tool | session or auxiliary model, tool results hidden / not trained (no logprobs) | spawn depth 2, agent_message 16,384, collect 65,536; no budgets | none |
| **verifiers chat program** | system + framing+summary (task lost) | none | `W−16,384` (floor `W/2`), after tool turns, `length` only in `complete` | policy / every branch trains (no edges in bash) | — | branches |
| **RE tasksets** | question + file path | `/workspace/context.txt`, answer file | via harness (98,304 in PR configs) | via harness | via harness | PR |
| **prime-rl** | — | — | — | — | — | branch samples, shared node once, one scalar, silent `seq_len` tail cut, cp ring/ulysses |
| **MiMo CC / Codex** | system + boundary + original issue + summary / user msgs ≤20k + summary | none (40k/48k middle cut); Codex PTC `store/load` | model `Compact` tool + 80% footer / 900k automatic | policy / in pre-compaction chain | CC sub-agents depth 1, 500 steps | prefix-hash chains, final-chain GRPO broadcast, `longest` selection (bad) |
| **verl** | full chat, left-truncated | none; tool output 256 chars | none | — | `max_parallel_calls=1` | silent `[:response_length]`; partial rollout in `generate()` |
| **slime** | black-box harness | — | harness-native | harness / per fork | — | message-tree forks (1,024 threshold), full reward per row |
| **AReaL** | full chat; Tongyi forces answer at 80% | untrained page summarizer | — | — | — | concat leaves or per-call rows with turn discount; per-token policy version |
| **SkyRL RLM** | system + metadata line; per-turn instruction popped | `context` var; output head-cut 20k | none | — | `rlm_query` children, no depth cap; `llm_query` untrained | step-wise; children prepended under root advantage |
| **SkyRL MemAgent** | system + problem + memory + chunk | memory as tool arg | each chunk (4,000 tokens) | policy / yes | — | prefix-merge, new datum on break |
| **rLLM** | full chat; deepresearch keeps the last 4 messages at 90% | — | — | — | role-keyed groups | prefix-merge, `steps_per_traj` metric |
| **NeMo-Gym** | compaction agent: recency (images/reasoning only); deepagents summary | MediaArena; `/conversation_history/{id}.md` | guards on server tokens; turn-chunked | identity/recency (no LLM) / evidence contract; terminus/stirrup LLM summaries | — | monotonic required except contract; one `/verify` per logical rollout |
| **Harbor terminus-2** | first user prompt + summary Q&A handoff | none; 10 KB head/tail; tmux | <8,000 free tokens; unwind newest to 4,000 | same model, 3 sub-agents / copied context not trained | — | ATIF segments `cont-N` |
| **Harbor dspy_rlm** | DSPy RLM state | REPL; 10,000-char output | DSPy-internal (not local) | — | 50 LLM calls, 20 iterations | none |
| **mini-swe-agent** | everything, linear | redirect-to-file hint; 5k+5k | none (abort) | — | step/cost limits; handoff file at N−20 steps | trajectory = context |
| **OpenEnv repl_env** | prompt + 500-char preview; linear runner | `context` var; 8,192 stdout; FINAL_VAR | none | — | depth 1, 8 workers, other caps None, cooperative timeout | capture graph: new root per rewrite |
| **FineEnvs** | harness-native | — | harness-native | — | 17 steps, 600 s | per-row (CC ×8 rows) |
| **agentenv** | agent-owned (A2A session_ref) | changelog per tool call; judge-side >1 KB offload | agent-owned | — | peer-agents, no limits | — |
| **autoresearch** | agent-native | log files + grep; `results.tsv`; git | agent-native | — | 5 min per experiment | — |
| **GitSwarm** | fresh per episode | git graph, MY_THOUGHTS/EVALUATION, dashboard | n/a (short episodes) | worker-written handoffs | 5 concurrent, 100-500 episodes | proposed only |
| **Crucible `RLMPolicy`** | system + verbatim task + summary + `[python-state]` + 3 turns | TASK/TOOLS/OBS/INBOX vars; `HISTORY` (capped outputs) | 0.8×128k on chars/4, before each call; no reactive | policy, 1 attempt / recorded `kind="compaction"` | query/batch depth 1; 256 calls, 4 concurrent, 2M sub tokens | segments + M6 per-call credit |

---

## 5. Failure modes catalogue (with the mitigation each source uses)

| Failure | Evidence | Mitigation seen |
|---|---|---|
| Context rot in a long transcript | RLM:17-31 | Data in kernel variables; output cap; sub-LLMs |
| Unbounded growth inside one run | Prime Agent compacts only at run boundaries (`D/auto_compaction.rs:32-62`) | nano-rlm checks after every tool result (`engine.py:995`) |
| Overflow with no recovery | PA allows 1 retry per run, then the goal fails (`D/overflow_compaction.rs:17`); Crucible ends `context_overflow` | nano-rlm: reactive compaction + 5 attempts + hollow-middle + clean `compaction_failed` stop |
| Summary loses the task | VF chat program, nano-rlm, Codex | MiMo CC `<original_user_issue>`; Crucible's task anchor; nano-rlm `h.user_messages()` |
| Summary misses evidence | PA summarizer blind to tool output; terminus-2 discards the pending observation | Feed capped tool results to the summarizer; terminus-2's questions sub-agent probes gaps |
| Summary drift / lag | PA marks "this summary may lag behind" retained messages | Update-mode summary + recent-state anchor; "do not invent state" (nano-rlm) |
| Lost variable names | all RLMs | Kernel note in the summary prompt + `[python-state]` row |
| Repeated actions / stalls | CH 1,003-turn stall, 842 identical payloads; browsecomp loop guard (≥6 of last 10) | Stall signatures (Crucible), no-progress cap 3 (PA), forced finish |
| Junk tool-call loops | MiMo 76.6% non-existent tools, 89.1% died on budget | Format-error cap (mini-swe 3); parser fixes; penalty masks |
| Duplicated work after compaction | nano-rlm warns explicitly | "Refresh resource metadata and inbox state before repeating work" |
| Blindly repeating an interrupted cell | nano-rlm, PA | Kernel-recovery notice; never replay side effects |
| Premature "done" | LHTB, PA goals | Goal continuation with audit; `continue_until_timeout` with verifier feedback |
| Running out of budget before answering | FS PortSim; Tongyi | Steps-left / time-left warnings; forced final call; salvage last text |
| Children that never reply | PA | Automatic no-reply / failed notices; `result(yield_after)` |
| Unbounded recursion cost | SkyRL (no depth cap), OpenEnv (caps None), PA (no budgets) | Tree-wide token, turn and spawn caps; per-depth reserved concurrency (nano-rlm) |
| Training on a summary the run never used | VF bash (no edges) | Accepted-only rule (`trace.py:515-553`) |
| Silent tail truncation in training | PR `batch.py:385-406`; slime `max_sample_tokens`; verl `[:response_length]` | Compaction threshold < `seq_len`; mask or flag over-length rows |
| Row multiplication skews the data | ART:1012-1014 | Per-logical-rollout weighting |
| Reward on the wrong segment | MiMo `longest` | `trajectory_selection=all`; final-chain stats broadcast |
| Judge failure scored 0 | SkyRL `evidence_rewards.py:345-361` | Mask (longpdfs flags rollouts instead, `longpdfs.py:290-293`) |
| Self-improvement regresses weak models | CH:443-451, 2023 | Measured A/B before keeping any harness edit (Crucible M14) |
| Exploit preserved as a skill | PA:632-638 | Edits pass the task's checks before persisting |
| Seed noise read as progress | KA (42→137 kept), GS paper:1949-1955 | Paired seeds; re-baseline |

---

## 6. Measured results relevant to very long horizons

| Result | Source |
|---|---|
| ARC-AGI-3: Prime Agent + Opus 5 95.5% vs 30.2% on the ARC harness; up to 3M output tokens per game | PA:364-376 |
| Factorio: 7 days, 23.4M output tokens, 633 depth-1 children in 149 waves, ≤7 concurrent, survived a world reset | PA:623-631 |
| OOLONG at 128k: 0.70 → 0.94 under Prime Agent | PA:396-411 |
| Rust rewrite: 2,209 agents, 228.7B tokens, 16,758 messages, depth ≤3, >100 kernels per 8-core node | RUST:28,56 |
| CH: runs past turn 170,000; sub-agent tokens ~10× below the orchestrator's | CH:1095,1937-1946 |
| RLM blog Oolong real: RLM > LLM up to ~1.5M chars (~300-400k tokens), none succeed beyond | rlm.txt:492 |
| RLM main-model context "much lower"; sub-LLMs carry most completion tokens | rlm.txt:382-384 |
| GitSwarm ProgramBench 79.4% vs 65.1% for one agent forced to continue | paper:339-342 |
| FineEnvs: harness swap moves a model 13 points; CC compaction 8 rows per rollout, 451M vs 162M tokens trained | ART:255-270,1012-1014,1274-1277 |
| Stirrup: truncation-recovery turns 15.8% of model time on GDPVal | `NG/responses_api_agents/stirrup_agent/app.py:1034-1044` |
| verl fully async 2.35-2.67×; AReaL 2.77× | `VL/.../fully_async_policy/README.md:9`; `AR/blog/AReaL_v0_3.md:10` |
| Crucible live: Sonnet 5.5 under the RLM harness 1.0 vs native 0.67 on the 12-ticket backlog (n=1 each) | `crucible/docs/research/improvements/M13-M15-RLM.md` |

**Not measured anywhere:**
- compaction count against success;
- summary quality against downstream reward;
- the cost of a compaction for prefix caching at RL scale;
- credit across a compaction boundary.

---

## 7. Design requirements for Crucible's very-long-horizon RLM harness (ranked)

Ranked by how much each one blocks horizons of thousands of steps and millions of tokens. Each requirement gives the
mode (DEPEND / PORT with licence / PATTERN / NEW), the parameters, and the gap against §1.

1. **An addressable, append-only episode ledger that the kernel can query.** *PATTERN from nano-rlm (MIT, the API
   shape may be ported).*
   - Every message is logged once with a stable index: system, user, assistant, the full tool output, the shortened
     context version linked by `source_message_index`, summaries, notices and sub-agent spawns.
   - Context windows are numbered with reason `start | compaction | restore`.
   - The kernel gets `history()` with `.messages[i]`, `.windows[w].messages[i]`, `.user_messages()`, `.search(regex)`
     and `.child(name)`.
   - The ledger is **on disk** (the attempt directory, streamed), not in kernel memory.
   - **Replaces** Crucible's in-kernel `HISTORY`, which holds only capped outputs and grows the kernel.
   - The post-compaction message names the API (`NR/src/rlm/engine.py:1371-1381`).
2. **Usage-anchored context accounting, checked after every tool result and before every call.**
   *PORT (MIT) from Prime Agent `SE/compaction.rs:39-295` + nano-rlm `engine.py:1133-1146`.*
   - Tokens = the last accepted call's server usage (`prompt + completion`) + chars/4 of what was appended since.
   - Usage from before a compaction never re-triggers it.
   - Threshold = `min(0.95·W, W − max_out − 16,384)`, floored at `W//2`.
   - **For training, also ≤ `seq_len − max_out`.** PR uses 98,304 under 131,072. No segment may ever reach the
     trainer's tail cut.
   - Replaces `approx_tokens(messages) ≥ 0.8 × budget`.
3. **A reactive overflow path and a robust summary loop.** *PORT (MIT) nano-rlm `engine.py:1195-1365`,
   `compaction.py:53-87,175-223`.*
   - **Triggers.** A 400/413 overflow, one of the 9 markers, PA's silent overflow (input > W, or a `length` stop with
     0 output and input ≥ 0.99·W, `R/crates/pa-ai/src/utils_inner/overflow.rs:8-98`), or a `length` stop past the
     threshold. Each discards the reply and compacts.
   - **The summary loop.**
     - Up to `max_compaction_attempts=5`.
     - On an overflowing summary request, `hollow_middle` the *summary input*: keep the start and end, whole tool
       exchanges, over-remove ×1.25.
     - Resample empty, tool-calling or reasoning-only replies.
     - Then a typed stop `compaction_failed` that ends cleanly, salvages the last assistant text or summary, and counts
       as truncation (not error).
   - Add `max_compactions` (default unbounded, but recorded).
   - Allow more than PA's single retry per run.
4. **The context after compaction:**
   - system;
   - the **verbatim task anchor**;
   - the summary, with nano-rlm's checkpoint prompt + REPL note, PA's update mode (`<previous-summary>`), and a
     recent-state anchor of ≤2,000 chars;
   - a `[python-state]` row: names **with types and sizes, capped** (e.g. 60 names, 2,000 chars), plus variables lost
     or pruned;
   - a boundary marker (count, pre/post tokens);
   - the recent tail: ~20k tokens or the last K complete turns, cut only before an assistant row so tool results stay
     paired.

   *PATTERN: MiMo `surgery.py:47-132` (MIT), PA `SE/compaction.rs:297-442` (MIT).*
   - **The summarizer must see tool outputs**, capped at 2,000 chars with a 500 tail each. PA's blindness to them is a
     defect.
   - No fabricated assistant turn, and no "another LLM" framing.
   - Framing: "retained messages are authoritative; this summary may lag".
5. **Observation discipline: cap what is printed, never lose the full output.** *PATTERN: nano-rlm shell jobs, PA
   native bash spill, mini-swe-agent template (MIT), RLM blog.*
   - Per-cell printed output stays at `max_output_chars=8,192`, head + tail.
   - The marker states the total size **and where the full text is**: ledger index plus a kernel handle, e.g.
     `OUT[i]` lazily loaded or `await out(i).read(cursor, max_bytes)`.
   - Background job output is retained up to 16 MiB with cursor reads.
   - The system prompt states the cap and says "slice, grep, or delegate".
   - Crucible already caps; the full output is what is missing.
6. **Long inputs are variables or files with a metadata line, never prompt text.** *PATTERN: SkyRL `env.py:137-159`,
   OpenEnv preview + length, RE `/workspace/context.txt`.*
   - `TASK`/`TOOLS`/`OBS`/`INBOX` stay.
   - Add a `CONTEXT` handle for packs with large inputs: type, total chars, chunk lengths, a preview of ≤500 chars,
     backed by a file for multi-GB data.
   - Answers are a call (`rlm.finish`) with a salvage fallback.
7. **Tree-wide budgets enforced by a supervisor, all recorded as typed truncation stops.** *PORT (MIT) nano-rlm
   `config.py:51-93`, `supervisor.py:146-158,439-456`.*
   - Budgets: `max_total_new_tokens` (nano-rlm 1M; VF 10M), `max_total_turns`, `max_subagent_calls`, `max_depth`
     (default 1, allow 2), and `max_concurrent_subagents` with **per-depth reserved capacity**.
   - Per-cell `exec_timeout` (Crucible 120 s; nano-rlm 300 s, capped at 600 s), measured on own compute only
     (Crucible already excludes tool waits).
   - Wall-clock budget per child.
   - Summary calls count toward tokens, not turns.
   - Child usage is billed to the tree but never to the parent's context count (`SE/rlm_usage.rs:32-38`).
   - Raise the episode `max_turns` from 500 to a budget-derived value (thousands).
8. **Persistent child agents (depth 2) next to the synchronous `rlm.query`/`batch`.** *PATTERN from nano-rlm and PA
   (MIT).*
   - `rlm.agent.spawn(task, name, persistent)` returns a handle: `result(yield_after)`, `send`, `steer`,
     `history()[start:end]`, `cancel()` (cascades).
   - `watch(child, every_turns=10 | every_tokens=50_000)` progress events.
   - Automatic `agent.completed` (last 4 KiB) / failed / no-reply notices; an inbox of ≤10,000 events with payloads
     ≤65,536 bytes.
   - Each child has a fresh context (a self-contained brief), its own kernel and ledger directory under the parent's,
     and a shared workspace for fan-in.
   - Every child call is a recorded model call with `parent` and a `subagent_call` edge.
9. **Training records as a message graph with typed semantic edges.** *DEPEND or PORT (MIT) verifiers
   `graph.py`/`trace.py`; PATTERN prime-rl `trajectories.py:105-191`.*
   - Edges: `continuation`, `compaction_attempt`, `compaction`, `subagent_call`, `subagent_return`, `agent_message`,
     `subagent_cancel`.
   - Delta tokens per node.
   - A compaction attempt is trainable only when accepted; a shared node is trained once.
   - The exact-prefix check inside a segment uses NG `required_prefix_token_ids` (*PORT, Apache-2.0*,
     `history.py:906-913`).
   - Crucible's `model_calls` (`kind`, `segment`, `parent`) maps onto this; make the edges explicit.
10. **Credit and weighting across compaction and children.**
    - GRPO statistics come from **logical rollouts**, never rows.
    - Loss weight is per logical rollout (token-mean pooled over its rows), so a rollout that compacts 8× does not
      dominate (ART:1012-1014).
    - M6 per-call ΔΦ already crosses segments.
    - Optional role-keyed baselines (`task_id:role`, rLLM `transform.py:115-132`) for children.
    - Sub-calls answered by a frozen model are environment calls (`sub_trainable=False`, already in Crucible).
    - **Never** silent tail-cut: an over-length row is masked and counted.
    - Judge or infra failures mask; they never score 0.
11. **Goal persistence and stall control.** *PATTERN: PA goals + CH, browsecomp guard, mini-swe ProgramBench.*
    - The objective is durable state outside the prompt (the task anchor).
    - Continuation with "audit the current state against every requirement" (Crucible has this).
    - A no-progress cap of 3.
    - Stall signatures: Crucible's, plus "same (tool, args) ≥6 of the last 10".
    - Steps-left and time-left notices at N−20 steps and T−600 s, asking for a handoff note in the workspace.
    - A post-compaction "refresh state before repeating work" line.
12. **Checkpoint and restore that scale with the horizon.**
    - Keep Crucible's exact journal replay, but bound its cost: a periodic kernel snapshot (dill, PA limits of
      256 MiB / 16 MiB per variable, `PY/repl.py:40-41`) plus the journal *suffix* since that snapshot.
    - Replay checks the namespace digest; if a snapshot cannot be restored exactly, fall back to full replay.
    - Checkpoint index = tool-call position, as in AE changelog `000042`.
    - The current segment's context (≤ one window) plus the ledger offset make a resume cheap.
    - After any kernel loss, inject the recovery notice ("variables lost; never blindly repeat the interrupted cell",
      `NR/src/rlm/prompt.py:119-122`).
13. **Optional policy-decided compaction, with a pressure signal.** *PORT (MIT) MiMo `Compact` + `<context_usage>`
    footer at 80% (`cc_agent.py:72,227-244`); PA `compact.status()/run(instructions)`.*
    - Always backed by the hard threshold (requirement 2).
    - The mode and threshold are recorded in `harness@version`, one per GRPO group (`ContextConfig.mode="action"`
      exists).
14. **Context-health and scaffold-use metrics, never in the reward.**
    - `num_compactions`, `turns_between_compactions`, chars dropped / summary chars, compaction failures
      (`NR/src/rlm/types.py:137-173`).
    - `steps_per_traj` and the merge ratio (rLLM).
    - Truncation hits, re-reads of `OUT`/`history()` after compaction, and the delegation rate.
    - A weight-0 rubric from `RE/judges/rlm/rubrics.toml` (bounded_output, stateful_variables,
      programmatic_tool_calling, subagent_delegation).
    - These are the only way to learn whether compaction quality matters, which no source measures.
15. **A long-context bench ladder to validate all of the above.** *PORT (Apache-2.0) the RE oolong / mrcr_v2 /
    longpdfs generators and scorers.*
    - Rungs 8k → 4M tokens.
    - Data delivered as a variable or file.
    - RLM vs native, compaction on vs off, depth 1 vs 2.
    - Plus a step-count ladder on a World pack (100 → 5,000 tool calls) with milestone Φ, so compaction loss is
      visible as ΔΦ per segment.
16. **Pruning before summarizing.** *PATTERN NG/HB.* Drop old images (keep 3, charge 1,300 tokens each) and old
    reasoning blocks before an LLM summary. The lossless copy stays in the ledger.
17. **A cache-stable static system prefix, with memories and digests in a separate user row** (PA). At eval only, an
    auxiliary summarizer model is allowed so the main model's cache is not broken. In training, the policy writes the
    summary so inference and training match.
18. **Kernel isolation and de-ambient credentials.** Crucible already does this: the Docker kernel, `safe_environ`,
    and nano-rlm's explicit `kernel_env`. Every sub-agent kernel follows the same rule.
19. **Shared memory across agents as files or git with provenance**, never a policy-side memory in train or eval (the
    B.15 threat). *PATTERN GitSwarm `informed_by`, `MY_THOUGHTS.md`; autoresearch `results.tsv`.* Φ never reads
    notes.
20. **`continue_until_timeout` re-entry with fast public checks only** (LHTB pattern; Harbor lacks it), declared to
    A13 as a feedback channel.

### What to avoid (from the sources)

- **Compaction behaviour:**
  - compacting only at run boundaries, or allowing only one overflow retry (PA);
  - a summarizer that cannot see tool output (PA);
  - dropping the newest turns or the pending observation at handoff (terminus-2);
  - losing the task in the rebuilt context (VF chat, nano-rlm, Codex);
  - fabricated assistant turns (computer-1);
  - "another language model" framing (VF).
- **Truncation:**
  - head-only truncation with no retained copy (PA kernel, SkyRL);
  - verl's 256-char tool default.
- **Recursion budgets:** unbounded recursion or `None` caps (SkyRL, OpenEnv, PA); cooperative-only child timeouts
  (OpenEnv).
- **Training:**
  - per-turn instruction pops that break every prefix (SkyRL);
  - `trajectory_selection: longest` (MiMo);
  - silent `seq_len` tail cuts (PR, slime, verl);
  - one row = one unit of weight (FineEnvs CC skew);
  - scoring judge failures as 0 (SkyRL);
  - cumulative-usage "context" accounting (rLLM deepresearch bug);
  - chars/4 as the trigger when server usage is available.
- **Self-improvement:** greedy keep-if-better on one noisy run (autoresearch).

### Open questions this study cannot answer from the sources

- Does training the summary with the episode advantage improve it, or does it need its own signal (e.g. ΔΦ
  retention across the boundary)? No source measures it.
- Keep a recent tail (PA ~20k, Crucible 3 turns) or not (nano-rlm none)? With an addressable ledger, the tail mostly
  buys cache reuse and local coherence. Ablate it on the ladder (requirement 15).
- What is the best depth? Factorio used only depth 1 (633 children). The Rust rewrite used depth ≤3.
- nano-rlm's whole-tree token budget vs per-child wall time for RL cost control: measure on the step ladder.
