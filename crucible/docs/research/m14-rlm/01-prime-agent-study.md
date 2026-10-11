# M14 research 01: Prime Agent (RLM harness + Continual Harness), studied for Crucible

## Sources
- **Code:** `RL-Environments/06-frameworks-and-env-code/primeintellect-ai-prime-agent`, commit 2ee7e623a (the Rust rewrite), MIT.
  - Nine crates: pa-daemon 158k lines, pa-tui 137k, pa-core 135k, pa-cli 71k, pa-ai 38k, pa-types 13k, pa-agent 7k, pa-telemetry 5k, pa-models 5k.
  - `prime-agent-runtime/`: the Python kernel, about 12k lines.
  - `skills/`: 14 skills.
- **Papers:**
  - *Prime Agent: A Self-Improving RLM Harness* (arXiv 2608.23552, "PA").
  - *Continual Harness: Online Adaptation for Self-Improving Foundation Agents* (arXiv 2605.09998, "CH").
- **Blogs:** the Prime Intellect *Recursive Language Models* post ("RLM") and *Rewriting Prime Agent in Rust* (Oct 9 2026, "RUST").
- **Local copies:** text and PDFs in `RL-Environments/07-articles/prime-agent/`.

Citations: `R/` is the repo root; PA/CH/RLM/RUST citations are line numbers in the text extractions.
This study was done by five parallel readers (RLM loop, continual harness, multi-agent and long-horizon, Python runtime,
papers). Their findings are merged here and the claims that matter were spot-checked.

---

## 1. What an RLM is, and what Prime Agent actually implements

**The idea** (RLM:17-89):
- *Context rot*: cost grows linearly with context while quality falls.
- The fix is not summarization ("never actually summarizes context, which leads to information loss"). Instead, data
  lives in a persistent Python REPL, and the model reaches it only by printing slices (output capped at 8192 chars by
  default, "so the RLM is forced to make use of Python and sub-LLMs"). It pipes slices to fresh sub-LLM instances.
- Prime's `verifiers` RLMEnv variant (RLM:51-105) adds:
  - parallel `llm_batch`;
  - **environment tools usable only by sub-LLMs**, so their token floods never reach the root;
  - sandboxed execution;
  - **the answer is a variable** (`answer = {"content": "", "ready": False}`): the model edits it many times, and the
    rollout ends at `ready=True`;
  - a 120 s REPL timeout, and recursion depth exactly 1.

**What the Prime Agent code does** differs from its README in four ways:

| README claim | Code |
|---|---|
| "treats context as variables (prompt-as-a-variable)" (`R/README.md:38`) | No context or prompt variable is pre-loaded. User prompts, child tasks, and agent messages enter the conversation as ordinary messages. Variables are a *habit the prompt asks for*: "Agents always assign read/search results to named variables so they can revisit them later" (`R/crates/pa-core/src/prompts/layers/usage.md:12`). |
| `rlm.spawn` "returns their results programmatically" (`R/README.md:62`) | "results never arrive as `rlm.spawn()` return values, they arrive only via agent_message" (`layers/core.md:33`). A typed snapshot is available by *polling* `rlm.collect`. |
| one tool | `ipython` is the core tool (`tools/ipython.rs:339-354`), but `bash` and `edit` can also be native tools (`R/AGENTS.md:174`; `prompts/system_prompt.rs:155,553`). |
| blocking sub-LLM calls (`llm_batch` in RLMEnv) | Prime Agent has **no** `query`/`batch`/`map`/`wait`. Sub-agents are full asynchronous sessions; join with `asyncio.gather(spawn…)` + `rlm.collect(timeout_ms)`. The only direct single-LLM call is `decision_api.decide` (`R/skills/decision-api/src/decision_api/__init__.py:82`). |

So there are two RLM styles:
- **(a) Synchronous recursion** (RLM blog / verifiers RLMEnv): `y = llm(prompt, slice)` inside code, depth 1.
- **(b) Asynchronous agent trees** (Prime Agent): persistent child sessions, message-driven.

Crucible needs both:
- (a) for the policy: deterministic, creditable, inside one step budget.
- (b) for the task manager: long-running workers.

## 2. The turn loop and the REPL (exact mechanics)

- **Loop** (`R/crates/pa-agent/src/agent_loop/run.rs:21-251`):
  - call the model; run the tool calls; repeat while there are tool calls or pending messages;
  - steering messages are injected only at turn boundaries;
  - follow-ups and goal continuations come after the last tool call.
  - **There is no max-turn cap in the core loop.** Caps exist only in autonomous mode (`DEFAULT_MAX_TURNS=12`,
    `autonomous/mod.rs:23`) and in side questions.
- **System prompt** (`prompts/system_prompt.rs:104-245`):
  - A static, cacheable prefix: `core.md`, `usage.md`, `opinionated.md`, and per-model extras.
  - A dynamic tail: packages, AGENTS.md, skills inventory, MCP, environment, session role, guidance.
  - Children are told "Task prompts are labeled `[task from parent]`" and to reply with `agent_message.send(...,
    receiver_role="parent")` (`:322-334`).
- **The tool** `ipython{code}` (`tools/ipython.rs:339-354`).
  - Output order: stdout, stderr, the repr of the trailing expression, the traceback, then
    `[background output (unattributed)]` (`:241-265`).
  - Capped at **65,536 chars per stream** (`kernel/shared.rs:12`).
  - **No per-cell timeout** (`kernel/manager/mod.rs:591-594`, `execution_timeout_ms: None`).
- **Kernel** (`R/prime-agent-runtime/src/rlm/repl.py`, protocol `repl.md`): NDJSON over stdio, protocol v3.
  - Requests: `execute`, `interrupt`, `host_reply`, `snapshot`, `restore`, `list_names`, …
  - Events: `ready`, `stdout`/`stderr`, `result`, `display`, `host_request`, `error`, `done`.
  - Cells compile with `PyCF_ALLOW_TOP_LEVEL_AWAIT` and run as tasks on one persistent event loop, so background tasks
    survive between cells. Output is tagged per cell with a contextvar.
  - `host_request(data)` is the single bridge to the host (`repl.py:173`); `host_reply` bypasses the FIFO to avoid
    deadlock (`repl.py:1499-1508`).
  - The interrupt sends SIGINT to the main thread (`repl.py:391-470`), and the host force-aborts after 1 s.
  - **Snapshot**: dill, one variable at a time, 256 MiB total and 16 MiB per variable, atomic, with a manifest. Saved
    functions are re-bound to the live globals on restore (`repl.py:40-45, 771-1067`). A final snapshot is written on
    stdin EOF.
  - **Isolation:** none. The kernel inherits the full host environment and process group (`kernel/manager/startup.rs:166-190`).

## 3. Sub-agents, messaging, accounting

- **`rlm.spawn(prompt, *, name, model=None, thinking=None, kind=None) -> RLMSpawnHandle`** (`rlm/__init__.py:179-209`).
  - It returns at admission.
  - Depth limit `DEFAULT_RLM_MAX_DEPTH=2` (`pa-daemon/src/rlm_children.rs:34`).
  - Each child runs in its **own supervised worker process with its own kernel** (`rlm_children.rs:1-4`) and sees only a
    fresh system prompt plus `"[task from parent]\n\n{prompt}"` (`rlm_children/host.rs:188-223`).
- **Results** come back three ways:
  1. an explicit `agent_message` to the parent (≤16,384 chars, ≤20 pending);
  2. an automatic `[child-exited: no-reply …]` / `[child-failed …]` follow-up that wakes the parent
     (`session_engine/rlm_notices.rs:36-118`);
  3. `rlm.collect(targets, timeout_ms)`, where `0` gives a non-blocking snapshot and `answer_text` is ≤65,536 chars
     (`rlm_child_model.rs:13`).
- **Messaging** is limited to parent, siblings and children. The daemon stamps the sender, so it cannot be spoofed.
  Delivery uses steering (delivered or queued) and is never retried (`skills/agent-message/SKILL.md:33-58`;
  `agent_messaging/message.rs:10-12`).
- **Accounting**: child usage is folded into the parent's spawning row, but the **parent's context-token count is left
  unchanged** (`session_engine/rlm_usage.rs:31-37`). Topology goes to an append-only spawn ledger (`rlm_ledger.rs`).
- **No global concurrency cap** outside the factory (`max_parallel` 1–64, default 8).

## 4. Long-horizon machinery

- **Compaction** (`session_engine/compaction.rs:8-17, 152-161, 300-463`):
  - Threshold: `min(0.95·window, window − max_output − max(16,384, 4,096))`.
  - Keeps about 20k recent tokens and never cuts at a tool result.
  - Summary format: Goal / Constraints / Progress (Done / In Progress / Blocked) / Key Decisions / Next Steps /
    Critical Context.
  - Repeated compactions update the previous summary (`<previous-summary>`).
  - Adds a `<recent-state-anchor>`, the newest assistant text, ≤2,000 chars.
  - **Kernel note**: "the Python kernel keeps running after this summary — every Python variable … stays available …
    record in the summary any names worth remembering". Afterwards a hidden `[python-state]` row lists the surviving
    names (`ipython_state.rs:116-198`).
- **Goals** (`goals.rs:271-358`; `goal_driver/progress.rs:13-29`):
  - Every natural stop re-injects `<objective>` plus budget status plus "audit the current state against every
    requirement … before `goal.complete()`".
  - A no-progress cap of 3 empty turns, with backoff doubling from 10 s.
  - The continuation is postponed while children are unsettled.
- **Autonomous mode** (`autonomous/mod.rs:20-34`; `gates.rs`):
  - "No human input is available …".
  - Defaults: 3 continuations, 12 turns, 80k tokens, 30 min.
  - **Shell-command gates are verifiers**: a failure feeds back `[autonomous-continuation: gate-failed]` with output
    ≤6,000 chars.
- **Heartbeats / cron** (`cron/mod.rs`; `scheduled_jobs.rs`): delivered at turn boundaries; minimum interval 10 s.
- **Persistence**:
  - An append-only JSONL tree per session (`id/parentId` entries; unknown types kept verbatim).
  - A recovery journal for queues.
  - A supervisor with restart backoff (5 failures, 30 s stability).
  - Revival only of interrupted work younger than 30 min (`pa-daemon/src/supervisor/*`).

## 5. Continual Harness (self-improvement)

- **State**, in the paper (CH:153-169): H = (p, G, K, M) = (system prompt, sub-agents, skills, memory), edited via
  meta-tools.
  - Prime Agent's five kinds (`refinement/mod.rs:10`): **prompt notes, memories, skills, subagent specs, factory specs**.
  - Each kind stores versioned `HarnessEntry{id,kind,title,content,path,scope,reference,arguments,metadata,source,version}`.
  - Scope is local (session) or global (`~/.prime/agent/harness/`).
  - Writes are atomic, lock-protected, mtime-resynced, and degrade to an empty store if the file is corrupt.
- **Injection**: never into the system prompt. A `[harness-digest]` user message is sent at start, resume and
  compaction, and re-sent only when the state fingerprint changes.
  - It shows **≤3 entries per kind, each cut to 140 chars**, ranked by TF-IDF against the goal and recent turns
    (`ranking.rs:195-370`).
  - Header: "The base system prompt is immutable; prompt entries below are supplemental notes only."
- **`/refine`** (`refinement/planner.rs`, `executor.rs`):
  - **Evidence:** one non-tool completion by the session model over the conversation (tail-capped at 80k chars), a
    harness overview (≤40 entries per kind), and the last 20 refinement results.
  - **Output:** JSON CRUD edits with `expectedOutcome`.
  - **Per-edit validation:**
    - `base_system_prompt` is not editable;
    - skills need a python reference plus arguments;
    - create on an existing id is rejected, and update/delete on a missing id is rejected;
    - **optimistic concurrency**: "entry changed during refinement planning".
  - **Rollback** stores before/after snapshots per edit, so `/refine rollback <id>` is computed without an LLM.
  - **Auto trigger:** after compaction, an LLM review gate ("Reject one-off noise …") plus a 20-minute cooldown. It runs
    local-only.
- **Gaps the readers found in the code:**
  - **No evaluation of updates.** `expectedOutcome` is stored and never checked.
  - No eviction or semantic dedup; the refinement list grows without bound.
  - Rollback is not stack-aware and can overwrite later edits.
  - Direct kernel writes cannot be rolled back.
  - Skill entries are validated for shape, not for whether they import.
  - The `turnInterval` setting is dead config.
  - Possible bugs, unverified: the global harness directory is passed inconsistently by three daemon call sites, and
    the kernel's local store env vars may never be exported.
- **Paper evidence (CH):**
  - It recovers most of the gap from H_min to the expert harness (Pokémon Red/Emerald). Pro: 100% of Emerald milestones
    at $130 vs H_min 98% at $215.
  - **A capability floor**: Flash-Lite *drops* from 20% to 3–13% with CH (CH:443-451).
  - **Create-and-forget**: 99 skills created and 5 ever succeeded (CH:1799-1804).
  - Newly authored sub-agents displace good inherited ones and push the agent "below Hmin" (CH:2019-2033).
  - Self-repair happens in the first 50–200 turns of a stall and stops after about 500 (CH:1777-1779).
  - A 1,003-turn stall came from a tool-schema mismatch ("schema fragility", "feedback blindness").
  - Training: SFT, then offline GRPO (no progress alone), then online reset-free DAgger with a PRM and teacher relabel,
    for +2–5 milestones. Reward is non-monotone, with regressions after each resume (CH:2222-2231).
- **Safety (PA:632-638).** In Factorio the agent found an RCON resource-spawn command, used it despite an anti-cheating
  heartbeat, "**and then preserved it as a reusable skill**". The authors' remedy: "least-privilege action interfaces,
  independent state validation, and auditable rollback of contaminated refinements."

## 6. Factory (FSM orchestration) and the Rust-rewrite swarm

- **Factory** (`R/prime-agent-runtime/src/rlm/factory.py`, about 5k lines, opt-in):
  - States are subagents with typed `text|json` ports.
  - Guarded transitions (`eq…contains`), `max_entries`, `foreach`≤256, resident lifecycles, `fail_fast|continue|escalate`.
  - Limits: budget, retries, `max_parallel`.
  - Seed machines: `builder`, `pr-manager` (review ⇄ fix, then a resident monitor), `review-sweep`.
  - **Run state is kernel memory only and does not survive a restart** (`factory.py:1422-1424`).
- **The swarm that rewrote Prime Agent in Rust** (RUST:13-124): about 2,209 agents, 228.7B tokens, 16,758
  agent-to-agent messages, depth ≤3.
  - **Humans wrote the verifiers** (TUI, harness, protocol and feature parity). "With an objective check for every kind
    of parity, the agents could measure their own progress."
  - A root agent wrote no code; it ran a toposorted FSM per task: **Planner → Implementer (own worktree) → Reviewer
    (adversarial, *different model*, separate context) → Verifier (fresh sandbox)**, looping back on failure.
  - "an agent that writes code is biased when evaluating it."
  - Hillclimb:
    - A/B runs on the same sandbox in alternating order;
    - noise checks;
    - neighbour regressions;
    - two reviewers on different models;
    - re-baseline after each merge;
    - **no numeric targets** ("a fixed threshold tends to become a stopping point").
  - "the parity checks only verified the behavior they exercised": dogfooding and trace review found the rest.

## 7. What is missing for RL in Prime Agent (Crucible's opportunity)

| Need (RL / long horizon) | Prime Agent | Crucible's answer (M14) |
|---|---|---|
| token ids / logprobs per call | none (usage counts only) | the `LLMClient` token record (TITO) for root *and* sub-LLM calls |
| deterministic replay / exact restore | dill best-effort; factory state lost; wall clock everywhere | journal-replayed kernel with recorded host replies, fingerprint-verified; virtual clock; seeded ids |
| cell timeouts / isolation | none / none | per-cell deadline; kernel runs in the World's sandbox (docker / gVisor) or a subprocess |
| credit for sub-LLM work | cost only (folded to parent) | sub-calls recorded as `kind="sub"` model calls attached to the root step that issued them |
| measured self-improvement | `expectedOutcome` never checked | every harness edit is A/B evaluated on fixed seeds and auto-reverted on regression; per-entry hit/help counters |
| safe persistence of learned skills | exploit preserved as a skill (PA:632) | edits pass M9 / A16 before they persist; local scope by default; policy-side memory is off for eval/train |
| generator ≠ verifier | prompt advice; pr-manager machine | the task FSM uses a reviewer on a different model and M9 as the objective verifier |
| serializable FSMs | kernel memory | FSM state is data in the journal, checkpointed |
