# Swarm pipeline, study 01: how Prime Agent runs a swarm for long-horizon work (deep read at HEAD 7cbffa77b)

- **Question.** How exactly does Prime Agent organise many agents for long work (factory machines, persistent
  children, messaging, verification in the loop), and what should Crucible's automated long-horizon coding task pipeline
  take from it?
- **Read:** 2026-10-17. Local checkouts only; no network. Prime Agent is MIT. Delta base: study
  `docs/research/task-factory/01-prime-agent-factory.md` (commit 2ee7e623a; 73 commits since, 575 files changed).
- **Path keys.** `PA/` = `RL-Environments/06-frameworks-and-env-code/primeintellect-ai-prime-agent/`. `F` =
  `PA/prime-agent-runtime/src/rlm/factory.py` (5,278 lines). `RLM` = `PA/prime-agent-runtime/src/rlm/__init__.py`.
  `MX/` = `PA/prime-agent-runtime/src/rlm/machines/`. `PAC/` = `PA/crates/pa-core/src/`. `PAD/` = `PA/crates/pa-daemon/src/`.
  `LAY/` = `PAC/prompts/layers/`. `SC/` = `PA/scripts/`. `TF` = `PA/prime-agent-runtime/test/test_factory.py`.
  `RUST` = `RL-Environments/07-articles/prime-agent/prime-agent-rust.txt`, `RLMB` = `.../prime-agent/rlm.txt`.
  `CR/` = `crucible/src/crucible/`.

## 0. Bottom line

1. **The factory is still a generic guarded state machine over spawned child sessions; it has no notion of sandbox,
   worktree, verifier or reviewer.** States spawn children with `rlm.spawn(prompt, name, model, thinking)` (`F:2798-2805`);
   a "verifier gates a transition" only because a child prints a fenced JSON verdict and a guard reads it
   (`F:1292-1347`, `MX/pr-manager/MACHINE.md:128-147`). The Planner/Implementer/Reviewer/Verifier machine is blog-only
   (`RUST:62-72`): "We are building this pattern into Prime Agent" (`RUST:72`).
2. **What changed in the factory since 2ee7e623a** (commits f2425dc1b, 8d58e5b6f, d356a1c69; +383/-80 lines in `F`, 13 new
   executor tests, `TF` now 309 tests):
   - **Escalate holds the failed entry unsettled; resume re-admits it.** Under `escalate` the failed settle is no longer
     queued, so no transition fires and no downstream entry is spent (`F:3339-3346`); `resume` first calls
     `_readmit_held_entries` (`F:1739-1745`, `:1763-1806`): same rendered prompt, same entry slot, no extra `max_entries`,
     fresh `retries` (`retry_base` moves, `F:1368-1371`, `:3252-3256`). Previously resume continued "the remaining states"
     and the failed state stayed `error`.
   - **Worker deaths are infra, not task failure: verified, then one free respawn.** An error whose text matches
     `child worker unreachable` / `supervisor link` (`F:1095-1116`) is re-read after a quiet period of one poll slice
     (`F:1089`, `:2990-3017`); if still dead and not yet respawned, the zombie is deleted and the same prompt re-queued
     without spending `retries` or `max_entries` (`F:3019-3084`, `infra_respawned` `F:1372-1373`). A second death takes
     the normal retries/policy path (`F:2894-2895`). Paired daemon fix: delete a dead-worker child without a worker
     handshake (d356a1c69, `PAD/supervisor/routing.rs`).
   - **An output that never binds fails the producer, not the dependent.** After one bounded re-capture window
     (`F:3171-3222`), an unbound declared port is a task failure of the producing entry (retries, then policy), with the
     answer kept as provisional for `needs_verify` (`F:2969-2988`, `:3123-3169`, `capture_failed` `F:1374-1376`).
     Previously the producer settled valueless and the dependent failed at bind.
   - **Residents reject `retries`** (`F:318-321`), and the authoring guide gained the **"retry re-renders the same prompt
     into a changed world"** rule (`F:5099-5107`); the pr-manager fixer prompt now reconciles the worktree first
     (`MX/pr-manager/MACHINE.md:88`).
   - **Unchanged:** loop exhaustion still ends `done` with no approval. A fire onto a state at `max_entries` is only a
     `transition_blocked` event (`F:2469-2483`), `_finalize` sets `done` when no entry errored (`F:3515-3541`), and a test
     pins it (`TF:5450-5477`, "blocked transition quiesces done"). The pr-manager seed's claim that "the run pauses
     (escalates) instead of spinning" (`MX/pr-manager/MACHINE.md:30-31`) holds only for `max_transitions`: its loop
     reaches at most 7 transitions (< 24) and then ends `done` unapproved.
3. **The swarm is a tree of persistent agent sessions, each with its own CPython kernel**, depth-bounded (default 2,
   `PAC/session_engine/rlm_host.rs:265`), talking only within the nuclear family (`LAY/core.md:29`). Results of a spawn
   never come back as a return value; they come by `agent_message` or files (`LAY/core.md:33`, `LAY/usage.md:32`).
4. **Prime verifies its own factory with hosted receipts, not just unit tests**: a real daemon binary against a scripted
   local OpenAI-compatible provider, a JSON receipt with binary SHA-256 and an explicit `not_verified` list, and a unit
   test that mutates the receipt to prove the checker rejects wrong outcomes (`SC/review_factory_resumption_parity.py`,
   `SC/test_review_factory_resumption_parity.py:13-20`). This is the pattern to copy for verifying Crucible's own pipeline.
5. **Crucible's `CR/factory/machine.py` is already ahead of Prime on the semantics that matter for trust** (journal,
   USD cap, exhaustion = escalate, cross-family by config, staging + canary), and already had "resume re-runs the
   failed entry with fresh retries" (`CR/factory/machine.py:959`, `:1195`). Port the two new Prime ideas it lacks:
   the infra/task failure split with one verified free respawn, and a declared recovery posture for state-mutating
   states. Add `foreach` + join only if per-task fan-out moves into the machine (section 6).

## 1. The factory, from the code

### 1.1 Where it lives

| Layer | File | Role |
|---|---|---|
| Kernel executor | `F` (`FactoryExecutor`, `F:1515`) | validates, canonicalises, runs; owns run state in kernel memory (`F:23-26`, `:1453-1456`) |
| Kernel API | `RLM:569-630` (`rlm.factory.run/status/stop/resume/graph/watch/help`) | thin wrappers; `help()` returns `FACTORY_HELP` (`F:4890-5278`) |
| Spec store | `PA/prime-agent-runtime/src/rlm/harness.py:34-39, :360-377` | harness kind `factory`; every write (incl. refinement edits) runs the validator; gated by `factory.enabled` |
| Host bridge | `PAC/session_engine/factory_host.rs:1-21, :250-287` | daemon/TUI -> kernel lane; `preflight_run` resolves every declared `model` (catalog, allowlist pin, auth) before any child spawns |
| Daemon lane | `PAD/factory_activity.rs:1-60` | `factory_activity` capability advertised only while `factory.enabled` (`:19-41`) |
| Library | `F:4141-4846`, `MX/*/MACHINE.md` | `MACHINE.md` = frontmatter + one fenced `machine-spec`; two levels (bundled or `PRIME_AGENT_MACHINES_DIR`, then personal), earlier wins (`F:5236-5251`) |
| Skill | `PA/skills/factory/SKILL.md:1-57` | points at `rlm.factory.help()`; opt-in gate (`/factory on`) |

### 1.2 Machine schema (validated at write time, `validate_factory_machine` `F:489-685`)

| Element | Fields and rules | Lines |
|---|---|---|
| `run` | `budget_ms` (>0), `failure_policy` in `fail_fast/continue/escalate` (default `escalate`), `max_parallel` 1..64 (default 8, run-wide concurrency), `max_transitions` <= 10,000 (default 10 x states), `max_children` <= 1,000,000 (default 10,000 total admissions) | `F:56-77`, `:222-255`, `:838-851` |
| state | `id` slug `^[a-z0-9][a-z0-9-]{0,63}$`, unique, 1..1024 states; `entry: bool` (>= 1 entry; entry states have no inputs); `subagent` = harness subagent id/title or inline `{prompt, name?, model?, thinking?}` (name <= 64, unique, must not collide with another state's `-i<n>/-a<n>` labels); `max_entries` >= 1 (default 1); `retries` 0..10; `budget_ms` <= run budget; `failure_policy`; `lifecycle` `task/resident`; `wait` rejected (not built) | `F:258-321`, `:521-616`, `:1148-1214` |
| ports | `inputs[{name, type text/json, from "<state>.<output>", optional?}]`, `outputs[{name, type}]`; types must match; no reading from a resident; a required self-input is rejected, an optional self-input is the loop form | `F:328-406`, `:602-616` |
| `foreach` | `{over: <json input>, max 1..256}`: one child per list item; empty list settles with no children | `F:408-430`, `:2734-2757` |
| resident | no outputs, foreach, retries or outgoing transitions; nothing reads it; stays alive after the run | `F:318-332`, `:391-393`, `:679-680` |
| transition | `{from, to, on: "settled", when?}`; `from` may be a list = join (fires once per source-settle combination; no guard); self-loops and cycles legal | `F:624-685`, `:5042-5062` |
| guard | `{output, path?, op, value?}`, ops `eq ne gt gte lt lte exists contains`; `path` needs a json port; `eq/ne` scalar and JSON-strict (bool != number); comparisons numeric; `contains` non-empty list; finite JSON, bounded depth | `F:438-486`, `:1274-1347` |
| dag sugar | `nodes` with `depends_on`/`inputs`; each node -> state entered once; full dependency set -> ONE join | `F:706-798` |

### 1.3 The run loop (`_launch_run` `F:1610-1641`, `_loop_body` `F:3573-3695`)

1. `run(spec_id)` re-validates, canonicalises, resolves every subagent, reporting all failures in one `ValueError` and
   starting nothing (`F:1556-1584`); `run_machine` does the same for a library template without a harness entry
   (`F:1586-1608`). Entry states are entered, instances admitted up to `max_parallel`, and the call **returns at once**;
   a background asyncio task continues (`F:1622-1641`, `:3545-3552`).
2. Each loop pass: `rlm.collect(in_flight, timeout 2 s)` (`F:3592-3594`, `POLL_TIMEOUT_MS` `F:1042`); apply settlements;
   evaluate queued settles; finalise on quiescence; check run budget; admit ready instances; fail loudly on dead
   configurations (`F:3637-3691`). A `loop_generation` counter kills a stale loop after `resume` (`F:1493-1497`, `:3605`).
3. **Settlement** (`_apply_settlement` `F:2871-2988`): infra recheck/respawn first (`F:2876-2895`); child error, cancel or
   odd status -> provisional exit answer kept, `retries` path (`F:2907-2924`); state `budget_ms` exceeded -> failure with
   no retry (`F:2925-2939`); else capture the full answer (cap 8,192 chars, `ANSWER_BINDING_CAP` `F:1018-1031`; the host
   carries up to 64 KiB `answer_text`) and, when every instance of the entry is done, capture ports (`F:3093-3121`):
   a `text` port takes the joined answer, a `json` port the last fenced ```json block whose object holds the port name
   (`F:1217-1246`); foreach instances are joined with blank lines (`F:3086-3091`).
4. **Evaluation** (`_evaluate_settles` `F:2408-2514`): every settle evaluated once; **every passing guard fires**
   (fan-out legal); a join fires once all sources have a settle and once per combination (`F:2433-2461`); a fire onto a
   state at `max_entries` is `transition_blocked` (`F:2469-2483`); hitting `max_transitions` pauses mid-settle and records
   the transition index so resume does not re-fire (`F:2484-2496`).
5. **Binding** (`_prepare_entry` `F:2647-2757`): a required input over an unsettled source waits; over a settled source
   with no value it fails the entry (binding failures never retry, `F:2654-2656`); an optional input binds `null`/`None`,
   except that an optional input over a *different* live state waits (`F:2687-2702`). Values render into `{name}` in one
   pass; unplaced inputs append under `## Inputs` (`F:1249-1271`).
6. **Admission** (`_admit` `F:2774-2867`): child label `<run6>-<name>[-i<n>][-a<n>]` (`F:1148-1193`); rate limits defer
   with a run-level backoff deadline `1 s * 2^k` capped 60 s, 5 strikes fail the node (`F:1073-1087`, `:2807-2840`); a
   spawn that lands after `stop()` is retracted (`F:2845-2851`).
7. **Finalise** (`F:3485-3541`): quiescence = no queued settle, no live entry, no pending instance (running residents
   excepted); any errored entry -> `failed`, else `done`. Dead configurations end `failed` with the reason (`F:3637-3691`).

### 1.4 How child outputs feed successors, and loop-back with findings

- A successor's prompt is the template with `{port}` values substituted from **the source state's latest settle**
  (`F:2661-2739`, `latest_settle` `F:1446-1450`). There is no message passing between states inside a machine: the
  rendered prompt is the only channel, so **isolation is by prompt content** (no staging, no canary in `F`).
- **Loop-back = back edge + typed port.** pr-manager: `reviewing.verdict` -> `fixing` prompt `{verdict}`;
  `fixing.fix_report` -> `reviewing` optional input that binds null on round 1 and the real report later
  (`MX/pr-manager/MACHINE.md:57-106`). Findings travel only through ports; **retries re-run the identical rendered
  prompt** (`F:1362`, `:3252-3267`), so a retry never sees why the attempt failed.

### 1.5 Failure, retry, resume, escalate (current semantics)

| Event | What happens | Lines |
|---|---|---|
| child error / exit | provisional answer kept; retry while `attempt - retry_base <= retries`; then policy | `F:2907-2924`, `:3224-3274` |
| worker death (infra) | quiet period, re-read; still dead -> delete zombie, respawn same prompt once, free; 2nd death -> normal path | `F:2876-2895`, `:2990-3084` |
| output not bound | one bounded re-capture; then producer task failure, answer provisional | `F:2969-2988`, `:3123-3222` |
| state `budget_ms` exceeded | failure, no retry | `F:2925-2939` |
| `fail_fast` | cancel every child, run `failed` | `F:3351-3368` |
| `continue` | entry `error`; its error settle fires guard-less transitions; required inputs over it fail dependents; optional bind null; run ends `failed` | `F:3339-3346`, `:3369-3370` |
| `escalate` (default) | run `paused` before the failure propagates; status carries `pause_reason`, `last_error`, `remedy`; `resume` re-admits the held entry (fresh retries, fresh free respawn, no extra entry); a binding failure over a broken upstream re-pauses | `F:3371-3380`, `:1682-1690`, `:1717-1806` |
| run `budget_ms`, `max_children`, `max_transitions` | each pauses **once**; resume continues past it as an operator decision; in-flight children keep running | `F:2569-2614`, `:2484-2496` |
| `needs_verify` | entry failed but kept a provisional answer (exit text, capture failure, last words after respawn); surfaced in `status()` | `F:1668-1678`, `:3286-3315` |
| loop exhaustion | `transition_blocked` event only; run ends `done` | `F:2469-2483`, `:3531`; `TF:5450-5477` |

### 1.6 Budgets, persistence, monitoring

- No token or dollar budget in the factory; only wall clock, admissions, transitions, concurrency (`F:5071-5098`).
  Per-child token usage is attributed to the parent transcript (d592d5e77), so cost is recoverable outside the factory.
- **Run state is kernel memory only**: "a kernel restart loses `status` for old runs, but the children keep running"
  (`F:5270-5273`). Specs persist in the harness JSON; sessions persist as JSONL under the daemon (`PA/AGENTS.md`,
  "Reliability model").
- `status()` returns per-state rows, the last 200 ledger events (`EVENT_WINDOW` `F:1033-1040`), usage, `needs_verify`
  (`F:1643-1692`); `graph()` fuses structure and live overlay; `watch(run_id, <=60 s)` blocks until the state/instance
  shape changes (`F:2107-2156`, `:5190-5218`). Milestones (paused, budget, failed, finished) are pushed once per kind as
  quiet notices into the parent conversation (`F:2367-2389`), so the root does not poll.

### 1.7 Authoring guide, the rules that matter (quoted from `FACTORY_HELP`)

- "Use it when a workflow needs shape: fan-out, bounded loops (review/fix until a verdict approves), joins, or one child
  per list item." (`F:4898-4900`)
- "Both worked examples bound their emitted payloads in the prompt ... an unbounded payload truncates at the cap, fails
  to capture, and fails the producing state" (`F:4994-4999`).
- "A declared port that binds no value ... is retried once and then FAILS the producing entry as a task failure ... so a
  valueless settle never reaches a dependent's bind." (`F:5031-5036`)
- "`escalate` (the default: pause the run BEFORE the failed settle propagates ... and `resume` re-admits the failed entry
  as a fresh attempt" (`F:5088-5093`).
- "A child whose worker died or whose supervisor link was lost is infra, not the task's failure ... re-spawns the same
  prompt once without spending `retries` or `max_entries`" (`F:5093-5098`).
- "Retry re-renders the same prompt into a changed world ... reconcile (start from what the earlier attempt left with
  `git status` and `git diff`, keep the changes that already serve the task, revert half-finished edits) or reset to a
  clean baseline, never fail on the old state or duplicate finished work. Retries with no recovery note still validate:
  validation cannot judge prose." (`F:5099-5107`)
- "Dead configurations fail loudly, never wedge" (`F:5108-5113`); "Bound loops with `max_entries`, `max_transitions`,
  and `run.max_children`" (`F:5261-5264`); keep outputs compact and "let the full answer live in the child's session"
  (`F:5265-5269`).

## 2. The machine library (all bundled seeds)

| Machine | Form | Graph | Pattern | Notes |
|---|---|---|---|---|
| `builder` (`MX/builder/MACHINE.md`) | dag | `scope` (outline json, <=5 lines) -> `notes` foreach(outline, max 8) text `section` -> `collect` text `merged` | **fan-out / fan-in** by foreach text join | no guards, no loops (`:17-19`). Defect a lint could catch: `collect` asks for "open questions" but binds only `sections` (`:79-87`) |
| `pr-manager` (`MX/pr-manager/MACHINE.md`) | machine | `entry` -> `reviewing` (max 4); `reviewing -> fixing` when `verdict.approved == false`; `reviewing -> monitoring` (resident) when `== true`; `fixing (max 3) -> reviewing`; run `max_transitions 24`, `budget_ms 1.8M` | **review/fix loop until approve**, resident watcher | reviewer prompt: "When a fix report is bound below, verify the described fixes landed" (`:60`); fixer reconciles worktree (`:88`); exhaustion ends `done` unapproved (section 0.2) |
| `review-sweep` (`MX/review-sweep/MACHINE.md`) | dag | `files` (json list) -> `review` foreach(files, max 256) one line per file -> `report` (json issues + clean count), which also reads `files.files` | **sweep**: map over a list, reduce | `max_parallel 8` bounds concurrency, `max 256` only pathological diffs (`:17-21`) |

All three use one model for every state (no `model` field), and none has a deterministic gate state: every check is
an LLM child's judgement.

## 3. Subagents and the swarm

### 3.1 Primitives (kernel API, `RLM`, surface pinned by `PA/AGENTS.md` "Surface contract")

| Call | Semantics | Lines |
|---|---|---|
| `rlm.spawn(prompt, *, name, model?, thinking?)` | child one level deeper; returns a handle (`rlm_child_id, name, session_dir, model`) once admitted, never the answer; name unique among siblings; the `kind="decision"` argument was removed with the Decision API (cfed5a777) | `RLM:179-202`; `LAY/core.md:33` |
| `rlm.collect(targets?, timeout_ms=0)` | typed snapshots (`status queued/running/done/error/cancelled`, `settled`, `answer_text`, `error`, `tool_use_count`); never steers; settled children keep results until deleted | `RLM:409-443` |
| `rlm.list_subagents()` | direct children with activity, progress note, staleness; the recovery path after compaction or kernel restart | `RLM:339-345`; `LAY/usage.md:31` |
| `rlm.delete_subagent`, `rlm.rename` | retire / rename a child | `RLM:478-519` |
| `rlm.progress_note(msg)` | <= 512 UTF-16 units, ~1 per 10 s, reaches the parent's roster without steering it | `RLM:446-475` |
| `rlm.create_session(prompt, cwd=...)` | new **depth-0** daemon session (a peer root, with its own `cwd`); depth-0 daemon sessions only | `RLM:217-241` |
| `agent_message.send(msg, receiver_role, receiver_name)` | parent / sibling / child only ("Roots are siblings"); wakes an idle child for a follow-up turn in the same context; `"all"` broadcasts; receipt `delivered` or `queued` | `PA/skills/agent-message/SKILL.md:8-48` |
| `agent_observe.list_agents/get_agent/recent_messages` | roster and transcript preview (`limit` 1-50, `max_chars` 80-2000) | `LAY/core.md:69-71` |

- **Messaging limits** (`PAC/session_engine/agent_messaging.rs:16-22`): 16,384 chars per message, 20 pending per
  session, per-sender token bucket of 3 with one refill per 1,000 ms. Delivered as `[agent-message from <role>:<name>]`
  (`:254-266`). Sender identity is daemon-derived, not spoofable (`PA/skills/agent-message/SKILL.md:58`). Queued messages
  now survive every abort shape and cleared ones bounce back to the sender (7d191b47e).
- **Depth.** Default `DEFAULT_RLM_MAX_DEPTH = 2` (`PAC/session_engine/rlm_host.rs:265`); precedence chat override >
  inherited > global setting > `RLM_MAX_DEPTH` env > default (`:357-382`); refusal text at `:269-273`. The rewrite ran
  to depth 3 (`RUST:17`). A daemon started inside an agent shell no longer inherits a stale `RLM_DEPTH` (e090862aa).
- **Kernel model.** One tool, `python_repl`, a persistent CPython REPL; everything else is a programmatic tool called
  from Python (`LAY/core.md:3`). Each subagent is "persistent, just like the root session" (`LAY/core.md:29`); the
  rewrite ran 100+ concurrent subagents and their kernels per 8-core node (`RUST:56`). "Python is the orchestration
  language" (`LAY/usage.md:19`). The research RLM (verifiers) differs: sub-LLMs via `llm_batch`, tools only for
  sub-LLMs, answer via a variable, recursion depth exactly 1 (`RLMB:39-58`, `:602`).
- **Print mode now has inline (daemonless) subagents** and the run waits for RLM quiescence before the final result
  (9eec445aa). This matters for batch pipelines: a headless `prime-agent -p` process can host a whole tree.

### 3.2 How the root orchestrates without writing code

- The prompt makes delegation the default: "When delegation is available and useful, an agent assigns independent
  substantive tasks to separate workers ... let them run in parallel" (`LAY/usage.md:15`); "Delegate parallel
  context-heavy research or independent implementation; do a single known lookup, edit, or command inline"
  (`LAY/usage.md:33`); "Have children write files and read those files for fan-in" (`LAY/usage.md:32`); no polling
  with sleep, end the turn and wait for notifications (`LAY/usage.md:16`).
- The rewrite's root "divided the work into a topological ordering of tasks ... wrote no product code, keeping it free
  to monitor every task, merge finished work, and maintain an overview" (`RUST:62`). In code the root's tools are
  `rlm.factory.run/watch/status/resume`, the milestone notices (`F:2367-2389`) and `rlm.collect`; nothing in code
  forbids the root from editing.
- **Persistent child = reusable context.** A settled child can be asked a follow-up by message and answers in the same
  context (`PA/skills/agent-message/SKILL.md:33-36`); residents keep a factory child alive after its run (`F:5067-5070`).

### 3.3 Context management for long runs

- Auto-compaction threshold is now configurable: `threshold_percent` (default 95) floored by
  `keep_recent_tokens + 2 * reserve_tokens + 1` so a low percentage cannot re-summarise every turn, capped by the
  combined input+output ceiling (`PAC/session_engine/compaction.rs:14-45`, `:152-185`; a498cbd49). Compaction now also
  fires at tool-turn thresholds inside one run (6465da8ed).
- The kernel survives compaction; only variables > 16 MiB serialised are dropped (`LAY/core.md:139`). Agents can
  `compact.run(instructions)` at a boundary (`PA/skills/compact/SKILL.md:14-41`).
- **Goals**: continuation prompt demands an audit "against every requirement ... Do not rely on intent, partial
  progress, memory of earlier work, or a plausible final answer" (`PAC/goals.rs:368`). No-progress streak cap 3 now
  uses a doubling backoff from 10 s with a durable wake cron before the goal ends (`PAC/session_engine/goal_driver/progress.rs:11-29`).
- **Autonomous mode** (missed by study 01): shell-command **gates** run after each turn; defaults 12 turns, 80k tokens,
  30 min, gate retries 3, gate timeout 5 min (`PAC/autonomous/mod.rs:20-27`); continuation text "the verifier/evaluator
  decides completion when configured gates pass" (`:20`); a failed gate is **not re-run over an unchanged workspace**
  (git status + diff + untracked hash snapshot, `PAC/autonomous/gates.rs:1-2`, `:153-176`, `mod.rs:155-160`).

### 3.4 Worktrees and sandboxes

- No worktree, sandbox or `cwd` field on `rlm.spawn` or on a factory state (`RLM:179-202`, `F:258-321`). A child shares
  its parent's filesystem view unless its prompt runs `git worktree add`; only `create_session(cwd=...)` sets a working
  directory, and only from depth 0 (`RLM:217-241`). The respawn code names the risk: delete the old worker "so the
  respawn cannot run beside its zombie in the same worktree" (`F:3036-3042`).
- Child session dirs live under the parent's artifact tree (`PAC/session_engine/rlm_host.rs:298`). Workspace
  snapshots (content-addressed, manifest-verified, credential-name denylist) exist as plumbing, "nothing wires it to a
  user-facing toggle yet" (`PAC/workspace_snapshot/mod.rs:1-24`).
- "Fresh Prime Sandbox" verifiers and "dedicated worktree" implementers are process description (`RUST:56`, `:66-70`).

## 4. Verification inside the loop, and how Prime verifies the factory

### 4.1 Verdict gates

- The gate is a guard over a json port: `{"output":"verdict","path":"approved","op":"eq","value":true}`
  (`MX/pr-manager/MACHINE.md:139-147`). Strictness: missing/unparseable port fails every op except `exists`
  (`F:1318-1322`); bool never equals number (`F:1274-1289`). The executor runs no check itself and trusts the child.
- Anti-trust signals in code: provisional exits and capture failures mark `needs_verify` (`F:3286-3315`); a capture
  failure is now a task failure (section 0.2). Different reviewer model is a per-state `model` field; nothing enforces
  it (`F:295-298`, `:2250`); the host only checks the model exists, is allowed and authenticated (`factory_host.rs:250-287`).
- The deterministic gate in Prime is outside the factory: autonomous gates (3.3). The blog's oracle was human-built
  parity suites: TUI frame diffs against the TS binary under a scripted model, transcript and provider-request diffs,
  protocol type parity, and an agent audit classing features matching/partial/missing (`RUST:44-54`).

### 4.2 The `review_factory_*_parity.py` scripts (how they verify the factory itself)

| Script | What it proves | Method |
|---|---|---|
| `SC/review_factory_validation_parity.py` | a real harness write through the kernel rejects `retries: 1` on a resident ("resident node watcher cannot declare retries"), stores `retries: 0`, rejects `'bad'` | kernel cell executed by a real daemon; asserts on stored/not stored (`:24-36`, `:97-104`) |
| `SC/review_factory_resumption_parity.py` | escalate -> resume re-admits the same entry: child 1 returns no JSON (capture failure -> pause), resume, child 2 returns `{"data":{"ok":true}}`, guard fires `verified`; asserts `work.entries_used == 1, attempts == 2`, `verified.attempts == 1`, exactly 3 child requests | scripted local provider routes parent vs child by marker strings (`:24-36`, `:55-85`, `:103-112`, `:176`) |
| `SC/review_factory_navigation_parity.py` | TUI factory page Left/Right keys over a fixture run, raw PTY evidence | protocol fixture injects a graph (`:1-30`) |

Common receipt discipline: Linux-only, the TS archive SHA-256 pinned, both binaries hashed, a `scope` string and an
explicit `not_verified` list (e.g. "worker-death respawn (separate executor regression)", `review_factory_resumption_parity.py:192-194`);
"TS has no advertised factory lane" is asserted, not assumed (`:144`). Each script has a `test_*` twin that mutates
the probe (`entries_used 2`, `attempts 1/3`, `status error`, duplicate or historical tool receipt) and requires the
validator to reject it (`SC/test_review_factory_resumption_parity.py:13-41`): **the checker is itself mutation-tested**.
The repo rule: "Verifiers over self-assessment ... No lane merges without its verifier passing, rerun by the reviewer
where feasible" (`PA/AGENTS.md`, "Tests").

## 5. Continual harness and self-improvement

- Five editable kinds: prompt, memory, skill, subagent, **factory** (`PAC/refinement/mod.rs:10`). The refiner model may
  create/update/delete factory specs (`PAC/refinement/planner.rs:10`, "factory: declarative state-machine workflow
  specs"); its JSON carries `rationale` and `expectedOutcome`, and it may roll back earlier faulty edits (`planner.rs:10`,
  `rollback_of` `planner.rs:293`).
- Auto-refine: every 25 turns, 20-minute cooldown, behind an LLM **review gate** that must approve
  (`PAC/session_engine/refine.rs:34-45`, `planner.rs:12`); local scope by default, global only for durable lessons
  (`planner.rs:10`).
- Guards against bad persisted edits (no exploit detection exists): (a) every factory write, including a refinement
  edit, runs the full validator and the `factory.enabled` gate (`harness.py:360-377`, `PAC/refinement/executor.rs:282-294`);
  (b) a new global master switch `harnessRefinements` that refuses every harness read/write in the kernel
  (`harness.py:136-145`, `:438`, `:459`, `:541`, `:628`, `:1350`) and strips harness digest and refinement rows from model context and
  compaction input (`PAC/session_engine/refinement_gate.rs:1-41`, `compaction_arms.rs:266-268`; f0c2d328a);
  (c) versioned entries and refinement history. `expectedOutcome` is still never measured. The documented exploit
  (an agent preserving a Factorio RCON shortcut as a skill) remains unaddressed in code (study 01 §2.10).

## 6. Mapping to Crucible's long-horizon coding task pipeline

### 6.1 What Crucible already has (`CR/factory/`), and where `machine.py` diverged from Prime

| Aspect | Prime `F` @ 7cbffa77b | Crucible `CR/factory/machine.py` | Verdict |
|---|---|---|---|
| Tokens | many: fan-out, joins, foreach, residents | one token; joins/foreach refused; first passing guard fires (`:25-27`) | keep; parallelism in state functions and `root.py` |
| Exhaustion | `done` silently (`F:3531`) | `exhausted` escalation with decision card (`:1118-1123`) | Crucible right |
| Escalate resume | re-admits held entry, fresh retries (new) | `resume_entry` re-runs same entry, fresh retries (`:959`, `:1195`, `:1132`) | equal; Crucible had it first |
| Infra vs task failure | verified death -> one free respawn (new) | every exception is an attempt (`:1160`); only budget/credit errors stop (`:1167-1170`) | **PORT** the split |
| Capture failure | producer task failure (new) | `_check_outputs` fails the attempt on missing/undeclared/oversize ports (`:1038-1059`) | equal |
| Retry recovery posture | prose rule, unenforced (`F:5099-5107`) | none | **PATTERN, enforced**: `mutates: true` states must declare `recovery: reconcile|reset`; reset = restore workspace snapshot |
| Persistence | kernel memory | journal + replay + torn tail (`:806-906`) | Crucible right |
| Budgets | ms, admissions, transitions | USD ledger reserve/settle, seconds, transitions (`:982-987`, `:1142-1166`) | Crucible right |
| Reviewer model | free field | `different_family_from` refused by validator (`:18-19`) | Crucible right |
| Isolation | prompt content only | `isolated_from` + canary + shingle screen (`:20-24`, `:1003-1030`) | Crucible right |
| Guards | leaf only | leaf + `all/any/not` + expression parser (`:39-42`, `:160-293`) | Crucible superset |
| Roles | full agent sessions (kernel, tools, files) | in-process handlers, mostly single LLM calls (`CR/factory/driver.py:1-25`) | **gap** for agentic world-building |
| Stale PA line refs | — | comments cite old lines (e.g. `PA:3248-3277` is now `F:3515-3541`, guards `PA:1270-1325` now `F:1292-1347`) | refresh refs |

Crucible also already has the per-task machine (`CR/factory/driver.py`: spec -> author -> realism -> oracle -> verifier ->
ledger -> floors -> probes -> hackers -> second_verifier -> solvers -> differential -> audit -> flake -> horizon ->
calibrate -> gate -> select), the root curator with leases, yield/diversity stops and persistent specialists
(`CR/factory/root.py:1-21`), the long-horizon window lanes (`CR/factory/lanes/windows*.py`), and the M16 RLM tree
(`CR/actors/rlm/tree.py`).

### 6.2 Role-by-role use of each Prime mechanism

| Role | Prime mechanism to use | How, in Crucible |
|---|---|---|
| Root curator | root that writes no code; milestone notices; `watch` | `root.py` already; add a `watch`-style blocking status and push notices on pause/exhaustion instead of polling |
| Planner (spec + milestone plan + verifier plan before code) | Planner writes "machine specification ... ground-truth ... and verifier" first (`RUST:64`) | `spec` state emits intent, horizon target, milestone DAG and the verifier plan as json ports; later states are checked against it (ledger lint) |
| World-builder (repo window at A, no network) | persistent agentic child with its own workspace; recovery posture | make it an agentic state (RLM child with tools) in its own git worktree; `mutates: true, recovery: reset` (snapshot restore) so a retry starts clean |
| Verifier-writer (hidden tests) | separate context; loop-back with findings through a port | `isolated_from: [oracle]` already; findings from probes/hackers come back via `carry` |
| Oracle-builder (reference at B) | Implementer in a dedicated worktree (`RUST:66`) | own worktree, canary on outputs; `recovery: reconcile` (keep commits that pass milestones) |
| Reviewer | different model, adversarial, separate context (`RUST:68`) | `different_family_from: [author, verifier]` (validator-enforced) |
| Hacker (N in parallel) | `foreach` fan-out + text join (review-sweep shape) | inside the `hackers` state function (one token), or add `foreach` + join to the machine if per-hacker retries/needs_verify are wanted |
| Solver ladder | sweep over models | same as hackers; results reduced by a deterministic state, never an LLM aggregator |
| Auditor / gate | Prime has none in-factory; autonomous gates are the deterministic analogue | gate is `accept_task` (deterministic); add "do not re-run a failed gate over an unchanged workspace" (`PAC/autonomous/gates.rs:153-176`) as a cache key on the workspace hash |
| Family specialist | persistent child answered by message; residents | `root.Specialists` already; messages screened by canary/zone like carried findings |

### 6.3 PORT vs PATTERN (licence MIT, credit in `THIRD_PARTY.md`)

- **PORT** (small, pure, already partly ported): the infra-error classifier and verified-respawn rule (`F:1095-1116`,
  `:2990-3084`) adapted to our runner errors (Harbor/docker container death, SSH drop, provider 5xx are infra; test
  failure, bad JSON, timeout are task); `_suffixed_spawn_form` label-collision check (`F:1196-1214`) if states get named
  children; `MACHINE.md` parse/render/list/import/export (`F:4141-4846`) as a machine library for task families.
- **PATTERN**: hosted receipts with a `not_verified` list and mutation-tested receipt checkers (section 4.2) for every
  pipeline milestone (e.g. "escalate resume re-runs one entry", "exhaustion pauses", "isolated state never sees the
  oracle canary") run against the real driver with scripted LLM clients; the graph/watch live view; quiet milestone
  notices; the authoring guide as a `help()` string that is the single source of truth.
- **Do not copy**: silent exhaustion (`F:3531`), kernel-memory run state (`F:23-26`), prompt-only isolation, unenforced
  reviewer model, the refiner's ability to rewrite factory specs without a measured A/B.

### 6.4 Design rules the code implies for the pipeline

1. A check state's verdict must be deterministic where possible; an LLM verdict only routes, and the final accept is
   `accept_task` (Prime trusts the child; Crucible must not).
2. Every state that mutates a workspace declares how a retry recovers, and the validator refuses one that does not;
   Prime admits "validation cannot judge prose" (`F:5107`), so make it a field, not prose.
3. Separate infra failure from task failure in the ledger and in retries; one free verified respawn, never a free
   retry on a task failure (otherwise solver/hacker pass rates are biased by flaky hosts).
4. Bound every payload that crosses a port in the producing prompt and keep full artefacts on disk (`F:4994-4999`,
   `LAY/usage.md:32`); for long-horizon tasks that means World/oracle/tests as files, ports carry paths + hashes.
5. Fan-in by files, not by LLM merging: review-sweep's LLM `report` is acceptable for advice, not for gates.

## 7. What study 01 got wrong or missed

- It said the factory fires `escalate` resume "continues the remaining states"; now the failed entry is held and
  re-admitted (section 0.2). It cited `_finalize` at `:3248-3277` and guards at `:1270-1325`; now `F:3515-3541`,
  `F:1292-1347`.
- Missed: autonomous gates with workspace-unchanged suppression (`PAC/autonomous/`); the `factory_host.rs` run
  preflight of every declared model (`:250-287`); print-mode inline subagents; the `harnessRefinements` kill switch;
  the configurable compaction threshold with a retention floor; the receipt-based verification of the factory and its
  mutation-tested checkers; that refinement may edit `factory` specs (`PAC/refinement/mod.rs:10`).
- It reported 3 no-progress continuations end a goal; the cap is still 3 but now with doubling backoff and a wake cron.
- New defect noted: the pr-manager seed's "pauses (escalates) instead of spinning" text is false for loop exhaustion.

## 8. Not verified

- `PAD/supervisor/routing.rs` dead-child delete internals (d356a1c69) read by commit summary only.
- Whether the Rust TUI factory page shows `transition_blocked`; not read.
- The blog's numbers (2,209 agents, 16,758 messages, 228.70B tokens, `RUST:28`) are not reproducible from code.
- `PAC/refinement/executor.rs` apply logic read only at `:280-294`.
