# Task factory, study 01: Prime Agent's factory, the Rust-rewrite process and GitSwarm, mapped to Crucible's forge

- **Question.** What can a swarm-based RL task factory (prompt + World + verifier + intent, many tasks, each verifier
  rigorously validated) reuse from Prime Agent's orchestration, and what does Crucible already have?
- **Read:** 2026-10-14, local checkouts, against Crucible HEAD 1f89e2a. Every claim cites `file:line`. Prime Agent is MIT.
- **Path keys.** `PA/` = `RL-Environments/06-frameworks-and-env-code/primeintellect-ai-prime-agent/`.
  `PAR/` = `PA/prime-agent-runtime/src/rlm/`. `PAC/` = `PA/crates/pa-core/src/`. `ART/` = `RL-Environments/07-articles/`.
  `RUST` = `ART/prime-agent/prime-agent-rust.txt`, `PAP` = `ART/prime-agent/2608.23552.txt`,
  `CH` = `ART/prime-agent/2605.09998.txt`, `GS` = `ART/gitswarm-2610.04862.txt`, `SWARM` = `ART/pi-on-the-nature-of-the-swarm.md`.
  Crucible paths are relative to `crucible/src/crucible/`.

## 0. Bottom line

1. Prime Agent's factory executor is a **generic guarded state machine over spawned child sessions**. It has no verifier,
   sandbox, worktree or reviewer concept in code. The Planner / Implementer / Reviewer / Verifier machine is described in
   the Rust blog (`RUST:62-70`), and **not shipped**: the bundled machines are `builder`, `pr-manager`, `review-sweep`
   (`PAR/machines/`). A verifier "gates" a transition only because a child prints a JSON verdict and a guard reads it.
2. Crucible already has more of the factory than Prime Agent ships: the DAG orchestrator with staging, canaries, journal
   and budget (`forge/orchestrator.py`), GitSwarm-style swarm stages (`forge/swarm.py`), zoned repo, dedup, ladder, curator,
   economy, judge, red-team (`forge/*`), acceptance A1-A18 (`accept/checks/*`), a Planner/Implementer/Reviewer/Verifier
   FSM for repairing tasks (`accept/repair.py:459`) and measured harness edits (`actors/rlm/harness_state.py:169`).
3. The gap is **integration and a declarative machine**: `SwarmStage`, `YieldBook`, `Judge`, `SpecialistHacker`, `TaskFSM`
   and `RLMTaskManager` are exercised by unit tests only (grep over `src packs scripts`: no driver builds them). No
   executable path runs seed -> swarm -> accept -> ladder -> curate -> retire. Section 5 ranks what is missing.
4. Reuse verdict: PORT (MIT) the machine validator, guard evaluator and canonicalizer from `PAR/factory.py`; PATTERN the
   rest. Do not DEPEND on `factory.py` (5k lines, kernel-memory state, supervisor-coupled).

## 1. How Prime Agent's FSM factory works, from the code

### 1.1 A machine is data: states, ports, guarded transitions

`PAR/factory.py` (4,975 lines) validates, canonicalizes and runs a declarative machine. Module doc: `factory.py:1-44`.
A spec lives in a continual-harness entry of kind `factory` (`PAR/harness.py:26-33`, validated at write time, an invalid
spec is never stored) or in a `MACHINE.md` file with one fenced `machine-spec` block (`factory.py:3904-4195`).

The machine JSON (documented `factory.py:4619-4975`, `FACTORY_HELP`):

```json
{"run": {"budget_ms": 1800000, "max_parallel": 8, "max_transitions": 24, "failure_policy": "escalate", "max_children": 10000},
 "states": [{"id": "reviewing", "subagent": {"prompt": "...{pr_url}...", "model": "...", "thinking": "..."},
             "inputs":  [{"name": "pr_url", "type": "json", "from": "entry.pr_url"},
                         {"name": "fix_report", "type": "json", "from": "fixing.fix_report", "optional": true}],
             "outputs": [{"name": "verdict", "type": "json"}],
             "max_entries": 4, "retries": 0, "budget_ms": 240000, "failure_policy": "escalate", "lifecycle": "task"}],
 "transitions": [{"from": "reviewing", "to": "fixing",
                  "when": {"output": "verdict", "path": "approved", "op": "eq", "value": false}},
                 {"from": ["a", "b"], "to": "join"}]}
```

Rules enforced at write time (`validate_factory_machine`, `factory.py:487-688`; `_validate_state_fields`, `:258-433`):

- 1..1024 states (`MAX_NODES`, `:61`), slug ids `^[a-z0-9][a-z0-9-]{0,63}$`, at least one `entry: true` state, entry
  states declare no inputs. **Cycles are legal** (no acyclicity rule, `:498-500`).
- Ports are typed `text` or `json`. An input binds `from: "<state>.<output>"`, types must match, nothing reads a resident.
  A required self-input is rejected (it can never bind); an **optional** self-input is the loop form: the first entry binds
  a null sentinel, a re-entry binds the previous settle (`:333-346`).
- A guard is `{output, path?, op, value}` over **the from-state's latest settle**, ops `eq ne gt gte lt lte exists contains`
  (`GUARD_OPS`, `:60`; `_guard_passes`, `:1270-1325`). A missing or unparseable port fails every op except `exists`;
  `eq` compares JSON-strictly (bool never equals number). A join (`from` is a list) fires once all sources settled and
  **cannot carry a guard** (`:629-633`).
- `foreach {over, max<=256}` fans one entry into one child per item of a json input. `resident` states live under the parent
  after the run, declare no outputs or transitions, and nothing reads them.
- `DAG` form is sugar: it compiles to machine form, fan-in becomes ONE join transition (`compile_factory_dag`, `:704-801`).

Key runtime dataclasses (verbatim shapes, `factory.py:1335-1480`):

```python
class _NodeInstance:  index, prompt (rendered once, re-spawns reuse it), status (pending|running|done|error|cancelled),
                      attempt, rate_limit_streak, child_id, spawned_at, duration_ms, answer, provisional, error, tool_uses
class _StateEntry:    index, status, instances, answer, outputs, output_errors, is_settle, consumed, needs_verify
class _StateRun:      state_id, spec, prompt_template, name, model, thinking, max_entries, entries_used, entries, error
class FactoryRun:     run_id, spec_id, machine, state (running|stopping|paused|done|failed|stopped), max_parallel,
                      max_transitions, max_children, run_budget_ms, pause_reason, last_error, states, transitions_from,
                      pending_evaluations, events, spawn_count, settle_count, transitions_fired, join_fired, revision
```

An **entry** is one activation of a state; re-entry creates a fresh entry. An entry settles when all its instances settle;
the settle captures declared outputs from the child's final answer (a fenced ```json block), capped at 8,192 chars
(`ANSWER_BINDING_CAP`, `:1016`; a longer block fails to bind and the dependent entry fails).

### 1.2 Loop-back with findings, retries and limits

- **Loop-back is a back edge plus a typed port.** The shipped `pr-manager` (`factory.py:4640-4709`): `reviewing`
  outputs `verdict {approved, findings[<=3]}`; `reviewing -> fixing` when `approved == false`, `reviewing -> monitoring`
  when true, `fixing -> reviewing`. The fixer's prompt contains `{verdict}`; the reviewer's optional input
  `fix_report` re-binds on re-entry. `max_entries` bounds each state (reviewing 4, fixing 3).
- **Evaluation** (`_evaluate_settles`, `:2322-2430`): each settle is queued and evaluated exactly once; every passing guard
  fires (fan-out is legal); a fire onto a state at `max_entries` is only an event, `transition_blocked`
  (`:2381-2398`). `max_transitions` (default 10 per state, cap 10,000, `:70`) pauses the run once, mid-settle, and a resume
  continues after the transitions already fired.
- **Defect to avoid.** Loop exhaustion is silent success. If the last `reviewing` entry is not approved, its transition to
  `fixing` is blocked, no state errored, nothing is pending, and `_finalize` sets the run to `done` (`:3248-3277`) with no
  approval and no escalation. Crucible's `TaskFSM` already ends such a loop as `failed` ("escalate to a person",
  `accept/repair.py:517`); keep that.
- **Retries** (`retries` 0..10, `MAX_RETRIES :63`): a failed child is re-spawned with the **same rendered prompt**, attempt+1
  (`_apply_instance_failure`, `:2984-3027`). Findings never feed a retry; only a loop-back state carries findings.
- **Failure policy** per state or run (`FAILURE_POLICIES :56`, `_apply_entry_failure_policy :3029-3113`): `fail_fast`
  (cancel every child, run `failed`), `continue` (entry stays `error`, dependents with optional inputs proceed, run ends
  `failed`), `escalate` (default, run `paused` with `pause_reason`, `last_error` and a resume remedy).
- **Rate limits**: `BACKOFF_MAX_ATTEMPTS = 5`, delay `1s * 2^k` capped 60s, recorded as a run-level deadline so the control
  loop keeps collecting while admission waits (`_admit`, `:2688-2783`; `:1071-1087`).
- **Dead configurations fail loudly** (`_loop_body`, `:3380-3430`): nothing in flight and nothing pending, a pending entry
  whose input source never settled, or a `max_parallel` cap held by never-settling residents each end the run `failed`
  with the reason in the ledger.

### 1.3 Roles get separate contexts and models

- Every state spawns a **child session** through `rlm.spawn(prompt, name, model, thinking)` (`PAR/__init__.py:179-210`): own
  context, own IPython kernel, own history (`PAP:204-212`). The state's `subagent` is an inline `{prompt, name?, model?,
  thinking?}` or a harness `subagent` entry (`content` = prompt template, `metadata.model/thinking`; `_resolve_subagents`,
  `factory.py:2144-2186`). So per-role model choice is one field per state.
- Inputs are rendered into `{placeholders}` in one pass; inputs without a placeholder are appended under `## Inputs`.
  **Isolation is by what the prompt contains**, not by a staging layer: the factory has no "visible inputs" control and no
  canaries (Crucible's `Node.visible` plus canary breach is stricter, section 4).
- Child names are run-scoped (`<run6>-name`, `-i<n>` for re-entry/foreach, `-a<n>` for retries, `:1097-1190`). Names that
  could collide with suffixed forms are rejected at write time (`:535-587`).
- **Nothing in code enforces "reviewer uses a different model."** It is a convention in the blog (`RUST:68`): author sets
  `model` differently on the reviewer state.

### 1.4 How a verifier gates a transition, and fresh sandboxes

- In the executor a verifier is another state. The gate is a guard on its json output, e.g.
  `{"output":"verdict","path":"approved","op":"eq","value":true}`. The executor never runs a check itself; it trusts the
  child's final answer. A child that exits without settling is `provisional` and marks the state `needs_verify`
  (`factory.py:1349-1355`, `:2800-2860`), the only anti-trust mechanism in the executor.
- **Fresh sandbox per verification is not in `factory.py`**. The blog says the Verifier "compiles and runs the feature's
  parity checks and tests in a fresh Prime Sandbox" and the Implementer works "in a dedicated worktree" (`RUST:66-70`);
  there is no `worktree`/`sandbox` field in the machine schema (grep of `factory.py` and `PAR/__init__.py` finds none).
  Heavy work was outsourced via agent utilities, on two 8-core nodes with 100+ concurrent subagents each (`RUST:56`).
- The objective check was human-written per parity kind: TUI frame diffs against the TypeScript build under a scripted
  model, transcript and provider-request diffs, protocol message types, and agent-audited feature lists classed
  matching / partial / missing (`RUST:46-52`).

### 1.5 Root orchestrator: assign, monitor, merge; persistent subagents

- The root is an ordinary agent whose Python kernel calls `rlm.factory.run/status/stop/resume/graph/watch`
  (`PAR/__init__.py:576-636`) and `rlm.spawn/collect/list_subagents/delete_subagent`. "The root agent wrote no product
  code, keeping it free to monitor every task, merge finished work, and maintain an overview" (`RUST:62`). Merge is the
  Implementer's PR merging after review and verification pass; there is no merge logic in the executor.
- **Monitoring**: `run()` returns at once and a background control loop polls `collect(timeout 2s)` (`POLL_TIMEOUT_MS :1040`).
  `status()` returns per-state rows, the last 200 ledger events (`EVENT_WINDOW :1031`), usage, `needs_verify`,
  `pause_reason`. `watch(run_id, timeout<=60s)` blocks until the run's state/instance shape changes. Pause and failure
  notices arrive as quiet notices once per kind per run, with the resume call spelled out.
- **Persistent subagents.** A child is "a persistent concurrent session, not a stateless completion" (`PAP:964`).
  `rlm.spawn` returns a handle before the child finishes; `rlm.list_subagents` recovers handles; the parent follows up with
  `agent_message.send(msg, receiver_role="child", receiver_name=...)`; a child replies to `"parent"`. Daemon-mediated
  queues hold messages until the recipient is active (`PAP:253-256`). Limits (`PAC/session_engine/agent_messaging.rs:14-22`):
  16,384 chars per message, 20 pending per session, a per-sender token bucket of 3 with one refill per 1,000 ms. Messages
  arrive as `[agent-message from <role>:<name>]` prompts (`:254-268`). Residents in a machine stay alive after the run.
- Scale reported: 2,209 agents, 16,758 agent-to-agent messages, 228.70B tokens, depth <= 3 (`RUST:28`).

### 1.6 Budgets, persistence, resume

- Budgets: `run.budget_ms` (pauses once; in-flight children keep running, `factory.py:2483-2503`), `max_children`
  (total admissions, default 10,000, `:2505-2528`), `max_transitions`, `max_parallel` (1..64, default 8, concurrent
  instances run-wide), per state `budget_ms` (admission to settlement; exceeded = failed attempt, no retry), `retries`,
  `max_entries`. There is **no dollar or token budget in the factory**; token budgets live in goals (below).
- **Run state is kernel memory only** and does not survive a kernel restart (`factory.py:19-23, :1422-1424`); children keep
  running under the supervisor. `resume` only un-pauses a paused run (it raises on any other state). Spec and harness state
  persist (JSON file with a lock directory, `PAR/harness.py:507-523`); sessions persist on disk and clients reattach
  (`RUST` architecture section).
- **Goals** (persistent objective across turns): `GoalState {status idle|active|paused|budget_limited|complete|error,
  token_budget, tokens_used, time_used_seconds, continuations_used, no_progress_streak}`
  (`PA/crates/pa-types/src/goal.rs:39-64`). The continuation prompt tells the agent to "audit the current state against every
  requirement in the objective. Do not rely on intent, partial progress, memory of earlier work, or a plausible final answer
  as proof" before `goal.complete()` (`PAC/goals.rs:342`). Three consecutive no-output continuations end the goal
  (`PAC/session_engine/goal_driver/progress.rs:13`). Autonomous mode runs an end-condition test after each turn, bounded by
  turn, token and wall-clock limits (`PAP:311-320`).
- **Refinement** (Continual Harness runtime): `/refine` emits create/update/delete edits over five kinds (prompt, memory,
  skill, subagent, factory), applied at a turn boundary with trigger and `expectedOutcome` recorded, versioned, with
  `rollback_proposal` (`PAC/refinement/planner.rs:505`). Auto-refine gates: every 25 turns, 20-minute cooldown, an
  approving **review gate** before any refine runs (`PAC/session_engine/refine.rs:35-45`, prompt `planner.rs:12`).
  `expectedOutcome` is recorded and never checked (m14 study, section 7).

## 2. Process lessons that transfer to task curation

Numbers are from `RUST` unless stated.

1. **Objective checks before autonomy.** "The human work required was to setup proper verification to enable autonomous
   deployment at scale"; with an objective check per parity kind "the agents could measure their own progress and catch
   regressions before changes merged, which let us cut back on human review" (`RUST:46-54`). For a factory: the acceptance
   contract (A1-A18) must exist and be deterministic before any generator swarm runs.
2. **Generation separate from verification.** "An agent that writes code is biased when evaluating it" (`RUST:62`). Four
   agents per task: Planner (machine spec, ground truth, the verifier), Implementer (own worktree), Reviewer (adversarial,
   different model, separate context), Verifier (fresh sandbox). Failed review or verification returns to the Implementer
   with findings; merge needs both (`RUST:62-70`).
3. **A different-model reviewer.** The hillclimb used two reviewers, each on a different frontier model, checking
   behaviour parity, byte-identical output where claimed, and the model-facing surface (`RUST:118-120`).
4. **Differential and parity checks** as the oracle: terminal-frame diff under a scripted model, session-transcript plus
   provider-request diff, protocol-type parity, audited feature parity (`RUST:46-52`). The transfer: a task's verifier
   should be checked against an independent implementation (a second reference solution, or a second verifier) on the same
   states, not only against its own oracle.
5. **"The parity checks only verified the behavior they exercised"** (`RUST:90`). Coverage gaps were found by dogfooding:
   the team moved to the Rust build, "we had agents review logs and traces from all beta users, allowing us to
   automatically diagnose and resolve runtime and agent issues" (`RUST:90`). Transfer: a trace-review role that reads
   real solver trajectories on accepted tasks and files verifier-coverage defects.
6. **Hill-climbing with no numeric target** (`RUST:112-124`): a second orchestrator ran three days; per experiment, (a) profile
   and build a hypothesis backlog, (b) a worker builds baseline and candidate **on the same sandbox, in alternating order,
   with neighbouring benchmarks** to catch regressions, (c) two reviewers on different models, (d) merge and **re-baseline**.
   "We deliberately gave the loop no numeric targets, since a fixed threshold tends to become a stopping point." Output:
   144+ experiment and audit records, 69+ merged changes; examples 14.18x cold start (736.1 -> 51.9 ms), 4.76x memory.
   The benchmark ran on a fresh 4-core 8 GB sandbox per benchmark with **noise checks that withhold any result too unstable
   to compare** (`RUST:112`).
7. **Generation scale and cost**: ~2,000 agents, 10,000+ sandboxes, 200B+ tokens, two weeks (`RUST:13`); root + 1,981 agents
   = 192.99B tokens for the rewrite, 228 agents = 35.70B for the hillclimb.
8. **Parity checks need a spec first**: the Planner writes "overall machine specification, including the feature design,
   ground-truth TypeScript behavior, and verifier (parity check)" before code (`RUST:64`). In a factory: intent plus verifier
   plan are produced before the World and oracle (Crucible's M8 already orders it that way: tasks before data).
9. **Continual Harness edit loop** (`CH`): every `F` steps after warm-up `W`, a Refiner reads the trajectory window and runs
   four passes (prompt; sub-agents: create for repeated patterns, delete those not invoked productively; skills: codify
   successes, repair code that raised; memory: add, update, demote stale) (`CH:100-104, :248-262`). Findings:
   - Gain depends on capability. On Pro, from-scratch harness reached 100% of milestones at $130 median vs 98% for $215
     (~40% cheaper); Flash-Lite fell to 3-13% vs 20% for the minimal harness (`CH:444-450`). A capability floor exists.
   - Reset-free refinement "triages": it repairs skills the agent depends on, **tolerates regressions on unused ones, and
     accepts a long create-and-forget tail** (`CH:1875`). Most authored skills are never invoked.
   - Regressions follow resume: the first iteration after a checkpoint regresses and recovers in 2-3 iterations (`CH:2227`);
     newly authored sub-agents overtake repaired inherited ones before they have been repaired (`CH:2028-2032`).
   - >= 3 seeds, seed medians reported (`CH:346-349`).
10. **Persisted exploits** (`PAP:632-638`): an agent found an RCON shortcut, used it despite an anti-cheating heartbeat,
    "and then preserved it as a reusable skill." Remedy: "least-privilege action interfaces, independent state validation,
    and auditable rollback of contaminated refinements." For a factory: a generator that learns from solver traces will
    learn exploits; every harness edit needs A16 and a measured A/B before it persists (Crucible's `HarnessState.evaluate`
    already requires paired seeds, a noise re-run and guards, `actors/rlm/harness_state.py:169-215`).

## 3. The swarm article and GitSwarm: what matters for a factory

### 3.1 `SWARM` (Prime Intellect, 2026-10-06)

- **Compaction is a bet; sub-agents write the summary after the question is known** (`SWARM` "Multi-agent systems"). But
  one-shot sub-agents forget; disk offload "remembers but can't think".
- **Persistent sub-agent = reusable context**: keep it alive; ask it a new question later; your context grows only by
  question and answer. Total context capacity "grows to (roughly) one window per agent"; agents are "parallel over context,
  too" (`SWARM` "Hire the sub-agent"). Transfer: a long-lived per-family specialist (a Test writer that has read the family's
  World and past failures) is cheaper than re-reading; Crucible has this only for the hacker (`forge/specialists.py`).
- **Communication is the price**: every handoff is squeezed into a message, "costs tokens and is itself lossy"; if the hard
  part needs everything in one place, splitting does not help (`SWARM` "No free lunch", footnote 13). Transfer: the
  Test writer cannot be blind to the oracle AND share its context, so the factory's asymmetric information is a deliberate
  communication cost, correct for verifier writing.
- **Tree vs mesh**: a tree routes every cross-branch question through the common ancestor and fills the root's context; either
  every branch rediscovers the same specialist, or only one branch benefits. A mesh lets anyone ask the specialist.
  "With 10 agents, a tree is probably fine. With 10,000 ... peers" (`SWARM` "From trees to swarms"). Add hierarchy "wherever
  it pays for itself" (Coase). Transfer: tens of tasks per run = tree plus a shared repo; thousands = mesh over the repo.
- **Specialization by context**: "nobody has to assign roles; they just happen" because the agent that read the frontend first
  answers the next frontend question. Prime's counterpoint for us: do not let this decide *information access*; roles that
  must not see the oracle are fixed by staging (`forge/orchestrator.py:9-12`).
- Seed with human structure, then let agents change it (`SWARM` end).

### 3.2 GitSwarm (`GS`, arXiv 2610.04862; no licence on the repo, PATTERN only)

Design (`GS:160-230`): one repo per task as memory; atomic immutable commits; Git parent plus a **mandatory `informed_by=`**
set (semantic inheritance across branches); homogeneous workers with no planner or roles; up to `C` workers (5) in flight,
replacement on exit, budget `B` charged episodes; each episode ends with exactly one of **Contribute / Nominate / Abstain**;
abstention is free in `B`; a worker announces scope **and abort criteria** before editing via `gs_inflight`; `gs_dashboard`
shows budget, in-flight workers and a nomination tally labelled "not evidence of correctness"; **validation checks only
publication rules, not correctness** (`GS:178-185`); readout = most nominations, deterministic tie-break.

Measured results that matter:

| Finding | Number | Source |
|---|---|---|
| Compounding beats sequential continuation | ProgramBench 79.4% (500 workers) vs 65.1% Forced Codex; 73.1% at 100 workers; baseline flat 63.1-65.1% | `GS:341-346` |
| Work is reused | 94.7% of ProgramBench contributions (44.4% on proofs) are built upon; selected solution's ancestry covers 81.6-92.6% of the graph; Git-parent ancestry only ~1/4, the rest is declared cross-branch | `GS:131-133, :496-498, :541-543` |
| **Duplicate waste** | ~1 in 6 ProgramBench workers abstains as redundant (<2.7% on proofs); abstainers cost 6-7% of input/output tokens | `GS:719-722, :2890-2900` |
| Abstention cannot recover spent compute | 407/419 collision workers (97.1%) saw the conflicting commit land after their episode began; 99.0% had already run tests | `GS:2905-2910` |
| Async vs synchronous waves | 38.0 min vs 49.3 min (1.30x); score 62.06 +- 2.44 vs 63.11 +- 0.58 | `GS:3217-3248` |
| **Selector vs synthesizer** | central **selector** 64.40 +- 1.49 (+2.34 pp, 34.9 min); central **aggregator** 58.19 +- 4.53 (-3.87 pp, 43.9 min); decentralized nomination 62.06 +- 2.44. 9 tasks, 3 seeds, B=50, not significant | `GS:729-731, :3225-3252` |
| Nomination is not verification | "a worker preference, not an independent verification of correctness" | `GS:2865` |
| **Herding** | share of commits building on one adapter rose from 17% to ~90% in 9 h ("over-concentration") | `GS:2074` |
| Duplicate prevented by shared in-flight state | w0033 abstained after seeing w0032's identical announced experiment | `GS:2190` |
| Cheap evaluation misranks | 128-example evals disagreed with the full set on 5 of 12 pairs and with 512-example on 15 of 50; the swarm adopted a 128 -> 512 -> 4,873 funnel | `GS:2061-2063` |
| Persistence reduces rediscovery | shared-memory coverage stays high; independent baselines ~70-75% (2 long workers) and 20-23% (short workers) despite 4Q/3 queries | `GS:1441-1446` |
| Roles can still be configured | one research run assigned experimenter/synthesizer vs critic/verifier at 3:1; it "supports adaptive work selection within those roles, not specialization without predefined roles" | `GS:2198-2200` |
| Scaling costs | gains diminish at large budgets; input tokens grow as workers inspect a larger repo; workers inspect 5-7 candidate commits (code), ~3 (proofs) | `GS:344, :721, :766` |

Caveats from the paper: no causal attribution of gain to persistence or `informed_by`; small tradeoff studies; the
ProgramBench result is a 50-task subset (`GS:750-770`).

## 4. Mapping table (Prime Agent / GitSwarm mechanism -> Crucible)

Action: **DEPEND** = import the package; **PORT** = copy/adapt code (MIT, credit in `THIRD_PARTY.md`); **PATTERN** = reimplement
the idea. GitSwarm has no licence, so PATTERN only. "Built" = code exists; "unit only" = no driver outside tests.

| # | Mechanism | Crucible equivalent | Status | Action |
|---|---|---|---|---|
| 1 | Machine spec: states, typed ports, entry, `max_entries`, `foreach` (`factory.py:487-688`) | `forge/orchestrator.py` `Plan`/`Node` is an **acyclic** DAG (`:62-107`, cycle = `ValueError`); `TaskFSM` is hard-coded Python (`accept/repair.py:459`); harness kind `fsm` exists (`actors/rlm/harness_state.py:37`) with no executor | missing (declarative, cyclic) | **PORT** validator + canonicalizer; own executor |
| 2 | Guards over settle output (`_guard_passes :1270`) | none; `TaskFSM` branches in Python on `ver["passed"]` and `rev["approved"]` (`accept/repair.py:538-547`) | missing | **PORT** (~50 lines, pure) |
| 3 | Loop-back with findings, bounded re-entry | validators -> `feedback` -> next attempt, 3 attempts (`orchestrator.py:350-395`); `TaskFSM` rounds with findings (`repair.py:514-547`) | built; exhaustion = `failed`/escalate (better than Prime's silent `done`) | **PATTERN** |
| 4 | Retries (same prompt), infra vs content | `RoleInfraError` x3 with backoff, content attempts carry findings, per-node `timeout_s` (`orchestrator.py:350-380`) | built | PATTERN |
| 5 | Failure policy `fail_fast/continue/escalate`, pause + `resume` | plan-level: `rejected` / `abandoned` states only (`orchestrator.py:303`); `TaskFSM` `failed` = "escalate" | partial; no pausable run | **PATTERN** |
| 6 | Budgets: wall-clock, `max_children`, `max_transitions`, `max_parallel` | USD reserve/settle ledger (`forge/budget.py`), `run_plans(concurrency)`, `TreeSupervisor` turns/tokens/spawns/depth (`actors/rlm/tree.py:169`), `StopRules` per generator (`forge/economy.py`) | USD and tokens: better than Prime; no wall-clock or transition caps | PATTERN |
| 7 | Rate-limit backoff run-level deadline (`factory.py:2688`) | `actors/llm.py` backoff with jitter (THIRD_PARTY.md:40) | built | none |
| 8 | Per-state `model`/`thinking`; reviewer on a different model | `Role` protocol, `roles` dict (`orchestrator.py:50-56`); `llm_reviewer` docstring says "use a different model" (`repair.py:576`) | built, **not enforced** | PATTERN (config check) |
| 9 | Separate child context per role | staging: a role gets only `visible` inputs, never the store; canary breach = hard fail (`orchestrator.py:9-12, :250-255`); zoned repo with shingle scan (`forge/repo.py`); Harbor wrapper sandbox per role (`interop/harbor/role_runner.py`) | built, stronger than Prime | DEPEND (Harbor) / done |
| 10 | Verifier in a fresh sandbox | acceptance checks grade in fresh Worlds, content-bound cache key (`accept/run.py`); `ForgeWorkspace.verify` runs M9 on a candidate version, `commit` refuses an unverified one (`repair.py:380-420`) | built | done |
| 11 | Root orchestrator that writes no code; assigns, monitors, merges | `RLMTaskManager` repairs failing tasks (`repair.py:612`); `Orchestrator.run_plans` runs plans in parallel with no backlog, claim or dependency order across tasks | **unit only / missing** for new-task backlog | **PATTERN** |
| 12 | Persistent subagents + `agent_message` (parent/child/sibling, 20 pending, 3 burst) | `ChildRegistry.spawn/result/send/steer/cancel/events` with wake of idle children (`actors/rlm/tree.py:429-760`); factory roles are stateless single calls | built for RLM trees, **not wired to factory roles** | PATTERN (done in M16) |
| 13 | Resident states, `watch`/`graph` live view | `Dashboard` in `forge/swarm.py:69`; Studio explorer (M15) | partial | PATTERN |
| 14 | Run state persistence and resume | `Journal` (`events.jsonl`, input-hash reuse, torn-tail repair, late-breach revoke, `orchestrator.py:129-170`); `FSMState` journal (`repair.py:423-480`). Prime keeps run state in kernel memory only | built, **better than Prime** | PATTERN |
| 15 | Goals: persistent objective, budget-limited wrap-up, audit-before-complete, 3-strike no-progress | goal continuation in `RLMAgent` (M14), stall detector (`actors/rlm/stall.py`) | built | PATTERN |
| 16 | `/refine` typed edits, rollback, review gate, cooldown | `HarnessState.propose/evaluate/rollback_to` with paired seeds, alternating order, noise re-run, guards, >= 3 seeds (`harness_state.py:169`) | built, **measured** (Prime's `expectedOutcome` is not checked) | PATTERN |
| 17 | Differential / parity verification | A1 replay, A3 mutants, A16 property audit "end-state differential against the oracle" (`accept/checks/redteam.py:3-9`), `interop/replay.py`; no second-implementation cross-check of a verifier | partial | **PATTERN** |
| 18 | Hillclimb: alternating order, noise gate, neighbours, re-baseline, no numeric target | `harness_state.evaluate`; `learnability.measure` Wilson 4->8->16 (`forge/learnability.py:75`); `bench/compare.py` paired tests; ladder monotonicity (`forge/ladder.py:81`) | built for harness edits; **not applied to generator prompts / role versions** | PATTERN |
| 19 | Dogfooding traces -> coverage defects | A16 `MutationHacker` + `SandboxedHacker` promote hacks to A13 attacks and `registry/defects` (`forge/hacker.py:145-187`); M4 signals; audit hunts | partial; **no trace-review role over solver runs on accepted tasks** | PATTERN |
| 20 | GitSwarm repo memory, `informed_by`, immutable commits | `forge/repo.py` zoned git repo with `informed_by`, breach scan, per-role views | built, stronger (zones) | PATTERN |
| 21 | Announce scope + abort criteria, collision cancel, abstain free | `SwarmStage` `Announcement`, Jaccard collision, `Abstain`, `max_episodes = 4B` (`forge/swarm.py:36-240`) | **unit only** | PATTERN |
| 22 | Dashboard, "not correctness evidence" | `Dashboard` (`swarm.py:69`) | unit only | PATTERN |
| 23 | Nomination readout vs central selector vs aggregator | `swarm.winner` (`:234`), `Judge` = selector, "selecting gained 2.34, synthesizing lost 3.87" (`forge/judge.py:1-4`) | unit only | PATTERN |
| 24 | Duplicate waste 1 in 6 | `dedup.py` L0-L3 incl. seed-time dedup; announce-time collisions | built | PATTERN |
| 25 | Herding 17% -> 90% | `YieldBook.admit` quota 30% per operator and seed (`forge/economy.py`) | unit only | PATTERN |
| 26 | Async scheduling, no waves | `run_plans` semaphore `C=5`, `SwarmStage` replacement | built | PATTERN |
| 27 | Specialization by persistent context | `FamilyMemory`/`SpecialistHacker` (strategies only, `forge/specialists.py`) | hacker only | PATTERN |
| 28 | Cheap-to-full evaluation funnel (128 -> 512 -> 4,873) | A12 cost tiers; `learnability` sequential sampling; cost-class ordered check plan (`accept/plan.py`) | built | PATTERN |
| 29 | Ladder/difficulty steering | `ladder.py`, `curator.py` (band 0.1-0.7 train, frontier ub <= 0.2 bench), `tighten.py`, `compose.py`; retire -> recompose | built, `Curator` used only by tests and `bench/contamination.py` | done |
| 30 | `MACHINE.md` library, import/export, `list` | `packs` (M14) for task packs; no machine library | missing | PORT (format) |

## 5. Missing pieces for a task factory, ranked

Ranked by how much each blocks a mass-production loop whose output is trustworthy.

1. **An end-to-end factory driver that wires the built parts.** One run should go: seed/generator -> `SwarmStage` (Architect,
   Env builder, Test writer, Adversary as homogeneous workers into the zoned repo) -> `Orchestrator` assemble ->
   `accept_task` (A1-A18) -> `run_ladder` -> `Curator` placement -> `YieldBook` stop rules -> retire/recompose. Today each
   piece has unit tests and no caller (section 0, item 3). Without the driver, none of the yield, quota or stop rules has
   evidence from a real run.
2. **A declarative, serializable, cyclic factory machine and executor** (PORT of `factory.py` validator, guards,
   canonicalizer, `MACHINE.md`). It replaces the acyclic `Plan` plus hard-coded `TaskFSM`: states = roles, ports =
   staged documents, guards on `AuditReport.sound`, `a1.pass`, `a16.hacks`, `ladder.steer`; loop-back to the **owning role**
   with findings (the §12.5 diagram of `docs/research/m7-m8/02-forge-synthesis.md:737-753` draws this; the code does it
   only inside a node's validator loop). Persist the run in the journal (Prime's kernel-memory state is the defect to avoid)
   and make exhaustion a pausable `escalate`, never `done`.
3. **A verifier-validation stage that is independent of the verifier's author.** A1-A18 tests the verifier against the
   oracle, negatives and hacker mutants that the same factory produced. Add: (a) a **second, independently generated
   verifier** (or second reference solution) run differentially on the same end states, disagreements = defects (Rust
   parity pattern, lesson 4); (b) a **trace-review role** that reads real solver trajectories on accepted tasks and files
   defect candidates to `accept/registry/defects` (lesson 5); (c) a **cross-model reviewer** enforced by config
   (`reviewer.model_family != implementer.model_family`), recorded in the report, since today it is a docstring
   (`repair.py:576`).
4. **Spec-first planning stage with a verifier plan as an artifact.** A Planner that emits intent, stated properties, verifier
   plan and the expected failure taxonomy *before* World/oracle build, and a deterministic check that the later artifacts
   realize it (A7 two-way coverage partly does; `docs/modules/M08-forge.md` toy path has the Architect). Needed so that long
   horizon tasks (M16) get a horizon target and milestone set up front.
5. **Hill-climb loop for the factory's own roles** (the second orchestrator of `RUST:112-124`): backlog of hypotheses about
   role prompts, operators and knobs, A/B on the same seeds in alternating order with a noise gate, neighbour regressions
   (other families), two reviewers on different models, re-baseline after each merge, no numeric stop target. `HarnessState.
   evaluate` is the reusable core; the metric is accepted-tasks per dollar plus A16 hack-free rate (`forge/economy.py`).
   Guard against Factorio-style exploit persistence: no edit persists without A16 and a held-out family.
6. **A root factory manager over a task backlog.** Prime's root assigns toposorted tasks, monitors, merges, and does not write
   code. Crucible's `RLMTaskManager` only repairs existing tasks. Needs: backlog and dependency order across seeds
   (composition `chain`/`fan_in` makes dependencies real), claim/lease per task, live status (`watch`-like), pause/resume
   per run, and a merge step that is only `commit` after verification (`ForgeWorkspace.commit` already refuses otherwise).
7. **Persistent family specialists beyond the hacker.** A per-family Test writer and Architect kept as long-lived children
   (`ChildRegistry`) that answer follow-ups ("why does this check exist?") instead of cold re-reads; subject to the
   information rule (a Test writer's persistent context must never ingest oracle material; reuse the canary and zone scan on
   every message it receives).
8. **Run-level pause/resume and wall-clock caps.** Add `budget_ms`, `max_transitions`, `max_children` equivalents to the
   Orchestrator/driver, and `needs_verify` marking for a role that exited without a clean settle (Prime
   `factory.py:1349-1355`), so a half-finished Test writer is never silently counted.
9. **Machine library and live view.** `MACHINE.md` import/export/list (`factory.py:3904-4575`) and a graph/watch view in the
   Studio explorer. Low risk, high leverage once item 2 exists.
10. **Mesh option.** GitSwarm's "no planner" scale applies at hundreds of workers per family. `SwarmStage` already supports
    it; add `gs_inflight`-style peer inspection and an *informed_by completeness* metric (share of contributions built upon;
    GitSwarm's 94.7%) as a swarm health signal. Defer until item 1 produces data.

## 6. Design rules the sources imply (short)

- Roles that must not see the oracle are fixed by staging, not by prompts or specialization (`forge/orchestrator.py:9-12`).
  Persistence and mesh add *reuse*, never access.
- Use a **selector, not a synthesizer**, as the final stage: pick among accepted variants (`Judge`), never merge them
  (`GS:3225-3252`). A merged task re-enters acceptance as a new task.
- Count duplicate waste at announce time and with a diversity quota (1 in 6; 17% -> 90%).
- A nomination, vote or reviewer approval is a coordination signal; only the deterministic acceptance report decides
  (`GS:2865`; Crucible: "a swarm only proposes", `forge/swarm.py:25`).
- Cheap checks run first, with a funnel (128 -> 512 -> 4,873 evaluations; A-check cost classes) because small evaluations
  misrank (`GS:2061`).
- Never give the factory a numeric stop target on task count or yield alone; stop on marginal yield and diversity
  (`RUST:124`, `forge/economy.py` stop rules).

## 7. Not read, or uncertain

- Not read: `PA/crates/pa-daemon/src/factory_activity.rs`, the TUI factory view, `pa-core` compaction. The Python executor is
  the factory's complete semantics (the Rust crate bridges to it, `PAC/session_engine/factory_host.rs:1-22`).
- `PAP:964` (persistent child quote) is from the appendix example; `PAP:253-256` for queue semantics.
- The Prime blog's "fresh Prime Sandbox" verifier and per-task worktree are unverifiable in code; treat as process description.
- GitSwarm's tradeoff study is small (9 tasks, 3 seeds); the +2.34 / -3.87 pp are directional, as the authors state
  (`GS:3250-3252`).
- A reference [2] in `RUST`, "Automatic generation of high-performance RL environments" (Karten et al., COLM 2026), is the
  closest prior work on generating RL environments by translation with differential verification; it is not in the local
  corpus and was not read.
